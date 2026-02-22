# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # 06 — Apple M4 Pro Performance Lab
#
# Standalone notebook that progressively builds a modern transformer and
# benchmarks **GPU utilization** and **memory bandwidth** on Apple Silicon.
#
# - **No data loading, no tokenizer, no HuggingFace** — pure fake data
# - Every phase is independently runnable after Phase 0
# - Uses **MLX** (Apple's native ML framework) for optimal Metal GPU performance
#
# ### Apple M4 Pro specs
# | Spec | Value |
# |------|-------|
# | Memory | 48 GB unified |
# | GPU cores | 20 |
# | Peak bf16/fp16 TFLOPS | ~17.2 |
# | Peak fp32 TFLOPS | ~8.6 |
# | Memory bandwidth | 273 GB/s |
# | Arithmetic intensity | 17.2e12 / 273e9 ≈ 63 FLOPs/byte |
#
# > **Key difference from TPU v6e:** M4 Pro is **bandwidth-limited** (63 vs 574
# > FLOPs/byte), so memory-bound ops hurt more and compute utilization ceilings
# > are lower.  M4 Pro (Apple GPU family 9+) has native bf16 support — bf16 and
# > fp16 run at the same speed (~2x fp32).  We use **bf16** as default dtype,
# > matching the TPU version.

# %%
# !uv pip install "mlx>=0.22.0"

# %% [markdown]
# ## Phase 0 — Setup & Utilities

# %%
import functools as ft
import time
import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

# Apple M4 Pro constants
PEAK_TFLOPS = 17.2            # bf16/fp16 peak compute (same speed on M4 Pro)
PEAK_TFLOPS_FP32 = 8.6        # fp32 peak compute
MEM_BW_GBS = 273              # memory bandwidth in GB/s
MEM_GB = 48                   # unified memory

print(f"MLX version : {mx.__version__}")
print(f"Device info : {mx.device_info()}")
print(f"Peak TFLOPS : {PEAK_TFLOPS} (bf16/fp16), {PEAK_TFLOPS_FP32} (fp32)")

# %%
# === Benchmark harness ===

def benchmark(fn, *args, warmup=3, repeats=10, flop_count=None,
              mem_bytes=None, label="", peak_tflops=PEAK_TFLOPS):
    """Run fn repeatedly and report wall time, TFLOP/s, GPU%, memory bandwidth%.

    Args:
        fn: callable
        *args: arguments forwarded to fn
        warmup: number of warmup calls
        repeats: number of timed calls
        flop_count: manual FLOP count (int); None = skip GPU% calculation
        mem_bytes: total bytes read+written per call (int); None = skip BW calc
        label: display label for printing
        peak_tflops: peak TFLOPS for utilization calculation

    Returns:
        dict with wall_ms, tflops, gpu_pct, mem_bw_gbs, mem_bw_pct
    """
    # Warmup
    for _ in range(warmup):
        out = fn(*args)
        mx.eval(out)

    # Timed runs
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn(*args)
        mx.eval(out)
        times.append(time.perf_counter() - t0)

    wall_s = sum(times) / len(times)
    wall_ms = wall_s * 1000

    # FLOP/s and GPU utilization
    tflops = flop_count / (wall_s * 1e12) if flop_count else 0.0
    gpu_pct = (tflops / peak_tflops * 100
               if flop_count and peak_tflops else 0.0)

    # Memory bandwidth utilization
    mem_bw_gbs = mem_bytes / (wall_s * 1e9) if mem_bytes else 0.0
    mem_bw_pct = mem_bw_gbs / MEM_BW_GBS * 100 if mem_bytes else 0.0

    result = dict(label=label, wall_ms=wall_ms, tflops=tflops,
                  gpu_pct=gpu_pct, mem_bw_gbs=mem_bw_gbs,
                  mem_bw_pct=mem_bw_pct)

    # Print single row
    gpu_str = f"{gpu_pct:5.1f}%" if flop_count else "  n/a"
    tflop_str = f"{tflops:6.1f}" if flop_count else "   n/a"
    bw_str = f"{mem_bw_pct:5.1f}%" if mem_bytes else "  n/a"
    print(f"  {label:<40s}  {wall_ms:8.2f} ms  {tflop_str} TFLOP/s  GPU {gpu_str}  "
          f"BW {bw_str}")
    return result


def print_summary(results):
    """Print a formatted comparison table from benchmark results."""
    print(f"\n  {'Label':<40s}  {'Wall ms':>8s}  {'TFLOP/s':>8s}  {'GPU %':>6s}  "
          f"{'BW%':>7s}")
    print("  " + "-" * 80)
    for r in results:
        gpu = f"{r['gpu_pct']:5.1f}%" if r['tflops'] > 0 else "  n/a"
        tf = f"{r['tflops']:7.1f}" if r['tflops'] > 0 else "    n/a"
        bw = f"{r['mem_bw_pct']:5.1f}%" if r['mem_bw_pct'] > 0 else "  n/a"
        print(f"  {r['label']:<40s}  {r['wall_ms']:8.2f}  {tf}  {gpu}  "
              f"{bw:>7s}")
    print()

# %%
# === Fake data generators ===

def fake_tokens(batch_size, seq_len, vocab_size=32768, seed=0):
    mx.random.seed(seed)
    return mx.random.randint(0, vocab_size, (batch_size, seq_len))

def fake_hidden(batch_size, seq_len, n_embd, seed=0, dtype=mx.bfloat16):
    mx.random.seed(seed)
    return mx.random.normal((batch_size, seq_len, n_embd)).astype(dtype)

# %%
# === FLOP counting helpers ===

def matmul_flops(M, N, K, batch=1):
    """FLOPs for [M,K] @ [K,N].  2*M*N*K per batch element."""
    return 2 * batch * M * N * K

def attention_flops(B, H, T, D):
    """FLOPs for QK^T + AV (full T×T, not causal-halved)."""
    return 2 * (2 * B * H * T * T * D)

def layer_flops(B, T, E, H, KV, D, MLP):
    """GPU-relevant FLOPs for one transformer layer."""
    tok = B * T
    q  = 2 * tok * E * H * D
    k  = 2 * tok * E * KV * D
    v  = 2 * tok * E * KV * D
    att = attention_flops(B, H, T, D)
    proj = 2 * tok * H * D * E
    gate = 2 * tok * E * MLP
    up   = 2 * tok * E * MLP
    down = 2 * tok * MLP * E
    return q + k + v + att + proj + gate + up + down

# %%
# === PerfConfig ===

@dataclass(kw_only=True, frozen=True)
class PerfConfig:
    """Model dimensions.  Defaults sized for M4 Pro benchmarking.

    Smaller than the TPU v6e config (05_tpu_perf.py) because M4 Pro has
    ~50x less compute (17 vs 918 TFLOPS).  Keeps total runtime manageable
    while still being large enough for meaningful GPU utilization numbers.
    """
    batch_size: int = 4
    seq_len: int = 2048
    n_head: int = 4
    n_kv_head: int = 2
    head_dim: int = 128
    n_embd: int = 512        # n_head * head_dim
    mlp_dim: int = 1536      # 3x expansion for SwiGLU
    vocab_size: int = 32768
    n_layer: int = 8
    softcap: float = 15.0
    num_lm_head_chunks: int = 8

cfg = PerfConfig()
assert cfg.vocab_size % 256 == 0, f"vocab_size must be divisible by 256, got {cfg.vocab_size}"
assert cfg.n_embd == cfg.n_head * cfg.head_dim, \
    f'n_embd ({cfg.n_embd}) must equal n_head * head_dim ({cfg.n_head * cfg.head_dim})'
print(f"Config: B={cfg.batch_size}, T={cfg.seq_len}, E={cfg.n_embd}, "
      f"H={cfg.n_head}, KV={cfg.n_kv_head}, D={cfg.head_dim}, "
      f"MLP={cfg.mlp_dim}, V={cfg.vocab_size}, L={cfg.n_layer}")

# %% [markdown]
# ## Phase 1 — Matmul Baseline
#
# Establish the GPU compute ceiling with pure matmuls.
# - Aligned sizes should hit best utilization
# - bf16 vs fp32 dtype comparison

# %%
# 1a. Square matmul — aligned sizes
print("=== Square matmul (aligned) ===")
results_1a = []
for size in [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]:
    mx.random.seed(0)
    a = mx.random.normal((size, size)).astype(mx.bfloat16)
    mx.random.seed(1)
    b = mx.random.normal((size, size)).astype(mx.bfloat16)

    def mm(a, b):
        return a @ b

    # bytes: read A + read B + write C, all fp16 (2 bytes each)
    mem = 3 * size * size * 2
    r = benchmark(mm, a, b, flop_count=matmul_flops(size, size, size),
                  mem_bytes=mem, label=f"matmul {size}x{size}")
    results_1a.append(r)

print_summary(results_1a)

# %%
# 1c. Batched matmul — simulating transformer projections
print("=== Batched matmul (transformer-shaped) ===")
results_1c = []
shapes = [
    (4, 2048, 512, 512, "B=4 hidden->hidden"),
    (4, 2048, 512, 1536, "B=4 hidden->mlp"),
    (4, 2048, 1536, 512, "B=4 mlp->hidden"),
    (4, 2048, 512, 32768, "B=4 hidden->vocab"),
]
for B, M, K, N, desc in shapes:
    mx.random.seed(0)
    a = mx.random.normal((B * M, K)).astype(mx.bfloat16)
    mx.random.seed(1)
    b = mx.random.normal((K, N)).astype(mx.bfloat16)

    def mm(a, b):
        return a @ b

    mem = (B * M * K + K * N + B * M * N) * 2
    r = benchmark(mm, a, b, flop_count=matmul_flops(B * M, N, K),
                  mem_bytes=mem, label=desc)
    results_1c.append(r)
print_summary(results_1c)

# %%
# 1b. Dtype comparison — bf16 vs fp32
print("=== Dtype comparison (4096x4096 matmul) ===")
results_1d = []
size = 4096
for dtype, dtype_name, bytes_per, peak in [
    (mx.bfloat16, "bf16", 2, PEAK_TFLOPS),
    (mx.float32, "fp32", 4, PEAK_TFLOPS_FP32),
]:
    mx.random.seed(0)
    a = mx.random.normal((size, size)).astype(dtype)
    mx.random.seed(1)
    b = mx.random.normal((size, size)).astype(dtype)

    def mm(a, b):
        return a @ b

    mem = 3 * size * size * bytes_per
    r = benchmark(mm, a, b, flop_count=matmul_flops(size, size, size),
                  mem_bytes=mem, label=f"matmul {size}x{size} ({dtype_name})",
                  peak_tflops=peak)
    results_1d.append(r)
print_summary(results_1d)

# %% [markdown]
# ### Ideas to try
# - **int8 matmul**: `mx.quantize` for int4/int8 weight quantization
# - **Rectangular aspect ratios**: tall-skinny vs short-wide
# - **`mx.compile`**: does explicit compilation help matmul throughput?

# %% [markdown]
# ## Phase 2 — Individual Transformer Components
#
# Isolate each building block and measure independently.
# - **RMSNorm / RoPE**: memory-bound (expect low GPU%)
# - **MLP**: compute-heavy (3 large matmuls)
# - **Attention**: mixed (projections = compute, softmax = memory)
#
# For RMSNorm and RoPE we compare manual implementations with
# `mx.fast.*` fused kernels.

# %%
# === Model primitives ===

def rms_norm_manual(x, weight, eps=1e-6):
    """RMSNorm — manual implementation."""
    return weight * x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)


def precompute_rope(seq_len, head_dim, base=10000):
    """Precompute rotary embedding cos/sin tables."""
    channel_range = mx.arange(0, head_dim, 2).astype(mx.float32)
    inv_freq = 1.0 / (base ** (channel_range / head_dim))
    t = mx.arange(seq_len).astype(mx.float32)
    freqs = mx.outer(t, inv_freq)
    cos = mx.cos(freqs).astype(mx.bfloat16)
    sin = mx.sin(freqs).astype(mx.bfloat16)
    return cos, sin


def apply_rope_manual(x, cos, sin):
    """Apply rotary embeddings manually. x: (B, H, T, D), cos/sin: (1, 1, T, D/2)"""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return mx.concatenate([y1, y2], axis=-1)

# %%
# 2a. RMSNorm — manual vs mx.fast.rms_norm
print("=== RMSNorm ===")
x = fake_hidden(cfg.batch_size, cfg.seq_len, cfg.n_embd)
norm_weight = mx.ones((cfg.n_embd,), dtype=mx.bfloat16)

def bench_rmsnorm_manual(x, w):
    return rms_norm_manual(x, w)

def bench_rmsnorm_fast(x, w):
    return mx.fast.rms_norm(x, w, 1e-6)

r_norm_manual = benchmark(bench_rmsnorm_manual, x, norm_weight,
                          flop_count=None, label="RMSNorm (manual)")
r_norm_fast = benchmark(bench_rmsnorm_fast, x, norm_weight,
                        flop_count=None, label="RMSNorm (mx.fast)")
print(f"  Speedup from mx.fast.rms_norm: {r_norm_manual['wall_ms'] / max(r_norm_fast['wall_ms'], 0.001):.2f}x")

# %%
# 2b. RoPE — manual vs mx.fast.rope
print("\n=== RoPE ===")
cos, sin = precompute_rope(cfg.seq_len, cfg.head_dim)
cos_b = cos[None, None, :, :]
sin_b = sin[None, None, :, :]

mx.random.seed(0)
q = mx.random.normal((cfg.batch_size, cfg.n_head, cfg.seq_len, cfg.head_dim)).astype(mx.bfloat16)

def bench_rope_manual(q, cos, sin):
    return apply_rope_manual(q, cos, sin)

def bench_rope_fast(q):
    return mx.fast.rope(q, dims=cfg.head_dim, traditional=False,
                        base=10000.0, scale=1.0, offset=0)

r_rope_manual = benchmark(bench_rope_manual, q, cos_b, sin_b,
                          flop_count=None, label="RoPE (manual)")
r_rope_fast = benchmark(bench_rope_fast, q,
                        flop_count=None, label="RoPE (mx.fast)")
print(f"  Speedup from mx.fast.rope: {r_rope_manual['wall_ms'] / max(r_rope_fast['wall_ms'], 0.001):.2f}x")

# %%
# 2c. SwiGLU MLP — 3 large matmuls (gate, up, down)
print("\n=== SwiGLU MLP ===")

def init_mlp_params(cfg, seed=42):
    mx.random.seed(seed)
    s = (3.0 ** 0.5) * (cfg.n_embd ** -0.5)
    return dict(
        w_gate=mx.random.uniform(-s, s, (cfg.n_embd, cfg.mlp_dim)).astype(mx.bfloat16),
        w_up=mx.random.uniform(-s, s, (cfg.n_embd, cfg.mlp_dim)).astype(mx.bfloat16),
        w_down=mx.zeros((cfg.mlp_dim, cfg.n_embd), dtype=mx.bfloat16),
        norm_w=mx.ones((cfg.n_embd,), dtype=mx.bfloat16),
    )

mlp_params = init_mlp_params(cfg)
x = fake_hidden(cfg.batch_size, cfg.seq_len, cfg.n_embd)

def bench_mlp(x, params):
    h = mx.fast.rms_norm(x, params['norm_w'], 1e-6)
    gate = nn.silu(mx.einsum('btd,dh->bth', h, params['w_gate']))
    up = mx.einsum('btd,dh->bth', h, params['w_up'])
    return mx.einsum('bth,hd->btd', gate * up, params['w_down'])

tok = cfg.batch_size * cfg.seq_len
mlp_flops = 3 * 2 * tok * cfg.n_embd * cfg.mlp_dim
r_mlp = benchmark(bench_mlp, x, mlp_params, flop_count=mlp_flops, label="SwiGLU MLP")

# %%
# 2d. Attention — einsum variant (manual QK^T + softmax + AV)
print("\n=== Attention (einsum) ===")

def init_attn_params(cfg, seed=42):
    mx.random.seed(seed)
    s = (3.0 ** 0.5) * (cfg.n_embd ** -0.5)
    return dict(
        c_q=mx.random.uniform(-s, s, (cfg.n_embd, cfg.n_head, cfg.head_dim)).astype(mx.bfloat16),
        c_k=mx.random.uniform(-s, s, (cfg.n_embd, cfg.n_kv_head, cfg.head_dim)).astype(mx.bfloat16),
        c_v=mx.random.uniform(-s, s, (cfg.n_embd, cfg.n_kv_head, cfg.head_dim)).astype(mx.bfloat16),
        c_proj=mx.zeros((cfg.n_head, cfg.head_dim, cfg.n_embd), dtype=mx.bfloat16),
        norm_w=mx.ones((cfg.n_embd,), dtype=mx.bfloat16),
    )

attn_params = init_attn_params(cfg)

def _expand_kv(k, v, n_head, n_kv_head):
    """Repeat KV heads to match Q head count for einsum backend."""
    if n_kv_head == n_head:
        return k, v
    ratio = n_head // n_kv_head
    # k/v: (B, KV, T, D) -> (B, H, T, D) via repeat
    k = mx.repeat(k, ratio, axis=1)
    v = mx.repeat(v, ratio, axis=1)
    return k, v

def bench_attn_einsum(x, params):
    h = mx.fast.rms_norm(x, params['norm_w'], 1e-6)
    q = mx.einsum('btd,dhk->bhtk', h, params['c_q'])
    k = mx.einsum('btd,dhk->bhtk', h, params['c_k'])
    v = mx.einsum('btd,dhk->bhtk', h, params['c_v'])
    q = mx.fast.rope(q, dims=cfg.head_dim, traditional=False, base=10000.0, scale=1.0, offset=0)
    k = mx.fast.rope(k, dims=cfg.head_dim, traditional=False, base=10000.0, scale=1.0, offset=0)
    # QK norm
    qn_w = mx.ones((cfg.head_dim,), dtype=mx.bfloat16)
    q = mx.fast.rms_norm(q, qn_w, 1e-6)
    k = mx.fast.rms_norm(k, qn_w, 1e-6)
    k_exp, v_exp = _expand_kv(k, v, cfg.n_head, cfg.n_kv_head)
    scale = cfg.head_dim ** -0.5
    T = x.shape[1]
    scores = mx.einsum('bhtd,bhsd->bhts', q, k_exp) * scale
    # Causal mask
    rows = mx.arange(T)[:, None]
    cols = mx.arange(T)[None, :]
    mask = cols <= rows
    scores = mx.where(mask[None, None, :, :], scores, mx.array(float('-inf')))
    attn_weights = mx.softmax(scores, axis=-1)
    attn_out = mx.einsum('bhts,bhsd->bhtd', attn_weights, v_exp)
    return mx.einsum('bhtd,hde->bte', attn_out, params['c_proj'])

tok = cfg.batch_size * cfg.seq_len
proj_flops = (2 * tok * cfg.n_embd * cfg.n_head * cfg.head_dim +
              2 * tok * cfg.n_embd * cfg.n_kv_head * cfg.head_dim +
              2 * tok * cfg.n_embd * cfg.n_kv_head * cfg.head_dim +
              2 * tok * cfg.n_head * cfg.head_dim * cfg.n_embd)
attn_core = attention_flops(cfg.batch_size, cfg.n_head, cfg.seq_len, cfg.head_dim)
total_attn_flops = proj_flops + attn_core

x = fake_hidden(cfg.batch_size, cfg.seq_len, cfg.n_embd)
r_attn_ein = benchmark(bench_attn_einsum, x, attn_params,
                        flop_count=total_attn_flops, label="Attention (einsum)")

# %%
# 2e. Attention — mx.fast.scaled_dot_product_attention (SDPA)
print("\n=== Attention (SDPA) ===")

def bench_attn_sdpa(x, params):
    h = mx.fast.rms_norm(x, params['norm_w'], 1e-6)
    q = mx.einsum('btd,dhk->bhtk', h, params['c_q'])
    k = mx.einsum('btd,dhk->bhtk', h, params['c_k'])
    v = mx.einsum('btd,dhk->bhtk', h, params['c_v'])
    q = mx.fast.rope(q, dims=cfg.head_dim, traditional=False, base=10000.0, scale=1.0, offset=0)
    k = mx.fast.rope(k, dims=cfg.head_dim, traditional=False, base=10000.0, scale=1.0, offset=0)
    # QK norm
    qn_w = mx.ones((cfg.head_dim,), dtype=mx.bfloat16)
    q = mx.fast.rms_norm(q, qn_w, 1e-6)
    k = mx.fast.rms_norm(k, qn_w, 1e-6)
    # SDPA handles GQA natively — pass K/V with n_kv heads directly
    T = x.shape[1]
    mask = mx.triu(mx.full((T, T), float('-inf'), dtype=q.dtype), k=1)
    attn_out = mx.fast.scaled_dot_product_attention(
        q, k, v, scale=cfg.head_dim ** -0.5, mask=mask)
    return mx.einsum('bhtd,hde->bte', attn_out, params['c_proj'])

r_attn_sdpa = benchmark(bench_attn_sdpa, x, attn_params,
                         flop_count=total_attn_flops, label="Attention (SDPA)")

# %%
# 2f. Component comparison
print("\n=== Phase 2 Summary ===")
print_summary([r_norm_manual, r_norm_fast, r_rope_manual, r_rope_fast,
               r_mlp, r_attn_ein, r_attn_sdpa])

# %% [markdown]
# ### Ideas to try
# - **Remove QK-norm** from attention — saves 2 RMSNorm calls on Q and K
# - **Vary head_dim**: try 64, 128, 256, 512 — how does per-component GPU% change?
# - **GQA within attention**: try n_kv_head = 1 (MQA), 2, 4 (MHA)
# - **`mx.compile`** wrapping individual components — does it help?

# %% [markdown]
# ## Phase 3 — Single Transformer Layer
#
# Assemble: pre-norm + attention + residual + pre-norm + MLP + residual.
# Compare full layer time with sum of Phase 2 parts.

# %%
# === Single layer functions ===

def init_layer_params(cfg, seed=42):
    """Initialize params for one transformer layer."""
    mx.random.seed(seed)
    s = (3.0 ** 0.5) * (cfg.n_embd ** -0.5)
    return dict(
        c_q=mx.random.uniform(-s, s, (cfg.n_embd, cfg.n_head, cfg.head_dim)).astype(mx.bfloat16),
        c_k=mx.random.uniform(-s, s, (cfg.n_embd, cfg.n_kv_head, cfg.head_dim)).astype(mx.bfloat16),
        c_v=mx.random.uniform(-s, s, (cfg.n_embd, cfg.n_kv_head, cfg.head_dim)).astype(mx.bfloat16),
        c_proj=mx.zeros((cfg.n_head, cfg.head_dim, cfg.n_embd), dtype=mx.bfloat16),
        w_gate=mx.random.uniform(-s, s, (cfg.n_embd, cfg.mlp_dim)).astype(mx.bfloat16),
        w_up=mx.random.uniform(-s, s, (cfg.n_embd, cfg.mlp_dim)).astype(mx.bfloat16),
        w_down=mx.zeros((cfg.mlp_dim, cfg.n_embd), dtype=mx.bfloat16),
        attn_norm_w=mx.ones((cfg.n_embd,), dtype=mx.bfloat16),
        mlp_norm_w=mx.ones((cfg.n_embd,), dtype=mx.bfloat16),
        qk_norm_w=mx.ones((cfg.head_dim,), dtype=mx.bfloat16),
    )


def single_layer_forward(cfg, layer, x, *, attn_impl='sdpa',
                         use_rope=True, use_qk_norm=True):
    """Forward pass for one transformer layer."""
    h = mx.fast.rms_norm(x, layer['attn_norm_w'], 1e-6)

    # --- Attention ---
    q = mx.einsum('btd,dhk->bhtk', h, layer['c_q'])
    k = mx.einsum('btd,dhk->bhtk', h, layer['c_k'])
    v = mx.einsum('btd,dhk->bhtk', h, layer['c_v'])

    if use_rope:
        q = mx.fast.rope(q, dims=cfg.head_dim, traditional=False,
                         base=10000.0, scale=1.0, offset=0)
        k = mx.fast.rope(k, dims=cfg.head_dim, traditional=False,
                         base=10000.0, scale=1.0, offset=0)
    if use_qk_norm:
        q = mx.fast.rms_norm(q, layer['qk_norm_w'], 1e-6)
        k = mx.fast.rms_norm(k, layer['qk_norm_w'], 1e-6)

    T = x.shape[1]
    if attn_impl == 'sdpa':
        mask = mx.triu(mx.full((T, T), float('-inf'), dtype=q.dtype), k=1)
        attn_out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=cfg.head_dim ** -0.5, mask=mask)
    else:  # einsum
        k_exp, v_exp = _expand_kv(k, v, cfg.n_head, cfg.n_kv_head)
        scale = cfg.head_dim ** -0.5
        scores = mx.einsum('bhtd,bhsd->bhts', q, k_exp) * scale
        rows = mx.arange(T)[:, None]
        cols = mx.arange(T)[None, :]
        causal = cols <= rows
        scores = mx.where(causal[None, None, :, :], scores, mx.array(float('-inf')))
        attn_weights = mx.softmax(scores, axis=-1)
        attn_out = mx.einsum('bhts,bhsd->bhtd', attn_weights, v_exp)

    attn_out = mx.einsum('bhtd,hde->bte', attn_out, layer['c_proj'])
    x = x + attn_out

    # --- SwiGLU MLP ---
    h2 = mx.fast.rms_norm(x, layer['mlp_norm_w'], 1e-6)
    gate = nn.silu(mx.einsum('btd,dh->bth', h2, layer['w_gate']))
    up = mx.einsum('btd,dh->bth', h2, layer['w_up'])
    mlp_out = mx.einsum('bth,hd->btd', gate * up, layer['w_down'])
    x = x + mlp_out
    return x

# %%
# 3a. Full layer benchmark
print("=== Single layer (SDPA) ===")
layer_p = init_layer_params(cfg)
x = fake_hidden(cfg.batch_size, cfg.seq_len, cfg.n_embd)
lf = layer_flops(cfg.batch_size, cfg.seq_len, cfg.n_embd,
                 cfg.n_head, cfg.n_kv_head, cfg.head_dim, cfg.mlp_dim)

def bench_layer(x, layer):
    return single_layer_forward(cfg, layer, x, attn_impl='sdpa')

r_layer = benchmark(bench_layer, x, layer_p,
                    flop_count=lf, label="Full layer (SDPA)")

print(f"\n  Sum of parts (MLP + attn_sdpa):  "
      f"{r_mlp['wall_ms'] + r_attn_sdpa['wall_ms']:.2f} ms")
print(f"  Full layer:                      {r_layer['wall_ms']:.2f} ms")
print(f"  Delta (overhead / fusion):       "
      f"{r_layer['wall_ms'] - r_mlp['wall_ms'] - r_attn_sdpa['wall_ms']:.2f} ms")

# %%
# 3b. Layer ablations
print("\n=== Layer ablations ===")
results_3b = []

for label, kw in [
    ("sdpa, +rope, +qknorm",  dict(attn_impl='sdpa',   use_rope=True,  use_qk_norm=True)),
    ("sdpa, -rope, +qknorm",  dict(attn_impl='sdpa',   use_rope=False, use_qk_norm=True)),
    ("sdpa, +rope, -qknorm",  dict(attn_impl='sdpa',   use_rope=True,  use_qk_norm=False)),
    ("sdpa, -rope, -qknorm",  dict(attn_impl='sdpa',   use_rope=False, use_qk_norm=False)),
    ("einsum, +rope, +qknorm", dict(attn_impl='einsum', use_rope=True,  use_qk_norm=True)),
]:
    def bench_fn(x, layer, _kw=kw):
        return single_layer_forward(cfg, layer, x, **_kw)

    r = benchmark(bench_fn, x, layer_p, flop_count=lf, label=label)
    results_3b.append(r)

print_summary(results_3b)

# %% [markdown]
# ### Ideas to try
# - **Remove RMSNorm entirely** (unsafe for training but measures its overhead)
# - **`mx.compile`** on the full layer — does Metal kernel fusion kick in?
# - **Two consecutive layers** — does MLX pipeline them?

# %% [markdown]
# ## Phase 4 — Stacking Layers
#
# Does GPU% change with depth? How does memory scale?

# %%
# === Multi-layer model ===

def init_all_layers(cfg, n_layers, seed=42):
    layers = {}
    for i in range(n_layers):
        layers[i] = init_layer_params(cfg, seed=seed + i * 7)
    return layers


def multi_layer_forward(cfg, layers, n_layers, x, attn_impl='sdpa'):
    for i in range(n_layers):
        x = single_layer_forward(cfg, layers[i], x, attn_impl=attn_impl)
    norm_w = mx.ones((cfg.n_embd,), dtype=mx.bfloat16)
    return mx.fast.rms_norm(x, norm_w, 1e-6)

# %%
# 4a. Depth sweep
print("=== Depth sweep ===")
results_4a = []
x = fake_hidden(cfg.batch_size, cfg.seq_len, cfg.n_embd)
final_norm_w = mx.ones((cfg.n_embd,), dtype=mx.bfloat16)

for n_layers in [1, 2, 4, 8]:
    layers = init_all_layers(cfg, n_layers)
    fl = n_layers * layer_flops(cfg.batch_size, cfg.seq_len, cfg.n_embd,
                                cfg.n_head, cfg.n_kv_head, cfg.head_dim, cfg.mlp_dim)

    def bench_fn(x, layers, _n=n_layers):
        return multi_layer_forward(cfg, layers, _n, x)

    mx.reset_peak_memory()
    r = benchmark(bench_fn, x, layers, flop_count=fl, label=f"{n_layers} layers")
    r['ms_per_layer'] = r['wall_ms'] / n_layers
    r['peak_mem_mb'] = mx.get_peak_memory() / 1e6
    results_4a.append(r)

print_summary(results_4a)
print("  Per-layer time & memory:")
for r in results_4a:
    print(f"    {r['label']:<20s}  {r['ms_per_layer']:.2f} ms/layer  "
          f"peak mem: {r['peak_mem_mb']:.0f} MB")

# %% [markdown]
# ### Ideas to try
# - **`mx.checkpoint` (remat)** on each layer — how many more layers fit?
# - **Max batch_size × n_layers grid**: find the OOM boundary
# - **`mx.compile`** on the full multi-layer forward

# %% [markdown]
# ## Phase 5 — Embedding & LM Head
#
# - Embedding: pure memory lookup (0% GPU compute)
# - LM head: large matmul `(B*T, n_embd) @ (n_embd, vocab)` — high GPU%
# - Chunked vs non-chunked loss comparison

# %%
# 5a. Embedding lookup
print("=== Embedding ===")
mx.random.seed(0)
wte = mx.random.normal((cfg.vocab_size, cfg.n_embd)).astype(mx.bfloat16)
emb_norm_w = mx.ones((cfg.n_embd,), dtype=mx.bfloat16)
tokens = fake_tokens(cfg.batch_size, cfg.seq_len)

def bench_embed(tokens, wte, norm_w):
    return mx.fast.rms_norm(wte[tokens], norm_w, 1e-6)

r_embed = benchmark(bench_embed, tokens, wte, emb_norm_w, flop_count=None,
                    label="Embedding + norm")

# %%
# 5b. LM head — non-chunked
print("\n=== LM head (non-chunked) ===")
mx.random.seed(1)
lm_head = mx.random.normal((cfg.n_embd, cfg.vocab_size)).astype(mx.bfloat16) * 0.001
hidden = fake_hidden(cfg.batch_size, cfg.seq_len, cfg.n_embd)
labels = fake_tokens(cfg.batch_size, cfg.seq_len)

def bench_lm_head(hidden, lm_head, labels):
    logits = mx.einsum('btd,dv->btv', hidden, lm_head)
    logits = logits.astype(mx.float32)
    logits = cfg.softcap * mx.tanh(logits / cfg.softcap)
    return mx.mean(nn.losses.cross_entropy(logits, labels))

lm_flops = matmul_flops(cfg.batch_size * cfg.seq_len, cfg.vocab_size, cfg.n_embd)
r_lm = benchmark(bench_lm_head, hidden, lm_head, labels,
                 flop_count=lm_flops, label="LM head (non-chunked)")

# %%
# 5c. Chunked LM head loss
print("\n=== LM head (chunked, 8 chunks) ===")

def chunked_lm_head_loss(hidden, lm_head, labels, config):
    """Chunked LM head to reduce peak memory from full vocab logits."""
    B, T, D = hidden.shape
    N = config.num_lm_head_chunks
    S = B * T // N
    hidden_flat = hidden.reshape(N, S, D)
    labels_flat = labels.reshape(N, S)

    total_loss = mx.array(0.0)
    for i in range(N):
        logits = hidden_flat[i] @ lm_head          # (S, vocab_size)
        logits = logits.astype(mx.float32)
        logits = config.softcap * mx.tanh(logits / config.softcap)
        total_loss = total_loss + mx.sum(nn.losses.cross_entropy(logits, labels_flat[i]))
    return total_loss / (B * T)

def bench_lm_chunked(hidden, lm_head, labels):
    return chunked_lm_head_loss(hidden, lm_head, labels, cfg)

r_lm_chunked = benchmark(bench_lm_chunked, hidden, lm_head, labels,
                          flop_count=lm_flops, label="LM head (chunked, 8)")

# %%
# 5d. Comparison
print("\n=== Phase 5 Summary ===")
print_summary([r_embed, r_lm, r_lm_chunked])

# %% [markdown]
# ### Ideas to try
# - **Vary `num_lm_head_chunks`**: 1, 2, 4, 8, 16 — speed vs memory tradeoff
# - **Vocab alignment**: 32768 vs 50257 (GPT-2, non-aligned) — how much padding waste?
# - **Weight-tied embedding** (`wte.T` as lm_head) — saves memory

# %% [markdown]
# ## Phase 6 — Forward vs Forward+Backward
#
# The backward pass typically costs 2-3x the forward.
# Gradient checkpointing (remat) trades compute for memory.

# %%
# === Full model for fwd/bwd testing ===

def init_full_model(cfg, seed=42):
    """Initialize all model params (embed + layers + lm_head)."""
    mx.random.seed(seed)
    params = {}
    params['wte'] = mx.random.normal((cfg.vocab_size, cfg.n_embd)).astype(mx.bfloat16)
    params['lm_head'] = mx.random.normal((cfg.n_embd, cfg.vocab_size)).astype(mx.bfloat16) * 0.001
    params['emb_norm_w'] = mx.ones((cfg.n_embd,), dtype=mx.bfloat16)
    params['final_norm_w'] = mx.ones((cfg.n_embd,), dtype=mx.bfloat16)
    params['layers'] = init_all_layers(cfg, cfg.n_layer, seed=seed + 100)
    return params


def model_forward(cfg, params, tokens):
    """Full forward: embed -> layers -> final_norm.  Returns hidden (B,T,E)."""
    x = mx.fast.rms_norm(params['wte'][tokens], params['emb_norm_w'], 1e-6)
    for i in range(cfg.n_layer):
        x = single_layer_forward(cfg, params['layers'][i], x, attn_impl='sdpa')
    return mx.fast.rms_norm(x, params['final_norm_w'], 1e-6)


def model_forward_remat(cfg, params, tokens):
    """Same as model_forward but with mx.checkpoint on each layer."""
    x = mx.fast.rms_norm(params['wte'][tokens], params['emb_norm_w'], 1e-6)
    for i in range(cfg.n_layer):
        layer_fn = ft.partial(single_layer_forward, cfg, attn_impl='sdpa')
        x = mx.checkpoint(layer_fn)(params['layers'][i], x)
    return mx.fast.rms_norm(x, params['final_norm_w'], 1e-6)


full_params = init_full_model(cfg)
tokens = fake_tokens(cfg.batch_size, cfg.seq_len)
labels = fake_tokens(cfg.batch_size, cfg.seq_len)

total_model_flops = (cfg.n_layer * layer_flops(cfg.batch_size, cfg.seq_len, cfg.n_embd,
                     cfg.n_head, cfg.n_kv_head, cfg.head_dim, cfg.mlp_dim) +
                     matmul_flops(cfg.batch_size * cfg.seq_len, cfg.vocab_size, cfg.n_embd))

# %%
# 6a. Forward only
print("=== Forward only ===")

def bench_fwd(params, tokens, labels):
    hidden = model_forward(cfg, params, tokens)
    return chunked_lm_head_loss(hidden, params['lm_head'], labels, cfg)

r_fwd = benchmark(bench_fwd, full_params, tokens, labels, warmup=2, repeats=5,
                  flop_count=total_model_flops, label="Forward only")

# %%
# 6b. Forward + backward
print("\n=== Forward + Backward ===")

def bench_fwd_bwd(params, tokens, labels):
    def loss_fn(p):
        hidden = model_forward(cfg, p, tokens)
        return chunked_lm_head_loss(hidden, p['lm_head'], labels, cfg)
    return mx.value_and_grad(loss_fn)(params)

bwd_flops = 3 * total_model_flops
r_fwd_bwd = benchmark(bench_fwd_bwd, full_params, tokens, labels, warmup=2, repeats=5,
                       flop_count=bwd_flops, label="Forward+Backward")

# %%
# 6c. Forward + backward with remat
print("\n=== Forward + Backward (remat) ===")

def bench_fwd_bwd_remat(params, tokens, labels):
    def loss_fn(p):
        hidden = model_forward_remat(cfg, p, tokens)
        return chunked_lm_head_loss(hidden, p['lm_head'], labels, cfg)
    return mx.value_and_grad(loss_fn)(params)

r_remat = benchmark(bench_fwd_bwd_remat, full_params, tokens, labels, warmup=2, repeats=5,
                    flop_count=bwd_flops, label="Fwd+Bwd (remat)")

# %%
# 6d. Comparison
print("\n=== Phase 6 Summary ===")
print_summary([r_fwd, r_fwd_bwd, r_remat])
print(f"  Backward / Forward ratio:  {r_fwd_bwd['wall_ms'] / max(r_fwd['wall_ms'], 0.01):.2f}x")
print(f"  Remat overhead vs no-remat: {r_remat['wall_ms'] / max(r_fwd_bwd['wall_ms'], 0.01):.2f}x")

# %% [markdown]
# ### Ideas to try
# - **Partial remat**: checkpoint every other layer, or only attention
# - **Remat at different batch sizes** — at small batch size memory may not be the bottleneck
# - **`mx.compile`** on loss_fn — does it help backward pass?

# %% [markdown]
# ## Phase 7 — Optimization Experiments
#
# Systematic sweeps over key parameters to find the performance sweet spots.

# %%
# 7a. Batch size sweep (forward+backward)
print("=== Batch size sweep ===")
results_7a = []

for bs in [1, 2, 4, 8]:
    cfg_bs = PerfConfig(batch_size=bs)
    p = init_full_model(cfg_bs)
    tok = fake_tokens(bs, cfg_bs.seq_len)
    lab = fake_tokens(bs, cfg_bs.seq_len)
    fl = 3 * (cfg_bs.n_layer * layer_flops(bs, cfg_bs.seq_len, cfg_bs.n_embd,
              cfg_bs.n_head, cfg_bs.n_kv_head, cfg_bs.head_dim, cfg_bs.mlp_dim) +
              matmul_flops(bs * cfg_bs.seq_len, cfg_bs.vocab_size, cfg_bs.n_embd))

    def bench(p, tok, _lab=lab, _cfg=cfg_bs):
        def loss_fn(params):
            hidden = model_forward(_cfg, params, tok)
            return chunked_lm_head_loss(hidden, params['lm_head'], _lab, _cfg)
        return mx.value_and_grad(loss_fn)(p)

    r = benchmark(bench, p, tok, warmup=2, repeats=3, flop_count=fl, label=f"bs={bs}")
    r['tok_per_sec'] = bs * cfg_bs.seq_len / (r['wall_ms'] / 1000)
    results_7a.append(r)

print_summary(results_7a)
print("  Throughput:")
for r in results_7a:
    print(f"    {r['label']:<20s}  {r['tok_per_sec']:,.0f} tok/s")

# %%
# 7b. Sequence length sweep
print("\n=== Sequence length sweep ===")
results_7b = []

for sl in [512, 1024, 2048, 4096]:
    cfg_sl = PerfConfig(seq_len=sl)
    p = init_full_model(cfg_sl)
    tok = fake_tokens(cfg_sl.batch_size, sl)
    lab = fake_tokens(cfg_sl.batch_size, sl)
    fl = 3 * (cfg_sl.n_layer * layer_flops(cfg_sl.batch_size, sl, cfg_sl.n_embd,
              cfg_sl.n_head, cfg_sl.n_kv_head, cfg_sl.head_dim, cfg_sl.mlp_dim) +
              matmul_flops(cfg_sl.batch_size * sl, cfg_sl.vocab_size, cfg_sl.n_embd))

    def bench(p, tok, _lab=lab, _cfg=cfg_sl):
        def loss_fn(params):
            hidden = model_forward(_cfg, params, tok)
            return chunked_lm_head_loss(hidden, params['lm_head'], _lab, _cfg)
        return mx.value_and_grad(loss_fn)(p)

    r = benchmark(bench, p, tok, warmup=2, repeats=3, flop_count=fl, label=f"seq_len={sl}")
    r['tok_per_sec'] = cfg_sl.batch_size * sl / (r['wall_ms'] / 1000)
    results_7b.append(r)

print_summary(results_7b)
print("  Throughput:")
for r in results_7b:
    print(f"    {r['label']:<20s}  {r['tok_per_sec']:,.0f} tok/s")

# %%
# 7c. Head dim alignment: 64 (16 heads) vs 128 (8 heads) vs 256 (4 heads)
print("\n=== Head dim alignment ===")
results_7c = []

for hd, nh in [(64, 16), (128, 8), (256, 4)]:
    cfg_hd = PerfConfig(head_dim=hd, n_head=nh, n_kv_head=max(1, nh // 4),
                         n_embd=nh * hd)
    p = init_full_model(cfg_hd)
    tok = fake_tokens(cfg_hd.batch_size, cfg_hd.seq_len)
    lab = fake_tokens(cfg_hd.batch_size, cfg_hd.seq_len)
    fl = 3 * (cfg_hd.n_layer * layer_flops(cfg_hd.batch_size, cfg_hd.seq_len,
              cfg_hd.n_embd, cfg_hd.n_head, cfg_hd.n_kv_head,
              cfg_hd.head_dim, cfg_hd.mlp_dim) +
              matmul_flops(cfg_hd.batch_size * cfg_hd.seq_len,
                           cfg_hd.vocab_size, cfg_hd.n_embd))

    def bench(p, tok, _lab=lab, _cfg=cfg_hd):
        def loss_fn(params):
            hidden = model_forward(_cfg, params, tok)
            return chunked_lm_head_loss(hidden, params['lm_head'], _lab, _cfg)
        return mx.value_and_grad(loss_fn)(p)

    r = benchmark(bench, p, tok, warmup=2, repeats=3, flop_count=fl,
                  label=f"head_dim={hd}, n_head={nh}, E={nh*hd}")
    r['tok_per_sec'] = cfg_hd.batch_size * cfg_hd.seq_len / (r['wall_ms'] / 1000)
    results_7c.append(r)

print_summary(results_7c)

# %%
# 7d. GQA ratio comparison
print("\n=== GQA ratio ===")
results_7d = []

for n_kv in [1, 2, 4]:
    cfg_gqa = PerfConfig(n_kv_head=n_kv)
    p = init_full_model(cfg_gqa)
    tok = fake_tokens(cfg_gqa.batch_size, cfg_gqa.seq_len)
    lab = fake_tokens(cfg_gqa.batch_size, cfg_gqa.seq_len)
    fl = 3 * (cfg_gqa.n_layer * layer_flops(cfg_gqa.batch_size, cfg_gqa.seq_len,
              cfg_gqa.n_embd, cfg_gqa.n_head, n_kv,
              cfg_gqa.head_dim, cfg_gqa.mlp_dim) +
              matmul_flops(cfg_gqa.batch_size * cfg_gqa.seq_len,
                           cfg_gqa.vocab_size, cfg_gqa.n_embd))

    def bench(p, tok, _lab=lab, _cfg=cfg_gqa):
        def loss_fn(params):
            hidden = model_forward(_cfg, params, tok)
            return chunked_lm_head_loss(hidden, params['lm_head'], _lab, _cfg)
        return mx.value_and_grad(loss_fn)(p)

    r = benchmark(bench, p, tok, warmup=2, repeats=3, flop_count=fl,
                  label=f"n_kv_head={n_kv} (ratio {cfg.n_head}:{n_kv})")
    r['tok_per_sec'] = cfg_gqa.batch_size * cfg_gqa.seq_len / (r['wall_ms'] / 1000)
    results_7d.append(r)

print_summary(results_7d)

# %%
# 7e. Attention implementation comparison (single layer)
print("\n=== Attention implementation comparison (single layer) ===")
results_7e = []
x = fake_hidden(cfg.batch_size, cfg.seq_len, cfg.n_embd)
layer_p = init_layer_params(cfg)

for impl in ['einsum', 'sdpa']:
    def bench_fn(x, layer, _impl=impl):
        return single_layer_forward(cfg, layer, x, attn_impl=_impl)

    r = benchmark(bench_fn, x, layer_p,
                  flop_count=lf, label=f"attn_impl={impl}")
    results_7e.append(r)

print_summary(results_7e)

# %% [markdown]
# ### Ideas to try
# - **Combined batch_size × seq_len grid** — find the throughput-maximizing combo
# - **`n_embd` sweep**: 512, 768, 1024, 1536, 2048
# - **`mlp_dim` expansion ratio**: 2x, 3x, 4x
# - **`n_layer` vs `n_embd`**: given a fixed param budget, is it better to go deep or wide?
# - **`mx.compile`** on the full training step

# %% [markdown]
# ## Phase 8 — Advanced Optimization Ideas
#
# Concepts for further experimentation on Apple Silicon.

# %% [markdown]
# ### 8.1 Custom Metal Kernels via `mx.fast.metal_kernel()`
#
# MLX allows writing custom Metal shaders that run directly on the GPU.
# A fused RMSNorm + Linear kernel would save one memory roundtrip per layer:
#
# ```python
# source = """
# // Metal shader for fused RMSNorm + Linear
# // Read x from device memory once, normalize in registers, multiply by W
# uint elem = thread_position_in_grid.x;
# float x_val = x[elem];
# float rms = ...; // compute RMS across the row
# out[elem] = (x_val / rms) * w[elem % D];
# """
# kernel = mx.fast.metal_kernel(
#     name="fused_norm_linear",
#     input_names=["x", "w"],
#     output_names=["out"],
#     source=source,
# )
# ```
#
# **Expected benefit**: Saves ~2 × B × T × E × 2 bytes per layer per call.
# At 24 layers with 4 norm+linear pairs each, that's significant bandwidth saved.

# %% [markdown]
# ### 8.2 Quantization: `mx.quantize` for int4/int8
#
# MLX has built-in quantization support for inference and QLoRA-style training:
#
# ```python
# # Quantize a weight matrix to 4-bit
# w_quant, scales, biases = mx.quantize(weight, bits=4, group_size=64)
#
# # Dequantize for matmul (or use mx.dequantize in the forward pass)
# w_approx = mx.dequantize(w_quant, scales, biases, bits=4, group_size=64)
# out = x @ w_approx
# ```
#
# **Key insight**: On M4 Pro (bandwidth-limited), quantized weights reduce memory
# traffic by 4-8x, which can more than compensate for any dequantization overhead.

# %% [markdown]
# ### 8.3 `mx.compile` Optimization Tips
#
# `mx.compile` fuses operations into optimized Metal compute graphs:
#
# ```python
# @mx.compile
# def optimized_layer(params, x):
#     return single_layer_forward(cfg, params, x)
#
# # Or compile the full training step:
# @mx.compile
# def train_step(params, tokens, labels):
#     def loss_fn(p):
#         hidden = model_forward(cfg, p, tokens)
#         return chunked_lm_head_loss(hidden, p['lm_head'], labels, cfg)
#     return mx.value_and_grad(loss_fn)(params)
# ```
#
# **Caveats**: `mx.compile` requires all shapes to be static (no dynamic control flow).
# The first call traces and compiles; subsequent calls reuse the compiled graph.

# %% [markdown]
# ### 8.4 Gradient Accumulation for Large Effective Batch
#
# With 48 GB unified memory, we can fit larger batches than TPU's 32 GB, but
# for very large effective batch sizes, gradient accumulation is still useful:
#
# ```python
# micro_bs = cfg.batch_size // 4
# grad_acc = None
# for i in range(4):
#     mb_tokens = tokens[i*micro_bs : (i+1)*micro_bs]
#     mb_labels = labels[i*micro_bs : (i+1)*micro_bs]
#     _, g = mx.value_and_grad(loss_fn)(params, mb_tokens, mb_labels)
#     mx.eval(g)  # eval each micro-batch to free activations
#     if grad_acc is None:
#         grad_acc = g
#     else:
#         grad_acc = {k: grad_acc[k] + g[k] for k in g}
# # Average gradients
# grads = {k: v / 4 for k, v in grad_acc.items()}
# ```

# %% [markdown]
# ### 8.5 Mixed Precision (bf16 compute, fp32 optimizer)
#
# The model keeps all params and activations in **bfloat16** (same as TPU),
# with selective upcasting to float32 for numerically sensitive ops:
#
# - **Currently f32**: logit softcap (`tanh`), cross-entropy loss
# - **Worth trying in f32**: attention scores (before softmax), optimizer moments
# - **Keep bf16**: all matmuls (projections, MLP), RMSNorm, RoPE
#
# M4 Pro runs bf16 at the same speed as fp16 (~2x fp32), and bf16 has
# better dynamic range (same exponent as fp32), making it the preferred
# training dtype — matching the TPU workflow.

# %% [markdown]
# ### 8.6 Memory Layout — Row-Major, `(B, H, T, D)` for Attention
#
# MLX uses row-major (C-contiguous) layout by default.
# For attention, the `(B, H, T, D)` layout is optimal because:
# - `mx.fast.scaled_dot_product_attention` expects this layout
# - The T dimension is contiguous with D for good locality
# - Einsum `'bhtd,bhsd->bhts'` has good access patterns
#
# Our codebase produces Q/K/V in `(B, H, T, D)` directly via
# `einsum('btd,dhk->bhtk', ...)` — the einsum fuses the reshape, avoiding
# explicit transposes.

# %% [markdown]
# ### 8.7 Metal GPU Capture for Profiling
#
# MLX supports Metal frame capture for Xcode Instruments analysis:
#
# ```python
# mx.metal.start_capture("trace.gputrace")
# for _ in range(5):
#     out = bench_fn(*args)
#     mx.eval(out)
# mx.metal.stop_capture()
# # Open trace.gputrace in Xcode Instruments → Metal System Trace
# ```
#
# This shows:
# - GPU shader utilization and occupancy
# - Memory bandwidth utilization
# - Kernel launch overhead
# - Buffer allocation patterns

# %% [markdown]
# ### 8.8 Unified Memory Advantage
#
# Apple Silicon's unified memory architecture means:
# - **No host-device copies** — CPU and GPU share the same physical memory
# - **No PCIe bottleneck** — data is already "on device"
# - **Larger effective memory** — 48 GB for both model params and activations
#
# This means data loading pipelines don't need prefetching or double-buffering
# for host→device transfer (the main bottleneck on discrete GPUs and TPUs).
# The bottleneck shifts to disk I/O and tokenization throughput.

# %% [markdown]
# ## Phase 9 — `mx.compile` and MFU Measurement
#
# Key optimization for bandwidth-limited M4 Pro: `mx.compile` fuses operations
# into optimized Metal compute graphs, eliminating intermediate memory roundtrips.
#
# **MFU formula** (matching 08_tpu_ablations):
# - `step_flops = 3 * fwd_flops` (forward + 2x backward)
# - `MFU% = step_flops / (PEAK_TFLOPS * 1e12 * step_time_s) * 100`

# %%
# === Phase 9: mx.compile + MFU benchmarks ===
print("=" * 80)
print("  Phase 9 — mx.compile + MFU Measurement")
print("=" * 80)

# FLOP counting helpers (matching 08_tpu_ablations)
def compute_layer_flops(B, T, D, N, K, H, F):
    """GLU MLP: 3 matmuls (gate + up + down)."""
    tok = B * T
    q    = 2 * tok * D * N * H
    k    = 2 * tok * D * K * H
    v    = 2 * tok * D * K * H
    att  = attention_flops(B, N, T, H)
    proj = 2 * tok * N * H * D
    mlp  = 3 * (2 * tok * D * F)
    return q + k + v + att + proj + mlp

def compute_mfu(step_time_s, step_flops):
    """Compute MFU% from step time and total step FLOPs."""
    return (step_flops / (PEAK_TFLOPS * 1e12 * step_time_s)) * 100

# %%
# 9a. Compiled vs non-compiled forward+backward MFU
print("\n=== 9a. mx.compile impact on fwd+bwd MFU ===")
print(f"Config: E={cfg.n_embd}, L={cfg.n_layer}, B={cfg.batch_size}, T={cfg.seq_len}")

# Compute FLOPs for this config
lf = compute_layer_flops(cfg.batch_size, cfg.seq_len, cfg.n_embd,
                          cfg.n_head, cfg.n_kv_head, cfg.head_dim, cfg.mlp_dim)
lm_flops = matmul_flops(cfg.batch_size * cfg.seq_len, cfg.vocab_size, cfg.n_embd)
fwd_flops = cfg.n_layer * lf + lm_flops
step_flops = 3 * fwd_flops  # fwd + 2x bwd
flops_per_tok = step_flops / (cfg.batch_size * cfg.seq_len)

print(f"Forward FLOPs: {fwd_flops/1e12:.3f} TFLOP")
print(f"Step FLOPs (3x fwd): {step_flops/1e12:.3f} TFLOP")
print(f"FLOPs/token: {flops_per_tok/1e6:.1f}M")
ideal_tok_s = PEAK_TFLOPS * 1e12 / flops_per_tok
print(f"Ideal tok/s (100% MFU): {int(ideal_tok_s):,}")

tok = fake_tokens(cfg.batch_size, cfg.seq_len + 1)
labels = tok[:, 1:]
tok = tok[:, :-1]
p = init_full_model(cfg)
mx.eval(p)

# Non-compiled fwd+bwd
def fwd_bwd_no_compile(params, tokens, labels):
    def loss_fn(p):
        hidden = model_forward(cfg, p, tokens)
        return chunked_lm_head_loss(hidden, p['lm_head'], labels, cfg)
    return mx.value_and_grad(loss_fn)(params)

# Warmup
for _ in range(2):
    loss, grads = fwd_bwd_no_compile(p, tok, labels)
    mx.eval(loss, grads)

times_no_compile = []
for _ in range(5):
    t0 = time.time()
    loss, grads = fwd_bwd_no_compile(p, tok, labels)
    mx.eval(loss, grads)
    times_no_compile.append(time.time() - t0)
avg_no_compile = sum(times_no_compile) / len(times_no_compile)
tps_no_compile = int(cfg.batch_size * cfg.seq_len / avg_no_compile)
mfu_no_compile = compute_mfu(avg_no_compile, step_flops)

# Compiled fwd+bwd
@mx.compile
def fwd_bwd_compiled(params, tokens, labels):
    def loss_fn(p):
        hidden = model_forward(cfg, p, tokens)
        return chunked_lm_head_loss(hidden, p['lm_head'], labels, cfg)
    return mx.value_and_grad(loss_fn)(params)

for _ in range(2):
    loss, grads = fwd_bwd_compiled(p, tok, labels)
    mx.eval(loss, grads)

times_compiled = []
for _ in range(5):
    t0 = time.time()
    loss, grads = fwd_bwd_compiled(p, tok, labels)
    mx.eval(loss, grads)
    times_compiled.append(time.time() - t0)
avg_compiled = sum(times_compiled) / len(times_compiled)
tps_compiled = int(cfg.batch_size * cfg.seq_len / avg_compiled)
mfu_compiled = compute_mfu(avg_compiled, step_flops)

print(f"  No compile:   {avg_no_compile*1000:8.1f}ms  {tps_no_compile:>8,} tok/s  MFU: {mfu_no_compile:.1f}%")
print(f"  Compiled:     {avg_compiled*1000:8.1f}ms  {tps_compiled:>8,} tok/s  MFU: {mfu_compiled:.1f}%")
print(f"  Speedup: {avg_no_compile/avg_compiled:.2f}x")

# %%
# 9b. Batch size sweep with mx.compile
print("\n=== 9b. Batch size sweep (compiled, E={}, L={}) ===".format(cfg.n_embd, cfg.n_layer))

results_9b = []
for B in [4, 8, 16, 32]:
    _cfg = PerfConfig(batch_size=B, n_embd=cfg.n_embd, n_head=cfg.n_head,
                      n_kv_head=cfg.n_kv_head, head_dim=cfg.head_dim,
                      mlp_dim=cfg.mlp_dim, n_layer=cfg.n_layer,
                      seq_len=cfg.seq_len)
    _lf = compute_layer_flops(B, _cfg.seq_len, _cfg.n_embd,
                               _cfg.n_head, _cfg.n_kv_head, _cfg.head_dim, _cfg.mlp_dim)
    _lm = matmul_flops(B * _cfg.seq_len, _cfg.vocab_size, _cfg.n_embd)
    _step_flops = 3 * (_cfg.n_layer * _lf + _lm)

    _tok = fake_tokens(B, _cfg.seq_len + 1)
    _lab = _tok[:, 1:]
    _tok = _tok[:, :-1]
    _p = init_full_model(_cfg)
    mx.eval(_p)

    @mx.compile
    def _fwd_bwd(_params, _tokens, _labels):
        def loss_fn(p):
            hidden = model_forward(_cfg, p, _tokens)
            return chunked_lm_head_loss(hidden, p['lm_head'], _labels, _cfg)
        return mx.value_and_grad(loss_fn)(_params)

    for _ in range(2):
        _loss, _grads = _fwd_bwd(_p, _tok, _lab)
        mx.eval(_loss, _grads)

    times = []
    for _ in range(5):
        t0 = time.time()
        _loss, _grads = _fwd_bwd(_p, _tok, _lab)
        mx.eval(_loss, _grads)
        times.append(time.time() - t0)

    avg_t = sum(times) / len(times)
    tps = int(B * _cfg.seq_len / avg_t)
    mfu = compute_mfu(avg_t, _step_flops)
    results_9b.append({'B': B, 'ms': avg_t*1000, 'tok_s': tps, 'mfu': mfu})
    print(f"  B={B:>2}:  {avg_t*1000:8.1f}ms  {tps:>8,} tok/s  MFU: {mfu:.1f}%")

# %%
# 9c. Model width sweep with mx.compile
print("\n=== 9c. Model width sweep (compiled, B={}, L={}) ===".format(cfg.batch_size, cfg.n_layer))

results_9c = []
for E in [512, 768, 1024, 1536]:
    H = E // cfg.head_dim
    MLP = int(E * 8 / 3 / 64) * 64  # SwiGLU standard ratio, rounded to 64
    _cfg = PerfConfig(batch_size=cfg.batch_size, n_embd=E, n_head=H,
                      n_kv_head=1, head_dim=cfg.head_dim,
                      mlp_dim=MLP, n_layer=cfg.n_layer,
                      seq_len=cfg.seq_len)

    _lf = compute_layer_flops(_cfg.batch_size, _cfg.seq_len, E,
                               H, 1, cfg.head_dim, MLP)
    _lm = matmul_flops(_cfg.batch_size * _cfg.seq_len, _cfg.vocab_size, E)
    _step_flops = 3 * (_cfg.n_layer * _lf + _lm)

    _tok = fake_tokens(_cfg.batch_size, _cfg.seq_len + 1)
    _lab = _tok[:, 1:]
    _tok = _tok[:, :-1]
    _p = init_full_model(_cfg)
    mx.eval(_p)

    @mx.compile
    def _fwd_bwd(_params, _tokens, _labels):
        def loss_fn(p):
            hidden = model_forward(_cfg, p, _tokens)
            return chunked_lm_head_loss(hidden, p['lm_head'], _labels, _cfg)
        return mx.value_and_grad(loss_fn)(_params)

    for _ in range(2):
        _loss, _grads = _fwd_bwd(_p, _tok, _lab)
        mx.eval(_loss, _grads)

    times = []
    for _ in range(5):
        t0 = time.time()
        _loss, _grads = _fwd_bwd(_p, _tok, _lab)
        mx.eval(_loss, _grads)
        times.append(time.time() - t0)

    avg_t = sum(times) / len(times)
    tps = int(_cfg.batch_size * _cfg.seq_len / avg_t)
    mfu = compute_mfu(avg_t, _step_flops)
    results_9c.append({'E': E, 'H': H, 'MLP': MLP, 'ms': avg_t*1000, 'tok_s': tps, 'mfu': mfu})
    print(f"  E={E:>4}, H={H}, MLP={MLP:>4}:  {avg_t*1000:8.1f}ms  {tps:>8,} tok/s  MFU: {mfu:.1f}%")

# %%
# 9d. Chunking ablation with mx.compile
print("\n=== 9d. Chunk sweep (compiled, E={}, L={}, B={}) ===".format(
    cfg.n_embd, cfg.n_layer, cfg.batch_size))

results_9d = []
for chunks in [1, 2, 4, 8]:
    _cfg = PerfConfig(batch_size=cfg.batch_size, n_embd=cfg.n_embd, n_head=cfg.n_head,
                      n_kv_head=cfg.n_kv_head, head_dim=cfg.head_dim,
                      mlp_dim=cfg.mlp_dim, n_layer=cfg.n_layer,
                      seq_len=cfg.seq_len, num_lm_head_chunks=chunks)

    _tok = fake_tokens(_cfg.batch_size, _cfg.seq_len + 1)
    _lab = _tok[:, 1:]
    _tok = _tok[:, :-1]
    _p = init_full_model(_cfg)
    mx.eval(_p)

    @mx.compile
    def _fwd_bwd(_params, _tokens, _labels):
        def loss_fn(p):
            hidden = model_forward(_cfg, p, _tokens)
            return chunked_lm_head_loss(hidden, p['lm_head'], _labels, _cfg)
        return mx.value_and_grad(loss_fn)(_params)

    for _ in range(2):
        _loss, _grads = _fwd_bwd(_p, _tok, _lab)
        mx.eval(_loss, _grads)

    times = []
    for _ in range(5):
        t0 = time.time()
        _loss, _grads = _fwd_bwd(_p, _tok, _lab)
        mx.eval(_loss, _grads)
        times.append(time.time() - t0)

    avg_t = sum(times) / len(times)
    tps = int(_cfg.batch_size * _cfg.seq_len / avg_t)
    mfu = compute_mfu(avg_t, step_flops)  # same model, only chunking differs
    results_9d.append({'chunks': chunks, 'ms': avg_t*1000, 'tok_s': tps, 'mfu': mfu})
    print(f"  chunks={chunks}:  {avg_t*1000:8.1f}ms  {tps:>8,} tok/s  MFU: {mfu:.1f}%")

# %%
# === Final summary ===
print("=" * 90)
print("  FULL SESSION SUMMARY")
print("=" * 90)

all_results = []
for name, rlist in [
    ("Phase 1 — Matmul", results_1a[-2:]),
    ("Phase 2 — Components", [r_mlp, r_attn_sdpa]),
    ("Phase 3 — Single layer", [r_layer]),
    ("Phase 4 — Depth", results_4a[-1:]),
    ("Phase 5 — LM head", [r_lm, r_lm_chunked]),
    ("Phase 6 — Fwd/Bwd", [r_fwd, r_fwd_bwd, r_remat]),
]:
    for r in rlist:
        all_results.append(r)

print_summary(all_results)

print("\n--- Phase 9 — mx.compile + MFU ---")
print(f"  {'Config':<35s}  {'ms':>8s}  {'tok/s':>10s}  {'MFU%':>6s}")
print(f"  {'─'*35}  {'─'*8}  {'─'*10}  {'─'*6}")
print(f"  {'fwd+bwd (no compile)':<35s}  {avg_no_compile*1000:8.1f}  {tps_no_compile:>10,}  {mfu_no_compile:5.1f}%")
print(f"  {'fwd+bwd (compiled)':<35s}  {avg_compiled*1000:8.1f}  {tps_compiled:>10,}  {mfu_compiled:5.1f}%")
for r in results_9b:
    print(f"  {'B=' + str(r['B']) + ' (compiled)':<35s}  {r['ms']:8.1f}  {r['tok_s']:>10,}  {r['mfu']:5.1f}%")
for r in results_9c:
    print(f"  {'E=' + str(r['E']) + ' (compiled)':<35s}  {r['ms']:8.1f}  {r['tok_s']:>10,}  {r['mfu']:5.1f}%")
for r in results_9d:
    print(f"  {'chunks=' + str(r['chunks']) + ' (compiled)':<35s}  {r['ms']:8.1f}  {r['tok_s']:>10,}  {r['mfu']:5.1f}%")

print("\nDone!  Use Metal GPU Capture (Phase 8.7) for detailed hardware traces.")
