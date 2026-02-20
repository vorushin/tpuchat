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
# # 05 — TPU v6e Performance Lab
#
# Standalone notebook that progressively builds a modern transformer and
# benchmarks **MFU** (Model FLOPs Utilization) and **HBM usage** at each stage.
#
# - **No data loading, no tokenizer, no HuggingFace** — pure fake data
# - Every phase is independently runnable after Phase 0
# - All matrix dims default to 256-aligned (optimal for v6e MXU)
#
# ### TPU v6e-1 specs
# | Spec | Value |
# |------|-------|
# | HBM | 32 GB |
# | MXU | 256×256 systolic array (bfloat16) |
# | Peak bf16 TFLOPS | 918 |
# | HBM capacity | 32 GB |
# | HBM bandwidth | 1600 GB/s |
# | Arithmetic intensity | 918e12 / 1600e9 ≈ 574 FLOPs/byte |
#
# > **MFU%** (Model FLOPs Utilization): `analytical_matmul_FLOPs / (peak_TFLOPS × wall_time)`.
# > We count every matmul individually — Q/K/V projections, attention (QK^T + AV),
# > output projection, SwiGLU gate/up/down, lm_head — then multiply by 3× for
# > fwd+bwd. This is more accurate than the common `6·N·B·T` shorthand which
# > misses attention FLOPs and doesn't reflect GQA savings. **MXU%** in this
# > notebook refers to XProf hardware measurements (not computed here).
#
# > **HBM BW%:** shows what fraction of the 1600 GB/s peak bandwidth
# > is utilized, computed from (bytes read+written) / wall_time.

# %%
# !pip install -q "jax[tpu]" optax

# %% [markdown]
# ## Phase 0 — Setup & Utilities

# %%
import functools as ft
import time
import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax

# TPU v6e-1 constants (from https://docs.cloud.google.com/tpu/docs/v6e)
PEAK_TFLOPS = 918          # bf16 peak compute per chip
HBM_GB = 32
HBM_BW_GBS = 1600         # HBM bandwidth in GB/s
MXU_DIM = 256              # 256×256 systolic array

ALL_RESULTS = []   # global collector — every benchmark() appends here

print(f"JAX version : {jax.__version__}")
print(f"Devices     : {jax.devices()}")
print(f"Peak TFLOPS : {PEAK_TFLOPS} (bf16, from v6e docs)")

# %%
# === dot_dict: JAX-compatible mutable dictionary ===

@jax.tree_util.register_pytree_with_keys_class
class dot_dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__

    def tree_flatten_with_keys(self):
        keys = tuple(sorted(self))
        return tuple((jax.tree_util.DictKey(k), self[k]) for k in keys), keys

    @classmethod
    def tree_unflatten(cls, keys, values):
        return cls(zip(keys, values))

# %%
# === Benchmark harness ===

def benchmark(fn, *args, warmup=3, repeats=10, flop_count=None,
              hbm_bytes=None, label=""):
    """Run fn repeatedly and report wall time, TFLOP/s, MFU%, HBM bandwidth%.

    Args:
        fn: callable (JIT-compiled or not — warmup handles compilation)
        *args: arguments forwarded to fn
        warmup: number of warmup calls (absorbs JIT compilation)
        repeats: number of timed calls
        flop_count: manual FLOP count (int); None = skip MFU calculation
        hbm_bytes: total bytes read+written per call (int); None = skip BW calc
        label: display label for printing

    Returns:
        dict with wall_ms, tflops, mfu_pct, hbm_bw_gbs, hbm_bw_pct
    """
    # Warmup (triggers JIT + XLA compilation)
    for _ in range(warmup):
        out = fn(*args)
        jax.block_until_ready(out)

    # Timed runs
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)

    wall_s = sum(times) / len(times)
    wall_ms = wall_s * 1000

    # FLOP/s and MFU
    tflops = flop_count / (wall_s * 1e12) if flop_count else 0.0
    mfu_pct = (tflops / PEAK_TFLOPS * 100
               if flop_count and PEAK_TFLOPS else 0.0)

    # HBM bandwidth utilization
    hbm_bw_gbs = hbm_bytes / (wall_s * 1e9) if hbm_bytes else 0.0
    hbm_bw_pct = hbm_bw_gbs / HBM_BW_GBS * 100 if hbm_bytes else 0.0

    result = dict(label=label, wall_ms=wall_ms, tflops=tflops,
                  mfu_pct=mfu_pct, hbm_bw_gbs=hbm_bw_gbs,
                  hbm_bw_pct=hbm_bw_pct)
    ALL_RESULTS.append(result)

    # Print single row
    mfu_str = f"{mfu_pct:5.1f}%" if flop_count else "  n/a"
    tflop_str = f"{tflops:6.1f}" if flop_count else "   n/a"
    bw_str = f"{hbm_bw_pct:5.1f}%" if hbm_bytes else "  n/a"
    print(f"  {label:<40s}  {wall_ms:8.2f} ms  {tflop_str} TFLOP/s  MFU {mfu_str}  "
          f"HBM BW {bw_str}")
    return result


def print_summary(results):
    """Print a formatted comparison table from benchmark results."""
    print(f"\n  {'Label':<40s}  {'Wall ms':>8s}  {'TFLOP/s':>8s}  {'MFU %':>6s}  "
          f"{'HBM BW%':>7s}")
    print("  " + "-" * 88)
    for r in results:
        mxu = f"{r['mfu_pct']:5.1f}%" if r['tflops'] > 0 else "  n/a"
        tf = f"{r['tflops']:7.1f}" if r['tflops'] > 0 else "    n/a"
        bw = f"{r['hbm_bw_pct']:5.1f}%" if r['hbm_bw_pct'] > 0 else "  n/a"
        print(f"  {r['label']:<40s}  {r['wall_ms']:8.2f}  {tf}  {mxu}  "
              f"{bw:>7s}")
    print()

# %%
# === Fake data generators ===

def fake_tokens(batch_size, seq_len, vocab_size=32768, seed=0):
    return jax.random.randint(jax.random.key(seed),
                              (batch_size, seq_len), 0, vocab_size, dtype=jnp.int32)

def fake_hidden(batch_size, seq_len, n_embd, seed=0):
    return jax.random.normal(jax.random.key(seed),
                             (batch_size, seq_len, n_embd), dtype=jnp.bfloat16)

# %%
# === FLOP counting helpers ===
# Dimension notation follows "How to Scale Your Model" (jax-ml/scaling-book):
#   B=batch, T=seq_len, D=d_model, N=n_heads, K=n_kv_heads,
#   H=head_dim, F=d_ff, L=n_layers, V=vocab_size

def matmul_flops(M, N, K, batch=1):
    """FLOPs for [M,K] @ [K,N].  2*M*N*K per batch element."""
    return 2 * batch * M * N * K

def attention_flops(B, N, T, H):
    """FLOPs for QK^T + AV (full T×T, not causal-halved).

    Counts full attention matrix. Causal kernels (e.g. splash) skip the
    upper triangle, so actual MXU work is ~half this — meaning MFU% for
    attention is overestimated by ~2x.
    """
    return 2 * (2 * B * N * T * T * H)   # QK^T + AV

def layer_flops(B, T, D, N, K, H, F):
    """MXU-relevant FLOPs for one transformer layer.

    Counts only matmul FLOPs (projections + attention core + MLP).
    Excludes elementwise ops (RMSNorm, RoPE, softmax, SiLU) which
    run on the vector unit, not the MXU.
    """
    tok = B * T
    q  = 2 * tok * D * N * H             # Q projection
    k  = 2 * tok * D * K * H             # K projection
    v  = 2 * tok * D * K * H             # V projection
    att = attention_flops(B, N, T, H)     # core attention
    proj = 2 * tok * N * H * D           # output projection
    gate = 2 * tok * D * F               # SwiGLU gate
    up   = 2 * tok * D * F               # SwiGLU up
    down = 2 * tok * F * D               # SwiGLU down
    return q + k + v + att + proj + gate + up + down

# %% [markdown]
# ## Phase 1 — Matmul Baseline
#
# Establish the MFU ceiling with pure matmuls at 256-aligned sizes.

# %%
# 1a. Square matmul — aligned sizes
print("=== Square matmul (256-aligned) ===")
results_1a = []
for size in [128, 256, 512, 1024, 2048, 4096, 8192]:
    a = jax.random.normal(jax.random.key(0), (size, size), dtype=jnp.bfloat16)
    b = jax.random.normal(jax.random.key(1), (size, size), dtype=jnp.bfloat16)

    @jax.jit
    def mm(a, b):
        return a @ b

    # HBM bytes: read A + read B + write C, all bf16 (2 bytes each)
    hbm = 3 * size * size * 2
    r = benchmark(mm, a, b, flop_count=matmul_flops(size, size, size),
                  hbm_bytes=hbm, label=f"matmul {size}x{size}")
    results_1a.append(r)

print_summary(results_1a)

# %%
# 1c. Batched matmul — simulating transformer projections
print("=== Batched matmul (transformer-shaped) ===")
results_1c = []
shapes = [
    # (B, M, K, N) -> (B*M, K) @ (K, N)
    (8, 2048, 1024, 1024, "B=8 hidden->hidden"),
    (8, 2048, 1024, 3072, "B=8 hidden->mlp"),
    (8, 2048, 3072, 1024, "B=8 mlp->hidden"),
    (8, 2048, 1024, 32768, "B=8 hidden->vocab"),
]
for B, M, K, N, desc in shapes:
    a = jax.random.normal(jax.random.key(0), (B * M, K), dtype=jnp.bfloat16)
    b = jax.random.normal(jax.random.key(1), (K, N), dtype=jnp.bfloat16)

    @jax.jit
    def mm(a, b):
        return a @ b

    # read A (B*M×K) + read B (K×N) + write C (B*M×N), all bf16
    hbm = (B * M * K + K * N + B * M * N) * 2
    r = benchmark(mm, a, b, flop_count=matmul_flops(B * M, N, K),
                  hbm_bytes=hbm, label=desc)
    results_1c.append(r)
print_summary(results_1c)

# %% [markdown]
# ### Ideas to try
# - **float32 vs bfloat16**: float32 matmul should be ~2x slower (MXU does bf16 natively)
# - **int8 matmul**: `jax.lax.dot_general` with `preferred_element_type=jnp.int32` — TPU v6e int8 = 2x bf16 throughput
# - **Rectangular aspect ratios**: tall-skinny (e.g. 16384×256 @ 256×256) vs short-wide
# - **`jnp.matmul` vs `jnp.einsum` vs `jax.lax.dot_general`**: should be identical after XLA compilation

# %% [markdown]
# ## Phase 2 — Individual Transformer Components
#
# Isolate each building block and measure MFU independently.
# - **RMSNorm / RoPE**: memory-bound (expect ~0% MFU)
# - **MLP**: compute-heavy (3 large matmuls)
# - **Attention**: mixed (projections = compute, softmax = memory)

# %%
# === Model primitives (from 04_maxtext.py) ===

def rms_norm(x):
    """RMSNorm with no learnable parameters."""
    return x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + 1e-6)


def precompute_rope(seq_len, head_dim, base=10000):
    """Precompute rotary embedding cos/sin tables."""
    channel_range = jnp.arange(0, head_dim, 2, dtype=jnp.float32)
    inv_freq = 1.0 / (base ** (channel_range / head_dim))
    t = jnp.arange(seq_len, dtype=jnp.float32)
    freqs = jnp.outer(t, inv_freq)
    cos = jnp.cos(freqs).astype(jnp.bfloat16)
    sin = jnp.sin(freqs).astype(jnp.bfloat16)
    return cos, sin


def apply_rope(x, cos, sin):
    """Apply rotary embeddings. x: (B, H, T, D), cos/sin: (1, 1, T, D/2)"""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return jnp.concatenate([y1, y2], axis=-1)


def _expand_kv(k, v, n_head, n_kv_head):
    """Repeat KV heads to match Q head count for non-splash backends."""
    if n_kv_head == n_head:
        return k, v
    ratio = n_head // n_kv_head
    return jnp.repeat(k, ratio, axis=1), jnp.repeat(v, ratio, axis=1)

# %%
# === PerfConfig ===

@jax.tree_util.register_static
@dataclass(kw_only=True, frozen=True)
class PerfConfig:
    """All dims 256-aligned for MXU.  Matches 04_maxtext.py defaults."""
    batch_size: int = 8
    seq_len: int = 2048
    n_head: int = 4
    n_kv_head: int = 2
    head_dim: int = 256
    n_embd: int = 1024       # n_head * head_dim
    mlp_dim: int = 3072      # 3x expansion for SwiGLU
    vocab_size: int = 32768
    n_layer: int = 24
    softcap: float = 15.0
    splash_block_size: int = 1024
    num_lm_head_chunks: int = 8

    @property
    def padded_vocab(self):
        return ((self.vocab_size + 63) // 64) * 64

cfg = PerfConfig()
assert cfg.n_embd == cfg.n_head * cfg.head_dim, \
    f'n_embd ({cfg.n_embd}) must equal n_head * head_dim ({cfg.n_head * cfg.head_dim})'
print(f"Config: B={cfg.batch_size}, T={cfg.seq_len}, D={cfg.n_embd}, "
      f"N={cfg.n_head}, K={cfg.n_kv_head}, H={cfg.head_dim}, "
      f"F={cfg.mlp_dim}, V={cfg.vocab_size}, L={cfg.n_layer}")

# %%
# 2a. RMSNorm — pure elementwise, expect ~0% MFU
print("=== RMSNorm ===")
x = fake_hidden(cfg.batch_size, cfg.seq_len, cfg.n_embd)

@jax.jit
def bench_rmsnorm(x):
    return rms_norm(x)

r_norm = benchmark(bench_rmsnorm, x, flop_count=None, label="RMSNorm")

# %%
# 2b. RoPE — elementwise multiply/concat, expect ~0% MFU
print("=== RoPE ===")
cos, sin = precompute_rope(cfg.seq_len, cfg.head_dim)
cos_b = cos[None, None, :, :]
sin_b = sin[None, None, :, :]

q = jax.random.normal(jax.random.key(0),
    (cfg.batch_size, cfg.n_head, cfg.seq_len, cfg.head_dim), dtype=jnp.bfloat16)

@jax.jit
def bench_rope(q, cos, sin):
    return apply_rope(q, cos, sin)

r_rope = benchmark(bench_rope, q, cos_b, sin_b, flop_count=None, label="RoPE")

# %%
# 2c. SwiGLU MLP — 3 large matmuls (gate, up, down)
print("=== SwiGLU MLP ===")

def init_mlp_params(cfg, seed=42):
    key = jax.random.key(seed)
    k1, k2, k3 = jax.random.split(key, 3)
    s = (3.0 ** 0.5) * (cfg.n_embd ** -0.5)
    return dot_dict(
        w_gate=jax.random.uniform(k1, (cfg.n_embd, cfg.mlp_dim),
                                   dtype=jnp.bfloat16, minval=-s, maxval=s),
        w_up=jax.random.uniform(k2, (cfg.n_embd, cfg.mlp_dim),
                                 dtype=jnp.bfloat16, minval=-s, maxval=s),
        w_down=jnp.zeros((cfg.mlp_dim, cfg.n_embd), dtype=jnp.bfloat16),
    )

mlp_params = init_mlp_params(cfg)
x = fake_hidden(cfg.batch_size, cfg.seq_len, cfg.n_embd)

@jax.jit
def bench_mlp(x, params):
    with jax.named_scope('mlp'):
        h = rms_norm(x)
        gate = jax.nn.silu(jnp.einsum('btd,dh->bth', h, params.w_gate))
        up = jnp.einsum('btd,dh->bth', h, params.w_up)
        return jnp.einsum('bth,hd->btd', gate * up, params.w_down)

tok = cfg.batch_size * cfg.seq_len
mlp_flops = 3 * 2 * tok * cfg.n_embd * cfg.mlp_dim
r_mlp = benchmark(bench_mlp, x, mlp_params, flop_count=mlp_flops, label="SwiGLU MLP")

# %%
# 2d. Attention — einsum variant (manual QK^T + softmax + AV)
print("=== Attention (einsum) ===")

def init_attn_params(cfg, seed=42):
    key = jax.random.key(seed)
    keys = jax.random.split(key, 4)
    s = (3.0 ** 0.5) * (cfg.n_embd ** -0.5)
    return dot_dict(
        c_q=jax.random.uniform(keys[0], (cfg.n_embd, cfg.n_head, cfg.head_dim),
                                dtype=jnp.bfloat16, minval=-s, maxval=s),
        c_k=jax.random.uniform(keys[1], (cfg.n_embd, cfg.n_kv_head, cfg.head_dim),
                                dtype=jnp.bfloat16, minval=-s, maxval=s),
        c_v=jax.random.uniform(keys[2], (cfg.n_embd, cfg.n_kv_head, cfg.head_dim),
                                dtype=jnp.bfloat16, minval=-s, maxval=s),
        c_proj=jnp.zeros((cfg.n_head, cfg.head_dim, cfg.n_embd), dtype=jnp.bfloat16),
    )

attn_params = init_attn_params(cfg)

@jax.jit
def bench_attn_einsum(x, params, cos, sin):
    with jax.named_scope('attn_einsum'):
        h = rms_norm(x)
        q = jnp.einsum('btd,dhk->bhtk', h, params.c_q)
        k = jnp.einsum('btd,dhk->bhtk', h, params.c_k)
        v = jnp.einsum('btd,dhk->bhtk', h, params.c_v)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        q = rms_norm(q)
        k = rms_norm(k)
        k_exp, v_exp = _expand_kv(k, v, cfg.n_head, cfg.n_kv_head)
        scale = cfg.head_dim ** -0.5
        T = x.shape[1]
        scores = jnp.einsum('bhtd,bhsd->bhts', q, k_exp) * scale
        rows = jnp.arange(T)[:, None]
        cols = jnp.arange(T)[None, :]
        mask = cols <= rows
        scores = jnp.where(mask[None, None, :, :], scores,
                           jnp.finfo(scores.dtype).min)
        attn_weights = jax.nn.softmax(scores, axis=-1)
        attn_out = jnp.einsum('bhts,bhsd->bhtd', attn_weights, v_exp)
        return jnp.einsum('bhtd,hde->bte', attn_out, params.c_proj)

tok = cfg.batch_size * cfg.seq_len
proj_flops = (2 * tok * cfg.n_embd * cfg.n_head * cfg.head_dim +
              2 * tok * cfg.n_embd * cfg.n_kv_head * cfg.head_dim +
              2 * tok * cfg.n_embd * cfg.n_kv_head * cfg.head_dim +
              2 * tok * cfg.n_head * cfg.head_dim * cfg.n_embd)
attn_core = attention_flops(cfg.batch_size, cfg.n_head, cfg.seq_len, cfg.head_dim)
total_attn_flops = proj_flops + attn_core

x = fake_hidden(cfg.batch_size, cfg.seq_len, cfg.n_embd)
r_attn_ein = benchmark(bench_attn_einsum, x, attn_params, cos_b, sin_b,
                        flop_count=total_attn_flops, label="Attention (einsum)")

# %%
# 2e. Attention — jax.nn.dot_product_attention
print("=== Attention (jax.nn) ===")

@jax.jit
def bench_attn_jax(x, params, cos, sin):
    with jax.named_scope('attn_jax'):
        h = rms_norm(x)
        q = jnp.einsum('btd,dhk->bhtk', h, params.c_q)
        k = jnp.einsum('btd,dhk->bhtk', h, params.c_k)
        v = jnp.einsum('btd,dhk->bhtk', h, params.c_v)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        q = rms_norm(q)
        k = rms_norm(k)
        k_exp, v_exp = _expand_kv(k, v, cfg.n_head, cfg.n_kv_head)
        attn_out = jax.nn.dot_product_attention(
            q, k_exp, v_exp, is_causal=True, implementation='xla')
        return jnp.einsum('bhtd,hde->bte', attn_out, params.c_proj)

r_attn_jax = benchmark(bench_attn_jax, x, attn_params, cos_b, sin_b,
                        flop_count=total_attn_flops, label="Attention (jax.nn)")

# %%
# 2f. Attention — Pallas splash kernel
print("=== Attention (splash) ===")

from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_mask, splash_attention_kernel)

@jax.jit
def bench_attn_splash(x, params, cos, sin):
    with jax.named_scope('attn_splash'):
        h = rms_norm(x)
        q = jnp.einsum('btd,dhk->bhtk', h, params.c_q)
        k = jnp.einsum('btd,dhk->bhtk', h, params.c_k)
        v = jnp.einsum('btd,dhk->bhtk', h, params.c_v)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        q = rms_norm(q)
        k = rms_norm(k)
        T = x.shape[1]
        smask = splash_attention_mask.CausalMask(shape=(T, T))
        mh_mask = splash_attention_mask.MultiHeadMask(masks=[smask] * cfg.n_head)
        bs = min(cfg.splash_block_size, T)
        block_sizes = splash_attention_kernel.BlockSizes(
            block_q=bs, block_kv=bs,
            block_q_dkv=bs, block_kv_dkv=bs,
            block_q_dq=bs, block_kv_dq=bs)
        kernel = splash_attention_kernel.make_splash_mha(
            mask=mh_mask, head_shards=1, q_seq_shards=1,
            block_sizes=block_sizes)
        attn_out = jax.vmap(kernel)(q, k, v)
        return jnp.einsum('bhtd,hde->bte', attn_out, params.c_proj)

r_attn_splash = benchmark(bench_attn_splash, x, attn_params, cos_b, sin_b,
                           flop_count=total_attn_flops, label="Attention (splash)")

# %%
# 2g. Component comparison
print("\n=== Phase 2 Summary ===")
print_summary([r_norm, r_rope, r_mlp, r_attn_ein, r_attn_jax, r_attn_splash])

# %% [markdown]
# ### Ideas to try
# - **Fused RMSNorm+Linear** as a single Pallas kernel (saves one HBM read/write roundtrip)
# - **Remove QK-norm** from attention — saves 2 RMSNorm calls on Q and K
# - **Vary head_dim**: try 64, 128, 256, 512 — how does per-component MFU change?
# - **GQA within attention**: try n_kv_head = 1 (MQA), 2, 4 (MHA)

# %% [markdown]
# ## Phase 3 — Single Transformer Layer
#
# Assemble: pre-norm + attention + residual + pre-norm + MLP + residual.
# Compare full layer time with sum of Phase 2 parts.

# %%
# === Single layer functions ===

def init_layer_params(cfg, seed=42):
    """Initialize params for one transformer layer."""
    key = jax.random.key(seed)
    keys = jax.random.split(key, 7)
    s = (3.0 ** 0.5) * (cfg.n_embd ** -0.5)
    layer = dot_dict()
    layer.c_q = jax.random.uniform(keys[0], (cfg.n_embd, cfg.n_head, cfg.head_dim),
                                    dtype=jnp.bfloat16, minval=-s, maxval=s)
    layer.c_k = jax.random.uniform(keys[1], (cfg.n_embd, cfg.n_kv_head, cfg.head_dim),
                                    dtype=jnp.bfloat16, minval=-s, maxval=s)
    layer.c_v = jax.random.uniform(keys[2], (cfg.n_embd, cfg.n_kv_head, cfg.head_dim),
                                    dtype=jnp.bfloat16, minval=-s, maxval=s)
    layer.c_proj = jnp.zeros((cfg.n_head, cfg.head_dim, cfg.n_embd), dtype=jnp.bfloat16)
    layer.w_gate = jax.random.uniform(keys[3], (cfg.n_embd, cfg.mlp_dim),
                                       dtype=jnp.bfloat16, minval=-s, maxval=s)
    layer.w_up = jax.random.uniform(keys[4], (cfg.n_embd, cfg.mlp_dim),
                                     dtype=jnp.bfloat16, minval=-s, maxval=s)
    layer.w_down = jnp.zeros((cfg.mlp_dim, cfg.n_embd), dtype=jnp.bfloat16)
    return layer


def single_layer_forward(cfg, layer, x, cos, sin, *, attn_impl='splash',
                         use_rope=True, use_qk_norm=True):
    """Forward pass for one transformer layer."""
    h = rms_norm(x)

    # --- Attention ---
    q = jnp.einsum('btd,dhk->bhtk', h, layer.c_q)
    k = jnp.einsum('btd,dhk->bhtk', h, layer.c_k)
    v = jnp.einsum('btd,dhk->bhtk', h, layer.c_v)

    if use_rope:
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
    if use_qk_norm:
        q = rms_norm(q)
        k = rms_norm(k)

    T = x.shape[1]
    if attn_impl == 'splash':
        smask = splash_attention_mask.CausalMask(shape=(T, T))
        mh_mask = splash_attention_mask.MultiHeadMask(masks=[smask] * cfg.n_head)
        bs = min(cfg.splash_block_size, T)
        block_sizes = splash_attention_kernel.BlockSizes(
            block_q=bs, block_kv=bs, block_q_dkv=bs, block_kv_dkv=bs,
            block_q_dq=bs, block_kv_dq=bs)
        kernel = splash_attention_kernel.make_splash_mha(
            mask=mh_mask, head_shards=1, q_seq_shards=1,
            block_sizes=block_sizes)
        attn_out = jax.vmap(kernel)(q, k, v)
    elif attn_impl == 'einsum':
        k_exp, v_exp = _expand_kv(k, v, cfg.n_head, cfg.n_kv_head)
        scale = cfg.head_dim ** -0.5
        scores = jnp.einsum('bhtd,bhsd->bhts', q, k_exp) * scale
        rows = jnp.arange(T)[:, None]
        cols = jnp.arange(T)[None, :]
        mask = cols <= rows
        scores = jnp.where(mask[None, None, :, :], scores,
                           jnp.finfo(scores.dtype).min)
        attn_weights = jax.nn.softmax(scores, axis=-1)
        attn_out = jnp.einsum('bhts,bhsd->bhtd', attn_weights, v_exp)
    else:
        k_exp, v_exp = _expand_kv(k, v, cfg.n_head, cfg.n_kv_head)
        attn_out = jax.nn.dot_product_attention(
            q, k_exp, v_exp, is_causal=True, implementation='xla')

    attn_out = jnp.einsum('bhtd,hde->bte', attn_out, layer.c_proj)
    x = x + attn_out

    # --- SwiGLU MLP ---
    h2 = rms_norm(x)
    gate = jax.nn.silu(jnp.einsum('btd,dh->bth', h2, layer.w_gate))
    up = jnp.einsum('btd,dh->bth', h2, layer.w_up)
    mlp_out = jnp.einsum('bth,hd->btd', gate * up, layer.w_down)
    x = x + mlp_out
    return x

# %%
# 3a. Full layer benchmark
print("=== Single layer (splash) ===")
layer_p = init_layer_params(cfg)
x = fake_hidden(cfg.batch_size, cfg.seq_len, cfg.n_embd)
lf = layer_flops(cfg.batch_size, cfg.seq_len, cfg.n_embd,
                 cfg.n_head, cfg.n_kv_head, cfg.head_dim, cfg.mlp_dim)

@jax.jit
def bench_layer(x, layer, cos, sin):
    return single_layer_forward(cfg, layer, x, cos, sin, attn_impl='splash')

r_layer = benchmark(bench_layer, x, layer_p, cos_b, sin_b,
                    flop_count=lf, label="Full layer (splash)")

print(f"\n  Sum of parts (MLP + attn_splash):  "
      f"{r_mlp['wall_ms'] + r_attn_splash['wall_ms']:.2f} ms")
print(f"  Full layer:                        {r_layer['wall_ms']:.2f} ms")
print(f"  Delta (overhead / fusion benefit):  "
      f"{r_layer['wall_ms'] - r_mlp['wall_ms'] - r_attn_splash['wall_ms']:.2f} ms")

# %%
# 3b. Layer ablations
print("\n=== Layer ablations ===")
results_3b = []

for label, kw in [
    ("splash, +rope, +qknorm", dict(attn_impl='splash', use_rope=True, use_qk_norm=True)),
    ("splash, -rope, +qknorm", dict(attn_impl='splash', use_rope=False, use_qk_norm=True)),
    ("splash, +rope, -qknorm", dict(attn_impl='splash', use_rope=True, use_qk_norm=False)),
    ("splash, -rope, -qknorm", dict(attn_impl='splash', use_rope=False, use_qk_norm=False)),
    ("einsum, +rope, +qknorm", dict(attn_impl='einsum', use_rope=True, use_qk_norm=True)),
]:
    @jax.jit
    def bench_fn(x, layer, cos, sin, _kw=kw):
        return single_layer_forward(cfg, layer, x, cos, sin, **_kw)

    r = benchmark(bench_fn, x, layer_p, cos_b, sin_b, flop_count=lf, label=label)
    results_3b.append(r)

print_summary(results_3b)

# %% [markdown]
# ### Ideas to try
# - **Remove RMSNorm entirely** (unsafe for training but measures its overhead)
# - **Remove softcap** — softcap uses `tanh` which is slow on MXU
# - **Capture XProf trace**: wrap a benchmark call with `jax.profiler.start_trace` / `stop_trace`
# - **Two consecutive layers** — does XLA pipeline them better?

# %% [markdown]
# ## Phase 4 — Stacking Layers
#
# Does MFU change with depth? How does HBM scale?

# %%
# === Multi-layer model ===

def init_all_layers(cfg, n_layers, seed=42):
    layers = dot_dict()
    for i in range(n_layers):
        layers[i] = init_layer_params(cfg, seed=seed + i * 7)
    return layers


def multi_layer_forward(cfg, layers, n_layers, x, cos, sin, attn_impl='splash'):
    for i in range(n_layers):
        x = single_layer_forward(cfg, layers[i], x, cos, sin, attn_impl=attn_impl)
    return rms_norm(x)

# %%
# 4a. Depth sweep
print("=== Depth sweep ===")
results_4a = []
x = fake_hidden(cfg.batch_size, cfg.seq_len, cfg.n_embd)

for n_layers in [1, 4, 8, 16, 24]:
    layers = init_all_layers(cfg, n_layers)
    fl = n_layers * layer_flops(cfg.batch_size, cfg.seq_len, cfg.n_embd,
                                cfg.n_head, cfg.n_kv_head, cfg.head_dim, cfg.mlp_dim)

    @jax.jit
    def bench_fn(x, layers, cos, sin, _n=n_layers):
        return multi_layer_forward(cfg, layers, _n, x, cos, sin)

    r = benchmark(bench_fn, x, layers, cos_b, sin_b,
                  flop_count=fl, label=f"{n_layers} layers")
    r['ms_per_layer'] = r['wall_ms'] / n_layers
    results_4a.append(r)

print_summary(results_4a)
print("  Per-layer time:")
for r in results_4a:
    print(f"    {r['label']:<20s}  {r['ms_per_layer']:.2f} ms/layer")

# %% [markdown]
# ### Ideas to try
# - **`jax.checkpoint` (remat)** on each layer — how many more layers fit?
#   ```python
#   x = jax.checkpoint(single_layer_forward)(cfg, layers[i], x, cos, sin)
#   ```
# - **`jax.lax.scan`** over layers with stacked params — reduces compilation time:
#   ```python
#   def scan_body(x, layer_params):
#       return single_layer_forward(cfg, layer_params, x, cos, sin), None
#   x, _ = jax.lax.scan(scan_body, x, stacked_layers)
#   ```
# - **Max batch_size × n_layers grid**: find the OOM boundary

# %% [markdown]
# ## Phase 5 — Embedding & LM Head
#
# - Embedding: pure memory lookup (0% MFU)
# - LM head: large matmul `(B*T, n_embd) @ (n_embd, vocab)` — high MFU
# - Chunked vs non-chunked loss comparison

# %%
# 5a. Embedding lookup
print("=== Embedding ===")
wte = jax.random.normal(jax.random.key(0),
    (cfg.padded_vocab, cfg.n_embd), dtype=jnp.bfloat16)
tokens = fake_tokens(cfg.batch_size, cfg.seq_len)

@jax.jit
def bench_embed(tokens, wte):
    return rms_norm(wte[tokens])

r_embed = benchmark(bench_embed, tokens, wte, flop_count=None,
                    label="Embedding + norm")

# %%
# 5b. LM head — non-chunked
print("=== LM head (non-chunked) ===")
lm_head = jax.random.normal(jax.random.key(1),
    (cfg.n_embd, cfg.padded_vocab), dtype=jnp.bfloat16) * 0.001
hidden = fake_hidden(cfg.batch_size, cfg.seq_len, cfg.n_embd)
labels = fake_tokens(cfg.batch_size, cfg.seq_len)

@jax.jit
def bench_lm_head(hidden, lm_head, labels):
    logits = jnp.einsum('btd,dv->btv', hidden, lm_head)
    logits = logits[:, :, :cfg.vocab_size].astype(jnp.float32)
    logits = cfg.softcap * jnp.tanh(logits / cfg.softcap)
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, labels))

lm_flops = matmul_flops(cfg.batch_size * cfg.seq_len, cfg.padded_vocab, cfg.n_embd)
r_lm = benchmark(bench_lm_head, hidden, lm_head, labels,
                 flop_count=lm_flops, label="LM head (non-chunked)")

# %%
# 5c. Chunked LM head loss (from 04_maxtext.py)

def _logits_from_chunk(h_chunk, lm_head, config):
    logits = jnp.einsum('td,dv->tv', h_chunk, lm_head)
    logits = logits[:, :config.vocab_size]
    logits = logits.astype(jnp.float32)
    return config.softcap * jnp.tanh(logits / config.softcap)


@ft.partial(jax.custom_vjp, nondiff_argnums=(3,))
def chunked_lm_head_loss(hidden, lm_head, labels, config):
    B, T, D = hidden.shape
    N = config.num_lm_head_chunks
    S = B * T // N
    hidden_chunks = hidden.reshape(N, S, D)
    labels_chunks = labels.reshape(N, S)

    def fwd_body(_, data):
        h_chunk, l_chunk = data
        return None, jnp.sum(
            optax.softmax_cross_entropy_with_integer_labels(
                _logits_from_chunk(h_chunk, lm_head, config), l_chunk))

    _, chunk_losses = jax.lax.scan(fwd_body, None, (hidden_chunks, labels_chunks))
    return jnp.sum(chunk_losses) / (B * T)


def _chunked_loss_fwd(hidden, lm_head, labels, config):
    loss = chunked_lm_head_loss(hidden, lm_head, labels, config)
    return loss, (hidden, lm_head, labels)


def _chunked_loss_bwd(config, residuals, g):
    hidden, lm_head, labels = residuals
    B, T, D = hidden.shape
    N = config.num_lm_head_chunks
    S = B * T // N
    hidden_chunks = hidden.reshape(N, S, D)
    labels_chunks = labels.reshape(N, S)

    def bwd_body(d_lm_head_acc, data):
        h_chunk, l_chunk = data

        def chunk_loss(h, w):
            return jnp.sum(
                optax.softmax_cross_entropy_with_integer_labels(
                    _logits_from_chunk(h, w, config), l_chunk))

        _, vjp_fn = jax.vjp(chunk_loss, h_chunk, lm_head)
        d_h, d_w = vjp_fn(g / (B * T))
        return d_lm_head_acc + d_w, d_h

    d_lm_head_init = jnp.zeros_like(lm_head)
    d_lm_head, d_hidden_chunks = jax.lax.scan(
        bwd_body, d_lm_head_init, (hidden_chunks, labels_chunks))
    return d_hidden_chunks.reshape(B, T, D), d_lm_head, jnp.zeros_like(labels)


chunked_lm_head_loss.defvjp(_chunked_loss_fwd, _chunked_loss_bwd)

print("=== LM head (chunked, 8 chunks) ===")

@jax.jit
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
# - **Vocab alignment**: 32768 (256-aligned) vs 50257 (GPT-2, non-aligned) — how much padding waste?
# - **Weight-tied embedding** (`wte.T` as lm_head) — saves HBM but may hurt convergence (see LOG.md)

# %% [markdown]
# ## Phase 6 — Forward vs Forward+Backward
#
# The backward pass typically costs 2-3x the forward.
# Gradient checkpointing (remat) trades compute for memory.

# %%
# === Full model for fwd/bwd testing ===

def init_full_model(cfg, seed=42):
    """Initialize all model params (embed + layers + lm_head + rope)."""
    key = jax.random.key(seed)
    params = dot_dict()
    key, k1, k2 = jax.random.split(key, 3)
    params.wte = jax.random.normal(k1, (cfg.padded_vocab, cfg.n_embd), dtype=jnp.bfloat16)
    params.lm_head = jax.random.normal(k2, (cfg.n_embd, cfg.padded_vocab),
                                        dtype=jnp.bfloat16) * 0.001
    params.rope_cos, params.rope_sin = precompute_rope(cfg.seq_len, cfg.head_dim)
    params.layers = init_all_layers(cfg, cfg.n_layer, seed=seed + 100)
    return params


def model_forward(cfg, params, tokens):
    """Full forward: embed -> layers -> final_norm.  Returns hidden (B,T,E)."""
    B, T = tokens.shape
    cos = params.rope_cos[:T][None, None, :, :]
    sin = params.rope_sin[:T][None, None, :, :]
    x = rms_norm(params.wte[tokens])
    for i in range(cfg.n_layer):
        x = single_layer_forward(cfg, params.layers[i], x, cos, sin, attn_impl='splash')
    return rms_norm(x)


def model_forward_remat(cfg, params, tokens):
    """Same as model_forward but with jax.checkpoint on each layer."""
    B, T = tokens.shape
    cos = params.rope_cos[:T][None, None, :, :]
    sin = params.rope_sin[:T][None, None, :, :]
    x = rms_norm(params.wte[tokens])
    layer_fn = ft.partial(single_layer_forward, cfg, attn_impl='splash')
    for i in range(cfg.n_layer):
        x = jax.checkpoint(layer_fn)(params.layers[i], x, cos, sin)
    return rms_norm(x)


# %%
# === Utilities for optimizer benchmarks ===

def split_trainable(params):
    """Split params into trainable and static (non-differentiable).
    From 02_train.py — rope_cos/rope_sin are precomputed, not trained."""
    trainable = dot_dict()
    static = dot_dict()
    for k, v in params.items():
        if k in ('rope_cos', 'rope_sin'):
            static[k] = v
        else:
            trainable[k] = v
    return trainable, static


def merge_params(trainable, static):
    """Merge trainable and static params back together."""
    merged = dot_dict()
    merged.update(trainable)
    merged.update(static)
    return merged


def init_adam_state(param):
    """Initialize Adam optimizer state for a single parameter."""
    return dot_dict(
        mu=jnp.zeros_like(param),
        nu=jnp.zeros_like(param),
        count=jnp.array(0, dtype=jnp.int32),
    )


def adamw_step(lr, beta1, beta2, eps, wd, lr_mult, param, grad, state):
    """AdamW update with explicit hyperparams. Returns (new_param, new_state).
    Note: weight_decay applied only to 2D+ params (matching 02_train.py).
    optax applies weight_decay uniformly — minor semantic difference on bias/scalar params."""
    new_count = state.count + 1
    new_mu = beta1 * state.mu + (1 - beta1) * grad
    new_nu = beta2 * state.nu + (1 - beta2) * grad ** 2

    mu_hat = new_mu / (1 - beta1 ** new_count)
    nu_hat = new_nu / (1 - beta2 ** new_count)

    lr_eff = lr * lr_mult
    update = mu_hat / (jnp.sqrt(nu_hat) + eps)

    # Weight decay for 2D+ params only (matches 02_train.py)
    wd_eff = jnp.where(param.ndim >= 2, wd, 0.0)
    new_param = param - lr_eff * (update + wd_eff * param)

    new_state = dot_dict(mu=new_mu, nu=new_nu, count=new_count)
    return new_param, new_state


def count_params(params):
    """Count total trainable parameters (excludes rope_cos/rope_sin)."""
    trainable, _ = split_trainable(params)
    return sum(p.size for p in jax.tree.leaves(trainable) if isinstance(p, jax.Array))


def count_non_embed_params(params):
    """Non-embedding params (unembed + layers). Excludes wte (lookup table)."""
    return count_params(params) - params.wte.size



# %%
full_params = init_full_model(cfg)
tokens = fake_tokens(cfg.batch_size, cfg.seq_len)
labels = fake_tokens(cfg.batch_size, cfg.seq_len)

total_model_flops = (cfg.n_layer * layer_flops(cfg.batch_size, cfg.seq_len, cfg.n_embd,
                     cfg.n_head, cfg.n_kv_head, cfg.head_dim, cfg.mlp_dim) +
                     matmul_flops(cfg.batch_size * cfg.seq_len, cfg.padded_vocab, cfg.n_embd))

# %%
# 6a. Forward only
print("=== Forward only ===")

@jax.jit
def bench_fwd(params, tokens):
    hidden = model_forward(cfg, params, tokens)
    return chunked_lm_head_loss(hidden, params.lm_head, labels, cfg)

r_fwd = benchmark(bench_fwd, full_params, tokens,
                  flop_count=total_model_flops, label="Forward only")

# %%
# 6b. Forward + backward
print("=== Forward + Backward ===")

@jax.jit
def bench_fwd_bwd(params, tokens):
    def loss_fn(p):
        hidden = model_forward(cfg, p, tokens)
        return chunked_lm_head_loss(hidden, p.lm_head, labels, cfg)
    return jax.value_and_grad(loss_fn)(params)

# ~3x forward FLOPs (fwd + 2x bwd)
bwd_flops = 3 * total_model_flops
r_fwd_bwd = benchmark(bench_fwd_bwd, full_params, tokens,
                       flop_count=bwd_flops, label="Forward+Backward")

# %%
# 6c. Forward + backward with remat
print("=== Forward + Backward (remat) ===")

@jax.jit
def bench_fwd_bwd_remat(params, tokens):
    def loss_fn(p):
        hidden = model_forward_remat(cfg, p, tokens)
        return chunked_lm_head_loss(hidden, p.lm_head, labels, cfg)
    return jax.value_and_grad(loss_fn)(params)

r_remat = benchmark(bench_fwd_bwd_remat, full_params, tokens,
                    flop_count=bwd_flops, label="Fwd+Bwd (remat)")

# %%
# 6d. Comparison
print("\n=== Phase 6 Summary ===")
print_summary([r_fwd, r_fwd_bwd, r_remat])
print(f"  Backward / Forward ratio:  {r_fwd_bwd['wall_ms'] / max(r_fwd['wall_ms'], 0.01):.2f}x")
print(f"  Remat overhead vs no-remat: {r_remat['wall_ms'] / max(r_fwd_bwd['wall_ms'], 0.01):.2f}x")

n_params = count_params(full_params)
print(f"\n  Trainable params: {n_params:,}")
for r in [r_fwd_bwd, r_remat]:
    print(f"  {r['label']:<30s}  MFU {r['mfu_pct']:5.1f}%")

# %% [markdown]
# ### Ideas to try
# - **Partial remat**: checkpoint every other layer, or only attention
# - **Checkpoint policies**: `jax.checkpoint(fn, policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable)`
# - **Remat at different batch sizes** — at small batch size memory may not be the bottleneck

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
              matmul_flops(bs * cfg_bs.seq_len, cfg_bs.padded_vocab, cfg_bs.n_embd))

    @jax.jit
    def bench(p, tok, _lab=lab, _cfg=cfg_bs):
        def loss_fn(params):
            hidden = model_forward(_cfg, params, tok)
            return chunked_lm_head_loss(hidden, params.lm_head, _lab, _cfg)
        return jax.value_and_grad(loss_fn)(p)

    r = benchmark(bench, p, tok, flop_count=fl, label=f"bs={bs}")
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

for sl in [1024, 2048, 4096]:
    cfg_sl = PerfConfig(seq_len=sl)
    p = init_full_model(cfg_sl)
    tok = fake_tokens(cfg_sl.batch_size, sl)
    lab = fake_tokens(cfg_sl.batch_size, sl)
    fl = 3 * (cfg_sl.n_layer * layer_flops(cfg_sl.batch_size, sl, cfg_sl.n_embd,
              cfg_sl.n_head, cfg_sl.n_kv_head, cfg_sl.head_dim, cfg_sl.mlp_dim) +
              matmul_flops(cfg_sl.batch_size * sl, cfg_sl.padded_vocab, cfg_sl.n_embd))

    @jax.jit
    def bench(p, tok, _lab=lab, _cfg=cfg_sl):
        def loss_fn(params):
            hidden = model_forward(_cfg, params, tok)
            return chunked_lm_head_loss(hidden, params.lm_head, _lab, _cfg)
        return jax.value_and_grad(loss_fn)(p)

    r = benchmark(bench, p, tok, flop_count=fl, label=f"seq_len={sl}")
    r['tok_per_sec'] = cfg_sl.batch_size * sl / (r['wall_ms'] / 1000)
    results_7b.append(r)

print_summary(results_7b)
print("  Throughput:")
for r in results_7b:
    print(f"    {r['label']:<20s}  {r['tok_per_sec']:,.0f} tok/s")

# %%
# 7c. Head dim alignment: 128 (8 heads) vs 256 (4 heads)
print("\n=== Head dim alignment ===")
results_7c = []

for hd, nh in [(128, 8), (256, 4)]:
    cfg_hd = PerfConfig(head_dim=hd, n_head=nh, n_kv_head=max(1, nh // 4),
                         n_embd=nh * hd)
    p = init_full_model(cfg_hd)
    tok = fake_tokens(cfg_hd.batch_size, cfg_hd.seq_len)
    lab = fake_tokens(cfg_hd.batch_size, cfg_hd.seq_len)
    fl = 3 * (cfg_hd.n_layer * layer_flops(cfg_hd.batch_size, cfg_hd.seq_len,
              cfg_hd.n_embd, cfg_hd.n_head, cfg_hd.n_kv_head,
              cfg_hd.head_dim, cfg_hd.mlp_dim) +
              matmul_flops(cfg_hd.batch_size * cfg_hd.seq_len,
                           cfg_hd.padded_vocab, cfg_hd.n_embd))

    @jax.jit
    def bench(p, tok, _lab=lab, _cfg=cfg_hd):
        def loss_fn(params):
            hidden = model_forward(_cfg, params, tok)
            return chunked_lm_head_loss(hidden, params.lm_head, _lab, _cfg)
        return jax.value_and_grad(loss_fn)(p)

    r = benchmark(bench, p, tok, flop_count=fl,
                  label=f"head_dim={hd}, n_head={nh}, D={nh*hd}")
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
                           cfg_gqa.padded_vocab, cfg_gqa.n_embd))

    @jax.jit
    def bench(p, tok, _lab=lab, _cfg=cfg_gqa):
        def loss_fn(params):
            hidden = model_forward(_cfg, params, tok)
            return chunked_lm_head_loss(hidden, params.lm_head, _lab, _cfg)
        return jax.value_and_grad(loss_fn)(p)

    r = benchmark(bench, p, tok, flop_count=fl,
                  label=f"n_kv_head={n_kv} (ratio {cfg.n_head}:{n_kv})")
    r['tok_per_sec'] = cfg_gqa.batch_size * cfg_gqa.seq_len / (r['wall_ms'] / 1000)
    results_7d.append(r)

print_summary(results_7d)

# %%
# 7e. Splash block size sweep (single layer)
print("\n=== Splash block size sweep (single layer) ===")
results_7e = []
x = fake_hidden(cfg.batch_size, cfg.seq_len, cfg.n_embd)
lf = layer_flops(cfg.batch_size, cfg.seq_len, cfg.n_embd,
                 cfg.n_head, cfg.n_kv_head, cfg.head_dim, cfg.mlp_dim)

for bs in [256, 512, 1024]:
    cfg_bs = PerfConfig(splash_block_size=bs)
    layer_p = init_layer_params(cfg_bs)

    @jax.jit
    def bench_fn(x, layer, cos, sin, _cfg=cfg_bs):
        return single_layer_forward(_cfg, layer, x, cos, sin, attn_impl='splash')

    r = benchmark(bench_fn, x, layer_p, cos_b, sin_b,
                  flop_count=lf, label=f"splash block_size={bs}")
    results_7e.append(r)

print_summary(results_7e)

# %%
# 7f. Attention implementation comparison (single layer)
print("\n=== Attention implementation comparison (single layer) ===")
results_7f = []
x = fake_hidden(cfg.batch_size, cfg.seq_len, cfg.n_embd)
layer_p = init_layer_params(cfg)

for impl in ['einsum', 'jax', 'splash']:
    @jax.jit
    def bench_fn(x, layer, cos, sin, _impl=impl):
        return single_layer_forward(cfg, layer, x, cos, sin, attn_impl=_impl)

    r = benchmark(bench_fn, x, layer_p, cos_b, sin_b,
                  flop_count=lf, label=f"attn_impl={impl}")
    results_7f.append(r)

print_summary(results_7f)

# %% [markdown]
# ### Ideas to try
# - **Combined batch_size × seq_len grid** — find the throughput-maximizing combo
# - **`n_embd` sweep**: 512, 768, 1024, 1536, 2048 (all 256-aligned)
# - **`mlp_dim` expansion ratio**: 2x, 3x, 4x — how does MLP FLOPs fraction affect overall MFU?
# - **`n_layer` vs `n_embd`**: given a fixed param budget, is it better to go deep or wide?
# - **Tok/s vs MFU%**: these are different! MFU measures compute efficiency, tok/s measures throughput

# %% [markdown]
# ## Phase 8 — Advanced Optimization Ideas
#
# Concepts with starter code for experimentation.

# %% [markdown]
# ### 8.1 Custom Pallas Kernel: Fused RMSNorm + Linear
#
# RMSNorm reads all of `x` from HBM, writes normalized `x` back, then the next matmul
# reads it again.  A fused Pallas kernel could normalize and multiply in one pass,
# saving one full HBM roundtrip (~2 × B × T × E bytes).
#
# ```python
# from jax.experimental import pallas as pl
#
# def fused_norm_linear_kernel(x_ref, w_ref, out_ref):
#     """Fused RMSNorm + linear projection in a single Pallas kernel."""
#     x = x_ref[...]
#     # RMSNorm in-register
#     x_norm = x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + 1e-6)
#     # Linear projection
#     out_ref[...] = x_norm @ w_ref[...]
#
# # Call with:
# # out = pl.pallas_call(fused_norm_linear_kernel, ...)(x, w)
# ```
#
# **Expected benefit**: Saves ~2 × 8 × 2048 × 1024 × 2 = 64 MB per layer per call.
# At 24 layers with 4 norm+linear pairs each, that's ~6 GB less HBM traffic.

# %% [markdown]
# ### 8.2 Int8 Matmul Experiments
#
# TPU v6e supports int8 computation with ~2x the bf16 throughput.
# XLA can auto-quantize via `jax.lax.dot_general` with `preferred_element_type`.
#
# ```python
# # Manual quantization
# scale = jnp.max(jnp.abs(a)) / 127.0
# a_i8 = jnp.int8(jnp.round(a / scale))
# b_i8 = jnp.int8(jnp.round(b / scale))
#
# result_i32 = jax.lax.dot_general(
#     a_i8, b_i8,
#     dimension_numbers=(((1,), (0,)), ((), ())),
#     preferred_element_type=jnp.int32)
# result_bf16 = (result_i32 * scale * scale).astype(jnp.bfloat16)
# ```
#
# **Caveats**: Quantization noise hurts attention scores more than MLP.
# Consider quantizing only the MLP weights (gate/up/down) first.

# %% [markdown]
# ### 8.3 Scan-based Layer Stacking
#
# `jax.lax.scan` over layers reduces XLA compilation time from O(n_layer) to O(1)
# and makes remat trivial.  Requires stacking all layer params into arrays with
# a leading layer dimension.
#
# ```python
# # Stack params: each weight becomes (n_layer, ...)
# stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *[layers[i] for i in range(n)])
#
# def scan_body(x, layer_params):
#     x = single_layer_forward(cfg, layer_params, x, cos, sin)
#     return x, None
#
# x, _ = jax.lax.scan(scan_body, x_init, stacked)
# ```
#
# **Bonus**: Combine with `jax.checkpoint` for free per-layer remat:
# `x, _ = jax.lax.scan(jax.checkpoint(scan_body), x_init, stacked)`

# %% [markdown]
# ### 8.4 Pipeline / Microbatch Strategies
#
# Split the batch into micro-batches and overlap compute of one with data movement
# of the next.  On a single chip the benefit is limited, but it helps when the model
# barely fits:
#
# ```python
# micro_bs = cfg.batch_size // 4
# micro_grads = []
# for i in range(4):
#     mb = tokens[i*micro_bs : (i+1)*micro_bs]
#     _, g = jax.value_and_grad(loss_fn)(params, mb)
#     micro_grads.append(g)
# grads = jax.tree.map(lambda *gs: sum(gs) / 4, *micro_grads)
# ```
#
# This is essentially gradient accumulation — useful if you need a large effective
# batch but can only fit a small micro-batch in HBM.

# %% [markdown]
# ### 8.5 Mixed Precision Strategies
#
# The model keeps all params and activations in **bfloat16**, with selective
# upcasting to float32 for numerically sensitive ops:
#
# - **Currently f32**: logit softcap (`tanh`), cross-entropy loss
# - **Worth trying in f32**: attention scores (before softmax), optimizer moments
# - **Keep bf16**: all matmuls (projections, MLP), RMSNorm, RoPE
#
# The key insight: bf16 matmul already accumulates internally in f32 on MXU,
# so the matmul output is accurate even though inputs are bf16.
# The danger zones are **reductions** (softmax, loss) and **small values** (optimizer eps).

# %% [markdown]
# ### 8.6 Memory Layout Optimization
#
# JAX/XLA uses row-major (C-contiguous) layout by default.
# For attention, the `(B, H, T, D)` layout is optimal because:
# - Splash/Pallas kernels expect this layout natively
# - The T dimension (which we iterate over in causal masking) is contiguous with D
# - Einsum `'bhtd,bhsd->bhts'` has good locality
#
# The alternative `(B, T, H, D)` (common in PyTorch) requires a transpose before
# attention, which is a pure memory operation with 0% MFU.
#
# Our codebase avoids transposes entirely by producing Q/K/V in `(B, H, T, D)`
# directly via `einsum('btd,dhk->bhtk', ...)` — the einsum fuses the reshape.

# %% [markdown]
# ### 8.7 Double Buffering / Async Data Transfer
#
# For real training (not this fake-data notebook), overlapping host→device
# transfer with compute is critical:
#
# ```python
# class PrefetchDataLoader:
#     def _worker(self):
#         for x, y in self.iterator:
#             # Transfer to HBM in background thread
#             item = (jax.device_put(jnp.array(x)),
#                     jax.device_put(jnp.array(y)))
#             self.queue.put(item)
# ```
#
# See `04_maxtext.py` for the full implementation.
# With prefetching, data loading adds ~0ms overhead per step.

# %% [markdown]
# ### 8.8 Sharding for Multi-Chip (Future)
#
# When scaling beyond a single TPU v6e, use `jax.sharding`:
#
# - **Tensor parallelism**: shard along the head dimension (natural for GQA).
#   Each chip handles `n_head / num_chips` heads.
# - **FSDP**: shard parameters and gather them just-in-time.
#   `jax.experimental.mesh_utils.create_device_mesh` + `NamedSharding`.
# - **Pipeline parallelism**: assign layers to different chips.
#   `jax.lax.ppermute` for inter-chip communication.
#
# ```python
# from jax.sharding import NamedSharding, PartitionSpec as P, Mesh
# mesh = Mesh(jax.devices(), ('dp',))
# # Shard batch dimension across devices
# x_sharded = jax.device_put(x, NamedSharding(mesh, P('dp', None)))
# ```

# %% [markdown]
# ### 8.9 XProf Trace Capture
#
# Wrap any benchmark with XProf to get a detailed hardware trace:
#
# ```python
# jax.profiler.start_trace('my_trace_dir')
# for _ in range(5):
#     out = bench_fn(*args)
#     jax.block_until_ready(out)
# jax.profiler.stop_trace()
# # Then: %load_ext tensorboard  /  %tensorboard --logdir my_trace_dir
# ```
#
# XProf shows:
# - MFU (actual vs peak)
# - Memory timeline (stack, heap, fragmentation)
# - Op-level breakdown (which ops are slowest)
# - Idle gaps (input-bound vs compute-bound)

# %% [markdown]
# ## Phase 9 — Optimizer Step Benchmarks
#
# Measure full train step (fwd + bwd with remat + optimizer update).
# Compare manual AdamW (matching 02_train.py) vs optax.adamw variants.

# %%
# === Phase 9: Optimizer step benchmarks ===
print("=== Phase 9: Optimizer Step Benchmarks ===")
print(f"Config: B={cfg.batch_size}, T={cfg.seq_len}, D={cfg.n_embd}, L={cfg.n_layer}")

n_params = count_params(full_params)
print(f"Trainable params: {n_params:,}")

# Shared: tokens and labels for all optimizer benchmarks
opt_tokens = fake_tokens(cfg.batch_size, cfg.seq_len)
opt_labels = fake_tokens(cfg.batch_size, cfg.seq_len)

# Optimizer hyperparams (matching 02_train.py defaults)
OPT_LR = 3e-4
OPT_BETA1 = 0.9
OPT_BETA2 = 0.95
OPT_EPS = 1e-8
OPT_WD = 0.1

# %%
# 9a. Manual AdamW (per-leaf loop, matching 02_train.py)
print("\n--- 9a. Manual AdamW ---")

manual_params = init_full_model(cfg)
manual_trainable, manual_static = split_trainable(manual_params)
manual_opt_state = jax.tree.map(init_adam_state, manual_trainable)

@jax.jit
def bench_manual_adamw(params, opt_state, tokens, lr_mult):
    trainable, static = split_trainable(params)

    def loss_fn(t):
        full = merge_params(t, static)
        hidden = model_forward_remat(cfg, full, tokens)
        return chunked_lm_head_loss(hidden, full.lm_head, opt_labels, cfg)

    loss, grads = jax.value_and_grad(loss_fn)(trainable)

    # Per-leaf AdamW update (matching 02_train.py pattern)
    is_opt_leaf = lambda x: isinstance(x, dot_dict) and 'mu' in x
    t_leaves, t_treedef = jax.tree.flatten(trainable)
    g_leaves, _ = jax.tree.flatten(grads)
    o_leaves, o_treedef = jax.tree.flatten(opt_state, is_leaf=is_opt_leaf)

    new_t_leaves, new_o_leaves = [], []
    for p, g, s in zip(t_leaves, g_leaves, o_leaves):
        new_p, new_s = adamw_step(OPT_LR, OPT_BETA1, OPT_BETA2, OPT_EPS,
                                   OPT_WD, lr_mult, p, g, s)
        new_t_leaves.append(new_p)
        new_o_leaves.append(new_s)

    new_trainable = t_treedef.unflatten(new_t_leaves)
    new_opt_state = o_treedef.unflatten(new_o_leaves)
    new_params = merge_params(new_trainable, static)
    return loss, new_params, new_opt_state

lr_mult = jnp.array(1.0, dtype=jnp.float32)
r_manual = benchmark(bench_manual_adamw, manual_params, manual_opt_state,
                     opt_tokens, lr_mult, flop_count=bwd_flops,
                     label="Manual AdamW (full step)")

# %%
# 9b. optax.adamw (f32 moments)
print("\n--- 9b. optax.adamw (f32 mu) ---")

optax_params_f32 = init_full_model(cfg)
optax_trainable_f32, optax_static_f32 = split_trainable(optax_params_f32)
schedule_f32 = optax.adamw(OPT_LR, b1=OPT_BETA1, b2=OPT_BETA2,
                            eps=OPT_EPS, weight_decay=OPT_WD)
optax_state_f32 = schedule_f32.init(optax_trainable_f32)

@jax.jit
def bench_optax_f32(params, opt_state, tokens):
    trainable, static = split_trainable(params)

    def loss_fn(t):
        full = merge_params(t, static)
        hidden = model_forward_remat(cfg, full, tokens)
        return chunked_lm_head_loss(hidden, full.lm_head, opt_labels, cfg)

    loss, grads = jax.value_and_grad(loss_fn)(trainable)
    updates, new_opt_state = schedule_f32.update(grads, opt_state, trainable)
    new_trainable = optax.apply_updates(trainable, updates)
    new_params = merge_params(new_trainable, static)
    return loss, new_params, new_opt_state

r_optax_f32 = benchmark(bench_optax_f32, optax_params_f32, optax_state_f32,
                         opt_tokens, flop_count=bwd_flops,
                         label="optax.adamw f32 (full step)")

# %%
# 9c. optax.adamw (bf16 moments — MaxText style)
print("\n--- 9c. optax.adamw (bf16 mu) ---")

optax_params_bf16 = init_full_model(cfg)
optax_trainable_bf16, optax_static_bf16 = split_trainable(optax_params_bf16)
schedule_bf16 = optax.adamw(OPT_LR, b1=OPT_BETA1, b2=OPT_BETA2,
                             eps=OPT_EPS, weight_decay=OPT_WD,
                             mu_dtype=jnp.bfloat16)
optax_state_bf16 = schedule_bf16.init(optax_trainable_bf16)

@jax.jit
def bench_optax_bf16(params, opt_state, tokens):
    trainable, static = split_trainable(params)

    def loss_fn(t):
        full = merge_params(t, static)
        hidden = model_forward_remat(cfg, full, tokens)
        return chunked_lm_head_loss(hidden, full.lm_head, opt_labels, cfg)

    loss, grads = jax.value_and_grad(loss_fn)(trainable)
    updates, new_opt_state = schedule_bf16.update(grads, opt_state, trainable)
    new_trainable = optax.apply_updates(trainable, updates)
    new_params = merge_params(new_trainable, static)
    return loss, new_params, new_opt_state

r_optax_bf16 = benchmark(bench_optax_bf16, optax_params_bf16, optax_state_bf16,
                          opt_tokens, flop_count=bwd_flops,
                          label="optax.adamw bf16 (full step)")

# %%
# 9d. Phase 9 Summary
print("\n=== Phase 9 Summary ===")
phase9_results = [r_remat, r_manual, r_optax_f32, r_optax_bf16]
print(f"\n  {'Label':<40s}  {'Wall ms':>8s}  {'MFU%':>6s}  {'tok/s':>10s}")
print("  " + "-" * 72)
for r in phase9_results:
    tok_s = cfg.batch_size * cfg.seq_len / (r['wall_ms'] / 1000)
    print(f"  {r['label']:<40s}  {r['wall_ms']:8.2f}  {r['mfu_pct']:5.1f}%  "
          f"{tok_s:>10,.0f}")

# Optimizer overhead vs fwd+bwd only
for r in [r_manual, r_optax_f32, r_optax_bf16]:
    overhead = r['wall_ms'] - r_remat['wall_ms']
    pct = overhead / r_remat['wall_ms'] * 100
    print(f"  {r['label']:<40s}  optimizer overhead: {overhead:+.2f} ms ({pct:+.1f}%)")

# %% [markdown]
# ## Phase 10 — ~100M Param Config Sweep
#
# Full train step (fwd+bwd+remat+optimizer) for configs targeting ~100M params.
# Uses batch_size=16 (smaller model → more batch fits in HBM).

# %%
# === Phase 10: ~100M param config sweep ===
print("=== Phase 10: ~100M Param Config Sweep ===")

sweep_configs = [
    # (label, n_head, n_kv_head, head_dim, n_embd, mlp_dim, n_layer, batch_size)
    ("D768 N3 L8",   3, 1, 256, 768,  2048, 8,  16),
    ("D512 N2 L16",  2, 1, 256, 512,  1536, 16, 16),
    ("D512 N2 L24",  2, 1, 256, 512,  1536, 24, 16),
]

results_10 = []
for label, nh, nkv, hd, ne, mlp, nl, bs in sweep_configs:
    print(f"\n--- {label}: D={ne}, N={nh}, K={nkv}, H={hd}, F={mlp}, L={nl}, B={bs} ---")
    cfg_s = PerfConfig(batch_size=bs, n_head=nh, n_kv_head=nkv, head_dim=hd,
                       n_embd=ne, mlp_dim=mlp, n_layer=nl)
    p_s = init_full_model(cfg_s)
    n_p = count_params(p_s)
    print(f"  Trainable params: {n_p:,}")

    tok_s = fake_tokens(bs, cfg_s.seq_len)
    lab_s = fake_tokens(bs, cfg_s.seq_len)

    # Use optax.adamw f32 (likely best from Phase 9)
    tr_s, st_s = split_trainable(p_s)
    sched_s = optax.adamw(OPT_LR, b1=OPT_BETA1, b2=OPT_BETA2,
                           eps=OPT_EPS, weight_decay=OPT_WD)
    ostate_s = sched_s.init(tr_s)

    fl_s = 3 * (cfg_s.n_layer * layer_flops(bs, cfg_s.seq_len, cfg_s.n_embd,
                cfg_s.n_head, cfg_s.n_kv_head, cfg_s.head_dim, cfg_s.mlp_dim) +
                matmul_flops(bs * cfg_s.seq_len, cfg_s.padded_vocab, cfg_s.n_embd))

    @jax.jit
    def bench_sweep(params, opt_state, tokens, _cfg=cfg_s, _sched=sched_s, _lab=lab_s):
        trainable, static = split_trainable(params)

        def loss_fn(t):
            full = merge_params(t, static)
            hidden = model_forward_remat(_cfg, full, tokens)
            return chunked_lm_head_loss(hidden, full.lm_head, _lab, _cfg)

        loss, grads = jax.value_and_grad(loss_fn)(trainable)
        updates, new_opt_state = _sched.update(grads, opt_state, trainable)
        new_trainable = optax.apply_updates(trainable, updates)
        new_params = merge_params(new_trainable, static)
        return loss, new_params, new_opt_state

    r = benchmark(bench_sweep, p_s, ostate_s, tok_s, flop_count=fl_s,
                  label=f"{label} (B={bs})")
    r['n_params'] = n_p
    r['tok_per_sec'] = bs * cfg_s.seq_len / (r['wall_ms'] / 1000)

    # Estimated wall time for 20 tokens/param training run
    total_tokens = 20 * n_p
    total_steps = total_tokens / (bs * cfg_s.seq_len)
    est_hours = total_steps * (r['wall_ms'] / 1000) / 3600
    r['est_hours_20x'] = est_hours
    results_10.append(r)

# %%
# 10b. Phase 10 Summary
print("\n=== Phase 10 Summary ===")
print(f"\n  {'Label':<25s}  {'Params':>10s}  {'Wall ms':>8s}  {'MFU%':>6s}  "
      f"{'tok/s':>10s}  {'20x hrs':>8s}")
print("  " + "-" * 82)
for r in results_10:
    print(f"  {r['label']:<25s}  {r['n_params']:>10,}  {r['wall_ms']:8.2f}  "
          f"{r['mfu_pct']:5.1f}%  {r['tok_per_sec']:>10,.0f}  "
          f"{r['est_hours_20x']:7.1f}h")

# %% [markdown]
# ## Phase 11 — ~100M Non-Embed Param Architecture Sweep
#
# Fix L=8 and systematically vary D, H, F, and K to find what architecture
# choices matter most for ~100M non-embedding parameter models on TPU v6e.
#
# **Parameter counting convention (Chinchilla/Kaplan):**
# - **Embedding** (V×D) is a pure lookup table — 0 matmul FLOPs, **not counted**
# - **Unembedding** (D×V) is a real matmul (our biggest at ~78% MFU) — **counted**
# - Non-embed params = unembed (D×V) + layer params
# - For ~100M non-embed models at L=8, embed is 20-30% of total params
#   (wasted budget if not tying, but tying hurts convergence)
# - `N` in scaling laws = non-embedding params

# %%
# === Phase 11: ~100M non-embed param architecture sweep ===
print("=== Phase 11: ~100M Non-Embed Param Architecture Sweep ===")

sweep_configs_11 = [
    # (label, n_head, n_kv_head, head_dim, n_embd, mlp_dim, n_layer, batch_size)
    ("D768-F3328-B4",    3, 1, 256, 768,  3328, 8, 4),    # 99M non-embed baseline
    ("D1024-F3072-B2",   3, 1, 256, 1024, 3072, 8, 2),    # 126M, small batch
    ("D1024-F3072-B4",   3, 1, 256, 1024, 3072, 8, 4),    # 126M, peak MFU
    ("D1024-N4-F3072-B4",4, 1, 256, 1024, 3072, 8, 4),    # 130M, D=N×H (square attn proj)
    ("D1024-F3072-B8",   3, 1, 256, 1024, 3072, 8, 8),    # 126M, batch scaling
    ("D1024-F3072-B16",  3, 1, 256, 1024, 3072, 8, 16),   # 126M, batch scaling
    ("D1024-F3072-B32",  3, 1, 256, 1024, 3072, 8, 32),   # 126M, large batch
]

results_11 = []
for label, nh, nkv, hd, ne, mlp, nl, bs in sweep_configs_11:
    print(f"\n--- {label}: D={ne}, N={nh}, K={nkv}, H={hd}, F={mlp}, L={nl}, B={bs} ---")
    cfg_s = PerfConfig(batch_size=bs, n_head=nh, n_kv_head=nkv, head_dim=hd,
                       n_embd=ne, mlp_dim=mlp, n_layer=nl)
    p_s = init_full_model(cfg_s)
    n_p = count_params(p_s)
    n_ne = count_non_embed_params(p_s)
    print(f"  Total params: {n_p:,}  Non-embed params: {n_ne:,}")

    tok_s = fake_tokens(bs, cfg_s.seq_len)
    lab_s = fake_tokens(bs, cfg_s.seq_len)

    # Use optax.adamw f32 (best from Phase 9)
    tr_s, st_s = split_trainable(p_s)
    sched_s = optax.adamw(OPT_LR, b1=OPT_BETA1, b2=OPT_BETA2,
                           eps=OPT_EPS, weight_decay=OPT_WD)
    ostate_s = sched_s.init(tr_s)

    fl_s = 3 * (cfg_s.n_layer * layer_flops(bs, cfg_s.seq_len, cfg_s.n_embd,
                cfg_s.n_head, cfg_s.n_kv_head, cfg_s.head_dim, cfg_s.mlp_dim) +
                matmul_flops(bs * cfg_s.seq_len, cfg_s.padded_vocab, cfg_s.n_embd))

    @jax.jit
    def bench_sweep_11(params, opt_state, tokens, _cfg=cfg_s, _sched=sched_s, _lab=lab_s):
        trainable, static = split_trainable(params)

        def loss_fn(t):
            full = merge_params(t, static)
            hidden = model_forward_remat(_cfg, full, tokens)
            return chunked_lm_head_loss(hidden, full.lm_head, _lab, _cfg)

        loss, grads = jax.value_and_grad(loss_fn)(trainable)
        updates, new_opt_state = _sched.update(grads, opt_state, trainable)
        new_trainable = optax.apply_updates(trainable, updates)
        new_params = merge_params(new_trainable, static)
        return loss, new_params, new_opt_state

    r = benchmark(bench_sweep_11, p_s, ostate_s, tok_s, flop_count=fl_s,
                  label=f"{label} (B={bs})")
    r['n_params'] = n_p
    r['n_non_embed'] = n_ne
    r['tok_per_sec'] = bs * cfg_s.seq_len / (r['wall_ms'] / 1000)

    # Estimated wall time for 20 tokens/param training run (using non-embed count)
    total_tokens = 20 * n_ne
    total_steps = total_tokens / (bs * cfg_s.seq_len)
    est_hours = total_steps * (r['wall_ms'] / 1000) / 3600
    r['est_hours_20x'] = est_hours
    results_11.append(r)

# %%
# 11b. Phase 11 Summary
print("\n=== Phase 11 Summary ===")
print(f"\n  {'Label':<20s}  {'Non-embed':>10s}  {'Total':>10s}  {'Wall ms':>8s}  "
      f"{'MFU%':>6s}  {'tok/s':>10s}  {'20x hrs':>8s}")
print("  " + "-" * 92)
for r in results_11:
    print(f"  {r['label']:<20s}  {r['n_non_embed']:>10,}  {r['n_params']:>10,}  "
          f"{r['wall_ms']:8.2f}  {r['mfu_pct']:5.1f}%  "
          f"{r['tok_per_sec']:>10,.0f}  {r['est_hours_20x']:7.1f}h")

# %% [markdown]
# ## Ideas from scaling-book
#
# Reference: `jax-ml/scaling-book` (cloned locally in `scaling-book/`)
#
# **Remat FLOPs accounting** — Block remat costs ~8ND FLOPs (vs 6ND without remat)
# because it recomputes the forward pass during backward. Our MFU% uses analytical
# FLOP counts (3× forward for fwd+bwd) which correctly reflects actual compute work.
#
# **Arithmetic intensity roofline** — The book defines `intensity = FLOPs / bytes`.
# For our matmul benchmarks we already compute HBM bytes — could plot a roofline
# diagram showing which ops are compute-bound vs memory-bound. Critical intensity
# for v6e = 918e12 / 1600e9 ≈ 574 FLOPs/byte. Ops above this line are compute-bound.
#
# **Attention dominance threshold** — Book says attention FLOPs dominate when
# `T > 8D` (assuming standard `F=4D` and `D=NH`). For our default config (D=1024),
# that's `T > 8192`. Our T=2048 is well below — attention is a small fraction of
# total FLOPs, MLP and projections dominate.
#
# **`jax.checkpoint` policies** — `dots_with_no_batch_dims_saveable` saves only
# big matmul outputs (~7 checkpoints/layer vs 1 for block remat, ~20 for no remat).
# This trades off between the 6ND and 8ND extremes — worth benchmarking.
#
# **int8 inference matmuls** — TPU v6e int8 = 2× bf16 throughput. The critical
# batch size for compute-bound regime stays the same (proportional reduction in
# both FLOPs and bytes).

# %%
# === All outputs — copy/paste this cell's output into Claude Code ===
print("=" * 90)
print("  ALL BENCHMARK RESULTS")
print("=" * 90)
print()
print(f"Config: B={cfg.batch_size}, T={cfg.seq_len}, D={cfg.n_embd}, "
      f"N={cfg.n_head}, K={cfg.n_kv_head}, H={cfg.head_dim}, "
      f"F={cfg.mlp_dim}, V={cfg.vocab_size}, L={cfg.n_layer}, "
      f"softcap={cfg.softcap}, splash_bs={cfg.splash_block_size}, "
      f"lm_chunks={cfg.num_lm_head_chunks}")
print(f"TPU: peak={PEAK_TFLOPS} TFLOPS, HBM={HBM_GB} GB, BW={HBM_BW_GBS} GB/s")
print()
print(f"{'#':<4s} {'Label':<45s} {'Wall ms':>8s} {'TFLOP/s':>8s} {'MFU%':>6s} {'BW%':>6s}")
print("-" * 84)
for i, r in enumerate(ALL_RESULTS):
    mxu = f"{r['mfu_pct']:5.1f}" if r['tflops'] > 0 else "  n/a"
    tf = f"{r['tflops']:7.1f}" if r['tflops'] > 0 else "    n/a"
    bw = f"{r['hbm_bw_pct']:5.1f}" if r['hbm_bw_pct'] > 0 else "  n/a"
    print(f"{i:<4d} {r['label']:<45s} {r['wall_ms']:8.2f} {tf} {mxu} {bw}")
print()
print(f"Total benchmarks: {len(ALL_RESULTS)}")
print("=" * 90)
