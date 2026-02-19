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
# benchmarks **MXU utilization** and **HBM usage** at each stage.
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
# > **HBM BW%:** shows what fraction of the ~820 GB/s peak bandwidth
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
    """Run fn repeatedly and report wall time, TFLOP/s, MXU%, HBM bandwidth%.

    Args:
        fn: callable (JIT-compiled or not — warmup handles compilation)
        *args: arguments forwarded to fn
        warmup: number of warmup calls (absorbs JIT compilation)
        repeats: number of timed calls
        flop_count: manual FLOP count (int); None = skip MXU calculation
        hbm_bytes: total bytes read+written per call (int); None = skip BW calc
        label: display label for printing

    Returns:
        dict with wall_ms, tflops, mxu_pct, hbm_bw_gbs, hbm_bw_pct
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

    # FLOP/s and MXU
    tflops = flop_count / (wall_s * 1e12) if flop_count else 0.0
    mxu_pct = (tflops / PEAK_TFLOPS * 100
               if flop_count and PEAK_TFLOPS else 0.0)

    # HBM bandwidth utilization
    hbm_bw_gbs = hbm_bytes / (wall_s * 1e9) if hbm_bytes else 0.0
    hbm_bw_pct = hbm_bw_gbs / HBM_BW_GBS * 100 if hbm_bytes else 0.0

    result = dict(label=label, wall_ms=wall_ms, tflops=tflops,
                  mxu_pct=mxu_pct, hbm_bw_gbs=hbm_bw_gbs,
                  hbm_bw_pct=hbm_bw_pct)

    # Print single row
    mxu_str = f"{mxu_pct:5.1f}%" if flop_count else "  n/a"
    tflop_str = f"{tflops:6.1f}" if flop_count else "   n/a"
    bw_str = f"{hbm_bw_pct:5.1f}%" if hbm_bytes else "  n/a"
    print(f"  {label:<40s}  {wall_ms:8.2f} ms  {tflop_str} TFLOP/s  MXU {mxu_str}  "
          f"HBM BW {bw_str}")
    return result


def print_summary(results):
    """Print a formatted comparison table from benchmark results."""
    print(f"\n  {'Label':<40s}  {'Wall ms':>8s}  {'TFLOP/s':>8s}  {'MXU %':>6s}  "
          f"{'HBM BW%':>7s}")
    print("  " + "-" * 88)
    for r in results:
        mxu = f"{r['mxu_pct']:5.1f}%" if r['tflops'] > 0 else "  n/a"
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

def matmul_flops(M, N, K, batch=1):
    """FLOPs for [M,K] @ [K,N].  2*M*N*K per batch element."""
    return 2 * batch * M * N * K

def attention_flops(B, H, T, D):
    """Approximate FLOPs for QK^T + softmax + AV (full, not causal-halved)."""
    return 2 * (2 * B * H * T * T * D)   # QK^T + AV

def layer_flops(B, T, E, H, KV, D, MLP):
    """Approximate FLOPs for one transformer layer."""
    tok = B * T
    q  = 2 * tok * E * H * D             # Q projection
    k  = 2 * tok * E * KV * D            # K projection
    v  = 2 * tok * E * KV * D            # V projection
    att = attention_flops(B, H, T, D)     # core attention
    proj = 2 * tok * H * D * E           # output projection
    gate = 2 * tok * E * MLP             # SwiGLU gate
    up   = 2 * tok * E * MLP             # SwiGLU up
    down = 2 * tok * MLP * E             # SwiGLU down
    return q + k + v + att + proj + gate + up + down

# %% [markdown]
# ## Phase 1 — Matmul Baseline
#
# Establish the MXU ceiling with pure matmuls.
# - 256-aligned sizes should hit best utilization
# - Non-aligned sizes show the padding penalty

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
# 1b. Non-aligned sizes — alignment penalty
print("=== Non-aligned matmul ===")
results_1b = []
for size in [255, 256, 257, 300, 384, 500, 512, 1000, 1024]:
    a = jax.random.normal(jax.random.key(0), (size, size), dtype=jnp.bfloat16)
    b = jax.random.normal(jax.random.key(1), (size, size), dtype=jnp.bfloat16)

    @jax.jit
    def mm(a, b):
        return a @ b

    hbm = 3 * size * size * 2
    r = benchmark(mm, a, b, flop_count=matmul_flops(size, size, size),
                  hbm_bytes=hbm, label=f"matmul {size}x{size}")
    results_1b.append(r)
print_summary(results_1b)

# %%
# 1c. Batched matmul — simulating transformer projections
print("=== Batched matmul (transformer-shaped) ===")
results_1c = []
shapes = [
    (8, 2048, 1024, 1024, "B=8 hidden->hidden"),
    (8, 2048, 3072, 1024, "B=8 hidden->mlp"),
    (8, 2048, 1024, 3072, "B=8 mlp->hidden"),
    (8, 2048, 32768, 1024, "B=8 hidden->vocab"),
]
for B, M, N, K, desc in shapes:
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
# Isolate each building block and measure MXU independently.
# - **RMSNorm / RoPE**: memory-bound (expect ~0% MXU)
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
print(f"Config: B={cfg.batch_size}, T={cfg.seq_len}, E={cfg.n_embd}, "
      f"H={cfg.n_head}, KV={cfg.n_kv_head}, D={cfg.head_dim}, "
      f"MLP={cfg.mlp_dim}, V={cfg.vocab_size}, L={cfg.n_layer}")

# %%
# 2a. RMSNorm — pure elementwise, expect ~0% MXU
print("=== RMSNorm ===")
x = fake_hidden(cfg.batch_size, cfg.seq_len, cfg.n_embd)

@jax.jit
def bench_rmsnorm(x):
    return rms_norm(x)

r_norm = benchmark(bench_rmsnorm, x, flop_count=None, label="RMSNorm")

# %%
# 2b. RoPE — elementwise multiply/concat, expect ~0% MXU
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
        bs = cfg.splash_block_size
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
# 2g. Attention — Pallas flash attention
print("=== Attention (pallas flash) ===")

from jax.experimental.pallas.ops.tpu.flash_attention import flash_attention

@jax.jit
def bench_attn_pallas(x, params, cos, sin):
    with jax.named_scope('attn_pallas'):
        h = rms_norm(x)
        q = jnp.einsum('btd,dhk->bhtk', h, params.c_q)
        k = jnp.einsum('btd,dhk->bhtk', h, params.c_k)
        v = jnp.einsum('btd,dhk->bhtk', h, params.c_v)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        q = rms_norm(q)
        k = rms_norm(k)
        k_exp, v_exp = _expand_kv(k, v, cfg.n_head, cfg.n_kv_head)
        attn_out = flash_attention(
            q, k_exp, v_exp, causal=True, sm_scale=cfg.head_dim ** -0.5)
        return jnp.einsum('bhtd,hde->bte', attn_out, params.c_proj)

r_attn_pallas = benchmark(bench_attn_pallas, x, attn_params, cos_b, sin_b,
                           flop_count=total_attn_flops, label="Attention (pallas flash)")

# %%
# 2h. Component comparison
print("\n=== Phase 2 Summary ===")
print_summary([r_norm, r_rope, r_mlp, r_attn_ein, r_attn_jax, r_attn_splash, r_attn_pallas])

# %% [markdown]
# ### Ideas to try
# - **Fused RMSNorm+Linear** as a single Pallas kernel (saves one HBM read/write roundtrip)
# - **Remove QK-norm** from attention — saves 2 RMSNorm calls on Q and K
# - **Vary head_dim**: try 64, 128, 256, 512 — how does per-component MXU change?
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
        bs = cfg.splash_block_size
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
# Does MXU change with depth? How does HBM scale?

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
# - Embedding: pure memory lookup (0% MXU)
# - LM head: large matmul `(B*T, n_embd) @ (n_embd, vocab)` — high MXU
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
    for i in range(cfg.n_layer):
        x = jax.checkpoint(single_layer_forward)(
            cfg, params.layers[i], x, cos, sin, attn_impl='splash')
    return rms_norm(x)


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
print(f"  Remat HBM savings:          "
      f"{r_fwd_bwd['hbm_peak_gb'] - r_remat['hbm_peak_gb']:.2f} GiB")

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

for sl in [512, 1024, 2048]:
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

for bs in [128, 256, 512, 1024]:
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
# - **`mlp_dim` expansion ratio**: 2x, 3x, 4x — how does MLP FLOPs fraction affect overall MXU?
# - **`n_layer` vs `n_embd`**: given a fixed param budget, is it better to go deep or wide?
# - **Tok/s vs MXU%**: these are different! MXU measures compute efficiency, tok/s measures throughput

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
# attention, which is a pure memory operation with 0% MXU.
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
# - MXU utilization (actual vs peak)
# - Memory timeline (stack, heap, fragmentation)
# - Op-level breakdown (which ops are slowest)
# - Idle gaps (input-bound vs compute-bound)

# %%
# === Final summary ===
print("=" * 90)
print("  FULL SESSION SUMMARY")
print("=" * 90)

all_results = []
for name, rlist in [
    ("Phase 1 — Matmul", results_1a[-2:]),      # last two large sizes
    ("Phase 2 — Components", [r_mlp, r_attn_splash]),
    ("Phase 3 — Single layer", [r_layer]),
    ("Phase 4 — Depth", results_4a[-1:]),         # 24 layers
    ("Phase 5 — LM head", [r_lm, r_lm_chunked]),
    ("Phase 6 — Fwd/Bwd", [r_fwd, r_fwd_bwd, r_remat]),
]:
    for r in rlist:
        all_results.append(r)

print_summary(all_results)
print("Done!  Use XProf (Phase 8.9) for detailed hardware traces.")
