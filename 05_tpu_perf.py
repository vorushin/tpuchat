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
#   accelerator: TPU
#   colab:
#     gpuType: V6E1
#     machine_shape: hm
# ---

# %% [markdown]
# <a href="https://colab.research.google.com/github/vorushin/tpuchat/blob/master/05_tpu_perf.ipynb?flush_caches=true" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
#
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
# !pip install -q "jax[tpu]" optax tensorboard tensorboard-plugin-profile

# %% [markdown]
# ## Phase 0 — Prerequisites
#
# All imports, constants, utility classes, model primitives, and function
# definitions live here. Run this section first, then jump to any phase.

# %%
import functools as ft
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_mask, splash_attention_kernel)

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

# %%
# === Model primitives ===

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
    """All dims 256-aligned for MXU.  164M param model (130M non-embed)."""
    batch_size: int = 64
    microbatch_size: int = 4
    seq_len: int = 2048
    n_head: int = 4
    n_kv_head: int = 1
    head_dim: int = 256
    n_embd: int = 1024       # n_head * head_dim
    mlp_dim: int = 3072      # 3x expansion for SwiGLU
    vocab_size: int = 32768
    n_layer: int = 8
    softcap: float = 15.0
    logit_dtype: str = 'bf16'    # 'bf16' or 'fp32' — bf16 is ~4% faster
    splash_block_size: int = 1024
    num_lm_head_chunks: int = 8

    @property
    def num_microbatches(self):
        return self.batch_size // self.microbatch_size

cfg = PerfConfig()
assert cfg.vocab_size % 256 == 0, f"vocab_size must be divisible by 256, got {cfg.vocab_size}"
assert cfg.n_embd == cfg.n_head * cfg.head_dim, \
    f'n_embd ({cfg.n_embd}) must equal n_head * head_dim ({cfg.n_head * cfg.head_dim})'
print(f"Config: B={cfg.batch_size}, mb={cfg.microbatch_size}, T={cfg.seq_len}, "
      f"D={cfg.n_embd}, N={cfg.n_head}, K={cfg.n_kv_head}, H={cfg.head_dim}, "
      f"F={cfg.mlp_dim}, V={cfg.vocab_size}, L={cfg.n_layer}")

# %%
# === Precomputed RoPE tables (used by multiple phases) ===
cos, sin = precompute_rope(cfg.seq_len, cfg.head_dim)
cos_b = cos[None, None, :, :]
sin_b = sin[None, None, :, :]

# %%
# === Param initializers ===

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


def init_all_layers(cfg, n_layers, seed=42):
    layers = dot_dict()
    for i in range(n_layers):
        layers[i] = init_layer_params(cfg, seed=seed + i * 7)
    return layers

# %%
# === Single layer forward ===

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
# === Multi-layer model ===

def multi_layer_forward(cfg, layers, n_layers, x, cos, sin, attn_impl='splash'):
    for i in range(n_layers):
        x = single_layer_forward(cfg, layers[i], x, cos, sin, attn_impl=attn_impl)
    return rms_norm(x)

# %%
# === Chunked LM head loss ===

def _logit_dtype(config):
    return jnp.float32 if config.logit_dtype == 'fp32' else jnp.bfloat16


def _logits_from_chunk(h_chunk, lm_head, config):
    logits = jnp.einsum('td,dv->tv', h_chunk, lm_head,
                        preferred_element_type=_logit_dtype(config))
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

# %%
# === Full model ===

def init_full_model(cfg, seed=42):
    """Initialize all model params (embed + layers + lm_head)."""
    key = jax.random.key(seed)
    params = dot_dict()
    key, k1, k2 = jax.random.split(key, 3)
    params.wte = jax.random.normal(k1, (cfg.vocab_size, cfg.n_embd), dtype=jnp.bfloat16)
    params.lm_head = jax.random.normal(k2, (cfg.n_embd, cfg.vocab_size),
                                        dtype=jnp.bfloat16) * 0.001
    params.layers = init_all_layers(cfg, cfg.n_layer, seed=seed + 100)
    return params


def model_forward(cfg, params, tokens, attn_impl='splash'):
    """Full forward: embed -> layers -> final_norm.  Returns hidden (B,T,E)."""
    B, T = tokens.shape
    cos, sin = precompute_rope(T, cfg.head_dim)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    x = rms_norm(params.wte[tokens])
    for i in range(cfg.n_layer):
        x = single_layer_forward(cfg, params.layers[i], x, cos, sin, attn_impl=attn_impl)
    return rms_norm(x)


def model_forward_remat(cfg, params, tokens):
    """Same as model_forward but with jax.checkpoint on each layer."""
    B, T = tokens.shape
    cos, sin = precompute_rope(T, cfg.head_dim)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    x = rms_norm(params.wte[tokens])
    layer_fn = ft.partial(single_layer_forward, cfg, attn_impl='splash')
    for i in range(cfg.n_layer):
        x = jax.checkpoint(layer_fn)(params.layers[i], x, cos, sin)
    return rms_norm(x)

# %%
# === Optimizer utilities ===

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
    """Count total parameters."""
    return sum(p.size for p in jax.tree.leaves(params) if isinstance(p, jax.Array))


def count_non_embed_params(params):
    """Non-embedding params (unembed + layers). Excludes wte (lookup table)."""
    return count_params(params) - params.wte.size

# %%
# === Optimizer hyperparams (matching 02_train.py defaults) ===
OPT_LR = 3e-4
OPT_BETA1 = 0.9
OPT_BETA2 = 0.95
OPT_EPS = 1e-8
OPT_WD = 0.1

# %%
# === Microbatched train step ===

def make_bench_train_step(optimizer, cfg, labels, attn_impl='splash'):
    """Create a JIT-compiled train step with microbatch gradient accumulation."""
    @jax.jit
    def train_step(params, opt_state, tokens, _opt=optimizer):
        num_mb = cfg.num_microbatches
        x_micro = tokens.reshape(num_mb, cfg.microbatch_size, cfg.seq_len)
        y_micro = labels.reshape(num_mb, cfg.microbatch_size, cfg.seq_len)

        def loss_fn(params, x_mb, y_mb):
            hidden = model_forward(cfg, params, x_mb, attn_impl=attn_impl)
            return chunked_lm_head_loss(hidden, params.lm_head, y_mb, cfg)

        def microbatch_step(grad_acc, data):
            x_mb, y_mb = data
            loss, grads = jax.value_and_grad(loss_fn)(params, x_mb, y_mb)
            grad_acc = jax.tree.map(jax.lax.add, grad_acc, grads)
            return grad_acc, loss

        grad_init = jax.tree.map(jnp.zeros_like, params)
        grads, losses = jax.lax.scan(microbatch_step, grad_init, (x_micro, y_micro))
        grads = jax.tree.map(lambda g: g / num_mb, grads)
        loss = jnp.mean(losses)

        updates, new_opt_state = _opt.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return loss, new_params, new_opt_state
    return train_step

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
# Component benchmarks use microbatch_size (B=4) to avoid OOM on attention matrices
B_comp = cfg.microbatch_size

# 2a. RMSNorm — pure elementwise, expect ~0% MFU
print("=== RMSNorm ===")
x = fake_hidden(B_comp, cfg.seq_len, cfg.n_embd)

@jax.jit
def bench_rmsnorm(x):
    return rms_norm(x)

r_norm = benchmark(bench_rmsnorm, x, flop_count=None, label="RMSNorm")

# %%
# 2b. RoPE — elementwise multiply/concat, expect ~0% MFU
print("=== RoPE ===")

q = jax.random.normal(jax.random.key(0),
    (B_comp, cfg.n_head, cfg.seq_len, cfg.head_dim), dtype=jnp.bfloat16)

@jax.jit
def bench_rope(q, cos, sin):
    return apply_rope(q, cos, sin)

r_rope = benchmark(bench_rope, q, cos_b, sin_b, flop_count=None, label="RoPE")

# %%
# 2c. SwiGLU MLP — 3 large matmuls (gate, up, down)
print("=== SwiGLU MLP ===")

mlp_params = init_mlp_params(cfg)
x = fake_hidden(B_comp, cfg.seq_len, cfg.n_embd)

@jax.jit
def bench_mlp(x, params):
    with jax.named_scope('mlp'):
        h = rms_norm(x)
        gate = jax.nn.silu(jnp.einsum('btd,dh->bth', h, params.w_gate))
        up = jnp.einsum('btd,dh->bth', h, params.w_up)
        return jnp.einsum('bth,hd->btd', gate * up, params.w_down)

tok = B_comp * cfg.seq_len
mlp_flops = 3 * 2 * tok * cfg.n_embd * cfg.mlp_dim
r_mlp = benchmark(bench_mlp, x, mlp_params, flop_count=mlp_flops, label="SwiGLU MLP")

# %%
# 2d. Attention — einsum variant (manual QK^T + softmax + AV)
print("=== Attention (einsum) ===")

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

tok = B_comp * cfg.seq_len
proj_flops = (2 * tok * cfg.n_embd * cfg.n_head * cfg.head_dim +
              2 * tok * cfg.n_embd * cfg.n_kv_head * cfg.head_dim +
              2 * tok * cfg.n_embd * cfg.n_kv_head * cfg.head_dim +
              2 * tok * cfg.n_head * cfg.head_dim * cfg.n_embd)
attn_core = attention_flops(B_comp, cfg.n_head, cfg.seq_len, cfg.head_dim)
total_attn_flops = proj_flops + attn_core

x = fake_hidden(B_comp, cfg.seq_len, cfg.n_embd)
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

# %%
# 3a. Full layer benchmark
print("=== Single layer (splash) ===")
layer_p = init_layer_params(cfg)
x = fake_hidden(B_comp, cfg.seq_len, cfg.n_embd)
lf = layer_flops(B_comp, cfg.seq_len, cfg.n_embd,
                 cfg.n_head, cfg.n_kv_head, cfg.head_dim, cfg.mlp_dim)

@jax.jit
def bench_layer(x, layer, cos, sin):
    return single_layer_forward(cfg, layer, x, cos, sin, attn_impl='splash')

r_layer = benchmark(bench_layer, x, layer_p, cos_b, sin_b,
                    flop_count=lf, label="Full layer (splash)")

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
# 4a. Depth sweep
print("=== Depth sweep ===")
results_4a = []
x = fake_hidden(B_comp, cfg.seq_len, cfg.n_embd)
lf = layer_flops(B_comp, cfg.seq_len, cfg.n_embd,
                 cfg.n_head, cfg.n_kv_head, cfg.head_dim, cfg.mlp_dim)

for n_layers in [1, 2, 4, 8, 12]:
    layers = init_all_layers(cfg, n_layers)
    fl = n_layers * lf

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
# Use microbatch_size for component benchmarks (non-chunked lm_head would OOM at B=64)
B5 = cfg.microbatch_size

# 5a. Embedding lookup
print("=== Embedding ===")
wte = jax.random.normal(jax.random.key(0),
    (cfg.vocab_size, cfg.n_embd), dtype=jnp.bfloat16)
tokens = fake_tokens(B5, cfg.seq_len)

@jax.jit
def bench_embed(tokens, wte):
    return rms_norm(wte[tokens])

r_embed = benchmark(bench_embed, tokens, wte, flop_count=None,
                    label="Embedding + norm")

# %%
# 5b. LM head — non-chunked
print("=== LM head (non-chunked) ===")
lm_head = jax.random.normal(jax.random.key(1),
    (cfg.n_embd, cfg.vocab_size), dtype=jnp.bfloat16) * 0.001
hidden = fake_hidden(B5, cfg.seq_len, cfg.n_embd)
labels = fake_tokens(B5, cfg.seq_len)

@jax.jit
def bench_lm_head(hidden, lm_head, labels):
    logits = jnp.einsum('btd,dv->btv', hidden, lm_head,
                        preferred_element_type=_logit_dtype(cfg))
    logits = cfg.softcap * jnp.tanh(logits / cfg.softcap)
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, labels))

lm_flops = matmul_flops(B5 * cfg.seq_len, cfg.vocab_size, cfg.n_embd)
r_lm = benchmark(bench_lm_head, hidden, lm_head, labels,
                 flop_count=lm_flops, label="LM head (non-chunked)")

# %%
# 5c. Chunked LM head loss
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
# Use microbatch_size as batch for single-pass benchmarks (no gradient accumulation)
cfg_6 = PerfConfig(batch_size=cfg.microbatch_size, microbatch_size=cfg.microbatch_size)
full_params = init_full_model(cfg_6)
tokens = fake_tokens(cfg_6.batch_size, cfg_6.seq_len)
labels = fake_tokens(cfg_6.batch_size, cfg_6.seq_len)

total_model_flops = (cfg_6.n_layer * layer_flops(cfg_6.batch_size, cfg_6.seq_len, cfg_6.n_embd,
                     cfg_6.n_head, cfg_6.n_kv_head, cfg_6.head_dim, cfg_6.mlp_dim) +
                     matmul_flops(cfg_6.batch_size * cfg_6.seq_len, cfg_6.vocab_size, cfg_6.n_embd))
bwd_flops = 3 * total_model_flops

# %%
# 6a. Forward only
print(f"=== Forward only (B={cfg_6.batch_size}) ===")

@jax.jit
def bench_fwd(params, tokens):
    hidden = model_forward(cfg_6, params, tokens)
    return chunked_lm_head_loss(hidden, params.lm_head, labels, cfg_6)

r_fwd = benchmark(bench_fwd, full_params, tokens,
                  flop_count=total_model_flops, label="Forward only")

# %%
# 6b. Forward + backward
print("=== Forward + Backward ===")

@jax.jit
def bench_fwd_bwd(params, tokens):
    def loss_fn(p):
        hidden = model_forward(cfg_6, p, tokens)
        return chunked_lm_head_loss(hidden, p.lm_head, labels, cfg_6)
    return jax.value_and_grad(loss_fn)(params)

r_fwd_bwd = benchmark(bench_fwd_bwd, full_params, tokens,
                       flop_count=bwd_flops, label="Forward+Backward")

# %%
# 6c. Forward + backward with remat
print("=== Forward + Backward (remat) ===")

@jax.jit
def bench_fwd_bwd_remat(params, tokens):
    def loss_fn(p):
        hidden = model_forward_remat(cfg_6, p, tokens)
        return chunked_lm_head_loss(hidden, p.lm_head, labels, cfg_6)
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
print(f"\n  Total params: {n_params:,}")
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
    cfg_bs = PerfConfig(batch_size=bs, microbatch_size=bs)
    p = init_full_model(cfg_bs)
    tok = fake_tokens(bs, cfg_bs.seq_len)
    lab = fake_tokens(bs, cfg_bs.seq_len)
    fl = 3 * (cfg_bs.n_layer * layer_flops(bs, cfg_bs.seq_len, cfg_bs.n_embd,
              cfg_bs.n_head, cfg_bs.n_kv_head, cfg_bs.head_dim, cfg_bs.mlp_dim) +
              matmul_flops(bs * cfg_bs.seq_len, cfg_bs.vocab_size, cfg_bs.n_embd))

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
# 7b. Head dim alignment: 128 (8 heads) vs 256 (4 heads)
print("\n=== Head dim alignment ===")
results_7b = []

for hd, nh in [(128, 8), (256, 4)]:
    cfg_hd = PerfConfig(batch_size=4, microbatch_size=4, head_dim=hd,
                         n_head=nh, n_kv_head=max(1, nh // 4), n_embd=nh * hd)
    p = init_full_model(cfg_hd)
    tok = fake_tokens(cfg_hd.batch_size, cfg_hd.seq_len)
    lab = fake_tokens(cfg_hd.batch_size, cfg_hd.seq_len)
    fl = 3 * (cfg_hd.n_layer * layer_flops(cfg_hd.batch_size, cfg_hd.seq_len,
              cfg_hd.n_embd, cfg_hd.n_head, cfg_hd.n_kv_head,
              cfg_hd.head_dim, cfg_hd.mlp_dim) +
              matmul_flops(cfg_hd.batch_size * cfg_hd.seq_len,
                           cfg_hd.vocab_size, cfg_hd.n_embd))

    @jax.jit
    def bench(p, tok, _lab=lab, _cfg=cfg_hd):
        def loss_fn(params):
            hidden = model_forward(_cfg, params, tok)
            return chunked_lm_head_loss(hidden, params.lm_head, _lab, _cfg)
        return jax.value_and_grad(loss_fn)(p)

    r = benchmark(bench, p, tok, flop_count=fl,
                  label=f"head_dim={hd}, n_head={nh}, D={nh*hd}")
    r['tok_per_sec'] = cfg_hd.batch_size * cfg_hd.seq_len / (r['wall_ms'] / 1000)
    results_7b.append(r)

print_summary(results_7b)

# %%
# 7c. GQA ratio comparison
print("\n=== GQA ratio ===")
results_7c = []

for n_kv in [1, 2, 4]:
    cfg_gqa = PerfConfig(batch_size=4, microbatch_size=4, n_kv_head=n_kv)
    p = init_full_model(cfg_gqa)
    tok = fake_tokens(cfg_gqa.batch_size, cfg_gqa.seq_len)
    lab = fake_tokens(cfg_gqa.batch_size, cfg_gqa.seq_len)
    fl = 3 * (cfg_gqa.n_layer * layer_flops(cfg_gqa.batch_size, cfg_gqa.seq_len,
              cfg_gqa.n_embd, cfg_gqa.n_head, n_kv,
              cfg_gqa.head_dim, cfg_gqa.mlp_dim) +
              matmul_flops(cfg_gqa.batch_size * cfg_gqa.seq_len,
                           cfg_gqa.vocab_size, cfg_gqa.n_embd))

    @jax.jit
    def bench(p, tok, _lab=lab, _cfg=cfg_gqa):
        def loss_fn(params):
            hidden = model_forward(_cfg, params, tok)
            return chunked_lm_head_loss(hidden, params.lm_head, _lab, _cfg)
        return jax.value_and_grad(loss_fn)(p)

    r = benchmark(bench, p, tok, flop_count=fl,
                  label=f"n_kv_head={n_kv} (ratio {cfg.n_head}:{n_kv})")
    r['tok_per_sec'] = cfg_gqa.batch_size * cfg_gqa.seq_len / (r['wall_ms'] / 1000)
    results_7c.append(r)

print_summary(results_7c)

# %%
# 7d. Attention implementation comparison (single layer)
print("\n=== Attention implementation comparison (single layer) ===")
results_7d = []
x = fake_hidden(B_comp, cfg.seq_len, cfg.n_embd)
layer_p = init_layer_params(cfg)

for impl in ['einsum', 'jax', 'splash']:
    @jax.jit
    def bench_fn(x, layer, cos, sin, _impl=impl):
        return single_layer_forward(cfg, layer, x, cos, sin, attn_impl=_impl)

    r = benchmark(bench_fn, x, layer_p, cos_b, sin_b,
                  flop_count=lf, label=f"attn_impl={impl}")
    results_7d.append(r)

print_summary(results_7d)

# %%
# 7e. Attention implementation — full train step (fwd+bwd+optimizer, microbatched)
print("\n=== Attention implementation (full train step, B=64 microbatched) ===")
results_7e = []

fl_7e = 3 * (cfg.n_layer * layer_flops(cfg.batch_size, cfg.seq_len, cfg.n_embd,
              cfg.n_head, cfg.n_kv_head, cfg.head_dim, cfg.mlp_dim) +
              matmul_flops(cfg.batch_size * cfg.seq_len, cfg.vocab_size, cfg.n_embd))

for impl in ['splash', 'jax', 'einsum']:
    p = init_full_model(cfg)
    tok = fake_tokens(cfg.batch_size, cfg.seq_len)
    lab = fake_tokens(cfg.batch_size, cfg.seq_len)

    sched = optax.adamw(OPT_LR, b1=OPT_BETA1, b2=OPT_BETA2,
                         eps=OPT_EPS, weight_decay=OPT_WD)
    ostate = sched.init(p)
    step_fn = make_bench_train_step(sched, cfg, lab, attn_impl=impl)

    r = benchmark(step_fn, p, ostate, tok, flop_count=fl_7e,
                  label=f"train_step attn={impl}")
    r['tok_per_sec'] = cfg.batch_size * cfg.seq_len / (r['wall_ms'] / 1000)
    results_7e.append(r)

print_summary(results_7e)
print("  Throughput:")
for r in results_7e:
    print(f"    {r['label']:<40s}  {r['tok_per_sec']:,.0f} tok/s")

# %% [markdown]
# ### Ideas to try
# - **Combined batch_size × seq_len grid** — find the throughput-maximizing combo
# - **`n_embd` sweep**: 512, 768, 1024, 1536, 2048 (all 256-aligned)
# - **`mlp_dim` expansion ratio**: 2x, 3x, 4x — how does MLP FLOPs fraction affect overall MFU?
# - **`n_layer` vs `n_embd`**: given a fixed param budget, is it better to go deep or wide?
# - **Tok/s vs MFU%**: these are different! MFU measures compute efficiency, tok/s measures throughput

# %% [markdown]
# ## Phase 8 — Optimizer Step Benchmarks
#
# Measure full train step (fwd + bwd + optimizer) with microbatching (B=64, mb=4).
# Compare manual AdamW (per-leaf loop) vs optax.adamw variants.
# With L=8 + B=64, the manual loop overhead should be dramatic (~1.5-2x).

# %%
# === Phase 8: Optimizer step benchmarks ===
print("=== Phase 8: Optimizer Step Benchmarks ===")
print(f"Config: B={cfg.batch_size}, mb={cfg.microbatch_size}, T={cfg.seq_len}, "
      f"D={cfg.n_embd}, L={cfg.n_layer}")

full_params_9 = init_full_model(cfg)
n_params_9 = count_params(full_params_9)
print(f"Total params: {n_params_9:,}")

opt_tokens = fake_tokens(cfg.batch_size, cfg.seq_len)
opt_labels = fake_tokens(cfg.batch_size, cfg.seq_len)

total_model_flops_9 = (cfg.n_layer * layer_flops(cfg.batch_size, cfg.seq_len, cfg.n_embd,
                       cfg.n_head, cfg.n_kv_head, cfg.head_dim, cfg.mlp_dim) +
                       matmul_flops(cfg.batch_size * cfg.seq_len, cfg.vocab_size, cfg.n_embd))
bwd_flops_9 = 3 * total_model_flops_9

# %%
# 8a-baseline. Fwd+Bwd only (with microbatching, no optimizer)
print("\n--- 9a-baseline. Fwd+Bwd (microbatched) baseline ---")

@jax.jit
def bench_fwd_bwd_9(params, tokens):
    num_mb = cfg.num_microbatches
    x_micro = tokens.reshape(num_mb, cfg.microbatch_size, cfg.seq_len)
    y_micro = opt_labels.reshape(num_mb, cfg.microbatch_size, cfg.seq_len)

    def loss_fn(params, x_mb, y_mb):
        hidden = model_forward(cfg, params, x_mb)
        return chunked_lm_head_loss(hidden, params.lm_head, y_mb, cfg)

    def microbatch_step(grad_acc, data):
        x_mb, y_mb = data
        loss, grads = jax.value_and_grad(loss_fn)(params, x_mb, y_mb)
        grad_acc = jax.tree.map(jax.lax.add, grad_acc, grads)
        return grad_acc, loss

    grad_init = jax.tree.map(jnp.zeros_like, params)
    grads, losses = jax.lax.scan(microbatch_step, grad_init, (x_micro, y_micro))
    grads = jax.tree.map(lambda g: g / num_mb, grads)
    return jnp.mean(losses), grads

r_fwd_bwd_9 = benchmark(bench_fwd_bwd_9, full_params_9, opt_tokens,
                         flop_count=bwd_flops_9, label="Fwd+Bwd (microbatched)")

# %%
# 8a. Manual AdamW (per-leaf loop — traces ~58 separate update ops into XLA)
print("\n--- 9a. Manual AdamW ---")

manual_params = init_full_model(cfg)
manual_opt_state = jax.tree.map(init_adam_state, manual_params)

@jax.jit
def bench_manual_adamw(params, opt_state, tokens, lr_mult):
    num_mb = cfg.num_microbatches
    x_micro = tokens.reshape(num_mb, cfg.microbatch_size, cfg.seq_len)
    y_micro = opt_labels.reshape(num_mb, cfg.microbatch_size, cfg.seq_len)

    def loss_fn(params, x_mb, y_mb):
        hidden = model_forward(cfg, params, x_mb)
        return chunked_lm_head_loss(hidden, params.lm_head, y_mb, cfg)

    def microbatch_step(grad_acc, data):
        x_mb, y_mb = data
        loss, grads = jax.value_and_grad(loss_fn)(params, x_mb, y_mb)
        grad_acc = jax.tree.map(jax.lax.add, grad_acc, grads)
        return grad_acc, loss

    grad_init = jax.tree.map(jnp.zeros_like, params)
    grads, losses = jax.lax.scan(microbatch_step, grad_init, (x_micro, y_micro))
    grads = jax.tree.map(lambda g: g / num_mb, grads)
    loss = jnp.mean(losses)

    # Per-leaf AdamW update (traces ~58 separate update ops)
    is_opt_leaf = lambda x: isinstance(x, dot_dict) and 'mu' in x
    t_leaves, t_treedef = jax.tree.flatten(params)
    g_leaves, _ = jax.tree.flatten(grads)
    o_leaves, o_treedef = jax.tree.flatten(opt_state, is_leaf=is_opt_leaf)

    new_t_leaves, new_o_leaves = [], []
    for p, g, s in zip(t_leaves, g_leaves, o_leaves):
        new_p, new_s = adamw_step(OPT_LR, OPT_BETA1, OPT_BETA2, OPT_EPS,
                                   OPT_WD, lr_mult, p, g, s)
        new_t_leaves.append(new_p)
        new_o_leaves.append(new_s)

    new_params = t_treedef.unflatten(new_t_leaves)
    new_opt_state = o_treedef.unflatten(new_o_leaves)
    return loss, new_params, new_opt_state

lr_mult = jnp.array(1.0, dtype=jnp.float32)
r_manual = benchmark(bench_manual_adamw, manual_params, manual_opt_state,
                     opt_tokens, lr_mult, flop_count=bwd_flops_9,
                     label="Manual AdamW (full step)")

# %%
# 8b. optax.adamw (f32 moments)
print("\n--- 9b. optax.adamw (f32 mu) ---")

optax_params_f32 = init_full_model(cfg)
schedule_f32 = optax.adamw(OPT_LR, b1=OPT_BETA1, b2=OPT_BETA2,
                            eps=OPT_EPS, weight_decay=OPT_WD)
optax_state_f32 = schedule_f32.init(optax_params_f32)
train_step_f32 = make_bench_train_step(schedule_f32, cfg, opt_labels)

r_optax_f32 = benchmark(train_step_f32, optax_params_f32, optax_state_f32,
                         opt_tokens, flop_count=bwd_flops_9,
                         label="optax.adamw f32 (full step)")

# %%
# 8c. optax.adamw (bf16 moments)
print("\n--- 9c. optax.adamw (bf16 mu) ---")

optax_params_bf16 = init_full_model(cfg)
schedule_bf16 = optax.adamw(OPT_LR, b1=OPT_BETA1, b2=OPT_BETA2,
                             eps=OPT_EPS, weight_decay=OPT_WD,
                             mu_dtype=jnp.bfloat16)
optax_state_bf16 = schedule_bf16.init(optax_params_bf16)
train_step_bf16 = make_bench_train_step(schedule_bf16, cfg, opt_labels)

r_optax_bf16 = benchmark(train_step_bf16, optax_params_bf16, optax_state_bf16,
                          opt_tokens, flop_count=bwd_flops_9,
                          label="optax.adamw bf16 (full step)")

# %%
# 8d. Phase 8 Summary
print("\n=== Phase 8 Summary ===")
phase9_results = [r_fwd_bwd_9, r_manual, r_optax_f32, r_optax_bf16]
print(f"\n  {'Label':<40s}  {'Wall ms':>8s}  {'MFU%':>6s}  {'tok/s':>10s}")
print("  " + "-" * 72)
for r in phase9_results:
    tok_s = cfg.batch_size * cfg.seq_len / (r['wall_ms'] / 1000)
    print(f"  {r['label']:<40s}  {r['wall_ms']:8.2f}  {r['mfu_pct']:5.1f}%  "
          f"{tok_s:>10,.0f}")

# Optimizer overhead vs fwd+bwd only
for r in [r_manual, r_optax_f32, r_optax_bf16]:
    overhead = r['wall_ms'] - r_fwd_bwd_9['wall_ms']
    pct = overhead / r_fwd_bwd_9['wall_ms'] * 100
    print(f"  {r['label']:<40s}  optimizer overhead: {overhead:+.2f} ms ({pct:+.1f}%)")

# %% [markdown]
# ## Phase 9 — ~100M Non-Embed Param Architecture Sweep
#
# Fix L=8 and systematically vary D, H, F, and K to find what architecture
# choices matter most for ~100M non-embedding parameter models on TPU v6e.
# All configs use microbatching (mb=4) and optax.adamw.
#
# **Parameter counting convention (Chinchilla/Kaplan):**
# - **Embedding** (V×D) is a pure lookup table — 0 matmul FLOPs, **not counted**
# - **Unembedding** (D×V) is a real matmul (our biggest at ~78% MFU) — **counted**
# - Non-embed params = unembed (D×V) + layer params
# - `N` in scaling laws = non-embedding params

# %%
# === Phase 9: ~100M non-embed param architecture sweep ===
print("=== Phase 9: ~100M Non-Embed Param Architecture Sweep ===")

sweep_configs_9 = [
    # (label, n_head, n_kv_head, head_dim, n_embd, mlp_dim, n_layer, batch_size, microbatch_size)
    ("D1024-N4-K1-B64",  4, 1, 256, 1024, 3072, 8, 64, 4),    # 130M — our final config
    ("D1024-N3-K1-B64",  3, 1, 256, 1024, 3072, 8, 64, 4),    # 126M — fewer Q heads
    ("D768-N3-K1-B64",   3, 1, 256, 768,  3328, 8, 64, 4),    # 99M — smaller model
    ("D1024-N4-K1-B32",  4, 1, 256, 1024, 3072, 8, 32, 4),    # 130M — smaller batch
]

results_9 = []
for label, nh, nkv, hd, ne, mlp, nl, bs, mbs in sweep_configs_9:
    print(f"\n--- {label}: D={ne}, N={nh}, K={nkv}, H={hd}, F={mlp}, L={nl}, B={bs}, mb={mbs} ---")
    cfg_s = PerfConfig(batch_size=bs, microbatch_size=mbs, n_head=nh, n_kv_head=nkv,
                       head_dim=hd, n_embd=ne, mlp_dim=mlp, n_layer=nl)
    p_s = init_full_model(cfg_s)
    n_p = count_params(p_s)
    n_ne = count_non_embed_params(p_s)
    print(f"  Total params: {n_p:,}  Non-embed params: {n_ne:,}")

    tok_s = fake_tokens(bs, cfg_s.seq_len)
    lab_s = fake_tokens(bs, cfg_s.seq_len)

    sched_s = optax.adamw(OPT_LR, b1=OPT_BETA1, b2=OPT_BETA2,
                           eps=OPT_EPS, weight_decay=OPT_WD)
    ostate_s = sched_s.init(p_s)

    fl_s = 3 * (cfg_s.n_layer * layer_flops(bs, cfg_s.seq_len, cfg_s.n_embd,
                cfg_s.n_head, cfg_s.n_kv_head, cfg_s.head_dim, cfg_s.mlp_dim) +
                matmul_flops(bs * cfg_s.seq_len, cfg_s.vocab_size, cfg_s.n_embd))

    step_fn = make_bench_train_step(sched_s, cfg_s, lab_s)

    r = benchmark(step_fn, p_s, ostate_s, tok_s, flop_count=fl_s,
                  label=f"{label}")
    r['n_params'] = n_p
    r['n_non_embed'] = n_ne
    r['tok_per_sec'] = bs * cfg_s.seq_len / (r['wall_ms'] / 1000)

    # Estimated wall time for 20 tokens/param training run (using non-embed count)
    total_tokens = 20 * n_ne
    total_steps = total_tokens / (bs * cfg_s.seq_len)
    est_hours = total_steps * (r['wall_ms'] / 1000) / 3600
    r['est_hours_20x'] = est_hours
    results_9.append(r)

# %%
# 9b. Phase 9 Summary
print("\n=== Phase 9 Summary ===")
print(f"\n  {'Label':<20s}  {'Non-embed':>10s}  {'Total':>10s}  {'Wall ms':>8s}  "
      f"{'MFU%':>6s}  {'tok/s':>10s}  {'20x hrs':>8s}")
print("  " + "-" * 92)
for r in results_9:
    print(f"  {r['label']:<20s}  {r['n_non_embed']:>10,}  {r['n_params']:>10,}  "
          f"{r['wall_ms']:8.2f}  {r['mfu_pct']:5.1f}%  "
          f"{r['tok_per_sec']:>10,.0f}  {r['est_hours_20x']:7.1f}h")

# %% [markdown]
# ## Phase 10 — XProf Trace
#
# Capture an XProf hardware trace for the final config (D1024-N4-K1-B64).
# Shows MXU utilization, memory timeline, op-level breakdown, and idle gaps.
#
# - Warmup runs first (absorbs JIT compilation)
# - Then 5 profiled steps are captured
# - View the trace in TensorBoard below

# %%
# === Phase 10: XProf profiling ===
print("=== Phase 10: XProf Trace Capture ===")
print(f"Profiling config: B={cfg.batch_size}, mb={cfg.microbatch_size}, "
      f"T={cfg.seq_len}, D={cfg.n_embd}, N={cfg.n_head}, K={cfg.n_kv_head}, "
      f"H={cfg.head_dim}, F={cfg.mlp_dim}, L={cfg.n_layer}")

prof_params = init_full_model(cfg)
prof_tokens = fake_tokens(cfg.batch_size, cfg.seq_len)
prof_labels = fake_tokens(cfg.batch_size, cfg.seq_len)

prof_sched = optax.adamw(OPT_LR, b1=OPT_BETA1, b2=OPT_BETA2,
                          eps=OPT_EPS, weight_decay=OPT_WD)
prof_opt_state = prof_sched.init(prof_params)
prof_train_step = make_bench_train_step(prof_sched, cfg, prof_labels)

# Warmup (JIT compilation)
print("Warming up (JIT compile)...")
for i in range(3):
    loss, prof_params, prof_opt_state = prof_train_step(
        prof_params, prof_opt_state, prof_tokens)
    jax.block_until_ready(loss)
    print(f"  warmup {i}: loss={float(loss):.4f}")

# Profiled steps
PROF_DIR = '/content/log_dir/xprof_perf'
print(f"\nCapturing XProf trace to '{PROF_DIR}'...")
jax.profiler.start_trace(PROF_DIR)
for i in range(5):
    loss, prof_params, prof_opt_state = prof_train_step(
        prof_params, prof_opt_state, prof_tokens)
    jax.block_until_ready(loss)
    print(f"  profiled step {i}: loss={float(loss):.4f}")
jax.profiler.stop_trace()
print(f"Trace saved to '{PROF_DIR}'.")

# %%
# === View XProf Results ===
# - **Overview page**: shows MXU% (hardware-measured) and step time breakdown
# - **Trace viewer**: zoom into individual ops (attention, MLP, optimizer)
# - **Memory viewer**: HBM usage timeline, peak usage, fragmentation
# - Large gaps between "Device Execution" blocks = INPUT BOUND
# - Tightly packed blocks = COMPUTE BOUND (good!)

# %load_ext tensorboard
# %tensorboard --logdir /content/log_dir/xprof_perf

# %%
# === All outputs — copy/paste this cell's output into Claude Code ===
print("=" * 90)
print("  ALL BENCHMARK RESULTS")
print("=" * 90)
print()
print(f"Config: B={cfg.batch_size}, mb={cfg.microbatch_size}, T={cfg.seq_len}, "
      f"D={cfg.n_embd}, N={cfg.n_head}, K={cfg.n_kv_head}, H={cfg.head_dim}, "
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
