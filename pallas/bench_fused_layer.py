# Full layer with fused RMSNorm+Linear vs baseline
# Replace rms_norm(x) -> projection with our Pallas kernel
import gc
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_mask, splash_attention_kernel)

# Fused RMSNorm+Linear kernel (from bench_rmsnorm_linear.py)
def _rmsnorm_linear_kernel(x_ref, w_ref, out_ref):
    x = x_ref[...].astype(jnp.float32)
    d = x.shape[-1]
    rms = jnp.sum(x * x, axis=-1, keepdims=True) / d
    x_norm = (x * jax.lax.rsqrt(rms + 1e-6)).astype(jnp.bfloat16)
    out_ref[...] = jnp.dot(x_norm, w_ref[...],
                           preferred_element_type=jnp.float32).astype(jnp.bfloat16)

def _rmsnorm_linear_pallas(x, w, block_m=1024, block_n=None):
    M, D = x.shape
    _, N = w.shape
    if block_n is None:
        block_n = N
    return pl.pallas_call(
        _rmsnorm_linear_kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), jnp.bfloat16),
        grid=(M // block_m, N // block_n),
        in_specs=[
            pl.BlockSpec((block_m, D), lambda i, j: (i, 0)),
            pl.BlockSpec((D, block_n), lambda i, j: (0, j)),
        ],
        out_specs=pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=['parallel', 'parallel'],
        ),
    )(x, w)

@jax.custom_vjp
def rmsnorm_linear(x, w):
    return _rmsnorm_linear_pallas(x, w)

def _rmsnorm_linear_fwd(x, w):
    return rmsnorm_linear(x, w), (x, w)

def _rmsnorm_linear_bwd(res, g):
    x, w = res
    def unfused(x, w):
        x_f32 = x.astype(jnp.float32)
        x_norm = (x_f32 * jax.lax.rsqrt(
            jnp.mean(x_f32 * x_f32, axis=-1, keepdims=True) + 1e-6
        )).astype(jnp.bfloat16)
        return jnp.dot(x_norm, w, preferred_element_type=jnp.float32).astype(jnp.bfloat16)
    _, vjp_fn = jax.vjp(unfused, x, w)
    return vjp_fn(g)

rmsnorm_linear.defvjp(_rmsnorm_linear_fwd, _rmsnorm_linear_bwd)


# Baseline layer (current code)
def layer_baseline(config, layer, x, cos, sin):
    h = rms_norm(x)
    q = jnp.einsum('btd,dnh->bnth', h, layer.c_q)
    k = jnp.einsum('btd,dnh->bnth', h, layer.c_k)
    v = jnp.einsum('btd,dnh->bnth', h, layer.c_v)
    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)
    if config.qk_norm:
        q = rms_norm(q)
        k = rms_norm(k)
    seq_len = x.shape[1]
    smask = splash_attention_mask.CausalMask(shape=(seq_len, seq_len))
    mh_mask = splash_attention_mask.MultiHeadMask(masks=[smask] * config.n_head)
    bs = min(config.splash_block_size, seq_len)
    block_sizes = splash_attention_kernel.BlockSizes(
        block_q=bs, block_kv=bs, block_q_dkv=bs, block_kv_dkv=bs,
        block_q_dq=bs, block_kv_dq=bs)
    kern = splash_attention_kernel.make_splash_mha(
        mask=mh_mask, head_shards=1, q_seq_shards=1, block_sizes=block_sizes)
    attn_out = jax.vmap(kern)(q, k, v)
    attn_out = jnp.einsum('bnth,nhd->btd', attn_out, layer.c_proj)
    x = x + attn_out
    h2 = rms_norm(x)
    gate = jax.nn.silu(jnp.einsum('btd,df->btf', h2, layer.w_gate))
    up = jnp.einsum('btd,df->btf', h2, layer.w_up)
    mlp_out = jnp.einsum('btf,fd->btd', gate * up, layer.w_down)
    x = x + mlp_out
    return x


# Fused layer: replace rms_norm+projection with fused kernel
def layer_fused(config, layer, x, cos, sin):
    B, T, D = x.shape
    M = B * T  # 8192

    # Pre-attention: fused rms_norm + QKV projections
    # Pack Q(D, N*H=1024), K(D, K*H=256), V(D, K*H=256) into one weight (D, 1536)
    W_qkv = jnp.concatenate([
        layer.c_q.reshape(D, -1),
        layer.c_k.reshape(D, -1),
        layer.c_v.reshape(D, -1),
    ], axis=1)

    qkv_flat = rmsnorm_linear(x.reshape(M, D), W_qkv)
    qkv = qkv_flat.reshape(B, T, -1)

    NH = config.n_head * config.head_dim
    KH = config.n_kv_head * config.head_dim
    q = qkv[:, :, :NH].reshape(B, T, config.n_head, config.head_dim).transpose(0, 2, 1, 3)
    k = qkv[:, :, NH:NH+KH].reshape(B, T, config.n_kv_head, config.head_dim).transpose(0, 2, 1, 3)
    v = qkv[:, :, NH+KH:].reshape(B, T, config.n_kv_head, config.head_dim).transpose(0, 2, 1, 3)

    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)
    if config.qk_norm:
        q = rms_norm(q)
        k = rms_norm(k)

    seq_len = T
    smask = splash_attention_mask.CausalMask(shape=(seq_len, seq_len))
    mh_mask = splash_attention_mask.MultiHeadMask(masks=[smask] * config.n_head)
    bs = min(config.splash_block_size, seq_len)
    block_sizes = splash_attention_kernel.BlockSizes(
        block_q=bs, block_kv=bs, block_q_dkv=bs, block_kv_dkv=bs,
        block_q_dq=bs, block_kv_dq=bs)
    kern = splash_attention_kernel.make_splash_mha(
        mask=mh_mask, head_shards=1, q_seq_shards=1, block_sizes=block_sizes)
    attn_out = jax.vmap(kern)(q, k, v)
    attn_out = jnp.einsum('bnth,nhd->btd', attn_out, layer.c_proj)
    x = x + attn_out

    # Pre-MLP: fused rms_norm + gate+up projection
    # Pack gate(D,F) and up(D,F) into one (D, 2F)
    W_gate_up = jnp.concatenate([layer.w_gate, layer.w_up], axis=1)
    gate_up_flat = rmsnorm_linear(x.reshape(M, D), W_gate_up)
    gate_up = gate_up_flat.reshape(B, T, 2, config.mlp_dim)
    gate = jax.nn.silu(gate_up[:, :, 0, :])
    up = gate_up[:, :, 1, :]
    mlp_out = jnp.einsum('btf,fd->btd', gate * up, layer.w_down)
    x = x + mlp_out
    return x


B, T, D = cfg.microbatch_size, cfg.seq_len, cfg.n_embd
x = jax.random.normal(jax.random.key(0), (B, T, D), dtype=jnp.bfloat16)
layer = params.layers[0]
cos, sin = precompute_rope(T, cfg.head_dim)
cos, sin = cos[None, None, :, :], sin[None, None, :, :]

flops_layer = layer_flops(B, T, D, cfg.n_head, cfg.n_kv_head, cfg.head_dim, cfg.mlp_dim) * 3

# Correctness check
ref = jax.jit(lambda x, l, c, s: layer_baseline(cfg, l, x, c, s))(x, layer, cos, sin)
fused = jax.jit(lambda x, l, c, s: layer_fused(cfg, l, x, c, s))(x, layer, cos, sin)
err = jnp.max(jnp.abs(ref - fused)).item()
print(f"Correctness: max abs error = {err:.2e}\n")

print("=== Single layer fwd+bwd: baseline vs fused ===\n")

fn_base = jax.jit(lambda x, l, c, s: jax.grad(
    lambda x: jnp.sum(layer_baseline(cfg, l, x, c, s)))(x))
benchmark(fn_base, x, layer, cos, sin, flop_count=flops_layer, label="baseline (separate norm+proj)")
del fn_base; gc.collect()

fn_fused = jax.jit(lambda x, l, c, s: jax.grad(
    lambda x: jnp.sum(layer_fused(cfg, l, x, c, s)))(x))
benchmark(fn_fused, x, layer, cos, sin, flop_count=flops_layer, label="fused (Pallas norm+proj)")
del fn_fused; gc.collect()

print_summary(ALL_RESULTS)
