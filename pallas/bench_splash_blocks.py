# Benchmark splash attention block sizes on a single layer fwd+bwd
import gc
from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_mask, splash_attention_kernel)

def single_layer_forward(config, layer, x, cos, sin, layer_idx=0):
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

B, T, D = cfg.microbatch_size, cfg.seq_len, cfg.n_embd
x = jax.random.normal(jax.random.key(0), (B, T, D), dtype=jnp.bfloat16)
layer = params.layers[0]
cos, sin = precompute_rope(T, cfg.head_dim)
cos, sin = cos[None, None, :, :], sin[None, None, :, :]

flops_layer = layer_flops(B, T, D, cfg.n_head, cfg.n_kv_head, cfg.head_dim, cfg.mlp_dim) * 3

print("=== Splash attention block size sweep (single layer fwd+bwd) ===\n")
for splash_bs in [256, 512, 1024, 2048]:
    test_cfg = Config(splash_block_size=splash_bs)
    fn = jax.jit(lambda x, ly, c, s, _c=test_cfg: jax.grad(
        lambda x: jnp.sum(single_layer_forward(_c, ly, x, c, s)))(x))
    try:
        benchmark(fn, x, layer, cos, sin,
                  flop_count=flops_layer,
                  label=f"splash_bs={splash_bs}")
    except Exception as e:
        print(f"  splash_bs={splash_bs}: ERROR {str(e)[:80]}")
    del fn; gc.collect()

print_summary(ALL_RESULTS)
