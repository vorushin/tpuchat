# Test if larger microbatch sizes improve MFU
# Bigger matmuls = better MXU utilization
# Trade-off: fewer microbatches means less gradient accumulation overlap
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

def model_forward(config, params, tokens):
    B, T = tokens.shape
    cos, sin = precompute_rope(T, config.head_dim)
    cos, sin = cos[None, None, :, :], sin[None, None, :, :]
    x = rms_norm(params.wte[tokens])
    for i in range(config.n_layer):
        x = single_layer_forward(config, params.layers[i], x, cos, sin, layer_idx=i)
    return rms_norm(x)

def loss_fn(config, params, tokens):
    hidden = model_forward(config, params, tokens)
    logits = jnp.einsum('btd,dv->btv', hidden, params.lm_head)
    logits = config.softcap * jnp.tanh(logits / config.softcap)
    targets = tokens[:, 1:]
    logits = logits[:, :-1, :]
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.take_along_axis(log_probs, targets[..., None], axis=-1))

def train_step_accumulated(config, params, all_tokens):
    def scan_body(carry, tokens):
        loss, grads = jax.value_and_grad(loss_fn, argnums=1)(config, carry, tokens)
        return carry, (loss, grads)
    _, (losses, grads) = jax.lax.scan(scan_body, params, all_tokens)
    return jnp.mean(losses), jax.tree.map(lambda g: jnp.mean(g, axis=0), grads)

D = cfg.n_embd
T = cfg.seq_len

print("=== Microbatch size sweep (total batch=64, splash) ===\n")

# Test configs: (microbatch_size, num_microbatches)
for mb_size, n_mb in [(2, 32), (4, 16), (8, 8), (16, 4)]:
    test_cfg = Config(microbatch_size=mb_size, batch_size=64)
    all_tokens = jax.random.randint(jax.random.key(0),
        (n_mb, mb_size, T), 0, cfg.vocab_size, dtype=jnp.int32)

    flops_per_layer = layer_flops(mb_size, T, D, cfg.n_head, cfg.n_kv_head,
                                   cfg.head_dim, cfg.mlp_dim)
    flops_lm = 2 * mb_size * T * D * cfg.vocab_size
    total_flops = ((flops_per_layer * cfg.n_layer + flops_lm) * 3) * n_mb

    try:
        fn = jax.jit(train_step_accumulated, static_argnums=0)
        benchmark(fn, test_cfg, params, all_tokens,
                  flop_count=total_flops,
                  label=f"mb={mb_size} x {n_mb} (batch=64)")
    except Exception as e:
        print(f"  mb={mb_size} x {n_mb}: ERROR {str(e)[:80]}")
    del fn, all_tokens; gc.collect()

print_summary(ALL_RESULTS)
