# Lightweight step breakdown: forward, backward, lm_head, optimizer
# Runs one benchmark at a time, deletes intermediates to save RAM

import gc
import optax
from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_mask, splash_attention_kernel)

def _expand_kv(k, v, n_head, n_kv_head):
    if n_kv_head == n_head:
        return k, v
    ratio = n_head // n_kv_head
    return jnp.repeat(k, ratio, axis=1), jnp.repeat(v, ratio, axis=1)

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
    logits = cfg.softcap * jnp.tanh(logits / cfg.softcap)
    targets = tokens[:, 1:]
    logits = logits[:, :-1, :]
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.take_along_axis(log_probs, targets[..., None], axis=-1))

B, T, D = cfg.microbatch_size, cfg.seq_len, cfg.n_embd
tokens = jax.random.randint(jax.random.key(0), (B, T), 0, cfg.vocab_size, dtype=jnp.int32)

flops_per_layer = layer_flops(B, T, D, cfg.n_head, cfg.n_kv_head, cfg.head_dim, cfg.mlp_dim)
flops_lm_head = 2 * B * T * D * cfg.vocab_size
flops_fwd = flops_per_layer * cfg.n_layer + flops_lm_head

print("=== Step breakdown (B=4, T=2048, L=8, splash) ===\n")

# Forward only
fwd_jit = jax.jit(lambda p, t: loss_fn(cfg, p, t))
benchmark(fwd_jit, params, tokens, flop_count=flops_fwd, label="1. forward only")
del fwd_jit; gc.collect()

# Forward + backward
fwd_bwd_jit = jax.jit(lambda p, t: jax.value_and_grad(loss_fn, argnums=1)(cfg, p, t))
r = benchmark(fwd_bwd_jit, params, tokens, flop_count=flops_fwd * 3, label="2. forward + backward")
# Keep grads for optimizer test
_, grads = fwd_bwd_jit(params, tokens)
del fwd_bwd_jit; gc.collect()

# lm_head + loss only (fwd+bwd)
hidden = jax.random.normal(jax.random.key(1), (B, T, D), dtype=jnp.bfloat16)
def lm_loss(hidden, lm_head, tokens):
    logits = jnp.einsum('btd,dv->btv', hidden, lm_head)
    logits = cfg.softcap * jnp.tanh(logits / cfg.softcap)
    targets = tokens[:, 1:]
    logits = logits[:, :-1, :]
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.take_along_axis(log_probs, targets[..., None], axis=-1))

lm_jit = jax.jit(lambda h, w, t: jax.value_and_grad(lm_loss)(h, w, t))
benchmark(lm_jit, hidden, params.lm_head, tokens,
          flop_count=flops_lm_head * 3, label="3. lm_head+loss fwd+bwd")
del lm_jit, hidden; gc.collect()

# Optimizer update
optimizer = optax.adamw(learning_rate=1e-4, b1=0.9, b2=0.95, weight_decay=0.1)
opt_state = optimizer.init(params)
opt_jit = jax.jit(lambda p, g, s: optimizer.update(g, s, p))
benchmark(opt_jit, params, grads, opt_state, label="4. optimizer update (adamw)")
del opt_jit, opt_state, grads; gc.collect()

print_summary(ALL_RESULTS[-4:])
print("Interpretation:")
print("  backward_only ~= (fwd+bwd) - forward")
print("  non-compute overhead = full_step - (fwd+bwd)*16 - optimizer")
