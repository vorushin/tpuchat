# Break down the full training step: forward, backward, optimizer
# Find where the non-MXU time goes

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
    if config.attn_impl == 'splash':
        smask = splash_attention_mask.CausalMask(shape=(seq_len, seq_len))
        mh_mask = splash_attention_mask.MultiHeadMask(masks=[smask] * config.n_head)
        bs = min(config.splash_block_size, seq_len)
        block_sizes = splash_attention_kernel.BlockSizes(
            block_q=bs, block_kv=bs, block_q_dkv=bs, block_kv_dkv=bs,
            block_q_dq=bs, block_kv_dq=bs)
        kernel = splash_attention_kernel.make_splash_mha(
            mask=mh_mask, head_shards=1, q_seq_shards=1, block_sizes=block_sizes)
        attn_out = jax.vmap(kernel)(q, k, v)
    elif config.attn_impl == 'einsum':
        k_exp, v_exp = _expand_kv(k, v, config.n_head, config.n_kv_head)
        scale = config.head_dim ** -0.5
        scores = jnp.einsum('bnth,bnsh->bnts', q, k_exp) * scale
        mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
        scores = jnp.where(mask[None, None], scores, jnp.finfo(scores.dtype).min)
        attn_w = jax.nn.softmax(scores, axis=-1)
        attn_out = jnp.einsum('bnts,bnsh->bnth', attn_w, v_exp)
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
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
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
    loss = -jnp.mean(jnp.take_along_axis(log_probs, targets[..., None], axis=-1))
    return loss

B = cfg.microbatch_size
T = cfg.seq_len
D = cfg.n_embd
tokens = jax.random.randint(jax.random.key(0), (B, T), 0, cfg.vocab_size, dtype=jnp.int32)

flops_per_layer = layer_flops(B, T, D, cfg.n_head, cfg.n_kv_head, cfg.head_dim, cfg.mlp_dim)
flops_lm_head = 2 * B * T * D * cfg.vocab_size
flops_fwd = flops_per_layer * cfg.n_layer + flops_lm_head

# 1. Forward only
@jax.jit
def fwd_only(params, tokens):
    return loss_fn(cfg, params, tokens)

print("=== Step breakdown (splash, B=4, T=2048, L=8) ===\n")
benchmark(fwd_only, params, tokens, flop_count=flops_fwd, label="forward only")

# 2. Forward + backward (grad computation)
@jax.jit
def fwd_bwd(params, tokens):
    return jax.value_and_grad(loss_fn, argnums=1)(cfg, params, tokens)

benchmark(fwd_bwd, params, tokens, flop_count=flops_fwd * 3, label="forward + backward")

# 3. Just the lm_head + loss (no layers)
hidden = jax.random.normal(jax.random.key(1), (B, T, D), dtype=jnp.bfloat16)

@jax.jit
def lm_head_loss(hidden, lm_head, tokens):
    logits = jnp.einsum('btd,dv->btv', hidden, lm_head)
    logits = cfg.softcap * jnp.tanh(logits / cfg.softcap)
    targets = tokens[:, 1:]
    logits = logits[:, :-1, :]
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.take_along_axis(log_probs, targets[..., None], axis=-1))

@jax.jit
def lm_head_loss_grad(hidden, lm_head, tokens):
    return jax.value_and_grad(lm_head_loss)(hidden, lm_head, tokens)

benchmark(lm_head_loss, hidden, params.lm_head, tokens,
          flop_count=flops_lm_head, label="lm_head+loss fwd")
benchmark(lm_head_loss_grad, hidden, params.lm_head, tokens,
          flop_count=flops_lm_head * 3, label="lm_head+loss fwd+bwd")

# 4. Optimizer step
optimizer = optax.adamw(learning_rate=1e-4, b1=0.9, b2=0.95,
                        weight_decay=0.1)
opt_state = optimizer.init(params)
_, grads = fwd_bwd(params, tokens)

@jax.jit
def optimizer_step(params, grads, opt_state):
    updates, new_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_state

benchmark(optimizer_step, params, grads, opt_state, label="optimizer update")

# 5. Splash attention block sizes
print("\n=== Splash attention block size sweep ===")
cos, sin = precompute_rope(T, cfg.head_dim)
cos = cos[None, None, :, :]
sin = sin[None, None, :, :]
layer = params.layers[0]
x = jax.random.normal(jax.random.key(2), (B, T, D), dtype=jnp.bfloat16)

for splash_bs in [256, 512, 1024, 2048]:
    test_cfg = Config(splash_block_size=splash_bs)
    @jax.jit
    def layer_fwd_bwd(x, layer, cos, sin, _cfg=test_cfg):
        def fwd(x):
            return jnp.sum(single_layer_forward(_cfg, layer, x, cos, sin))
        return jax.grad(fwd)(x)
    try:
        benchmark(layer_fwd_bwd, x, layer, cos, sin,
                  flop_count=flops_per_layer * 3,
                  label=f"layer fwd+bwd splash_bs={splash_bs}")
    except Exception as e:
        print(f"  splash_bs={splash_bs}: ERROR {str(e)[:60]}")

print_summary(ALL_RESULTS[-8:])
