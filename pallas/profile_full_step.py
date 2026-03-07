# Profile the full training step to understand where time goes
# This measures the actual forward+backward with gradient accumulation

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
    if config.mlp_type == 'glu':
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

def train_step_microbatch(config, params, tokens):
    loss, grads = jax.value_and_grad(loss_fn, argnums=1)(config, params, tokens)
    return loss, grads

# Create fake data
tokens = jax.random.randint(jax.random.key(0),
    (cfg.microbatch_size, cfg.seq_len), 0, cfg.vocab_size, dtype=jnp.int32)

# Benchmark single microbatch fwd+bwd with splash attention
print("=== Single microbatch fwd+bwd (splash attention) ===")
train_step_jit = jax.jit(train_step_microbatch, static_argnums=0)

# FLOP count: 3x (fwd+bwd) per layer x L layers + embed + lm_head
B, T, D = cfg.microbatch_size, cfg.seq_len, cfg.n_embd
flops_per_layer = layer_flops(B, T, D, cfg.n_head, cfg.n_kv_head, cfg.head_dim, cfg.mlp_dim)
flops_embed = 0  # lookup, no matmul
flops_lm_head = 2 * B * T * D * cfg.vocab_size
total_flops = (flops_per_layer * cfg.n_layer + flops_lm_head) * 3  # 3x for fwd+bwd

benchmark(train_step_jit, cfg, params, tokens,
          flop_count=total_flops, label=f"microbatch fwd+bwd (splash, L={cfg.n_layer})")

# Also benchmark with einsum attention for comparison
cfg_einsum = Config(attn_impl='einsum')
train_step_einsum = jax.jit(train_step_microbatch, static_argnums=0)
print("\n=== Single microbatch fwd+bwd (einsum attention) ===")
benchmark(train_step_einsum, cfg_einsum, params, tokens,
          flop_count=total_flops, label=f"microbatch fwd+bwd (einsum, L={cfg.n_layer})")

# Gradient accumulation via scan
def train_step_accumulated(config, params, all_tokens):
    def scan_body(carry, tokens):
        loss, grads = jax.value_and_grad(loss_fn, argnums=1)(config, carry, tokens)
        return carry, (loss, grads)
    _, (losses, grads) = jax.lax.scan(scan_body, params, all_tokens)
    avg_loss = jnp.mean(losses)
    avg_grads = jax.tree.map(lambda g: jnp.mean(g, axis=0), grads)
    return avg_loss, avg_grads

# Full batch = 16 microbatches
all_tokens = jax.random.randint(jax.random.key(1),
    (cfg.num_microbatches, cfg.microbatch_size, cfg.seq_len),
    0, cfg.vocab_size, dtype=jnp.int32)

print(f"\n=== Full step: {cfg.num_microbatches} microbatches x B={cfg.microbatch_size} (splash) ===")
full_step_jit = jax.jit(train_step_accumulated, static_argnums=0)
total_flops_full = total_flops * cfg.num_microbatches

benchmark(full_step_jit, cfg, params, all_tokens,
          flop_count=total_flops_full,
          label=f"full step ({cfg.num_microbatches}x{cfg.microbatch_size}, splash)")

print_summary(ALL_RESULTS[-3:])
