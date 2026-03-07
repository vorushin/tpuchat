# Profile the ~31ms gap in training step breakdown
# Known: 16 * 8 layers * 1.35ms = 172.8ms, 16 * 5.21ms = 83.4ms lm_head, 2.67ms optim
# Unaccounted: ~31ms — where does it go?
import gc
from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_mask, splash_attention_kernel)

B, T, D = cfg.microbatch_size, cfg.seq_len, cfg.n_embd
V = cfg.vocab_size
n_mb = cfg.num_microbatches

tokens = jax.random.randint(jax.random.key(0), (B, T), 0, V, dtype=jnp.int32)
x = jax.random.normal(jax.random.key(1), (B, T, D), dtype=jnp.bfloat16)

print("=== Profiling the gap: per-microbatch overhead components ===\n")
print(f"  B={B}, T={T}, D={D}, V={V}, n_mb={n_mb}, n_layer={cfg.n_layer}\n")

# 1. Embedding lookup: wte[tokens]
def embed_fwd(wte, tok):
    return jnp.sum(wte[tok])  # sum to get scalar for grad
fn = jax.jit(jax.value_and_grad(embed_fwd))
benchmark(fn, params.wte, tokens, flop_count=0, label="embedding lookup fwd+bwd")
del fn; gc.collect()

# 2. Initial RMSNorm (applied to embeddings)
def norm_fwd(x):
    return jnp.sum(rms_norm(x))
fn = jax.jit(jax.value_and_grad(norm_fwd))
benchmark(fn, x, flop_count=0, label="rms_norm fwd+bwd (B,T,D)")
del fn; gc.collect()

# 3. RoPE precomputation (no grad, just forward)
fn = jax.jit(lambda: precompute_rope(T, cfg.head_dim))
benchmark(fn, flop_count=0, label="precompute_rope (forward only)")
del fn; gc.collect()

# 4. apply_rope on Q and K
q = jax.random.normal(jax.random.key(2), (B, cfg.n_head, T, cfg.head_dim), dtype=jnp.bfloat16)
cos, sin = precompute_rope(T, cfg.head_dim)
cos_b, sin_b = cos[None, None, :, :], sin[None, None, :, :]

def rope_fwd(q, cos, sin):
    return jnp.sum(apply_rope(q, cos, sin))
fn = jax.jit(jax.value_and_grad(rope_fwd))
benchmark(fn, q, cos_b, sin_b, flop_count=0, label="apply_rope fwd+bwd (one head group)")
del fn, q; gc.collect()

# 5. Gradient averaging: tree_map mean over axis=0 for 16 microbatches
# Simulate: stack of 16 grad trees, then mean
fake_grads = jax.tree.map(
    lambda p: jnp.broadcast_to(p[None], (n_mb,) + p.shape), params)
fn = jax.jit(lambda g: jax.tree.map(lambda x: jnp.mean(x, axis=0), g))
benchmark(fn, fake_grads, flop_count=0, label=f"grad averaging (tree mean, {n_mb} microbatches)")
del fn, fake_grads; gc.collect()

# 6. lax.scan overhead: trivial body
def scan_trivial(_, tok):
    return None, jnp.float32(0.0)
fn = jax.jit(lambda t: jax.lax.scan(scan_trivial, None, t))
all_tokens = jax.random.randint(jax.random.key(0), (n_mb, B, T), 0, V, dtype=jnp.int32)
benchmark(fn, all_tokens, flop_count=0, label=f"lax.scan overhead ({n_mb} iters, trivial body)")
del fn; gc.collect()

# 7. Scan with just embedding + 2x rms_norm (no layers, no lm_head)
def scan_embed_only(wte, all_tokens):
    def body(carry, tok):
        x = rms_norm(carry[tok])
        x = rms_norm(x)  # final norm
        loss = jnp.mean(x)
        return carry, loss
    _, losses = jax.lax.scan(body, wte, all_tokens)
    return jnp.mean(losses)
fn = jax.jit(jax.value_and_grad(scan_embed_only))
benchmark(fn, params.wte, all_tokens, flop_count=0,
          label=f"scan({n_mb}) x [embed + 2*rms_norm] fwd+bwd")
del fn; gc.collect()

# 8. Full model_forward WITHOUT layers (embed + init_norm + final_norm + lm_head + loss)
def model_no_layers(config, params, tokens):
    B, T = tokens.shape
    x = rms_norm(params.wte[tokens])
    x = rms_norm(x)  # final norm (skip layers)
    logits = jnp.einsum('btd,dv->btv', x, params.lm_head)
    logits = config.softcap * jnp.tanh(logits / config.softcap)
    targets = tokens[:, 1:]
    logits = logits[:, :-1, :]
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.take_along_axis(log_probs, targets[..., None], axis=-1))

def step_no_layers(params, all_tokens):
    def body(carry, tok):
        loss, grads = jax.value_and_grad(model_no_layers, argnums=1)(cfg, carry, tok)
        return carry, (loss, grads)
    _, (losses, grads) = jax.lax.scan(body, params, all_tokens)
    return jnp.mean(losses), jax.tree.map(lambda g: jnp.mean(g, axis=0), grads)

fn = jax.jit(step_no_layers)
flops_lm = 2 * B * T * D * V
total_no_layers = flops_lm * 3 * n_mb
benchmark(fn, params, all_tokens, flop_count=total_no_layers,
          label=f"full step WITHOUT layers (embed+norms+lm_head+loss)")
del fn; gc.collect()

# 9. Full step reference (with layers)
def single_layer_forward(config, layer, x, cos, sin):
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

def model_forward_full(config, params, tokens):
    B, T = tokens.shape
    cos, sin = precompute_rope(T, config.head_dim)
    cos, sin = cos[None, None, :, :], sin[None, None, :, :]
    x = rms_norm(params.wte[tokens])
    for i in range(config.n_layer):
        x = single_layer_forward(config, params.layers[i], x, cos, sin)
    return rms_norm(x)

def loss_full(config, params, tokens):
    hidden = model_forward_full(config, params, tokens)
    logits = jnp.einsum('btd,dv->btv', hidden, params.lm_head)
    logits = config.softcap * jnp.tanh(logits / config.softcap)
    targets = tokens[:, 1:]
    logits = logits[:, :-1, :]
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.take_along_axis(log_probs, targets[..., None], axis=-1))

def step_full(params, all_tokens):
    def body(carry, tok):
        loss, grads = jax.value_and_grad(loss_full, argnums=1)(cfg, carry, tok)
        return carry, (loss, grads)
    _, (losses, grads) = jax.lax.scan(body, params, all_tokens)
    return jnp.mean(losses), jax.tree.map(lambda g: jnp.mean(g, axis=0), grads)

flops_per_layer = layer_flops(B, T, D, cfg.n_head, cfg.n_kv_head, cfg.head_dim, cfg.mlp_dim)
total_flops = ((flops_per_layer * cfg.n_layer + flops_lm) * 3) * n_mb
fn = jax.jit(step_full)
benchmark(fn, params, all_tokens, flop_count=total_flops, label="full step (reference)")
del fn; gc.collect()

print_summary(ALL_RESULTS)
