# Benchmark potential optimizations for the ~31ms gap
import gc
from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_mask, splash_attention_kernel)

B, T, D = cfg.microbatch_size, cfg.seq_len, cfg.n_embd
V = cfg.vocab_size
n_mb = cfg.num_microbatches
all_tokens = jax.random.randint(jax.random.key(0),
    (n_mb, B, T), 0, V, dtype=jnp.int32)

flops_per_layer = layer_flops(B, T, D, cfg.n_head, cfg.n_kv_head, cfg.head_dim, cfg.mlp_dim)
flops_lm = 2 * B * T * D * V
total_flops = ((flops_per_layer * cfg.n_layer + flops_lm) * 3) * n_mb

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

print("=== Optimization benchmarks ===\n")

# --- BASELINE: current approach (Python loop, RoPE inside, post-hoc grad mean) ---
def loss_baseline(config, params, tokens):
    B, T = tokens.shape
    cos, sin = precompute_rope(T, config.head_dim)
    cos, sin = cos[None, None, :, :], sin[None, None, :, :]
    x = rms_norm(params.wte[tokens])
    for i in range(config.n_layer):
        x = single_layer_forward(config, params.layers[i], x, cos, sin)
    hidden = rms_norm(x)
    logits = jnp.einsum('btd,dv->btv', hidden, params.lm_head)
    logits = config.softcap * jnp.tanh(logits / config.softcap)
    targets = tokens[:, 1:]
    logits = logits[:, :-1, :]
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.take_along_axis(log_probs, targets[..., None], axis=-1))

def step_baseline(params, all_tokens):
    def body(carry, tok):
        loss, grads = jax.value_and_grad(loss_baseline, argnums=1)(cfg, carry, tok)
        return carry, (loss, grads)
    _, (losses, grads) = jax.lax.scan(body, params, all_tokens)
    return jnp.mean(losses), jax.tree.map(lambda g: jnp.mean(g, axis=0), grads)

fn = jax.jit(step_baseline)
benchmark(fn, params, all_tokens, flop_count=total_flops, label="baseline (current)")
del fn; gc.collect()

# --- OPT 1: Accumulate grads inside scan (avoid post-hoc tree mean) ---
def step_accum_grads(params, all_tokens):
    def body(carry, tok):
        p, g_acc = carry
        loss, grads = jax.value_and_grad(loss_baseline, argnums=1)(cfg, p, tok)
        g_acc = jax.tree.map(lambda a, g: a + g, g_acc, grads)
        return (p, g_acc), loss
    g_init = jax.tree.map(jnp.zeros_like, params)
    (_, g_sum), losses = jax.lax.scan(body, (params, g_init), all_tokens)
    avg_grads = jax.tree.map(lambda g: g / n_mb, g_sum)
    return jnp.mean(losses), avg_grads

fn = jax.jit(step_accum_grads)
benchmark(fn, params, all_tokens, flop_count=total_flops,
          label="opt1: accumulate grads in scan")
del fn; gc.collect()

# --- OPT 2: lax.scan for layers instead of Python for loop ---
def loss_scan_layers(config, params, tokens):
    B, T = tokens.shape
    cos, sin = precompute_rope(T, config.head_dim)
    cos, sin = cos[None, None, :, :], sin[None, None, :, :]
    x = rms_norm(params.wte[tokens])

    # Stack layer params into arrays for scan
    stacked_layers = jax.tree.map(lambda *arrs: jnp.stack(arrs), *params.layers)

    def layer_body(x, layer):
        return single_layer_forward(config, layer, x, cos, sin), None
    x, _ = jax.lax.scan(layer_body, x, stacked_layers)

    hidden = rms_norm(x)
    logits = jnp.einsum('btd,dv->btv', hidden, params.lm_head)
    logits = config.softcap * jnp.tanh(logits / config.softcap)
    targets = tokens[:, 1:]
    logits = logits[:, :-1, :]
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.take_along_axis(log_probs, targets[..., None], axis=-1))

def step_scan_layers(params, all_tokens):
    def body(carry, tok):
        loss, grads = jax.value_and_grad(loss_scan_layers, argnums=1)(cfg, carry, tok)
        return carry, (loss, grads)
    _, (losses, grads) = jax.lax.scan(body, params, all_tokens)
    return jnp.mean(losses), jax.tree.map(lambda g: jnp.mean(g, axis=0), grads)

fn = jax.jit(step_scan_layers)
benchmark(fn, params, all_tokens, flop_count=total_flops,
          label="opt2: lax.scan for layers")
del fn; gc.collect()

# --- OPT 3: Both (scan layers + accumulate grads) ---
def step_both(params, all_tokens):
    def body(carry, tok):
        p, g_acc = carry
        loss, grads = jax.value_and_grad(loss_scan_layers, argnums=1)(cfg, p, tok)
        g_acc = jax.tree.map(lambda a, g: a + g, g_acc, grads)
        return (p, g_acc), loss
    g_init = jax.tree.map(jnp.zeros_like, params)
    (_, g_sum), losses = jax.lax.scan(body, (params, g_init), all_tokens)
    avg_grads = jax.tree.map(lambda g: g / n_mb, g_sum)
    return jnp.mean(losses), avg_grads

fn = jax.jit(step_both)
benchmark(fn, params, all_tokens, flop_count=total_flops,
          label="opt3: scan layers + accum grads")
del fn; gc.collect()

print_summary(ALL_RESULTS)
