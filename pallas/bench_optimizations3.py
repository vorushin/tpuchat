# Fix: scan layers benchmarks must return grads to prevent DCE
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

def single_layer_forward_dict(config, layer, x, cos, sin):
    h = rms_norm(x)
    q = jnp.einsum('btd,dnh->bnth', h, layer['c_q'])
    k = jnp.einsum('btd,dnh->bnth', h, layer['c_k'])
    v = jnp.einsum('btd,dnh->bnth', h, layer['c_v'])
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
    attn_out = jnp.einsum('bnth,nhd->btd', attn_out, layer['c_proj'])
    x = x + attn_out
    h2 = rms_norm(x)
    gate = jax.nn.silu(jnp.einsum('btd,df->btf', h2, layer['w_gate']))
    up = jnp.einsum('btd,df->btf', h2, layer['w_up'])
    mlp_out = jnp.einsum('btf,fd->btd', gate * up, layer['w_down'])
    x = x + mlp_out
    return x

def layers_to_stacked_dict(layers, n_layer):
    keys = ['c_q', 'c_k', 'c_v', 'c_proj', 'w_gate', 'w_up', 'w_down']
    return {k: jnp.stack([getattr(layers[i], k) for i in range(n_layer)]) for k in keys}

stacked = layers_to_stacked_dict(params.layers, cfg.n_layer)

print("=== Scan layers benchmarks (fixed: return grads) ===\n")

# --- OPT 2: lax.scan for layers, post-hoc grad mean ---
def loss_scan_layers(config, params_flat, stacked_layers, tokens):
    B, T = tokens.shape
    cos, sin = precompute_rope(T, config.head_dim)
    cos, sin = cos[None, None, :, :], sin[None, None, :, :]
    x = rms_norm(params_flat['wte'][tokens])
    def layer_body(x, layer):
        return single_layer_forward_dict(config, layer, x, cos, sin), None
    x, _ = jax.lax.scan(layer_body, x, stacked_layers)
    hidden = rms_norm(x)
    logits = jnp.einsum('btd,dv->btv', hidden, params_flat['lm_head'])
    logits = config.softcap * jnp.tanh(logits / config.softcap)
    targets = tokens[:, 1:]
    logits = logits[:, :-1, :]
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.take_along_axis(log_probs, targets[..., None], axis=-1))

# Flatten non-layer params into a plain dict
params_flat = {'wte': params.wte, 'lm_head': params.lm_head}

def step_scan_layers(pf, sl, all_tokens):
    def body(carry, tok):
        loss, grads = jax.value_and_grad(
            loss_scan_layers, argnums=(1, 2))(cfg, carry[0], carry[1], tok)
        return carry, (loss, grads)
    _, (losses, grads) = jax.lax.scan(body, (pf, sl), all_tokens)
    avg_g = jax.tree.map(lambda g: jnp.mean(g, axis=0), grads)
    return jnp.mean(losses), avg_g

fn = jax.jit(step_scan_layers)
benchmark(fn, params_flat, stacked, all_tokens, flop_count=total_flops,
          label="opt2: scan layers + post-hoc grad mean")
del fn; gc.collect()

# --- OPT 3: scan layers + accumulate grads in scan ---
def step_scan_layers_accum(pf, sl, all_tokens):
    def body(carry, tok):
        pf, sl, g_pf_acc, g_sl_acc = carry
        loss, (g_pf, g_sl) = jax.value_and_grad(
            loss_scan_layers, argnums=(1, 2))(cfg, pf, sl, tok)
        g_pf_acc = jax.tree.map(lambda a, g: a + g, g_pf_acc, g_pf)
        g_sl_acc = jax.tree.map(lambda a, g: a + g, g_sl_acc, g_sl)
        return (pf, sl, g_pf_acc, g_sl_acc), loss
    g_pf_init = jax.tree.map(jnp.zeros_like, pf)
    g_sl_init = jax.tree.map(jnp.zeros_like, sl)
    (_, _, g_pf_sum, g_sl_sum), losses = jax.lax.scan(
        body, (pf, sl, g_pf_init, g_sl_init), all_tokens)
    return jnp.mean(losses), jax.tree.map(lambda g: g / n_mb, g_pf_sum), \
           jax.tree.map(lambda g: g / n_mb, g_sl_sum)

fn = jax.jit(step_scan_layers_accum)
benchmark(fn, params_flat, stacked, all_tokens, flop_count=total_flops,
          label="opt3: scan layers + accum grads")
del fn; gc.collect()

print_summary(ALL_RESULTS)
