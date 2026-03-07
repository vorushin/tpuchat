# Full training step: compare num_lm_head_chunks=8 (current) vs 2 (optimal)
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

def loss_fn_full(config, params, tokens):
    hidden = model_forward(config, params, tokens)
    logits = jnp.einsum('btd,dv->btv', hidden, params.lm_head)
    logits = config.softcap * jnp.tanh(logits / config.softcap)
    targets = tokens[:, 1:]
    logits = logits[:, :-1, :]
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.take_along_axis(log_probs, targets[..., None], axis=-1))

def loss_fn_chunked(config, params, tokens, num_chunks=2):
    hidden = model_forward(config, params, tokens)
    hidden = hidden[:, :-1, :]
    targets = tokens[:, 1:]
    B, T, D = hidden.shape
    V = config.vocab_size
    chunk_size = V // num_chunks

    total_lse = jnp.full((B, T), -jnp.inf, dtype=jnp.float32)
    target_logits = jnp.zeros((B, T), dtype=jnp.float32)

    for c in range(num_chunks):
        v_start = c * chunk_size
        v_end = v_start + chunk_size
        w_chunk = params.lm_head[:, v_start:v_end]
        logits_c = jnp.einsum('btd,dc->btc', hidden, w_chunk)
        logits_c = config.softcap * jnp.tanh(logits_c / config.softcap)
        total_lse = jnp.logaddexp(total_lse, jax.nn.logsumexp(logits_c, axis=-1))
        chunk_targets = targets - v_start
        in_chunk = (chunk_targets >= 0) & (chunk_targets < chunk_size)
        safe_targets = jnp.where(in_chunk, chunk_targets, 0)
        gathered = jnp.take_along_axis(logits_c, safe_targets[..., None], axis=-1)[..., 0]
        target_logits = target_logits + jnp.where(in_chunk, gathered, 0.0)

    return -jnp.mean(target_logits - total_lse)

def train_step(config, params, all_tokens, loss_fn):
    def scan_body(carry, tokens):
        loss, grads = jax.value_and_grad(loss_fn, argnums=1)(config, carry, tokens)
        return carry, (loss, grads)
    _, (losses, grads) = jax.lax.scan(scan_body, params, all_tokens)
    return jnp.mean(losses), jax.tree.map(lambda g: jnp.mean(g, axis=0), grads)

B, T, D = cfg.microbatch_size, cfg.seq_len, cfg.n_embd
n_mb = cfg.num_microbatches
all_tokens = jax.random.randint(jax.random.key(0),
    (n_mb, B, T), 0, cfg.vocab_size, dtype=jnp.int32)

flops_per_layer = layer_flops(B, T, D, cfg.n_head, cfg.n_kv_head, cfg.head_dim, cfg.mlp_dim)
flops_lm = 2 * B * T * D * cfg.vocab_size
total_flops = ((flops_per_layer * cfg.n_layer + flops_lm) * 3) * n_mb

print("=== Full training step: full vs chunked lm_head ===\n")

fn_full = jax.jit(lambda p, t: train_step(cfg, p, t, loss_fn_full))
benchmark(fn_full, params, all_tokens, flop_count=total_flops,
          label="full logits (current)")
del fn_full; gc.collect()

fn_c2 = jax.jit(lambda p, t: train_step(cfg, p, t,
    lambda c, p, tok: loss_fn_chunked(c, p, tok, 2)))
benchmark(fn_c2, params, all_tokens, flop_count=total_flops,
          label="chunked (2 chunks)")
del fn_c2; gc.collect()

fn_c4 = jax.jit(lambda p, t: train_step(cfg, p, t,
    lambda c, p, tok: loss_fn_chunked(c, p, tok, 4)))
benchmark(fn_c4, params, all_tokens, flop_count=total_flops,
          label="chunked (4 chunks)")
del fn_c4; gc.collect()

print_summary(ALL_RESULTS)
