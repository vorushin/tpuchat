# Full training step: no chunking vs current (num_lm_head_chunks=8)
import gc
import optax
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

# No chunking: full matmul + loss
def loss_no_chunk(config, params, tokens):
    hidden = model_forward(config, params, tokens)
    logits = jnp.einsum('btd,dv->btv', hidden, params.lm_head)
    logits = config.softcap * jnp.tanh(logits / config.softcap)
    targets = tokens[:, 1:]
    logits = logits[:, :-1, :]
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.take_along_axis(log_probs, targets[..., None], axis=-1))

# Chunked (current code approach, batch-dim)
def _logits_from_chunk(h_chunk, lm_head, config):
    return config.softcap * jnp.tanh(
        jnp.einsum('td,dv->tv', h_chunk, lm_head) / config.softcap)

@ft.partial(jax.custom_vjp, nondiff_argnums=(3,))
def chunked_lm_head_loss(hidden, lm_head, labels, config):
    B, T, D = hidden.shape
    N = config.num_lm_head_chunks
    S = B * T // N
    def fwd_body(_, data):
        h_chunk, l_chunk = data
        return None, jnp.sum(optax.softmax_cross_entropy_with_integer_labels(
            _logits_from_chunk(h_chunk, lm_head, config), l_chunk))
    _, losses = jax.lax.scan(fwd_body, None,
        (hidden.reshape(N, S, D), labels.reshape(N, S)))
    return jnp.sum(losses) / (B * T)

def _fwd(hidden, lm_head, labels, config):
    return chunked_lm_head_loss(hidden, lm_head, labels, config), (hidden, lm_head, labels)

def _bwd(config, res, g):
    hidden, lm_head, labels = res
    B, T, D = hidden.shape
    N = config.num_lm_head_chunks
    S = B * T // N
    def bwd_body(d_w_acc, data):
        h_chunk, l_chunk = data
        def chunk_loss(h, w):
            return jnp.sum(optax.softmax_cross_entropy_with_integer_labels(
                _logits_from_chunk(h, w, config), l_chunk))
        _, vjp_fn = jax.vjp(chunk_loss, h_chunk, lm_head)
        d_h, d_w = vjp_fn(g / (B * T))
        return d_w_acc + d_w, d_h
    d_w, d_h_chunks = jax.lax.scan(bwd_body, jnp.zeros_like(lm_head),
        (hidden.reshape(N, S, D), labels.reshape(N, S)))
    return d_h_chunks.reshape(B, T, D), d_w, jnp.zeros_like(labels)

chunked_lm_head_loss.defvjp(_fwd, _bwd)

def loss_chunked(config, params, tokens):
    hidden = model_forward(config, params, tokens)
    return chunked_lm_head_loss(hidden, params.lm_head, tokens, config)

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

print("=== Full step: chunked (current) vs no chunking ===\n")

fn_chunk = jax.jit(lambda p, t: train_step(cfg, p, t, loss_chunked))
benchmark(fn_chunk, params, all_tokens, flop_count=total_flops,
          label=f"chunked (8 chunks, current)")
del fn_chunk; gc.collect()

fn_nochunk = jax.jit(lambda p, t: train_step(cfg, p, t, loss_no_chunk))
benchmark(fn_nochunk, params, all_tokens, flop_count=total_flops,
          label="no chunking (full lm_head)")
del fn_nochunk; gc.collect()

print_summary(ALL_RESULTS)
