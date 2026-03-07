# Benchmark batch-dimension chunking (matching actual 08_tpu_ablations code)
# Tests chunked_lm_head_loss with different num_lm_head_chunks values
import gc
import optax

def _logits_from_chunk(h_chunk, lm_head, config):
    logits = jnp.einsum('td,dv->tv', h_chunk, lm_head)
    return config.softcap * jnp.tanh(logits / config.softcap)

@ft.partial(jax.custom_vjp, nondiff_argnums=(3,))
def chunked_lm_head_loss(hidden, lm_head, labels, config):
    B, T, D = hidden.shape
    N = config.num_lm_head_chunks
    S = B * T // N
    hidden_chunks = hidden.reshape(N, S, D)
    labels_chunks = labels.reshape(N, S)
    def fwd_body(_, data):
        h_chunk, l_chunk = data
        return None, jnp.sum(
            optax.softmax_cross_entropy_with_integer_labels(
                _logits_from_chunk(h_chunk, lm_head, config), l_chunk))
    _, chunk_losses = jax.lax.scan(fwd_body, None, (hidden_chunks, labels_chunks))
    return jnp.sum(chunk_losses) / (B * T)

def _chunked_loss_fwd(hidden, lm_head, labels, config):
    loss = chunked_lm_head_loss(hidden, lm_head, labels, config)
    return loss, (hidden, lm_head, labels)

def _chunked_loss_bwd(config, residuals, g):
    hidden, lm_head, labels = residuals
    B, T, D = hidden.shape
    N = config.num_lm_head_chunks
    S = B * T // N
    hidden_chunks = hidden.reshape(N, S, D)
    labels_chunks = labels.reshape(N, S)
    def bwd_body(d_lm_head_acc, data):
        h_chunk, l_chunk = data
        def chunk_loss(h, w):
            return jnp.sum(
                optax.softmax_cross_entropy_with_integer_labels(
                    _logits_from_chunk(h, w, config), l_chunk))
        _, vjp_fn = jax.vjp(chunk_loss, h_chunk, lm_head)
        d_h, d_w = vjp_fn(g / (B * T))
        return d_lm_head_acc + d_w, d_h
    d_lm_head_init = jnp.zeros_like(lm_head)
    d_lm_head, d_hidden_chunks = jax.lax.scan(
        bwd_body, d_lm_head_init, (hidden_chunks, labels_chunks))
    return d_hidden_chunks.reshape(B, T, D), d_lm_head, jnp.zeros_like(labels)

chunked_lm_head_loss.defvjp(_chunked_loss_fwd, _chunked_loss_bwd)

# Also test non-chunked (full matmul) baseline
def full_lm_head_loss(hidden, lm_head, labels, config):
    B, T, D = hidden.shape
    logits = jnp.einsum('btd,dv->btv', hidden, lm_head)
    logits = config.softcap * jnp.tanh(logits / config.softcap)
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(
        logits.reshape(B * T, -1), labels.reshape(B * T)))

B, T, D = cfg.microbatch_size, cfg.seq_len, cfg.n_embd
hidden = jax.random.normal(jax.random.key(0), (B, T, D), dtype=jnp.bfloat16)
labels = jax.random.randint(jax.random.key(1), (B, T), 0, cfg.vocab_size, dtype=jnp.int32)

flops_lm = 2 * B * T * D * cfg.vocab_size * 3

print("=== Batch-dim chunked lm_head (matching 08 code) fwd+bwd ===\n")

# Full (no chunking)
fn = jax.jit(lambda h, w, l: jax.value_and_grad(full_lm_head_loss)(h, w, l, cfg))
benchmark(fn, hidden, params.lm_head, labels,
          flop_count=flops_lm, label="no chunking (full)")
del fn; gc.collect()

# Batch-dim chunks: 1, 2, 4, 8, 16
for nc in [1, 2, 4, 8, 16]:
    test_cfg = Config(num_lm_head_chunks=nc)
    fn = jax.jit(lambda h, w, l, _c=test_cfg: jax.value_and_grad(
        chunked_lm_head_loss, argnums=(0, 1))(h, w, l, _c))
    try:
        benchmark(fn, hidden, params.lm_head, labels,
                  flop_count=flops_lm, label=f"batch-chunked ({nc} chunks)")
    except Exception as e:
        print(f"  batch-chunked ({nc}): ERROR {str(e)[:80]}")
    del fn; gc.collect()

print_summary(ALL_RESULTS)
