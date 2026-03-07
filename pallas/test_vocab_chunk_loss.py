# Test vocab-dim chunked loss (lax.scan + custom_vjp) matches non-chunked reference
import jax
import jax.numpy as jnp
import functools as ft

softcap = 15.0

# Reference: full logits, no chunking
def loss_reference(hidden, lm_head, labels):
    logits = jnp.einsum('btd,dv->btv', hidden, lm_head)
    logits = softcap * jnp.tanh(logits / softcap)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.take_along_axis(log_probs, labels[..., None], axis=-1))

# Vocab-dim chunked with lax.scan + custom_vjp (matching 08_tpu_ablations.py)
def _reshape_lm_chunks(lm_head, num_chunks):
    D, V = lm_head.shape
    return lm_head.reshape(D, num_chunks, V // num_chunks).transpose(1, 0, 2)

@ft.partial(jax.custom_vjp, nondiff_argnums=(3,))
def loss_chunked(hidden, lm_head, labels, num_chunks):
    B, T, D = hidden.shape
    V = lm_head.shape[1]
    chunk_size = V // num_chunks
    lm_chunks = _reshape_lm_chunks(lm_head, num_chunks)
    v_starts = jnp.arange(num_chunks) * chunk_size

    def fwd_body(carry, data):
        total_lse, target_logits = carry
        w_chunk, v_start = data
        logits_c = jnp.einsum('btd,dc->btc', hidden, w_chunk)
        logits_c = softcap * jnp.tanh(logits_c / softcap)
        total_lse = jnp.logaddexp(total_lse, jax.nn.logsumexp(logits_c, axis=-1))
        chunk_targets = labels - v_start
        in_chunk = (chunk_targets >= 0) & (chunk_targets < chunk_size)
        safe_targets = jnp.where(in_chunk, chunk_targets, 0)
        gathered = jnp.take_along_axis(logits_c, safe_targets[..., None], axis=-1)[..., 0]
        target_logits = target_logits + jnp.where(in_chunk, gathered, 0.0)
        return (total_lse, target_logits), None

    init = (jnp.full((B, T), -jnp.inf, dtype=jnp.float32),
            jnp.zeros((B, T), dtype=jnp.float32))
    (total_lse, target_logits), _ = jax.lax.scan(fwd_body, init, (lm_chunks, v_starts))
    return -jnp.mean(target_logits - total_lse)

def _fwd(hidden, lm_head, labels, num_chunks):
    loss = loss_chunked(hidden, lm_head, labels, num_chunks)
    return loss, (hidden, lm_head, labels)

def _bwd(num_chunks, residuals, g):
    hidden, lm_head, labels = residuals
    B, T, D = hidden.shape
    V = lm_head.shape[1]
    chunk_size = V // num_chunks
    lm_chunks = _reshape_lm_chunks(lm_head, num_chunks)
    v_starts = jnp.arange(num_chunks) * chunk_size

    # Pass 1: recompute total_lse
    def lse_body(total_lse, w_chunk):
        logits_c = jnp.einsum('btd,dc->btc', hidden, w_chunk)
        logits_c = softcap * jnp.tanh(logits_c / softcap)
        return jnp.logaddexp(total_lse, jax.nn.logsumexp(logits_c, axis=-1)), None
    total_lse, _ = jax.lax.scan(
        lse_body, jnp.full((B, T), -jnp.inf, dtype=jnp.float32), lm_chunks)

    # Pass 2: gradients
    scale = g / (B * T)
    def grad_body(d_hidden, data):
        w_chunk, v_start = data
        logits_c = jnp.einsum('btd,dc->btc', hidden, w_chunk)
        logits_c = softcap * jnp.tanh(logits_c / softcap)
        probs_c = jnp.exp(logits_c - total_lse[..., None])
        chunk_targets = labels - v_start
        in_chunk = (chunk_targets >= 0) & (chunk_targets < chunk_size)
        safe_targets = jnp.where(in_chunk, chunk_targets, 0)
        one_hot_c = jax.nn.one_hot(safe_targets, chunk_size) * in_chunk[..., None]
        d_logits_c = scale * (probs_c - one_hot_c)
        softcap_deriv = 1.0 - (logits_c / softcap) ** 2
        d_raw_c = d_logits_c * softcap_deriv
        d_hidden = d_hidden + jnp.einsum('btc,dc->btd', d_raw_c, w_chunk)
        d_lm_chunk = jnp.einsum('btd,btc->dc', hidden, d_raw_c)
        return d_hidden, d_lm_chunk

    d_hidden, d_lm_chunks = jax.lax.scan(
        grad_body, jnp.zeros_like(hidden), (lm_chunks, v_starts))
    d_lm_head = d_lm_chunks.transpose(1, 0, 2).reshape(D, V)
    return d_hidden, d_lm_head, jnp.zeros_like(labels)

loss_chunked.defvjp(_fwd, _bwd)

# Test
B, T, D, V = 2, 16, 32, 128
key = jax.random.key(42)
hidden = jax.random.normal(key, (B, T, D), dtype=jnp.float32)
lm_head = jax.random.normal(jax.random.key(1), (D, V), dtype=jnp.float32)
labels = jax.random.randint(jax.random.key(2), (B, T), 0, V, dtype=jnp.int32)

# Forward
loss_ref = loss_reference(hidden, lm_head, labels)
loss_c2 = loss_chunked(hidden, lm_head, labels, 2)
loss_c4 = loss_chunked(hidden, lm_head, labels, 4)

print(f"Reference loss:  {loss_ref:.6f}")
print(f"Chunked (2):     {loss_c2:.6f}  (diff: {abs(loss_ref - loss_c2):.2e})")
print(f"Chunked (4):     {loss_c4:.6f}  (diff: {abs(loss_ref - loss_c4):.2e})")

# Gradients
g_ref = jax.grad(loss_reference, argnums=(0, 1))(hidden, lm_head, labels)
g_c2 = jax.grad(loss_chunked, argnums=(0, 1))(hidden, lm_head, labels, 2)
g_c4 = jax.grad(loss_chunked, argnums=(0, 1))(hidden, lm_head, labels, 4)

for name, g_test in [("Chunked (2)", g_c2), ("Chunked (4)", g_c4)]:
    d_h_err = jnp.max(jnp.abs(g_ref[0] - g_test[0]))
    d_w_err = jnp.max(jnp.abs(g_ref[1] - g_test[1]))
    print(f"{name} grad: d_hidden max_err={d_h_err:.2e}, d_lm_head max_err={d_w_err:.2e}")

max_err = max(
    jnp.max(jnp.abs(g_ref[0] - g_c2[0])),
    jnp.max(jnp.abs(g_ref[1] - g_c2[1])),
)
print(f"\n{'PASS' if max_err < 1e-5 else 'FAIL'} (max gradient error: {max_err:.2e})")
