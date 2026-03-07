# Chunked lm_head + loss: process logits in chunks to reduce peak memory
# The (B*T, V) = (8192, 32768) logit tensor is 512MB in bf16 — chunking avoids this
import gc

B, T, D, V = cfg.microbatch_size, cfg.seq_len, cfg.n_embd, cfg.vocab_size
hidden = jax.random.normal(jax.random.key(0), (B, T, D), dtype=jnp.bfloat16)
tokens = jax.random.randint(jax.random.key(1), (B, T), 0, V, dtype=jnp.int32)
lm_head_w = params.lm_head

# Baseline: full logits materialized
def loss_full(hidden, lm_head, tokens):
    logits = jnp.einsum('btd,dv->btv', hidden, lm_head)
    logits = cfg.softcap * jnp.tanh(logits / cfg.softcap)
    targets = tokens[:, 1:]
    logits = logits[:, :-1, :]
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.take_along_axis(log_probs, targets[..., None], axis=-1))

# Chunked: split vocabulary into chunks, compute loss per chunk
def loss_chunked(hidden, lm_head, tokens, num_chunks):
    hidden = hidden[:, :-1, :]  # (B, T-1, D)
    targets = tokens[:, 1:]      # (B, T-1)
    B_, T_, D_ = hidden.shape
    V_ = lm_head.shape[1]
    chunk_size = V_ // num_chunks

    # Compute loss in chunks along vocab dimension
    total_log_sum_exp = jnp.zeros((B_, T_), dtype=jnp.float32)
    target_logits = jnp.zeros((B_, T_), dtype=jnp.float32)

    for c in range(num_chunks):
        v_start = c * chunk_size
        v_end = v_start + chunk_size
        w_chunk = lm_head[:, v_start:v_end]
        logits_chunk = jnp.einsum('btd,dc->btc', hidden, w_chunk)
        logits_chunk = cfg.softcap * jnp.tanh(logits_chunk / cfg.softcap)

        # Accumulate logsumexp
        total_log_sum_exp = jnp.logaddexp(
            total_log_sum_exp,
            jax.nn.logsumexp(logits_chunk, axis=-1))

        # Gather target logits for chunks that contain them
        chunk_targets = targets - v_start
        in_chunk = (chunk_targets >= 0) & (chunk_targets < chunk_size)
        safe_targets = jnp.where(in_chunk, chunk_targets, 0)
        gathered = jnp.take_along_axis(logits_chunk, safe_targets[..., None], axis=-1)[..., 0]
        target_logits = target_logits + jnp.where(in_chunk, gathered, 0.0)

    log_probs = target_logits - total_log_sum_exp
    return -jnp.mean(log_probs)

flops_lm = 2 * B * (T - 1) * D * V * 3  # fwd+bwd

print("=== lm_head + loss: full vs chunked (fwd+bwd) ===\n")

fn_full = jax.jit(lambda h, w, t: jax.value_and_grad(loss_full)(h, w, t))
benchmark(fn_full, hidden, lm_head_w, tokens,
          flop_count=flops_lm, label="full logits (no chunks)")
del fn_full; gc.collect()

for nc in [2, 4, 8, 16]:
    fn_chunk = jax.jit(lambda h, w, t, _nc=nc: jax.value_and_grad(
        lambda h, w, t: loss_chunked(h, w, t, _nc))(h, w, t))
    benchmark(fn_chunk, hidden, lm_head_w, tokens,
              flop_count=flops_lm, label=f"chunked ({nc} chunks)")
    del fn_chunk; gc.collect()

# Correctness check
loss_ref = jax.jit(loss_full)(hidden, lm_head_w, tokens)
loss_c8 = jax.jit(lambda h, w, t: loss_chunked(h, w, t, 8))(hidden, lm_head_w, tokens)
print(f"\nCorrectness: full={loss_ref.item():.6f}, chunked(8)={loss_c8.item():.6f}, "
      f"diff={abs(loss_ref.item() - loss_c8.item()):.2e}")

print_summary(ALL_RESULTS)
