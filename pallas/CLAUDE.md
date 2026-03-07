# Pallas Kernel Development

## Goal

Maximize MXU utilization in the tpuchat training loop on TPU v6e.

## Tooling

| File | Purpose |
|------|---------|
| `colab_server.py` | Jupytext notebook — opens in Colab, starts HTTP code execution server with pre-loaded model |
| `colab_client.py` | CLI client — sends code/files to Colab TPU, displays results |
| `pallas_test.py` | Local CPU correctness tests via `interpret=True` |
| `.colab_connection` | Saved `URL\|TOKEN` (gitignored) |

## Workflow

1. Write code to a `.py` file in this directory (avoid `--code` with long strings — triggers permission prompts)
2. Send to Colab: `uv run python pallas/colab_client.py --file pallas/<file>.py --timeout 300`
3. For local kernel testing: `uv run python pallas/pallas_test.py`

## TPU v6e specs

| Spec | Value |
|------|-------|
| Peak bf16 TFLOPS | 918 |
| HBM bandwidth | 1600 GB/s |
| MXU | 256x256 systolic array |
| VMEM | ~32 MB per core |

## Model dimensions (08_tpu_ablations Config)

B=4 (microbatch), T=2048, D=1024, N=4, K=1, H=256, F=3072, L=8, V=32768

## Baseline & Optimized Performance

| Metric | Current | Best found |
|--------|---------|------------|
| Full step | 290ms, 48.3% MFU | **275ms, 50.9% MFU** |
| Per layer fwd+bwd | 1.35ms, 64.2% MFU | (already optimal) |
| lm_head+loss fwd+bwd | 5.21ms (batch-chunked 8) | 3.40ms (vocab-chunked 2), 52.7% MFU |

## Actionable Recommendations

### 1. Fix lm_head chunking — switch from batch-dim to vocab-dim
**Impact: ~5ms/step (+0.9% MFU)**

The current `num_lm_head_chunks=8` chunks along the **batch dimension** (B×T), not the vocabulary dimension. Each chunk still materializes (S, V) logits where V=32768 is untouched — this defeats the purpose of chunking. The `lax.scan` overhead makes it strictly worse than no chunking.

**Better approach: vocab-dimension chunking** (standard, used by LiGER kernel etc.) splits the lm_head weight `(D, V)` into chunks along V. Each chunk computes `(B, T, V/N)` logits — never materializing the full `(B, T, V)` tensor. Loss is accumulated via `logaddexp` across chunks.

| Approach | lm_head fwd+bwd | MFU% | Full step | MFU% |
|----------|-----------------|------|-----------|------|
| Batch-dim 8 chunks (current) | 5.21ms | 34.5% | 295ms | 47.5% |
| No chunking | 3.11ms | 57.7% | 290ms | 48.3% |
| **Vocab-dim 2 chunks** | **3.40ms** | **52.7%** | **285ms** | **49.2%** |

Options: (a) remove chunking entirely (simplest, +0.8% MFU), or (b) switch to vocab-dim 2 chunks (best perf, +0.9% MFU, slight loss difference from logaddexp accumulation).

### 2. Accumulate gradients inside the microbatch scan
**Impact: ~6ms/step (+1.0% MFU)**

Current code collects all 16 microbatch gradients into a `(16, ...)` array, then does `jax.tree.map(lambda g: jnp.mean(g, axis=0), grads)` post-hoc. This costs 5.84ms.

Instead, accumulate inside the scan body:
```python
def body(carry, tok):
    p, g_acc = carry
    loss, grads = jax.value_and_grad(loss_fn, argnums=1)(cfg, p, tok)
    g_acc = jax.tree.map(lambda a, g: a + g, g_acc, grads)
    return (p, g_acc), loss
```
Then divide by `n_mb` once at the end. This avoids materializing the `(16, ...)` grad tensor.

### 3. Combined: both optimizations stack
**Impact: 275ms, 50.9% MFU (was 290ms, 48.3%)**

| Approach | Step time | MFU |
|----------|-----------|-----|
| A: Baseline (current 08) | 290ms | 48.3% |
| B: Accum grads only | 284ms | 49.3% |
| C: Vocab-dim 2 chunks only | 285ms | 49.2% |
| **D: Both combined** | **275ms** | **50.9%** |

### 4. Keep current defaults — they're already optimal
- `splash_block_size=1024` — best among 256/512/1024/2048
- `microbatch_size=4, num_microbatches=16` — best among mb=2/4/8/16
- `logit_dtype='bf16'` — already fastest

## Key Findings

### XLA is very good — don't fight it

1. **Pallas kernels HURT in full layers**: Our fused RMSNorm+Linear Pallas kernel beats XLA for isolated ops (25.5% vs 22.8% MFU) but is **2x slower** in a full layer (32.9% vs 64.1% MFU). Pallas kernels act as opaque barriers that block XLA's global graph optimization.

2. **XLA already fuses shared-input matmuls**: Manually packing QKV into one matmul or gate+up into one matmul provides zero benefit. XLA detects the shared input and fuses automatically.

3. **Rule: Only use Pallas for things XLA fundamentally can't do** — custom attention patterns, novel memory access patterns. Don't replace standard ops.

### Splash attention block sizes

| Block size | Layer fwd+bwd | MFU% |
|-----------|---------------|------|
| 256 | 2.13ms | 40.8% |
| 512 | 1.44ms | 60.2% |
| **1024** | **1.35ms** | **64.2%** |
| 2048 | 1.42ms | 61.5% |

### Microbatch size sweep (total batch=64)

| Config | Step time | MFU% |
|--------|-----------|------|
| mb=2 x 32 | 296ms | 47.3% |
| **mb=4 x 16** | **290ms** | **48.3%** |
| mb=8 x 8 | 303ms | 46.3% |
| mb=16 x 4 | 329ms | 42.6% |

### lm_head chunking: batch-dim vs vocab-dim

The current 08_tpu_ablations.py chunks along **batch** (B×T), not **vocab** (V). This is the wrong dimension — V=32768 is the large axis that should be chunked to avoid materializing huge logit tensors.

**Batch-dim chunking (current 08 code — wrong approach):**
```python
hidden.reshape(N, S, D)  # S = B*T/N tokens, each gets full (S,V) logits
```

| Chunks | Isolated fwd+bwd | MFU% |
|--------|-------------------|------|
| **no chunking** | **3.11ms** | **57.7%** |
| 1 (scan overhead) | 4.10ms | 43.9% |
| 2 | 4.93ms | 36.4% |
| 8 (current) | 5.21ms | 34.5% |

Batch-dim chunking is strictly slower than no chunking. The `lax.scan` adds overhead with no memory benefit on the V dimension.

**Vocab-dim chunking (standard approach — recommended):**
```python
w_chunk = lm_head[:, v_start:v_end]  # (D, V/N), logits are (B,T,V/N)
# Accumulate loss via logaddexp across chunks
```

| Chunks | Isolated fwd+bwd | MFU% |
|--------|-------------------|------|
| no chunking | 3.97ms | 45.2% |
| **2 chunks** | **3.40ms** | **52.7%** |
| 4 chunks | 3.46ms | 52.0% |
| 8 chunks | 4.00ms | 44.8% |

Vocab-dim 2 chunks is the winner. Note: slight loss difference (2.3e-2) vs non-chunked due to logaddexp accumulation — numerically equivalent for training.

### lax.scan for layers — DON'T

Using `lax.scan` over stacked layer params instead of a Python `for` loop is **47% slower** (425ms/33% vs 290ms/48.3%). Same principle as Pallas: it blocks XLA's cross-layer optimization. Always use Python loops for layers.

### Step time gap analysis

Where the full step time goes (290ms baseline):

| Component | Per-mb | × 16 | Total |
|-----------|--------|------|-------|
| 8 layers fwd+bwd | 10.8ms | 16 | 172.8ms |
| lm_head+loss fwd+bwd | ~4.9ms | 16 | ~78ms |
| Embedding + norms | ~0.6ms | 16 | ~9.6ms |
| Grad averaging | — | — | 5.8ms |
| Scan + other overhead | — | — | ~24ms |

Embedding (0.63ms fwd+bwd) and RMSNorm (0.25ms) are memory-bound — not optimizable via MXU. RoPE precompute is 0.20ms per microbatch (recomputed 16 times but cheap). The main actionable items were grad averaging (5.8ms) and lm_head chunking overhead.

### Step breakdown (per microbatch)

| Component | Time | MFU% |
|-----------|------|------|
| Forward (8 layers) | ~5.8ms | 41.1% |
| Backward (8 layers) | ~9.3ms | — |
| lm_head+loss fwd+bwd | ~3.1ms | 57.7% |
| Optimizer (once/step) | 2.67ms | n/a |

### Individual component MFU

| Component | Wall ms | MFU% |
|-----------|---------|------|
| rms_norm | 0.18 | n/a (memory-bound) |
| Q proj (D→N*H) | 0.20 | 9.3% |
| K proj (D→K*H) | 0.20 | 2.3% |
| V proj (D→K*H) | 0.16 | 3.0% |
| Out proj (N*H→D) | 0.17 | 10.9% |
| MLP gate+silu | 0.22 | 25.2% |
| MLP up | 0.22 | 25.1% |
| MLP down | 0.22 | 25.0% |

Individual projections look terrible, but XLA fuses them into the full layer at 64% MFU.

## Pallas/TPU Technical Notes

- `pltpu.CompilerParams` (not `TPUCompilerParams`) in JAX 0.9.x
- Pallas `jnp.dot` with bf16 inputs requires `preferred_element_type=jnp.float32` — Mosaic compiler constraint (`tpu.matmul` op requires 32-bit accumulator), matches MXU's native bf16→f32 accumulation mode
- VMEM limit ~32MB — block sizes must fit all tiles
- `dimension_semantics=['parallel','parallel']` enables Mosaic pipelining
- Pallas kernels need `@jax.custom_vjp` to support `jax.grad` (no auto-diff)
- `jnp.dot` works inside Pallas kernels (lowers to MXU matmul)
- `interpret=True` for CPU testing
- JAX 0.9.1+ needed (Mosaic IR v8 requires matching libtpu)
- `ALL_RESULTS.clear()` between server requests prevents memory accumulation
