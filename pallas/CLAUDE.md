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
| Arithmetic intensity | 574 FLOPs/byte |

## Model dimensions (08_tpu_ablations Config)

B=4 (microbatch), T=2048, D=1024, N=4, K=1, H=256, F=3072, L=8, V=32768

## Baseline Performance

Full training step: **290ms, 48.3% MFU** (splash attention, 16x4 microbatches, batch=64)

### Step breakdown (per microbatch, 19ms total)

| Component | Time | % of step | MFU% |
|-----------|------|-----------|------|
| Forward (8 layers) | ~5.8ms | 32% | 41.1% |
| Backward (8 layers) | ~9.3ms | 51% | — |
| lm_head+loss fwd+bwd | ~3.96ms | 21% | 45.3% |
| Optimizer (once/step) | 2.67ms | <1% | n/a |

Per-layer fwd+bwd: 1.35ms at **64.2% MFU** (splash_bs=1024)

## Key Findings

### 1. Splash attention block sizes

| Block size | Layer fwd+bwd | MFU% |
|-----------|---------------|------|
| 256 | 2.13ms | 40.8% |
| 512 | 1.44ms | 60.2% |
| **1024** | **1.35ms** | **64.2%** |
| 2048 | 1.42ms | 61.5% |

splash_bs=1024 (current default) is optimal.

### 2. Microbatch size sweep (total batch=64)

| Config | Step time | MFU% |
|--------|-----------|------|
| mb=2 x 32 | 296ms | 47.3% |
| **mb=4 x 16** | **290ms** | **48.3%** |
| mb=8 x 8 | 303ms | 46.3% |
| mb=16 x 4 | 329ms | 42.6% |

mb=4 (current default) is optimal. Larger microbatches increase memory pressure.

### 3. XLA already fuses shared-input matmuls

Manually fusing QKV or gate+up into single matmuls provides NO benefit — XLA already does this. The gate+up fusion actually *hurt* (0.31ms → 0.44ms) because the (D, 2F=6144) output width doesn't tile as well.

### 4. Pallas kernels HURT inside full layers

| Variant | Layer fwd+bwd | MFU% |
|---------|---------------|------|
| **Baseline (XLA)** | **1.36ms** | **64.1%** |
| Fused Pallas norm+proj | 2.65ms | 32.9% |

Pallas kernels act as **opaque barriers** that break XLA's global optimization. XLA can fuse norms, matmuls, activations, and residuals across the entire layer computation graph. A Pallas kernel inserts a boundary that prevents this fusion. The isolated rmsnorm+linear kernel beats XLA (25.5% vs 22.8% MFU), but embedded in a full layer it's 2x slower.

**Rule: Only use Pallas for operations XLA fundamentally can't optimize** (e.g., custom attention patterns, novel memory access patterns). Don't replace standard ops that XLA handles well.

### 5. Individual component MFU

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
