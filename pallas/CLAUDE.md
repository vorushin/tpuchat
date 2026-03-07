# Pallas Kernel Development

## Goal

Maximize MXU utilization in the tpuchat training loop by writing custom fused Pallas kernels for TPU v6e.

## Tooling

| File | Purpose |
|------|---------|
| `colab_server.py` | Jupytext notebook — opens in Colab, starts HTTP code execution server with pre-loaded model |
| `colab_client.py` | CLI client — sends code/files to Colab TPU, displays results |
| `pallas_test.py` | Local CPU correctness tests via `interpret=True` |
| `bench_rmsnorm_linear.py` | TPU benchmark for fused RMSNorm+Linear kernel |
| `profile_step.py` | TPU profiling script — breaks down single layer into components |
| `.colab_connection` | Saved `URL\|TOKEN` (gitignored) |

## Workflow

1. Edit kernel in `pallas_test.py`, run locally: `uv run python pallas/pallas_test.py`
2. Write TPU benchmark to a `.py` file in this directory
3. Send to Colab: `uv run python pallas/colab_client.py --file pallas/<file>.py --timeout 300`
4. Always write code to files first (avoid `--code` with long strings — triggers permission prompts)

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

## Progress

### Profiling results (single layer components, B=4 microbatch)

| Component | Wall ms | MFU% | Notes |
|-----------|---------|------|-------|
| rms_norm | 0.18 | n/a | Pure memory-bound, no MXU |
| Q proj (D→N*H) | 0.20 | 9.3% | Low MFU — 3D einsum overhead |
| K proj (D→K*H) | 0.20 | 2.3% | Very low — K=1, tiny matmul |
| V proj (D→K*H) | 0.16 | 3.0% | Same issue as K |
| Out proj (N*H→D) | 0.17 | 10.9% | |
| MLP gate+silu (D→F) | 0.22 | 25.2% | Best individual MFU |
| MLP up (D→F) | 0.22 | 25.1% | |
| MLP down (F→D) | 0.22 | 25.0% | |
| **Full layer fwd** | **0.90** | **32.4%** | XLA fuses some ops |
| **Full layer fwd+bwd** | **2.01** | **43.4%** | Backward is more compute-dense |

Sum of individual components: 1.57ms. Full layer: 0.90ms. XLA already saves 0.67ms through fusion.

### Kernel 1: Fused RMSNorm + Linear

Eliminates intermediate normalized tensor write/read from HBM.

| Variant | Wall ms | MFU% |
|---------|---------|------|
| Unfused (XLA) | 0.25 | 22.8% |
| **Fused Pallas (1024x3072)** | **0.22** | **25.5%** |

Key tuning findings:
- `block_n=full_output_width` (3072) avoids redundant x loads across column blocks
- `block_m=1024` keeps MXU busy
- `dimension_semantics=['parallel','parallel']` enables Mosaic pipelining
- TPU MXU requires float32 accumulator (`preferred_element_type=jnp.float32`)
- JAX 0.9.1+ needed (Mosaic IR v8 requires matching libtpu)

## Key Pallas/TPU lessons

- `pltpu.CompilerParams` (not `TPUCompilerParams`) in JAX 0.9.x
- TPU MXU only does bf16 inputs → f32 accumulator. Cannot accumulate in bf16.
- VMEM limit ~32MB — block sizes must fit x_tile + w_tile + out_tile in VMEM
- Larger blocks = better MFU (fewer launches, better MXU utilization)
- `jnp.dot` works inside Pallas kernels (lowers to MXU matmul)
- `interpret=True` for CPU testing — supports all JAX ops
