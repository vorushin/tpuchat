# JAX vs PyTorch: pretraining a 164M GPT on a single Colab G4 GPU

**Date:** 2026-07-22 · **Hardware:** NVIDIA RTX PRO 6000 Blackwell Server Edition
(Colab G4 runtime, 96 GB GDDR7, sm_120, driver 580.82.07) · **Software:** jax 0.7.2
(cuda12 plugin) vs torch 2.11.0+cu128, both Colab-preinstalled ·
**Notebooks:** [`11_gpu_jax.py`](../11_gpu_jax.py) / [`12_gpu_torch.py`](../12_gpu_torch.py) rev 3 ·
**Raw data:** [`results/metrics_jax.json`](results/metrics_jax.json), [`results/metrics_torch.json`](results/metrics_torch.json)

Both notebooks train the **identical model** — D=1024, N=8, K=2, H=128, F=3072
SwiGLU, L=8, T=2048, V=32768 (163.6M params, 130.0M non-embed; bit-identical
param count and FLOPs to the TPU config in `08_tpu_ablations.py`) — on the same
FineWeb-Edu shards with the same BPE tokenizer, pure bf16 (params, grads, Adam
moments; no autocast, no fp32 master weights), same effective batch
64×2048 = 131,072 tokens/step with no gradient accumulation. Each framework
uses its best idioms: JAX = XLA/jit + cuDNN flash attention + optax; PyTorch =
torch.compile + SDPA (cuDNN backend) + fused AdamW.

## Headline: PyTorch wins by ~15% end-to-end (~9% on pure compute)

| Metric (300-step clean run) | JAX | PyTorch | torch/jax |
|---|---|---|---|
| tok/s | 254,495 | 292,028 | **1.15x** |
| median step | 517.1 ms | 450.9 ms | 0.87x |
| isolated train step (Section B) | 480.6 ms | 440.1 ms | 0.92x |
| MFU (vs measured 392 TFLOP/s dense peak) | 63.5% | 73.1% | 1.15x |
| MFU (vs quoted 960 TFLOP/s) | 26.0% | 29.9% | |
| warmup (first step, warm compile caches) | 5.0 s | 0.4 s | |
| val loss @ step 300 | **6.375** | 6.434 | (RNG noise) |

Val loss trajectories match within run-to-run noise (JAX/torch @100: 7.053/6.988,
@200: 6.491/6.503, @300: 6.375/6.434) — both implementations train correctly and
equivalently; this is purely a systems comparison.

## Finding 0: Colab's "960 BF16 TFLOPs" is a sparse figure

Large dense bf16 matmuls top out at **392 TFLOP/s** in both frameworks
(16384³: JAX 390.6, torch 392.8; both stacks route to cuBLAS/cutlass). That is
0.41× the quoted number — consistent with 960 being the 2:4-sparsity marketing
figure for GB202. All MFU numbers here use the measured dense peak; against it,
both frameworks run this model at a healthy 63-73%.

## Where the 40 ms compute gap comes from (component benchmarks)

Labels are measured identically in both notebooks (same shapes, same sync
discipline; warmup absorbs compilation):

| Component | JAX ms | torch ms | faster |
|---|---|---|---|
| attention fwd (cuDNN flash, B=64) | 1.88 | 1.85 | par |
| attention fwd+bwd | 9.12 | 10.03 | **JAX 1.10x** |
| layer fwd | 13.84 | 13.76 | par |
| layer fwd+bwd | 42.75 | 45.20 | **JAX 1.06x** |
| model fwd (8 layers + embed) | 110.32 | 110.18 | par |
| model fwd+bwd incl. lm_head+CE loss | 478.04 | 433.53 | **torch 1.10x** |
| optimizer step (AdamW, 164M params) | 2.53 | 1.72 | **torch 1.48x** |
| full train step | 480.63 | 440.08 | **torch 1.09x** |

The striking pattern: **JAX is faster per transformer layer** (its attention
backward and layer-level fusion beat Inductor by 6-10%), yet loses the full
step. Subtracting 8× layer fwd+bwd from model fwd+bwd isolates the
embed + final-norm + **lm_head + cross-entropy** region:

- JAX: 478.0 − 8×42.75 = **136 ms**
- torch: 433.5 − 8×45.20 = **72 ms**

The (131072 × 1024) @ (1024 × 32768) lm_head matmul costs ~23.4 ms per pass
(~70 ms floor for fwd + 2 bwd passes in both stacks). PyTorch's Inductor fuses
the softcap-tanh + log-softmax + CE over the 8.6 GB bf16 logits tensor into a
single Triton reduction (`triton_red_fused__log_softmax...` in the trace),
paying almost nothing beyond the matmuls. XLA runs the softmax/CE as separate
memory-bound passes over the logits and spends **~64 ms more per step**. That
one fusion difference is the bulk of the framework gap; fused AdamW (0.8 ms)
adds a little more.

## What "squeezing" found (accepted → defaults, rejected → documented)

**JAX**
- `num_lm_head_chunks` 8 → **1**: 517→481 ms (−7%). The maxtext-style chunked
  custom_vjp scan exists for 32 GB TPUs; on 96 GB it only adds scan overhead.
  chunks=1 exactly matches a plain unchunked loss (480.9 ms). **Accepted.**
- fp32 logits: 641 ms (+33%). **Rejected** — bf16 logits confirmed.
- XLA flags (latency-hiding scheduler, command buffers, `triton_gemm_any`,
  autotune level 4 + exhaustive tiling): **all within ±1 ms of no flags.**
  jax 0.7.2's GPU defaults are already well-tuned; nothing to gain here.

**PyTorch**
- `torch.compile` default mode: **1.63x over eager** (716 → 440 ms). The single
  biggest lever in either framework.
- max-autotune: −0.8% step time for +93 s warmup — not worth it. Worse, with
  the unchunked CE fusion it **crashed with a CUDA illegal memory access**
  (autotuned kernel bug on sm_120). **Rejected.**
- CE chunking: torch is insensitive (chunks=8: 443.7, chunks=1/full: 439.6-441.5)
  — Inductor fuses the chunk loop either way. Default chunks=1 for symmetry.
- SDPA backend: cuDNN 439.6 vs FlashAttention 442.8 — cuDNN kept.

**Fairness note:** every accepted change was attempted on the other side —
chunking (JAX gains, torch indifferent), flags (JAX-only concept, no-op),
compile mode (torch-only concept).

## Input pipeline: a real-world dispatch difference

Both notebooks share an identical background-thread loader (tokenize FineWeb
on the fly → transfer to device). Diagnostics on the JAX side show the queue
never starves (0.1 ms wait; the tokenizer sustains 4M tok/s, 16× the demand),
yet the loader's mere presence costs JAX **~33 ms/step** (488 ms fixed-batch →
521 ms with loader) vs **~11 ms** for torch (440 → 451). Moving `device_put`
out of the worker thread changes nothing (516.5 ms) — the contention is the
tokenizer's GIL-heavy burst (~32 ms/batch) colliding with step dispatch, to
which JAX's dispatch path is measurably more sensitive. Fix for both (out of
scope here): move tokenization to a subprocess. This is why the end-to-end gap
(15%) exceeds the pure-compute gap (9%).

## Trace-level notes (sm_120 software maturity, mid-2026)

- Both stacks' top gemm kernels are `cutlass_80_tensorop_bf16_s16816gemm_*` —
  **Ampere-generation kernels running on Blackwell**. cuBLAS/cutlass evidently
  has no sm_120-specialized tiles for these shapes yet; the 392 TFLOP/s
  "achievable peak" already bakes that in.
- cuDNN attention: forward runs an **sm120-native** kernel
  (`..._sdpa_sm120_flash_fprop...`), backward falls back to **sm80**
  (`..._sdpa_sm80_flash_bprop...`) in both frameworks — one reason attention
  fwd hits ~150% of the "causal-halved" FLOP estimate while fwd+bwd doesn't.
- torch.compile max-autotune producing an illegal memory access on the CE
  fusion is the same theme: sm_120 is functional everywhere but tuned nowhere.

## Context: vs the TPU v6e baseline

The TPU run of the same-FLOPs model (`08_tpu_ablations.py`): ~435k tok/s at
46.5% MFU of a 918 TFLOP/s peak (~302 ms/step with 16×4 grad accumulation).
The G4 delivers 254-292k tok/s at 63-73% of its real 392 TFLOP/s peak. In other
words: **the GPU software stacks are more efficient relative to their silicon,
but the v6e simply has ~2.3× the dense bf16 compute** and wins on absolute
throughput for this workload.

## Caveats

- One model size (164M), T=2048, B=64, 300-step runs — findings may not
  transfer to larger models, longer contexts, or multi-GPU.
- Warmup times are with warm persistent compile caches (`/content/jax_cache`,
  Inductor cache) and after Section B pre-compiled similar shapes; cold-start
  compile is several minutes for both.
- Different RNG streams (JAX vs torch generators, same distributions); loss
  differences at ±0.06 are noise.
- torch's `F.cross_entropy` accumulates in fp32 internally; optax's CE on bf16
  logits computes log-softmax in bf16 — a negligible, documented asymmetry.
- Attention MFU >100% is expected: `attention_flops` counts the full T×T
  matrix while causal kernels skip half of it.
- Untested headroom: B=128+ batch sizes (96 GB allows far larger batches),
  FP8/NVFP4 (Blackwell hardware supports it; excluded by design).

## Reproducing

```bash
uv tool install google-colab-cli   # then one-time interactive OAuth: colab sessions
bash gpu/run_bench.sh setup        # provisions a G4 session
bash gpu/run_bench.sh jax && bash gpu/run_bench.sh fetch jax
bash gpu/run_bench.sh torch && bash gpu/run_bench.sh fetch torch
uv run python gpu/compare.py gpu/results/metrics_jax.json gpu/results/metrics_torch.json
bash gpu/run_bench.sh stop         # G4 bills compute units — always stop when idle
```
