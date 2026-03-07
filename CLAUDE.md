# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Local development

Use `uv` for Python package management. Run scripts with `uv run` (e.g. `uv run python script.py`).

## Project overview

**tpuchat** is a JAX-native GPT pretraining harness for a single Colab Pro+ TPU v6e (32 GB HBM). Port of Karpathy's NanoChat — raw JAX, no Flax/Orbax.

164M param model (130M non-embed): D=1024, N=4, K=1, H=256, F=3072, L=8, B=64 (16×4 microbatch). RoPE, QK-norm, SwiGLU MLP, splash attention, logit softcap. Trained on FineWeb-Edu-100B-Shuffle (50 shards, tokenized on-the-fly with custom BPE vocab 32768).

## Editing workflow

1. Edit the `.py` file (source of truth) — jupytext percent format
2. Run `bash update_notebooks.sh` to regenerate `.ipynb` via jupytext
3. Commit both `.py` and `.ipynb`
4. User re-opens notebook in Colab from GitHub

There is no test suite, linter, or build system. Validation is done via training metrics (val loss, throughput, MXU%) in Colab.

### Notebook revisions

Notebooks with a `REVISION = N` constant (currently 08) get auto-incremented by `update_notebooks.sh` when content changes. The revision appears in the notebook title and is included in checkpoint metadata uploaded to HuggingFace. The script uses `.notebook_hashes` (gitignored) to track content changes.

### Colab secrets

All notebooks use `google.colab.userdata` for authentication (no interactive prompts):
- `HF_TOKEN` — HuggingFace Hub login
- `WANDB_TOKEN` — Weights & Biases login (notebooks 03, 08, 09)

## File map

| File | Purpose |
|------|---------|
| `02_train.py` | **Main** — full model, optimizer, training loop (~992 lines) |
| `01_tokenizer.py` | Train BPE tokenizer, upload to HF Hub |
| `03_worker.py` | wandb hyperparameter sweep worker |
| `04_maxtext.py` | MaxText-inspired ~370M variant (SwiGLU, chunked lm_head loss) |
| `05_tpu_perf.py` | TPU v6e performance benchmarks (MXU%, HBM bandwidth) |
| `08_tpu_ablations.py` | Ablation lab — quick training, wandb sweep, hero run with HF upload |
| `09_moe.py` | MoE training lab — 8 experts, top-2 routing, capacity-based dispatch, wandb sweep, hero run |
| `LOG.md` | Chronological dev log — append with `Agent:` prefix after significant work |
| `update_notebooks.sh` | `jupytext --to ipynb` + auto-increment REVISION for changed notebooks |

## Architecture (02_train.py)

**Config** — frozen dataclass, registered as JAX static type (changes trigger recompilation). `n_head` is the primary scaling knob: `n_embd = n_head × head_dim`, `depth = n_embd / aspect_ratio`.

**dot_dict** — custom JAX pytree that supports dot-notation access. Used for params, optimizer state, and per-layer weights.

**Model (`model_apply`)** — forward pass: token embed → RMSNorm → per-layer (attention + MLP with residual) → final norm → lm_head + softcap. QKV weights shaped `(n_embd, n_head, head_dim)` — einsums produce multi-head shapes directly, no reshapes.

**Attention dispatch** — 4 backends via `config.attn_impl`: `'einsum'` (manual), `'jax'` (dot_product_attention), `'splash'` (Pallas splash kernel), `'pallas'` (flash attention).

**split_trainable / merge_params** — RoPE cos/sin are precomputed and non-trainable. Separated before training step, merged back after. This pattern is incompatible with `donate_argnums`.

**PrefetchDataLoader** — background thread overlaps tokenization + `jax.device_put` with compute.

**Optimizer** — optax AdamW with warmup/warmdown LR schedule. Always use `optax.adamw()` — manual per-leaf loops kill MFU on TPU (see LOG.md).

**Gradient accumulation** — `jax.lax.scan` over microbatches (microbatch_size=4, num_microbatches=16 for batch_size=64).

**Profiling** — `jax.named_scope` on all components (embedding, layer_N/attention, layer_N/mlp, lm_head) for XProf/TensorBoard traces. Steps 15-20 captured by default.

## Dimension letters

Single-letter notation from [How to Scale Your Model](https://jax-ml.github.io/scaling-book/transformers/). Use these consistently in comments, formulas, print statements, and FLOP counting functions.

| Letter | Meaning | Config field |
|--------|---------|-------------|
| B | Batch size | `batch_size` |
| T | Sequence length | `seq_len` |
| D | Model dimension | `n_embd` |
| N | Number of query heads | `n_head` |
| K | Number of KV heads | `n_kv_head` |
| H | Head dimension | `head_dim` |
| F | FFN hidden dimension | `mlp_dim` / `expert_mlp_dim` |
| L | Number of layers | `n_layer` |
| V | Vocabulary size | `vocab_size` |
| E | Number of experts | `n_experts` |
| k | Active experts per token | `n_active_experts` |

## Pallas kernel development (pallas/)

- `pallas/pallas_test.py` — local CPU correctness tests (Pallas interpret mode)
- `pallas/colab_server.py` — jupytext notebook, open in Colab to start code execution server
- `pallas/colab_client.py` — send code to Colab TPU, get results
- Connection: save `URL|TOKEN` from Colab server output to `pallas/.colab_connection`

## Key conventions

- **No requirements.txt** — dependencies installed via `!pip install` cells in notebooks (jax[tpu], optax, tiktoken, pyarrow, huggingface_hub, etc.)
- **Param shapes are explicit** — QKV: `(n_embd, n_head, head_dim)`, c_proj: `(n_head, head_dim, n_embd)`, MLP: `(n_embd, mlp_dim)`
- **LOG.md** — always append after significant work, prefix with `Agent:`, include metrics when relevant
- **HuggingFace Hub** — tokenizer and checkpoints stored at `vorushin/tpuchat`
- **Data paths assume Colab** — `/content/base_data/`, `/content/tokenizer/`, `/content/log_dir/`
- **Checkpoint format** — `params.pkl` (JAX arrays → numpy pickle) + `config.json` (Config dict + revision + training metrics), uploaded via `HfApi.upload_folder()`
- **Runtime shutdown** — hero runs end with `runtime.unassign()` to auto-disconnect Colab and stop billing
