# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

**tpuchat** is a JAX-native GPT pretraining harness for a single Colab Pro+ TPU v6e (32 GB HBM). Port of Karpathy's NanoChat — raw JAX, no Flax/Orbax.

168M param model: RoPE, QK-norm, ReLU² MLP, sliding window attention, logit softcap, x0 residual connections. Trained on FineWeb-Edu-100B-Shuffle (50 shards, tokenized on-the-fly with custom BPE vocab 32768).

## Editing workflow

1. Edit the `.py` file (source of truth) — jupytext percent format
2. Run `bash update_notebooks.sh` to regenerate `.ipynb` via jupytext
3. Commit both `.py` and `.ipynb`
4. User re-opens notebook in Colab from GitHub

There is no test suite, linter, or build system. Validation is done via training metrics (val loss, throughput, MXU%) in Colab.

## File map

| File | Purpose |
|------|---------|
| `02_train.py` | **Main** — full model, optimizer, training loop (~992 lines) |
| `01_tokenizer.py` | Train BPE tokenizer, upload to HF Hub |
| `03_worker.py` | wandb hyperparameter sweep worker |
| `04_maxtext.py` | MaxText-inspired ~370M variant (SwiGLU, chunked lm_head loss) |
| `05_tpu_perf.py` | TPU v6e performance benchmarks (MXU%, HBM bandwidth) |
| `LOG.md` | Chronological dev log — append with `Agent:` prefix after significant work |
| `update_notebooks.sh` | `jupytext --to ipynb` for all numbered .py files |

## Architecture (02_train.py)

**Config** — frozen dataclass, registered as JAX static type (changes trigger recompilation). `n_head` is the primary scaling knob: `n_embd = n_head × head_dim`, `depth = n_embd / aspect_ratio`.

**dot_dict** — custom JAX pytree that supports dot-notation access. Used for params, optimizer state, and per-layer weights.

**Model (`model_apply`)** — forward pass: token embed → RMSNorm → per-layer (attention + MLP with residual) → final norm → lm_head + softcap. QKV weights shaped `(n_embd, n_head, head_dim)` — einsums produce multi-head shapes directly, no reshapes.

**Attention dispatch** — 4 backends via `config.attn_impl`: `'einsum'` (manual), `'jax'` (dot_product_attention), `'splash'` (Pallas splash kernel), `'pallas'` (flash attention).

**split_trainable / merge_params** — RoPE cos/sin are precomputed and non-trainable. Separated before training step, merged back after. This pattern is incompatible with `donate_argnums`.

**PrefetchDataLoader** — background thread overlaps tokenization + `jax.device_put` with compute.

**Optimizer** — explicit AdamW with warmup/warmdown LR schedule. No optax optimizer wrapper — step function is manual.

**Profiling** — `jax.named_scope` on all components (embedding, layer_N/attention, layer_N/mlp, lm_head) for XProf/TensorBoard traces. Steps 15-20 captured by default.

## Key conventions

- **No requirements.txt** — dependencies installed via `!pip install` cells in notebooks (jax[tpu], optax, tiktoken, pyarrow, huggingface_hub, etc.)
- **Param shapes are explicit** — QKV: `(n_embd, n_head, head_dim)`, c_proj: `(n_head, head_dim, n_embd)`, MLP: `(n_embd, 4*n_embd)`
- **LOG.md** — always append after significant work, prefix with `Agent:`, include metrics when relevant
- **HuggingFace Hub** — tokenizer and checkpoints stored at `vorushin/tpuchat`
- **Data paths assume Colab** — `/content/base_data/`, `/content/tokenizer/`, `/content/log_dir/`
