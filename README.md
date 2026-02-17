# tpuchat — GPT pretraining on a single TPU

Port of [NanoChat](https://github.com/karpathy/nanochat) (Andrej Karpathy's GPT-2 speedrun) to train on a **single Colab Pro+ TPU v6e** (32 GB HBM) using raw JAX (no Flax/Orbax).

## Current status (Feb 16, 2026)

**Model:** 168M params (n_head=8, n_embd=1024, depth=16, head_dim=128, seq_len=2048)
- Architecture: RoPE, QK-norm, ReLU² MLP, sliding window attention (SSSL pattern), logit softcap (15.0), x0 residual connections
- `n_head` is the primary scaling knob — `n_embd = n_head × head_dim`, `depth = n_embd / aspect_ratio`
- No value embeddings (removed — param cost outweighed benefit at this scale)
- Separate wte (embed) + lm_head (unembed) — weight tying was tried but regressed loss due to init scale × softcap interaction

**Training:** 1K steps, device_batch_size=4, ~65K tok/s on v6e, val loss ~5.9 at step 1000
- AdamW optimizer (lr=3e-4, warmup 2%, warmdown 50%)
- Data: 50 shards from FineWeb-Edu-100B, tokenized on-the-fly with custom BPE (vocab 32768)
- Profiling: XProf annotations (`jax.named_scope`) on all model components, TensorBoard integration

**Attention:** 4 switchable implementations via `Config(attn_impl=...)`:
- `'einsum'` — manual QK^T/softmax/AV, supports sliding window (default)
- `'jax'` — `jax.nn.dot_product_attention`, supports sliding window via mask
- `'splash'` — Pallas splash kernel (used in MaxText/Gemma), supports sliding window
- `'pallas'` — Pallas flash attention, causal only

**MXU utilization:** ~14.5% at device_batch_size=4 — model is memory-bandwidth-bound, not compute-bound at this scale. Larger batch sizes help but hit HBM limits (batch_size=6 uses 28.8 GiB / 32 GiB).

## Repo structure

```
02_train.py          Main training notebook (jupytext percent format)
03_worker.py         Hyperparameter sweep worker (wandb sweeps)
01_tokenizer.py      Tokenizer training notebook
LOG.md               Chronological development log (Roman: / Agent: entries)
update_notebooks.sh  Converts .py → .ipynb via jupytext
nanochat/            Reference nanochat repo (not committed, .gitignored)
```

## Agent workflow

### Editing code
1. Edit the `.py` file (e.g. `02_train.py`) — this is the source of truth
2. Run `bash update_notebooks.sh` to regenerate `.ipynb` files
3. `git add -A && git commit -m "..." && git push`
4. User re-opens the notebook in Colab from GitHub to pick up changes

### Key conventions
- **LOG.md** — append entries after significant work. Prefix with `Agent:`. Roman prefixes his with `Roman:`.
- **Config is frozen** — all model/training params in the `Config` dataclass. `Config` is registered as a JAX static type, so changes trigger recompilation.
- **Weights use explicit head dims** — QKV weights are `(n_embd, n_head, head_dim)`, c_proj is `(n_head, head_dim, n_embd)`. Einsums produce multi-head shapes directly, no reshapes.
- **split_trainable/merge_params** — RoPE parameters (rope_cos, rope_sin) are separated as non-trainable. This pattern is incompatible with `donate_argnums`.
- **PrefetchDataLoader** — background thread does `jax.device_put` to overlap host→device transfer with compute.
- **Param counting** — reported as embed / attn / mlp / lm_head separately for analysis.

## Notebooks

Notebooks are stored as `.py` files in [jupytext percent format](https://jupytext.readthedocs.io/en/latest/formats-scripts.html#the-percent-format) for readable diffs. The corresponding `.ipynb` files are also committed so you can open them directly in Colab from GitHub.

| Notebook | Open in Colab | Description |
|----------|--------------|-------------|
| `01_tokenizer.py` | [Open](https://colab.research.google.com/github/vorushin/tpuchat/blob/master/01_tokenizer.ipynb) | Train BPE tokenizer (vocab 32768), upload to HuggingFace Hub |
| `02_train.py` | [Open](https://colab.research.google.com/github/vorushin/tpuchat/blob/master/02_train.ipynb) | Pretrain GPT model in raw JAX on single TPU |
| `03_worker.py` | [Open](https://colab.research.google.com/github/vorushin/tpuchat/blob/master/03_worker.ipynb) | wandb hyperparameter sweep worker |
| `04_maxtext.py` | [Open](https://colab.research.google.com/github/vorushin/tpuchat/blob/master/04_maxtext.ipynb) | MaxText-inspired ~370M model (SwiGLU, 256-aligned dims) |