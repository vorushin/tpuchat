Feb 15. 

Roman: Starting to work on the project.01
Roman: GPT-2 speedrun is trained on 370 FineWeb-Edu-100B-Shuffle shards (eash shard is ~100MB). The tokenizer - on 8 shards.
Agent: Created 01_tokenizer.ipynb — self-contained Colab notebook that downloads 8 shards, trains RustBPE tokenizer (vocab 32768), saves tokenizer.pkl + token_bytes.pt to Google Drive, and evaluates compression vs GPT-2/GPT-4. No nanochat imports needed, all code is inline.
Roman: Running 01_tokenzier.ipynb on a CPU kernel. Using Google Drive is suboptimal - need to give access to the whole my Google Drive (no way to select a folder), switching to HF hub.
Agent: Switched notebook storage to jupytext percent format. We version-control .py files and generate .ipynb via `jupytext --to ipynb`. Added *.ipynb to .gitignore.
Roman: Yes, directly editing ipynb isn't comfortable for the models - it's full of JSON strings with a lot of escaping. Glad there is a .py format support - similar to what we have internally at Google.
Roman: The tokenizer is uploaded to https://huggingface.co/vorushin/tpuchat/tree/main/tokenizer Woo-hoo!

                Vocab     english        code        math   avg ratio
----------------------------------------------------------------------------
GPT-2           50257        6.05        1.94        2.55        3.51
GPT-4          100277        6.05        2.85        3.08        4.00
Ours            32768        6.21        1.84        2.52        3.52

(Ratio = bytes/token. Higher is better = more compression)

Roman: Good to go for the next step. 1 hour of time spent so far.
Agent: Created 02_train.py — raw JAX training notebook (no Flax). Ports the full nanochat GPT architecture: RoPE, QK-norm, GQA, ReLU^2, value embeddings, sliding window (SSSL), logit softcap. Uses AdamW (not Muon) with warmup+warmdown LR schedule. Follows the JAX training cookbook pattern (dot_dict pytrees, functional updates, jax.jit). Starts with depth=12 (~85M params), 50 data shards.