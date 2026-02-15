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
Agent (prompted by Roman): First 1K-step debug run on free TPU v5e-1 (12GB HBM). Using gradient accumulation (device_batch_size=2, grad_accum_steps=8). Val loss: 10.40 → 5.27 in 18 minutes. Model generates repetitive text ("the first to be the first to be...") — expected for 1K steps.
Roman: batch_size=8 doesn't fit into 12G HBM. Thus using gradient accumulaiton.
Agent: Added JAX profiler (steps 15-20) + TensorBoard to visualize traces. Profile revealed two bottlenecks: (1) input-bound — TPU idle waiting for data, (2) slow scatter-add in backward pass from manual log_softmax+indexing loss. Fixed both: added PrefetchDataLoader (background thread tokenization), replaced manual loss with optax.softmax_cross_entropy_with_integer_labels (fused kernel).
Roman: Switched from free TPU v5e-1 to Colab Pro+ TPU v6e (32GB HBM). batch_size=8 fits now — no more gradient accumulation needed.
Agent: Holistic cleanup of 02_train.py (-125 lines). Removed gradient accumulation, consolidated compute_grads + apply_optimizer + train_step_accum into a single @jax.jit train_step. Fixed several bugs: undefined avg_loss variable, missing import optax, accidentally deleted train_step function. Moved TensorBoard cell above training loop for easy profile reload. device_batch_size=8, total_batch_size=16384 tokens/step.