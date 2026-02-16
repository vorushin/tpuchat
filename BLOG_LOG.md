## Feb 15 2026

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

Roman: v6e1, random data
step 00010/01000 (1.0%) | loss: 10.0966 | lr_mult: 0.550 | dt: 153ms | tok/s: 107,322
Profiling started...
Profiling stopped. Trace saved to 'log_dir'.
step 00020/01000 (2.0%) | loss: 9.1600 | lr_mult: 1.000 | dt: 153ms | tok/s: 107,406 | eta: 2.5m
step 00030/01000 (3.0%) | loss: 7.6790 | lr_mult: 1.000 | dt: 152ms | tok/s: 107,783 | eta: 2.5m
step 00040/01000 (4.0%) | loss: 6.0094 | lr_mult: 1.000 | dt: 153ms | tok/s: 107,235 | eta: 2.4m
step 00050/01000 (5.0%) | loss: 4.3529 | lr_mult: 1.000 | dt: 153ms | tok/s: 107,039 | eta: 2.4m

real data:
step 00020/01000 (2.0%) | loss: 9.4643 | lr_mult: 1.000 | dt: 152ms | tok/s: 107,623 | eta: 2.5m
step 00030/01000 (3.0%) | loss: 8.4961 | lr_mult: 1.000 | dt: 153ms | tok/s: 106,876 | eta: 2.5m
step 00040/01000 (4.0%) | loss: 8.0154 | lr_mult: 1.000 | dt: 153ms | tok/s: 107,076 | eta: 2.4m
step 00050/01000 (5.0%) | loss: 7.8481 | lr_mult: 1.000 | dt: 153ms | tok/s: 107,004 | eta: 2.4m
step 00060/01000 (6.0%) | loss: 7.7605 | lr_mult: 1.000 | dt: 153ms | tok/s: 106,966 | eta: 2.4m
step 00070/01000 (7.0%) | loss: 7.7359 | lr_mult: 1.000 | dt: 153ms | tok/s: 107,054 | eta: 2.4m
step 00080/01000 (8.0%) | loss: 7.7449 | lr_mult: 1.000 | dt: 153ms | tok/s: 107,271 | eta: 2.3m

The data loading is clearly not a bottleneck.
Agent: Added jax.named_scope XProf annotations throughout model (embedding, layer_N/attention, layer_N/mlp, lm_head) and train_step (forward_backward, optimizer). Added use_random_data checkbox — pre-generates random tensors on HBM to isolate pure compute from data loading in profiles.
Agent: Performance audit against JAX/TPU checklist. donate_argnums didn't work (split_trainable/merge_params pattern incompatible). Applied two working optimizations: (1) moved jax.device_put into PrefetchDataLoader background thread to overlap host→device transfer with compute, (2) JIT-compiled eval_step replacing per-call closure that retraced every eval.
Agent: Scaled model — n_head is now the primary scaling knob (was depth). n_head=8 → n_embd=1024, depth=16 (derived as n_embd/aspect_ratio). All tensor dims now 128-aligned for TPU MXU. ~419M params, fits in v6e 32GB HBM.
Agent: XProf showed attention taking 5x longer than MLP. Root cause: jax.nn.dot_product_attention uses vmap internally. Replaced with direct einsum: scores=einsum('bthd,bshd->bhts'), softmax, attn_out=einsum('bhts,bshd->bthd'). Eliminates vmap dispatch overhead.

## Feb 16 2026
Roman: 
Value Embeddings (VE) uses a lot of params. Let's disable (manually in a Colab interface, later have to be done in the repo).
****With VE on**:

device_batch_size = 4

step 00080/01000 (8.0%) | loss: 7.5212 | lr_mult: 1.000 | dt: 140ms | tok/s: 58,375 | eta: 2.1m

**With VE off**:
step 00080/01000 (8.0%) | loss: 7.5225 | lr_mult: 1.000 | dt: 126ms | tok/s: 64,808 | eta: 1.9m

Utilization of TPU Matrix Units	
14.5%

Profile-window Peak Memory Usage
stack + heap, within profiling window
22.07 GiB
Timestamp: 520.4 ms 
Stack Reservation: 12.67 GiB 
Heap Allocation: 9.40 GiB 
Free Memory: 9.18 GiB 
Fragmentation: 43.18%

Nice win!

Next thing to try is to share embed/unebmed tensors. Should save some HBM.
Agent: Removed Value Embeddings (ve_embed, ve_gate, has_ve) entirely. Tied embed/unembed weights — removed separate lm_head param, using wte.T in forward pass via einsum('btd,vd->btv'). Standard practice from GPT-2/T5/LLaMA. Init stays normal(0,1) for the shared wte.

Roman:
device_batch_size = 4

step 00080/01000 (8.0%) | loss: 12.9113 | lr_mult: 1.000 | dt: 127ms | tok/s: 64,490 | eta: 1.9m

We didn't expect a speed-up, but expected lower HBM usage.

Profile-window Peak Memory Usage
stack + heap, within profiling window
21.68 GiB
Timestamp: 520.3 ms 
Stack Reservation: 13.05 GiB 
Heap Allocation: 8.63 GiB 
Free Memory: 9.57 GiB 
Fragmentation: 49.41% 

We use slightly less HBM now.

device_batch_size = 8 fits now (most likely because of removed VE), but tok/s is surprisingly smaller

step 00080/01000 (8.0%) | loss: 12.7541 | lr_mult: 1.000 | dt: 288ms | tok/s: 56,924 | eta: 4.4m

With the shared embed/unembed tensors the learning is happening much much slower (see the diff in the loss values at step 80). I can also see that the decoding of the examples every 250 steps is pretty dump. Interesting.

Agent: The quickest fix would be to scale the logits by 1/√n_embd when using tied weights. This brings initial logit magnitude from ~32 back down to ~1, well within the softcap's linear regime.

Roman:

step 00080/01000 (8.0%) | loss: 8.3550 | lr_mult: 1.000 | dt: 289ms | tok/s: 56,695 | eta: 4.4m

step 00490/01000 (49.0%) | loss: 6.5318 | lr_mult: 1.000 | dt: 288ms | tok/s: 56,972 | eta: 2.5m
Step 00500 | Val loss: 6.5308 (best: 6.5308)

Reverting to the separate embed/unembed.

step 00490/01000 (49.0%) | loss: 6.0491 | lr_mult: 1.000 | dt: 291ms | tok/s: 56,397 | eta: 2.5m
Step 00500 | Val loss: 6.0524 (best: 6.0524)

Let's stick with it for now.

Let's experiment with device_batch_size values:

**device_batch_size = 1**

step 00990/01000 (99.0%) | loss: 6.6651 | lr_mult: 0.020 | dt: 37ms | tok/s: 56,048 | eta: 0.0m
Step 01000 | Val loss: 6.6013 (best: 6.6013)

**device_batch_size = 2**

step 00990/01000 (99.0%) | loss: 6.2589 | lr_mult: 0.020 | dt: 65ms | tok/s: 63,266 | eta: 0.0m
Step 01000 | Val loss: 6.2527 (best: 6.2527)

Profile-window Peak Memory Usage
stack + heap, within profiling window
13.26 GiB
Timestamp: 267.5 ms 
Stack Reservation: 6.04 GiB 
Heap Allocation: 7.21 GiB 
Free Memory: 17.99 GiB 
Fragmentation: 0.56% 

**device_batch_size = 4**

step 00990/01000 (99.0%) | loss: 5.8556 | lr_mult: 0.020 | dt: 126ms | tok/s: 64,765 | eta: 0.0m
Step 01000 | Val loss: 5.9217 (best: 5.9217)

Profile-window Peak Memory Usage
stack + heap, within profiling window
19.87 GiB
Timestamp: 524.1 ms 
Stack Reservation: 12.67 GiB 
Heap Allocation: 7.20 GiB 
Free Memory: 11.37 GiB 
Fragmentation: 1.12% 

**device_batch_size = 6**

step 00090/01000 (9.0%) | loss: 7.4390 | lr_mult: 1.000 | dt: 198ms | tok/s: 62,095 | eta: 3.0m

Profile-window Peak Memory Usage
stack + heap, within profiling window
28.78 GiB
Timestamp: 809.6 ms 
Stack Reservation: 21.86 GiB 
Heap Allocation: 6.92 GiB 
Free Memory: 2.47 GiB 
Fragmentation: 5.71% 


Trained batch_size=4, 20000 steps, it's not bad:

step 19900/20000 (99.5%) | loss: 3.8105 | lr_mult: 0.010 | dt: 129ms | tok/s: 63,493 | eta: 0.2m
Step 20000 | Val loss: 3.8809 (best: 3.8809)

--- Samples (step 20000) ---
Prompt: The capital of France is
Output: <|bos|>The capital of France is one of the richest and most productive of all cities in Europe and this is also the largest city in the world. The capital of France is in the heart of the town of Dicas. It is located in the town of Cologne, now named after the city of Dicas. It is the world

Prompt: In a distant galaxy, scientists discovered
Output: <|bos|>In a distant galaxy, scientists discovered a new object: the star that once covered the solar system.
This is a bright star that is about 12 light years from the Sun. This was the first time scientists discovered that this young star is too light to be seen, by the star's optical system.
The team, who led the study, says the

Prompt: The quick brown fox
Output: <|bos|>The quick brown fox (cuckus pomentus) is a large, small, bird-like cat.
The white male is often seen with a white bill, which is a small black bill. The brown is also known as the red wolf, but may be found in the southern part of the United States.
In the wild,

Prompt: Machine learning is
Output: <|bos|>Machine learning is a research and learning method that is used to define and construct theories and ideas of ideas and to build knowledge. The research techniques that are used to form this understanding can be used to create knowledge and knowledge that can be obtained through the research methods.
The research methods for collecting data from a variety of different sources have been described

----------------------------

Training complete. Total time: 43.4m
Best val loss: 3.8809


Btw, TPU v5e-1 has MXU utilization around 25% - has lower arithmetic intensity. Back to v6e-1 - asked Agent to add optimized attention kernels, 3 diffirent variants. Let's benchmark them.

