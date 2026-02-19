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

### Attention implemenations

Btw, TPU v5e-1 has MXU utilization around 25% - has lower arithmetic intensity. Back to v6e-1 - asked Agent to add optimized attention kernels, 3 different variants. Let's benchmark them.

device_batch_size=4, v6e-1

**default (as before) 'einsum':**

step 00990/01000 (99.0%) | loss: 5.8589 | lr_mult: 0.020 | dt: 127ms | tok/s: 64,455 | eta: 0.0m
Step 01000 | Val loss: 5.9245 (best: 5.9245)

FLOPS Utilization
higher is better.
Utilization of TPU Matrix Units	
14.5%

Profile-window Peak Memory Usage
stack + heap, within profiling window
20.03 GiB
Timestamp: 519.2 ms 
Stack Reservation: 12.75 GiB 
Heap Allocation: 7.27 GiB 
Free Memory: 11.22 GiB 
Fragmentation: 0.00% 

**jax.nn.dot_product_attention:**

step 00990/01000 (99.0%) | loss: 5.8565 | lr_mult: 0.020 | dt: 126ms | tok/s: 65,015 | eta: 0.0m
Step 01000 | Val loss: 5.9222 (best: 5.9222)

Utilization of TPU Matrix Units	
14.6%

Profile-window Peak Memory Usage
stack + heap, within profiling window
20.06 GiB
Timestamp: 518.4 ms 
Stack Reservation: 12.75 GiB 
Heap Allocation: 7.31 GiB 
Free Memory: 11.18 GiB 
Fragmentation: 0.78% 

**Pallas splash kernel**

step 00240/01000 (24.0%) | loss: 6.6512 | lr_mult: 1.000 | dt: 400ms | tok/s: 40,971 | eta: 5.1m

Utilization of TPU Matrix Units	
8.8%

Profile-window Peak Memory Usage
stack + heap, within profiling window
23.80 GiB
Timestamp: 1620.6 ms 
Stack Reservation: 16.01 GiB 
Heap Allocation: 7.79 GiB 
Free Memory: 7.45 GiB 
Fragmentation: 0.32%

block_sizes: 128 -> 256

step 00060/01000 (6.0%) | loss: 7.6033 | lr_mult: 1.000 | dt: 208ms | tok/s: 78,904 | eta: 3.3m

Utilization of TPU Matrix Units	
17.2%

Profile-window Peak Memory Usage
stack + heap, within profiling window
23.31 GiB
Timestamp: 850.0 ms 
Stack Reservation: 16.01 GiB 
Heap Allocation: 7.30 GiB 
Free Memory: 7.94 GiB 
Fragmentation: 1.08% 

block_sizes: 512

step 00190/01000 (19.0%) | loss: 6.8646 | lr_mult: 1.000 | dt: 167ms | tok/s: 98,326 | eta: 2.2m

Utilization of TPU Matrix Units	
22.3%

step 00240/01000 (24.0%) | loss: 6.6497 | lr_mult: 1.000 | dt: 161ms | tok/s: 102,030 | eta: 2.0m

block_sizes: 1024

Utilization of TPU Matrix Units	
25.3%

block_sizes: 2048

Scoped allocation with size 34.08M and limit 32.00M exceeded scoped vmem limit by 2.08M. It should not be possible to run out of scoped vmem - please ...

Roman: 
apply_rope is taking a long time. Changing the implemenation slightly - based on maxtext - didn't help. Reduced number of KV heads 8 -> 2. 8 Q heads still. helped a ton:

step 00490/01000 (49.0%) | loss: 6.0802 | lr_mult: 1.000 | dt: 137ms | tok/s: 119,698 | eta: 1.2m

Utilization of TPU Matrix Units	
27.7%

SSSL pattern -> LLLL
step 00090/01000 (9.0%) | loss: 7.4665 | lr_mult: 1.000 | dt: 137ms | tok/s: 119,945 | eta: 2.1m

Utilization of TPU Matrix Units	
27.8%

block size 1024 -> 512
step 00100/01000 (10.0%) | loss: 7.3906 | lr_mult: 1.000 | dt: 144ms | tok/s: 114,068 | eta: 2.2m

1024 is still optimal.

Checked with Claude Code for the possible optimizations. The biggest issue is head_dim = 128 when v6e MMX unit accepts 256*256 matrices. Trying head_dim = 256, n_head = 4, n_kv_head = 1

step 00090/01000 (9.0%) | loss: 7.4782 | lr_mult: 1.000 | dt: 119ms | tok/s: 137,825 | eta: 1.8m

Utilization of TPU Matrix Units	
25.5%

Nice speedup - lower utilization because before we multipled zeros a lot.

trying diff block sizes: 256, SSSL

step 00500/01000 (50.0%) | loss: 6.1184 | lr_mult: 1.000 | dt: 141ms | tok/s: 116,013 | eta: 1.2m

trying diff block sizes: 256, LLLL

step 00070/01000 (7.0%) | loss: 7.5517 | lr_mult: 1.000 | dt: 153ms | tok/s: 107,009 | eta: 2.4m

let's keep 1024, LLLL

Running a longer run on 100k steps * batch_size 8 * seq 2k = 1.6B tokens

=== Starting training for 100,000 steps ===

Step 00000 | Val loss: 10.3975 (best: 10.3975)
step 00000/100000 (0.0%) | loss: 10.3980 | lr_mult: 0.001 | dt: 33512ms | tok/s: 488
Profiling started...
Profiling stopped. Trace saved to 'log_dir'.
Step 01000 | Val loss: 6.2024 (best: 6.2024)
step 01000/100000 (1.0%) | loss: 6.1779 | lr_mult: 0.501 | dt: 118ms | tok/s: 138,962 | eta: 195.6m

step 79000/100000 (79.0%) | loss: 3.3970 | lr_mult: 0.420 | dt: 117ms | tok/s: 139,858 | eta: 41.4m
Step 80000 | Val loss: 3.3618 (best: 3.3618)

--- Samples (step 80000) ---
Prompt: The capital of France is
Output: <|bos|>The capital of France is Paris. Paris is the most populous city in Europe, with a population of 9.7 million people. It is the capital of France. It is a centre of industrial activity and for a long time it has been a city of merchants and merchants, for trade and trade. At the time of its founding, Paris

Prompt: In a distant galaxy, scientists discovered
Output: <|bos|>In a distant galaxy, scientists discovered a supermassive black hole that has a mass of about 2 billion times the mass of the Sun. The black hole’s mass is approximately 10 times larger than the galaxy’s own Sun, and the mass of the black hole is only 0.2 suns.
The black hole’s mass is about 

Prompt: The quick brown fox
Output: <|bos|>The quick brown fox is found in the Southeast of North Carolina in the Great Plains and its northern ranges. The fox is a common type of fox that is known to breed all over the United States, from the Mexican to the American west.
The fox has a long, narrow, or flat bottom that allows it to reach heights of 30

Prompt: Machine learning is
Output: <|bos|>Machine learning is an area of research that involves the use of machine learning to improve a particular technology or system. This area of research can include fields such as machine learning, image classification, machine learning, data analytics, and other types of artificial intelligence. The purpose of machine learning is to improve an system’s performance by finding out what people


Much better than after prev run at 20k steps (bs 4) - 0.1 of the current larger run.

step 99000/100000 (99.0%) | loss: 3.2519 | lr_mult: 0.020 | dt: 118ms | tok/s: 139,213 | eta: 2.0m
Step 100000 | Val loss: 3.2509 (best: 3.2509)

--- Samples (step 100000) ---
Prompt: The capital of France is
Output: <|bos|>The capital of France is the capital of the United Kingdom. It is a country of considerable economic and social importance. The capital is Paris. There are 44 cities, and the city of Paris is the largest in France. The capital is Paris. The capital of the Republic of France is Paris.
The capital of France is Paris. It is

Prompt: In a distant galaxy, scientists discovered
Output: <|bos|>In a distant galaxy, scientists discovered that the first humans lived in the far distant past – 3.5 billion years ago.
“It was a big surprise to me to see that, more than 2.5 billion years ago, we lived in a galaxy that’s now 1.5 billion years older,” says Stephen Doy, a professor at

Prompt: The quick brown fox
Output: <|bos|>The quick brown fox is one of the most common and widespread species of fox, although there are currently only 2 subspecies found in Australia. Their range is primarily in South East Asia, Central and South East Asia, the Indo-Australian region and South East Asia. Its range is also in the Indian subcontinent, from the coast of

Prompt: Machine learning is
Output: <|bos|>Machine learning is the use of machine learning to make machines learn from data. Machine learning is the study of data and machine learning is the use of data to make machines learn. The most basic form of machine learning is a deep learning.
We shall see how machine learning works and how machine learning can be applied to other fields in the

----------------------------

Training complete. Total time: 197.3m
Best val loss: 3.2509

## Feb 17 2026
Roman:
Let's dig for better ways of integration with Colab - can we have a way where the Agent can launch and track experiments without Roman reloading and running things manually. We have a Colab Pro+ account with a lot of cool features. Let's figure out how to use them for this project.

While Agent is working on this, I am quadrupling the size of the LLM: 8 query heads, 32 layers, batch_size 2. Let's see if this changes the MXU utilization. HBM OOM. Trying device_batch_size=1. Still OOM. Trying a smaller model.

Model: depth=24, n_embd=1536, n_head=6, n_kv_head=1, head_dim=256, vocab=32768 (padded=32768)
Params: 685.8M total (embed: 50.3M, attn: 132.1M, mlp: 453.0M, lm_head: 50.3M)
Total batch size: 2,048 tokens/step (1×2048)
Scaling params: 685,768,704
Target tokens: 7,200,571,392
Num iterations: 100
Estimated training tokens: 204,800

step 00090/00100 (90.0%) | loss: 7.6009 | lr_mult: 0.200 | dt: 43ms | tok/s: 47,366 | eta: 0.0m
Step 00100 | Val loss: 7.6175 (best: 7.6175)

Utilization of TPU Matrix Units	
24.7%

The larger model is not leading to the better MXU utilization.

Agent: Created 03_worker.py — wandb hyperparameter sweep worker notebook. Self-contained jupytext notebook with all model/training code copied from 02_train.py. Uses wandb Sweeps for coordination: Bayesian search over learning_rate, n_head, n_kv_head, head_dim, device_batch_size, warmup_ratio, warmdown_ratio. Hyperband early termination (min_iter=3, eta=3) kills bad runs early. Run cap of 30. Two-phase usage: (1) run once to create sweep and get sweep_id, (2) paste sweep_id into agent cell on each Colab Pro+ instance. Invalid configs (e.g. n_head % n_kv_head != 0) log val_loss=999 and return immediately. Updated README.md with notebook entry.

Roman: Running a first sweep:

sweep_config = {
    "name": "tpuchat-hparam-sweep",
    "method": "bayes",
    "metric": {"goal": "minimize", "name": "val_loss"},
    "early_terminate": {"type": "hyperband", "min_iter": 3, "eta": 3},
    "run_cap": 30,
    "parameters": {
        "learning_rate": {"distribution": "log_uniform_values", "min": 5e-5, "max": 1e-3},
        "weight_decay": {"distribution": "uniform", "min": 0.01, "max": 0.2},
    },
}

https://wandb.ai/vorushin/tpuchat/sweeps/e67oc6lq

Running 10 runs times 2 colab instances.

While it's running trying to reach higher MMX utilization in 02_train.py - editing in Colab UI.

TPU v6e bfloat16 arithmetic intensity is ~560 FLOPs/byte.
Trying to boost the sizes of all tensors.

Agent: Recommendation: learning_rate=7e-4, weight_decay=0.1

  This is right in the middle of the winning cluster. The current default (3e-4) is on the low side — bumping to 7e-4 should give ~0.1 lower val_loss based on the
  sweep. Weight decay can stay at 0.1 (current default) since it's not a significant factor in this range.

  The top 5 runs are within 0.04 of each other, so there's not much to squeeze from further tuning these two params. The next sweep should explore architectural knobs
   (n_head, head_dim, n_kv_head) which likely have bigger impact.

Roman: Not much luck with getting to the higher MXU in the meantime.
Roman: Asked for a next notebook with maxtext-inspired 200M-400M model with the dimensions optimal for v6e-1.
Agent: Created 04_maxtext.py — MaxText-inspired ~370M param model. 5 changes from 02_train.py: (1) SwiGLU MLP replaces ReLU^2 (w_gate/w_up/w_down), (2) explicit dimensions in Config (n_embd=1024, n_layer=24, mlp_dim=3072) with 256-alignment assertions instead of derived aspect_ratio, (3) standard pre-norm residual (no x0 connection), (4) lr=7e-4 from sweep results, (5) updated param names. All matrix dims are multiples of 256 for perfect v6e MXU tiling. Everything else (data pipeline, tokenizer, optimizer, splash attention, RoPE, QK-norm, softcap, profiling, eval, plotting, checkpoint) copied verbatim.


## Feb 18 2026
Roman:
Before chunking logits: 

Profile-window Peak Memory Usage
stack + heap, within profiling window
29.80 GiB
Timestamp: 843.6 ms 
Stack Reservation: 20.83 GiB 
Heap Allocation: 8.97 GiB 
Free Memory: 1.44 GiB 
Fragmentation: 8.65% 

After chunking logits:

With 8 chunks:

Profile-window Peak Memory Usage
stack + heap, within profiling window
28.39 GiB
Timestamp: 813.9 ms 
Stack Reservation: 18.93 GiB 
Heap Allocation: 9.46 GiB 
Free Memory: 2.86 GiB 
Fragmentation: 0.03% 

And no more huge tensors in the Memory Viewer - used to domitate everything by the size.
Agent: Added chunked lm_head loss to 04_maxtext.py (commit 456233e). Factored final norm + lm_head out of model_apply — model now returns hidden state (B, T, n_embd). New apply_lm_head used by eval_step/predict_step. New chunked_lm_head_loss tiles B×T into num_lm_head_chunks=8 slices, computes cross-entropy per chunk via jax.lax.scan, and uses a custom VJP that saves only hidden/lm_head/labels as residuals (no logits) — recomputing logits chunk-by-chunk in the backward pass. Pattern adapted from MaxText vocabulary_tiling.py. Stack reduction ~2 GiB (20.83→18.93 GiB), fragmentation dropped from 8.65% to 0.03%, free memory up from 1.44→2.86 GiB.

## Feb 19 2026
Agent: Fixed 05_tpu_perf.py benchmark harness — two issues. (1) MXU was over 100% because PEAK_TFLOPS was hardcoded to 197 (wrong estimate). Updated to 918 TFLOPS from official v6e docs (https://docs.cloud.google.com/tpu/docs/v6e). (2) HBM reporting showed bytes_in_use (capacity) instead of bandwidth utilization. Replaced with HBM BW% = (bytes read+written) / wall_time / 1600 GB/s × 100. Added hbm_bytes parameter to benchmark(). Phase 1 matmuls now report both MXU% and HBM BW%. The 8192×8192 matmul at ~622 TFLOP/s should now show ~68% MXU instead of 316%.
