# NanoChat ported to Colab Pro+ TPU

nanochat folder (not commited) contains the Andrej Karpathy project NanoChat - small GPT-2 model with its own tokenizer, trained on 8*H100 in 3 hours.

I am porting it to train on a TPU v6e-1 available on a Colab Pro+ plan.

I am writing small snippets as I go along the work to BLOG_LOG.md. The agent is also welcome to add a few lines of its notes to the log after some significant amount of work is done. Annotate it with "Agent:". I will annotate my lines with "Roman:".

## Notebooks

Notebooks are stored as `.py` files in [jupytext percent format](https://jupytext.readthedocs.io/en/latest/formats-scripts.html#the-percent-format). To convert to `.ipynb` for Colab:

```bash
pip install jupytext
jupytext --to ipynb 01_tokenizer.py
```

- **01_tokenizer.py** — Train a BPE tokenizer (vocab 32768) on FineWeb-Edu data, upload to HuggingFace Hub