# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# <a href="https://colab.research.google.com/github/vorushin/tpuchat/blob/master/01_tokenizer.ipynb?flush_caches=true" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
#
# # 01 — Train BPE Tokenizer
#
# Self-contained notebook that:
# 1. Downloads 8 data shards from FineWeb-Edu-100B-Shuffle (~800MB)
# 2. Trains a BPE tokenizer (vocab 32768) on ~2B characters
# 3. Evaluates compression ratio vs GPT-2 and GPT-4 tokenizers
# 4. Saves tokenizer + `token_bytes.pt` to HuggingFace Hub
#
# Ported from [nanochat](https://github.com/karpathy/nanochat) by Andrej Karpathy.

# %%
# !pip install -q rustbpe tiktoken tokenizers pyarrow requests torch huggingface_hub

# %%
# Login to HuggingFace Hub for saving the trained tokenizer
from huggingface_hub import login, HfApi
from google.colab import userdata
login(token=userdata.get("HF_TOKEN"))

HF_REPO_ID = 'vorushin/tpuchat'  # change this to your HF username/repo
LOCAL_TOKENIZER_DIR = '/content/tokenizer'

import os
os.makedirs(LOCAL_TOKENIZER_DIR, exist_ok=True)
print(f'Tokenizer will be saved locally to: {LOCAL_TOKENIZER_DIR}')
print(f'Then uploaded to: https://huggingface.co/{HF_REPO_ID}')

# %%
# Download 8 data shards from HuggingFace (~800MB total)
# Each shard is ~100MB compressed parquet, ~250M chars of text

import time
import requests
from multiprocessing import Pool

BASE_URL = 'https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main'
DATA_DIR = '/content/base_data'
NUM_SHARDS = 8
os.makedirs(DATA_DIR, exist_ok=True)

def download_shard(index):
    filename = f'shard_{index:05d}.parquet'
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        print(f'Skipping {filename} (exists)')
        return True
    url = f'{BASE_URL}/{filename}'
    print(f'Downloading {filename}...')
    for attempt in range(1, 4):
        try:
            resp = requests.get(url, stream=True, timeout=60)
            resp.raise_for_status()
            tmp = filepath + '.tmp'
            with open(tmp, 'wb') as f:
                for chunk in resp.iter_content(1024 * 1024):
                    if chunk:
                        f.write(chunk)
            os.rename(tmp, filepath)
            print(f'Downloaded {filename}')
            return True
        except Exception as e:
            print(f'Attempt {attempt}/3 failed: {e}')
            for p in [filepath + '.tmp', filepath]:
                if os.path.exists(p):
                    os.remove(p)
            if attempt < 3:
                time.sleep(2 ** attempt)
    return False

t0 = time.time()
with Pool(4) as pool:
    results = pool.map(download_shard, range(NUM_SHARDS))
print(f'\nDownloaded {sum(results)}/{NUM_SHARDS} shards in {time.time()-t0:.1f}s')

# %%
# Tokenizer constants and class (self-contained port from nanochat/tokenizer.py)

import pickle
import rustbpe
import tiktoken
from functools import lru_cache

SPECIAL_TOKENS = [
    '<|bos|>',
    '<|user_start|>', '<|user_end|>',
    '<|assistant_start|>', '<|assistant_end|>',
    '<|python_start|>', '<|python_end|>',
    '<|output_start|>', '<|output_end|>',
]

# GPT-4 style split pattern (with 1,2 instead of 1,3 for smaller vocab)
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class RustBPETokenizer:
    """BPE tokenizer: trains with rustbpe, uses tiktoken for fast inference."""

    def __init__(self, enc, bos_token):
        self.enc = enc
        self.bos_token_id = self.enc.encode_single_token(bos_token)

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        tokenizer = rustbpe.Tokenizer()
        vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
        assert vocab_size_no_special >= 256
        tokenizer.train_from_iterator(
            text_iterator, vocab_size_no_special, pattern=SPLIT_PATTERN
        )
        # Build tiktoken Encoding from trained merge table
        pattern = tokenizer.get_pattern()
        mergeable_ranks_list = tokenizer.get_mergeable_ranks()
        mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list}
        tokens_offset = len(mergeable_ranks)
        special_tokens = {
            name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)
        }
        enc = tiktoken.Encoding(
            name='rustbpe',
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens,
        )
        bos_token = '<|bos|>'
        return cls(enc, bos_token)

    @classmethod
    def from_directory(cls, tokenizer_dir):
        with open(os.path.join(tokenizer_dir, 'tokenizer.pkl'), 'rb') as f:
            enc = pickle.load(f)
        bos_token = '<|bos|>'
        return cls(enc, bos_token)

    @classmethod
    def from_pretrained(cls, tiktoken_name):
        enc = tiktoken.get_encoding(tiktoken_name)
        # GPT-2 uses endoftext as its BOS/document delimiter
        eot = '<|endoftext|>'
        return cls(enc, eot)

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_special_tokens(self):
        return self.enc.special_tokens_set

    def encode(self, text, num_threads=8):
        if isinstance(text, str):
            return self.enc.encode_ordinary(text)
        elif isinstance(text, list):
            return self.enc.encode_ordinary_batch(text, num_threads=num_threads)
        else:
            raise ValueError(f'Invalid input type: {type(text)}')

    def decode(self, ids):
        return self.enc.decode(ids)

    def save(self, tokenizer_dir):
        os.makedirs(tokenizer_dir, exist_ok=True)
        pickle_path = os.path.join(tokenizer_dir, 'tokenizer.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(self.enc, f)
        print(f'Saved tokenizer encoding to {pickle_path}')

print('RustBPETokenizer class defined.')

# %%
# Train the BPE tokenizer on ~2B characters from the downloaded shards

import pyarrow.parquet as pq

MAX_CHARS = 2_000_000_000  # 2B characters
DOC_CAP = 10_000           # max chars per document
VOCAB_SIZE = 32768         # 2^15

def text_iterator():
    """Iterate over documents from parquet shards, yielding capped text."""
    parquet_files = sorted([
        f for f in os.listdir(DATA_DIR)
        if f.endswith('.parquet') and not f.endswith('.tmp')
    ])
    nchars = 0
    for fname in parquet_files:
        filepath = os.path.join(DATA_DIR, fname)
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column('text').to_pylist()
            for doc in texts:
                if len(doc) > DOC_CAP:
                    doc = doc[:DOC_CAP]
                nchars += len(doc)
                yield doc
                if nchars > MAX_CHARS:
                    return

print(f'Training tokenizer with vocab_size={VOCAB_SIZE} on up to {MAX_CHARS:,} chars...')
t0 = time.time()
tokenizer = RustBPETokenizer.train_from_iterator(text_iterator(), VOCAB_SIZE)
train_time = time.time() - t0
print(f'Training completed in {train_time:.1f}s')
print(f'Vocab size: {tokenizer.get_vocab_size()}')

# %%
# Save tokenizer and token_bytes.pt, then upload to HuggingFace Hub

import torch

# 1) Save the tokenizer encoding (pickle) locally
tokenizer.save(LOCAL_TOKENIZER_DIR)

# 2) Quick sanity check: encode/decode roundtrip
test_text = "Hello world! Numbers: 42. The quick brown fox jumps over the lazy dog."
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded)
assert decoded == test_text, f"Roundtrip failed: {decoded!r} != {test_text!r}"
print(f'Sanity check passed: {len(encoded)} tokens')

# 3) Compute and save token_bytes.pt (token_id -> num UTF-8 bytes)
# Used later for computing bits-per-byte (BPB) evaluation metric
vocab_size = tokenizer.get_vocab_size()
special_set = tokenizer.get_special_tokens()
token_bytes_list = []
for token_id in range(vocab_size):
    token_str = tokenizer.decode([token_id])
    if token_str in special_set:
        token_bytes_list.append(0)  # special tokens don't count
    else:
        token_bytes_list.append(len(token_str.encode('utf-8')))

token_bytes = torch.tensor(token_bytes_list, dtype=torch.int32)
token_bytes_path = os.path.join(LOCAL_TOKENIZER_DIR, 'token_bytes.pt')
torch.save(token_bytes, token_bytes_path)
print(f'Saved token_bytes.pt to {token_bytes_path}')

# Stats
nonzero = token_bytes[token_bytes > 0].float()
print(f'Token byte stats: min={nonzero.min().item():.0f}, max={nonzero.max().item():.0f}, '
      f'mean={nonzero.mean().item():.2f}, std={nonzero.std().item():.2f}')

# 4) Upload to HuggingFace Hub
api = HfApi()
api.create_repo(HF_REPO_ID, repo_type='model', exist_ok=True)
api.upload_folder(
    folder_path=LOCAL_TOKENIZER_DIR,
    repo_id=HF_REPO_ID,
    path_in_repo='tokenizer',
    commit_message='Upload trained BPE tokenizer (vocab 32768)',
)
print(f'\nUploaded to https://huggingface.co/{HF_REPO_ID}/tree/main/tokenizer')

# %%
# Evaluate compression ratio vs GPT-2 and GPT-4 tokenizers

sample_texts = {
    'english': (
        "The quick brown fox jumps over the lazy dog. "
        "Machine learning models have revolutionized natural language processing, "
        "enabling tasks such as translation, summarization, and question answering "
        "to be performed with unprecedented accuracy."
    ),
    'code': '''def fibonacci(n):
    """Calculate the nth Fibonacci number."""
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

# Test it
for i in range(10):
    print(f"F({i}) = {fibonacci(i)}")
''',
    'math': (
        "The Fourier transform of f(x) is F(k) = integral from -inf to +inf "
        "of f(x) * exp(-2*pi*i*k*x) dx. For the Gaussian f(x) = exp(-pi*x^2), "
        "we get F(k) = exp(-pi*k^2), showing that the Gaussian is its own Fourier transform."
    ),
}

# Compare our tokenizer with GPT-2 and GPT-4
tokenizers_to_compare = {
    'GPT-2': RustBPETokenizer.from_pretrained('gpt2'),
    'GPT-4': RustBPETokenizer.from_pretrained('cl100k_base'),
    'Ours': tokenizer,
}

print(f"{'':12s} {'Vocab':>8s}", end='')
for name in sample_texts:
    print(f"  {name:>10s}", end='')
print(f"  {'avg ratio':>10s}")
print("-" * 76)

for tok_name, tok in tokenizers_to_compare.items():
    ratios = []
    print(f"{tok_name:12s} {tok.get_vocab_size():8d}", end='')
    for text_name, text in sample_texts.items():
        encoded = tok.encode(text)
        text_bytes = len(text.encode('utf-8'))
        ratio = text_bytes / len(encoded)
        ratios.append(ratio)
        print(f"  {ratio:10.2f}", end='')
    avg_ratio = sum(ratios) / len(ratios)
    print(f"  {avg_ratio:10.2f}")

print("\n(Ratio = bytes/token. Higher is better = more compression)")
