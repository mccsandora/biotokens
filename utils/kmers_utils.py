import os
import re
import pandas as pd
import sentencepiece as spm
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

def generate_kmers(sequence, k):
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

def save_kmers_to_file(kmers, file_path):
    with open(file_path, 'w') as f:
        for kmer in kmers:
            f.write(kmer + '\n')
    print(f"K-mers saved to {file_path}")

def train_sentencepiece_tokenizer_kmer(input_file, model_prefix, vocab_size):
    """Train SentencePiece tokenizer on kmers."""
    try:
        spm.SentencePieceTrainer.train(
            input=input_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type='bpe',
            character_coverage=0.9995,
            max_sentence_length=5000,
            hard_vocab_limit=False
        )
        if os.path.exists(model_prefix + ".model") and os.path.exists(model_prefix + ".vocab"):
            print(f"model and vocab files created: {model_prefix}.model, {model_prefix}.vocab")
        else:
            print(f"model or vocab file not created for {model_prefix}")
    except Exception as e:
        print(f"error occurred during training: {e}")
        raise e

def load_tokenizer_vocab(model_prefix):
    vocab_file = model_prefix + ".vocab"
    vocab = pd.read_csv(vocab_file, sep='\t', header=None)
    vocab = vocab[0].tolist()
    vocab = [v for v in vocab if v and re.match(r'^[ACGT]+$', v)] 
    return vocab