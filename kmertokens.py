import os
import time
import cProfile
import pstats
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, decoders
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ["TOKENIZERS_PARALLELISM"] = "false"

genome_sequences_dir = "genome_sequences"
tokenizers_dir = "tokenizers"
subsampled_genomes_dir = "subsampled_genomes"
os.makedirs(subsampled_genomes_dir, exist_ok=True)
os.makedirs(tokenizers_dir, exist_ok=True)


def load_genome_sequence(filename):
    start_time = time.time()
    with open(os.path.join(genome_sequences_dir, filename), 'r') as f:
        genome = f.read().upper().replace('\n', '')
    end_time = time.time()
    print(f"loaded the sequence in {end_time - start_time:.2f} secs")
    return genome

def subsample_genome(genome, size=10**6):
    return genome[:min(size, len(genome))]

def save_subsampled_genome(org, subsample):
    subsample_path = os.path.join(subsampled_genomes_dir, f"{org.replace(' ', '_')}_subsampled.txt")
    with open(subsample_path, 'w') as f:
        f.write(subsample)
    return subsample_path


def generate_kmers(sequence, k):
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

def calculate_kmer_frequencies(sequence, k):
    start_time = time.time()
    kmer_freqs = defaultdict(int)
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k]
        kmer_freqs[kmer] += 1
    end_time = time.time()
    print(f"Calculated k-mer frequencies in {end_time - start_time:.2f} seconds")
    return kmer_freqs

def create_vocabulary(kmer_freqs, vocab_size):
    sorted_kmers = sorted(kmer_freqs.items(), key=lambda x: x[1], reverse=True)
    return [kmer for kmer, _ in sorted_kmers[:vocab_size]]

def calculate_compression(genome_sequence, tokens, vocab):
    total_length = len(genome_sequence)
    encoded_length = len(tokens)
    vocab_length = sum(len(token.replace('##', '')) for token in vocab) 
    return (encoded_length + vocab_length) / total_length

def train_and_save_wordpiece_tokenizer(kmers, vocab_size, save_dir):
    start_time = time.time()
    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]", max_input_chars_per_word=10**15))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.decoder = decoders.WordPiece()
    trainer = trainers.WordPieceTrainer(vocab_size=vocab_size, special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]", "[N]"])
    tokenizer.train_from_iterator(kmers, trainer=trainer)
    tokenizer.post_processor = processors.BertProcessing(
        ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ("[CLS]", tokenizer.token_to_id("[CLS]")),
    )
    save_path = os.path.join(save_dir, f"wordpiece_tokenizer_{vocab_size}.json")
    tokenizer.save(save_path)
    end_time = time.time()
    print(f"trained and saved tokenizer for vocab size {vocab_size} in {end_time - start_time:.2f} secs")
    return save_path

def tokenize_file(genome_sequence, tokenizer):
    start_time = time.time()
    encoded = tokenizer.encode(genome_sequence)
    end_time = time.time()
    print(f"Tokenized genome sequence in {end_time - start_time:.2f} seconds")
    return encoded.tokens

def evaluate_vocab_size(kmer_freqs, genome_sequence, vocab_size, tokenizers_dir):
    start_time = time.time()
    vocab = create_vocabulary(kmer_freqs, vocab_size)
    tokenizer_save_path = train_and_save_wordpiece_tokenizer(vocab, vocab_size, tokenizers_dir)
    tokenizer = Tokenizer.from_file(tokenizer_save_path)
    tokens = tokenize_file(genome_sequence, tokenizer)
    compression = calculate_compression(genome_sequence, tokens, vocab)
    end_time = time.time()
    print(f"evaluated vocab size {vocab_size} in {end_time - start_time:.2f} secs")
    return vocab_size, compression

def heuristic_search_optimal_vocab_size(kmer_freqs, genome_sequence, initial_vocab_size=50, step_size=1000, max_vocab_size=80000, parallel=True):
    vocab_sizes = list(range(initial_vocab_size, max_vocab_size + step_size, step_size))
    best_vocab_size = initial_vocab_size
    best_compression_factor = float('inf')
    compression_factors = []

    if parallel:
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(evaluate_vocab_size, kmer_freqs, genome_sequence, vocab_size, tokenizers_dir): vocab_size for vocab_size in vocab_sizes}
            for future in as_completed(futures):
                try:
                    vocab_size, compression = future.result()
                    compression_factors.append((vocab_size, compression))
                    print(f"Vocab size: {vocab_size}, compression factor: {compression}")
                    if compression < best_compression_factor:
                        best_compression_factor = compression
                        best_vocab_size = vocab_size
                except Exception as e:
                    print(f"Error with vocab size {futures[future]}: {e}")
    else:
        for vocab_size in vocab_sizes:
            try:
                start_time = time.time()
                vocab_size, compression = evaluate_vocab_size(kmer_freqs, genome_sequence, vocab_size, tokenizers_dir)
                end_time = time.time()
                print(f"evaluated vocab size {vocab_size} in {end_time - start_time:.2f} secs")
                compression_factors.append((vocab_size, compression))
                print(f"Vocab size: {vocab_size}, Compression factor: {compression}")
                if compression < best_compression_factor:
                    best_compression_factor = compression
                    best_vocab_size = vocab_size
            except Exception as e:
                print(f"error w vocab size {vocab_size}: {e}")

    return best_vocab_size, best_compression_factor, compression_factors

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()

    org = "Nanobdella aerobiophila"
    genome_sequence = load_genome_sequence(f"{org.replace(' ', '_')}_cleaned.txt")
    subsampled_sequence = subsample_genome(genome_sequence)
    save_subsampled_genome(org, subsampled_sequence)

    k = 6
    kmer_freqs = calculate_kmer_frequencies(subsampled_sequence, k)

    start_time = time.time()
    optimal_vocab_size, optimal_compression_factor, compression_factors = heuristic_search_optimal_vocab_size(
        kmer_freqs, subsampled_sequence, initial_vocab_size=50, step_size=1000, max_vocab_size=80000, parallel=True)
    end_time = time.time()
    print(f"search completed in {end_time - start_time:.2f} seconds")

    print(f"Optimal vocab size for {org}: {optimal_vocab_size}, compression factor: {optimal_compression_factor}")

    vocab_sizes = [cf[0] for cf in compression_factors]
    compression_values = [cf[1] for cf in compression_factors]

    plt.plot(vocab_sizes, compression_values)
    plt.xlabel('vocab size')
    plt.ylabel('compression factor')
    plt.title(f'compression factor vs vocab size for {org}')
    plt.show()

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(10)  