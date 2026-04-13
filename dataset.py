import numpy as np
import os
from collections import Counter
import random

def load_text8(path):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    words = text.split()
    return words

def build_vocab(words, min_count=5):
    raw_counts = Counter(words)

    vocab = [w for w in raw_counts if raw_counts[w] >= min_count]

    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}

    word_counts = {w: raw_counts[w] for w in vocab}
    
    return vocab, word2idx, idx2word, word_counts

def words_to_indices(words, word2idx):
    return [word2idx[w] for w in words if w in word2idx]

def subsample(indices, word_counts, word2idx, t=1e-5):
    total_count = sum(word_counts.values())
    
    freqs = np.array([word_counts[w] / total_count for w in word2idx])
    
    discard_prob = 1 - np.sqrt(t / freqs)
    discard_prob = np.clip(discard_prob, 0, 1)
    
    new_indices = []
    
    for idx in indices:
        if random.random() > discard_prob[idx]:
            new_indices.append(idx)
            
    return new_indices

def negative_sampling_distribution(word_counts, word2idx):
    vocab_size = len(word2idx)
    freqs = np.zeros(vocab_size)
    
    for word, idx in word2idx.items():
        freqs[idx] = word_counts[word]
    
    freqs = freqs ** 0.75
    freqs /= np.sum(freqs)
    
    return freqs


class Word2VecDataset:
    def __init__(self, indices, word_counts, word2idx, window_size=5, neg_samples=5):
        self.indices = indices
        self.word_counts = word_counts
        self.word2idx = word2idx
        self.window_size = window_size
        self.neg_samples = neg_samples
        
        self.neg_distribution = negative_sampling_distribution(
            word_counts, word2idx
        )
        self.vocab_size = len(word2idx)
    
    def sample_negative(self):
        return np.random.choice(
            self.vocab_size,
            size=self.neg_samples,
            p=self.neg_distribution
        )
    
    def get_batch(self, batch_size):
        centers = []
        positives = []
        negatives = []
        
        for _ in range(batch_size):
            i = random.randint(0, len(self.indices) - 1)
            center = self.indices[i]
            
            window = random.randint(1, self.window_size)
            start = max(0, i - window)
            end = min(len(self.indices), i + window + 1)
            
            context_indices = [
                self.indices[j]
                for j in range(start, end)
                if j != i
            ]
            
            if not context_indices:
                continue
            
            positive = random.choice(context_indices)
            negative = self.sample_negative()
            
            centers.append(center)
            positives.append(positive)
            negatives.append(negative)
        
        return (
            np.array(centers),
            np.array(positives),
            np.array(negatives)
        )