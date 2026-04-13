import numpy as np
import time

from dataset import (
    load_text8,
    build_vocab,
    words_to_indices,
    subsample,
    Word2VecDataset
)

from model import SkipGramNS

def normalize_embeddings(W):
    norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-10
    return W / norms

def cosine_similarity(vec, W):
    return np.dot(W, vec)


def most_similar(word, W, word2idx, idx2word, top_k=10):
    if word not in word2idx:
        print(f"{word}       vocabulary.")
        return []

    vec = W[word2idx[word]]
    sims = cosine_similarity(vec, W)

    best_indices = np.argsort(-sims)[:top_k + 1]

    results = []
    for idx in best_indices:
        results.append((idx2word[idx], sims[idx]))

    return results

def train():
    #Hyperparameters
    corpus_path = "data/text8"     
    min_count = 5
    embed_dim = 100
    window_size = 5
    neg_samples = 5
    batch_size = 64
    num_steps = 100000
    lr = 0.025

    #Data preparation
    words = load_text8(corpus_path)
    vocab, word2idx, idx2word, word_counts = build_vocab(words,min_count=min_count)
    indices = words_to_indices(words, word2idx)
    indices = subsample(indices, word_counts, word2idx)

    dataset = Word2VecDataset(
        indices=indices,
        word_counts=word_counts,
        word2idx=word2idx,
        window_size=window_size,
        neg_samples=neg_samples
    )

    model = SkipGramNS(
        vocab_size=len(word2idx),
        embed_dim=embed_dim
    )

    #Training 
    start_time = time.time()
    for step in range(1, num_steps + 1):

        centers, positives, negatives = dataset.get_batch(batch_size)

        loss, grad_w_i, grad_w_o, grad_w_j = model.forward_backward(
            centers,
            positives,
            negatives
        )

        model.update(
            centers,
            positives,
            negatives,
            grad_w_i,
            grad_w_o,
            grad_w_j,
            lr
        )

        if step % 1000 == 0:
            elapsed = time.time() - start_time
            print(f"Step {step}/{num_steps} | Loss: {loss:.4f} | Time: {elapsed:.2f}s")
            
    np.save("word2vec_input_embeddings.npy", model.W)
    np.save("word2vec_output_embeddings.npy", model.W_prim)

    return model, word2idx, idx2word

if __name__ == "__main__":
    model, word2idx, idx2word = train()
    
    W = normalize_embeddings(model.W)
    print(most_similar("king", W, word2idx, idx2word))