# Word2Vec (Skip-Gram with Negative Sampling) — NumPy Implementation

This repository contains a clean, educational implementation of the Word2Vec Skip-Gram model with Negative Sampling (SGNS) written using only NumPy. The code focuses on clarity and follows the mathematical derivations from the literature so it is suitable for learning and experimentation.

## Contents

- `train.py` — Main entry point. Handles hyperparameters, data loading, training loop, learning-rate decay and saving final embeddings.
- `model.py` — `SkipGramNS` class: weight initialization, `forward_backward` pass, gradient computations and parameter update logic.
- `dataset.py` — `Word2VecDataset` class: vocabulary construction, word↔index mappings, subsampling, and batch generator.
- `data/text8` — Example raw text dataset (not included by this repo in some distributions); used for training experiments.
- `word2vec_input_embeddings.npy`, `word2vec_output_embeddings.npy` — Example or output embedding matrices saved by training.

## Requirements

- Python 3.8+ (developed and tested with CPython 3.11)
- NumPy

Install requirements:

```powershell
pip install numpy
```

## Quick start


To run this code, please follow these steps:
1. Download the dataset from [text8 dataset (Matt Mahoney)](http://mattmahoney.net/dc/textdata.html)
2. Create a folder named `data/` in the root directory of this project.
3. Edit hyperparameters in `train.py` if desired (embedding size, window, negative samples, batch size, learning rate, epochs).
4. Run training:

```powershell
python train.py
```

Trained embeddings will be saved as NumPy `.npy` files (see `word2vec_input_embeddings.npy` and `word2vec_output_embeddings.npy`).

## Implementation notes & references

- Core model and negative sampling formulation based on: Mikolov et al., "Distributed Representations of Words and Phrases and their Compositionality" (2013).
- Gradient derivations and backpropagation follow: Xin Rong, "word2vec Parameter Learning Explained" (2014).
- This repository emphasizes clarity and faithful translation of the math into NumPy operations rather than production-level performance.

## Paper attribution

This repository includes an annotated copy of the paper "word2vec Parameter Learning Explained" for personal study. I went through the math in detail, checked dimensions, and added notes to verify my understanding. The paper itself is not my work; all rights belong to the original author and publisher. The original paper is available at: https://arxiv.org/abs/1411.2738

## License

Use and modify freely for educational and research purposes. No warranty provided.
