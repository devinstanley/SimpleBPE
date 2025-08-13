import math, random
import numpy as np
from collections import defaultdict

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

class SimpleSkipGramEmbeddings:
    def __init__(self, vocab_size, dim):
        self.dim = dim
        self.vocab_size = vocab_size

        # Input Embeddings
        self.E = np.random.uniform(-0.5/dim, 0.5/dim, (vocab_size, dim))

        # Output Embeddings
        self.C = np.random.uniform(-0.5/dim, 0.5,dim, (vocab_size, dim))

        # Track Word Frequencies for Sampling
        self.word_counts = defaultdict(int)
        self.total_words = 0

    def update_word_counts(self, token_ids):
        # Update Word Frequency for Negative Sampling
        for token_id in token_ids:
            self.word_counts[token_id] += 1
            self.total_words += 1

    def negative_sample(self, positive_context, num_samples=5):
        # Fallback to Random Uniform Sampling
        if not self.word_counts:
            return np.random.choice(self.vocab_size, num_samples, replace=False)
        
        words = list(self.word_counts.keys())
        probs = np.array([self.word_counts[w] ** 0.75 for w in words])
        probs /= probs.sum()

        negatives = []
        while len(negatives) < num_samples:
            sample = np.random.choice(words, p=probs)
            if sample != positive_context and sample not in negatives:
                negatives.append(sample)

        return negatives[:num_samples]

    def train_pair(self, center, context, lr=0.01, negative_samples=5):
        # Center Word Embedding
        v_c = self.E[center]

        # Context Word Embedding
        v_o = self.C[context]

        # Positive Samples
        pos_score = np.dot(v_c, v_o)
        pos_sigmoid = sigmoid(pos_score)
        pos_grad = 1 - pos_sigmoid

        # Update Embeddings From Positive Sample
        v_c_grad = pos_grad * v_o
        v_o_grad = pos_grad * v_c

        # Negative Sampling
        negatives = self.negative_sample(context, negative_samples)
        for negative_context in negatives:
            v_neg = self.C[negative_context]
            neg_score = np.dot(v_c, v_neg)
            neg_sigmoid = sigmoid(neg_score)

            neg_grad = -neg_sigmoid

            # Accumulate Negative Sample Gradients
            v_c_grad += neg_grad * v_neg
            self.C[negative_context] += lr * neg_grad * v_c
        
        # Apply Gradients
        self.E[center] += lr * v_c_grad
        self.C[context] += lr * v_o_grad

    def train_on_tokens(self, token_ids, window_size=2, epochs=1, lr=0.01):
        # Update Word Counts for Negative Samples
        self.update_word_counts(token_ids)

        # Training Loop
        for epoch in range(epochs):
            for i, center_token in enumerate(token_ids):
                # Create Context Window
                start = max(0, i - window_size)
                end = min(len(token_ids), i + window_size + 1)

                for j in range(start, end):
                    # Skip Center Word
                    if i != j:
                        context_token = token_ids[j]
                        self.train_pair(center_token, context_token, lr)

    def get_embedding(self, token_id):
        # Return Embedding if Exists, Otherwise 0 Vector
        if token_id < self.vocab_size:
            return self.E[token_id]
        return np.zeros(self.dim)


