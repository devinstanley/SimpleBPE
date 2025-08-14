import math, random
import numpy as np
from collections import defaultdict
from tqdm import tqdm

def sigmoid(x):
    # Clip to Prevent Overflow
    x = np.clip(x, -500, 500)
    return 1 / (1 + math.exp(-x))

class SimpleSkipGramEmbeddings:
    def __init__(self, vocab_size, dim, verbosity = 0):
        self.dim = dim
        self.vocab_size = vocab_size
        self.verbosity = verbosity

        # Input Embeddings
        self.E = np.random.uniform(-0.5/dim, 0.5/dim, (vocab_size, dim))

        # Output Embeddings
        self.C = np.random.uniform(-0.5/dim, 0.5/dim, (vocab_size, dim))

        # For Negative Sampling
        self.word_counts = defaultdict(int)
        self.total_words = 0
        self.neg_sampling_table = None

    def update_word_counts(self, token_ids):
        # Update Word Frequency for Negative Sampling
        for token_id in token_ids:
            self.word_counts[token_id] += 1
            self.total_words += 1

    def build_negative_sampling_table(self, table_size = 1e8):
        if not self.word_counts:
            return
        
        if self.verbosity > 0:
            print("Building negative sampling table...")

        table_size = int(table_size)

        # Calculate Probabilities
        words = list(self.word_counts.keys())
        probs = np.array(self.word_counts[w] ** 0.75 for w in words)
        probs /= probs.sum()

        # Build Table
        self.neg_sampling_table = []
        for i, word in enumerate(words):
            count = int(probs[i] * table_size)
            self.neg_sampling_table.extend([word] * max(1, count))

        self.neg_sampling_table = np.array(self.neg_sampling_table)

        if (self.verbosity > 0):
            print(f"Negative sampling table built with {len(self.neg_sampling_table)}")


    def negative_sample(self, positive_context, num_samples=5):
        if self.neg_sampling_table is None or len(self.neg_sampling_table) == 0:
            batch_size = len(positive_context) if hasattr(positive_context, '__len__') else 1
            return np.random.choice(self.vocab_sie, (batch_size, num_samples))
        
        if not hasattr(positive_context, '__len__'):
            positive_context = [positive_context]

        batch_size = len(positive_context)
        negatives = np.zeros((batch_size, num_samples), dtype=np.int32)

        for i, pos_context in enumerate(positive_context):
            neg_samples = set()
            attemps = 0
            max_attempts = num_samples * 10 # Prevent Infinite Loop

            while len(neg_samples) < num_samples and attemps < max_attempts:
                candidate = np.random.choice(self.neg_sampling_table)
                if candidate != pos_context:
                    neg_samples.add(candidate)
                attempts += 1
            
            # Fill Remaining with Randoms in Case
            neg_list = list(neg_samples)
            while len(neg_list) < num_samples:
                candidate = np.random.randint(0, self.vocab_size)
                if candidate != pos_context and candidate not in neg_list:
                    neg_list.append(candidate)
            negatives[i] = neg_list[:num_samples]

        return negatives if batch_size > 1 else negatives[0]

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
        print(f"Training on {len(token_ids)} tokens...")

        # Optional: Subsample frequent words (like "the", "a", etc.)
        subsampled_tokens = self._subsample_frequent_words(token_ids)
        print(f"After subsampling: {len(subsampled_tokens)} tokens")

        # Training Loop
        for epoch in range(epochs):
            print(f"Starting Epoch: {epoch}")
            for i, center_token in tqdm(enumerate(subsampled_tokens)):
                # Create Context Window
                start = max(0, i - window_size)
                end = min(len(subsampled_tokens), i + window_size + 1)

                for j in range(start, end):
                    # Skip Center Word
                    if i != j:
                        context_token = subsampled_tokens[j]
                        self.train_pair(center_token, context_token, lr)

    def _subsample_frequent_words(self, token_ids, threshold=1e-5):
        """Subsample frequent words to speed up training"""
        if not self.word_counts:
            return token_ids
            
        subsampled = []
        for token_id in token_ids:
            freq = self.word_counts[token_id] / self.total_words
            
            # Probability of keeping the word
            if freq <= threshold:
                prob_keep = 1.0
            else:
                prob_keep = (np.sqrt(freq / threshold) + 1) * (threshold / freq)
            
            if random.random() < prob_keep:
                subsampled.append(token_id)
                
        return subsampled

    def get_embedding(self, token_id):
        # Return Embedding if Exists, Otherwise 0 Vector
        if token_id < self.vocab_size:
            return self.E[token_id]
        return np.zeros(self.dim)
    
    def similarity(self, token1, token2):
        # Compute Cosine Similarity Between Two Tokens
        v1 = self.get_embedding(token1)
        v2 = self.get_embedding(token2)

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(v1, v2) / (norm1 * norm2)
    
    def most_similar(self, token_id, top_k = 5):
        # Return Empty Set if Token Not In Embedding
        if token_id > self.vocab_size:
            return []
        
        target_embedding = self.E[token_id]
        similarities = []

        for i in range(self.vocab_size):
            # Skip Identity
            if i != token_id:
                sim = self.similarity(token_id, i)
                similarities.append((i, sim))
        
        # Find Highest Similarities
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


