import math, random
import numpy as np
from collections import defaultdict
from tqdm import tqdm

def sigmoid(x):
    # Clip to Prevent Overflow
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

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
        probs = np.array([self.word_counts[w] ** 0.75 for w in words])
        probs /= probs.sum()

        # Build Table
        self.neg_sampling_table = []
        for i, word in enumerate(words):
            count = int(probs[i] * table_size)
            self.neg_sampling_table.extend([word] * max(1, count))

        self.neg_sampling_table = np.array(self.neg_sampling_table)

        if (self.verbosity > 0):
            print(f"Negative sampling table built with {len(self.neg_sampling_table)}")


    def negative_sample(self, positive_contexts, num_samples=5):
        if self.neg_sampling_table is None or len(self.neg_sampling_table) == 0:
            batch_size = len(positive_contexts) if hasattr(positive_contexts, '__len__') else 1
            return np.random.choice(self.vocab_size, (batch_size, num_samples))
        
        if not hasattr(positive_contexts, '__len__'):
            positive_contexts = [positive_contexts]

        batch_size = len(positive_contexts)
        negatives = np.zeros((batch_size, num_samples), dtype=np.int32)

        for i, pos_context in enumerate(positive_contexts):
            neg_samples = set()
            attempts = 0
            max_attempts = num_samples * 10 # Prevent Infinite Loop

            while len(neg_samples) < num_samples and attempts < max_attempts:
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
    
    def train_batch(self, center_batch, context_batch, lr=0.01, negative_samples=5):
        center_batch = np.array(center_batch, dtype=np.int32)
        context_batch = np.array(context_batch, dtype=np.int32)
        batch_size = len(center_batch)

        # Get Embeddings
        v_centers = self.E[center_batch]
        v_contexts = self.C[context_batch]

        # Positive Samples
        pos_scores = np.sum(v_centers * v_contexts, axis=1)
        pos_sigmoids = sigmoid(pos_scores)
        pos_grads = (1 - pos_sigmoids).astype(np.float32)

        # Gradient Accumulators
        center_grads = pos_grads[:, np.newaxis] * v_contexts
        context_grads = pos_grads[:, np.newaxis] * v_centers

        # Negative Samples
        neg_samples = self.negative_sample(context_batch.tolist(), negative_samples)

        for i in range(batch_size):
            center_embed = v_centers[i]
            neg_indices = neg_samples[i] if batch_size > 1 else neg_samples

            v_negs = self.C[neg_indices]
            neg_scores = np.dot(v_negs, center_embed)
            neg_sigmoids = sigmoid(neg_scores)
            neg_grads = - neg_sigmoids.astype(np.float32)

            # Update Negative Embeddings
            self.C[neg_indices] += lr * neg_grads[:, np.newaxis] * center_embed

            # Accumulate Gradients
            center_grads[i] += np.sum(neg_grads[:, np.newaxis] * v_negs, axis=0)

        # Apply Gradients
        self.E[center_batch] += lr * center_grads
        self.C[context_batch] += lr * context_grads

    def create_training_pairs(self, token_ids, window_size=2, subsample_threshold=1e-5):
        # Subsample Tokens First if Enabled
        if subsample_threshold > 0:
            token_ids = self.subsample_frequent_words(token_ids, subsample_threshold)

        pairs = []
        for i in range(len(token_ids)):
            center_token = token_ids[i]

            # Dynamic Window Sizing
            actual_window = np.random.randint(1, window_size + 1)

            start = max(0, i - actual_window)
            end = min(len(token_ids), i + actual_window + 1)

            for j in range(start, end):
                if i != j:
                    context_token = token_ids[j]
                    pairs.append((center_token, context_token))
        
        return pairs

    def subsample_frequent_words(self, token_ids, threshold=1e-5):
        # Subsample Frequent Words
        if not self.word_counts or self.total_words == 0:
            return token_ids
            
        subsampled = []
        for token_id in token_ids:
            freq = self.word_counts[token_id] / self.total_words
            
            # Word2Vec Formula
            if freq <= threshold:
                prob_keep = 1.0
            else:
                prob_keep = (np.sqrt(freq / threshold) + 1) * (threshold / freq)
            
            if np.random.random() < prob_keep:
                subsampled.append(token_id)
        
        return subsampled
    
    def train_on_tokens(self, token_ids, window_size=2, epochs=1, lr=0.01, batch_size=1000, negative_samples = 5, subsample_threshold = 1e-5):
        if self.verbosity > 0:
            print(f"Training on {len(token_ids)} tokens...")

        # Update Word Counts for Negative Samples
        self.update_word_counts(token_ids)
        self.build_negative_sampling_table()

        for epoch in range(epochs):
            if self.verbosity > 0:
                print(f"Starting Epoch {epoch + 1} / {epochs}")

            # Generate Training Pairs
            pairs = self.create_training_pairs(token_ids, window_size, subsample_threshold)
            if self.verbosity > 0:
                print(f"Generated {len(pairs)} training pairs")

            # Randomize for Better Training
            np.random.shuffle(pairs)

            # Batch Training
            total_batches = (len(pairs) + batch_size - 1) // batch_size

            for batch_idx in tqdm(range(total_batches), desc="Training batches"):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(pairs))
                batch_pairs = pairs[start_idx:end_idx]

                if not batch_pairs:
                    continue
                
                # Split Into Centers and Contexts
                centers, contexts = zip(*batch_pairs)

                self.train_batch(list(centers), list(contexts), lr, negative_samples)

            if self.verbosity > 0:
                print(f"Epoch {epoch + 1} completed")


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
        if token_id >= self.vocab_size:
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


