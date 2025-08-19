from collections import Counter, defaultdict
import re, json
from tqdm import tqdm
import numpy as np

class SimpleBPETokenizer:
    def __init__(self, special_tokens = None):
        self.vocab = {}
        self.merges = []
        self.special_tokens = special_tokens or []
        self.id_to_token = {}
        self.merge_lookup = {}

    def get_pairs(self, word_tokens):
        pair_freqs = Counter()

        for word_token_list in word_tokens:
            if len(word_token_list) < 2:
                continue

            # Convert to Numpy for Vectorization
            word_array = np.array(word_token_list, dtype=object)

            # Create Pairs w Array Slicing
            if len(word_array) > 1:
                pairs = list(zip(word_array[:-1], word_array[1:]))
                pair_freqs.update(pairs)

        return pair_freqs


    def apply_merge(self, word_tokens, merge_pair, new_token):
        new_word_tokens = []
        frequency_changes = defaultdict(int)
        a, b = merge_pair

        for word_idx, word_token_list in enumerate(word_tokens):
            if len(word_token_list) < 2:
                new_word_tokens.append(word_token_list)
                continue

            # Convert to Numpy for Vectorization
            word_array = np.array(word_token_list, dtype=object)

            # Find Merge Positions
            merge_positions = []
            for i in range(len(word_array) - 1):
                if word_array[i] == a and word_array[i + 1] == b:
                    merge_positions.append(i)

            if not merge_positions:
                new_word_tokens.append(word_token_list)
                continue

            # Track Frequency Positions
            for pos in merge_positions:
                # Remove Merged Pair
                frequency_changes[(a, b)] -= 1

                # Update Neighbors
                if pos > 0:
                    left_token = word_array[pos - 1]
                    frequency_changes[(left_token, a)] -= 1
                    frequency_changes[(left_token, new_token)] += 1

                if pos + 2 < len(word_array):
                    right_token = word_array[pos + 2]
                    frequency_changes[(b, right_token)] -= 1
                    frequency_changes[(new_token, right_token)] += 1

            # Apply Merges, Reverse Order to Maintain Indices
            merged_list = list(word_array)
            for pos in reversed(merge_positions):
                merged_list[pos:pos+2] = [new_token]

            new_word_tokens.append(merged_list)

        return new_word_tokens, frequency_changes
    
    def update_pair_frequencies(self, pair_freqs, frequency_changes):
        for pair, change in frequency_changes.items():
            if change == 0:
                continue

            if pair in pair_freqs:
                new_freq = pair_freqs[pair] + change
                if new_freq <= 0:
                    del pair_freqs[pair]
                else:
                    pair_freqs[pair] = new_freq
            elif change > 0:
                pair_freqs[pair] = change


    def train(self, text, vocab_size=100, min_frequency=3, verbosity=0):
        # Breakup Input Text to Tokens
        original_words = re.findall(r"\w+|\s+|[^\w\s]", text)
        word_tokens = [list(word) for word in original_words]

        # Get Character Frequencies
        char_counts = Counter()
        for word in original_words:
            char_counts.update(word)

        # Initialize Vocab w/ Chars
        self.vocab.update(
            {token: idx + len(self.special_tokens)
             for idx, token in enumerate(char_counts.keys())}
        )

        # Initialize Pair Prequencies
        pair_freqs = self.get_pairs(word_tokens)

        # Pre-Allocated
        max_pair = None
        max_freq = 0

        # Begin BPE Training Loop
        iterations = 0
        total_iterations = vocab_size - len(self.vocab)
        with tqdm(total=total_iterations, desc="Training BPE") as pbar:
            while len(self.vocab) < vocab_size and pair_freqs:
                if verbosity > 0 and (verbosity > 1 or iterations % 10 == 0):
                    print(f"Iteration {iterations}: {len(self.vocab)} / {vocab_size}")

                # Find Best Pair - Cached
                if not max_pair or max_pair not in pair_freqs:
                    max_pair = max(pair_freqs, key=pair_freqs.get)
                    max_freq = pair_freqs[max_pair]
                else:
                    current_freq = pair_freqs.get(max_pair, 0)
                    if current_freq != max_freq or current_freq == 0:
                        max_pair = max(pair_freqs, key=pair_freqs.get)
                        max_freq = pair_freqs[max_pair]

                best_pair = max_pair
                best_freq = max_freq

                if verbosity > 0 and (verbosity > 1 or iterations % 10 == 0):
                    print(f"\tBest Pair: {best_pair} with frequency {best_freq}")

                # Check Threshold
                if best_freq < min_frequency:
                    if verbosity > 0:
                        print(f"\tFrequency {best_freq} < min_frequency {min_frequency}, exiting...")
                    break

                # Create New Token
                new_token = "".join(best_pair)

                # Skip Token if Already Present
                if new_token in self.vocab:
                    if verbosity > 0:
                        print(f"\tToken '{new_token}' already exists in vocab, skipping...")
                    del pair_freqs[best_pair]
                    continue

                # Add to Vocab and Merges
                self.vocab[new_token] = len(self.vocab)
                self.merges.append(best_pair)
                self.merge_lookup[best_pair] = new_token

                if verbosity > 0 and (verbosity > 1 or iterations % 10 == 0):
                    print(f"\tAdded New Token: '{new_token}'")

                # Apply Merge and Get Frequency Changes
                word_tokens, frequency_changes = self.apply_merge(word_tokens, best_pair, new_token)

                # Update Frequencies
                self.update_pair_frequencies(pair_freqs, frequency_changes)

                iterations += 1
                pbar.update(1)

                # Safety Break
                if iterations > vocab_size * 2:
                    print("Too many iterations, exiting...")
                    break

        # Finalize
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        print(f"Training complete! Final vocab size: {len(self.vocab)}")

    def encode(self, text):
        print("Encoding...")
        # Breakup Input Text
        words = re.findall(r"\w+|\s+|[^\w\s]", text)

        # Convert Each Word to a List of Chars
        word_tokens = [list(word) for word in words]

        # Apply Merges In Learned Order
        for merge_pair in tqdm(self.merges, desc="Applying Merges"):
            new_token = self.merge_lookup[merge_pair]
            word_tokens, _ = self.apply_merge(word_tokens, merge_pair, new_token)

        # Convert Tokens to IDs
        token_ids = []
        for word_token_list in tqdm(word_tokens, desc="Converting Tokens"):
            for token in word_token_list:
                if token in self.vocab:
                    token_ids.append(self.vocab[token])
                else:
                    # TODO:: Handle Unknown Tokens Here
                    # For now, attempt to go char by char
                    for char in token:
                        if char in self.vocab:
                            token_ids.append(self.vocab[char])
        print("Encoded!")
        return token_ids
                    

    def decode(self, token_ids):
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                tokens.append(self.id_to_token[token_id])
            else:
                # TODO:: Handle Unknowns Here
                # For now, append UNK
                tokens.append("<UNK>")
        return "".join(tokens)

    def save(self, filepath):
        # Pack Into Dictionary
        tokenizer_data = {
            'vocab': self.vocab,
            'merges': self.merges,
            'special_tokens': self.special_tokens
        }

        # Store as JSON
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath):
        # Load File
        with open(filepath, 'r', encoding='utf-8') as f:
            tokenizer_data = json.load(f)

        # Construct Class
        tokenizer = cls(special_tokens=tokenizer_data['special_tokens'])
        tokenizer.vocab = tokenizer_data['vocab']
        tokenizer.merges = [(pair[0], pair[1]) if isinstance(pair, list) else pair 
                           for pair in tokenizer_data['merges']]
        tokenizer.id_to_token = {v: k for k, v in tokenizer.vocab.items()}
        tokenizer.merge_lookup = {pair: "".join(pair) for pair in tokenizer.merges}

        return tokenizer
