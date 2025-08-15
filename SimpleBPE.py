from collections import Counter, defaultdict
import re, json
from tqdm import tqdm

class SimpleBPETokenizer:
    def __init__(self, special_tokens = None):
        self.vocab = {}
        self.merges = []
        self.special_tokens = special_tokens or []
        self.id_to_token = {}
        self.merge_lookup = {}

    def apply_merge(self, word_tokens, merge_pair, new_token):
        new_word_tokens = []
        affected_pairs = defaultdict(int)
        a, b = merge_pair

        for word_token_list in word_tokens:
            i = 0
            merged_list = []

            while i < len(word_token_list):
                if (
                    i < len(word_token_list) - 1 and
                    word_token_list[i] == a and
                    word_token_list[i + 1] == b
                ):
                    # Track Removed Pairs
                    affected_pairs[(a, b)] -= 1

                    # Track Left/Right Neighbor Changes
                    if i > 0:
                        left_token = word_token_list[i - 1]
                        affected_pairs[(left_token, a)] -= 1
                        affected_pairs[(left_token, new_token)] += 1

                    if i < len(word_token_list) - 2:
                        right_token = word_token_list[i + 2]
                        affected_pairs[(b, right_token)] -= 1
                        affected_pairs[(new_token, right_token)] += 1

                    merged_list.append(new_token)
                    i += 2
                else:
                    merged_list.append(word_token_list[i])
                    i += 1
            new_word_tokens.append(merged_list)
        return new_word_tokens, affected_pairs

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
        pair_freqs = Counter()
        for word in word_tokens:
            for i in range(len(word) - 1):
                pair_freqs[(word[i], word[i + 1])] += 1

        # Begin BPE Training Loop
        iterations = 0
        while len(self.vocab) < vocab_size and pair_freqs:
            # Debug Statement Frequency Depending on Verbosity
            if verbosity > 0:
                if verbosity > 1 or iterations % 10 == 0:
                    print(f"Iteration {iterations}: {len(self.vocab)} / {vocab_size}")

            # Pick Most Frequent Pair
            best_pair, best_freq = max(pair_freqs.items(), key=lambda x: x[1])

            # Debug Statement Frequency Depending on Verbosity
            if verbosity > 0:
                if verbosity > 1 or iterations % 10 == 0:
                    print(f"\tBest Pair: {best_pair} with frequency {best_freq}")

            if best_freq < min_frequency:
                print(f"\tFrequency {best_freq} < min_frequency {min_frequency}, exiting...")
                break

            # Merge Token
            new_token = "".join(best_pair)

            # Ensure Token Not Already Found
            if new_token in self.vocab:
                if verbosity > 0:
                    print(f"\tToken '{new_token}' already exists in vocab, skipping...")
                # Remove this pair and continue
                del pair_freqs[best_pair]
                continue

            # Add New Token to Vocab and Merges
            self.vocab[new_token] = len(self.vocab)
            self.merges.append(best_pair)

            # For Faster Encoding
            self.merge_lookup[best_pair] = new_token

            # Debug Statement Frequency Depending on Verbosity
            if verbosity > 0:
                if verbosity > 1 or iterations % 10 == 0:
                    print(f"\tAdded New Token: '{new_token}'")

            # Apply merge and get frequency updates
            word_tokens, affected_pairs = self.apply_merge(
                word_tokens, best_pair, new_token
            )

            # Update pair frequencies efficiently
            for pair, change in affected_pairs.items():
                if pair in pair_freqs:
                    pair_freqs[pair] += change
                    if pair_freqs[pair] <= 0:
                        del pair_freqs[pair]
                elif change > 0:
                    pair_freqs[pair] = change

            iterations += 1
            
            # Safety Break
            if iterations > vocab_size * 2:
                print("Too many iterations, exiting...")

        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.merge_lookup = {pair: "".join(pair) for pair in self.merges} # Faster Encoding

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
