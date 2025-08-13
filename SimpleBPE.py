from collections import Counter
import re, json

class SimpleBPETokenizer:
    def __init__(self, special_tokens = None):
        self.vocab = {}
        self.merges = []
        self.special_tokens = special_tokens or []
        self.id_to_token = {}

    def train(self, text, vocab_size=100, min_frequency=3, verbosity=0):
        # Breakup Input Text
        original_words = re.findall(r"\w+|\s+|[^\w\s]", text)

        # Convert Each Word to a List of Chars
        word_tokens = [list(word) for word in original_words]

        # Get Character Frequencies
        char_counts = Counter()
        for word in original_words:
            char_counts.update(word)

        self.vocab.update(
            {token: idx + len(self.special_tokens)
             for idx, token in enumerate(char_counts.keys())}
        )

        # Begin BPE Training Loop
        ii = 0
        while len(self.vocab) < vocab_size:
            # Debug Statement Frequency Depending on Verbosity
            if verbosity > 0:
                if verbosity > 1:
                    print(f"Iteration {ii}: {len(self.vocab)} / {vocab_size}")
                elif ii % 10 == 0:
                    print(f"Iteration {ii}: {len(self.vocab)} / {vocab_size}")

            pairs = Counter()
            for words_token_list in word_tokens:                            # Iterate Over All Word's Char List
                for i in range(len(words_token_list) - 1):                  # Iterate Over Each Char in Word's Char List
                    pair = (words_token_list[i], words_token_list[i + 1])   # Create Pairs From Each Word
                    pairs[pair] += 1                                            # Increment Frequency

            # Ensure Progress Continues
            if not pairs:
                print("No more pairs found - exiting...")
                break

            best_pair = max(pairs.items(), key=lambda x: x[1])[0]
            best_freq = pairs[best_pair]

            # Debug Statement Frequency Depending on Verbosity
            if verbosity > 0:
                if verbosity > 1:
                    print(f"\tBest Pair: {best_pair} with frequency {best_freq}")
                elif ii % 10 == 0:
                    print(f"\tBest Pair: {best_pair} with frequency {best_freq}")

            if best_freq < min_frequency:
                print(f"\tFrequency {best_freq} < min_frequency {min_frequency}, exiting...")
                break

            new_token = "".join(best_pair)

            # Ensure Token Not Already Found
            if new_token in self.vocab:
                print(f"\tToken '{new_token}' already exists in vocab, exiting...")
                break

            # Add New Token to Vocab and Merges
            self.vocab[new_token] = len(self.vocab)
            self.merges.append(best_pair)

            # Debug Statement Frequency Depending on Verbosity
            if verbosity > 0:
                if verbosity > 1:
                    print(f"\tAdded New Token: '{new_token}'")
                elif ii % 10 == 0:
                    print(f"\tAdded New Token: '{new_token}'")

            # Apply Merge to All Word Token Lists
            merge_count = 0
            new_word_tokens = []

            for word_token_list in word_tokens:
                new_tokens = []
                i = 0

                while i < len(word_token_list):
                    # Check if Tokens Can be Merged
                    if (i < len(word_token_list) - 1 and
                        word_token_list[i] == best_pair[0] and
                        word_token_list[i + 1] == best_pair[1]):
                        
                        # Merge Pair
                        new_tokens.append(new_token)
                        i += 2 # Skip Forward
                        merge_count += 1
                    else:
                        # Keep Current Token
                        new_tokens.append(word_token_list[i])
                        i += 1

                new_word_tokens.append(new_tokens)
            
            # Update word_tokens for Next Iteration
            word_tokens = new_word_tokens

            ii += 1
            
            # Safety Break
            if ii > vocab_size * 2:
                print("Too many iterations, exiting...")

        self.id_to_token = {v: k for k, v in self.vocab.items()}

    def encode(self, text):
        # Breakup Input Text
        words = re.findall(r"\w+|\s+|[^\w\s]", text)

        # Convert Each Word to a List of Chars
        word_tokens = [list(word) for word in words]

        # Apply Merges In Learned Order
        for merge_pair in self.merges:
            new_word_tokens = []
            for word_token_list in word_tokens:
                new_tokens = []
                i = 0
                while i < len(word_token_list):
                    if (
                        i < len(word_token_list) - 1 and
                        word_token_list[i] == merge_pair[0] and
                        word_token_list[i + 1] == merge_pair[1]
                    ):
                        merged_token = "".join(merge_pair)
                        new_tokens.append(merged_token)
                        i += 2
                    else:
                        new_tokens.append(word_token_list[i])
                        i += 1
                new_word_tokens.append(new_tokens)
            word_tokens = new_word_tokens

        # Convert Tokens to IDs
        token_ids = []
        for word_token_list in word_tokens:
            for token in word_token_list:
                if token in self.vocab:
                    token_ids.append(self.vocab[token])
                else:
                    # TODO:: Handle Unknown Tokens Here
                    # For now, attempt to go char by char
                    for char in token:
                        if char in self.vocab:
                            token_ids.append(self.vocab[char])

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

        return tokenizer
