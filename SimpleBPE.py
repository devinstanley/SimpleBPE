class SimpleBPETokenizer:
    def __init__(self, special_tokens = None):
        self.vocab = {}
        self.merges = {}
        self.special_tokens = special_tokens or []

    def train(self, text, vocab_size=100, min_frequency=3):
        pass

    def encode(self, text):
        pass

    def decode(self, token_ids):
        pass

    def save(self, filepath):
        pass

    @classmethod
    def load(cls, filepath):
        pass