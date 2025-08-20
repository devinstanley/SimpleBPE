import requests
import numpy as np
from SimpleBPE import SimpleBPETokenizer
from SimpleEmbeddings import SimpleSkipGramEmbeddings

tests = {
    'small': {
        'vocab_size': 8000,
        'min_frequency': 3,
        'embedding_dimension': 256,
        'embedding_lr': 0.025,
        'embedding_epochs': 3,
        'embedding_window': 10,
        'embedding_negative_samples': 10,
        'embedding_subsample_thresh': 1e-4
    }
}

class Tester:
    def __init__(self, url = None, verbosity = 0):
        self.url = url or "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        self.verbosity = verbosity
        self.load_data()
    
    def load_data(self):
        response = requests.get(self.url)
        self.text = response.text

    def run_tests(self, test):
        bpe = self.run_BPE(test)
        bpe.save("tokens.pht")
        #bpe = SimpleBPETokenizer.load("tokens.pht")
        self.test_encoding(bpe)
        self.embed_tokens(bpe, test)

    # BPE Testing
    def run_BPE(self, test):
        bpe = SimpleBPETokenizer()
        bpe.train(self.text, test['vocab_size'], test['min_frequency'], verbosity=self.verbosity)

        assert len(bpe.vocab) == test['vocab_size']

        return bpe

    def test_encoding(self, bpe):
        text_to_encode = self.text[0:40]
        tokens = bpe.encode(text_to_encode)
        decoded_text = bpe.decode(tokens)

        print(f"Encoded Text:\n {text_to_encode}\n")
        print(f"Tokenized Text:\n {tokens}\n")
        print(f"Decoded Text:\n {decoded_text}")

        assert text_to_encode == decoded_text

    # Embed Testing
    def embed_tokens(self, bpe, test):
        embedding = SimpleSkipGramEmbeddings(test['vocab_size'], test['embedding_dimension'], self.verbosity)
        tokens = bpe.encode(self.text)
        embedding.train_on_tokens(
            tokens, 
            window_size=test['embedding_window'], 
            epochs=test['embedding_epochs'], 
            lr=test['embedding_lr'], 
            negative_samples=test['embedding_negative_samples'],
            subsample_threshold=test['embedding_subsample_thresh']
        )

        if len(tokens) > 10:
            for i in range(40):
                test_token = np.random.choice(tokens)
                decoded_token = bpe.decode([test_token])
                similar = embedding.most_similar(test_token, top_k=3)
                decoded_similar = [bpe.decode([s[0]]) for s in similar]
                print(f"{i}. Most similar to token {test_token}: {similar}")
                print(f"\tMost similar to decoded token {decoded_token}: {decoded_similar}")

if __name__ == "__main__":
    print("Starting Tester...")
    tester = Tester(None, 1)
    for test in tests.keys():
        tester.run_tests(tests[test])