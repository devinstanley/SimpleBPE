import requests
from SimpleBPE import SimpleBPETokenizer

class Tester:
    def __init__(self, url = None, verbosity = 0):
        self.url = url or "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        self.verbosity = verbosity
    
    def load_data(self):
        response = requests.get(self.url)
        self.text = response.text

    def run_small_BPE(self):
        bpe = SimpleBPETokenizer()
        bpe.train(self.text, 200, 3, verbosity=self.verbosity)
        self.test_encoding(bpe)

    def test_encoding(self, bpe):
        text_to_encode = self.text[0:40]
        tokens = bpe.encode(text_to_encode)
        decoded_text = bpe.decode(tokens)

        print(f"Encoded Text:\n {text_to_encode}\n")
        print(f"Tokenized Text:\n {tokens}\n")
        print(f"Decoded Text:\n {decoded_text}")

        assert text_to_encode == decoded_text

if __name__ == "__main__":
    print("Starting Tester...")
    tester = Tester(None, 1)
    tester.load_data()
    tester.run_small_BPE()