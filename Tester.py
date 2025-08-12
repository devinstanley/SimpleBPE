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


if __name__ == "__main__":
    print("Starting Tester...")
    tester = Tester(None, 1)
    tester.load_data()
    tester.run_small_BPE()