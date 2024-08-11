import pickle
import sys

if __name__ == '__main__':
    name = sys.argv[1]
    # with open(f'bpe_out/{name}/merges.pkl', 'rb') as f:
    #     merges = pickle.load(f)
    with open(f'bpe_out/{name}/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)

    # Process vocab however you want