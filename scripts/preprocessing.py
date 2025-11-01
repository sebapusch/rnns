import numpy as np
from os import path
import re
import string

import pickle

import datasets
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import normalize

DATASET_NAME = "stanfordnlp/sst2"
EMBEDDINGS_URL = path.join('assets', 'embeddings', "wiki_giga_2024_100_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05.050_combined.txt")
EMBEDDING_SIZE = 100

TrainVal = tuple[np.ndarray, np.ndarray, np.ndarray]


def load_glove_embeddings():
    embeddings = {}
    with open(EMBEDDINGS_URL, 'r', encoding="utf8") as f:
        for line in f:
            parts = line.strip().split()
            word  = ''.join(parts[:len(parts) - EMBEDDING_SIZE])

            embeddings[word] = np.array(parts[-EMBEDDING_SIZE:], dtype=np.float32)
    return embeddings

class Embedder:
    def __init__(self, embeddings: dict, rules_path: str | None = None) -> None:
        self.embeddings = embeddings
        self.rules = self._load_rules(rules_path) if rules_path else {}

        # track missing tokens for later inspection
        self.missing: list[str] = []

    def _load_rules(self, rules_path: str) -> dict[str, list[str]]:
        """
        Load token correction/splitting rules from a text file.
        Format:
            original_token corrected_token1 [corrected_token2 ...]
        """
        rules = {}
        if not path.exists(rules_path):
            print(f"[WARN] No token rules file found at: {rules_path}")
            return rules

        with open(rules_path, "r", encoding="utf8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    key = parts[0]
                    values = parts[1:]
                    rules[key] = values
        print(f"[INFO] Loaded {len(rules)} token rules from {rules_path}")
        return rules

    def tokenize(self, sentence: str) -> list[str]:
        sentence = sentence.lower()
        sentence = sentence.replace('-', ' ')
        # remove punctuation (keep apostrophes if desired)
        sentence = re.sub(f"[{re.escape(string.punctuation)}]", "", sentence)
        return sentence.split()

    def __call__(self, sentence: str) -> np.ndarray:
        tokens = self.tokenize(sentence)
        sequence_embeddings = []

        for token in tokens:
            # Apply manual correction/splitting rules if available
            if token in self.rules:
                rule_tokens = self.rules[token]
            else:
                rule_tokens = [token]

            # For each resulting token, look up or assign <unk>
            for t in rule_tokens:
                if t in self.embeddings:
                    sequence_embeddings.append(self.embeddings[t])
                else:
                    self.missing.append(t)
                    sequence_embeddings.append(np.zeros(EMBEDDING_SIZE, dtype=np.float32))

        # fallback if empty (e.g., all unknown tokens)
        if not sequence_embeddings:
            sequence_embeddings = [np.zeros(EMBEDDING_SIZE, dtype=np.float32)]

        return np.stack(sequence_embeddings)

    

    @staticmethod
    def generate_or_load() -> 'Embedder':
        if path.exists('embedder.pickle'):
            return Embedder.load()
        
        embedder = Embedder(load_glove_embeddings())
        embedder.save()
        return embedder

    @staticmethod
    def load() -> 'Embedder':
        with open(path.join('assets', 'embeddings', 'embedder.pickle'), 'rb') as f:
            embeddings = pickle.load(f)
            return Embedder(embeddings)

    def save(self) -> None:
        with open(path.join('assets', 'embeddings', 'embedder.pickle'), 'wb') as f:
            pickle.dump(self.embeddings, f)
    
            

def balanced_sample(split: datasets.Dataset, size: int, seed: int) -> datasets.Dataset:
    """
    Randomly select a balanced subset with equal number of label 0 and label 1.
    Raises a clear error if not enough examples exist.
    """
    if size <= 0:
        raise ValueError("Requested size must be > 0.")

    zeros = split.filter(lambda ex: ex["label"] == 0)
    ones  = split.filter(lambda ex: ex["label"] == 1)

    per_class = size // 2
    if per_class == 0:
        raise ValueError(f"Requested size {size} too small to balance into two classes.")

    if len(zeros) < per_class or len(ones) < per_class:
        raise ValueError(
            f"Not enough examples for a balanced sample of size {size}: "
            f"class 0 has {len(zeros)}, class 1 has {len(ones)}"
        )

    zeros = zeros.shuffle(seed=seed).select(range(per_class))
    ones  = ones.shuffle(seed=seed + 1).select(range(per_class))

    subset = datasets.concatenate_datasets([zeros, ones]).shuffle(seed=seed + 2)

    return subset

def preprocess(embedder: Embedder, dataset: datasets.Dataset) -> tuple[list, np.ndarray, int]:
    x = []
    y = []

    max_size = -1
    print(f'preprocessing {len(dataset)} samples...')
    for datapoint in tqdm(dataset):
        embeddings = embedder(datapoint['sentence']) # type: ignore
        x.append(normalize(embeddings, 'l2'))
        y.append(datapoint['label']) # type: ignore
        
        max_size = max(max_size, len(embeddings))

    return x, np.stack(y), max_size

def pad(sequence: list, max_size: int) -> tuple[np.ndarray, np.ndarray]:
    print(f'applying padding and computing sequence length...')
    sequence_length = np.zeros(len(sequence), dtype=int)
    for i, embeddings in tqdm(enumerate(sequence)):
        if len(embeddings) == max_size:
            sequence_length[i] = max_size
            continue
        sequence_length[i] = len(embeddings)
        sequence[i] = np.concatenate((embeddings, np.zeros((max_size - len(embeddings), 100))))

    return np.stack(sequence), sequence_length

def load_data(size: int, seed: int) -> TrainVal:
    np.random.seed(seed)

    print('loading embeddings...')
    embedder = Embedder.generate_or_load()
    print('embeddings loaded...')

    dataset = datasets.load_dataset(DATASET_NAME)['train']

    assert type(dataset) == datasets.Dataset

    data = balanced_sample(dataset, size, seed)

    X, Y, max_size = preprocess(embedder, data)
    X, S = pad(X, max_size)

    return (X, Y, S)

if __name__ == '__main__':
    X, Y, S = load_data(58000, 11)

    permutation = np.random.permutation(len(X))
    X = X[permutation]
    Y = Y[permutation]
    S = S[permutation]
    
    train = (X[:40000], Y[:40000], S[:40000])
    validation = (X[40000:45000], Y[40000:45000], S[40000:45000])
    test = (X[45000:], Y[45000:], S[45000:])

    np.save(path.join('data', 'test-x'), test[0])
    np.save(path.join('data', 'test-y'), test[1])
    np.save(path.join('data', 'test-s'), test[2])

    np.save(path.join('data', 'validation-x'), validation[0])
    np.save(path.join('data', 'validation-y'), validation[1])
    np.save(path.join('data', 'validation-s'), validation[2])

    np.save(path.join('data', 'train-x'), train[0])
    np.save(path.join('data', 'train-y'), train[1])
    np.save(path.join('data', 'train-s'), train[2])