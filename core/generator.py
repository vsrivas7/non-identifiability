import numpy as np
import random

VOCAB = [f"w{i}" for i in range(2000)]

def make_output(length=80):
    return " ".join(np.random.choice(VOCAB, size=length))

def corrupt(text, strength=0.1):
    tokens = text.split()
    k = max(1, int(len(tokens) * strength))
    idx = np.random.choice(len(tokens), k, replace=False)

    for i in idx:
        tokens[i] = random.choice(VOCAB)

    return " ".join(tokens)

def generate_dataset(n=1000, delta=0.1):
    p = 0.05
    texts, labels = [], []

    for _ in range(n):
        base = make_output()

        if random.random() < (p + delta):
            texts.append(corrupt(base))
            labels.append(1)
        else:
            texts.append(base)
            labels.append(0)

    return texts, np.array(labels)