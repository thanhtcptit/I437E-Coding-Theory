import numpy as np


def bsc(u, p):
    return [i if np.random.choice([0, 1], p=[p, 1 - p]) == 1 else (1 - i)
            for i in u]

H = np.array([
    [1, 0, 1, 1, 1, 0, 0],
    [1, 1, 0, 1, 0, 1, 0],
    [1, 1, 1, 0, 0, 0, 1]
])

G = np.array([
    [1, 0, 0, 0, 1, 1, 1],
    [0, 1, 0, 0, 0, 1, 1],
    [0, 0, 1, 0, 1, 0, 1],
    [0, 0, 0, 1, 1, 1, 0]
])

syndrome_tables = {
    (0, 0, 0): np.array([0, 0, 0, 0, 0, 0, 0]),
    (0, 0, 1): np.array([0, 0, 0, 0, 0, 0, 1]),
    (0, 1, 0): np.array([0, 0, 0, 0, 0, 1, 0]),
    (0, 1, 1): np.array([0, 1, 0, 0, 0, 0, 0]),
    (1, 0, 0): np.array([0, 0, 0, 0, 1, 0, 0]),
    (1, 0, 1): np.array([0, 0, 1, 0, 0, 0, 0]),
    (1, 1, 0): np.array([0, 0, 0, 1, 0, 0, 0]),
    (1, 1, 1): np.array([1, 0, 0, 0, 0, 0, 0])
}

N = int(1e4)
p_list = [0.1, 0.025, 0.0025]

for p in p_list:
    wer = 0
    for _ in range(N):
        u = np.random.choice([0, 1], size=4)
        c = u.dot(G) % 2
        y = bsc(c, p)
        s = tuple((H.dot(y) % 2).tolist())
        e = syndrome_tables[s]
        c_hat = (y + e) % 2

        if not np.array_equal(c, c_hat):
            wer += 1

    print(f"P: {p} - WER: {wer / N}")
