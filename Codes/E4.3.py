import itertools
import numpy as np
import matplotlib.pyplot as plt


def convert_symbol_to_input(c):
    return np.array([1 - 2 * i for i in c])


def bawgn(c, sigma2):
    x = convert_symbol_to_input(c)
    return [np.random.normal(i, np.sqrt(sigma2)) for i in x]


def snr_db_to_noise_var(snr):
    return (1 / (10 ** (snr / 10))) / 2


G = np.array([
    [1, 0, 0, 0, 1, 1, 1],
    [0, 1, 0, 0, 0, 1, 1],
    [0, 0, 1, 0, 1, 0, 1],
    [0, 0, 0, 1, 1, 1, 0]
])

U = np.array(list(itertools.product([0, 1], repeat=4)))
C = [
    np.sum(np.array(u)[:, None] * G, axis=0) % 2 for u in U
]

N = int(1e5)
snr_db_list = [i for i in range(1, 10)]

wer_list, ber_list, uncoded_ber_list = [], [], []
for snr in snr_db_list:
    sigma2 = snr_db_to_noise_var(snr)
    wer = ber = uncoded_ber = 0
    for _ in range(N):
        u = np.random.choice([0, 1], size=4)
        c = u.dot(G) % 2
        y = bawgn(c, sigma2)

        diffs = [
            np.sum(np.power(convert_symbol_to_input(i) - y, 2), axis=-1) for i in C
        ]
        c_hat = C[np.argsort(diffs)[0]]
        if not np.array_equal(c, c_hat):
            wer += 1
            ber += sum([1 for i, j in zip(c, c_hat) if i != j])
        
        y = bawgn(u, sigma2)
        diffs = [
            np.sum(np.power(convert_symbol_to_input(i) - y, 2), axis=-1) for i in U
        ]
        u_hat = U[np.argsort(diffs)[0]]
        if not np.array_equal(u, u_hat):
            uncoded_ber += sum([1 for i, j in zip(u, u_hat) if i != j])

    wer_list.append(wer / N)
    ber_list.append(ber / (7 * N))
    uncoded_ber_list.append(uncoded_ber / (4 * N))
    print(f"SNR/Sigma2: {snr}/{sigma2:.04} - WER: {wer / N:0.4} - "
          f"BER: {ber / (7 * N):0.4} - Uncoded BER: {uncoded_ber / (4 * N):0.4}")

print(wer_list)
print(ber_list)

plt.plot(snr_db_list, wer_list, color='blue')
plt.plot(snr_db_list, ber_list, color='red')
plt.plot(snr_db_list, uncoded_ber_list, color='green')
plt.xlabel("SNR dB")
plt.legend(["WER", "BER", "Uncoded BER"])
plt.yscale('log')
plt.savefig('4.3.jpg')
