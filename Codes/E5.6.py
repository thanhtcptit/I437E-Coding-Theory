import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm


def convert_symbol_to_input(c):
    return np.array([1 - 2 * i for i in c])


def bawgn(c, sigma2):
    x = convert_symbol_to_input(c)
    return np.random.normal(x, np.sqrt(sigma2))


def eb_snr_db_to_noise_var(snr, rate):
    return 1 / (2 * (10 ** (snr / 10)) * rate)


def sum_product_decoding(y, H, sigma2, check_nei, var_nei, max_iter):
    L = (2 / sigma2) * y
    Q = [L[check_nei[check_i]] for check_i in range(m)]
    Q_calc = [np.prod(np.tanh(np.array(Q[check_i]) / 2))
              for check_i in range(m)]

    n_iter = 0
    while n_iter < max_iter:
        R = [
            [2 * np.arctanh(Q_calc[check_i] / np.tanh(Q[check_i][check_nei[check_i].index(var_i)] / 2))
             for check_i in var_nei[var_i]]
            for var_i in range(n)
        ]
        R_calc = [L[var_i] + np.sum(R[var_i]) for var_i in range(n)]

        Q = [
            [R_calc[var_i] - R[var_i][var_nei[var_i].index(check_i)]
             for var_i in check_nei[check_i]]
            for check_i in range(m)
        ]
        Q_calc = [np.prod(np.tanh(np.array(Q[check_i]) / 2)) for check_i in range(m)]

        c_hat = np.array([0 if v >= 0 else 1 for v in R_calc])
        if np.array_equal((c_hat.dot(H.T) % 2), np.zeros(m)):
            break
        n_iter += 1
    return c_hat


n_mcmc = int(1e3)
eb_snr_db_list = [i for i in range(1, 11)]
max_iter = 5

H = np.array([
    [1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1],
    [1, 0, 0, 1, 0, 1, 0, 1],
    [0, 1, 1, 0, 1, 0, 1, 0]
])

m, n = H.shape
c = [0] * n
rate = 1 - (np.linalg.matrix_rank(H) / n)

var_nei = [[r for r in range(m) if H[r, c] == 1] for c in range(n)]
check_nei = [[c for c in range(n) if H[r, c] == 1] for r in range(m)]

wer_list, ber_list = [], []
for snr in eb_snr_db_list:
    sigma2 = eb_snr_db_to_noise_var(snr, rate)
    wer = ber = 0
    for _ in tqdm(range(n_mcmc)):
        y = bawgn(c, sigma2)
        c_hat = sum_product_decoding(y, H, sigma2, check_nei, var_nei, max_iter)

        if not np.array_equal(c, c_hat):
            wer += 1
            ber += sum([1 for i, j in zip(c, c_hat) if i != j])

    wer_list.append(wer / n_mcmc)
    ber_list.append(ber / (n * n_mcmc))
    print(f"SNR/Sigma2: {snr}/{sigma2:.04} - WER: {wer / n_mcmc:0.4} - BER: {ber / (n * n_mcmc):0.4}")

print(wer_list)
print(ber_list)

plt.plot(eb_snr_db_list, wer_list, color='blue')
plt.plot(eb_snr_db_list, ber_list, color='red')
plt.xlabel("Eb / N0 SNR dB")
plt.legend(["WER", "BER"])
plt.yscale('log')
plt.savefig('5.6.jpg')
