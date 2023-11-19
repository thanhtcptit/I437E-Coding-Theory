import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm


def create_quasi_cyclic_ldpc(z, proto):
    P = np.identity(z)
    H = []
    for r in range(proto.shape[0]):
        H_row = []
        for c in range(proto.shape[1]):
            if proto[r, c] == -1:
                H_row.append(np.zeros((z, z)))
            else:
                H_row.append(np.roll(P, proto[r, c], axis=1))
        H.append(np.hstack(H_row))
    return np.vstack(H)


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


n, k = 648, 432
rate = 1 - (k / n)

z = 27
proto = np.array([
    [25, 26, 14, -1, 20, -1,  2, -1,  4, -1, -1,  8, -1, 16, -1, 18,  1,  0, -1, -1, -1, -1, -1, -1],
    [10, 9, 15, 11, -1, 0, -1,  1, -1, -1, 18, -1, 8, -1, 10, -1, -1, 0, 0, -1, -1, -1, -1, -1,],
    [16, 2, 20, 26, 21, -1, 6, -1, 1, 26, -1, 7, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1],
    [10, 13, 5, 0, -1, 3, -1, 7, -1, -1, 26, -1, -1, 13, -1, 16, -1, -1, -1, 0, 0, -1, -1, -1],
    [23, 14, 24, -1, 12, -1, 19, -1, 17, -1, -1, -1, 20, -1, 21, -1, 0, -1, -1, -1, 0, 0, -1, -1],
    [6, 22, 9, 20, -1, 25, -1, 17, -1, 8, -1, 14, -1, 18, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1],
    [14, 23, 21, 11, 20, -1, 24, -1, 18, -1, 19, -1, -1, -1, -1, 22, -1, -1, -1, -1, -1, -1, 0, 0],
    [17, 11, 11, 20, -1, 21, -1, 26, -1, 3, -1, -1, 18, -1, 26, -1, 1, -1, -1, -1, -1, -1, -1, 0]
])
H = create_quasi_cyclic_ldpc(z, proto)
m = H.shape[0]
check_node_neighbors = [[c for c in range(n) if H[r, c] == 1] for r in range(m)]
var_node_neighbors = [[r for r in range(m) if H[r, c] == 1] for c in range(n)]

n_mcmc = int(1e4)
max_iter = 5
u = np.array([0] * k)
c = np.array([0] * n)
snr_db_list = [i for i in range(1, 11)]

wer_list, ber_list, uncoded_ber_list = [], [], []
for snr in snr_db_list:
    sigma2 = eb_snr_db_to_noise_var(snr, rate)
    wer = ber = uncoded_ber = 0
    for _ in tqdm(range(n_mcmc)):
        y = bawgn(c, sigma2)
        c_hat = sum_product_decoding(
            y, H, sigma2, check_node_neighbors, var_node_neighbors, max_iter)
        if not np.array_equal(c, c_hat):
            wer += 1
            ber += sum([1 for i, j in zip(c, c_hat) if i != j])

        y = bawgn(u, sigma2)
        u_hat = np.array([0 if y[i] >= 0 else 1 for i in range(k)])
        if not np.array_equal(u, u_hat):
            uncoded_ber += sum([1 for i, j in zip(u, u_hat) if i != j])

    wer_list.append(wer / n_mcmc)
    ber_list.append(ber / (n * n_mcmc))
    uncoded_ber_list.append(uncoded_ber / (k * n_mcmc))
    print(f"SNR/Sigma2: {snr}/{sigma2:.04} - WER: {wer / n_mcmc:0.4} - "
          f"BER: {ber / (n * n_mcmc):0.4} - Uncoded BER: {uncoded_ber / (k * n_mcmc):0.4}")

print(wer_list)
print(ber_list)
print(uncoded_ber_list)

plt.plot(snr_db_list, wer_list, color='blue')
plt.plot(snr_db_list, ber_list, color='red')
plt.plot(snr_db_list, uncoded_ber_list, color='green')
plt.xlabel("SNR dB")
plt.legend(["WER", "BER", "Uncoded BER"])
plt.yscale('log')
plt.savefig('6.3.jpg')
