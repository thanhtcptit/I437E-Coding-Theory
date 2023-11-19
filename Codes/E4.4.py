import scipy
import itertools
import numpy as np
import matplotlib.pyplot as plt


def eb_snr_db_to_noise_var(snr, R):
    return (1 / (2 * (10 ** (snr / 10)) * R))


G = np.array([
    [1, 0, 0, 0, 1, 1, 1],
    [0, 1, 0, 0, 0, 1, 1],
    [0, 0, 1, 0, 1, 0, 1],
    [0, 0, 0, 1, 1, 1, 0]
])
R = 4 / 7

U = np.array(list(itertools.product([0, 1], repeat=4)))
C = [
    np.sum(np.array(u)[:, None] * G, axis=0) % 2 for u in U
]
weight_dist = {
    w: len([c for c in C if np.sum(c) == w]) for w in range(1, 8)
}
weight_dist = list(filter(lambda x: x[1] != 0, sorted(weight_dist.items(), key=lambda x: x[0])))

N = int(1e5)
eb_snr_db_list = [i for i in range(1, 11)]

union_bound_list, union_bound_est_list = [], []
for snr in eb_snr_db_list:
    sigma2 = eb_snr_db_to_noise_var(snr, R)
    union_bound_list.append(
        0.5 * np.sum([c * scipy.special.erfc(np.sqrt(w / (2 * sigma2)))
                      for w, c in weight_dist])
    )
    union_bound_est_list.append(
        0.5 * weight_dist[0][1] * scipy.special.erfc(np.sqrt(weight_dist[0][0] / (2 * sigma2))))

    print(f"SNR/Sigma2: {snr}/{sigma2:.04} - Union bound: {union_bound_list[-1]:0.4} - "
          f"Union bound estimate: {union_bound_est_list[-1]:0.4}")


plt.plot(eb_snr_db_list, union_bound_list, '--', linewidth=2, color='red')
plt.plot(eb_snr_db_list, union_bound_est_list, color='green')
plt.xlabel("Eb / N0 SNR dB")
plt.ylabel("WER")
plt.legend(["Union bound", "Union bound estimate"])
plt.yscale('log')
plt.savefig('4.4.jpg')