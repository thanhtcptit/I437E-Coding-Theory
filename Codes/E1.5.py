import numpy as np
import scipy


def channels(x, sigma2):
    return np.random.normal(x, np.sqrt(sigma2))


def prob_error(sigma2):
    return 0.5 * scipy.special.erfc(1 / np.sqrt(2 * sigma2))


N = 10000
sigma2_list = [1, 0.5, 0.15]

X = [-1, 1]
p_X = [0.5, 0.5]

for sigma2 in sigma2_list:
    N_err = 0
    for _ in range(N):
        x = np.random.choice(X, size=1, p=p_X)
        y = channels(x, sigma2)
        x_hat = -1 if y < 0 else 1
        if x != x_hat:
            N_err += 1
    print(f"Sigma2: {sigma2} - Estimated P_e: {N_err / N} - "
          f"Analytical P_e: {prob_error(sigma2)}")
