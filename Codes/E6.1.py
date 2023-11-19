import numpy as np
import matplotlib.pyplot as plt


def check_node_out(d, v):
    return 1 - (1 - v) ** (d - 1)


def variable_node_out(d, sigma, u):
    return sigma * u ** (d - 1)


code_pairs = [(2, 6), (3, 9), (4, 12), (5, 15)]
max_iter = 1e5
for i, (dv, du) in enumerate(code_pairs):
    sigma_range = (0.0, 0.5)
    while True:
        _sigma = v_u = (sigma_range[0] + sigma_range[1]) / 2
        n_iter = 0
        converged = False
        while n_iter < max_iter:
            u_v = check_node_out(du, v_u)
            v_u = variable_node_out(dv, _sigma, u_v)
            if v_u < 1e-10:
                converged = True
                break
            n_iter += 1
        if converged:
            sigma_range = (_sigma, sigma_range[1])
        else:
            sigma_range = (sigma_range[0], _sigma)
        new_sigma = (sigma_range[0] + sigma_range[1]) / 2
        if np.round(new_sigma, 5) == np.round(_sigma, 5):
            break
    print(f"Noise threshold of {(dv, du)}: {_sigma}")


# sigma = 0.2

# v_range = np.linspace(0, 0.5, num=100)
# u_range = np.linspace(0, 1, num=100)

# for i, (dv, du) in enumerate(code_pairs):
#     u_v = [check_node_out(du, v) for v in v_range]
#     v_u = [variable_node_out(dv, sigma, u) for u in u_range]

#     fig = plt.figure(figsize=(8, 8))
#     plt.plot(v_range, u_v, color='red')
#     plt.plot(v_u, u_range, color='blue')
#     plt.xlabel("v")
#     plt.ylabel("u")
#     plt.legend(["var node", "check node"])
#     fig.savefig(f'6.1.{i}.jpg')

