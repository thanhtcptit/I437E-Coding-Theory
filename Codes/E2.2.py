import matplotlib.pyplot as plt


G = [
    (i, j) for i in range(0, 7) for j in range(0, 7)
]
H = [
    (3 * x, 3 * y) for x in range(0, 3) for y in range(0, 3)
]
G_H = set(G) & set(H)

plt.scatter(
    [x for x, _ in set(G) - set(G_H)], [y for _, y in set(G) - set(G_H)],
    marker="o", c=[[1, 0, 0]]
)
plt.scatter(
    [x for x, _ in G_H], [y for _, y in G_H],
    marker="o", c=[[0, 0, 1]]
)
plt.legend(["G", "G & H"])
plt.show()
plt.savefig("2.2.jpg")