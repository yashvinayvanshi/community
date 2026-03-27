import numpy as np
import matplotlib.pyplot as plt
import os

# ─────────────────────────────────────────────
#  Graph definition  (20 nodes, 3 clusters)
# ─────────────────────────────────────────────
#
#  TOP-LEFT cluster  : 0 – 6
#  RIGHT cluster     : 7 – 12
#  BOTTOM cluster    : 13 – 19
#
#  Inter-cluster edges:
#      6 -- 7   (top-left <-> right)
#      6 -- 13  (top-left <-> bottom)
#     12 -- 13  (right    <-> bottom)

N = 20

edges = [
    # TOP-LEFT cluster (0-6)
    (0,1),(0,2),(0,3),(0,4),
    (1,2),(1,4),(1,6),
    (2,3),(2,5),
    (3,4),(3,5),(3,6),
    (4,5),(4,6),
    (5,6),
    # RIGHT cluster (7-12)
    (7,8),(7,9),(7,10),(7,11),
    (8,9),(8,11),(8,12),
    (9,10),(9,11),(9,12),
    (10,11),(10,12),
    (11,12),
    # BOTTOM cluster (13-19)
    (13,14),(13,15),(13,16),(13,17),
    (14,15),(14,16),(14,18),
    (15,16),(15,17),(15,19),
    (16,17),(16,18),
    (17,18),(17,19),
    (18,19),
    # INTER-CLUSTER edges
    (6,7),(6,13),(12,13),
]

# Build adjacency matrix
A = np.zeros((N, N), dtype=np.int64)
for u, v in edges:
    A[u][v] = 1
    A[v][u] = 1

# ─────────────────────────────────────────────
#  Print adjacency list (matching C++ output)
# ─────────────────────────────────────────────

print("+" + "=" * 38 + "+")
print(f"|        ADJACENCY LIST  (N = {N})       |")
print("+" + "=" * 38 + "+")
for i in range(N):
    neighbors = list(np.where(A[i] == 1)[0])
    print(f"  Node {i:2d} -> [{', '.join(str(x) for x in neighbors)}]")

# ─────────────────────────────────────────────
#  Get k from user
# ─────────────────────────────────────────────

k = int(input("\nEnter k (compute A^1 through A^k): "))
assert k >= 1, "k must be >= 1"

# ─────────────────────────────────────────────
#  Compute powers and cumulative sums
# ─────────────────────────────────────────────

powers = [A.copy()]          # powers[0] = A^1
cumulative_sums = [A.copy()] # cumulative_sums[0] = A^1

power = A.copy()
cum_sum = A.copy()

for p in range(2, k + 1):
    power = power @ A         # A^p
    cum_sum = cum_sum + power  # A^1 + A^2 + ... + A^p
    powers.append(power.copy())
    cumulative_sums.append(cum_sum.copy())

# ─────────────────────────────────────────────
#  Print matrices (matching C++ format)
# ─────────────────────────────────────────────

def print_matrix(M, name, w=6):
    print(f"\n== {name} ==")
    header = "     " + "".join(f" {j:>{w}} " for j in range(N))
    print(header)
    print("     " + "-" * (N * (w + 2)))
    for i in range(N):
        row = f"{i:3d} |" + "".join(f" {M[i][j]:>{w}} " for j in range(N))
        print(row)

print_matrix(A, "A^1  (Adjacency Matrix)")
for p in range(1, k):
    print_matrix(powers[p], f"A^{p+1}")

sum_label = f"SUM  ( A^1 + A^2 + ... + A^{k} )"
print_matrix(cumulative_sums[-1], sum_label, w=8)

# ─────────────────────────────────────────────
#  Heatmap generation
# ─────────────────────────────────────────────

script_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(script_dir, "outputs")
os.makedirs(out_dir, exist_ok=True)

# Community boundaries for annotation lines
boundaries = [6.5, 12.5]  # between node 6-7 and 12-13


def save_heatmap(matrix, title, filename):
    fig, ax = plt.subplots(figsize=(8, 7))

    vmax = matrix.max()
    vmin = matrix.min()

    im = ax.imshow(matrix, cmap="YlOrRd", aspect="equal",
                   vmin=vmin, vmax=vmax, interpolation="nearest")

    # Community boundary lines
    for b in boundaries:
        ax.axhline(y=b, color="black", linewidth=2, linestyle="--", alpha=0.7)
        ax.axvline(x=b, color="black", linewidth=2, linestyle="--", alpha=0.7)

    # Annotate community regions
    ax.text(3, -1.2, "Cluster 1\n(0-6)", ha="center", fontsize=9,
            fontweight="bold", color="#333")
    ax.text(9.5, -1.2, "Cluster 2\n(7-12)", ha="center", fontsize=9,
            fontweight="bold", color="#333")
    ax.text(16, -1.2, "Cluster 3\n(13-19)", ha="center", fontsize=9,
            fontweight="bold", color="#333")

    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xticklabels(range(N), fontsize=7)
    ax.set_yticklabels(range(N), fontsize=7)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=30)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("Value", fontsize=10)

    plt.tight_layout()
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


print("\n--- Generating heatmaps ---\n")

# Individual powers: A^1, A^2, ..., A^k
for p in range(k):
    save_heatmap(
        powers[p],
        f"$A^{p+1}$",
        f"power_A{p+1}.png"
    )

# Cumulative sums: A^1, A^1+A^2, ..., A^1+...+A^k
for s in range(k):
    terms = " + ".join(f"A^{i+1}" for i in range(s + 1))
    save_heatmap(
        cumulative_sums[s],
        f"Sum: ${terms}$",
        f"cumsum_upto_A{s+1}.png"
    )

# ─────────────────────────────────────────────
#  CSV export of final sum
# ─────────────────────────────────────────────

csv_path = os.path.join(out_dir, f"sum_A1_to_A{k}.csv")
np.savetxt(csv_path, cumulative_sums[-1], delimiter=",", fmt="%d")
print(f"\n  CSV exported -> {csv_path}")

print("\nDone.")
