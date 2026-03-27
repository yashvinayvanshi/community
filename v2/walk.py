import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import sys

# ─────────────────────────────────────────────
#  Paths relative to this script
# ─────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR  = os.path.join(SCRIPT_DIR, "inputs")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")

# ─────────────────────────────────────────────
#  Input file parser
# ─────────────────────────────────────────────
#
#  FORMAT:
#    # comment
#    nodes: 20
#    cluster: 0-6 Top-Left       (optional, for visualization)
#    cluster: 7-12 Right
#    edges:
#    0 1
#    2 3
#    ...

def parse_input(filepath):
    """Parse a graph input file. Returns (N, edges, clusters)."""
    N = None
    edges = []
    clusters = []       # list of (start, end, name)
    reading_edges = False

    with open(filepath) as f:
        for raw_line in f:
            line = raw_line.strip()

            # skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            # node count
            if line.lower().startswith("nodes:"):
                N = int(line.split(":")[1].strip())
                reading_edges = False
                continue

            # cluster definition
            if line.lower().startswith("cluster:"):
                rest = line.split(":", 1)[1].strip()       # "0-6 Top-Left"
                parts = rest.split(None, 1)                 # ["0-6", "Top-Left"]
                rng = parts[0]                              # "0-6"
                name = parts[1] if len(parts) > 1 else rng  # "Top-Left"
                a, b = map(int, rng.split("-"))
                clusters.append((a, b, name))
                reading_edges = False
                continue

            # edge section marker
            if line.lower().startswith("edges"):
                reading_edges = True
                continue

            # edge data
            if reading_edges:
                tokens = line.split()
                if len(tokens) >= 2:
                    u, v = int(tokens[0]), int(tokens[1])
                    edges.append((u, v))

    if N is None:
        # infer N from edges
        N = max(max(u, v) for u, v in edges) + 1
        print(f"  (node count inferred from edges: {N})")

    return N, edges, clusters

# ─────────────────────────────────────────────
#  Graph plotting (force-directed layout,
#  no external dependencies beyond numpy/mpl)
# ─────────────────────────────────────────────

def force_directed_layout(N, edges, iterations=500):
    """Simple Fruchterman-Reingold-style layout."""
    np.random.seed(42)
    pos = np.random.rand(N, 2) * 10

    # build adjacency for quick lookup
    adj = [[] for _ in range(N)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    k_ideal = np.sqrt(100.0 / max(N, 1))   # ideal spring length
    temp = 5.0

    for it in range(iterations):
        disp = np.zeros((N, 2))

        # repulsive forces (all pairs)
        for i in range(N):
            for j in range(i + 1, N):
                delta = pos[i] - pos[j]
                dist = max(np.linalg.norm(delta), 0.01)
                force = (k_ideal ** 2) / dist
                direction = delta / dist
                disp[i] += direction * force
                disp[j] -= direction * force

        # attractive forces (edges)
        for u, v in edges:
            delta = pos[u] - pos[v]
            dist = max(np.linalg.norm(delta), 0.01)
            force = (dist ** 2) / k_ideal
            direction = delta / dist
            disp[u] -= direction * force
            disp[v] += direction * force

        # apply with temperature cooling
        for i in range(N):
            mag = max(np.linalg.norm(disp[i]), 0.01)
            pos[i] += (disp[i] / mag) * min(mag, temp)

        temp *= 0.97  # cool down

    # normalize to [0, 1]
    pos -= pos.min(axis=0)
    rng = pos.max(axis=0)
    rng[rng == 0] = 1
    pos /= rng

    return pos


def save_graph_plot(N, edges, clusters, out_path):
    """Draw and save the graph with nodes colored by cluster."""
    pos = force_directed_layout(N, edges)

    # assign colors by cluster
    palette = [
        "#7F77DD", "#1D9E75", "#D85A30", "#D4537E",
        "#378ADD", "#639922", "#BA7517", "#E24B4A", "#888780"
    ]

    node_colors = ["#888780"] * N       # default gray
    node_cluster = [-1] * N
    for ci, (a, b, name) in enumerate(clusters):
        for node in range(a, b + 1):
            if node < N:
                node_colors[node] = palette[ci % len(palette)]
                node_cluster[node] = ci

    fig, ax = plt.subplots(figsize=(10, 8))

    # draw edges
    for u, v in edges:
        same_cluster = (node_cluster[u] == node_cluster[v] and node_cluster[u] >= 0)
        color = "#B4B2A9" if not same_cluster else "#D3D1C7"
        width = 1.5 if same_cluster else 2.0
        style = "-" if same_cluster else "--"
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color=color, linewidth=width, linestyle=style, zorder=1)

    # draw nodes
    node_size = max(80, 400 - N * 5)   # scale down for large graphs
    font_size = max(5, 10 - N // 15)
    for i in range(N):
        ax.scatter(pos[i][0], pos[i][1], s=node_size, c=node_colors[i],
                   edgecolors="white", linewidths=2, zorder=3)
        ax.text(pos[i][0], pos[i][1], str(i), ha="center", va="center",
                fontsize=font_size, fontweight="bold", color="white", zorder=4)

    # legend
    if clusters:
        handles = []
        for ci, (a, b, name) in enumerate(clusters):
            handles.append(mpatches.Patch(
                color=palette[ci % len(palette)],
                label=f"{name} (nodes {a}-{b})"
            ))
        ax.legend(handles=handles, loc="upper left", fontsize=9,
                  framealpha=0.9, edgecolor="#D3D1C7")

    ax.set_title(f"Graph  ({N} nodes, {len(edges)} edges)",
                 fontsize=14, fontweight="bold")
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved graph plot: {out_path}")

# ─────────────────────────────────────────────
#  Heatmap saver
# ─────────────────────────────────────────────

def save_heatmap(matrix, title, filepath, N, clusters):
    """Save a heatmap with optional cluster boundary lines."""
    fig, ax = plt.subplots(figsize=(8, 7))

    vmax = matrix.max()
    vmin = matrix.min()

    im = ax.imshow(matrix, cmap="YlOrRd", aspect="equal",
                   vmin=vmin, vmax=vmax, interpolation="nearest")

    # cluster boundary lines and labels
    if clusters:
        for ci, (a, b, name) in enumerate(clusters):
            if b + 1 < N:
                boundary = b + 0.5
                ax.axhline(y=boundary, color="black", linewidth=2,
                           linestyle="--", alpha=0.7)
                ax.axvline(x=boundary, color="black", linewidth=2,
                           linestyle="--", alpha=0.7)

            center_x = (a + b) / 2
            ax.text(center_x, -1.5, f"{name}\n({a}-{b})", ha="center",
                    va="center", fontsize=8, fontweight="bold", color="#333")

    tick_size = max(4, 9 - N // 10)
    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xticklabels(range(N), fontsize=tick_size)
    ax.set_yticklabels(range(N), fontsize=tick_size)
    ax.set_xlabel("Node j", fontsize=11)
    ax.set_ylabel("Node i", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=30)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("Walk count", fontsize=10)

    plt.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filepath}")

# ─────────────────────────────────────────────
#  Console matrix printer
# ─────────────────────────────────────────────

def print_matrix(M, name, N, w=6):
    if N > 30:
        print(f"\n== {name} ==  ({N}x{N}, too large to print — see heatmap)")
        return
    print(f"\n== {name} ==")
    header = "     " + "".join(f" {j:>{w}} " for j in range(N))
    print(header)
    print("     " + "-" * (N * (w + 2)))
    for i in range(N):
        row = f"{i:3d} |" + "".join(f" {M[i][j]:>{w}} " for j in range(N))
        print(row)

# ═════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════

def main():
    # ensure inputs folder exists
    os.makedirs(INPUT_DIR, exist_ok=True)

    # list available input files
    files = sorted([f for f in os.listdir(INPUT_DIR)
                    if os.path.isfile(os.path.join(INPUT_DIR, f))
                    and not f.startswith(".")])

    if not files:
        print(f"\nNo input files found in:  {INPUT_DIR}/")
        print(f"Place a graph file there and re-run.")
        print(f"\nExpected format:")
        print(f"  nodes: 20")
        print(f"  cluster: 0-6 Cluster-A    (optional)")
        print(f"  edges:")
        print(f"  0 1")
        print(f"  0 2")
        print(f"  ...")
        sys.exit(1)

    # ── File selection ─────────────────────────
    print("\n" + "=" * 50)
    print("  Available input files")
    print("=" * 50)
    for i, f in enumerate(files, 1):
        print(f"  [{i}] {f}")
    print()

    choice = input("Enter file number or name: ").strip()

    # accept number or filename
    if choice.isdigit() and 1 <= int(choice) <= len(files):
        filename = files[int(choice) - 1]
    elif choice in files:
        filename = choice
    else:
        print(f"Invalid choice: '{choice}'")
        sys.exit(1)

    filepath = os.path.join(INPUT_DIR, filename)
    basename = os.path.splitext(filename)[0]

    print(f"\nReading: {filepath}")

    # ── Parse ──────────────────────────────────
    N, edges, clusters = parse_input(filepath)

    # ── Build adjacency matrix ─────────────────
    A = np.zeros((N, N), dtype=np.int64)
    for u, v in edges:
        A[u][v] = 1
        A[v][u] = 1

    # ── Print adjacency list ───────────────────
    print("\n" + "=" * 50)
    print(f"  ADJACENCY LIST  (N = {N}, E = {len(edges)})")
    print("=" * 50)
    if N <= 40:
        for i in range(N):
            neighbors = list(np.where(A[i] == 1)[0])
            print(f"  Node {i:3d} -> [{', '.join(str(x) for x in neighbors)}]")
    else:
        print(f"  (large graph — showing first 10 and last 5 nodes)")
        for i in list(range(10)) + ["..."] + list(range(N - 5, N)):
            if i == "...":
                print(f"  ...")
                continue
            neighbors = list(np.where(A[i] == 1)[0])
            print(f"  Node {i:3d} -> [{', '.join(str(x) for x in neighbors)}]")

    if clusters:
        print(f"\n  Clusters:")
        for a, b, name in clusters:
            print(f"    {name}: nodes {a} – {b}  ({b - a + 1} nodes)")

    # ── Get k from user ────────────────────────
    k = int(input(f"\nEnter k (compute A^1 through A^k): "))
    assert k >= 1, "k must be >= 1"

    # ── Create output directory ────────────────
    out_dir = os.path.join(OUTPUT_DIR, basename)
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nOutput directory: {out_dir}/")

    # ── Compute powers and cumulative sums ─────
    print("\nComputing matrix powers...")
    powers = [A.copy()]
    cumulative_sums = [A.copy()]

    power = A.copy()
    cum_sum = A.copy()

    for p in range(2, k + 1):
        power = power @ A
        cum_sum = cum_sum + power
        powers.append(power.copy())
        cumulative_sums.append(cum_sum.copy())
        print(f"  A^{p} computed  (max entry: {power.max()})")

    # ── Print matrices to console ──────────────
    print_matrix(A, "A^1  (Adjacency Matrix)", N)
    for p in range(1, k):
        print_matrix(powers[p], f"A^{p+1}", N)

    sum_label = f"SUM  ( A^1 + A^2 + ... + A^{k} )"
    print_matrix(cumulative_sums[-1], sum_label, N, w=8)

    # ── Save graph plot ────────────────────────
    print("\n--- Generating graph plot ---\n")
    save_graph_plot(N, edges, clusters,
                    os.path.join(out_dir, "graph.png"))

    # ── Save power heatmaps ────────────────────
    print("\n--- Generating power heatmaps ---\n")
    for p in range(k):
        save_heatmap(
            powers[p],
            f"$A^{{{p+1}}}$  (walks of length {p+1})",
            os.path.join(out_dir, f"power_A{p+1}.png"),
            N, clusters
        )

    # ── Save cumulative sum heatmaps ───────────
    print("\n--- Generating cumulative sum heatmaps ---\n")
    for s in range(k):
        if s == 0:
            terms = "$A^1$"
        elif s <= 3:
            terms = " + ".join(f"$A^{{{i+1}}}$" for i in range(s + 1))
        else:
            terms = f"$A^1 + A^2 + ... + A^{{{s+1}}}$"
        save_heatmap(
            cumulative_sums[s],
            f"Sum: {terms}",
            os.path.join(out_dir, f"cumsum_upto_A{s+1}.png"),
            N, clusters
        )

    # ── CSV export of final sum ────────────────
    csv_path = os.path.join(out_dir, f"sum_A1_to_A{k}.csv")
    np.savetxt(csv_path, cumulative_sums[-1], delimiter=",", fmt="%d")
    print(f"\n  CSV exported -> {csv_path}")

    print(f"\nDone. All outputs saved in:  {out_dir}/")


if __name__ == "__main__":
    main()
