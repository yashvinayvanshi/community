#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <cassert>
#include <iomanip>

using namespace std;

// ============================================================
// SECTION 1: MATRIX UTILITIES
// We work with dense matrices for small graphs.
// For large graphs, replace with sparse representations.
// ============================================================

using Matrix = vector<vector<double>>;

/**
 * Multiply two dense matrices A (m x n) and B (n x p).
 * Returns C (m x p) where C = A * B.
 */
Matrix matmul(const Matrix& A, const Matrix& B) {
    int m = A.size(), n = B.size(), p = B[0].size();
    Matrix C(m, vector<double>(p, 0.0));
    for (int i = 0; i < m; i++)
        for (int k = 0; k < n; k++)
            for (int j = 0; j < p; j++)
                C[i][j] += A[i][k] * B[k][j];
    return C;
}

/**
 * Add two matrices of the same dimensions.
 */
Matrix matadd(const Matrix& A, const Matrix& B) {
    int n = A.size(), m = A[0].size();
    Matrix C(n, vector<double>(m));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            C[i][j] = A[i][j] + B[i][j];
    return C;
}

/**
 * Build the adjacency matrix from an adjacency list.
 */
Matrix build_adjacency_matrix(const vector<vector<int>>& adj) {
    int n = adj.size();
    Matrix A(n, vector<double>(n, 0.0));
    for (int u = 0; u < n; u++)
        for (int v : adj[u])
            A[u][v] = 1.0;
    return A;
}

// ============================================================
// SECTION 2: COMPUTE CUMULATIVE WALK-MASS MATRIX
// M = A + A^2 + A^3 + ... + A^k
// This matrix encodes all walks of length 1 through k.
// ============================================================

/**
 * Compute M = A + A^2 + ... + A^k by repeated multiplication.
 * A_power tracks the current power of A.
 * We accumulate into M at each step.
 */
Matrix compute_walk_mass_matrix(const Matrix& A, int k) {
    int n = A.size();
    Matrix M(n, vector<double>(n, 0.0));
    Matrix A_power = A; // starts as A^1

    for (int j = 1; j <= k; j++) {
        M = matadd(M, A_power);
        if (j < k)
            A_power = matmul(A_power, A);
    }
    return M;
}

// ============================================================
// SECTION 3: EIGENVECTOR COMPUTATION
// We use power iteration with deflation to extract
// the top-ell eigenvectors of M.
// These eigenvectors capture the dominant community structure.
// ============================================================

/**
 * Compute the L2 norm of a vector.
 */
double vec_norm(const vector<double>& v) {
    double s = 0.0;
    for (double x : v) s += x * x;
    return sqrt(s);
}

/**
 * Dot product of two vectors.
 */
double vec_dot(const vector<double>& a, const vector<double>& b) {
    double s = 0.0;
    for (int i = 0; i < (int)a.size(); i++) s += a[i] * b[i];
    return s;
}

/**
 * Multiply matrix M by vector v: result = M * v.
 */
vector<double> matvec(const Matrix& M, const vector<double>& v) {
    int n = M.size();
    vector<double> result(n, 0.0);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            result[i] += M[i][j] * v[j];
    return result;
}

/**
 * Extract top-ell eigenpairs of symmetric matrix M using
 * power iteration with deflation.
 *
 * Power iteration finds the dominant eigenvector by repeatedly
 * multiplying a random vector by M. After finding each eigenvector,
 * we "deflate" M by subtracting the contribution of that eigenvector,
 * so the next iteration finds the next one.
 *
 * Returns: pair of (eigenvalues, eigenvectors)
 *   eigenvalues[i] = i-th largest eigenvalue
 *   eigenvectors[i] = corresponding eigenvector
 */
pair<vector<double>, vector<vector<double>>>
compute_top_eigenpairs(Matrix M, int ell, int max_iter = 1000, double tol = 1e-10) {
    int n = M.size();
    vector<double> eigenvalues;
    vector<vector<double>> eigenvectors;
    mt19937 rng(42);
    normal_distribution<double> dist(0.0, 1.0);

    for (int e = 0; e < ell; e++) {
        // Initialize random vector
        vector<double> v(n);
        for (int i = 0; i < n; i++) v[i] = dist(rng);
        double norm = vec_norm(v);
        for (int i = 0; i < n; i++) v[i] /= norm;

        double eigenvalue = 0.0;

        // Power iteration: v <- M*v / ||M*v||
        for (int iter = 0; iter < max_iter; iter++) {
            vector<double> Mv = matvec(M, v);
            double new_eigenvalue = vec_dot(v, Mv);
            norm = vec_norm(Mv);
            if (norm < 1e-15) break;
            for (int i = 0; i < n; i++) Mv[i] /= norm;

            // Check convergence
            if (abs(new_eigenvalue - eigenvalue) < tol) {
                eigenvalue = new_eigenvalue;
                v = Mv;
                break;
            }
            eigenvalue = new_eigenvalue;
            v = Mv;
        }

        eigenvalues.push_back(eigenvalue);
        eigenvectors.push_back(v);

        // Deflation: remove contribution of found eigenvector
        // M <- M - eigenvalue * v * v^T
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                M[i][j] -= eigenvalue * v[i] * v[j];
    }

    return {eigenvalues, eigenvectors};
}

// ============================================================
// SECTION 4: SPECTRAL EMBEDDING
// Each vertex gets a position in ell-dimensional space.
// The coordinates are weighted by sqrt(mu_i) so that
// dot products in embedding space approximate walk-mass.
//
// x_v = (sqrt(mu_1)*q_1(v), sqrt(mu_2)*q_2(v), ..., sqrt(mu_ell)*q_ell(v))
//
// Then: M_{vu} ≈ x_v · x_u
// ============================================================

/**
 * Build the spectral embedding for all vertices.
 * embedding[v] = position of vertex v in R^ell.
 */
vector<vector<double>> build_spectral_embedding(
    const vector<double>& eigenvalues,
    const vector<vector<double>>& eigenvectors
) {
    int n = eigenvectors[0].size();
    int ell = eigenvalues.size();
    vector<vector<double>> embedding(n, vector<double>(ell));

    for (int i = 0; i < ell; i++) {
        // Weight by sqrt of eigenvalue magnitude
        // Use abs because some eigenvalues might be negative
        double weight = sqrt(max(0.0, eigenvalues[i]));
        for (int v = 0; v < n; v++) {
            embedding[v][i] = weight * eigenvectors[i][v];
        }
    }
    return embedding;
}

// ============================================================
// SECTION 5: K-MEANS CLUSTERING IN EMBEDDING SPACE
// After embedding, vertices in the same community are close
// together. K-means finds these clusters.
// ============================================================

/**
 * Euclidean distance squared between two points.
 */
double dist_sq(const vector<double>& a, const vector<double>& b) {
    double s = 0.0;
    for (int i = 0; i < (int)a.size(); i++) {
        double d = a[i] - b[i];
        s += d * d;
    }
    return s;
}

/**
 * K-means clustering on the spectral embeddings.
 * Returns assignment[v] = cluster index for vertex v.
 */
vector<int> kmeans(
    const vector<vector<double>>& points,
    int num_clusters,
    int max_iter = 100
) {
    int n = points.size();
    int dim = points[0].size();
    vector<int> assignment(n, 0);
    vector<vector<double>> centers(num_clusters, vector<double>(dim, 0.0));

    // Initialize centers using k-means++ strategy
    mt19937 rng(42);
    uniform_int_distribution<int> pick(0, n - 1);
    centers[0] = points[pick(rng)];

    for (int c = 1; c < num_clusters; c++) {
        // For each point, find distance to nearest existing center
        vector<double> dists(n);
        for (int i = 0; i < n; i++) {
            double min_d = 1e18;
            for (int j = 0; j < c; j++)
                min_d = min(min_d, dist_sq(points[i], centers[j]));
            dists[i] = min_d;
        }
        // Pick next center with probability proportional to distance squared
        discrete_distribution<int> weighted(dists.begin(), dists.end());
        centers[c] = points[weighted(rng)];
    }

    // Iterate: assign points to nearest center, then recompute centers
    for (int iter = 0; iter < max_iter; iter++) {
        bool changed = false;

        // Assign each point to nearest center
        for (int i = 0; i < n; i++) {
            int best = 0;
            double best_d = dist_sq(points[i], centers[0]);
            for (int c = 1; c < num_clusters; c++) {
                double d = dist_sq(points[i], centers[c]);
                if (d < best_d) { best_d = d; best = c; }
            }
            if (assignment[i] != best) { assignment[i] = best; changed = true; }
        }

        if (!changed) break;

        // Recompute centers as mean of assigned points
        vector<int> counts(num_clusters, 0);
        for (auto& c : centers) fill(c.begin(), c.end(), 0.0);
        for (int i = 0; i < n; i++) {
            counts[assignment[i]]++;
            for (int d = 0; d < dim; d++)
                centers[assignment[i]][d] += points[i][d];
        }
        for (int c = 0; c < num_clusters; c++)
            if (counts[c] > 0)
                for (int d = 0; d < dim; d++)
                    centers[c][d] /= counts[c];
    }

    return assignment;
}

// ============================================================
// SECTION 6: COMMUNITY EXTRACTION AND VERIFICATION
// From each cluster, extract the top-s vertices closest to
// the cluster center. Then verify the exact walk-trapping
// condition using the actual matrix M.
//
// The spectral method gives candidates.
// Verification checks the actual definition:
//   for all v in S: sum_{u in S} M[v][u] > r_v / 2
// ============================================================

/**
 * Compute the trapping ratio for vertex v with respect to set S.
 * rho(v, S) = sum_{u in S} M[v][u] / sum_{u in V} M[v][u]
 *
 * If rho > 0.5 for ALL v in S, then S is a community.
 */
double trapping_ratio(int v, const vector<int>& S, const Matrix& M) {
    int n = M.size();
    double internal = 0.0, total = 0.0;
    for (int u = 0; u < n; u++) {
        total += M[v][u];
    }
    for (int u : S) {
        internal += M[v][u];
    }
    if (total < 1e-15) return 0.0;
    return internal / total;
}

/**
 * Community quality score: the minimum trapping ratio among all members.
 * Q(S) = min_{v in S} rho(v, S)
 *
 * Higher Q means every member is strongly trapped.
 * Q > 0.5 means S satisfies the community definition.
 */
double community_quality(const vector<int>& S, const Matrix& M) {
    double min_rho = 1.0;
    for (int v : S) {
        double rho = trapping_ratio(v, S, M);
        min_rho = min(min_rho, rho);
    }
    return min_rho;
}

/**
 * A community with its member vertices and quality score.
 */
struct Community {
    vector<int> members;
    double quality; // min trapping ratio
};

/**
 * Extract a community of size s from a cluster.
 *
 * Strategy: find the cluster center, then take the s vertices
 * closest to it. This gives the tightest core of the cluster,
 * which should have the highest trapping ratio.
 */
Community extract_community_from_cluster(
    const vector<int>& cluster_members,
    int community_size,
    const vector<vector<double>>& embedding,
    const Matrix& M
) {
    if ((int)cluster_members.size() < community_size) {
        // Cluster too small, return what we have
        Community c;
        c.members = cluster_members;
        c.quality = community_quality(cluster_members, M);
        return c;
    }

    int dim = embedding[0].size();

    // Compute cluster center
    vector<double> center(dim, 0.0);
    for (int v : cluster_members)
        for (int d = 0; d < dim; d++)
            center[d] += embedding[v][d];
    for (int d = 0; d < dim; d++)
        center[d] /= cluster_members.size();

    // Sort members by distance to center (closest first)
    vector<pair<double, int>> dists;
    for (int v : cluster_members)
        dists.push_back({dist_sq(embedding[v], center), v});
    sort(dists.begin(), dists.end());

    // Take the closest s vertices as the initial community
    vector<int> S;
    for (int i = 0; i < community_size && i < (int)dists.size(); i++)
        S.push_back(dists[i].second);

    // Local swap refinement:
    // Try swapping each member with each non-member in the cluster.
    // Accept the swap if it improves the quality score.
    bool improved = true;
    while (improved) {
        improved = false;
        for (int i = 0; i < (int)S.size(); i++) {
            for (int j = community_size; j < (int)dists.size(); j++) {
                // Try swapping S[i] with dists[j].second
                vector<int> S_new = S;
                S_new[i] = dists[j].second;

                double old_q = community_quality(S, M);
                double new_q = community_quality(S_new, M);

                if (new_q > old_q) {
                    S = S_new;
                    improved = true;
                }
            }
        }
    }

    Community c;
    c.members = S;
    sort(c.members.begin(), c.members.end());
    c.quality = community_quality(S, M);
    return c;
}

// ============================================================
// SECTION 7: TOP-LEVEL ALGORITHM
// Ties everything together:
//   1. Compute walk-mass matrix M
//   2. Extract top eigenvectors
//   3. Build spectral embedding
//   4. Cluster in embedding space
//   5. Extract communities from each cluster
//   6. Verify against exact definition
//   7. Return top-t communities ranked by quality
// ============================================================

/**
 * Find the top-t communities of size s using walks of length k.
 *
 * @param adj           Adjacency list of the graph
 * @param walk_length   Walk length k (controls resolution)
 * @param com_size      Desired community size s
 * @param num_top       Number of top communities to return
 * @return              Vector of communities sorted by quality
 */
vector<Community> find_top_communities(
    const vector<vector<int>>& adj,
    int walk_length,
    int com_size,
    int num_top
) {
    int n = adj.size();

    // Step 1: Build adjacency matrix and compute walk-mass matrix
    cout << "Step 1: Computing walk-mass matrix M = A + A^2 + ... + A^"
         << walk_length << endl;
    Matrix A = build_adjacency_matrix(adj);
    Matrix M = compute_walk_mass_matrix(A, walk_length);

    // Step 2: Compute top eigenvectors
    // We use more eigenvectors than communities for better separation
    int num_eigenvectors = min(n, num_top + 5);
    cout << "Step 2: Computing top " << num_eigenvectors
         << " eigenvectors via power iteration" << endl;
    auto [eigenvalues, eigenvectors] = compute_top_eigenpairs(M, num_eigenvectors);

    cout << "  Eigenvalues: ";
    for (double ev : eigenvalues) cout << fixed << setprecision(1) << ev << " ";
    cout << endl;

    // Step 3: Build spectral embedding
    cout << "Step 3: Building spectral embedding in R^"
         << num_eigenvectors << endl;
    auto embedding = build_spectral_embedding(eigenvalues, eigenvectors);

    // Step 4: Cluster in embedding space
    // Use more clusters than needed, then keep the best ones
    int num_clusters = min(n, max(num_top, 3));
    cout << "Step 4: Running k-means with " << num_clusters << " clusters" << endl;
    auto assignment = kmeans(embedding, num_clusters);

    // Group vertices by cluster
    vector<vector<int>> clusters(num_clusters);
    for (int v = 0; v < n; v++)
        clusters[assignment[v]].push_back(v);

    // Step 5: Extract a community from each cluster
    cout << "Step 5: Extracting and refining communities of size "
         << com_size << endl;
    vector<Community> all_communities;
    for (int c = 0; c < num_clusters; c++) {
        if ((int)clusters[c].size() < com_size) continue;
        Community com = extract_community_from_cluster(
            clusters[c], com_size, embedding, M
        );
        all_communities.push_back(com);
    }

    // Step 6: Sort by quality and return top-t
    sort(all_communities.begin(), all_communities.end(),
         [](const Community& a, const Community& b) {
             return a.quality > b.quality;
         });

    // Keep only top-t
    vector<Community> top;
    for (int i = 0; i < num_top && i < (int)all_communities.size(); i++)
        top.push_back(all_communities[i]);

    return top;
}

// ============================================================
// SECTION 8: GRAPH CONSTRUCTION AND MAIN
// Build a test graph with 15 vertices and 3 clear clusters:
//
//   Cluster A: {0, 1, 2, 3, 4}    - densely connected
//   Cluster B: {5, 6, 7, 8, 9}    - densely connected
//   Cluster C: {10, 11, 12, 13, 14} - densely connected
//
//   A few weak edges between clusters to make it connected.
//
//    [0]-[1]-[2]       [5]-[6]-[7]       [10]-[11]-[12]
//     |\ /|\ /|         |\ /|\ /|          |\ / |\ / |
//    [3]--[4]           [8]--[9]           [13]--[14]
//         |                  |                    |
//         +------[bridge]----+-------[bridge]-----+
// ============================================================

/**
 * Add an undirected edge between u and v.
 */
void add_edge(vector<vector<int>>& adj, int u, int v) {
    adj[u].push_back(v);
    adj[v].push_back(u);
}

/**
 * Build a test graph with 3 clusters of 5 vertices each.
 * Internal edges are dense, cross-cluster edges are sparse.
 */
vector<vector<int>> build_test_graph() {
    int n = 16;
    vector<vector<int>> adj(n);

    // check 3, 5, 3 vs 3, 6, 3
    // Cluster A: vertices 0-4 (near-clique)
    add_edge(adj, 0, 1);
    add_edge(adj, 0, 2);
    add_edge(adj, 0, 3);
    add_edge(adj, 0, 4);
    add_edge(adj, 1, 2);
    add_edge(adj, 1, 3);
    add_edge(adj, 1, 4);
    add_edge(adj, 2, 3);
    add_edge(adj, 2, 4);
    add_edge(adj, 3, 4);

    // Cluster B: vertices 5-9 (near-clique)
    add_edge(adj, 5, 6);
    add_edge(adj, 5, 7);
    add_edge(adj, 5, 8);
    add_edge(adj, 5, 9);
    add_edge(adj, 6, 7);
    add_edge(adj, 6, 8);
    add_edge(adj, 6, 9);
    add_edge(adj, 7, 8);
    add_edge(adj, 7, 9);
    add_edge(adj, 8, 9);

    // Cluster C: vertices 10-14 (near-clique)
    add_edge(adj, 10, 11);
    add_edge(adj, 10, 12);
    add_edge(adj, 10, 13);
    add_edge(adj, 10, 14);
    add_edge(adj, 11, 12);
    add_edge(adj, 11, 13);
    add_edge(adj, 11, 14);
    add_edge(adj, 12, 13);
    add_edge(adj, 12, 14);
    add_edge(adj, 13, 14);
    add_edge(adj, 14, 15);

    // Sparse cross-cluster bridges
    add_edge(adj, 4, 5);   // A -- B
    add_edge(adj, 9, 10);  // B -- C

    return adj;

    // two 5-clqiues sharing an edge (overlapping communities example)    
    // int n = 8;
    // vector<vector<int>> adj(n);

    // // Clique A: vertices 0-4
    // add_edge(adj, 0, 1);
    // add_edge(adj, 0, 2);
    // add_edge(adj, 0, 3);
    // add_edge(adj, 0, 4);
    // add_edge(adj, 1, 2);
    // add_edge(adj, 1, 3);
    // add_edge(adj, 1, 4);
    // add_edge(adj, 2, 3);
    // add_edge(adj, 2, 4);
    // add_edge(adj, 3, 4);

    // // Clique B: vertices 3-7 (shares edge 3-4 with A)
    // add_edge(adj, 3, 5);
    // add_edge(adj, 3, 6);
    // add_edge(adj, 3, 7);
    // add_edge(adj, 4, 5);
    // add_edge(adj, 4, 6);
    // add_edge(adj, 4, 7);
    // add_edge(adj, 5, 6);
    // add_edge(adj, 5, 7);
    // add_edge(adj, 6, 7);

    // return adj;
}

/**
 * Print the adjacency list for visualization.
 */
void print_graph(const vector<vector<int>>& adj) {
    cout << "Graph (" << adj.size() << " vertices):" << endl;
    for (int u = 0; u < (int)adj.size(); u++) {
        cout << "  " << u << " -> ";
        for (int v : adj[u]) cout << v << " ";
        cout << endl;
    }
}

int main() {
    cout << "============================================" << endl;
    cout << "  Walk-Trapping Community Detection" << endl;
    cout << "  Spectral Method Implementation" << endl;
    cout << "============================================" << endl << endl;

    /*
        NOTES:
        Do we need to normalise the walk mass with number of paths of degree k ?
    */

    // Build test graph: 15 vertices, 3 clusters of 5
    vector<vector<int>> adj = build_test_graph();
    print_graph(adj);
    cout << endl;

    // Parameters
    int walk_length = 1;      // k: walk length (resolution)
    int community_size = 3;   // s: desired community size
    int num_top = 10;          // t: number of top communities to find

    cout << "Parameters:" << endl;
    cout << "  Walk length k = " << walk_length << endl;
    cout << "  Community size s = " << community_size << endl;
    cout << "  Top communities t = " << num_top << endl;
    cout << endl;

    // Run the algorithm
    auto communities = find_top_communities(adj, walk_length, community_size, num_top);

    // Report results
    cout << endl;
    cout << "============================================" << endl;
    cout << "  RESULTS: Top " << communities.size() << " communities" << endl;
    cout << "============================================" << endl << endl;

    // Precompute M for verification printing
    Matrix A_verify = build_adjacency_matrix(adj);
    Matrix M_verify = compute_walk_mass_matrix(A_verify, walk_length);

    for (int i = 0; i < (int)communities.size(); i++) {
        const auto& com = communities[i];
        cout << "Community #" << (i + 1) << ":" << endl;
        cout << "  Members: { ";
        for (int v : com.members) cout << v << " ";
        cout << "}" << endl;
        cout << "  Quality Q(S) = " << fixed << setprecision(4) << com.quality << endl;
        cout << "  Satisfies definition: "
             << (com.quality > 0.5 ? "YES" : "NO") << endl;

        // Print per-vertex trapping ratios
        cout << "  Per-vertex trapping ratios:" << endl;
        for (int v : com.members) {
            double rho = trapping_ratio(v, com.members, M_verify);
            cout << "    vertex " << v << ": rho = " << fixed
                 << setprecision(4) << rho
                 << (rho > 0.5 ? " (trapped)" : " (leaking)") << endl;
        }
        cout << endl;
    }

    return 0;
}
