#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <string>
#include <numeric>
#include <filesystem>

using namespace std;

// ─────────────────────────────────────────────
//  Matrix helpers
// ─────────────────────────────────────────────

using Matrix = vector<vector<long long>>;

Matrix makeZero(int n) { return Matrix(n, vector<long long>(n, 0)); }

Matrix multiply(const Matrix &A, const Matrix &B, int n)
{
    Matrix C = makeZero(n);
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < n; ++k)
            if (A[i][k])
                for (int j = 0; j < n; ++j)
                    C[i][j] += A[i][k] * B[k][j];
    return C;
}

Matrix add(const Matrix &A, const Matrix &B, int n)
{
    Matrix C = makeZero(n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            C[i][j] = A[i][j] + B[i][j];
    return C;
}

// ─────────────────────────────────────────────
//  Pretty printer
// ─────────────────────────────────────────────

void printMatrix(const Matrix &M, const string &name, int n, int w = 6)
{
    cout << "\n╔══ " << name << " ══╗\n";

    // Column header
    cout << "     ";
    for (int j = 0; j < n; ++j)
        cout << " " << setw(w) << j << " ";
    cout << "\n     ";
    for (int j = 0; j < n; ++j)
        cout << string(w + 2, '-');
    cout << "\n";

    // Rows
    for (int i = 0; i < n; ++i)
    {
        cout << setw(3) << i << " |";
        for (int j = 0; j < n; ++j)
            cout << " " << setw(w) << M[i][j] << " ";
        cout << "\n";
    }
}

// ─────────────────────────────────────────────
//  CSV export
// ─────────────────────────────────────────────

void exportCSV(const Matrix &M, int n, const string &path)
{
    ofstream f(path);
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            if (j) f << ",";
            f << M[i][j];
        }
        f << "\n";
    }
    cout << "\n✓ CSV exported → " << path << "\n";
}

// ─────────────────────────────────────────────
//  Graph definition  (20 nodes, 3 clusters)
// ─────────────────────────────────────────────
//
//  Node labelling (see diagram):
//
//  TOP-LEFT cluster  : 0 – 6
//      0                 (top-centre)
//      1,2               (upper-left, upper-right)
//      3,4               (middle-left, middle-right)
//      5,6               (lower-left, lower-right)
//
//  RIGHT cluster     : 7 – 12
//      7,8               (upper-left, upper-right)
//      9,10              (middle-left, middle-right)
//      11,12             (lower-left, lower-right)
//
//  BOTTOM cluster    : 13 – 19
//      13                (top-centre)
//      14,15             (upper-left, upper-right)
//      16,17             (middle-left, middle-right)
//      18,19             (lower-left, lower-right)
//
//  Inter-cluster (light-gray) edges:
//      6 -- 7   (top-left ↔ right)
//      6 -- 13  (top-left ↔ bottom)
//     12 -- 13  (right    ↔ bottom)

int main()
{
    // ── adjacency list ──────────────────────────────────────────────────────
    const int N = 20;

    vector<vector<int>> adj(N);

    auto addEdge = [&](int u, int v)
    {
        adj[u].push_back(v);
        adj[v].push_back(u);
    };

    // ── TOP-LEFT cluster (0-6), nearly complete ──────────────────────────
    addEdge(0, 1);
    addEdge(0, 2);
    addEdge(0, 3);
    addEdge(0, 4);
    addEdge(1, 2);
    addEdge(1, 4);
    addEdge(1, 6);
    addEdge(2, 3);
    addEdge(2, 5);
    addEdge(3, 4);
    addEdge(3, 5);
    addEdge(3, 6);
    addEdge(4, 5);
    addEdge(4, 6);
    addEdge(5, 6);

    // ── RIGHT cluster (7-12), nearly complete ────────────────────────────
    addEdge(7, 8);
    addEdge(7, 9);
    addEdge(7, 10);
    addEdge(7, 11);
    addEdge(8, 9);
    addEdge(8, 11);
    addEdge(8, 12);
    addEdge(9, 10);
    addEdge(9, 11);
    addEdge(9, 12);
    addEdge(10, 11);
    addEdge(10, 12);
    addEdge(11, 12);

    // ── BOTTOM cluster (13-19), nearly complete ──────────────────────────
    addEdge(13, 14);
    addEdge(13, 15);
    addEdge(13, 16);
    addEdge(13, 17);
    addEdge(14, 15);
    addEdge(14, 16);
    addEdge(14, 18);
    addEdge(15, 16);
    addEdge(15, 17);
    addEdge(15, 19);
    addEdge(16, 17);
    addEdge(16, 18);
    addEdge(17, 18);
    addEdge(17, 19);
    addEdge(18, 19);

    // ── INTER-CLUSTER (light-gray) edges ─────────────────────────────────
    addEdge(6, 7);
    addEdge(6, 13);
    addEdge(12, 13);

    // ── Print adjacency list ─────────────────────────────────────────────
    cout << "╔══════════════════════════════════════╗\n";
    cout << "║        ADJACENCY LIST  (N = " << N << ")       ║\n";
    cout << "╚══════════════════════════════════════╝\n";
    for (int i = 0; i < N; ++i)
    {
        cout << "  Node " << setw(2) << i << " → [";
        for (int idx = 0; idx < (int)adj[i].size(); ++idx)
        {
            cout << adj[i][idx];
            if (idx + 1 < (int)adj[i].size())
                cout << ", ";
        }
        cout << "]\n";
    }

    // ── Build adjacency matrix A ─────────────────────────────────────────
    Matrix A = makeZero(N);
    for (int i = 0; i < N; ++i)
        for (int j : adj[i])
            A[i][j] = 1;

    // ── Get k from user ──────────────────────────────────────────────────
    int k;
    cout << "\nEnter k (compute A^1 through A^k): ";
    cin >> k;

    if (k < 1)
    {
        cerr << "k must be >= 1.\n";
        return 1;
    }

    // ── Compute and display A^1 … A^k and their sum ──────────────────────
    printMatrix(A, "A^1  (Adjacency Matrix)", N);

    Matrix power = A;
    Matrix sumM = A; // starts as A^1

    for (int p = 2; p <= k; ++p)
    {
        power = multiply(power, A, N);
        string label = "A^" + to_string(p);
        printMatrix(power, label, N);

        sumM = add(sumM, power, N);
    }

    // ── Final sum ────────────────────────────────────────────────────────
    string sumLabel = "SUM  ( A^1 + A^2 + ... + A^" + to_string(k) + " )";
    printMatrix(sumM, sumLabel, N, 8);

    // ── Export sum matrix as CSV next to this source file ────────────────
    namespace fs = std::filesystem;
    fs::path srcDir = fs::path(__FILE__).parent_path();
    string csvName = "sum_A1_to_A" + to_string(k) + ".csv";
    exportCSV(sumM, N, (srcDir / csvName).string());

    cout << "✓ Done.\n";
    return 0;
}