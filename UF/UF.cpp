/******************************************************************************
 *  Compilation: make
 *  Execution: UF < input.txt
 *  Data files: tinyUF.txt mediumUF.txt largeUF.txt
 *  Implementation of Union Find data structure
 *  This implementation is based on weighted quick union with path compression
 *  This implementation supports the union and find operations, along with
 *  methods for determining whether two sites are in the same component and
 *  the total number of components.
 *  % UF < tinyUF.txt
 *  4 3
 *  3 8
 *  6 5
 *  9 4
 *  2 1
 *  5 0
 *  7 2
 *  6 1
 *  2 components
 *  History:
 *  11-29-15: adapted from http://algs4.cs.princeton.edu/15uf/UF.java.html
 ******************************************************************************/
#include <iostream>
#include <fstream>
#include <vector>

using std::vector;
using std::cout;
using std::cin;

class UF {
public:
  UF(int n): parent(n), sz(n), count{n} {
    for (int i = 0; i < n; ++i) {
      parent[i] = i;
      sz[i] = i;
    }
  }
  int getCount() {
    return count;
  }
  int find(int p) {
    while (p != parent[p]) {
      parent[p] = parent[parent[p]]; // path compression by halving
      p = parent[p];
    }
    return p;
  }
  bool connected(int p, int q) {
    return find(p) == find(q);
  }
  void join(int p, int q) {
    int root_p = find(p);
    int root_q = find(q);
    if (root_p == root_q) { return; }

    // make smaller root point to larger one
    if (sz[root_p] < sz[root_q]) {
      parent[root_p] = root_q;
      sz[root_q] += sz[root_p];
    } else {
      parent[root_q] = root_p;
      sz[root_p] += sz[root_q];
    }
    --count;
  }
private:
  vector<int> parent;
  vector<int> sz;
  int count;
};

int main() {
  int n, p, q;
  cin >> n;
  UF uf(n);
  while ((cin >> p) && (cin >> q)) {
    if (uf.connected(p, q)) continue;
    uf.join(p, q);
    cout << p << " " << q << "\n";
  }
  cout << uf.getCount() << " components\n";
  return 0;
}
