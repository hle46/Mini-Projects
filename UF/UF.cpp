#include <iostream>
#include <fstream>
#include <vector>

/* Union Find data structure
 * Weighted quick union with path compression
 * The class represents a union-find data structure.
 * It supports the union and find operations, along with methods for determining
 * whether two sites are in the same component and the total number of components
 * 
 */
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
  std::vector<int> parent;
  std::vector<int> sz;
  int count;
};

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cout << "Usage: " << argv[0] << " file" << "\n";
    exit(-1);
  }
  std::ifstream ifs(argv[1], std::ifstream::in);
  if (!ifs.is_open()) {
    std::cout << "Cannot open file \n";
    exit(-1);
  }
  int n, p, q;
  ifs >> n;
  UF uf(n);
  while ((ifs >> p) && (ifs >> q)) {
    if (uf.connected(p, q)) continue;
    uf.join(p, q);
  }
  std::cout << uf.getCount() << " components\n";
  return 0;
}
