#include <iostream>
#include <limits>
#include <vector>
#include <cassert>
#include <cmath>

using std::swap;
using std::cout;
using std::vector;

struct Node {
  Node() = default;
  Node(Node *left, Node *right, float length1, float length2)
      : childs{left, right}, branch_length{length1, length2} {}
  ~Node() = default;
  vector<Node *> childs;
  vector<float> branch_length;
};

class NJ {
public:
  NJ(float *_mat, int _num_seqs) : mat{_mat}, num_seqs{_num_seqs} {
    vector<Node *> nodes(num_seqs);
    for (int i = 0; i < num_seqs; ++i) {
      nodes[i] = new Node(nullptr, nullptr, 0.0f, 0.0f);
    }
    float *q = (float *)malloc(sizeof(float) * num_seqs * num_seqs);
    vector<float> s(num_seqs);
    int root_idx = -1;
    for (int remain = num_seqs; remain > 2; --remain) {
      // calculate sums over row
      for (int i = 0; i < num_seqs; ++i) {
        s[i] = 0.0f;
        for (int j = 0; j < num_seqs; ++j) {
          s[i] += isinf(mat[i * num_seqs + j]) ? 0.0f : mat[i * num_seqs + j];
        }
      }

      // calculate q matrix;
      for (int i = 0; i < num_seqs; ++i) {
        for (int j = 0; j < num_seqs; ++j) {
          q[i * num_seqs + j] =
              isinf(mat[i * num_seqs + j])
                  ? INFINITY
                  : (remain - 2) * mat[i * num_seqs + j] - s[i] - s[j];
        }
      }

      for (int i = 0; i < num_seqs; ++i) {
        for (int j = 0; j < num_seqs; ++j) {
          cout << q[i * num_seqs + j] << ",\t";
        }
        cout << "\n";
      }
      cout << "--------------------------------------\n";

      int idx = getMinIdx(q, num_seqs * num_seqs);
      int idx1 = idx / num_seqs;
      int idx2 = idx % num_seqs;
      if (idx1 > idx2) {
        swap(idx1, idx2);
      }
      cout << idx1 << ", " << idx2 << "\n";
      float length = mat[idx1 * num_seqs + idx2];

      float branch_length1 =
          length / 2 + (s[idx1] - s[idx2]) / ((remain - 2) * 2);
      float branch_length2 = length - branch_length1;
      root = new Node(nodes[idx1], nodes[idx2], branch_length1, branch_length2);
      update(idx1, idx2);

      for (int i = 0; i < num_seqs; ++i) {
        for (int j = 0; j < num_seqs; ++j) {
          cout << mat[num_seqs * i + j] << ",\t";
        }
        cout << "\n";
      }
      cout << "--------------------------------------\n";
      root_idx = idx1;
      nodes[idx1] = root;
      nodes[idx2] = nullptr;
    }

    Node *other_root = nullptr;
    int other_root_idx = -1;
    for (int i = 0; i < num_seqs; ++i) {
      if (nodes[i] != nullptr && nodes[i] != root) {
        other_root = nodes[i];
        other_root_idx = i;
        break;
      }
    }
    root->childs.push_back(other_root);
    root->branch_length.push_back(mat[root_idx * num_seqs + other_root_idx]);

    // Free memory
    free(q);
  }

  void print() {
    print(root);
    cout << "\n";
  }

private:
  float *mat;
  int num_seqs;
  Node *root = nullptr;

  int getMinIdx(float *a, int n) {
    float val = INFINITY;
    int idx = -1;
    for (int i = 0; i < n; ++i) {
      if (a[i] < val) {
        idx = i;
        val = a[i];
      }
    }
    return idx;
  }

  void update(int idx1, int idx2) {
    float d = mat[num_seqs * idx1 + idx2];
    for (int i = 0; i < num_seqs; ++i) {
      float val = mat[num_seqs * idx1 + i];
      if (isinf(val)) {
        continue;
      }
      float new_val = (val + mat[num_seqs * idx2 + i] - d) / 2;
      mat[num_seqs * idx1 + i] = new_val;
      mat[num_seqs * idx2 + i] = INFINITY;
      mat[num_seqs * i + idx1] = new_val;
      mat[num_seqs * i + idx2] = INFINITY;
    }
  }

  void cleanup(Node *node) {
    if (node == nullptr) {
      return;
    }
    int num_childs = node->childs.size();
    for (int i = 0; i < num_childs; ++i) {
      cleanup(node->childs[i]);
    }
    delete node;
  }

  void print(Node *node) {
    int num_childs = node->childs.size();
    // Reach the leaf
    if (num_childs == 2 && node->childs[0] == nullptr &&
        node->childs[1] == nullptr) {
      return;
    }
    cout << "(";
    for (int i = 0; i < num_childs - 1; ++i) {
      print(node->childs[i]);
      cout << ": " << node->branch_length[i] << ", ";
    }
    print(node->childs[num_childs - 1]);
    cout << ": " << node->branch_length[num_childs - 1] << ")";
  }
};

int main() {
  const int num_seqs = 5;
  float a[num_seqs][num_seqs]{{INFINITY, 5.0f, 9.0f, 9.0f, 8.0f},
                              {5.0f, INFINITY, 10.0f, 10.0f, 9.0f},
                              {9.0f, 10.0f, INFINITY, 8.0f, 7.0f},
                              {9.0f, 10.0f, 8.0f, INFINITY, 3.0f},
                              {8.0f, 9.0f, 7.0f, 3.0f, INFINITY}};

  assert(num_seqs > 2);
  NJ nj((float *)a, num_seqs);
  nj.print();
  return 0;
}
