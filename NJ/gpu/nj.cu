#include <iostream>
#include <limits>
#include <vector>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>

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

__global__ void sum(float *input, int n, float *output_val) {
  __shared__ float smem_val[BLOCK_SIZE];
  __shared__ int smem_idx[BLOCK_SIZE];

  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int i = tx + bx * BLOCK_SIZE * 8;

  float s = 0.0f;

  if (i < n) {
    float a1, a2, a3, a4, a5, a6, a7, a8;
    a1 = input[i];

    a2 = (i + BLOCK_SIZE) < n ? input[i + BLOCK_SIZE] : 0.0f;

    a3 = (i + 2 * BLOCK_SIZE) < n ? input[i + 2 * BLOCK_SIZE] : 0.0f;

    a4 = (i + 3 * BLOCK_SIZE) < n ? input[i + 3 * BLOCK_SIZE] : 0.0f;

    a5 = (i + 4 * BLOCK_SIZE) < n ? input[i + 4 * BLOCK_SIZE] : 0.0f;

    a6 = (i + 5 * BLOCK_SIZE) < n ? input[i + 5 * BLOCK_SIZE] : 0.0f;

    a7 = (i + 6 * BLOCK_SIZE) < n ? input[i + 6 * BLOCK_SIZE] : 0.0f;

    a8 = (i + 7 * BLOCK_SIZE) < n ? input[i + 7 * BLOCK_SIZE] : 0.0f;

    s = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8;
  }

  smem_val[tx] = s;
  __syncthreads();

  // in-place reduction in shared memory
  if (blockDim.x >= 1024 && tx < 512) {
    smem_val[tx] = s = s + smem_val[tx + 512];
  }
  __syncthreads();

  if (blockDim.x >= 512 && tx < 256) {
    smem_val[tx] = s = s + smem_val[tx + 256];
  }
  __syncthreads();

  if (blockDim.x >= 256 && tx < 128) {
    smem_val[tx] = s = s + smem_val[tx + 128];
  }
  __syncthreads();

  if (blockDim.x >= 128 && tx < 64) {
    smem_val[tx] = s = s + smem_val[tx + 64];
  }
  __syncthreads();

  // unrolling warp
  if (tx < 32) {
    volatile float *vsmem_val = smem_val;
    volatile int *vsmem_idx = smem_idx;
    if (vsmem_val[tx + 32] < vsmem_val[tx]) {
      vsmem_val[tx] = min_val = vsmem_val[tx + 32];
      vsmem_idx[tx] = min_idx = vsmem_idx[tx + 32];
    }
    if (vsmem_val[tx + 16] < vsmem_val[tx]) {
      vsmem_val[tx] = min_val = vsmem_val[tx + 16];
      vsmem_idx[tx] = min_idx = vsmem_idx[tx + 16];
    }
    if (vsmem_val[tx + 8] < vsmem_val[tx]) {
      vsmem_val[tx] = min_val = vsmem_val[tx + 8];
      vsmem_idx[tx] = min_idx = vsmem_idx[tx + 8];
    }
    if (vsmem_val[tx + 4] < vsmem_val[tx]) {
      vsmem_val[tx] = min_val = vsmem_val[tx + 4];
      vsmem_idx[tx] = min_idx = vsmem_idx[tx + 4];
    }
    if (vsmem_val[tx + 2] < vsmem_val[tx]) {
      vsmem_val[tx] = min_val = vsmem_val[tx + 2];
      vsmem_idx[tx] = min_idx = vsmem_idx[tx + 2];
    }
    if (vsmem_val[tx + 1] < vsmem_val[tx]) {
      vsmem_val[tx] = min_val = vsmem_val[tx + 1];
      vsmem_idx[tx] = min_idx = vsmem_idx[tx + 1];
    }
  }

  if (tx == 0) {
    output_val[bx] = min_val;
    output_idx[bx] = (input_idx == nullptr) ? min_idx : input_idx[min_idx];
  }
}

__global__ void update(float *mat, int n, int idx1, int idx2, int num_nodes1,
                       int num_nodes2) {
  int tx = threadIdx.x;
  int i = tx + blockDim.x * blockIdx.x;
  if (i >= n) {
    return;
  }
  float val = mat[n * idx1 + i];
  if (isinf(val)) {
    return;
  }
  int total_nodes = num_nodes1 + num_nodes2;
  float new_val =
      (val * num_nodes1 + mat[n * idx2 + i] * num_nodes2) / total_nodes;
  mat[n * idx1 + i] = new_val;
  mat[n * idx2 + i] = INFINITY;
  mat[n * i + idx1] = new_val;
  mat[n * i + idx2] = INFINITY;
}

class NJ {
public:
  NJ(float *_mat, int _num_seqs) : h_mat{_mat}, num_seqs{_num_seqs} {
    vector<Node *> nodes(num_seqs);
    for (int i = 0; i < num_seqs; ++i) {
      nodes[i] = new Node(nullptr, nullptr, 0.0f, 0.0f);
    }

    int n = num_seqs * num_seqs;

    // Allocate device variables
    float *d_mat; // Device matrix
    float *d_q; // Device q matrix
    CHECK(cudaMalloc((void **)&d_mat, sizeof(float) * n));
    CHECK(cudaMalloc((void **)&d_q, sizeof(float) * n));
    
    float *q = (float *)malloc(sizeof(float) * num_seqs * num_seqs);
    vector<float> s(num_seqs);
    int root_idx = -1;
    for (int remain = num_seqs; remain > 2; --remain) {
      // Calculate sums over row on GPU
      
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
  float *h_mat;
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
