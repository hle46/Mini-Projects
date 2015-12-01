#include <iostream>
#include <algorithm>
#include <vector>
#include <iomanip>
#include <cuda.h>
#include <cuda_runtime.h>

using std::swap;
using std::cout;
using std::vector;

#define BLOCK_SIZE 128
#define CHECK(call)                                                            \
  {                                                                            \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess) {                                                \
      fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                   \
      fprintf(stderr, "code: %d, reason: %s\n", error,                         \
              cudaGetErrorString(error));                                      \
      exit(1);                                                                 \
    }                                                                          \
  }

struct Node {
  int num_nodes;          // number of nodes in the subtree
  Node *left;             // left subtree
  Node *right;            // right subtree
  float total_length;     // total length of the subtree
  float branch_length[2]; // lengths of left and right subtrees
  Node(int _num_nodes, float _length, Node *_left, Node *_right, float length1,
       float length2)
      : num_nodes(_num_nodes), left(_left), right(_right),
        total_length(_length) {
    branch_length[0] = length1;
    branch_length[1] = length2;
  }
};

__global__ void getMin(float *input, int *input_idx, int n, float *output_val,
                       int *output_idx) {
  __shared__ float smem_val[BLOCK_SIZE];
  __shared__ int smem_idx[BLOCK_SIZE];

  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int i = tx + bx * BLOCK_SIZE * 8;

  float min_val = INFINITY;
  int min_idx = i;

  if (i < n) {
    float a1, a2, a3, a4, a5, a6, a7, a8;
    a1 = input[i];

    a2 = (i + BLOCK_SIZE) < n ? input[i + BLOCK_SIZE] : INFINITY;

    a3 = (i + 2 * BLOCK_SIZE) < n ? input[i + 2 * BLOCK_SIZE] : INFINITY;

    a4 = (i + 3 * BLOCK_SIZE) < n ? input[i + 3 * BLOCK_SIZE] : INFINITY;

    a5 = (i + 4 * BLOCK_SIZE) < n ? input[i + 4 * BLOCK_SIZE] : INFINITY;

    a6 = (i + 5 * BLOCK_SIZE) < n ? input[i + 5 * BLOCK_SIZE] : INFINITY;

    a7 = (i + 6 * BLOCK_SIZE) < n ? input[i + 6 * BLOCK_SIZE] : INFINITY;

    a8 = (i + 7 * BLOCK_SIZE) < n ? input[i + 7 * BLOCK_SIZE] : INFINITY;

    min_val = a1;
    min_idx = i;
    if (a2 < min_val) {
      min_val = a2;
      min_idx = i + BLOCK_SIZE;
    }
    if (a3 < min_val) {
      min_val = a3;
      min_idx = i + 2 * BLOCK_SIZE;
    }
    if (a4 < min_val) {
      min_val = a4;
      min_idx = i + 3 * BLOCK_SIZE;
    }
    if (a5 < min_val) {
      min_val = a5;
      min_idx = i + 4 * BLOCK_SIZE;
    }
    if (a6 < min_val) {
      min_val = a6;
      min_idx = i + 5 * BLOCK_SIZE;
    }
    if (a7 < min_val) {
      min_val = a7;
      min_idx = i + 6 * BLOCK_SIZE;
    }
    if (a8 < min_val) {
      min_val = a8;
      min_idx = i + 7 * BLOCK_SIZE;
    }
  }

  smem_val[tx] = min_val;
  smem_idx[tx] = min_idx;
  __syncthreads();

  // in-place reduction in shared memory
  if (blockDim.x >= 1024 && tx < 512 && smem_val[tx + 512] < min_val) {
    smem_val[tx] = min_val = smem_val[tx + 512];
    smem_idx[tx] = min_idx = smem_idx[tx + 512];
  }
  __syncthreads();

  if (blockDim.x >= 512 && tx < 256 && smem_val[tx + 256] < min_val) {
    smem_val[tx] = min_val = smem_val[tx + 256];
    smem_idx[tx] = min_idx = smem_idx[tx + 256];
  }
  __syncthreads();

  if (blockDim.x >= 256 && tx < 128 && smem_val[tx + 128] < min_val) {
    smem_val[tx] = min_val = smem_val[tx + 128];
    smem_idx[tx] = min_idx = smem_idx[tx + 128];
  }
  __syncthreads();

  if (blockDim.x >= 128 && tx < 64 && smem_val[tx + 64] < min_val) {
    smem_val[tx] = min_val = smem_val[tx + 64];
    smem_idx[tx] = min_idx = smem_idx[tx + 64];
  }
  __syncthreads();

  // unrolling warp
  if (tx < 32) {
    volatile float *vsmem_val = smem_val;
    volatile int *vsmem_idx = smem_idx;
    if (vsmem_val[tx + 32] < min_val) {
      vsmem_val[tx] = min_val = vsmem_val[tx + 32];
      vsmem_idx[tx] = min_idx = vsmem_idx[tx + 32];
    }
    if (vsmem_val[tx + 16] < min_val) {
      vsmem_val[tx] = min_val = vsmem_val[tx + 16];
      vsmem_idx[tx] = min_idx = vsmem_idx[tx + 16];
    }
    if (vsmem_val[tx + 8] < min_val) {
      vsmem_val[tx] = min_val = vsmem_val[tx + 8];
      vsmem_idx[tx] = min_idx = vsmem_idx[tx + 8];
    }
    if (vsmem_val[tx + 4] < min_val) {
      vsmem_val[tx] = min_val = vsmem_val[tx + 4];
      vsmem_idx[tx] = min_idx = vsmem_idx[tx + 4];
    }
    if (vsmem_val[tx + 2] < min_val) {
      vsmem_val[tx] = min_val = vsmem_val[tx + 2];
      vsmem_idx[tx] = min_idx = vsmem_idx[tx + 2];
    }
    if (vsmem_val[tx + 1] < min_val) {
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

class UPGMA {
public:
  UPGMA(float *_mat, int _num_seqs) {
    h_mat = _mat;
    num_seqs = _num_seqs;

    int n = num_seqs * num_seqs;
    int n_out_level0 = ceil((float)n / (BLOCK_SIZE * 8));
    int n_out_level1 = ceil((float)n_out_level0 / (BLOCK_SIZE * 8));

    // Allocate host variables
    // Result values after level 1 reduction for final reduction
    float *h_val_level1 = (float *)malloc(sizeof(float) * n_out_level1);
    // Result indexes after level 1 reduction for final reduction
    int *h_idx_level1 = (int *)malloc(sizeof(int) * n_out_level1);

    // Allocate device variables
    float *d_mat;                       // Device matrix
    float *d_val_level0, *d_val_level1; // Device result values
    int *d_idx_level0, *d_idx_level1;   // Device index values
    CHECK(cudaMalloc((void **)&d_mat, sizeof(float) * n));
    CHECK(cudaMalloc((void **)&d_val_level0, sizeof(float) * n_out_level0));
    CHECK(cudaMalloc((void **)&d_idx_level0, sizeof(int) * n_out_level0));
    CHECK(cudaMalloc((void **)&d_val_level1, sizeof(float) * n_out_level1));
    CHECK(cudaMalloc((void **)&d_idx_level1, sizeof(int) * n_out_level1));

    // Copy from host to device
    CHECK(cudaMemcpy(d_mat, h_mat, sizeof(float) * n, cudaMemcpyHostToDevice));

    vector<Node *> nodes(num_seqs);
    for (int i = 0; i < num_seqs; ++i) {
      nodes[i] = new Node(1, 0.0f, nullptr, nullptr, 0.0f, 0.0f);
    }

    for (int remain = num_seqs; remain >= 2; --remain) {
      // Reduction round 1
      getMin<<<n_out_level0, BLOCK_SIZE>>>(d_mat, nullptr, n, d_val_level0,
                                           d_idx_level0);

      CHECK(cudaDeviceSynchronize());

      // Reduction round 2
      getMin<<<n_out_level1, BLOCK_SIZE>>>(
          d_val_level0, d_idx_level0, n_out_level0, d_val_level1, d_idx_level1);

      CHECK(cudaDeviceSynchronize());

      // Copy results and indexes back
      CHECK(cudaMemcpy(h_val_level1, d_val_level1, sizeof(float) * n_out_level1,
                       cudaMemcpyDeviceToHost));
      CHECK(cudaMemcpy(h_idx_level1, d_idx_level1, sizeof(int) * n_out_level1,
                       cudaMemcpyDeviceToHost));

      float val = h_val_level1[0];
      int idx = h_idx_level1[0];
      for (int i = 0; i < n_out_level1; ++i) {
        if (h_val_level1[i] < val) {
          val = h_val_level1[i];
          idx = h_idx_level1[i];
        }
      }

      int idx1 = idx / num_seqs;
      int idx2 = idx % num_seqs;
      if (idx1 > idx2) {
        swap(idx1, idx2);
      }

      update<<<num_seqs, BLOCK_SIZE>>>(d_mat, num_seqs, idx1, idx2,
                                       nodes[idx1]->num_nodes,
                                       nodes[idx2]->num_nodes);

      float length = val;
      root = new Node(nodes[idx1]->num_nodes + nodes[idx2]->num_nodes,
                      length / 2, nodes[idx1], nodes[idx2],
                      length / 2 - nodes[idx1]->total_length,
                      length / 2 - nodes[idx2]->total_length);

      nodes[idx1] = root;
      nodes[idx2] = nullptr;

      CHECK(cudaDeviceSynchronize());
    }

    // Free device memory
    CHECK(cudaFree(d_mat));
    CHECK(cudaFree(d_val_level0));
    CHECK(cudaFree(d_idx_level0));
    CHECK(cudaFree(d_val_level1));
    CHECK(cudaFree(d_idx_level1));

    // Free host memory
    free(h_val_level1);
    free(h_idx_level1);
  }

  ~UPGMA() { cleanup(root); }

  void print() {
    print(root);
    cout << "\n";
  }

private:
  float *h_mat;
  int num_seqs;
  Node *root;

  void cleanup(Node *node) {
    if (node == nullptr) {
      return;
    }
    cleanup(node->left);
    cleanup(node->right);
    delete node;
  }

  void print(Node *node) {
    // Reach the leaf
    if (node->left == nullptr && node->right == nullptr) {
      return;
    }
    cout << "(";
    print(node->left);
    cout << ": " << std::fixed << node->branch_length[0] << ", ";
    print(node->right);
    cout << ": " << std::fixed << node->branch_length[1] << ")";
  }
};

int main() {
  /*
  const int num_seqs = 7;
  float h_a[num_seqs][num_seqs]{
      {INFINITY, 19.0f, 27.0f, 8.0f, 33.0f, 18.0f, 13.0f},
      {19.0f, INFINITY, 31.0f, 18.0f, 36.0f, 1.0f, 13.0f},
      {27.0f, 31.0f, INFINITY, 26.0f, 41.0f, 32.0f, 29.0f},
      {8.0f, 18.0f, 26.0f, INFINITY, 31.0f, 17.0f, 14.0f},
      {33.0f, 36.0f, 41.0f, 31.0f, INFINITY, 35.0f, 28.0f},
      {18.0f, 1.0f, 32.0f, 17.0f, 35.0f, INFINITY, 12.0f},
      {13.0f, 13.0f, 29.0f, 14.0f, 28.0f, 12.0f, INFINITY}}; */

  /*
  const int num_seqs = 6;
  float h_a[num_seqs][num_seqs]{
      {INFINITY, 2, 4, 6, 6, 8}, {2, INFINITY, 4, 6, 6, 8},
      {4, 4, INFINITY, 6, 6, 8}, {6, 6, 6, INFINITY, 4, 8},
      {6, 6, 6, 4, INFINITY, 8}, {8, 8, 8, 8, 8, INFINITY}}; */
  const int num_seqs = 4096;
  float *a = new float[num_seqs * num_seqs];
  srand(0);
  for (int i = 0; i < num_seqs; ++i) {
    for (int j = 0; j < i; ++j) {
      a[i * num_seqs + j] = rand() / (float)RAND_MAX;
      a[j * num_seqs + i] = a[i * num_seqs + j];
    }
    a[i * num_seqs + i] = INFINITY;
  }

  UPGMA upgma(a, num_seqs);
  upgma.print();
  return 0;
}
