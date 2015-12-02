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

#define BLOCK_SIZE 128 // Block size should be multiple of 64
#define Q_BLOCK_SIZE 16

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
  Node() = default;
  Node(int _id, Node *left, Node *right, float length1, float length2)
      : id{_id}, childs{left, right}, branch_length{length1, length2} {}
  ~Node() = default;
  int id;
  vector<Node *> childs;
  vector<float> branch_length;
};

__global__ void sum_level0(float *input, int n_e, int n_b, float *output_val) {
  __shared__ float smem_val[BLOCK_SIZE];

  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int bs = blockDim.x;
  int i = (bx / n_b) * n_e + tx +
          (bx % n_b) * BLOCK_SIZE * 8; // (bx / n_b) * n_e is offset
  int n = ((bx / n_b) + 1) * n_e;
  float val = 0.0f;

  if (i < n) {
    float a1, a2, a3, a4, a5, a6, a7, a8;
    a1 = input[i];
    a1 = isinf(a1) ? 0.0f : a1;

    a2 = (i + BLOCK_SIZE) < n ? input[i + BLOCK_SIZE] : 0.0f;
    a2 = isinf(a2) ? 0.0f : a2;

    a3 = (i + 2 * BLOCK_SIZE) < n ? input[i + 2 * BLOCK_SIZE] : 0.0f;
    a3 = isinf(a3) ? 0.0f : a3;

    a4 = (i + 3 * BLOCK_SIZE) < n ? input[i + 3 * BLOCK_SIZE] : 0.0f;
    a4 = isinf(a4) ? 0.0f : a4;

    a5 = (i + 4 * BLOCK_SIZE) < n ? input[i + 4 * BLOCK_SIZE] : 0.0f;
    a5 = isinf(a5) ? 0.0f : a5;

    a6 = (i + 5 * BLOCK_SIZE) < n ? input[i + 5 * BLOCK_SIZE] : 0.0f;
    a6 = isinf(a6) ? 0.0f : a6;

    a7 = (i + 6 * BLOCK_SIZE) < n ? input[i + 6 * BLOCK_SIZE] : 0.0f;
    a7 = isinf(a7) ? 0.0f : a7;

    a8 = (i + 7 * BLOCK_SIZE) < n ? input[i + 7 * BLOCK_SIZE] : 0.0f;
    a8 = isinf(a8) ? 0.0f : a8;

    val = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8;
  }

  smem_val[tx] = val;
  __syncthreads();

  // in-place reduction in shared memory
  if (bs >= 1024 && tx < 512) {
    smem_val[tx] = val = val + smem_val[tx + 512];
  }
  __syncthreads();

  if (bs >= 512 && tx < 256) {
    smem_val[tx] = val = val + smem_val[tx + 256];
  }
  __syncthreads();

  if (bs >= 256 && tx < 128) {
    smem_val[tx] = val = val + smem_val[tx + 128];
  }
  __syncthreads();

  if (bs >= 128 && tx < 64) {
    smem_val[tx] = val = val + smem_val[tx + 64];
  }
  __syncthreads();

  // unrolling warp
  if (tx < 32) {
    volatile float *vsmem_val = smem_val;
    vsmem_val[tx] = val = val + vsmem_val[tx + 32];
    vsmem_val[tx] = val = val + vsmem_val[tx + 16];
    vsmem_val[tx] = val = val + vsmem_val[tx + 8];
    vsmem_val[tx] = val = val + vsmem_val[tx + 4];
    vsmem_val[tx] = val = val + vsmem_val[tx + 2];
    vsmem_val[tx] = val = val + vsmem_val[tx + 1];
  }

  if (tx == 0) {
    //printf("Block: %d, val: %f\n", bx, val);
    output_val[(bx / n_b) + (bx % n_b) * n_e] = val;
  }
}

__global__ void sum_level1(float *input, int n_e, int n_b, float *output) {
  int tx = threadIdx.x;
  int i = tx + blockIdx.x * blockDim.x;
  float val = 0.0f;
  if (i >= n_e) {
    return;
  }
  for (int j = 0; j < n_b; ++j) {
    val += input[i + j * n_e];
  }
  output[i] = val;
  //printf("%d, %f\n", i, val);
}

__global__ void calculate_q(float *mat, float *s, int n, int remain, float *q) {
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int i = ty + blockIdx.y * blockDim.y;
  int j = tx + blockIdx.x * blockDim.x;
  if (i >= n || j >= n) {
    return;
  }
  __shared__ float smem_i[Q_BLOCK_SIZE];
  __shared__ float smem_j[Q_BLOCK_SIZE];
  if (tx == 0) {
    smem_i[ty] = s[i];
  }
  if (ty == 0) {
    smem_j[tx] = s[j];
  }
  __syncthreads();

  float val = mat[i * n + j];
  q[i * n + j] =
      isinf(val) ? INFINITY : ((remain - 2) * val - smem_i[ty] - smem_j[tx]);
}

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

/*
 void update(int idx1, int idx2) {
    float d = h_mat[num_seqs * idx1 + idx2];
    for (int i = 0; i < num_seqs; ++i) {
      float val = h_mat[num_seqs * idx1 + i];
      if (isinf(val)) {
        continue;
      }
      float new_val = (val + h_mat[num_seqs * idx2 + i] - d) / 2;
      h_mat[num_seqs * idx1 + i] = new_val;
      h_mat[num_seqs * idx2 + i] = INFINITY;
      h_mat[num_seqs * i + idx1] = new_val;
      h_mat[num_seqs * i + idx2] = INFINITY;
    }
  }
*/

__global__ void update(float *mat, int n, float d, int idx1, int idx2) {
  int tx = threadIdx.x;
  int i = tx + blockDim.x * blockIdx.x;
  if (i >= n) {
    return;
  }
  if (i == idx2) {
    mat[n * idx1 + idx2] = INFINITY;
    mat[n * idx2 + idx1] = INFINITY;
    return;
  }
  float val = mat[n * idx1 + i];
  if (isinf(val)) {
    return;
  }
  float new_val =
    (val + mat[n * idx2 + i] - d) / 2.0;
  mat[n * idx1 + i] = new_val;
  mat[n * idx2 + i] = INFINITY;
  mat[n * i + idx1] = new_val;
  mat[n * i + idx2] = INFINITY;
}

class NJ {
public:
  NJ(float *_mat, int _num_seqs)
      : h_mat{_mat}, num_seqs{_num_seqs}, root{nullptr} {
    vector<Node *> nodes(num_seqs);
    for (int i = 0; i < num_seqs; ++i) {
      nodes[i] = new Node(i, nullptr, nullptr, 0.0f, 0.0f);
    }

    int n = num_seqs * num_seqs;
    // number of blocks to calculate a row
    int n_blocks_per_row = ceil(num_seqs / (float)(BLOCK_SIZE * 8));
    int n_out = n_blocks_per_row * num_seqs;
    
    int n_out_level0 = ceil((float)n / (BLOCK_SIZE * 8));
    int n_out_level1 = ceil((float)n_out_level0 / (BLOCK_SIZE * 8));

    // Allocate host variables
    // Result values after level 1 reduction for final reduction
    float *h_val_level1 = (float *)malloc(sizeof(float) * n_out_level1);
    // Result indexes after level 1 reduction for final reduction
    int *h_idx_level1 = (int *)malloc(sizeof(int) * n_out_level1);

    // Allocate device variables
    float *d_mat;                   // Device matrix
    float *d_q;                     // Device q matrix
    float *d_s_level0, *d_s_level1; // Device s matrix
    float *d_val_level0, *d_val_level1; // Device min result values
    int *d_idx_level0, *d_idx_level1;   // Device min index values
    CHECK(cudaMalloc((void **)&d_mat, sizeof(float) * n));
    CHECK(cudaMalloc((void **)&d_q, sizeof(float) * n));
    CHECK(cudaMalloc((void **)&d_s_level0, sizeof(float) * n_out));
    CHECK(cudaMalloc((void **)&d_s_level1, sizeof(float) * num_seqs));
    CHECK(cudaMalloc((void **)&d_val_level0, sizeof(float) * n_out_level0));
    CHECK(cudaMalloc((void **)&d_idx_level0, sizeof(int) * n_out_level0));
    CHECK(cudaMalloc((void **)&d_val_level1, sizeof(float) * n_out_level1));
    CHECK(cudaMalloc((void **)&d_idx_level1, sizeof(int) * n_out_level1));

    CHECK(cudaMemcpy(d_mat, h_mat, sizeof(float) * n, cudaMemcpyHostToDevice));

    float *q = (float *)malloc(sizeof(float) * num_seqs * num_seqs);
    int root_idx = -1;
    for (int remain = num_seqs; remain > 2; --remain) {
      // Calculate sums over row on GPU
      sum_level0<<<n_out, BLOCK_SIZE>>>(d_mat, num_seqs, n_blocks_per_row,
                                        d_s_level0);
      CHECK(cudaDeviceSynchronize());

      sum_level1<<<ceil(num_seqs / (float)BLOCK_SIZE), BLOCK_SIZE>>>(
          d_s_level0, num_seqs, n_blocks_per_row, d_s_level1);
      CHECK(cudaDeviceSynchronize());

      // Calculate q matrix on GPU
      calculate_q<<<dim3(ceil(num_seqs / (float)Q_BLOCK_SIZE),
                         ceil(num_seqs / (float)Q_BLOCK_SIZE), 1),
                    dim3(Q_BLOCK_SIZE, Q_BLOCK_SIZE, 1)>>>(
          d_mat, d_s_level1, num_seqs, remain, d_q);

      CHECK(cudaDeviceSynchronize());

      /*
      // Copy back device q back to host q to check
      CHECK(cudaMemcpy(q, d_q, sizeof(float) * num_seqs * num_seqs, cudaMemcpyDeviceToHost));
      for (int i = 0; i < num_seqs; ++i) {
        for (int j = 0; j < num_seqs; ++j) {
          cout << q[i * num_seqs + j] << ",\t";
        }
        cout << "\n";
      }
      cout << "--------------------------------------\n";*/

      // Get min on GPU
      // Reduction round 1
      getMin<<<n_out_level0, BLOCK_SIZE>>>(d_q, nullptr, n, d_val_level0,
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
      // cout << idx1 << ", " << idx2 << "\n";

      float length;
      CHECK(cudaMemcpy(&length, &d_mat[idx1 * num_seqs + idx2], sizeof(float),
                       cudaMemcpyDeviceToHost));
      float s1, s2;
      CHECK(cudaMemcpy(&s1, &d_s_level1[idx1], sizeof(float), cudaMemcpyDeviceToHost));
      CHECK(cudaMemcpy(&s2, &d_s_level1[idx2], sizeof(float), cudaMemcpyDeviceToHost));

      update<<<ceil(num_seqs / (float)BLOCK_SIZE), BLOCK_SIZE>>>(d_mat, num_seqs, length, idx1, idx2);

      float branch_length1 =
          length / 2 + (s1 - s2) / ((remain - 2) * 2);
      float branch_length2 = length - branch_length1;
      if (nodes[idx1] == nullptr || nodes[idx2] == nullptr) {
	cout << idx1 << ", " << idx2 << " Fuck\n";
      }
      root = new Node(-1, nodes[idx1], nodes[idx2], branch_length1, branch_length2);
      root_idx = idx1;
      nodes[idx1] = root;
      nodes[idx2] = nullptr;

      CHECK(cudaDeviceSynchronize());
      
      /*
      // Copy device mat back to host mat to check
      CHECK(cudaMemcpy(h_mat, d_mat, sizeof(float) * num_seqs * num_seqs, cudaMemcpyDeviceToHost));
      for (int i = 0; i < num_seqs; ++i) {
        for (int j = 0; j < num_seqs; ++j) {
          cout << h_mat[num_seqs * i + j] << ",\t";
        }
        cout << "\n";
      }
      cout << "--------------------------------------\n"; */
    }

    Node *other_root = nullptr;
    int other_root_idx = -1;
    for (int i = 0; i < num_seqs; ++i) {
      if (nodes[i] != nullptr && nodes[i] != root) {
        other_root = nodes[i];
        other_root_idx = i;
	nodes[i] = nullptr;
        break;
      }
    }
    
    float length;
    CHECK(cudaMemcpy(&length, &d_mat[root_idx * num_seqs + other_root_idx], sizeof(float), cudaMemcpyDeviceToHost));

    if (root_idx < other_root_idx) {
      root->childs.push_back(other_root);
      root->branch_length.push_back(length);
    } else {
      other_root->childs.push_back(root);
      other_root->branch_length.push_back(length);
    }

    // Free device memory
    CHECK(cudaFree(d_mat));
    CHECK(cudaFree(d_q));
    CHECK(cudaFree(d_s_level0));
    CHECK(cudaFree(d_s_level1));
    CHECK(cudaFree(d_val_level0));
    CHECK(cudaFree(d_idx_level0));
    CHECK(cudaFree(d_val_level1));
    CHECK(cudaFree(d_idx_level1));

    // Free host memory
    free(h_val_level1);
    free(h_idx_level1);
    free(q);
  }

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
    int num_childs = node->childs.size();
    for (int i = 0; i < num_childs; ++i) {
      cleanup(node->childs[i]);
    }
    delete node;
  }

  void print(Node *node) {
    if (node == nullptr) {
      cout << "Oops! Null pointer\n";
    }
    int num_childs = node->childs.size();
    // Reach the leaf
    if (num_childs == 2 && node->childs[0] == nullptr &&
        node->childs[1] == nullptr) {
      cout << "A" + std::to_string(node->id);
      return;
    }
    cout << "(";
    for (int i = 0; i < num_childs - 1; ++i) {
      print(node->childs[i]);
      cout << ":" << node->branch_length[i] << ",";
    }
    print(node->childs[num_childs - 1]);
    cout << ":" << node->branch_length[num_childs - 1] << ")";
  }
};

int main() {
  /*
  const int num_seqs = 5;
  float a[num_seqs][num_seqs]{{INFINITY, 5.0f, 9.0f, 9.0f, 8.0f},
                              {5.0f, INFINITY, 10.0f, 10.0f, 9.0f},
                              {9.0f, 10.0f, INFINITY, 8.0f, 7.0f},
                              {9.0f, 10.0f, 8.0f, INFINITY, 3.0f},
                              {8.0f, 9.0f, 7.0f, 3.0f, INFINITY}};*/

  const int num_seqs = 2048;
  float *a = new float[num_seqs * num_seqs];
  srand(0);
  for (int i = 0; i < num_seqs; ++i) {
    for (int j = 0; j < i; ++j) {
      a[i * num_seqs + j] = rand() / (float)RAND_MAX;
      a[j * num_seqs + i] = a[i * num_seqs + j];
    }
    a[i * num_seqs + i] = INFINITY;
  }

  /*
  for (int i = 0; i < num_seqs; ++i) {
    for (int j = 0; j < num_seqs; ++j) {
      cout << a[num_seqs * i + j] << ",\t";
    }
    cout << "\n";
  }
  cout << "--------------------------------------\n";*/

  assert(num_seqs > 2);
  NJ nj((float *)a, num_seqs);
  nj.print();
  return 0;
}
