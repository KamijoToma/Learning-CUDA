#include <vector>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "../tester/utils.h"

// Helpers for mixed-precision load/store
template <typename T>
__device__ __forceinline__ float flashToFloat(T x) {
  return static_cast<float>(x);
}

template <>
__device__ __forceinline__ float flashToFloat<half>(half x) {
  return __half2float(x);
}

template <typename T>
__device__ __forceinline__ T flashFromFloat(float x) {
  return static_cast<T>(x);
}

template <>
__device__ __forceinline__ half flashFromFloat<half>(float x) {
  return __float2half(x);
}

// Simple FlashAttention kernel using shared-memory tiling and online softmax
template <typename T, int BLOCK_SEQ, int MAX_D>
__global__ void flashAttentionKernel(const T* __restrict__ Q, const T* __restrict__ K,
                                      const T* __restrict__ V, T* __restrict__ O,
                                      int T_len, int S_len, int D, int QH, int KH,
                                      int stride_q_b, int stride_q_t, int stride_q_h,
                                      int stride_k_b, int stride_k_s, int stride_k_h,
                                      int stride_v_b, int stride_v_s, int stride_v_h,
                                      int stride_o_b, int stride_o_t, int stride_o_h,
                                      bool causal) {
  extern __shared__ float smem[];
  float* K_tile = smem;
  float* V_tile = smem + BLOCK_SEQ * D;

  const int b = blockIdx.z;
  const int qh = blockIdx.y;
  const int t_idx = blockIdx.x * blockDim.x + threadIdx.x;

  const int group_size = QH / KH;
  const int kh = qh / group_size;

  const T* Q_ptr = Q + b * stride_q_b + qh * stride_q_h;
  const T* K_ptr = K + b * stride_k_b + kh * stride_k_h;
  const T* V_ptr = V + b * stride_v_b + kh * stride_v_h;
  T* O_ptr = O + b * stride_o_b + qh * stride_o_h;

  float q_reg[MAX_D];
  float o_reg[MAX_D];
  float m_i = -INFINITY;
  float l_i = 0.0f;

  if (t_idx < T_len) {
    #pragma unroll
    for (int d = 0; d < MAX_D; ++d) {
      o_reg[d] = 0.0f;
    }

    #pragma unroll
    for (int d = 0; d < MAX_D; ++d) {
      if (d < D) {
        q_reg[d] = flashToFloat(Q_ptr[t_idx * stride_q_t + d]);
      } else {
        q_reg[d] = 0.0f;
      }
    }
  }

  const float scale = rsqrtf(static_cast<float>(D));

  for (int s_base = 0; s_base < S_len; s_base += BLOCK_SEQ) {
    const int tile_elems = BLOCK_SEQ * D;
    for (int idx = threadIdx.x; idx < tile_elems; idx += blockDim.x) {
      const int row = idx / D;
      const int col = idx - row * D;
      const int s_idx = s_base + row;
      if (s_idx < S_len && col < D) {
        K_tile[idx] = flashToFloat(K_ptr[s_idx * stride_k_s + col]);
        V_tile[idx] = flashToFloat(V_ptr[s_idx * stride_v_s + col]);
      } else {
        K_tile[idx] = 0.0f;
        V_tile[idx] = 0.0f;
      }
    }
    __syncthreads();

    if (t_idx < T_len) {
      const int chunk = min(BLOCK_SEQ, S_len - s_base);
      for (int k = 0; k < chunk; ++k) {
        const int s_idx = s_base + k;
        if (causal && s_idx > t_idx) {
          continue;
        }

        float score = 0.0f;
        #pragma unroll
        for (int d = 0; d < MAX_D; ++d) {
          if (d < D) {
            score += q_reg[d] * K_tile[k * D + d];
          }
        }
        score *= scale;

        const float m_prev = m_i;
        m_i = fmaxf(m_prev, score);
        const float alpha = (m_prev == -INFINITY) ? 0.0f : expf(m_prev - m_i);
        const float p_val = expf(score - m_i);
        l_i = l_i * alpha + p_val;

        #pragma unroll
        for (int d = 0; d < MAX_D; ++d) {
          if (d < D) {
            o_reg[d] = o_reg[d] * alpha + p_val * V_tile[k * D + d];
          }
        }
      }
    }
    __syncthreads();
  }

  if (t_idx < T_len) {
    const float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
    #pragma unroll
    for (int d = 0; d < MAX_D; ++d) {
      if (d < D) {
        O_ptr[t_idx * stride_o_t + d] = flashFromFloat<T>(o_reg[d] * inv_l);
      }
    }
  }
}

/**
 * @brief A naive kernel implementation of the trace function.
 * @tparam T The numeric type of matrix elements (e.g., float, int).
 * @param d_input Pointer to the input matrix in device memory.
 * @param d_output Pointer to the temp array of main diagonal elements in device memory.
 * @param diag_len Length of the main diagonal (min(rows, cols)).
 * @param cols Number of columns in the matrix.
 */
template <typename T>
__global__ void traceKernel(const T* d_input, T* d_output, size_t diag_len, size_t cols) {
  // Calculate the global thread index
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  // if the thread index is within the diagonal length, copy the diagonal element
  if (idx < diag_len) {
    d_output[idx] = d_input[idx * cols + idx];
  }
}

/**
 * @brief A improved kernel using shared memory and parallel reduction
 * @tparam T The numeric type of matrix elements (e.g., float, int).
 */
template <typename T>
__global__ void traceKernelWithReduction(const T* d_input, T* d_output, size_t diag_len, size_t cols) {
  // Shared memory for partial sums - use char array to avoid template redeclaration issues
  extern __shared__ char shared_mem[];
  T* sdata = (T*)shared_mem;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Load diagonal elements into shared memory
  if (idx < diag_len) {
    sdata[threadIdx.x] = d_input[idx * cols + idx];
  } else {
    // Initialize out-of-bounds threads to zero
    sdata[threadIdx.x] = T(0);
  }
  __syncthreads();

  // Parallel reduction within the block
  for (size_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      sdata[threadIdx.x] += sdata[threadIdx.x + stride];
    }
    __syncthreads();
  }

  // Write the block's partial sum to global memory
  if (threadIdx.x == 0) {
    d_output[blockIdx.x] = sdata[0];
  }
}

/**
 * @brief Computes the trace of a matrix.
 *
 * The trace of a matrix is defined as the sum of its diagonal elements.
 * This function expects a flattened row-major matrix stored in a
 * std::vector. If the matrix is not square, the trace will sum up
 * elements along the main diagonal up to the smaller of rows or cols.
 * 
 * The input can not be modified so pinning memory is not applicable here.
 * Sothat CUDA Streams cannot be used in this implementation.
 *
 * According to the performance report, sometimes the native kernel
 * performs better than the reduction kernel.
 *
 * @tparam T The numeric type of matrix elements (e.g., float, int).
 * @param h_input A flattened matrix of size rows * cols.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @return The trace (sum of diagonal values) of the matrix.
 */
template <typename T>
T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
  // TODO: Implement the trace function
  auto diag_len = (rows < cols) ? rows : cols;
  int threadsPerBlock = 256;
  int blocksPerGrid = (diag_len + threadsPerBlock - 1) / threadsPerBlock;
  T* d_input, * d_output;
  size_t input_size = h_input.size() * sizeof(T);
  size_t output_size = diag_len * sizeof(T);
  // Allocate device memory
  cudaMalloc((void**)&d_input, input_size);
  cudaMalloc((void**)&d_output, output_size);
  // Copy input data from host to device
  cudaMemcpy(d_input, h_input.data(), input_size, cudaMemcpyHostToDevice);
  // Launch the kernel
  // ====== native ======
  // traceKernel<T><<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, diag_len, cols);
  // ====== with reduction ======
  size_t shared_mem_size = threadsPerBlock * sizeof(T);
  traceKernelWithReduction<T><<<blocksPerGrid, threadsPerBlock, shared_mem_size>>>(d_input, d_output, diag_len, cols);
  cudaDeviceSynchronize();
  // Copy the diagonal elements back to host
  std::vector<T> h_output(diag_len);
  cudaMemcpy(h_output.data(), d_output, output_size, cudaMemcpyDeviceToHost);
  // Free device memory
  cudaFree(d_input);
  cudaFree(d_output);
  // Compute the sum of diagonal elements
  T trace_sum = T(0);
  for(const auto&val:h_output){
    trace_sum += val;
  }
  // Return the result
  return trace_sum;
}

/**
 * @brief Computes flash attention for given query, key, and value tensors.
 * 
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length  
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */
template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {       
  constexpr int BLOCK_SEQ = 32;
  constexpr int MAX_D = 1024;

  if (head_dim > MAX_D) {
    throw std::runtime_error("head_dim too large for simple FlashAttention kernel");
  }

  // Host side setup
  const size_t size_q = h_q.size() * sizeof(T);
  const size_t size_k = h_k.size() * sizeof(T);
  const size_t size_v = h_v.size() * sizeof(T);
  const size_t size_o = h_o.size() * sizeof(T);

  T *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_o = nullptr;
  RUNTIME_CHECK(cudaMalloc(&d_q, size_q));
  RUNTIME_CHECK(cudaMalloc(&d_k, size_k));
  RUNTIME_CHECK(cudaMalloc(&d_v, size_v));
  RUNTIME_CHECK(cudaMalloc(&d_o, size_o));

  RUNTIME_CHECK(cudaMemcpy(d_q, h_q.data(), size_q, cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemcpy(d_k, h_k.data(), size_k, cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemcpy(d_v, h_v.data(), size_v, cudaMemcpyHostToDevice));

  // Strides for layout [B, T, H, D] or [B, S, H, D]
  const int stride_q_h = head_dim;
  const int stride_q_t = query_heads * head_dim;
  const int stride_q_b = target_seq_len * stride_q_t;

  const int stride_k_h = head_dim;
  const int stride_k_s = kv_heads * head_dim;
  const int stride_k_b = src_seq_len * stride_k_s;

  const int stride_v_h = head_dim;
  const int stride_v_s = kv_heads * head_dim;
  const int stride_v_b = src_seq_len * stride_v_s;

  const int stride_o_h = head_dim;
  const int stride_o_t = query_heads * head_dim;
  const int stride_o_b = target_seq_len * stride_o_t;

  const int block_size = 128; // threads per block along time dimension
  const dim3 grid((target_seq_len + block_size - 1) / block_size, query_heads, batch_size);
  const size_t shared_bytes = 2 * BLOCK_SEQ * head_dim * sizeof(float);

  flashAttentionKernel<T, BLOCK_SEQ, MAX_D><<<grid, block_size, shared_bytes>>>(
      d_q, d_k, d_v, d_o,
      target_seq_len, src_seq_len, head_dim,
      query_heads, kv_heads,
      stride_q_b, stride_q_t, stride_q_h,
      stride_k_b, stride_k_s, stride_k_h,
      stride_v_b, stride_v_s, stride_v_h,
      stride_o_b, stride_o_t, stride_o_h,
      is_causal);

  RUNTIME_CHECK(cudaGetLastError());
  RUNTIME_CHECK(cudaDeviceSynchronize());

  RUNTIME_CHECK(cudaMemcpy(h_o.data(), d_o, size_o, cudaMemcpyDeviceToHost));

  cudaFree(d_q);
  cudaFree(d_k);
  cudaFree(d_v);
  cudaFree(d_o);
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int trace<int>(const std::vector<int>&, size_t, size_t);
template float trace<float>(const std::vector<float>&, size_t, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);
