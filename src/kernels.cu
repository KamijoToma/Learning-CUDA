#include <vector>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "../tester/utils.h"

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
  // Shared memory for partial sums
  extern __shared__ T shared_data[];
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Load diagonal elements into shared memory
  if (idx < diag_len) {
    shared_data[threadIdx.x] = d_input[idx * cols + idx];
  } else {
    // Initialize out-of-bounds threads to zero
    shared_data[threadIdx.x] = T(0);
  }
  __syncthreads();

  // Parallel reduction within the block
  for (size_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      shared_data[threadIdx.x] += shared_data[threadIdx.x + stride];
    }
    __syncthreads();
  }

  // Write the block's partial sum to global memory
  if (threadIdx.x == 0) {
    d_output[blockIdx.x] = shared_data[0];
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
  traceKernel<T><<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, diag_len, cols);
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
  // TODO: Implement the flash attention function
  // Quick fail since not implemented
  throw std::runtime_error("flashAttention function is not implemented yet.");
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
