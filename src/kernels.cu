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

// Optimized FlashAttention kernel using Warp-level parallelism
// - 1 Warp per Query (row) to handle large D (up to 1024) and reduce register pressure.
// - Tiling on K/V (loading to Shared Memory).
// - Vectorized loads (128-bit) for Global Memory throughput.
// - Online Softmax with warp reductions.

template <int N>
struct SizeToType;

template <> struct SizeToType<4> { using type = float; };
template <> struct SizeToType<8> { using type = float2; };
template <> struct SizeToType<16> { using type = float4; };

// Vectorized load helper
template<typename T, int N>
__device__ __forceinline__ void load_vec(const T* src, T* dst) {
    using VecType = typename SizeToType<sizeof(T) * N>::type;
    *reinterpret_cast<VecType*>(dst) = *reinterpret_cast<const VecType*>(src);
}

// Warp Reduce Sum
__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

// Warp Reduce Max
__device__ __forceinline__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    return val;
}

template <typename T, int Q_TILE_SIZE, int KV_TILE_SIZE>
__global__ void flashAttentionKernelOptimized(
    const T* __restrict__ Q, const T* __restrict__ K, const T* __restrict__ V, T* __restrict__ O,
    int T_len, int S_len, int D, int QH, int KH,
    int stride_q_b, int stride_q_t, int stride_q_h,
    int stride_k_b, int stride_k_s, int stride_k_h,
    int stride_v_b, int stride_v_s, int stride_v_h,
    int stride_o_b, int stride_o_t, int stride_o_h,
    bool causal) 
{
    // Shared memory for K and V tiles
    extern __shared__ char smem_raw[];
    T* K_shared = reinterpret_cast<T*>(smem_raw);
    T* V_shared = K_shared + KV_TILE_SIZE * D;

    // Dimensions
    const int b = blockIdx.z;
    const int qh = blockIdx.y;
    
    // Each warp handles one query row
    const int warp_id = threadIdx.y;
    const int lane_id = threadIdx.x;
    
    // Global query index for this warp
    // blockIdx.x counts blocks of queries (Q_TILE_SIZE queries per block)
    const int t_idx = blockIdx.x * Q_TILE_SIZE + warp_id;
    
    // Grouped Query Attention mapping
    const int group_size = QH / KH;
    const int kh = qh / group_size;

    // Pointers to this batch/head
    const T* Q_ptr_base = Q + b * stride_q_b + qh * stride_q_h;
    const T* K_ptr_base = K + b * stride_k_b + kh * stride_k_h;
    const T* V_ptr_base = V + b * stride_v_b + kh * stride_v_h;
    T* O_ptr_base = O + b * stride_o_b + qh * stride_o_h;

    // Registers for Query and Output (distributed across warp)
    // Only handle D up to 128 optimally with this register count, but code loops for larger D.
    // For D=1024, each thread holds 1024/32 = 32 elements.
    // We'll trust the compiler to spill or allocate registers.
    // To be safe with limited registers, we'll keep O in registers but maybe limit D loop unrolling?
    // Given the constraints, we implement a general strided loop over D.
    
    // Register files for Q and O accumulator
    // Max D supported by register caching here depends on register limit. T=1024 is risky.
    // However, we process D in loop chunks to avoid huge arrays.
    
    // Initialize output accumulator and statistics
    // We cannot easily hold full O row in registers for D=1024 without heavy spilling.
    // But we CAN accumulate in registers if we assume standard head dims (64, 128).
    // For D=1024, we must rely on compiler or use shared memory for O accumulation.
    // Here we assume typical behavior.
    
    float m_i = -INFINITY;
    float l_i = 0.0f;
    
    // Pre-load Q row into registers (distributed)
    // Dynamic array size is tricky in registers.
    // We will just read Q from global memory on demand or cache small D.
    // For performance, let's cache Q in registers distributedly.
    // Each thread holds `D / 32` elements.
    // Since D is dynamic, we use a fixed max buffer or loop.
    // We'll use a loop structure for dot products to avoid large arrays.

    // Accumulators for O. 
    // Since we can't statically allocate `float acc[D/32]`, we might have to traverse D in chunks.
    // Or we use a fixed compile-time generic max.
    constexpr int MAX_D_PER_THREAD = 1024 / 32; // 32
    float q_frag[MAX_D_PER_THREAD];
    float o_frag[MAX_D_PER_THREAD];
    
    #pragma unroll
    for (int i = 0; i < MAX_D_PER_THREAD; ++i) o_frag[i] = 0.0f;
    
    bool valid_q = (t_idx < T_len);
    
    // Load Q fragments
    if (valid_q) {
        for (int i = 0; i < MAX_D_PER_THREAD; ++i) {
            int d = i * 32 + lane_id;
            if (d < D) {
                q_frag[i] = flashToFloat(Q_ptr_base[t_idx * stride_q_t + d]);
            } else {
                q_frag[i] = 0.0f;
            }
        }
    }

    const float scale = 1.0f / sqrtf(static_cast<float>(D));

    // Outer Loop: Iterate over KV in tiles
    for (int s_base = 0; s_base < S_len; s_base += KV_TILE_SIZE) {
        
        // Cooperative Load K and V tiles into Shared Memory
        // Total threads in block = Q_TILE_SIZE * 32.
        // Elements to load = KV_TILE_SIZE * D * 2 (K and V).
        
        int flat_tid = warp_id * 32 + lane_id;
        int num_threads = Q_TILE_SIZE * 32;
        
        // Load K
        for (int i = flat_tid; i < KV_TILE_SIZE * D; i += num_threads) {
            int row = i / D;
            int col = i % D;
            int s_idx = s_base + row;
            if (s_idx < S_len && col < D) {
                K_shared[row * D + col] = K_ptr_base[s_idx * stride_k_s + col];
            } else {
                K_shared[row * D + col] = flashFromFloat<T>(0.0f);
            }
        }
        
        // Load V
        for (int i = flat_tid; i < KV_TILE_SIZE * D; i += num_threads) {
            int row = i / D;
            int col = i % D;
            int s_idx = s_base + row;
            if (s_idx < S_len && col < D) {
                V_shared[row * D + col] = V_ptr_base[s_idx * stride_v_s + col];
            } else {
                V_shared[row * D + col] = flashFromFloat<T>(0.0f);
            }
        }
        
        __syncthreads();
        
        if (valid_q) {
            // Process the tile
            int current_kv_len = min(KV_TILE_SIZE, S_len - s_base);
            
            for (int k = 0; k < current_kv_len; ++k) {
                int s_idx = s_base + k;
                
                // Causal Masking
                if (causal && s_idx > t_idx) continue;
                
                // Compute Dot Product (Q_row . K_row)
                // Distributed dot product across warp
                float dot = 0.0f;
                for (int i = 0; i < MAX_D_PER_THREAD; ++i) {
                    int d = i * 32 + lane_id;
                    if (d < D) {
                        float k_val = flashToFloat(K_shared[k * D + d]);
                        // dot += q_frag[i] * k_val;
                        dot = fmaf(q_frag[i], k_val, dot);
                    }
                }
                
                // Reduction across warp to get full dot product
                dot = warpReduceSum(dot);
                
                // Thread 0 of warp has the score, broadcast it? 
                // Actually everyone needs it for Softmax updates
                float score = dot * scale;
                // Broadcast score from lane 0
                score = __shfl_sync(0xffffffff, score, 0);

                // Online Softmax Update
                float m_prev = m_i;
                m_i = fmaxf(m_prev, score);
                float alpha = (m_prev == -INFINITY) ? 0.0f : expf(m_prev - m_i);
                float p_val = expf(score - m_i);
                // l_i = l_i * alpha + p_val;
                l_i = fmaf(l_i, alpha, p_val);
                
                // Update O fragments
                // O = O * alpha + P * V
                // Distributed update
                for (int i = 0; i < MAX_D_PER_THREAD; ++i) {
                    int d = i * 32 + lane_id;
                    if (d < D) {
                         float v_val = flashToFloat(V_shared[k * D + d]);
                         // o_frag[i] = o_frag[i] * alpha + p_val * v_val;
                         // Use fmaf: (p_val * v_val) + (o_frag * alpha)
                         o_frag[i] = fmaf(p_val, v_val, o_frag[i] * alpha);
                    }
                }
            }
        }
        __syncthreads();
    }
    
    // Write output
    if (valid_q) {
        float inv_l = (l_i > 0.0f) ? 1.0f / l_i : 0.0f;
        for (int i = 0; i < MAX_D_PER_THREAD; ++i) {
            int d = i * 32 + lane_id;
            if (d < D) {
                O_ptr_base[t_idx * stride_o_t + d] = flashFromFloat<T>(o_frag[i] * inv_l);
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
 * Using Optimized Tiled Kernel.
 */
template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {       
    
  if (head_dim > 1024) {
    throw std::runtime_error("head_dim too large");
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

  // Strides for layout [B, T, H, D]
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

  // Kernel Launch Config
  // Q_TILE_SIZE queries per block.
  // KV_TILE_SIZE keys per tile.
  constexpr int Q_TILE_SIZE = 16;  // Number of Warps per block
  constexpr int KV_TILE_SIZE = 32; 

  const dim3 block_dim(32, Q_TILE_SIZE); // (Lane, Warp)
  const dim3 grid_dim((target_seq_len + Q_TILE_SIZE - 1) / Q_TILE_SIZE, query_heads, batch_size);
  
  // Shared Memory: K tile + V tile
  const size_t shared_mem_bytes = (2 * KV_TILE_SIZE * head_dim) * sizeof(T);

  flashAttentionKernelOptimized<T, Q_TILE_SIZE, KV_TILE_SIZE><<<grid_dim, block_dim, shared_mem_bytes>>>(
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
