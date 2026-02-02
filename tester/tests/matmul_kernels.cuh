#pragma once

#include "../utils.h"

#include <cuda_runtime.h>

// ========================================
// CUDA Kernel Declarations
// ========================================

/**
 * @brief Naive matrix multiplication kernel: C = A * B
 * @param A Input matrix A (M x K)
 * @param B Input matrix B (K x N)
 * @param C Output matrix C (M x N)
 * @param M Number of rows in A
 * @param K Number of columns in A (rows in B)
 * @param N Number of columns in B
 */
template <typename T>
__global__ void matmulNaiveKernel(const T* A, const T* B, T* C, 
                                  int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        T sum = 0;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

/**
 * @brief Optimized matrix multiplication kernel using shared memory
 * @param A Input matrix A (M x K)
 * @param B Input matrix B (K x N)
 * @param C Output matrix C (M x N)
 * @param M Number of rows in A
 * @param K Number of columns in A (rows in B)
 * @param N Number of columns in B
 */
template <typename T, int TILE_SIZE = 16>
__global__ void matmulTiledKernel(const T* A, const T* B, T* C,
                                  int M, int K, int N) {
    __shared__ T As[TILE_SIZE][TILE_SIZE];
    __shared__ T Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    T sum = 0;
    
    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tile from A
        if (row < M && (t * TILE_SIZE + tx) < K) {
            As[ty][tx] = A[row * K + t * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0;
        }
        
        // Load tile from B
        if ((t * TILE_SIZE + ty) < K && col < N) {
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0;
        }
        
        __syncthreads();
        
        // Compute partial product
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ========================================
// CUDA Wrapper Functions
// ========================================

template <typename T>
void matmulNaiveCuda(const T* A, const T* B, T* C, int M, int K, int N) {
    T *d_A, *d_B, *d_C;
    size_t size_A = M * K * sizeof(T);
    size_t size_B = K * N * sizeof(T);
    size_t size_C = M * N * sizeof(T);
    RUNTIME_CHECK(cudaMalloc(&d_A, size_A));
    RUNTIME_CHECK(cudaMalloc(&d_B, size_B));
    RUNTIME_CHECK(cudaMalloc(&d_C, size_C));
    
    // Copy data to device
    RUNTIME_CHECK(cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice));
    RUNTIME_CHECK(cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice));
    
    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
                  (M + blockSize.y - 1) / blockSize.y);
    
    matmulNaiveKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, K, N);
    RUNTIME_CHECK(cudaGetLastError());
    
    // Copy result back
    RUNTIME_CHECK(cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost));
    RUNTIME_CHECK(cudaDeviceSynchronize());
    
    // Free memory
    RUNTIME_CHECK(cudaFree(d_A));
    RUNTIME_CHECK(cudaFree(d_B));
    RUNTIME_CHECK(cudaFree(d_C));
}

template <typename T>
void matmulTiledCuda(const T* A, const T* B, T* C, int M, int K, int N) {
    T *d_A, *d_B, *d_C;
    size_t size_A = M * K * sizeof(T);
    size_t size_B = K * N * sizeof(T);
    size_t size_C = M * N * sizeof(T);
    
    constexpr int TILE_SIZE = 16;
    RUNTIME_CHECK(cudaMalloc(&d_A, size_A));
    RUNTIME_CHECK(cudaMalloc(&d_B, size_B));
    RUNTIME_CHECK(cudaMalloc(&d_C, size_C));
    
    RUNTIME_CHECK(cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice));
    RUNTIME_CHECK(cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice));
    
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((N + TILE_SIZE - 1) / TILE_SIZE,
                  (M + TILE_SIZE - 1) / TILE_SIZE);
    
    matmulTiledKernel<T, TILE_SIZE><<<gridSize, blockSize>>>(d_A, d_B, d_C, M, K, N);
    RUNTIME_CHECK(cudaGetLastError());
    
    RUNTIME_CHECK(cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost));
    RUNTIME_CHECK(cudaDeviceSynchronize());
    
    RUNTIME_CHECK(cudaFree(d_A));
    RUNTIME_CHECK(cudaFree(d_B));
    RUNTIME_CHECK(cudaFree(d_C));
}
