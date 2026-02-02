#include "../test_framework.h"
#include "matmul_kernels.cuh"
#include <random>
#include <algorithm>

// ========================================
// CPU Reference Implementation
// ========================================

template <typename T>
void matmulCPU(const T* A, const T* B, T* C, int M, int K, int N) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            T sum = 0;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// ========================================
// Matrix Multiplication Test Class
// ========================================

class MatmulTest : public BaseTest {
private:
    // Test configuration
    static constexpr int M = 1024;   // Rows in A
    static constexpr int K = 1024;   // Cols in A, Rows in B
    static constexpr int N = 1024;   // Cols in B
    
    // Test data
    std::vector<float> A_;
    std::vector<float> B_;
    std::vector<float> C_cpu_;
    std::vector<float> C_cuda_naive_;
    std::vector<float> C_cuda_tiled_;
    
    void initializeData() {
        size_t size_A = M * K;
        size_t size_B = K * N;
        size_t size_C = M * N;
        
        A_.resize(size_A);
        B_.resize(size_B);
        C_cpu_.resize(size_C);
        C_cuda_naive_.resize(size_C);
        C_cuda_tiled_.resize(size_C);
        
        // Initialize with random values
        std::random_device rd;
        std::mt19937 gen(42);  // Fixed seed for reproducibility
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        
        for (size_t i = 0; i < size_A; ++i) {
            A_[i] = dis(gen);
        }
        
        for (size_t i = 0; i < size_B; ++i) {
            B_[i] = dis(gen);
        }
    }
    
public:
    std::string getName() const override {
        return "MatrixMultiplication (M=" + std::to_string(M) + 
               ", K=" + std::to_string(K) + ", N=" + std::to_string(N) + ")";
    }
    
    TestResult runAccuracyTest() override {
        TestResult result;
        result.name = "Accuracy";
        result.passed = true;
        
        try {
            // Initialize test data
            initializeData();
            
            // Compute CPU reference
            Timer cpu_timer;
            cpu_timer.start();
            matmulCPU(A_.data(), B_.data(), C_cpu_.data(), M, K, N);
            double cpu_time = cpu_timer.elapsed_ms();
            
            // Compute CUDA naive
            matmulNaiveCuda(A_.data(), B_.data(), C_cuda_naive_.data(), M, K, N);
            
            // Compute CUDA tiled
            matmulTiledCuda(A_.data(), B_.data(), C_cuda_tiled_.data(), M, K, N);
            
            // Compare results
            bool naive_matches = compareArrays(C_cpu_.data(), C_cuda_naive_.data(), 
                                              M * N, 1e-3);
            bool tiled_matches = compareArrays(C_cpu_.data(), C_cuda_tiled_.data(), 
                                              M * N, 1e-3);
            
            if (!naive_matches) {
                result.passed = false;
                result.message = "CUDA naive implementation mismatch with CPU";
            } else if (!tiled_matches) {
                result.passed = false;
                result.message = "CUDA tiled implementation mismatch with CPU";
            } else {
                result.message = "All implementations match (CPU ref: " + 
                               std::to_string(cpu_time) + " ms)";
            }
            
        } catch (const std::exception& e) {
            result.passed = false;
            result.message = std::string("Exception: ") + e.what();
        }
        
        return result;
    }
    
    TestResult runPerformanceTest() override {
        TestResult result;
        result.name = "Performance";
        result.passed = true;
        
        try {
            // Benchmark CPU
            auto cpu_stats = benchmark([this]() {
                matmulCPU(A_.data(), B_.data(), C_cpu_.data(), M, K, N);
            }, 1, 5);  // 1 warmup, 5 iterations
            
            // Benchmark CUDA naive
            auto cuda_naive_stats = benchmark([this]() {
                matmulNaiveCuda(A_.data(), B_.data(), C_cuda_naive_.data(), M, K, N);
            }, 3, 10);  // 3 warmup, 10 iterations
            
            // Benchmark CUDA tiled
            auto cuda_tiled_stats = benchmark([this]() {
                matmulTiledCuda(A_.data(), B_.data(), C_cuda_tiled_.data(), M, K, N);
            }, 3, 10);  // 3 warmup, 10 iterations
            
            // Print results
            printPerformanceStats("CPU", cpu_stats);
            printPerformanceStats("CUDA Naive", cuda_naive_stats);
            printPerformanceStats("CUDA Tiled", cuda_tiled_stats);
            
            // Calculate speedup
            double speedup_naive = cpu_stats.mean_ms / cuda_naive_stats.mean_ms;
            double speedup_tiled = cpu_stats.mean_ms / cuda_tiled_stats.mean_ms;
            
            result.message = "Speedup - Naive: " + std::to_string(speedup_naive) + "x, " +
                           "Tiled: " + std::to_string(speedup_tiled) + "x";
            
        } catch (const std::exception& e) {
            result.passed = false;
            result.message = std::string("Exception: ") + e.what();
        }
        
        return result;
    }
};

// Register the test
REGISTER_TEST(MatmulTest)

// ========================================
// Main Function (for standalone execution)
// ========================================

int main(int argc, char** argv) {
    bool verbose = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--verbose" || arg == "-v") {
            verbose = true;
        }
    }
    
    // Run all registered tests
    TestRegistry::runAllTests(verbose);
    
    return 0;
}
