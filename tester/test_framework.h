#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <functional>
#include <chrono>
#include <cmath>
#include <memory>
#include <map>
#include "utils.h"

// ========================================
// Test Result & Statistics
// ========================================

struct TestResult {
    std::string name;
    bool passed;
    double time_ms;
    std::string message;
};

struct PerformanceStats {
    double mean_ms;
    double min_ms;
    double max_ms;
    double stddev_ms;
    size_t iterations;
};

// ========================================
// Accuracy Testing Utilities
// ========================================

template <typename T>
bool compareArrays(const T* a, const T* b, size_t n, double rtol = 1e-3, double atol = 1e-5) {
    for (size_t i = 0; i < n; ++i) {
        double av = static_cast<double>(a[i]);
        double bv = static_cast<double>(b[i]);
        double diff = std::abs(av - bv);
        double magnitude = std::max(std::abs(av), std::abs(bv));
        double threshold = atol + rtol * magnitude;
        
        if (diff > threshold) {
            double relative_error = magnitude > 1e-10 ? diff / magnitude : diff;
            std::cerr << "Mismatch at index " << i << ": " 
                      << a[i] << " vs " << b[i] 
                      << " (diff: " << diff << ", rtol: " << rtol 
                      << ", atol: " << atol 
                      << ", relative error: " << relative_error << ")" << std::endl;
            return false;
        }
    }
    return true;
}

// ========================================
// Performance Timing Utilities
// ========================================

class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_;
    
public:
    void start() {
        start_ = std::chrono::high_resolution_clock::now();
    }
    
    double elapsed_ms() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }
};

// CUDA Event-based timer for accurate GPU timing
class CudaTimer {
private:
#if defined(PLATFORM_NVIDIA) || defined(PLATFORM_ILUVATAR)
    cudaEvent_t start_, stop_;
#elif defined(PLATFORM_MOORE)
    musaEvent_t start_, stop_;
#elif defined(PLATFORM_METAX)
    mcEvent_t start_, stop_;
#endif

public:
    CudaTimer() {
#if defined(PLATFORM_NVIDIA) || defined(PLATFORM_ILUVATAR)
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
#elif defined(PLATFORM_MOORE)
        musaEventCreate(&start_);
        musaEventCreate(&stop_);
#elif defined(PLATFORM_METAX)
        mcEventCreate(&start_);
        mcEventCreate(&stop_);
#endif
    }
    
    ~CudaTimer() {
#if defined(PLATFORM_NVIDIA) || defined(PLATFORM_ILUVATAR)
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
#elif defined(PLATFORM_MOORE)
        musaEventDestroy(start_);
        musaEventDestroy(stop_);
#elif defined(PLATFORM_METAX)
        mcEventDestroy(start_);
        mcEventDestroy(stop_);
#endif
    }
    
    void start() {
#if defined(PLATFORM_NVIDIA) || defined(PLATFORM_ILUVATAR)
        cudaEventRecord(start_);
#elif defined(PLATFORM_MOORE)
        musaEventRecord(start_);
#elif defined(PLATFORM_METAX)
        mcEventRecord(start_);
#endif
    }
    
    double elapsed_ms() {
#if defined(PLATFORM_NVIDIA) || defined(PLATFORM_ILUVATAR)
        cudaEventRecord(stop_);
        cudaEventSynchronize(stop_);
        float ms = 0;
        cudaEventElapsedTime(&ms, start_, stop_);
        return static_cast<double>(ms);
#elif defined(PLATFORM_MOORE)
        musaEventRecord(stop_);
        musaEventSynchronize(stop_);
        float ms = 0;
        musaEventElapsedTime(&ms, start_, stop_);
        return static_cast<double>(ms);
#elif defined(PLATFORM_METAX)
        mcEventRecord(stop_);
        mcEventSynchronize(stop_);
        float ms = 0;
        mcEventElapsedTime(&ms, start_, stop_);
        return static_cast<double>(ms);
#endif
    }
};

// Run performance benchmark
template <typename Func>
PerformanceStats benchmark(Func&& func, size_t warmup = 3, size_t iterations = 10) {
    // Warmup runs
    for (size_t i = 0; i < warmup; ++i) {
        func();
    }
    
    // Benchmark runs
    std::vector<double> times;
    times.reserve(iterations);
    
    for (size_t i = 0; i < iterations; ++i) {
        Timer timer;
        timer.start();
        func();
        double elapsed = timer.elapsed_ms();
        times.push_back(elapsed);
    }
    
    // Calculate statistics
    double sum = 0.0;
    double min_val = times[0];
    double max_val = times[0];
    
    for (double t : times) {
        sum += t;
        min_val = std::min(min_val, t);
        max_val = std::max(max_val, t);
    }
    
    double mean = sum / iterations;
    
    // Calculate standard deviation
    double variance = 0.0;
    for (double t : times) {
        variance += (t - mean) * (t - mean);
    }
    double stddev = std::sqrt(variance / iterations);
    
    return PerformanceStats{mean, min_val, max_val, stddev, iterations};
}

// ========================================
// Base Test Class
// ========================================

class BaseTest {
public:
    virtual ~BaseTest() = default;
    virtual std::string getName() const = 0;
    virtual TestResult runAccuracyTest() = 0;
    virtual TestResult runPerformanceTest() = 0;
    
    void printPerformanceStats(const std::string& name, const PerformanceStats& stats) {
        std::cout << "  [" << name << "] Performance: "
                  << stats.mean_ms << " ± " << stats.stddev_ms << " ms "
                  << "(min: " << stats.min_ms << " ms, max: " << stats.max_ms << " ms, "
                  << "n=" << stats.iterations << ")" << std::endl;
    }
};

// ========================================
// Test Registry
// ========================================

class TestRegistry {
private:
    static TestRegistry& instance() {
        static TestRegistry registry;
        return registry;
    }
    
    std::map<std::string, std::function<std::unique_ptr<BaseTest>()>> tests_;
    
    TestRegistry() = default;
    
public:
    static void registerTest(const std::string& name, 
                            std::function<std::unique_ptr<BaseTest>()> factory) {
        instance().tests_[name] = factory;
    }
    
    static std::vector<std::unique_ptr<BaseTest>> getAllTests() {
        std::vector<std::unique_ptr<BaseTest>> result;
        for (auto& [name, factory] : instance().tests_) {
            result.push_back(factory());
        }
        return result;
    }
    
    static void runAllTests(bool verbose = false) {
        auto tests = getAllTests();
        int passed = 0;
        int failed = 0;
        
        std::cout << "\n========================================" << std::endl;
        std::cout << "Running Test Suite (" << tests.size() << " tests)" << std::endl;
        std::cout << "========================================\n" << std::endl;
        
        for (auto& test : tests) {
            std::cout << "Test: " << test->getName() << std::endl;
            
            // Run accuracy test
            auto acc_result = test->runAccuracyTest();
            if (acc_result.passed) {
                std::cout << "  ✓ Accuracy: PASSED";
                if (verbose && !acc_result.message.empty()) {
                    std::cout << " (" << acc_result.message << ")";
                }
                std::cout << std::endl;
                passed++;
            } else {
                std::cout << "  ✗ Accuracy: FAILED";
                if (!acc_result.message.empty()) {
                    std::cout << " - " << acc_result.message;
                }
                std::cout << std::endl;
                failed++;
            }
            
            // Run performance test only if accuracy passed
            if (acc_result.passed) {
                auto perf_result = test->runPerformanceTest();
                if (verbose || !perf_result.passed) {
                    std::cout << "  " << (perf_result.passed ? "✓" : "✗") 
                              << " Performance: " << perf_result.message << std::endl;
                }
            }
            
            std::cout << std::endl;
        }
        
        std::cout << "========================================" << std::endl;
        std::cout << "Summary: " << passed << " passed, " << failed << " failed" << std::endl;
        std::cout << "========================================\n" << std::endl;
    }
};

// Helper macro to auto-register tests
#define REGISTER_TEST(TestClass) \
    namespace { \
        struct TestClass##Registrar { \
            TestClass##Registrar() { \
                TestRegistry::registerTest(#TestClass, []() -> std::unique_ptr<BaseTest> { \
                    return std::make_unique<TestClass>(); \
                }); \
            } \
        }; \
        static TestClass##Registrar global_##TestClass##Registrar; \
    }
