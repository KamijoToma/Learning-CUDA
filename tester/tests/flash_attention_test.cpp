#define ENABLE_FLASHATTN_SIMPLE_CUDA
#define ENABLE_FLASHATTN_CUDA
#include <cuda_runtime.h>
#include "../test_framework.h"

#include <random>
#include <limits>
#include <cmath>

// ========================================
// CPU Reference Implementation
// ========================================

static inline size_t idx_q(int b, int t, int qh, int d,
                           int T, int QH, int D) {
    return static_cast<size_t>(((b * T + t) * QH + qh) * D + d);
}

static inline size_t idx_kv(int b, int s, int kh, int d,
                            int S, int KH, int D) {
    return static_cast<size_t>(((b * S + s) * KH + kh) * D + d);
}

static inline size_t idx_o(int b, int t, int qh, int d,
                           int T, int QH, int D) {
    return static_cast<size_t>(((b * T + t) * QH + qh) * D + d);
}

void flashAttentionCPU(const std::vector<float>& h_q,
                       const std::vector<float>& h_k,
                       const std::vector<float>& h_v,
                       std::vector<float>& h_o,
                       int batch_size, int target_seq_len, int src_seq_len,
                       int query_heads, int kv_heads, int head_dim,
                       bool is_causal) {
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const int group_size = query_heads / kv_heads;

    for (int b = 0; b < batch_size; ++b) {
        for (int t = 0; t < target_seq_len; ++t) {
            for (int qh = 0; qh < query_heads; ++qh) {
                const int kh = qh / group_size;

                // Compute attention scores
                std::vector<float> scores(src_seq_len, 0.0f);
                float max_score = -std::numeric_limits<float>::infinity();

                for (int s = 0; s < src_seq_len; ++s) {
                    if (is_causal && s > t) {
                        scores[s] = -std::numeric_limits<float>::infinity();
                        continue;
                    }

                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; ++d) {
                        float q = h_q[idx_q(b, t, qh, d, target_seq_len, query_heads, head_dim)];
                        float k = h_k[idx_kv(b, s, kh, d, src_seq_len, kv_heads, head_dim)];
                        dot += q * k;
                    }

                    float score = dot * scale;
                    scores[s] = score;
                    if (score > max_score) {
                        max_score = score;
                    }
                }

                // Softmax
                float denom = 0.0f;
                for (int s = 0; s < src_seq_len; ++s) {
                    float ex = std::exp(scores[s] - max_score);
                    scores[s] = ex;
                    denom += ex;
                }

                float inv_denom = (denom > 0.0f) ? (1.0f / denom) : 0.0f;

                // Weighted sum
                for (int d = 0; d < head_dim; ++d) {
                    float acc = 0.0f;
                    for (int s = 0; s < src_seq_len; ++s) {
                        float p = scores[s] * inv_denom;
                        float v = h_v[idx_kv(b, s, kh, d, src_seq_len, kv_heads, head_dim)];
                        acc += p * v;
                    }
                    h_o[idx_o(b, t, qh, d, target_seq_len, query_heads, head_dim)] = acc;
                }
            }
        }
    }
}

// ========================================
// CUDA FlashAttention Declaration
// ========================================

#ifdef ENABLE_FLASHATTN_CUDA

#endif

// ========================================
// CUDA FlashAttention Simple (Stub)
// ========================================

#ifdef ENABLE_FLASHATTN_SIMPLE_CUDA

// Error checking helper
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while (0)

__global__ void flash_attention_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int T, int S, int D,
    int QH, int KH, // Query Heads, Key/Value Heads
    // Strides (Layout: [Batch, Seq, Head, Dim])
    int stride_q_b, int stride_q_t, int stride_q_h,
    int stride_k_b, int stride_k_s, int stride_k_h,
    int stride_v_b, int stride_v_s, int stride_v_h,
    int stride_o_b, int stride_o_t, int stride_o_h,
    bool is_causal)
{
    // ==========================================
    // 1. 设置 Shared Memory
    // ==========================================
    // 我们定义 K/V 的分块大小 BC。
    // 在这个简单示例中，我们取 32。
    const int BC = 32; 
    
    // Shared Memory 指针
    // 我们需要装下 K的一个块 (BC x D) 和 V的一个块 (BC x D)
    extern __shared__ float smem[];
    float* K_shared = smem;           // K 块的起始位置
    float* V_shared = smem + BC * D;  // V 块的起始位置

    // ==========================================
    // 2. 线程定位与分块 (新增: 多 Batch 多 Head)
    // ==========================================
    // Grid.z = Batch, Grid.y = Query Heads
    int b = blockIdx.z;
    int qh = blockIdx.y;

    // Handle GQA/MQA: Map query head to kv head
    // group_size = QH / KH (Assuming QH is multiple of KH)
    int kh = qh / (QH / KH);

    // 每个线程负责处理 Query 序列中的一个 token。
    // 也就是处理一行 Q，算出针对所有 K 的注意力，最后得到一行 O。
    int tx = threadIdx.x;
    int t = blockIdx.x * blockDim.x + tx; // 当前处理的是第 t 个 query

    // ==========================================
    // 3. 多维指针计算 (新增)
    // ==========================================
    // 为了支持 batch 和 heads，我们需要正确跳跃指针
    // 指向对应 batch 对应 head 的矩阵起始位置
    const float* Q_ptr = Q + b * stride_q_b + qh * stride_q_h;
    const float* K_ptr = K + b * stride_k_b + kh * stride_k_h;
    const float* V_ptr = V + b * stride_v_b + kh * stride_v_h;
    float* O_ptr = O + b * stride_o_b + qh * stride_o_h;

    // ==========================================
    // 4. 寄存器初始化
    // ==========================================
    // 我们假设 D 比较小 (<= 64)，可以直接把 Q 的一行和 O 的一行放进寄存器。
    // 这样计算 Dot Product 时非常快。
    constexpr int MAX_D = 64; 
    float q_reg[MAX_D];             // 存储 Q[t]
    float o_reg[MAX_D] = {0.0f};    // 累积 O[t]

    // Online Softmax 的统计量
    float m_i = -INFINITY; // 当前最大的 score
    float l_i = 0.0f;      // 当前 exponent 的和 (denominator)

    // 让线程加载自己负责的那一行 Q 到寄存器
    if (t < T) {
        for (int d = 0; d < D; ++d) {
             // 注意：这里的内存访问也需要考虑 sequence stride
            if (d < MAX_D) q_reg[d] = Q_ptr[t * stride_q_t + d];
        }
    }

    // 缩放因子 1/sqrt(d)
    float scale = 1.0f / sqrtf((float)D);

    // ==========================================
    // 5. 外层循环：遍历 K 和 V 的分块
    // ==========================================
    // 我们把 Key/Value 序列切成长度为 BC 的小块。
    // 每次迭代处理一个块。
    for (int s_base = 0; s_base < S; s_base += BC) {
        
        // 5.1. 协作加载 K, V 到 Shared Memory
        // ----------------------------------
        // 一个 Block 有 blockDim.x 个线程，大家一起要把 BC*D 大小的数据搬进来。
        int elements_to_load = BC * D;
        
        for (int i = tx; i < elements_to_load; i += blockDim.x) {
            int row = i / D; // 在块内的行号 (0 ~ BC-1)
            int col = i % D; // 维度索引
            int s = s_base + row; // 全局序列索引
            
            if (s < S) {
                 // 注意：需要使用 K_ptr 和 stride_k_s
                K_shared[row * D + col] = K_ptr[s * stride_k_s + col];
                V_shared[row * D + col] = V_ptr[s * stride_v_s + col];
            } else {
                // 越界补0 (Safety)
                K_shared[row * D + col] = 0.0f;
                V_shared[row * D + col] = 0.0f;
            }
        }

        // 同步！必须等大家把数据都搬进 Shared Memory
        __syncthreads();

        // 5.2. 计算 Attention (核心部分)
        // ----------------------------------
        if (t < T) { // 只计算有效的 query
            // 当前块的实际长度 (可能在最后一个块不足 BC)
            int current_chunk_len = (S - s_base < BC) ? (S - s_base) : BC;

            // 遍历当前块里的每一个 Key
            for (int k = 0; k < current_chunk_len; ++k) {
                int s = s_base + k; // Global source index

                // Causal Masking (因果遮蔽)
                // 如果启用，且 source > target，则这一项无效
                if (is_causal && s > t) {
                    continue; 
                }

                // A. 计算分数 Score = Q[t] dot K[s]
                float score = 0.0f;
                for (int d = 0; d < D; ++d) {
                    // Q 在寄存器，K 在 Shared Memory
                    if (d < MAX_D) score += q_reg[d] * K_shared[k * D + d];
                }
                score *= scale;

                // B. Online Softmax 更新
                // 这是 FlashAttention 的精髓：无需存储所有分数
                // 公式推导：
                // m_new = max(m_old, score)
                // alpha = exp(m_old - m_new)
                // P_val = exp(score - m_new)
                
                float m_prev = m_i;
                m_i = fmaxf(m_prev, score);
                
                // Be careful with -inf
                float alpha = (m_prev == -INFINITY) ? 0.0f : expf(m_prev - m_i);
                float P_val = expf(score - m_i);

                l_i = l_i * alpha + P_val;

                // 更新分子 (Weighted Sum)
                // O_new = O_old * alpha + P_val * V[s]
                for (int d = 0; d < D; ++d) {
                    if (d < MAX_D) {
                        o_reg[d] = o_reg[d] * alpha + P_val * V_shared[k * D + d];
                    }
                }
            }
        }
        // 同步！确保大家算完了，下一轮循环可以覆盖 Shared Memory
        __syncthreads();
    }

    // ==========================================
    // 6. 最终写回
    // ==========================================
    // 现在的 o_reg 只是加权和，还需要除以分母 l_i
    if (t < T) {
        float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
        for (int d = 0; d < D; ++d) {
             // 写入全局内存时使用 O_ptr 和 stride_o_t
             if (d < MAX_D) O_ptr[t * stride_o_t + d] = o_reg[d] * inv_l;
        }
    }
}

// Unified Host Implementation
void flashAttentionCUDA(const std::vector<float>& h_q,
                        const std::vector<float>& h_k,
                        const std::vector<float>& h_v,
                        std::vector<float>& h_o,
                        int batch_size, int target_seq_len, int src_seq_len,
                        int query_heads, int kv_heads, int head_dim,
                        bool is_causal) {
    
    // Allocate Device Memory
    float *d_q, *d_k, *d_v, *d_o;
    size_t size_q = h_q.size() * sizeof(float);
    size_t size_k = h_k.size() * sizeof(float);
    size_t size_v = h_v.size() * sizeof(float);
    size_t size_o = h_o.size() * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_q, size_q));
    CUDA_CHECK(cudaMalloc(&d_k, size_k));
    CUDA_CHECK(cudaMalloc(&d_v, size_v));
    CUDA_CHECK(cudaMalloc(&d_o, size_o));

    CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), size_q, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), size_k, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, h_v.data(), size_v, cudaMemcpyHostToDevice));

    // Strides Calculation: Layout [B, T, H, D]
    // Q: [B, T, QH, D]
    int stride_q_h = head_dim;
    int stride_q_t = query_heads * head_dim;
    int stride_q_b = target_seq_len * stride_q_t;

    // K, V: [B, S, KH, D]
    int stride_k_h = head_dim;
    int stride_k_s = kv_heads * head_dim;
    int stride_k_b = src_seq_len * stride_k_s;

    // V: same as K
    int stride_v_h = head_dim;
    int stride_v_s = kv_heads * head_dim;
    int stride_v_b = src_seq_len * stride_v_s;

    // O: [B, T, QH, D]
    int stride_o_h = head_dim;
    int stride_o_t = query_heads * head_dim;
    int stride_o_b = target_seq_len * stride_o_t;

    // Kernel Configuration
    int block_size = 32;
    int BC = 32;
    size_t shared_mem_size = 2 * BC * head_dim * sizeof(float);
    
    // 3D Grid: x=Time, y=QueryHeads, z=Batch
    dim3 grid((target_seq_len + block_size - 1) / block_size, query_heads, batch_size);

    flash_attention_kernel<<<grid, block_size, shared_mem_size>>>(
        d_q, d_k, d_v, d_o,
        target_seq_len, src_seq_len, head_dim,
        query_heads, kv_heads,
        stride_q_b, stride_q_t, stride_q_h,
        stride_k_b, stride_k_s, stride_k_h,
        stride_v_b, stride_v_s, stride_v_h,
        stride_o_b, stride_o_t, stride_o_h,
        is_causal
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_o.data(), d_o, size_o, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_o));
}

// Wrapper for the simple test
void flashAttentionSimpleCUDA(const std::vector<float>& h_q,
                              const std::vector<float>& h_k,
                              const std::vector<float>& h_v,
                              std::vector<float>& h_o,
                              int target_seq_len, int src_seq_len,
                              int head_dim, bool is_causal) {
    // Call the general version with B=1, QH=1, KH=1
    flashAttentionCUDA(h_q, h_k, h_v, h_o, 1, target_seq_len, src_seq_len, 1, 1, head_dim, is_causal);
}
#endif

// Enable CUDA for the main test as well
void flashAttention(const std::vector<float>& q, const std::vector<float>& k, const std::vector<float>& v, std::vector<float>& o,
                    int B, int T, int S, int QH, int KH, int D, bool causal) {
#ifdef ENABLE_FLASHATTN_SIMPLE_CUDA   
    flashAttentionCUDA(q, k, v, o, B, T, S, QH, KH, D, causal);
#endif
}

// ========================================
// FlashAttention Tests
// ========================================

class FlashAttentionTest : public BaseTest {
public:
    std::string getName() const override {
        return "FlashAttention (CPU ref, CUDA optional)";
    }

    TestResult runAccuracyTest() override {
        TestResult result;
        result.name = "Accuracy";
        result.passed = true;

        const int B = 2;
        const int T = 8;
        const int S = 8;
        const int QH = 4;
        const int KH = 2;
        const int D = 16;
        const bool causal = true;

        std::vector<float> q(B * T * QH * D);
        std::vector<float> k(B * S * KH * D);
        std::vector<float> v(B * S * KH * D);
        std::vector<float> o_cpu(B * T * QH * D, 0.0f);

        std::mt19937 gen(123);
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        for (auto& x : q) x = dis(gen);
        for (auto& x : k) x = dis(gen);
        for (auto& x : v) x = dis(gen);

        flashAttentionCPU(q, k, v, o_cpu, B, T, S, QH, KH, D, causal);

#ifdef ENABLE_FLASHATTN_CUDA
        std::vector<float> o_cuda(B * T * QH * D, 0.0f);
        try {
            flashAttention(q, k, v, o_cuda, B, T, S, QH, KH, D, causal);
        } catch (const std::exception& e) {
            result.passed = false;
            result.message = std::string("CUDA flashAttention threw: ") + e.what();
            return result;
        }

        bool ok = compareArrays(o_cpu.data(), o_cuda.data(), o_cpu.size(), 1e-3, 1e-5);
        if (!ok) {
            result.passed = false;
            result.message = "CUDA output mismatch with CPU reference";
            return result;
        }
        result.message = "CPU vs CUDA match";
#else
        result.message = "CUDA comparison skipped (define ENABLE_FLASHATTN_CUDA to enable)";
#endif

        return result;
    }

    TestResult runPerformanceTest() override {
        TestResult result;
        result.name = "Performance";
        result.passed = true;

        const int B = 2;
        const int T = 8;
        const int S = 8;
        const int QH = 4;
        const int KH = 2;
        const int D = 16;
        const bool causal = true;

        std::vector<float> q(B * T * QH * D);
        std::vector<float> k(B * S * KH * D);
        std::vector<float> v(B * S * KH * D);
        std::vector<float> o_cpu(B * T * QH * D, 0.0f);

        std::mt19937 gen(321);
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        for (auto& x : q) x = dis(gen);
        for (auto& x : k) x = dis(gen);
        for (auto& x : v) x = dis(gen);

        auto cpu_stats = benchmark([&]() {
            flashAttentionCPU(q, k, v, o_cpu, B, T, S, QH, KH, D, causal);
        }, 1, 3);

        printPerformanceStats("CPU", cpu_stats);
        result.message = "CPU benchmark complete";

        return result;
    }
};

REGISTER_TEST(FlashAttentionTest)

// ========================================
// Simplified FlashAttention Test
// batch_size=1, heads=1
// ========================================

class FlashAttentionSimpleTest : public BaseTest {
public:
    std::string getName() const override {
        return "FlashAttentionSimple (B=1, H=1)";
    }

    TestResult runAccuracyTest() override {
        TestResult result;
        result.name = "Accuracy";
        result.passed = true;

        const int B = 1;
        const int T = 4;
        const int S = 4;
        const int QH = 1;
        const int KH = 1;
        const int D = 8;
        const bool causal = false;

        std::vector<float> q(B * T * QH * D, 0.0f);
        std::vector<float> k(B * S * KH * D, 0.0f);
        std::vector<float> v(B * S * KH * D, 0.0f);
        std::vector<float> o_cpu(B * T * QH * D, 0.0f);

        // Simple deterministic data for learning
        for (size_t i = 0; i < q.size(); ++i) q[i] = static_cast<float>(static_cast<int>(i % 7) - 3);
        for (size_t i = 0; i < k.size(); ++i) k[i] = static_cast<float>(static_cast<int>(i % 5) - 2);
        for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<float>(static_cast<int>(i % 9) - 4);

        flashAttentionCPU(q, k, v, o_cpu, B, T, S, QH, KH, D, causal);

        // Self-consistency check: output should be finite
        for (auto val : o_cpu) {
            if (!std::isfinite(val)) {
                result.passed = false;
                result.message = "Non-finite value in CPU output";
                return result;
            }
        }

#ifdef ENABLE_FLASHATTN_SIMPLE_CUDA
        std::vector<float> o_cuda(B * T * QH * D, 0.0f);
        flashAttentionSimpleCUDA(q, k, v, o_cuda, T, S, D, causal);
        bool ok = compareArrays(o_cpu.data(), o_cuda.data(), o_cpu.size(), 1e-3, 1e-5);
        if (!ok) {
            result.passed = false;
            result.message = "CUDA simple output mismatch with CPU reference";
            return result;
        }
        result.message = "CPU vs CUDA simple match";
#else
        result.message = "CPU output is finite (CUDA simple skipped)";
#endif
        return result;
    }

    TestResult runPerformanceTest() override {
        TestResult result;
        result.name = "Performance";
        result.passed = true;

        const int B = 1;
        const int T = 4;
        const int S = 4;
        const int QH = 1;
        const int KH = 1;
        const int D = 8;
        const bool causal = false;

        std::vector<float> q(B * T * QH * D, 0.0f);
        std::vector<float> k(B * S * KH * D, 0.0f);
        std::vector<float> v(B * S * KH * D, 0.0f);
        std::vector<float> o_cpu(B * T * QH * D, 0.0f);

        for (size_t i = 0; i < q.size(); ++i) q[i] = static_cast<float>(static_cast<int>(i % 7) - 3);
        for (size_t i = 0; i < k.size(); ++i) k[i] = static_cast<float>(static_cast<int>(i % 5) - 2);
        for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<float>(static_cast<int>(i % 9) - 4);

        auto cpu_stats = benchmark([&]() {
            flashAttentionCPU(q, k, v, o_cpu, B, T, S, QH, KH, D, causal);
        }, 1, 3);

        printPerformanceStats("CPU", cpu_stats);
        result.message = "CPU benchmark complete";
        return result;
    }
};

REGISTER_TEST(FlashAttentionSimpleTest)

// ========================================
// Main
// ========================================

int main(int argc, char** argv) {
    bool verbose = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--verbose" || arg == "-v") {
            verbose = true;
        }
    }

    TestRegistry::runAllTests(verbose);
    return 0;
}