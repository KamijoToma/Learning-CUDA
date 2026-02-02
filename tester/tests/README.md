# Test Infrastructure

本目录提供可扩展的测试基础设施，支持 CPU 与 CUDA（或其他平台）实现的精度与性能测试，并与现有 Makefile 体系解耦。

## 快速开始

- 构建并运行所有测试（不影响原 Makefile 行为）：
  - make -f Makefile.tests
- 只构建测试：
  - make -f Makefile.tests build-tests
- 运行测试（默认 verbose）：
  - make -f Makefile.tests run-tests
- 清理测试产物：
  - make -f Makefile.tests clean-tests

## Profiling（Nsight）

- Nsight Systems：
  - make -f Makefile.tests nsys-profile TEST=matmul_test
- Nsight Compute（完整）：
  - make -f Makefile.tests ncu-profile TEST=matmul_test
- Nsight Compute（快速指标）：
  - make -f Makefile.tests ncu-quick TEST=matmul_test

## 新增测试的最小步骤

1) 添加测试源文件：
   - 在本目录新增 *_test.cpp 文件
   - 继承 BaseTest，并实现 getName/runAccuracyTest/runPerformanceTest
   - 使用 REGISTER_TEST(MyTestClass) 注册

2) 编译：
   - make -f Makefile.tests build-tests

## Matmul 示例说明

- matmul_test.cpp：包含 CPU 参考实现、CUDA naive/tiled 实现的精度与性能测试
- matmul_kernels.cuh：CUDA kernel 与 wrapper 封装

## 与 clangd 的配合

- compile_commands.json 已包含测试编译条目
- 若后续新增测试源文件，可追加新的 compile_commands 条目以获得更完整的补全/跳转体验
