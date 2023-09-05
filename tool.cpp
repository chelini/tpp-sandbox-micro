#include "Container.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/ToolOutputFile.h"
#include <benchmark/benchmark.h>
#include <random>

using namespace std;

extern "C" {
void _mlir_ciface_small_gemm(MemRef<float, 2> *inputA, MemRef<float, 2> *inputB,
                             MemRef<float, 2> *outputC);
}

static void DoSetup(const benchmark::State &state) {
  static const char structuredOpOdsHeaderFormat[] = R"FMT(

func.func @small_gemm(%arg0: tensor<{0}x{0}xf32>, %arg1: tensor<{0}x{0}xf32>, 
                      %arg2: tensor<{0}x{0}xf32>) -> tensor<{0}x{0}xf32> attributes {{llvm.emit_c_interface}} {{
  %0 = linalg.matmul ins(%arg0, %arg1: tensor<{0}x{0}xf32>, tensor<{0}x{0}xf32>)
                     outs(%arg2: tensor<{0}x{0}xf32>) -> tensor<{0}x{0}xf32>
  return %0 : tensor<{0}x{0}xf32>
}}

  )FMT";

  // Open the file.
  std::string outputCppImplFilename = "fat-gemm.mlir";
  std::string errorMessage;
  std::unique_ptr<llvm::ToolOutputFile> outputCppImpl;
  if (!outputCppImplFilename.empty()) {
    outputCppImpl = mlir::openOutputFile(outputCppImplFilename, &errorMessage);
    if (!outputCppImpl) {
      llvm::errs() << errorMessage << "\n";
      return;
    }
  }

  // Inject into the file.
  outputCppImpl->os() << llvm::formatv(structuredOpOdsHeaderFormat,
                                       state.range(0));
  outputCppImpl->keep();
}

static void DoTeardown(const benchmark::State &state) {}

static void BM_func(benchmark::State &state) {
  intptr_t sizesInput[2] = {state.range(0), state.range(0)};
  MemRef<float, 2> inputA(sizesInput);
  MemRef<float, 2> inputB(sizesInput);

  intptr_t sizesOutput[2] = {state.range(0), state.range(0)};
  MemRef<float, 2> outputC(sizesOutput);

  random_device rd;
  mt19937 gen(rd());
  uniform_real_distribution<> dis(0.0, 1.0);

  for (int i = inputA.getSize(); i >= 0; i--) {
    inputA[i] = dis(gen);
  }

  for (int i = inputB.getSize(); i >= 0; i--) {
    inputB[i] = dis(gen);
  }

  for (int i = outputC.getSize(); i >= 0; i--) {
    outputC[i] = dis(gen);
  }

  for (auto _ : state) {
    _mlir_ciface_small_gemm(&inputA, &inputB, &outputC);
  }
}

BENCHMARK(BM_func)->Arg(2048)->Arg(2048)->Setup(DoSetup)->Teardown(DoTeardown);
BENCHMARK_MAIN();
