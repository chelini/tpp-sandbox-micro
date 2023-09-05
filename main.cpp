#include "Container.h"
#include <benchmark/benchmark.h>
#include <iostream>
#include <random>

using namespace std;

extern "C" {
void _mlir_ciface_small_gemm(MemRef<float, 2> *inputA, MemRef<float, 2> *inputB,
                             MemRef<float, 2> *outputC);
void _mlir_ciface_large_gemm(MemRef<float, 2> *inputA, MemRef<float, 2> *inputB,
                             MemRef<float, 2> *outputC);
void _mlir_ciface_large_gemm_with_fill(MemRef<float, 2> *inputA,
                                       MemRef<float, 2> *inputB,
                                       MemRef<float, 2> *outputC);
void _mlir_ciface_mha_projection_v(MemRef<float, 3> *inputA,
                                   MemRef<float, 4> *output);
void _mlir_ciface_mha_projection_q(MemRef<float, 3> *inputA,
                                   MemRef<float, 4> *output);
void _mlir_ciface_q_times_k(MemRef<float, 4> *inputA, MemRef<float, 4> *inputB,
                            MemRef<float, 4> *output);
void _mlir_ciface_q_times_k_with_transpose(MemRef<float, 4> *inputA,
                                           MemRef<float, 4> *inputB,
                                           MemRef<float, 4> *output);
void _mlir_ciface_q_times_k_transposed(MemRef<float, 4> *inputA,
                                       MemRef<float, 4> *inputB,
                                       MemRef<float, 4> *output);
void _mlir_ciface_softmax_times_v(MemRef<float, 4> *inputA,
                                  MemRef<float, 4> *inputB,
                                  MemRef<float, 4> *output);
void _mlir_ciface_mha_tensorflow(MemRef<float, 3> *Q, MemRef<float, 3> *K,
                                 MemRef<float, 3> *V, MemRef<float, 3> *output);
void _mlir_ciface_mha_tensorflow_seq_len_1024(MemRef<float, 3> *Q,
                                              MemRef<float, 3> *K,
                                              MemRef<float, 3> *V,
                                              MemRef<float, 3> *output);
void _mlir_ciface_mha_tensorflow_seq_len_256(MemRef<float, 3> *Q,
                                             MemRef<float, 3> *K,
                                             MemRef<float, 3> *V,
                                             MemRef<float, 3> *output);
void _mlir_ciface_mha_tensorflow_bytedance(MemRef<float, 3> *Q,
                                           MemRef<float, 3> *K,
                                           MemRef<float, 3> *V,
                                           MemRef<float, 3> *output);
void _mlir_ciface_mlp_single_layer(MemRef<float, 2> *In, MemRef<float, 2> *Out);
void _mlir_ciface_pack_gemm_operand_a(MemRef<float, 2> *In,
                                      MemRef<float, 4> *Out);
void _mlir_ciface_pack_gemm_operand_b(MemRef<float, 2> *In,
                                      MemRef<float, 4> *Out);
void _mlir_ciface_large_gemm_pre_packed(MemRef<float, 4> *inputA,
                                        MemRef<float, 4> *inputB,
                                        MemRef<float, 4> *outputC);
}

static void BM_pack_gemm_operand_a(benchmark::State &state) {
  intptr_t sizesIn[2] = {512, 1024};
  MemRef<float, 2> in(sizesIn);

  intptr_t sizesOut[4] = {16, 32, 32, 32};
  MemRef<float, 4> out(sizesOut);

  random_device rd;
  mt19937 gen(rd());
  uniform_real_distribution<> dis(0.0, 1.0);

  for (int i = in.getSize(); i >= 0; i--) {
    in[i] = dis(gen);
  }

  for (int i = out.getSize(); i >= 0; i--) {
    out[i] = 0.0;
  }

  for (auto _ : state) {
    _mlir_ciface_pack_gemm_operand_a(&in, &out);
  }
}

static void BM_pack_gemm_operand_b(benchmark::State &state) {
  intptr_t sizesIn[2] = {1024, 512};
  MemRef<float, 2> in(sizesIn);

  intptr_t sizesOut[4] = {16, 32, 32, 32};
  MemRef<float, 4> out(sizesOut);

  random_device rd;
  mt19937 gen(rd());
  uniform_real_distribution<> dis(0.0, 1.0);

  for (int i = in.getSize(); i >= 0; i--) {
    in[i] = dis(gen);
  }

  for (int i = out.getSize(); i >= 0; i--) {
    out[i] = 0.0;
  }

  for (auto _ : state) {
    _mlir_ciface_pack_gemm_operand_b(&in, &out);
  }
}

static void BM_mlp_single_layer(benchmark::State &state) {
  intptr_t sizes[2] = {256, 512};
  MemRef<float, 2> in(sizes);
  MemRef<float, 2> out(sizes);

  random_device rd;
  mt19937 gen(rd());
  uniform_real_distribution<> dis(0.0, 1.0);

  for (int i = in.getSize(); i >= 0; i--) {
    in[i] = dis(gen);
  }

  for (int i = out.getSize(); i >= 0; i--) {
    out[i] = 0.0;
  }

  for (auto _ : state) {
    _mlir_ciface_mlp_single_layer(&in, &out);
  }
}

static void BM_small_gemm(benchmark::State &state) {
  intptr_t sizesInput[2] = {32, 32};
  MemRef<float, 2> inputA(sizesInput);
  MemRef<float, 2> inputB(sizesInput);

  intptr_t sizesOutput[2] = {32, 32};
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

static void BM_large_gemm(benchmark::State &state) {
  intptr_t sizesInputA[2] = {512, 1024};
  MemRef<float, 2> inputA(sizesInputA);
  intptr_t sizesInputB[2] = {1024, 512};
  MemRef<float, 2> inputB(sizesInputB);

  intptr_t sizesOutput[2] = {512, 512};
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
    _mlir_ciface_large_gemm(&inputA, &inputB, &outputC);
  }
}

static void BM_large_gemm_with_fill(benchmark::State &state) {
  intptr_t sizesInputA[2] = {512, 1024};
  MemRef<float, 2> inputA(sizesInputA);
  intptr_t sizesInputB[2] = {1024, 512};
  MemRef<float, 2> inputB(sizesInputB);

  intptr_t sizesOutput[2] = {512, 512};
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
    outputC[i] = 0;
  }

  for (auto _ : state) {
    _mlir_ciface_large_gemm_with_fill(&inputA, &inputB, &outputC);
  }
}

static void BM_large_gemm_pre_packed(benchmark::State &state) {
  intptr_t sizesInput[4] = {16, 32, 32, 32};
  MemRef<float, 4> inputA(sizesInput);
  MemRef<float, 4> inputB(sizesInput);

  intptr_t sizesOutput[4] = {16, 16, 32, 32};
  MemRef<float, 4> outputC(sizesOutput);

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
    _mlir_ciface_large_gemm_pre_packed(&inputA, &inputB, &outputC);
  }
}

static void BM_mha_projection_v(benchmark::State &state) {
  intptr_t sizesInput[3] = {64, 32, 512};
  MemRef<float, 3> inputA(sizesInput);

  intptr_t sizesOutput[4] = {64, 32, 8, 64};
  MemRef<float, 4> output(sizesOutput);

  random_device rd;
  mt19937 gen(rd());
  uniform_real_distribution<> dis(0.0, 1.0);

  for (int i = inputA.getSize(); i >= 0; i--) {
    inputA[i] = dis(gen);
  }

  for (auto _ : state) {
    _mlir_ciface_mha_projection_v(&inputA, &output);
  }
}

static void BM_mha_projection_q(benchmark::State &state) {
  intptr_t sizesInput[3] = {64, 32, 512};
  MemRef<float, 3> inputA(sizesInput);

  intptr_t sizesOutput[4] = {64, 32, 8, 64};
  MemRef<float, 4> output(sizesOutput);

  random_device rd;
  mt19937 gen(rd());
  uniform_real_distribution<> dis(0.0, 1.0);

  for (int i = inputA.getSize(); i >= 0; i--) {
    inputA[i] = dis(gen);
  }

  for (auto _ : state) {
    _mlir_ciface_mha_projection_q(&inputA, &output);
  }
}

static void BM_mha_q_times_k(benchmark::State &state) {
  intptr_t sizesInput[4] = {64, 32, 8, 64};
  MemRef<float, 4> inputA(sizesInput);
  MemRef<float, 4> inputB(sizesInput);
  MemRef<float, 4> output(sizesInput);

  random_device rd;
  mt19937 gen(rd());
  uniform_real_distribution<> dis(0.0, 1.0);

  for (int i = inputA.getSize(); i >= 0; i--) {
    inputA[i] = dis(gen);
  }

  for (int i = inputB.getSize(); i >= 0; i--) {
    inputB[i] = dis(gen);
  }

  for (auto _ : state) {
    _mlir_ciface_q_times_k(&inputA, &inputB, &output);
  }
}

static void BM_mha_q_times_k_with_transpose(benchmark::State &state) {
  intptr_t sizesInput[4] = {64, 32, 8, 64};
  MemRef<float, 4> inputA(sizesInput);
  MemRef<float, 4> inputB(sizesInput);
  MemRef<float, 4> output(sizesInput);

  random_device rd;
  mt19937 gen(rd());
  uniform_real_distribution<> dis(0.0, 1.0);

  for (int i = inputA.getSize(); i >= 0; i--) {
    inputA[i] = dis(gen);
  }

  for (int i = inputB.getSize(); i >= 0; i--) {
    inputB[i] = dis(gen);
  }

  for (auto _ : state) {
    _mlir_ciface_q_times_k_with_transpose(&inputA, &inputB, &output);
  }
}

static void BM_mha_q_times_k_transposed(benchmark::State &state) {
  intptr_t sizesInputA[4] = {64, 8, 32, 64};
  intptr_t sizesInputB[4] = {64, 8, 64, 32};
  intptr_t sizesInputC[4] = {64, 8, 32, 32};
  MemRef<float, 4> inputA(sizesInputA);
  MemRef<float, 4> inputB(sizesInputB);
  MemRef<float, 4> output(sizesInputC);

  random_device rd;
  mt19937 gen(rd());
  uniform_real_distribution<> dis(0.0, 1.0);

  for (int i = inputA.getSize(); i >= 0; i--) {
    inputA[i] = dis(gen);
  }

  for (int i = inputB.getSize(); i >= 0; i--) {
    inputB[i] = dis(gen);
  }

  for (auto _ : state) {
    _mlir_ciface_q_times_k_transposed(&inputA, &inputB, &output);
  }
}

static void BM_mha_softmax_times_v(benchmark::State &state) {
  intptr_t sizesInputA[4] = {64, 8, 32, 32};
  MemRef<float, 4> inputA(sizesInputA);
  intptr_t sizesInputB[4] = {64, 32, 8, 64};
  MemRef<float, 4> inputB(sizesInputB);
  MemRef<float, 4> output(sizesInputB);

  random_device rd;
  mt19937 gen(rd());
  uniform_real_distribution<> dis(0.0, 1.0);

  for (int i = inputA.getSize(); i >= 0; i--) {
    inputA[i] = dis(gen);
  }

  for (int i = inputB.getSize(); i >= 0; i--) {
    inputB[i] = dis(gen);
  }

  for (auto _ : state) {
    _mlir_ciface_softmax_times_v(&inputA, &inputB, &output);
  }
}

static void BM_mha_tensorflow(benchmark::State &state) {
  intptr_t sizesInputQKV[3] = {64, 32, 512};
  MemRef<float, 3> Q(sizesInputQKV);
  MemRef<float, 3> K(sizesInputQKV);
  MemRef<float, 3> V(sizesInputQKV);
  MemRef<float, 3> output(sizesInputQKV);

  random_device rd;
  mt19937 gen(rd());
  uniform_real_distribution<> dis(0.0, 1.0);

  for (int i = Q.getSize(); i >= 0; i--) {
    Q[i] = dis(gen);
    K[i] = dis(gen);
    V[i] = dis(gen);
  }

  for (auto _ : state) {
    _mlir_ciface_mha_tensorflow(&Q, &V, &K, &output);
  }
}

static void BM_mha_tensorflow_bytedance(benchmark::State &state) {
  intptr_t sizesInputQKV[3] = {64, 32, 512};
  MemRef<float, 3> Q(sizesInputQKV);
  MemRef<float, 3> K(sizesInputQKV);
  MemRef<float, 3> V(sizesInputQKV);
  MemRef<float, 3> output(sizesInputQKV);

  random_device rd;
  mt19937 gen(rd());
  uniform_real_distribution<> dis(0.0, 1.0);

  for (int i = Q.getSize(); i >= 0; i--) {
    Q[i] = dis(gen);
    K[i] = dis(gen);
    V[i] = dis(gen);
  }

  for (auto _ : state) {
    _mlir_ciface_mha_tensorflow_bytedance(&Q, &V, &K, &output);
  }
}

static void BM_mha_tensorflow_seq_len_1024(benchmark::State &state) {
  intptr_t sizesInputQKV[3] = {64, 1024, 512};
  MemRef<float, 3> Q(sizesInputQKV);
  MemRef<float, 3> K(sizesInputQKV);
  MemRef<float, 3> V(sizesInputQKV);
  MemRef<float, 3> output(sizesInputQKV);

  random_device rd;
  mt19937 gen(rd());
  uniform_real_distribution<> dis(0.0, 1.0);

  for (int i = Q.getSize(); i >= 0; i--) {
    Q[i] = dis(gen);
    K[i] = dis(gen);
    V[i] = dis(gen);
  }

  for (auto _ : state) {
    _mlir_ciface_mha_tensorflow_seq_len_1024(&Q, &V, &K, &output);
  }
}

static void BM_mha_tensorflow_seq_len_256(benchmark::State &state) {
  intptr_t sizesInputQKV[3] = {64, 256, 512};
  MemRef<float, 3> Q(sizesInputQKV);
  MemRef<float, 3> K(sizesInputQKV);
  MemRef<float, 3> V(sizesInputQKV);
  MemRef<float, 3> output(sizesInputQKV);

  random_device rd;
  mt19937 gen(rd());
  uniform_real_distribution<> dis(0.0, 1.0);

  for (int i = Q.getSize(); i >= 0; i--) {
    Q[i] = dis(gen);
    K[i] = dis(gen);
    V[i] = dis(gen);
  }

  for (auto _ : state) {
    _mlir_ciface_mha_tensorflow_seq_len_256(&Q, &V, &K, &output);
  }
}

#if 0
BENCHMARK(BM_small_gemm);
BENCHMARK(BM_large_gemm);
BENCHMARK(BM_large_gemm_with_fill);
BENCHMARK(BM_mlp_single_layer);
BENCHMARK(BM_pack_gemm_operand_a);
BENCHMARK(BM_pack_gemm_operand_b);
BENCHMARK(BM_large_gemm_pre_packed);
BENCHMARK(BM_mha_projection_v);
BENCHMARK(BM_mha_projection_q);
BENCHMARK(BM_mha_q_times_k);
BENCHMARK(BM_mha_softmax_times_v);
BENCHMARK(BM_mha_q_times_k);
BENCHMARK(BM_mha_q_times_k_transposed);
BENCHMARK(BM_mha_q_times_k_with_transpose);
#endif
BENCHMARK(BM_small_gemm);
BENCHMARK(BM_large_gemm);
BENCHMARK(BM_mha_tensorflow);
BENCHMARK(BM_mha_tensorflow_bytedance);
BENCHMARK(BM_mha_tensorflow_seq_len_1024);
// BENCHMARK(BM_mha_tensorflow_seq_len_256);

BENCHMARK_MAIN();
