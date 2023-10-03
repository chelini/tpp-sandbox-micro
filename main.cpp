#include "Container.h"
#include <benchmark/benchmark.h>
#include <iostream>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

using namespace std;

extern "C" {
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
void _mlir_ciface_gemm_32x32x32(MemRef<float, 2> *A, MemRef<float, 2> *B,
                                MemRef<float, 2> *C);
void _mlir_ciface_gemm_32x32x64(MemRef<float, 2> *A, MemRef<float, 2> *B,
                                MemRef<float, 2> *C);
void _mlir_ciface_gemm_64x64x64(MemRef<float, 2> *A, MemRef<float, 2> *B,
                                MemRef<float, 2> *C);
void _mlir_ciface_gemm_128x128x128(MemRef<float, 2> *A, MemRef<float, 2> *B,
                                   MemRef<float, 2> *C);
void _mlir_ciface_gemm_1024x1024x1024(MemRef<float, 2> *A, MemRef<float, 2> *B,
                                      MemRef<float, 2> *C);
void _mlir_ciface_unpack_gemm_operand(MemRef<float, 4> *A, MemRef<float, 2> *B);
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

  int64_t numberOfElem = 512 * 1024;
  state.SetBytesProcessed(int64_t(state.iterations()) * sizeof(float) *
                          numberOfElem);
}

static void BM_unpack_gemm_operand(benchmark::State &state) {
  intptr_t sizesIn[4] = {16, 16, 32, 32};
  MemRef<float, 4> in(sizesIn);

  intptr_t sizesOut[2] = {512, 512};
  MemRef<float, 2> out(sizesOut);

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
    _mlir_ciface_unpack_gemm_operand(&in, &out);
  }

  int64_t numberOfElem = 512 * 512;
  state.SetBytesProcessed(int64_t(state.iterations()) * sizeof(float) *
                          numberOfElem);
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

static void BM_gemm_32x32x32(benchmark::State &state) {
  intptr_t sizes[2] = {32, 32};
  MemRef<float, 2> A(sizes);
  MemRef<float, 2> B(sizes);
  MemRef<float, 2> C(sizes);

  random_device rd;
  mt19937 gen(rd());
  uniform_real_distribution<> dis(0.0, 1.0);

  for (int i = A.getSize(); i >= 0; i--) {
    A[i] = dis(gen);
    B[i] = dis(gen);
    C[i] = dis(gen);
  }

  for (auto _ : state) {
    _mlir_ciface_gemm_32x32x32(&A, &B, &C);
  }
}

static void BM_gemm_32x32x64(benchmark::State &state) {
  intptr_t sizesA[2] = {32, 64};
  MemRef<float, 2> A(sizesA);
  intptr_t sizesB[2] = {64, 32};
  MemRef<float, 2> B(sizesB);
  intptr_t sizesC[2] = {32, 32};
  MemRef<float, 2> C(sizesC);

  random_device rd;
  mt19937 gen(rd());
  uniform_real_distribution<> dis(0.0, 1.0);

  for (int i = A.getSize(); i >= 0; i--) {
    A[i] = dis(gen);
  }

  for (int i = B.getSize(); i >= 0; i--) {
    B[i] = dis(gen);
  }

  for (int i = C.getSize(); i >= 0; i--) {
    C[i] = dis(gen);
  }

  for (auto _ : state) {
    _mlir_ciface_gemm_32x32x64(&A, &B, &C);
  }
}

static void BM_gemm_64x64x64(benchmark::State &state) {
  intptr_t sizes[2] = {64, 64};
  MemRef<float, 2> A(sizes);
  MemRef<float, 2> B(sizes);
  MemRef<float, 2> C(sizes);

  random_device rd;
  mt19937 gen(rd());
  uniform_real_distribution<> dis(0.0, 1.0);

  for (int i = A.getSize(); i >= 0; i--) {
    A[i] = dis(gen);
    B[i] = dis(gen);
    C[i] = dis(gen);
  }

  for (auto _ : state) {
    _mlir_ciface_gemm_64x64x64(&A, &B, &C);
  }
}

static void BM_gemm_128x128x128(benchmark::State &state) {
  intptr_t sizes[2] = {128, 128};
  MemRef<float, 2> A(sizes);
  MemRef<float, 2> B(sizes);
  MemRef<float, 2> C(sizes);

  random_device rd;
  mt19937 gen(rd());
  uniform_real_distribution<> dis(0.0, 1.0);

  for (int i = A.getSize(); i >= 0; i--) {
    A[i] = dis(gen);
    B[i] = dis(gen);
    C[i] = dis(gen);
  }

  for (auto _ : state) {
    _mlir_ciface_gemm_128x128x128(&A, &B, &C);
  }
}

static void BM_gemm_1024x1024x1024(benchmark::State &state) {
  intptr_t sizes[2] = {1024, 1024};
  MemRef<float, 2> A(sizes);
  MemRef<float, 2> B(sizes);
  MemRef<float, 2> C(sizes);

  random_device rd;
  mt19937 gen(rd());
  uniform_real_distribution<> dis(0.0, 1.0);

  for (int i = A.getSize(); i >= 0; i--) {
    A[i] = dis(gen);
    B[i] = dis(gen);
    C[i] = dis(gen);
  }

  for (auto _ : state) {
    _mlir_ciface_gemm_1024x1024x1024(&A, &B, &C);
  }
}

static void BM_memcpy(benchmark::State &state) {
  float *src = new float[state.range(0)];
  float *dst = new float[state.range(0)];
  memset(src, 'x', state.range(0));
  for (auto _ : state)
    memcpy(dst, src, state.range(0));
  benchmark::DoNotOptimize(src);
  benchmark::DoNotOptimize(dst);
  benchmark::ClobberMemory();
  state.SetBytesProcessed(int64_t(state.iterations()) * sizeof(float) *
                          int64_t(state.range(0)));
  delete[] src;
  delete[] dst;
}
BENCHMARK(BM_memcpy)->Arg(262144);

// BENCHMARK(BM_gemm_32x32x32);
//  BENCHMARK(BM_gemm_32x32x64);
//  BENCHMARK(BM_gemm_64x64x64);
//  BENCHMARK(BM_gemm_128x128x128);
//  BENCHMARK(BM_gemm_1024x1024x1024);
//   BENCHMARK(BM_large_gemm);
//   BENCHMARK(BM_mid_gemm);
//   BENCHMARK(BM_big_k_gemm);
//   BENCHMARK(BM_big_m_gemm);
//   BENCHMARK(BM_big_n_gemm);
//   BENCHMARK(BM_mha_tensorflow);
//  BENCHMARK(BM_mha_tensorflow_bytedance);
// BENCHMARK(BM_mha_projection_v);
// BENCHMARK(BM_mha_q_times_k);
// BENCHMARK(BM_mha_softmax_times_v);
// BENCHMARK(BM_mha_tensorflow);
//  BENCHMARK(BM_mha_tensorflow_seq_len_1024);
//  BENCHMARK(BM_mha_tensorflow_seq_len_256);
//  BENCHMARK(BM_mlp_single_layer);
BENCHMARK(BM_pack_gemm_operand_a);
BENCHMARK(BM_unpack_gemm_operand);

BENCHMARK_MAIN();
