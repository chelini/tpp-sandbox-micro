#include "Container.h"
#include <iostream>
#include <benchmark/benchmark.h>
#include <random>

using namespace std;

extern "C" {
void _mlir_ciface_fat_gemm(MemRef<float, 2> *inputA, 
                           MemRef<float, 2> *inputB, MemRef<float, 2> *outputC);
void _mlir_ciface_mha_projection_v(MemRef<float, 3> *inputA, MemRef<float, 4> *output);
void _mlir_ciface_mha_projection_q(MemRef<float, 3> *inputA, MemRef<float, 4> *output);
void _mlir_ciface_q_times_k(MemRef<float, 4> *inputA, MemRef<float, 4> *inputB, MemRef<float, 4> *output);
void _mlir_ciface_softmax_times_v(MemRef<float, 4> *inputA, MemRef<float, 4> *inputB, MemRef<float, 4> *output);
}

static void BM_fat_gemm(benchmark::State& state) {
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
    _mlir_ciface_fat_gemm(&inputA, &inputB, &outputC);
  }
}

static void BM_mha_projection_v(benchmark::State& state) {
  intptr_t sizesInput[3] = {64,32,512};
  MemRef<float, 3> inputA(sizesInput);

  intptr_t sizesOutput[4] = {64,32,8,64};
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

static void BM_mha_projection_q(benchmark::State& state) {
  intptr_t sizesInput[3] = {64,32,512};
  MemRef<float, 3> inputA(sizesInput);

  intptr_t sizesOutput[4] = {64,32,8,64};
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

static void BM_mha_q_times_k(benchmark::State& state) {
  intptr_t sizesInput[4] = {64,32,8,64};
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

static void BM_softmax_times_v(benchmark::State& state) {
  intptr_t sizesInputA[4] = {64,8,32,32};
  MemRef<float, 4> inputA(sizesInputA);
  intptr_t sizesInputB[4] = {64,32,8,64};
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

BENCHMARK(BM_fat_gemm);
BENCHMARK(BM_mha_projection_v);
BENCHMARK(BM_mha_projection_q);
BENCHMARK(BM_mha_q_times_k);
BENCHMARK(BM_softmax_times_v);

BENCHMARK_MAIN();
