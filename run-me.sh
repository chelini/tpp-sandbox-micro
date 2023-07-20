#!/bin/bash

rm *.s
rm *.ll
rm main

export PATH=/home/lorenzo/llvm-project/build/bin:$PATH
export PATH=/home/lorenzo/tpp-sandbox/build/bin:$PATH
export LD_LIBRARY_PATH=/home/lorenzo/tpp-sandbox/build/lib

clang++ -std=c++11 -O3 -emit-llvm -fPIE -S -isystem benchmark/include main.cpp
llc main.ll

#fat matmul.
tpp-opt -default-pipeline mlir/fat-gemm.mlir > fat-gemm.llvm.mlir
mlir-translate fat-gemm.llvm.mlir -mlir-to-llvmir > fat-gemm.ll
llc fat-gemm.ll
# FLOPS = 32 * 32 * 32 * 2
# ~108 GFLOPs
# 100% peak (baseline)

#projection_v.
tpp-opt -tile-consumer-and-fuse-producers -convert-linalg-to-tpp -bufferize \
  -convert-linalg-to-xsmm -default-pipeline mlir/projection_v.mlir > projection_v.llvm.mlir
mlir-translate projection_v.llvm.mlir -mlir-to-llvmir > projection_v.ll
llc projection_v.ll
# FLOPS = 64 * 32 * 8 * 64 * 512 (red) * 2 = 1073741824
# ~52 GFLOPs
# 48% peak

#projection q.
tpp-opt -tile-consumer-and-fuse-producers -convert-linalg-to-tpp -bufferize \
  -convert-linalg-to-xsmm -default-pipeline mlir/projection_q.mlir > projection_q.llvm.mlir
mlir-translate projection_q.llvm.mlir -mlir-to-llvmir > projection_q.ll
llc projection_q.ll
# FLOPS = 64 * 32 * 8 * 64 * 512 (red) * 2 = 1073741824
# ~53 GFLOPs
# 49% peak

# q times k.
tpp-opt -tile-consumer-and-fuse-producers -convert-linalg-to-tpp -bufferize \
  -convert-linalg-to-xsmm -default-pipeline mlir/q_times_k.mlir > q_times_k.llvm.mlir
mlir-translate q_times_k.llvm.mlir -mlir-to-llvmir > q_times_k.ll
llc q_times_k.ll
# FLOPS = 64 * 32 * 8 * 64 * 32 (red) * 2 = 67108864
# ~59 GFLOPs
# 54% peak

# softmax times v
tpp-opt -tile-consumer-and-fuse-producers -convert-linalg-to-tpp -bufferize \
  -convert-linalg-to-xsmm -default-pipeline mlir/softmax_times_v.mlir > softmax_times_v.llvm.mlir
mlir-translate softmax_times_v.llvm.mlir -mlir-to-llvmir > softmax_times_v.ll
llc softmax_times_v.ll
# FLOPS = 64 * 32 * 8 * 64 * 32 (red) * 2 = 67108864
# ~68 GFLOPs
# 62% peak

tpp-opt -tile-consumer-and-fuse-producers -convert-linalg-to-tpp -bufferize \
  -convert-linalg-to-xsmm -default-pipeline mlir/mha.mlir > mha.llvm.mlir
mlir-translate mha.llvm.mlir -mlir-to-llvmir > mha.ll
llc mha.ll
# FLOPS = projQ      + projK      + projV      + Q_t_K    + s_t_V    + Wo
# FLOPS = 1073741824 + 1073741824 + 1073741824 + 67108864 + 67108864 + 1073741824
# FLOPS = 3221225472 + 134217728 = 3355443200
# ~40 GFLOPs
# 37% peak

clang -std=c++11 -O3 main.s fat-gemm.s projection_v.s projection_q.s q_times_k.s softmax_times_v.s mha.s \
  -Lbenchmark/build/src -L../tpp-sandbox/build/lib -no-pie -lstdc++ -lbenchmark -ltpp_c_runner_utils -lm -o main

taskset -c 1 ./main --benchmark_enable_random_interleaving=true --benchmark_repetitions=10 --benchmark_min_time=1s
