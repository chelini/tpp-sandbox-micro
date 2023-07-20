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

#projection_v.
tpp-opt -tile-consumer-and-fuse-producers -convert-linalg-to-tpp -bufferize \
  -convert-linalg-to-xsmm -default-pipeline mlir/projection_v.mlir > projection_v.llvm.mlir
mlir-translate projection_v.llvm.mlir -mlir-to-llvmir > projection_v.ll
llc projection_v.ll

#projection q.
tpp-opt -tile-consumer-and-fuse-producers -convert-linalg-to-tpp -bufferize \
  -convert-linalg-to-xsmm -default-pipeline mlir/projection_q.mlir > projection_q.llvm.mlir
mlir-translate projection_q.llvm.mlir -mlir-to-llvmir > projection_q.ll
llc projection_q.ll

# q times v
tpp-opt -tile-consumer-and-fuse-producers -convert-linalg-to-tpp -bufferize \
  -convert-linalg-to-xsmm -default-pipeline mlir/q_times_k.mlir > q_times_k.llvm.mlir
mlir-translate q_times_k.llvm.mlir -mlir-to-llvmir > q_times_k.ll
llc q_times_k.ll

clang -std=c++11 -O3 main.s fat-gemm.s projection_v.s projection_q.s q_times_k.s \
  -Lbenchmark/build/src -L../tpp-sandbox/build/lib -no-pie -lstdc++ -lbenchmark -ltpp_c_runner_utils -lm -o main

taskset -c 1 ./main --benchmark_enable_random_interleaving=true --benchmark_repetitions=10 --benchmark_min_time=1s
