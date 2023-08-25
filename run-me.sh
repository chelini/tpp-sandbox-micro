#!/bin/bash

rm *.s
rm *.ll
rm main

export PATH=/home/lorenzo/llvm-project/build/bin:$PATH
export PATH=/home/lorenzo/tpp-sandbox/build/bin:$PATH
export LD_LIBRARY_PATH=/home/lorenzo/tpp-sandbox/build/lib:/home/lorenzo/llvm-project/build/lib

clang++ -std=c++11 -O3 \
  -emit-llvm -fno-exceptions -fno-rtti -fPIE -S -isystem benchmark/include main.cpp
llc main.ll

BENCHS=("small_gemm" 
        #"large_gemm"
        "mlp_single_layer"
        "pack_gemm_operand_a"
        "pack_gemm_operand_b"
#        "mha_projection_v"
#        "mha_projection_q"
#        "mha_q_times_k"
#        "mha_softmax_times_v"
#        "mha_tensorflow"
       )

FLOPS=( 65536.000
        #268435456.000
        134479872.000
        2097152.000
        2097152.000
#        1073741824.000
#        1073741824.000
#        67108864.000
#        67108864.000
#        3355443200.000
      )

TPP_FLAGS="-tpp-mapping -bufferize \
  -convert-linalg-to-xsmm -default-pipeline"

for BENCH in ${BENCHS[@]}; do
  tpp-opt ${TPP_FLAGS} mlir/${BENCH}.mlir > ${BENCH}.llvm.mlir
  mlir-translate ${BENCH}.llvm.mlir -mlir-to-llvmir > ${BENCH}.ll
  llc ${BENCH}.ll
done

#Append .s to BENCHS
BENCHS_ASSEMBLY=("${BENCHS[@]/%/.s}")

# small_gemm.
# FLOPS = 32 * 32 * 32 * 2 = 65536
# ~108 GFLOPs
# 100% GEMM peak (baseline)

# large gemm.
# FLOPS = 512 * 512 * 1024 = 268435456

# mlp_single_layer.
# FLOPS = (256 * 512 * 512 * 2) + (256 * 512 * 2) = 134479872

# mha_projection_v.
# FLOPS = 64 * 32 * 8 * 64 * 512 (red) * 2 = 1073741824
# ~52 GFLOPs
# 48% peak

# mha_projection_q.
# FLOPS = 64 * 32 * 8 * 64 * 512 (red) * 2 = 1073741824
# ~53 GFLOPs
# 49% peak

# mha_q_times_k.
# FLOPS = 64 * 32 * 8 * 64 * 32 (red) * 2 = 67108864
# ~59 GFLOPs
# 54% peak

# mha_softmax_times_v.
# FLOPS = 64 * 32 * 8 * 64 * 32 (red) * 2 = 67108864
# ~68 GFLOPs
# 62% peak

# mha_tensorflow.
# FLOPS = projQ      + projK      + projV      + Q_t_K    + s_t_V    + Wo
# FLOPS = 1073741824 + 1073741824 + 1073741824 + 67108864 + 67108864 + 1073741824
# FLOPS = 3221225472 + 134217728 = 3355443200
# ~40 GFLOPs
# 37% peak 

clang -std=c++11 -O3 main.s ${BENCHS_ASSEMBLY[@]} \
  -Lbenchmark/build/src -L../tpp-sandbox/build/lib -no-pie -lstdc++ -lbenchmark -ltpp_c_runner_utils -lm -o main

taskset -c 1 ./main --benchmark_enable_random_interleaving=true --benchmark_repetitions=20 \
  --benchmark_min_time=1s --benchmark_report_aggregates_only=true --benchmark_format=json > dump.json

BENCHS_BENCH=("${BENCHS[@]/#/BM_}")
BENCHS_BENCH=("${BENCHS_BENCH[@]/%/_mean}")

for i in ${!BENCHS_BENCH[@]}; do
  TIME=$( jq '.benchmarks[] | select(.name=='"\"${BENCHS_BENCH[$i]}\""') .cpu_time' dump.json )
  awk "BEGIN {
    flops=${FLOPS[$i]}; time=${TIME}
    gflops=flops/time
    printf \"glops for ${BENCHS_BENCH[$i]} : %.3f\n\", gflops
  }" 
done
