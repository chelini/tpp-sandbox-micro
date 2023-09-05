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
        "mid_gemm"
        "large_gemm"
        "large_gemm_with_fill"
        "large_gemm_pre_packed"
        "mlp_single_layer"
        "pack_gemm_operand_a"
        "pack_gemm_operand_b"
        "mha_projection_v"
        "mha_projection_q"
        "mha_q_times_k"
        "mha_q_times_k_transposed"
        "mha_q_times_k_with_transpose"
        "mha_softmax_times_v"
        "mha_tensorflow"
        "mha_tensorflow_bytedance"
        "mha_tensorflow_seq_len_1024"
        "mha_tensorflow_seq_len_256"
       )

FLOPS=( 65536.000
        524288.000
        536870912.000
        536870912.000
        536870912.000
        134479872.000
        2097152.000
        2097152.000
        1073741824.000
        1073741824.000
        67108864.000
        67108864.000
        67108864.000
        67108864.000
        3355443200.000
        3355443200.000
        141733920768.000
        9126805504.000
      )

TPP_FLAGS="-tpp-mapping -bufferize \
  -convert-linalg-to-xsmm -fold-xsmm-flags -default-pipeline"

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

# mid_gemm.
# FLOPS = 64 * 64 * 64 * 2 = 524288

# large gemm and large_gemm_pre_packed.
# FLOPS = 512 * 512 * 1024 * 2 = 536870912

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

# mha_tensorflow (seq len 32)
# FLOPS = projQ      + projK      + projV      + Q_t_K    + s_t_V    + Wo
# FLOPS = 1073741824 + 1073741824 + 1073741824 + 67108864 + 67108864 + 1073741824
# FLOPS = 3221225472 + 134217728 = 3355443200
# ~40 GFLOPs
# 37% peak 

# mha_tensorflow (seq len 1024)
# FLOPS = projQ      + projK      + projV      + Q_t_K    + s_t_V    + Wo
# projQ = 64 * 1024 * 8 * 64 * 512 (red) * 2 = 34359738368
# Q_t_K = 64 * 1024 * 8 * 64 * 32 (red) * 2 = 2147483648
# FLOPS = 34359738368 * 4 + 2147483648 * 2 = 141733920768

# mha_tensorflow (seq len 256)
# FLOPS = projQ      + projK      + projV      + Q_t_K    + s_t_V    + Wo
# projQ = 64 * 256 * 8 * 64 * 512 (red) * 2 = 8589934592
# Q_t_K = 64 * 256 * 8 * 64 * 32 (red) * 2 = 536870912
# FLOPS = 34359738368 * 4 + 2147483648 * 2 = 9126805504

clang -std=c++11 -O3 main.s ${BENCHS_ASSEMBLY[@]} \
  -Lbenchmark/build/src -L../tpp-sandbox/build/lib -no-pie -lstdc++ -lbenchmark -ltpp_c_runner_utils -lm -o main

taskset -c 1 ./main --benchmark_enable_random_interleaving=true --benchmark_repetitions=15 \
  --benchmark_min_time=1s --benchmark_report_aggregates_only=true --benchmark_format=json > dump.json

BENCHS_BENCH=("${BENCHS[@]/#/BM_}")
BENCHS_BENCH=("${BENCHS_BENCH[@]/%/_median}")

for i in ${!BENCHS_BENCH[@]}; do
  TIME=$( jq '.benchmarks[] | select(.name=='"\"${BENCHS_BENCH[$i]}\""') .cpu_time' dump.json )
  if [ -z "$TIME" ]
  then
    echo "no time for ${BENCHS_BENCH[$i]}, skipping it"
  else 
    awk "BEGIN {
      flops=${FLOPS[$i]}; time=${TIME}
      gflops=flops/time
      printf \"gflops for ${BENCHS_BENCH[$i]} : %.3f\n\", gflops
    }"
  fi  
done
