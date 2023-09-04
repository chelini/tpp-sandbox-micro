func.func @large_gemm_pre_packed(%1: tensor<16x32x32x32xf32>, %2: tensor<16x32x32x32xf32>, %4: tensor<16x16x32x32xf32>) -> tensor<16x16x32x32xf32> attributes {llvm.emit_c_interface} {
  %5 = scf.forall (%arg3, %arg4) in (16, 16) shared_outs(%arg5 = %4) -> (tensor<16x16x32x32xf32>) {
      %extracted_slice = tensor.extract_slice %1[%arg3, 0, 0, 0] [1, 32, 32, 32] [1, 1, 1, 1] : tensor<16x32x32x32xf32> to tensor<32x32x32xf32>
      %extracted_slice_0 = tensor.extract_slice %2[%arg4, 0, 0, 0] [1, 32, 32, 32] [1, 1, 1, 1] : tensor<16x32x32x32xf32> to tensor<32x32x32xf32>
      %extracted_slice_1 = tensor.extract_slice %arg5[%arg3, %arg4, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<16x16x32x32xf32> to tensor<32x32xf32>
      %8 = linalg.batch_reduce_matmul ins(%extracted_slice, %extracted_slice_0 : tensor<32x32x32xf32>, tensor<32x32x32xf32>) outs(%extracted_slice_1 : tensor<32x32xf32>) -> tensor<32x32xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %8 into %arg5[%arg3, %arg4, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<32x32xf32> into tensor<16x16x32x32xf32>
      }
    }
  return %5: tensor<16x16x32x32xf32>
}
