func.func @gemm_64x64x64(%arg0: tensor<64x64xf32>, 
                         %arg1: tensor<64x64xf32>,
                         %arg2: tensor<64x64xf32>) -> tensor<64x64xf32> attributes {llvm.emit_c_interface} {
  %0 = linalg.matmul ins(%arg0, %arg1: tensor<64x64xf32>, tensor<64x64xf32>)
                     outs(%arg2: tensor<64x64xf32>) -> tensor<64x64xf32>
  return %0 : tensor<64x64xf32>
}
