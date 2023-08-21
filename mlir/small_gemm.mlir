func.func @small_gemm(%arg0: tensor<32x32xf32>, 
    %arg1: tensor<32x32xf32>, %arg2: tensor<32x32xf32>) -> tensor<32x32xf32> attributes {llvm.emit_c_interface} {
  %0 = linalg.matmul ins(%arg0, %arg1: tensor<32x32xf32>, tensor<32x32xf32>)
                     outs(%arg2: tensor<32x32xf32>) -> tensor<32x32xf32>
  return %0 : tensor<32x32xf32>
}
