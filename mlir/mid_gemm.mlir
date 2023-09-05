func.func @mid_gemm(%arg0: tensor<64x64xf64>, 
    %arg1: tensor<64x64xf64>, %arg2: tensor<64x64xf64>) -> tensor<64x64xf64> attributes {llvm.emit_c_interface} {
  %0 = linalg.matmul ins(%arg0, %arg1: tensor<64x64xf64>, tensor<64x64xf64>)
                     outs(%arg2: tensor<64x64xf64>) -> tensor<64x64xf64>
  return %0 : tensor<64x64xf64>
}
