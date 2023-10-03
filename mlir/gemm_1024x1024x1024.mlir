func.func @gemm_1024x1024x1024(%arg0: tensor<1024x1024xf32>, 
                         %arg1: tensor<1024x1024xf32>,
                         %arg2: tensor<1024x1024xf32>) -> tensor<1024x1024xf32> attributes {llvm.emit_c_interface} {
  %0 = linalg.matmul ins(%arg0, %arg1: tensor<1024x1024xf32>, tensor<1024x1024xf32>)
                     outs(%arg2: tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  return %0 : tensor<1024x1024xf32>
}
