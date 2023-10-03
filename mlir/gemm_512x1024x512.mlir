func.func @gemm_512x512x1024(%arg0: tensor<512x1024xf32>, 
                         %arg1: tensor<1024x512xf32>,
                         %arg2: tensor<512x512xf32>) -> tensor<512x512xf32> attributes {llvm.emit_c_interface} {
  %0 = linalg.matmul ins(%arg0, %arg1: tensor<512x1024xf32>, tensor<1024x512xf32>)
                     outs(%arg2: tensor<512x512xf32>) -> tensor<512x512xf32>
  return %0 : tensor<512x512xf32>
}
