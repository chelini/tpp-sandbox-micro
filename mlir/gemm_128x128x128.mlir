func.func @gemm_128x128x128(%arg0: tensor<128x128xf32>, 
                         %arg1: tensor<128x128xf32>,
                         %arg2: tensor<128x128xf32>) -> tensor<128x128xf32> attributes {llvm.emit_c_interface} {
  %0 = linalg.matmul ins(%arg0, %arg1: tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg2: tensor<128x128xf32>) -> tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}
