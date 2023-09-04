func.func @large_gemm_with_fill(%arg0: tensor<512x1024xf32>, 
    %arg1: tensor<1024x512xf32>, %arg2: tensor<512x512xf32>) -> tensor<512x512xf32> attributes {llvm.emit_c_interface} {
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%arg2: tensor<512x512xf32>) -> tensor<512x512xf32>
  %0 = linalg.matmul ins(%arg0, %arg1: tensor<512x1024xf32>, tensor<1024x512xf32>)
                     outs(%fill: tensor<512x512xf32>) -> tensor<512x512xf32>
  return %0 : tensor<512x512xf32>
}
