func.func @pack_gemm_operand_b(%arg0: tensor<1024x512xf32>, %arg1: tensor<16x32x32x32xf32>) -> tensor<16x32x32x32xf32> attributes {llvm.emit_c_interface} {
  %0 = tensor.pack %arg0 outer_dims_perm = [1, 0] 
                         inner_dims_pos = [0, 1] 
                         inner_tiles = [32, 32] 
    into %arg1 : tensor<1024x512xf32> -> tensor<16x32x32x32xf32>
  return %0 : tensor<16x32x32x32xf32>
}
