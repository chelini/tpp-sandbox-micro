func.func @pack_gemm_operand_a(%arg0: tensor<512x1024xf32>, %arg1: tensor<16x32x32x32xf32>) -> tensor<16x32x32x32xf32> attributes {llvm.emit_c_interface} {
  %pack = tensor.pack %arg0 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %arg1 : tensor<512x1024xf32> -> tensor<16x32x32x32xf32>
  return %pack : tensor<16x32x32x32xf32>
}
