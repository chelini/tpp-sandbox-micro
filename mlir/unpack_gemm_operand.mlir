func.func @unpack_gemm_operand(%arg0: tensor<16x16x32x32xf32>, %arg1: tensor<512x512xf32>) -> tensor<512x512xf32> attributes {llvm.emit_c_interface} {
  %unpack = tensor.unpack %arg0 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %arg1 : tensor<16x16x32x32xf32> -> tensor<512x512xf32>
  return %unpack : tensor<512x512xf32>
}
