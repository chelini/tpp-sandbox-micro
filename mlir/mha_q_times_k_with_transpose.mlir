#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>
#map5 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>
#map6 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>

func.func @q_times_k_with_transpose(%arg0: tensor<64x32x8x64xf32>, %arg1: tensor<64x32x8x64xf32>, %out_b: tensor<64x8x32x32xf32>) -> tensor<64x8x32x32xf32> attributes {llvm.emit_c_interface} {
  %cst_1 = arith.constant 0.0 : f32
  %7 = linalg.fill ins(%cst_1 : f32) outs(%out_b : tensor<64x8x32x32xf32>) -> tensor<64x8x32x32xf32>
  
  %et = tensor.empty() : tensor<64x8x32x64xf32>
  %t = linalg.transpose ins(%arg0: tensor<64x32x8x64xf32>) outs(%et : tensor<64x8x32x64xf32>) permutation = [0, 2, 1, 3]
  
  %eet = tensor.empty() : tensor<64x8x64x32xf32>
  %tt = linalg.transpose ins(%arg1: tensor<64x32x8x64xf32>) outs(%eet : tensor<64x8x64x32xf32>) permutation = [0, 2, 3, 1]

  %8 = linalg.generic {
    indexing_maps = [#map4, #map5, #map6], 
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} 
    ins(%t, %tt : tensor<64x8x32x64xf32>, tensor<64x8x64x32xf32>) outs(%7 : tensor<64x8x32x32xf32>) {
    ^bb0(%in: f32, %in_8: f32, %out: f32):
      %23 = arith.mulf %in, %in_8 : f32
      %24 = arith.addf %out, %23 : f32
      linalg.yield %24 : f32
  } -> tensor<64x8x32x32xf32>
 
  return %8 : tensor<64x8x32x32xf32>
}
