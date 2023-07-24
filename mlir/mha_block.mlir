#map = affine_map<(b, h, s, w, r) -> (b, h, s, r)>
#map1 = affine_map<(b, h, s, w, r) -> (b, h, r, w)>
#map2 = affine_map<(b, h, s, w, r) -> (b, h, s, w)>

func.func @mha_block(%arg0: tensor<64x8x32x64xf32>, %arg1: tensor<64x8x64x32xf32>,
      %arg2: tensor<64x8x32x64xf32>, %arg3: tensor<64x8x32x64xf32>) -> tensor<64x8x32x64xf32> {
  
  %cst_1 = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<64x8x32x32xf32>
  
  %1 = linalg.fill ins(%cst_1 : f32) outs(%empty : tensor<64x8x32x32xf32>) -> tensor<64x8x32x32xf32>
  %2 = linalg.generic {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]}
    ins(%arg0, %arg1: tensor<64x8x32x64xf32>, tensor<64x8x64x32xf32>)
    outs(%1: tensor<64x8x32x32xf32>) {
    ^bb0(%in: f32, %in_8: f32, %out: f32):
      %23 = arith.mulf %in, %in_8 : f32
      %24 = arith.addf %out, %23 : f32
      linalg.yield %24 : f32
  } -> tensor<64x8x32x32xf32>
  
  %3 = linalg.fill ins(%cst_1 : f32) outs(%arg3 : tensor<64x8x32x64xf32>) -> tensor<64x8x32x64xf32>
  %4 = linalg.generic {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]}
    ins(%2, %arg2: tensor<64x8x32x32xf32>, tensor<64x8x32x64xf32>)
    outs(%arg3: tensor<64x8x32x64xf32>) {
    ^bb0(%in: f32, %in_8: f32, %out: f32):
      %23 = arith.mulf %in, %in_8 : f32
      %24 = arith.addf %out, %23 : f32
      linalg.yield %24 : f32
  } -> tensor<64x8x32x64xf32>

  return %4: tensor<64x8x32x64xf32>
}
