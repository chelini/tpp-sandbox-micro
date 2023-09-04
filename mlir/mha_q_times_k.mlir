#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
#map5 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d2, d3)>
#map6 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4, d1)>

func.func @q_times_k(%arg0: tensor<64x32x8x64xf32>, %arg1: tensor<64x32x8x64xf32>, %out_b: tensor<64x8x32x32xf32>) -> tensor<64x8x32x32xf32> attributes {llvm.emit_c_interface} {
  %cst_1 = arith.constant 0.0 : f32
  %7 = linalg.fill ins(%cst_1 : f32) outs(%out_b : tensor<64x8x32x32xf32>) -> tensor<64x8x32x32xf32>
  %8 = linalg.generic {
    indexing_maps = [#map4, #map5, #map6], 
    iterator_types = ["parallel", "parallel", "parallel", "reduction", "parallel"]} 
    ins(%arg0, %arg1 : tensor<64x32x8x64xf32>, tensor<64x32x8x64xf32>) outs(%7 : tensor<64x8x32x32xf32>) {
    ^bb0(%in: f32, %in_8: f32, %out: f32):
      %23 = arith.mulf %in, %in_8 : f32
      %24 = arith.addf %out, %23 : f32
      linalg.yield %24 : f32
    } -> tensor<64x8x32x32xf32>
  return %8 : tensor<64x8x32x32xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  transform.structured.interchange %0 iterator_interchange = [0, 2, 1, 4, 3] : (!transform.any_op) -> !transform.any_op
}
