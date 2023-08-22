#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1) -> (d1)>

func.func @mlp_single_layer(%arg0: tensor<256x512xf32>, %output: tensor<256x512xf32>) -> tensor<256x512xf32> attributes {llvm.emit_c_interface} {
  %cst = arith.constant 0.000000e+00 : f32
 
  %cst_3 = arith.constant

  %bias3 = arith.constant dense<0.06> : tensor<512xf32>

  %zero_out_2 = linalg.fill ins(%cst : f32) outs(%output : tensor<256x512xf32>) -> tensor<256x512xf32>
  %8 = linalg.matmul ins(%arg0, %cst_3 : tensor<256x512xf32>, tensor<512x512xf32>) 
                     outs(%zero_out_2 : tensor<256x512xf32>) -> tensor<256x512xf32>
 
  %e = tensor.empty() : tensor<256x512xf32> 
  %10 = linalg.generic {
    indexing_maps = [#map3, #map4, #map3], 
    iterator_types = ["parallel", "parallel"]}
    ins(%8, %bias3 : tensor<256x512xf32>, tensor<512xf32>) outs(%e : tensor<256x512xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %add = arith.addf %in, %in_1 : f32
      linalg.yield %add : f32
  } -> tensor<256x512xf32>

  %ee = tensor.empty() : tensor<256x512xf32>
  %11 = linalg.generic {
    indexing_maps = [#map3, #map3], 
    iterator_types = ["parallel", "parallel"]} 
    ins(%10 : tensor<256x512xf32>) outs(%ee : tensor<256x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %max = arith.maxf %in, %cst : f32
      linalg.yield %max : f32
  } -> tensor<256x512xf32>

  return %11 : tensor<256x512xf32>
}