func.func @attention_2_512_128_64_32_dtype_f16_f16_f16_f16(%query: tensor<2x512x128xf16>, %key: tensor<2x64x128xf16>, %value: tensor<2x64x32xf16>, %scale: f32) -> tensor<2x512x32xf16> {
  %result0 = tensor.empty(): tensor<2x512x32xf16>
  %scale_f16 = arith.truncf %scale : f32 to f16 
  %result1 = iree_linalg_ext.attention {
      indexing_maps = [affine_map<(batch, m, n, k1, k2) -> (batch, m, k1)>,
                       affine_map<(batch, m, n, k1, k2) -> (batch, k2, k1)>,
                       affine_map<(batch, m, n, k1, k2) -> (batch, k2, n)>,
                       affine_map<(batch, m, n, k1, k2) -> ()>,
                       affine_map<(batch, m, n, k1, k2) -> (batch, m, n)>]
}      ins(%query, %key, %value, %scale_f16: tensor<2x512x128xf16>, tensor<2x64x128xf16>, tensor<2x64x32xf16>, f16)
      outs(%result0: tensor<2x512x32xf16>) {
   ^bb0(%score: f32): 
   iree_linalg_ext.yield %score : f32
 } -> tensor<2x512x32xf16>
 return %result1: tensor<2x512x32xf16>
}

