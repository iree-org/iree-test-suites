func.func @attention_2_DYN_16_64_DYN_dtype_f16_f16_f16_f16(%query: tensor<2x?x64xf16>, %key: tensor<2x?x64xf16>, %value: tensor<2x?x16xf16>, %scale: f32) -> tensor<2x?x16xf16> {
  %c1 = arith.constant 1 : index
  %m_in = tensor.dim %query, %c1 : tensor<2x?x64xf16>
  %k2_in = tensor.dim %key, %c1 : tensor<2x?x64xf16>
  %m = util.assume.int %m_in<udiv = 16> : index
  %k2 = util.assume.int %k2_in<udiv = 16> : index
  %query_mixed = flow.tensor.tie_shape %query : tensor<2x?x64xf16>{%m}
  %key_mixed = flow.tensor.tie_shape %key : tensor<2x?x64xf16>{%k2}
  %value_mixed = flow.tensor.tie_shape %value : tensor<2x?x16xf16>{%k2}
  %result0 = tensor.empty(%m): tensor<2x?x16xf16>
  %scale_f16 = arith.truncf %scale : f32 to f16
  %result1 = iree_linalg_ext.attention {
      indexing_maps = [affine_map<(batch, m, n, k1, k2) -> (batch, m, k1)>,
                       affine_map<(batch, m, n, k1, k2) -> (batch, k2, k1)>,
                       affine_map<(batch, m, n, k1, k2) -> (batch, k2, n)>,
                       affine_map<(batch, m, n, k1, k2) -> ()>,
                       affine_map<(batch, m, n, k1, k2) -> (batch, m, n)>]
}  ins(%query_mixed, %key_mixed, %value_mixed, %scale_f16: tensor<2x?x64xf16>, tensor<2x?x64xf16>, tensor<2x?x16xf16>, f16)
      outs(%result0: tensor<2x?x16xf16>) {
   ^bb0(%score: f32):
   iree_linalg_ext.yield %score : f32
 } -> tensor<2x?x16xf16>
 return %result1: tensor<2x?x16xf16>
}
func.func @attention_2_256_16_64_32_dtype_f16_f16_f16_f16(%query: tensor<2x256x64xf16>, %key: tensor<2x32x64xf16>, %value: tensor<2x32x16xf16>, %scale: f32) -> tensor<2x256x16xf16> {
  %result0 = tensor.empty(): tensor<2x256x16xf16>
  %scale_f16 = arith.truncf %scale : f32 to f16
  %result1 = iree_linalg_ext.attention {
      indexing_maps = [affine_map<(batch, m, n, k1, k2) -> (batch, m, k1)>,
                       affine_map<(batch, m, n, k1, k2) -> (batch, k2, k1)>,
                       affine_map<(batch, m, n, k1, k2) -> (batch, k2, n)>,
                       affine_map<(batch, m, n, k1, k2) -> ()>,
                       affine_map<(batch, m, n, k1, k2) -> (batch, m, n)>]
}  ins(%query, %key, %value, %scale_f16: tensor<2x256x64xf16>, tensor<2x32x64xf16>, tensor<2x32x16xf16>, f16)
      outs(%result0: tensor<2x256x16xf16>) {
   ^bb0(%score: f32):
   iree_linalg_ext.yield %score : f32
 } -> tensor<2x256x16xf16>
 return %result1: tensor<2x256x16xf16>
}
