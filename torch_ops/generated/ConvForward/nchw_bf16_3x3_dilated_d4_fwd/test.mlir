#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d4, d2 + d5 * 4, d3 + d6 * 4)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d4, d5, d6)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
module {
  func.func public @fused_op_layout_cus_e66ca4caf0fe622a0498eb149598510f3cb5f14f_16x48x48x32xbfloat16_48x48x3x3xbfloat16(%arg0: !torch.vtensor<[16,48,48,32],bf16>, %arg1: !torch.vtensor<[48,48,3,3],bf16>) -> !torch.vtensor<[16,48,48,32],bf16> {
    %0 = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[16,48,48,32],bf16> -> tensor<16x48x48x32xbf16>
    %1 = torch_c.to_builtin_tensor %arg1 : !torch.vtensor<[48,48,3,3],bf16> -> tensor<48x48x3x3xbf16>
    %2 = call @conv_2d_bfloat16_forward_16x48x48x32_nchw_48x48x3x3_fchw_nfhw_1x1s_4x4p_4x4d_1g(%0, %1) : (tensor<16x48x48x32xbf16>, tensor<48x48x3x3xbf16>) -> tensor<16x48x48x32xbf16>
    %3 = torch_c.from_builtin_tensor %2 : tensor<16x48x48x32xbf16> -> !torch.vtensor<[16,48,48,32],bf16>
    return %3 : !torch.vtensor<[16,48,48,32],bf16>
  }
  func.func private @conv_2d_bfloat16_forward_16x48x48x32_nchw_48x48x3x3_fchw_nfhw_1x1s_4x4p_4x4d_1g(%arg0: tensor<16x48x48x32xbf16>, %arg1: tensor<48x48x3x3xbf16>) -> tensor<16x48x48x32xbf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch_c.from_builtin_tensor %arg1 : tensor<48x48x3x3xbf16> -> !torch.vtensor<[48,48,3,3],bf16>
    %1 = torch_c.from_builtin_tensor %arg0 : tensor<16x48x48x32xbf16> -> !torch.vtensor<[16,48,48,32],bf16>
    %false = torch.constant.bool false
    %int15 = torch.constant.int 15
    %none = torch.constant.none
    %int4 = torch.constant.int 4
    %int0 = torch.constant.int 0
    %2 = torch.prim.ListConstruct %int4, %int4, %int4, %int4, %int0, %int0, %int0, %int0 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %3 = torch.aten.constant_pad_nd %1, %2, %int0 : !torch.vtensor<[16,48,48,32],bf16>, !torch.list<int>, !torch.int -> !torch.vtensor<[16,48,56,40],bf16>
    %4 = torch_c.to_builtin_tensor %3 : !torch.vtensor<[16,48,56,40],bf16> -> tensor<16x48x56x40xbf16>
    %5 = torch_c.to_builtin_tensor %0 : !torch.vtensor<[48,48,3,3],bf16> -> tensor<48x48x3x3xbf16>
    %6 = util.call @generic_conv_16x48x56x40xbf16_48x48x3x3xbf16_1x1S_4x4D_nchw_fchw_nfhw_f32(%4, %5) : (tensor<16x48x56x40xbf16>, tensor<48x48x3x3xbf16>) -> tensor<16x48x48x32xf32>
    %7 = torch_c.from_builtin_tensor %6 : tensor<16x48x48x32xf32> -> !torch.vtensor<[16,48,48,32],f32>
    %8 = torch.aten.to.dtype %7, %int15, %false, %false, %none : !torch.vtensor<[16,48,48,32],f32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[16,48,48,32],bf16>
    %9 = torch_c.to_builtin_tensor %8 : !torch.vtensor<[16,48,48,32],bf16> -> tensor<16x48x48x32xbf16>
    return %9 : tensor<16x48x48x32xbf16>
  }
  util.func private @generic_conv_16x48x56x40xbf16_48x48x3x3xbf16_1x1S_4x4D_nchw_fchw_nfhw_f32(%arg0: tensor<16x48x56x40xbf16>, %arg1: tensor<48x48x3x3xbf16>) -> tensor<16x48x48x32xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<16x48x48x32xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<16x48x48x32xf32>) -> tensor<16x48x48x32xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<16x48x56x40xbf16>, tensor<48x48x3x3xbf16>) outs(%1 : tensor<16x48x48x32xf32>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: f32):
      %3 = arith.extf %in : bf16 to f32
      %4 = arith.extf %in_0 : bf16 to f32
      %5 = arith.mulf %3, %4 : f32
      %6 = arith.addf %out, %5 : f32
      linalg.yield %6 : f32
    } -> tensor<16x48x48x32xf32>
    util.return %2 : tensor<16x48x48x32xf32>
  }
}
