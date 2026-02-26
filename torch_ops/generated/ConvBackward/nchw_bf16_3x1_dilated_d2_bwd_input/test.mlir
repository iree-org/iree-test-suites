#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d4, d2 + d5 * 2, d3 + d6 * 2)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d1, d5, d6)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
module {
  func.func public @fused_op_layout_cus_8a2acfcacde2ec47037c0f35501a699697d88821_16x96x48x32xbfloat16_16x96x48x32xbfloat16_96x96x3x1xbfloat16(%arg0: !torch.vtensor<[16,96,48,32],bf16>, %arg1: !torch.vtensor<[16,96,48,32],bf16>, %arg2: !torch.vtensor<[96,96,3,1],bf16>) -> !torch.vtensor<[16,96,48,32],bf16> {
    %0 = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[16,96,48,32],bf16> -> tensor<16x96x48x32xbf16>
    %1 = torch_c.to_builtin_tensor %arg1 : !torch.vtensor<[16,96,48,32],bf16> -> tensor<16x96x48x32xbf16>
    %2 = torch_c.to_builtin_tensor %arg2 : !torch.vtensor<[96,96,3,1],bf16> -> tensor<96x96x3x1xbf16>
    %3 = call @conv_2d_bfloat16_input_backward_16x96x48x32_nchw_96x96x3x1_fchw_nfhw_1x1s_2x0p_2x2d_1g(%0, %1, %2) : (tensor<16x96x48x32xbf16>, tensor<16x96x48x32xbf16>, tensor<96x96x3x1xbf16>) -> tensor<16x96x48x32xbf16>
    %4 = torch_c.from_builtin_tensor %3 : tensor<16x96x48x32xbf16> -> !torch.vtensor<[16,96,48,32],bf16>
    %none = torch.constant.none
    return %4 : !torch.vtensor<[16,96,48,32],bf16>
  }
  func.func private @conv_2d_bfloat16_input_backward_16x96x48x32_nchw_96x96x3x1_fchw_nfhw_1x1s_2x0p_2x2d_1g(%arg0: tensor<16x96x48x32xbf16>, %arg1: tensor<16x96x48x32xbf16>, %arg2: tensor<96x96x3x1xbf16>) -> tensor<16x96x48x32xbf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch_c.from_builtin_tensor %arg2 : tensor<96x96x3x1xbf16> -> !torch.vtensor<[96,96,3,1],bf16>
    %1 = torch_c.from_builtin_tensor %arg0 : tensor<16x96x48x32xbf16> -> !torch.vtensor<[16,96,48,32],bf16>
    %false = torch.constant.bool false
    %int15 = torch.constant.int 15
    %none = torch.constant.none
    %int0 = torch.constant.int 0
    %int2 = torch.constant.int 2
    %2 = torch.prim.ListConstruct %int2 : (!torch.int) -> !torch.list<int>
    %3 = torch.aten.flip %0, %2 : !torch.vtensor<[96,96,3,1],bf16>, !torch.list<int> -> !torch.vtensor<[96,96,3,1],bf16>
    %4 = torch.prim.ListConstruct %int0, %int0, %int2, %int2, %int0, %int0, %int0, %int0 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %5 = torch.aten.constant_pad_nd %1, %4, %int0 : !torch.vtensor<[16,96,48,32],bf16>, !torch.list<int>, !torch.int -> !torch.vtensor<[16,96,52,32],bf16>
    %6 = torch_c.to_builtin_tensor %5 : !torch.vtensor<[16,96,52,32],bf16> -> tensor<16x96x52x32xbf16>
    %7 = torch_c.to_builtin_tensor %3 : !torch.vtensor<[96,96,3,1],bf16> -> tensor<96x96x3x1xbf16>
    %8 = util.call @generic_conv_16x96x52x32xbf16_96x96x3x1xbf16_1x1S_2x2D_nchw_cfhw_nfhw_f32(%6, %7) : (tensor<16x96x52x32xbf16>, tensor<96x96x3x1xbf16>) -> tensor<16x96x48x32xf32>
    %9 = torch_c.from_builtin_tensor %8 : tensor<16x96x48x32xf32> -> !torch.vtensor<[16,96,48,32],f32>
    %10 = torch.aten.to.dtype %9, %int15, %false, %false, %none : !torch.vtensor<[16,96,48,32],f32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[16,96,48,32],bf16>
    %11 = torch_c.to_builtin_tensor %10 : !torch.vtensor<[16,96,48,32],bf16> -> tensor<16x96x48x32xbf16>
    return %11 : tensor<16x96x48x32xbf16>
  }
  util.func private @generic_conv_16x96x52x32xbf16_96x96x3x1xbf16_1x1S_2x2D_nchw_cfhw_nfhw_f32(%arg0: tensor<16x96x52x32xbf16>, %arg1: tensor<96x96x3x1xbf16>) -> tensor<16x96x48x32xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<16x96x48x32xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<16x96x48x32xf32>) -> tensor<16x96x48x32xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<16x96x52x32xbf16>, tensor<96x96x3x1xbf16>) outs(%1 : tensor<16x96x48x32xf32>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: f32):
      %3 = arith.extf %in : bf16 to f32
      %4 = arith.extf %in_0 : bf16 to f32
      %5 = arith.mulf %3, %4 : f32
      %6 = arith.addf %out, %5 : f32
      linalg.yield %6 : f32
    } -> tensor<16x96x48x32xf32>
    util.return %2 : tensor<16x96x48x32xf32>
  }
}
