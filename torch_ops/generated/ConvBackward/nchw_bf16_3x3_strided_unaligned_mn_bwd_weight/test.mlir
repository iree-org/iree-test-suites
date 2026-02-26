#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d1, d2 + d5 * 3, d3 + d6 * 3)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d0, d5, d6)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
module {
  func.func public @fused_op_layout_cus_e1e0b9db3cdd498d4f2b06aa16a6ccc2f3c40689_16x64x75x75xbfloat16_16x64x225x225xbfloat16_64x64x3x3xbfloat16(%arg0: !torch.vtensor<[16,64,75,75],bf16>, %arg1: !torch.vtensor<[16,64,225,225],bf16>, %arg2: !torch.vtensor<[64,64,3,3],bf16>) -> !torch.vtensor<[64,64,3,3],bf16> {
    %0 = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[16,64,75,75],bf16> -> tensor<16x64x75x75xbf16>
    %1 = torch_c.to_builtin_tensor %arg1 : !torch.vtensor<[16,64,225,225],bf16> -> tensor<16x64x225x225xbf16>
    %2 = torch_c.to_builtin_tensor %arg2 : !torch.vtensor<[64,64,3,3],bf16> -> tensor<64x64x3x3xbf16>
    %3 = call @conv_2d_bfloat16_weight_backward_16x64x225x225_nchw_64x64x3x3_fchw_nfhw_3x3s_1x1p_1x1d_1g(%0, %1, %2) : (tensor<16x64x75x75xbf16>, tensor<16x64x225x225xbf16>, tensor<64x64x3x3xbf16>) -> tensor<64x64x3x3xbf16>
    %4 = torch_c.from_builtin_tensor %3 : tensor<64x64x3x3xbf16> -> !torch.vtensor<[64,64,3,3],bf16>
    %none = torch.constant.none
    return %4 : !torch.vtensor<[64,64,3,3],bf16>
  }
  func.func private @conv_2d_bfloat16_weight_backward_16x64x225x225_nchw_64x64x3x3_fchw_nfhw_3x3s_1x1p_1x1d_1g(%arg0: tensor<16x64x75x75xbf16>, %arg1: tensor<16x64x225x225xbf16>, %arg2: tensor<64x64x3x3xbf16>) -> tensor<64x64x3x3xbf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch_c.from_builtin_tensor %arg1 : tensor<16x64x225x225xbf16> -> !torch.vtensor<[16,64,225,225],bf16>
    %1 = torch_c.from_builtin_tensor %arg0 : tensor<16x64x75x75xbf16> -> !torch.vtensor<[16,64,75,75],bf16>
    %false = torch.constant.bool false
    %int15 = torch.constant.int 15
    %none = torch.constant.none
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %2 = torch.prim.ListConstruct %int1, %int1, %int1, %int1, %int0, %int0, %int0, %int0 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %3 = torch.aten.constant_pad_nd %0, %2, %int0 : !torch.vtensor<[16,64,225,225],bf16>, !torch.list<int>, !torch.int -> !torch.vtensor<[16,64,227,227],bf16>
    %4 = torch_c.to_builtin_tensor %3 : !torch.vtensor<[16,64,227,227],bf16> -> tensor<16x64x227x227xbf16>
    %5 = torch_c.to_builtin_tensor %1 : !torch.vtensor<[16,64,75,75],bf16> -> tensor<16x64x75x75xbf16>
    %6 = util.call @generic_conv_16x64x227x227xbf16_16x64x75x75xbf16_1x1S_3x3D_cnhw_cfhw_fnhw_f32(%4, %5) : (tensor<16x64x227x227xbf16>, tensor<16x64x75x75xbf16>) -> tensor<64x64x3x3xf32>
    %7 = torch_c.from_builtin_tensor %6 : tensor<64x64x3x3xf32> -> !torch.vtensor<[64,64,3,3],f32>
    %8 = torch.aten.to.dtype %7, %int15, %false, %false, %none : !torch.vtensor<[64,64,3,3],f32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[64,64,3,3],bf16>
    %9 = torch_c.to_builtin_tensor %8 : !torch.vtensor<[64,64,3,3],bf16> -> tensor<64x64x3x3xbf16>
    return %9 : tensor<64x64x3x3xbf16>
  }
  util.func private @generic_conv_16x64x227x227xbf16_16x64x75x75xbf16_1x1S_3x3D_cnhw_cfhw_fnhw_f32(%arg0: tensor<16x64x227x227xbf16>, %arg1: tensor<16x64x75x75xbf16>) -> tensor<64x64x3x3xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<64x64x3x3xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<16x64x227x227xbf16>, tensor<16x64x75x75xbf16>) outs(%1 : tensor<64x64x3x3xf32>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: f32):
      %3 = arith.extf %in : bf16 to f32
      %4 = arith.extf %in_0 : bf16 to f32
      %5 = arith.mulf %3, %4 : f32
      %6 = arith.addf %out, %5 : f32
      linalg.yield %6 : f32
    } -> tensor<64x64x3x3xf32>
    util.return %2 : tensor<64x64x3x3xf32>
  }
}
