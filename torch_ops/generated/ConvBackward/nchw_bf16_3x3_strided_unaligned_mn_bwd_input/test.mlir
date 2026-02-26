#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d4, d2 + d5, d3 + d6)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d1, d5, d6)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
module {
  func.func public @fused_op_layout_cus_f75bbe5bbd611c0854c075c62904b2b2177b94d8_16x64x75x75xbfloat16_16x64x225x225xbfloat16_64x64x3x3xbfloat16(%arg0: !torch.vtensor<[16,64,75,75],bf16>, %arg1: !torch.vtensor<[16,64,225,225],bf16>, %arg2: !torch.vtensor<[64,64,3,3],bf16>) -> !torch.vtensor<[16,64,225,225],bf16> {
    %0 = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[16,64,75,75],bf16> -> tensor<16x64x75x75xbf16>
    %1 = torch_c.to_builtin_tensor %arg1 : !torch.vtensor<[16,64,225,225],bf16> -> tensor<16x64x225x225xbf16>
    %2 = torch_c.to_builtin_tensor %arg2 : !torch.vtensor<[64,64,3,3],bf16> -> tensor<64x64x3x3xbf16>
    %3 = call @conv_2d_bfloat16_input_backward_16x64x225x225_nchw_64x64x3x3_fchw_nfhw_3x3s_1x1p_1x1d_1g(%0, %1, %2) : (tensor<16x64x75x75xbf16>, tensor<16x64x225x225xbf16>, tensor<64x64x3x3xbf16>) -> tensor<16x64x225x225xbf16>
    %4 = torch_c.from_builtin_tensor %3 : tensor<16x64x225x225xbf16> -> !torch.vtensor<[16,64,225,225],bf16>
    %none = torch.constant.none
    return %4 : !torch.vtensor<[16,64,225,225],bf16>
  }
  func.func private @conv_2d_bfloat16_input_backward_16x64x225x225_nchw_64x64x3x3_fchw_nfhw_3x3s_1x1p_1x1d_1g(%arg0: tensor<16x64x75x75xbf16>, %arg1: tensor<16x64x225x225xbf16>, %arg2: tensor<64x64x3x3xbf16>) -> tensor<16x64x225x225xbf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch_c.from_builtin_tensor %arg2 : tensor<64x64x3x3xbf16> -> !torch.vtensor<[64,64,3,3],bf16>
    %1 = torch_c.from_builtin_tensor %arg0 : tensor<16x64x75x75xbf16> -> !torch.vtensor<[16,64,75,75],bf16>
    %2 = torch.vtensor.literal(dense<0.000000e+00> : tensor<16x64x227x227xbf16>) : !torch.vtensor<[16,64,227,227],bf16>
    %false = torch.constant.bool false
    %none = torch.constant.none
    %int15 = torch.constant.int 15
    %int2 = torch.constant.int 2
    %int3 = torch.constant.int 3
    %3 = torch.prim.ListConstruct %int2, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
    %4 = torch.aten.flip %0, %3 : !torch.vtensor<[64,64,3,3],bf16>, !torch.list<int> -> !torch.vtensor<[64,64,3,3],bf16>
    %5 = torch_c.to_builtin_tensor %1 : !torch.vtensor<[16,64,75,75],bf16> -> tensor<16x64x75x75xbf16>
    %6 = torch_c.to_builtin_tensor %2 : !torch.vtensor<[16,64,227,227],bf16> -> tensor<16x64x227x227xbf16>
    %7 = util.call @insert_slice_16x64x75x75xbf16_into_16x64x227x227xbf16_0_0_1_1_offset_1_1_3_3_stride(%5, %6) : (tensor<16x64x75x75xbf16>, tensor<16x64x227x227xbf16>) -> tensor<16x64x227x227xbf16>
    %8 = torch_c.from_builtin_tensor %7 : tensor<16x64x227x227xbf16> -> !torch.vtensor<[16,64,227,227],bf16>
    %9 = torch_c.to_builtin_tensor %8 : !torch.vtensor<[16,64,227,227],bf16> -> tensor<16x64x227x227xbf16>
    %10 = torch_c.to_builtin_tensor %4 : !torch.vtensor<[64,64,3,3],bf16> -> tensor<64x64x3x3xbf16>
    %11 = util.call @generic_conv_16x64x227x227xbf16_64x64x3x3xbf16_1x1S_1x1D_nchw_cfhw_nfhw_f32(%9, %10) : (tensor<16x64x227x227xbf16>, tensor<64x64x3x3xbf16>) -> tensor<16x64x225x225xf32>
    %12 = torch_c.from_builtin_tensor %11 : tensor<16x64x225x225xf32> -> !torch.vtensor<[16,64,225,225],f32>
    %13 = torch.aten.to.dtype %12, %int15, %false, %false, %none : !torch.vtensor<[16,64,225,225],f32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[16,64,225,225],bf16>
    %14 = torch_c.to_builtin_tensor %13 : !torch.vtensor<[16,64,225,225],bf16> -> tensor<16x64x225x225xbf16>
    return %14 : tensor<16x64x225x225xbf16>
  }
  util.func private @insert_slice_16x64x75x75xbf16_into_16x64x227x227xbf16_0_0_1_1_offset_1_1_3_3_stride(%arg0: tensor<16x64x75x75xbf16>, %arg1: tensor<16x64x227x227xbf16>) -> tensor<16x64x227x227xbf16> {
    %inserted_slice = tensor.insert_slice %arg0 into %arg1[0, 0, 1, 1] [16, 64, 75, 75] [1, 1, 3, 3] : tensor<16x64x75x75xbf16> into tensor<16x64x227x227xbf16>
    util.return %inserted_slice : tensor<16x64x227x227xbf16>
  }
  util.func private @generic_conv_16x64x227x227xbf16_64x64x3x3xbf16_1x1S_1x1D_nchw_cfhw_nfhw_f32(%arg0: tensor<16x64x227x227xbf16>, %arg1: tensor<64x64x3x3xbf16>) -> tensor<16x64x225x225xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<16x64x225x225xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<16x64x225x225xf32>) -> tensor<16x64x225x225xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<16x64x227x227xbf16>, tensor<64x64x3x3xbf16>) outs(%1 : tensor<16x64x225x225xf32>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: f32):
      %3 = arith.extf %in : bf16 to f32
      %4 = arith.extf %in_0 : bf16 to f32
      %5 = arith.mulf %3, %4 : f32
      %6 = arith.addf %out, %5 : f32
      linalg.yield %6 : f32
    } -> tensor<16x64x225x225xf32>
    util.return %2 : tensor<16x64x225x225xf32>
  }
}
