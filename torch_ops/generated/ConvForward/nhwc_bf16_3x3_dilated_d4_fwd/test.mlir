#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4 * 4, d2 + d5 * 4, d6)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d5, d6)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
module {
  func.func public @fused_op_layout_cus_2e6de8cc7aa9449e1ee3554de6f8b2fdaa35c0db_16x48x32x48xbfloat16_48x3x3x48xbfloat16(%arg0: !torch.vtensor<[16,48,32,48],bf16>, %arg1: !torch.vtensor<[48,3,3,48],bf16>) -> !torch.vtensor<[16,48,32,48],bf16> {
    %int0 = torch.constant.int 0
    %int3 = torch.constant.int 3
    %int1 = torch.constant.int 1
    %int2 = torch.constant.int 2
    %0 = torch.prim.ListConstruct %int0, %int3, %int1, %int2 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.permute %arg0, %0 : !torch.vtensor<[16,48,32,48],bf16>, !torch.list<int> -> !torch.vtensor<[16,48,48,32],bf16>
    %int0_0 = torch.constant.int 0
    %int3_1 = torch.constant.int 3
    %int1_2 = torch.constant.int 1
    %int2_3 = torch.constant.int 2
    %2 = torch.prim.ListConstruct %int0_0, %int3_1, %int1_2, %int2_3 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %3 = torch.aten.permute %arg1, %2 : !torch.vtensor<[48,3,3,48],bf16>, !torch.list<int> -> !torch.vtensor<[48,48,3,3],bf16>
    %int0_4 = torch.constant.int 0
    %int2_5 = torch.constant.int 2
    %int3_6 = torch.constant.int 3
    %int1_7 = torch.constant.int 1
    %4 = torch.prim.ListConstruct %int0_4, %int2_5, %int3_6, %int1_7 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %5 = torch.aten.permute %1, %4 : !torch.vtensor<[16,48,48,32],bf16>, !torch.list<int> -> !torch.vtensor<[16,48,32,48],bf16>
    %int0_8 = torch.constant.int 0
    %int2_9 = torch.constant.int 2
    %int3_10 = torch.constant.int 3
    %int1_11 = torch.constant.int 1
    %6 = torch.prim.ListConstruct %int0_8, %int2_9, %int3_10, %int1_11 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %7 = torch.aten.permute %3, %6 : !torch.vtensor<[48,48,3,3],bf16>, !torch.list<int> -> !torch.vtensor<[48,3,3,48],bf16>
    %8 = torch_c.to_builtin_tensor %5 : !torch.vtensor<[16,48,32,48],bf16> -> tensor<16x48x32x48xbf16>
    %9 = torch_c.to_builtin_tensor %7 : !torch.vtensor<[48,3,3,48],bf16> -> tensor<48x3x3x48xbf16>
    %10 = call @conv_2d_bfloat16_forward_16x48x32x48_nhwc_48x3x3x48_fhwc_nhwf_1x1s_4x4p_4x4d_1g(%8, %9) : (tensor<16x48x32x48xbf16>, tensor<48x3x3x48xbf16>) -> tensor<16x48x32x48xbf16>
    %11 = torch_c.from_builtin_tensor %10 : tensor<16x48x32x48xbf16> -> !torch.vtensor<[16,48,32,48],bf16>
    %int0_12 = torch.constant.int 0
    %int3_13 = torch.constant.int 3
    %int1_14 = torch.constant.int 1
    %int2_15 = torch.constant.int 2
    %12 = torch.prim.ListConstruct %int0_12, %int3_13, %int1_14, %int2_15 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %13 = torch.aten.permute %11, %12 : !torch.vtensor<[16,48,32,48],bf16>, !torch.list<int> -> !torch.vtensor<[16,48,48,32],bf16>
    %int0_16 = torch.constant.int 0
    %int2_17 = torch.constant.int 2
    %int3_18 = torch.constant.int 3
    %int1_19 = torch.constant.int 1
    %14 = torch.prim.ListConstruct %int0_16, %int2_17, %int3_18, %int1_19 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %15 = torch.aten.permute %13, %14 : !torch.vtensor<[16,48,48,32],bf16>, !torch.list<int> -> !torch.vtensor<[16,48,32,48],bf16>
    return %15 : !torch.vtensor<[16,48,32,48],bf16>
  }
  func.func private @conv_2d_bfloat16_forward_16x48x32x48_nhwc_48x3x3x48_fhwc_nhwf_1x1s_4x4p_4x4d_1g(%arg0: tensor<16x48x32x48xbf16>, %arg1: tensor<48x3x3x48xbf16>) -> tensor<16x48x32x48xbf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch_c.from_builtin_tensor %arg1 : tensor<48x3x3x48xbf16> -> !torch.vtensor<[48,3,3,48],bf16>
    %1 = torch_c.from_builtin_tensor %arg0 : tensor<16x48x32x48xbf16> -> !torch.vtensor<[16,48,32,48],bf16>
    %false = torch.constant.bool false
    %int15 = torch.constant.int 15
    %none = torch.constant.none
    %int0 = torch.constant.int 0
    %int4 = torch.constant.int 4
    %2 = torch.prim.ListConstruct %int0, %int0, %int4, %int4, %int4, %int4, %int0, %int0 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %3 = torch.aten.constant_pad_nd %1, %2, %int0 : !torch.vtensor<[16,48,32,48],bf16>, !torch.list<int>, !torch.int -> !torch.vtensor<[16,56,40,48],bf16>
    %4 = torch_c.to_builtin_tensor %3 : !torch.vtensor<[16,56,40,48],bf16> -> tensor<16x56x40x48xbf16>
    %5 = torch_c.to_builtin_tensor %0 : !torch.vtensor<[48,3,3,48],bf16> -> tensor<48x3x3x48xbf16>
    %6 = util.call @generic_conv_16x56x40x48xbf16_48x3x3x48xbf16_1x1S_4x4D_nhwc_fhwc_nhwf_f32(%4, %5) : (tensor<16x56x40x48xbf16>, tensor<48x3x3x48xbf16>) -> tensor<16x48x32x48xf32>
    %7 = torch_c.from_builtin_tensor %6 : tensor<16x48x32x48xf32> -> !torch.vtensor<[16,48,32,48],f32>
    %8 = torch.aten.to.dtype %7, %int15, %false, %false, %none : !torch.vtensor<[16,48,32,48],f32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[16,48,32,48],bf16>
    %9 = torch_c.to_builtin_tensor %8 : !torch.vtensor<[16,48,32,48],bf16> -> tensor<16x48x32x48xbf16>
    return %9 : tensor<16x48x32x48xbf16>
  }
  util.func private @generic_conv_16x56x40x48xbf16_48x3x3x48xbf16_1x1S_4x4D_nhwc_fhwc_nhwf_f32(%arg0: tensor<16x56x40x48xbf16>, %arg1: tensor<48x3x3x48xbf16>) -> tensor<16x48x32x48xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<16x48x32x48xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<16x48x32x48xf32>) -> tensor<16x48x32x48xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<16x56x40x48xbf16>, tensor<48x3x3x48xbf16>) outs(%1 : tensor<16x48x32x48xf32>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: f32):
      %3 = arith.extf %in : bf16 to f32
      %4 = arith.extf %in_0 : bf16 to f32
      %5 = arith.mulf %3, %4 : f32
      %6 = arith.addf %out, %5 : f32
      linalg.yield %6 : f32
    } -> tensor<16x48x32x48xf32>
    util.return %2 : tensor<16x48x32x48xf32>
  }
}
