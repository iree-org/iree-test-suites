#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d1 + d5 * 3, d2 + d6 * 3, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d0)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
module {
  func.func public @fused_op_layout_cus_10218f58bac941df596e80e7ef9d23650d3b3541_16x75x75x64xbfloat16_16x225x225x64xbfloat16_64x3x3x64xbfloat16(%arg0: !torch.vtensor<[16,75,75,64],bf16>, %arg1: !torch.vtensor<[16,225,225,64],bf16>, %arg2: !torch.vtensor<[64,3,3,64],bf16>) -> !torch.vtensor<[64,3,3,64],bf16> {
    %int0 = torch.constant.int 0
    %int3 = torch.constant.int 3
    %int1 = torch.constant.int 1
    %int2 = torch.constant.int 2
    %0 = torch.prim.ListConstruct %int0, %int3, %int1, %int2 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.permute %arg0, %0 : !torch.vtensor<[16,75,75,64],bf16>, !torch.list<int> -> !torch.vtensor<[16,64,75,75],bf16>
    %int0_0 = torch.constant.int 0
    %int3_1 = torch.constant.int 3
    %int1_2 = torch.constant.int 1
    %int2_3 = torch.constant.int 2
    %2 = torch.prim.ListConstruct %int0_0, %int3_1, %int1_2, %int2_3 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %3 = torch.aten.permute %arg1, %2 : !torch.vtensor<[16,225,225,64],bf16>, !torch.list<int> -> !torch.vtensor<[16,64,225,225],bf16>
    %int0_4 = torch.constant.int 0
    %int3_5 = torch.constant.int 3
    %int1_6 = torch.constant.int 1
    %int2_7 = torch.constant.int 2
    %4 = torch.prim.ListConstruct %int0_4, %int3_5, %int1_6, %int2_7 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %5 = torch.aten.permute %arg2, %4 : !torch.vtensor<[64,3,3,64],bf16>, !torch.list<int> -> !torch.vtensor<[64,64,3,3],bf16>
    %int0_8 = torch.constant.int 0
    %int2_9 = torch.constant.int 2
    %int3_10 = torch.constant.int 3
    %int1_11 = torch.constant.int 1
    %6 = torch.prim.ListConstruct %int0_8, %int2_9, %int3_10, %int1_11 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %7 = torch.aten.permute %1, %6 : !torch.vtensor<[16,64,75,75],bf16>, !torch.list<int> -> !torch.vtensor<[16,75,75,64],bf16>
    %int0_12 = torch.constant.int 0
    %int2_13 = torch.constant.int 2
    %int3_14 = torch.constant.int 3
    %int1_15 = torch.constant.int 1
    %8 = torch.prim.ListConstruct %int0_12, %int2_13, %int3_14, %int1_15 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %9 = torch.aten.permute %3, %8 : !torch.vtensor<[16,64,225,225],bf16>, !torch.list<int> -> !torch.vtensor<[16,225,225,64],bf16>
    %int0_16 = torch.constant.int 0
    %int2_17 = torch.constant.int 2
    %int3_18 = torch.constant.int 3
    %int1_19 = torch.constant.int 1
    %10 = torch.prim.ListConstruct %int0_16, %int2_17, %int3_18, %int1_19 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %11 = torch.aten.permute %5, %10 : !torch.vtensor<[64,64,3,3],bf16>, !torch.list<int> -> !torch.vtensor<[64,3,3,64],bf16>
    %12 = torch_c.to_builtin_tensor %7 : !torch.vtensor<[16,75,75,64],bf16> -> tensor<16x75x75x64xbf16>
    %13 = torch_c.to_builtin_tensor %9 : !torch.vtensor<[16,225,225,64],bf16> -> tensor<16x225x225x64xbf16>
    %14 = torch_c.to_builtin_tensor %11 : !torch.vtensor<[64,3,3,64],bf16> -> tensor<64x3x3x64xbf16>
    %15 = call @conv_2d_bfloat16_weight_backward_16x225x225x64_nhwc_64x3x3x64_fhwc_nhwf_3x3s_1x1p_1x1d_1g(%12, %13, %14) : (tensor<16x75x75x64xbf16>, tensor<16x225x225x64xbf16>, tensor<64x3x3x64xbf16>) -> tensor<64x3x3x64xbf16>
    %16 = torch_c.from_builtin_tensor %15 : tensor<64x3x3x64xbf16> -> !torch.vtensor<[64,3,3,64],bf16>
    %none = torch.constant.none
    %int0_20 = torch.constant.int 0
    %int3_21 = torch.constant.int 3
    %int1_22 = torch.constant.int 1
    %int2_23 = torch.constant.int 2
    %17 = torch.prim.ListConstruct %int0_20, %int3_21, %int1_22, %int2_23 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %18 = torch.aten.permute %16, %17 : !torch.vtensor<[64,3,3,64],bf16>, !torch.list<int> -> !torch.vtensor<[64,64,3,3],bf16>
    %int0_24 = torch.constant.int 0
    %int2_25 = torch.constant.int 2
    %int3_26 = torch.constant.int 3
    %int1_27 = torch.constant.int 1
    %19 = torch.prim.ListConstruct %int0_24, %int2_25, %int3_26, %int1_27 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %20 = torch.aten.permute %18, %19 : !torch.vtensor<[64,64,3,3],bf16>, !torch.list<int> -> !torch.vtensor<[64,3,3,64],bf16>
    return %20 : !torch.vtensor<[64,3,3,64],bf16>
  }
  func.func private @conv_2d_bfloat16_weight_backward_16x225x225x64_nhwc_64x3x3x64_fhwc_nhwf_3x3s_1x1p_1x1d_1g(%arg0: tensor<16x75x75x64xbf16>, %arg1: tensor<16x225x225x64xbf16>, %arg2: tensor<64x3x3x64xbf16>) -> tensor<64x3x3x64xbf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch_c.from_builtin_tensor %arg1 : tensor<16x225x225x64xbf16> -> !torch.vtensor<[16,225,225,64],bf16>
    %1 = torch_c.from_builtin_tensor %arg0 : tensor<16x75x75x64xbf16> -> !torch.vtensor<[16,75,75,64],bf16>
    %false = torch.constant.bool false
    %int15 = torch.constant.int 15
    %none = torch.constant.none
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %2 = torch.prim.ListConstruct %int0, %int0, %int1, %int1, %int1, %int1, %int0, %int0 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %3 = torch.aten.constant_pad_nd %0, %2, %int0 : !torch.vtensor<[16,225,225,64],bf16>, !torch.list<int>, !torch.int -> !torch.vtensor<[16,227,227,64],bf16>
    %4 = torch_c.to_builtin_tensor %3 : !torch.vtensor<[16,227,227,64],bf16> -> tensor<16x227x227x64xbf16>
    %5 = torch_c.to_builtin_tensor %1 : !torch.vtensor<[16,75,75,64],bf16> -> tensor<16x75x75x64xbf16>
    %6 = util.call @generic_conv_16x227x227x64xbf16_16x75x75x64xbf16_1x1S_3x3D_chwn_chwf_fhwn_f32(%4, %5) : (tensor<16x227x227x64xbf16>, tensor<16x75x75x64xbf16>) -> tensor<64x3x3x64xf32>
    %7 = torch_c.from_builtin_tensor %6 : tensor<64x3x3x64xf32> -> !torch.vtensor<[64,3,3,64],f32>
    %8 = torch.aten.to.dtype %7, %int15, %false, %false, %none : !torch.vtensor<[64,3,3,64],f32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[64,3,3,64],bf16>
    %9 = torch_c.to_builtin_tensor %8 : !torch.vtensor<[64,3,3,64],bf16> -> tensor<64x3x3x64xbf16>
    return %9 : tensor<64x3x3x64xbf16>
  }
  util.func private @generic_conv_16x227x227x64xbf16_16x75x75x64xbf16_1x1S_3x3D_chwn_chwf_fhwn_f32(%arg0: tensor<16x227x227x64xbf16>, %arg1: tensor<16x75x75x64xbf16>) -> tensor<64x3x3x64xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<64x3x3x64xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<64x3x3x64xf32>) -> tensor<64x3x3x64xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<16x227x227x64xbf16>, tensor<16x75x75x64xbf16>) outs(%1 : tensor<64x3x3x64xf32>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: f32):
      %3 = arith.extf %in : bf16 to f32
      %4 = arith.extf %in_0 : bf16 to f32
      %5 = arith.mulf %3, %4 : f32
      %6 = arith.addf %out, %5 : f32
      linalg.yield %6 : f32
    } -> tensor<64x3x3x64xf32>
    util.return %2 : tensor<64x3x3x64xf32>
  }
}
