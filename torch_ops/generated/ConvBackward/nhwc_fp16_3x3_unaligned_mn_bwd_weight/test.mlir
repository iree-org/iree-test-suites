#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d1 + d5, d2 + d6, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d0)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
module {
  func.func public @fused_op_layout_cus_3d6cae6e609965dae93f0fd2b743cba78a9c6844_32x25x25x256xfloat16_32x25x25x256xfloat16_256x3x3x256xfloat16(%arg0: !torch.vtensor<[32,25,25,256],f16>, %arg1: !torch.vtensor<[32,25,25,256],f16>, %arg2: !torch.vtensor<[256,3,3,256],f16>) -> !torch.vtensor<[256,3,3,256],f16> {
    %int0 = torch.constant.int 0
    %int3 = torch.constant.int 3
    %int1 = torch.constant.int 1
    %int2 = torch.constant.int 2
    %0 = torch.prim.ListConstruct %int0, %int3, %int1, %int2 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.permute %arg0, %0 : !torch.vtensor<[32,25,25,256],f16>, !torch.list<int> -> !torch.vtensor<[32,256,25,25],f16>
    %int0_0 = torch.constant.int 0
    %int3_1 = torch.constant.int 3
    %int1_2 = torch.constant.int 1
    %int2_3 = torch.constant.int 2
    %2 = torch.prim.ListConstruct %int0_0, %int3_1, %int1_2, %int2_3 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %3 = torch.aten.permute %arg1, %2 : !torch.vtensor<[32,25,25,256],f16>, !torch.list<int> -> !torch.vtensor<[32,256,25,25],f16>
    %int0_4 = torch.constant.int 0
    %int3_5 = torch.constant.int 3
    %int1_6 = torch.constant.int 1
    %int2_7 = torch.constant.int 2
    %4 = torch.prim.ListConstruct %int0_4, %int3_5, %int1_6, %int2_7 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %5 = torch.aten.permute %arg2, %4 : !torch.vtensor<[256,3,3,256],f16>, !torch.list<int> -> !torch.vtensor<[256,256,3,3],f16>
    %int0_8 = torch.constant.int 0
    %int2_9 = torch.constant.int 2
    %int3_10 = torch.constant.int 3
    %int1_11 = torch.constant.int 1
    %6 = torch.prim.ListConstruct %int0_8, %int2_9, %int3_10, %int1_11 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %7 = torch.aten.permute %1, %6 : !torch.vtensor<[32,256,25,25],f16>, !torch.list<int> -> !torch.vtensor<[32,25,25,256],f16>
    %int0_12 = torch.constant.int 0
    %int2_13 = torch.constant.int 2
    %int3_14 = torch.constant.int 3
    %int1_15 = torch.constant.int 1
    %8 = torch.prim.ListConstruct %int0_12, %int2_13, %int3_14, %int1_15 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %9 = torch.aten.permute %3, %8 : !torch.vtensor<[32,256,25,25],f16>, !torch.list<int> -> !torch.vtensor<[32,25,25,256],f16>
    %int0_16 = torch.constant.int 0
    %int2_17 = torch.constant.int 2
    %int3_18 = torch.constant.int 3
    %int1_19 = torch.constant.int 1
    %10 = torch.prim.ListConstruct %int0_16, %int2_17, %int3_18, %int1_19 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %11 = torch.aten.permute %5, %10 : !torch.vtensor<[256,256,3,3],f16>, !torch.list<int> -> !torch.vtensor<[256,3,3,256],f16>
    %12 = torch_c.to_builtin_tensor %7 : !torch.vtensor<[32,25,25,256],f16> -> tensor<32x25x25x256xf16>
    %13 = torch_c.to_builtin_tensor %9 : !torch.vtensor<[32,25,25,256],f16> -> tensor<32x25x25x256xf16>
    %14 = torch_c.to_builtin_tensor %11 : !torch.vtensor<[256,3,3,256],f16> -> tensor<256x3x3x256xf16>
    %15 = call @conv_2d_float16_weight_backward_32x25x25x256_nhwc_256x3x3x256_fhwc_nhwf_1x1s_1x1p_1x1d_1g(%12, %13, %14) : (tensor<32x25x25x256xf16>, tensor<32x25x25x256xf16>, tensor<256x3x3x256xf16>) -> tensor<256x3x3x256xf16>
    %16 = torch_c.from_builtin_tensor %15 : tensor<256x3x3x256xf16> -> !torch.vtensor<[256,3,3,256],f16>
    %none = torch.constant.none
    %int0_20 = torch.constant.int 0
    %int3_21 = torch.constant.int 3
    %int1_22 = torch.constant.int 1
    %int2_23 = torch.constant.int 2
    %17 = torch.prim.ListConstruct %int0_20, %int3_21, %int1_22, %int2_23 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %18 = torch.aten.permute %16, %17 : !torch.vtensor<[256,3,3,256],f16>, !torch.list<int> -> !torch.vtensor<[256,256,3,3],f16>
    %int0_24 = torch.constant.int 0
    %int2_25 = torch.constant.int 2
    %int3_26 = torch.constant.int 3
    %int1_27 = torch.constant.int 1
    %19 = torch.prim.ListConstruct %int0_24, %int2_25, %int3_26, %int1_27 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %20 = torch.aten.permute %18, %19 : !torch.vtensor<[256,256,3,3],f16>, !torch.list<int> -> !torch.vtensor<[256,3,3,256],f16>
    return %20 : !torch.vtensor<[256,3,3,256],f16>
  }
  func.func private @conv_2d_float16_weight_backward_32x25x25x256_nhwc_256x3x3x256_fhwc_nhwf_1x1s_1x1p_1x1d_1g(%arg0: tensor<32x25x25x256xf16>, %arg1: tensor<32x25x25x256xf16>, %arg2: tensor<256x3x3x256xf16>) -> tensor<256x3x3x256xf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch_c.from_builtin_tensor %arg1 : tensor<32x25x25x256xf16> -> !torch.vtensor<[32,25,25,256],f16>
    %1 = torch_c.from_builtin_tensor %arg0 : tensor<32x25x25x256xf16> -> !torch.vtensor<[32,25,25,256],f16>
    %false = torch.constant.bool false
    %int5 = torch.constant.int 5
    %none = torch.constant.none
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %2 = torch.prim.ListConstruct %int0, %int0, %int1, %int1, %int1, %int1, %int0, %int0 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %3 = torch.aten.constant_pad_nd %0, %2, %int0 : !torch.vtensor<[32,25,25,256],f16>, !torch.list<int>, !torch.int -> !torch.vtensor<[32,27,27,256],f16>
    %4 = torch_c.to_builtin_tensor %3 : !torch.vtensor<[32,27,27,256],f16> -> tensor<32x27x27x256xf16>
    %5 = torch_c.to_builtin_tensor %1 : !torch.vtensor<[32,25,25,256],f16> -> tensor<32x25x25x256xf16>
    %6 = util.call @generic_conv_32x27x27x256xf16_32x25x25x256xf16_1x1S_1x1D_chwn_chwf_fhwn_f32(%4, %5) : (tensor<32x27x27x256xf16>, tensor<32x25x25x256xf16>) -> tensor<256x3x3x256xf32>
    %7 = torch_c.from_builtin_tensor %6 : tensor<256x3x3x256xf32> -> !torch.vtensor<[256,3,3,256],f32>
    %8 = torch.aten.to.dtype %7, %int5, %false, %false, %none : !torch.vtensor<[256,3,3,256],f32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[256,3,3,256],f16>
    %9 = torch_c.to_builtin_tensor %8 : !torch.vtensor<[256,3,3,256],f16> -> tensor<256x3x3x256xf16>
    return %9 : tensor<256x3x3x256xf16>
  }
  util.func private @generic_conv_32x27x27x256xf16_32x25x25x256xf16_1x1S_1x1D_chwn_chwf_fhwn_f32(%arg0: tensor<32x27x27x256xf16>, %arg1: tensor<32x25x25x256xf16>) -> tensor<256x3x3x256xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<256x3x3x256xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<256x3x3x256xf32>) -> tensor<256x3x3x256xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<32x27x27x256xf16>, tensor<32x25x25x256xf16>) outs(%1 : tensor<256x3x3x256xf32>) {
    ^bb0(%in: f16, %in_0: f16, %out: f32):
      %3 = arith.extf %in : f16 to f32
      %4 = arith.extf %in_0 : f16 to f32
      %5 = arith.mulf %3, %4 : f32
      %6 = arith.addf %out, %5 : f32
      linalg.yield %6 : f32
    } -> tensor<256x3x3x256xf32>
    util.return %2 : tensor<256x3x3x256xf32>
  }
}
