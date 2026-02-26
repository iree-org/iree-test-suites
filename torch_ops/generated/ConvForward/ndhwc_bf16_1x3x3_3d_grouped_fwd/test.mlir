#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d0, d1 + d6, d2 + d7, d3 + d8, d4, d9)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d4, d5, d6, d7, d8, d9)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d0, d1, d2, d3, d4, d5)>
module {
  func.func public @fused_op_layout_cus_a1c8a5d197f7991863461ff95bef31c45fd1f778_16x8x48x32x288xbfloat16_288x1x3x3x96xbfloat16(%arg0: !torch.vtensor<[16,8,48,32,288],bf16>, %arg1: !torch.vtensor<[288,1,3,3,96],bf16>) -> !torch.vtensor<[16,8,48,32,288],bf16> {
    %int0 = torch.constant.int 0
    %int4 = torch.constant.int 4
    %int1 = torch.constant.int 1
    %int2 = torch.constant.int 2
    %int3 = torch.constant.int 3
    %0 = torch.prim.ListConstruct %int0, %int4, %int1, %int2, %int3 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.permute %arg0, %0 : !torch.vtensor<[16,8,48,32,288],bf16>, !torch.list<int> -> !torch.vtensor<[16,288,8,48,32],bf16>
    %int0_0 = torch.constant.int 0
    %int4_1 = torch.constant.int 4
    %int1_2 = torch.constant.int 1
    %int2_3 = torch.constant.int 2
    %int3_4 = torch.constant.int 3
    %2 = torch.prim.ListConstruct %int0_0, %int4_1, %int1_2, %int2_3, %int3_4 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %3 = torch.aten.permute %arg1, %2 : !torch.vtensor<[288,1,3,3,96],bf16>, !torch.list<int> -> !torch.vtensor<[288,96,1,3,3],bf16>
    %int0_5 = torch.constant.int 0
    %int2_6 = torch.constant.int 2
    %int3_7 = torch.constant.int 3
    %int4_8 = torch.constant.int 4
    %int1_9 = torch.constant.int 1
    %4 = torch.prim.ListConstruct %int0_5, %int2_6, %int3_7, %int4_8, %int1_9 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %5 = torch.aten.permute %1, %4 : !torch.vtensor<[16,288,8,48,32],bf16>, !torch.list<int> -> !torch.vtensor<[16,8,48,32,288],bf16>
    %int0_10 = torch.constant.int 0
    %int2_11 = torch.constant.int 2
    %int3_12 = torch.constant.int 3
    %int4_13 = torch.constant.int 4
    %int1_14 = torch.constant.int 1
    %6 = torch.prim.ListConstruct %int0_10, %int2_11, %int3_12, %int4_13, %int1_14 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %7 = torch.aten.permute %3, %6 : !torch.vtensor<[288,96,1,3,3],bf16>, !torch.list<int> -> !torch.vtensor<[288,1,3,3,96],bf16>
    %8 = torch_c.to_builtin_tensor %5 : !torch.vtensor<[16,8,48,32,288],bf16> -> tensor<16x8x48x32x288xbf16>
    %9 = torch_c.to_builtin_tensor %7 : !torch.vtensor<[288,1,3,3,96],bf16> -> tensor<288x1x3x3x96xbf16>
    %10 = call @conv_3d_bfloat16_forward_16x8x48x32x288_ndhwc_288x1x3x3x96_fdhwc_ndhwf_1x1x1s_0x1x1p_1x1x1d_3g(%8, %9) : (tensor<16x8x48x32x288xbf16>, tensor<288x1x3x3x96xbf16>) -> tensor<16x8x48x32x288xbf16>
    %11 = torch_c.from_builtin_tensor %10 : tensor<16x8x48x32x288xbf16> -> !torch.vtensor<[16,8,48,32,288],bf16>
    %int0_15 = torch.constant.int 0
    %int4_16 = torch.constant.int 4
    %int1_17 = torch.constant.int 1
    %int2_18 = torch.constant.int 2
    %int3_19 = torch.constant.int 3
    %12 = torch.prim.ListConstruct %int0_15, %int4_16, %int1_17, %int2_18, %int3_19 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %13 = torch.aten.permute %11, %12 : !torch.vtensor<[16,8,48,32,288],bf16>, !torch.list<int> -> !torch.vtensor<[16,288,8,48,32],bf16>
    %int0_20 = torch.constant.int 0
    %int2_21 = torch.constant.int 2
    %int3_22 = torch.constant.int 3
    %int4_23 = torch.constant.int 4
    %int1_24 = torch.constant.int 1
    %14 = torch.prim.ListConstruct %int0_20, %int2_21, %int3_22, %int4_23, %int1_24 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %15 = torch.aten.permute %13, %14 : !torch.vtensor<[16,288,8,48,32],bf16>, !torch.list<int> -> !torch.vtensor<[16,8,48,32,288],bf16>
    return %15 : !torch.vtensor<[16,8,48,32,288],bf16>
  }
  func.func private @conv_3d_bfloat16_forward_16x8x48x32x288_ndhwc_288x1x3x3x96_fdhwc_ndhwf_1x1x1s_0x1x1p_1x1x1d_3g(%arg0: tensor<16x8x48x32x288xbf16>, %arg1: tensor<288x1x3x3x96xbf16>) -> tensor<16x8x48x32x288xbf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch_c.from_builtin_tensor %arg1 : tensor<288x1x3x3x96xbf16> -> !torch.vtensor<[288,1,3,3,96],bf16>
    %1 = torch_c.from_builtin_tensor %arg0 : tensor<16x8x48x32x288xbf16> -> !torch.vtensor<[16,8,48,32,288],bf16>
    %int5 = torch.constant.int 5
    %false = torch.constant.bool false
    %int15 = torch.constant.int 15
    %none = torch.constant.none
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %int4 = torch.constant.int 4
    %int3 = torch.constant.int 3
    %int-1 = torch.constant.int -1
    %2 = torch.prim.ListConstruct %int3, %int-1 : (!torch.int, !torch.int) -> !torch.list<int>
    %3 = torch.aten.unflatten.int %1, %int4, %2 : !torch.vtensor<[16,8,48,32,288],bf16>, !torch.int, !torch.list<int> -> !torch.vtensor<[16,8,48,32,3,96],bf16>
    %4 = torch.prim.ListConstruct %int3, %int-1 : (!torch.int, !torch.int) -> !torch.list<int>
    %5 = torch.aten.unflatten.int %0, %int0, %4 : !torch.vtensor<[288,1,3,3,96],bf16>, !torch.int, !torch.list<int> -> !torch.vtensor<[3,96,1,3,3,96],bf16>
    %6 = torch.prim.ListConstruct %int0, %int0, %int0, %int0, %int1, %int1, %int1, %int1, %int0, %int0, %int0, %int0 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %7 = torch.aten.constant_pad_nd %3, %6, %int0 : !torch.vtensor<[16,8,48,32,3,96],bf16>, !torch.list<int>, !torch.int -> !torch.vtensor<[16,8,50,34,3,96],bf16>
    %8 = torch_c.to_builtin_tensor %7 : !torch.vtensor<[16,8,50,34,3,96],bf16> -> tensor<16x8x50x34x3x96xbf16>
    %9 = torch_c.to_builtin_tensor %5 : !torch.vtensor<[3,96,1,3,3,96],bf16> -> tensor<3x96x1x3x3x96xbf16>
    %10 = util.call @generic_conv_16x8x50x34x3x96xbf16_3x96x1x3x3x96xbf16_1x1x1S_1x1x1D_ndhwgc_gfdhwc_ndhwgf_f32(%8, %9) : (tensor<16x8x50x34x3x96xbf16>, tensor<3x96x1x3x3x96xbf16>) -> tensor<16x8x48x32x3x96xf32>
    %11 = torch_c.from_builtin_tensor %10 : tensor<16x8x48x32x3x96xf32> -> !torch.vtensor<[16,8,48,32,3,96],f32>
    %12 = torch.aten.to.dtype %11, %int15, %false, %false, %none : !torch.vtensor<[16,8,48,32,3,96],f32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[16,8,48,32,3,96],bf16>
    %13 = torch.aten.flatten.using_ints %12, %int4, %int5 : !torch.vtensor<[16,8,48,32,3,96],bf16>, !torch.int, !torch.int -> !torch.vtensor<[16,8,48,32,288],bf16>
    %14 = torch_c.to_builtin_tensor %13 : !torch.vtensor<[16,8,48,32,288],bf16> -> tensor<16x8x48x32x288xbf16>
    return %14 : tensor<16x8x48x32x288xbf16>
  }
  util.func private @generic_conv_16x8x50x34x3x96xbf16_3x96x1x3x3x96xbf16_1x1x1S_1x1x1D_ndhwgc_gfdhwc_ndhwgf_f32(%arg0: tensor<16x8x50x34x3x96xbf16>, %arg1: tensor<3x96x1x3x3x96xbf16>) -> tensor<16x8x48x32x3x96xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<16x8x48x32x3x96xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<16x8x48x32x3x96xf32>) -> tensor<16x8x48x32x3x96xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<16x8x50x34x3x96xbf16>, tensor<3x96x1x3x3x96xbf16>) outs(%1 : tensor<16x8x48x32x3x96xf32>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: f32):
      %3 = arith.extf %in : bf16 to f32
      %4 = arith.extf %in_0 : bf16 to f32
      %5 = arith.mulf %3, %4 : f32
      %6 = arith.addf %out, %5 : f32
      linalg.yield %6 : f32
    } -> tensor<16x8x48x32x3x96xf32>
    util.return %2 : tensor<16x8x48x32x3x96xf32>
  }
}
