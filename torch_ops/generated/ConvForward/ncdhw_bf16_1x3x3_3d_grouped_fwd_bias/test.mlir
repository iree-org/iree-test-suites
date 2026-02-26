#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d0, d1, d6, d3 + d7, d4 + d8, d5 + d9)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d1, d2, d6, d7, d8, d9)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d0, d1, d2, d3, d4, d5)>
module {
  func.func public @fused_op_layout_cus_cda22d7a32e4f4aa1e421e66a02fd23f35fef850_16x288x8x48x32xbfloat16_288x96x1x3x3xbfloat16_288xbfloat16(%arg0: !torch.vtensor<[16,288,8,48,32],bf16>, %arg1: !torch.vtensor<[288,96,1,3,3],bf16>, %arg2: !torch.vtensor<[288],bf16>) -> !torch.vtensor<[16,288,8,48,32],bf16> {
    %0 = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[16,288,8,48,32],bf16> -> tensor<16x288x8x48x32xbf16>
    %1 = torch_c.to_builtin_tensor %arg1 : !torch.vtensor<[288,96,1,3,3],bf16> -> tensor<288x96x1x3x3xbf16>
    %2 = torch_c.to_builtin_tensor %arg2 : !torch.vtensor<[288],bf16> -> tensor<288xbf16>
    %3 = call @conv_3d_bfloat16_forward_b_16x288x8x48x32_ncdhw_288x96x1x3x3_fcdhw_nfdhw_1x1x1s_0x1x1p_1x1x1d_3g(%0, %1, %2) : (tensor<16x288x8x48x32xbf16>, tensor<288x96x1x3x3xbf16>, tensor<288xbf16>) -> tensor<16x288x8x48x32xbf16>
    %4 = torch_c.from_builtin_tensor %3 : tensor<16x288x8x48x32xbf16> -> !torch.vtensor<[16,288,8,48,32],bf16>
    return %4 : !torch.vtensor<[16,288,8,48,32],bf16>
  }
  func.func private @conv_3d_bfloat16_forward_b_16x288x8x48x32_ncdhw_288x96x1x3x3_fcdhw_nfdhw_1x1x1s_0x1x1p_1x1x1d_3g(%arg0: tensor<16x288x8x48x32xbf16>, %arg1: tensor<288x96x1x3x3xbf16>, %arg2: tensor<288xbf16>) -> tensor<16x288x8x48x32xbf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch_c.from_builtin_tensor %arg2 : tensor<288xbf16> -> !torch.vtensor<[288],bf16>
    %1 = torch_c.from_builtin_tensor %arg1 : tensor<288x96x1x3x3xbf16> -> !torch.vtensor<[288,96,1,3,3],bf16>
    %2 = torch_c.from_builtin_tensor %arg0 : tensor<16x288x8x48x32xbf16> -> !torch.vtensor<[16,288,8,48,32],bf16>
    %int2 = torch.constant.int 2
    %false = torch.constant.bool false
    %int15 = torch.constant.int 15
    %none = torch.constant.none
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %int3 = torch.constant.int 3
    %int-1 = torch.constant.int -1
    %3 = torch.prim.ListConstruct %int3, %int-1 : (!torch.int, !torch.int) -> !torch.list<int>
    %4 = torch.aten.unflatten.int %2, %int1, %3 : !torch.vtensor<[16,288,8,48,32],bf16>, !torch.int, !torch.list<int> -> !torch.vtensor<[16,3,96,8,48,32],bf16>
    %5 = torch.prim.ListConstruct %int3, %int-1 : (!torch.int, !torch.int) -> !torch.list<int>
    %6 = torch.aten.unflatten.int %1, %int0, %5 : !torch.vtensor<[288,96,1,3,3],bf16>, !torch.int, !torch.list<int> -> !torch.vtensor<[3,96,96,1,3,3],bf16>
    %7 = torch.prim.ListConstruct %int1, %int1, %int1, %int1, %int0, %int0, %int0, %int0, %int0, %int0, %int0, %int0 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %8 = torch.aten.constant_pad_nd %4, %7, %int0 : !torch.vtensor<[16,3,96,8,48,32],bf16>, !torch.list<int>, !torch.int -> !torch.vtensor<[16,3,96,8,50,34],bf16>
    %9 = torch_c.to_builtin_tensor %8 : !torch.vtensor<[16,3,96,8,50,34],bf16> -> tensor<16x3x96x8x50x34xbf16>
    %10 = torch_c.to_builtin_tensor %6 : !torch.vtensor<[3,96,96,1,3,3],bf16> -> tensor<3x96x96x1x3x3xbf16>
    %11 = util.call @generic_conv_16x3x96x8x50x34xbf16_3x96x96x1x3x3xbf16_1x1x1S_1x1x1D_ngcdhw_gfcdhw_ngfdhw_f32(%9, %10) : (tensor<16x3x96x8x50x34xbf16>, tensor<3x96x96x1x3x3xbf16>) -> tensor<16x3x96x8x48x32xf32>
    %12 = torch_c.from_builtin_tensor %11 : tensor<16x3x96x8x48x32xf32> -> !torch.vtensor<[16,3,96,8,48,32],f32>
    %13 = torch.aten.to.dtype %12, %int15, %false, %false, %none : !torch.vtensor<[16,3,96,8,48,32],f32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[16,3,96,8,48,32],bf16>
    %14 = torch.aten.flatten.using_ints %13, %int1, %int2 : !torch.vtensor<[16,3,96,8,48,32],bf16>, !torch.int, !torch.int -> !torch.vtensor<[16,288,8,48,32],bf16>
    %15 = torch.prim.ListConstruct %int-1, %int1, %int1, %int1 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %16 = torch.aten.unflatten.int %0, %int0, %15 : !torch.vtensor<[288],bf16>, !torch.int, !torch.list<int> -> !torch.vtensor<[288,1,1,1],bf16>
    %17 = torch.aten.add.Tensor %14, %16, %int1 : !torch.vtensor<[16,288,8,48,32],bf16>, !torch.vtensor<[288,1,1,1],bf16>, !torch.int -> !torch.vtensor<[16,288,8,48,32],bf16>
    %18 = torch_c.to_builtin_tensor %17 : !torch.vtensor<[16,288,8,48,32],bf16> -> tensor<16x288x8x48x32xbf16>
    return %18 : tensor<16x288x8x48x32xbf16>
  }
  util.func private @generic_conv_16x3x96x8x50x34xbf16_3x96x96x1x3x3xbf16_1x1x1S_1x1x1D_ngcdhw_gfcdhw_ngfdhw_f32(%arg0: tensor<16x3x96x8x50x34xbf16>, %arg1: tensor<3x96x96x1x3x3xbf16>) -> tensor<16x3x96x8x48x32xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<16x3x96x8x48x32xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<16x3x96x8x48x32xf32>) -> tensor<16x3x96x8x48x32xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<16x3x96x8x50x34xbf16>, tensor<3x96x96x1x3x3xbf16>) outs(%1 : tensor<16x3x96x8x48x32xf32>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: f32):
      %3 = arith.extf %in : bf16 to f32
      %4 = arith.extf %in_0 : bf16 to f32
      %5 = arith.mulf %3, %4 : f32
      %6 = arith.addf %out, %5 : f32
      linalg.yield %6 : f32
    } -> tensor<16x3x96x8x48x32xf32>
    util.return %2 : tensor<16x3x96x8x48x32xf32>
  }
}
