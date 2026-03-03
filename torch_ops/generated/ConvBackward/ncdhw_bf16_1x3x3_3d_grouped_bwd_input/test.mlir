#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d0, d1, d6, d3 + d7, d4 + d8, d5 + d9)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d1, d6, d2, d7, d8, d9)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d0, d1, d2, d3, d4, d5)>
module {
  func.func public @fused_op_layout_cus_358aed60e5ce1bd96721d1473b08c558ba75f403_16x288x8x48x32xbfloat16_16x288x8x48x32xbfloat16_288x96x1x3x3xbfloat16(%arg0: !torch.vtensor<[16,288,8,48,32],bf16>, %arg1: !torch.vtensor<[16,288,8,48,32],bf16>, %arg2: !torch.vtensor<[288,96,1,3,3],bf16>) -> !torch.vtensor<[16,288,8,48,32],bf16> {
    %0 = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[16,288,8,48,32],bf16> -> tensor<16x288x8x48x32xbf16>
    %1 = torch_c.to_builtin_tensor %arg1 : !torch.vtensor<[16,288,8,48,32],bf16> -> tensor<16x288x8x48x32xbf16>
    %2 = torch_c.to_builtin_tensor %arg2 : !torch.vtensor<[288,96,1,3,3],bf16> -> tensor<288x96x1x3x3xbf16>
    %3 = call @conv_3d_bfloat16_input_backward_16x288x8x48x32_ncdhw_288x96x1x3x3_fcdhw_nfdhw_1x1x1s_0x1x1p_1x1x1d_3g(%0, %1, %2) : (tensor<16x288x8x48x32xbf16>, tensor<16x288x8x48x32xbf16>, tensor<288x96x1x3x3xbf16>) -> tensor<16x288x8x48x32xbf16>
    %4 = torch_c.from_builtin_tensor %3 : tensor<16x288x8x48x32xbf16> -> !torch.vtensor<[16,288,8,48,32],bf16>
    %none = torch.constant.none
    return %4 : !torch.vtensor<[16,288,8,48,32],bf16>
  }
  func.func private @conv_3d_bfloat16_input_backward_16x288x8x48x32_ncdhw_288x96x1x3x3_fcdhw_nfdhw_1x1x1s_0x1x1p_1x1x1d_3g(%arg0: tensor<16x288x8x48x32xbf16>, %arg1: tensor<16x288x8x48x32xbf16>, %arg2: tensor<288x96x1x3x3xbf16>) -> tensor<16x288x8x48x32xbf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch_c.from_builtin_tensor %arg2 : tensor<288x96x1x3x3xbf16> -> !torch.vtensor<[288,96,1,3,3],bf16>
    %1 = torch_c.from_builtin_tensor %arg0 : tensor<16x288x8x48x32xbf16> -> !torch.vtensor<[16,288,8,48,32],bf16>
    %int288 = torch.constant.int 288
    %false = torch.constant.bool false
    %none = torch.constant.none
    %int15 = torch.constant.int 15
    %int0 = torch.constant.int 0
    %int5 = torch.constant.int 5
    %int4 = torch.constant.int 4
    %int1 = torch.constant.int 1
    %int16 = torch.constant.int 16
    %int3 = torch.constant.int 3
    %int96 = torch.constant.int 96
    %int8 = torch.constant.int 8
    %int48 = torch.constant.int 48
    %int32 = torch.constant.int 32
    %2 = torch.prim.ListConstruct %int16, %int3, %int96, %int8, %int48, %int32 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %3 = torch.aten.view %1, %2 : !torch.vtensor<[16,288,8,48,32],bf16>, !torch.list<int> -> !torch.vtensor<[16,3,96,8,48,32],bf16>
    %4 = torch.prim.ListConstruct %int3, %int96, %int96, %int1, %int3, %int3 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %5 = torch.aten.view %0, %4 : !torch.vtensor<[288,96,1,3,3],bf16>, !torch.list<int> -> !torch.vtensor<[3,96,96,1,3,3],bf16>
    %6 = torch.prim.ListConstruct %int4, %int5 : (!torch.int, !torch.int) -> !torch.list<int>
    %7 = torch.aten.flip %5, %6 : !torch.vtensor<[3,96,96,1,3,3],bf16>, !torch.list<int> -> !torch.vtensor<[3,96,96,1,3,3],bf16>
    %8 = torch.prim.ListConstruct %int1, %int1, %int1, %int1, %int0, %int0, %int0, %int0, %int0, %int0, %int0, %int0 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %9 = torch.aten.constant_pad_nd %3, %8, %int0 : !torch.vtensor<[16,3,96,8,48,32],bf16>, !torch.list<int>, !torch.int -> !torch.vtensor<[16,3,96,8,50,34],bf16>
    %10 = torch_c.to_builtin_tensor %9 : !torch.vtensor<[16,3,96,8,50,34],bf16> -> tensor<16x3x96x8x50x34xbf16>
    %11 = torch_c.to_builtin_tensor %7 : !torch.vtensor<[3,96,96,1,3,3],bf16> -> tensor<3x96x96x1x3x3xbf16>
    %12 = util.call @generic_conv_16x3x96x8x50x34xbf16_3x96x96x1x3x3xbf16_1x1x1S_1x1x1D_ngcdhw_gcfdhw_ngfdhw_f32(%10, %11) : (tensor<16x3x96x8x50x34xbf16>, tensor<3x96x96x1x3x3xbf16>) -> tensor<16x3x96x8x48x32xf32>
    %13 = torch_c.from_builtin_tensor %12 : tensor<16x3x96x8x48x32xf32> -> !torch.vtensor<[16,3,96,8,48,32],f32>
    %14 = torch.aten._to_copy %13, %int15, %none, %none, %none, %false, %none : !torch.vtensor<[16,3,96,8,48,32],f32>, !torch.int, !torch.none, !torch.none, !torch.none, !torch.bool, !torch.none -> !torch.vtensor<[16,3,96,8,48,32],bf16>
    %15 = torch.prim.ListConstruct %int16, %int288, %int8, %int48, %int32 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %16 = torch.aten.view %14, %15 : !torch.vtensor<[16,3,96,8,48,32],bf16>, !torch.list<int> -> !torch.vtensor<[16,288,8,48,32],bf16>
    %17 = torch_c.to_builtin_tensor %16 : !torch.vtensor<[16,288,8,48,32],bf16> -> tensor<16x288x8x48x32xbf16>
    return %17 : tensor<16x288x8x48x32xbf16>
  }
  util.func private @generic_conv_16x3x96x8x50x34xbf16_3x96x96x1x3x3xbf16_1x1x1S_1x1x1D_ngcdhw_gcfdhw_ngfdhw_f32(%arg0: tensor<16x3x96x8x50x34xbf16>, %arg1: tensor<3x96x96x1x3x3xbf16>) -> tensor<16x3x96x8x48x32xf32> {
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
