#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d5, d0, d2, d3 + d6, d4 + d7)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d5, d0, d1, d6, d7)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>
module {
  func.func public @fused_op_layout_cus_be4c3d3ab9f4c15d00b81ee14f0a6f67f6cd6040_128x384x48x32xbfloat16_128x384x48x32xbfloat16_384x64x3x3xbfloat16(%arg0: !torch.vtensor<[128,384,48,32],bf16>, %arg1: !torch.vtensor<[128,384,48,32],bf16>, %arg2: !torch.vtensor<[384,64,3,3],bf16>) -> !torch.vtensor<[384,64,3,3],bf16> {
    %0 = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[128,384,48,32],bf16> -> tensor<128x384x48x32xbf16>
    %1 = torch_c.to_builtin_tensor %arg1 : !torch.vtensor<[128,384,48,32],bf16> -> tensor<128x384x48x32xbf16>
    %2 = torch_c.to_builtin_tensor %arg2 : !torch.vtensor<[384,64,3,3],bf16> -> tensor<384x64x3x3xbf16>
    %3 = call @conv_2d_bfloat16_weight_backward_128x384x48x32_nchw_384x64x3x3_fchw_nfhw_1x1s_1x1p_1x1d_6g(%0, %1, %2) : (tensor<128x384x48x32xbf16>, tensor<128x384x48x32xbf16>, tensor<384x64x3x3xbf16>) -> tensor<384x64x3x3xbf16>
    %4 = torch_c.from_builtin_tensor %3 : tensor<384x64x3x3xbf16> -> !torch.vtensor<[384,64,3,3],bf16>
    %none = torch.constant.none
    return %4 : !torch.vtensor<[384,64,3,3],bf16>
  }
  func.func private @conv_2d_bfloat16_weight_backward_128x384x48x32_nchw_384x64x3x3_fchw_nfhw_1x1s_1x1p_1x1d_6g(%arg0: tensor<128x384x48x32xbf16>, %arg1: tensor<128x384x48x32xbf16>, %arg2: tensor<384x64x3x3xbf16>) -> tensor<384x64x3x3xbf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch_c.from_builtin_tensor %arg1 : tensor<128x384x48x32xbf16> -> !torch.vtensor<[128,384,48,32],bf16>
    %1 = torch_c.from_builtin_tensor %arg0 : tensor<128x384x48x32xbf16> -> !torch.vtensor<[128,384,48,32],bf16>
    %false = torch.constant.bool false
    %int15 = torch.constant.int 15
    %none = torch.constant.none
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %int6 = torch.constant.int 6
    %int-1 = torch.constant.int -1
    %2 = torch.prim.ListConstruct %int6, %int-1 : (!torch.int, !torch.int) -> !torch.list<int>
    %3 = torch.aten.unflatten.int %0, %int1, %2 : !torch.vtensor<[128,384,48,32],bf16>, !torch.int, !torch.list<int> -> !torch.vtensor<[128,6,64,48,32],bf16>
    %4 = torch.prim.ListConstruct %int6, %int-1 : (!torch.int, !torch.int) -> !torch.list<int>
    %5 = torch.aten.unflatten.int %1, %int1, %4 : !torch.vtensor<[128,384,48,32],bf16>, !torch.int, !torch.list<int> -> !torch.vtensor<[128,6,64,48,32],bf16>
    %6 = torch.prim.ListConstruct %int1, %int1, %int1, %int1, %int0, %int0, %int0, %int0, %int0, %int0 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %7 = torch.aten.constant_pad_nd %3, %6, %int0 : !torch.vtensor<[128,6,64,48,32],bf16>, !torch.list<int>, !torch.int -> !torch.vtensor<[128,6,64,50,34],bf16>
    %8 = torch_c.to_builtin_tensor %7 : !torch.vtensor<[128,6,64,50,34],bf16> -> tensor<128x6x64x50x34xbf16>
    %9 = torch_c.to_builtin_tensor %5 : !torch.vtensor<[128,6,64,48,32],bf16> -> tensor<128x6x64x48x32xbf16>
    %10 = util.call @generic_conv_128x6x64x50x34xbf16_128x6x64x48x32xbf16_1x1S_1x1D_cgnhw_cgfhw_gfnhw_f32(%8, %9) : (tensor<128x6x64x50x34xbf16>, tensor<128x6x64x48x32xbf16>) -> tensor<6x64x64x3x3xf32>
    %11 = torch_c.from_builtin_tensor %10 : tensor<6x64x64x3x3xf32> -> !torch.vtensor<[6,64,64,3,3],f32>
    %12 = torch.aten.to.dtype %11, %int15, %false, %false, %none : !torch.vtensor<[6,64,64,3,3],f32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[6,64,64,3,3],bf16>
    %13 = torch.aten.flatten.using_ints %12, %int0, %int1 : !torch.vtensor<[6,64,64,3,3],bf16>, !torch.int, !torch.int -> !torch.vtensor<[384,64,3,3],bf16>
    %14 = torch_c.to_builtin_tensor %13 : !torch.vtensor<[384,64,3,3],bf16> -> tensor<384x64x3x3xbf16>
    return %14 : tensor<384x64x3x3xbf16>
  }
  util.func private @generic_conv_128x6x64x50x34xbf16_128x6x64x48x32xbf16_1x1S_1x1D_cgnhw_cgfhw_gfnhw_f32(%arg0: tensor<128x6x64x50x34xbf16>, %arg1: tensor<128x6x64x48x32xbf16>) -> tensor<6x64x64x3x3xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<6x64x64x3x3xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<6x64x64x3x3xf32>) -> tensor<6x64x64x3x3xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<128x6x64x50x34xbf16>, tensor<128x6x64x48x32xbf16>) outs(%1 : tensor<6x64x64x3x3xf32>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: f32):
      %3 = arith.extf %in : bf16 to f32
      %4 = arith.extf %in_0 : bf16 to f32
      %5 = arith.mulf %3, %4 : f32
      %6 = arith.addf %out, %5 : f32
      linalg.yield %6 : f32
    } -> tensor<6x64x64x3x3xf32>
    util.return %2 : tensor<6x64x64x3x3xf32>
  }
}
