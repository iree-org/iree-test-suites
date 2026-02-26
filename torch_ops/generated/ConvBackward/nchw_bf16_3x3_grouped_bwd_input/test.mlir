#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d5, d3 + d6, d4 + d7)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d1, d5, d2, d6, d7)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>
module {
  func.func public @fused_op_layout_cus_b69aba9691a161457341f1c7b69acb51cc67b3f4_128x384x48x32xbfloat16_128x384x48x32xbfloat16_384x64x3x3xbfloat16(%arg0: !torch.vtensor<[128,384,48,32],bf16>, %arg1: !torch.vtensor<[128,384,48,32],bf16>, %arg2: !torch.vtensor<[384,64,3,3],bf16>) -> !torch.vtensor<[128,384,48,32],bf16> {
    %0 = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[128,384,48,32],bf16> -> tensor<128x384x48x32xbf16>
    %1 = torch_c.to_builtin_tensor %arg1 : !torch.vtensor<[128,384,48,32],bf16> -> tensor<128x384x48x32xbf16>
    %2 = torch_c.to_builtin_tensor %arg2 : !torch.vtensor<[384,64,3,3],bf16> -> tensor<384x64x3x3xbf16>
    %3 = call @conv_2d_bfloat16_input_backward_128x384x48x32_nchw_384x64x3x3_fchw_nfhw_1x1s_1x1p_1x1d_6g(%0, %1, %2) : (tensor<128x384x48x32xbf16>, tensor<128x384x48x32xbf16>, tensor<384x64x3x3xbf16>) -> tensor<128x384x48x32xbf16>
    %4 = torch_c.from_builtin_tensor %3 : tensor<128x384x48x32xbf16> -> !torch.vtensor<[128,384,48,32],bf16>
    %none = torch.constant.none
    return %4 : !torch.vtensor<[128,384,48,32],bf16>
  }
  func.func private @conv_2d_bfloat16_input_backward_128x384x48x32_nchw_384x64x3x3_fchw_nfhw_1x1s_1x1p_1x1d_6g(%arg0: tensor<128x384x48x32xbf16>, %arg1: tensor<128x384x48x32xbf16>, %arg2: tensor<384x64x3x3xbf16>) -> tensor<128x384x48x32xbf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch_c.from_builtin_tensor %arg2 : tensor<384x64x3x3xbf16> -> !torch.vtensor<[384,64,3,3],bf16>
    %1 = torch_c.from_builtin_tensor %arg0 : tensor<128x384x48x32xbf16> -> !torch.vtensor<[128,384,48,32],bf16>
    %int2 = torch.constant.int 2
    %false = torch.constant.bool false
    %int15 = torch.constant.int 15
    %none = torch.constant.none
    %int4 = torch.constant.int 4
    %int3 = torch.constant.int 3
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %int6 = torch.constant.int 6
    %int-1 = torch.constant.int -1
    %2 = torch.prim.ListConstruct %int6, %int-1 : (!torch.int, !torch.int) -> !torch.list<int>
    %3 = torch.aten.unflatten.int %1, %int1, %2 : !torch.vtensor<[128,384,48,32],bf16>, !torch.int, !torch.list<int> -> !torch.vtensor<[128,6,64,48,32],bf16>
    %4 = torch.prim.ListConstruct %int6, %int-1 : (!torch.int, !torch.int) -> !torch.list<int>
    %5 = torch.aten.unflatten.int %0, %int0, %4 : !torch.vtensor<[384,64,3,3],bf16>, !torch.int, !torch.list<int> -> !torch.vtensor<[6,64,64,3,3],bf16>
    %6 = torch.prim.ListConstruct %int3, %int4 : (!torch.int, !torch.int) -> !torch.list<int>
    %7 = torch.aten.flip %5, %6 : !torch.vtensor<[6,64,64,3,3],bf16>, !torch.list<int> -> !torch.vtensor<[6,64,64,3,3],bf16>
    %8 = torch.prim.ListConstruct %int1, %int1, %int1, %int1, %int0, %int0, %int0, %int0, %int0, %int0 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %9 = torch.aten.constant_pad_nd %3, %8, %int0 : !torch.vtensor<[128,6,64,48,32],bf16>, !torch.list<int>, !torch.int -> !torch.vtensor<[128,6,64,50,34],bf16>
    %10 = torch_c.to_builtin_tensor %9 : !torch.vtensor<[128,6,64,50,34],bf16> -> tensor<128x6x64x50x34xbf16>
    %11 = torch_c.to_builtin_tensor %7 : !torch.vtensor<[6,64,64,3,3],bf16> -> tensor<6x64x64x3x3xbf16>
    %12 = util.call @generic_conv_128x6x64x50x34xbf16_6x64x64x3x3xbf16_1x1S_1x1D_ngchw_gcfhw_ngfhw_f32(%10, %11) : (tensor<128x6x64x50x34xbf16>, tensor<6x64x64x3x3xbf16>) -> tensor<128x6x64x48x32xf32>
    %13 = torch_c.from_builtin_tensor %12 : tensor<128x6x64x48x32xf32> -> !torch.vtensor<[128,6,64,48,32],f32>
    %14 = torch.aten.to.dtype %13, %int15, %false, %false, %none : !torch.vtensor<[128,6,64,48,32],f32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[128,6,64,48,32],bf16>
    %15 = torch.aten.flatten.using_ints %14, %int1, %int2 : !torch.vtensor<[128,6,64,48,32],bf16>, !torch.int, !torch.int -> !torch.vtensor<[128,384,48,32],bf16>
    %16 = torch_c.to_builtin_tensor %15 : !torch.vtensor<[128,384,48,32],bf16> -> tensor<128x384x48x32xbf16>
    return %16 : tensor<128x384x48x32xbf16>
  }
  util.func private @generic_conv_128x6x64x50x34xbf16_6x64x64x3x3xbf16_1x1S_1x1D_ngchw_gcfhw_ngfhw_f32(%arg0: tensor<128x6x64x50x34xbf16>, %arg1: tensor<6x64x64x3x3xbf16>) -> tensor<128x6x64x48x32xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<128x6x64x48x32xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<128x6x64x48x32xf32>) -> tensor<128x6x64x48x32xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<128x6x64x50x34xbf16>, tensor<6x64x64x3x3xbf16>) outs(%1 : tensor<128x6x64x48x32xf32>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: f32):
      %3 = arith.extf %in : bf16 to f32
      %4 = arith.extf %in_0 : bf16 to f32
      %5 = arith.mulf %3, %4 : f32
      %6 = arith.addf %out, %5 : f32
      linalg.yield %6 : f32
    } -> tensor<128x6x64x48x32xf32>
    util.return %2 : tensor<128x6x64x48x32xf32>
  }
}
