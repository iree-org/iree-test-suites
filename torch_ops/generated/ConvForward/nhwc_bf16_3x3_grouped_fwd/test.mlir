#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1 + d5, d2 + d6, d3, d7)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d3, d4, d5, d6, d7)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>
module {
  func.func public @fused_op_layout_cus_72462e9a541052a7edff526c4c8402566ca36ae8_128x48x32x384xbfloat16_384x3x3x64xbfloat16(%arg0: !torch.vtensor<[128,48,32,384],bf16>, %arg1: !torch.vtensor<[384,3,3,64],bf16>) -> !torch.vtensor<[128,48,32,384],bf16> {
    %int0 = torch.constant.int 0
    %int3 = torch.constant.int 3
    %int1 = torch.constant.int 1
    %int2 = torch.constant.int 2
    %0 = torch.prim.ListConstruct %int0, %int3, %int1, %int2 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.permute %arg0, %0 : !torch.vtensor<[128,48,32,384],bf16>, !torch.list<int> -> !torch.vtensor<[128,384,48,32],bf16>
    %int0_0 = torch.constant.int 0
    %int3_1 = torch.constant.int 3
    %int1_2 = torch.constant.int 1
    %int2_3 = torch.constant.int 2
    %2 = torch.prim.ListConstruct %int0_0, %int3_1, %int1_2, %int2_3 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %3 = torch.aten.permute %arg1, %2 : !torch.vtensor<[384,3,3,64],bf16>, !torch.list<int> -> !torch.vtensor<[384,64,3,3],bf16>
    %int0_4 = torch.constant.int 0
    %int2_5 = torch.constant.int 2
    %int3_6 = torch.constant.int 3
    %int1_7 = torch.constant.int 1
    %4 = torch.prim.ListConstruct %int0_4, %int2_5, %int3_6, %int1_7 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %5 = torch.aten.permute %1, %4 : !torch.vtensor<[128,384,48,32],bf16>, !torch.list<int> -> !torch.vtensor<[128,48,32,384],bf16>
    %int0_8 = torch.constant.int 0
    %int2_9 = torch.constant.int 2
    %int3_10 = torch.constant.int 3
    %int1_11 = torch.constant.int 1
    %6 = torch.prim.ListConstruct %int0_8, %int2_9, %int3_10, %int1_11 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %7 = torch.aten.permute %3, %6 : !torch.vtensor<[384,64,3,3],bf16>, !torch.list<int> -> !torch.vtensor<[384,3,3,64],bf16>
    %8 = torch_c.to_builtin_tensor %5 : !torch.vtensor<[128,48,32,384],bf16> -> tensor<128x48x32x384xbf16>
    %9 = torch_c.to_builtin_tensor %7 : !torch.vtensor<[384,3,3,64],bf16> -> tensor<384x3x3x64xbf16>
    %10 = call @conv_2d_bfloat16_forward_128x48x32x384_nhwc_384x3x3x64_fhwc_nhwf_1x1s_1x1p_1x1d_6g(%8, %9) : (tensor<128x48x32x384xbf16>, tensor<384x3x3x64xbf16>) -> tensor<128x48x32x384xbf16>
    %11 = torch_c.from_builtin_tensor %10 : tensor<128x48x32x384xbf16> -> !torch.vtensor<[128,48,32,384],bf16>
    %int0_12 = torch.constant.int 0
    %int3_13 = torch.constant.int 3
    %int1_14 = torch.constant.int 1
    %int2_15 = torch.constant.int 2
    %12 = torch.prim.ListConstruct %int0_12, %int3_13, %int1_14, %int2_15 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %13 = torch.aten.permute %11, %12 : !torch.vtensor<[128,48,32,384],bf16>, !torch.list<int> -> !torch.vtensor<[128,384,48,32],bf16>
    %int0_16 = torch.constant.int 0
    %int2_17 = torch.constant.int 2
    %int3_18 = torch.constant.int 3
    %int1_19 = torch.constant.int 1
    %14 = torch.prim.ListConstruct %int0_16, %int2_17, %int3_18, %int1_19 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %15 = torch.aten.permute %13, %14 : !torch.vtensor<[128,384,48,32],bf16>, !torch.list<int> -> !torch.vtensor<[128,48,32,384],bf16>
    return %15 : !torch.vtensor<[128,48,32,384],bf16>
  }
  func.func private @conv_2d_bfloat16_forward_128x48x32x384_nhwc_384x3x3x64_fhwc_nhwf_1x1s_1x1p_1x1d_6g(%arg0: tensor<128x48x32x384xbf16>, %arg1: tensor<384x3x3x64xbf16>) -> tensor<128x48x32x384xbf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch_c.from_builtin_tensor %arg1 : tensor<384x3x3x64xbf16> -> !torch.vtensor<[384,3,3,64],bf16>
    %1 = torch_c.from_builtin_tensor %arg0 : tensor<128x48x32x384xbf16> -> !torch.vtensor<[128,48,32,384],bf16>
    %int4 = torch.constant.int 4
    %false = torch.constant.bool false
    %int15 = torch.constant.int 15
    %none = torch.constant.none
    %int1 = torch.constant.int 1
    %int0 = torch.constant.int 0
    %int3 = torch.constant.int 3
    %int6 = torch.constant.int 6
    %int-1 = torch.constant.int -1
    %2 = torch.prim.ListConstruct %int6, %int-1 : (!torch.int, !torch.int) -> !torch.list<int>
    %3 = torch.aten.unflatten.int %1, %int3, %2 : !torch.vtensor<[128,48,32,384],bf16>, !torch.int, !torch.list<int> -> !torch.vtensor<[128,48,32,6,64],bf16>
    %4 = torch.prim.ListConstruct %int6, %int-1 : (!torch.int, !torch.int) -> !torch.list<int>
    %5 = torch.aten.unflatten.int %0, %int0, %4 : !torch.vtensor<[384,3,3,64],bf16>, !torch.int, !torch.list<int> -> !torch.vtensor<[6,64,3,3,64],bf16>
    %6 = torch.prim.ListConstruct %int0, %int0, %int0, %int0, %int1, %int1, %int1, %int1, %int0, %int0 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %7 = torch.aten.constant_pad_nd %3, %6, %int0 : !torch.vtensor<[128,48,32,6,64],bf16>, !torch.list<int>, !torch.int -> !torch.vtensor<[128,50,34,6,64],bf16>
    %8 = torch_c.to_builtin_tensor %7 : !torch.vtensor<[128,50,34,6,64],bf16> -> tensor<128x50x34x6x64xbf16>
    %9 = torch_c.to_builtin_tensor %5 : !torch.vtensor<[6,64,3,3,64],bf16> -> tensor<6x64x3x3x64xbf16>
    %10 = util.call @generic_conv_128x50x34x6x64xbf16_6x64x3x3x64xbf16_1x1S_1x1D_nhwgc_gfhwc_nhwgf_f32(%8, %9) : (tensor<128x50x34x6x64xbf16>, tensor<6x64x3x3x64xbf16>) -> tensor<128x48x32x6x64xf32>
    %11 = torch_c.from_builtin_tensor %10 : tensor<128x48x32x6x64xf32> -> !torch.vtensor<[128,48,32,6,64],f32>
    %12 = torch.aten.to.dtype %11, %int15, %false, %false, %none : !torch.vtensor<[128,48,32,6,64],f32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[128,48,32,6,64],bf16>
    %13 = torch.aten.flatten.using_ints %12, %int3, %int4 : !torch.vtensor<[128,48,32,6,64],bf16>, !torch.int, !torch.int -> !torch.vtensor<[128,48,32,384],bf16>
    %14 = torch_c.to_builtin_tensor %13 : !torch.vtensor<[128,48,32,384],bf16> -> tensor<128x48x32x384xbf16>
    return %14 : tensor<128x48x32x384xbf16>
  }
  util.func private @generic_conv_128x50x34x6x64xbf16_6x64x3x3x64xbf16_1x1S_1x1D_nhwgc_gfhwc_nhwgf_f32(%arg0: tensor<128x50x34x6x64xbf16>, %arg1: tensor<6x64x3x3x64xbf16>) -> tensor<128x48x32x6x64xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<128x48x32x6x64xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<128x48x32x6x64xf32>) -> tensor<128x48x32x6x64xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<128x50x34x6x64xbf16>, tensor<6x64x3x3x64xbf16>) outs(%1 : tensor<128x48x32x6x64xf32>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: f32):
      %3 = arith.extf %in : bf16 to f32
      %4 = arith.extf %in_0 : bf16 to f32
      %5 = arith.mulf %3, %4 : f32
      %6 = arith.addf %out, %5 : f32
      linalg.yield %6 : f32
    } -> tensor<128x48x32x6x64xf32>
    util.return %2 : tensor<128x48x32x6x64xf32>
  }
}
