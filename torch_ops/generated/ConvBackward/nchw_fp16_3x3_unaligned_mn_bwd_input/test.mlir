#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d4, d2 + d5, d3 + d6)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d1, d5, d6)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
module {
  func.func public @fused_op_layout_cus_8ce1f182e1fefbb60c50702abd992d86ad1a44c8_32x256x25x25xfloat16_32x256x25x25xfloat16_256x256x3x3xfloat16(%arg0: !torch.vtensor<[32,256,25,25],f16>, %arg1: !torch.vtensor<[32,256,25,25],f16>, %arg2: !torch.vtensor<[256,256,3,3],f16>) -> !torch.vtensor<[32,256,25,25],f16> {
    %0 = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[32,256,25,25],f16> -> tensor<32x256x25x25xf16>
    %1 = torch_c.to_builtin_tensor %arg1 : !torch.vtensor<[32,256,25,25],f16> -> tensor<32x256x25x25xf16>
    %2 = torch_c.to_builtin_tensor %arg2 : !torch.vtensor<[256,256,3,3],f16> -> tensor<256x256x3x3xf16>
    %3 = call @conv_2d_float16_input_backward_32x256x25x25_nchw_256x256x3x3_fchw_nfhw_1x1s_1x1p_1x1d_1g(%0, %1, %2) : (tensor<32x256x25x25xf16>, tensor<32x256x25x25xf16>, tensor<256x256x3x3xf16>) -> tensor<32x256x25x25xf16>
    %4 = torch_c.from_builtin_tensor %3 : tensor<32x256x25x25xf16> -> !torch.vtensor<[32,256,25,25],f16>
    %none = torch.constant.none
    return %4 : !torch.vtensor<[32,256,25,25],f16>
  }
  func.func private @conv_2d_float16_input_backward_32x256x25x25_nchw_256x256x3x3_fchw_nfhw_1x1s_1x1p_1x1d_1g(%arg0: tensor<32x256x25x25xf16>, %arg1: tensor<32x256x25x25xf16>, %arg2: tensor<256x256x3x3xf16>) -> tensor<32x256x25x25xf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch_c.from_builtin_tensor %arg2 : tensor<256x256x3x3xf16> -> !torch.vtensor<[256,256,3,3],f16>
    %1 = torch_c.from_builtin_tensor %arg0 : tensor<32x256x25x25xf16> -> !torch.vtensor<[32,256,25,25],f16>
    %false = torch.constant.bool false
    %int5 = torch.constant.int 5
    %none = torch.constant.none
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %int2 = torch.constant.int 2
    %int3 = torch.constant.int 3
    %2 = torch.prim.ListConstruct %int2, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
    %3 = torch.aten.flip %0, %2 : !torch.vtensor<[256,256,3,3],f16>, !torch.list<int> -> !torch.vtensor<[256,256,3,3],f16>
    %4 = torch.prim.ListConstruct %int1, %int1, %int1, %int1, %int0, %int0, %int0, %int0 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %5 = torch.aten.constant_pad_nd %1, %4, %int0 : !torch.vtensor<[32,256,25,25],f16>, !torch.list<int>, !torch.int -> !torch.vtensor<[32,256,27,27],f16>
    %6 = torch_c.to_builtin_tensor %5 : !torch.vtensor<[32,256,27,27],f16> -> tensor<32x256x27x27xf16>
    %7 = torch_c.to_builtin_tensor %3 : !torch.vtensor<[256,256,3,3],f16> -> tensor<256x256x3x3xf16>
    %8 = util.call @generic_conv_32x256x27x27xf16_256x256x3x3xf16_1x1S_1x1D_nchw_cfhw_nfhw_f32(%6, %7) : (tensor<32x256x27x27xf16>, tensor<256x256x3x3xf16>) -> tensor<32x256x25x25xf32>
    %9 = torch_c.from_builtin_tensor %8 : tensor<32x256x25x25xf32> -> !torch.vtensor<[32,256,25,25],f32>
    %10 = torch.aten.to.dtype %9, %int5, %false, %false, %none : !torch.vtensor<[32,256,25,25],f32>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[32,256,25,25],f16>
    %11 = torch_c.to_builtin_tensor %10 : !torch.vtensor<[32,256,25,25],f16> -> tensor<32x256x25x25xf16>
    return %11 : tensor<32x256x25x25xf16>
  }
  util.func private @generic_conv_32x256x27x27xf16_256x256x3x3xf16_1x1S_1x1D_nchw_cfhw_nfhw_f32(%arg0: tensor<32x256x27x27xf16>, %arg1: tensor<256x256x3x3xf16>) -> tensor<32x256x25x25xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<32x256x25x25xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<32x256x25x25xf32>) -> tensor<32x256x25x25xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<32x256x27x27xf16>, tensor<256x256x3x3xf16>) outs(%1 : tensor<32x256x25x25xf32>) {
    ^bb0(%in: f16, %in_0: f16, %out: f32):
      %3 = arith.extf %in : f16 to f32
      %4 = arith.extf %in_0 : f16 to f32
      %5 = arith.mulf %3, %4 : f32
      %6 = arith.addf %out, %5 : f32
      linalg.yield %6 : f32
    } -> tensor<32x256x25x25xf32>
    util.return %2 : tensor<32x256x25x25xf32>
  }
}
