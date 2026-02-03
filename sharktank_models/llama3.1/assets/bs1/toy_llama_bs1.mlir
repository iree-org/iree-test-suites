#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module @module {
  util.global private @__auto.token_embd.weight = #stream.parameter.named<"model"::"token_embd.weight"> : tensor<256x256xf32>
  util.global private @__auto.blk.0.attn_norm.weight = #stream.parameter.named<"model"::"blk.0.attn_norm.weight"> : tensor<256xf32>
  util.global private @__auto.blk.0.attn_q.weight = #stream.parameter.named<"model"::"blk.0.attn_q.weight"> : tensor<256x256xf32>
  util.global private @__auto.blk.0.attn_k.weight = #stream.parameter.named<"model"::"blk.0.attn_k.weight"> : tensor<128x256xf32>
  util.global private @__auto.blk.0.attn_v.weight = #stream.parameter.named<"model"::"blk.0.attn_v.weight"> : tensor<128x256xf32>
  util.global private @__auto.blk.0.attn_output.weight = #stream.parameter.named<"model"::"blk.0.attn_output.weight"> : tensor<256x256xf32>
  util.global private @__auto.blk.0.ffn_norm.weight = #stream.parameter.named<"model"::"blk.0.ffn_norm.weight"> : tensor<256xf32>
  util.global private @__auto.blk.0.ffn_gate.weight = #stream.parameter.named<"model"::"blk.0.ffn_gate.weight"> : tensor<23x256xf32>
  util.global private @__auto.blk.0.ffn_up.weight = #stream.parameter.named<"model"::"blk.0.ffn_up.weight"> : tensor<23x256xf32>
  util.global private @__auto.blk.0.ffn_down.weight = #stream.parameter.named<"model"::"blk.0.ffn_down.weight"> : tensor<256x23xf32>
  util.global private @__auto.blk.1.attn_norm.weight = #stream.parameter.named<"model"::"blk.1.attn_norm.weight"> : tensor<256xf32>
  util.global private @__auto.blk.1.attn_q.weight = #stream.parameter.named<"model"::"blk.1.attn_q.weight"> : tensor<256x256xf32>
  util.global private @__auto.blk.1.attn_k.weight = #stream.parameter.named<"model"::"blk.1.attn_k.weight"> : tensor<128x256xf32>
  util.global private @__auto.blk.1.attn_v.weight = #stream.parameter.named<"model"::"blk.1.attn_v.weight"> : tensor<128x256xf32>
  util.global private @__auto.blk.1.attn_output.weight = #stream.parameter.named<"model"::"blk.1.attn_output.weight"> : tensor<256x256xf32>
  util.global private @__auto.blk.1.ffn_norm.weight = #stream.parameter.named<"model"::"blk.1.ffn_norm.weight"> : tensor<256xf32>
  util.global private @__auto.blk.1.ffn_gate.weight = #stream.parameter.named<"model"::"blk.1.ffn_gate.weight"> : tensor<23x256xf32>
  util.global private @__auto.blk.1.ffn_up.weight = #stream.parameter.named<"model"::"blk.1.ffn_up.weight"> : tensor<23x256xf32>
  util.global private @__auto.blk.1.ffn_down.weight = #stream.parameter.named<"model"::"blk.1.ffn_down.weight"> : tensor<256x23xf32>
  util.global private @__auto.blk.2.attn_norm.weight = #stream.parameter.named<"model"::"blk.2.attn_norm.weight"> : tensor<256xf32>
  util.global private @__auto.blk.2.attn_q.weight = #stream.parameter.named<"model"::"blk.2.attn_q.weight"> : tensor<256x256xf32>
  util.global private @__auto.blk.2.attn_k.weight = #stream.parameter.named<"model"::"blk.2.attn_k.weight"> : tensor<128x256xf32>
  util.global private @__auto.blk.2.attn_v.weight = #stream.parameter.named<"model"::"blk.2.attn_v.weight"> : tensor<128x256xf32>
  util.global private @__auto.blk.2.attn_output.weight = #stream.parameter.named<"model"::"blk.2.attn_output.weight"> : tensor<256x256xf32>
  util.global private @__auto.blk.2.ffn_norm.weight = #stream.parameter.named<"model"::"blk.2.ffn_norm.weight"> : tensor<256xf32>
  util.global private @__auto.blk.2.ffn_gate.weight = #stream.parameter.named<"model"::"blk.2.ffn_gate.weight"> : tensor<23x256xf32>
  util.global private @__auto.blk.2.ffn_up.weight = #stream.parameter.named<"model"::"blk.2.ffn_up.weight"> : tensor<23x256xf32>
  util.global private @__auto.blk.2.ffn_down.weight = #stream.parameter.named<"model"::"blk.2.ffn_down.weight"> : tensor<256x23xf32>
  util.global private @__auto.output_norm.weight = #stream.parameter.named<"model"::"output_norm.weight"> : tensor<1x256xf32>
  util.global private @__auto.output.weight = #stream.parameter.named<"model"::"output.weight"> : tensor<256x256xf32>
  func.func @prefill_bs1(%arg0: !torch.vtensor<[1,?],si64>, %arg1: !torch.vtensor<[1],si64>, %arg2: !torch.vtensor<[1,?],si64>, %arg3: !torch.tensor<[?,24576],f16>) -> !torch.vtensor<[1,?,256],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %__auto.token_embd.weight = util.global.load @__auto.token_embd.weight : tensor<256x256xf32>
    %0 = torch_c.from_builtin_tensor %__auto.token_embd.weight : tensor<256x256xf32> -> !torch.vtensor<[256,256],f32>
    %__auto.blk.0.attn_norm.weight = util.global.load @__auto.blk.0.attn_norm.weight : tensor<256xf32>
    %1 = torch_c.from_builtin_tensor %__auto.blk.0.attn_norm.weight : tensor<256xf32> -> !torch.vtensor<[256],f32>
    %__auto.blk.0.attn_q.weight = util.global.load @__auto.blk.0.attn_q.weight : tensor<256x256xf32>
    %2 = torch_c.from_builtin_tensor %__auto.blk.0.attn_q.weight : tensor<256x256xf32> -> !torch.vtensor<[256,256],f32>
    %__auto.blk.0.attn_k.weight = util.global.load @__auto.blk.0.attn_k.weight : tensor<128x256xf32>
    %3 = torch_c.from_builtin_tensor %__auto.blk.0.attn_k.weight : tensor<128x256xf32> -> !torch.vtensor<[128,256],f32>
    %__auto.blk.0.attn_v.weight = util.global.load @__auto.blk.0.attn_v.weight : tensor<128x256xf32>
    %4 = torch_c.from_builtin_tensor %__auto.blk.0.attn_v.weight : tensor<128x256xf32> -> !torch.vtensor<[128,256],f32>
    %__auto.blk.0.attn_output.weight = util.global.load @__auto.blk.0.attn_output.weight : tensor<256x256xf32>
    %5 = torch_c.from_builtin_tensor %__auto.blk.0.attn_output.weight : tensor<256x256xf32> -> !torch.vtensor<[256,256],f32>
    %__auto.blk.0.ffn_norm.weight = util.global.load @__auto.blk.0.ffn_norm.weight : tensor<256xf32>
    %6 = torch_c.from_builtin_tensor %__auto.blk.0.ffn_norm.weight : tensor<256xf32> -> !torch.vtensor<[256],f32>
    %__auto.blk.0.ffn_gate.weight = util.global.load @__auto.blk.0.ffn_gate.weight : tensor<23x256xf32>
    %7 = torch_c.from_builtin_tensor %__auto.blk.0.ffn_gate.weight : tensor<23x256xf32> -> !torch.vtensor<[23,256],f32>
    %__auto.blk.0.ffn_up.weight = util.global.load @__auto.blk.0.ffn_up.weight : tensor<23x256xf32>
    %8 = torch_c.from_builtin_tensor %__auto.blk.0.ffn_up.weight : tensor<23x256xf32> -> !torch.vtensor<[23,256],f32>
    %__auto.blk.0.ffn_down.weight = util.global.load @__auto.blk.0.ffn_down.weight : tensor<256x23xf32>
    %9 = torch_c.from_builtin_tensor %__auto.blk.0.ffn_down.weight : tensor<256x23xf32> -> !torch.vtensor<[256,23],f32>
    %__auto.blk.1.attn_norm.weight = util.global.load @__auto.blk.1.attn_norm.weight : tensor<256xf32>
    %10 = torch_c.from_builtin_tensor %__auto.blk.1.attn_norm.weight : tensor<256xf32> -> !torch.vtensor<[256],f32>
    %__auto.blk.1.attn_q.weight = util.global.load @__auto.blk.1.attn_q.weight : tensor<256x256xf32>
    %11 = torch_c.from_builtin_tensor %__auto.blk.1.attn_q.weight : tensor<256x256xf32> -> !torch.vtensor<[256,256],f32>
    %__auto.blk.1.attn_k.weight = util.global.load @__auto.blk.1.attn_k.weight : tensor<128x256xf32>
    %12 = torch_c.from_builtin_tensor %__auto.blk.1.attn_k.weight : tensor<128x256xf32> -> !torch.vtensor<[128,256],f32>
    %__auto.blk.1.attn_v.weight = util.global.load @__auto.blk.1.attn_v.weight : tensor<128x256xf32>
    %13 = torch_c.from_builtin_tensor %__auto.blk.1.attn_v.weight : tensor<128x256xf32> -> !torch.vtensor<[128,256],f32>
    %__auto.blk.1.attn_output.weight = util.global.load @__auto.blk.1.attn_output.weight : tensor<256x256xf32>
    %14 = torch_c.from_builtin_tensor %__auto.blk.1.attn_output.weight : tensor<256x256xf32> -> !torch.vtensor<[256,256],f32>
    %__auto.blk.1.ffn_norm.weight = util.global.load @__auto.blk.1.ffn_norm.weight : tensor<256xf32>
    %15 = torch_c.from_builtin_tensor %__auto.blk.1.ffn_norm.weight : tensor<256xf32> -> !torch.vtensor<[256],f32>
    %__auto.blk.1.ffn_gate.weight = util.global.load @__auto.blk.1.ffn_gate.weight : tensor<23x256xf32>
    %16 = torch_c.from_builtin_tensor %__auto.blk.1.ffn_gate.weight : tensor<23x256xf32> -> !torch.vtensor<[23,256],f32>
    %__auto.blk.1.ffn_up.weight = util.global.load @__auto.blk.1.ffn_up.weight : tensor<23x256xf32>
    %17 = torch_c.from_builtin_tensor %__auto.blk.1.ffn_up.weight : tensor<23x256xf32> -> !torch.vtensor<[23,256],f32>
    %__auto.blk.1.ffn_down.weight = util.global.load @__auto.blk.1.ffn_down.weight : tensor<256x23xf32>
    %18 = torch_c.from_builtin_tensor %__auto.blk.1.ffn_down.weight : tensor<256x23xf32> -> !torch.vtensor<[256,23],f32>
    %__auto.blk.2.attn_norm.weight = util.global.load @__auto.blk.2.attn_norm.weight : tensor<256xf32>
    %19 = torch_c.from_builtin_tensor %__auto.blk.2.attn_norm.weight : tensor<256xf32> -> !torch.vtensor<[256],f32>
    %__auto.blk.2.attn_q.weight = util.global.load @__auto.blk.2.attn_q.weight : tensor<256x256xf32>
    %20 = torch_c.from_builtin_tensor %__auto.blk.2.attn_q.weight : tensor<256x256xf32> -> !torch.vtensor<[256,256],f32>
    %__auto.blk.2.attn_k.weight = util.global.load @__auto.blk.2.attn_k.weight : tensor<128x256xf32>
    %21 = torch_c.from_builtin_tensor %__auto.blk.2.attn_k.weight : tensor<128x256xf32> -> !torch.vtensor<[128,256],f32>
    %__auto.blk.2.attn_v.weight = util.global.load @__auto.blk.2.attn_v.weight : tensor<128x256xf32>
    %22 = torch_c.from_builtin_tensor %__auto.blk.2.attn_v.weight : tensor<128x256xf32> -> !torch.vtensor<[128,256],f32>
    %__auto.blk.2.attn_output.weight = util.global.load @__auto.blk.2.attn_output.weight : tensor<256x256xf32>
    %23 = torch_c.from_builtin_tensor %__auto.blk.2.attn_output.weight : tensor<256x256xf32> -> !torch.vtensor<[256,256],f32>
    %__auto.blk.2.ffn_norm.weight = util.global.load @__auto.blk.2.ffn_norm.weight : tensor<256xf32>
    %24 = torch_c.from_builtin_tensor %__auto.blk.2.ffn_norm.weight : tensor<256xf32> -> !torch.vtensor<[256],f32>
    %__auto.blk.2.ffn_gate.weight = util.global.load @__auto.blk.2.ffn_gate.weight : tensor<23x256xf32>
    %25 = torch_c.from_builtin_tensor %__auto.blk.2.ffn_gate.weight : tensor<23x256xf32> -> !torch.vtensor<[23,256],f32>
    %__auto.blk.2.ffn_up.weight = util.global.load @__auto.blk.2.ffn_up.weight : tensor<23x256xf32>
    %26 = torch_c.from_builtin_tensor %__auto.blk.2.ffn_up.weight : tensor<23x256xf32> -> !torch.vtensor<[23,256],f32>
    %__auto.blk.2.ffn_down.weight = util.global.load @__auto.blk.2.ffn_down.weight : tensor<256x23xf32>
    %27 = torch_c.from_builtin_tensor %__auto.blk.2.ffn_down.weight : tensor<256x23xf32> -> !torch.vtensor<[256,23],f32>
    %__auto.output_norm.weight = util.global.load @__auto.output_norm.weight : tensor<1x256xf32>
    %28 = torch_c.from_builtin_tensor %__auto.output_norm.weight : tensor<1x256xf32> -> !torch.vtensor<[1,256],f32>
    %__auto.output.weight = util.global.load @__auto.output.weight : tensor<256x256xf32>
    %29 = torch_c.from_builtin_tensor %__auto.output.weight : tensor<256x256xf32> -> !torch.vtensor<[256,256],f32>
    %30 = torch.copy.to_vtensor %arg3 : !torch.vtensor<[?,24576],f16>
    %31 = torch.symbolic_int "s1" {min_val = 2, max_val = 3} : !torch.int
    %32 = torch.symbolic_int "s2" {min_val = 2, max_val = 9223372036854775806} : !torch.int
    torch.bind_symbolic_shape %arg0, [%31], affine_map<()[s0] -> (1, s0 * 32)> : !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %arg2, [%31], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %30, [%32], affine_map<()[s0] -> (s0, 24576)> : !torch.vtensor<[?,24576],f16>
    %int5 = torch.constant.int 5
    %33 = torch.prims.convert_element_type %0, %int5 : !torch.vtensor<[256,256],f32>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int-1 = torch.constant.int -1
    %false = torch.constant.bool false
    %false_0 = torch.constant.bool false
    %34 = torch.aten.embedding %33, %arg0, %int-1, %false, %false_0 : !torch.vtensor<[256,256],f16>, !torch.vtensor<[1,?],si64>, !torch.int, !torch.bool, !torch.bool -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %34, [%31], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int6 = torch.constant.int 6
    %35 = torch.prims.convert_element_type %34, %int6 : !torch.vtensor<[1,?,256],f16>, !torch.int -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %35, [%31], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int2 = torch.constant.int 2
    %36 = torch.aten.pow.Tensor_Scalar %35, %int2 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %36, [%31], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int-1_1 = torch.constant.int -1
    %37 = torch.prim.ListConstruct %int-1_1 : (!torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %none = torch.constant.none
    %38 = torch.aten.mean.dim %36, %37, %true, %none : !torch.vtensor<[1,?,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %38, [%31], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %float1.000000e-02 = torch.constant.float 1.000000e-02
    %int1 = torch.constant.int 1
    %39 = torch.aten.add.Scalar %38, %float1.000000e-02, %int1 : !torch.vtensor<[1,?,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %39, [%31], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %40 = torch.aten.rsqrt %39 : !torch.vtensor<[1,?,1],f32> -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %40, [%31], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %41 = torch.aten.mul.Tensor %35, %40 : !torch.vtensor<[1,?,256],f32>, !torch.vtensor<[1,?,1],f32> -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %41, [%31], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int5_2 = torch.constant.int 5
    %42 = torch.prims.convert_element_type %41, %int5_2 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %42, [%31], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %43 = torch.aten.mul.Tensor %1, %42 : !torch.vtensor<[256],f32>, !torch.vtensor<[1,?,256],f16> -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %43, [%31], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int5_3 = torch.constant.int 5
    %44 = torch.prims.convert_element_type %43, %int5_3 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %44, [%31], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int5_4 = torch.constant.int 5
    %45 = torch.prims.convert_element_type %2, %int5_4 : !torch.vtensor<[256,256],f32>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int-2 = torch.constant.int -2
    %int-1_5 = torch.constant.int -1
    %46 = torch.aten.transpose.int %45, %int-2, %int-1_5 : !torch.vtensor<[256,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,256],f16>
    %int1_6 = torch.constant.int 1
    %47 = torch.aten.size.int %arg0, %int1_6 : !torch.vtensor<[1,?],si64>, !torch.int -> !torch.int
    %int256 = torch.constant.int 256
    %48 = torch.prim.ListConstruct %47, %int256 : (!torch.int, !torch.int) -> !torch.list<int>
    %49 = torch.aten.view %44, %48 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %49, [%31], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %50 = torch.aten.mm %49, %46 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,256],f16> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %50, [%31], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %int1_7 = torch.constant.int 1
    %int256_8 = torch.constant.int 256
    %51 = torch.prim.ListConstruct %int1_7, %47, %int256_8 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %52 = torch.aten.view %50, %51 : !torch.vtensor<[?,256],f16>, !torch.list<int> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %52, [%31], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int5_9 = torch.constant.int 5
    %53 = torch.prims.convert_element_type %3, %int5_9 : !torch.vtensor<[128,256],f32>, !torch.int -> !torch.vtensor<[128,256],f16>
    %int-2_10 = torch.constant.int -2
    %int-1_11 = torch.constant.int -1
    %54 = torch.aten.transpose.int %53, %int-2_10, %int-1_11 : !torch.vtensor<[128,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,128],f16>
    %int256_12 = torch.constant.int 256
    %55 = torch.prim.ListConstruct %47, %int256_12 : (!torch.int, !torch.int) -> !torch.list<int>
    %56 = torch.aten.view %44, %55 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %56, [%31], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %57 = torch.aten.mm %56, %54 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,128],f16> -> !torch.vtensor<[?,128],f16>
    torch.bind_symbolic_shape %57, [%31], affine_map<()[s0] -> (s0 * 32, 128)> : !torch.vtensor<[?,128],f16>
    %int1_13 = torch.constant.int 1
    %int128 = torch.constant.int 128
    %58 = torch.prim.ListConstruct %int1_13, %47, %int128 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %59 = torch.aten.view %57, %58 : !torch.vtensor<[?,128],f16>, !torch.list<int> -> !torch.vtensor<[1,?,128],f16>
    torch.bind_symbolic_shape %59, [%31], affine_map<()[s0] -> (1, s0 * 32, 128)> : !torch.vtensor<[1,?,128],f16>
    %int5_14 = torch.constant.int 5
    %60 = torch.prims.convert_element_type %4, %int5_14 : !torch.vtensor<[128,256],f32>, !torch.int -> !torch.vtensor<[128,256],f16>
    %int-2_15 = torch.constant.int -2
    %int-1_16 = torch.constant.int -1
    %61 = torch.aten.transpose.int %60, %int-2_15, %int-1_16 : !torch.vtensor<[128,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,128],f16>
    %int256_17 = torch.constant.int 256
    %62 = torch.prim.ListConstruct %47, %int256_17 : (!torch.int, !torch.int) -> !torch.list<int>
    %63 = torch.aten.view %44, %62 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %63, [%31], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %64 = torch.aten.mm %63, %61 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,128],f16> -> !torch.vtensor<[?,128],f16>
    torch.bind_symbolic_shape %64, [%31], affine_map<()[s0] -> (s0 * 32, 128)> : !torch.vtensor<[?,128],f16>
    %int1_18 = torch.constant.int 1
    %int128_19 = torch.constant.int 128
    %65 = torch.prim.ListConstruct %int1_18, %47, %int128_19 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %66 = torch.aten.view %64, %65 : !torch.vtensor<[?,128],f16>, !torch.list<int> -> !torch.vtensor<[1,?,128],f16>
    torch.bind_symbolic_shape %66, [%31], affine_map<()[s0] -> (1, s0 * 32, 128)> : !torch.vtensor<[1,?,128],f16>
    %int1_20 = torch.constant.int 1
    %int8 = torch.constant.int 8
    %int32 = torch.constant.int 32
    %67 = torch.prim.ListConstruct %int1_20, %47, %int8, %int32 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %68 = torch.aten.view %52, %67 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[1,?,8,32],f16>
    torch.bind_symbolic_shape %68, [%31], affine_map<()[s0] -> (1, s0 * 32, 8, 32)> : !torch.vtensor<[1,?,8,32],f16>
    %int1_21 = torch.constant.int 1
    %int4 = torch.constant.int 4
    %int32_22 = torch.constant.int 32
    %69 = torch.prim.ListConstruct %int1_21, %47, %int4, %int32_22 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %70 = torch.aten.view %59, %69 : !torch.vtensor<[1,?,128],f16>, !torch.list<int> -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %70, [%31], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int1_23 = torch.constant.int 1
    %int4_24 = torch.constant.int 4
    %int32_25 = torch.constant.int 32
    %71 = torch.prim.ListConstruct %int1_23, %47, %int4_24, %int32_25 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %72 = torch.aten.view %66, %71 : !torch.vtensor<[1,?,128],f16>, !torch.list<int> -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %72, [%31], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int128_26 = torch.constant.int 128
    %none_27 = torch.constant.none
    %none_28 = torch.constant.none
    %cpu = torch.constant.device "cpu"
    %false_29 = torch.constant.bool false
    %73 = torch.aten.arange %int128_26, %none_27, %none_28, %cpu, %false_29 : !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[128],si64>
    %int0 = torch.constant.int 0
    %int32_30 = torch.constant.int 32
    %none_31 = torch.constant.none
    %none_32 = torch.constant.none
    %cpu_33 = torch.constant.device "cpu"
    %false_34 = torch.constant.bool false
    %74 = torch.aten.arange.start %int0, %int32_30, %none_31, %none_32, %cpu_33, %false_34 : !torch.int, !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[32],si64>
    %int2_35 = torch.constant.int 2
    %75 = torch.aten.floor_divide.Scalar %74, %int2_35 : !torch.vtensor<[32],si64>, !torch.int -> !torch.vtensor<[32],si64>
    %int6_36 = torch.constant.int 6
    %76 = torch.prims.convert_element_type %75, %int6_36 : !torch.vtensor<[32],si64>, !torch.int -> !torch.vtensor<[32],f32>
    %int32_37 = torch.constant.int 32
    %77 = torch.aten.div.Scalar %76, %int32_37 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[32],f32>
    %float2.000000e00 = torch.constant.float 2.000000e+00
    %78 = torch.aten.mul.Scalar %77, %float2.000000e00 : !torch.vtensor<[32],f32>, !torch.float -> !torch.vtensor<[32],f32>
    %float5.000000e05 = torch.constant.float 5.000000e+05
    %79 = torch.aten.pow.Scalar %float5.000000e05, %78 : !torch.float, !torch.vtensor<[32],f32> -> !torch.vtensor<[32],f32>
    %80 = torch.aten.reciprocal %79 : !torch.vtensor<[32],f32> -> !torch.vtensor<[32],f32>
    %float1.000000e00 = torch.constant.float 1.000000e+00
    %81 = torch.aten.mul.Scalar %80, %float1.000000e00 : !torch.vtensor<[32],f32>, !torch.float -> !torch.vtensor<[32],f32>
    %int1_38 = torch.constant.int 1
    %82 = torch.aten.unsqueeze %73, %int1_38 : !torch.vtensor<[128],si64>, !torch.int -> !torch.vtensor<[128,1],si64>
    %int0_39 = torch.constant.int 0
    %83 = torch.aten.unsqueeze %81, %int0_39 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[1,32],f32>
    %84 = torch.aten.mul.Tensor %82, %83 : !torch.vtensor<[128,1],si64>, !torch.vtensor<[1,32],f32> -> !torch.vtensor<[128,32],f32>
    %int1_40 = torch.constant.int 1
    %85 = torch.aten.size.int %52, %int1_40 : !torch.vtensor<[1,?,256],f16>, !torch.int -> !torch.int
    %int0_41 = torch.constant.int 0
    %86 = torch.aten.add.int %int0_41, %85 : !torch.int, !torch.int -> !torch.int
    %int0_42 = torch.constant.int 0
    %int0_43 = torch.constant.int 0
    %int1_44 = torch.constant.int 1
    %87 = torch.aten.slice.Tensor %84, %int0_42, %int0_43, %86, %int1_44 : !torch.vtensor<[128,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %87, [%31], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int1_45 = torch.constant.int 1
    %int0_46 = torch.constant.int 0
    %int9223372036854775807 = torch.constant.int 9223372036854775807
    %int1_47 = torch.constant.int 1
    %88 = torch.aten.slice.Tensor %87, %int1_45, %int0_46, %int9223372036854775807, %int1_47 : !torch.vtensor<[?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %88, [%31], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int1_48 = torch.constant.int 1
    %int0_49 = torch.constant.int 0
    %int9223372036854775807_50 = torch.constant.int 9223372036854775807
    %int1_51 = torch.constant.int 1
    %89 = torch.aten.slice.Tensor %88, %int1_48, %int0_49, %int9223372036854775807_50, %int1_51 : !torch.vtensor<[?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %89, [%31], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int0_52 = torch.constant.int 0
    %90 = torch.aten.unsqueeze %89, %int0_52 : !torch.vtensor<[?,32],f32>, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %90, [%31], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int1_53 = torch.constant.int 1
    %int0_54 = torch.constant.int 0
    %int9223372036854775807_55 = torch.constant.int 9223372036854775807
    %int1_56 = torch.constant.int 1
    %91 = torch.aten.slice.Tensor %90, %int1_53, %int0_54, %int9223372036854775807_55, %int1_56 : !torch.vtensor<[1,?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %91, [%31], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int2_57 = torch.constant.int 2
    %int0_58 = torch.constant.int 0
    %int9223372036854775807_59 = torch.constant.int 9223372036854775807
    %int1_60 = torch.constant.int 1
    %92 = torch.aten.slice.Tensor %91, %int2_57, %int0_58, %int9223372036854775807_59, %int1_60 : !torch.vtensor<[1,?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %92, [%31], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int1_61 = torch.constant.int 1
    %int1_62 = torch.constant.int 1
    %int1_63 = torch.constant.int 1
    %93 = torch.prim.ListConstruct %int1_61, %int1_62, %int1_63 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %94 = torch.aten.repeat %92, %93 : !torch.vtensor<[1,?,32],f32>, !torch.list<int> -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %94, [%31], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int6_64 = torch.constant.int 6
    %95 = torch.prims.convert_element_type %68, %int6_64 : !torch.vtensor<[1,?,8,32],f16>, !torch.int -> !torch.vtensor<[1,?,8,32],f32>
    torch.bind_symbolic_shape %95, [%31], affine_map<()[s0] -> (1, s0 * 32, 8, 32)> : !torch.vtensor<[1,?,8,32],f32>
    %96 = torch_c.to_builtin_tensor %95 : !torch.vtensor<[1,?,8,32],f32> -> tensor<1x?x8x32xf32>
    %97 = torch_c.to_builtin_tensor %94 : !torch.vtensor<[1,?,32],f32> -> tensor<1x?x32xf32>
    %98 = util.call @sharktank_rotary_embedding_1_D_8_32_f32(%96, %97) : (tensor<1x?x8x32xf32>, tensor<1x?x32xf32>) -> tensor<1x?x8x32xf32>
    %99 = torch_c.from_builtin_tensor %98 : tensor<1x?x8x32xf32> -> !torch.vtensor<[1,?,8,32],f32>
    torch.bind_symbolic_shape %99, [%31], affine_map<()[s0] -> (1, s0 * 32, 8, 32)> : !torch.vtensor<[1,?,8,32],f32>
    %int5_65 = torch.constant.int 5
    %100 = torch.prims.convert_element_type %99, %int5_65 : !torch.vtensor<[1,?,8,32],f32>, !torch.int -> !torch.vtensor<[1,?,8,32],f16>
    torch.bind_symbolic_shape %100, [%31], affine_map<()[s0] -> (1, s0 * 32, 8, 32)> : !torch.vtensor<[1,?,8,32],f16>
    %int128_66 = torch.constant.int 128
    %none_67 = torch.constant.none
    %none_68 = torch.constant.none
    %cpu_69 = torch.constant.device "cpu"
    %false_70 = torch.constant.bool false
    %101 = torch.aten.arange %int128_66, %none_67, %none_68, %cpu_69, %false_70 : !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[128],si64>
    %int0_71 = torch.constant.int 0
    %int32_72 = torch.constant.int 32
    %none_73 = torch.constant.none
    %none_74 = torch.constant.none
    %cpu_75 = torch.constant.device "cpu"
    %false_76 = torch.constant.bool false
    %102 = torch.aten.arange.start %int0_71, %int32_72, %none_73, %none_74, %cpu_75, %false_76 : !torch.int, !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[32],si64>
    %int2_77 = torch.constant.int 2
    %103 = torch.aten.floor_divide.Scalar %102, %int2_77 : !torch.vtensor<[32],si64>, !torch.int -> !torch.vtensor<[32],si64>
    %int6_78 = torch.constant.int 6
    %104 = torch.prims.convert_element_type %103, %int6_78 : !torch.vtensor<[32],si64>, !torch.int -> !torch.vtensor<[32],f32>
    %int32_79 = torch.constant.int 32
    %105 = torch.aten.div.Scalar %104, %int32_79 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[32],f32>
    %float2.000000e00_80 = torch.constant.float 2.000000e+00
    %106 = torch.aten.mul.Scalar %105, %float2.000000e00_80 : !torch.vtensor<[32],f32>, !torch.float -> !torch.vtensor<[32],f32>
    %float5.000000e05_81 = torch.constant.float 5.000000e+05
    %107 = torch.aten.pow.Scalar %float5.000000e05_81, %106 : !torch.float, !torch.vtensor<[32],f32> -> !torch.vtensor<[32],f32>
    %108 = torch.aten.reciprocal %107 : !torch.vtensor<[32],f32> -> !torch.vtensor<[32],f32>
    %float1.000000e00_82 = torch.constant.float 1.000000e+00
    %109 = torch.aten.mul.Scalar %108, %float1.000000e00_82 : !torch.vtensor<[32],f32>, !torch.float -> !torch.vtensor<[32],f32>
    %int1_83 = torch.constant.int 1
    %110 = torch.aten.unsqueeze %101, %int1_83 : !torch.vtensor<[128],si64>, !torch.int -> !torch.vtensor<[128,1],si64>
    %int0_84 = torch.constant.int 0
    %111 = torch.aten.unsqueeze %109, %int0_84 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[1,32],f32>
    %112 = torch.aten.mul.Tensor %110, %111 : !torch.vtensor<[128,1],si64>, !torch.vtensor<[1,32],f32> -> !torch.vtensor<[128,32],f32>
    %int1_85 = torch.constant.int 1
    %113 = torch.aten.size.int %59, %int1_85 : !torch.vtensor<[1,?,128],f16>, !torch.int -> !torch.int
    %int0_86 = torch.constant.int 0
    %114 = torch.aten.add.int %int0_86, %113 : !torch.int, !torch.int -> !torch.int
    %int0_87 = torch.constant.int 0
    %int0_88 = torch.constant.int 0
    %int1_89 = torch.constant.int 1
    %115 = torch.aten.slice.Tensor %112, %int0_87, %int0_88, %114, %int1_89 : !torch.vtensor<[128,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %115, [%31], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int1_90 = torch.constant.int 1
    %int0_91 = torch.constant.int 0
    %int9223372036854775807_92 = torch.constant.int 9223372036854775807
    %int1_93 = torch.constant.int 1
    %116 = torch.aten.slice.Tensor %115, %int1_90, %int0_91, %int9223372036854775807_92, %int1_93 : !torch.vtensor<[?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %116, [%31], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int1_94 = torch.constant.int 1
    %int0_95 = torch.constant.int 0
    %int9223372036854775807_96 = torch.constant.int 9223372036854775807
    %int1_97 = torch.constant.int 1
    %117 = torch.aten.slice.Tensor %116, %int1_94, %int0_95, %int9223372036854775807_96, %int1_97 : !torch.vtensor<[?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %117, [%31], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int0_98 = torch.constant.int 0
    %118 = torch.aten.unsqueeze %117, %int0_98 : !torch.vtensor<[?,32],f32>, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %118, [%31], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int1_99 = torch.constant.int 1
    %int0_100 = torch.constant.int 0
    %int9223372036854775807_101 = torch.constant.int 9223372036854775807
    %int1_102 = torch.constant.int 1
    %119 = torch.aten.slice.Tensor %118, %int1_99, %int0_100, %int9223372036854775807_101, %int1_102 : !torch.vtensor<[1,?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %119, [%31], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int2_103 = torch.constant.int 2
    %int0_104 = torch.constant.int 0
    %int9223372036854775807_105 = torch.constant.int 9223372036854775807
    %int1_106 = torch.constant.int 1
    %120 = torch.aten.slice.Tensor %119, %int2_103, %int0_104, %int9223372036854775807_105, %int1_106 : !torch.vtensor<[1,?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %120, [%31], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int1_107 = torch.constant.int 1
    %int1_108 = torch.constant.int 1
    %int1_109 = torch.constant.int 1
    %121 = torch.prim.ListConstruct %int1_107, %int1_108, %int1_109 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %122 = torch.aten.repeat %120, %121 : !torch.vtensor<[1,?,32],f32>, !torch.list<int> -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %122, [%31], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int6_110 = torch.constant.int 6
    %123 = torch.prims.convert_element_type %70, %int6_110 : !torch.vtensor<[1,?,4,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,32],f32>
    torch.bind_symbolic_shape %123, [%31], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f32>
    %124 = torch_c.to_builtin_tensor %123 : !torch.vtensor<[1,?,4,32],f32> -> tensor<1x?x4x32xf32>
    %125 = torch_c.to_builtin_tensor %122 : !torch.vtensor<[1,?,32],f32> -> tensor<1x?x32xf32>
    %126 = util.call @sharktank_rotary_embedding_1_D_4_32_f32(%124, %125) : (tensor<1x?x4x32xf32>, tensor<1x?x32xf32>) -> tensor<1x?x4x32xf32>
    %127 = torch_c.from_builtin_tensor %126 : tensor<1x?x4x32xf32> -> !torch.vtensor<[1,?,4,32],f32>
    torch.bind_symbolic_shape %127, [%31], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f32>
    %int5_111 = torch.constant.int 5
    %128 = torch.prims.convert_element_type %127, %int5_111 : !torch.vtensor<[1,?,4,32],f32>, !torch.int -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %128, [%31], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int0_112 = torch.constant.int 0
    %129 = torch.aten.size.int %30, %int0_112 : !torch.vtensor<[?,24576],f16>, !torch.int -> !torch.int
    %int3 = torch.constant.int 3
    %int2_113 = torch.constant.int 2
    %int32_114 = torch.constant.int 32
    %int4_115 = torch.constant.int 4
    %int32_116 = torch.constant.int 32
    %130 = torch.prim.ListConstruct %129, %int3, %int2_113, %int32_114, %int4_115, %int32_116 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %131 = torch.aten.view %30, %130 : !torch.vtensor<[?,24576],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,4,32],f16>
    torch.bind_symbolic_shape %131, [%32], affine_map<()[s0] -> (s0, 3, 2, 32, 4, 32)> : !torch.vtensor<[?,3,2,32,4,32],f16>
    %int3_117 = torch.constant.int 3
    %132 = torch.aten.mul.int %129, %int3_117 : !torch.int, !torch.int -> !torch.int
    %int2_118 = torch.constant.int 2
    %133 = torch.aten.mul.int %132, %int2_118 : !torch.int, !torch.int -> !torch.int
    %int32_119 = torch.constant.int 32
    %int4_120 = torch.constant.int 4
    %int32_121 = torch.constant.int 32
    %134 = torch.prim.ListConstruct %133, %int32_119, %int4_120, %int32_121 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %135 = torch.aten.view %131, %134 : !torch.vtensor<[?,3,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,32,4,32],f16>
    torch.bind_symbolic_shape %135, [%32], affine_map<()[s0] -> (s0 * 6, 32, 4, 32)> : !torch.vtensor<[?,32,4,32],f16>
    %int6_122 = torch.constant.int 6
    %136 = torch.aten.mul.Scalar %arg2, %int6_122 : !torch.vtensor<[1,?],si64>, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %136, [%31], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %int0_123 = torch.constant.int 0
    %int1_124 = torch.constant.int 1
    %137 = torch.aten.add.Scalar %136, %int0_123, %int1_124 : !torch.vtensor<[1,?],si64>, !torch.int, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %137, [%31], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %int1_125 = torch.constant.int 1
    %138 = torch.aten.size.int %arg2, %int1_125 : !torch.vtensor<[1,?],si64>, !torch.int -> !torch.int
    %int1_126 = torch.constant.int 1
    %int32_127 = torch.constant.int 32
    %int4_128 = torch.constant.int 4
    %int32_129 = torch.constant.int 32
    %139 = torch.prim.ListConstruct %int1_126, %138, %int32_127, %int4_128, %int32_129 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %140 = torch.aten.view %128, %139 : !torch.vtensor<[1,?,4,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,32,4,32],f16>
    torch.bind_symbolic_shape %140, [%31], affine_map<()[s0] -> (1, s0, 32, 4, 32)> : !torch.vtensor<[1,?,32,4,32],f16>
    %int32_130 = torch.constant.int 32
    %int4_131 = torch.constant.int 4
    %int32_132 = torch.constant.int 32
    %141 = torch.prim.ListConstruct %138, %int32_130, %int4_131, %int32_132 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %142 = torch.aten.view %140, %141 : !torch.vtensor<[1,?,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,32,4,32],f16>
    torch.bind_symbolic_shape %142, [%31], affine_map<()[s0] -> (s0, 32, 4, 32)> : !torch.vtensor<[?,32,4,32],f16>
    %143 = torch.prim.ListConstruct %138 : (!torch.int) -> !torch.list<int>
    %144 = torch.aten.view %137, %143 : !torch.vtensor<[1,?],si64>, !torch.list<int> -> !torch.vtensor<[?],si64>
    torch.bind_symbolic_shape %144, [%31], affine_map<()[s0] -> (s0)> : !torch.vtensor<[?],si64>
    %145 = torch.prim.ListConstruct %144 : (!torch.vtensor<[?],si64>) -> !torch.list<optional<vtensor>>
    %false_133 = torch.constant.bool false
    %146 = torch.aten.index_put %135, %145, %142, %false_133 : !torch.vtensor<[?,32,4,32],f16>, !torch.list<optional<vtensor>>, !torch.vtensor<[?,32,4,32],f16>, !torch.bool -> !torch.vtensor<[?,32,4,32],f16>
    torch.bind_symbolic_shape %146, [%32], affine_map<()[s0] -> (s0 * 6, 32, 4, 32)> : !torch.vtensor<[?,32,4,32],f16>
    %int3_134 = torch.constant.int 3
    %int2_135 = torch.constant.int 2
    %int32_136 = torch.constant.int 32
    %int4_137 = torch.constant.int 4
    %int32_138 = torch.constant.int 32
    %147 = torch.prim.ListConstruct %129, %int3_134, %int2_135, %int32_136, %int4_137, %int32_138 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %148 = torch.aten.view %146, %147 : !torch.vtensor<[?,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,4,32],f16>
    torch.bind_symbolic_shape %148, [%32], affine_map<()[s0] -> (s0, 3, 2, 32, 4, 32)> : !torch.vtensor<[?,3,2,32,4,32],f16>
    %int24576 = torch.constant.int 24576
    %149 = torch.prim.ListConstruct %129, %int24576 : (!torch.int, !torch.int) -> !torch.list<int>
    %150 = torch.aten.view %148, %149 : !torch.vtensor<[?,3,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,24576],f16>
    torch.bind_symbolic_shape %150, [%32], affine_map<()[s0] -> (s0, 24576)> : !torch.vtensor<[?,24576],f16>
    %int3_139 = torch.constant.int 3
    %int2_140 = torch.constant.int 2
    %int32_141 = torch.constant.int 32
    %int4_142 = torch.constant.int 4
    %int32_143 = torch.constant.int 32
    %151 = torch.prim.ListConstruct %129, %int3_139, %int2_140, %int32_141, %int4_142, %int32_143 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %152 = torch.aten.view %150, %151 : !torch.vtensor<[?,24576],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,4,32],f16>
    torch.bind_symbolic_shape %152, [%32], affine_map<()[s0] -> (s0, 3, 2, 32, 4, 32)> : !torch.vtensor<[?,3,2,32,4,32],f16>
    %int32_144 = torch.constant.int 32
    %int4_145 = torch.constant.int 4
    %int32_146 = torch.constant.int 32
    %153 = torch.prim.ListConstruct %133, %int32_144, %int4_145, %int32_146 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %154 = torch.aten.view %152, %153 : !torch.vtensor<[?,3,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,32,4,32],f16>
    torch.bind_symbolic_shape %154, [%32], affine_map<()[s0] -> (s0 * 6, 32, 4, 32)> : !torch.vtensor<[?,32,4,32],f16>
    %int1_147 = torch.constant.int 1
    %int32_148 = torch.constant.int 32
    %int4_149 = torch.constant.int 4
    %int32_150 = torch.constant.int 32
    %155 = torch.prim.ListConstruct %int1_147, %138, %int32_148, %int4_149, %int32_150 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %156 = torch.aten.view %72, %155 : !torch.vtensor<[1,?,4,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,32,4,32],f16>
    torch.bind_symbolic_shape %156, [%31], affine_map<()[s0] -> (1, s0, 32, 4, 32)> : !torch.vtensor<[1,?,32,4,32],f16>
    %int32_151 = torch.constant.int 32
    %int4_152 = torch.constant.int 4
    %int32_153 = torch.constant.int 32
    %157 = torch.prim.ListConstruct %138, %int32_151, %int4_152, %int32_153 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %158 = torch.aten.view %156, %157 : !torch.vtensor<[1,?,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,32,4,32],f16>
    torch.bind_symbolic_shape %158, [%31], affine_map<()[s0] -> (s0, 32, 4, 32)> : !torch.vtensor<[?,32,4,32],f16>
    %int1_154 = torch.constant.int 1
    %int1_155 = torch.constant.int 1
    %159 = torch.aten.add.Scalar %137, %int1_154, %int1_155 : !torch.vtensor<[1,?],si64>, !torch.int, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %159, [%31], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %160 = torch.prim.ListConstruct %138 : (!torch.int) -> !torch.list<int>
    %161 = torch.aten.view %159, %160 : !torch.vtensor<[1,?],si64>, !torch.list<int> -> !torch.vtensor<[?],si64>
    torch.bind_symbolic_shape %161, [%31], affine_map<()[s0] -> (s0)> : !torch.vtensor<[?],si64>
    %162 = torch.prim.ListConstruct %161 : (!torch.vtensor<[?],si64>) -> !torch.list<optional<vtensor>>
    %false_156 = torch.constant.bool false
    %163 = torch.aten.index_put %154, %162, %158, %false_156 : !torch.vtensor<[?,32,4,32],f16>, !torch.list<optional<vtensor>>, !torch.vtensor<[?,32,4,32],f16>, !torch.bool -> !torch.vtensor<[?,32,4,32],f16>
    torch.bind_symbolic_shape %163, [%32], affine_map<()[s0] -> (s0 * 6, 32, 4, 32)> : !torch.vtensor<[?,32,4,32],f16>
    %int3_157 = torch.constant.int 3
    %int2_158 = torch.constant.int 2
    %int32_159 = torch.constant.int 32
    %int4_160 = torch.constant.int 4
    %int32_161 = torch.constant.int 32
    %164 = torch.prim.ListConstruct %129, %int3_157, %int2_158, %int32_159, %int4_160, %int32_161 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %165 = torch.aten.view %163, %164 : !torch.vtensor<[?,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,4,32],f16>
    torch.bind_symbolic_shape %165, [%32], affine_map<()[s0] -> (s0, 3, 2, 32, 4, 32)> : !torch.vtensor<[?,3,2,32,4,32],f16>
    %int24576_162 = torch.constant.int 24576
    %166 = torch.prim.ListConstruct %129, %int24576_162 : (!torch.int, !torch.int) -> !torch.list<int>
    %167 = torch.aten.view %165, %166 : !torch.vtensor<[?,3,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,24576],f16>
    torch.bind_symbolic_shape %167, [%32], affine_map<()[s0] -> (s0, 24576)> : !torch.vtensor<[?,24576],f16>
    %int-2_163 = torch.constant.int -2
    %168 = torch.aten.unsqueeze %128, %int-2_163 : !torch.vtensor<[1,?,4,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,1,32],f16>
    torch.bind_symbolic_shape %168, [%31], affine_map<()[s0] -> (1, s0 * 32, 4, 1, 32)> : !torch.vtensor<[1,?,4,1,32],f16>
    %int1_164 = torch.constant.int 1
    %int4_165 = torch.constant.int 4
    %int2_166 = torch.constant.int 2
    %int32_167 = torch.constant.int 32
    %169 = torch.prim.ListConstruct %int1_164, %113, %int4_165, %int2_166, %int32_167 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_168 = torch.constant.bool false
    %170 = torch.aten.expand %168, %169, %false_168 : !torch.vtensor<[1,?,4,1,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,?,4,2,32],f16>
    torch.bind_symbolic_shape %170, [%31], affine_map<()[s0] -> (1, s0 * 32, 4, 2, 32)> : !torch.vtensor<[1,?,4,2,32],f16>
    %int0_169 = torch.constant.int 0
    %171 = torch.aten.clone %170, %int0_169 : !torch.vtensor<[1,?,4,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,2,32],f16>
    torch.bind_symbolic_shape %171, [%31], affine_map<()[s0] -> (1, s0 * 32, 4, 2, 32)> : !torch.vtensor<[1,?,4,2,32],f16>
    %int1_170 = torch.constant.int 1
    %int8_171 = torch.constant.int 8
    %int32_172 = torch.constant.int 32
    %172 = torch.prim.ListConstruct %int1_170, %113, %int8_171, %int32_172 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %173 = torch.aten._unsafe_view %171, %172 : !torch.vtensor<[1,?,4,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,8,32],f16>
    torch.bind_symbolic_shape %173, [%31], affine_map<()[s0] -> (1, s0 * 32, 8, 32)> : !torch.vtensor<[1,?,8,32],f16>
    %int-2_173 = torch.constant.int -2
    %174 = torch.aten.unsqueeze %72, %int-2_173 : !torch.vtensor<[1,?,4,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,1,32],f16>
    torch.bind_symbolic_shape %174, [%31], affine_map<()[s0] -> (1, s0 * 32, 4, 1, 32)> : !torch.vtensor<[1,?,4,1,32],f16>
    %int1_174 = torch.constant.int 1
    %175 = torch.aten.size.int %66, %int1_174 : !torch.vtensor<[1,?,128],f16>, !torch.int -> !torch.int
    %int1_175 = torch.constant.int 1
    %int4_176 = torch.constant.int 4
    %int2_177 = torch.constant.int 2
    %int32_178 = torch.constant.int 32
    %176 = torch.prim.ListConstruct %int1_175, %175, %int4_176, %int2_177, %int32_178 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_179 = torch.constant.bool false
    %177 = torch.aten.expand %174, %176, %false_179 : !torch.vtensor<[1,?,4,1,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,?,4,2,32],f16>
    torch.bind_symbolic_shape %177, [%31], affine_map<()[s0] -> (1, s0 * 32, 4, 2, 32)> : !torch.vtensor<[1,?,4,2,32],f16>
    %int0_180 = torch.constant.int 0
    %178 = torch.aten.clone %177, %int0_180 : !torch.vtensor<[1,?,4,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,2,32],f16>
    torch.bind_symbolic_shape %178, [%31], affine_map<()[s0] -> (1, s0 * 32, 4, 2, 32)> : !torch.vtensor<[1,?,4,2,32],f16>
    %int1_181 = torch.constant.int 1
    %int8_182 = torch.constant.int 8
    %int32_183 = torch.constant.int 32
    %179 = torch.prim.ListConstruct %int1_181, %175, %int8_182, %int32_183 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %180 = torch.aten._unsafe_view %178, %179 : !torch.vtensor<[1,?,4,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,8,32],f16>
    torch.bind_symbolic_shape %180, [%31], affine_map<()[s0] -> (1, s0 * 32, 8, 32)> : !torch.vtensor<[1,?,8,32],f16>
    %int1_184 = torch.constant.int 1
    %int2_185 = torch.constant.int 2
    %181 = torch.aten.transpose.int %100, %int1_184, %int2_185 : !torch.vtensor<[1,?,8,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,8,?,32],f16>
    torch.bind_symbolic_shape %181, [%31], affine_map<()[s0] -> (1, 8, s0 * 32, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %int1_186 = torch.constant.int 1
    %int2_187 = torch.constant.int 2
    %182 = torch.aten.transpose.int %173, %int1_186, %int2_187 : !torch.vtensor<[1,?,8,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,8,?,32],f16>
    torch.bind_symbolic_shape %182, [%31], affine_map<()[s0] -> (1, 8, s0 * 32, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %int1_188 = torch.constant.int 1
    %int2_189 = torch.constant.int 2
    %183 = torch.aten.transpose.int %180, %int1_188, %int2_189 : !torch.vtensor<[1,?,8,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,8,?,32],f16>
    torch.bind_symbolic_shape %183, [%31], affine_map<()[s0] -> (1, 8, s0 * 32, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %float0.000000e00 = torch.constant.float 0.000000e+00
    %true_190 = torch.constant.bool true
    %none_191 = torch.constant.none
    %none_192 = torch.constant.none
    %184:2 = torch.operator "torch.aten._scaled_dot_product_flash_attention_for_cpu"(%181, %182, %183, %float0.000000e00, %true_190, %none_191, %none_192) : (!torch.vtensor<[1,8,?,32],f16>, !torch.vtensor<[1,8,?,32],f16>, !torch.vtensor<[1,8,?,32],f16>, !torch.float, !torch.bool, !torch.none, !torch.none) -> (!torch.vtensor<[1,8,?,32],f16>, !torch.vtensor<[1,8,?],f32>) 
    torch.bind_symbolic_shape %184#0, [%31], affine_map<()[s0] -> (1, 8, s0 * 32, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %int1_193 = torch.constant.int 1
    %int2_194 = torch.constant.int 2
    %185 = torch.aten.transpose.int %184#0, %int1_193, %int2_194 : !torch.vtensor<[1,8,?,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,?,8,32],f16>
    torch.bind_symbolic_shape %185, [%31], affine_map<()[s0] -> (1, s0 * 32, 8, 32)> : !torch.vtensor<[1,?,8,32],f16>
    %int1_195 = torch.constant.int 1
    %int256_196 = torch.constant.int 256
    %186 = torch.prim.ListConstruct %int1_195, %85, %int256_196 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %187 = torch.aten.view %185, %186 : !torch.vtensor<[1,?,8,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %187, [%31], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int5_197 = torch.constant.int 5
    %188 = torch.prims.convert_element_type %5, %int5_197 : !torch.vtensor<[256,256],f32>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int-2_198 = torch.constant.int -2
    %int-1_199 = torch.constant.int -1
    %189 = torch.aten.transpose.int %188, %int-2_198, %int-1_199 : !torch.vtensor<[256,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,256],f16>
    %int256_200 = torch.constant.int 256
    %190 = torch.prim.ListConstruct %85, %int256_200 : (!torch.int, !torch.int) -> !torch.list<int>
    %191 = torch.aten.view %187, %190 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %191, [%31], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %192 = torch.aten.mm %191, %189 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,256],f16> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %192, [%31], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %int1_201 = torch.constant.int 1
    %int256_202 = torch.constant.int 256
    %193 = torch.prim.ListConstruct %int1_201, %85, %int256_202 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %194 = torch.aten.view %192, %193 : !torch.vtensor<[?,256],f16>, !torch.list<int> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %194, [%31], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int1_203 = torch.constant.int 1
    %195 = torch.aten.add.Tensor %34, %194, %int1_203 : !torch.vtensor<[1,?,256],f16>, !torch.vtensor<[1,?,256],f16>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %195, [%31], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int6_204 = torch.constant.int 6
    %196 = torch.prims.convert_element_type %195, %int6_204 : !torch.vtensor<[1,?,256],f16>, !torch.int -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %196, [%31], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int2_205 = torch.constant.int 2
    %197 = torch.aten.pow.Tensor_Scalar %196, %int2_205 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %197, [%31], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int-1_206 = torch.constant.int -1
    %198 = torch.prim.ListConstruct %int-1_206 : (!torch.int) -> !torch.list<int>
    %true_207 = torch.constant.bool true
    %none_208 = torch.constant.none
    %199 = torch.aten.mean.dim %197, %198, %true_207, %none_208 : !torch.vtensor<[1,?,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %199, [%31], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %float1.000000e-02_209 = torch.constant.float 1.000000e-02
    %int1_210 = torch.constant.int 1
    %200 = torch.aten.add.Scalar %199, %float1.000000e-02_209, %int1_210 : !torch.vtensor<[1,?,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %200, [%31], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %201 = torch.aten.rsqrt %200 : !torch.vtensor<[1,?,1],f32> -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %201, [%31], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %202 = torch.aten.mul.Tensor %196, %201 : !torch.vtensor<[1,?,256],f32>, !torch.vtensor<[1,?,1],f32> -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %202, [%31], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int5_211 = torch.constant.int 5
    %203 = torch.prims.convert_element_type %202, %int5_211 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %203, [%31], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %204 = torch.aten.mul.Tensor %6, %203 : !torch.vtensor<[256],f32>, !torch.vtensor<[1,?,256],f16> -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %204, [%31], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int5_212 = torch.constant.int 5
    %205 = torch.prims.convert_element_type %204, %int5_212 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %205, [%31], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int5_213 = torch.constant.int 5
    %206 = torch.prims.convert_element_type %7, %int5_213 : !torch.vtensor<[23,256],f32>, !torch.int -> !torch.vtensor<[23,256],f16>
    %int-2_214 = torch.constant.int -2
    %int-1_215 = torch.constant.int -1
    %207 = torch.aten.transpose.int %206, %int-2_214, %int-1_215 : !torch.vtensor<[23,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,23],f16>
    %int256_216 = torch.constant.int 256
    %208 = torch.prim.ListConstruct %47, %int256_216 : (!torch.int, !torch.int) -> !torch.list<int>
    %209 = torch.aten.view %205, %208 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %209, [%31], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %210 = torch.aten.mm %209, %207 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,23],f16> -> !torch.vtensor<[?,23],f16>
    torch.bind_symbolic_shape %210, [%31], affine_map<()[s0] -> (s0 * 32, 23)> : !torch.vtensor<[?,23],f16>
    %int1_217 = torch.constant.int 1
    %int23 = torch.constant.int 23
    %211 = torch.prim.ListConstruct %int1_217, %47, %int23 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %212 = torch.aten.view %210, %211 : !torch.vtensor<[?,23],f16>, !torch.list<int> -> !torch.vtensor<[1,?,23],f16>
    torch.bind_symbolic_shape %212, [%31], affine_map<()[s0] -> (1, s0 * 32, 23)> : !torch.vtensor<[1,?,23],f16>
    %213 = torch.aten.silu %212 : !torch.vtensor<[1,?,23],f16> -> !torch.vtensor<[1,?,23],f16>
    torch.bind_symbolic_shape %213, [%31], affine_map<()[s0] -> (1, s0 * 32, 23)> : !torch.vtensor<[1,?,23],f16>
    %int5_218 = torch.constant.int 5
    %214 = torch.prims.convert_element_type %8, %int5_218 : !torch.vtensor<[23,256],f32>, !torch.int -> !torch.vtensor<[23,256],f16>
    %int-2_219 = torch.constant.int -2
    %int-1_220 = torch.constant.int -1
    %215 = torch.aten.transpose.int %214, %int-2_219, %int-1_220 : !torch.vtensor<[23,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,23],f16>
    %int256_221 = torch.constant.int 256
    %216 = torch.prim.ListConstruct %47, %int256_221 : (!torch.int, !torch.int) -> !torch.list<int>
    %217 = torch.aten.view %205, %216 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %217, [%31], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %218 = torch.aten.mm %217, %215 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,23],f16> -> !torch.vtensor<[?,23],f16>
    torch.bind_symbolic_shape %218, [%31], affine_map<()[s0] -> (s0 * 32, 23)> : !torch.vtensor<[?,23],f16>
    %int1_222 = torch.constant.int 1
    %int23_223 = torch.constant.int 23
    %219 = torch.prim.ListConstruct %int1_222, %47, %int23_223 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %220 = torch.aten.view %218, %219 : !torch.vtensor<[?,23],f16>, !torch.list<int> -> !torch.vtensor<[1,?,23],f16>
    torch.bind_symbolic_shape %220, [%31], affine_map<()[s0] -> (1, s0 * 32, 23)> : !torch.vtensor<[1,?,23],f16>
    %221 = torch.aten.mul.Tensor %213, %220 : !torch.vtensor<[1,?,23],f16>, !torch.vtensor<[1,?,23],f16> -> !torch.vtensor<[1,?,23],f16>
    torch.bind_symbolic_shape %221, [%31], affine_map<()[s0] -> (1, s0 * 32, 23)> : !torch.vtensor<[1,?,23],f16>
    %int5_224 = torch.constant.int 5
    %222 = torch.prims.convert_element_type %9, %int5_224 : !torch.vtensor<[256,23],f32>, !torch.int -> !torch.vtensor<[256,23],f16>
    %int-2_225 = torch.constant.int -2
    %int-1_226 = torch.constant.int -1
    %223 = torch.aten.transpose.int %222, %int-2_225, %int-1_226 : !torch.vtensor<[256,23],f16>, !torch.int, !torch.int -> !torch.vtensor<[23,256],f16>
    %int1_227 = torch.constant.int 1
    %224 = torch.aten.size.int %212, %int1_227 : !torch.vtensor<[1,?,23],f16>, !torch.int -> !torch.int
    %int23_228 = torch.constant.int 23
    %225 = torch.prim.ListConstruct %224, %int23_228 : (!torch.int, !torch.int) -> !torch.list<int>
    %226 = torch.aten.view %221, %225 : !torch.vtensor<[1,?,23],f16>, !torch.list<int> -> !torch.vtensor<[?,23],f16>
    torch.bind_symbolic_shape %226, [%31], affine_map<()[s0] -> (s0 * 32, 23)> : !torch.vtensor<[?,23],f16>
    %227 = torch.aten.mm %226, %223 : !torch.vtensor<[?,23],f16>, !torch.vtensor<[23,256],f16> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %227, [%31], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %int1_229 = torch.constant.int 1
    %int256_230 = torch.constant.int 256
    %228 = torch.prim.ListConstruct %int1_229, %224, %int256_230 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %229 = torch.aten.view %227, %228 : !torch.vtensor<[?,256],f16>, !torch.list<int> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %229, [%31], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int1_231 = torch.constant.int 1
    %230 = torch.aten.add.Tensor %195, %229, %int1_231 : !torch.vtensor<[1,?,256],f16>, !torch.vtensor<[1,?,256],f16>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %230, [%31], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int6_232 = torch.constant.int 6
    %231 = torch.prims.convert_element_type %230, %int6_232 : !torch.vtensor<[1,?,256],f16>, !torch.int -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %231, [%31], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int2_233 = torch.constant.int 2
    %232 = torch.aten.pow.Tensor_Scalar %231, %int2_233 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %232, [%31], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int-1_234 = torch.constant.int -1
    %233 = torch.prim.ListConstruct %int-1_234 : (!torch.int) -> !torch.list<int>
    %true_235 = torch.constant.bool true
    %none_236 = torch.constant.none
    %234 = torch.aten.mean.dim %232, %233, %true_235, %none_236 : !torch.vtensor<[1,?,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %234, [%31], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %float1.000000e-02_237 = torch.constant.float 1.000000e-02
    %int1_238 = torch.constant.int 1
    %235 = torch.aten.add.Scalar %234, %float1.000000e-02_237, %int1_238 : !torch.vtensor<[1,?,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %235, [%31], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %236 = torch.aten.rsqrt %235 : !torch.vtensor<[1,?,1],f32> -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %236, [%31], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %237 = torch.aten.mul.Tensor %231, %236 : !torch.vtensor<[1,?,256],f32>, !torch.vtensor<[1,?,1],f32> -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %237, [%31], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int5_239 = torch.constant.int 5
    %238 = torch.prims.convert_element_type %237, %int5_239 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %238, [%31], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %239 = torch.aten.mul.Tensor %10, %238 : !torch.vtensor<[256],f32>, !torch.vtensor<[1,?,256],f16> -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %239, [%31], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int5_240 = torch.constant.int 5
    %240 = torch.prims.convert_element_type %239, %int5_240 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %240, [%31], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int5_241 = torch.constant.int 5
    %241 = torch.prims.convert_element_type %11, %int5_241 : !torch.vtensor<[256,256],f32>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int-2_242 = torch.constant.int -2
    %int-1_243 = torch.constant.int -1
    %242 = torch.aten.transpose.int %241, %int-2_242, %int-1_243 : !torch.vtensor<[256,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,256],f16>
    %int256_244 = torch.constant.int 256
    %243 = torch.prim.ListConstruct %47, %int256_244 : (!torch.int, !torch.int) -> !torch.list<int>
    %244 = torch.aten.view %240, %243 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %244, [%31], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %245 = torch.aten.mm %244, %242 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,256],f16> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %245, [%31], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %int1_245 = torch.constant.int 1
    %int256_246 = torch.constant.int 256
    %246 = torch.prim.ListConstruct %int1_245, %47, %int256_246 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %247 = torch.aten.view %245, %246 : !torch.vtensor<[?,256],f16>, !torch.list<int> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %247, [%31], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int5_247 = torch.constant.int 5
    %248 = torch.prims.convert_element_type %12, %int5_247 : !torch.vtensor<[128,256],f32>, !torch.int -> !torch.vtensor<[128,256],f16>
    %int-2_248 = torch.constant.int -2
    %int-1_249 = torch.constant.int -1
    %249 = torch.aten.transpose.int %248, %int-2_248, %int-1_249 : !torch.vtensor<[128,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,128],f16>
    %int256_250 = torch.constant.int 256
    %250 = torch.prim.ListConstruct %47, %int256_250 : (!torch.int, !torch.int) -> !torch.list<int>
    %251 = torch.aten.view %240, %250 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %251, [%31], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %252 = torch.aten.mm %251, %249 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,128],f16> -> !torch.vtensor<[?,128],f16>
    torch.bind_symbolic_shape %252, [%31], affine_map<()[s0] -> (s0 * 32, 128)> : !torch.vtensor<[?,128],f16>
    %int1_251 = torch.constant.int 1
    %int128_252 = torch.constant.int 128
    %253 = torch.prim.ListConstruct %int1_251, %47, %int128_252 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %254 = torch.aten.view %252, %253 : !torch.vtensor<[?,128],f16>, !torch.list<int> -> !torch.vtensor<[1,?,128],f16>
    torch.bind_symbolic_shape %254, [%31], affine_map<()[s0] -> (1, s0 * 32, 128)> : !torch.vtensor<[1,?,128],f16>
    %int5_253 = torch.constant.int 5
    %255 = torch.prims.convert_element_type %13, %int5_253 : !torch.vtensor<[128,256],f32>, !torch.int -> !torch.vtensor<[128,256],f16>
    %int-2_254 = torch.constant.int -2
    %int-1_255 = torch.constant.int -1
    %256 = torch.aten.transpose.int %255, %int-2_254, %int-1_255 : !torch.vtensor<[128,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,128],f16>
    %int256_256 = torch.constant.int 256
    %257 = torch.prim.ListConstruct %47, %int256_256 : (!torch.int, !torch.int) -> !torch.list<int>
    %258 = torch.aten.view %240, %257 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %258, [%31], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %259 = torch.aten.mm %258, %256 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,128],f16> -> !torch.vtensor<[?,128],f16>
    torch.bind_symbolic_shape %259, [%31], affine_map<()[s0] -> (s0 * 32, 128)> : !torch.vtensor<[?,128],f16>
    %int1_257 = torch.constant.int 1
    %int128_258 = torch.constant.int 128
    %260 = torch.prim.ListConstruct %int1_257, %47, %int128_258 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %261 = torch.aten.view %259, %260 : !torch.vtensor<[?,128],f16>, !torch.list<int> -> !torch.vtensor<[1,?,128],f16>
    torch.bind_symbolic_shape %261, [%31], affine_map<()[s0] -> (1, s0 * 32, 128)> : !torch.vtensor<[1,?,128],f16>
    %int1_259 = torch.constant.int 1
    %int8_260 = torch.constant.int 8
    %int32_261 = torch.constant.int 32
    %262 = torch.prim.ListConstruct %int1_259, %47, %int8_260, %int32_261 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %263 = torch.aten.view %247, %262 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[1,?,8,32],f16>
    torch.bind_symbolic_shape %263, [%31], affine_map<()[s0] -> (1, s0 * 32, 8, 32)> : !torch.vtensor<[1,?,8,32],f16>
    %int1_262 = torch.constant.int 1
    %int4_263 = torch.constant.int 4
    %int32_264 = torch.constant.int 32
    %264 = torch.prim.ListConstruct %int1_262, %47, %int4_263, %int32_264 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %265 = torch.aten.view %254, %264 : !torch.vtensor<[1,?,128],f16>, !torch.list<int> -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %265, [%31], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int1_265 = torch.constant.int 1
    %int4_266 = torch.constant.int 4
    %int32_267 = torch.constant.int 32
    %266 = torch.prim.ListConstruct %int1_265, %47, %int4_266, %int32_267 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %267 = torch.aten.view %261, %266 : !torch.vtensor<[1,?,128],f16>, !torch.list<int> -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %267, [%31], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int128_268 = torch.constant.int 128
    %none_269 = torch.constant.none
    %none_270 = torch.constant.none
    %cpu_271 = torch.constant.device "cpu"
    %false_272 = torch.constant.bool false
    %268 = torch.aten.arange %int128_268, %none_269, %none_270, %cpu_271, %false_272 : !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[128],si64>
    %int0_273 = torch.constant.int 0
    %int32_274 = torch.constant.int 32
    %none_275 = torch.constant.none
    %none_276 = torch.constant.none
    %cpu_277 = torch.constant.device "cpu"
    %false_278 = torch.constant.bool false
    %269 = torch.aten.arange.start %int0_273, %int32_274, %none_275, %none_276, %cpu_277, %false_278 : !torch.int, !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[32],si64>
    %int2_279 = torch.constant.int 2
    %270 = torch.aten.floor_divide.Scalar %269, %int2_279 : !torch.vtensor<[32],si64>, !torch.int -> !torch.vtensor<[32],si64>
    %int6_280 = torch.constant.int 6
    %271 = torch.prims.convert_element_type %270, %int6_280 : !torch.vtensor<[32],si64>, !torch.int -> !torch.vtensor<[32],f32>
    %int32_281 = torch.constant.int 32
    %272 = torch.aten.div.Scalar %271, %int32_281 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[32],f32>
    %float2.000000e00_282 = torch.constant.float 2.000000e+00
    %273 = torch.aten.mul.Scalar %272, %float2.000000e00_282 : !torch.vtensor<[32],f32>, !torch.float -> !torch.vtensor<[32],f32>
    %float5.000000e05_283 = torch.constant.float 5.000000e+05
    %274 = torch.aten.pow.Scalar %float5.000000e05_283, %273 : !torch.float, !torch.vtensor<[32],f32> -> !torch.vtensor<[32],f32>
    %275 = torch.aten.reciprocal %274 : !torch.vtensor<[32],f32> -> !torch.vtensor<[32],f32>
    %float1.000000e00_284 = torch.constant.float 1.000000e+00
    %276 = torch.aten.mul.Scalar %275, %float1.000000e00_284 : !torch.vtensor<[32],f32>, !torch.float -> !torch.vtensor<[32],f32>
    %int1_285 = torch.constant.int 1
    %277 = torch.aten.unsqueeze %268, %int1_285 : !torch.vtensor<[128],si64>, !torch.int -> !torch.vtensor<[128,1],si64>
    %int0_286 = torch.constant.int 0
    %278 = torch.aten.unsqueeze %276, %int0_286 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[1,32],f32>
    %279 = torch.aten.mul.Tensor %277, %278 : !torch.vtensor<[128,1],si64>, !torch.vtensor<[1,32],f32> -> !torch.vtensor<[128,32],f32>
    %int1_287 = torch.constant.int 1
    %280 = torch.aten.size.int %247, %int1_287 : !torch.vtensor<[1,?,256],f16>, !torch.int -> !torch.int
    %int0_288 = torch.constant.int 0
    %281 = torch.aten.add.int %int0_288, %280 : !torch.int, !torch.int -> !torch.int
    %int0_289 = torch.constant.int 0
    %int0_290 = torch.constant.int 0
    %int1_291 = torch.constant.int 1
    %282 = torch.aten.slice.Tensor %279, %int0_289, %int0_290, %281, %int1_291 : !torch.vtensor<[128,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %282, [%31], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int1_292 = torch.constant.int 1
    %int0_293 = torch.constant.int 0
    %int9223372036854775807_294 = torch.constant.int 9223372036854775807
    %int1_295 = torch.constant.int 1
    %283 = torch.aten.slice.Tensor %282, %int1_292, %int0_293, %int9223372036854775807_294, %int1_295 : !torch.vtensor<[?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %283, [%31], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int1_296 = torch.constant.int 1
    %int0_297 = torch.constant.int 0
    %int9223372036854775807_298 = torch.constant.int 9223372036854775807
    %int1_299 = torch.constant.int 1
    %284 = torch.aten.slice.Tensor %283, %int1_296, %int0_297, %int9223372036854775807_298, %int1_299 : !torch.vtensor<[?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %284, [%31], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int0_300 = torch.constant.int 0
    %285 = torch.aten.unsqueeze %284, %int0_300 : !torch.vtensor<[?,32],f32>, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %285, [%31], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int1_301 = torch.constant.int 1
    %int0_302 = torch.constant.int 0
    %int9223372036854775807_303 = torch.constant.int 9223372036854775807
    %int1_304 = torch.constant.int 1
    %286 = torch.aten.slice.Tensor %285, %int1_301, %int0_302, %int9223372036854775807_303, %int1_304 : !torch.vtensor<[1,?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %286, [%31], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int2_305 = torch.constant.int 2
    %int0_306 = torch.constant.int 0
    %int9223372036854775807_307 = torch.constant.int 9223372036854775807
    %int1_308 = torch.constant.int 1
    %287 = torch.aten.slice.Tensor %286, %int2_305, %int0_306, %int9223372036854775807_307, %int1_308 : !torch.vtensor<[1,?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %287, [%31], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int1_309 = torch.constant.int 1
    %int1_310 = torch.constant.int 1
    %int1_311 = torch.constant.int 1
    %288 = torch.prim.ListConstruct %int1_309, %int1_310, %int1_311 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %289 = torch.aten.repeat %287, %288 : !torch.vtensor<[1,?,32],f32>, !torch.list<int> -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %289, [%31], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int6_312 = torch.constant.int 6
    %290 = torch.prims.convert_element_type %263, %int6_312 : !torch.vtensor<[1,?,8,32],f16>, !torch.int -> !torch.vtensor<[1,?,8,32],f32>
    torch.bind_symbolic_shape %290, [%31], affine_map<()[s0] -> (1, s0 * 32, 8, 32)> : !torch.vtensor<[1,?,8,32],f32>
    %291 = torch_c.to_builtin_tensor %290 : !torch.vtensor<[1,?,8,32],f32> -> tensor<1x?x8x32xf32>
    %292 = torch_c.to_builtin_tensor %289 : !torch.vtensor<[1,?,32],f32> -> tensor<1x?x32xf32>
    %293 = util.call @sharktank_rotary_embedding_1_D_8_32_f32(%291, %292) : (tensor<1x?x8x32xf32>, tensor<1x?x32xf32>) -> tensor<1x?x8x32xf32>
    %294 = torch_c.from_builtin_tensor %293 : tensor<1x?x8x32xf32> -> !torch.vtensor<[1,?,8,32],f32>
    torch.bind_symbolic_shape %294, [%31], affine_map<()[s0] -> (1, s0 * 32, 8, 32)> : !torch.vtensor<[1,?,8,32],f32>
    %int5_313 = torch.constant.int 5
    %295 = torch.prims.convert_element_type %294, %int5_313 : !torch.vtensor<[1,?,8,32],f32>, !torch.int -> !torch.vtensor<[1,?,8,32],f16>
    torch.bind_symbolic_shape %295, [%31], affine_map<()[s0] -> (1, s0 * 32, 8, 32)> : !torch.vtensor<[1,?,8,32],f16>
    %int128_314 = torch.constant.int 128
    %none_315 = torch.constant.none
    %none_316 = torch.constant.none
    %cpu_317 = torch.constant.device "cpu"
    %false_318 = torch.constant.bool false
    %296 = torch.aten.arange %int128_314, %none_315, %none_316, %cpu_317, %false_318 : !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[128],si64>
    %int0_319 = torch.constant.int 0
    %int32_320 = torch.constant.int 32
    %none_321 = torch.constant.none
    %none_322 = torch.constant.none
    %cpu_323 = torch.constant.device "cpu"
    %false_324 = torch.constant.bool false
    %297 = torch.aten.arange.start %int0_319, %int32_320, %none_321, %none_322, %cpu_323, %false_324 : !torch.int, !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[32],si64>
    %int2_325 = torch.constant.int 2
    %298 = torch.aten.floor_divide.Scalar %297, %int2_325 : !torch.vtensor<[32],si64>, !torch.int -> !torch.vtensor<[32],si64>
    %int6_326 = torch.constant.int 6
    %299 = torch.prims.convert_element_type %298, %int6_326 : !torch.vtensor<[32],si64>, !torch.int -> !torch.vtensor<[32],f32>
    %int32_327 = torch.constant.int 32
    %300 = torch.aten.div.Scalar %299, %int32_327 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[32],f32>
    %float2.000000e00_328 = torch.constant.float 2.000000e+00
    %301 = torch.aten.mul.Scalar %300, %float2.000000e00_328 : !torch.vtensor<[32],f32>, !torch.float -> !torch.vtensor<[32],f32>
    %float5.000000e05_329 = torch.constant.float 5.000000e+05
    %302 = torch.aten.pow.Scalar %float5.000000e05_329, %301 : !torch.float, !torch.vtensor<[32],f32> -> !torch.vtensor<[32],f32>
    %303 = torch.aten.reciprocal %302 : !torch.vtensor<[32],f32> -> !torch.vtensor<[32],f32>
    %float1.000000e00_330 = torch.constant.float 1.000000e+00
    %304 = torch.aten.mul.Scalar %303, %float1.000000e00_330 : !torch.vtensor<[32],f32>, !torch.float -> !torch.vtensor<[32],f32>
    %int1_331 = torch.constant.int 1
    %305 = torch.aten.unsqueeze %296, %int1_331 : !torch.vtensor<[128],si64>, !torch.int -> !torch.vtensor<[128,1],si64>
    %int0_332 = torch.constant.int 0
    %306 = torch.aten.unsqueeze %304, %int0_332 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[1,32],f32>
    %307 = torch.aten.mul.Tensor %305, %306 : !torch.vtensor<[128,1],si64>, !torch.vtensor<[1,32],f32> -> !torch.vtensor<[128,32],f32>
    %int1_333 = torch.constant.int 1
    %308 = torch.aten.size.int %254, %int1_333 : !torch.vtensor<[1,?,128],f16>, !torch.int -> !torch.int
    %int0_334 = torch.constant.int 0
    %309 = torch.aten.add.int %int0_334, %308 : !torch.int, !torch.int -> !torch.int
    %int0_335 = torch.constant.int 0
    %int0_336 = torch.constant.int 0
    %int1_337 = torch.constant.int 1
    %310 = torch.aten.slice.Tensor %307, %int0_335, %int0_336, %309, %int1_337 : !torch.vtensor<[128,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %310, [%31], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int1_338 = torch.constant.int 1
    %int0_339 = torch.constant.int 0
    %int9223372036854775807_340 = torch.constant.int 9223372036854775807
    %int1_341 = torch.constant.int 1
    %311 = torch.aten.slice.Tensor %310, %int1_338, %int0_339, %int9223372036854775807_340, %int1_341 : !torch.vtensor<[?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %311, [%31], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int1_342 = torch.constant.int 1
    %int0_343 = torch.constant.int 0
    %int9223372036854775807_344 = torch.constant.int 9223372036854775807
    %int1_345 = torch.constant.int 1
    %312 = torch.aten.slice.Tensor %311, %int1_342, %int0_343, %int9223372036854775807_344, %int1_345 : !torch.vtensor<[?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %312, [%31], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int0_346 = torch.constant.int 0
    %313 = torch.aten.unsqueeze %312, %int0_346 : !torch.vtensor<[?,32],f32>, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %313, [%31], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int1_347 = torch.constant.int 1
    %int0_348 = torch.constant.int 0
    %int9223372036854775807_349 = torch.constant.int 9223372036854775807
    %int1_350 = torch.constant.int 1
    %314 = torch.aten.slice.Tensor %313, %int1_347, %int0_348, %int9223372036854775807_349, %int1_350 : !torch.vtensor<[1,?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %314, [%31], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int2_351 = torch.constant.int 2
    %int0_352 = torch.constant.int 0
    %int9223372036854775807_353 = torch.constant.int 9223372036854775807
    %int1_354 = torch.constant.int 1
    %315 = torch.aten.slice.Tensor %314, %int2_351, %int0_352, %int9223372036854775807_353, %int1_354 : !torch.vtensor<[1,?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %315, [%31], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int1_355 = torch.constant.int 1
    %int1_356 = torch.constant.int 1
    %int1_357 = torch.constant.int 1
    %316 = torch.prim.ListConstruct %int1_355, %int1_356, %int1_357 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %317 = torch.aten.repeat %315, %316 : !torch.vtensor<[1,?,32],f32>, !torch.list<int> -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %317, [%31], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int6_358 = torch.constant.int 6
    %318 = torch.prims.convert_element_type %265, %int6_358 : !torch.vtensor<[1,?,4,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,32],f32>
    torch.bind_symbolic_shape %318, [%31], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f32>
    %319 = torch_c.to_builtin_tensor %318 : !torch.vtensor<[1,?,4,32],f32> -> tensor<1x?x4x32xf32>
    %320 = torch_c.to_builtin_tensor %317 : !torch.vtensor<[1,?,32],f32> -> tensor<1x?x32xf32>
    %321 = util.call @sharktank_rotary_embedding_1_D_4_32_f32(%319, %320) : (tensor<1x?x4x32xf32>, tensor<1x?x32xf32>) -> tensor<1x?x4x32xf32>
    %322 = torch_c.from_builtin_tensor %321 : tensor<1x?x4x32xf32> -> !torch.vtensor<[1,?,4,32],f32>
    torch.bind_symbolic_shape %322, [%31], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f32>
    %int5_359 = torch.constant.int 5
    %323 = torch.prims.convert_element_type %322, %int5_359 : !torch.vtensor<[1,?,4,32],f32>, !torch.int -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %323, [%31], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int6_360 = torch.constant.int 6
    %324 = torch.aten.mul.Scalar %arg2, %int6_360 : !torch.vtensor<[1,?],si64>, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %324, [%31], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %int2_361 = torch.constant.int 2
    %int1_362 = torch.constant.int 1
    %325 = torch.aten.add.Scalar %324, %int2_361, %int1_362 : !torch.vtensor<[1,?],si64>, !torch.int, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %325, [%31], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %int1_363 = torch.constant.int 1
    %int32_364 = torch.constant.int 32
    %int4_365 = torch.constant.int 4
    %int32_366 = torch.constant.int 32
    %326 = torch.prim.ListConstruct %int1_363, %138, %int32_364, %int4_365, %int32_366 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %327 = torch.aten.view %323, %326 : !torch.vtensor<[1,?,4,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,32,4,32],f16>
    torch.bind_symbolic_shape %327, [%31], affine_map<()[s0] -> (1, s0, 32, 4, 32)> : !torch.vtensor<[1,?,32,4,32],f16>
    %int32_367 = torch.constant.int 32
    %int4_368 = torch.constant.int 4
    %int32_369 = torch.constant.int 32
    %328 = torch.prim.ListConstruct %138, %int32_367, %int4_368, %int32_369 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %329 = torch.aten.view %327, %328 : !torch.vtensor<[1,?,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,32,4,32],f16>
    torch.bind_symbolic_shape %329, [%31], affine_map<()[s0] -> (s0, 32, 4, 32)> : !torch.vtensor<[?,32,4,32],f16>
    %330 = torch.prim.ListConstruct %138 : (!torch.int) -> !torch.list<int>
    %331 = torch.aten.view %325, %330 : !torch.vtensor<[1,?],si64>, !torch.list<int> -> !torch.vtensor<[?],si64>
    torch.bind_symbolic_shape %331, [%31], affine_map<()[s0] -> (s0)> : !torch.vtensor<[?],si64>
    %int3_370 = torch.constant.int 3
    %int2_371 = torch.constant.int 2
    %int32_372 = torch.constant.int 32
    %int4_373 = torch.constant.int 4
    %int32_374 = torch.constant.int 32
    %332 = torch.prim.ListConstruct %129, %int3_370, %int2_371, %int32_372, %int4_373, %int32_374 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %333 = torch.aten.view %167, %332 : !torch.vtensor<[?,24576],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,4,32],f16>
    torch.bind_symbolic_shape %333, [%32], affine_map<()[s0] -> (s0, 3, 2, 32, 4, 32)> : !torch.vtensor<[?,3,2,32,4,32],f16>
    %int3_375 = torch.constant.int 3
    %334 = torch.aten.mul.int %129, %int3_375 : !torch.int, !torch.int -> !torch.int
    %int2_376 = torch.constant.int 2
    %335 = torch.aten.mul.int %334, %int2_376 : !torch.int, !torch.int -> !torch.int
    %int32_377 = torch.constant.int 32
    %int4_378 = torch.constant.int 4
    %int32_379 = torch.constant.int 32
    %336 = torch.prim.ListConstruct %335, %int32_377, %int4_378, %int32_379 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %337 = torch.aten.view %333, %336 : !torch.vtensor<[?,3,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,32,4,32],f16>
    torch.bind_symbolic_shape %337, [%32], affine_map<()[s0] -> (s0 * 6, 32, 4, 32)> : !torch.vtensor<[?,32,4,32],f16>
    %338 = torch.prim.ListConstruct %331 : (!torch.vtensor<[?],si64>) -> !torch.list<optional<vtensor>>
    %false_380 = torch.constant.bool false
    %339 = torch.aten.index_put %337, %338, %329, %false_380 : !torch.vtensor<[?,32,4,32],f16>, !torch.list<optional<vtensor>>, !torch.vtensor<[?,32,4,32],f16>, !torch.bool -> !torch.vtensor<[?,32,4,32],f16>
    torch.bind_symbolic_shape %339, [%32], affine_map<()[s0] -> (s0 * 6, 32, 4, 32)> : !torch.vtensor<[?,32,4,32],f16>
    %int3_381 = torch.constant.int 3
    %int2_382 = torch.constant.int 2
    %int32_383 = torch.constant.int 32
    %int4_384 = torch.constant.int 4
    %int32_385 = torch.constant.int 32
    %340 = torch.prim.ListConstruct %129, %int3_381, %int2_382, %int32_383, %int4_384, %int32_385 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %341 = torch.aten.view %339, %340 : !torch.vtensor<[?,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,4,32],f16>
    torch.bind_symbolic_shape %341, [%32], affine_map<()[s0] -> (s0, 3, 2, 32, 4, 32)> : !torch.vtensor<[?,3,2,32,4,32],f16>
    %int24576_386 = torch.constant.int 24576
    %342 = torch.prim.ListConstruct %129, %int24576_386 : (!torch.int, !torch.int) -> !torch.list<int>
    %343 = torch.aten.view %341, %342 : !torch.vtensor<[?,3,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,24576],f16>
    torch.bind_symbolic_shape %343, [%32], affine_map<()[s0] -> (s0, 24576)> : !torch.vtensor<[?,24576],f16>
    %int3_387 = torch.constant.int 3
    %int2_388 = torch.constant.int 2
    %int32_389 = torch.constant.int 32
    %int4_390 = torch.constant.int 4
    %int32_391 = torch.constant.int 32
    %344 = torch.prim.ListConstruct %129, %int3_387, %int2_388, %int32_389, %int4_390, %int32_391 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %345 = torch.aten.view %343, %344 : !torch.vtensor<[?,24576],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,4,32],f16>
    torch.bind_symbolic_shape %345, [%32], affine_map<()[s0] -> (s0, 3, 2, 32, 4, 32)> : !torch.vtensor<[?,3,2,32,4,32],f16>
    %int32_392 = torch.constant.int 32
    %int4_393 = torch.constant.int 4
    %int32_394 = torch.constant.int 32
    %346 = torch.prim.ListConstruct %335, %int32_392, %int4_393, %int32_394 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %347 = torch.aten.view %345, %346 : !torch.vtensor<[?,3,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,32,4,32],f16>
    torch.bind_symbolic_shape %347, [%32], affine_map<()[s0] -> (s0 * 6, 32, 4, 32)> : !torch.vtensor<[?,32,4,32],f16>
    %int1_395 = torch.constant.int 1
    %int32_396 = torch.constant.int 32
    %int4_397 = torch.constant.int 4
    %int32_398 = torch.constant.int 32
    %348 = torch.prim.ListConstruct %int1_395, %138, %int32_396, %int4_397, %int32_398 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %349 = torch.aten.view %267, %348 : !torch.vtensor<[1,?,4,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,32,4,32],f16>
    torch.bind_symbolic_shape %349, [%31], affine_map<()[s0] -> (1, s0, 32, 4, 32)> : !torch.vtensor<[1,?,32,4,32],f16>
    %int32_399 = torch.constant.int 32
    %int4_400 = torch.constant.int 4
    %int32_401 = torch.constant.int 32
    %350 = torch.prim.ListConstruct %138, %int32_399, %int4_400, %int32_401 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %351 = torch.aten.view %349, %350 : !torch.vtensor<[1,?,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,32,4,32],f16>
    torch.bind_symbolic_shape %351, [%31], affine_map<()[s0] -> (s0, 32, 4, 32)> : !torch.vtensor<[?,32,4,32],f16>
    %int1_402 = torch.constant.int 1
    %int1_403 = torch.constant.int 1
    %352 = torch.aten.add.Scalar %325, %int1_402, %int1_403 : !torch.vtensor<[1,?],si64>, !torch.int, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %352, [%31], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %353 = torch.prim.ListConstruct %138 : (!torch.int) -> !torch.list<int>
    %354 = torch.aten.view %352, %353 : !torch.vtensor<[1,?],si64>, !torch.list<int> -> !torch.vtensor<[?],si64>
    torch.bind_symbolic_shape %354, [%31], affine_map<()[s0] -> (s0)> : !torch.vtensor<[?],si64>
    %355 = torch.prim.ListConstruct %354 : (!torch.vtensor<[?],si64>) -> !torch.list<optional<vtensor>>
    %false_404 = torch.constant.bool false
    %356 = torch.aten.index_put %347, %355, %351, %false_404 : !torch.vtensor<[?,32,4,32],f16>, !torch.list<optional<vtensor>>, !torch.vtensor<[?,32,4,32],f16>, !torch.bool -> !torch.vtensor<[?,32,4,32],f16>
    torch.bind_symbolic_shape %356, [%32], affine_map<()[s0] -> (s0 * 6, 32, 4, 32)> : !torch.vtensor<[?,32,4,32],f16>
    %int3_405 = torch.constant.int 3
    %int2_406 = torch.constant.int 2
    %int32_407 = torch.constant.int 32
    %int4_408 = torch.constant.int 4
    %int32_409 = torch.constant.int 32
    %357 = torch.prim.ListConstruct %129, %int3_405, %int2_406, %int32_407, %int4_408, %int32_409 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %358 = torch.aten.view %356, %357 : !torch.vtensor<[?,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,4,32],f16>
    torch.bind_symbolic_shape %358, [%32], affine_map<()[s0] -> (s0, 3, 2, 32, 4, 32)> : !torch.vtensor<[?,3,2,32,4,32],f16>
    %int24576_410 = torch.constant.int 24576
    %359 = torch.prim.ListConstruct %129, %int24576_410 : (!torch.int, !torch.int) -> !torch.list<int>
    %360 = torch.aten.view %358, %359 : !torch.vtensor<[?,3,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,24576],f16>
    torch.bind_symbolic_shape %360, [%32], affine_map<()[s0] -> (s0, 24576)> : !torch.vtensor<[?,24576],f16>
    %int-2_411 = torch.constant.int -2
    %361 = torch.aten.unsqueeze %323, %int-2_411 : !torch.vtensor<[1,?,4,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,1,32],f16>
    torch.bind_symbolic_shape %361, [%31], affine_map<()[s0] -> (1, s0 * 32, 4, 1, 32)> : !torch.vtensor<[1,?,4,1,32],f16>
    %int1_412 = torch.constant.int 1
    %int4_413 = torch.constant.int 4
    %int2_414 = torch.constant.int 2
    %int32_415 = torch.constant.int 32
    %362 = torch.prim.ListConstruct %int1_412, %308, %int4_413, %int2_414, %int32_415 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_416 = torch.constant.bool false
    %363 = torch.aten.expand %361, %362, %false_416 : !torch.vtensor<[1,?,4,1,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,?,4,2,32],f16>
    torch.bind_symbolic_shape %363, [%31], affine_map<()[s0] -> (1, s0 * 32, 4, 2, 32)> : !torch.vtensor<[1,?,4,2,32],f16>
    %int0_417 = torch.constant.int 0
    %364 = torch.aten.clone %363, %int0_417 : !torch.vtensor<[1,?,4,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,2,32],f16>
    torch.bind_symbolic_shape %364, [%31], affine_map<()[s0] -> (1, s0 * 32, 4, 2, 32)> : !torch.vtensor<[1,?,4,2,32],f16>
    %int1_418 = torch.constant.int 1
    %int8_419 = torch.constant.int 8
    %int32_420 = torch.constant.int 32
    %365 = torch.prim.ListConstruct %int1_418, %308, %int8_419, %int32_420 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %366 = torch.aten._unsafe_view %364, %365 : !torch.vtensor<[1,?,4,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,8,32],f16>
    torch.bind_symbolic_shape %366, [%31], affine_map<()[s0] -> (1, s0 * 32, 8, 32)> : !torch.vtensor<[1,?,8,32],f16>
    %int-2_421 = torch.constant.int -2
    %367 = torch.aten.unsqueeze %267, %int-2_421 : !torch.vtensor<[1,?,4,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,1,32],f16>
    torch.bind_symbolic_shape %367, [%31], affine_map<()[s0] -> (1, s0 * 32, 4, 1, 32)> : !torch.vtensor<[1,?,4,1,32],f16>
    %int1_422 = torch.constant.int 1
    %368 = torch.aten.size.int %261, %int1_422 : !torch.vtensor<[1,?,128],f16>, !torch.int -> !torch.int
    %int1_423 = torch.constant.int 1
    %int4_424 = torch.constant.int 4
    %int2_425 = torch.constant.int 2
    %int32_426 = torch.constant.int 32
    %369 = torch.prim.ListConstruct %int1_423, %368, %int4_424, %int2_425, %int32_426 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_427 = torch.constant.bool false
    %370 = torch.aten.expand %367, %369, %false_427 : !torch.vtensor<[1,?,4,1,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,?,4,2,32],f16>
    torch.bind_symbolic_shape %370, [%31], affine_map<()[s0] -> (1, s0 * 32, 4, 2, 32)> : !torch.vtensor<[1,?,4,2,32],f16>
    %int0_428 = torch.constant.int 0
    %371 = torch.aten.clone %370, %int0_428 : !torch.vtensor<[1,?,4,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,2,32],f16>
    torch.bind_symbolic_shape %371, [%31], affine_map<()[s0] -> (1, s0 * 32, 4, 2, 32)> : !torch.vtensor<[1,?,4,2,32],f16>
    %int1_429 = torch.constant.int 1
    %int8_430 = torch.constant.int 8
    %int32_431 = torch.constant.int 32
    %372 = torch.prim.ListConstruct %int1_429, %368, %int8_430, %int32_431 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %373 = torch.aten._unsafe_view %371, %372 : !torch.vtensor<[1,?,4,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,8,32],f16>
    torch.bind_symbolic_shape %373, [%31], affine_map<()[s0] -> (1, s0 * 32, 8, 32)> : !torch.vtensor<[1,?,8,32],f16>
    %int1_432 = torch.constant.int 1
    %int2_433 = torch.constant.int 2
    %374 = torch.aten.transpose.int %295, %int1_432, %int2_433 : !torch.vtensor<[1,?,8,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,8,?,32],f16>
    torch.bind_symbolic_shape %374, [%31], affine_map<()[s0] -> (1, 8, s0 * 32, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %int1_434 = torch.constant.int 1
    %int2_435 = torch.constant.int 2
    %375 = torch.aten.transpose.int %366, %int1_434, %int2_435 : !torch.vtensor<[1,?,8,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,8,?,32],f16>
    torch.bind_symbolic_shape %375, [%31], affine_map<()[s0] -> (1, 8, s0 * 32, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %int1_436 = torch.constant.int 1
    %int2_437 = torch.constant.int 2
    %376 = torch.aten.transpose.int %373, %int1_436, %int2_437 : !torch.vtensor<[1,?,8,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,8,?,32],f16>
    torch.bind_symbolic_shape %376, [%31], affine_map<()[s0] -> (1, 8, s0 * 32, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %float0.000000e00_438 = torch.constant.float 0.000000e+00
    %true_439 = torch.constant.bool true
    %none_440 = torch.constant.none
    %none_441 = torch.constant.none
    %377:2 = torch.operator "torch.aten._scaled_dot_product_flash_attention_for_cpu"(%374, %375, %376, %float0.000000e00_438, %true_439, %none_440, %none_441) : (!torch.vtensor<[1,8,?,32],f16>, !torch.vtensor<[1,8,?,32],f16>, !torch.vtensor<[1,8,?,32],f16>, !torch.float, !torch.bool, !torch.none, !torch.none) -> (!torch.vtensor<[1,8,?,32],f16>, !torch.vtensor<[1,8,?],f32>) 
    torch.bind_symbolic_shape %377#0, [%31], affine_map<()[s0] -> (1, 8, s0 * 32, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %int1_442 = torch.constant.int 1
    %int2_443 = torch.constant.int 2
    %378 = torch.aten.transpose.int %377#0, %int1_442, %int2_443 : !torch.vtensor<[1,8,?,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,?,8,32],f16>
    torch.bind_symbolic_shape %378, [%31], affine_map<()[s0] -> (1, s0 * 32, 8, 32)> : !torch.vtensor<[1,?,8,32],f16>
    %int1_444 = torch.constant.int 1
    %int256_445 = torch.constant.int 256
    %379 = torch.prim.ListConstruct %int1_444, %280, %int256_445 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %380 = torch.aten.view %378, %379 : !torch.vtensor<[1,?,8,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %380, [%31], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int5_446 = torch.constant.int 5
    %381 = torch.prims.convert_element_type %14, %int5_446 : !torch.vtensor<[256,256],f32>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int-2_447 = torch.constant.int -2
    %int-1_448 = torch.constant.int -1
    %382 = torch.aten.transpose.int %381, %int-2_447, %int-1_448 : !torch.vtensor<[256,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,256],f16>
    %int256_449 = torch.constant.int 256
    %383 = torch.prim.ListConstruct %280, %int256_449 : (!torch.int, !torch.int) -> !torch.list<int>
    %384 = torch.aten.view %380, %383 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %384, [%31], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %385 = torch.aten.mm %384, %382 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,256],f16> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %385, [%31], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %int1_450 = torch.constant.int 1
    %int256_451 = torch.constant.int 256
    %386 = torch.prim.ListConstruct %int1_450, %280, %int256_451 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %387 = torch.aten.view %385, %386 : !torch.vtensor<[?,256],f16>, !torch.list<int> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %387, [%31], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int1_452 = torch.constant.int 1
    %388 = torch.aten.add.Tensor %230, %387, %int1_452 : !torch.vtensor<[1,?,256],f16>, !torch.vtensor<[1,?,256],f16>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %388, [%31], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int6_453 = torch.constant.int 6
    %389 = torch.prims.convert_element_type %388, %int6_453 : !torch.vtensor<[1,?,256],f16>, !torch.int -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %389, [%31], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int2_454 = torch.constant.int 2
    %390 = torch.aten.pow.Tensor_Scalar %389, %int2_454 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %390, [%31], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int-1_455 = torch.constant.int -1
    %391 = torch.prim.ListConstruct %int-1_455 : (!torch.int) -> !torch.list<int>
    %true_456 = torch.constant.bool true
    %none_457 = torch.constant.none
    %392 = torch.aten.mean.dim %390, %391, %true_456, %none_457 : !torch.vtensor<[1,?,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %392, [%31], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %float1.000000e-02_458 = torch.constant.float 1.000000e-02
    %int1_459 = torch.constant.int 1
    %393 = torch.aten.add.Scalar %392, %float1.000000e-02_458, %int1_459 : !torch.vtensor<[1,?,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %393, [%31], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %394 = torch.aten.rsqrt %393 : !torch.vtensor<[1,?,1],f32> -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %394, [%31], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %395 = torch.aten.mul.Tensor %389, %394 : !torch.vtensor<[1,?,256],f32>, !torch.vtensor<[1,?,1],f32> -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %395, [%31], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int5_460 = torch.constant.int 5
    %396 = torch.prims.convert_element_type %395, %int5_460 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %396, [%31], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %397 = torch.aten.mul.Tensor %15, %396 : !torch.vtensor<[256],f32>, !torch.vtensor<[1,?,256],f16> -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %397, [%31], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int5_461 = torch.constant.int 5
    %398 = torch.prims.convert_element_type %397, %int5_461 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %398, [%31], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int5_462 = torch.constant.int 5
    %399 = torch.prims.convert_element_type %16, %int5_462 : !torch.vtensor<[23,256],f32>, !torch.int -> !torch.vtensor<[23,256],f16>
    %int-2_463 = torch.constant.int -2
    %int-1_464 = torch.constant.int -1
    %400 = torch.aten.transpose.int %399, %int-2_463, %int-1_464 : !torch.vtensor<[23,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,23],f16>
    %int256_465 = torch.constant.int 256
    %401 = torch.prim.ListConstruct %47, %int256_465 : (!torch.int, !torch.int) -> !torch.list<int>
    %402 = torch.aten.view %398, %401 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %402, [%31], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %403 = torch.aten.mm %402, %400 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,23],f16> -> !torch.vtensor<[?,23],f16>
    torch.bind_symbolic_shape %403, [%31], affine_map<()[s0] -> (s0 * 32, 23)> : !torch.vtensor<[?,23],f16>
    %int1_466 = torch.constant.int 1
    %int23_467 = torch.constant.int 23
    %404 = torch.prim.ListConstruct %int1_466, %47, %int23_467 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %405 = torch.aten.view %403, %404 : !torch.vtensor<[?,23],f16>, !torch.list<int> -> !torch.vtensor<[1,?,23],f16>
    torch.bind_symbolic_shape %405, [%31], affine_map<()[s0] -> (1, s0 * 32, 23)> : !torch.vtensor<[1,?,23],f16>
    %406 = torch.aten.silu %405 : !torch.vtensor<[1,?,23],f16> -> !torch.vtensor<[1,?,23],f16>
    torch.bind_symbolic_shape %406, [%31], affine_map<()[s0] -> (1, s0 * 32, 23)> : !torch.vtensor<[1,?,23],f16>
    %int5_468 = torch.constant.int 5
    %407 = torch.prims.convert_element_type %17, %int5_468 : !torch.vtensor<[23,256],f32>, !torch.int -> !torch.vtensor<[23,256],f16>
    %int-2_469 = torch.constant.int -2
    %int-1_470 = torch.constant.int -1
    %408 = torch.aten.transpose.int %407, %int-2_469, %int-1_470 : !torch.vtensor<[23,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,23],f16>
    %int256_471 = torch.constant.int 256
    %409 = torch.prim.ListConstruct %47, %int256_471 : (!torch.int, !torch.int) -> !torch.list<int>
    %410 = torch.aten.view %398, %409 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %410, [%31], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %411 = torch.aten.mm %410, %408 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,23],f16> -> !torch.vtensor<[?,23],f16>
    torch.bind_symbolic_shape %411, [%31], affine_map<()[s0] -> (s0 * 32, 23)> : !torch.vtensor<[?,23],f16>
    %int1_472 = torch.constant.int 1
    %int23_473 = torch.constant.int 23
    %412 = torch.prim.ListConstruct %int1_472, %47, %int23_473 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %413 = torch.aten.view %411, %412 : !torch.vtensor<[?,23],f16>, !torch.list<int> -> !torch.vtensor<[1,?,23],f16>
    torch.bind_symbolic_shape %413, [%31], affine_map<()[s0] -> (1, s0 * 32, 23)> : !torch.vtensor<[1,?,23],f16>
    %414 = torch.aten.mul.Tensor %406, %413 : !torch.vtensor<[1,?,23],f16>, !torch.vtensor<[1,?,23],f16> -> !torch.vtensor<[1,?,23],f16>
    torch.bind_symbolic_shape %414, [%31], affine_map<()[s0] -> (1, s0 * 32, 23)> : !torch.vtensor<[1,?,23],f16>
    %int5_474 = torch.constant.int 5
    %415 = torch.prims.convert_element_type %18, %int5_474 : !torch.vtensor<[256,23],f32>, !torch.int -> !torch.vtensor<[256,23],f16>
    %int-2_475 = torch.constant.int -2
    %int-1_476 = torch.constant.int -1
    %416 = torch.aten.transpose.int %415, %int-2_475, %int-1_476 : !torch.vtensor<[256,23],f16>, !torch.int, !torch.int -> !torch.vtensor<[23,256],f16>
    %int1_477 = torch.constant.int 1
    %417 = torch.aten.size.int %405, %int1_477 : !torch.vtensor<[1,?,23],f16>, !torch.int -> !torch.int
    %int23_478 = torch.constant.int 23
    %418 = torch.prim.ListConstruct %417, %int23_478 : (!torch.int, !torch.int) -> !torch.list<int>
    %419 = torch.aten.view %414, %418 : !torch.vtensor<[1,?,23],f16>, !torch.list<int> -> !torch.vtensor<[?,23],f16>
    torch.bind_symbolic_shape %419, [%31], affine_map<()[s0] -> (s0 * 32, 23)> : !torch.vtensor<[?,23],f16>
    %420 = torch.aten.mm %419, %416 : !torch.vtensor<[?,23],f16>, !torch.vtensor<[23,256],f16> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %420, [%31], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %int1_479 = torch.constant.int 1
    %int256_480 = torch.constant.int 256
    %421 = torch.prim.ListConstruct %int1_479, %417, %int256_480 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %422 = torch.aten.view %420, %421 : !torch.vtensor<[?,256],f16>, !torch.list<int> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %422, [%31], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int1_481 = torch.constant.int 1
    %423 = torch.aten.add.Tensor %388, %422, %int1_481 : !torch.vtensor<[1,?,256],f16>, !torch.vtensor<[1,?,256],f16>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %423, [%31], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int6_482 = torch.constant.int 6
    %424 = torch.prims.convert_element_type %423, %int6_482 : !torch.vtensor<[1,?,256],f16>, !torch.int -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %424, [%31], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int2_483 = torch.constant.int 2
    %425 = torch.aten.pow.Tensor_Scalar %424, %int2_483 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %425, [%31], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int-1_484 = torch.constant.int -1
    %426 = torch.prim.ListConstruct %int-1_484 : (!torch.int) -> !torch.list<int>
    %true_485 = torch.constant.bool true
    %none_486 = torch.constant.none
    %427 = torch.aten.mean.dim %425, %426, %true_485, %none_486 : !torch.vtensor<[1,?,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %427, [%31], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %float1.000000e-02_487 = torch.constant.float 1.000000e-02
    %int1_488 = torch.constant.int 1
    %428 = torch.aten.add.Scalar %427, %float1.000000e-02_487, %int1_488 : !torch.vtensor<[1,?,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %428, [%31], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %429 = torch.aten.rsqrt %428 : !torch.vtensor<[1,?,1],f32> -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %429, [%31], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %430 = torch.aten.mul.Tensor %424, %429 : !torch.vtensor<[1,?,256],f32>, !torch.vtensor<[1,?,1],f32> -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %430, [%31], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int5_489 = torch.constant.int 5
    %431 = torch.prims.convert_element_type %430, %int5_489 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %431, [%31], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %432 = torch.aten.mul.Tensor %19, %431 : !torch.vtensor<[256],f32>, !torch.vtensor<[1,?,256],f16> -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %432, [%31], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int5_490 = torch.constant.int 5
    %433 = torch.prims.convert_element_type %432, %int5_490 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %433, [%31], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int5_491 = torch.constant.int 5
    %434 = torch.prims.convert_element_type %20, %int5_491 : !torch.vtensor<[256,256],f32>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int-2_492 = torch.constant.int -2
    %int-1_493 = torch.constant.int -1
    %435 = torch.aten.transpose.int %434, %int-2_492, %int-1_493 : !torch.vtensor<[256,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,256],f16>
    %int256_494 = torch.constant.int 256
    %436 = torch.prim.ListConstruct %47, %int256_494 : (!torch.int, !torch.int) -> !torch.list<int>
    %437 = torch.aten.view %433, %436 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %437, [%31], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %438 = torch.aten.mm %437, %435 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,256],f16> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %438, [%31], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %int1_495 = torch.constant.int 1
    %int256_496 = torch.constant.int 256
    %439 = torch.prim.ListConstruct %int1_495, %47, %int256_496 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %440 = torch.aten.view %438, %439 : !torch.vtensor<[?,256],f16>, !torch.list<int> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %440, [%31], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int5_497 = torch.constant.int 5
    %441 = torch.prims.convert_element_type %21, %int5_497 : !torch.vtensor<[128,256],f32>, !torch.int -> !torch.vtensor<[128,256],f16>
    %int-2_498 = torch.constant.int -2
    %int-1_499 = torch.constant.int -1
    %442 = torch.aten.transpose.int %441, %int-2_498, %int-1_499 : !torch.vtensor<[128,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,128],f16>
    %int256_500 = torch.constant.int 256
    %443 = torch.prim.ListConstruct %47, %int256_500 : (!torch.int, !torch.int) -> !torch.list<int>
    %444 = torch.aten.view %433, %443 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %444, [%31], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %445 = torch.aten.mm %444, %442 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,128],f16> -> !torch.vtensor<[?,128],f16>
    torch.bind_symbolic_shape %445, [%31], affine_map<()[s0] -> (s0 * 32, 128)> : !torch.vtensor<[?,128],f16>
    %int1_501 = torch.constant.int 1
    %int128_502 = torch.constant.int 128
    %446 = torch.prim.ListConstruct %int1_501, %47, %int128_502 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %447 = torch.aten.view %445, %446 : !torch.vtensor<[?,128],f16>, !torch.list<int> -> !torch.vtensor<[1,?,128],f16>
    torch.bind_symbolic_shape %447, [%31], affine_map<()[s0] -> (1, s0 * 32, 128)> : !torch.vtensor<[1,?,128],f16>
    %int5_503 = torch.constant.int 5
    %448 = torch.prims.convert_element_type %22, %int5_503 : !torch.vtensor<[128,256],f32>, !torch.int -> !torch.vtensor<[128,256],f16>
    %int-2_504 = torch.constant.int -2
    %int-1_505 = torch.constant.int -1
    %449 = torch.aten.transpose.int %448, %int-2_504, %int-1_505 : !torch.vtensor<[128,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,128],f16>
    %int256_506 = torch.constant.int 256
    %450 = torch.prim.ListConstruct %47, %int256_506 : (!torch.int, !torch.int) -> !torch.list<int>
    %451 = torch.aten.view %433, %450 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %451, [%31], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %452 = torch.aten.mm %451, %449 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,128],f16> -> !torch.vtensor<[?,128],f16>
    torch.bind_symbolic_shape %452, [%31], affine_map<()[s0] -> (s0 * 32, 128)> : !torch.vtensor<[?,128],f16>
    %int1_507 = torch.constant.int 1
    %int128_508 = torch.constant.int 128
    %453 = torch.prim.ListConstruct %int1_507, %47, %int128_508 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %454 = torch.aten.view %452, %453 : !torch.vtensor<[?,128],f16>, !torch.list<int> -> !torch.vtensor<[1,?,128],f16>
    torch.bind_symbolic_shape %454, [%31], affine_map<()[s0] -> (1, s0 * 32, 128)> : !torch.vtensor<[1,?,128],f16>
    %int1_509 = torch.constant.int 1
    %int8_510 = torch.constant.int 8
    %int32_511 = torch.constant.int 32
    %455 = torch.prim.ListConstruct %int1_509, %47, %int8_510, %int32_511 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %456 = torch.aten.view %440, %455 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[1,?,8,32],f16>
    torch.bind_symbolic_shape %456, [%31], affine_map<()[s0] -> (1, s0 * 32, 8, 32)> : !torch.vtensor<[1,?,8,32],f16>
    %int1_512 = torch.constant.int 1
    %int4_513 = torch.constant.int 4
    %int32_514 = torch.constant.int 32
    %457 = torch.prim.ListConstruct %int1_512, %47, %int4_513, %int32_514 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %458 = torch.aten.view %447, %457 : !torch.vtensor<[1,?,128],f16>, !torch.list<int> -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %458, [%31], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int1_515 = torch.constant.int 1
    %int4_516 = torch.constant.int 4
    %int32_517 = torch.constant.int 32
    %459 = torch.prim.ListConstruct %int1_515, %47, %int4_516, %int32_517 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %460 = torch.aten.view %454, %459 : !torch.vtensor<[1,?,128],f16>, !torch.list<int> -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %460, [%31], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int128_518 = torch.constant.int 128
    %none_519 = torch.constant.none
    %none_520 = torch.constant.none
    %cpu_521 = torch.constant.device "cpu"
    %false_522 = torch.constant.bool false
    %461 = torch.aten.arange %int128_518, %none_519, %none_520, %cpu_521, %false_522 : !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[128],si64>
    %int0_523 = torch.constant.int 0
    %int32_524 = torch.constant.int 32
    %none_525 = torch.constant.none
    %none_526 = torch.constant.none
    %cpu_527 = torch.constant.device "cpu"
    %false_528 = torch.constant.bool false
    %462 = torch.aten.arange.start %int0_523, %int32_524, %none_525, %none_526, %cpu_527, %false_528 : !torch.int, !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[32],si64>
    %int2_529 = torch.constant.int 2
    %463 = torch.aten.floor_divide.Scalar %462, %int2_529 : !torch.vtensor<[32],si64>, !torch.int -> !torch.vtensor<[32],si64>
    %int6_530 = torch.constant.int 6
    %464 = torch.prims.convert_element_type %463, %int6_530 : !torch.vtensor<[32],si64>, !torch.int -> !torch.vtensor<[32],f32>
    %int32_531 = torch.constant.int 32
    %465 = torch.aten.div.Scalar %464, %int32_531 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[32],f32>
    %float2.000000e00_532 = torch.constant.float 2.000000e+00
    %466 = torch.aten.mul.Scalar %465, %float2.000000e00_532 : !torch.vtensor<[32],f32>, !torch.float -> !torch.vtensor<[32],f32>
    %float5.000000e05_533 = torch.constant.float 5.000000e+05
    %467 = torch.aten.pow.Scalar %float5.000000e05_533, %466 : !torch.float, !torch.vtensor<[32],f32> -> !torch.vtensor<[32],f32>
    %468 = torch.aten.reciprocal %467 : !torch.vtensor<[32],f32> -> !torch.vtensor<[32],f32>
    %float1.000000e00_534 = torch.constant.float 1.000000e+00
    %469 = torch.aten.mul.Scalar %468, %float1.000000e00_534 : !torch.vtensor<[32],f32>, !torch.float -> !torch.vtensor<[32],f32>
    %int1_535 = torch.constant.int 1
    %470 = torch.aten.unsqueeze %461, %int1_535 : !torch.vtensor<[128],si64>, !torch.int -> !torch.vtensor<[128,1],si64>
    %int0_536 = torch.constant.int 0
    %471 = torch.aten.unsqueeze %469, %int0_536 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[1,32],f32>
    %472 = torch.aten.mul.Tensor %470, %471 : !torch.vtensor<[128,1],si64>, !torch.vtensor<[1,32],f32> -> !torch.vtensor<[128,32],f32>
    %int1_537 = torch.constant.int 1
    %473 = torch.aten.size.int %440, %int1_537 : !torch.vtensor<[1,?,256],f16>, !torch.int -> !torch.int
    %int0_538 = torch.constant.int 0
    %474 = torch.aten.add.int %int0_538, %473 : !torch.int, !torch.int -> !torch.int
    %int0_539 = torch.constant.int 0
    %int0_540 = torch.constant.int 0
    %int1_541 = torch.constant.int 1
    %475 = torch.aten.slice.Tensor %472, %int0_539, %int0_540, %474, %int1_541 : !torch.vtensor<[128,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %475, [%31], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int1_542 = torch.constant.int 1
    %int0_543 = torch.constant.int 0
    %int9223372036854775807_544 = torch.constant.int 9223372036854775807
    %int1_545 = torch.constant.int 1
    %476 = torch.aten.slice.Tensor %475, %int1_542, %int0_543, %int9223372036854775807_544, %int1_545 : !torch.vtensor<[?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %476, [%31], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int1_546 = torch.constant.int 1
    %int0_547 = torch.constant.int 0
    %int9223372036854775807_548 = torch.constant.int 9223372036854775807
    %int1_549 = torch.constant.int 1
    %477 = torch.aten.slice.Tensor %476, %int1_546, %int0_547, %int9223372036854775807_548, %int1_549 : !torch.vtensor<[?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %477, [%31], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int0_550 = torch.constant.int 0
    %478 = torch.aten.unsqueeze %477, %int0_550 : !torch.vtensor<[?,32],f32>, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %478, [%31], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int1_551 = torch.constant.int 1
    %int0_552 = torch.constant.int 0
    %int9223372036854775807_553 = torch.constant.int 9223372036854775807
    %int1_554 = torch.constant.int 1
    %479 = torch.aten.slice.Tensor %478, %int1_551, %int0_552, %int9223372036854775807_553, %int1_554 : !torch.vtensor<[1,?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %479, [%31], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int2_555 = torch.constant.int 2
    %int0_556 = torch.constant.int 0
    %int9223372036854775807_557 = torch.constant.int 9223372036854775807
    %int1_558 = torch.constant.int 1
    %480 = torch.aten.slice.Tensor %479, %int2_555, %int0_556, %int9223372036854775807_557, %int1_558 : !torch.vtensor<[1,?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %480, [%31], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int1_559 = torch.constant.int 1
    %int1_560 = torch.constant.int 1
    %int1_561 = torch.constant.int 1
    %481 = torch.prim.ListConstruct %int1_559, %int1_560, %int1_561 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %482 = torch.aten.repeat %480, %481 : !torch.vtensor<[1,?,32],f32>, !torch.list<int> -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %482, [%31], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int6_562 = torch.constant.int 6
    %483 = torch.prims.convert_element_type %456, %int6_562 : !torch.vtensor<[1,?,8,32],f16>, !torch.int -> !torch.vtensor<[1,?,8,32],f32>
    torch.bind_symbolic_shape %483, [%31], affine_map<()[s0] -> (1, s0 * 32, 8, 32)> : !torch.vtensor<[1,?,8,32],f32>
    %484 = torch_c.to_builtin_tensor %483 : !torch.vtensor<[1,?,8,32],f32> -> tensor<1x?x8x32xf32>
    %485 = torch_c.to_builtin_tensor %482 : !torch.vtensor<[1,?,32],f32> -> tensor<1x?x32xf32>
    %486 = util.call @sharktank_rotary_embedding_1_D_8_32_f32(%484, %485) : (tensor<1x?x8x32xf32>, tensor<1x?x32xf32>) -> tensor<1x?x8x32xf32>
    %487 = torch_c.from_builtin_tensor %486 : tensor<1x?x8x32xf32> -> !torch.vtensor<[1,?,8,32],f32>
    torch.bind_symbolic_shape %487, [%31], affine_map<()[s0] -> (1, s0 * 32, 8, 32)> : !torch.vtensor<[1,?,8,32],f32>
    %int5_563 = torch.constant.int 5
    %488 = torch.prims.convert_element_type %487, %int5_563 : !torch.vtensor<[1,?,8,32],f32>, !torch.int -> !torch.vtensor<[1,?,8,32],f16>
    torch.bind_symbolic_shape %488, [%31], affine_map<()[s0] -> (1, s0 * 32, 8, 32)> : !torch.vtensor<[1,?,8,32],f16>
    %int128_564 = torch.constant.int 128
    %none_565 = torch.constant.none
    %none_566 = torch.constant.none
    %cpu_567 = torch.constant.device "cpu"
    %false_568 = torch.constant.bool false
    %489 = torch.aten.arange %int128_564, %none_565, %none_566, %cpu_567, %false_568 : !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[128],si64>
    %int0_569 = torch.constant.int 0
    %int32_570 = torch.constant.int 32
    %none_571 = torch.constant.none
    %none_572 = torch.constant.none
    %cpu_573 = torch.constant.device "cpu"
    %false_574 = torch.constant.bool false
    %490 = torch.aten.arange.start %int0_569, %int32_570, %none_571, %none_572, %cpu_573, %false_574 : !torch.int, !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[32],si64>
    %int2_575 = torch.constant.int 2
    %491 = torch.aten.floor_divide.Scalar %490, %int2_575 : !torch.vtensor<[32],si64>, !torch.int -> !torch.vtensor<[32],si64>
    %int6_576 = torch.constant.int 6
    %492 = torch.prims.convert_element_type %491, %int6_576 : !torch.vtensor<[32],si64>, !torch.int -> !torch.vtensor<[32],f32>
    %int32_577 = torch.constant.int 32
    %493 = torch.aten.div.Scalar %492, %int32_577 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[32],f32>
    %float2.000000e00_578 = torch.constant.float 2.000000e+00
    %494 = torch.aten.mul.Scalar %493, %float2.000000e00_578 : !torch.vtensor<[32],f32>, !torch.float -> !torch.vtensor<[32],f32>
    %float5.000000e05_579 = torch.constant.float 5.000000e+05
    %495 = torch.aten.pow.Scalar %float5.000000e05_579, %494 : !torch.float, !torch.vtensor<[32],f32> -> !torch.vtensor<[32],f32>
    %496 = torch.aten.reciprocal %495 : !torch.vtensor<[32],f32> -> !torch.vtensor<[32],f32>
    %float1.000000e00_580 = torch.constant.float 1.000000e+00
    %497 = torch.aten.mul.Scalar %496, %float1.000000e00_580 : !torch.vtensor<[32],f32>, !torch.float -> !torch.vtensor<[32],f32>
    %int1_581 = torch.constant.int 1
    %498 = torch.aten.unsqueeze %489, %int1_581 : !torch.vtensor<[128],si64>, !torch.int -> !torch.vtensor<[128,1],si64>
    %int0_582 = torch.constant.int 0
    %499 = torch.aten.unsqueeze %497, %int0_582 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[1,32],f32>
    %500 = torch.aten.mul.Tensor %498, %499 : !torch.vtensor<[128,1],si64>, !torch.vtensor<[1,32],f32> -> !torch.vtensor<[128,32],f32>
    %int1_583 = torch.constant.int 1
    %501 = torch.aten.size.int %447, %int1_583 : !torch.vtensor<[1,?,128],f16>, !torch.int -> !torch.int
    %int0_584 = torch.constant.int 0
    %502 = torch.aten.add.int %int0_584, %501 : !torch.int, !torch.int -> !torch.int
    %int0_585 = torch.constant.int 0
    %int0_586 = torch.constant.int 0
    %int1_587 = torch.constant.int 1
    %503 = torch.aten.slice.Tensor %500, %int0_585, %int0_586, %502, %int1_587 : !torch.vtensor<[128,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %503, [%31], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int1_588 = torch.constant.int 1
    %int0_589 = torch.constant.int 0
    %int9223372036854775807_590 = torch.constant.int 9223372036854775807
    %int1_591 = torch.constant.int 1
    %504 = torch.aten.slice.Tensor %503, %int1_588, %int0_589, %int9223372036854775807_590, %int1_591 : !torch.vtensor<[?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %504, [%31], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int1_592 = torch.constant.int 1
    %int0_593 = torch.constant.int 0
    %int9223372036854775807_594 = torch.constant.int 9223372036854775807
    %int1_595 = torch.constant.int 1
    %505 = torch.aten.slice.Tensor %504, %int1_592, %int0_593, %int9223372036854775807_594, %int1_595 : !torch.vtensor<[?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %505, [%31], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int0_596 = torch.constant.int 0
    %506 = torch.aten.unsqueeze %505, %int0_596 : !torch.vtensor<[?,32],f32>, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %506, [%31], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int1_597 = torch.constant.int 1
    %int0_598 = torch.constant.int 0
    %int9223372036854775807_599 = torch.constant.int 9223372036854775807
    %int1_600 = torch.constant.int 1
    %507 = torch.aten.slice.Tensor %506, %int1_597, %int0_598, %int9223372036854775807_599, %int1_600 : !torch.vtensor<[1,?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %507, [%31], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int2_601 = torch.constant.int 2
    %int0_602 = torch.constant.int 0
    %int9223372036854775807_603 = torch.constant.int 9223372036854775807
    %int1_604 = torch.constant.int 1
    %508 = torch.aten.slice.Tensor %507, %int2_601, %int0_602, %int9223372036854775807_603, %int1_604 : !torch.vtensor<[1,?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %508, [%31], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int1_605 = torch.constant.int 1
    %int1_606 = torch.constant.int 1
    %int1_607 = torch.constant.int 1
    %509 = torch.prim.ListConstruct %int1_605, %int1_606, %int1_607 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %510 = torch.aten.repeat %508, %509 : !torch.vtensor<[1,?,32],f32>, !torch.list<int> -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %510, [%31], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int6_608 = torch.constant.int 6
    %511 = torch.prims.convert_element_type %458, %int6_608 : !torch.vtensor<[1,?,4,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,32],f32>
    torch.bind_symbolic_shape %511, [%31], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f32>
    %512 = torch_c.to_builtin_tensor %511 : !torch.vtensor<[1,?,4,32],f32> -> tensor<1x?x4x32xf32>
    %513 = torch_c.to_builtin_tensor %510 : !torch.vtensor<[1,?,32],f32> -> tensor<1x?x32xf32>
    %514 = util.call @sharktank_rotary_embedding_1_D_4_32_f32(%512, %513) : (tensor<1x?x4x32xf32>, tensor<1x?x32xf32>) -> tensor<1x?x4x32xf32>
    %515 = torch_c.from_builtin_tensor %514 : tensor<1x?x4x32xf32> -> !torch.vtensor<[1,?,4,32],f32>
    torch.bind_symbolic_shape %515, [%31], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f32>
    %int5_609 = torch.constant.int 5
    %516 = torch.prims.convert_element_type %515, %int5_609 : !torch.vtensor<[1,?,4,32],f32>, !torch.int -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %516, [%31], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int6_610 = torch.constant.int 6
    %517 = torch.aten.mul.Scalar %arg2, %int6_610 : !torch.vtensor<[1,?],si64>, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %517, [%31], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %int4_611 = torch.constant.int 4
    %int1_612 = torch.constant.int 1
    %518 = torch.aten.add.Scalar %517, %int4_611, %int1_612 : !torch.vtensor<[1,?],si64>, !torch.int, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %518, [%31], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %int1_613 = torch.constant.int 1
    %int32_614 = torch.constant.int 32
    %int4_615 = torch.constant.int 4
    %int32_616 = torch.constant.int 32
    %519 = torch.prim.ListConstruct %int1_613, %138, %int32_614, %int4_615, %int32_616 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %520 = torch.aten.view %516, %519 : !torch.vtensor<[1,?,4,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,32,4,32],f16>
    torch.bind_symbolic_shape %520, [%31], affine_map<()[s0] -> (1, s0, 32, 4, 32)> : !torch.vtensor<[1,?,32,4,32],f16>
    %int32_617 = torch.constant.int 32
    %int4_618 = torch.constant.int 4
    %int32_619 = torch.constant.int 32
    %521 = torch.prim.ListConstruct %138, %int32_617, %int4_618, %int32_619 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %522 = torch.aten.view %520, %521 : !torch.vtensor<[1,?,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,32,4,32],f16>
    torch.bind_symbolic_shape %522, [%31], affine_map<()[s0] -> (s0, 32, 4, 32)> : !torch.vtensor<[?,32,4,32],f16>
    %523 = torch.prim.ListConstruct %138 : (!torch.int) -> !torch.list<int>
    %524 = torch.aten.view %518, %523 : !torch.vtensor<[1,?],si64>, !torch.list<int> -> !torch.vtensor<[?],si64>
    torch.bind_symbolic_shape %524, [%31], affine_map<()[s0] -> (s0)> : !torch.vtensor<[?],si64>
    %int3_620 = torch.constant.int 3
    %int2_621 = torch.constant.int 2
    %int32_622 = torch.constant.int 32
    %int4_623 = torch.constant.int 4
    %int32_624 = torch.constant.int 32
    %525 = torch.prim.ListConstruct %129, %int3_620, %int2_621, %int32_622, %int4_623, %int32_624 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %526 = torch.aten.view %360, %525 : !torch.vtensor<[?,24576],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,4,32],f16>
    torch.bind_symbolic_shape %526, [%32], affine_map<()[s0] -> (s0, 3, 2, 32, 4, 32)> : !torch.vtensor<[?,3,2,32,4,32],f16>
    %int3_625 = torch.constant.int 3
    %527 = torch.aten.mul.int %129, %int3_625 : !torch.int, !torch.int -> !torch.int
    %int2_626 = torch.constant.int 2
    %528 = torch.aten.mul.int %527, %int2_626 : !torch.int, !torch.int -> !torch.int
    %int32_627 = torch.constant.int 32
    %int4_628 = torch.constant.int 4
    %int32_629 = torch.constant.int 32
    %529 = torch.prim.ListConstruct %528, %int32_627, %int4_628, %int32_629 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %530 = torch.aten.view %526, %529 : !torch.vtensor<[?,3,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,32,4,32],f16>
    torch.bind_symbolic_shape %530, [%32], affine_map<()[s0] -> (s0 * 6, 32, 4, 32)> : !torch.vtensor<[?,32,4,32],f16>
    %531 = torch.prim.ListConstruct %524 : (!torch.vtensor<[?],si64>) -> !torch.list<optional<vtensor>>
    %false_630 = torch.constant.bool false
    %532 = torch.aten.index_put %530, %531, %522, %false_630 : !torch.vtensor<[?,32,4,32],f16>, !torch.list<optional<vtensor>>, !torch.vtensor<[?,32,4,32],f16>, !torch.bool -> !torch.vtensor<[?,32,4,32],f16>
    torch.bind_symbolic_shape %532, [%32], affine_map<()[s0] -> (s0 * 6, 32, 4, 32)> : !torch.vtensor<[?,32,4,32],f16>
    %int3_631 = torch.constant.int 3
    %int2_632 = torch.constant.int 2
    %int32_633 = torch.constant.int 32
    %int4_634 = torch.constant.int 4
    %int32_635 = torch.constant.int 32
    %533 = torch.prim.ListConstruct %129, %int3_631, %int2_632, %int32_633, %int4_634, %int32_635 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %534 = torch.aten.view %532, %533 : !torch.vtensor<[?,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,4,32],f16>
    torch.bind_symbolic_shape %534, [%32], affine_map<()[s0] -> (s0, 3, 2, 32, 4, 32)> : !torch.vtensor<[?,3,2,32,4,32],f16>
    %int24576_636 = torch.constant.int 24576
    %535 = torch.prim.ListConstruct %129, %int24576_636 : (!torch.int, !torch.int) -> !torch.list<int>
    %536 = torch.aten.view %534, %535 : !torch.vtensor<[?,3,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,24576],f16>
    torch.bind_symbolic_shape %536, [%32], affine_map<()[s0] -> (s0, 24576)> : !torch.vtensor<[?,24576],f16>
    %int3_637 = torch.constant.int 3
    %int2_638 = torch.constant.int 2
    %int32_639 = torch.constant.int 32
    %int4_640 = torch.constant.int 4
    %int32_641 = torch.constant.int 32
    %537 = torch.prim.ListConstruct %129, %int3_637, %int2_638, %int32_639, %int4_640, %int32_641 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %538 = torch.aten.view %536, %537 : !torch.vtensor<[?,24576],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,4,32],f16>
    torch.bind_symbolic_shape %538, [%32], affine_map<()[s0] -> (s0, 3, 2, 32, 4, 32)> : !torch.vtensor<[?,3,2,32,4,32],f16>
    %int32_642 = torch.constant.int 32
    %int4_643 = torch.constant.int 4
    %int32_644 = torch.constant.int 32
    %539 = torch.prim.ListConstruct %528, %int32_642, %int4_643, %int32_644 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %540 = torch.aten.view %538, %539 : !torch.vtensor<[?,3,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,32,4,32],f16>
    torch.bind_symbolic_shape %540, [%32], affine_map<()[s0] -> (s0 * 6, 32, 4, 32)> : !torch.vtensor<[?,32,4,32],f16>
    %int1_645 = torch.constant.int 1
    %int32_646 = torch.constant.int 32
    %int4_647 = torch.constant.int 4
    %int32_648 = torch.constant.int 32
    %541 = torch.prim.ListConstruct %int1_645, %138, %int32_646, %int4_647, %int32_648 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %542 = torch.aten.view %460, %541 : !torch.vtensor<[1,?,4,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,32,4,32],f16>
    torch.bind_symbolic_shape %542, [%31], affine_map<()[s0] -> (1, s0, 32, 4, 32)> : !torch.vtensor<[1,?,32,4,32],f16>
    %int32_649 = torch.constant.int 32
    %int4_650 = torch.constant.int 4
    %int32_651 = torch.constant.int 32
    %543 = torch.prim.ListConstruct %138, %int32_649, %int4_650, %int32_651 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %544 = torch.aten.view %542, %543 : !torch.vtensor<[1,?,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,32,4,32],f16>
    torch.bind_symbolic_shape %544, [%31], affine_map<()[s0] -> (s0, 32, 4, 32)> : !torch.vtensor<[?,32,4,32],f16>
    %int1_652 = torch.constant.int 1
    %int1_653 = torch.constant.int 1
    %545 = torch.aten.add.Scalar %518, %int1_652, %int1_653 : !torch.vtensor<[1,?],si64>, !torch.int, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %545, [%31], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %546 = torch.prim.ListConstruct %138 : (!torch.int) -> !torch.list<int>
    %547 = torch.aten.view %545, %546 : !torch.vtensor<[1,?],si64>, !torch.list<int> -> !torch.vtensor<[?],si64>
    torch.bind_symbolic_shape %547, [%31], affine_map<()[s0] -> (s0)> : !torch.vtensor<[?],si64>
    %548 = torch.prim.ListConstruct %547 : (!torch.vtensor<[?],si64>) -> !torch.list<optional<vtensor>>
    %false_654 = torch.constant.bool false
    %549 = torch.aten.index_put %540, %548, %544, %false_654 : !torch.vtensor<[?,32,4,32],f16>, !torch.list<optional<vtensor>>, !torch.vtensor<[?,32,4,32],f16>, !torch.bool -> !torch.vtensor<[?,32,4,32],f16>
    torch.bind_symbolic_shape %549, [%32], affine_map<()[s0] -> (s0 * 6, 32, 4, 32)> : !torch.vtensor<[?,32,4,32],f16>
    %int3_655 = torch.constant.int 3
    %int2_656 = torch.constant.int 2
    %int32_657 = torch.constant.int 32
    %int4_658 = torch.constant.int 4
    %int32_659 = torch.constant.int 32
    %550 = torch.prim.ListConstruct %129, %int3_655, %int2_656, %int32_657, %int4_658, %int32_659 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %551 = torch.aten.view %549, %550 : !torch.vtensor<[?,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,4,32],f16>
    torch.bind_symbolic_shape %551, [%32], affine_map<()[s0] -> (s0, 3, 2, 32, 4, 32)> : !torch.vtensor<[?,3,2,32,4,32],f16>
    %int24576_660 = torch.constant.int 24576
    %552 = torch.prim.ListConstruct %129, %int24576_660 : (!torch.int, !torch.int) -> !torch.list<int>
    %553 = torch.aten.view %551, %552 : !torch.vtensor<[?,3,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,24576],f16>
    torch.overwrite.tensor.contents %553 overwrites %arg3 : !torch.vtensor<[?,24576],f16>, !torch.tensor<[?,24576],f16>
    torch.bind_symbolic_shape %553, [%32], affine_map<()[s0] -> (s0, 24576)> : !torch.vtensor<[?,24576],f16>
    %int-2_661 = torch.constant.int -2
    %554 = torch.aten.unsqueeze %516, %int-2_661 : !torch.vtensor<[1,?,4,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,1,32],f16>
    torch.bind_symbolic_shape %554, [%31], affine_map<()[s0] -> (1, s0 * 32, 4, 1, 32)> : !torch.vtensor<[1,?,4,1,32],f16>
    %int1_662 = torch.constant.int 1
    %int4_663 = torch.constant.int 4
    %int2_664 = torch.constant.int 2
    %int32_665 = torch.constant.int 32
    %555 = torch.prim.ListConstruct %int1_662, %501, %int4_663, %int2_664, %int32_665 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_666 = torch.constant.bool false
    %556 = torch.aten.expand %554, %555, %false_666 : !torch.vtensor<[1,?,4,1,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,?,4,2,32],f16>
    torch.bind_symbolic_shape %556, [%31], affine_map<()[s0] -> (1, s0 * 32, 4, 2, 32)> : !torch.vtensor<[1,?,4,2,32],f16>
    %int0_667 = torch.constant.int 0
    %557 = torch.aten.clone %556, %int0_667 : !torch.vtensor<[1,?,4,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,2,32],f16>
    torch.bind_symbolic_shape %557, [%31], affine_map<()[s0] -> (1, s0 * 32, 4, 2, 32)> : !torch.vtensor<[1,?,4,2,32],f16>
    %int1_668 = torch.constant.int 1
    %int8_669 = torch.constant.int 8
    %int32_670 = torch.constant.int 32
    %558 = torch.prim.ListConstruct %int1_668, %501, %int8_669, %int32_670 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %559 = torch.aten._unsafe_view %557, %558 : !torch.vtensor<[1,?,4,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,8,32],f16>
    torch.bind_symbolic_shape %559, [%31], affine_map<()[s0] -> (1, s0 * 32, 8, 32)> : !torch.vtensor<[1,?,8,32],f16>
    %int-2_671 = torch.constant.int -2
    %560 = torch.aten.unsqueeze %460, %int-2_671 : !torch.vtensor<[1,?,4,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,1,32],f16>
    torch.bind_symbolic_shape %560, [%31], affine_map<()[s0] -> (1, s0 * 32, 4, 1, 32)> : !torch.vtensor<[1,?,4,1,32],f16>
    %int1_672 = torch.constant.int 1
    %561 = torch.aten.size.int %454, %int1_672 : !torch.vtensor<[1,?,128],f16>, !torch.int -> !torch.int
    %int1_673 = torch.constant.int 1
    %int4_674 = torch.constant.int 4
    %int2_675 = torch.constant.int 2
    %int32_676 = torch.constant.int 32
    %562 = torch.prim.ListConstruct %int1_673, %561, %int4_674, %int2_675, %int32_676 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_677 = torch.constant.bool false
    %563 = torch.aten.expand %560, %562, %false_677 : !torch.vtensor<[1,?,4,1,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,?,4,2,32],f16>
    torch.bind_symbolic_shape %563, [%31], affine_map<()[s0] -> (1, s0 * 32, 4, 2, 32)> : !torch.vtensor<[1,?,4,2,32],f16>
    %int0_678 = torch.constant.int 0
    %564 = torch.aten.clone %563, %int0_678 : !torch.vtensor<[1,?,4,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,2,32],f16>
    torch.bind_symbolic_shape %564, [%31], affine_map<()[s0] -> (1, s0 * 32, 4, 2, 32)> : !torch.vtensor<[1,?,4,2,32],f16>
    %int1_679 = torch.constant.int 1
    %int8_680 = torch.constant.int 8
    %int32_681 = torch.constant.int 32
    %565 = torch.prim.ListConstruct %int1_679, %561, %int8_680, %int32_681 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %566 = torch.aten._unsafe_view %564, %565 : !torch.vtensor<[1,?,4,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,8,32],f16>
    torch.bind_symbolic_shape %566, [%31], affine_map<()[s0] -> (1, s0 * 32, 8, 32)> : !torch.vtensor<[1,?,8,32],f16>
    %int1_682 = torch.constant.int 1
    %int2_683 = torch.constant.int 2
    %567 = torch.aten.transpose.int %488, %int1_682, %int2_683 : !torch.vtensor<[1,?,8,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,8,?,32],f16>
    torch.bind_symbolic_shape %567, [%31], affine_map<()[s0] -> (1, 8, s0 * 32, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %int1_684 = torch.constant.int 1
    %int2_685 = torch.constant.int 2
    %568 = torch.aten.transpose.int %559, %int1_684, %int2_685 : !torch.vtensor<[1,?,8,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,8,?,32],f16>
    torch.bind_symbolic_shape %568, [%31], affine_map<()[s0] -> (1, 8, s0 * 32, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %int1_686 = torch.constant.int 1
    %int2_687 = torch.constant.int 2
    %569 = torch.aten.transpose.int %566, %int1_686, %int2_687 : !torch.vtensor<[1,?,8,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,8,?,32],f16>
    torch.bind_symbolic_shape %569, [%31], affine_map<()[s0] -> (1, 8, s0 * 32, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %float0.000000e00_688 = torch.constant.float 0.000000e+00
    %true_689 = torch.constant.bool true
    %none_690 = torch.constant.none
    %none_691 = torch.constant.none
    %570:2 = torch.operator "torch.aten._scaled_dot_product_flash_attention_for_cpu"(%567, %568, %569, %float0.000000e00_688, %true_689, %none_690, %none_691) : (!torch.vtensor<[1,8,?,32],f16>, !torch.vtensor<[1,8,?,32],f16>, !torch.vtensor<[1,8,?,32],f16>, !torch.float, !torch.bool, !torch.none, !torch.none) -> (!torch.vtensor<[1,8,?,32],f16>, !torch.vtensor<[1,8,?],f32>) 
    torch.bind_symbolic_shape %570#0, [%31], affine_map<()[s0] -> (1, 8, s0 * 32, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %int1_692 = torch.constant.int 1
    %int2_693 = torch.constant.int 2
    %571 = torch.aten.transpose.int %570#0, %int1_692, %int2_693 : !torch.vtensor<[1,8,?,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,?,8,32],f16>
    torch.bind_symbolic_shape %571, [%31], affine_map<()[s0] -> (1, s0 * 32, 8, 32)> : !torch.vtensor<[1,?,8,32],f16>
    %int1_694 = torch.constant.int 1
    %int256_695 = torch.constant.int 256
    %572 = torch.prim.ListConstruct %int1_694, %473, %int256_695 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %573 = torch.aten.view %571, %572 : !torch.vtensor<[1,?,8,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %573, [%31], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int5_696 = torch.constant.int 5
    %574 = torch.prims.convert_element_type %23, %int5_696 : !torch.vtensor<[256,256],f32>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int-2_697 = torch.constant.int -2
    %int-1_698 = torch.constant.int -1
    %575 = torch.aten.transpose.int %574, %int-2_697, %int-1_698 : !torch.vtensor<[256,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,256],f16>
    %int256_699 = torch.constant.int 256
    %576 = torch.prim.ListConstruct %473, %int256_699 : (!torch.int, !torch.int) -> !torch.list<int>
    %577 = torch.aten.view %573, %576 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %577, [%31], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %578 = torch.aten.mm %577, %575 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,256],f16> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %578, [%31], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %int1_700 = torch.constant.int 1
    %int256_701 = torch.constant.int 256
    %579 = torch.prim.ListConstruct %int1_700, %473, %int256_701 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %580 = torch.aten.view %578, %579 : !torch.vtensor<[?,256],f16>, !torch.list<int> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %580, [%31], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int1_702 = torch.constant.int 1
    %581 = torch.aten.add.Tensor %423, %580, %int1_702 : !torch.vtensor<[1,?,256],f16>, !torch.vtensor<[1,?,256],f16>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %581, [%31], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int6_703 = torch.constant.int 6
    %582 = torch.prims.convert_element_type %581, %int6_703 : !torch.vtensor<[1,?,256],f16>, !torch.int -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %582, [%31], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int2_704 = torch.constant.int 2
    %583 = torch.aten.pow.Tensor_Scalar %582, %int2_704 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %583, [%31], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int-1_705 = torch.constant.int -1
    %584 = torch.prim.ListConstruct %int-1_705 : (!torch.int) -> !torch.list<int>
    %true_706 = torch.constant.bool true
    %none_707 = torch.constant.none
    %585 = torch.aten.mean.dim %583, %584, %true_706, %none_707 : !torch.vtensor<[1,?,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %585, [%31], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %float1.000000e-02_708 = torch.constant.float 1.000000e-02
    %int1_709 = torch.constant.int 1
    %586 = torch.aten.add.Scalar %585, %float1.000000e-02_708, %int1_709 : !torch.vtensor<[1,?,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %586, [%31], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %587 = torch.aten.rsqrt %586 : !torch.vtensor<[1,?,1],f32> -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %587, [%31], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %588 = torch.aten.mul.Tensor %582, %587 : !torch.vtensor<[1,?,256],f32>, !torch.vtensor<[1,?,1],f32> -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %588, [%31], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int5_710 = torch.constant.int 5
    %589 = torch.prims.convert_element_type %588, %int5_710 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %589, [%31], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %590 = torch.aten.mul.Tensor %24, %589 : !torch.vtensor<[256],f32>, !torch.vtensor<[1,?,256],f16> -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %590, [%31], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int5_711 = torch.constant.int 5
    %591 = torch.prims.convert_element_type %590, %int5_711 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %591, [%31], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int5_712 = torch.constant.int 5
    %592 = torch.prims.convert_element_type %25, %int5_712 : !torch.vtensor<[23,256],f32>, !torch.int -> !torch.vtensor<[23,256],f16>
    %int-2_713 = torch.constant.int -2
    %int-1_714 = torch.constant.int -1
    %593 = torch.aten.transpose.int %592, %int-2_713, %int-1_714 : !torch.vtensor<[23,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,23],f16>
    %int256_715 = torch.constant.int 256
    %594 = torch.prim.ListConstruct %47, %int256_715 : (!torch.int, !torch.int) -> !torch.list<int>
    %595 = torch.aten.view %591, %594 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %595, [%31], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %596 = torch.aten.mm %595, %593 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,23],f16> -> !torch.vtensor<[?,23],f16>
    torch.bind_symbolic_shape %596, [%31], affine_map<()[s0] -> (s0 * 32, 23)> : !torch.vtensor<[?,23],f16>
    %int1_716 = torch.constant.int 1
    %int23_717 = torch.constant.int 23
    %597 = torch.prim.ListConstruct %int1_716, %47, %int23_717 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %598 = torch.aten.view %596, %597 : !torch.vtensor<[?,23],f16>, !torch.list<int> -> !torch.vtensor<[1,?,23],f16>
    torch.bind_symbolic_shape %598, [%31], affine_map<()[s0] -> (1, s0 * 32, 23)> : !torch.vtensor<[1,?,23],f16>
    %599 = torch.aten.silu %598 : !torch.vtensor<[1,?,23],f16> -> !torch.vtensor<[1,?,23],f16>
    torch.bind_symbolic_shape %599, [%31], affine_map<()[s0] -> (1, s0 * 32, 23)> : !torch.vtensor<[1,?,23],f16>
    %int5_718 = torch.constant.int 5
    %600 = torch.prims.convert_element_type %26, %int5_718 : !torch.vtensor<[23,256],f32>, !torch.int -> !torch.vtensor<[23,256],f16>
    %int-2_719 = torch.constant.int -2
    %int-1_720 = torch.constant.int -1
    %601 = torch.aten.transpose.int %600, %int-2_719, %int-1_720 : !torch.vtensor<[23,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,23],f16>
    %int256_721 = torch.constant.int 256
    %602 = torch.prim.ListConstruct %47, %int256_721 : (!torch.int, !torch.int) -> !torch.list<int>
    %603 = torch.aten.view %591, %602 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %603, [%31], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %604 = torch.aten.mm %603, %601 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,23],f16> -> !torch.vtensor<[?,23],f16>
    torch.bind_symbolic_shape %604, [%31], affine_map<()[s0] -> (s0 * 32, 23)> : !torch.vtensor<[?,23],f16>
    %int1_722 = torch.constant.int 1
    %int23_723 = torch.constant.int 23
    %605 = torch.prim.ListConstruct %int1_722, %47, %int23_723 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %606 = torch.aten.view %604, %605 : !torch.vtensor<[?,23],f16>, !torch.list<int> -> !torch.vtensor<[1,?,23],f16>
    torch.bind_symbolic_shape %606, [%31], affine_map<()[s0] -> (1, s0 * 32, 23)> : !torch.vtensor<[1,?,23],f16>
    %607 = torch.aten.mul.Tensor %599, %606 : !torch.vtensor<[1,?,23],f16>, !torch.vtensor<[1,?,23],f16> -> !torch.vtensor<[1,?,23],f16>
    torch.bind_symbolic_shape %607, [%31], affine_map<()[s0] -> (1, s0 * 32, 23)> : !torch.vtensor<[1,?,23],f16>
    %int5_724 = torch.constant.int 5
    %608 = torch.prims.convert_element_type %27, %int5_724 : !torch.vtensor<[256,23],f32>, !torch.int -> !torch.vtensor<[256,23],f16>
    %int-2_725 = torch.constant.int -2
    %int-1_726 = torch.constant.int -1
    %609 = torch.aten.transpose.int %608, %int-2_725, %int-1_726 : !torch.vtensor<[256,23],f16>, !torch.int, !torch.int -> !torch.vtensor<[23,256],f16>
    %int1_727 = torch.constant.int 1
    %610 = torch.aten.size.int %598, %int1_727 : !torch.vtensor<[1,?,23],f16>, !torch.int -> !torch.int
    %int23_728 = torch.constant.int 23
    %611 = torch.prim.ListConstruct %610, %int23_728 : (!torch.int, !torch.int) -> !torch.list<int>
    %612 = torch.aten.view %607, %611 : !torch.vtensor<[1,?,23],f16>, !torch.list<int> -> !torch.vtensor<[?,23],f16>
    torch.bind_symbolic_shape %612, [%31], affine_map<()[s0] -> (s0 * 32, 23)> : !torch.vtensor<[?,23],f16>
    %613 = torch.aten.mm %612, %609 : !torch.vtensor<[?,23],f16>, !torch.vtensor<[23,256],f16> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %613, [%31], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %int1_729 = torch.constant.int 1
    %int256_730 = torch.constant.int 256
    %614 = torch.prim.ListConstruct %int1_729, %610, %int256_730 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %615 = torch.aten.view %613, %614 : !torch.vtensor<[?,256],f16>, !torch.list<int> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %615, [%31], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int1_731 = torch.constant.int 1
    %616 = torch.aten.add.Tensor %581, %615, %int1_731 : !torch.vtensor<[1,?,256],f16>, !torch.vtensor<[1,?,256],f16>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %616, [%31], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int6_732 = torch.constant.int 6
    %617 = torch.prims.convert_element_type %616, %int6_732 : !torch.vtensor<[1,?,256],f16>, !torch.int -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %617, [%31], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int2_733 = torch.constant.int 2
    %618 = torch.aten.pow.Tensor_Scalar %617, %int2_733 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %618, [%31], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int-1_734 = torch.constant.int -1
    %619 = torch.prim.ListConstruct %int-1_734 : (!torch.int) -> !torch.list<int>
    %true_735 = torch.constant.bool true
    %none_736 = torch.constant.none
    %620 = torch.aten.mean.dim %618, %619, %true_735, %none_736 : !torch.vtensor<[1,?,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %620, [%31], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %float1.000000e-02_737 = torch.constant.float 1.000000e-02
    %int1_738 = torch.constant.int 1
    %621 = torch.aten.add.Scalar %620, %float1.000000e-02_737, %int1_738 : !torch.vtensor<[1,?,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %621, [%31], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %622 = torch.aten.rsqrt %621 : !torch.vtensor<[1,?,1],f32> -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %622, [%31], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %623 = torch.aten.mul.Tensor %617, %622 : !torch.vtensor<[1,?,256],f32>, !torch.vtensor<[1,?,1],f32> -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %623, [%31], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int5_739 = torch.constant.int 5
    %624 = torch.prims.convert_element_type %623, %int5_739 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %624, [%31], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %625 = torch.aten.mul.Tensor %28, %624 : !torch.vtensor<[1,256],f32>, !torch.vtensor<[1,?,256],f16> -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %625, [%31], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int5_740 = torch.constant.int 5
    %626 = torch.prims.convert_element_type %625, %int5_740 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %626, [%31], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int5_741 = torch.constant.int 5
    %627 = torch.prims.convert_element_type %29, %int5_741 : !torch.vtensor<[256,256],f32>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int-2_742 = torch.constant.int -2
    %int-1_743 = torch.constant.int -1
    %628 = torch.aten.transpose.int %627, %int-2_742, %int-1_743 : !torch.vtensor<[256,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,256],f16>
    %int256_744 = torch.constant.int 256
    %629 = torch.prim.ListConstruct %47, %int256_744 : (!torch.int, !torch.int) -> !torch.list<int>
    %630 = torch.aten.view %626, %629 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %630, [%31], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %631 = torch.aten.mm %630, %628 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,256],f16> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %631, [%31], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %int1_745 = torch.constant.int 1
    %int256_746 = torch.constant.int 256
    %632 = torch.prim.ListConstruct %int1_745, %47, %int256_746 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %633 = torch.aten.view %631, %632 : !torch.vtensor<[?,256],f16>, !torch.list<int> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %633, [%31], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    return %633 : !torch.vtensor<[1,?,256],f16>
  }
  func.func @decode_bs1(%arg0: !torch.vtensor<[1,1],si64>, %arg1: !torch.vtensor<[1],si64>, %arg2: !torch.vtensor<[1],si64>, %arg3: !torch.vtensor<[1,?],si64>, %arg4: !torch.tensor<[?,24576],f16>) -> !torch.vtensor<[1,1,256],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %__auto.token_embd.weight = util.global.load @__auto.token_embd.weight : tensor<256x256xf32>
    %0 = torch_c.from_builtin_tensor %__auto.token_embd.weight : tensor<256x256xf32> -> !torch.vtensor<[256,256],f32>
    %__auto.blk.0.attn_norm.weight = util.global.load @__auto.blk.0.attn_norm.weight : tensor<256xf32>
    %1 = torch_c.from_builtin_tensor %__auto.blk.0.attn_norm.weight : tensor<256xf32> -> !torch.vtensor<[256],f32>
    %__auto.blk.0.attn_q.weight = util.global.load @__auto.blk.0.attn_q.weight : tensor<256x256xf32>
    %2 = torch_c.from_builtin_tensor %__auto.blk.0.attn_q.weight : tensor<256x256xf32> -> !torch.vtensor<[256,256],f32>
    %__auto.blk.0.attn_k.weight = util.global.load @__auto.blk.0.attn_k.weight : tensor<128x256xf32>
    %3 = torch_c.from_builtin_tensor %__auto.blk.0.attn_k.weight : tensor<128x256xf32> -> !torch.vtensor<[128,256],f32>
    %__auto.blk.0.attn_v.weight = util.global.load @__auto.blk.0.attn_v.weight : tensor<128x256xf32>
    %4 = torch_c.from_builtin_tensor %__auto.blk.0.attn_v.weight : tensor<128x256xf32> -> !torch.vtensor<[128,256],f32>
    %5 = torch.vtensor.literal(dense<0> : tensor<si64>) : !torch.vtensor<[],si64>
    %6 = torch.vtensor.literal(dense<1> : tensor<si64>) : !torch.vtensor<[],si64>
    %__auto.blk.0.attn_output.weight = util.global.load @__auto.blk.0.attn_output.weight : tensor<256x256xf32>
    %7 = torch_c.from_builtin_tensor %__auto.blk.0.attn_output.weight : tensor<256x256xf32> -> !torch.vtensor<[256,256],f32>
    %__auto.blk.0.ffn_norm.weight = util.global.load @__auto.blk.0.ffn_norm.weight : tensor<256xf32>
    %8 = torch_c.from_builtin_tensor %__auto.blk.0.ffn_norm.weight : tensor<256xf32> -> !torch.vtensor<[256],f32>
    %__auto.blk.0.ffn_gate.weight = util.global.load @__auto.blk.0.ffn_gate.weight : tensor<23x256xf32>
    %9 = torch_c.from_builtin_tensor %__auto.blk.0.ffn_gate.weight : tensor<23x256xf32> -> !torch.vtensor<[23,256],f32>
    %__auto.blk.0.ffn_up.weight = util.global.load @__auto.blk.0.ffn_up.weight : tensor<23x256xf32>
    %10 = torch_c.from_builtin_tensor %__auto.blk.0.ffn_up.weight : tensor<23x256xf32> -> !torch.vtensor<[23,256],f32>
    %__auto.blk.0.ffn_down.weight = util.global.load @__auto.blk.0.ffn_down.weight : tensor<256x23xf32>
    %11 = torch_c.from_builtin_tensor %__auto.blk.0.ffn_down.weight : tensor<256x23xf32> -> !torch.vtensor<[256,23],f32>
    %__auto.blk.1.attn_norm.weight = util.global.load @__auto.blk.1.attn_norm.weight : tensor<256xf32>
    %12 = torch_c.from_builtin_tensor %__auto.blk.1.attn_norm.weight : tensor<256xf32> -> !torch.vtensor<[256],f32>
    %__auto.blk.1.attn_q.weight = util.global.load @__auto.blk.1.attn_q.weight : tensor<256x256xf32>
    %13 = torch_c.from_builtin_tensor %__auto.blk.1.attn_q.weight : tensor<256x256xf32> -> !torch.vtensor<[256,256],f32>
    %__auto.blk.1.attn_k.weight = util.global.load @__auto.blk.1.attn_k.weight : tensor<128x256xf32>
    %14 = torch_c.from_builtin_tensor %__auto.blk.1.attn_k.weight : tensor<128x256xf32> -> !torch.vtensor<[128,256],f32>
    %__auto.blk.1.attn_v.weight = util.global.load @__auto.blk.1.attn_v.weight : tensor<128x256xf32>
    %15 = torch_c.from_builtin_tensor %__auto.blk.1.attn_v.weight : tensor<128x256xf32> -> !torch.vtensor<[128,256],f32>
    %16 = torch.vtensor.literal(dense<0> : tensor<si64>) : !torch.vtensor<[],si64>
    %17 = torch.vtensor.literal(dense<1> : tensor<si64>) : !torch.vtensor<[],si64>
    %__auto.blk.1.attn_output.weight = util.global.load @__auto.blk.1.attn_output.weight : tensor<256x256xf32>
    %18 = torch_c.from_builtin_tensor %__auto.blk.1.attn_output.weight : tensor<256x256xf32> -> !torch.vtensor<[256,256],f32>
    %__auto.blk.1.ffn_norm.weight = util.global.load @__auto.blk.1.ffn_norm.weight : tensor<256xf32>
    %19 = torch_c.from_builtin_tensor %__auto.blk.1.ffn_norm.weight : tensor<256xf32> -> !torch.vtensor<[256],f32>
    %__auto.blk.1.ffn_gate.weight = util.global.load @__auto.blk.1.ffn_gate.weight : tensor<23x256xf32>
    %20 = torch_c.from_builtin_tensor %__auto.blk.1.ffn_gate.weight : tensor<23x256xf32> -> !torch.vtensor<[23,256],f32>
    %__auto.blk.1.ffn_up.weight = util.global.load @__auto.blk.1.ffn_up.weight : tensor<23x256xf32>
    %21 = torch_c.from_builtin_tensor %__auto.blk.1.ffn_up.weight : tensor<23x256xf32> -> !torch.vtensor<[23,256],f32>
    %__auto.blk.1.ffn_down.weight = util.global.load @__auto.blk.1.ffn_down.weight : tensor<256x23xf32>
    %22 = torch_c.from_builtin_tensor %__auto.blk.1.ffn_down.weight : tensor<256x23xf32> -> !torch.vtensor<[256,23],f32>
    %__auto.blk.2.attn_norm.weight = util.global.load @__auto.blk.2.attn_norm.weight : tensor<256xf32>
    %23 = torch_c.from_builtin_tensor %__auto.blk.2.attn_norm.weight : tensor<256xf32> -> !torch.vtensor<[256],f32>
    %__auto.blk.2.attn_q.weight = util.global.load @__auto.blk.2.attn_q.weight : tensor<256x256xf32>
    %24 = torch_c.from_builtin_tensor %__auto.blk.2.attn_q.weight : tensor<256x256xf32> -> !torch.vtensor<[256,256],f32>
    %__auto.blk.2.attn_k.weight = util.global.load @__auto.blk.2.attn_k.weight : tensor<128x256xf32>
    %25 = torch_c.from_builtin_tensor %__auto.blk.2.attn_k.weight : tensor<128x256xf32> -> !torch.vtensor<[128,256],f32>
    %__auto.blk.2.attn_v.weight = util.global.load @__auto.blk.2.attn_v.weight : tensor<128x256xf32>
    %26 = torch_c.from_builtin_tensor %__auto.blk.2.attn_v.weight : tensor<128x256xf32> -> !torch.vtensor<[128,256],f32>
    %27 = torch.vtensor.literal(dense<0> : tensor<si64>) : !torch.vtensor<[],si64>
    %28 = torch.vtensor.literal(dense<1> : tensor<si64>) : !torch.vtensor<[],si64>
    %__auto.blk.2.attn_output.weight = util.global.load @__auto.blk.2.attn_output.weight : tensor<256x256xf32>
    %29 = torch_c.from_builtin_tensor %__auto.blk.2.attn_output.weight : tensor<256x256xf32> -> !torch.vtensor<[256,256],f32>
    %__auto.blk.2.ffn_norm.weight = util.global.load @__auto.blk.2.ffn_norm.weight : tensor<256xf32>
    %30 = torch_c.from_builtin_tensor %__auto.blk.2.ffn_norm.weight : tensor<256xf32> -> !torch.vtensor<[256],f32>
    %__auto.blk.2.ffn_gate.weight = util.global.load @__auto.blk.2.ffn_gate.weight : tensor<23x256xf32>
    %31 = torch_c.from_builtin_tensor %__auto.blk.2.ffn_gate.weight : tensor<23x256xf32> -> !torch.vtensor<[23,256],f32>
    %__auto.blk.2.ffn_up.weight = util.global.load @__auto.blk.2.ffn_up.weight : tensor<23x256xf32>
    %32 = torch_c.from_builtin_tensor %__auto.blk.2.ffn_up.weight : tensor<23x256xf32> -> !torch.vtensor<[23,256],f32>
    %__auto.blk.2.ffn_down.weight = util.global.load @__auto.blk.2.ffn_down.weight : tensor<256x23xf32>
    %33 = torch_c.from_builtin_tensor %__auto.blk.2.ffn_down.weight : tensor<256x23xf32> -> !torch.vtensor<[256,23],f32>
    %__auto.output_norm.weight = util.global.load @__auto.output_norm.weight : tensor<1x256xf32>
    %34 = torch_c.from_builtin_tensor %__auto.output_norm.weight : tensor<1x256xf32> -> !torch.vtensor<[1,256],f32>
    %__auto.output.weight = util.global.load @__auto.output.weight : tensor<256x256xf32>
    %35 = torch_c.from_builtin_tensor %__auto.output.weight : tensor<256x256xf32> -> !torch.vtensor<[256,256],f32>
    %36 = torch.copy.to_vtensor %arg4 : !torch.vtensor<[?,24576],f16>
    %37 = torch.symbolic_int "s0" {min_val = 2, max_val = 3} : !torch.int
    %38 = torch.symbolic_int "s1" {min_val = 2, max_val = 9223372036854775806} : !torch.int
    torch.bind_symbolic_shape %arg3, [%37], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %36, [%38], affine_map<()[s0] -> (s0, 24576)> : !torch.vtensor<[?,24576],f16>
    %int1 = torch.constant.int 1
    %39 = torch.aten.size.int %arg3, %int1 : !torch.vtensor<[1,?],si64>, !torch.int -> !torch.int
    %int32 = torch.constant.int 32
    %40 = torch.aten.mul.int %39, %int32 : !torch.int, !torch.int -> !torch.int
    %int0 = torch.constant.int 0
    %int1_0 = torch.constant.int 1
    %none = torch.constant.none
    %none_1 = torch.constant.none
    %cpu = torch.constant.device "cpu"
    %false = torch.constant.bool false
    %41 = torch.aten.arange.start_step %int0, %40, %int1_0, %none, %none_1, %cpu, %false : !torch.int, !torch.int, !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[?],si64>
    torch.bind_symbolic_shape %41, [%37], affine_map<()[s0] -> (s0 * 32)> : !torch.vtensor<[?],si64>
    %int-1 = torch.constant.int -1
    %42 = torch.aten.unsqueeze %arg1, %int-1 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %43 = torch.aten.ge.Tensor %41, %42 : !torch.vtensor<[?],si64>, !torch.vtensor<[1,1],si64> -> !torch.vtensor<[1,?],i1>
    torch.bind_symbolic_shape %43, [%37], affine_map<()[s0] -> (1, s0 * 32)> : !torch.vtensor<[1,?],i1>
    %int0_2 = torch.constant.int 0
    %int6 = torch.constant.int 6
    %int0_3 = torch.constant.int 0
    %cpu_4 = torch.constant.device "cpu"
    %none_5 = torch.constant.none
    %44 = torch.aten.scalar_tensor %int0_2, %int6, %int0_3, %cpu_4, %none_5 : !torch.int, !torch.int, !torch.int, !torch.Device, !torch.none -> !torch.vtensor<[],f32>
    %float-Inf = torch.constant.float 0xFFF0000000000000
    %int6_6 = torch.constant.int 6
    %int0_7 = torch.constant.int 0
    %cpu_8 = torch.constant.device "cpu"
    %none_9 = torch.constant.none
    %45 = torch.aten.scalar_tensor %float-Inf, %int6_6, %int0_7, %cpu_8, %none_9 : !torch.float, !torch.int, !torch.int, !torch.Device, !torch.none -> !torch.vtensor<[],f32>
    %46 = torch.aten.where.self %43, %45, %44 : !torch.vtensor<[1,?],i1>, !torch.vtensor<[],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[1,?],f32>
    torch.bind_symbolic_shape %46, [%37], affine_map<()[s0] -> (1, s0 * 32)> : !torch.vtensor<[1,?],f32>
    %int5 = torch.constant.int 5
    %47 = torch.prims.convert_element_type %46, %int5 : !torch.vtensor<[1,?],f32>, !torch.int -> !torch.vtensor<[1,?],f16>
    torch.bind_symbolic_shape %47, [%37], affine_map<()[s0] -> (1, s0 * 32)> : !torch.vtensor<[1,?],f16>
    %int1_10 = torch.constant.int 1
    %48 = torch.aten.unsqueeze %47, %int1_10 : !torch.vtensor<[1,?],f16>, !torch.int -> !torch.vtensor<[1,1,?],f16>
    torch.bind_symbolic_shape %48, [%37], affine_map<()[s0] -> (1, 1, s0 * 32)> : !torch.vtensor<[1,1,?],f16>
    %int1_11 = torch.constant.int 1
    %49 = torch.aten.unsqueeze %48, %int1_11 : !torch.vtensor<[1,1,?],f16>, !torch.int -> !torch.vtensor<[1,1,1,?],f16>
    torch.bind_symbolic_shape %49, [%37], affine_map<()[s0] -> (1, 1, 1, s0 * 32)> : !torch.vtensor<[1,1,1,?],f16>
    %int0_12 = torch.constant.int 0
    %int1_13 = torch.constant.int 1
    %none_14 = torch.constant.none
    %none_15 = torch.constant.none
    %cpu_16 = torch.constant.device "cpu"
    %false_17 = torch.constant.bool false
    %50 = torch.aten.arange.start %int0_12, %int1_13, %none_14, %none_15, %cpu_16, %false_17 : !torch.int, !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[1],si64>
    %int0_18 = torch.constant.int 0
    %51 = torch.aten.unsqueeze %50, %int0_18 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_19 = torch.constant.int 1
    %52 = torch.aten.unsqueeze %arg2, %int1_19 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_20 = torch.constant.int 1
    %53 = torch.aten.add.Tensor %51, %52, %int1_20 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int128 = torch.constant.int 128
    %none_21 = torch.constant.none
    %none_22 = torch.constant.none
    %cpu_23 = torch.constant.device "cpu"
    %false_24 = torch.constant.bool false
    %54 = torch.aten.arange %int128, %none_21, %none_22, %cpu_23, %false_24 : !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[128],si64>
    %int0_25 = torch.constant.int 0
    %int32_26 = torch.constant.int 32
    %none_27 = torch.constant.none
    %none_28 = torch.constant.none
    %cpu_29 = torch.constant.device "cpu"
    %false_30 = torch.constant.bool false
    %55 = torch.aten.arange.start %int0_25, %int32_26, %none_27, %none_28, %cpu_29, %false_30 : !torch.int, !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[32],si64>
    %int2 = torch.constant.int 2
    %56 = torch.aten.floor_divide.Scalar %55, %int2 : !torch.vtensor<[32],si64>, !torch.int -> !torch.vtensor<[32],si64>
    %int6_31 = torch.constant.int 6
    %57 = torch.prims.convert_element_type %56, %int6_31 : !torch.vtensor<[32],si64>, !torch.int -> !torch.vtensor<[32],f32>
    %int32_32 = torch.constant.int 32
    %58 = torch.aten.div.Scalar %57, %int32_32 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[32],f32>
    %float2.000000e00 = torch.constant.float 2.000000e+00
    %59 = torch.aten.mul.Scalar %58, %float2.000000e00 : !torch.vtensor<[32],f32>, !torch.float -> !torch.vtensor<[32],f32>
    %float5.000000e05 = torch.constant.float 5.000000e+05
    %60 = torch.aten.pow.Scalar %float5.000000e05, %59 : !torch.float, !torch.vtensor<[32],f32> -> !torch.vtensor<[32],f32>
    %61 = torch.aten.reciprocal %60 : !torch.vtensor<[32],f32> -> !torch.vtensor<[32],f32>
    %float1.000000e00 = torch.constant.float 1.000000e+00
    %62 = torch.aten.mul.Scalar %61, %float1.000000e00 : !torch.vtensor<[32],f32>, !torch.float -> !torch.vtensor<[32],f32>
    %int1_33 = torch.constant.int 1
    %63 = torch.aten.unsqueeze %54, %int1_33 : !torch.vtensor<[128],si64>, !torch.int -> !torch.vtensor<[128,1],si64>
    %int0_34 = torch.constant.int 0
    %64 = torch.aten.unsqueeze %62, %int0_34 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[1,32],f32>
    %65 = torch.aten.mul.Tensor %63, %64 : !torch.vtensor<[128,1],si64>, !torch.vtensor<[1,32],f32> -> !torch.vtensor<[128,32],f32>
    %int1_35 = torch.constant.int 1
    %66 = torch.prim.ListConstruct %int1_35 : (!torch.int) -> !torch.list<int>
    %67 = torch.aten.view %53, %66 : !torch.vtensor<[1,1],si64>, !torch.list<int> -> !torch.vtensor<[1],si64>
    %68 = torch.prim.ListConstruct %67 : (!torch.vtensor<[1],si64>) -> !torch.list<optional<vtensor>>
    %69 = torch.aten.index.Tensor %65, %68 : !torch.vtensor<[128,32],f32>, !torch.list<optional<vtensor>> -> !torch.vtensor<[1,32],f32>
    %int1_36 = torch.constant.int 1
    %70 = torch.aten.unsqueeze %69, %int1_36 : !torch.vtensor<[1,32],f32>, !torch.int -> !torch.vtensor<[1,1,32],f32>
    %int5_37 = torch.constant.int 5
    %71 = torch.prims.convert_element_type %0, %int5_37 : !torch.vtensor<[256,256],f32>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int-1_38 = torch.constant.int -1
    %false_39 = torch.constant.bool false
    %false_40 = torch.constant.bool false
    %72 = torch.aten.embedding %71, %arg0, %int-1_38, %false_39, %false_40 : !torch.vtensor<[256,256],f16>, !torch.vtensor<[1,1],si64>, !torch.int, !torch.bool, !torch.bool -> !torch.vtensor<[1,1,256],f16>
    %int6_41 = torch.constant.int 6
    %73 = torch.prims.convert_element_type %72, %int6_41 : !torch.vtensor<[1,1,256],f16>, !torch.int -> !torch.vtensor<[1,1,256],f32>
    %int2_42 = torch.constant.int 2
    %74 = torch.aten.pow.Tensor_Scalar %73, %int2_42 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f32>
    %int-1_43 = torch.constant.int -1
    %75 = torch.prim.ListConstruct %int-1_43 : (!torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %none_44 = torch.constant.none
    %76 = torch.aten.mean.dim %74, %75, %true, %none_44 : !torch.vtensor<[1,1,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1],f32>
    %float1.000000e-02 = torch.constant.float 1.000000e-02
    %int1_45 = torch.constant.int 1
    %77 = torch.aten.add.Scalar %76, %float1.000000e-02, %int1_45 : !torch.vtensor<[1,1,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,1,1],f32>
    %78 = torch.aten.rsqrt %77 : !torch.vtensor<[1,1,1],f32> -> !torch.vtensor<[1,1,1],f32>
    %79 = torch.aten.mul.Tensor %73, %78 : !torch.vtensor<[1,1,256],f32>, !torch.vtensor<[1,1,1],f32> -> !torch.vtensor<[1,1,256],f32>
    %int5_46 = torch.constant.int 5
    %80 = torch.prims.convert_element_type %79, %int5_46 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %81 = torch.aten.mul.Tensor %1, %80 : !torch.vtensor<[256],f32>, !torch.vtensor<[1,1,256],f16> -> !torch.vtensor<[1,1,256],f32>
    %int5_47 = torch.constant.int 5
    %82 = torch.prims.convert_element_type %81, %int5_47 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %int5_48 = torch.constant.int 5
    %83 = torch.prims.convert_element_type %2, %int5_48 : !torch.vtensor<[256,256],f32>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int-2 = torch.constant.int -2
    %int-1_49 = torch.constant.int -1
    %84 = torch.aten.transpose.int %83, %int-2, %int-1_49 : !torch.vtensor<[256,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,256],f16>
    %int1_50 = torch.constant.int 1
    %int256 = torch.constant.int 256
    %85 = torch.prim.ListConstruct %int1_50, %int256 : (!torch.int, !torch.int) -> !torch.list<int>
    %86 = torch.aten.view %82, %85 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %87 = torch.aten.mm %86, %84 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,256],f16> -> !torch.vtensor<[1,256],f16>
    %int1_51 = torch.constant.int 1
    %int1_52 = torch.constant.int 1
    %int256_53 = torch.constant.int 256
    %88 = torch.prim.ListConstruct %int1_51, %int1_52, %int256_53 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %89 = torch.aten.view %87, %88 : !torch.vtensor<[1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,1,256],f16>
    %int5_54 = torch.constant.int 5
    %90 = torch.prims.convert_element_type %3, %int5_54 : !torch.vtensor<[128,256],f32>, !torch.int -> !torch.vtensor<[128,256],f16>
    %int-2_55 = torch.constant.int -2
    %int-1_56 = torch.constant.int -1
    %91 = torch.aten.transpose.int %90, %int-2_55, %int-1_56 : !torch.vtensor<[128,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,128],f16>
    %int1_57 = torch.constant.int 1
    %int256_58 = torch.constant.int 256
    %92 = torch.prim.ListConstruct %int1_57, %int256_58 : (!torch.int, !torch.int) -> !torch.list<int>
    %93 = torch.aten.view %82, %92 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %94 = torch.aten.mm %93, %91 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,128],f16> -> !torch.vtensor<[1,128],f16>
    %int1_59 = torch.constant.int 1
    %int1_60 = torch.constant.int 1
    %int128_61 = torch.constant.int 128
    %95 = torch.prim.ListConstruct %int1_59, %int1_60, %int128_61 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %96 = torch.aten.view %94, %95 : !torch.vtensor<[1,128],f16>, !torch.list<int> -> !torch.vtensor<[1,1,128],f16>
    %int5_62 = torch.constant.int 5
    %97 = torch.prims.convert_element_type %4, %int5_62 : !torch.vtensor<[128,256],f32>, !torch.int -> !torch.vtensor<[128,256],f16>
    %int-2_63 = torch.constant.int -2
    %int-1_64 = torch.constant.int -1
    %98 = torch.aten.transpose.int %97, %int-2_63, %int-1_64 : !torch.vtensor<[128,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,128],f16>
    %int1_65 = torch.constant.int 1
    %int256_66 = torch.constant.int 256
    %99 = torch.prim.ListConstruct %int1_65, %int256_66 : (!torch.int, !torch.int) -> !torch.list<int>
    %100 = torch.aten.view %82, %99 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %101 = torch.aten.mm %100, %98 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,128],f16> -> !torch.vtensor<[1,128],f16>
    %int1_67 = torch.constant.int 1
    %int1_68 = torch.constant.int 1
    %int128_69 = torch.constant.int 128
    %102 = torch.prim.ListConstruct %int1_67, %int1_68, %int128_69 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %103 = torch.aten.view %101, %102 : !torch.vtensor<[1,128],f16>, !torch.list<int> -> !torch.vtensor<[1,1,128],f16>
    %int1_70 = torch.constant.int 1
    %int1_71 = torch.constant.int 1
    %int8 = torch.constant.int 8
    %int32_72 = torch.constant.int 32
    %104 = torch.prim.ListConstruct %int1_70, %int1_71, %int8, %int32_72 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %105 = torch.aten.view %89, %104 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,1,8,32],f16>
    %int1_73 = torch.constant.int 1
    %int1_74 = torch.constant.int 1
    %int4 = torch.constant.int 4
    %int32_75 = torch.constant.int 32
    %106 = torch.prim.ListConstruct %int1_73, %int1_74, %int4, %int32_75 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %107 = torch.aten.view %96, %106 : !torch.vtensor<[1,1,128],f16>, !torch.list<int> -> !torch.vtensor<[1,1,4,32],f16>
    %int1_76 = torch.constant.int 1
    %int1_77 = torch.constant.int 1
    %int4_78 = torch.constant.int 4
    %int32_79 = torch.constant.int 32
    %108 = torch.prim.ListConstruct %int1_76, %int1_77, %int4_78, %int32_79 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %109 = torch.aten.view %103, %108 : !torch.vtensor<[1,1,128],f16>, !torch.list<int> -> !torch.vtensor<[1,1,4,32],f16>
    %int6_80 = torch.constant.int 6
    %110 = torch.prims.convert_element_type %105, %int6_80 : !torch.vtensor<[1,1,8,32],f16>, !torch.int -> !torch.vtensor<[1,1,8,32],f32>
    %111 = torch_c.to_builtin_tensor %110 : !torch.vtensor<[1,1,8,32],f32> -> tensor<1x1x8x32xf32>
    %112 = torch_c.to_builtin_tensor %70 : !torch.vtensor<[1,1,32],f32> -> tensor<1x1x32xf32>
    %113 = util.call @sharktank_rotary_embedding_1_1_8_32_f32(%111, %112) : (tensor<1x1x8x32xf32>, tensor<1x1x32xf32>) -> tensor<1x1x8x32xf32>
    %114 = torch_c.from_builtin_tensor %113 : tensor<1x1x8x32xf32> -> !torch.vtensor<[1,1,8,32],f32>
    %int5_81 = torch.constant.int 5
    %115 = torch.prims.convert_element_type %114, %int5_81 : !torch.vtensor<[1,1,8,32],f32>, !torch.int -> !torch.vtensor<[1,1,8,32],f16>
    %int6_82 = torch.constant.int 6
    %116 = torch.prims.convert_element_type %107, %int6_82 : !torch.vtensor<[1,1,4,32],f16>, !torch.int -> !torch.vtensor<[1,1,4,32],f32>
    %117 = torch_c.to_builtin_tensor %116 : !torch.vtensor<[1,1,4,32],f32> -> tensor<1x1x4x32xf32>
    %118 = torch_c.to_builtin_tensor %70 : !torch.vtensor<[1,1,32],f32> -> tensor<1x1x32xf32>
    %119 = util.call @sharktank_rotary_embedding_1_1_4_32_f32(%117, %118) : (tensor<1x1x4x32xf32>, tensor<1x1x32xf32>) -> tensor<1x1x4x32xf32>
    %120 = torch_c.from_builtin_tensor %119 : tensor<1x1x4x32xf32> -> !torch.vtensor<[1,1,4,32],f32>
    %int5_83 = torch.constant.int 5
    %121 = torch.prims.convert_element_type %120, %int5_83 : !torch.vtensor<[1,1,4,32],f32>, !torch.int -> !torch.vtensor<[1,1,4,32],f16>
    %int0_84 = torch.constant.int 0
    %122 = torch.aten.size.int %36, %int0_84 : !torch.vtensor<[?,24576],f16>, !torch.int -> !torch.int
    %int3 = torch.constant.int 3
    %int2_85 = torch.constant.int 2
    %int32_86 = torch.constant.int 32
    %int4_87 = torch.constant.int 4
    %int32_88 = torch.constant.int 32
    %123 = torch.prim.ListConstruct %122, %int3, %int2_85, %int32_86, %int4_87, %int32_88 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %124 = torch.aten.view %36, %123 : !torch.vtensor<[?,24576],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,4,32],f16>
    torch.bind_symbolic_shape %124, [%38], affine_map<()[s0] -> (s0, 3, 2, 32, 4, 32)> : !torch.vtensor<[?,3,2,32,4,32],f16>
    %int3_89 = torch.constant.int 3
    %125 = torch.aten.mul.int %122, %int3_89 : !torch.int, !torch.int -> !torch.int
    %int2_90 = torch.constant.int 2
    %126 = torch.aten.mul.int %125, %int2_90 : !torch.int, !torch.int -> !torch.int
    %int32_91 = torch.constant.int 32
    %127 = torch.aten.mul.int %126, %int32_91 : !torch.int, !torch.int -> !torch.int
    %int4_92 = torch.constant.int 4
    %int32_93 = torch.constant.int 32
    %128 = torch.prim.ListConstruct %127, %int4_92, %int32_93 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %129 = torch.aten.view %124, %128 : !torch.vtensor<[?,3,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,4,32],f16>
    torch.bind_symbolic_shape %129, [%38], affine_map<()[s0] -> (s0 * 192, 4, 32)> : !torch.vtensor<[?,4,32],f16>
    %int32_94 = torch.constant.int 32
    %130 = torch.aten.floor_divide.Scalar %arg2, %int32_94 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int1_95 = torch.constant.int 1
    %131 = torch.aten.unsqueeze %130, %int1_95 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_96 = torch.constant.int 1
    %false_97 = torch.constant.bool false
    %132 = torch.aten.gather %arg3, %int1_96, %131, %false_97 : !torch.vtensor<[1,?],si64>, !torch.int, !torch.vtensor<[1,1],si64>, !torch.bool -> !torch.vtensor<[1,1],si64>
    %int32_98 = torch.constant.int 32
    %133 = torch.aten.remainder.Scalar %arg2, %int32_98 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int1_99 = torch.constant.int 1
    %134 = torch.aten.unsqueeze %133, %int1_99 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %none_100 = torch.constant.none
    %135 = torch.aten.clone %5, %none_100 : !torch.vtensor<[],si64>, !torch.none -> !torch.vtensor<[],si64>
    %int0_101 = torch.constant.int 0
    %136 = torch.aten.unsqueeze %135, %int0_101 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int1_102 = torch.constant.int 1
    %int1_103 = torch.constant.int 1
    %137 = torch.prim.ListConstruct %int1_102, %int1_103 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_104 = torch.constant.int 1
    %int1_105 = torch.constant.int 1
    %138 = torch.prim.ListConstruct %int1_104, %int1_105 : (!torch.int, !torch.int) -> !torch.list<int>
    %int4_106 = torch.constant.int 4
    %int0_107 = torch.constant.int 0
    %cpu_108 = torch.constant.device "cpu"
    %false_109 = torch.constant.bool false
    %139 = torch.aten.empty_strided %137, %138, %int4_106, %int0_107, %cpu_108, %false_109 : !torch.list<int>, !torch.list<int>, !torch.int, !torch.int, !torch.Device, !torch.bool -> !torch.vtensor<[1,1],si64>
    %int0_110 = torch.constant.int 0
    %140 = torch.aten.fill.Scalar %139, %int0_110 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_111 = torch.constant.int 1
    %int1_112 = torch.constant.int 1
    %141 = torch.prim.ListConstruct %int1_111, %int1_112 : (!torch.int, !torch.int) -> !torch.list<int>
    %142 = torch.aten.repeat %136, %141 : !torch.vtensor<[1],si64>, !torch.list<int> -> !torch.vtensor<[1,1],si64>
    %int3_113 = torch.constant.int 3
    %143 = torch.aten.mul.Scalar %132, %int3_113 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_114 = torch.constant.int 1
    %144 = torch.aten.add.Tensor %143, %140, %int1_114 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int2_115 = torch.constant.int 2
    %145 = torch.aten.mul.Scalar %144, %int2_115 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_116 = torch.constant.int 1
    %146 = torch.aten.add.Tensor %145, %142, %int1_116 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int32_117 = torch.constant.int 32
    %147 = torch.aten.mul.Scalar %146, %int32_117 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_118 = torch.constant.int 1
    %148 = torch.aten.add.Tensor %147, %134, %int1_118 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %149 = torch.prim.ListConstruct %148 : (!torch.vtensor<[1,1],si64>) -> !torch.list<optional<vtensor>>
    %false_119 = torch.constant.bool false
    %150 = torch.aten.index_put %129, %149, %121, %false_119 : !torch.vtensor<[?,4,32],f16>, !torch.list<optional<vtensor>>, !torch.vtensor<[1,1,4,32],f16>, !torch.bool -> !torch.vtensor<[?,4,32],f16>
    torch.bind_symbolic_shape %150, [%38], affine_map<()[s0] -> (s0 * 192, 4, 32)> : !torch.vtensor<[?,4,32],f16>
    %int3_120 = torch.constant.int 3
    %int2_121 = torch.constant.int 2
    %int32_122 = torch.constant.int 32
    %int4_123 = torch.constant.int 4
    %int32_124 = torch.constant.int 32
    %151 = torch.prim.ListConstruct %122, %int3_120, %int2_121, %int32_122, %int4_123, %int32_124 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %152 = torch.aten.view %150, %151 : !torch.vtensor<[?,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,4,32],f16>
    torch.bind_symbolic_shape %152, [%38], affine_map<()[s0] -> (s0, 3, 2, 32, 4, 32)> : !torch.vtensor<[?,3,2,32,4,32],f16>
    %int24576 = torch.constant.int 24576
    %153 = torch.prim.ListConstruct %122, %int24576 : (!torch.int, !torch.int) -> !torch.list<int>
    %154 = torch.aten.view %152, %153 : !torch.vtensor<[?,3,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,24576],f16>
    torch.bind_symbolic_shape %154, [%38], affine_map<()[s0] -> (s0, 24576)> : !torch.vtensor<[?,24576],f16>
    %int3_125 = torch.constant.int 3
    %int2_126 = torch.constant.int 2
    %int32_127 = torch.constant.int 32
    %int4_128 = torch.constant.int 4
    %int32_129 = torch.constant.int 32
    %155 = torch.prim.ListConstruct %122, %int3_125, %int2_126, %int32_127, %int4_128, %int32_129 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %156 = torch.aten.view %154, %155 : !torch.vtensor<[?,24576],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,4,32],f16>
    torch.bind_symbolic_shape %156, [%38], affine_map<()[s0] -> (s0, 3, 2, 32, 4, 32)> : !torch.vtensor<[?,3,2,32,4,32],f16>
    %int4_130 = torch.constant.int 4
    %int32_131 = torch.constant.int 32
    %157 = torch.prim.ListConstruct %127, %int4_130, %int32_131 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %158 = torch.aten.view %156, %157 : !torch.vtensor<[?,3,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,4,32],f16>
    torch.bind_symbolic_shape %158, [%38], affine_map<()[s0] -> (s0 * 192, 4, 32)> : !torch.vtensor<[?,4,32],f16>
    %int32_132 = torch.constant.int 32
    %159 = torch.aten.floor_divide.Scalar %arg2, %int32_132 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int1_133 = torch.constant.int 1
    %160 = torch.aten.unsqueeze %159, %int1_133 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_134 = torch.constant.int 1
    %false_135 = torch.constant.bool false
    %161 = torch.aten.gather %arg3, %int1_134, %160, %false_135 : !torch.vtensor<[1,?],si64>, !torch.int, !torch.vtensor<[1,1],si64>, !torch.bool -> !torch.vtensor<[1,1],si64>
    %int32_136 = torch.constant.int 32
    %162 = torch.aten.remainder.Scalar %arg2, %int32_136 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int1_137 = torch.constant.int 1
    %163 = torch.aten.unsqueeze %162, %int1_137 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %none_138 = torch.constant.none
    %164 = torch.aten.clone %6, %none_138 : !torch.vtensor<[],si64>, !torch.none -> !torch.vtensor<[],si64>
    %int0_139 = torch.constant.int 0
    %165 = torch.aten.unsqueeze %164, %int0_139 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int1_140 = torch.constant.int 1
    %int1_141 = torch.constant.int 1
    %166 = torch.prim.ListConstruct %int1_140, %int1_141 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_142 = torch.constant.int 1
    %int1_143 = torch.constant.int 1
    %167 = torch.prim.ListConstruct %int1_142, %int1_143 : (!torch.int, !torch.int) -> !torch.list<int>
    %int4_144 = torch.constant.int 4
    %int0_145 = torch.constant.int 0
    %cpu_146 = torch.constant.device "cpu"
    %false_147 = torch.constant.bool false
    %168 = torch.aten.empty_strided %166, %167, %int4_144, %int0_145, %cpu_146, %false_147 : !torch.list<int>, !torch.list<int>, !torch.int, !torch.int, !torch.Device, !torch.bool -> !torch.vtensor<[1,1],si64>
    %int0_148 = torch.constant.int 0
    %169 = torch.aten.fill.Scalar %168, %int0_148 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_149 = torch.constant.int 1
    %int1_150 = torch.constant.int 1
    %170 = torch.prim.ListConstruct %int1_149, %int1_150 : (!torch.int, !torch.int) -> !torch.list<int>
    %171 = torch.aten.repeat %165, %170 : !torch.vtensor<[1],si64>, !torch.list<int> -> !torch.vtensor<[1,1],si64>
    %int3_151 = torch.constant.int 3
    %172 = torch.aten.mul.Scalar %161, %int3_151 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_152 = torch.constant.int 1
    %173 = torch.aten.add.Tensor %172, %169, %int1_152 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int2_153 = torch.constant.int 2
    %174 = torch.aten.mul.Scalar %173, %int2_153 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_154 = torch.constant.int 1
    %175 = torch.aten.add.Tensor %174, %171, %int1_154 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int32_155 = torch.constant.int 32
    %176 = torch.aten.mul.Scalar %175, %int32_155 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_156 = torch.constant.int 1
    %177 = torch.aten.add.Tensor %176, %163, %int1_156 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %178 = torch.prim.ListConstruct %177 : (!torch.vtensor<[1,1],si64>) -> !torch.list<optional<vtensor>>
    %false_157 = torch.constant.bool false
    %179 = torch.aten.index_put %158, %178, %109, %false_157 : !torch.vtensor<[?,4,32],f16>, !torch.list<optional<vtensor>>, !torch.vtensor<[1,1,4,32],f16>, !torch.bool -> !torch.vtensor<[?,4,32],f16>
    torch.bind_symbolic_shape %179, [%38], affine_map<()[s0] -> (s0 * 192, 4, 32)> : !torch.vtensor<[?,4,32],f16>
    %int3_158 = torch.constant.int 3
    %int2_159 = torch.constant.int 2
    %int32_160 = torch.constant.int 32
    %int4_161 = torch.constant.int 4
    %int32_162 = torch.constant.int 32
    %180 = torch.prim.ListConstruct %122, %int3_158, %int2_159, %int32_160, %int4_161, %int32_162 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %181 = torch.aten.view %179, %180 : !torch.vtensor<[?,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,4,32],f16>
    torch.bind_symbolic_shape %181, [%38], affine_map<()[s0] -> (s0, 3, 2, 32, 4, 32)> : !torch.vtensor<[?,3,2,32,4,32],f16>
    %int24576_163 = torch.constant.int 24576
    %182 = torch.prim.ListConstruct %122, %int24576_163 : (!torch.int, !torch.int) -> !torch.list<int>
    %183 = torch.aten.view %181, %182 : !torch.vtensor<[?,3,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,24576],f16>
    torch.bind_symbolic_shape %183, [%38], affine_map<()[s0] -> (s0, 24576)> : !torch.vtensor<[?,24576],f16>
    %int1_164 = torch.constant.int 1
    %184 = torch.prim.ListConstruct %int1_164, %39 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_165 = torch.constant.int 1
    %185 = torch.prim.ListConstruct %39, %int1_165 : (!torch.int, !torch.int) -> !torch.list<int>
    %int4_166 = torch.constant.int 4
    %int0_167 = torch.constant.int 0
    %cpu_168 = torch.constant.device "cpu"
    %false_169 = torch.constant.bool false
    %186 = torch.aten.empty_strided %184, %185, %int4_166, %int0_167, %cpu_168, %false_169 : !torch.list<int>, !torch.list<int>, !torch.int, !torch.int, !torch.Device, !torch.bool -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %186, [%37], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %int0_170 = torch.constant.int 0
    %187 = torch.aten.fill.Scalar %186, %int0_170 : !torch.vtensor<[1,?],si64>, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %187, [%37], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %int3_171 = torch.constant.int 3
    %188 = torch.aten.mul.Scalar %arg3, %int3_171 : !torch.vtensor<[1,?],si64>, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %188, [%37], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %int1_172 = torch.constant.int 1
    %189 = torch.aten.add.Tensor %188, %187, %int1_172 : !torch.vtensor<[1,?],si64>, !torch.vtensor<[1,?],si64>, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %189, [%37], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %190 = torch.prim.ListConstruct %39 : (!torch.int) -> !torch.list<int>
    %191 = torch.aten.view %189, %190 : !torch.vtensor<[1,?],si64>, !torch.list<int> -> !torch.vtensor<[?],si64>
    torch.bind_symbolic_shape %191, [%37], affine_map<()[s0] -> (s0)> : !torch.vtensor<[?],si64>
    %int3_173 = torch.constant.int 3
    %int2_174 = torch.constant.int 2
    %int32_175 = torch.constant.int 32
    %int4_176 = torch.constant.int 4
    %int32_177 = torch.constant.int 32
    %192 = torch.prim.ListConstruct %122, %int3_173, %int2_174, %int32_175, %int4_176, %int32_177 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %193 = torch.aten.view %183, %192 : !torch.vtensor<[?,24576],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,4,32],f16>
    torch.bind_symbolic_shape %193, [%38], affine_map<()[s0] -> (s0, 3, 2, 32, 4, 32)> : !torch.vtensor<[?,3,2,32,4,32],f16>
    %int3_178 = torch.constant.int 3
    %194 = torch.aten.mul.int %122, %int3_178 : !torch.int, !torch.int -> !torch.int
    %int2_179 = torch.constant.int 2
    %int32_180 = torch.constant.int 32
    %int4_181 = torch.constant.int 4
    %int32_182 = torch.constant.int 32
    %195 = torch.prim.ListConstruct %194, %int2_179, %int32_180, %int4_181, %int32_182 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %196 = torch.aten.view %193, %195 : !torch.vtensor<[?,3,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,2,32,4,32],f16>
    torch.bind_symbolic_shape %196, [%38], affine_map<()[s0] -> (s0 * 3, 2, 32, 4, 32)> : !torch.vtensor<[?,2,32,4,32],f16>
    %int0_183 = torch.constant.int 0
    %197 = torch.aten.index_select %196, %int0_183, %191 : !torch.vtensor<[?,2,32,4,32],f16>, !torch.int, !torch.vtensor<[?],si64> -> !torch.vtensor<[?,2,32,4,32],f16>
    torch.bind_symbolic_shape %197, [%37], affine_map<()[s0] -> (s0, 2, 32, 4, 32)> : !torch.vtensor<[?,2,32,4,32],f16>
    %int1_184 = torch.constant.int 1
    %int2_185 = torch.constant.int 2
    %int32_186 = torch.constant.int 32
    %int4_187 = torch.constant.int 4
    %int32_188 = torch.constant.int 32
    %198 = torch.prim.ListConstruct %int1_184, %39, %int2_185, %int32_186, %int4_187, %int32_188 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %199 = torch.aten.view %197, %198 : !torch.vtensor<[?,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,2,32,4,32],f16>
    torch.bind_symbolic_shape %199, [%37], affine_map<()[s0] -> (1, s0, 2, 32, 4, 32)> : !torch.vtensor<[1,?,2,32,4,32],f16>
    %int0_189 = torch.constant.int 0
    %int0_190 = torch.constant.int 0
    %int9223372036854775807 = torch.constant.int 9223372036854775807
    %int1_191 = torch.constant.int 1
    %200 = torch.aten.slice.Tensor %199, %int0_189, %int0_190, %int9223372036854775807, %int1_191 : !torch.vtensor<[1,?,2,32,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,2,32,4,32],f16>
    torch.bind_symbolic_shape %200, [%37], affine_map<()[s0] -> (1, s0, 2, 32, 4, 32)> : !torch.vtensor<[1,?,2,32,4,32],f16>
    %int1_192 = torch.constant.int 1
    %int0_193 = torch.constant.int 0
    %int9223372036854775807_194 = torch.constant.int 9223372036854775807
    %int1_195 = torch.constant.int 1
    %201 = torch.aten.slice.Tensor %200, %int1_192, %int0_193, %int9223372036854775807_194, %int1_195 : !torch.vtensor<[1,?,2,32,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,2,32,4,32],f16>
    torch.bind_symbolic_shape %201, [%37], affine_map<()[s0] -> (1, s0, 2, 32, 4, 32)> : !torch.vtensor<[1,?,2,32,4,32],f16>
    %int2_196 = torch.constant.int 2
    %int0_197 = torch.constant.int 0
    %202 = torch.aten.select.int %201, %int2_196, %int0_197 : !torch.vtensor<[1,?,2,32,4,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,?,32,4,32],f16>
    torch.bind_symbolic_shape %202, [%37], affine_map<()[s0] -> (1, s0, 32, 4, 32)> : !torch.vtensor<[1,?,32,4,32],f16>
    %int32_198 = torch.constant.int 32
    %203 = torch.aten.mul.int %39, %int32_198 : !torch.int, !torch.int -> !torch.int
    %int2_199 = torch.constant.int 2
    %int0_200 = torch.constant.int 0
    %int1_201 = torch.constant.int 1
    %204 = torch.aten.slice.Tensor %202, %int2_199, %int0_200, %203, %int1_201 : !torch.vtensor<[1,?,32,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32,4,32],f16>
    torch.bind_symbolic_shape %204, [%37], affine_map<()[s0] -> (1, s0, 32, 4, 32)> : !torch.vtensor<[1,?,32,4,32],f16>
    %int0_202 = torch.constant.int 0
    %205 = torch.aten.clone %204, %int0_202 : !torch.vtensor<[1,?,32,4,32],f16>, !torch.int -> !torch.vtensor<[1,?,32,4,32],f16>
    torch.bind_symbolic_shape %205, [%37], affine_map<()[s0] -> (1, s0, 32, 4, 32)> : !torch.vtensor<[1,?,32,4,32],f16>
    %int1_203 = torch.constant.int 1
    %206 = torch.aten.size.int %201, %int1_203 : !torch.vtensor<[1,?,2,32,4,32],f16>, !torch.int -> !torch.int
    %int32_204 = torch.constant.int 32
    %207 = torch.aten.mul.int %206, %int32_204 : !torch.int, !torch.int -> !torch.int
    %int1_205 = torch.constant.int 1
    %int4_206 = torch.constant.int 4
    %int32_207 = torch.constant.int 32
    %208 = torch.prim.ListConstruct %int1_205, %207, %int4_206, %int32_207 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %209 = torch.aten._unsafe_view %205, %208 : !torch.vtensor<[1,?,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %209, [%37], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int0_208 = torch.constant.int 0
    %int0_209 = torch.constant.int 0
    %int9223372036854775807_210 = torch.constant.int 9223372036854775807
    %int1_211 = torch.constant.int 1
    %210 = torch.aten.slice.Tensor %209, %int0_208, %int0_209, %int9223372036854775807_210, %int1_211 : !torch.vtensor<[1,?,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %210, [%37], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int0_212 = torch.constant.int 0
    %int0_213 = torch.constant.int 0
    %int9223372036854775807_214 = torch.constant.int 9223372036854775807
    %int1_215 = torch.constant.int 1
    %211 = torch.aten.slice.Tensor %199, %int0_212, %int0_213, %int9223372036854775807_214, %int1_215 : !torch.vtensor<[1,?,2,32,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,2,32,4,32],f16>
    torch.bind_symbolic_shape %211, [%37], affine_map<()[s0] -> (1, s0, 2, 32, 4, 32)> : !torch.vtensor<[1,?,2,32,4,32],f16>
    %int1_216 = torch.constant.int 1
    %int0_217 = torch.constant.int 0
    %int9223372036854775807_218 = torch.constant.int 9223372036854775807
    %int1_219 = torch.constant.int 1
    %212 = torch.aten.slice.Tensor %211, %int1_216, %int0_217, %int9223372036854775807_218, %int1_219 : !torch.vtensor<[1,?,2,32,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,2,32,4,32],f16>
    torch.bind_symbolic_shape %212, [%37], affine_map<()[s0] -> (1, s0, 2, 32, 4, 32)> : !torch.vtensor<[1,?,2,32,4,32],f16>
    %int2_220 = torch.constant.int 2
    %int1_221 = torch.constant.int 1
    %213 = torch.aten.select.int %212, %int2_220, %int1_221 : !torch.vtensor<[1,?,2,32,4,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,?,32,4,32],f16>
    torch.bind_symbolic_shape %213, [%37], affine_map<()[s0] -> (1, s0, 32, 4, 32)> : !torch.vtensor<[1,?,32,4,32],f16>
    %int2_222 = torch.constant.int 2
    %int0_223 = torch.constant.int 0
    %int1_224 = torch.constant.int 1
    %214 = torch.aten.slice.Tensor %213, %int2_222, %int0_223, %203, %int1_224 : !torch.vtensor<[1,?,32,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32,4,32],f16>
    torch.bind_symbolic_shape %214, [%37], affine_map<()[s0] -> (1, s0, 32, 4, 32)> : !torch.vtensor<[1,?,32,4,32],f16>
    %int0_225 = torch.constant.int 0
    %215 = torch.aten.clone %214, %int0_225 : !torch.vtensor<[1,?,32,4,32],f16>, !torch.int -> !torch.vtensor<[1,?,32,4,32],f16>
    torch.bind_symbolic_shape %215, [%37], affine_map<()[s0] -> (1, s0, 32, 4, 32)> : !torch.vtensor<[1,?,32,4,32],f16>
    %int1_226 = torch.constant.int 1
    %216 = torch.aten.size.int %212, %int1_226 : !torch.vtensor<[1,?,2,32,4,32],f16>, !torch.int -> !torch.int
    %int32_227 = torch.constant.int 32
    %217 = torch.aten.mul.int %216, %int32_227 : !torch.int, !torch.int -> !torch.int
    %int1_228 = torch.constant.int 1
    %int4_229 = torch.constant.int 4
    %int32_230 = torch.constant.int 32
    %218 = torch.prim.ListConstruct %int1_228, %217, %int4_229, %int32_230 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %219 = torch.aten._unsafe_view %215, %218 : !torch.vtensor<[1,?,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %219, [%37], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int0_231 = torch.constant.int 0
    %int0_232 = torch.constant.int 0
    %int9223372036854775807_233 = torch.constant.int 9223372036854775807
    %int1_234 = torch.constant.int 1
    %220 = torch.aten.slice.Tensor %219, %int0_231, %int0_232, %int9223372036854775807_233, %int1_234 : !torch.vtensor<[1,?,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %220, [%37], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int-2_235 = torch.constant.int -2
    %221 = torch.aten.unsqueeze %210, %int-2_235 : !torch.vtensor<[1,?,4,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,1,32],f16>
    torch.bind_symbolic_shape %221, [%37], affine_map<()[s0] -> (1, s0 * 32, 4, 1, 32)> : !torch.vtensor<[1,?,4,1,32],f16>
    %int1_236 = torch.constant.int 1
    %222 = torch.aten.size.int %209, %int1_236 : !torch.vtensor<[1,?,4,32],f16>, !torch.int -> !torch.int
    %int1_237 = torch.constant.int 1
    %int4_238 = torch.constant.int 4
    %int2_239 = torch.constant.int 2
    %int32_240 = torch.constant.int 32
    %223 = torch.prim.ListConstruct %int1_237, %222, %int4_238, %int2_239, %int32_240 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_241 = torch.constant.bool false
    %224 = torch.aten.expand %221, %223, %false_241 : !torch.vtensor<[1,?,4,1,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,?,4,2,32],f16>
    torch.bind_symbolic_shape %224, [%37], affine_map<()[s0] -> (1, s0 * 32, 4, 2, 32)> : !torch.vtensor<[1,?,4,2,32],f16>
    %int0_242 = torch.constant.int 0
    %225 = torch.aten.clone %224, %int0_242 : !torch.vtensor<[1,?,4,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,2,32],f16>
    torch.bind_symbolic_shape %225, [%37], affine_map<()[s0] -> (1, s0 * 32, 4, 2, 32)> : !torch.vtensor<[1,?,4,2,32],f16>
    %int1_243 = torch.constant.int 1
    %int8_244 = torch.constant.int 8
    %int32_245 = torch.constant.int 32
    %226 = torch.prim.ListConstruct %int1_243, %222, %int8_244, %int32_245 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %227 = torch.aten._unsafe_view %225, %226 : !torch.vtensor<[1,?,4,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,8,32],f16>
    torch.bind_symbolic_shape %227, [%37], affine_map<()[s0] -> (1, s0 * 32, 8, 32)> : !torch.vtensor<[1,?,8,32],f16>
    %int-2_246 = torch.constant.int -2
    %228 = torch.aten.unsqueeze %220, %int-2_246 : !torch.vtensor<[1,?,4,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,1,32],f16>
    torch.bind_symbolic_shape %228, [%37], affine_map<()[s0] -> (1, s0 * 32, 4, 1, 32)> : !torch.vtensor<[1,?,4,1,32],f16>
    %int1_247 = torch.constant.int 1
    %229 = torch.aten.size.int %219, %int1_247 : !torch.vtensor<[1,?,4,32],f16>, !torch.int -> !torch.int
    %int1_248 = torch.constant.int 1
    %int4_249 = torch.constant.int 4
    %int2_250 = torch.constant.int 2
    %int32_251 = torch.constant.int 32
    %230 = torch.prim.ListConstruct %int1_248, %229, %int4_249, %int2_250, %int32_251 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_252 = torch.constant.bool false
    %231 = torch.aten.expand %228, %230, %false_252 : !torch.vtensor<[1,?,4,1,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,?,4,2,32],f16>
    torch.bind_symbolic_shape %231, [%37], affine_map<()[s0] -> (1, s0 * 32, 4, 2, 32)> : !torch.vtensor<[1,?,4,2,32],f16>
    %int0_253 = torch.constant.int 0
    %232 = torch.aten.clone %231, %int0_253 : !torch.vtensor<[1,?,4,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,2,32],f16>
    torch.bind_symbolic_shape %232, [%37], affine_map<()[s0] -> (1, s0 * 32, 4, 2, 32)> : !torch.vtensor<[1,?,4,2,32],f16>
    %int1_254 = torch.constant.int 1
    %int8_255 = torch.constant.int 8
    %int32_256 = torch.constant.int 32
    %233 = torch.prim.ListConstruct %int1_254, %229, %int8_255, %int32_256 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %234 = torch.aten._unsafe_view %232, %233 : !torch.vtensor<[1,?,4,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,8,32],f16>
    torch.bind_symbolic_shape %234, [%37], affine_map<()[s0] -> (1, s0 * 32, 8, 32)> : !torch.vtensor<[1,?,8,32],f16>
    %int1_257 = torch.constant.int 1
    %int2_258 = torch.constant.int 2
    %235 = torch.aten.transpose.int %115, %int1_257, %int2_258 : !torch.vtensor<[1,1,8,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,8,1,32],f16>
    %int1_259 = torch.constant.int 1
    %int2_260 = torch.constant.int 2
    %236 = torch.aten.transpose.int %227, %int1_259, %int2_260 : !torch.vtensor<[1,?,8,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,8,?,32],f16>
    torch.bind_symbolic_shape %236, [%37], affine_map<()[s0] -> (1, 8, s0 * 32, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %int1_261 = torch.constant.int 1
    %int2_262 = torch.constant.int 2
    %237 = torch.aten.transpose.int %234, %int1_261, %int2_262 : !torch.vtensor<[1,?,8,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,8,?,32],f16>
    torch.bind_symbolic_shape %237, [%37], affine_map<()[s0] -> (1, 8, s0 * 32, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %float0.000000e00 = torch.constant.float 0.000000e+00
    %false_263 = torch.constant.bool false
    %none_264 = torch.constant.none
    %238:2 = torch.operator "torch.aten._scaled_dot_product_flash_attention_for_cpu"(%235, %236, %237, %float0.000000e00, %false_263, %49, %none_264) : (!torch.vtensor<[1,8,1,32],f16>, !torch.vtensor<[1,8,?,32],f16>, !torch.vtensor<[1,8,?,32],f16>, !torch.float, !torch.bool, !torch.vtensor<[1,1,1,?],f16>, !torch.none) -> (!torch.vtensor<[1,8,1,32],f16>, !torch.vtensor<[1,8,1],f32>) 
    %int1_265 = torch.constant.int 1
    %int2_266 = torch.constant.int 2
    %239 = torch.aten.transpose.int %238#0, %int1_265, %int2_266 : !torch.vtensor<[1,8,1,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,1,8,32],f16>
    %int1_267 = torch.constant.int 1
    %int1_268 = torch.constant.int 1
    %int256_269 = torch.constant.int 256
    %240 = torch.prim.ListConstruct %int1_267, %int1_268, %int256_269 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %241 = torch.aten.view %239, %240 : !torch.vtensor<[1,1,8,32],f16>, !torch.list<int> -> !torch.vtensor<[1,1,256],f16>
    %int5_270 = torch.constant.int 5
    %242 = torch.prims.convert_element_type %7, %int5_270 : !torch.vtensor<[256,256],f32>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int-2_271 = torch.constant.int -2
    %int-1_272 = torch.constant.int -1
    %243 = torch.aten.transpose.int %242, %int-2_271, %int-1_272 : !torch.vtensor<[256,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,256],f16>
    %int1_273 = torch.constant.int 1
    %int256_274 = torch.constant.int 256
    %244 = torch.prim.ListConstruct %int1_273, %int256_274 : (!torch.int, !torch.int) -> !torch.list<int>
    %245 = torch.aten.view %241, %244 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %246 = torch.aten.mm %245, %243 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,256],f16> -> !torch.vtensor<[1,256],f16>
    %int1_275 = torch.constant.int 1
    %int1_276 = torch.constant.int 1
    %int256_277 = torch.constant.int 256
    %247 = torch.prim.ListConstruct %int1_275, %int1_276, %int256_277 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %248 = torch.aten.view %246, %247 : !torch.vtensor<[1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,1,256],f16>
    %int1_278 = torch.constant.int 1
    %249 = torch.aten.add.Tensor %72, %248, %int1_278 : !torch.vtensor<[1,1,256],f16>, !torch.vtensor<[1,1,256],f16>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %int6_279 = torch.constant.int 6
    %250 = torch.prims.convert_element_type %249, %int6_279 : !torch.vtensor<[1,1,256],f16>, !torch.int -> !torch.vtensor<[1,1,256],f32>
    %int2_280 = torch.constant.int 2
    %251 = torch.aten.pow.Tensor_Scalar %250, %int2_280 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f32>
    %int-1_281 = torch.constant.int -1
    %252 = torch.prim.ListConstruct %int-1_281 : (!torch.int) -> !torch.list<int>
    %true_282 = torch.constant.bool true
    %none_283 = torch.constant.none
    %253 = torch.aten.mean.dim %251, %252, %true_282, %none_283 : !torch.vtensor<[1,1,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1],f32>
    %float1.000000e-02_284 = torch.constant.float 1.000000e-02
    %int1_285 = torch.constant.int 1
    %254 = torch.aten.add.Scalar %253, %float1.000000e-02_284, %int1_285 : !torch.vtensor<[1,1,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,1,1],f32>
    %255 = torch.aten.rsqrt %254 : !torch.vtensor<[1,1,1],f32> -> !torch.vtensor<[1,1,1],f32>
    %256 = torch.aten.mul.Tensor %250, %255 : !torch.vtensor<[1,1,256],f32>, !torch.vtensor<[1,1,1],f32> -> !torch.vtensor<[1,1,256],f32>
    %int5_286 = torch.constant.int 5
    %257 = torch.prims.convert_element_type %256, %int5_286 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %258 = torch.aten.mul.Tensor %8, %257 : !torch.vtensor<[256],f32>, !torch.vtensor<[1,1,256],f16> -> !torch.vtensor<[1,1,256],f32>
    %int5_287 = torch.constant.int 5
    %259 = torch.prims.convert_element_type %258, %int5_287 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %int5_288 = torch.constant.int 5
    %260 = torch.prims.convert_element_type %9, %int5_288 : !torch.vtensor<[23,256],f32>, !torch.int -> !torch.vtensor<[23,256],f16>
    %int-2_289 = torch.constant.int -2
    %int-1_290 = torch.constant.int -1
    %261 = torch.aten.transpose.int %260, %int-2_289, %int-1_290 : !torch.vtensor<[23,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,23],f16>
    %int1_291 = torch.constant.int 1
    %int256_292 = torch.constant.int 256
    %262 = torch.prim.ListConstruct %int1_291, %int256_292 : (!torch.int, !torch.int) -> !torch.list<int>
    %263 = torch.aten.view %259, %262 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %264 = torch.aten.mm %263, %261 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,23],f16> -> !torch.vtensor<[1,23],f16>
    %int1_293 = torch.constant.int 1
    %int1_294 = torch.constant.int 1
    %int23 = torch.constant.int 23
    %265 = torch.prim.ListConstruct %int1_293, %int1_294, %int23 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %266 = torch.aten.view %264, %265 : !torch.vtensor<[1,23],f16>, !torch.list<int> -> !torch.vtensor<[1,1,23],f16>
    %267 = torch.aten.silu %266 : !torch.vtensor<[1,1,23],f16> -> !torch.vtensor<[1,1,23],f16>
    %int5_295 = torch.constant.int 5
    %268 = torch.prims.convert_element_type %10, %int5_295 : !torch.vtensor<[23,256],f32>, !torch.int -> !torch.vtensor<[23,256],f16>
    %int-2_296 = torch.constant.int -2
    %int-1_297 = torch.constant.int -1
    %269 = torch.aten.transpose.int %268, %int-2_296, %int-1_297 : !torch.vtensor<[23,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,23],f16>
    %int1_298 = torch.constant.int 1
    %int256_299 = torch.constant.int 256
    %270 = torch.prim.ListConstruct %int1_298, %int256_299 : (!torch.int, !torch.int) -> !torch.list<int>
    %271 = torch.aten.view %259, %270 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %272 = torch.aten.mm %271, %269 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,23],f16> -> !torch.vtensor<[1,23],f16>
    %int1_300 = torch.constant.int 1
    %int1_301 = torch.constant.int 1
    %int23_302 = torch.constant.int 23
    %273 = torch.prim.ListConstruct %int1_300, %int1_301, %int23_302 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %274 = torch.aten.view %272, %273 : !torch.vtensor<[1,23],f16>, !torch.list<int> -> !torch.vtensor<[1,1,23],f16>
    %275 = torch.aten.mul.Tensor %267, %274 : !torch.vtensor<[1,1,23],f16>, !torch.vtensor<[1,1,23],f16> -> !torch.vtensor<[1,1,23],f16>
    %int5_303 = torch.constant.int 5
    %276 = torch.prims.convert_element_type %11, %int5_303 : !torch.vtensor<[256,23],f32>, !torch.int -> !torch.vtensor<[256,23],f16>
    %int-2_304 = torch.constant.int -2
    %int-1_305 = torch.constant.int -1
    %277 = torch.aten.transpose.int %276, %int-2_304, %int-1_305 : !torch.vtensor<[256,23],f16>, !torch.int, !torch.int -> !torch.vtensor<[23,256],f16>
    %int1_306 = torch.constant.int 1
    %int23_307 = torch.constant.int 23
    %278 = torch.prim.ListConstruct %int1_306, %int23_307 : (!torch.int, !torch.int) -> !torch.list<int>
    %279 = torch.aten.view %275, %278 : !torch.vtensor<[1,1,23],f16>, !torch.list<int> -> !torch.vtensor<[1,23],f16>
    %280 = torch.aten.mm %279, %277 : !torch.vtensor<[1,23],f16>, !torch.vtensor<[23,256],f16> -> !torch.vtensor<[1,256],f16>
    %int1_308 = torch.constant.int 1
    %int1_309 = torch.constant.int 1
    %int256_310 = torch.constant.int 256
    %281 = torch.prim.ListConstruct %int1_308, %int1_309, %int256_310 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %282 = torch.aten.view %280, %281 : !torch.vtensor<[1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,1,256],f16>
    %int1_311 = torch.constant.int 1
    %283 = torch.aten.add.Tensor %249, %282, %int1_311 : !torch.vtensor<[1,1,256],f16>, !torch.vtensor<[1,1,256],f16>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %int6_312 = torch.constant.int 6
    %284 = torch.prims.convert_element_type %283, %int6_312 : !torch.vtensor<[1,1,256],f16>, !torch.int -> !torch.vtensor<[1,1,256],f32>
    %int2_313 = torch.constant.int 2
    %285 = torch.aten.pow.Tensor_Scalar %284, %int2_313 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f32>
    %int-1_314 = torch.constant.int -1
    %286 = torch.prim.ListConstruct %int-1_314 : (!torch.int) -> !torch.list<int>
    %true_315 = torch.constant.bool true
    %none_316 = torch.constant.none
    %287 = torch.aten.mean.dim %285, %286, %true_315, %none_316 : !torch.vtensor<[1,1,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1],f32>
    %float1.000000e-02_317 = torch.constant.float 1.000000e-02
    %int1_318 = torch.constant.int 1
    %288 = torch.aten.add.Scalar %287, %float1.000000e-02_317, %int1_318 : !torch.vtensor<[1,1,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,1,1],f32>
    %289 = torch.aten.rsqrt %288 : !torch.vtensor<[1,1,1],f32> -> !torch.vtensor<[1,1,1],f32>
    %290 = torch.aten.mul.Tensor %284, %289 : !torch.vtensor<[1,1,256],f32>, !torch.vtensor<[1,1,1],f32> -> !torch.vtensor<[1,1,256],f32>
    %int5_319 = torch.constant.int 5
    %291 = torch.prims.convert_element_type %290, %int5_319 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %292 = torch.aten.mul.Tensor %12, %291 : !torch.vtensor<[256],f32>, !torch.vtensor<[1,1,256],f16> -> !torch.vtensor<[1,1,256],f32>
    %int5_320 = torch.constant.int 5
    %293 = torch.prims.convert_element_type %292, %int5_320 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %int5_321 = torch.constant.int 5
    %294 = torch.prims.convert_element_type %13, %int5_321 : !torch.vtensor<[256,256],f32>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int-2_322 = torch.constant.int -2
    %int-1_323 = torch.constant.int -1
    %295 = torch.aten.transpose.int %294, %int-2_322, %int-1_323 : !torch.vtensor<[256,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,256],f16>
    %int1_324 = torch.constant.int 1
    %int256_325 = torch.constant.int 256
    %296 = torch.prim.ListConstruct %int1_324, %int256_325 : (!torch.int, !torch.int) -> !torch.list<int>
    %297 = torch.aten.view %293, %296 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %298 = torch.aten.mm %297, %295 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,256],f16> -> !torch.vtensor<[1,256],f16>
    %int1_326 = torch.constant.int 1
    %int1_327 = torch.constant.int 1
    %int256_328 = torch.constant.int 256
    %299 = torch.prim.ListConstruct %int1_326, %int1_327, %int256_328 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %300 = torch.aten.view %298, %299 : !torch.vtensor<[1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,1,256],f16>
    %int5_329 = torch.constant.int 5
    %301 = torch.prims.convert_element_type %14, %int5_329 : !torch.vtensor<[128,256],f32>, !torch.int -> !torch.vtensor<[128,256],f16>
    %int-2_330 = torch.constant.int -2
    %int-1_331 = torch.constant.int -1
    %302 = torch.aten.transpose.int %301, %int-2_330, %int-1_331 : !torch.vtensor<[128,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,128],f16>
    %int1_332 = torch.constant.int 1
    %int256_333 = torch.constant.int 256
    %303 = torch.prim.ListConstruct %int1_332, %int256_333 : (!torch.int, !torch.int) -> !torch.list<int>
    %304 = torch.aten.view %293, %303 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %305 = torch.aten.mm %304, %302 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,128],f16> -> !torch.vtensor<[1,128],f16>
    %int1_334 = torch.constant.int 1
    %int1_335 = torch.constant.int 1
    %int128_336 = torch.constant.int 128
    %306 = torch.prim.ListConstruct %int1_334, %int1_335, %int128_336 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %307 = torch.aten.view %305, %306 : !torch.vtensor<[1,128],f16>, !torch.list<int> -> !torch.vtensor<[1,1,128],f16>
    %int5_337 = torch.constant.int 5
    %308 = torch.prims.convert_element_type %15, %int5_337 : !torch.vtensor<[128,256],f32>, !torch.int -> !torch.vtensor<[128,256],f16>
    %int-2_338 = torch.constant.int -2
    %int-1_339 = torch.constant.int -1
    %309 = torch.aten.transpose.int %308, %int-2_338, %int-1_339 : !torch.vtensor<[128,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,128],f16>
    %int1_340 = torch.constant.int 1
    %int256_341 = torch.constant.int 256
    %310 = torch.prim.ListConstruct %int1_340, %int256_341 : (!torch.int, !torch.int) -> !torch.list<int>
    %311 = torch.aten.view %293, %310 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %312 = torch.aten.mm %311, %309 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,128],f16> -> !torch.vtensor<[1,128],f16>
    %int1_342 = torch.constant.int 1
    %int1_343 = torch.constant.int 1
    %int128_344 = torch.constant.int 128
    %313 = torch.prim.ListConstruct %int1_342, %int1_343, %int128_344 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %314 = torch.aten.view %312, %313 : !torch.vtensor<[1,128],f16>, !torch.list<int> -> !torch.vtensor<[1,1,128],f16>
    %int1_345 = torch.constant.int 1
    %int1_346 = torch.constant.int 1
    %int8_347 = torch.constant.int 8
    %int32_348 = torch.constant.int 32
    %315 = torch.prim.ListConstruct %int1_345, %int1_346, %int8_347, %int32_348 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %316 = torch.aten.view %300, %315 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,1,8,32],f16>
    %int1_349 = torch.constant.int 1
    %int1_350 = torch.constant.int 1
    %int4_351 = torch.constant.int 4
    %int32_352 = torch.constant.int 32
    %317 = torch.prim.ListConstruct %int1_349, %int1_350, %int4_351, %int32_352 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %318 = torch.aten.view %307, %317 : !torch.vtensor<[1,1,128],f16>, !torch.list<int> -> !torch.vtensor<[1,1,4,32],f16>
    %int1_353 = torch.constant.int 1
    %int1_354 = torch.constant.int 1
    %int4_355 = torch.constant.int 4
    %int32_356 = torch.constant.int 32
    %319 = torch.prim.ListConstruct %int1_353, %int1_354, %int4_355, %int32_356 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %320 = torch.aten.view %314, %319 : !torch.vtensor<[1,1,128],f16>, !torch.list<int> -> !torch.vtensor<[1,1,4,32],f16>
    %int6_357 = torch.constant.int 6
    %321 = torch.prims.convert_element_type %316, %int6_357 : !torch.vtensor<[1,1,8,32],f16>, !torch.int -> !torch.vtensor<[1,1,8,32],f32>
    %322 = torch_c.to_builtin_tensor %321 : !torch.vtensor<[1,1,8,32],f32> -> tensor<1x1x8x32xf32>
    %323 = torch_c.to_builtin_tensor %70 : !torch.vtensor<[1,1,32],f32> -> tensor<1x1x32xf32>
    %324 = util.call @sharktank_rotary_embedding_1_1_8_32_f32(%322, %323) : (tensor<1x1x8x32xf32>, tensor<1x1x32xf32>) -> tensor<1x1x8x32xf32>
    %325 = torch_c.from_builtin_tensor %324 : tensor<1x1x8x32xf32> -> !torch.vtensor<[1,1,8,32],f32>
    %int5_358 = torch.constant.int 5
    %326 = torch.prims.convert_element_type %325, %int5_358 : !torch.vtensor<[1,1,8,32],f32>, !torch.int -> !torch.vtensor<[1,1,8,32],f16>
    %int6_359 = torch.constant.int 6
    %327 = torch.prims.convert_element_type %318, %int6_359 : !torch.vtensor<[1,1,4,32],f16>, !torch.int -> !torch.vtensor<[1,1,4,32],f32>
    %328 = torch_c.to_builtin_tensor %327 : !torch.vtensor<[1,1,4,32],f32> -> tensor<1x1x4x32xf32>
    %329 = torch_c.to_builtin_tensor %70 : !torch.vtensor<[1,1,32],f32> -> tensor<1x1x32xf32>
    %330 = util.call @sharktank_rotary_embedding_1_1_4_32_f32(%328, %329) : (tensor<1x1x4x32xf32>, tensor<1x1x32xf32>) -> tensor<1x1x4x32xf32>
    %331 = torch_c.from_builtin_tensor %330 : tensor<1x1x4x32xf32> -> !torch.vtensor<[1,1,4,32],f32>
    %int5_360 = torch.constant.int 5
    %332 = torch.prims.convert_element_type %331, %int5_360 : !torch.vtensor<[1,1,4,32],f32>, !torch.int -> !torch.vtensor<[1,1,4,32],f16>
    %int32_361 = torch.constant.int 32
    %333 = torch.aten.floor_divide.Scalar %arg2, %int32_361 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int1_362 = torch.constant.int 1
    %334 = torch.aten.unsqueeze %333, %int1_362 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_363 = torch.constant.int 1
    %false_364 = torch.constant.bool false
    %335 = torch.aten.gather %arg3, %int1_363, %334, %false_364 : !torch.vtensor<[1,?],si64>, !torch.int, !torch.vtensor<[1,1],si64>, !torch.bool -> !torch.vtensor<[1,1],si64>
    %int32_365 = torch.constant.int 32
    %336 = torch.aten.remainder.Scalar %arg2, %int32_365 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int1_366 = torch.constant.int 1
    %337 = torch.aten.unsqueeze %336, %int1_366 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %none_367 = torch.constant.none
    %338 = torch.aten.clone %16, %none_367 : !torch.vtensor<[],si64>, !torch.none -> !torch.vtensor<[],si64>
    %int0_368 = torch.constant.int 0
    %339 = torch.aten.unsqueeze %338, %int0_368 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int1_369 = torch.constant.int 1
    %int1_370 = torch.constant.int 1
    %340 = torch.prim.ListConstruct %int1_369, %int1_370 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_371 = torch.constant.int 1
    %int1_372 = torch.constant.int 1
    %341 = torch.prim.ListConstruct %int1_371, %int1_372 : (!torch.int, !torch.int) -> !torch.list<int>
    %int4_373 = torch.constant.int 4
    %int0_374 = torch.constant.int 0
    %cpu_375 = torch.constant.device "cpu"
    %false_376 = torch.constant.bool false
    %342 = torch.aten.empty_strided %340, %341, %int4_373, %int0_374, %cpu_375, %false_376 : !torch.list<int>, !torch.list<int>, !torch.int, !torch.int, !torch.Device, !torch.bool -> !torch.vtensor<[1,1],si64>
    %int1_377 = torch.constant.int 1
    %343 = torch.aten.fill.Scalar %342, %int1_377 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_378 = torch.constant.int 1
    %int1_379 = torch.constant.int 1
    %344 = torch.prim.ListConstruct %int1_378, %int1_379 : (!torch.int, !torch.int) -> !torch.list<int>
    %345 = torch.aten.repeat %339, %344 : !torch.vtensor<[1],si64>, !torch.list<int> -> !torch.vtensor<[1,1],si64>
    %int3_380 = torch.constant.int 3
    %346 = torch.aten.mul.Scalar %335, %int3_380 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_381 = torch.constant.int 1
    %347 = torch.aten.add.Tensor %346, %343, %int1_381 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int2_382 = torch.constant.int 2
    %348 = torch.aten.mul.Scalar %347, %int2_382 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_383 = torch.constant.int 1
    %349 = torch.aten.add.Tensor %348, %345, %int1_383 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int32_384 = torch.constant.int 32
    %350 = torch.aten.mul.Scalar %349, %int32_384 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_385 = torch.constant.int 1
    %351 = torch.aten.add.Tensor %350, %337, %int1_385 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int3_386 = torch.constant.int 3
    %int2_387 = torch.constant.int 2
    %int32_388 = torch.constant.int 32
    %int4_389 = torch.constant.int 4
    %int32_390 = torch.constant.int 32
    %352 = torch.prim.ListConstruct %122, %int3_386, %int2_387, %int32_388, %int4_389, %int32_390 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %353 = torch.aten.view %183, %352 : !torch.vtensor<[?,24576],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,4,32],f16>
    torch.bind_symbolic_shape %353, [%38], affine_map<()[s0] -> (s0, 3, 2, 32, 4, 32)> : !torch.vtensor<[?,3,2,32,4,32],f16>
    %int3_391 = torch.constant.int 3
    %354 = torch.aten.mul.int %122, %int3_391 : !torch.int, !torch.int -> !torch.int
    %int2_392 = torch.constant.int 2
    %355 = torch.aten.mul.int %354, %int2_392 : !torch.int, !torch.int -> !torch.int
    %int32_393 = torch.constant.int 32
    %356 = torch.aten.mul.int %355, %int32_393 : !torch.int, !torch.int -> !torch.int
    %int4_394 = torch.constant.int 4
    %int32_395 = torch.constant.int 32
    %357 = torch.prim.ListConstruct %356, %int4_394, %int32_395 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %358 = torch.aten.view %353, %357 : !torch.vtensor<[?,3,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,4,32],f16>
    torch.bind_symbolic_shape %358, [%38], affine_map<()[s0] -> (s0 * 192, 4, 32)> : !torch.vtensor<[?,4,32],f16>
    %359 = torch.prim.ListConstruct %351 : (!torch.vtensor<[1,1],si64>) -> !torch.list<optional<vtensor>>
    %false_396 = torch.constant.bool false
    %360 = torch.aten.index_put %358, %359, %332, %false_396 : !torch.vtensor<[?,4,32],f16>, !torch.list<optional<vtensor>>, !torch.vtensor<[1,1,4,32],f16>, !torch.bool -> !torch.vtensor<[?,4,32],f16>
    torch.bind_symbolic_shape %360, [%38], affine_map<()[s0] -> (s0 * 192, 4, 32)> : !torch.vtensor<[?,4,32],f16>
    %int3_397 = torch.constant.int 3
    %int2_398 = torch.constant.int 2
    %int32_399 = torch.constant.int 32
    %int4_400 = torch.constant.int 4
    %int32_401 = torch.constant.int 32
    %361 = torch.prim.ListConstruct %122, %int3_397, %int2_398, %int32_399, %int4_400, %int32_401 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %362 = torch.aten.view %360, %361 : !torch.vtensor<[?,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,4,32],f16>
    torch.bind_symbolic_shape %362, [%38], affine_map<()[s0] -> (s0, 3, 2, 32, 4, 32)> : !torch.vtensor<[?,3,2,32,4,32],f16>
    %int24576_402 = torch.constant.int 24576
    %363 = torch.prim.ListConstruct %122, %int24576_402 : (!torch.int, !torch.int) -> !torch.list<int>
    %364 = torch.aten.view %362, %363 : !torch.vtensor<[?,3,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,24576],f16>
    torch.bind_symbolic_shape %364, [%38], affine_map<()[s0] -> (s0, 24576)> : !torch.vtensor<[?,24576],f16>
    %int3_403 = torch.constant.int 3
    %int2_404 = torch.constant.int 2
    %int32_405 = torch.constant.int 32
    %int4_406 = torch.constant.int 4
    %int32_407 = torch.constant.int 32
    %365 = torch.prim.ListConstruct %122, %int3_403, %int2_404, %int32_405, %int4_406, %int32_407 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %366 = torch.aten.view %364, %365 : !torch.vtensor<[?,24576],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,4,32],f16>
    torch.bind_symbolic_shape %366, [%38], affine_map<()[s0] -> (s0, 3, 2, 32, 4, 32)> : !torch.vtensor<[?,3,2,32,4,32],f16>
    %int4_408 = torch.constant.int 4
    %int32_409 = torch.constant.int 32
    %367 = torch.prim.ListConstruct %356, %int4_408, %int32_409 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %368 = torch.aten.view %366, %367 : !torch.vtensor<[?,3,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,4,32],f16>
    torch.bind_symbolic_shape %368, [%38], affine_map<()[s0] -> (s0 * 192, 4, 32)> : !torch.vtensor<[?,4,32],f16>
    %int32_410 = torch.constant.int 32
    %369 = torch.aten.floor_divide.Scalar %arg2, %int32_410 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int1_411 = torch.constant.int 1
    %370 = torch.aten.unsqueeze %369, %int1_411 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_412 = torch.constant.int 1
    %false_413 = torch.constant.bool false
    %371 = torch.aten.gather %arg3, %int1_412, %370, %false_413 : !torch.vtensor<[1,?],si64>, !torch.int, !torch.vtensor<[1,1],si64>, !torch.bool -> !torch.vtensor<[1,1],si64>
    %int32_414 = torch.constant.int 32
    %372 = torch.aten.remainder.Scalar %arg2, %int32_414 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int1_415 = torch.constant.int 1
    %373 = torch.aten.unsqueeze %372, %int1_415 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %none_416 = torch.constant.none
    %374 = torch.aten.clone %17, %none_416 : !torch.vtensor<[],si64>, !torch.none -> !torch.vtensor<[],si64>
    %int0_417 = torch.constant.int 0
    %375 = torch.aten.unsqueeze %374, %int0_417 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int1_418 = torch.constant.int 1
    %int1_419 = torch.constant.int 1
    %376 = torch.prim.ListConstruct %int1_418, %int1_419 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_420 = torch.constant.int 1
    %int1_421 = torch.constant.int 1
    %377 = torch.prim.ListConstruct %int1_420, %int1_421 : (!torch.int, !torch.int) -> !torch.list<int>
    %int4_422 = torch.constant.int 4
    %int0_423 = torch.constant.int 0
    %cpu_424 = torch.constant.device "cpu"
    %false_425 = torch.constant.bool false
    %378 = torch.aten.empty_strided %376, %377, %int4_422, %int0_423, %cpu_424, %false_425 : !torch.list<int>, !torch.list<int>, !torch.int, !torch.int, !torch.Device, !torch.bool -> !torch.vtensor<[1,1],si64>
    %int1_426 = torch.constant.int 1
    %379 = torch.aten.fill.Scalar %378, %int1_426 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_427 = torch.constant.int 1
    %int1_428 = torch.constant.int 1
    %380 = torch.prim.ListConstruct %int1_427, %int1_428 : (!torch.int, !torch.int) -> !torch.list<int>
    %381 = torch.aten.repeat %375, %380 : !torch.vtensor<[1],si64>, !torch.list<int> -> !torch.vtensor<[1,1],si64>
    %int3_429 = torch.constant.int 3
    %382 = torch.aten.mul.Scalar %371, %int3_429 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_430 = torch.constant.int 1
    %383 = torch.aten.add.Tensor %382, %379, %int1_430 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int2_431 = torch.constant.int 2
    %384 = torch.aten.mul.Scalar %383, %int2_431 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_432 = torch.constant.int 1
    %385 = torch.aten.add.Tensor %384, %381, %int1_432 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int32_433 = torch.constant.int 32
    %386 = torch.aten.mul.Scalar %385, %int32_433 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_434 = torch.constant.int 1
    %387 = torch.aten.add.Tensor %386, %373, %int1_434 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %388 = torch.prim.ListConstruct %387 : (!torch.vtensor<[1,1],si64>) -> !torch.list<optional<vtensor>>
    %false_435 = torch.constant.bool false
    %389 = torch.aten.index_put %368, %388, %320, %false_435 : !torch.vtensor<[?,4,32],f16>, !torch.list<optional<vtensor>>, !torch.vtensor<[1,1,4,32],f16>, !torch.bool -> !torch.vtensor<[?,4,32],f16>
    torch.bind_symbolic_shape %389, [%38], affine_map<()[s0] -> (s0 * 192, 4, 32)> : !torch.vtensor<[?,4,32],f16>
    %int3_436 = torch.constant.int 3
    %int2_437 = torch.constant.int 2
    %int32_438 = torch.constant.int 32
    %int4_439 = torch.constant.int 4
    %int32_440 = torch.constant.int 32
    %390 = torch.prim.ListConstruct %122, %int3_436, %int2_437, %int32_438, %int4_439, %int32_440 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %391 = torch.aten.view %389, %390 : !torch.vtensor<[?,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,4,32],f16>
    torch.bind_symbolic_shape %391, [%38], affine_map<()[s0] -> (s0, 3, 2, 32, 4, 32)> : !torch.vtensor<[?,3,2,32,4,32],f16>
    %int24576_441 = torch.constant.int 24576
    %392 = torch.prim.ListConstruct %122, %int24576_441 : (!torch.int, !torch.int) -> !torch.list<int>
    %393 = torch.aten.view %391, %392 : !torch.vtensor<[?,3,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,24576],f16>
    torch.bind_symbolic_shape %393, [%38], affine_map<()[s0] -> (s0, 24576)> : !torch.vtensor<[?,24576],f16>
    %int1_442 = torch.constant.int 1
    %394 = torch.prim.ListConstruct %int1_442, %39 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_443 = torch.constant.int 1
    %395 = torch.prim.ListConstruct %39, %int1_443 : (!torch.int, !torch.int) -> !torch.list<int>
    %int4_444 = torch.constant.int 4
    %int0_445 = torch.constant.int 0
    %cpu_446 = torch.constant.device "cpu"
    %false_447 = torch.constant.bool false
    %396 = torch.aten.empty_strided %394, %395, %int4_444, %int0_445, %cpu_446, %false_447 : !torch.list<int>, !torch.list<int>, !torch.int, !torch.int, !torch.Device, !torch.bool -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %396, [%37], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %int1_448 = torch.constant.int 1
    %397 = torch.aten.fill.Scalar %396, %int1_448 : !torch.vtensor<[1,?],si64>, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %397, [%37], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %int3_449 = torch.constant.int 3
    %398 = torch.aten.mul.Scalar %arg3, %int3_449 : !torch.vtensor<[1,?],si64>, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %398, [%37], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %int1_450 = torch.constant.int 1
    %399 = torch.aten.add.Tensor %398, %397, %int1_450 : !torch.vtensor<[1,?],si64>, !torch.vtensor<[1,?],si64>, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %399, [%37], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %400 = torch.prim.ListConstruct %39 : (!torch.int) -> !torch.list<int>
    %401 = torch.aten.view %399, %400 : !torch.vtensor<[1,?],si64>, !torch.list<int> -> !torch.vtensor<[?],si64>
    torch.bind_symbolic_shape %401, [%37], affine_map<()[s0] -> (s0)> : !torch.vtensor<[?],si64>
    %int3_451 = torch.constant.int 3
    %int2_452 = torch.constant.int 2
    %int32_453 = torch.constant.int 32
    %int4_454 = torch.constant.int 4
    %int32_455 = torch.constant.int 32
    %402 = torch.prim.ListConstruct %122, %int3_451, %int2_452, %int32_453, %int4_454, %int32_455 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %403 = torch.aten.view %393, %402 : !torch.vtensor<[?,24576],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,4,32],f16>
    torch.bind_symbolic_shape %403, [%38], affine_map<()[s0] -> (s0, 3, 2, 32, 4, 32)> : !torch.vtensor<[?,3,2,32,4,32],f16>
    %int3_456 = torch.constant.int 3
    %404 = torch.aten.mul.int %122, %int3_456 : !torch.int, !torch.int -> !torch.int
    %int2_457 = torch.constant.int 2
    %int32_458 = torch.constant.int 32
    %int4_459 = torch.constant.int 4
    %int32_460 = torch.constant.int 32
    %405 = torch.prim.ListConstruct %404, %int2_457, %int32_458, %int4_459, %int32_460 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %406 = torch.aten.view %403, %405 : !torch.vtensor<[?,3,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,2,32,4,32],f16>
    torch.bind_symbolic_shape %406, [%38], affine_map<()[s0] -> (s0 * 3, 2, 32, 4, 32)> : !torch.vtensor<[?,2,32,4,32],f16>
    %int0_461 = torch.constant.int 0
    %407 = torch.aten.index_select %406, %int0_461, %401 : !torch.vtensor<[?,2,32,4,32],f16>, !torch.int, !torch.vtensor<[?],si64> -> !torch.vtensor<[?,2,32,4,32],f16>
    torch.bind_symbolic_shape %407, [%37], affine_map<()[s0] -> (s0, 2, 32, 4, 32)> : !torch.vtensor<[?,2,32,4,32],f16>
    %int1_462 = torch.constant.int 1
    %int2_463 = torch.constant.int 2
    %int32_464 = torch.constant.int 32
    %int4_465 = torch.constant.int 4
    %int32_466 = torch.constant.int 32
    %408 = torch.prim.ListConstruct %int1_462, %39, %int2_463, %int32_464, %int4_465, %int32_466 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %409 = torch.aten.view %407, %408 : !torch.vtensor<[?,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,2,32,4,32],f16>
    torch.bind_symbolic_shape %409, [%37], affine_map<()[s0] -> (1, s0, 2, 32, 4, 32)> : !torch.vtensor<[1,?,2,32,4,32],f16>
    %int0_467 = torch.constant.int 0
    %int0_468 = torch.constant.int 0
    %int9223372036854775807_469 = torch.constant.int 9223372036854775807
    %int1_470 = torch.constant.int 1
    %410 = torch.aten.slice.Tensor %409, %int0_467, %int0_468, %int9223372036854775807_469, %int1_470 : !torch.vtensor<[1,?,2,32,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,2,32,4,32],f16>
    torch.bind_symbolic_shape %410, [%37], affine_map<()[s0] -> (1, s0, 2, 32, 4, 32)> : !torch.vtensor<[1,?,2,32,4,32],f16>
    %int1_471 = torch.constant.int 1
    %int0_472 = torch.constant.int 0
    %int9223372036854775807_473 = torch.constant.int 9223372036854775807
    %int1_474 = torch.constant.int 1
    %411 = torch.aten.slice.Tensor %410, %int1_471, %int0_472, %int9223372036854775807_473, %int1_474 : !torch.vtensor<[1,?,2,32,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,2,32,4,32],f16>
    torch.bind_symbolic_shape %411, [%37], affine_map<()[s0] -> (1, s0, 2, 32, 4, 32)> : !torch.vtensor<[1,?,2,32,4,32],f16>
    %int2_475 = torch.constant.int 2
    %int0_476 = torch.constant.int 0
    %412 = torch.aten.select.int %411, %int2_475, %int0_476 : !torch.vtensor<[1,?,2,32,4,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,?,32,4,32],f16>
    torch.bind_symbolic_shape %412, [%37], affine_map<()[s0] -> (1, s0, 32, 4, 32)> : !torch.vtensor<[1,?,32,4,32],f16>
    %int32_477 = torch.constant.int 32
    %413 = torch.aten.mul.int %39, %int32_477 : !torch.int, !torch.int -> !torch.int
    %int2_478 = torch.constant.int 2
    %int0_479 = torch.constant.int 0
    %int1_480 = torch.constant.int 1
    %414 = torch.aten.slice.Tensor %412, %int2_478, %int0_479, %413, %int1_480 : !torch.vtensor<[1,?,32,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32,4,32],f16>
    torch.bind_symbolic_shape %414, [%37], affine_map<()[s0] -> (1, s0, 32, 4, 32)> : !torch.vtensor<[1,?,32,4,32],f16>
    %int0_481 = torch.constant.int 0
    %415 = torch.aten.clone %414, %int0_481 : !torch.vtensor<[1,?,32,4,32],f16>, !torch.int -> !torch.vtensor<[1,?,32,4,32],f16>
    torch.bind_symbolic_shape %415, [%37], affine_map<()[s0] -> (1, s0, 32, 4, 32)> : !torch.vtensor<[1,?,32,4,32],f16>
    %int1_482 = torch.constant.int 1
    %416 = torch.aten.size.int %411, %int1_482 : !torch.vtensor<[1,?,2,32,4,32],f16>, !torch.int -> !torch.int
    %int32_483 = torch.constant.int 32
    %417 = torch.aten.mul.int %416, %int32_483 : !torch.int, !torch.int -> !torch.int
    %int1_484 = torch.constant.int 1
    %int4_485 = torch.constant.int 4
    %int32_486 = torch.constant.int 32
    %418 = torch.prim.ListConstruct %int1_484, %417, %int4_485, %int32_486 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %419 = torch.aten._unsafe_view %415, %418 : !torch.vtensor<[1,?,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %419, [%37], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int0_487 = torch.constant.int 0
    %int0_488 = torch.constant.int 0
    %int9223372036854775807_489 = torch.constant.int 9223372036854775807
    %int1_490 = torch.constant.int 1
    %420 = torch.aten.slice.Tensor %419, %int0_487, %int0_488, %int9223372036854775807_489, %int1_490 : !torch.vtensor<[1,?,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %420, [%37], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int0_491 = torch.constant.int 0
    %int0_492 = torch.constant.int 0
    %int9223372036854775807_493 = torch.constant.int 9223372036854775807
    %int1_494 = torch.constant.int 1
    %421 = torch.aten.slice.Tensor %409, %int0_491, %int0_492, %int9223372036854775807_493, %int1_494 : !torch.vtensor<[1,?,2,32,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,2,32,4,32],f16>
    torch.bind_symbolic_shape %421, [%37], affine_map<()[s0] -> (1, s0, 2, 32, 4, 32)> : !torch.vtensor<[1,?,2,32,4,32],f16>
    %int1_495 = torch.constant.int 1
    %int0_496 = torch.constant.int 0
    %int9223372036854775807_497 = torch.constant.int 9223372036854775807
    %int1_498 = torch.constant.int 1
    %422 = torch.aten.slice.Tensor %421, %int1_495, %int0_496, %int9223372036854775807_497, %int1_498 : !torch.vtensor<[1,?,2,32,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,2,32,4,32],f16>
    torch.bind_symbolic_shape %422, [%37], affine_map<()[s0] -> (1, s0, 2, 32, 4, 32)> : !torch.vtensor<[1,?,2,32,4,32],f16>
    %int2_499 = torch.constant.int 2
    %int1_500 = torch.constant.int 1
    %423 = torch.aten.select.int %422, %int2_499, %int1_500 : !torch.vtensor<[1,?,2,32,4,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,?,32,4,32],f16>
    torch.bind_symbolic_shape %423, [%37], affine_map<()[s0] -> (1, s0, 32, 4, 32)> : !torch.vtensor<[1,?,32,4,32],f16>
    %int2_501 = torch.constant.int 2
    %int0_502 = torch.constant.int 0
    %int1_503 = torch.constant.int 1
    %424 = torch.aten.slice.Tensor %423, %int2_501, %int0_502, %413, %int1_503 : !torch.vtensor<[1,?,32,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32,4,32],f16>
    torch.bind_symbolic_shape %424, [%37], affine_map<()[s0] -> (1, s0, 32, 4, 32)> : !torch.vtensor<[1,?,32,4,32],f16>
    %int0_504 = torch.constant.int 0
    %425 = torch.aten.clone %424, %int0_504 : !torch.vtensor<[1,?,32,4,32],f16>, !torch.int -> !torch.vtensor<[1,?,32,4,32],f16>
    torch.bind_symbolic_shape %425, [%37], affine_map<()[s0] -> (1, s0, 32, 4, 32)> : !torch.vtensor<[1,?,32,4,32],f16>
    %int1_505 = torch.constant.int 1
    %426 = torch.aten.size.int %422, %int1_505 : !torch.vtensor<[1,?,2,32,4,32],f16>, !torch.int -> !torch.int
    %int32_506 = torch.constant.int 32
    %427 = torch.aten.mul.int %426, %int32_506 : !torch.int, !torch.int -> !torch.int
    %int1_507 = torch.constant.int 1
    %int4_508 = torch.constant.int 4
    %int32_509 = torch.constant.int 32
    %428 = torch.prim.ListConstruct %int1_507, %427, %int4_508, %int32_509 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %429 = torch.aten._unsafe_view %425, %428 : !torch.vtensor<[1,?,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %429, [%37], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int0_510 = torch.constant.int 0
    %int0_511 = torch.constant.int 0
    %int9223372036854775807_512 = torch.constant.int 9223372036854775807
    %int1_513 = torch.constant.int 1
    %430 = torch.aten.slice.Tensor %429, %int0_510, %int0_511, %int9223372036854775807_512, %int1_513 : !torch.vtensor<[1,?,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %430, [%37], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int-2_514 = torch.constant.int -2
    %431 = torch.aten.unsqueeze %420, %int-2_514 : !torch.vtensor<[1,?,4,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,1,32],f16>
    torch.bind_symbolic_shape %431, [%37], affine_map<()[s0] -> (1, s0 * 32, 4, 1, 32)> : !torch.vtensor<[1,?,4,1,32],f16>
    %int1_515 = torch.constant.int 1
    %432 = torch.aten.size.int %419, %int1_515 : !torch.vtensor<[1,?,4,32],f16>, !torch.int -> !torch.int
    %int1_516 = torch.constant.int 1
    %int4_517 = torch.constant.int 4
    %int2_518 = torch.constant.int 2
    %int32_519 = torch.constant.int 32
    %433 = torch.prim.ListConstruct %int1_516, %432, %int4_517, %int2_518, %int32_519 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_520 = torch.constant.bool false
    %434 = torch.aten.expand %431, %433, %false_520 : !torch.vtensor<[1,?,4,1,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,?,4,2,32],f16>
    torch.bind_symbolic_shape %434, [%37], affine_map<()[s0] -> (1, s0 * 32, 4, 2, 32)> : !torch.vtensor<[1,?,4,2,32],f16>
    %int0_521 = torch.constant.int 0
    %435 = torch.aten.clone %434, %int0_521 : !torch.vtensor<[1,?,4,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,2,32],f16>
    torch.bind_symbolic_shape %435, [%37], affine_map<()[s0] -> (1, s0 * 32, 4, 2, 32)> : !torch.vtensor<[1,?,4,2,32],f16>
    %int1_522 = torch.constant.int 1
    %int8_523 = torch.constant.int 8
    %int32_524 = torch.constant.int 32
    %436 = torch.prim.ListConstruct %int1_522, %432, %int8_523, %int32_524 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %437 = torch.aten._unsafe_view %435, %436 : !torch.vtensor<[1,?,4,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,8,32],f16>
    torch.bind_symbolic_shape %437, [%37], affine_map<()[s0] -> (1, s0 * 32, 8, 32)> : !torch.vtensor<[1,?,8,32],f16>
    %int-2_525 = torch.constant.int -2
    %438 = torch.aten.unsqueeze %430, %int-2_525 : !torch.vtensor<[1,?,4,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,1,32],f16>
    torch.bind_symbolic_shape %438, [%37], affine_map<()[s0] -> (1, s0 * 32, 4, 1, 32)> : !torch.vtensor<[1,?,4,1,32],f16>
    %int1_526 = torch.constant.int 1
    %439 = torch.aten.size.int %429, %int1_526 : !torch.vtensor<[1,?,4,32],f16>, !torch.int -> !torch.int
    %int1_527 = torch.constant.int 1
    %int4_528 = torch.constant.int 4
    %int2_529 = torch.constant.int 2
    %int32_530 = torch.constant.int 32
    %440 = torch.prim.ListConstruct %int1_527, %439, %int4_528, %int2_529, %int32_530 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_531 = torch.constant.bool false
    %441 = torch.aten.expand %438, %440, %false_531 : !torch.vtensor<[1,?,4,1,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,?,4,2,32],f16>
    torch.bind_symbolic_shape %441, [%37], affine_map<()[s0] -> (1, s0 * 32, 4, 2, 32)> : !torch.vtensor<[1,?,4,2,32],f16>
    %int0_532 = torch.constant.int 0
    %442 = torch.aten.clone %441, %int0_532 : !torch.vtensor<[1,?,4,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,2,32],f16>
    torch.bind_symbolic_shape %442, [%37], affine_map<()[s0] -> (1, s0 * 32, 4, 2, 32)> : !torch.vtensor<[1,?,4,2,32],f16>
    %int1_533 = torch.constant.int 1
    %int8_534 = torch.constant.int 8
    %int32_535 = torch.constant.int 32
    %443 = torch.prim.ListConstruct %int1_533, %439, %int8_534, %int32_535 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %444 = torch.aten._unsafe_view %442, %443 : !torch.vtensor<[1,?,4,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,8,32],f16>
    torch.bind_symbolic_shape %444, [%37], affine_map<()[s0] -> (1, s0 * 32, 8, 32)> : !torch.vtensor<[1,?,8,32],f16>
    %int1_536 = torch.constant.int 1
    %int2_537 = torch.constant.int 2
    %445 = torch.aten.transpose.int %326, %int1_536, %int2_537 : !torch.vtensor<[1,1,8,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,8,1,32],f16>
    %int1_538 = torch.constant.int 1
    %int2_539 = torch.constant.int 2
    %446 = torch.aten.transpose.int %437, %int1_538, %int2_539 : !torch.vtensor<[1,?,8,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,8,?,32],f16>
    torch.bind_symbolic_shape %446, [%37], affine_map<()[s0] -> (1, 8, s0 * 32, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %int1_540 = torch.constant.int 1
    %int2_541 = torch.constant.int 2
    %447 = torch.aten.transpose.int %444, %int1_540, %int2_541 : !torch.vtensor<[1,?,8,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,8,?,32],f16>
    torch.bind_symbolic_shape %447, [%37], affine_map<()[s0] -> (1, 8, s0 * 32, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %float0.000000e00_542 = torch.constant.float 0.000000e+00
    %false_543 = torch.constant.bool false
    %none_544 = torch.constant.none
    %448:2 = torch.operator "torch.aten._scaled_dot_product_flash_attention_for_cpu"(%445, %446, %447, %float0.000000e00_542, %false_543, %49, %none_544) : (!torch.vtensor<[1,8,1,32],f16>, !torch.vtensor<[1,8,?,32],f16>, !torch.vtensor<[1,8,?,32],f16>, !torch.float, !torch.bool, !torch.vtensor<[1,1,1,?],f16>, !torch.none) -> (!torch.vtensor<[1,8,1,32],f16>, !torch.vtensor<[1,8,1],f32>) 
    %int1_545 = torch.constant.int 1
    %int2_546 = torch.constant.int 2
    %449 = torch.aten.transpose.int %448#0, %int1_545, %int2_546 : !torch.vtensor<[1,8,1,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,1,8,32],f16>
    %int1_547 = torch.constant.int 1
    %int1_548 = torch.constant.int 1
    %int256_549 = torch.constant.int 256
    %450 = torch.prim.ListConstruct %int1_547, %int1_548, %int256_549 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %451 = torch.aten.view %449, %450 : !torch.vtensor<[1,1,8,32],f16>, !torch.list<int> -> !torch.vtensor<[1,1,256],f16>
    %int5_550 = torch.constant.int 5
    %452 = torch.prims.convert_element_type %18, %int5_550 : !torch.vtensor<[256,256],f32>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int-2_551 = torch.constant.int -2
    %int-1_552 = torch.constant.int -1
    %453 = torch.aten.transpose.int %452, %int-2_551, %int-1_552 : !torch.vtensor<[256,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,256],f16>
    %int1_553 = torch.constant.int 1
    %int256_554 = torch.constant.int 256
    %454 = torch.prim.ListConstruct %int1_553, %int256_554 : (!torch.int, !torch.int) -> !torch.list<int>
    %455 = torch.aten.view %451, %454 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %456 = torch.aten.mm %455, %453 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,256],f16> -> !torch.vtensor<[1,256],f16>
    %int1_555 = torch.constant.int 1
    %int1_556 = torch.constant.int 1
    %int256_557 = torch.constant.int 256
    %457 = torch.prim.ListConstruct %int1_555, %int1_556, %int256_557 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %458 = torch.aten.view %456, %457 : !torch.vtensor<[1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,1,256],f16>
    %int1_558 = torch.constant.int 1
    %459 = torch.aten.add.Tensor %283, %458, %int1_558 : !torch.vtensor<[1,1,256],f16>, !torch.vtensor<[1,1,256],f16>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %int6_559 = torch.constant.int 6
    %460 = torch.prims.convert_element_type %459, %int6_559 : !torch.vtensor<[1,1,256],f16>, !torch.int -> !torch.vtensor<[1,1,256],f32>
    %int2_560 = torch.constant.int 2
    %461 = torch.aten.pow.Tensor_Scalar %460, %int2_560 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f32>
    %int-1_561 = torch.constant.int -1
    %462 = torch.prim.ListConstruct %int-1_561 : (!torch.int) -> !torch.list<int>
    %true_562 = torch.constant.bool true
    %none_563 = torch.constant.none
    %463 = torch.aten.mean.dim %461, %462, %true_562, %none_563 : !torch.vtensor<[1,1,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1],f32>
    %float1.000000e-02_564 = torch.constant.float 1.000000e-02
    %int1_565 = torch.constant.int 1
    %464 = torch.aten.add.Scalar %463, %float1.000000e-02_564, %int1_565 : !torch.vtensor<[1,1,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,1,1],f32>
    %465 = torch.aten.rsqrt %464 : !torch.vtensor<[1,1,1],f32> -> !torch.vtensor<[1,1,1],f32>
    %466 = torch.aten.mul.Tensor %460, %465 : !torch.vtensor<[1,1,256],f32>, !torch.vtensor<[1,1,1],f32> -> !torch.vtensor<[1,1,256],f32>
    %int5_566 = torch.constant.int 5
    %467 = torch.prims.convert_element_type %466, %int5_566 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %468 = torch.aten.mul.Tensor %19, %467 : !torch.vtensor<[256],f32>, !torch.vtensor<[1,1,256],f16> -> !torch.vtensor<[1,1,256],f32>
    %int5_567 = torch.constant.int 5
    %469 = torch.prims.convert_element_type %468, %int5_567 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %int5_568 = torch.constant.int 5
    %470 = torch.prims.convert_element_type %20, %int5_568 : !torch.vtensor<[23,256],f32>, !torch.int -> !torch.vtensor<[23,256],f16>
    %int-2_569 = torch.constant.int -2
    %int-1_570 = torch.constant.int -1
    %471 = torch.aten.transpose.int %470, %int-2_569, %int-1_570 : !torch.vtensor<[23,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,23],f16>
    %int1_571 = torch.constant.int 1
    %int256_572 = torch.constant.int 256
    %472 = torch.prim.ListConstruct %int1_571, %int256_572 : (!torch.int, !torch.int) -> !torch.list<int>
    %473 = torch.aten.view %469, %472 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %474 = torch.aten.mm %473, %471 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,23],f16> -> !torch.vtensor<[1,23],f16>
    %int1_573 = torch.constant.int 1
    %int1_574 = torch.constant.int 1
    %int23_575 = torch.constant.int 23
    %475 = torch.prim.ListConstruct %int1_573, %int1_574, %int23_575 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %476 = torch.aten.view %474, %475 : !torch.vtensor<[1,23],f16>, !torch.list<int> -> !torch.vtensor<[1,1,23],f16>
    %477 = torch.aten.silu %476 : !torch.vtensor<[1,1,23],f16> -> !torch.vtensor<[1,1,23],f16>
    %int5_576 = torch.constant.int 5
    %478 = torch.prims.convert_element_type %21, %int5_576 : !torch.vtensor<[23,256],f32>, !torch.int -> !torch.vtensor<[23,256],f16>
    %int-2_577 = torch.constant.int -2
    %int-1_578 = torch.constant.int -1
    %479 = torch.aten.transpose.int %478, %int-2_577, %int-1_578 : !torch.vtensor<[23,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,23],f16>
    %int1_579 = torch.constant.int 1
    %int256_580 = torch.constant.int 256
    %480 = torch.prim.ListConstruct %int1_579, %int256_580 : (!torch.int, !torch.int) -> !torch.list<int>
    %481 = torch.aten.view %469, %480 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %482 = torch.aten.mm %481, %479 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,23],f16> -> !torch.vtensor<[1,23],f16>
    %int1_581 = torch.constant.int 1
    %int1_582 = torch.constant.int 1
    %int23_583 = torch.constant.int 23
    %483 = torch.prim.ListConstruct %int1_581, %int1_582, %int23_583 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %484 = torch.aten.view %482, %483 : !torch.vtensor<[1,23],f16>, !torch.list<int> -> !torch.vtensor<[1,1,23],f16>
    %485 = torch.aten.mul.Tensor %477, %484 : !torch.vtensor<[1,1,23],f16>, !torch.vtensor<[1,1,23],f16> -> !torch.vtensor<[1,1,23],f16>
    %int5_584 = torch.constant.int 5
    %486 = torch.prims.convert_element_type %22, %int5_584 : !torch.vtensor<[256,23],f32>, !torch.int -> !torch.vtensor<[256,23],f16>
    %int-2_585 = torch.constant.int -2
    %int-1_586 = torch.constant.int -1
    %487 = torch.aten.transpose.int %486, %int-2_585, %int-1_586 : !torch.vtensor<[256,23],f16>, !torch.int, !torch.int -> !torch.vtensor<[23,256],f16>
    %int1_587 = torch.constant.int 1
    %int23_588 = torch.constant.int 23
    %488 = torch.prim.ListConstruct %int1_587, %int23_588 : (!torch.int, !torch.int) -> !torch.list<int>
    %489 = torch.aten.view %485, %488 : !torch.vtensor<[1,1,23],f16>, !torch.list<int> -> !torch.vtensor<[1,23],f16>
    %490 = torch.aten.mm %489, %487 : !torch.vtensor<[1,23],f16>, !torch.vtensor<[23,256],f16> -> !torch.vtensor<[1,256],f16>
    %int1_589 = torch.constant.int 1
    %int1_590 = torch.constant.int 1
    %int256_591 = torch.constant.int 256
    %491 = torch.prim.ListConstruct %int1_589, %int1_590, %int256_591 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %492 = torch.aten.view %490, %491 : !torch.vtensor<[1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,1,256],f16>
    %int1_592 = torch.constant.int 1
    %493 = torch.aten.add.Tensor %459, %492, %int1_592 : !torch.vtensor<[1,1,256],f16>, !torch.vtensor<[1,1,256],f16>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %int6_593 = torch.constant.int 6
    %494 = torch.prims.convert_element_type %493, %int6_593 : !torch.vtensor<[1,1,256],f16>, !torch.int -> !torch.vtensor<[1,1,256],f32>
    %int2_594 = torch.constant.int 2
    %495 = torch.aten.pow.Tensor_Scalar %494, %int2_594 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f32>
    %int-1_595 = torch.constant.int -1
    %496 = torch.prim.ListConstruct %int-1_595 : (!torch.int) -> !torch.list<int>
    %true_596 = torch.constant.bool true
    %none_597 = torch.constant.none
    %497 = torch.aten.mean.dim %495, %496, %true_596, %none_597 : !torch.vtensor<[1,1,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1],f32>
    %float1.000000e-02_598 = torch.constant.float 1.000000e-02
    %int1_599 = torch.constant.int 1
    %498 = torch.aten.add.Scalar %497, %float1.000000e-02_598, %int1_599 : !torch.vtensor<[1,1,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,1,1],f32>
    %499 = torch.aten.rsqrt %498 : !torch.vtensor<[1,1,1],f32> -> !torch.vtensor<[1,1,1],f32>
    %500 = torch.aten.mul.Tensor %494, %499 : !torch.vtensor<[1,1,256],f32>, !torch.vtensor<[1,1,1],f32> -> !torch.vtensor<[1,1,256],f32>
    %int5_600 = torch.constant.int 5
    %501 = torch.prims.convert_element_type %500, %int5_600 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %502 = torch.aten.mul.Tensor %23, %501 : !torch.vtensor<[256],f32>, !torch.vtensor<[1,1,256],f16> -> !torch.vtensor<[1,1,256],f32>
    %int5_601 = torch.constant.int 5
    %503 = torch.prims.convert_element_type %502, %int5_601 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %int5_602 = torch.constant.int 5
    %504 = torch.prims.convert_element_type %24, %int5_602 : !torch.vtensor<[256,256],f32>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int-2_603 = torch.constant.int -2
    %int-1_604 = torch.constant.int -1
    %505 = torch.aten.transpose.int %504, %int-2_603, %int-1_604 : !torch.vtensor<[256,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,256],f16>
    %int1_605 = torch.constant.int 1
    %int256_606 = torch.constant.int 256
    %506 = torch.prim.ListConstruct %int1_605, %int256_606 : (!torch.int, !torch.int) -> !torch.list<int>
    %507 = torch.aten.view %503, %506 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %508 = torch.aten.mm %507, %505 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,256],f16> -> !torch.vtensor<[1,256],f16>
    %int1_607 = torch.constant.int 1
    %int1_608 = torch.constant.int 1
    %int256_609 = torch.constant.int 256
    %509 = torch.prim.ListConstruct %int1_607, %int1_608, %int256_609 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %510 = torch.aten.view %508, %509 : !torch.vtensor<[1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,1,256],f16>
    %int5_610 = torch.constant.int 5
    %511 = torch.prims.convert_element_type %25, %int5_610 : !torch.vtensor<[128,256],f32>, !torch.int -> !torch.vtensor<[128,256],f16>
    %int-2_611 = torch.constant.int -2
    %int-1_612 = torch.constant.int -1
    %512 = torch.aten.transpose.int %511, %int-2_611, %int-1_612 : !torch.vtensor<[128,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,128],f16>
    %int1_613 = torch.constant.int 1
    %int256_614 = torch.constant.int 256
    %513 = torch.prim.ListConstruct %int1_613, %int256_614 : (!torch.int, !torch.int) -> !torch.list<int>
    %514 = torch.aten.view %503, %513 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %515 = torch.aten.mm %514, %512 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,128],f16> -> !torch.vtensor<[1,128],f16>
    %int1_615 = torch.constant.int 1
    %int1_616 = torch.constant.int 1
    %int128_617 = torch.constant.int 128
    %516 = torch.prim.ListConstruct %int1_615, %int1_616, %int128_617 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %517 = torch.aten.view %515, %516 : !torch.vtensor<[1,128],f16>, !torch.list<int> -> !torch.vtensor<[1,1,128],f16>
    %int5_618 = torch.constant.int 5
    %518 = torch.prims.convert_element_type %26, %int5_618 : !torch.vtensor<[128,256],f32>, !torch.int -> !torch.vtensor<[128,256],f16>
    %int-2_619 = torch.constant.int -2
    %int-1_620 = torch.constant.int -1
    %519 = torch.aten.transpose.int %518, %int-2_619, %int-1_620 : !torch.vtensor<[128,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,128],f16>
    %int1_621 = torch.constant.int 1
    %int256_622 = torch.constant.int 256
    %520 = torch.prim.ListConstruct %int1_621, %int256_622 : (!torch.int, !torch.int) -> !torch.list<int>
    %521 = torch.aten.view %503, %520 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %522 = torch.aten.mm %521, %519 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,128],f16> -> !torch.vtensor<[1,128],f16>
    %int1_623 = torch.constant.int 1
    %int1_624 = torch.constant.int 1
    %int128_625 = torch.constant.int 128
    %523 = torch.prim.ListConstruct %int1_623, %int1_624, %int128_625 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %524 = torch.aten.view %522, %523 : !torch.vtensor<[1,128],f16>, !torch.list<int> -> !torch.vtensor<[1,1,128],f16>
    %int1_626 = torch.constant.int 1
    %int1_627 = torch.constant.int 1
    %int8_628 = torch.constant.int 8
    %int32_629 = torch.constant.int 32
    %525 = torch.prim.ListConstruct %int1_626, %int1_627, %int8_628, %int32_629 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %526 = torch.aten.view %510, %525 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,1,8,32],f16>
    %int1_630 = torch.constant.int 1
    %int1_631 = torch.constant.int 1
    %int4_632 = torch.constant.int 4
    %int32_633 = torch.constant.int 32
    %527 = torch.prim.ListConstruct %int1_630, %int1_631, %int4_632, %int32_633 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %528 = torch.aten.view %517, %527 : !torch.vtensor<[1,1,128],f16>, !torch.list<int> -> !torch.vtensor<[1,1,4,32],f16>
    %int1_634 = torch.constant.int 1
    %int1_635 = torch.constant.int 1
    %int4_636 = torch.constant.int 4
    %int32_637 = torch.constant.int 32
    %529 = torch.prim.ListConstruct %int1_634, %int1_635, %int4_636, %int32_637 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %530 = torch.aten.view %524, %529 : !torch.vtensor<[1,1,128],f16>, !torch.list<int> -> !torch.vtensor<[1,1,4,32],f16>
    %int6_638 = torch.constant.int 6
    %531 = torch.prims.convert_element_type %526, %int6_638 : !torch.vtensor<[1,1,8,32],f16>, !torch.int -> !torch.vtensor<[1,1,8,32],f32>
    %532 = torch_c.to_builtin_tensor %531 : !torch.vtensor<[1,1,8,32],f32> -> tensor<1x1x8x32xf32>
    %533 = torch_c.to_builtin_tensor %70 : !torch.vtensor<[1,1,32],f32> -> tensor<1x1x32xf32>
    %534 = util.call @sharktank_rotary_embedding_1_1_8_32_f32(%532, %533) : (tensor<1x1x8x32xf32>, tensor<1x1x32xf32>) -> tensor<1x1x8x32xf32>
    %535 = torch_c.from_builtin_tensor %534 : tensor<1x1x8x32xf32> -> !torch.vtensor<[1,1,8,32],f32>
    %int5_639 = torch.constant.int 5
    %536 = torch.prims.convert_element_type %535, %int5_639 : !torch.vtensor<[1,1,8,32],f32>, !torch.int -> !torch.vtensor<[1,1,8,32],f16>
    %int6_640 = torch.constant.int 6
    %537 = torch.prims.convert_element_type %528, %int6_640 : !torch.vtensor<[1,1,4,32],f16>, !torch.int -> !torch.vtensor<[1,1,4,32],f32>
    %538 = torch_c.to_builtin_tensor %537 : !torch.vtensor<[1,1,4,32],f32> -> tensor<1x1x4x32xf32>
    %539 = torch_c.to_builtin_tensor %70 : !torch.vtensor<[1,1,32],f32> -> tensor<1x1x32xf32>
    %540 = util.call @sharktank_rotary_embedding_1_1_4_32_f32(%538, %539) : (tensor<1x1x4x32xf32>, tensor<1x1x32xf32>) -> tensor<1x1x4x32xf32>
    %541 = torch_c.from_builtin_tensor %540 : tensor<1x1x4x32xf32> -> !torch.vtensor<[1,1,4,32],f32>
    %int5_641 = torch.constant.int 5
    %542 = torch.prims.convert_element_type %541, %int5_641 : !torch.vtensor<[1,1,4,32],f32>, !torch.int -> !torch.vtensor<[1,1,4,32],f16>
    %int32_642 = torch.constant.int 32
    %543 = torch.aten.floor_divide.Scalar %arg2, %int32_642 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int1_643 = torch.constant.int 1
    %544 = torch.aten.unsqueeze %543, %int1_643 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_644 = torch.constant.int 1
    %false_645 = torch.constant.bool false
    %545 = torch.aten.gather %arg3, %int1_644, %544, %false_645 : !torch.vtensor<[1,?],si64>, !torch.int, !torch.vtensor<[1,1],si64>, !torch.bool -> !torch.vtensor<[1,1],si64>
    %int32_646 = torch.constant.int 32
    %546 = torch.aten.remainder.Scalar %arg2, %int32_646 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int1_647 = torch.constant.int 1
    %547 = torch.aten.unsqueeze %546, %int1_647 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %none_648 = torch.constant.none
    %548 = torch.aten.clone %27, %none_648 : !torch.vtensor<[],si64>, !torch.none -> !torch.vtensor<[],si64>
    %int0_649 = torch.constant.int 0
    %549 = torch.aten.unsqueeze %548, %int0_649 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int1_650 = torch.constant.int 1
    %int1_651 = torch.constant.int 1
    %550 = torch.prim.ListConstruct %int1_650, %int1_651 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_652 = torch.constant.int 1
    %int1_653 = torch.constant.int 1
    %551 = torch.prim.ListConstruct %int1_652, %int1_653 : (!torch.int, !torch.int) -> !torch.list<int>
    %int4_654 = torch.constant.int 4
    %int0_655 = torch.constant.int 0
    %cpu_656 = torch.constant.device "cpu"
    %false_657 = torch.constant.bool false
    %552 = torch.aten.empty_strided %550, %551, %int4_654, %int0_655, %cpu_656, %false_657 : !torch.list<int>, !torch.list<int>, !torch.int, !torch.int, !torch.Device, !torch.bool -> !torch.vtensor<[1,1],si64>
    %int2_658 = torch.constant.int 2
    %553 = torch.aten.fill.Scalar %552, %int2_658 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_659 = torch.constant.int 1
    %int1_660 = torch.constant.int 1
    %554 = torch.prim.ListConstruct %int1_659, %int1_660 : (!torch.int, !torch.int) -> !torch.list<int>
    %555 = torch.aten.repeat %549, %554 : !torch.vtensor<[1],si64>, !torch.list<int> -> !torch.vtensor<[1,1],si64>
    %int3_661 = torch.constant.int 3
    %556 = torch.aten.mul.Scalar %545, %int3_661 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_662 = torch.constant.int 1
    %557 = torch.aten.add.Tensor %556, %553, %int1_662 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int2_663 = torch.constant.int 2
    %558 = torch.aten.mul.Scalar %557, %int2_663 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_664 = torch.constant.int 1
    %559 = torch.aten.add.Tensor %558, %555, %int1_664 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int32_665 = torch.constant.int 32
    %560 = torch.aten.mul.Scalar %559, %int32_665 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_666 = torch.constant.int 1
    %561 = torch.aten.add.Tensor %560, %547, %int1_666 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int3_667 = torch.constant.int 3
    %int2_668 = torch.constant.int 2
    %int32_669 = torch.constant.int 32
    %int4_670 = torch.constant.int 4
    %int32_671 = torch.constant.int 32
    %562 = torch.prim.ListConstruct %122, %int3_667, %int2_668, %int32_669, %int4_670, %int32_671 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %563 = torch.aten.view %393, %562 : !torch.vtensor<[?,24576],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,4,32],f16>
    torch.bind_symbolic_shape %563, [%38], affine_map<()[s0] -> (s0, 3, 2, 32, 4, 32)> : !torch.vtensor<[?,3,2,32,4,32],f16>
    %int3_672 = torch.constant.int 3
    %564 = torch.aten.mul.int %122, %int3_672 : !torch.int, !torch.int -> !torch.int
    %int2_673 = torch.constant.int 2
    %565 = torch.aten.mul.int %564, %int2_673 : !torch.int, !torch.int -> !torch.int
    %int32_674 = torch.constant.int 32
    %566 = torch.aten.mul.int %565, %int32_674 : !torch.int, !torch.int -> !torch.int
    %int4_675 = torch.constant.int 4
    %int32_676 = torch.constant.int 32
    %567 = torch.prim.ListConstruct %566, %int4_675, %int32_676 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %568 = torch.aten.view %563, %567 : !torch.vtensor<[?,3,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,4,32],f16>
    torch.bind_symbolic_shape %568, [%38], affine_map<()[s0] -> (s0 * 192, 4, 32)> : !torch.vtensor<[?,4,32],f16>
    %569 = torch.prim.ListConstruct %561 : (!torch.vtensor<[1,1],si64>) -> !torch.list<optional<vtensor>>
    %false_677 = torch.constant.bool false
    %570 = torch.aten.index_put %568, %569, %542, %false_677 : !torch.vtensor<[?,4,32],f16>, !torch.list<optional<vtensor>>, !torch.vtensor<[1,1,4,32],f16>, !torch.bool -> !torch.vtensor<[?,4,32],f16>
    torch.bind_symbolic_shape %570, [%38], affine_map<()[s0] -> (s0 * 192, 4, 32)> : !torch.vtensor<[?,4,32],f16>
    %int3_678 = torch.constant.int 3
    %int2_679 = torch.constant.int 2
    %int32_680 = torch.constant.int 32
    %int4_681 = torch.constant.int 4
    %int32_682 = torch.constant.int 32
    %571 = torch.prim.ListConstruct %122, %int3_678, %int2_679, %int32_680, %int4_681, %int32_682 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %572 = torch.aten.view %570, %571 : !torch.vtensor<[?,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,4,32],f16>
    torch.bind_symbolic_shape %572, [%38], affine_map<()[s0] -> (s0, 3, 2, 32, 4, 32)> : !torch.vtensor<[?,3,2,32,4,32],f16>
    %int24576_683 = torch.constant.int 24576
    %573 = torch.prim.ListConstruct %122, %int24576_683 : (!torch.int, !torch.int) -> !torch.list<int>
    %574 = torch.aten.view %572, %573 : !torch.vtensor<[?,3,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,24576],f16>
    torch.bind_symbolic_shape %574, [%38], affine_map<()[s0] -> (s0, 24576)> : !torch.vtensor<[?,24576],f16>
    %int3_684 = torch.constant.int 3
    %int2_685 = torch.constant.int 2
    %int32_686 = torch.constant.int 32
    %int4_687 = torch.constant.int 4
    %int32_688 = torch.constant.int 32
    %575 = torch.prim.ListConstruct %122, %int3_684, %int2_685, %int32_686, %int4_687, %int32_688 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %576 = torch.aten.view %574, %575 : !torch.vtensor<[?,24576],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,4,32],f16>
    torch.bind_symbolic_shape %576, [%38], affine_map<()[s0] -> (s0, 3, 2, 32, 4, 32)> : !torch.vtensor<[?,3,2,32,4,32],f16>
    %int4_689 = torch.constant.int 4
    %int32_690 = torch.constant.int 32
    %577 = torch.prim.ListConstruct %566, %int4_689, %int32_690 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %578 = torch.aten.view %576, %577 : !torch.vtensor<[?,3,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,4,32],f16>
    torch.bind_symbolic_shape %578, [%38], affine_map<()[s0] -> (s0 * 192, 4, 32)> : !torch.vtensor<[?,4,32],f16>
    %int32_691 = torch.constant.int 32
    %579 = torch.aten.floor_divide.Scalar %arg2, %int32_691 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int1_692 = torch.constant.int 1
    %580 = torch.aten.unsqueeze %579, %int1_692 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_693 = torch.constant.int 1
    %false_694 = torch.constant.bool false
    %581 = torch.aten.gather %arg3, %int1_693, %580, %false_694 : !torch.vtensor<[1,?],si64>, !torch.int, !torch.vtensor<[1,1],si64>, !torch.bool -> !torch.vtensor<[1,1],si64>
    %int32_695 = torch.constant.int 32
    %582 = torch.aten.remainder.Scalar %arg2, %int32_695 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int1_696 = torch.constant.int 1
    %583 = torch.aten.unsqueeze %582, %int1_696 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %none_697 = torch.constant.none
    %584 = torch.aten.clone %28, %none_697 : !torch.vtensor<[],si64>, !torch.none -> !torch.vtensor<[],si64>
    %int0_698 = torch.constant.int 0
    %585 = torch.aten.unsqueeze %584, %int0_698 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int1_699 = torch.constant.int 1
    %int1_700 = torch.constant.int 1
    %586 = torch.prim.ListConstruct %int1_699, %int1_700 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_701 = torch.constant.int 1
    %int1_702 = torch.constant.int 1
    %587 = torch.prim.ListConstruct %int1_701, %int1_702 : (!torch.int, !torch.int) -> !torch.list<int>
    %int4_703 = torch.constant.int 4
    %int0_704 = torch.constant.int 0
    %cpu_705 = torch.constant.device "cpu"
    %false_706 = torch.constant.bool false
    %588 = torch.aten.empty_strided %586, %587, %int4_703, %int0_704, %cpu_705, %false_706 : !torch.list<int>, !torch.list<int>, !torch.int, !torch.int, !torch.Device, !torch.bool -> !torch.vtensor<[1,1],si64>
    %int2_707 = torch.constant.int 2
    %589 = torch.aten.fill.Scalar %588, %int2_707 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_708 = torch.constant.int 1
    %int1_709 = torch.constant.int 1
    %590 = torch.prim.ListConstruct %int1_708, %int1_709 : (!torch.int, !torch.int) -> !torch.list<int>
    %591 = torch.aten.repeat %585, %590 : !torch.vtensor<[1],si64>, !torch.list<int> -> !torch.vtensor<[1,1],si64>
    %int3_710 = torch.constant.int 3
    %592 = torch.aten.mul.Scalar %581, %int3_710 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_711 = torch.constant.int 1
    %593 = torch.aten.add.Tensor %592, %589, %int1_711 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int2_712 = torch.constant.int 2
    %594 = torch.aten.mul.Scalar %593, %int2_712 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_713 = torch.constant.int 1
    %595 = torch.aten.add.Tensor %594, %591, %int1_713 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int32_714 = torch.constant.int 32
    %596 = torch.aten.mul.Scalar %595, %int32_714 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_715 = torch.constant.int 1
    %597 = torch.aten.add.Tensor %596, %583, %int1_715 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %598 = torch.prim.ListConstruct %597 : (!torch.vtensor<[1,1],si64>) -> !torch.list<optional<vtensor>>
    %false_716 = torch.constant.bool false
    %599 = torch.aten.index_put %578, %598, %530, %false_716 : !torch.vtensor<[?,4,32],f16>, !torch.list<optional<vtensor>>, !torch.vtensor<[1,1,4,32],f16>, !torch.bool -> !torch.vtensor<[?,4,32],f16>
    torch.bind_symbolic_shape %599, [%38], affine_map<()[s0] -> (s0 * 192, 4, 32)> : !torch.vtensor<[?,4,32],f16>
    %int3_717 = torch.constant.int 3
    %int2_718 = torch.constant.int 2
    %int32_719 = torch.constant.int 32
    %int4_720 = torch.constant.int 4
    %int32_721 = torch.constant.int 32
    %600 = torch.prim.ListConstruct %122, %int3_717, %int2_718, %int32_719, %int4_720, %int32_721 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %601 = torch.aten.view %599, %600 : !torch.vtensor<[?,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,4,32],f16>
    torch.bind_symbolic_shape %601, [%38], affine_map<()[s0] -> (s0, 3, 2, 32, 4, 32)> : !torch.vtensor<[?,3,2,32,4,32],f16>
    %int24576_722 = torch.constant.int 24576
    %602 = torch.prim.ListConstruct %122, %int24576_722 : (!torch.int, !torch.int) -> !torch.list<int>
    %603 = torch.aten.view %601, %602 : !torch.vtensor<[?,3,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,24576],f16>
    torch.overwrite.tensor.contents %603 overwrites %arg4 : !torch.vtensor<[?,24576],f16>, !torch.tensor<[?,24576],f16>
    torch.bind_symbolic_shape %603, [%38], affine_map<()[s0] -> (s0, 24576)> : !torch.vtensor<[?,24576],f16>
    %int1_723 = torch.constant.int 1
    %604 = torch.prim.ListConstruct %int1_723, %39 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_724 = torch.constant.int 1
    %605 = torch.prim.ListConstruct %39, %int1_724 : (!torch.int, !torch.int) -> !torch.list<int>
    %int4_725 = torch.constant.int 4
    %int0_726 = torch.constant.int 0
    %cpu_727 = torch.constant.device "cpu"
    %false_728 = torch.constant.bool false
    %606 = torch.aten.empty_strided %604, %605, %int4_725, %int0_726, %cpu_727, %false_728 : !torch.list<int>, !torch.list<int>, !torch.int, !torch.int, !torch.Device, !torch.bool -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %606, [%37], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %int2_729 = torch.constant.int 2
    %607 = torch.aten.fill.Scalar %606, %int2_729 : !torch.vtensor<[1,?],si64>, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %607, [%37], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %int3_730 = torch.constant.int 3
    %608 = torch.aten.mul.Scalar %arg3, %int3_730 : !torch.vtensor<[1,?],si64>, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %608, [%37], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %int1_731 = torch.constant.int 1
    %609 = torch.aten.add.Tensor %608, %607, %int1_731 : !torch.vtensor<[1,?],si64>, !torch.vtensor<[1,?],si64>, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %609, [%37], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %610 = torch.prim.ListConstruct %39 : (!torch.int) -> !torch.list<int>
    %611 = torch.aten.view %609, %610 : !torch.vtensor<[1,?],si64>, !torch.list<int> -> !torch.vtensor<[?],si64>
    torch.bind_symbolic_shape %611, [%37], affine_map<()[s0] -> (s0)> : !torch.vtensor<[?],si64>
    %int3_732 = torch.constant.int 3
    %int2_733 = torch.constant.int 2
    %int32_734 = torch.constant.int 32
    %int4_735 = torch.constant.int 4
    %int32_736 = torch.constant.int 32
    %612 = torch.prim.ListConstruct %122, %int3_732, %int2_733, %int32_734, %int4_735, %int32_736 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %613 = torch.aten.view %603, %612 : !torch.vtensor<[?,24576],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,4,32],f16>
    torch.bind_symbolic_shape %613, [%38], affine_map<()[s0] -> (s0, 3, 2, 32, 4, 32)> : !torch.vtensor<[?,3,2,32,4,32],f16>
    %int3_737 = torch.constant.int 3
    %614 = torch.aten.mul.int %122, %int3_737 : !torch.int, !torch.int -> !torch.int
    %int2_738 = torch.constant.int 2
    %int32_739 = torch.constant.int 32
    %int4_740 = torch.constant.int 4
    %int32_741 = torch.constant.int 32
    %615 = torch.prim.ListConstruct %614, %int2_738, %int32_739, %int4_740, %int32_741 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %616 = torch.aten.view %613, %615 : !torch.vtensor<[?,3,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,2,32,4,32],f16>
    torch.bind_symbolic_shape %616, [%38], affine_map<()[s0] -> (s0 * 3, 2, 32, 4, 32)> : !torch.vtensor<[?,2,32,4,32],f16>
    %int0_742 = torch.constant.int 0
    %617 = torch.aten.index_select %616, %int0_742, %611 : !torch.vtensor<[?,2,32,4,32],f16>, !torch.int, !torch.vtensor<[?],si64> -> !torch.vtensor<[?,2,32,4,32],f16>
    torch.bind_symbolic_shape %617, [%37], affine_map<()[s0] -> (s0, 2, 32, 4, 32)> : !torch.vtensor<[?,2,32,4,32],f16>
    %int1_743 = torch.constant.int 1
    %int2_744 = torch.constant.int 2
    %int32_745 = torch.constant.int 32
    %int4_746 = torch.constant.int 4
    %int32_747 = torch.constant.int 32
    %618 = torch.prim.ListConstruct %int1_743, %39, %int2_744, %int32_745, %int4_746, %int32_747 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %619 = torch.aten.view %617, %618 : !torch.vtensor<[?,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,2,32,4,32],f16>
    torch.bind_symbolic_shape %619, [%37], affine_map<()[s0] -> (1, s0, 2, 32, 4, 32)> : !torch.vtensor<[1,?,2,32,4,32],f16>
    %int0_748 = torch.constant.int 0
    %int0_749 = torch.constant.int 0
    %int9223372036854775807_750 = torch.constant.int 9223372036854775807
    %int1_751 = torch.constant.int 1
    %620 = torch.aten.slice.Tensor %619, %int0_748, %int0_749, %int9223372036854775807_750, %int1_751 : !torch.vtensor<[1,?,2,32,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,2,32,4,32],f16>
    torch.bind_symbolic_shape %620, [%37], affine_map<()[s0] -> (1, s0, 2, 32, 4, 32)> : !torch.vtensor<[1,?,2,32,4,32],f16>
    %int1_752 = torch.constant.int 1
    %int0_753 = torch.constant.int 0
    %int9223372036854775807_754 = torch.constant.int 9223372036854775807
    %int1_755 = torch.constant.int 1
    %621 = torch.aten.slice.Tensor %620, %int1_752, %int0_753, %int9223372036854775807_754, %int1_755 : !torch.vtensor<[1,?,2,32,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,2,32,4,32],f16>
    torch.bind_symbolic_shape %621, [%37], affine_map<()[s0] -> (1, s0, 2, 32, 4, 32)> : !torch.vtensor<[1,?,2,32,4,32],f16>
    %int2_756 = torch.constant.int 2
    %int0_757 = torch.constant.int 0
    %622 = torch.aten.select.int %621, %int2_756, %int0_757 : !torch.vtensor<[1,?,2,32,4,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,?,32,4,32],f16>
    torch.bind_symbolic_shape %622, [%37], affine_map<()[s0] -> (1, s0, 32, 4, 32)> : !torch.vtensor<[1,?,32,4,32],f16>
    %int32_758 = torch.constant.int 32
    %623 = torch.aten.mul.int %39, %int32_758 : !torch.int, !torch.int -> !torch.int
    %int2_759 = torch.constant.int 2
    %int0_760 = torch.constant.int 0
    %int1_761 = torch.constant.int 1
    %624 = torch.aten.slice.Tensor %622, %int2_759, %int0_760, %623, %int1_761 : !torch.vtensor<[1,?,32,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32,4,32],f16>
    torch.bind_symbolic_shape %624, [%37], affine_map<()[s0] -> (1, s0, 32, 4, 32)> : !torch.vtensor<[1,?,32,4,32],f16>
    %int0_762 = torch.constant.int 0
    %625 = torch.aten.clone %624, %int0_762 : !torch.vtensor<[1,?,32,4,32],f16>, !torch.int -> !torch.vtensor<[1,?,32,4,32],f16>
    torch.bind_symbolic_shape %625, [%37], affine_map<()[s0] -> (1, s0, 32, 4, 32)> : !torch.vtensor<[1,?,32,4,32],f16>
    %int1_763 = torch.constant.int 1
    %626 = torch.aten.size.int %621, %int1_763 : !torch.vtensor<[1,?,2,32,4,32],f16>, !torch.int -> !torch.int
    %int32_764 = torch.constant.int 32
    %627 = torch.aten.mul.int %626, %int32_764 : !torch.int, !torch.int -> !torch.int
    %int1_765 = torch.constant.int 1
    %int4_766 = torch.constant.int 4
    %int32_767 = torch.constant.int 32
    %628 = torch.prim.ListConstruct %int1_765, %627, %int4_766, %int32_767 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %629 = torch.aten._unsafe_view %625, %628 : !torch.vtensor<[1,?,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %629, [%37], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int0_768 = torch.constant.int 0
    %int0_769 = torch.constant.int 0
    %int9223372036854775807_770 = torch.constant.int 9223372036854775807
    %int1_771 = torch.constant.int 1
    %630 = torch.aten.slice.Tensor %629, %int0_768, %int0_769, %int9223372036854775807_770, %int1_771 : !torch.vtensor<[1,?,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %630, [%37], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int0_772 = torch.constant.int 0
    %int0_773 = torch.constant.int 0
    %int9223372036854775807_774 = torch.constant.int 9223372036854775807
    %int1_775 = torch.constant.int 1
    %631 = torch.aten.slice.Tensor %619, %int0_772, %int0_773, %int9223372036854775807_774, %int1_775 : !torch.vtensor<[1,?,2,32,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,2,32,4,32],f16>
    torch.bind_symbolic_shape %631, [%37], affine_map<()[s0] -> (1, s0, 2, 32, 4, 32)> : !torch.vtensor<[1,?,2,32,4,32],f16>
    %int1_776 = torch.constant.int 1
    %int0_777 = torch.constant.int 0
    %int9223372036854775807_778 = torch.constant.int 9223372036854775807
    %int1_779 = torch.constant.int 1
    %632 = torch.aten.slice.Tensor %631, %int1_776, %int0_777, %int9223372036854775807_778, %int1_779 : !torch.vtensor<[1,?,2,32,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,2,32,4,32],f16>
    torch.bind_symbolic_shape %632, [%37], affine_map<()[s0] -> (1, s0, 2, 32, 4, 32)> : !torch.vtensor<[1,?,2,32,4,32],f16>
    %int2_780 = torch.constant.int 2
    %int1_781 = torch.constant.int 1
    %633 = torch.aten.select.int %632, %int2_780, %int1_781 : !torch.vtensor<[1,?,2,32,4,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,?,32,4,32],f16>
    torch.bind_symbolic_shape %633, [%37], affine_map<()[s0] -> (1, s0, 32, 4, 32)> : !torch.vtensor<[1,?,32,4,32],f16>
    %int2_782 = torch.constant.int 2
    %int0_783 = torch.constant.int 0
    %int1_784 = torch.constant.int 1
    %634 = torch.aten.slice.Tensor %633, %int2_782, %int0_783, %623, %int1_784 : !torch.vtensor<[1,?,32,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32,4,32],f16>
    torch.bind_symbolic_shape %634, [%37], affine_map<()[s0] -> (1, s0, 32, 4, 32)> : !torch.vtensor<[1,?,32,4,32],f16>
    %int0_785 = torch.constant.int 0
    %635 = torch.aten.clone %634, %int0_785 : !torch.vtensor<[1,?,32,4,32],f16>, !torch.int -> !torch.vtensor<[1,?,32,4,32],f16>
    torch.bind_symbolic_shape %635, [%37], affine_map<()[s0] -> (1, s0, 32, 4, 32)> : !torch.vtensor<[1,?,32,4,32],f16>
    %int1_786 = torch.constant.int 1
    %636 = torch.aten.size.int %632, %int1_786 : !torch.vtensor<[1,?,2,32,4,32],f16>, !torch.int -> !torch.int
    %int32_787 = torch.constant.int 32
    %637 = torch.aten.mul.int %636, %int32_787 : !torch.int, !torch.int -> !torch.int
    %int1_788 = torch.constant.int 1
    %int4_789 = torch.constant.int 4
    %int32_790 = torch.constant.int 32
    %638 = torch.prim.ListConstruct %int1_788, %637, %int4_789, %int32_790 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %639 = torch.aten._unsafe_view %635, %638 : !torch.vtensor<[1,?,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %639, [%37], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int0_791 = torch.constant.int 0
    %int0_792 = torch.constant.int 0
    %int9223372036854775807_793 = torch.constant.int 9223372036854775807
    %int1_794 = torch.constant.int 1
    %640 = torch.aten.slice.Tensor %639, %int0_791, %int0_792, %int9223372036854775807_793, %int1_794 : !torch.vtensor<[1,?,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %640, [%37], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int-2_795 = torch.constant.int -2
    %641 = torch.aten.unsqueeze %630, %int-2_795 : !torch.vtensor<[1,?,4,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,1,32],f16>
    torch.bind_symbolic_shape %641, [%37], affine_map<()[s0] -> (1, s0 * 32, 4, 1, 32)> : !torch.vtensor<[1,?,4,1,32],f16>
    %int1_796 = torch.constant.int 1
    %642 = torch.aten.size.int %629, %int1_796 : !torch.vtensor<[1,?,4,32],f16>, !torch.int -> !torch.int
    %int1_797 = torch.constant.int 1
    %int4_798 = torch.constant.int 4
    %int2_799 = torch.constant.int 2
    %int32_800 = torch.constant.int 32
    %643 = torch.prim.ListConstruct %int1_797, %642, %int4_798, %int2_799, %int32_800 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_801 = torch.constant.bool false
    %644 = torch.aten.expand %641, %643, %false_801 : !torch.vtensor<[1,?,4,1,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,?,4,2,32],f16>
    torch.bind_symbolic_shape %644, [%37], affine_map<()[s0] -> (1, s0 * 32, 4, 2, 32)> : !torch.vtensor<[1,?,4,2,32],f16>
    %int0_802 = torch.constant.int 0
    %645 = torch.aten.clone %644, %int0_802 : !torch.vtensor<[1,?,4,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,2,32],f16>
    torch.bind_symbolic_shape %645, [%37], affine_map<()[s0] -> (1, s0 * 32, 4, 2, 32)> : !torch.vtensor<[1,?,4,2,32],f16>
    %int1_803 = torch.constant.int 1
    %int8_804 = torch.constant.int 8
    %int32_805 = torch.constant.int 32
    %646 = torch.prim.ListConstruct %int1_803, %642, %int8_804, %int32_805 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %647 = torch.aten._unsafe_view %645, %646 : !torch.vtensor<[1,?,4,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,8,32],f16>
    torch.bind_symbolic_shape %647, [%37], affine_map<()[s0] -> (1, s0 * 32, 8, 32)> : !torch.vtensor<[1,?,8,32],f16>
    %int-2_806 = torch.constant.int -2
    %648 = torch.aten.unsqueeze %640, %int-2_806 : !torch.vtensor<[1,?,4,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,1,32],f16>
    torch.bind_symbolic_shape %648, [%37], affine_map<()[s0] -> (1, s0 * 32, 4, 1, 32)> : !torch.vtensor<[1,?,4,1,32],f16>
    %int1_807 = torch.constant.int 1
    %649 = torch.aten.size.int %639, %int1_807 : !torch.vtensor<[1,?,4,32],f16>, !torch.int -> !torch.int
    %int1_808 = torch.constant.int 1
    %int4_809 = torch.constant.int 4
    %int2_810 = torch.constant.int 2
    %int32_811 = torch.constant.int 32
    %650 = torch.prim.ListConstruct %int1_808, %649, %int4_809, %int2_810, %int32_811 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_812 = torch.constant.bool false
    %651 = torch.aten.expand %648, %650, %false_812 : !torch.vtensor<[1,?,4,1,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,?,4,2,32],f16>
    torch.bind_symbolic_shape %651, [%37], affine_map<()[s0] -> (1, s0 * 32, 4, 2, 32)> : !torch.vtensor<[1,?,4,2,32],f16>
    %int0_813 = torch.constant.int 0
    %652 = torch.aten.clone %651, %int0_813 : !torch.vtensor<[1,?,4,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,2,32],f16>
    torch.bind_symbolic_shape %652, [%37], affine_map<()[s0] -> (1, s0 * 32, 4, 2, 32)> : !torch.vtensor<[1,?,4,2,32],f16>
    %int1_814 = torch.constant.int 1
    %int8_815 = torch.constant.int 8
    %int32_816 = torch.constant.int 32
    %653 = torch.prim.ListConstruct %int1_814, %649, %int8_815, %int32_816 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %654 = torch.aten._unsafe_view %652, %653 : !torch.vtensor<[1,?,4,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,8,32],f16>
    torch.bind_symbolic_shape %654, [%37], affine_map<()[s0] -> (1, s0 * 32, 8, 32)> : !torch.vtensor<[1,?,8,32],f16>
    %int1_817 = torch.constant.int 1
    %int2_818 = torch.constant.int 2
    %655 = torch.aten.transpose.int %536, %int1_817, %int2_818 : !torch.vtensor<[1,1,8,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,8,1,32],f16>
    %int1_819 = torch.constant.int 1
    %int2_820 = torch.constant.int 2
    %656 = torch.aten.transpose.int %647, %int1_819, %int2_820 : !torch.vtensor<[1,?,8,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,8,?,32],f16>
    torch.bind_symbolic_shape %656, [%37], affine_map<()[s0] -> (1, 8, s0 * 32, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %int1_821 = torch.constant.int 1
    %int2_822 = torch.constant.int 2
    %657 = torch.aten.transpose.int %654, %int1_821, %int2_822 : !torch.vtensor<[1,?,8,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,8,?,32],f16>
    torch.bind_symbolic_shape %657, [%37], affine_map<()[s0] -> (1, 8, s0 * 32, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %float0.000000e00_823 = torch.constant.float 0.000000e+00
    %false_824 = torch.constant.bool false
    %none_825 = torch.constant.none
    %658:2 = torch.operator "torch.aten._scaled_dot_product_flash_attention_for_cpu"(%655, %656, %657, %float0.000000e00_823, %false_824, %49, %none_825) : (!torch.vtensor<[1,8,1,32],f16>, !torch.vtensor<[1,8,?,32],f16>, !torch.vtensor<[1,8,?,32],f16>, !torch.float, !torch.bool, !torch.vtensor<[1,1,1,?],f16>, !torch.none) -> (!torch.vtensor<[1,8,1,32],f16>, !torch.vtensor<[1,8,1],f32>) 
    %int1_826 = torch.constant.int 1
    %int2_827 = torch.constant.int 2
    %659 = torch.aten.transpose.int %658#0, %int1_826, %int2_827 : !torch.vtensor<[1,8,1,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,1,8,32],f16>
    %int1_828 = torch.constant.int 1
    %int1_829 = torch.constant.int 1
    %int256_830 = torch.constant.int 256
    %660 = torch.prim.ListConstruct %int1_828, %int1_829, %int256_830 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %661 = torch.aten.view %659, %660 : !torch.vtensor<[1,1,8,32],f16>, !torch.list<int> -> !torch.vtensor<[1,1,256],f16>
    %int5_831 = torch.constant.int 5
    %662 = torch.prims.convert_element_type %29, %int5_831 : !torch.vtensor<[256,256],f32>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int-2_832 = torch.constant.int -2
    %int-1_833 = torch.constant.int -1
    %663 = torch.aten.transpose.int %662, %int-2_832, %int-1_833 : !torch.vtensor<[256,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,256],f16>
    %int1_834 = torch.constant.int 1
    %int256_835 = torch.constant.int 256
    %664 = torch.prim.ListConstruct %int1_834, %int256_835 : (!torch.int, !torch.int) -> !torch.list<int>
    %665 = torch.aten.view %661, %664 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %666 = torch.aten.mm %665, %663 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,256],f16> -> !torch.vtensor<[1,256],f16>
    %int1_836 = torch.constant.int 1
    %int1_837 = torch.constant.int 1
    %int256_838 = torch.constant.int 256
    %667 = torch.prim.ListConstruct %int1_836, %int1_837, %int256_838 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %668 = torch.aten.view %666, %667 : !torch.vtensor<[1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,1,256],f16>
    %int1_839 = torch.constant.int 1
    %669 = torch.aten.add.Tensor %493, %668, %int1_839 : !torch.vtensor<[1,1,256],f16>, !torch.vtensor<[1,1,256],f16>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %int6_840 = torch.constant.int 6
    %670 = torch.prims.convert_element_type %669, %int6_840 : !torch.vtensor<[1,1,256],f16>, !torch.int -> !torch.vtensor<[1,1,256],f32>
    %int2_841 = torch.constant.int 2
    %671 = torch.aten.pow.Tensor_Scalar %670, %int2_841 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f32>
    %int-1_842 = torch.constant.int -1
    %672 = torch.prim.ListConstruct %int-1_842 : (!torch.int) -> !torch.list<int>
    %true_843 = torch.constant.bool true
    %none_844 = torch.constant.none
    %673 = torch.aten.mean.dim %671, %672, %true_843, %none_844 : !torch.vtensor<[1,1,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1],f32>
    %float1.000000e-02_845 = torch.constant.float 1.000000e-02
    %int1_846 = torch.constant.int 1
    %674 = torch.aten.add.Scalar %673, %float1.000000e-02_845, %int1_846 : !torch.vtensor<[1,1,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,1,1],f32>
    %675 = torch.aten.rsqrt %674 : !torch.vtensor<[1,1,1],f32> -> !torch.vtensor<[1,1,1],f32>
    %676 = torch.aten.mul.Tensor %670, %675 : !torch.vtensor<[1,1,256],f32>, !torch.vtensor<[1,1,1],f32> -> !torch.vtensor<[1,1,256],f32>
    %int5_847 = torch.constant.int 5
    %677 = torch.prims.convert_element_type %676, %int5_847 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %678 = torch.aten.mul.Tensor %30, %677 : !torch.vtensor<[256],f32>, !torch.vtensor<[1,1,256],f16> -> !torch.vtensor<[1,1,256],f32>
    %int5_848 = torch.constant.int 5
    %679 = torch.prims.convert_element_type %678, %int5_848 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %int5_849 = torch.constant.int 5
    %680 = torch.prims.convert_element_type %31, %int5_849 : !torch.vtensor<[23,256],f32>, !torch.int -> !torch.vtensor<[23,256],f16>
    %int-2_850 = torch.constant.int -2
    %int-1_851 = torch.constant.int -1
    %681 = torch.aten.transpose.int %680, %int-2_850, %int-1_851 : !torch.vtensor<[23,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,23],f16>
    %int1_852 = torch.constant.int 1
    %int256_853 = torch.constant.int 256
    %682 = torch.prim.ListConstruct %int1_852, %int256_853 : (!torch.int, !torch.int) -> !torch.list<int>
    %683 = torch.aten.view %679, %682 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %684 = torch.aten.mm %683, %681 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,23],f16> -> !torch.vtensor<[1,23],f16>
    %int1_854 = torch.constant.int 1
    %int1_855 = torch.constant.int 1
    %int23_856 = torch.constant.int 23
    %685 = torch.prim.ListConstruct %int1_854, %int1_855, %int23_856 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %686 = torch.aten.view %684, %685 : !torch.vtensor<[1,23],f16>, !torch.list<int> -> !torch.vtensor<[1,1,23],f16>
    %687 = torch.aten.silu %686 : !torch.vtensor<[1,1,23],f16> -> !torch.vtensor<[1,1,23],f16>
    %int5_857 = torch.constant.int 5
    %688 = torch.prims.convert_element_type %32, %int5_857 : !torch.vtensor<[23,256],f32>, !torch.int -> !torch.vtensor<[23,256],f16>
    %int-2_858 = torch.constant.int -2
    %int-1_859 = torch.constant.int -1
    %689 = torch.aten.transpose.int %688, %int-2_858, %int-1_859 : !torch.vtensor<[23,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,23],f16>
    %int1_860 = torch.constant.int 1
    %int256_861 = torch.constant.int 256
    %690 = torch.prim.ListConstruct %int1_860, %int256_861 : (!torch.int, !torch.int) -> !torch.list<int>
    %691 = torch.aten.view %679, %690 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %692 = torch.aten.mm %691, %689 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,23],f16> -> !torch.vtensor<[1,23],f16>
    %int1_862 = torch.constant.int 1
    %int1_863 = torch.constant.int 1
    %int23_864 = torch.constant.int 23
    %693 = torch.prim.ListConstruct %int1_862, %int1_863, %int23_864 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %694 = torch.aten.view %692, %693 : !torch.vtensor<[1,23],f16>, !torch.list<int> -> !torch.vtensor<[1,1,23],f16>
    %695 = torch.aten.mul.Tensor %687, %694 : !torch.vtensor<[1,1,23],f16>, !torch.vtensor<[1,1,23],f16> -> !torch.vtensor<[1,1,23],f16>
    %int5_865 = torch.constant.int 5
    %696 = torch.prims.convert_element_type %33, %int5_865 : !torch.vtensor<[256,23],f32>, !torch.int -> !torch.vtensor<[256,23],f16>
    %int-2_866 = torch.constant.int -2
    %int-1_867 = torch.constant.int -1
    %697 = torch.aten.transpose.int %696, %int-2_866, %int-1_867 : !torch.vtensor<[256,23],f16>, !torch.int, !torch.int -> !torch.vtensor<[23,256],f16>
    %int1_868 = torch.constant.int 1
    %int23_869 = torch.constant.int 23
    %698 = torch.prim.ListConstruct %int1_868, %int23_869 : (!torch.int, !torch.int) -> !torch.list<int>
    %699 = torch.aten.view %695, %698 : !torch.vtensor<[1,1,23],f16>, !torch.list<int> -> !torch.vtensor<[1,23],f16>
    %700 = torch.aten.mm %699, %697 : !torch.vtensor<[1,23],f16>, !torch.vtensor<[23,256],f16> -> !torch.vtensor<[1,256],f16>
    %int1_870 = torch.constant.int 1
    %int1_871 = torch.constant.int 1
    %int256_872 = torch.constant.int 256
    %701 = torch.prim.ListConstruct %int1_870, %int1_871, %int256_872 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %702 = torch.aten.view %700, %701 : !torch.vtensor<[1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,1,256],f16>
    %int1_873 = torch.constant.int 1
    %703 = torch.aten.add.Tensor %669, %702, %int1_873 : !torch.vtensor<[1,1,256],f16>, !torch.vtensor<[1,1,256],f16>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %int6_874 = torch.constant.int 6
    %704 = torch.prims.convert_element_type %703, %int6_874 : !torch.vtensor<[1,1,256],f16>, !torch.int -> !torch.vtensor<[1,1,256],f32>
    %int2_875 = torch.constant.int 2
    %705 = torch.aten.pow.Tensor_Scalar %704, %int2_875 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f32>
    %int-1_876 = torch.constant.int -1
    %706 = torch.prim.ListConstruct %int-1_876 : (!torch.int) -> !torch.list<int>
    %true_877 = torch.constant.bool true
    %none_878 = torch.constant.none
    %707 = torch.aten.mean.dim %705, %706, %true_877, %none_878 : !torch.vtensor<[1,1,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1],f32>
    %float1.000000e-02_879 = torch.constant.float 1.000000e-02
    %int1_880 = torch.constant.int 1
    %708 = torch.aten.add.Scalar %707, %float1.000000e-02_879, %int1_880 : !torch.vtensor<[1,1,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,1,1],f32>
    %709 = torch.aten.rsqrt %708 : !torch.vtensor<[1,1,1],f32> -> !torch.vtensor<[1,1,1],f32>
    %710 = torch.aten.mul.Tensor %704, %709 : !torch.vtensor<[1,1,256],f32>, !torch.vtensor<[1,1,1],f32> -> !torch.vtensor<[1,1,256],f32>
    %int5_881 = torch.constant.int 5
    %711 = torch.prims.convert_element_type %710, %int5_881 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %712 = torch.aten.mul.Tensor %34, %711 : !torch.vtensor<[1,256],f32>, !torch.vtensor<[1,1,256],f16> -> !torch.vtensor<[1,1,256],f32>
    %int5_882 = torch.constant.int 5
    %713 = torch.prims.convert_element_type %712, %int5_882 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %int5_883 = torch.constant.int 5
    %714 = torch.prims.convert_element_type %35, %int5_883 : !torch.vtensor<[256,256],f32>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int-2_884 = torch.constant.int -2
    %int-1_885 = torch.constant.int -1
    %715 = torch.aten.transpose.int %714, %int-2_884, %int-1_885 : !torch.vtensor<[256,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,256],f16>
    %int1_886 = torch.constant.int 1
    %int256_887 = torch.constant.int 256
    %716 = torch.prim.ListConstruct %int1_886, %int256_887 : (!torch.int, !torch.int) -> !torch.list<int>
    %717 = torch.aten.view %713, %716 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %718 = torch.aten.mm %717, %715 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,256],f16> -> !torch.vtensor<[1,256],f16>
    %int1_888 = torch.constant.int 1
    %int1_889 = torch.constant.int 1
    %int256_890 = torch.constant.int 256
    %719 = torch.prim.ListConstruct %int1_888, %int1_889, %int256_890 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %720 = torch.aten.view %718, %719 : !torch.vtensor<[1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,1,256],f16>
    return %720 : !torch.vtensor<[1,1,256],f16>
  }
  util.func private @sharktank_rotary_embedding_1_D_8_32_f32(%arg0: tensor<1x?x8x32xf32>, %arg1: tensor<1x?x32xf32>) -> tensor<1x?x8x32xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %dim = tensor.dim %arg0, %c0 : tensor<1x?x8x32xf32>
    %dim_0 = tensor.dim %arg0, %c1 : tensor<1x?x8x32xf32>
    %dim_1 = tensor.dim %arg0, %c2 : tensor<1x?x8x32xf32>
    %dim_2 = tensor.dim %arg0, %c3 : tensor<1x?x8x32xf32>
    %0 = tensor.empty(%dim, %dim_0, %dim_1, %dim_2) : tensor<?x?x?x?xf32>
    %cast = tensor.cast %0 : tensor<?x?x?x?xf32> to tensor<1x?x8x32xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg1 : tensor<1x?x32xf32>) outs(%cast : tensor<1x?x8x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %2 = linalg.index 0 : index
      %3 = linalg.index 1 : index
      %4 = linalg.index 2 : index
      %5 = linalg.index 3 : index
      %6 = arith.divui %5, %c2 : index
      %7 = arith.remui %5, %c2 : index
      %8 = math.cos %in : f32
      %9 = math.sin %in : f32
      %10 = arith.muli %6, %c2 : index
      %11 = arith.addi %10, %c1 : index
      %extracted = tensor.extract %arg0[%2, %3, %4, %10] : tensor<1x?x8x32xf32>
      %extracted_3 = tensor.extract %arg0[%2, %3, %4, %11] : tensor<1x?x8x32xf32>
      %12 = arith.cmpi eq, %7, %c0 : index
      %13 = arith.mulf %extracted, %8 : f32
      %14 = arith.mulf %extracted_3, %9 : f32
      %15 = arith.subf %13, %14 : f32
      %16 = arith.mulf %extracted_3, %8 : f32
      %17 = arith.mulf %extracted, %9 : f32
      %18 = arith.addf %16, %17 : f32
      %19 = arith.select %12, %15, %18 : f32
      linalg.yield %19 : f32
    } -> tensor<1x?x8x32xf32>
    util.return %1 : tensor<1x?x8x32xf32>
  }
  util.func private @sharktank_rotary_embedding_1_D_4_32_f32(%arg0: tensor<1x?x4x32xf32>, %arg1: tensor<1x?x32xf32>) -> tensor<1x?x4x32xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %dim = tensor.dim %arg0, %c0 : tensor<1x?x4x32xf32>
    %dim_0 = tensor.dim %arg0, %c1 : tensor<1x?x4x32xf32>
    %dim_1 = tensor.dim %arg0, %c2 : tensor<1x?x4x32xf32>
    %dim_2 = tensor.dim %arg0, %c3 : tensor<1x?x4x32xf32>
    %0 = tensor.empty(%dim, %dim_0, %dim_1, %dim_2) : tensor<?x?x?x?xf32>
    %cast = tensor.cast %0 : tensor<?x?x?x?xf32> to tensor<1x?x4x32xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg1 : tensor<1x?x32xf32>) outs(%cast : tensor<1x?x4x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %2 = linalg.index 0 : index
      %3 = linalg.index 1 : index
      %4 = linalg.index 2 : index
      %5 = linalg.index 3 : index
      %6 = arith.divui %5, %c2 : index
      %7 = arith.remui %5, %c2 : index
      %8 = math.cos %in : f32
      %9 = math.sin %in : f32
      %10 = arith.muli %6, %c2 : index
      %11 = arith.addi %10, %c1 : index
      %extracted = tensor.extract %arg0[%2, %3, %4, %10] : tensor<1x?x4x32xf32>
      %extracted_3 = tensor.extract %arg0[%2, %3, %4, %11] : tensor<1x?x4x32xf32>
      %12 = arith.cmpi eq, %7, %c0 : index
      %13 = arith.mulf %extracted, %8 : f32
      %14 = arith.mulf %extracted_3, %9 : f32
      %15 = arith.subf %13, %14 : f32
      %16 = arith.mulf %extracted_3, %8 : f32
      %17 = arith.mulf %extracted, %9 : f32
      %18 = arith.addf %16, %17 : f32
      %19 = arith.select %12, %15, %18 : f32
      linalg.yield %19 : f32
    } -> tensor<1x?x4x32xf32>
    util.return %1 : tensor<1x?x4x32xf32>
  }
  util.func private @sharktank_rotary_embedding_1_1_8_32_f32(%arg0: tensor<1x1x8x32xf32>, %arg1: tensor<1x1x32xf32>) -> tensor<1x1x8x32xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %dim = tensor.dim %arg0, %c0 : tensor<1x1x8x32xf32>
    %dim_0 = tensor.dim %arg0, %c1 : tensor<1x1x8x32xf32>
    %dim_1 = tensor.dim %arg0, %c2 : tensor<1x1x8x32xf32>
    %dim_2 = tensor.dim %arg0, %c3 : tensor<1x1x8x32xf32>
    %0 = tensor.empty(%dim, %dim_0, %dim_1, %dim_2) : tensor<?x?x?x?xf32>
    %cast = tensor.cast %0 : tensor<?x?x?x?xf32> to tensor<1x1x8x32xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg1 : tensor<1x1x32xf32>) outs(%cast : tensor<1x1x8x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %2 = linalg.index 0 : index
      %3 = linalg.index 1 : index
      %4 = linalg.index 2 : index
      %5 = linalg.index 3 : index
      %6 = arith.divui %5, %c2 : index
      %7 = arith.remui %5, %c2 : index
      %8 = math.cos %in : f32
      %9 = math.sin %in : f32
      %10 = arith.muli %6, %c2 : index
      %11 = arith.addi %10, %c1 : index
      %extracted = tensor.extract %arg0[%2, %3, %4, %10] : tensor<1x1x8x32xf32>
      %extracted_3 = tensor.extract %arg0[%2, %3, %4, %11] : tensor<1x1x8x32xf32>
      %12 = arith.cmpi eq, %7, %c0 : index
      %13 = arith.mulf %extracted, %8 : f32
      %14 = arith.mulf %extracted_3, %9 : f32
      %15 = arith.subf %13, %14 : f32
      %16 = arith.mulf %extracted_3, %8 : f32
      %17 = arith.mulf %extracted, %9 : f32
      %18 = arith.addf %16, %17 : f32
      %19 = arith.select %12, %15, %18 : f32
      linalg.yield %19 : f32
    } -> tensor<1x1x8x32xf32>
    util.return %1 : tensor<1x1x8x32xf32>
  }
  util.func private @sharktank_rotary_embedding_1_1_4_32_f32(%arg0: tensor<1x1x4x32xf32>, %arg1: tensor<1x1x32xf32>) -> tensor<1x1x4x32xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %dim = tensor.dim %arg0, %c0 : tensor<1x1x4x32xf32>
    %dim_0 = tensor.dim %arg0, %c1 : tensor<1x1x4x32xf32>
    %dim_1 = tensor.dim %arg0, %c2 : tensor<1x1x4x32xf32>
    %dim_2 = tensor.dim %arg0, %c3 : tensor<1x1x4x32xf32>
    %0 = tensor.empty(%dim, %dim_0, %dim_1, %dim_2) : tensor<?x?x?x?xf32>
    %cast = tensor.cast %0 : tensor<?x?x?x?xf32> to tensor<1x1x4x32xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg1 : tensor<1x1x32xf32>) outs(%cast : tensor<1x1x4x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %2 = linalg.index 0 : index
      %3 = linalg.index 1 : index
      %4 = linalg.index 2 : index
      %5 = linalg.index 3 : index
      %6 = arith.divui %5, %c2 : index
      %7 = arith.remui %5, %c2 : index
      %8 = math.cos %in : f32
      %9 = math.sin %in : f32
      %10 = arith.muli %6, %c2 : index
      %11 = arith.addi %10, %c1 : index
      %extracted = tensor.extract %arg0[%2, %3, %4, %10] : tensor<1x1x4x32xf32>
      %extracted_3 = tensor.extract %arg0[%2, %3, %4, %11] : tensor<1x1x4x32xf32>
      %12 = arith.cmpi eq, %7, %c0 : index
      %13 = arith.mulf %extracted, %8 : f32
      %14 = arith.mulf %extracted_3, %9 : f32
      %15 = arith.subf %13, %14 : f32
      %16 = arith.mulf %extracted_3, %8 : f32
      %17 = arith.mulf %extracted, %9 : f32
      %18 = arith.addf %16, %17 : f32
      %19 = arith.select %12, %15, %18 : f32
      linalg.yield %19 : f32
    } -> tensor<1x1x4x32xf32>
    util.return %1 : tensor<1x1x4x32xf32>
  }
}
