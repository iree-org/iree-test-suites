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
  func.func @prefill_bs4(%arg0: !torch.vtensor<[4,?],si64>, %arg1: !torch.vtensor<[4],si64>, %arg2: !torch.vtensor<[4,?],si64>, %arg3: !torch.tensor<[?,24576],f16>) -> !torch.vtensor<[4,?,256],f16> attributes {torch.assume_strict_symbolic_shapes} {
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
    torch.bind_symbolic_shape %arg0, [%31], affine_map<()[s0] -> (4, s0 * 32)> : !torch.vtensor<[4,?],si64>
    torch.bind_symbolic_shape %arg2, [%31], affine_map<()[s0] -> (4, s0)> : !torch.vtensor<[4,?],si64>
    torch.bind_symbolic_shape %30, [%32], affine_map<()[s0] -> (s0, 24576)> : !torch.vtensor<[?,24576],f16>
    %int5 = torch.constant.int 5
    %33 = torch.prims.convert_element_type %0, %int5 : !torch.vtensor<[256,256],f32>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int-1 = torch.constant.int -1
    %false = torch.constant.bool false
    %false_0 = torch.constant.bool false
    %34 = torch.aten.embedding %33, %arg0, %int-1, %false, %false_0 : !torch.vtensor<[256,256],f16>, !torch.vtensor<[4,?],si64>, !torch.int, !torch.bool, !torch.bool -> !torch.vtensor<[4,?,256],f16>
    torch.bind_symbolic_shape %34, [%31], affine_map<()[s0] -> (4, s0 * 32, 256)> : !torch.vtensor<[4,?,256],f16>
    %int6 = torch.constant.int 6
    %35 = torch.prims.convert_element_type %34, %int6 : !torch.vtensor<[4,?,256],f16>, !torch.int -> !torch.vtensor<[4,?,256],f32>
    torch.bind_symbolic_shape %35, [%31], affine_map<()[s0] -> (4, s0 * 32, 256)> : !torch.vtensor<[4,?,256],f32>
    %int2 = torch.constant.int 2
    %36 = torch.aten.pow.Tensor_Scalar %35, %int2 : !torch.vtensor<[4,?,256],f32>, !torch.int -> !torch.vtensor<[4,?,256],f32>
    torch.bind_symbolic_shape %36, [%31], affine_map<()[s0] -> (4, s0 * 32, 256)> : !torch.vtensor<[4,?,256],f32>
    %int-1_1 = torch.constant.int -1
    %37 = torch.prim.ListConstruct %int-1_1 : (!torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %none = torch.constant.none
    %38 = torch.aten.mean.dim %36, %37, %true, %none : !torch.vtensor<[4,?,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[4,?,1],f32>
    torch.bind_symbolic_shape %38, [%31], affine_map<()[s0] -> (4, s0 * 32, 1)> : !torch.vtensor<[4,?,1],f32>
    %float1.000000e-02 = torch.constant.float 1.000000e-02
    %int1 = torch.constant.int 1
    %39 = torch.aten.add.Scalar %38, %float1.000000e-02, %int1 : !torch.vtensor<[4,?,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[4,?,1],f32>
    torch.bind_symbolic_shape %39, [%31], affine_map<()[s0] -> (4, s0 * 32, 1)> : !torch.vtensor<[4,?,1],f32>
    %40 = torch.aten.rsqrt %39 : !torch.vtensor<[4,?,1],f32> -> !torch.vtensor<[4,?,1],f32>
    torch.bind_symbolic_shape %40, [%31], affine_map<()[s0] -> (4, s0 * 32, 1)> : !torch.vtensor<[4,?,1],f32>
    %41 = torch.aten.mul.Tensor %35, %40 : !torch.vtensor<[4,?,256],f32>, !torch.vtensor<[4,?,1],f32> -> !torch.vtensor<[4,?,256],f32>
    torch.bind_symbolic_shape %41, [%31], affine_map<()[s0] -> (4, s0 * 32, 256)> : !torch.vtensor<[4,?,256],f32>
    %int5_2 = torch.constant.int 5
    %42 = torch.prims.convert_element_type %41, %int5_2 : !torch.vtensor<[4,?,256],f32>, !torch.int -> !torch.vtensor<[4,?,256],f16>
    torch.bind_symbolic_shape %42, [%31], affine_map<()[s0] -> (4, s0 * 32, 256)> : !torch.vtensor<[4,?,256],f16>
    %43 = torch.aten.mul.Tensor %1, %42 : !torch.vtensor<[256],f32>, !torch.vtensor<[4,?,256],f16> -> !torch.vtensor<[4,?,256],f32>
    torch.bind_symbolic_shape %43, [%31], affine_map<()[s0] -> (4, s0 * 32, 256)> : !torch.vtensor<[4,?,256],f32>
    %int5_3 = torch.constant.int 5
    %44 = torch.prims.convert_element_type %43, %int5_3 : !torch.vtensor<[4,?,256],f32>, !torch.int -> !torch.vtensor<[4,?,256],f16>
    torch.bind_symbolic_shape %44, [%31], affine_map<()[s0] -> (4, s0 * 32, 256)> : !torch.vtensor<[4,?,256],f16>
    %int5_4 = torch.constant.int 5
    %45 = torch.prims.convert_element_type %2, %int5_4 : !torch.vtensor<[256,256],f32>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int-2 = torch.constant.int -2
    %int-1_5 = torch.constant.int -1
    %46 = torch.aten.transpose.int %45, %int-2, %int-1_5 : !torch.vtensor<[256,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,256],f16>
    %int1_6 = torch.constant.int 1
    %47 = torch.aten.size.int %arg0, %int1_6 : !torch.vtensor<[4,?],si64>, !torch.int -> !torch.int
    %int4 = torch.constant.int 4
    %48 = torch.aten.mul.int %int4, %47 : !torch.int, !torch.int -> !torch.int
    %int256 = torch.constant.int 256
    %49 = torch.prim.ListConstruct %48, %int256 : (!torch.int, !torch.int) -> !torch.list<int>
    %50 = torch.aten.view %44, %49 : !torch.vtensor<[4,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %50, [%31], affine_map<()[s0] -> (s0 * 128, 256)> : !torch.vtensor<[?,256],f16>
    %51 = torch.aten.mm %50, %46 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,256],f16> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %51, [%31], affine_map<()[s0] -> (s0 * 128, 256)> : !torch.vtensor<[?,256],f16>
    %int4_7 = torch.constant.int 4
    %int256_8 = torch.constant.int 256
    %52 = torch.prim.ListConstruct %int4_7, %47, %int256_8 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %53 = torch.aten.view %51, %52 : !torch.vtensor<[?,256],f16>, !torch.list<int> -> !torch.vtensor<[4,?,256],f16>
    torch.bind_symbolic_shape %53, [%31], affine_map<()[s0] -> (4, s0 * 32, 256)> : !torch.vtensor<[4,?,256],f16>
    %int5_9 = torch.constant.int 5
    %54 = torch.prims.convert_element_type %3, %int5_9 : !torch.vtensor<[128,256],f32>, !torch.int -> !torch.vtensor<[128,256],f16>
    %int-2_10 = torch.constant.int -2
    %int-1_11 = torch.constant.int -1
    %55 = torch.aten.transpose.int %54, %int-2_10, %int-1_11 : !torch.vtensor<[128,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,128],f16>
    %int4_12 = torch.constant.int 4
    %56 = torch.aten.mul.int %int4_12, %47 : !torch.int, !torch.int -> !torch.int
    %int256_13 = torch.constant.int 256
    %57 = torch.prim.ListConstruct %56, %int256_13 : (!torch.int, !torch.int) -> !torch.list<int>
    %58 = torch.aten.view %44, %57 : !torch.vtensor<[4,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %58, [%31], affine_map<()[s0] -> (s0 * 128, 256)> : !torch.vtensor<[?,256],f16>
    %59 = torch.aten.mm %58, %55 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,128],f16> -> !torch.vtensor<[?,128],f16>
    torch.bind_symbolic_shape %59, [%31], affine_map<()[s0] -> (s0 * 128, 128)> : !torch.vtensor<[?,128],f16>
    %int4_14 = torch.constant.int 4
    %int128 = torch.constant.int 128
    %60 = torch.prim.ListConstruct %int4_14, %47, %int128 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %61 = torch.aten.view %59, %60 : !torch.vtensor<[?,128],f16>, !torch.list<int> -> !torch.vtensor<[4,?,128],f16>
    torch.bind_symbolic_shape %61, [%31], affine_map<()[s0] -> (4, s0 * 32, 128)> : !torch.vtensor<[4,?,128],f16>
    %int5_15 = torch.constant.int 5
    %62 = torch.prims.convert_element_type %4, %int5_15 : !torch.vtensor<[128,256],f32>, !torch.int -> !torch.vtensor<[128,256],f16>
    %int-2_16 = torch.constant.int -2
    %int-1_17 = torch.constant.int -1
    %63 = torch.aten.transpose.int %62, %int-2_16, %int-1_17 : !torch.vtensor<[128,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,128],f16>
    %int4_18 = torch.constant.int 4
    %64 = torch.aten.mul.int %int4_18, %47 : !torch.int, !torch.int -> !torch.int
    %int256_19 = torch.constant.int 256
    %65 = torch.prim.ListConstruct %64, %int256_19 : (!torch.int, !torch.int) -> !torch.list<int>
    %66 = torch.aten.view %44, %65 : !torch.vtensor<[4,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %66, [%31], affine_map<()[s0] -> (s0 * 128, 256)> : !torch.vtensor<[?,256],f16>
    %67 = torch.aten.mm %66, %63 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,128],f16> -> !torch.vtensor<[?,128],f16>
    torch.bind_symbolic_shape %67, [%31], affine_map<()[s0] -> (s0 * 128, 128)> : !torch.vtensor<[?,128],f16>
    %int4_20 = torch.constant.int 4
    %int128_21 = torch.constant.int 128
    %68 = torch.prim.ListConstruct %int4_20, %47, %int128_21 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %69 = torch.aten.view %67, %68 : !torch.vtensor<[?,128],f16>, !torch.list<int> -> !torch.vtensor<[4,?,128],f16>
    torch.bind_symbolic_shape %69, [%31], affine_map<()[s0] -> (4, s0 * 32, 128)> : !torch.vtensor<[4,?,128],f16>
    %int4_22 = torch.constant.int 4
    %int8 = torch.constant.int 8
    %int32 = torch.constant.int 32
    %70 = torch.prim.ListConstruct %int4_22, %47, %int8, %int32 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %71 = torch.aten.view %53, %70 : !torch.vtensor<[4,?,256],f16>, !torch.list<int> -> !torch.vtensor<[4,?,8,32],f16>
    torch.bind_symbolic_shape %71, [%31], affine_map<()[s0] -> (4, s0 * 32, 8, 32)> : !torch.vtensor<[4,?,8,32],f16>
    %int4_23 = torch.constant.int 4
    %int4_24 = torch.constant.int 4
    %int32_25 = torch.constant.int 32
    %72 = torch.prim.ListConstruct %int4_23, %47, %int4_24, %int32_25 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %73 = torch.aten.view %61, %72 : !torch.vtensor<[4,?,128],f16>, !torch.list<int> -> !torch.vtensor<[4,?,4,32],f16>
    torch.bind_symbolic_shape %73, [%31], affine_map<()[s0] -> (4, s0 * 32, 4, 32)> : !torch.vtensor<[4,?,4,32],f16>
    %int4_26 = torch.constant.int 4
    %int4_27 = torch.constant.int 4
    %int32_28 = torch.constant.int 32
    %74 = torch.prim.ListConstruct %int4_26, %47, %int4_27, %int32_28 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %75 = torch.aten.view %69, %74 : !torch.vtensor<[4,?,128],f16>, !torch.list<int> -> !torch.vtensor<[4,?,4,32],f16>
    torch.bind_symbolic_shape %75, [%31], affine_map<()[s0] -> (4, s0 * 32, 4, 32)> : !torch.vtensor<[4,?,4,32],f16>
    %int128_29 = torch.constant.int 128
    %none_30 = torch.constant.none
    %none_31 = torch.constant.none
    %cpu = torch.constant.device "cpu"
    %false_32 = torch.constant.bool false
    %76 = torch.aten.arange %int128_29, %none_30, %none_31, %cpu, %false_32 : !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[128],si64>
    %int0 = torch.constant.int 0
    %int32_33 = torch.constant.int 32
    %none_34 = torch.constant.none
    %none_35 = torch.constant.none
    %cpu_36 = torch.constant.device "cpu"
    %false_37 = torch.constant.bool false
    %77 = torch.aten.arange.start %int0, %int32_33, %none_34, %none_35, %cpu_36, %false_37 : !torch.int, !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[32],si64>
    %int2_38 = torch.constant.int 2
    %78 = torch.aten.floor_divide.Scalar %77, %int2_38 : !torch.vtensor<[32],si64>, !torch.int -> !torch.vtensor<[32],si64>
    %int6_39 = torch.constant.int 6
    %79 = torch.prims.convert_element_type %78, %int6_39 : !torch.vtensor<[32],si64>, !torch.int -> !torch.vtensor<[32],f32>
    %int32_40 = torch.constant.int 32
    %80 = torch.aten.div.Scalar %79, %int32_40 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[32],f32>
    %float2.000000e00 = torch.constant.float 2.000000e+00
    %81 = torch.aten.mul.Scalar %80, %float2.000000e00 : !torch.vtensor<[32],f32>, !torch.float -> !torch.vtensor<[32],f32>
    %float5.000000e05 = torch.constant.float 5.000000e+05
    %82 = torch.aten.pow.Scalar %float5.000000e05, %81 : !torch.float, !torch.vtensor<[32],f32> -> !torch.vtensor<[32],f32>
    %83 = torch.aten.reciprocal %82 : !torch.vtensor<[32],f32> -> !torch.vtensor<[32],f32>
    %float1.000000e00 = torch.constant.float 1.000000e+00
    %84 = torch.aten.mul.Scalar %83, %float1.000000e00 : !torch.vtensor<[32],f32>, !torch.float -> !torch.vtensor<[32],f32>
    %int1_41 = torch.constant.int 1
    %85 = torch.aten.unsqueeze %76, %int1_41 : !torch.vtensor<[128],si64>, !torch.int -> !torch.vtensor<[128,1],si64>
    %int0_42 = torch.constant.int 0
    %86 = torch.aten.unsqueeze %84, %int0_42 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[1,32],f32>
    %87 = torch.aten.mul.Tensor %85, %86 : !torch.vtensor<[128,1],si64>, !torch.vtensor<[1,32],f32> -> !torch.vtensor<[128,32],f32>
    %int1_43 = torch.constant.int 1
    %88 = torch.aten.size.int %53, %int1_43 : !torch.vtensor<[4,?,256],f16>, !torch.int -> !torch.int
    %int0_44 = torch.constant.int 0
    %89 = torch.aten.add.int %int0_44, %88 : !torch.int, !torch.int -> !torch.int
    %int0_45 = torch.constant.int 0
    %int0_46 = torch.constant.int 0
    %int1_47 = torch.constant.int 1
    %90 = torch.aten.slice.Tensor %87, %int0_45, %int0_46, %89, %int1_47 : !torch.vtensor<[128,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %90, [%31], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int1_48 = torch.constant.int 1
    %int0_49 = torch.constant.int 0
    %int9223372036854775807 = torch.constant.int 9223372036854775807
    %int1_50 = torch.constant.int 1
    %91 = torch.aten.slice.Tensor %90, %int1_48, %int0_49, %int9223372036854775807, %int1_50 : !torch.vtensor<[?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %91, [%31], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int1_51 = torch.constant.int 1
    %int0_52 = torch.constant.int 0
    %int9223372036854775807_53 = torch.constant.int 9223372036854775807
    %int1_54 = torch.constant.int 1
    %92 = torch.aten.slice.Tensor %91, %int1_51, %int0_52, %int9223372036854775807_53, %int1_54 : !torch.vtensor<[?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %92, [%31], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int0_55 = torch.constant.int 0
    %93 = torch.aten.unsqueeze %92, %int0_55 : !torch.vtensor<[?,32],f32>, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %93, [%31], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int1_56 = torch.constant.int 1
    %int0_57 = torch.constant.int 0
    %int9223372036854775807_58 = torch.constant.int 9223372036854775807
    %int1_59 = torch.constant.int 1
    %94 = torch.aten.slice.Tensor %93, %int1_56, %int0_57, %int9223372036854775807_58, %int1_59 : !torch.vtensor<[1,?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %94, [%31], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int2_60 = torch.constant.int 2
    %int0_61 = torch.constant.int 0
    %int9223372036854775807_62 = torch.constant.int 9223372036854775807
    %int1_63 = torch.constant.int 1
    %95 = torch.aten.slice.Tensor %94, %int2_60, %int0_61, %int9223372036854775807_62, %int1_63 : !torch.vtensor<[1,?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %95, [%31], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int4_64 = torch.constant.int 4
    %int1_65 = torch.constant.int 1
    %int1_66 = torch.constant.int 1
    %96 = torch.prim.ListConstruct %int4_64, %int1_65, %int1_66 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %97 = torch.aten.repeat %95, %96 : !torch.vtensor<[1,?,32],f32>, !torch.list<int> -> !torch.vtensor<[4,?,32],f32>
    torch.bind_symbolic_shape %97, [%31], affine_map<()[s0] -> (4, s0 * 32, 32)> : !torch.vtensor<[4,?,32],f32>
    %int6_67 = torch.constant.int 6
    %98 = torch.prims.convert_element_type %71, %int6_67 : !torch.vtensor<[4,?,8,32],f16>, !torch.int -> !torch.vtensor<[4,?,8,32],f32>
    torch.bind_symbolic_shape %98, [%31], affine_map<()[s0] -> (4, s0 * 32, 8, 32)> : !torch.vtensor<[4,?,8,32],f32>
    %99 = torch_c.to_builtin_tensor %98 : !torch.vtensor<[4,?,8,32],f32> -> tensor<4x?x8x32xf32>
    %100 = torch_c.to_builtin_tensor %97 : !torch.vtensor<[4,?,32],f32> -> tensor<4x?x32xf32>
    %101 = util.call @sharktank_rotary_embedding_4_D_8_32_f32(%99, %100) : (tensor<4x?x8x32xf32>, tensor<4x?x32xf32>) -> tensor<4x?x8x32xf32>
    %102 = torch_c.from_builtin_tensor %101 : tensor<4x?x8x32xf32> -> !torch.vtensor<[4,?,8,32],f32>
    torch.bind_symbolic_shape %102, [%31], affine_map<()[s0] -> (4, s0 * 32, 8, 32)> : !torch.vtensor<[4,?,8,32],f32>
    %int5_68 = torch.constant.int 5
    %103 = torch.prims.convert_element_type %102, %int5_68 : !torch.vtensor<[4,?,8,32],f32>, !torch.int -> !torch.vtensor<[4,?,8,32],f16>
    torch.bind_symbolic_shape %103, [%31], affine_map<()[s0] -> (4, s0 * 32, 8, 32)> : !torch.vtensor<[4,?,8,32],f16>
    %int128_69 = torch.constant.int 128
    %none_70 = torch.constant.none
    %none_71 = torch.constant.none
    %cpu_72 = torch.constant.device "cpu"
    %false_73 = torch.constant.bool false
    %104 = torch.aten.arange %int128_69, %none_70, %none_71, %cpu_72, %false_73 : !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[128],si64>
    %int0_74 = torch.constant.int 0
    %int32_75 = torch.constant.int 32
    %none_76 = torch.constant.none
    %none_77 = torch.constant.none
    %cpu_78 = torch.constant.device "cpu"
    %false_79 = torch.constant.bool false
    %105 = torch.aten.arange.start %int0_74, %int32_75, %none_76, %none_77, %cpu_78, %false_79 : !torch.int, !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[32],si64>
    %int2_80 = torch.constant.int 2
    %106 = torch.aten.floor_divide.Scalar %105, %int2_80 : !torch.vtensor<[32],si64>, !torch.int -> !torch.vtensor<[32],si64>
    %int6_81 = torch.constant.int 6
    %107 = torch.prims.convert_element_type %106, %int6_81 : !torch.vtensor<[32],si64>, !torch.int -> !torch.vtensor<[32],f32>
    %int32_82 = torch.constant.int 32
    %108 = torch.aten.div.Scalar %107, %int32_82 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[32],f32>
    %float2.000000e00_83 = torch.constant.float 2.000000e+00
    %109 = torch.aten.mul.Scalar %108, %float2.000000e00_83 : !torch.vtensor<[32],f32>, !torch.float -> !torch.vtensor<[32],f32>
    %float5.000000e05_84 = torch.constant.float 5.000000e+05
    %110 = torch.aten.pow.Scalar %float5.000000e05_84, %109 : !torch.float, !torch.vtensor<[32],f32> -> !torch.vtensor<[32],f32>
    %111 = torch.aten.reciprocal %110 : !torch.vtensor<[32],f32> -> !torch.vtensor<[32],f32>
    %float1.000000e00_85 = torch.constant.float 1.000000e+00
    %112 = torch.aten.mul.Scalar %111, %float1.000000e00_85 : !torch.vtensor<[32],f32>, !torch.float -> !torch.vtensor<[32],f32>
    %int1_86 = torch.constant.int 1
    %113 = torch.aten.unsqueeze %104, %int1_86 : !torch.vtensor<[128],si64>, !torch.int -> !torch.vtensor<[128,1],si64>
    %int0_87 = torch.constant.int 0
    %114 = torch.aten.unsqueeze %112, %int0_87 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[1,32],f32>
    %115 = torch.aten.mul.Tensor %113, %114 : !torch.vtensor<[128,1],si64>, !torch.vtensor<[1,32],f32> -> !torch.vtensor<[128,32],f32>
    %int1_88 = torch.constant.int 1
    %116 = torch.aten.size.int %61, %int1_88 : !torch.vtensor<[4,?,128],f16>, !torch.int -> !torch.int
    %int0_89 = torch.constant.int 0
    %117 = torch.aten.add.int %int0_89, %116 : !torch.int, !torch.int -> !torch.int
    %int0_90 = torch.constant.int 0
    %int0_91 = torch.constant.int 0
    %int1_92 = torch.constant.int 1
    %118 = torch.aten.slice.Tensor %115, %int0_90, %int0_91, %117, %int1_92 : !torch.vtensor<[128,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %118, [%31], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int1_93 = torch.constant.int 1
    %int0_94 = torch.constant.int 0
    %int9223372036854775807_95 = torch.constant.int 9223372036854775807
    %int1_96 = torch.constant.int 1
    %119 = torch.aten.slice.Tensor %118, %int1_93, %int0_94, %int9223372036854775807_95, %int1_96 : !torch.vtensor<[?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %119, [%31], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int1_97 = torch.constant.int 1
    %int0_98 = torch.constant.int 0
    %int9223372036854775807_99 = torch.constant.int 9223372036854775807
    %int1_100 = torch.constant.int 1
    %120 = torch.aten.slice.Tensor %119, %int1_97, %int0_98, %int9223372036854775807_99, %int1_100 : !torch.vtensor<[?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %120, [%31], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int0_101 = torch.constant.int 0
    %121 = torch.aten.unsqueeze %120, %int0_101 : !torch.vtensor<[?,32],f32>, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %121, [%31], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int1_102 = torch.constant.int 1
    %int0_103 = torch.constant.int 0
    %int9223372036854775807_104 = torch.constant.int 9223372036854775807
    %int1_105 = torch.constant.int 1
    %122 = torch.aten.slice.Tensor %121, %int1_102, %int0_103, %int9223372036854775807_104, %int1_105 : !torch.vtensor<[1,?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %122, [%31], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int2_106 = torch.constant.int 2
    %int0_107 = torch.constant.int 0
    %int9223372036854775807_108 = torch.constant.int 9223372036854775807
    %int1_109 = torch.constant.int 1
    %123 = torch.aten.slice.Tensor %122, %int2_106, %int0_107, %int9223372036854775807_108, %int1_109 : !torch.vtensor<[1,?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %123, [%31], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int4_110 = torch.constant.int 4
    %int1_111 = torch.constant.int 1
    %int1_112 = torch.constant.int 1
    %124 = torch.prim.ListConstruct %int4_110, %int1_111, %int1_112 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %125 = torch.aten.repeat %123, %124 : !torch.vtensor<[1,?,32],f32>, !torch.list<int> -> !torch.vtensor<[4,?,32],f32>
    torch.bind_symbolic_shape %125, [%31], affine_map<()[s0] -> (4, s0 * 32, 32)> : !torch.vtensor<[4,?,32],f32>
    %int6_113 = torch.constant.int 6
    %126 = torch.prims.convert_element_type %73, %int6_113 : !torch.vtensor<[4,?,4,32],f16>, !torch.int -> !torch.vtensor<[4,?,4,32],f32>
    torch.bind_symbolic_shape %126, [%31], affine_map<()[s0] -> (4, s0 * 32, 4, 32)> : !torch.vtensor<[4,?,4,32],f32>
    %127 = torch_c.to_builtin_tensor %126 : !torch.vtensor<[4,?,4,32],f32> -> tensor<4x?x4x32xf32>
    %128 = torch_c.to_builtin_tensor %125 : !torch.vtensor<[4,?,32],f32> -> tensor<4x?x32xf32>
    %129 = util.call @sharktank_rotary_embedding_4_D_4_32_f32(%127, %128) : (tensor<4x?x4x32xf32>, tensor<4x?x32xf32>) -> tensor<4x?x4x32xf32>
    %130 = torch_c.from_builtin_tensor %129 : tensor<4x?x4x32xf32> -> !torch.vtensor<[4,?,4,32],f32>
    torch.bind_symbolic_shape %130, [%31], affine_map<()[s0] -> (4, s0 * 32, 4, 32)> : !torch.vtensor<[4,?,4,32],f32>
    %int5_114 = torch.constant.int 5
    %131 = torch.prims.convert_element_type %130, %int5_114 : !torch.vtensor<[4,?,4,32],f32>, !torch.int -> !torch.vtensor<[4,?,4,32],f16>
    torch.bind_symbolic_shape %131, [%31], affine_map<()[s0] -> (4, s0 * 32, 4, 32)> : !torch.vtensor<[4,?,4,32],f16>
    %int0_115 = torch.constant.int 0
    %132 = torch.aten.size.int %30, %int0_115 : !torch.vtensor<[?,24576],f16>, !torch.int -> !torch.int
    %int3 = torch.constant.int 3
    %int2_116 = torch.constant.int 2
    %int32_117 = torch.constant.int 32
    %int4_118 = torch.constant.int 4
    %int32_119 = torch.constant.int 32
    %133 = torch.prim.ListConstruct %132, %int3, %int2_116, %int32_117, %int4_118, %int32_119 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %134 = torch.aten.view %30, %133 : !torch.vtensor<[?,24576],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,4,32],f16>
    torch.bind_symbolic_shape %134, [%32], affine_map<()[s0] -> (s0, 3, 2, 32, 4, 32)> : !torch.vtensor<[?,3,2,32,4,32],f16>
    %int3_120 = torch.constant.int 3
    %135 = torch.aten.mul.int %132, %int3_120 : !torch.int, !torch.int -> !torch.int
    %int2_121 = torch.constant.int 2
    %136 = torch.aten.mul.int %135, %int2_121 : !torch.int, !torch.int -> !torch.int
    %int32_122 = torch.constant.int 32
    %int4_123 = torch.constant.int 4
    %int32_124 = torch.constant.int 32
    %137 = torch.prim.ListConstruct %136, %int32_122, %int4_123, %int32_124 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %138 = torch.aten.view %134, %137 : !torch.vtensor<[?,3,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,32,4,32],f16>
    torch.bind_symbolic_shape %138, [%32], affine_map<()[s0] -> (s0 * 6, 32, 4, 32)> : !torch.vtensor<[?,32,4,32],f16>
    %int6_125 = torch.constant.int 6
    %139 = torch.aten.mul.Scalar %arg2, %int6_125 : !torch.vtensor<[4,?],si64>, !torch.int -> !torch.vtensor<[4,?],si64>
    torch.bind_symbolic_shape %139, [%31], affine_map<()[s0] -> (4, s0)> : !torch.vtensor<[4,?],si64>
    %int0_126 = torch.constant.int 0
    %int1_127 = torch.constant.int 1
    %140 = torch.aten.add.Scalar %139, %int0_126, %int1_127 : !torch.vtensor<[4,?],si64>, !torch.int, !torch.int -> !torch.vtensor<[4,?],si64>
    torch.bind_symbolic_shape %140, [%31], affine_map<()[s0] -> (4, s0)> : !torch.vtensor<[4,?],si64>
    %int1_128 = torch.constant.int 1
    %141 = torch.aten.size.int %arg2, %int1_128 : !torch.vtensor<[4,?],si64>, !torch.int -> !torch.int
    %int4_129 = torch.constant.int 4
    %int32_130 = torch.constant.int 32
    %int4_131 = torch.constant.int 4
    %int32_132 = torch.constant.int 32
    %142 = torch.prim.ListConstruct %int4_129, %141, %int32_130, %int4_131, %int32_132 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %143 = torch.aten.view %131, %142 : !torch.vtensor<[4,?,4,32],f16>, !torch.list<int> -> !torch.vtensor<[4,?,32,4,32],f16>
    torch.bind_symbolic_shape %143, [%31], affine_map<()[s0] -> (4, s0, 32, 4, 32)> : !torch.vtensor<[4,?,32,4,32],f16>
    %int4_133 = torch.constant.int 4
    %144 = torch.aten.mul.int %int4_133, %141 : !torch.int, !torch.int -> !torch.int
    %int32_134 = torch.constant.int 32
    %int4_135 = torch.constant.int 4
    %int32_136 = torch.constant.int 32
    %145 = torch.prim.ListConstruct %144, %int32_134, %int4_135, %int32_136 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %146 = torch.aten.view %143, %145 : !torch.vtensor<[4,?,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,32,4,32],f16>
    torch.bind_symbolic_shape %146, [%31], affine_map<()[s0] -> (s0 * 4, 32, 4, 32)> : !torch.vtensor<[?,32,4,32],f16>
    %int4_137 = torch.constant.int 4
    %147 = torch.aten.mul.int %int4_137, %141 : !torch.int, !torch.int -> !torch.int
    %148 = torch.prim.ListConstruct %147 : (!torch.int) -> !torch.list<int>
    %149 = torch.aten.view %140, %148 : !torch.vtensor<[4,?],si64>, !torch.list<int> -> !torch.vtensor<[?],si64>
    torch.bind_symbolic_shape %149, [%31], affine_map<()[s0] -> (s0 * 4)> : !torch.vtensor<[?],si64>
    %150 = torch.prim.ListConstruct %149 : (!torch.vtensor<[?],si64>) -> !torch.list<optional<vtensor>>
    %false_138 = torch.constant.bool false
    %151 = torch.aten.index_put %138, %150, %146, %false_138 : !torch.vtensor<[?,32,4,32],f16>, !torch.list<optional<vtensor>>, !torch.vtensor<[?,32,4,32],f16>, !torch.bool -> !torch.vtensor<[?,32,4,32],f16>
    torch.bind_symbolic_shape %151, [%32], affine_map<()[s0] -> (s0 * 6, 32, 4, 32)> : !torch.vtensor<[?,32,4,32],f16>
    %int3_139 = torch.constant.int 3
    %int2_140 = torch.constant.int 2
    %int32_141 = torch.constant.int 32
    %int4_142 = torch.constant.int 4
    %int32_143 = torch.constant.int 32
    %152 = torch.prim.ListConstruct %132, %int3_139, %int2_140, %int32_141, %int4_142, %int32_143 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %153 = torch.aten.view %151, %152 : !torch.vtensor<[?,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,4,32],f16>
    torch.bind_symbolic_shape %153, [%32], affine_map<()[s0] -> (s0, 3, 2, 32, 4, 32)> : !torch.vtensor<[?,3,2,32,4,32],f16>
    %int24576 = torch.constant.int 24576
    %154 = torch.prim.ListConstruct %132, %int24576 : (!torch.int, !torch.int) -> !torch.list<int>
    %155 = torch.aten.view %153, %154 : !torch.vtensor<[?,3,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,24576],f16>
    torch.bind_symbolic_shape %155, [%32], affine_map<()[s0] -> (s0, 24576)> : !torch.vtensor<[?,24576],f16>
    %int3_144 = torch.constant.int 3
    %int2_145 = torch.constant.int 2
    %int32_146 = torch.constant.int 32
    %int4_147 = torch.constant.int 4
    %int32_148 = torch.constant.int 32
    %156 = torch.prim.ListConstruct %132, %int3_144, %int2_145, %int32_146, %int4_147, %int32_148 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %157 = torch.aten.view %155, %156 : !torch.vtensor<[?,24576],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,4,32],f16>
    torch.bind_symbolic_shape %157, [%32], affine_map<()[s0] -> (s0, 3, 2, 32, 4, 32)> : !torch.vtensor<[?,3,2,32,4,32],f16>
    %int32_149 = torch.constant.int 32
    %int4_150 = torch.constant.int 4
    %int32_151 = torch.constant.int 32
    %158 = torch.prim.ListConstruct %136, %int32_149, %int4_150, %int32_151 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %159 = torch.aten.view %157, %158 : !torch.vtensor<[?,3,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,32,4,32],f16>
    torch.bind_symbolic_shape %159, [%32], affine_map<()[s0] -> (s0 * 6, 32, 4, 32)> : !torch.vtensor<[?,32,4,32],f16>
    %int4_152 = torch.constant.int 4
    %int32_153 = torch.constant.int 32
    %int4_154 = torch.constant.int 4
    %int32_155 = torch.constant.int 32
    %160 = torch.prim.ListConstruct %int4_152, %141, %int32_153, %int4_154, %int32_155 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %161 = torch.aten.view %75, %160 : !torch.vtensor<[4,?,4,32],f16>, !torch.list<int> -> !torch.vtensor<[4,?,32,4,32],f16>
    torch.bind_symbolic_shape %161, [%31], affine_map<()[s0] -> (4, s0, 32, 4, 32)> : !torch.vtensor<[4,?,32,4,32],f16>
    %int4_156 = torch.constant.int 4
    %162 = torch.aten.mul.int %int4_156, %141 : !torch.int, !torch.int -> !torch.int
    %int32_157 = torch.constant.int 32
    %int4_158 = torch.constant.int 4
    %int32_159 = torch.constant.int 32
    %163 = torch.prim.ListConstruct %162, %int32_157, %int4_158, %int32_159 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %164 = torch.aten.view %161, %163 : !torch.vtensor<[4,?,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,32,4,32],f16>
    torch.bind_symbolic_shape %164, [%31], affine_map<()[s0] -> (s0 * 4, 32, 4, 32)> : !torch.vtensor<[?,32,4,32],f16>
    %int1_160 = torch.constant.int 1
    %int1_161 = torch.constant.int 1
    %165 = torch.aten.add.Scalar %140, %int1_160, %int1_161 : !torch.vtensor<[4,?],si64>, !torch.int, !torch.int -> !torch.vtensor<[4,?],si64>
    torch.bind_symbolic_shape %165, [%31], affine_map<()[s0] -> (4, s0)> : !torch.vtensor<[4,?],si64>
    %int4_162 = torch.constant.int 4
    %166 = torch.aten.mul.int %int4_162, %141 : !torch.int, !torch.int -> !torch.int
    %167 = torch.prim.ListConstruct %166 : (!torch.int) -> !torch.list<int>
    %168 = torch.aten.view %165, %167 : !torch.vtensor<[4,?],si64>, !torch.list<int> -> !torch.vtensor<[?],si64>
    torch.bind_symbolic_shape %168, [%31], affine_map<()[s0] -> (s0 * 4)> : !torch.vtensor<[?],si64>
    %169 = torch.prim.ListConstruct %168 : (!torch.vtensor<[?],si64>) -> !torch.list<optional<vtensor>>
    %false_163 = torch.constant.bool false
    %170 = torch.aten.index_put %159, %169, %164, %false_163 : !torch.vtensor<[?,32,4,32],f16>, !torch.list<optional<vtensor>>, !torch.vtensor<[?,32,4,32],f16>, !torch.bool -> !torch.vtensor<[?,32,4,32],f16>
    torch.bind_symbolic_shape %170, [%32], affine_map<()[s0] -> (s0 * 6, 32, 4, 32)> : !torch.vtensor<[?,32,4,32],f16>
    %int3_164 = torch.constant.int 3
    %int2_165 = torch.constant.int 2
    %int32_166 = torch.constant.int 32
    %int4_167 = torch.constant.int 4
    %int32_168 = torch.constant.int 32
    %171 = torch.prim.ListConstruct %132, %int3_164, %int2_165, %int32_166, %int4_167, %int32_168 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %172 = torch.aten.view %170, %171 : !torch.vtensor<[?,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,4,32],f16>
    torch.bind_symbolic_shape %172, [%32], affine_map<()[s0] -> (s0, 3, 2, 32, 4, 32)> : !torch.vtensor<[?,3,2,32,4,32],f16>
    %int24576_169 = torch.constant.int 24576
    %173 = torch.prim.ListConstruct %132, %int24576_169 : (!torch.int, !torch.int) -> !torch.list<int>
    %174 = torch.aten.view %172, %173 : !torch.vtensor<[?,3,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,24576],f16>
    torch.bind_symbolic_shape %174, [%32], affine_map<()[s0] -> (s0, 24576)> : !torch.vtensor<[?,24576],f16>
    %int-2_170 = torch.constant.int -2
    %175 = torch.aten.unsqueeze %131, %int-2_170 : !torch.vtensor<[4,?,4,32],f16>, !torch.int -> !torch.vtensor<[4,?,4,1,32],f16>
    torch.bind_symbolic_shape %175, [%31], affine_map<()[s0] -> (4, s0 * 32, 4, 1, 32)> : !torch.vtensor<[4,?,4,1,32],f16>
    %int4_171 = torch.constant.int 4
    %int4_172 = torch.constant.int 4
    %int2_173 = torch.constant.int 2
    %int32_174 = torch.constant.int 32
    %176 = torch.prim.ListConstruct %int4_171, %116, %int4_172, %int2_173, %int32_174 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_175 = torch.constant.bool false
    %177 = torch.aten.expand %175, %176, %false_175 : !torch.vtensor<[4,?,4,1,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[4,?,4,2,32],f16>
    torch.bind_symbolic_shape %177, [%31], affine_map<()[s0] -> (4, s0 * 32, 4, 2, 32)> : !torch.vtensor<[4,?,4,2,32],f16>
    %int0_176 = torch.constant.int 0
    %178 = torch.aten.clone %177, %int0_176 : !torch.vtensor<[4,?,4,2,32],f16>, !torch.int -> !torch.vtensor<[4,?,4,2,32],f16>
    torch.bind_symbolic_shape %178, [%31], affine_map<()[s0] -> (4, s0 * 32, 4, 2, 32)> : !torch.vtensor<[4,?,4,2,32],f16>
    %int4_177 = torch.constant.int 4
    %int8_178 = torch.constant.int 8
    %int32_179 = torch.constant.int 32
    %179 = torch.prim.ListConstruct %int4_177, %116, %int8_178, %int32_179 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %180 = torch.aten._unsafe_view %178, %179 : !torch.vtensor<[4,?,4,2,32],f16>, !torch.list<int> -> !torch.vtensor<[4,?,8,32],f16>
    torch.bind_symbolic_shape %180, [%31], affine_map<()[s0] -> (4, s0 * 32, 8, 32)> : !torch.vtensor<[4,?,8,32],f16>
    %int-2_180 = torch.constant.int -2
    %181 = torch.aten.unsqueeze %75, %int-2_180 : !torch.vtensor<[4,?,4,32],f16>, !torch.int -> !torch.vtensor<[4,?,4,1,32],f16>
    torch.bind_symbolic_shape %181, [%31], affine_map<()[s0] -> (4, s0 * 32, 4, 1, 32)> : !torch.vtensor<[4,?,4,1,32],f16>
    %int1_181 = torch.constant.int 1
    %182 = torch.aten.size.int %69, %int1_181 : !torch.vtensor<[4,?,128],f16>, !torch.int -> !torch.int
    %int4_182 = torch.constant.int 4
    %int4_183 = torch.constant.int 4
    %int2_184 = torch.constant.int 2
    %int32_185 = torch.constant.int 32
    %183 = torch.prim.ListConstruct %int4_182, %182, %int4_183, %int2_184, %int32_185 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_186 = torch.constant.bool false
    %184 = torch.aten.expand %181, %183, %false_186 : !torch.vtensor<[4,?,4,1,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[4,?,4,2,32],f16>
    torch.bind_symbolic_shape %184, [%31], affine_map<()[s0] -> (4, s0 * 32, 4, 2, 32)> : !torch.vtensor<[4,?,4,2,32],f16>
    %int0_187 = torch.constant.int 0
    %185 = torch.aten.clone %184, %int0_187 : !torch.vtensor<[4,?,4,2,32],f16>, !torch.int -> !torch.vtensor<[4,?,4,2,32],f16>
    torch.bind_symbolic_shape %185, [%31], affine_map<()[s0] -> (4, s0 * 32, 4, 2, 32)> : !torch.vtensor<[4,?,4,2,32],f16>
    %int4_188 = torch.constant.int 4
    %int8_189 = torch.constant.int 8
    %int32_190 = torch.constant.int 32
    %186 = torch.prim.ListConstruct %int4_188, %182, %int8_189, %int32_190 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %187 = torch.aten._unsafe_view %185, %186 : !torch.vtensor<[4,?,4,2,32],f16>, !torch.list<int> -> !torch.vtensor<[4,?,8,32],f16>
    torch.bind_symbolic_shape %187, [%31], affine_map<()[s0] -> (4, s0 * 32, 8, 32)> : !torch.vtensor<[4,?,8,32],f16>
    %int1_191 = torch.constant.int 1
    %int2_192 = torch.constant.int 2
    %188 = torch.aten.transpose.int %103, %int1_191, %int2_192 : !torch.vtensor<[4,?,8,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[4,8,?,32],f16>
    torch.bind_symbolic_shape %188, [%31], affine_map<()[s0] -> (4, 8, s0 * 32, 32)> : !torch.vtensor<[4,8,?,32],f16>
    %int1_193 = torch.constant.int 1
    %int2_194 = torch.constant.int 2
    %189 = torch.aten.transpose.int %180, %int1_193, %int2_194 : !torch.vtensor<[4,?,8,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[4,8,?,32],f16>
    torch.bind_symbolic_shape %189, [%31], affine_map<()[s0] -> (4, 8, s0 * 32, 32)> : !torch.vtensor<[4,8,?,32],f16>
    %int1_195 = torch.constant.int 1
    %int2_196 = torch.constant.int 2
    %190 = torch.aten.transpose.int %187, %int1_195, %int2_196 : !torch.vtensor<[4,?,8,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[4,8,?,32],f16>
    torch.bind_symbolic_shape %190, [%31], affine_map<()[s0] -> (4, 8, s0 * 32, 32)> : !torch.vtensor<[4,8,?,32],f16>
    %float0.000000e00 = torch.constant.float 0.000000e+00
    %true_197 = torch.constant.bool true
    %none_198 = torch.constant.none
    %none_199 = torch.constant.none
    %191:2 = torch.operator "torch.aten._scaled_dot_product_flash_attention_for_cpu"(%188, %189, %190, %float0.000000e00, %true_197, %none_198, %none_199) : (!torch.vtensor<[4,8,?,32],f16>, !torch.vtensor<[4,8,?,32],f16>, !torch.vtensor<[4,8,?,32],f16>, !torch.float, !torch.bool, !torch.none, !torch.none) -> (!torch.vtensor<[4,8,?,32],f16>, !torch.vtensor<[4,8,?],f32>) 
    torch.bind_symbolic_shape %191#0, [%31], affine_map<()[s0] -> (4, 8, s0 * 32, 32)> : !torch.vtensor<[4,8,?,32],f16>
    %int1_200 = torch.constant.int 1
    %int2_201 = torch.constant.int 2
    %192 = torch.aten.transpose.int %191#0, %int1_200, %int2_201 : !torch.vtensor<[4,8,?,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[4,?,8,32],f16>
    torch.bind_symbolic_shape %192, [%31], affine_map<()[s0] -> (4, s0 * 32, 8, 32)> : !torch.vtensor<[4,?,8,32],f16>
    %int4_202 = torch.constant.int 4
    %int256_203 = torch.constant.int 256
    %193 = torch.prim.ListConstruct %int4_202, %88, %int256_203 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %194 = torch.aten.view %192, %193 : !torch.vtensor<[4,?,8,32],f16>, !torch.list<int> -> !torch.vtensor<[4,?,256],f16>
    torch.bind_symbolic_shape %194, [%31], affine_map<()[s0] -> (4, s0 * 32, 256)> : !torch.vtensor<[4,?,256],f16>
    %int5_204 = torch.constant.int 5
    %195 = torch.prims.convert_element_type %5, %int5_204 : !torch.vtensor<[256,256],f32>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int-2_205 = torch.constant.int -2
    %int-1_206 = torch.constant.int -1
    %196 = torch.aten.transpose.int %195, %int-2_205, %int-1_206 : !torch.vtensor<[256,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,256],f16>
    %int4_207 = torch.constant.int 4
    %197 = torch.aten.mul.int %int4_207, %88 : !torch.int, !torch.int -> !torch.int
    %int256_208 = torch.constant.int 256
    %198 = torch.prim.ListConstruct %197, %int256_208 : (!torch.int, !torch.int) -> !torch.list<int>
    %199 = torch.aten.view %194, %198 : !torch.vtensor<[4,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %199, [%31], affine_map<()[s0] -> (s0 * 128, 256)> : !torch.vtensor<[?,256],f16>
    %200 = torch.aten.mm %199, %196 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,256],f16> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %200, [%31], affine_map<()[s0] -> (s0 * 128, 256)> : !torch.vtensor<[?,256],f16>
    %int4_209 = torch.constant.int 4
    %int256_210 = torch.constant.int 256
    %201 = torch.prim.ListConstruct %int4_209, %88, %int256_210 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %202 = torch.aten.view %200, %201 : !torch.vtensor<[?,256],f16>, !torch.list<int> -> !torch.vtensor<[4,?,256],f16>
    torch.bind_symbolic_shape %202, [%31], affine_map<()[s0] -> (4, s0 * 32, 256)> : !torch.vtensor<[4,?,256],f16>
    %int1_211 = torch.constant.int 1
    %203 = torch.aten.add.Tensor %34, %202, %int1_211 : !torch.vtensor<[4,?,256],f16>, !torch.vtensor<[4,?,256],f16>, !torch.int -> !torch.vtensor<[4,?,256],f16>
    torch.bind_symbolic_shape %203, [%31], affine_map<()[s0] -> (4, s0 * 32, 256)> : !torch.vtensor<[4,?,256],f16>
    %int6_212 = torch.constant.int 6
    %204 = torch.prims.convert_element_type %203, %int6_212 : !torch.vtensor<[4,?,256],f16>, !torch.int -> !torch.vtensor<[4,?,256],f32>
    torch.bind_symbolic_shape %204, [%31], affine_map<()[s0] -> (4, s0 * 32, 256)> : !torch.vtensor<[4,?,256],f32>
    %int2_213 = torch.constant.int 2
    %205 = torch.aten.pow.Tensor_Scalar %204, %int2_213 : !torch.vtensor<[4,?,256],f32>, !torch.int -> !torch.vtensor<[4,?,256],f32>
    torch.bind_symbolic_shape %205, [%31], affine_map<()[s0] -> (4, s0 * 32, 256)> : !torch.vtensor<[4,?,256],f32>
    %int-1_214 = torch.constant.int -1
    %206 = torch.prim.ListConstruct %int-1_214 : (!torch.int) -> !torch.list<int>
    %true_215 = torch.constant.bool true
    %none_216 = torch.constant.none
    %207 = torch.aten.mean.dim %205, %206, %true_215, %none_216 : !torch.vtensor<[4,?,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[4,?,1],f32>
    torch.bind_symbolic_shape %207, [%31], affine_map<()[s0] -> (4, s0 * 32, 1)> : !torch.vtensor<[4,?,1],f32>
    %float1.000000e-02_217 = torch.constant.float 1.000000e-02
    %int1_218 = torch.constant.int 1
    %208 = torch.aten.add.Scalar %207, %float1.000000e-02_217, %int1_218 : !torch.vtensor<[4,?,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[4,?,1],f32>
    torch.bind_symbolic_shape %208, [%31], affine_map<()[s0] -> (4, s0 * 32, 1)> : !torch.vtensor<[4,?,1],f32>
    %209 = torch.aten.rsqrt %208 : !torch.vtensor<[4,?,1],f32> -> !torch.vtensor<[4,?,1],f32>
    torch.bind_symbolic_shape %209, [%31], affine_map<()[s0] -> (4, s0 * 32, 1)> : !torch.vtensor<[4,?,1],f32>
    %210 = torch.aten.mul.Tensor %204, %209 : !torch.vtensor<[4,?,256],f32>, !torch.vtensor<[4,?,1],f32> -> !torch.vtensor<[4,?,256],f32>
    torch.bind_symbolic_shape %210, [%31], affine_map<()[s0] -> (4, s0 * 32, 256)> : !torch.vtensor<[4,?,256],f32>
    %int5_219 = torch.constant.int 5
    %211 = torch.prims.convert_element_type %210, %int5_219 : !torch.vtensor<[4,?,256],f32>, !torch.int -> !torch.vtensor<[4,?,256],f16>
    torch.bind_symbolic_shape %211, [%31], affine_map<()[s0] -> (4, s0 * 32, 256)> : !torch.vtensor<[4,?,256],f16>
    %212 = torch.aten.mul.Tensor %6, %211 : !torch.vtensor<[256],f32>, !torch.vtensor<[4,?,256],f16> -> !torch.vtensor<[4,?,256],f32>
    torch.bind_symbolic_shape %212, [%31], affine_map<()[s0] -> (4, s0 * 32, 256)> : !torch.vtensor<[4,?,256],f32>
    %int5_220 = torch.constant.int 5
    %213 = torch.prims.convert_element_type %212, %int5_220 : !torch.vtensor<[4,?,256],f32>, !torch.int -> !torch.vtensor<[4,?,256],f16>
    torch.bind_symbolic_shape %213, [%31], affine_map<()[s0] -> (4, s0 * 32, 256)> : !torch.vtensor<[4,?,256],f16>
    %int5_221 = torch.constant.int 5
    %214 = torch.prims.convert_element_type %7, %int5_221 : !torch.vtensor<[23,256],f32>, !torch.int -> !torch.vtensor<[23,256],f16>
    %int-2_222 = torch.constant.int -2
    %int-1_223 = torch.constant.int -1
    %215 = torch.aten.transpose.int %214, %int-2_222, %int-1_223 : !torch.vtensor<[23,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,23],f16>
    %int4_224 = torch.constant.int 4
    %216 = torch.aten.mul.int %int4_224, %47 : !torch.int, !torch.int -> !torch.int
    %int256_225 = torch.constant.int 256
    %217 = torch.prim.ListConstruct %216, %int256_225 : (!torch.int, !torch.int) -> !torch.list<int>
    %218 = torch.aten.view %213, %217 : !torch.vtensor<[4,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %218, [%31], affine_map<()[s0] -> (s0 * 128, 256)> : !torch.vtensor<[?,256],f16>
    %219 = torch.aten.mm %218, %215 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,23],f16> -> !torch.vtensor<[?,23],f16>
    torch.bind_symbolic_shape %219, [%31], affine_map<()[s0] -> (s0 * 128, 23)> : !torch.vtensor<[?,23],f16>
    %int4_226 = torch.constant.int 4
    %int23 = torch.constant.int 23
    %220 = torch.prim.ListConstruct %int4_226, %47, %int23 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %221 = torch.aten.view %219, %220 : !torch.vtensor<[?,23],f16>, !torch.list<int> -> !torch.vtensor<[4,?,23],f16>
    torch.bind_symbolic_shape %221, [%31], affine_map<()[s0] -> (4, s0 * 32, 23)> : !torch.vtensor<[4,?,23],f16>
    %222 = torch.aten.silu %221 : !torch.vtensor<[4,?,23],f16> -> !torch.vtensor<[4,?,23],f16>
    torch.bind_symbolic_shape %222, [%31], affine_map<()[s0] -> (4, s0 * 32, 23)> : !torch.vtensor<[4,?,23],f16>
    %int5_227 = torch.constant.int 5
    %223 = torch.prims.convert_element_type %8, %int5_227 : !torch.vtensor<[23,256],f32>, !torch.int -> !torch.vtensor<[23,256],f16>
    %int-2_228 = torch.constant.int -2
    %int-1_229 = torch.constant.int -1
    %224 = torch.aten.transpose.int %223, %int-2_228, %int-1_229 : !torch.vtensor<[23,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,23],f16>
    %int4_230 = torch.constant.int 4
    %225 = torch.aten.mul.int %int4_230, %47 : !torch.int, !torch.int -> !torch.int
    %int256_231 = torch.constant.int 256
    %226 = torch.prim.ListConstruct %225, %int256_231 : (!torch.int, !torch.int) -> !torch.list<int>
    %227 = torch.aten.view %213, %226 : !torch.vtensor<[4,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %227, [%31], affine_map<()[s0] -> (s0 * 128, 256)> : !torch.vtensor<[?,256],f16>
    %228 = torch.aten.mm %227, %224 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,23],f16> -> !torch.vtensor<[?,23],f16>
    torch.bind_symbolic_shape %228, [%31], affine_map<()[s0] -> (s0 * 128, 23)> : !torch.vtensor<[?,23],f16>
    %int4_232 = torch.constant.int 4
    %int23_233 = torch.constant.int 23
    %229 = torch.prim.ListConstruct %int4_232, %47, %int23_233 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %230 = torch.aten.view %228, %229 : !torch.vtensor<[?,23],f16>, !torch.list<int> -> !torch.vtensor<[4,?,23],f16>
    torch.bind_symbolic_shape %230, [%31], affine_map<()[s0] -> (4, s0 * 32, 23)> : !torch.vtensor<[4,?,23],f16>
    %231 = torch.aten.mul.Tensor %222, %230 : !torch.vtensor<[4,?,23],f16>, !torch.vtensor<[4,?,23],f16> -> !torch.vtensor<[4,?,23],f16>
    torch.bind_symbolic_shape %231, [%31], affine_map<()[s0] -> (4, s0 * 32, 23)> : !torch.vtensor<[4,?,23],f16>
    %int5_234 = torch.constant.int 5
    %232 = torch.prims.convert_element_type %9, %int5_234 : !torch.vtensor<[256,23],f32>, !torch.int -> !torch.vtensor<[256,23],f16>
    %int-2_235 = torch.constant.int -2
    %int-1_236 = torch.constant.int -1
    %233 = torch.aten.transpose.int %232, %int-2_235, %int-1_236 : !torch.vtensor<[256,23],f16>, !torch.int, !torch.int -> !torch.vtensor<[23,256],f16>
    %int1_237 = torch.constant.int 1
    %234 = torch.aten.size.int %221, %int1_237 : !torch.vtensor<[4,?,23],f16>, !torch.int -> !torch.int
    %int4_238 = torch.constant.int 4
    %235 = torch.aten.mul.int %int4_238, %234 : !torch.int, !torch.int -> !torch.int
    %int23_239 = torch.constant.int 23
    %236 = torch.prim.ListConstruct %235, %int23_239 : (!torch.int, !torch.int) -> !torch.list<int>
    %237 = torch.aten.view %231, %236 : !torch.vtensor<[4,?,23],f16>, !torch.list<int> -> !torch.vtensor<[?,23],f16>
    torch.bind_symbolic_shape %237, [%31], affine_map<()[s0] -> (s0 * 128, 23)> : !torch.vtensor<[?,23],f16>
    %238 = torch.aten.mm %237, %233 : !torch.vtensor<[?,23],f16>, !torch.vtensor<[23,256],f16> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %238, [%31], affine_map<()[s0] -> (s0 * 128, 256)> : !torch.vtensor<[?,256],f16>
    %int4_240 = torch.constant.int 4
    %int256_241 = torch.constant.int 256
    %239 = torch.prim.ListConstruct %int4_240, %234, %int256_241 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %240 = torch.aten.view %238, %239 : !torch.vtensor<[?,256],f16>, !torch.list<int> -> !torch.vtensor<[4,?,256],f16>
    torch.bind_symbolic_shape %240, [%31], affine_map<()[s0] -> (4, s0 * 32, 256)> : !torch.vtensor<[4,?,256],f16>
    %int1_242 = torch.constant.int 1
    %241 = torch.aten.add.Tensor %203, %240, %int1_242 : !torch.vtensor<[4,?,256],f16>, !torch.vtensor<[4,?,256],f16>, !torch.int -> !torch.vtensor<[4,?,256],f16>
    torch.bind_symbolic_shape %241, [%31], affine_map<()[s0] -> (4, s0 * 32, 256)> : !torch.vtensor<[4,?,256],f16>
    %int6_243 = torch.constant.int 6
    %242 = torch.prims.convert_element_type %241, %int6_243 : !torch.vtensor<[4,?,256],f16>, !torch.int -> !torch.vtensor<[4,?,256],f32>
    torch.bind_symbolic_shape %242, [%31], affine_map<()[s0] -> (4, s0 * 32, 256)> : !torch.vtensor<[4,?,256],f32>
    %int2_244 = torch.constant.int 2
    %243 = torch.aten.pow.Tensor_Scalar %242, %int2_244 : !torch.vtensor<[4,?,256],f32>, !torch.int -> !torch.vtensor<[4,?,256],f32>
    torch.bind_symbolic_shape %243, [%31], affine_map<()[s0] -> (4, s0 * 32, 256)> : !torch.vtensor<[4,?,256],f32>
    %int-1_245 = torch.constant.int -1
    %244 = torch.prim.ListConstruct %int-1_245 : (!torch.int) -> !torch.list<int>
    %true_246 = torch.constant.bool true
    %none_247 = torch.constant.none
    %245 = torch.aten.mean.dim %243, %244, %true_246, %none_247 : !torch.vtensor<[4,?,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[4,?,1],f32>
    torch.bind_symbolic_shape %245, [%31], affine_map<()[s0] -> (4, s0 * 32, 1)> : !torch.vtensor<[4,?,1],f32>
    %float1.000000e-02_248 = torch.constant.float 1.000000e-02
    %int1_249 = torch.constant.int 1
    %246 = torch.aten.add.Scalar %245, %float1.000000e-02_248, %int1_249 : !torch.vtensor<[4,?,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[4,?,1],f32>
    torch.bind_symbolic_shape %246, [%31], affine_map<()[s0] -> (4, s0 * 32, 1)> : !torch.vtensor<[4,?,1],f32>
    %247 = torch.aten.rsqrt %246 : !torch.vtensor<[4,?,1],f32> -> !torch.vtensor<[4,?,1],f32>
    torch.bind_symbolic_shape %247, [%31], affine_map<()[s0] -> (4, s0 * 32, 1)> : !torch.vtensor<[4,?,1],f32>
    %248 = torch.aten.mul.Tensor %242, %247 : !torch.vtensor<[4,?,256],f32>, !torch.vtensor<[4,?,1],f32> -> !torch.vtensor<[4,?,256],f32>
    torch.bind_symbolic_shape %248, [%31], affine_map<()[s0] -> (4, s0 * 32, 256)> : !torch.vtensor<[4,?,256],f32>
    %int5_250 = torch.constant.int 5
    %249 = torch.prims.convert_element_type %248, %int5_250 : !torch.vtensor<[4,?,256],f32>, !torch.int -> !torch.vtensor<[4,?,256],f16>
    torch.bind_symbolic_shape %249, [%31], affine_map<()[s0] -> (4, s0 * 32, 256)> : !torch.vtensor<[4,?,256],f16>
    %250 = torch.aten.mul.Tensor %10, %249 : !torch.vtensor<[256],f32>, !torch.vtensor<[4,?,256],f16> -> !torch.vtensor<[4,?,256],f32>
    torch.bind_symbolic_shape %250, [%31], affine_map<()[s0] -> (4, s0 * 32, 256)> : !torch.vtensor<[4,?,256],f32>
    %int5_251 = torch.constant.int 5
    %251 = torch.prims.convert_element_type %250, %int5_251 : !torch.vtensor<[4,?,256],f32>, !torch.int -> !torch.vtensor<[4,?,256],f16>
    torch.bind_symbolic_shape %251, [%31], affine_map<()[s0] -> (4, s0 * 32, 256)> : !torch.vtensor<[4,?,256],f16>
    %int5_252 = torch.constant.int 5
    %252 = torch.prims.convert_element_type %11, %int5_252 : !torch.vtensor<[256,256],f32>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int-2_253 = torch.constant.int -2
    %int-1_254 = torch.constant.int -1
    %253 = torch.aten.transpose.int %252, %int-2_253, %int-1_254 : !torch.vtensor<[256,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,256],f16>
    %int4_255 = torch.constant.int 4
    %254 = torch.aten.mul.int %int4_255, %47 : !torch.int, !torch.int -> !torch.int
    %int256_256 = torch.constant.int 256
    %255 = torch.prim.ListConstruct %254, %int256_256 : (!torch.int, !torch.int) -> !torch.list<int>
    %256 = torch.aten.view %251, %255 : !torch.vtensor<[4,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %256, [%31], affine_map<()[s0] -> (s0 * 128, 256)> : !torch.vtensor<[?,256],f16>
    %257 = torch.aten.mm %256, %253 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,256],f16> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %257, [%31], affine_map<()[s0] -> (s0 * 128, 256)> : !torch.vtensor<[?,256],f16>
    %int4_257 = torch.constant.int 4
    %int256_258 = torch.constant.int 256
    %258 = torch.prim.ListConstruct %int4_257, %47, %int256_258 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %259 = torch.aten.view %257, %258 : !torch.vtensor<[?,256],f16>, !torch.list<int> -> !torch.vtensor<[4,?,256],f16>
    torch.bind_symbolic_shape %259, [%31], affine_map<()[s0] -> (4, s0 * 32, 256)> : !torch.vtensor<[4,?,256],f16>
    %int5_259 = torch.constant.int 5
    %260 = torch.prims.convert_element_type %12, %int5_259 : !torch.vtensor<[128,256],f32>, !torch.int -> !torch.vtensor<[128,256],f16>
    %int-2_260 = torch.constant.int -2
    %int-1_261 = torch.constant.int -1
    %261 = torch.aten.transpose.int %260, %int-2_260, %int-1_261 : !torch.vtensor<[128,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,128],f16>
    %int4_262 = torch.constant.int 4
    %262 = torch.aten.mul.int %int4_262, %47 : !torch.int, !torch.int -> !torch.int
    %int256_263 = torch.constant.int 256
    %263 = torch.prim.ListConstruct %262, %int256_263 : (!torch.int, !torch.int) -> !torch.list<int>
    %264 = torch.aten.view %251, %263 : !torch.vtensor<[4,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %264, [%31], affine_map<()[s0] -> (s0 * 128, 256)> : !torch.vtensor<[?,256],f16>
    %265 = torch.aten.mm %264, %261 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,128],f16> -> !torch.vtensor<[?,128],f16>
    torch.bind_symbolic_shape %265, [%31], affine_map<()[s0] -> (s0 * 128, 128)> : !torch.vtensor<[?,128],f16>
    %int4_264 = torch.constant.int 4
    %int128_265 = torch.constant.int 128
    %266 = torch.prim.ListConstruct %int4_264, %47, %int128_265 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %267 = torch.aten.view %265, %266 : !torch.vtensor<[?,128],f16>, !torch.list<int> -> !torch.vtensor<[4,?,128],f16>
    torch.bind_symbolic_shape %267, [%31], affine_map<()[s0] -> (4, s0 * 32, 128)> : !torch.vtensor<[4,?,128],f16>
    %int5_266 = torch.constant.int 5
    %268 = torch.prims.convert_element_type %13, %int5_266 : !torch.vtensor<[128,256],f32>, !torch.int -> !torch.vtensor<[128,256],f16>
    %int-2_267 = torch.constant.int -2
    %int-1_268 = torch.constant.int -1
    %269 = torch.aten.transpose.int %268, %int-2_267, %int-1_268 : !torch.vtensor<[128,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,128],f16>
    %int4_269 = torch.constant.int 4
    %270 = torch.aten.mul.int %int4_269, %47 : !torch.int, !torch.int -> !torch.int
    %int256_270 = torch.constant.int 256
    %271 = torch.prim.ListConstruct %270, %int256_270 : (!torch.int, !torch.int) -> !torch.list<int>
    %272 = torch.aten.view %251, %271 : !torch.vtensor<[4,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %272, [%31], affine_map<()[s0] -> (s0 * 128, 256)> : !torch.vtensor<[?,256],f16>
    %273 = torch.aten.mm %272, %269 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,128],f16> -> !torch.vtensor<[?,128],f16>
    torch.bind_symbolic_shape %273, [%31], affine_map<()[s0] -> (s0 * 128, 128)> : !torch.vtensor<[?,128],f16>
    %int4_271 = torch.constant.int 4
    %int128_272 = torch.constant.int 128
    %274 = torch.prim.ListConstruct %int4_271, %47, %int128_272 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %275 = torch.aten.view %273, %274 : !torch.vtensor<[?,128],f16>, !torch.list<int> -> !torch.vtensor<[4,?,128],f16>
    torch.bind_symbolic_shape %275, [%31], affine_map<()[s0] -> (4, s0 * 32, 128)> : !torch.vtensor<[4,?,128],f16>
    %int4_273 = torch.constant.int 4
    %int8_274 = torch.constant.int 8
    %int32_275 = torch.constant.int 32
    %276 = torch.prim.ListConstruct %int4_273, %47, %int8_274, %int32_275 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %277 = torch.aten.view %259, %276 : !torch.vtensor<[4,?,256],f16>, !torch.list<int> -> !torch.vtensor<[4,?,8,32],f16>
    torch.bind_symbolic_shape %277, [%31], affine_map<()[s0] -> (4, s0 * 32, 8, 32)> : !torch.vtensor<[4,?,8,32],f16>
    %int4_276 = torch.constant.int 4
    %int4_277 = torch.constant.int 4
    %int32_278 = torch.constant.int 32
    %278 = torch.prim.ListConstruct %int4_276, %47, %int4_277, %int32_278 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %279 = torch.aten.view %267, %278 : !torch.vtensor<[4,?,128],f16>, !torch.list<int> -> !torch.vtensor<[4,?,4,32],f16>
    torch.bind_symbolic_shape %279, [%31], affine_map<()[s0] -> (4, s0 * 32, 4, 32)> : !torch.vtensor<[4,?,4,32],f16>
    %int4_279 = torch.constant.int 4
    %int4_280 = torch.constant.int 4
    %int32_281 = torch.constant.int 32
    %280 = torch.prim.ListConstruct %int4_279, %47, %int4_280, %int32_281 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %281 = torch.aten.view %275, %280 : !torch.vtensor<[4,?,128],f16>, !torch.list<int> -> !torch.vtensor<[4,?,4,32],f16>
    torch.bind_symbolic_shape %281, [%31], affine_map<()[s0] -> (4, s0 * 32, 4, 32)> : !torch.vtensor<[4,?,4,32],f16>
    %int128_282 = torch.constant.int 128
    %none_283 = torch.constant.none
    %none_284 = torch.constant.none
    %cpu_285 = torch.constant.device "cpu"
    %false_286 = torch.constant.bool false
    %282 = torch.aten.arange %int128_282, %none_283, %none_284, %cpu_285, %false_286 : !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[128],si64>
    %int0_287 = torch.constant.int 0
    %int32_288 = torch.constant.int 32
    %none_289 = torch.constant.none
    %none_290 = torch.constant.none
    %cpu_291 = torch.constant.device "cpu"
    %false_292 = torch.constant.bool false
    %283 = torch.aten.arange.start %int0_287, %int32_288, %none_289, %none_290, %cpu_291, %false_292 : !torch.int, !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[32],si64>
    %int2_293 = torch.constant.int 2
    %284 = torch.aten.floor_divide.Scalar %283, %int2_293 : !torch.vtensor<[32],si64>, !torch.int -> !torch.vtensor<[32],si64>
    %int6_294 = torch.constant.int 6
    %285 = torch.prims.convert_element_type %284, %int6_294 : !torch.vtensor<[32],si64>, !torch.int -> !torch.vtensor<[32],f32>
    %int32_295 = torch.constant.int 32
    %286 = torch.aten.div.Scalar %285, %int32_295 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[32],f32>
    %float2.000000e00_296 = torch.constant.float 2.000000e+00
    %287 = torch.aten.mul.Scalar %286, %float2.000000e00_296 : !torch.vtensor<[32],f32>, !torch.float -> !torch.vtensor<[32],f32>
    %float5.000000e05_297 = torch.constant.float 5.000000e+05
    %288 = torch.aten.pow.Scalar %float5.000000e05_297, %287 : !torch.float, !torch.vtensor<[32],f32> -> !torch.vtensor<[32],f32>
    %289 = torch.aten.reciprocal %288 : !torch.vtensor<[32],f32> -> !torch.vtensor<[32],f32>
    %float1.000000e00_298 = torch.constant.float 1.000000e+00
    %290 = torch.aten.mul.Scalar %289, %float1.000000e00_298 : !torch.vtensor<[32],f32>, !torch.float -> !torch.vtensor<[32],f32>
    %int1_299 = torch.constant.int 1
    %291 = torch.aten.unsqueeze %282, %int1_299 : !torch.vtensor<[128],si64>, !torch.int -> !torch.vtensor<[128,1],si64>
    %int0_300 = torch.constant.int 0
    %292 = torch.aten.unsqueeze %290, %int0_300 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[1,32],f32>
    %293 = torch.aten.mul.Tensor %291, %292 : !torch.vtensor<[128,1],si64>, !torch.vtensor<[1,32],f32> -> !torch.vtensor<[128,32],f32>
    %int1_301 = torch.constant.int 1
    %294 = torch.aten.size.int %259, %int1_301 : !torch.vtensor<[4,?,256],f16>, !torch.int -> !torch.int
    %int0_302 = torch.constant.int 0
    %295 = torch.aten.add.int %int0_302, %294 : !torch.int, !torch.int -> !torch.int
    %int0_303 = torch.constant.int 0
    %int0_304 = torch.constant.int 0
    %int1_305 = torch.constant.int 1
    %296 = torch.aten.slice.Tensor %293, %int0_303, %int0_304, %295, %int1_305 : !torch.vtensor<[128,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %296, [%31], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int1_306 = torch.constant.int 1
    %int0_307 = torch.constant.int 0
    %int9223372036854775807_308 = torch.constant.int 9223372036854775807
    %int1_309 = torch.constant.int 1
    %297 = torch.aten.slice.Tensor %296, %int1_306, %int0_307, %int9223372036854775807_308, %int1_309 : !torch.vtensor<[?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %297, [%31], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int1_310 = torch.constant.int 1
    %int0_311 = torch.constant.int 0
    %int9223372036854775807_312 = torch.constant.int 9223372036854775807
    %int1_313 = torch.constant.int 1
    %298 = torch.aten.slice.Tensor %297, %int1_310, %int0_311, %int9223372036854775807_312, %int1_313 : !torch.vtensor<[?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %298, [%31], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int0_314 = torch.constant.int 0
    %299 = torch.aten.unsqueeze %298, %int0_314 : !torch.vtensor<[?,32],f32>, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %299, [%31], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int1_315 = torch.constant.int 1
    %int0_316 = torch.constant.int 0
    %int9223372036854775807_317 = torch.constant.int 9223372036854775807
    %int1_318 = torch.constant.int 1
    %300 = torch.aten.slice.Tensor %299, %int1_315, %int0_316, %int9223372036854775807_317, %int1_318 : !torch.vtensor<[1,?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %300, [%31], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int2_319 = torch.constant.int 2
    %int0_320 = torch.constant.int 0
    %int9223372036854775807_321 = torch.constant.int 9223372036854775807
    %int1_322 = torch.constant.int 1
    %301 = torch.aten.slice.Tensor %300, %int2_319, %int0_320, %int9223372036854775807_321, %int1_322 : !torch.vtensor<[1,?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %301, [%31], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int4_323 = torch.constant.int 4
    %int1_324 = torch.constant.int 1
    %int1_325 = torch.constant.int 1
    %302 = torch.prim.ListConstruct %int4_323, %int1_324, %int1_325 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %303 = torch.aten.repeat %301, %302 : !torch.vtensor<[1,?,32],f32>, !torch.list<int> -> !torch.vtensor<[4,?,32],f32>
    torch.bind_symbolic_shape %303, [%31], affine_map<()[s0] -> (4, s0 * 32, 32)> : !torch.vtensor<[4,?,32],f32>
    %int6_326 = torch.constant.int 6
    %304 = torch.prims.convert_element_type %277, %int6_326 : !torch.vtensor<[4,?,8,32],f16>, !torch.int -> !torch.vtensor<[4,?,8,32],f32>
    torch.bind_symbolic_shape %304, [%31], affine_map<()[s0] -> (4, s0 * 32, 8, 32)> : !torch.vtensor<[4,?,8,32],f32>
    %305 = torch_c.to_builtin_tensor %304 : !torch.vtensor<[4,?,8,32],f32> -> tensor<4x?x8x32xf32>
    %306 = torch_c.to_builtin_tensor %303 : !torch.vtensor<[4,?,32],f32> -> tensor<4x?x32xf32>
    %307 = util.call @sharktank_rotary_embedding_4_D_8_32_f32(%305, %306) : (tensor<4x?x8x32xf32>, tensor<4x?x32xf32>) -> tensor<4x?x8x32xf32>
    %308 = torch_c.from_builtin_tensor %307 : tensor<4x?x8x32xf32> -> !torch.vtensor<[4,?,8,32],f32>
    torch.bind_symbolic_shape %308, [%31], affine_map<()[s0] -> (4, s0 * 32, 8, 32)> : !torch.vtensor<[4,?,8,32],f32>
    %int5_327 = torch.constant.int 5
    %309 = torch.prims.convert_element_type %308, %int5_327 : !torch.vtensor<[4,?,8,32],f32>, !torch.int -> !torch.vtensor<[4,?,8,32],f16>
    torch.bind_symbolic_shape %309, [%31], affine_map<()[s0] -> (4, s0 * 32, 8, 32)> : !torch.vtensor<[4,?,8,32],f16>
    %int128_328 = torch.constant.int 128
    %none_329 = torch.constant.none
    %none_330 = torch.constant.none
    %cpu_331 = torch.constant.device "cpu"
    %false_332 = torch.constant.bool false
    %310 = torch.aten.arange %int128_328, %none_329, %none_330, %cpu_331, %false_332 : !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[128],si64>
    %int0_333 = torch.constant.int 0
    %int32_334 = torch.constant.int 32
    %none_335 = torch.constant.none
    %none_336 = torch.constant.none
    %cpu_337 = torch.constant.device "cpu"
    %false_338 = torch.constant.bool false
    %311 = torch.aten.arange.start %int0_333, %int32_334, %none_335, %none_336, %cpu_337, %false_338 : !torch.int, !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[32],si64>
    %int2_339 = torch.constant.int 2
    %312 = torch.aten.floor_divide.Scalar %311, %int2_339 : !torch.vtensor<[32],si64>, !torch.int -> !torch.vtensor<[32],si64>
    %int6_340 = torch.constant.int 6
    %313 = torch.prims.convert_element_type %312, %int6_340 : !torch.vtensor<[32],si64>, !torch.int -> !torch.vtensor<[32],f32>
    %int32_341 = torch.constant.int 32
    %314 = torch.aten.div.Scalar %313, %int32_341 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[32],f32>
    %float2.000000e00_342 = torch.constant.float 2.000000e+00
    %315 = torch.aten.mul.Scalar %314, %float2.000000e00_342 : !torch.vtensor<[32],f32>, !torch.float -> !torch.vtensor<[32],f32>
    %float5.000000e05_343 = torch.constant.float 5.000000e+05
    %316 = torch.aten.pow.Scalar %float5.000000e05_343, %315 : !torch.float, !torch.vtensor<[32],f32> -> !torch.vtensor<[32],f32>
    %317 = torch.aten.reciprocal %316 : !torch.vtensor<[32],f32> -> !torch.vtensor<[32],f32>
    %float1.000000e00_344 = torch.constant.float 1.000000e+00
    %318 = torch.aten.mul.Scalar %317, %float1.000000e00_344 : !torch.vtensor<[32],f32>, !torch.float -> !torch.vtensor<[32],f32>
    %int1_345 = torch.constant.int 1
    %319 = torch.aten.unsqueeze %310, %int1_345 : !torch.vtensor<[128],si64>, !torch.int -> !torch.vtensor<[128,1],si64>
    %int0_346 = torch.constant.int 0
    %320 = torch.aten.unsqueeze %318, %int0_346 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[1,32],f32>
    %321 = torch.aten.mul.Tensor %319, %320 : !torch.vtensor<[128,1],si64>, !torch.vtensor<[1,32],f32> -> !torch.vtensor<[128,32],f32>
    %int1_347 = torch.constant.int 1
    %322 = torch.aten.size.int %267, %int1_347 : !torch.vtensor<[4,?,128],f16>, !torch.int -> !torch.int
    %int0_348 = torch.constant.int 0
    %323 = torch.aten.add.int %int0_348, %322 : !torch.int, !torch.int -> !torch.int
    %int0_349 = torch.constant.int 0
    %int0_350 = torch.constant.int 0
    %int1_351 = torch.constant.int 1
    %324 = torch.aten.slice.Tensor %321, %int0_349, %int0_350, %323, %int1_351 : !torch.vtensor<[128,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %324, [%31], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int1_352 = torch.constant.int 1
    %int0_353 = torch.constant.int 0
    %int9223372036854775807_354 = torch.constant.int 9223372036854775807
    %int1_355 = torch.constant.int 1
    %325 = torch.aten.slice.Tensor %324, %int1_352, %int0_353, %int9223372036854775807_354, %int1_355 : !torch.vtensor<[?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %325, [%31], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int1_356 = torch.constant.int 1
    %int0_357 = torch.constant.int 0
    %int9223372036854775807_358 = torch.constant.int 9223372036854775807
    %int1_359 = torch.constant.int 1
    %326 = torch.aten.slice.Tensor %325, %int1_356, %int0_357, %int9223372036854775807_358, %int1_359 : !torch.vtensor<[?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %326, [%31], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int0_360 = torch.constant.int 0
    %327 = torch.aten.unsqueeze %326, %int0_360 : !torch.vtensor<[?,32],f32>, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %327, [%31], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int1_361 = torch.constant.int 1
    %int0_362 = torch.constant.int 0
    %int9223372036854775807_363 = torch.constant.int 9223372036854775807
    %int1_364 = torch.constant.int 1
    %328 = torch.aten.slice.Tensor %327, %int1_361, %int0_362, %int9223372036854775807_363, %int1_364 : !torch.vtensor<[1,?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %328, [%31], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int2_365 = torch.constant.int 2
    %int0_366 = torch.constant.int 0
    %int9223372036854775807_367 = torch.constant.int 9223372036854775807
    %int1_368 = torch.constant.int 1
    %329 = torch.aten.slice.Tensor %328, %int2_365, %int0_366, %int9223372036854775807_367, %int1_368 : !torch.vtensor<[1,?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %329, [%31], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int4_369 = torch.constant.int 4
    %int1_370 = torch.constant.int 1
    %int1_371 = torch.constant.int 1
    %330 = torch.prim.ListConstruct %int4_369, %int1_370, %int1_371 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %331 = torch.aten.repeat %329, %330 : !torch.vtensor<[1,?,32],f32>, !torch.list<int> -> !torch.vtensor<[4,?,32],f32>
    torch.bind_symbolic_shape %331, [%31], affine_map<()[s0] -> (4, s0 * 32, 32)> : !torch.vtensor<[4,?,32],f32>
    %int6_372 = torch.constant.int 6
    %332 = torch.prims.convert_element_type %279, %int6_372 : !torch.vtensor<[4,?,4,32],f16>, !torch.int -> !torch.vtensor<[4,?,4,32],f32>
    torch.bind_symbolic_shape %332, [%31], affine_map<()[s0] -> (4, s0 * 32, 4, 32)> : !torch.vtensor<[4,?,4,32],f32>
    %333 = torch_c.to_builtin_tensor %332 : !torch.vtensor<[4,?,4,32],f32> -> tensor<4x?x4x32xf32>
    %334 = torch_c.to_builtin_tensor %331 : !torch.vtensor<[4,?,32],f32> -> tensor<4x?x32xf32>
    %335 = util.call @sharktank_rotary_embedding_4_D_4_32_f32(%333, %334) : (tensor<4x?x4x32xf32>, tensor<4x?x32xf32>) -> tensor<4x?x4x32xf32>
    %336 = torch_c.from_builtin_tensor %335 : tensor<4x?x4x32xf32> -> !torch.vtensor<[4,?,4,32],f32>
    torch.bind_symbolic_shape %336, [%31], affine_map<()[s0] -> (4, s0 * 32, 4, 32)> : !torch.vtensor<[4,?,4,32],f32>
    %int5_373 = torch.constant.int 5
    %337 = torch.prims.convert_element_type %336, %int5_373 : !torch.vtensor<[4,?,4,32],f32>, !torch.int -> !torch.vtensor<[4,?,4,32],f16>
    torch.bind_symbolic_shape %337, [%31], affine_map<()[s0] -> (4, s0 * 32, 4, 32)> : !torch.vtensor<[4,?,4,32],f16>
    %int6_374 = torch.constant.int 6
    %338 = torch.aten.mul.Scalar %arg2, %int6_374 : !torch.vtensor<[4,?],si64>, !torch.int -> !torch.vtensor<[4,?],si64>
    torch.bind_symbolic_shape %338, [%31], affine_map<()[s0] -> (4, s0)> : !torch.vtensor<[4,?],si64>
    %int2_375 = torch.constant.int 2
    %int1_376 = torch.constant.int 1
    %339 = torch.aten.add.Scalar %338, %int2_375, %int1_376 : !torch.vtensor<[4,?],si64>, !torch.int, !torch.int -> !torch.vtensor<[4,?],si64>
    torch.bind_symbolic_shape %339, [%31], affine_map<()[s0] -> (4, s0)> : !torch.vtensor<[4,?],si64>
    %int4_377 = torch.constant.int 4
    %int32_378 = torch.constant.int 32
    %int4_379 = torch.constant.int 4
    %int32_380 = torch.constant.int 32
    %340 = torch.prim.ListConstruct %int4_377, %141, %int32_378, %int4_379, %int32_380 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %341 = torch.aten.view %337, %340 : !torch.vtensor<[4,?,4,32],f16>, !torch.list<int> -> !torch.vtensor<[4,?,32,4,32],f16>
    torch.bind_symbolic_shape %341, [%31], affine_map<()[s0] -> (4, s0, 32, 4, 32)> : !torch.vtensor<[4,?,32,4,32],f16>
    %int4_381 = torch.constant.int 4
    %342 = torch.aten.mul.int %int4_381, %141 : !torch.int, !torch.int -> !torch.int
    %int32_382 = torch.constant.int 32
    %int4_383 = torch.constant.int 4
    %int32_384 = torch.constant.int 32
    %343 = torch.prim.ListConstruct %342, %int32_382, %int4_383, %int32_384 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %344 = torch.aten.view %341, %343 : !torch.vtensor<[4,?,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,32,4,32],f16>
    torch.bind_symbolic_shape %344, [%31], affine_map<()[s0] -> (s0 * 4, 32, 4, 32)> : !torch.vtensor<[?,32,4,32],f16>
    %int4_385 = torch.constant.int 4
    %345 = torch.aten.mul.int %int4_385, %141 : !torch.int, !torch.int -> !torch.int
    %346 = torch.prim.ListConstruct %345 : (!torch.int) -> !torch.list<int>
    %347 = torch.aten.view %339, %346 : !torch.vtensor<[4,?],si64>, !torch.list<int> -> !torch.vtensor<[?],si64>
    torch.bind_symbolic_shape %347, [%31], affine_map<()[s0] -> (s0 * 4)> : !torch.vtensor<[?],si64>
    %int3_386 = torch.constant.int 3
    %int2_387 = torch.constant.int 2
    %int32_388 = torch.constant.int 32
    %int4_389 = torch.constant.int 4
    %int32_390 = torch.constant.int 32
    %348 = torch.prim.ListConstruct %132, %int3_386, %int2_387, %int32_388, %int4_389, %int32_390 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %349 = torch.aten.view %174, %348 : !torch.vtensor<[?,24576],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,4,32],f16>
    torch.bind_symbolic_shape %349, [%32], affine_map<()[s0] -> (s0, 3, 2, 32, 4, 32)> : !torch.vtensor<[?,3,2,32,4,32],f16>
    %int3_391 = torch.constant.int 3
    %350 = torch.aten.mul.int %132, %int3_391 : !torch.int, !torch.int -> !torch.int
    %int2_392 = torch.constant.int 2
    %351 = torch.aten.mul.int %350, %int2_392 : !torch.int, !torch.int -> !torch.int
    %int32_393 = torch.constant.int 32
    %int4_394 = torch.constant.int 4
    %int32_395 = torch.constant.int 32
    %352 = torch.prim.ListConstruct %351, %int32_393, %int4_394, %int32_395 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %353 = torch.aten.view %349, %352 : !torch.vtensor<[?,3,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,32,4,32],f16>
    torch.bind_symbolic_shape %353, [%32], affine_map<()[s0] -> (s0 * 6, 32, 4, 32)> : !torch.vtensor<[?,32,4,32],f16>
    %354 = torch.prim.ListConstruct %347 : (!torch.vtensor<[?],si64>) -> !torch.list<optional<vtensor>>
    %false_396 = torch.constant.bool false
    %355 = torch.aten.index_put %353, %354, %344, %false_396 : !torch.vtensor<[?,32,4,32],f16>, !torch.list<optional<vtensor>>, !torch.vtensor<[?,32,4,32],f16>, !torch.bool -> !torch.vtensor<[?,32,4,32],f16>
    torch.bind_symbolic_shape %355, [%32], affine_map<()[s0] -> (s0 * 6, 32, 4, 32)> : !torch.vtensor<[?,32,4,32],f16>
    %int3_397 = torch.constant.int 3
    %int2_398 = torch.constant.int 2
    %int32_399 = torch.constant.int 32
    %int4_400 = torch.constant.int 4
    %int32_401 = torch.constant.int 32
    %356 = torch.prim.ListConstruct %132, %int3_397, %int2_398, %int32_399, %int4_400, %int32_401 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %357 = torch.aten.view %355, %356 : !torch.vtensor<[?,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,4,32],f16>
    torch.bind_symbolic_shape %357, [%32], affine_map<()[s0] -> (s0, 3, 2, 32, 4, 32)> : !torch.vtensor<[?,3,2,32,4,32],f16>
    %int24576_402 = torch.constant.int 24576
    %358 = torch.prim.ListConstruct %132, %int24576_402 : (!torch.int, !torch.int) -> !torch.list<int>
    %359 = torch.aten.view %357, %358 : !torch.vtensor<[?,3,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,24576],f16>
    torch.bind_symbolic_shape %359, [%32], affine_map<()[s0] -> (s0, 24576)> : !torch.vtensor<[?,24576],f16>
    %int3_403 = torch.constant.int 3
    %int2_404 = torch.constant.int 2
    %int32_405 = torch.constant.int 32
    %int4_406 = torch.constant.int 4
    %int32_407 = torch.constant.int 32
    %360 = torch.prim.ListConstruct %132, %int3_403, %int2_404, %int32_405, %int4_406, %int32_407 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %361 = torch.aten.view %359, %360 : !torch.vtensor<[?,24576],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,4,32],f16>
    torch.bind_symbolic_shape %361, [%32], affine_map<()[s0] -> (s0, 3, 2, 32, 4, 32)> : !torch.vtensor<[?,3,2,32,4,32],f16>
    %int32_408 = torch.constant.int 32
    %int4_409 = torch.constant.int 4
    %int32_410 = torch.constant.int 32
    %362 = torch.prim.ListConstruct %351, %int32_408, %int4_409, %int32_410 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %363 = torch.aten.view %361, %362 : !torch.vtensor<[?,3,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,32,4,32],f16>
    torch.bind_symbolic_shape %363, [%32], affine_map<()[s0] -> (s0 * 6, 32, 4, 32)> : !torch.vtensor<[?,32,4,32],f16>
    %int4_411 = torch.constant.int 4
    %int32_412 = torch.constant.int 32
    %int4_413 = torch.constant.int 4
    %int32_414 = torch.constant.int 32
    %364 = torch.prim.ListConstruct %int4_411, %141, %int32_412, %int4_413, %int32_414 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %365 = torch.aten.view %281, %364 : !torch.vtensor<[4,?,4,32],f16>, !torch.list<int> -> !torch.vtensor<[4,?,32,4,32],f16>
    torch.bind_symbolic_shape %365, [%31], affine_map<()[s0] -> (4, s0, 32, 4, 32)> : !torch.vtensor<[4,?,32,4,32],f16>
    %int4_415 = torch.constant.int 4
    %366 = torch.aten.mul.int %int4_415, %141 : !torch.int, !torch.int -> !torch.int
    %int32_416 = torch.constant.int 32
    %int4_417 = torch.constant.int 4
    %int32_418 = torch.constant.int 32
    %367 = torch.prim.ListConstruct %366, %int32_416, %int4_417, %int32_418 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %368 = torch.aten.view %365, %367 : !torch.vtensor<[4,?,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,32,4,32],f16>
    torch.bind_symbolic_shape %368, [%31], affine_map<()[s0] -> (s0 * 4, 32, 4, 32)> : !torch.vtensor<[?,32,4,32],f16>
    %int1_419 = torch.constant.int 1
    %int1_420 = torch.constant.int 1
    %369 = torch.aten.add.Scalar %339, %int1_419, %int1_420 : !torch.vtensor<[4,?],si64>, !torch.int, !torch.int -> !torch.vtensor<[4,?],si64>
    torch.bind_symbolic_shape %369, [%31], affine_map<()[s0] -> (4, s0)> : !torch.vtensor<[4,?],si64>
    %int4_421 = torch.constant.int 4
    %370 = torch.aten.mul.int %int4_421, %141 : !torch.int, !torch.int -> !torch.int
    %371 = torch.prim.ListConstruct %370 : (!torch.int) -> !torch.list<int>
    %372 = torch.aten.view %369, %371 : !torch.vtensor<[4,?],si64>, !torch.list<int> -> !torch.vtensor<[?],si64>
    torch.bind_symbolic_shape %372, [%31], affine_map<()[s0] -> (s0 * 4)> : !torch.vtensor<[?],si64>
    %373 = torch.prim.ListConstruct %372 : (!torch.vtensor<[?],si64>) -> !torch.list<optional<vtensor>>
    %false_422 = torch.constant.bool false
    %374 = torch.aten.index_put %363, %373, %368, %false_422 : !torch.vtensor<[?,32,4,32],f16>, !torch.list<optional<vtensor>>, !torch.vtensor<[?,32,4,32],f16>, !torch.bool -> !torch.vtensor<[?,32,4,32],f16>
    torch.bind_symbolic_shape %374, [%32], affine_map<()[s0] -> (s0 * 6, 32, 4, 32)> : !torch.vtensor<[?,32,4,32],f16>
    %int3_423 = torch.constant.int 3
    %int2_424 = torch.constant.int 2
    %int32_425 = torch.constant.int 32
    %int4_426 = torch.constant.int 4
    %int32_427 = torch.constant.int 32
    %375 = torch.prim.ListConstruct %132, %int3_423, %int2_424, %int32_425, %int4_426, %int32_427 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %376 = torch.aten.view %374, %375 : !torch.vtensor<[?,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,4,32],f16>
    torch.bind_symbolic_shape %376, [%32], affine_map<()[s0] -> (s0, 3, 2, 32, 4, 32)> : !torch.vtensor<[?,3,2,32,4,32],f16>
    %int24576_428 = torch.constant.int 24576
    %377 = torch.prim.ListConstruct %132, %int24576_428 : (!torch.int, !torch.int) -> !torch.list<int>
    %378 = torch.aten.view %376, %377 : !torch.vtensor<[?,3,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,24576],f16>
    torch.bind_symbolic_shape %378, [%32], affine_map<()[s0] -> (s0, 24576)> : !torch.vtensor<[?,24576],f16>
    %int-2_429 = torch.constant.int -2
    %379 = torch.aten.unsqueeze %337, %int-2_429 : !torch.vtensor<[4,?,4,32],f16>, !torch.int -> !torch.vtensor<[4,?,4,1,32],f16>
    torch.bind_symbolic_shape %379, [%31], affine_map<()[s0] -> (4, s0 * 32, 4, 1, 32)> : !torch.vtensor<[4,?,4,1,32],f16>
    %int4_430 = torch.constant.int 4
    %int4_431 = torch.constant.int 4
    %int2_432 = torch.constant.int 2
    %int32_433 = torch.constant.int 32
    %380 = torch.prim.ListConstruct %int4_430, %322, %int4_431, %int2_432, %int32_433 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_434 = torch.constant.bool false
    %381 = torch.aten.expand %379, %380, %false_434 : !torch.vtensor<[4,?,4,1,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[4,?,4,2,32],f16>
    torch.bind_symbolic_shape %381, [%31], affine_map<()[s0] -> (4, s0 * 32, 4, 2, 32)> : !torch.vtensor<[4,?,4,2,32],f16>
    %int0_435 = torch.constant.int 0
    %382 = torch.aten.clone %381, %int0_435 : !torch.vtensor<[4,?,4,2,32],f16>, !torch.int -> !torch.vtensor<[4,?,4,2,32],f16>
    torch.bind_symbolic_shape %382, [%31], affine_map<()[s0] -> (4, s0 * 32, 4, 2, 32)> : !torch.vtensor<[4,?,4,2,32],f16>
    %int4_436 = torch.constant.int 4
    %int8_437 = torch.constant.int 8
    %int32_438 = torch.constant.int 32
    %383 = torch.prim.ListConstruct %int4_436, %322, %int8_437, %int32_438 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %384 = torch.aten._unsafe_view %382, %383 : !torch.vtensor<[4,?,4,2,32],f16>, !torch.list<int> -> !torch.vtensor<[4,?,8,32],f16>
    torch.bind_symbolic_shape %384, [%31], affine_map<()[s0] -> (4, s0 * 32, 8, 32)> : !torch.vtensor<[4,?,8,32],f16>
    %int-2_439 = torch.constant.int -2
    %385 = torch.aten.unsqueeze %281, %int-2_439 : !torch.vtensor<[4,?,4,32],f16>, !torch.int -> !torch.vtensor<[4,?,4,1,32],f16>
    torch.bind_symbolic_shape %385, [%31], affine_map<()[s0] -> (4, s0 * 32, 4, 1, 32)> : !torch.vtensor<[4,?,4,1,32],f16>
    %int1_440 = torch.constant.int 1
    %386 = torch.aten.size.int %275, %int1_440 : !torch.vtensor<[4,?,128],f16>, !torch.int -> !torch.int
    %int4_441 = torch.constant.int 4
    %int4_442 = torch.constant.int 4
    %int2_443 = torch.constant.int 2
    %int32_444 = torch.constant.int 32
    %387 = torch.prim.ListConstruct %int4_441, %386, %int4_442, %int2_443, %int32_444 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_445 = torch.constant.bool false
    %388 = torch.aten.expand %385, %387, %false_445 : !torch.vtensor<[4,?,4,1,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[4,?,4,2,32],f16>
    torch.bind_symbolic_shape %388, [%31], affine_map<()[s0] -> (4, s0 * 32, 4, 2, 32)> : !torch.vtensor<[4,?,4,2,32],f16>
    %int0_446 = torch.constant.int 0
    %389 = torch.aten.clone %388, %int0_446 : !torch.vtensor<[4,?,4,2,32],f16>, !torch.int -> !torch.vtensor<[4,?,4,2,32],f16>
    torch.bind_symbolic_shape %389, [%31], affine_map<()[s0] -> (4, s0 * 32, 4, 2, 32)> : !torch.vtensor<[4,?,4,2,32],f16>
    %int4_447 = torch.constant.int 4
    %int8_448 = torch.constant.int 8
    %int32_449 = torch.constant.int 32
    %390 = torch.prim.ListConstruct %int4_447, %386, %int8_448, %int32_449 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %391 = torch.aten._unsafe_view %389, %390 : !torch.vtensor<[4,?,4,2,32],f16>, !torch.list<int> -> !torch.vtensor<[4,?,8,32],f16>
    torch.bind_symbolic_shape %391, [%31], affine_map<()[s0] -> (4, s0 * 32, 8, 32)> : !torch.vtensor<[4,?,8,32],f16>
    %int1_450 = torch.constant.int 1
    %int2_451 = torch.constant.int 2
    %392 = torch.aten.transpose.int %309, %int1_450, %int2_451 : !torch.vtensor<[4,?,8,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[4,8,?,32],f16>
    torch.bind_symbolic_shape %392, [%31], affine_map<()[s0] -> (4, 8, s0 * 32, 32)> : !torch.vtensor<[4,8,?,32],f16>
    %int1_452 = torch.constant.int 1
    %int2_453 = torch.constant.int 2
    %393 = torch.aten.transpose.int %384, %int1_452, %int2_453 : !torch.vtensor<[4,?,8,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[4,8,?,32],f16>
    torch.bind_symbolic_shape %393, [%31], affine_map<()[s0] -> (4, 8, s0 * 32, 32)> : !torch.vtensor<[4,8,?,32],f16>
    %int1_454 = torch.constant.int 1
    %int2_455 = torch.constant.int 2
    %394 = torch.aten.transpose.int %391, %int1_454, %int2_455 : !torch.vtensor<[4,?,8,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[4,8,?,32],f16>
    torch.bind_symbolic_shape %394, [%31], affine_map<()[s0] -> (4, 8, s0 * 32, 32)> : !torch.vtensor<[4,8,?,32],f16>
    %float0.000000e00_456 = torch.constant.float 0.000000e+00
    %true_457 = torch.constant.bool true
    %none_458 = torch.constant.none
    %none_459 = torch.constant.none
    %395:2 = torch.operator "torch.aten._scaled_dot_product_flash_attention_for_cpu"(%392, %393, %394, %float0.000000e00_456, %true_457, %none_458, %none_459) : (!torch.vtensor<[4,8,?,32],f16>, !torch.vtensor<[4,8,?,32],f16>, !torch.vtensor<[4,8,?,32],f16>, !torch.float, !torch.bool, !torch.none, !torch.none) -> (!torch.vtensor<[4,8,?,32],f16>, !torch.vtensor<[4,8,?],f32>) 
    torch.bind_symbolic_shape %395#0, [%31], affine_map<()[s0] -> (4, 8, s0 * 32, 32)> : !torch.vtensor<[4,8,?,32],f16>
    %int1_460 = torch.constant.int 1
    %int2_461 = torch.constant.int 2
    %396 = torch.aten.transpose.int %395#0, %int1_460, %int2_461 : !torch.vtensor<[4,8,?,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[4,?,8,32],f16>
    torch.bind_symbolic_shape %396, [%31], affine_map<()[s0] -> (4, s0 * 32, 8, 32)> : !torch.vtensor<[4,?,8,32],f16>
    %int4_462 = torch.constant.int 4
    %int256_463 = torch.constant.int 256
    %397 = torch.prim.ListConstruct %int4_462, %294, %int256_463 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %398 = torch.aten.view %396, %397 : !torch.vtensor<[4,?,8,32],f16>, !torch.list<int> -> !torch.vtensor<[4,?,256],f16>
    torch.bind_symbolic_shape %398, [%31], affine_map<()[s0] -> (4, s0 * 32, 256)> : !torch.vtensor<[4,?,256],f16>
    %int5_464 = torch.constant.int 5
    %399 = torch.prims.convert_element_type %14, %int5_464 : !torch.vtensor<[256,256],f32>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int-2_465 = torch.constant.int -2
    %int-1_466 = torch.constant.int -1
    %400 = torch.aten.transpose.int %399, %int-2_465, %int-1_466 : !torch.vtensor<[256,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,256],f16>
    %int4_467 = torch.constant.int 4
    %401 = torch.aten.mul.int %int4_467, %294 : !torch.int, !torch.int -> !torch.int
    %int256_468 = torch.constant.int 256
    %402 = torch.prim.ListConstruct %401, %int256_468 : (!torch.int, !torch.int) -> !torch.list<int>
    %403 = torch.aten.view %398, %402 : !torch.vtensor<[4,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %403, [%31], affine_map<()[s0] -> (s0 * 128, 256)> : !torch.vtensor<[?,256],f16>
    %404 = torch.aten.mm %403, %400 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,256],f16> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %404, [%31], affine_map<()[s0] -> (s0 * 128, 256)> : !torch.vtensor<[?,256],f16>
    %int4_469 = torch.constant.int 4
    %int256_470 = torch.constant.int 256
    %405 = torch.prim.ListConstruct %int4_469, %294, %int256_470 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %406 = torch.aten.view %404, %405 : !torch.vtensor<[?,256],f16>, !torch.list<int> -> !torch.vtensor<[4,?,256],f16>
    torch.bind_symbolic_shape %406, [%31], affine_map<()[s0] -> (4, s0 * 32, 256)> : !torch.vtensor<[4,?,256],f16>
    %int1_471 = torch.constant.int 1
    %407 = torch.aten.add.Tensor %241, %406, %int1_471 : !torch.vtensor<[4,?,256],f16>, !torch.vtensor<[4,?,256],f16>, !torch.int -> !torch.vtensor<[4,?,256],f16>
    torch.bind_symbolic_shape %407, [%31], affine_map<()[s0] -> (4, s0 * 32, 256)> : !torch.vtensor<[4,?,256],f16>
    %int6_472 = torch.constant.int 6
    %408 = torch.prims.convert_element_type %407, %int6_472 : !torch.vtensor<[4,?,256],f16>, !torch.int -> !torch.vtensor<[4,?,256],f32>
    torch.bind_symbolic_shape %408, [%31], affine_map<()[s0] -> (4, s0 * 32, 256)> : !torch.vtensor<[4,?,256],f32>
    %int2_473 = torch.constant.int 2
    %409 = torch.aten.pow.Tensor_Scalar %408, %int2_473 : !torch.vtensor<[4,?,256],f32>, !torch.int -> !torch.vtensor<[4,?,256],f32>
    torch.bind_symbolic_shape %409, [%31], affine_map<()[s0] -> (4, s0 * 32, 256)> : !torch.vtensor<[4,?,256],f32>
    %int-1_474 = torch.constant.int -1
    %410 = torch.prim.ListConstruct %int-1_474 : (!torch.int) -> !torch.list<int>
    %true_475 = torch.constant.bool true
    %none_476 = torch.constant.none
    %411 = torch.aten.mean.dim %409, %410, %true_475, %none_476 : !torch.vtensor<[4,?,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[4,?,1],f32>
    torch.bind_symbolic_shape %411, [%31], affine_map<()[s0] -> (4, s0 * 32, 1)> : !torch.vtensor<[4,?,1],f32>
    %float1.000000e-02_477 = torch.constant.float 1.000000e-02
    %int1_478 = torch.constant.int 1
    %412 = torch.aten.add.Scalar %411, %float1.000000e-02_477, %int1_478 : !torch.vtensor<[4,?,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[4,?,1],f32>
    torch.bind_symbolic_shape %412, [%31], affine_map<()[s0] -> (4, s0 * 32, 1)> : !torch.vtensor<[4,?,1],f32>
    %413 = torch.aten.rsqrt %412 : !torch.vtensor<[4,?,1],f32> -> !torch.vtensor<[4,?,1],f32>
    torch.bind_symbolic_shape %413, [%31], affine_map<()[s0] -> (4, s0 * 32, 1)> : !torch.vtensor<[4,?,1],f32>
    %414 = torch.aten.mul.Tensor %408, %413 : !torch.vtensor<[4,?,256],f32>, !torch.vtensor<[4,?,1],f32> -> !torch.vtensor<[4,?,256],f32>
    torch.bind_symbolic_shape %414, [%31], affine_map<()[s0] -> (4, s0 * 32, 256)> : !torch.vtensor<[4,?,256],f32>
    %int5_479 = torch.constant.int 5
    %415 = torch.prims.convert_element_type %414, %int5_479 : !torch.vtensor<[4,?,256],f32>, !torch.int -> !torch.vtensor<[4,?,256],f16>
    torch.bind_symbolic_shape %415, [%31], affine_map<()[s0] -> (4, s0 * 32, 256)> : !torch.vtensor<[4,?,256],f16>
    %416 = torch.aten.mul.Tensor %15, %415 : !torch.vtensor<[256],f32>, !torch.vtensor<[4,?,256],f16> -> !torch.vtensor<[4,?,256],f32>
    torch.bind_symbolic_shape %416, [%31], affine_map<()[s0] -> (4, s0 * 32, 256)> : !torch.vtensor<[4,?,256],f32>
    %int5_480 = torch.constant.int 5
    %417 = torch.prims.convert_element_type %416, %int5_480 : !torch.vtensor<[4,?,256],f32>, !torch.int -> !torch.vtensor<[4,?,256],f16>
    torch.bind_symbolic_shape %417, [%31], affine_map<()[s0] -> (4, s0 * 32, 256)> : !torch.vtensor<[4,?,256],f16>
    %int5_481 = torch.constant.int 5
    %418 = torch.prims.convert_element_type %16, %int5_481 : !torch.vtensor<[23,256],f32>, !torch.int -> !torch.vtensor<[23,256],f16>
    %int-2_482 = torch.constant.int -2
    %int-1_483 = torch.constant.int -1
    %419 = torch.aten.transpose.int %418, %int-2_482, %int-1_483 : !torch.vtensor<[23,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,23],f16>
    %int4_484 = torch.constant.int 4
    %420 = torch.aten.mul.int %int4_484, %47 : !torch.int, !torch.int -> !torch.int
    %int256_485 = torch.constant.int 256
    %421 = torch.prim.ListConstruct %420, %int256_485 : (!torch.int, !torch.int) -> !torch.list<int>
    %422 = torch.aten.view %417, %421 : !torch.vtensor<[4,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %422, [%31], affine_map<()[s0] -> (s0 * 128, 256)> : !torch.vtensor<[?,256],f16>
    %423 = torch.aten.mm %422, %419 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,23],f16> -> !torch.vtensor<[?,23],f16>
    torch.bind_symbolic_shape %423, [%31], affine_map<()[s0] -> (s0 * 128, 23)> : !torch.vtensor<[?,23],f16>
    %int4_486 = torch.constant.int 4
    %int23_487 = torch.constant.int 23
    %424 = torch.prim.ListConstruct %int4_486, %47, %int23_487 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %425 = torch.aten.view %423, %424 : !torch.vtensor<[?,23],f16>, !torch.list<int> -> !torch.vtensor<[4,?,23],f16>
    torch.bind_symbolic_shape %425, [%31], affine_map<()[s0] -> (4, s0 * 32, 23)> : !torch.vtensor<[4,?,23],f16>
    %426 = torch.aten.silu %425 : !torch.vtensor<[4,?,23],f16> -> !torch.vtensor<[4,?,23],f16>
    torch.bind_symbolic_shape %426, [%31], affine_map<()[s0] -> (4, s0 * 32, 23)> : !torch.vtensor<[4,?,23],f16>
    %int5_488 = torch.constant.int 5
    %427 = torch.prims.convert_element_type %17, %int5_488 : !torch.vtensor<[23,256],f32>, !torch.int -> !torch.vtensor<[23,256],f16>
    %int-2_489 = torch.constant.int -2
    %int-1_490 = torch.constant.int -1
    %428 = torch.aten.transpose.int %427, %int-2_489, %int-1_490 : !torch.vtensor<[23,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,23],f16>
    %int4_491 = torch.constant.int 4
    %429 = torch.aten.mul.int %int4_491, %47 : !torch.int, !torch.int -> !torch.int
    %int256_492 = torch.constant.int 256
    %430 = torch.prim.ListConstruct %429, %int256_492 : (!torch.int, !torch.int) -> !torch.list<int>
    %431 = torch.aten.view %417, %430 : !torch.vtensor<[4,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %431, [%31], affine_map<()[s0] -> (s0 * 128, 256)> : !torch.vtensor<[?,256],f16>
    %432 = torch.aten.mm %431, %428 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,23],f16> -> !torch.vtensor<[?,23],f16>
    torch.bind_symbolic_shape %432, [%31], affine_map<()[s0] -> (s0 * 128, 23)> : !torch.vtensor<[?,23],f16>
    %int4_493 = torch.constant.int 4
    %int23_494 = torch.constant.int 23
    %433 = torch.prim.ListConstruct %int4_493, %47, %int23_494 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %434 = torch.aten.view %432, %433 : !torch.vtensor<[?,23],f16>, !torch.list<int> -> !torch.vtensor<[4,?,23],f16>
    torch.bind_symbolic_shape %434, [%31], affine_map<()[s0] -> (4, s0 * 32, 23)> : !torch.vtensor<[4,?,23],f16>
    %435 = torch.aten.mul.Tensor %426, %434 : !torch.vtensor<[4,?,23],f16>, !torch.vtensor<[4,?,23],f16> -> !torch.vtensor<[4,?,23],f16>
    torch.bind_symbolic_shape %435, [%31], affine_map<()[s0] -> (4, s0 * 32, 23)> : !torch.vtensor<[4,?,23],f16>
    %int5_495 = torch.constant.int 5
    %436 = torch.prims.convert_element_type %18, %int5_495 : !torch.vtensor<[256,23],f32>, !torch.int -> !torch.vtensor<[256,23],f16>
    %int-2_496 = torch.constant.int -2
    %int-1_497 = torch.constant.int -1
    %437 = torch.aten.transpose.int %436, %int-2_496, %int-1_497 : !torch.vtensor<[256,23],f16>, !torch.int, !torch.int -> !torch.vtensor<[23,256],f16>
    %int1_498 = torch.constant.int 1
    %438 = torch.aten.size.int %425, %int1_498 : !torch.vtensor<[4,?,23],f16>, !torch.int -> !torch.int
    %int4_499 = torch.constant.int 4
    %439 = torch.aten.mul.int %int4_499, %438 : !torch.int, !torch.int -> !torch.int
    %int23_500 = torch.constant.int 23
    %440 = torch.prim.ListConstruct %439, %int23_500 : (!torch.int, !torch.int) -> !torch.list<int>
    %441 = torch.aten.view %435, %440 : !torch.vtensor<[4,?,23],f16>, !torch.list<int> -> !torch.vtensor<[?,23],f16>
    torch.bind_symbolic_shape %441, [%31], affine_map<()[s0] -> (s0 * 128, 23)> : !torch.vtensor<[?,23],f16>
    %442 = torch.aten.mm %441, %437 : !torch.vtensor<[?,23],f16>, !torch.vtensor<[23,256],f16> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %442, [%31], affine_map<()[s0] -> (s0 * 128, 256)> : !torch.vtensor<[?,256],f16>
    %int4_501 = torch.constant.int 4
    %int256_502 = torch.constant.int 256
    %443 = torch.prim.ListConstruct %int4_501, %438, %int256_502 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %444 = torch.aten.view %442, %443 : !torch.vtensor<[?,256],f16>, !torch.list<int> -> !torch.vtensor<[4,?,256],f16>
    torch.bind_symbolic_shape %444, [%31], affine_map<()[s0] -> (4, s0 * 32, 256)> : !torch.vtensor<[4,?,256],f16>
    %int1_503 = torch.constant.int 1
    %445 = torch.aten.add.Tensor %407, %444, %int1_503 : !torch.vtensor<[4,?,256],f16>, !torch.vtensor<[4,?,256],f16>, !torch.int -> !torch.vtensor<[4,?,256],f16>
    torch.bind_symbolic_shape %445, [%31], affine_map<()[s0] -> (4, s0 * 32, 256)> : !torch.vtensor<[4,?,256],f16>
    %int6_504 = torch.constant.int 6
    %446 = torch.prims.convert_element_type %445, %int6_504 : !torch.vtensor<[4,?,256],f16>, !torch.int -> !torch.vtensor<[4,?,256],f32>
    torch.bind_symbolic_shape %446, [%31], affine_map<()[s0] -> (4, s0 * 32, 256)> : !torch.vtensor<[4,?,256],f32>
    %int2_505 = torch.constant.int 2
    %447 = torch.aten.pow.Tensor_Scalar %446, %int2_505 : !torch.vtensor<[4,?,256],f32>, !torch.int -> !torch.vtensor<[4,?,256],f32>
    torch.bind_symbolic_shape %447, [%31], affine_map<()[s0] -> (4, s0 * 32, 256)> : !torch.vtensor<[4,?,256],f32>
    %int-1_506 = torch.constant.int -1
    %448 = torch.prim.ListConstruct %int-1_506 : (!torch.int) -> !torch.list<int>
    %true_507 = torch.constant.bool true
    %none_508 = torch.constant.none
    %449 = torch.aten.mean.dim %447, %448, %true_507, %none_508 : !torch.vtensor<[4,?,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[4,?,1],f32>
    torch.bind_symbolic_shape %449, [%31], affine_map<()[s0] -> (4, s0 * 32, 1)> : !torch.vtensor<[4,?,1],f32>
    %float1.000000e-02_509 = torch.constant.float 1.000000e-02
    %int1_510 = torch.constant.int 1
    %450 = torch.aten.add.Scalar %449, %float1.000000e-02_509, %int1_510 : !torch.vtensor<[4,?,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[4,?,1],f32>
    torch.bind_symbolic_shape %450, [%31], affine_map<()[s0] -> (4, s0 * 32, 1)> : !torch.vtensor<[4,?,1],f32>
    %451 = torch.aten.rsqrt %450 : !torch.vtensor<[4,?,1],f32> -> !torch.vtensor<[4,?,1],f32>
    torch.bind_symbolic_shape %451, [%31], affine_map<()[s0] -> (4, s0 * 32, 1)> : !torch.vtensor<[4,?,1],f32>
    %452 = torch.aten.mul.Tensor %446, %451 : !torch.vtensor<[4,?,256],f32>, !torch.vtensor<[4,?,1],f32> -> !torch.vtensor<[4,?,256],f32>
    torch.bind_symbolic_shape %452, [%31], affine_map<()[s0] -> (4, s0 * 32, 256)> : !torch.vtensor<[4,?,256],f32>
    %int5_511 = torch.constant.int 5
    %453 = torch.prims.convert_element_type %452, %int5_511 : !torch.vtensor<[4,?,256],f32>, !torch.int -> !torch.vtensor<[4,?,256],f16>
    torch.bind_symbolic_shape %453, [%31], affine_map<()[s0] -> (4, s0 * 32, 256)> : !torch.vtensor<[4,?,256],f16>
    %454 = torch.aten.mul.Tensor %19, %453 : !torch.vtensor<[256],f32>, !torch.vtensor<[4,?,256],f16> -> !torch.vtensor<[4,?,256],f32>
    torch.bind_symbolic_shape %454, [%31], affine_map<()[s0] -> (4, s0 * 32, 256)> : !torch.vtensor<[4,?,256],f32>
    %int5_512 = torch.constant.int 5
    %455 = torch.prims.convert_element_type %454, %int5_512 : !torch.vtensor<[4,?,256],f32>, !torch.int -> !torch.vtensor<[4,?,256],f16>
    torch.bind_symbolic_shape %455, [%31], affine_map<()[s0] -> (4, s0 * 32, 256)> : !torch.vtensor<[4,?,256],f16>
    %int5_513 = torch.constant.int 5
    %456 = torch.prims.convert_element_type %20, %int5_513 : !torch.vtensor<[256,256],f32>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int-2_514 = torch.constant.int -2
    %int-1_515 = torch.constant.int -1
    %457 = torch.aten.transpose.int %456, %int-2_514, %int-1_515 : !torch.vtensor<[256,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,256],f16>
    %int4_516 = torch.constant.int 4
    %458 = torch.aten.mul.int %int4_516, %47 : !torch.int, !torch.int -> !torch.int
    %int256_517 = torch.constant.int 256
    %459 = torch.prim.ListConstruct %458, %int256_517 : (!torch.int, !torch.int) -> !torch.list<int>
    %460 = torch.aten.view %455, %459 : !torch.vtensor<[4,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %460, [%31], affine_map<()[s0] -> (s0 * 128, 256)> : !torch.vtensor<[?,256],f16>
    %461 = torch.aten.mm %460, %457 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,256],f16> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %461, [%31], affine_map<()[s0] -> (s0 * 128, 256)> : !torch.vtensor<[?,256],f16>
    %int4_518 = torch.constant.int 4
    %int256_519 = torch.constant.int 256
    %462 = torch.prim.ListConstruct %int4_518, %47, %int256_519 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %463 = torch.aten.view %461, %462 : !torch.vtensor<[?,256],f16>, !torch.list<int> -> !torch.vtensor<[4,?,256],f16>
    torch.bind_symbolic_shape %463, [%31], affine_map<()[s0] -> (4, s0 * 32, 256)> : !torch.vtensor<[4,?,256],f16>
    %int5_520 = torch.constant.int 5
    %464 = torch.prims.convert_element_type %21, %int5_520 : !torch.vtensor<[128,256],f32>, !torch.int -> !torch.vtensor<[128,256],f16>
    %int-2_521 = torch.constant.int -2
    %int-1_522 = torch.constant.int -1
    %465 = torch.aten.transpose.int %464, %int-2_521, %int-1_522 : !torch.vtensor<[128,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,128],f16>
    %int4_523 = torch.constant.int 4
    %466 = torch.aten.mul.int %int4_523, %47 : !torch.int, !torch.int -> !torch.int
    %int256_524 = torch.constant.int 256
    %467 = torch.prim.ListConstruct %466, %int256_524 : (!torch.int, !torch.int) -> !torch.list<int>
    %468 = torch.aten.view %455, %467 : !torch.vtensor<[4,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %468, [%31], affine_map<()[s0] -> (s0 * 128, 256)> : !torch.vtensor<[?,256],f16>
    %469 = torch.aten.mm %468, %465 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,128],f16> -> !torch.vtensor<[?,128],f16>
    torch.bind_symbolic_shape %469, [%31], affine_map<()[s0] -> (s0 * 128, 128)> : !torch.vtensor<[?,128],f16>
    %int4_525 = torch.constant.int 4
    %int128_526 = torch.constant.int 128
    %470 = torch.prim.ListConstruct %int4_525, %47, %int128_526 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %471 = torch.aten.view %469, %470 : !torch.vtensor<[?,128],f16>, !torch.list<int> -> !torch.vtensor<[4,?,128],f16>
    torch.bind_symbolic_shape %471, [%31], affine_map<()[s0] -> (4, s0 * 32, 128)> : !torch.vtensor<[4,?,128],f16>
    %int5_527 = torch.constant.int 5
    %472 = torch.prims.convert_element_type %22, %int5_527 : !torch.vtensor<[128,256],f32>, !torch.int -> !torch.vtensor<[128,256],f16>
    %int-2_528 = torch.constant.int -2
    %int-1_529 = torch.constant.int -1
    %473 = torch.aten.transpose.int %472, %int-2_528, %int-1_529 : !torch.vtensor<[128,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,128],f16>
    %int4_530 = torch.constant.int 4
    %474 = torch.aten.mul.int %int4_530, %47 : !torch.int, !torch.int -> !torch.int
    %int256_531 = torch.constant.int 256
    %475 = torch.prim.ListConstruct %474, %int256_531 : (!torch.int, !torch.int) -> !torch.list<int>
    %476 = torch.aten.view %455, %475 : !torch.vtensor<[4,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %476, [%31], affine_map<()[s0] -> (s0 * 128, 256)> : !torch.vtensor<[?,256],f16>
    %477 = torch.aten.mm %476, %473 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,128],f16> -> !torch.vtensor<[?,128],f16>
    torch.bind_symbolic_shape %477, [%31], affine_map<()[s0] -> (s0 * 128, 128)> : !torch.vtensor<[?,128],f16>
    %int4_532 = torch.constant.int 4
    %int128_533 = torch.constant.int 128
    %478 = torch.prim.ListConstruct %int4_532, %47, %int128_533 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %479 = torch.aten.view %477, %478 : !torch.vtensor<[?,128],f16>, !torch.list<int> -> !torch.vtensor<[4,?,128],f16>
    torch.bind_symbolic_shape %479, [%31], affine_map<()[s0] -> (4, s0 * 32, 128)> : !torch.vtensor<[4,?,128],f16>
    %int4_534 = torch.constant.int 4
    %int8_535 = torch.constant.int 8
    %int32_536 = torch.constant.int 32
    %480 = torch.prim.ListConstruct %int4_534, %47, %int8_535, %int32_536 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %481 = torch.aten.view %463, %480 : !torch.vtensor<[4,?,256],f16>, !torch.list<int> -> !torch.vtensor<[4,?,8,32],f16>
    torch.bind_symbolic_shape %481, [%31], affine_map<()[s0] -> (4, s0 * 32, 8, 32)> : !torch.vtensor<[4,?,8,32],f16>
    %int4_537 = torch.constant.int 4
    %int4_538 = torch.constant.int 4
    %int32_539 = torch.constant.int 32
    %482 = torch.prim.ListConstruct %int4_537, %47, %int4_538, %int32_539 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %483 = torch.aten.view %471, %482 : !torch.vtensor<[4,?,128],f16>, !torch.list<int> -> !torch.vtensor<[4,?,4,32],f16>
    torch.bind_symbolic_shape %483, [%31], affine_map<()[s0] -> (4, s0 * 32, 4, 32)> : !torch.vtensor<[4,?,4,32],f16>
    %int4_540 = torch.constant.int 4
    %int4_541 = torch.constant.int 4
    %int32_542 = torch.constant.int 32
    %484 = torch.prim.ListConstruct %int4_540, %47, %int4_541, %int32_542 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %485 = torch.aten.view %479, %484 : !torch.vtensor<[4,?,128],f16>, !torch.list<int> -> !torch.vtensor<[4,?,4,32],f16>
    torch.bind_symbolic_shape %485, [%31], affine_map<()[s0] -> (4, s0 * 32, 4, 32)> : !torch.vtensor<[4,?,4,32],f16>
    %int128_543 = torch.constant.int 128
    %none_544 = torch.constant.none
    %none_545 = torch.constant.none
    %cpu_546 = torch.constant.device "cpu"
    %false_547 = torch.constant.bool false
    %486 = torch.aten.arange %int128_543, %none_544, %none_545, %cpu_546, %false_547 : !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[128],si64>
    %int0_548 = torch.constant.int 0
    %int32_549 = torch.constant.int 32
    %none_550 = torch.constant.none
    %none_551 = torch.constant.none
    %cpu_552 = torch.constant.device "cpu"
    %false_553 = torch.constant.bool false
    %487 = torch.aten.arange.start %int0_548, %int32_549, %none_550, %none_551, %cpu_552, %false_553 : !torch.int, !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[32],si64>
    %int2_554 = torch.constant.int 2
    %488 = torch.aten.floor_divide.Scalar %487, %int2_554 : !torch.vtensor<[32],si64>, !torch.int -> !torch.vtensor<[32],si64>
    %int6_555 = torch.constant.int 6
    %489 = torch.prims.convert_element_type %488, %int6_555 : !torch.vtensor<[32],si64>, !torch.int -> !torch.vtensor<[32],f32>
    %int32_556 = torch.constant.int 32
    %490 = torch.aten.div.Scalar %489, %int32_556 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[32],f32>
    %float2.000000e00_557 = torch.constant.float 2.000000e+00
    %491 = torch.aten.mul.Scalar %490, %float2.000000e00_557 : !torch.vtensor<[32],f32>, !torch.float -> !torch.vtensor<[32],f32>
    %float5.000000e05_558 = torch.constant.float 5.000000e+05
    %492 = torch.aten.pow.Scalar %float5.000000e05_558, %491 : !torch.float, !torch.vtensor<[32],f32> -> !torch.vtensor<[32],f32>
    %493 = torch.aten.reciprocal %492 : !torch.vtensor<[32],f32> -> !torch.vtensor<[32],f32>
    %float1.000000e00_559 = torch.constant.float 1.000000e+00
    %494 = torch.aten.mul.Scalar %493, %float1.000000e00_559 : !torch.vtensor<[32],f32>, !torch.float -> !torch.vtensor<[32],f32>
    %int1_560 = torch.constant.int 1
    %495 = torch.aten.unsqueeze %486, %int1_560 : !torch.vtensor<[128],si64>, !torch.int -> !torch.vtensor<[128,1],si64>
    %int0_561 = torch.constant.int 0
    %496 = torch.aten.unsqueeze %494, %int0_561 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[1,32],f32>
    %497 = torch.aten.mul.Tensor %495, %496 : !torch.vtensor<[128,1],si64>, !torch.vtensor<[1,32],f32> -> !torch.vtensor<[128,32],f32>
    %int1_562 = torch.constant.int 1
    %498 = torch.aten.size.int %463, %int1_562 : !torch.vtensor<[4,?,256],f16>, !torch.int -> !torch.int
    %int0_563 = torch.constant.int 0
    %499 = torch.aten.add.int %int0_563, %498 : !torch.int, !torch.int -> !torch.int
    %int0_564 = torch.constant.int 0
    %int0_565 = torch.constant.int 0
    %int1_566 = torch.constant.int 1
    %500 = torch.aten.slice.Tensor %497, %int0_564, %int0_565, %499, %int1_566 : !torch.vtensor<[128,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %500, [%31], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int1_567 = torch.constant.int 1
    %int0_568 = torch.constant.int 0
    %int9223372036854775807_569 = torch.constant.int 9223372036854775807
    %int1_570 = torch.constant.int 1
    %501 = torch.aten.slice.Tensor %500, %int1_567, %int0_568, %int9223372036854775807_569, %int1_570 : !torch.vtensor<[?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %501, [%31], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int1_571 = torch.constant.int 1
    %int0_572 = torch.constant.int 0
    %int9223372036854775807_573 = torch.constant.int 9223372036854775807
    %int1_574 = torch.constant.int 1
    %502 = torch.aten.slice.Tensor %501, %int1_571, %int0_572, %int9223372036854775807_573, %int1_574 : !torch.vtensor<[?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %502, [%31], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int0_575 = torch.constant.int 0
    %503 = torch.aten.unsqueeze %502, %int0_575 : !torch.vtensor<[?,32],f32>, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %503, [%31], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int1_576 = torch.constant.int 1
    %int0_577 = torch.constant.int 0
    %int9223372036854775807_578 = torch.constant.int 9223372036854775807
    %int1_579 = torch.constant.int 1
    %504 = torch.aten.slice.Tensor %503, %int1_576, %int0_577, %int9223372036854775807_578, %int1_579 : !torch.vtensor<[1,?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %504, [%31], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int2_580 = torch.constant.int 2
    %int0_581 = torch.constant.int 0
    %int9223372036854775807_582 = torch.constant.int 9223372036854775807
    %int1_583 = torch.constant.int 1
    %505 = torch.aten.slice.Tensor %504, %int2_580, %int0_581, %int9223372036854775807_582, %int1_583 : !torch.vtensor<[1,?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %505, [%31], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int4_584 = torch.constant.int 4
    %int1_585 = torch.constant.int 1
    %int1_586 = torch.constant.int 1
    %506 = torch.prim.ListConstruct %int4_584, %int1_585, %int1_586 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %507 = torch.aten.repeat %505, %506 : !torch.vtensor<[1,?,32],f32>, !torch.list<int> -> !torch.vtensor<[4,?,32],f32>
    torch.bind_symbolic_shape %507, [%31], affine_map<()[s0] -> (4, s0 * 32, 32)> : !torch.vtensor<[4,?,32],f32>
    %int6_587 = torch.constant.int 6
    %508 = torch.prims.convert_element_type %481, %int6_587 : !torch.vtensor<[4,?,8,32],f16>, !torch.int -> !torch.vtensor<[4,?,8,32],f32>
    torch.bind_symbolic_shape %508, [%31], affine_map<()[s0] -> (4, s0 * 32, 8, 32)> : !torch.vtensor<[4,?,8,32],f32>
    %509 = torch_c.to_builtin_tensor %508 : !torch.vtensor<[4,?,8,32],f32> -> tensor<4x?x8x32xf32>
    %510 = torch_c.to_builtin_tensor %507 : !torch.vtensor<[4,?,32],f32> -> tensor<4x?x32xf32>
    %511 = util.call @sharktank_rotary_embedding_4_D_8_32_f32(%509, %510) : (tensor<4x?x8x32xf32>, tensor<4x?x32xf32>) -> tensor<4x?x8x32xf32>
    %512 = torch_c.from_builtin_tensor %511 : tensor<4x?x8x32xf32> -> !torch.vtensor<[4,?,8,32],f32>
    torch.bind_symbolic_shape %512, [%31], affine_map<()[s0] -> (4, s0 * 32, 8, 32)> : !torch.vtensor<[4,?,8,32],f32>
    %int5_588 = torch.constant.int 5
    %513 = torch.prims.convert_element_type %512, %int5_588 : !torch.vtensor<[4,?,8,32],f32>, !torch.int -> !torch.vtensor<[4,?,8,32],f16>
    torch.bind_symbolic_shape %513, [%31], affine_map<()[s0] -> (4, s0 * 32, 8, 32)> : !torch.vtensor<[4,?,8,32],f16>
    %int128_589 = torch.constant.int 128
    %none_590 = torch.constant.none
    %none_591 = torch.constant.none
    %cpu_592 = torch.constant.device "cpu"
    %false_593 = torch.constant.bool false
    %514 = torch.aten.arange %int128_589, %none_590, %none_591, %cpu_592, %false_593 : !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[128],si64>
    %int0_594 = torch.constant.int 0
    %int32_595 = torch.constant.int 32
    %none_596 = torch.constant.none
    %none_597 = torch.constant.none
    %cpu_598 = torch.constant.device "cpu"
    %false_599 = torch.constant.bool false
    %515 = torch.aten.arange.start %int0_594, %int32_595, %none_596, %none_597, %cpu_598, %false_599 : !torch.int, !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[32],si64>
    %int2_600 = torch.constant.int 2
    %516 = torch.aten.floor_divide.Scalar %515, %int2_600 : !torch.vtensor<[32],si64>, !torch.int -> !torch.vtensor<[32],si64>
    %int6_601 = torch.constant.int 6
    %517 = torch.prims.convert_element_type %516, %int6_601 : !torch.vtensor<[32],si64>, !torch.int -> !torch.vtensor<[32],f32>
    %int32_602 = torch.constant.int 32
    %518 = torch.aten.div.Scalar %517, %int32_602 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[32],f32>
    %float2.000000e00_603 = torch.constant.float 2.000000e+00
    %519 = torch.aten.mul.Scalar %518, %float2.000000e00_603 : !torch.vtensor<[32],f32>, !torch.float -> !torch.vtensor<[32],f32>
    %float5.000000e05_604 = torch.constant.float 5.000000e+05
    %520 = torch.aten.pow.Scalar %float5.000000e05_604, %519 : !torch.float, !torch.vtensor<[32],f32> -> !torch.vtensor<[32],f32>
    %521 = torch.aten.reciprocal %520 : !torch.vtensor<[32],f32> -> !torch.vtensor<[32],f32>
    %float1.000000e00_605 = torch.constant.float 1.000000e+00
    %522 = torch.aten.mul.Scalar %521, %float1.000000e00_605 : !torch.vtensor<[32],f32>, !torch.float -> !torch.vtensor<[32],f32>
    %int1_606 = torch.constant.int 1
    %523 = torch.aten.unsqueeze %514, %int1_606 : !torch.vtensor<[128],si64>, !torch.int -> !torch.vtensor<[128,1],si64>
    %int0_607 = torch.constant.int 0
    %524 = torch.aten.unsqueeze %522, %int0_607 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[1,32],f32>
    %525 = torch.aten.mul.Tensor %523, %524 : !torch.vtensor<[128,1],si64>, !torch.vtensor<[1,32],f32> -> !torch.vtensor<[128,32],f32>
    %int1_608 = torch.constant.int 1
    %526 = torch.aten.size.int %471, %int1_608 : !torch.vtensor<[4,?,128],f16>, !torch.int -> !torch.int
    %int0_609 = torch.constant.int 0
    %527 = torch.aten.add.int %int0_609, %526 : !torch.int, !torch.int -> !torch.int
    %int0_610 = torch.constant.int 0
    %int0_611 = torch.constant.int 0
    %int1_612 = torch.constant.int 1
    %528 = torch.aten.slice.Tensor %525, %int0_610, %int0_611, %527, %int1_612 : !torch.vtensor<[128,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %528, [%31], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int1_613 = torch.constant.int 1
    %int0_614 = torch.constant.int 0
    %int9223372036854775807_615 = torch.constant.int 9223372036854775807
    %int1_616 = torch.constant.int 1
    %529 = torch.aten.slice.Tensor %528, %int1_613, %int0_614, %int9223372036854775807_615, %int1_616 : !torch.vtensor<[?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %529, [%31], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int1_617 = torch.constant.int 1
    %int0_618 = torch.constant.int 0
    %int9223372036854775807_619 = torch.constant.int 9223372036854775807
    %int1_620 = torch.constant.int 1
    %530 = torch.aten.slice.Tensor %529, %int1_617, %int0_618, %int9223372036854775807_619, %int1_620 : !torch.vtensor<[?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %530, [%31], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int0_621 = torch.constant.int 0
    %531 = torch.aten.unsqueeze %530, %int0_621 : !torch.vtensor<[?,32],f32>, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %531, [%31], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int1_622 = torch.constant.int 1
    %int0_623 = torch.constant.int 0
    %int9223372036854775807_624 = torch.constant.int 9223372036854775807
    %int1_625 = torch.constant.int 1
    %532 = torch.aten.slice.Tensor %531, %int1_622, %int0_623, %int9223372036854775807_624, %int1_625 : !torch.vtensor<[1,?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %532, [%31], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int2_626 = torch.constant.int 2
    %int0_627 = torch.constant.int 0
    %int9223372036854775807_628 = torch.constant.int 9223372036854775807
    %int1_629 = torch.constant.int 1
    %533 = torch.aten.slice.Tensor %532, %int2_626, %int0_627, %int9223372036854775807_628, %int1_629 : !torch.vtensor<[1,?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %533, [%31], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int4_630 = torch.constant.int 4
    %int1_631 = torch.constant.int 1
    %int1_632 = torch.constant.int 1
    %534 = torch.prim.ListConstruct %int4_630, %int1_631, %int1_632 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %535 = torch.aten.repeat %533, %534 : !torch.vtensor<[1,?,32],f32>, !torch.list<int> -> !torch.vtensor<[4,?,32],f32>
    torch.bind_symbolic_shape %535, [%31], affine_map<()[s0] -> (4, s0 * 32, 32)> : !torch.vtensor<[4,?,32],f32>
    %int6_633 = torch.constant.int 6
    %536 = torch.prims.convert_element_type %483, %int6_633 : !torch.vtensor<[4,?,4,32],f16>, !torch.int -> !torch.vtensor<[4,?,4,32],f32>
    torch.bind_symbolic_shape %536, [%31], affine_map<()[s0] -> (4, s0 * 32, 4, 32)> : !torch.vtensor<[4,?,4,32],f32>
    %537 = torch_c.to_builtin_tensor %536 : !torch.vtensor<[4,?,4,32],f32> -> tensor<4x?x4x32xf32>
    %538 = torch_c.to_builtin_tensor %535 : !torch.vtensor<[4,?,32],f32> -> tensor<4x?x32xf32>
    %539 = util.call @sharktank_rotary_embedding_4_D_4_32_f32(%537, %538) : (tensor<4x?x4x32xf32>, tensor<4x?x32xf32>) -> tensor<4x?x4x32xf32>
    %540 = torch_c.from_builtin_tensor %539 : tensor<4x?x4x32xf32> -> !torch.vtensor<[4,?,4,32],f32>
    torch.bind_symbolic_shape %540, [%31], affine_map<()[s0] -> (4, s0 * 32, 4, 32)> : !torch.vtensor<[4,?,4,32],f32>
    %int5_634 = torch.constant.int 5
    %541 = torch.prims.convert_element_type %540, %int5_634 : !torch.vtensor<[4,?,4,32],f32>, !torch.int -> !torch.vtensor<[4,?,4,32],f16>
    torch.bind_symbolic_shape %541, [%31], affine_map<()[s0] -> (4, s0 * 32, 4, 32)> : !torch.vtensor<[4,?,4,32],f16>
    %int6_635 = torch.constant.int 6
    %542 = torch.aten.mul.Scalar %arg2, %int6_635 : !torch.vtensor<[4,?],si64>, !torch.int -> !torch.vtensor<[4,?],si64>
    torch.bind_symbolic_shape %542, [%31], affine_map<()[s0] -> (4, s0)> : !torch.vtensor<[4,?],si64>
    %int4_636 = torch.constant.int 4
    %int1_637 = torch.constant.int 1
    %543 = torch.aten.add.Scalar %542, %int4_636, %int1_637 : !torch.vtensor<[4,?],si64>, !torch.int, !torch.int -> !torch.vtensor<[4,?],si64>
    torch.bind_symbolic_shape %543, [%31], affine_map<()[s0] -> (4, s0)> : !torch.vtensor<[4,?],si64>
    %int4_638 = torch.constant.int 4
    %int32_639 = torch.constant.int 32
    %int4_640 = torch.constant.int 4
    %int32_641 = torch.constant.int 32
    %544 = torch.prim.ListConstruct %int4_638, %141, %int32_639, %int4_640, %int32_641 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %545 = torch.aten.view %541, %544 : !torch.vtensor<[4,?,4,32],f16>, !torch.list<int> -> !torch.vtensor<[4,?,32,4,32],f16>
    torch.bind_symbolic_shape %545, [%31], affine_map<()[s0] -> (4, s0, 32, 4, 32)> : !torch.vtensor<[4,?,32,4,32],f16>
    %int4_642 = torch.constant.int 4
    %546 = torch.aten.mul.int %int4_642, %141 : !torch.int, !torch.int -> !torch.int
    %int32_643 = torch.constant.int 32
    %int4_644 = torch.constant.int 4
    %int32_645 = torch.constant.int 32
    %547 = torch.prim.ListConstruct %546, %int32_643, %int4_644, %int32_645 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %548 = torch.aten.view %545, %547 : !torch.vtensor<[4,?,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,32,4,32],f16>
    torch.bind_symbolic_shape %548, [%31], affine_map<()[s0] -> (s0 * 4, 32, 4, 32)> : !torch.vtensor<[?,32,4,32],f16>
    %int4_646 = torch.constant.int 4
    %549 = torch.aten.mul.int %int4_646, %141 : !torch.int, !torch.int -> !torch.int
    %550 = torch.prim.ListConstruct %549 : (!torch.int) -> !torch.list<int>
    %551 = torch.aten.view %543, %550 : !torch.vtensor<[4,?],si64>, !torch.list<int> -> !torch.vtensor<[?],si64>
    torch.bind_symbolic_shape %551, [%31], affine_map<()[s0] -> (s0 * 4)> : !torch.vtensor<[?],si64>
    %int3_647 = torch.constant.int 3
    %int2_648 = torch.constant.int 2
    %int32_649 = torch.constant.int 32
    %int4_650 = torch.constant.int 4
    %int32_651 = torch.constant.int 32
    %552 = torch.prim.ListConstruct %132, %int3_647, %int2_648, %int32_649, %int4_650, %int32_651 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %553 = torch.aten.view %378, %552 : !torch.vtensor<[?,24576],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,4,32],f16>
    torch.bind_symbolic_shape %553, [%32], affine_map<()[s0] -> (s0, 3, 2, 32, 4, 32)> : !torch.vtensor<[?,3,2,32,4,32],f16>
    %int3_652 = torch.constant.int 3
    %554 = torch.aten.mul.int %132, %int3_652 : !torch.int, !torch.int -> !torch.int
    %int2_653 = torch.constant.int 2
    %555 = torch.aten.mul.int %554, %int2_653 : !torch.int, !torch.int -> !torch.int
    %int32_654 = torch.constant.int 32
    %int4_655 = torch.constant.int 4
    %int32_656 = torch.constant.int 32
    %556 = torch.prim.ListConstruct %555, %int32_654, %int4_655, %int32_656 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %557 = torch.aten.view %553, %556 : !torch.vtensor<[?,3,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,32,4,32],f16>
    torch.bind_symbolic_shape %557, [%32], affine_map<()[s0] -> (s0 * 6, 32, 4, 32)> : !torch.vtensor<[?,32,4,32],f16>
    %558 = torch.prim.ListConstruct %551 : (!torch.vtensor<[?],si64>) -> !torch.list<optional<vtensor>>
    %false_657 = torch.constant.bool false
    %559 = torch.aten.index_put %557, %558, %548, %false_657 : !torch.vtensor<[?,32,4,32],f16>, !torch.list<optional<vtensor>>, !torch.vtensor<[?,32,4,32],f16>, !torch.bool -> !torch.vtensor<[?,32,4,32],f16>
    torch.bind_symbolic_shape %559, [%32], affine_map<()[s0] -> (s0 * 6, 32, 4, 32)> : !torch.vtensor<[?,32,4,32],f16>
    %int3_658 = torch.constant.int 3
    %int2_659 = torch.constant.int 2
    %int32_660 = torch.constant.int 32
    %int4_661 = torch.constant.int 4
    %int32_662 = torch.constant.int 32
    %560 = torch.prim.ListConstruct %132, %int3_658, %int2_659, %int32_660, %int4_661, %int32_662 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %561 = torch.aten.view %559, %560 : !torch.vtensor<[?,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,4,32],f16>
    torch.bind_symbolic_shape %561, [%32], affine_map<()[s0] -> (s0, 3, 2, 32, 4, 32)> : !torch.vtensor<[?,3,2,32,4,32],f16>
    %int24576_663 = torch.constant.int 24576
    %562 = torch.prim.ListConstruct %132, %int24576_663 : (!torch.int, !torch.int) -> !torch.list<int>
    %563 = torch.aten.view %561, %562 : !torch.vtensor<[?,3,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,24576],f16>
    torch.bind_symbolic_shape %563, [%32], affine_map<()[s0] -> (s0, 24576)> : !torch.vtensor<[?,24576],f16>
    %int3_664 = torch.constant.int 3
    %int2_665 = torch.constant.int 2
    %int32_666 = torch.constant.int 32
    %int4_667 = torch.constant.int 4
    %int32_668 = torch.constant.int 32
    %564 = torch.prim.ListConstruct %132, %int3_664, %int2_665, %int32_666, %int4_667, %int32_668 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %565 = torch.aten.view %563, %564 : !torch.vtensor<[?,24576],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,4,32],f16>
    torch.bind_symbolic_shape %565, [%32], affine_map<()[s0] -> (s0, 3, 2, 32, 4, 32)> : !torch.vtensor<[?,3,2,32,4,32],f16>
    %int32_669 = torch.constant.int 32
    %int4_670 = torch.constant.int 4
    %int32_671 = torch.constant.int 32
    %566 = torch.prim.ListConstruct %555, %int32_669, %int4_670, %int32_671 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %567 = torch.aten.view %565, %566 : !torch.vtensor<[?,3,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,32,4,32],f16>
    torch.bind_symbolic_shape %567, [%32], affine_map<()[s0] -> (s0 * 6, 32, 4, 32)> : !torch.vtensor<[?,32,4,32],f16>
    %int4_672 = torch.constant.int 4
    %int32_673 = torch.constant.int 32
    %int4_674 = torch.constant.int 4
    %int32_675 = torch.constant.int 32
    %568 = torch.prim.ListConstruct %int4_672, %141, %int32_673, %int4_674, %int32_675 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %569 = torch.aten.view %485, %568 : !torch.vtensor<[4,?,4,32],f16>, !torch.list<int> -> !torch.vtensor<[4,?,32,4,32],f16>
    torch.bind_symbolic_shape %569, [%31], affine_map<()[s0] -> (4, s0, 32, 4, 32)> : !torch.vtensor<[4,?,32,4,32],f16>
    %int4_676 = torch.constant.int 4
    %570 = torch.aten.mul.int %int4_676, %141 : !torch.int, !torch.int -> !torch.int
    %int32_677 = torch.constant.int 32
    %int4_678 = torch.constant.int 4
    %int32_679 = torch.constant.int 32
    %571 = torch.prim.ListConstruct %570, %int32_677, %int4_678, %int32_679 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %572 = torch.aten.view %569, %571 : !torch.vtensor<[4,?,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,32,4,32],f16>
    torch.bind_symbolic_shape %572, [%31], affine_map<()[s0] -> (s0 * 4, 32, 4, 32)> : !torch.vtensor<[?,32,4,32],f16>
    %int1_680 = torch.constant.int 1
    %int1_681 = torch.constant.int 1
    %573 = torch.aten.add.Scalar %543, %int1_680, %int1_681 : !torch.vtensor<[4,?],si64>, !torch.int, !torch.int -> !torch.vtensor<[4,?],si64>
    torch.bind_symbolic_shape %573, [%31], affine_map<()[s0] -> (4, s0)> : !torch.vtensor<[4,?],si64>
    %int4_682 = torch.constant.int 4
    %574 = torch.aten.mul.int %int4_682, %141 : !torch.int, !torch.int -> !torch.int
    %575 = torch.prim.ListConstruct %574 : (!torch.int) -> !torch.list<int>
    %576 = torch.aten.view %573, %575 : !torch.vtensor<[4,?],si64>, !torch.list<int> -> !torch.vtensor<[?],si64>
    torch.bind_symbolic_shape %576, [%31], affine_map<()[s0] -> (s0 * 4)> : !torch.vtensor<[?],si64>
    %577 = torch.prim.ListConstruct %576 : (!torch.vtensor<[?],si64>) -> !torch.list<optional<vtensor>>
    %false_683 = torch.constant.bool false
    %578 = torch.aten.index_put %567, %577, %572, %false_683 : !torch.vtensor<[?,32,4,32],f16>, !torch.list<optional<vtensor>>, !torch.vtensor<[?,32,4,32],f16>, !torch.bool -> !torch.vtensor<[?,32,4,32],f16>
    torch.bind_symbolic_shape %578, [%32], affine_map<()[s0] -> (s0 * 6, 32, 4, 32)> : !torch.vtensor<[?,32,4,32],f16>
    %int3_684 = torch.constant.int 3
    %int2_685 = torch.constant.int 2
    %int32_686 = torch.constant.int 32
    %int4_687 = torch.constant.int 4
    %int32_688 = torch.constant.int 32
    %579 = torch.prim.ListConstruct %132, %int3_684, %int2_685, %int32_686, %int4_687, %int32_688 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %580 = torch.aten.view %578, %579 : !torch.vtensor<[?,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,4,32],f16>
    torch.bind_symbolic_shape %580, [%32], affine_map<()[s0] -> (s0, 3, 2, 32, 4, 32)> : !torch.vtensor<[?,3,2,32,4,32],f16>
    %int24576_689 = torch.constant.int 24576
    %581 = torch.prim.ListConstruct %132, %int24576_689 : (!torch.int, !torch.int) -> !torch.list<int>
    %582 = torch.aten.view %580, %581 : !torch.vtensor<[?,3,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,24576],f16>
    torch.overwrite.tensor.contents %582 overwrites %arg3 : !torch.vtensor<[?,24576],f16>, !torch.tensor<[?,24576],f16>
    torch.bind_symbolic_shape %582, [%32], affine_map<()[s0] -> (s0, 24576)> : !torch.vtensor<[?,24576],f16>
    %int-2_690 = torch.constant.int -2
    %583 = torch.aten.unsqueeze %541, %int-2_690 : !torch.vtensor<[4,?,4,32],f16>, !torch.int -> !torch.vtensor<[4,?,4,1,32],f16>
    torch.bind_symbolic_shape %583, [%31], affine_map<()[s0] -> (4, s0 * 32, 4, 1, 32)> : !torch.vtensor<[4,?,4,1,32],f16>
    %int4_691 = torch.constant.int 4
    %int4_692 = torch.constant.int 4
    %int2_693 = torch.constant.int 2
    %int32_694 = torch.constant.int 32
    %584 = torch.prim.ListConstruct %int4_691, %526, %int4_692, %int2_693, %int32_694 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_695 = torch.constant.bool false
    %585 = torch.aten.expand %583, %584, %false_695 : !torch.vtensor<[4,?,4,1,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[4,?,4,2,32],f16>
    torch.bind_symbolic_shape %585, [%31], affine_map<()[s0] -> (4, s0 * 32, 4, 2, 32)> : !torch.vtensor<[4,?,4,2,32],f16>
    %int0_696 = torch.constant.int 0
    %586 = torch.aten.clone %585, %int0_696 : !torch.vtensor<[4,?,4,2,32],f16>, !torch.int -> !torch.vtensor<[4,?,4,2,32],f16>
    torch.bind_symbolic_shape %586, [%31], affine_map<()[s0] -> (4, s0 * 32, 4, 2, 32)> : !torch.vtensor<[4,?,4,2,32],f16>
    %int4_697 = torch.constant.int 4
    %int8_698 = torch.constant.int 8
    %int32_699 = torch.constant.int 32
    %587 = torch.prim.ListConstruct %int4_697, %526, %int8_698, %int32_699 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %588 = torch.aten._unsafe_view %586, %587 : !torch.vtensor<[4,?,4,2,32],f16>, !torch.list<int> -> !torch.vtensor<[4,?,8,32],f16>
    torch.bind_symbolic_shape %588, [%31], affine_map<()[s0] -> (4, s0 * 32, 8, 32)> : !torch.vtensor<[4,?,8,32],f16>
    %int-2_700 = torch.constant.int -2
    %589 = torch.aten.unsqueeze %485, %int-2_700 : !torch.vtensor<[4,?,4,32],f16>, !torch.int -> !torch.vtensor<[4,?,4,1,32],f16>
    torch.bind_symbolic_shape %589, [%31], affine_map<()[s0] -> (4, s0 * 32, 4, 1, 32)> : !torch.vtensor<[4,?,4,1,32],f16>
    %int1_701 = torch.constant.int 1
    %590 = torch.aten.size.int %479, %int1_701 : !torch.vtensor<[4,?,128],f16>, !torch.int -> !torch.int
    %int4_702 = torch.constant.int 4
    %int4_703 = torch.constant.int 4
    %int2_704 = torch.constant.int 2
    %int32_705 = torch.constant.int 32
    %591 = torch.prim.ListConstruct %int4_702, %590, %int4_703, %int2_704, %int32_705 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_706 = torch.constant.bool false
    %592 = torch.aten.expand %589, %591, %false_706 : !torch.vtensor<[4,?,4,1,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[4,?,4,2,32],f16>
    torch.bind_symbolic_shape %592, [%31], affine_map<()[s0] -> (4, s0 * 32, 4, 2, 32)> : !torch.vtensor<[4,?,4,2,32],f16>
    %int0_707 = torch.constant.int 0
    %593 = torch.aten.clone %592, %int0_707 : !torch.vtensor<[4,?,4,2,32],f16>, !torch.int -> !torch.vtensor<[4,?,4,2,32],f16>
    torch.bind_symbolic_shape %593, [%31], affine_map<()[s0] -> (4, s0 * 32, 4, 2, 32)> : !torch.vtensor<[4,?,4,2,32],f16>
    %int4_708 = torch.constant.int 4
    %int8_709 = torch.constant.int 8
    %int32_710 = torch.constant.int 32
    %594 = torch.prim.ListConstruct %int4_708, %590, %int8_709, %int32_710 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %595 = torch.aten._unsafe_view %593, %594 : !torch.vtensor<[4,?,4,2,32],f16>, !torch.list<int> -> !torch.vtensor<[4,?,8,32],f16>
    torch.bind_symbolic_shape %595, [%31], affine_map<()[s0] -> (4, s0 * 32, 8, 32)> : !torch.vtensor<[4,?,8,32],f16>
    %int1_711 = torch.constant.int 1
    %int2_712 = torch.constant.int 2
    %596 = torch.aten.transpose.int %513, %int1_711, %int2_712 : !torch.vtensor<[4,?,8,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[4,8,?,32],f16>
    torch.bind_symbolic_shape %596, [%31], affine_map<()[s0] -> (4, 8, s0 * 32, 32)> : !torch.vtensor<[4,8,?,32],f16>
    %int1_713 = torch.constant.int 1
    %int2_714 = torch.constant.int 2
    %597 = torch.aten.transpose.int %588, %int1_713, %int2_714 : !torch.vtensor<[4,?,8,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[4,8,?,32],f16>
    torch.bind_symbolic_shape %597, [%31], affine_map<()[s0] -> (4, 8, s0 * 32, 32)> : !torch.vtensor<[4,8,?,32],f16>
    %int1_715 = torch.constant.int 1
    %int2_716 = torch.constant.int 2
    %598 = torch.aten.transpose.int %595, %int1_715, %int2_716 : !torch.vtensor<[4,?,8,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[4,8,?,32],f16>
    torch.bind_symbolic_shape %598, [%31], affine_map<()[s0] -> (4, 8, s0 * 32, 32)> : !torch.vtensor<[4,8,?,32],f16>
    %float0.000000e00_717 = torch.constant.float 0.000000e+00
    %true_718 = torch.constant.bool true
    %none_719 = torch.constant.none
    %none_720 = torch.constant.none
    %599:2 = torch.operator "torch.aten._scaled_dot_product_flash_attention_for_cpu"(%596, %597, %598, %float0.000000e00_717, %true_718, %none_719, %none_720) : (!torch.vtensor<[4,8,?,32],f16>, !torch.vtensor<[4,8,?,32],f16>, !torch.vtensor<[4,8,?,32],f16>, !torch.float, !torch.bool, !torch.none, !torch.none) -> (!torch.vtensor<[4,8,?,32],f16>, !torch.vtensor<[4,8,?],f32>) 
    torch.bind_symbolic_shape %599#0, [%31], affine_map<()[s0] -> (4, 8, s0 * 32, 32)> : !torch.vtensor<[4,8,?,32],f16>
    %int1_721 = torch.constant.int 1
    %int2_722 = torch.constant.int 2
    %600 = torch.aten.transpose.int %599#0, %int1_721, %int2_722 : !torch.vtensor<[4,8,?,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[4,?,8,32],f16>
    torch.bind_symbolic_shape %600, [%31], affine_map<()[s0] -> (4, s0 * 32, 8, 32)> : !torch.vtensor<[4,?,8,32],f16>
    %int4_723 = torch.constant.int 4
    %int256_724 = torch.constant.int 256
    %601 = torch.prim.ListConstruct %int4_723, %498, %int256_724 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %602 = torch.aten.view %600, %601 : !torch.vtensor<[4,?,8,32],f16>, !torch.list<int> -> !torch.vtensor<[4,?,256],f16>
    torch.bind_symbolic_shape %602, [%31], affine_map<()[s0] -> (4, s0 * 32, 256)> : !torch.vtensor<[4,?,256],f16>
    %int5_725 = torch.constant.int 5
    %603 = torch.prims.convert_element_type %23, %int5_725 : !torch.vtensor<[256,256],f32>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int-2_726 = torch.constant.int -2
    %int-1_727 = torch.constant.int -1
    %604 = torch.aten.transpose.int %603, %int-2_726, %int-1_727 : !torch.vtensor<[256,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,256],f16>
    %int4_728 = torch.constant.int 4
    %605 = torch.aten.mul.int %int4_728, %498 : !torch.int, !torch.int -> !torch.int
    %int256_729 = torch.constant.int 256
    %606 = torch.prim.ListConstruct %605, %int256_729 : (!torch.int, !torch.int) -> !torch.list<int>
    %607 = torch.aten.view %602, %606 : !torch.vtensor<[4,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %607, [%31], affine_map<()[s0] -> (s0 * 128, 256)> : !torch.vtensor<[?,256],f16>
    %608 = torch.aten.mm %607, %604 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,256],f16> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %608, [%31], affine_map<()[s0] -> (s0 * 128, 256)> : !torch.vtensor<[?,256],f16>
    %int4_730 = torch.constant.int 4
    %int256_731 = torch.constant.int 256
    %609 = torch.prim.ListConstruct %int4_730, %498, %int256_731 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %610 = torch.aten.view %608, %609 : !torch.vtensor<[?,256],f16>, !torch.list<int> -> !torch.vtensor<[4,?,256],f16>
    torch.bind_symbolic_shape %610, [%31], affine_map<()[s0] -> (4, s0 * 32, 256)> : !torch.vtensor<[4,?,256],f16>
    %int1_732 = torch.constant.int 1
    %611 = torch.aten.add.Tensor %445, %610, %int1_732 : !torch.vtensor<[4,?,256],f16>, !torch.vtensor<[4,?,256],f16>, !torch.int -> !torch.vtensor<[4,?,256],f16>
    torch.bind_symbolic_shape %611, [%31], affine_map<()[s0] -> (4, s0 * 32, 256)> : !torch.vtensor<[4,?,256],f16>
    %int6_733 = torch.constant.int 6
    %612 = torch.prims.convert_element_type %611, %int6_733 : !torch.vtensor<[4,?,256],f16>, !torch.int -> !torch.vtensor<[4,?,256],f32>
    torch.bind_symbolic_shape %612, [%31], affine_map<()[s0] -> (4, s0 * 32, 256)> : !torch.vtensor<[4,?,256],f32>
    %int2_734 = torch.constant.int 2
    %613 = torch.aten.pow.Tensor_Scalar %612, %int2_734 : !torch.vtensor<[4,?,256],f32>, !torch.int -> !torch.vtensor<[4,?,256],f32>
    torch.bind_symbolic_shape %613, [%31], affine_map<()[s0] -> (4, s0 * 32, 256)> : !torch.vtensor<[4,?,256],f32>
    %int-1_735 = torch.constant.int -1
    %614 = torch.prim.ListConstruct %int-1_735 : (!torch.int) -> !torch.list<int>
    %true_736 = torch.constant.bool true
    %none_737 = torch.constant.none
    %615 = torch.aten.mean.dim %613, %614, %true_736, %none_737 : !torch.vtensor<[4,?,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[4,?,1],f32>
    torch.bind_symbolic_shape %615, [%31], affine_map<()[s0] -> (4, s0 * 32, 1)> : !torch.vtensor<[4,?,1],f32>
    %float1.000000e-02_738 = torch.constant.float 1.000000e-02
    %int1_739 = torch.constant.int 1
    %616 = torch.aten.add.Scalar %615, %float1.000000e-02_738, %int1_739 : !torch.vtensor<[4,?,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[4,?,1],f32>
    torch.bind_symbolic_shape %616, [%31], affine_map<()[s0] -> (4, s0 * 32, 1)> : !torch.vtensor<[4,?,1],f32>
    %617 = torch.aten.rsqrt %616 : !torch.vtensor<[4,?,1],f32> -> !torch.vtensor<[4,?,1],f32>
    torch.bind_symbolic_shape %617, [%31], affine_map<()[s0] -> (4, s0 * 32, 1)> : !torch.vtensor<[4,?,1],f32>
    %618 = torch.aten.mul.Tensor %612, %617 : !torch.vtensor<[4,?,256],f32>, !torch.vtensor<[4,?,1],f32> -> !torch.vtensor<[4,?,256],f32>
    torch.bind_symbolic_shape %618, [%31], affine_map<()[s0] -> (4, s0 * 32, 256)> : !torch.vtensor<[4,?,256],f32>
    %int5_740 = torch.constant.int 5
    %619 = torch.prims.convert_element_type %618, %int5_740 : !torch.vtensor<[4,?,256],f32>, !torch.int -> !torch.vtensor<[4,?,256],f16>
    torch.bind_symbolic_shape %619, [%31], affine_map<()[s0] -> (4, s0 * 32, 256)> : !torch.vtensor<[4,?,256],f16>
    %620 = torch.aten.mul.Tensor %24, %619 : !torch.vtensor<[256],f32>, !torch.vtensor<[4,?,256],f16> -> !torch.vtensor<[4,?,256],f32>
    torch.bind_symbolic_shape %620, [%31], affine_map<()[s0] -> (4, s0 * 32, 256)> : !torch.vtensor<[4,?,256],f32>
    %int5_741 = torch.constant.int 5
    %621 = torch.prims.convert_element_type %620, %int5_741 : !torch.vtensor<[4,?,256],f32>, !torch.int -> !torch.vtensor<[4,?,256],f16>
    torch.bind_symbolic_shape %621, [%31], affine_map<()[s0] -> (4, s0 * 32, 256)> : !torch.vtensor<[4,?,256],f16>
    %int5_742 = torch.constant.int 5
    %622 = torch.prims.convert_element_type %25, %int5_742 : !torch.vtensor<[23,256],f32>, !torch.int -> !torch.vtensor<[23,256],f16>
    %int-2_743 = torch.constant.int -2
    %int-1_744 = torch.constant.int -1
    %623 = torch.aten.transpose.int %622, %int-2_743, %int-1_744 : !torch.vtensor<[23,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,23],f16>
    %int4_745 = torch.constant.int 4
    %624 = torch.aten.mul.int %int4_745, %47 : !torch.int, !torch.int -> !torch.int
    %int256_746 = torch.constant.int 256
    %625 = torch.prim.ListConstruct %624, %int256_746 : (!torch.int, !torch.int) -> !torch.list<int>
    %626 = torch.aten.view %621, %625 : !torch.vtensor<[4,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %626, [%31], affine_map<()[s0] -> (s0 * 128, 256)> : !torch.vtensor<[?,256],f16>
    %627 = torch.aten.mm %626, %623 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,23],f16> -> !torch.vtensor<[?,23],f16>
    torch.bind_symbolic_shape %627, [%31], affine_map<()[s0] -> (s0 * 128, 23)> : !torch.vtensor<[?,23],f16>
    %int4_747 = torch.constant.int 4
    %int23_748 = torch.constant.int 23
    %628 = torch.prim.ListConstruct %int4_747, %47, %int23_748 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %629 = torch.aten.view %627, %628 : !torch.vtensor<[?,23],f16>, !torch.list<int> -> !torch.vtensor<[4,?,23],f16>
    torch.bind_symbolic_shape %629, [%31], affine_map<()[s0] -> (4, s0 * 32, 23)> : !torch.vtensor<[4,?,23],f16>
    %630 = torch.aten.silu %629 : !torch.vtensor<[4,?,23],f16> -> !torch.vtensor<[4,?,23],f16>
    torch.bind_symbolic_shape %630, [%31], affine_map<()[s0] -> (4, s0 * 32, 23)> : !torch.vtensor<[4,?,23],f16>
    %int5_749 = torch.constant.int 5
    %631 = torch.prims.convert_element_type %26, %int5_749 : !torch.vtensor<[23,256],f32>, !torch.int -> !torch.vtensor<[23,256],f16>
    %int-2_750 = torch.constant.int -2
    %int-1_751 = torch.constant.int -1
    %632 = torch.aten.transpose.int %631, %int-2_750, %int-1_751 : !torch.vtensor<[23,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,23],f16>
    %int4_752 = torch.constant.int 4
    %633 = torch.aten.mul.int %int4_752, %47 : !torch.int, !torch.int -> !torch.int
    %int256_753 = torch.constant.int 256
    %634 = torch.prim.ListConstruct %633, %int256_753 : (!torch.int, !torch.int) -> !torch.list<int>
    %635 = torch.aten.view %621, %634 : !torch.vtensor<[4,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %635, [%31], affine_map<()[s0] -> (s0 * 128, 256)> : !torch.vtensor<[?,256],f16>
    %636 = torch.aten.mm %635, %632 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,23],f16> -> !torch.vtensor<[?,23],f16>
    torch.bind_symbolic_shape %636, [%31], affine_map<()[s0] -> (s0 * 128, 23)> : !torch.vtensor<[?,23],f16>
    %int4_754 = torch.constant.int 4
    %int23_755 = torch.constant.int 23
    %637 = torch.prim.ListConstruct %int4_754, %47, %int23_755 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %638 = torch.aten.view %636, %637 : !torch.vtensor<[?,23],f16>, !torch.list<int> -> !torch.vtensor<[4,?,23],f16>
    torch.bind_symbolic_shape %638, [%31], affine_map<()[s0] -> (4, s0 * 32, 23)> : !torch.vtensor<[4,?,23],f16>
    %639 = torch.aten.mul.Tensor %630, %638 : !torch.vtensor<[4,?,23],f16>, !torch.vtensor<[4,?,23],f16> -> !torch.vtensor<[4,?,23],f16>
    torch.bind_symbolic_shape %639, [%31], affine_map<()[s0] -> (4, s0 * 32, 23)> : !torch.vtensor<[4,?,23],f16>
    %int5_756 = torch.constant.int 5
    %640 = torch.prims.convert_element_type %27, %int5_756 : !torch.vtensor<[256,23],f32>, !torch.int -> !torch.vtensor<[256,23],f16>
    %int-2_757 = torch.constant.int -2
    %int-1_758 = torch.constant.int -1
    %641 = torch.aten.transpose.int %640, %int-2_757, %int-1_758 : !torch.vtensor<[256,23],f16>, !torch.int, !torch.int -> !torch.vtensor<[23,256],f16>
    %int1_759 = torch.constant.int 1
    %642 = torch.aten.size.int %629, %int1_759 : !torch.vtensor<[4,?,23],f16>, !torch.int -> !torch.int
    %int4_760 = torch.constant.int 4
    %643 = torch.aten.mul.int %int4_760, %642 : !torch.int, !torch.int -> !torch.int
    %int23_761 = torch.constant.int 23
    %644 = torch.prim.ListConstruct %643, %int23_761 : (!torch.int, !torch.int) -> !torch.list<int>
    %645 = torch.aten.view %639, %644 : !torch.vtensor<[4,?,23],f16>, !torch.list<int> -> !torch.vtensor<[?,23],f16>
    torch.bind_symbolic_shape %645, [%31], affine_map<()[s0] -> (s0 * 128, 23)> : !torch.vtensor<[?,23],f16>
    %646 = torch.aten.mm %645, %641 : !torch.vtensor<[?,23],f16>, !torch.vtensor<[23,256],f16> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %646, [%31], affine_map<()[s0] -> (s0 * 128, 256)> : !torch.vtensor<[?,256],f16>
    %int4_762 = torch.constant.int 4
    %int256_763 = torch.constant.int 256
    %647 = torch.prim.ListConstruct %int4_762, %642, %int256_763 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %648 = torch.aten.view %646, %647 : !torch.vtensor<[?,256],f16>, !torch.list<int> -> !torch.vtensor<[4,?,256],f16>
    torch.bind_symbolic_shape %648, [%31], affine_map<()[s0] -> (4, s0 * 32, 256)> : !torch.vtensor<[4,?,256],f16>
    %int1_764 = torch.constant.int 1
    %649 = torch.aten.add.Tensor %611, %648, %int1_764 : !torch.vtensor<[4,?,256],f16>, !torch.vtensor<[4,?,256],f16>, !torch.int -> !torch.vtensor<[4,?,256],f16>
    torch.bind_symbolic_shape %649, [%31], affine_map<()[s0] -> (4, s0 * 32, 256)> : !torch.vtensor<[4,?,256],f16>
    %int6_765 = torch.constant.int 6
    %650 = torch.prims.convert_element_type %649, %int6_765 : !torch.vtensor<[4,?,256],f16>, !torch.int -> !torch.vtensor<[4,?,256],f32>
    torch.bind_symbolic_shape %650, [%31], affine_map<()[s0] -> (4, s0 * 32, 256)> : !torch.vtensor<[4,?,256],f32>
    %int2_766 = torch.constant.int 2
    %651 = torch.aten.pow.Tensor_Scalar %650, %int2_766 : !torch.vtensor<[4,?,256],f32>, !torch.int -> !torch.vtensor<[4,?,256],f32>
    torch.bind_symbolic_shape %651, [%31], affine_map<()[s0] -> (4, s0 * 32, 256)> : !torch.vtensor<[4,?,256],f32>
    %int-1_767 = torch.constant.int -1
    %652 = torch.prim.ListConstruct %int-1_767 : (!torch.int) -> !torch.list<int>
    %true_768 = torch.constant.bool true
    %none_769 = torch.constant.none
    %653 = torch.aten.mean.dim %651, %652, %true_768, %none_769 : !torch.vtensor<[4,?,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[4,?,1],f32>
    torch.bind_symbolic_shape %653, [%31], affine_map<()[s0] -> (4, s0 * 32, 1)> : !torch.vtensor<[4,?,1],f32>
    %float1.000000e-02_770 = torch.constant.float 1.000000e-02
    %int1_771 = torch.constant.int 1
    %654 = torch.aten.add.Scalar %653, %float1.000000e-02_770, %int1_771 : !torch.vtensor<[4,?,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[4,?,1],f32>
    torch.bind_symbolic_shape %654, [%31], affine_map<()[s0] -> (4, s0 * 32, 1)> : !torch.vtensor<[4,?,1],f32>
    %655 = torch.aten.rsqrt %654 : !torch.vtensor<[4,?,1],f32> -> !torch.vtensor<[4,?,1],f32>
    torch.bind_symbolic_shape %655, [%31], affine_map<()[s0] -> (4, s0 * 32, 1)> : !torch.vtensor<[4,?,1],f32>
    %656 = torch.aten.mul.Tensor %650, %655 : !torch.vtensor<[4,?,256],f32>, !torch.vtensor<[4,?,1],f32> -> !torch.vtensor<[4,?,256],f32>
    torch.bind_symbolic_shape %656, [%31], affine_map<()[s0] -> (4, s0 * 32, 256)> : !torch.vtensor<[4,?,256],f32>
    %int5_772 = torch.constant.int 5
    %657 = torch.prims.convert_element_type %656, %int5_772 : !torch.vtensor<[4,?,256],f32>, !torch.int -> !torch.vtensor<[4,?,256],f16>
    torch.bind_symbolic_shape %657, [%31], affine_map<()[s0] -> (4, s0 * 32, 256)> : !torch.vtensor<[4,?,256],f16>
    %658 = torch.aten.mul.Tensor %28, %657 : !torch.vtensor<[1,256],f32>, !torch.vtensor<[4,?,256],f16> -> !torch.vtensor<[4,?,256],f32>
    torch.bind_symbolic_shape %658, [%31], affine_map<()[s0] -> (4, s0 * 32, 256)> : !torch.vtensor<[4,?,256],f32>
    %int5_773 = torch.constant.int 5
    %659 = torch.prims.convert_element_type %658, %int5_773 : !torch.vtensor<[4,?,256],f32>, !torch.int -> !torch.vtensor<[4,?,256],f16>
    torch.bind_symbolic_shape %659, [%31], affine_map<()[s0] -> (4, s0 * 32, 256)> : !torch.vtensor<[4,?,256],f16>
    %int5_774 = torch.constant.int 5
    %660 = torch.prims.convert_element_type %29, %int5_774 : !torch.vtensor<[256,256],f32>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int-2_775 = torch.constant.int -2
    %int-1_776 = torch.constant.int -1
    %661 = torch.aten.transpose.int %660, %int-2_775, %int-1_776 : !torch.vtensor<[256,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,256],f16>
    %int4_777 = torch.constant.int 4
    %662 = torch.aten.mul.int %int4_777, %47 : !torch.int, !torch.int -> !torch.int
    %int256_778 = torch.constant.int 256
    %663 = torch.prim.ListConstruct %662, %int256_778 : (!torch.int, !torch.int) -> !torch.list<int>
    %664 = torch.aten.view %659, %663 : !torch.vtensor<[4,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %664, [%31], affine_map<()[s0] -> (s0 * 128, 256)> : !torch.vtensor<[?,256],f16>
    %665 = torch.aten.mm %664, %661 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,256],f16> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %665, [%31], affine_map<()[s0] -> (s0 * 128, 256)> : !torch.vtensor<[?,256],f16>
    %int4_779 = torch.constant.int 4
    %int256_780 = torch.constant.int 256
    %666 = torch.prim.ListConstruct %int4_779, %47, %int256_780 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %667 = torch.aten.view %665, %666 : !torch.vtensor<[?,256],f16>, !torch.list<int> -> !torch.vtensor<[4,?,256],f16>
    torch.bind_symbolic_shape %667, [%31], affine_map<()[s0] -> (4, s0 * 32, 256)> : !torch.vtensor<[4,?,256],f16>
    return %667 : !torch.vtensor<[4,?,256],f16>
  }
  func.func @decode_bs4(%arg0: !torch.vtensor<[4,1],si64>, %arg1: !torch.vtensor<[4],si64>, %arg2: !torch.vtensor<[4],si64>, %arg3: !torch.vtensor<[4,?],si64>, %arg4: !torch.tensor<[?,24576],f16>) -> !torch.vtensor<[4,1,256],f16> attributes {torch.assume_strict_symbolic_shapes} {
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
    torch.bind_symbolic_shape %arg3, [%37], affine_map<()[s0] -> (4, s0)> : !torch.vtensor<[4,?],si64>
    torch.bind_symbolic_shape %36, [%38], affine_map<()[s0] -> (s0, 24576)> : !torch.vtensor<[?,24576],f16>
    %int1 = torch.constant.int 1
    %39 = torch.aten.size.int %arg3, %int1 : !torch.vtensor<[4,?],si64>, !torch.int -> !torch.int
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
    %42 = torch.aten.unsqueeze %arg1, %int-1 : !torch.vtensor<[4],si64>, !torch.int -> !torch.vtensor<[4,1],si64>
    %43 = torch.aten.ge.Tensor %41, %42 : !torch.vtensor<[?],si64>, !torch.vtensor<[4,1],si64> -> !torch.vtensor<[4,?],i1>
    torch.bind_symbolic_shape %43, [%37], affine_map<()[s0] -> (4, s0 * 32)> : !torch.vtensor<[4,?],i1>
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
    %46 = torch.aten.where.self %43, %45, %44 : !torch.vtensor<[4,?],i1>, !torch.vtensor<[],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[4,?],f32>
    torch.bind_symbolic_shape %46, [%37], affine_map<()[s0] -> (4, s0 * 32)> : !torch.vtensor<[4,?],f32>
    %int5 = torch.constant.int 5
    %47 = torch.prims.convert_element_type %46, %int5 : !torch.vtensor<[4,?],f32>, !torch.int -> !torch.vtensor<[4,?],f16>
    torch.bind_symbolic_shape %47, [%37], affine_map<()[s0] -> (4, s0 * 32)> : !torch.vtensor<[4,?],f16>
    %int1_10 = torch.constant.int 1
    %48 = torch.aten.unsqueeze %47, %int1_10 : !torch.vtensor<[4,?],f16>, !torch.int -> !torch.vtensor<[4,1,?],f16>
    torch.bind_symbolic_shape %48, [%37], affine_map<()[s0] -> (4, 1, s0 * 32)> : !torch.vtensor<[4,1,?],f16>
    %int1_11 = torch.constant.int 1
    %49 = torch.aten.unsqueeze %48, %int1_11 : !torch.vtensor<[4,1,?],f16>, !torch.int -> !torch.vtensor<[4,1,1,?],f16>
    torch.bind_symbolic_shape %49, [%37], affine_map<()[s0] -> (4, 1, 1, s0 * 32)> : !torch.vtensor<[4,1,1,?],f16>
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
    %52 = torch.aten.unsqueeze %arg2, %int1_19 : !torch.vtensor<[4],si64>, !torch.int -> !torch.vtensor<[4,1],si64>
    %int1_20 = torch.constant.int 1
    %53 = torch.aten.add.Tensor %51, %52, %int1_20 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[4,1],si64>, !torch.int -> !torch.vtensor<[4,1],si64>
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
    %int4 = torch.constant.int 4
    %66 = torch.prim.ListConstruct %int4 : (!torch.int) -> !torch.list<int>
    %67 = torch.aten.view %53, %66 : !torch.vtensor<[4,1],si64>, !torch.list<int> -> !torch.vtensor<[4],si64>
    %68 = torch.prim.ListConstruct %67 : (!torch.vtensor<[4],si64>) -> !torch.list<optional<vtensor>>
    %69 = torch.aten.index.Tensor %65, %68 : !torch.vtensor<[128,32],f32>, !torch.list<optional<vtensor>> -> !torch.vtensor<[4,32],f32>
    %int1_35 = torch.constant.int 1
    %70 = torch.aten.unsqueeze %69, %int1_35 : !torch.vtensor<[4,32],f32>, !torch.int -> !torch.vtensor<[4,1,32],f32>
    %int5_36 = torch.constant.int 5
    %71 = torch.prims.convert_element_type %0, %int5_36 : !torch.vtensor<[256,256],f32>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int-1_37 = torch.constant.int -1
    %false_38 = torch.constant.bool false
    %false_39 = torch.constant.bool false
    %72 = torch.aten.embedding %71, %arg0, %int-1_37, %false_38, %false_39 : !torch.vtensor<[256,256],f16>, !torch.vtensor<[4,1],si64>, !torch.int, !torch.bool, !torch.bool -> !torch.vtensor<[4,1,256],f16>
    %int6_40 = torch.constant.int 6
    %73 = torch.prims.convert_element_type %72, %int6_40 : !torch.vtensor<[4,1,256],f16>, !torch.int -> !torch.vtensor<[4,1,256],f32>
    %int2_41 = torch.constant.int 2
    %74 = torch.aten.pow.Tensor_Scalar %73, %int2_41 : !torch.vtensor<[4,1,256],f32>, !torch.int -> !torch.vtensor<[4,1,256],f32>
    %int-1_42 = torch.constant.int -1
    %75 = torch.prim.ListConstruct %int-1_42 : (!torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %none_43 = torch.constant.none
    %76 = torch.aten.mean.dim %74, %75, %true, %none_43 : !torch.vtensor<[4,1,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[4,1,1],f32>
    %float1.000000e-02 = torch.constant.float 1.000000e-02
    %int1_44 = torch.constant.int 1
    %77 = torch.aten.add.Scalar %76, %float1.000000e-02, %int1_44 : !torch.vtensor<[4,1,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[4,1,1],f32>
    %78 = torch.aten.rsqrt %77 : !torch.vtensor<[4,1,1],f32> -> !torch.vtensor<[4,1,1],f32>
    %79 = torch.aten.mul.Tensor %73, %78 : !torch.vtensor<[4,1,256],f32>, !torch.vtensor<[4,1,1],f32> -> !torch.vtensor<[4,1,256],f32>
    %int5_45 = torch.constant.int 5
    %80 = torch.prims.convert_element_type %79, %int5_45 : !torch.vtensor<[4,1,256],f32>, !torch.int -> !torch.vtensor<[4,1,256],f16>
    %81 = torch.aten.mul.Tensor %1, %80 : !torch.vtensor<[256],f32>, !torch.vtensor<[4,1,256],f16> -> !torch.vtensor<[4,1,256],f32>
    %int5_46 = torch.constant.int 5
    %82 = torch.prims.convert_element_type %81, %int5_46 : !torch.vtensor<[4,1,256],f32>, !torch.int -> !torch.vtensor<[4,1,256],f16>
    %int5_47 = torch.constant.int 5
    %83 = torch.prims.convert_element_type %2, %int5_47 : !torch.vtensor<[256,256],f32>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int-2 = torch.constant.int -2
    %int-1_48 = torch.constant.int -1
    %84 = torch.aten.transpose.int %83, %int-2, %int-1_48 : !torch.vtensor<[256,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,256],f16>
    %int4_49 = torch.constant.int 4
    %int256 = torch.constant.int 256
    %85 = torch.prim.ListConstruct %int4_49, %int256 : (!torch.int, !torch.int) -> !torch.list<int>
    %86 = torch.aten.view %82, %85 : !torch.vtensor<[4,1,256],f16>, !torch.list<int> -> !torch.vtensor<[4,256],f16>
    %87 = torch.aten.mm %86, %84 : !torch.vtensor<[4,256],f16>, !torch.vtensor<[256,256],f16> -> !torch.vtensor<[4,256],f16>
    %int4_50 = torch.constant.int 4
    %int1_51 = torch.constant.int 1
    %int256_52 = torch.constant.int 256
    %88 = torch.prim.ListConstruct %int4_50, %int1_51, %int256_52 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %89 = torch.aten.view %87, %88 : !torch.vtensor<[4,256],f16>, !torch.list<int> -> !torch.vtensor<[4,1,256],f16>
    %int5_53 = torch.constant.int 5
    %90 = torch.prims.convert_element_type %3, %int5_53 : !torch.vtensor<[128,256],f32>, !torch.int -> !torch.vtensor<[128,256],f16>
    %int-2_54 = torch.constant.int -2
    %int-1_55 = torch.constant.int -1
    %91 = torch.aten.transpose.int %90, %int-2_54, %int-1_55 : !torch.vtensor<[128,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,128],f16>
    %int4_56 = torch.constant.int 4
    %int256_57 = torch.constant.int 256
    %92 = torch.prim.ListConstruct %int4_56, %int256_57 : (!torch.int, !torch.int) -> !torch.list<int>
    %93 = torch.aten.view %82, %92 : !torch.vtensor<[4,1,256],f16>, !torch.list<int> -> !torch.vtensor<[4,256],f16>
    %94 = torch.aten.mm %93, %91 : !torch.vtensor<[4,256],f16>, !torch.vtensor<[256,128],f16> -> !torch.vtensor<[4,128],f16>
    %int4_58 = torch.constant.int 4
    %int1_59 = torch.constant.int 1
    %int128_60 = torch.constant.int 128
    %95 = torch.prim.ListConstruct %int4_58, %int1_59, %int128_60 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %96 = torch.aten.view %94, %95 : !torch.vtensor<[4,128],f16>, !torch.list<int> -> !torch.vtensor<[4,1,128],f16>
    %int5_61 = torch.constant.int 5
    %97 = torch.prims.convert_element_type %4, %int5_61 : !torch.vtensor<[128,256],f32>, !torch.int -> !torch.vtensor<[128,256],f16>
    %int-2_62 = torch.constant.int -2
    %int-1_63 = torch.constant.int -1
    %98 = torch.aten.transpose.int %97, %int-2_62, %int-1_63 : !torch.vtensor<[128,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,128],f16>
    %int4_64 = torch.constant.int 4
    %int256_65 = torch.constant.int 256
    %99 = torch.prim.ListConstruct %int4_64, %int256_65 : (!torch.int, !torch.int) -> !torch.list<int>
    %100 = torch.aten.view %82, %99 : !torch.vtensor<[4,1,256],f16>, !torch.list<int> -> !torch.vtensor<[4,256],f16>
    %101 = torch.aten.mm %100, %98 : !torch.vtensor<[4,256],f16>, !torch.vtensor<[256,128],f16> -> !torch.vtensor<[4,128],f16>
    %int4_66 = torch.constant.int 4
    %int1_67 = torch.constant.int 1
    %int128_68 = torch.constant.int 128
    %102 = torch.prim.ListConstruct %int4_66, %int1_67, %int128_68 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %103 = torch.aten.view %101, %102 : !torch.vtensor<[4,128],f16>, !torch.list<int> -> !torch.vtensor<[4,1,128],f16>
    %int4_69 = torch.constant.int 4
    %int1_70 = torch.constant.int 1
    %int8 = torch.constant.int 8
    %int32_71 = torch.constant.int 32
    %104 = torch.prim.ListConstruct %int4_69, %int1_70, %int8, %int32_71 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %105 = torch.aten.view %89, %104 : !torch.vtensor<[4,1,256],f16>, !torch.list<int> -> !torch.vtensor<[4,1,8,32],f16>
    %int4_72 = torch.constant.int 4
    %int1_73 = torch.constant.int 1
    %int4_74 = torch.constant.int 4
    %int32_75 = torch.constant.int 32
    %106 = torch.prim.ListConstruct %int4_72, %int1_73, %int4_74, %int32_75 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %107 = torch.aten.view %96, %106 : !torch.vtensor<[4,1,128],f16>, !torch.list<int> -> !torch.vtensor<[4,1,4,32],f16>
    %int4_76 = torch.constant.int 4
    %int1_77 = torch.constant.int 1
    %int4_78 = torch.constant.int 4
    %int32_79 = torch.constant.int 32
    %108 = torch.prim.ListConstruct %int4_76, %int1_77, %int4_78, %int32_79 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %109 = torch.aten.view %103, %108 : !torch.vtensor<[4,1,128],f16>, !torch.list<int> -> !torch.vtensor<[4,1,4,32],f16>
    %int6_80 = torch.constant.int 6
    %110 = torch.prims.convert_element_type %105, %int6_80 : !torch.vtensor<[4,1,8,32],f16>, !torch.int -> !torch.vtensor<[4,1,8,32],f32>
    %111 = torch_c.to_builtin_tensor %110 : !torch.vtensor<[4,1,8,32],f32> -> tensor<4x1x8x32xf32>
    %112 = torch_c.to_builtin_tensor %70 : !torch.vtensor<[4,1,32],f32> -> tensor<4x1x32xf32>
    %113 = util.call @sharktank_rotary_embedding_4_1_8_32_f32(%111, %112) : (tensor<4x1x8x32xf32>, tensor<4x1x32xf32>) -> tensor<4x1x8x32xf32>
    %114 = torch_c.from_builtin_tensor %113 : tensor<4x1x8x32xf32> -> !torch.vtensor<[4,1,8,32],f32>
    %int5_81 = torch.constant.int 5
    %115 = torch.prims.convert_element_type %114, %int5_81 : !torch.vtensor<[4,1,8,32],f32>, !torch.int -> !torch.vtensor<[4,1,8,32],f16>
    %int6_82 = torch.constant.int 6
    %116 = torch.prims.convert_element_type %107, %int6_82 : !torch.vtensor<[4,1,4,32],f16>, !torch.int -> !torch.vtensor<[4,1,4,32],f32>
    %117 = torch_c.to_builtin_tensor %116 : !torch.vtensor<[4,1,4,32],f32> -> tensor<4x1x4x32xf32>
    %118 = torch_c.to_builtin_tensor %70 : !torch.vtensor<[4,1,32],f32> -> tensor<4x1x32xf32>
    %119 = util.call @sharktank_rotary_embedding_4_1_4_32_f32(%117, %118) : (tensor<4x1x4x32xf32>, tensor<4x1x32xf32>) -> tensor<4x1x4x32xf32>
    %120 = torch_c.from_builtin_tensor %119 : tensor<4x1x4x32xf32> -> !torch.vtensor<[4,1,4,32],f32>
    %int5_83 = torch.constant.int 5
    %121 = torch.prims.convert_element_type %120, %int5_83 : !torch.vtensor<[4,1,4,32],f32>, !torch.int -> !torch.vtensor<[4,1,4,32],f16>
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
    %130 = torch.aten.floor_divide.Scalar %arg2, %int32_94 : !torch.vtensor<[4],si64>, !torch.int -> !torch.vtensor<[4],si64>
    %int1_95 = torch.constant.int 1
    %131 = torch.aten.unsqueeze %130, %int1_95 : !torch.vtensor<[4],si64>, !torch.int -> !torch.vtensor<[4,1],si64>
    %int1_96 = torch.constant.int 1
    %false_97 = torch.constant.bool false
    %132 = torch.aten.gather %arg3, %int1_96, %131, %false_97 : !torch.vtensor<[4,?],si64>, !torch.int, !torch.vtensor<[4,1],si64>, !torch.bool -> !torch.vtensor<[4,1],si64>
    %int32_98 = torch.constant.int 32
    %133 = torch.aten.remainder.Scalar %arg2, %int32_98 : !torch.vtensor<[4],si64>, !torch.int -> !torch.vtensor<[4],si64>
    %int1_99 = torch.constant.int 1
    %134 = torch.aten.unsqueeze %133, %int1_99 : !torch.vtensor<[4],si64>, !torch.int -> !torch.vtensor<[4,1],si64>
    %none_100 = torch.constant.none
    %135 = torch.aten.clone %5, %none_100 : !torch.vtensor<[],si64>, !torch.none -> !torch.vtensor<[],si64>
    %int0_101 = torch.constant.int 0
    %136 = torch.aten.unsqueeze %135, %int0_101 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int4_102 = torch.constant.int 4
    %int1_103 = torch.constant.int 1
    %137 = torch.prim.ListConstruct %int4_102, %int1_103 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_104 = torch.constant.int 1
    %int1_105 = torch.constant.int 1
    %138 = torch.prim.ListConstruct %int1_104, %int1_105 : (!torch.int, !torch.int) -> !torch.list<int>
    %int4_106 = torch.constant.int 4
    %int0_107 = torch.constant.int 0
    %cpu_108 = torch.constant.device "cpu"
    %false_109 = torch.constant.bool false
    %139 = torch.aten.empty_strided %137, %138, %int4_106, %int0_107, %cpu_108, %false_109 : !torch.list<int>, !torch.list<int>, !torch.int, !torch.int, !torch.Device, !torch.bool -> !torch.vtensor<[4,1],si64>
    %int0_110 = torch.constant.int 0
    %140 = torch.aten.fill.Scalar %139, %int0_110 : !torch.vtensor<[4,1],si64>, !torch.int -> !torch.vtensor<[4,1],si64>
    %int4_111 = torch.constant.int 4
    %int1_112 = torch.constant.int 1
    %141 = torch.prim.ListConstruct %int4_111, %int1_112 : (!torch.int, !torch.int) -> !torch.list<int>
    %142 = torch.aten.repeat %136, %141 : !torch.vtensor<[1],si64>, !torch.list<int> -> !torch.vtensor<[4,1],si64>
    %int3_113 = torch.constant.int 3
    %143 = torch.aten.mul.Scalar %132, %int3_113 : !torch.vtensor<[4,1],si64>, !torch.int -> !torch.vtensor<[4,1],si64>
    %int1_114 = torch.constant.int 1
    %144 = torch.aten.add.Tensor %143, %140, %int1_114 : !torch.vtensor<[4,1],si64>, !torch.vtensor<[4,1],si64>, !torch.int -> !torch.vtensor<[4,1],si64>
    %int2_115 = torch.constant.int 2
    %145 = torch.aten.mul.Scalar %144, %int2_115 : !torch.vtensor<[4,1],si64>, !torch.int -> !torch.vtensor<[4,1],si64>
    %int1_116 = torch.constant.int 1
    %146 = torch.aten.add.Tensor %145, %142, %int1_116 : !torch.vtensor<[4,1],si64>, !torch.vtensor<[4,1],si64>, !torch.int -> !torch.vtensor<[4,1],si64>
    %int32_117 = torch.constant.int 32
    %147 = torch.aten.mul.Scalar %146, %int32_117 : !torch.vtensor<[4,1],si64>, !torch.int -> !torch.vtensor<[4,1],si64>
    %int1_118 = torch.constant.int 1
    %148 = torch.aten.add.Tensor %147, %134, %int1_118 : !torch.vtensor<[4,1],si64>, !torch.vtensor<[4,1],si64>, !torch.int -> !torch.vtensor<[4,1],si64>
    %149 = torch.prim.ListConstruct %148 : (!torch.vtensor<[4,1],si64>) -> !torch.list<optional<vtensor>>
    %false_119 = torch.constant.bool false
    %150 = torch.aten.index_put %129, %149, %121, %false_119 : !torch.vtensor<[?,4,32],f16>, !torch.list<optional<vtensor>>, !torch.vtensor<[4,1,4,32],f16>, !torch.bool -> !torch.vtensor<[?,4,32],f16>
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
    %159 = torch.aten.floor_divide.Scalar %arg2, %int32_132 : !torch.vtensor<[4],si64>, !torch.int -> !torch.vtensor<[4],si64>
    %int1_133 = torch.constant.int 1
    %160 = torch.aten.unsqueeze %159, %int1_133 : !torch.vtensor<[4],si64>, !torch.int -> !torch.vtensor<[4,1],si64>
    %int1_134 = torch.constant.int 1
    %false_135 = torch.constant.bool false
    %161 = torch.aten.gather %arg3, %int1_134, %160, %false_135 : !torch.vtensor<[4,?],si64>, !torch.int, !torch.vtensor<[4,1],si64>, !torch.bool -> !torch.vtensor<[4,1],si64>
    %int32_136 = torch.constant.int 32
    %162 = torch.aten.remainder.Scalar %arg2, %int32_136 : !torch.vtensor<[4],si64>, !torch.int -> !torch.vtensor<[4],si64>
    %int1_137 = torch.constant.int 1
    %163 = torch.aten.unsqueeze %162, %int1_137 : !torch.vtensor<[4],si64>, !torch.int -> !torch.vtensor<[4,1],si64>
    %none_138 = torch.constant.none
    %164 = torch.aten.clone %6, %none_138 : !torch.vtensor<[],si64>, !torch.none -> !torch.vtensor<[],si64>
    %int0_139 = torch.constant.int 0
    %165 = torch.aten.unsqueeze %164, %int0_139 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int4_140 = torch.constant.int 4
    %int1_141 = torch.constant.int 1
    %166 = torch.prim.ListConstruct %int4_140, %int1_141 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_142 = torch.constant.int 1
    %int1_143 = torch.constant.int 1
    %167 = torch.prim.ListConstruct %int1_142, %int1_143 : (!torch.int, !torch.int) -> !torch.list<int>
    %int4_144 = torch.constant.int 4
    %int0_145 = torch.constant.int 0
    %cpu_146 = torch.constant.device "cpu"
    %false_147 = torch.constant.bool false
    %168 = torch.aten.empty_strided %166, %167, %int4_144, %int0_145, %cpu_146, %false_147 : !torch.list<int>, !torch.list<int>, !torch.int, !torch.int, !torch.Device, !torch.bool -> !torch.vtensor<[4,1],si64>
    %int0_148 = torch.constant.int 0
    %169 = torch.aten.fill.Scalar %168, %int0_148 : !torch.vtensor<[4,1],si64>, !torch.int -> !torch.vtensor<[4,1],si64>
    %int4_149 = torch.constant.int 4
    %int1_150 = torch.constant.int 1
    %170 = torch.prim.ListConstruct %int4_149, %int1_150 : (!torch.int, !torch.int) -> !torch.list<int>
    %171 = torch.aten.repeat %165, %170 : !torch.vtensor<[1],si64>, !torch.list<int> -> !torch.vtensor<[4,1],si64>
    %int3_151 = torch.constant.int 3
    %172 = torch.aten.mul.Scalar %161, %int3_151 : !torch.vtensor<[4,1],si64>, !torch.int -> !torch.vtensor<[4,1],si64>
    %int1_152 = torch.constant.int 1
    %173 = torch.aten.add.Tensor %172, %169, %int1_152 : !torch.vtensor<[4,1],si64>, !torch.vtensor<[4,1],si64>, !torch.int -> !torch.vtensor<[4,1],si64>
    %int2_153 = torch.constant.int 2
    %174 = torch.aten.mul.Scalar %173, %int2_153 : !torch.vtensor<[4,1],si64>, !torch.int -> !torch.vtensor<[4,1],si64>
    %int1_154 = torch.constant.int 1
    %175 = torch.aten.add.Tensor %174, %171, %int1_154 : !torch.vtensor<[4,1],si64>, !torch.vtensor<[4,1],si64>, !torch.int -> !torch.vtensor<[4,1],si64>
    %int32_155 = torch.constant.int 32
    %176 = torch.aten.mul.Scalar %175, %int32_155 : !torch.vtensor<[4,1],si64>, !torch.int -> !torch.vtensor<[4,1],si64>
    %int1_156 = torch.constant.int 1
    %177 = torch.aten.add.Tensor %176, %163, %int1_156 : !torch.vtensor<[4,1],si64>, !torch.vtensor<[4,1],si64>, !torch.int -> !torch.vtensor<[4,1],si64>
    %178 = torch.prim.ListConstruct %177 : (!torch.vtensor<[4,1],si64>) -> !torch.list<optional<vtensor>>
    %false_157 = torch.constant.bool false
    %179 = torch.aten.index_put %158, %178, %109, %false_157 : !torch.vtensor<[?,4,32],f16>, !torch.list<optional<vtensor>>, !torch.vtensor<[4,1,4,32],f16>, !torch.bool -> !torch.vtensor<[?,4,32],f16>
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
    %int4_164 = torch.constant.int 4
    %184 = torch.prim.ListConstruct %int4_164, %39 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_165 = torch.constant.int 1
    %185 = torch.prim.ListConstruct %39, %int1_165 : (!torch.int, !torch.int) -> !torch.list<int>
    %int4_166 = torch.constant.int 4
    %int0_167 = torch.constant.int 0
    %cpu_168 = torch.constant.device "cpu"
    %false_169 = torch.constant.bool false
    %186 = torch.aten.empty_strided %184, %185, %int4_166, %int0_167, %cpu_168, %false_169 : !torch.list<int>, !torch.list<int>, !torch.int, !torch.int, !torch.Device, !torch.bool -> !torch.vtensor<[4,?],si64>
    torch.bind_symbolic_shape %186, [%37], affine_map<()[s0] -> (4, s0)> : !torch.vtensor<[4,?],si64>
    %int0_170 = torch.constant.int 0
    %187 = torch.aten.fill.Scalar %186, %int0_170 : !torch.vtensor<[4,?],si64>, !torch.int -> !torch.vtensor<[4,?],si64>
    torch.bind_symbolic_shape %187, [%37], affine_map<()[s0] -> (4, s0)> : !torch.vtensor<[4,?],si64>
    %int3_171 = torch.constant.int 3
    %188 = torch.aten.mul.Scalar %arg3, %int3_171 : !torch.vtensor<[4,?],si64>, !torch.int -> !torch.vtensor<[4,?],si64>
    torch.bind_symbolic_shape %188, [%37], affine_map<()[s0] -> (4, s0)> : !torch.vtensor<[4,?],si64>
    %int1_172 = torch.constant.int 1
    %189 = torch.aten.add.Tensor %188, %187, %int1_172 : !torch.vtensor<[4,?],si64>, !torch.vtensor<[4,?],si64>, !torch.int -> !torch.vtensor<[4,?],si64>
    torch.bind_symbolic_shape %189, [%37], affine_map<()[s0] -> (4, s0)> : !torch.vtensor<[4,?],si64>
    %int4_173 = torch.constant.int 4
    %190 = torch.aten.mul.int %int4_173, %39 : !torch.int, !torch.int -> !torch.int
    %191 = torch.prim.ListConstruct %190 : (!torch.int) -> !torch.list<int>
    %192 = torch.aten.view %189, %191 : !torch.vtensor<[4,?],si64>, !torch.list<int> -> !torch.vtensor<[?],si64>
    torch.bind_symbolic_shape %192, [%37], affine_map<()[s0] -> (s0 * 4)> : !torch.vtensor<[?],si64>
    %int3_174 = torch.constant.int 3
    %int2_175 = torch.constant.int 2
    %int32_176 = torch.constant.int 32
    %int4_177 = torch.constant.int 4
    %int32_178 = torch.constant.int 32
    %193 = torch.prim.ListConstruct %122, %int3_174, %int2_175, %int32_176, %int4_177, %int32_178 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %194 = torch.aten.view %183, %193 : !torch.vtensor<[?,24576],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,4,32],f16>
    torch.bind_symbolic_shape %194, [%38], affine_map<()[s0] -> (s0, 3, 2, 32, 4, 32)> : !torch.vtensor<[?,3,2,32,4,32],f16>
    %int3_179 = torch.constant.int 3
    %195 = torch.aten.mul.int %122, %int3_179 : !torch.int, !torch.int -> !torch.int
    %int2_180 = torch.constant.int 2
    %int32_181 = torch.constant.int 32
    %int4_182 = torch.constant.int 4
    %int32_183 = torch.constant.int 32
    %196 = torch.prim.ListConstruct %195, %int2_180, %int32_181, %int4_182, %int32_183 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %197 = torch.aten.view %194, %196 : !torch.vtensor<[?,3,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,2,32,4,32],f16>
    torch.bind_symbolic_shape %197, [%38], affine_map<()[s0] -> (s0 * 3, 2, 32, 4, 32)> : !torch.vtensor<[?,2,32,4,32],f16>
    %int0_184 = torch.constant.int 0
    %198 = torch.aten.index_select %197, %int0_184, %192 : !torch.vtensor<[?,2,32,4,32],f16>, !torch.int, !torch.vtensor<[?],si64> -> !torch.vtensor<[?,2,32,4,32],f16>
    torch.bind_symbolic_shape %198, [%37], affine_map<()[s0] -> (s0 * 4, 2, 32, 4, 32)> : !torch.vtensor<[?,2,32,4,32],f16>
    %int4_185 = torch.constant.int 4
    %int2_186 = torch.constant.int 2
    %int32_187 = torch.constant.int 32
    %int4_188 = torch.constant.int 4
    %int32_189 = torch.constant.int 32
    %199 = torch.prim.ListConstruct %int4_185, %39, %int2_186, %int32_187, %int4_188, %int32_189 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %200 = torch.aten.view %198, %199 : !torch.vtensor<[?,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[4,?,2,32,4,32],f16>
    torch.bind_symbolic_shape %200, [%37], affine_map<()[s0] -> (4, s0, 2, 32, 4, 32)> : !torch.vtensor<[4,?,2,32,4,32],f16>
    %int0_190 = torch.constant.int 0
    %int0_191 = torch.constant.int 0
    %int9223372036854775807 = torch.constant.int 9223372036854775807
    %int1_192 = torch.constant.int 1
    %201 = torch.aten.slice.Tensor %200, %int0_190, %int0_191, %int9223372036854775807, %int1_192 : !torch.vtensor<[4,?,2,32,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,?,2,32,4,32],f16>
    torch.bind_symbolic_shape %201, [%37], affine_map<()[s0] -> (4, s0, 2, 32, 4, 32)> : !torch.vtensor<[4,?,2,32,4,32],f16>
    %int1_193 = torch.constant.int 1
    %int0_194 = torch.constant.int 0
    %int9223372036854775807_195 = torch.constant.int 9223372036854775807
    %int1_196 = torch.constant.int 1
    %202 = torch.aten.slice.Tensor %201, %int1_193, %int0_194, %int9223372036854775807_195, %int1_196 : !torch.vtensor<[4,?,2,32,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,?,2,32,4,32],f16>
    torch.bind_symbolic_shape %202, [%37], affine_map<()[s0] -> (4, s0, 2, 32, 4, 32)> : !torch.vtensor<[4,?,2,32,4,32],f16>
    %int2_197 = torch.constant.int 2
    %int0_198 = torch.constant.int 0
    %203 = torch.aten.select.int %202, %int2_197, %int0_198 : !torch.vtensor<[4,?,2,32,4,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[4,?,32,4,32],f16>
    torch.bind_symbolic_shape %203, [%37], affine_map<()[s0] -> (4, s0, 32, 4, 32)> : !torch.vtensor<[4,?,32,4,32],f16>
    %int32_199 = torch.constant.int 32
    %204 = torch.aten.mul.int %39, %int32_199 : !torch.int, !torch.int -> !torch.int
    %int2_200 = torch.constant.int 2
    %int0_201 = torch.constant.int 0
    %int1_202 = torch.constant.int 1
    %205 = torch.aten.slice.Tensor %203, %int2_200, %int0_201, %204, %int1_202 : !torch.vtensor<[4,?,32,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,?,32,4,32],f16>
    torch.bind_symbolic_shape %205, [%37], affine_map<()[s0] -> (4, s0, 32, 4, 32)> : !torch.vtensor<[4,?,32,4,32],f16>
    %int0_203 = torch.constant.int 0
    %206 = torch.aten.clone %205, %int0_203 : !torch.vtensor<[4,?,32,4,32],f16>, !torch.int -> !torch.vtensor<[4,?,32,4,32],f16>
    torch.bind_symbolic_shape %206, [%37], affine_map<()[s0] -> (4, s0, 32, 4, 32)> : !torch.vtensor<[4,?,32,4,32],f16>
    %int1_204 = torch.constant.int 1
    %207 = torch.aten.size.int %202, %int1_204 : !torch.vtensor<[4,?,2,32,4,32],f16>, !torch.int -> !torch.int
    %int32_205 = torch.constant.int 32
    %208 = torch.aten.mul.int %207, %int32_205 : !torch.int, !torch.int -> !torch.int
    %int4_206 = torch.constant.int 4
    %int4_207 = torch.constant.int 4
    %int32_208 = torch.constant.int 32
    %209 = torch.prim.ListConstruct %int4_206, %208, %int4_207, %int32_208 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %210 = torch.aten._unsafe_view %206, %209 : !torch.vtensor<[4,?,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[4,?,4,32],f16>
    torch.bind_symbolic_shape %210, [%37], affine_map<()[s0] -> (4, s0 * 32, 4, 32)> : !torch.vtensor<[4,?,4,32],f16>
    %int0_209 = torch.constant.int 0
    %int0_210 = torch.constant.int 0
    %int9223372036854775807_211 = torch.constant.int 9223372036854775807
    %int1_212 = torch.constant.int 1
    %211 = torch.aten.slice.Tensor %210, %int0_209, %int0_210, %int9223372036854775807_211, %int1_212 : !torch.vtensor<[4,?,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,?,4,32],f16>
    torch.bind_symbolic_shape %211, [%37], affine_map<()[s0] -> (4, s0 * 32, 4, 32)> : !torch.vtensor<[4,?,4,32],f16>
    %int0_213 = torch.constant.int 0
    %int0_214 = torch.constant.int 0
    %int9223372036854775807_215 = torch.constant.int 9223372036854775807
    %int1_216 = torch.constant.int 1
    %212 = torch.aten.slice.Tensor %200, %int0_213, %int0_214, %int9223372036854775807_215, %int1_216 : !torch.vtensor<[4,?,2,32,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,?,2,32,4,32],f16>
    torch.bind_symbolic_shape %212, [%37], affine_map<()[s0] -> (4, s0, 2, 32, 4, 32)> : !torch.vtensor<[4,?,2,32,4,32],f16>
    %int1_217 = torch.constant.int 1
    %int0_218 = torch.constant.int 0
    %int9223372036854775807_219 = torch.constant.int 9223372036854775807
    %int1_220 = torch.constant.int 1
    %213 = torch.aten.slice.Tensor %212, %int1_217, %int0_218, %int9223372036854775807_219, %int1_220 : !torch.vtensor<[4,?,2,32,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,?,2,32,4,32],f16>
    torch.bind_symbolic_shape %213, [%37], affine_map<()[s0] -> (4, s0, 2, 32, 4, 32)> : !torch.vtensor<[4,?,2,32,4,32],f16>
    %int2_221 = torch.constant.int 2
    %int1_222 = torch.constant.int 1
    %214 = torch.aten.select.int %213, %int2_221, %int1_222 : !torch.vtensor<[4,?,2,32,4,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[4,?,32,4,32],f16>
    torch.bind_symbolic_shape %214, [%37], affine_map<()[s0] -> (4, s0, 32, 4, 32)> : !torch.vtensor<[4,?,32,4,32],f16>
    %int2_223 = torch.constant.int 2
    %int0_224 = torch.constant.int 0
    %int1_225 = torch.constant.int 1
    %215 = torch.aten.slice.Tensor %214, %int2_223, %int0_224, %204, %int1_225 : !torch.vtensor<[4,?,32,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,?,32,4,32],f16>
    torch.bind_symbolic_shape %215, [%37], affine_map<()[s0] -> (4, s0, 32, 4, 32)> : !torch.vtensor<[4,?,32,4,32],f16>
    %int0_226 = torch.constant.int 0
    %216 = torch.aten.clone %215, %int0_226 : !torch.vtensor<[4,?,32,4,32],f16>, !torch.int -> !torch.vtensor<[4,?,32,4,32],f16>
    torch.bind_symbolic_shape %216, [%37], affine_map<()[s0] -> (4, s0, 32, 4, 32)> : !torch.vtensor<[4,?,32,4,32],f16>
    %int1_227 = torch.constant.int 1
    %217 = torch.aten.size.int %213, %int1_227 : !torch.vtensor<[4,?,2,32,4,32],f16>, !torch.int -> !torch.int
    %int32_228 = torch.constant.int 32
    %218 = torch.aten.mul.int %217, %int32_228 : !torch.int, !torch.int -> !torch.int
    %int4_229 = torch.constant.int 4
    %int4_230 = torch.constant.int 4
    %int32_231 = torch.constant.int 32
    %219 = torch.prim.ListConstruct %int4_229, %218, %int4_230, %int32_231 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %220 = torch.aten._unsafe_view %216, %219 : !torch.vtensor<[4,?,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[4,?,4,32],f16>
    torch.bind_symbolic_shape %220, [%37], affine_map<()[s0] -> (4, s0 * 32, 4, 32)> : !torch.vtensor<[4,?,4,32],f16>
    %int0_232 = torch.constant.int 0
    %int0_233 = torch.constant.int 0
    %int9223372036854775807_234 = torch.constant.int 9223372036854775807
    %int1_235 = torch.constant.int 1
    %221 = torch.aten.slice.Tensor %220, %int0_232, %int0_233, %int9223372036854775807_234, %int1_235 : !torch.vtensor<[4,?,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,?,4,32],f16>
    torch.bind_symbolic_shape %221, [%37], affine_map<()[s0] -> (4, s0 * 32, 4, 32)> : !torch.vtensor<[4,?,4,32],f16>
    %int-2_236 = torch.constant.int -2
    %222 = torch.aten.unsqueeze %211, %int-2_236 : !torch.vtensor<[4,?,4,32],f16>, !torch.int -> !torch.vtensor<[4,?,4,1,32],f16>
    torch.bind_symbolic_shape %222, [%37], affine_map<()[s0] -> (4, s0 * 32, 4, 1, 32)> : !torch.vtensor<[4,?,4,1,32],f16>
    %int1_237 = torch.constant.int 1
    %223 = torch.aten.size.int %210, %int1_237 : !torch.vtensor<[4,?,4,32],f16>, !torch.int -> !torch.int
    %int4_238 = torch.constant.int 4
    %int4_239 = torch.constant.int 4
    %int2_240 = torch.constant.int 2
    %int32_241 = torch.constant.int 32
    %224 = torch.prim.ListConstruct %int4_238, %223, %int4_239, %int2_240, %int32_241 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_242 = torch.constant.bool false
    %225 = torch.aten.expand %222, %224, %false_242 : !torch.vtensor<[4,?,4,1,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[4,?,4,2,32],f16>
    torch.bind_symbolic_shape %225, [%37], affine_map<()[s0] -> (4, s0 * 32, 4, 2, 32)> : !torch.vtensor<[4,?,4,2,32],f16>
    %int0_243 = torch.constant.int 0
    %226 = torch.aten.clone %225, %int0_243 : !torch.vtensor<[4,?,4,2,32],f16>, !torch.int -> !torch.vtensor<[4,?,4,2,32],f16>
    torch.bind_symbolic_shape %226, [%37], affine_map<()[s0] -> (4, s0 * 32, 4, 2, 32)> : !torch.vtensor<[4,?,4,2,32],f16>
    %int4_244 = torch.constant.int 4
    %int8_245 = torch.constant.int 8
    %int32_246 = torch.constant.int 32
    %227 = torch.prim.ListConstruct %int4_244, %223, %int8_245, %int32_246 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %228 = torch.aten._unsafe_view %226, %227 : !torch.vtensor<[4,?,4,2,32],f16>, !torch.list<int> -> !torch.vtensor<[4,?,8,32],f16>
    torch.bind_symbolic_shape %228, [%37], affine_map<()[s0] -> (4, s0 * 32, 8, 32)> : !torch.vtensor<[4,?,8,32],f16>
    %int-2_247 = torch.constant.int -2
    %229 = torch.aten.unsqueeze %221, %int-2_247 : !torch.vtensor<[4,?,4,32],f16>, !torch.int -> !torch.vtensor<[4,?,4,1,32],f16>
    torch.bind_symbolic_shape %229, [%37], affine_map<()[s0] -> (4, s0 * 32, 4, 1, 32)> : !torch.vtensor<[4,?,4,1,32],f16>
    %int1_248 = torch.constant.int 1
    %230 = torch.aten.size.int %220, %int1_248 : !torch.vtensor<[4,?,4,32],f16>, !torch.int -> !torch.int
    %int4_249 = torch.constant.int 4
    %int4_250 = torch.constant.int 4
    %int2_251 = torch.constant.int 2
    %int32_252 = torch.constant.int 32
    %231 = torch.prim.ListConstruct %int4_249, %230, %int4_250, %int2_251, %int32_252 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_253 = torch.constant.bool false
    %232 = torch.aten.expand %229, %231, %false_253 : !torch.vtensor<[4,?,4,1,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[4,?,4,2,32],f16>
    torch.bind_symbolic_shape %232, [%37], affine_map<()[s0] -> (4, s0 * 32, 4, 2, 32)> : !torch.vtensor<[4,?,4,2,32],f16>
    %int0_254 = torch.constant.int 0
    %233 = torch.aten.clone %232, %int0_254 : !torch.vtensor<[4,?,4,2,32],f16>, !torch.int -> !torch.vtensor<[4,?,4,2,32],f16>
    torch.bind_symbolic_shape %233, [%37], affine_map<()[s0] -> (4, s0 * 32, 4, 2, 32)> : !torch.vtensor<[4,?,4,2,32],f16>
    %int4_255 = torch.constant.int 4
    %int8_256 = torch.constant.int 8
    %int32_257 = torch.constant.int 32
    %234 = torch.prim.ListConstruct %int4_255, %230, %int8_256, %int32_257 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %235 = torch.aten._unsafe_view %233, %234 : !torch.vtensor<[4,?,4,2,32],f16>, !torch.list<int> -> !torch.vtensor<[4,?,8,32],f16>
    torch.bind_symbolic_shape %235, [%37], affine_map<()[s0] -> (4, s0 * 32, 8, 32)> : !torch.vtensor<[4,?,8,32],f16>
    %int1_258 = torch.constant.int 1
    %int2_259 = torch.constant.int 2
    %236 = torch.aten.transpose.int %115, %int1_258, %int2_259 : !torch.vtensor<[4,1,8,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[4,8,1,32],f16>
    %int1_260 = torch.constant.int 1
    %int2_261 = torch.constant.int 2
    %237 = torch.aten.transpose.int %228, %int1_260, %int2_261 : !torch.vtensor<[4,?,8,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[4,8,?,32],f16>
    torch.bind_symbolic_shape %237, [%37], affine_map<()[s0] -> (4, 8, s0 * 32, 32)> : !torch.vtensor<[4,8,?,32],f16>
    %int1_262 = torch.constant.int 1
    %int2_263 = torch.constant.int 2
    %238 = torch.aten.transpose.int %235, %int1_262, %int2_263 : !torch.vtensor<[4,?,8,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[4,8,?,32],f16>
    torch.bind_symbolic_shape %238, [%37], affine_map<()[s0] -> (4, 8, s0 * 32, 32)> : !torch.vtensor<[4,8,?,32],f16>
    %float0.000000e00 = torch.constant.float 0.000000e+00
    %false_264 = torch.constant.bool false
    %none_265 = torch.constant.none
    %239:2 = torch.operator "torch.aten._scaled_dot_product_flash_attention_for_cpu"(%236, %237, %238, %float0.000000e00, %false_264, %49, %none_265) : (!torch.vtensor<[4,8,1,32],f16>, !torch.vtensor<[4,8,?,32],f16>, !torch.vtensor<[4,8,?,32],f16>, !torch.float, !torch.bool, !torch.vtensor<[4,1,1,?],f16>, !torch.none) -> (!torch.vtensor<[4,8,1,32],f16>, !torch.vtensor<[4,8,1],f32>) 
    %int1_266 = torch.constant.int 1
    %int2_267 = torch.constant.int 2
    %240 = torch.aten.transpose.int %239#0, %int1_266, %int2_267 : !torch.vtensor<[4,8,1,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[4,1,8,32],f16>
    %int4_268 = torch.constant.int 4
    %int1_269 = torch.constant.int 1
    %int256_270 = torch.constant.int 256
    %241 = torch.prim.ListConstruct %int4_268, %int1_269, %int256_270 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %242 = torch.aten.view %240, %241 : !torch.vtensor<[4,1,8,32],f16>, !torch.list<int> -> !torch.vtensor<[4,1,256],f16>
    %int5_271 = torch.constant.int 5
    %243 = torch.prims.convert_element_type %7, %int5_271 : !torch.vtensor<[256,256],f32>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int-2_272 = torch.constant.int -2
    %int-1_273 = torch.constant.int -1
    %244 = torch.aten.transpose.int %243, %int-2_272, %int-1_273 : !torch.vtensor<[256,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,256],f16>
    %int4_274 = torch.constant.int 4
    %int256_275 = torch.constant.int 256
    %245 = torch.prim.ListConstruct %int4_274, %int256_275 : (!torch.int, !torch.int) -> !torch.list<int>
    %246 = torch.aten.view %242, %245 : !torch.vtensor<[4,1,256],f16>, !torch.list<int> -> !torch.vtensor<[4,256],f16>
    %247 = torch.aten.mm %246, %244 : !torch.vtensor<[4,256],f16>, !torch.vtensor<[256,256],f16> -> !torch.vtensor<[4,256],f16>
    %int4_276 = torch.constant.int 4
    %int1_277 = torch.constant.int 1
    %int256_278 = torch.constant.int 256
    %248 = torch.prim.ListConstruct %int4_276, %int1_277, %int256_278 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %249 = torch.aten.view %247, %248 : !torch.vtensor<[4,256],f16>, !torch.list<int> -> !torch.vtensor<[4,1,256],f16>
    %int1_279 = torch.constant.int 1
    %250 = torch.aten.add.Tensor %72, %249, %int1_279 : !torch.vtensor<[4,1,256],f16>, !torch.vtensor<[4,1,256],f16>, !torch.int -> !torch.vtensor<[4,1,256],f16>
    %int6_280 = torch.constant.int 6
    %251 = torch.prims.convert_element_type %250, %int6_280 : !torch.vtensor<[4,1,256],f16>, !torch.int -> !torch.vtensor<[4,1,256],f32>
    %int2_281 = torch.constant.int 2
    %252 = torch.aten.pow.Tensor_Scalar %251, %int2_281 : !torch.vtensor<[4,1,256],f32>, !torch.int -> !torch.vtensor<[4,1,256],f32>
    %int-1_282 = torch.constant.int -1
    %253 = torch.prim.ListConstruct %int-1_282 : (!torch.int) -> !torch.list<int>
    %true_283 = torch.constant.bool true
    %none_284 = torch.constant.none
    %254 = torch.aten.mean.dim %252, %253, %true_283, %none_284 : !torch.vtensor<[4,1,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[4,1,1],f32>
    %float1.000000e-02_285 = torch.constant.float 1.000000e-02
    %int1_286 = torch.constant.int 1
    %255 = torch.aten.add.Scalar %254, %float1.000000e-02_285, %int1_286 : !torch.vtensor<[4,1,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[4,1,1],f32>
    %256 = torch.aten.rsqrt %255 : !torch.vtensor<[4,1,1],f32> -> !torch.vtensor<[4,1,1],f32>
    %257 = torch.aten.mul.Tensor %251, %256 : !torch.vtensor<[4,1,256],f32>, !torch.vtensor<[4,1,1],f32> -> !torch.vtensor<[4,1,256],f32>
    %int5_287 = torch.constant.int 5
    %258 = torch.prims.convert_element_type %257, %int5_287 : !torch.vtensor<[4,1,256],f32>, !torch.int -> !torch.vtensor<[4,1,256],f16>
    %259 = torch.aten.mul.Tensor %8, %258 : !torch.vtensor<[256],f32>, !torch.vtensor<[4,1,256],f16> -> !torch.vtensor<[4,1,256],f32>
    %int5_288 = torch.constant.int 5
    %260 = torch.prims.convert_element_type %259, %int5_288 : !torch.vtensor<[4,1,256],f32>, !torch.int -> !torch.vtensor<[4,1,256],f16>
    %int5_289 = torch.constant.int 5
    %261 = torch.prims.convert_element_type %9, %int5_289 : !torch.vtensor<[23,256],f32>, !torch.int -> !torch.vtensor<[23,256],f16>
    %int-2_290 = torch.constant.int -2
    %int-1_291 = torch.constant.int -1
    %262 = torch.aten.transpose.int %261, %int-2_290, %int-1_291 : !torch.vtensor<[23,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,23],f16>
    %int4_292 = torch.constant.int 4
    %int256_293 = torch.constant.int 256
    %263 = torch.prim.ListConstruct %int4_292, %int256_293 : (!torch.int, !torch.int) -> !torch.list<int>
    %264 = torch.aten.view %260, %263 : !torch.vtensor<[4,1,256],f16>, !torch.list<int> -> !torch.vtensor<[4,256],f16>
    %265 = torch.aten.mm %264, %262 : !torch.vtensor<[4,256],f16>, !torch.vtensor<[256,23],f16> -> !torch.vtensor<[4,23],f16>
    %int4_294 = torch.constant.int 4
    %int1_295 = torch.constant.int 1
    %int23 = torch.constant.int 23
    %266 = torch.prim.ListConstruct %int4_294, %int1_295, %int23 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %267 = torch.aten.view %265, %266 : !torch.vtensor<[4,23],f16>, !torch.list<int> -> !torch.vtensor<[4,1,23],f16>
    %268 = torch.aten.silu %267 : !torch.vtensor<[4,1,23],f16> -> !torch.vtensor<[4,1,23],f16>
    %int5_296 = torch.constant.int 5
    %269 = torch.prims.convert_element_type %10, %int5_296 : !torch.vtensor<[23,256],f32>, !torch.int -> !torch.vtensor<[23,256],f16>
    %int-2_297 = torch.constant.int -2
    %int-1_298 = torch.constant.int -1
    %270 = torch.aten.transpose.int %269, %int-2_297, %int-1_298 : !torch.vtensor<[23,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,23],f16>
    %int4_299 = torch.constant.int 4
    %int256_300 = torch.constant.int 256
    %271 = torch.prim.ListConstruct %int4_299, %int256_300 : (!torch.int, !torch.int) -> !torch.list<int>
    %272 = torch.aten.view %260, %271 : !torch.vtensor<[4,1,256],f16>, !torch.list<int> -> !torch.vtensor<[4,256],f16>
    %273 = torch.aten.mm %272, %270 : !torch.vtensor<[4,256],f16>, !torch.vtensor<[256,23],f16> -> !torch.vtensor<[4,23],f16>
    %int4_301 = torch.constant.int 4
    %int1_302 = torch.constant.int 1
    %int23_303 = torch.constant.int 23
    %274 = torch.prim.ListConstruct %int4_301, %int1_302, %int23_303 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %275 = torch.aten.view %273, %274 : !torch.vtensor<[4,23],f16>, !torch.list<int> -> !torch.vtensor<[4,1,23],f16>
    %276 = torch.aten.mul.Tensor %268, %275 : !torch.vtensor<[4,1,23],f16>, !torch.vtensor<[4,1,23],f16> -> !torch.vtensor<[4,1,23],f16>
    %int5_304 = torch.constant.int 5
    %277 = torch.prims.convert_element_type %11, %int5_304 : !torch.vtensor<[256,23],f32>, !torch.int -> !torch.vtensor<[256,23],f16>
    %int-2_305 = torch.constant.int -2
    %int-1_306 = torch.constant.int -1
    %278 = torch.aten.transpose.int %277, %int-2_305, %int-1_306 : !torch.vtensor<[256,23],f16>, !torch.int, !torch.int -> !torch.vtensor<[23,256],f16>
    %int4_307 = torch.constant.int 4
    %int23_308 = torch.constant.int 23
    %279 = torch.prim.ListConstruct %int4_307, %int23_308 : (!torch.int, !torch.int) -> !torch.list<int>
    %280 = torch.aten.view %276, %279 : !torch.vtensor<[4,1,23],f16>, !torch.list<int> -> !torch.vtensor<[4,23],f16>
    %281 = torch.aten.mm %280, %278 : !torch.vtensor<[4,23],f16>, !torch.vtensor<[23,256],f16> -> !torch.vtensor<[4,256],f16>
    %int4_309 = torch.constant.int 4
    %int1_310 = torch.constant.int 1
    %int256_311 = torch.constant.int 256
    %282 = torch.prim.ListConstruct %int4_309, %int1_310, %int256_311 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %283 = torch.aten.view %281, %282 : !torch.vtensor<[4,256],f16>, !torch.list<int> -> !torch.vtensor<[4,1,256],f16>
    %int1_312 = torch.constant.int 1
    %284 = torch.aten.add.Tensor %250, %283, %int1_312 : !torch.vtensor<[4,1,256],f16>, !torch.vtensor<[4,1,256],f16>, !torch.int -> !torch.vtensor<[4,1,256],f16>
    %int6_313 = torch.constant.int 6
    %285 = torch.prims.convert_element_type %284, %int6_313 : !torch.vtensor<[4,1,256],f16>, !torch.int -> !torch.vtensor<[4,1,256],f32>
    %int2_314 = torch.constant.int 2
    %286 = torch.aten.pow.Tensor_Scalar %285, %int2_314 : !torch.vtensor<[4,1,256],f32>, !torch.int -> !torch.vtensor<[4,1,256],f32>
    %int-1_315 = torch.constant.int -1
    %287 = torch.prim.ListConstruct %int-1_315 : (!torch.int) -> !torch.list<int>
    %true_316 = torch.constant.bool true
    %none_317 = torch.constant.none
    %288 = torch.aten.mean.dim %286, %287, %true_316, %none_317 : !torch.vtensor<[4,1,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[4,1,1],f32>
    %float1.000000e-02_318 = torch.constant.float 1.000000e-02
    %int1_319 = torch.constant.int 1
    %289 = torch.aten.add.Scalar %288, %float1.000000e-02_318, %int1_319 : !torch.vtensor<[4,1,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[4,1,1],f32>
    %290 = torch.aten.rsqrt %289 : !torch.vtensor<[4,1,1],f32> -> !torch.vtensor<[4,1,1],f32>
    %291 = torch.aten.mul.Tensor %285, %290 : !torch.vtensor<[4,1,256],f32>, !torch.vtensor<[4,1,1],f32> -> !torch.vtensor<[4,1,256],f32>
    %int5_320 = torch.constant.int 5
    %292 = torch.prims.convert_element_type %291, %int5_320 : !torch.vtensor<[4,1,256],f32>, !torch.int -> !torch.vtensor<[4,1,256],f16>
    %293 = torch.aten.mul.Tensor %12, %292 : !torch.vtensor<[256],f32>, !torch.vtensor<[4,1,256],f16> -> !torch.vtensor<[4,1,256],f32>
    %int5_321 = torch.constant.int 5
    %294 = torch.prims.convert_element_type %293, %int5_321 : !torch.vtensor<[4,1,256],f32>, !torch.int -> !torch.vtensor<[4,1,256],f16>
    %int5_322 = torch.constant.int 5
    %295 = torch.prims.convert_element_type %13, %int5_322 : !torch.vtensor<[256,256],f32>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int-2_323 = torch.constant.int -2
    %int-1_324 = torch.constant.int -1
    %296 = torch.aten.transpose.int %295, %int-2_323, %int-1_324 : !torch.vtensor<[256,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,256],f16>
    %int4_325 = torch.constant.int 4
    %int256_326 = torch.constant.int 256
    %297 = torch.prim.ListConstruct %int4_325, %int256_326 : (!torch.int, !torch.int) -> !torch.list<int>
    %298 = torch.aten.view %294, %297 : !torch.vtensor<[4,1,256],f16>, !torch.list<int> -> !torch.vtensor<[4,256],f16>
    %299 = torch.aten.mm %298, %296 : !torch.vtensor<[4,256],f16>, !torch.vtensor<[256,256],f16> -> !torch.vtensor<[4,256],f16>
    %int4_327 = torch.constant.int 4
    %int1_328 = torch.constant.int 1
    %int256_329 = torch.constant.int 256
    %300 = torch.prim.ListConstruct %int4_327, %int1_328, %int256_329 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %301 = torch.aten.view %299, %300 : !torch.vtensor<[4,256],f16>, !torch.list<int> -> !torch.vtensor<[4,1,256],f16>
    %int5_330 = torch.constant.int 5
    %302 = torch.prims.convert_element_type %14, %int5_330 : !torch.vtensor<[128,256],f32>, !torch.int -> !torch.vtensor<[128,256],f16>
    %int-2_331 = torch.constant.int -2
    %int-1_332 = torch.constant.int -1
    %303 = torch.aten.transpose.int %302, %int-2_331, %int-1_332 : !torch.vtensor<[128,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,128],f16>
    %int4_333 = torch.constant.int 4
    %int256_334 = torch.constant.int 256
    %304 = torch.prim.ListConstruct %int4_333, %int256_334 : (!torch.int, !torch.int) -> !torch.list<int>
    %305 = torch.aten.view %294, %304 : !torch.vtensor<[4,1,256],f16>, !torch.list<int> -> !torch.vtensor<[4,256],f16>
    %306 = torch.aten.mm %305, %303 : !torch.vtensor<[4,256],f16>, !torch.vtensor<[256,128],f16> -> !torch.vtensor<[4,128],f16>
    %int4_335 = torch.constant.int 4
    %int1_336 = torch.constant.int 1
    %int128_337 = torch.constant.int 128
    %307 = torch.prim.ListConstruct %int4_335, %int1_336, %int128_337 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %308 = torch.aten.view %306, %307 : !torch.vtensor<[4,128],f16>, !torch.list<int> -> !torch.vtensor<[4,1,128],f16>
    %int5_338 = torch.constant.int 5
    %309 = torch.prims.convert_element_type %15, %int5_338 : !torch.vtensor<[128,256],f32>, !torch.int -> !torch.vtensor<[128,256],f16>
    %int-2_339 = torch.constant.int -2
    %int-1_340 = torch.constant.int -1
    %310 = torch.aten.transpose.int %309, %int-2_339, %int-1_340 : !torch.vtensor<[128,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,128],f16>
    %int4_341 = torch.constant.int 4
    %int256_342 = torch.constant.int 256
    %311 = torch.prim.ListConstruct %int4_341, %int256_342 : (!torch.int, !torch.int) -> !torch.list<int>
    %312 = torch.aten.view %294, %311 : !torch.vtensor<[4,1,256],f16>, !torch.list<int> -> !torch.vtensor<[4,256],f16>
    %313 = torch.aten.mm %312, %310 : !torch.vtensor<[4,256],f16>, !torch.vtensor<[256,128],f16> -> !torch.vtensor<[4,128],f16>
    %int4_343 = torch.constant.int 4
    %int1_344 = torch.constant.int 1
    %int128_345 = torch.constant.int 128
    %314 = torch.prim.ListConstruct %int4_343, %int1_344, %int128_345 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %315 = torch.aten.view %313, %314 : !torch.vtensor<[4,128],f16>, !torch.list<int> -> !torch.vtensor<[4,1,128],f16>
    %int4_346 = torch.constant.int 4
    %int1_347 = torch.constant.int 1
    %int8_348 = torch.constant.int 8
    %int32_349 = torch.constant.int 32
    %316 = torch.prim.ListConstruct %int4_346, %int1_347, %int8_348, %int32_349 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %317 = torch.aten.view %301, %316 : !torch.vtensor<[4,1,256],f16>, !torch.list<int> -> !torch.vtensor<[4,1,8,32],f16>
    %int4_350 = torch.constant.int 4
    %int1_351 = torch.constant.int 1
    %int4_352 = torch.constant.int 4
    %int32_353 = torch.constant.int 32
    %318 = torch.prim.ListConstruct %int4_350, %int1_351, %int4_352, %int32_353 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %319 = torch.aten.view %308, %318 : !torch.vtensor<[4,1,128],f16>, !torch.list<int> -> !torch.vtensor<[4,1,4,32],f16>
    %int4_354 = torch.constant.int 4
    %int1_355 = torch.constant.int 1
    %int4_356 = torch.constant.int 4
    %int32_357 = torch.constant.int 32
    %320 = torch.prim.ListConstruct %int4_354, %int1_355, %int4_356, %int32_357 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %321 = torch.aten.view %315, %320 : !torch.vtensor<[4,1,128],f16>, !torch.list<int> -> !torch.vtensor<[4,1,4,32],f16>
    %int6_358 = torch.constant.int 6
    %322 = torch.prims.convert_element_type %317, %int6_358 : !torch.vtensor<[4,1,8,32],f16>, !torch.int -> !torch.vtensor<[4,1,8,32],f32>
    %323 = torch_c.to_builtin_tensor %322 : !torch.vtensor<[4,1,8,32],f32> -> tensor<4x1x8x32xf32>
    %324 = torch_c.to_builtin_tensor %70 : !torch.vtensor<[4,1,32],f32> -> tensor<4x1x32xf32>
    %325 = util.call @sharktank_rotary_embedding_4_1_8_32_f32(%323, %324) : (tensor<4x1x8x32xf32>, tensor<4x1x32xf32>) -> tensor<4x1x8x32xf32>
    %326 = torch_c.from_builtin_tensor %325 : tensor<4x1x8x32xf32> -> !torch.vtensor<[4,1,8,32],f32>
    %int5_359 = torch.constant.int 5
    %327 = torch.prims.convert_element_type %326, %int5_359 : !torch.vtensor<[4,1,8,32],f32>, !torch.int -> !torch.vtensor<[4,1,8,32],f16>
    %int6_360 = torch.constant.int 6
    %328 = torch.prims.convert_element_type %319, %int6_360 : !torch.vtensor<[4,1,4,32],f16>, !torch.int -> !torch.vtensor<[4,1,4,32],f32>
    %329 = torch_c.to_builtin_tensor %328 : !torch.vtensor<[4,1,4,32],f32> -> tensor<4x1x4x32xf32>
    %330 = torch_c.to_builtin_tensor %70 : !torch.vtensor<[4,1,32],f32> -> tensor<4x1x32xf32>
    %331 = util.call @sharktank_rotary_embedding_4_1_4_32_f32(%329, %330) : (tensor<4x1x4x32xf32>, tensor<4x1x32xf32>) -> tensor<4x1x4x32xf32>
    %332 = torch_c.from_builtin_tensor %331 : tensor<4x1x4x32xf32> -> !torch.vtensor<[4,1,4,32],f32>
    %int5_361 = torch.constant.int 5
    %333 = torch.prims.convert_element_type %332, %int5_361 : !torch.vtensor<[4,1,4,32],f32>, !torch.int -> !torch.vtensor<[4,1,4,32],f16>
    %int32_362 = torch.constant.int 32
    %334 = torch.aten.floor_divide.Scalar %arg2, %int32_362 : !torch.vtensor<[4],si64>, !torch.int -> !torch.vtensor<[4],si64>
    %int1_363 = torch.constant.int 1
    %335 = torch.aten.unsqueeze %334, %int1_363 : !torch.vtensor<[4],si64>, !torch.int -> !torch.vtensor<[4,1],si64>
    %int1_364 = torch.constant.int 1
    %false_365 = torch.constant.bool false
    %336 = torch.aten.gather %arg3, %int1_364, %335, %false_365 : !torch.vtensor<[4,?],si64>, !torch.int, !torch.vtensor<[4,1],si64>, !torch.bool -> !torch.vtensor<[4,1],si64>
    %int32_366 = torch.constant.int 32
    %337 = torch.aten.remainder.Scalar %arg2, %int32_366 : !torch.vtensor<[4],si64>, !torch.int -> !torch.vtensor<[4],si64>
    %int1_367 = torch.constant.int 1
    %338 = torch.aten.unsqueeze %337, %int1_367 : !torch.vtensor<[4],si64>, !torch.int -> !torch.vtensor<[4,1],si64>
    %none_368 = torch.constant.none
    %339 = torch.aten.clone %16, %none_368 : !torch.vtensor<[],si64>, !torch.none -> !torch.vtensor<[],si64>
    %int0_369 = torch.constant.int 0
    %340 = torch.aten.unsqueeze %339, %int0_369 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int4_370 = torch.constant.int 4
    %int1_371 = torch.constant.int 1
    %341 = torch.prim.ListConstruct %int4_370, %int1_371 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_372 = torch.constant.int 1
    %int1_373 = torch.constant.int 1
    %342 = torch.prim.ListConstruct %int1_372, %int1_373 : (!torch.int, !torch.int) -> !torch.list<int>
    %int4_374 = torch.constant.int 4
    %int0_375 = torch.constant.int 0
    %cpu_376 = torch.constant.device "cpu"
    %false_377 = torch.constant.bool false
    %343 = torch.aten.empty_strided %341, %342, %int4_374, %int0_375, %cpu_376, %false_377 : !torch.list<int>, !torch.list<int>, !torch.int, !torch.int, !torch.Device, !torch.bool -> !torch.vtensor<[4,1],si64>
    %int1_378 = torch.constant.int 1
    %344 = torch.aten.fill.Scalar %343, %int1_378 : !torch.vtensor<[4,1],si64>, !torch.int -> !torch.vtensor<[4,1],si64>
    %int4_379 = torch.constant.int 4
    %int1_380 = torch.constant.int 1
    %345 = torch.prim.ListConstruct %int4_379, %int1_380 : (!torch.int, !torch.int) -> !torch.list<int>
    %346 = torch.aten.repeat %340, %345 : !torch.vtensor<[1],si64>, !torch.list<int> -> !torch.vtensor<[4,1],si64>
    %int3_381 = torch.constant.int 3
    %347 = torch.aten.mul.Scalar %336, %int3_381 : !torch.vtensor<[4,1],si64>, !torch.int -> !torch.vtensor<[4,1],si64>
    %int1_382 = torch.constant.int 1
    %348 = torch.aten.add.Tensor %347, %344, %int1_382 : !torch.vtensor<[4,1],si64>, !torch.vtensor<[4,1],si64>, !torch.int -> !torch.vtensor<[4,1],si64>
    %int2_383 = torch.constant.int 2
    %349 = torch.aten.mul.Scalar %348, %int2_383 : !torch.vtensor<[4,1],si64>, !torch.int -> !torch.vtensor<[4,1],si64>
    %int1_384 = torch.constant.int 1
    %350 = torch.aten.add.Tensor %349, %346, %int1_384 : !torch.vtensor<[4,1],si64>, !torch.vtensor<[4,1],si64>, !torch.int -> !torch.vtensor<[4,1],si64>
    %int32_385 = torch.constant.int 32
    %351 = torch.aten.mul.Scalar %350, %int32_385 : !torch.vtensor<[4,1],si64>, !torch.int -> !torch.vtensor<[4,1],si64>
    %int1_386 = torch.constant.int 1
    %352 = torch.aten.add.Tensor %351, %338, %int1_386 : !torch.vtensor<[4,1],si64>, !torch.vtensor<[4,1],si64>, !torch.int -> !torch.vtensor<[4,1],si64>
    %int3_387 = torch.constant.int 3
    %int2_388 = torch.constant.int 2
    %int32_389 = torch.constant.int 32
    %int4_390 = torch.constant.int 4
    %int32_391 = torch.constant.int 32
    %353 = torch.prim.ListConstruct %122, %int3_387, %int2_388, %int32_389, %int4_390, %int32_391 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %354 = torch.aten.view %183, %353 : !torch.vtensor<[?,24576],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,4,32],f16>
    torch.bind_symbolic_shape %354, [%38], affine_map<()[s0] -> (s0, 3, 2, 32, 4, 32)> : !torch.vtensor<[?,3,2,32,4,32],f16>
    %int3_392 = torch.constant.int 3
    %355 = torch.aten.mul.int %122, %int3_392 : !torch.int, !torch.int -> !torch.int
    %int2_393 = torch.constant.int 2
    %356 = torch.aten.mul.int %355, %int2_393 : !torch.int, !torch.int -> !torch.int
    %int32_394 = torch.constant.int 32
    %357 = torch.aten.mul.int %356, %int32_394 : !torch.int, !torch.int -> !torch.int
    %int4_395 = torch.constant.int 4
    %int32_396 = torch.constant.int 32
    %358 = torch.prim.ListConstruct %357, %int4_395, %int32_396 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %359 = torch.aten.view %354, %358 : !torch.vtensor<[?,3,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,4,32],f16>
    torch.bind_symbolic_shape %359, [%38], affine_map<()[s0] -> (s0 * 192, 4, 32)> : !torch.vtensor<[?,4,32],f16>
    %360 = torch.prim.ListConstruct %352 : (!torch.vtensor<[4,1],si64>) -> !torch.list<optional<vtensor>>
    %false_397 = torch.constant.bool false
    %361 = torch.aten.index_put %359, %360, %333, %false_397 : !torch.vtensor<[?,4,32],f16>, !torch.list<optional<vtensor>>, !torch.vtensor<[4,1,4,32],f16>, !torch.bool -> !torch.vtensor<[?,4,32],f16>
    torch.bind_symbolic_shape %361, [%38], affine_map<()[s0] -> (s0 * 192, 4, 32)> : !torch.vtensor<[?,4,32],f16>
    %int3_398 = torch.constant.int 3
    %int2_399 = torch.constant.int 2
    %int32_400 = torch.constant.int 32
    %int4_401 = torch.constant.int 4
    %int32_402 = torch.constant.int 32
    %362 = torch.prim.ListConstruct %122, %int3_398, %int2_399, %int32_400, %int4_401, %int32_402 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %363 = torch.aten.view %361, %362 : !torch.vtensor<[?,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,4,32],f16>
    torch.bind_symbolic_shape %363, [%38], affine_map<()[s0] -> (s0, 3, 2, 32, 4, 32)> : !torch.vtensor<[?,3,2,32,4,32],f16>
    %int24576_403 = torch.constant.int 24576
    %364 = torch.prim.ListConstruct %122, %int24576_403 : (!torch.int, !torch.int) -> !torch.list<int>
    %365 = torch.aten.view %363, %364 : !torch.vtensor<[?,3,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,24576],f16>
    torch.bind_symbolic_shape %365, [%38], affine_map<()[s0] -> (s0, 24576)> : !torch.vtensor<[?,24576],f16>
    %int3_404 = torch.constant.int 3
    %int2_405 = torch.constant.int 2
    %int32_406 = torch.constant.int 32
    %int4_407 = torch.constant.int 4
    %int32_408 = torch.constant.int 32
    %366 = torch.prim.ListConstruct %122, %int3_404, %int2_405, %int32_406, %int4_407, %int32_408 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %367 = torch.aten.view %365, %366 : !torch.vtensor<[?,24576],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,4,32],f16>
    torch.bind_symbolic_shape %367, [%38], affine_map<()[s0] -> (s0, 3, 2, 32, 4, 32)> : !torch.vtensor<[?,3,2,32,4,32],f16>
    %int4_409 = torch.constant.int 4
    %int32_410 = torch.constant.int 32
    %368 = torch.prim.ListConstruct %357, %int4_409, %int32_410 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %369 = torch.aten.view %367, %368 : !torch.vtensor<[?,3,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,4,32],f16>
    torch.bind_symbolic_shape %369, [%38], affine_map<()[s0] -> (s0 * 192, 4, 32)> : !torch.vtensor<[?,4,32],f16>
    %int32_411 = torch.constant.int 32
    %370 = torch.aten.floor_divide.Scalar %arg2, %int32_411 : !torch.vtensor<[4],si64>, !torch.int -> !torch.vtensor<[4],si64>
    %int1_412 = torch.constant.int 1
    %371 = torch.aten.unsqueeze %370, %int1_412 : !torch.vtensor<[4],si64>, !torch.int -> !torch.vtensor<[4,1],si64>
    %int1_413 = torch.constant.int 1
    %false_414 = torch.constant.bool false
    %372 = torch.aten.gather %arg3, %int1_413, %371, %false_414 : !torch.vtensor<[4,?],si64>, !torch.int, !torch.vtensor<[4,1],si64>, !torch.bool -> !torch.vtensor<[4,1],si64>
    %int32_415 = torch.constant.int 32
    %373 = torch.aten.remainder.Scalar %arg2, %int32_415 : !torch.vtensor<[4],si64>, !torch.int -> !torch.vtensor<[4],si64>
    %int1_416 = torch.constant.int 1
    %374 = torch.aten.unsqueeze %373, %int1_416 : !torch.vtensor<[4],si64>, !torch.int -> !torch.vtensor<[4,1],si64>
    %none_417 = torch.constant.none
    %375 = torch.aten.clone %17, %none_417 : !torch.vtensor<[],si64>, !torch.none -> !torch.vtensor<[],si64>
    %int0_418 = torch.constant.int 0
    %376 = torch.aten.unsqueeze %375, %int0_418 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int4_419 = torch.constant.int 4
    %int1_420 = torch.constant.int 1
    %377 = torch.prim.ListConstruct %int4_419, %int1_420 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_421 = torch.constant.int 1
    %int1_422 = torch.constant.int 1
    %378 = torch.prim.ListConstruct %int1_421, %int1_422 : (!torch.int, !torch.int) -> !torch.list<int>
    %int4_423 = torch.constant.int 4
    %int0_424 = torch.constant.int 0
    %cpu_425 = torch.constant.device "cpu"
    %false_426 = torch.constant.bool false
    %379 = torch.aten.empty_strided %377, %378, %int4_423, %int0_424, %cpu_425, %false_426 : !torch.list<int>, !torch.list<int>, !torch.int, !torch.int, !torch.Device, !torch.bool -> !torch.vtensor<[4,1],si64>
    %int1_427 = torch.constant.int 1
    %380 = torch.aten.fill.Scalar %379, %int1_427 : !torch.vtensor<[4,1],si64>, !torch.int -> !torch.vtensor<[4,1],si64>
    %int4_428 = torch.constant.int 4
    %int1_429 = torch.constant.int 1
    %381 = torch.prim.ListConstruct %int4_428, %int1_429 : (!torch.int, !torch.int) -> !torch.list<int>
    %382 = torch.aten.repeat %376, %381 : !torch.vtensor<[1],si64>, !torch.list<int> -> !torch.vtensor<[4,1],si64>
    %int3_430 = torch.constant.int 3
    %383 = torch.aten.mul.Scalar %372, %int3_430 : !torch.vtensor<[4,1],si64>, !torch.int -> !torch.vtensor<[4,1],si64>
    %int1_431 = torch.constant.int 1
    %384 = torch.aten.add.Tensor %383, %380, %int1_431 : !torch.vtensor<[4,1],si64>, !torch.vtensor<[4,1],si64>, !torch.int -> !torch.vtensor<[4,1],si64>
    %int2_432 = torch.constant.int 2
    %385 = torch.aten.mul.Scalar %384, %int2_432 : !torch.vtensor<[4,1],si64>, !torch.int -> !torch.vtensor<[4,1],si64>
    %int1_433 = torch.constant.int 1
    %386 = torch.aten.add.Tensor %385, %382, %int1_433 : !torch.vtensor<[4,1],si64>, !torch.vtensor<[4,1],si64>, !torch.int -> !torch.vtensor<[4,1],si64>
    %int32_434 = torch.constant.int 32
    %387 = torch.aten.mul.Scalar %386, %int32_434 : !torch.vtensor<[4,1],si64>, !torch.int -> !torch.vtensor<[4,1],si64>
    %int1_435 = torch.constant.int 1
    %388 = torch.aten.add.Tensor %387, %374, %int1_435 : !torch.vtensor<[4,1],si64>, !torch.vtensor<[4,1],si64>, !torch.int -> !torch.vtensor<[4,1],si64>
    %389 = torch.prim.ListConstruct %388 : (!torch.vtensor<[4,1],si64>) -> !torch.list<optional<vtensor>>
    %false_436 = torch.constant.bool false
    %390 = torch.aten.index_put %369, %389, %321, %false_436 : !torch.vtensor<[?,4,32],f16>, !torch.list<optional<vtensor>>, !torch.vtensor<[4,1,4,32],f16>, !torch.bool -> !torch.vtensor<[?,4,32],f16>
    torch.bind_symbolic_shape %390, [%38], affine_map<()[s0] -> (s0 * 192, 4, 32)> : !torch.vtensor<[?,4,32],f16>
    %int3_437 = torch.constant.int 3
    %int2_438 = torch.constant.int 2
    %int32_439 = torch.constant.int 32
    %int4_440 = torch.constant.int 4
    %int32_441 = torch.constant.int 32
    %391 = torch.prim.ListConstruct %122, %int3_437, %int2_438, %int32_439, %int4_440, %int32_441 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %392 = torch.aten.view %390, %391 : !torch.vtensor<[?,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,4,32],f16>
    torch.bind_symbolic_shape %392, [%38], affine_map<()[s0] -> (s0, 3, 2, 32, 4, 32)> : !torch.vtensor<[?,3,2,32,4,32],f16>
    %int24576_442 = torch.constant.int 24576
    %393 = torch.prim.ListConstruct %122, %int24576_442 : (!torch.int, !torch.int) -> !torch.list<int>
    %394 = torch.aten.view %392, %393 : !torch.vtensor<[?,3,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,24576],f16>
    torch.bind_symbolic_shape %394, [%38], affine_map<()[s0] -> (s0, 24576)> : !torch.vtensor<[?,24576],f16>
    %int4_443 = torch.constant.int 4
    %395 = torch.prim.ListConstruct %int4_443, %39 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_444 = torch.constant.int 1
    %396 = torch.prim.ListConstruct %39, %int1_444 : (!torch.int, !torch.int) -> !torch.list<int>
    %int4_445 = torch.constant.int 4
    %int0_446 = torch.constant.int 0
    %cpu_447 = torch.constant.device "cpu"
    %false_448 = torch.constant.bool false
    %397 = torch.aten.empty_strided %395, %396, %int4_445, %int0_446, %cpu_447, %false_448 : !torch.list<int>, !torch.list<int>, !torch.int, !torch.int, !torch.Device, !torch.bool -> !torch.vtensor<[4,?],si64>
    torch.bind_symbolic_shape %397, [%37], affine_map<()[s0] -> (4, s0)> : !torch.vtensor<[4,?],si64>
    %int1_449 = torch.constant.int 1
    %398 = torch.aten.fill.Scalar %397, %int1_449 : !torch.vtensor<[4,?],si64>, !torch.int -> !torch.vtensor<[4,?],si64>
    torch.bind_symbolic_shape %398, [%37], affine_map<()[s0] -> (4, s0)> : !torch.vtensor<[4,?],si64>
    %int3_450 = torch.constant.int 3
    %399 = torch.aten.mul.Scalar %arg3, %int3_450 : !torch.vtensor<[4,?],si64>, !torch.int -> !torch.vtensor<[4,?],si64>
    torch.bind_symbolic_shape %399, [%37], affine_map<()[s0] -> (4, s0)> : !torch.vtensor<[4,?],si64>
    %int1_451 = torch.constant.int 1
    %400 = torch.aten.add.Tensor %399, %398, %int1_451 : !torch.vtensor<[4,?],si64>, !torch.vtensor<[4,?],si64>, !torch.int -> !torch.vtensor<[4,?],si64>
    torch.bind_symbolic_shape %400, [%37], affine_map<()[s0] -> (4, s0)> : !torch.vtensor<[4,?],si64>
    %int4_452 = torch.constant.int 4
    %401 = torch.aten.mul.int %int4_452, %39 : !torch.int, !torch.int -> !torch.int
    %402 = torch.prim.ListConstruct %401 : (!torch.int) -> !torch.list<int>
    %403 = torch.aten.view %400, %402 : !torch.vtensor<[4,?],si64>, !torch.list<int> -> !torch.vtensor<[?],si64>
    torch.bind_symbolic_shape %403, [%37], affine_map<()[s0] -> (s0 * 4)> : !torch.vtensor<[?],si64>
    %int3_453 = torch.constant.int 3
    %int2_454 = torch.constant.int 2
    %int32_455 = torch.constant.int 32
    %int4_456 = torch.constant.int 4
    %int32_457 = torch.constant.int 32
    %404 = torch.prim.ListConstruct %122, %int3_453, %int2_454, %int32_455, %int4_456, %int32_457 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %405 = torch.aten.view %394, %404 : !torch.vtensor<[?,24576],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,4,32],f16>
    torch.bind_symbolic_shape %405, [%38], affine_map<()[s0] -> (s0, 3, 2, 32, 4, 32)> : !torch.vtensor<[?,3,2,32,4,32],f16>
    %int3_458 = torch.constant.int 3
    %406 = torch.aten.mul.int %122, %int3_458 : !torch.int, !torch.int -> !torch.int
    %int2_459 = torch.constant.int 2
    %int32_460 = torch.constant.int 32
    %int4_461 = torch.constant.int 4
    %int32_462 = torch.constant.int 32
    %407 = torch.prim.ListConstruct %406, %int2_459, %int32_460, %int4_461, %int32_462 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %408 = torch.aten.view %405, %407 : !torch.vtensor<[?,3,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,2,32,4,32],f16>
    torch.bind_symbolic_shape %408, [%38], affine_map<()[s0] -> (s0 * 3, 2, 32, 4, 32)> : !torch.vtensor<[?,2,32,4,32],f16>
    %int0_463 = torch.constant.int 0
    %409 = torch.aten.index_select %408, %int0_463, %403 : !torch.vtensor<[?,2,32,4,32],f16>, !torch.int, !torch.vtensor<[?],si64> -> !torch.vtensor<[?,2,32,4,32],f16>
    torch.bind_symbolic_shape %409, [%37], affine_map<()[s0] -> (s0 * 4, 2, 32, 4, 32)> : !torch.vtensor<[?,2,32,4,32],f16>
    %int4_464 = torch.constant.int 4
    %int2_465 = torch.constant.int 2
    %int32_466 = torch.constant.int 32
    %int4_467 = torch.constant.int 4
    %int32_468 = torch.constant.int 32
    %410 = torch.prim.ListConstruct %int4_464, %39, %int2_465, %int32_466, %int4_467, %int32_468 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %411 = torch.aten.view %409, %410 : !torch.vtensor<[?,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[4,?,2,32,4,32],f16>
    torch.bind_symbolic_shape %411, [%37], affine_map<()[s0] -> (4, s0, 2, 32, 4, 32)> : !torch.vtensor<[4,?,2,32,4,32],f16>
    %int0_469 = torch.constant.int 0
    %int0_470 = torch.constant.int 0
    %int9223372036854775807_471 = torch.constant.int 9223372036854775807
    %int1_472 = torch.constant.int 1
    %412 = torch.aten.slice.Tensor %411, %int0_469, %int0_470, %int9223372036854775807_471, %int1_472 : !torch.vtensor<[4,?,2,32,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,?,2,32,4,32],f16>
    torch.bind_symbolic_shape %412, [%37], affine_map<()[s0] -> (4, s0, 2, 32, 4, 32)> : !torch.vtensor<[4,?,2,32,4,32],f16>
    %int1_473 = torch.constant.int 1
    %int0_474 = torch.constant.int 0
    %int9223372036854775807_475 = torch.constant.int 9223372036854775807
    %int1_476 = torch.constant.int 1
    %413 = torch.aten.slice.Tensor %412, %int1_473, %int0_474, %int9223372036854775807_475, %int1_476 : !torch.vtensor<[4,?,2,32,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,?,2,32,4,32],f16>
    torch.bind_symbolic_shape %413, [%37], affine_map<()[s0] -> (4, s0, 2, 32, 4, 32)> : !torch.vtensor<[4,?,2,32,4,32],f16>
    %int2_477 = torch.constant.int 2
    %int0_478 = torch.constant.int 0
    %414 = torch.aten.select.int %413, %int2_477, %int0_478 : !torch.vtensor<[4,?,2,32,4,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[4,?,32,4,32],f16>
    torch.bind_symbolic_shape %414, [%37], affine_map<()[s0] -> (4, s0, 32, 4, 32)> : !torch.vtensor<[4,?,32,4,32],f16>
    %int32_479 = torch.constant.int 32
    %415 = torch.aten.mul.int %39, %int32_479 : !torch.int, !torch.int -> !torch.int
    %int2_480 = torch.constant.int 2
    %int0_481 = torch.constant.int 0
    %int1_482 = torch.constant.int 1
    %416 = torch.aten.slice.Tensor %414, %int2_480, %int0_481, %415, %int1_482 : !torch.vtensor<[4,?,32,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,?,32,4,32],f16>
    torch.bind_symbolic_shape %416, [%37], affine_map<()[s0] -> (4, s0, 32, 4, 32)> : !torch.vtensor<[4,?,32,4,32],f16>
    %int0_483 = torch.constant.int 0
    %417 = torch.aten.clone %416, %int0_483 : !torch.vtensor<[4,?,32,4,32],f16>, !torch.int -> !torch.vtensor<[4,?,32,4,32],f16>
    torch.bind_symbolic_shape %417, [%37], affine_map<()[s0] -> (4, s0, 32, 4, 32)> : !torch.vtensor<[4,?,32,4,32],f16>
    %int1_484 = torch.constant.int 1
    %418 = torch.aten.size.int %413, %int1_484 : !torch.vtensor<[4,?,2,32,4,32],f16>, !torch.int -> !torch.int
    %int32_485 = torch.constant.int 32
    %419 = torch.aten.mul.int %418, %int32_485 : !torch.int, !torch.int -> !torch.int
    %int4_486 = torch.constant.int 4
    %int4_487 = torch.constant.int 4
    %int32_488 = torch.constant.int 32
    %420 = torch.prim.ListConstruct %int4_486, %419, %int4_487, %int32_488 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %421 = torch.aten._unsafe_view %417, %420 : !torch.vtensor<[4,?,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[4,?,4,32],f16>
    torch.bind_symbolic_shape %421, [%37], affine_map<()[s0] -> (4, s0 * 32, 4, 32)> : !torch.vtensor<[4,?,4,32],f16>
    %int0_489 = torch.constant.int 0
    %int0_490 = torch.constant.int 0
    %int9223372036854775807_491 = torch.constant.int 9223372036854775807
    %int1_492 = torch.constant.int 1
    %422 = torch.aten.slice.Tensor %421, %int0_489, %int0_490, %int9223372036854775807_491, %int1_492 : !torch.vtensor<[4,?,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,?,4,32],f16>
    torch.bind_symbolic_shape %422, [%37], affine_map<()[s0] -> (4, s0 * 32, 4, 32)> : !torch.vtensor<[4,?,4,32],f16>
    %int0_493 = torch.constant.int 0
    %int0_494 = torch.constant.int 0
    %int9223372036854775807_495 = torch.constant.int 9223372036854775807
    %int1_496 = torch.constant.int 1
    %423 = torch.aten.slice.Tensor %411, %int0_493, %int0_494, %int9223372036854775807_495, %int1_496 : !torch.vtensor<[4,?,2,32,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,?,2,32,4,32],f16>
    torch.bind_symbolic_shape %423, [%37], affine_map<()[s0] -> (4, s0, 2, 32, 4, 32)> : !torch.vtensor<[4,?,2,32,4,32],f16>
    %int1_497 = torch.constant.int 1
    %int0_498 = torch.constant.int 0
    %int9223372036854775807_499 = torch.constant.int 9223372036854775807
    %int1_500 = torch.constant.int 1
    %424 = torch.aten.slice.Tensor %423, %int1_497, %int0_498, %int9223372036854775807_499, %int1_500 : !torch.vtensor<[4,?,2,32,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,?,2,32,4,32],f16>
    torch.bind_symbolic_shape %424, [%37], affine_map<()[s0] -> (4, s0, 2, 32, 4, 32)> : !torch.vtensor<[4,?,2,32,4,32],f16>
    %int2_501 = torch.constant.int 2
    %int1_502 = torch.constant.int 1
    %425 = torch.aten.select.int %424, %int2_501, %int1_502 : !torch.vtensor<[4,?,2,32,4,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[4,?,32,4,32],f16>
    torch.bind_symbolic_shape %425, [%37], affine_map<()[s0] -> (4, s0, 32, 4, 32)> : !torch.vtensor<[4,?,32,4,32],f16>
    %int2_503 = torch.constant.int 2
    %int0_504 = torch.constant.int 0
    %int1_505 = torch.constant.int 1
    %426 = torch.aten.slice.Tensor %425, %int2_503, %int0_504, %415, %int1_505 : !torch.vtensor<[4,?,32,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,?,32,4,32],f16>
    torch.bind_symbolic_shape %426, [%37], affine_map<()[s0] -> (4, s0, 32, 4, 32)> : !torch.vtensor<[4,?,32,4,32],f16>
    %int0_506 = torch.constant.int 0
    %427 = torch.aten.clone %426, %int0_506 : !torch.vtensor<[4,?,32,4,32],f16>, !torch.int -> !torch.vtensor<[4,?,32,4,32],f16>
    torch.bind_symbolic_shape %427, [%37], affine_map<()[s0] -> (4, s0, 32, 4, 32)> : !torch.vtensor<[4,?,32,4,32],f16>
    %int1_507 = torch.constant.int 1
    %428 = torch.aten.size.int %424, %int1_507 : !torch.vtensor<[4,?,2,32,4,32],f16>, !torch.int -> !torch.int
    %int32_508 = torch.constant.int 32
    %429 = torch.aten.mul.int %428, %int32_508 : !torch.int, !torch.int -> !torch.int
    %int4_509 = torch.constant.int 4
    %int4_510 = torch.constant.int 4
    %int32_511 = torch.constant.int 32
    %430 = torch.prim.ListConstruct %int4_509, %429, %int4_510, %int32_511 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %431 = torch.aten._unsafe_view %427, %430 : !torch.vtensor<[4,?,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[4,?,4,32],f16>
    torch.bind_symbolic_shape %431, [%37], affine_map<()[s0] -> (4, s0 * 32, 4, 32)> : !torch.vtensor<[4,?,4,32],f16>
    %int0_512 = torch.constant.int 0
    %int0_513 = torch.constant.int 0
    %int9223372036854775807_514 = torch.constant.int 9223372036854775807
    %int1_515 = torch.constant.int 1
    %432 = torch.aten.slice.Tensor %431, %int0_512, %int0_513, %int9223372036854775807_514, %int1_515 : !torch.vtensor<[4,?,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,?,4,32],f16>
    torch.bind_symbolic_shape %432, [%37], affine_map<()[s0] -> (4, s0 * 32, 4, 32)> : !torch.vtensor<[4,?,4,32],f16>
    %int-2_516 = torch.constant.int -2
    %433 = torch.aten.unsqueeze %422, %int-2_516 : !torch.vtensor<[4,?,4,32],f16>, !torch.int -> !torch.vtensor<[4,?,4,1,32],f16>
    torch.bind_symbolic_shape %433, [%37], affine_map<()[s0] -> (4, s0 * 32, 4, 1, 32)> : !torch.vtensor<[4,?,4,1,32],f16>
    %int1_517 = torch.constant.int 1
    %434 = torch.aten.size.int %421, %int1_517 : !torch.vtensor<[4,?,4,32],f16>, !torch.int -> !torch.int
    %int4_518 = torch.constant.int 4
    %int4_519 = torch.constant.int 4
    %int2_520 = torch.constant.int 2
    %int32_521 = torch.constant.int 32
    %435 = torch.prim.ListConstruct %int4_518, %434, %int4_519, %int2_520, %int32_521 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_522 = torch.constant.bool false
    %436 = torch.aten.expand %433, %435, %false_522 : !torch.vtensor<[4,?,4,1,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[4,?,4,2,32],f16>
    torch.bind_symbolic_shape %436, [%37], affine_map<()[s0] -> (4, s0 * 32, 4, 2, 32)> : !torch.vtensor<[4,?,4,2,32],f16>
    %int0_523 = torch.constant.int 0
    %437 = torch.aten.clone %436, %int0_523 : !torch.vtensor<[4,?,4,2,32],f16>, !torch.int -> !torch.vtensor<[4,?,4,2,32],f16>
    torch.bind_symbolic_shape %437, [%37], affine_map<()[s0] -> (4, s0 * 32, 4, 2, 32)> : !torch.vtensor<[4,?,4,2,32],f16>
    %int4_524 = torch.constant.int 4
    %int8_525 = torch.constant.int 8
    %int32_526 = torch.constant.int 32
    %438 = torch.prim.ListConstruct %int4_524, %434, %int8_525, %int32_526 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %439 = torch.aten._unsafe_view %437, %438 : !torch.vtensor<[4,?,4,2,32],f16>, !torch.list<int> -> !torch.vtensor<[4,?,8,32],f16>
    torch.bind_symbolic_shape %439, [%37], affine_map<()[s0] -> (4, s0 * 32, 8, 32)> : !torch.vtensor<[4,?,8,32],f16>
    %int-2_527 = torch.constant.int -2
    %440 = torch.aten.unsqueeze %432, %int-2_527 : !torch.vtensor<[4,?,4,32],f16>, !torch.int -> !torch.vtensor<[4,?,4,1,32],f16>
    torch.bind_symbolic_shape %440, [%37], affine_map<()[s0] -> (4, s0 * 32, 4, 1, 32)> : !torch.vtensor<[4,?,4,1,32],f16>
    %int1_528 = torch.constant.int 1
    %441 = torch.aten.size.int %431, %int1_528 : !torch.vtensor<[4,?,4,32],f16>, !torch.int -> !torch.int
    %int4_529 = torch.constant.int 4
    %int4_530 = torch.constant.int 4
    %int2_531 = torch.constant.int 2
    %int32_532 = torch.constant.int 32
    %442 = torch.prim.ListConstruct %int4_529, %441, %int4_530, %int2_531, %int32_532 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_533 = torch.constant.bool false
    %443 = torch.aten.expand %440, %442, %false_533 : !torch.vtensor<[4,?,4,1,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[4,?,4,2,32],f16>
    torch.bind_symbolic_shape %443, [%37], affine_map<()[s0] -> (4, s0 * 32, 4, 2, 32)> : !torch.vtensor<[4,?,4,2,32],f16>
    %int0_534 = torch.constant.int 0
    %444 = torch.aten.clone %443, %int0_534 : !torch.vtensor<[4,?,4,2,32],f16>, !torch.int -> !torch.vtensor<[4,?,4,2,32],f16>
    torch.bind_symbolic_shape %444, [%37], affine_map<()[s0] -> (4, s0 * 32, 4, 2, 32)> : !torch.vtensor<[4,?,4,2,32],f16>
    %int4_535 = torch.constant.int 4
    %int8_536 = torch.constant.int 8
    %int32_537 = torch.constant.int 32
    %445 = torch.prim.ListConstruct %int4_535, %441, %int8_536, %int32_537 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %446 = torch.aten._unsafe_view %444, %445 : !torch.vtensor<[4,?,4,2,32],f16>, !torch.list<int> -> !torch.vtensor<[4,?,8,32],f16>
    torch.bind_symbolic_shape %446, [%37], affine_map<()[s0] -> (4, s0 * 32, 8, 32)> : !torch.vtensor<[4,?,8,32],f16>
    %int1_538 = torch.constant.int 1
    %int2_539 = torch.constant.int 2
    %447 = torch.aten.transpose.int %327, %int1_538, %int2_539 : !torch.vtensor<[4,1,8,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[4,8,1,32],f16>
    %int1_540 = torch.constant.int 1
    %int2_541 = torch.constant.int 2
    %448 = torch.aten.transpose.int %439, %int1_540, %int2_541 : !torch.vtensor<[4,?,8,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[4,8,?,32],f16>
    torch.bind_symbolic_shape %448, [%37], affine_map<()[s0] -> (4, 8, s0 * 32, 32)> : !torch.vtensor<[4,8,?,32],f16>
    %int1_542 = torch.constant.int 1
    %int2_543 = torch.constant.int 2
    %449 = torch.aten.transpose.int %446, %int1_542, %int2_543 : !torch.vtensor<[4,?,8,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[4,8,?,32],f16>
    torch.bind_symbolic_shape %449, [%37], affine_map<()[s0] -> (4, 8, s0 * 32, 32)> : !torch.vtensor<[4,8,?,32],f16>
    %float0.000000e00_544 = torch.constant.float 0.000000e+00
    %false_545 = torch.constant.bool false
    %none_546 = torch.constant.none
    %450:2 = torch.operator "torch.aten._scaled_dot_product_flash_attention_for_cpu"(%447, %448, %449, %float0.000000e00_544, %false_545, %49, %none_546) : (!torch.vtensor<[4,8,1,32],f16>, !torch.vtensor<[4,8,?,32],f16>, !torch.vtensor<[4,8,?,32],f16>, !torch.float, !torch.bool, !torch.vtensor<[4,1,1,?],f16>, !torch.none) -> (!torch.vtensor<[4,8,1,32],f16>, !torch.vtensor<[4,8,1],f32>) 
    %int1_547 = torch.constant.int 1
    %int2_548 = torch.constant.int 2
    %451 = torch.aten.transpose.int %450#0, %int1_547, %int2_548 : !torch.vtensor<[4,8,1,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[4,1,8,32],f16>
    %int4_549 = torch.constant.int 4
    %int1_550 = torch.constant.int 1
    %int256_551 = torch.constant.int 256
    %452 = torch.prim.ListConstruct %int4_549, %int1_550, %int256_551 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %453 = torch.aten.view %451, %452 : !torch.vtensor<[4,1,8,32],f16>, !torch.list<int> -> !torch.vtensor<[4,1,256],f16>
    %int5_552 = torch.constant.int 5
    %454 = torch.prims.convert_element_type %18, %int5_552 : !torch.vtensor<[256,256],f32>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int-2_553 = torch.constant.int -2
    %int-1_554 = torch.constant.int -1
    %455 = torch.aten.transpose.int %454, %int-2_553, %int-1_554 : !torch.vtensor<[256,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,256],f16>
    %int4_555 = torch.constant.int 4
    %int256_556 = torch.constant.int 256
    %456 = torch.prim.ListConstruct %int4_555, %int256_556 : (!torch.int, !torch.int) -> !torch.list<int>
    %457 = torch.aten.view %453, %456 : !torch.vtensor<[4,1,256],f16>, !torch.list<int> -> !torch.vtensor<[4,256],f16>
    %458 = torch.aten.mm %457, %455 : !torch.vtensor<[4,256],f16>, !torch.vtensor<[256,256],f16> -> !torch.vtensor<[4,256],f16>
    %int4_557 = torch.constant.int 4
    %int1_558 = torch.constant.int 1
    %int256_559 = torch.constant.int 256
    %459 = torch.prim.ListConstruct %int4_557, %int1_558, %int256_559 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %460 = torch.aten.view %458, %459 : !torch.vtensor<[4,256],f16>, !torch.list<int> -> !torch.vtensor<[4,1,256],f16>
    %int1_560 = torch.constant.int 1
    %461 = torch.aten.add.Tensor %284, %460, %int1_560 : !torch.vtensor<[4,1,256],f16>, !torch.vtensor<[4,1,256],f16>, !torch.int -> !torch.vtensor<[4,1,256],f16>
    %int6_561 = torch.constant.int 6
    %462 = torch.prims.convert_element_type %461, %int6_561 : !torch.vtensor<[4,1,256],f16>, !torch.int -> !torch.vtensor<[4,1,256],f32>
    %int2_562 = torch.constant.int 2
    %463 = torch.aten.pow.Tensor_Scalar %462, %int2_562 : !torch.vtensor<[4,1,256],f32>, !torch.int -> !torch.vtensor<[4,1,256],f32>
    %int-1_563 = torch.constant.int -1
    %464 = torch.prim.ListConstruct %int-1_563 : (!torch.int) -> !torch.list<int>
    %true_564 = torch.constant.bool true
    %none_565 = torch.constant.none
    %465 = torch.aten.mean.dim %463, %464, %true_564, %none_565 : !torch.vtensor<[4,1,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[4,1,1],f32>
    %float1.000000e-02_566 = torch.constant.float 1.000000e-02
    %int1_567 = torch.constant.int 1
    %466 = torch.aten.add.Scalar %465, %float1.000000e-02_566, %int1_567 : !torch.vtensor<[4,1,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[4,1,1],f32>
    %467 = torch.aten.rsqrt %466 : !torch.vtensor<[4,1,1],f32> -> !torch.vtensor<[4,1,1],f32>
    %468 = torch.aten.mul.Tensor %462, %467 : !torch.vtensor<[4,1,256],f32>, !torch.vtensor<[4,1,1],f32> -> !torch.vtensor<[4,1,256],f32>
    %int5_568 = torch.constant.int 5
    %469 = torch.prims.convert_element_type %468, %int5_568 : !torch.vtensor<[4,1,256],f32>, !torch.int -> !torch.vtensor<[4,1,256],f16>
    %470 = torch.aten.mul.Tensor %19, %469 : !torch.vtensor<[256],f32>, !torch.vtensor<[4,1,256],f16> -> !torch.vtensor<[4,1,256],f32>
    %int5_569 = torch.constant.int 5
    %471 = torch.prims.convert_element_type %470, %int5_569 : !torch.vtensor<[4,1,256],f32>, !torch.int -> !torch.vtensor<[4,1,256],f16>
    %int5_570 = torch.constant.int 5
    %472 = torch.prims.convert_element_type %20, %int5_570 : !torch.vtensor<[23,256],f32>, !torch.int -> !torch.vtensor<[23,256],f16>
    %int-2_571 = torch.constant.int -2
    %int-1_572 = torch.constant.int -1
    %473 = torch.aten.transpose.int %472, %int-2_571, %int-1_572 : !torch.vtensor<[23,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,23],f16>
    %int4_573 = torch.constant.int 4
    %int256_574 = torch.constant.int 256
    %474 = torch.prim.ListConstruct %int4_573, %int256_574 : (!torch.int, !torch.int) -> !torch.list<int>
    %475 = torch.aten.view %471, %474 : !torch.vtensor<[4,1,256],f16>, !torch.list<int> -> !torch.vtensor<[4,256],f16>
    %476 = torch.aten.mm %475, %473 : !torch.vtensor<[4,256],f16>, !torch.vtensor<[256,23],f16> -> !torch.vtensor<[4,23],f16>
    %int4_575 = torch.constant.int 4
    %int1_576 = torch.constant.int 1
    %int23_577 = torch.constant.int 23
    %477 = torch.prim.ListConstruct %int4_575, %int1_576, %int23_577 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %478 = torch.aten.view %476, %477 : !torch.vtensor<[4,23],f16>, !torch.list<int> -> !torch.vtensor<[4,1,23],f16>
    %479 = torch.aten.silu %478 : !torch.vtensor<[4,1,23],f16> -> !torch.vtensor<[4,1,23],f16>
    %int5_578 = torch.constant.int 5
    %480 = torch.prims.convert_element_type %21, %int5_578 : !torch.vtensor<[23,256],f32>, !torch.int -> !torch.vtensor<[23,256],f16>
    %int-2_579 = torch.constant.int -2
    %int-1_580 = torch.constant.int -1
    %481 = torch.aten.transpose.int %480, %int-2_579, %int-1_580 : !torch.vtensor<[23,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,23],f16>
    %int4_581 = torch.constant.int 4
    %int256_582 = torch.constant.int 256
    %482 = torch.prim.ListConstruct %int4_581, %int256_582 : (!torch.int, !torch.int) -> !torch.list<int>
    %483 = torch.aten.view %471, %482 : !torch.vtensor<[4,1,256],f16>, !torch.list<int> -> !torch.vtensor<[4,256],f16>
    %484 = torch.aten.mm %483, %481 : !torch.vtensor<[4,256],f16>, !torch.vtensor<[256,23],f16> -> !torch.vtensor<[4,23],f16>
    %int4_583 = torch.constant.int 4
    %int1_584 = torch.constant.int 1
    %int23_585 = torch.constant.int 23
    %485 = torch.prim.ListConstruct %int4_583, %int1_584, %int23_585 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %486 = torch.aten.view %484, %485 : !torch.vtensor<[4,23],f16>, !torch.list<int> -> !torch.vtensor<[4,1,23],f16>
    %487 = torch.aten.mul.Tensor %479, %486 : !torch.vtensor<[4,1,23],f16>, !torch.vtensor<[4,1,23],f16> -> !torch.vtensor<[4,1,23],f16>
    %int5_586 = torch.constant.int 5
    %488 = torch.prims.convert_element_type %22, %int5_586 : !torch.vtensor<[256,23],f32>, !torch.int -> !torch.vtensor<[256,23],f16>
    %int-2_587 = torch.constant.int -2
    %int-1_588 = torch.constant.int -1
    %489 = torch.aten.transpose.int %488, %int-2_587, %int-1_588 : !torch.vtensor<[256,23],f16>, !torch.int, !torch.int -> !torch.vtensor<[23,256],f16>
    %int4_589 = torch.constant.int 4
    %int23_590 = torch.constant.int 23
    %490 = torch.prim.ListConstruct %int4_589, %int23_590 : (!torch.int, !torch.int) -> !torch.list<int>
    %491 = torch.aten.view %487, %490 : !torch.vtensor<[4,1,23],f16>, !torch.list<int> -> !torch.vtensor<[4,23],f16>
    %492 = torch.aten.mm %491, %489 : !torch.vtensor<[4,23],f16>, !torch.vtensor<[23,256],f16> -> !torch.vtensor<[4,256],f16>
    %int4_591 = torch.constant.int 4
    %int1_592 = torch.constant.int 1
    %int256_593 = torch.constant.int 256
    %493 = torch.prim.ListConstruct %int4_591, %int1_592, %int256_593 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %494 = torch.aten.view %492, %493 : !torch.vtensor<[4,256],f16>, !torch.list<int> -> !torch.vtensor<[4,1,256],f16>
    %int1_594 = torch.constant.int 1
    %495 = torch.aten.add.Tensor %461, %494, %int1_594 : !torch.vtensor<[4,1,256],f16>, !torch.vtensor<[4,1,256],f16>, !torch.int -> !torch.vtensor<[4,1,256],f16>
    %int6_595 = torch.constant.int 6
    %496 = torch.prims.convert_element_type %495, %int6_595 : !torch.vtensor<[4,1,256],f16>, !torch.int -> !torch.vtensor<[4,1,256],f32>
    %int2_596 = torch.constant.int 2
    %497 = torch.aten.pow.Tensor_Scalar %496, %int2_596 : !torch.vtensor<[4,1,256],f32>, !torch.int -> !torch.vtensor<[4,1,256],f32>
    %int-1_597 = torch.constant.int -1
    %498 = torch.prim.ListConstruct %int-1_597 : (!torch.int) -> !torch.list<int>
    %true_598 = torch.constant.bool true
    %none_599 = torch.constant.none
    %499 = torch.aten.mean.dim %497, %498, %true_598, %none_599 : !torch.vtensor<[4,1,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[4,1,1],f32>
    %float1.000000e-02_600 = torch.constant.float 1.000000e-02
    %int1_601 = torch.constant.int 1
    %500 = torch.aten.add.Scalar %499, %float1.000000e-02_600, %int1_601 : !torch.vtensor<[4,1,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[4,1,1],f32>
    %501 = torch.aten.rsqrt %500 : !torch.vtensor<[4,1,1],f32> -> !torch.vtensor<[4,1,1],f32>
    %502 = torch.aten.mul.Tensor %496, %501 : !torch.vtensor<[4,1,256],f32>, !torch.vtensor<[4,1,1],f32> -> !torch.vtensor<[4,1,256],f32>
    %int5_602 = torch.constant.int 5
    %503 = torch.prims.convert_element_type %502, %int5_602 : !torch.vtensor<[4,1,256],f32>, !torch.int -> !torch.vtensor<[4,1,256],f16>
    %504 = torch.aten.mul.Tensor %23, %503 : !torch.vtensor<[256],f32>, !torch.vtensor<[4,1,256],f16> -> !torch.vtensor<[4,1,256],f32>
    %int5_603 = torch.constant.int 5
    %505 = torch.prims.convert_element_type %504, %int5_603 : !torch.vtensor<[4,1,256],f32>, !torch.int -> !torch.vtensor<[4,1,256],f16>
    %int5_604 = torch.constant.int 5
    %506 = torch.prims.convert_element_type %24, %int5_604 : !torch.vtensor<[256,256],f32>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int-2_605 = torch.constant.int -2
    %int-1_606 = torch.constant.int -1
    %507 = torch.aten.transpose.int %506, %int-2_605, %int-1_606 : !torch.vtensor<[256,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,256],f16>
    %int4_607 = torch.constant.int 4
    %int256_608 = torch.constant.int 256
    %508 = torch.prim.ListConstruct %int4_607, %int256_608 : (!torch.int, !torch.int) -> !torch.list<int>
    %509 = torch.aten.view %505, %508 : !torch.vtensor<[4,1,256],f16>, !torch.list<int> -> !torch.vtensor<[4,256],f16>
    %510 = torch.aten.mm %509, %507 : !torch.vtensor<[4,256],f16>, !torch.vtensor<[256,256],f16> -> !torch.vtensor<[4,256],f16>
    %int4_609 = torch.constant.int 4
    %int1_610 = torch.constant.int 1
    %int256_611 = torch.constant.int 256
    %511 = torch.prim.ListConstruct %int4_609, %int1_610, %int256_611 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %512 = torch.aten.view %510, %511 : !torch.vtensor<[4,256],f16>, !torch.list<int> -> !torch.vtensor<[4,1,256],f16>
    %int5_612 = torch.constant.int 5
    %513 = torch.prims.convert_element_type %25, %int5_612 : !torch.vtensor<[128,256],f32>, !torch.int -> !torch.vtensor<[128,256],f16>
    %int-2_613 = torch.constant.int -2
    %int-1_614 = torch.constant.int -1
    %514 = torch.aten.transpose.int %513, %int-2_613, %int-1_614 : !torch.vtensor<[128,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,128],f16>
    %int4_615 = torch.constant.int 4
    %int256_616 = torch.constant.int 256
    %515 = torch.prim.ListConstruct %int4_615, %int256_616 : (!torch.int, !torch.int) -> !torch.list<int>
    %516 = torch.aten.view %505, %515 : !torch.vtensor<[4,1,256],f16>, !torch.list<int> -> !torch.vtensor<[4,256],f16>
    %517 = torch.aten.mm %516, %514 : !torch.vtensor<[4,256],f16>, !torch.vtensor<[256,128],f16> -> !torch.vtensor<[4,128],f16>
    %int4_617 = torch.constant.int 4
    %int1_618 = torch.constant.int 1
    %int128_619 = torch.constant.int 128
    %518 = torch.prim.ListConstruct %int4_617, %int1_618, %int128_619 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %519 = torch.aten.view %517, %518 : !torch.vtensor<[4,128],f16>, !torch.list<int> -> !torch.vtensor<[4,1,128],f16>
    %int5_620 = torch.constant.int 5
    %520 = torch.prims.convert_element_type %26, %int5_620 : !torch.vtensor<[128,256],f32>, !torch.int -> !torch.vtensor<[128,256],f16>
    %int-2_621 = torch.constant.int -2
    %int-1_622 = torch.constant.int -1
    %521 = torch.aten.transpose.int %520, %int-2_621, %int-1_622 : !torch.vtensor<[128,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,128],f16>
    %int4_623 = torch.constant.int 4
    %int256_624 = torch.constant.int 256
    %522 = torch.prim.ListConstruct %int4_623, %int256_624 : (!torch.int, !torch.int) -> !torch.list<int>
    %523 = torch.aten.view %505, %522 : !torch.vtensor<[4,1,256],f16>, !torch.list<int> -> !torch.vtensor<[4,256],f16>
    %524 = torch.aten.mm %523, %521 : !torch.vtensor<[4,256],f16>, !torch.vtensor<[256,128],f16> -> !torch.vtensor<[4,128],f16>
    %int4_625 = torch.constant.int 4
    %int1_626 = torch.constant.int 1
    %int128_627 = torch.constant.int 128
    %525 = torch.prim.ListConstruct %int4_625, %int1_626, %int128_627 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %526 = torch.aten.view %524, %525 : !torch.vtensor<[4,128],f16>, !torch.list<int> -> !torch.vtensor<[4,1,128],f16>
    %int4_628 = torch.constant.int 4
    %int1_629 = torch.constant.int 1
    %int8_630 = torch.constant.int 8
    %int32_631 = torch.constant.int 32
    %527 = torch.prim.ListConstruct %int4_628, %int1_629, %int8_630, %int32_631 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %528 = torch.aten.view %512, %527 : !torch.vtensor<[4,1,256],f16>, !torch.list<int> -> !torch.vtensor<[4,1,8,32],f16>
    %int4_632 = torch.constant.int 4
    %int1_633 = torch.constant.int 1
    %int4_634 = torch.constant.int 4
    %int32_635 = torch.constant.int 32
    %529 = torch.prim.ListConstruct %int4_632, %int1_633, %int4_634, %int32_635 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %530 = torch.aten.view %519, %529 : !torch.vtensor<[4,1,128],f16>, !torch.list<int> -> !torch.vtensor<[4,1,4,32],f16>
    %int4_636 = torch.constant.int 4
    %int1_637 = torch.constant.int 1
    %int4_638 = torch.constant.int 4
    %int32_639 = torch.constant.int 32
    %531 = torch.prim.ListConstruct %int4_636, %int1_637, %int4_638, %int32_639 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %532 = torch.aten.view %526, %531 : !torch.vtensor<[4,1,128],f16>, !torch.list<int> -> !torch.vtensor<[4,1,4,32],f16>
    %int6_640 = torch.constant.int 6
    %533 = torch.prims.convert_element_type %528, %int6_640 : !torch.vtensor<[4,1,8,32],f16>, !torch.int -> !torch.vtensor<[4,1,8,32],f32>
    %534 = torch_c.to_builtin_tensor %533 : !torch.vtensor<[4,1,8,32],f32> -> tensor<4x1x8x32xf32>
    %535 = torch_c.to_builtin_tensor %70 : !torch.vtensor<[4,1,32],f32> -> tensor<4x1x32xf32>
    %536 = util.call @sharktank_rotary_embedding_4_1_8_32_f32(%534, %535) : (tensor<4x1x8x32xf32>, tensor<4x1x32xf32>) -> tensor<4x1x8x32xf32>
    %537 = torch_c.from_builtin_tensor %536 : tensor<4x1x8x32xf32> -> !torch.vtensor<[4,1,8,32],f32>
    %int5_641 = torch.constant.int 5
    %538 = torch.prims.convert_element_type %537, %int5_641 : !torch.vtensor<[4,1,8,32],f32>, !torch.int -> !torch.vtensor<[4,1,8,32],f16>
    %int6_642 = torch.constant.int 6
    %539 = torch.prims.convert_element_type %530, %int6_642 : !torch.vtensor<[4,1,4,32],f16>, !torch.int -> !torch.vtensor<[4,1,4,32],f32>
    %540 = torch_c.to_builtin_tensor %539 : !torch.vtensor<[4,1,4,32],f32> -> tensor<4x1x4x32xf32>
    %541 = torch_c.to_builtin_tensor %70 : !torch.vtensor<[4,1,32],f32> -> tensor<4x1x32xf32>
    %542 = util.call @sharktank_rotary_embedding_4_1_4_32_f32(%540, %541) : (tensor<4x1x4x32xf32>, tensor<4x1x32xf32>) -> tensor<4x1x4x32xf32>
    %543 = torch_c.from_builtin_tensor %542 : tensor<4x1x4x32xf32> -> !torch.vtensor<[4,1,4,32],f32>
    %int5_643 = torch.constant.int 5
    %544 = torch.prims.convert_element_type %543, %int5_643 : !torch.vtensor<[4,1,4,32],f32>, !torch.int -> !torch.vtensor<[4,1,4,32],f16>
    %int32_644 = torch.constant.int 32
    %545 = torch.aten.floor_divide.Scalar %arg2, %int32_644 : !torch.vtensor<[4],si64>, !torch.int -> !torch.vtensor<[4],si64>
    %int1_645 = torch.constant.int 1
    %546 = torch.aten.unsqueeze %545, %int1_645 : !torch.vtensor<[4],si64>, !torch.int -> !torch.vtensor<[4,1],si64>
    %int1_646 = torch.constant.int 1
    %false_647 = torch.constant.bool false
    %547 = torch.aten.gather %arg3, %int1_646, %546, %false_647 : !torch.vtensor<[4,?],si64>, !torch.int, !torch.vtensor<[4,1],si64>, !torch.bool -> !torch.vtensor<[4,1],si64>
    %int32_648 = torch.constant.int 32
    %548 = torch.aten.remainder.Scalar %arg2, %int32_648 : !torch.vtensor<[4],si64>, !torch.int -> !torch.vtensor<[4],si64>
    %int1_649 = torch.constant.int 1
    %549 = torch.aten.unsqueeze %548, %int1_649 : !torch.vtensor<[4],si64>, !torch.int -> !torch.vtensor<[4,1],si64>
    %none_650 = torch.constant.none
    %550 = torch.aten.clone %27, %none_650 : !torch.vtensor<[],si64>, !torch.none -> !torch.vtensor<[],si64>
    %int0_651 = torch.constant.int 0
    %551 = torch.aten.unsqueeze %550, %int0_651 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int4_652 = torch.constant.int 4
    %int1_653 = torch.constant.int 1
    %552 = torch.prim.ListConstruct %int4_652, %int1_653 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_654 = torch.constant.int 1
    %int1_655 = torch.constant.int 1
    %553 = torch.prim.ListConstruct %int1_654, %int1_655 : (!torch.int, !torch.int) -> !torch.list<int>
    %int4_656 = torch.constant.int 4
    %int0_657 = torch.constant.int 0
    %cpu_658 = torch.constant.device "cpu"
    %false_659 = torch.constant.bool false
    %554 = torch.aten.empty_strided %552, %553, %int4_656, %int0_657, %cpu_658, %false_659 : !torch.list<int>, !torch.list<int>, !torch.int, !torch.int, !torch.Device, !torch.bool -> !torch.vtensor<[4,1],si64>
    %int2_660 = torch.constant.int 2
    %555 = torch.aten.fill.Scalar %554, %int2_660 : !torch.vtensor<[4,1],si64>, !torch.int -> !torch.vtensor<[4,1],si64>
    %int4_661 = torch.constant.int 4
    %int1_662 = torch.constant.int 1
    %556 = torch.prim.ListConstruct %int4_661, %int1_662 : (!torch.int, !torch.int) -> !torch.list<int>
    %557 = torch.aten.repeat %551, %556 : !torch.vtensor<[1],si64>, !torch.list<int> -> !torch.vtensor<[4,1],si64>
    %int3_663 = torch.constant.int 3
    %558 = torch.aten.mul.Scalar %547, %int3_663 : !torch.vtensor<[4,1],si64>, !torch.int -> !torch.vtensor<[4,1],si64>
    %int1_664 = torch.constant.int 1
    %559 = torch.aten.add.Tensor %558, %555, %int1_664 : !torch.vtensor<[4,1],si64>, !torch.vtensor<[4,1],si64>, !torch.int -> !torch.vtensor<[4,1],si64>
    %int2_665 = torch.constant.int 2
    %560 = torch.aten.mul.Scalar %559, %int2_665 : !torch.vtensor<[4,1],si64>, !torch.int -> !torch.vtensor<[4,1],si64>
    %int1_666 = torch.constant.int 1
    %561 = torch.aten.add.Tensor %560, %557, %int1_666 : !torch.vtensor<[4,1],si64>, !torch.vtensor<[4,1],si64>, !torch.int -> !torch.vtensor<[4,1],si64>
    %int32_667 = torch.constant.int 32
    %562 = torch.aten.mul.Scalar %561, %int32_667 : !torch.vtensor<[4,1],si64>, !torch.int -> !torch.vtensor<[4,1],si64>
    %int1_668 = torch.constant.int 1
    %563 = torch.aten.add.Tensor %562, %549, %int1_668 : !torch.vtensor<[4,1],si64>, !torch.vtensor<[4,1],si64>, !torch.int -> !torch.vtensor<[4,1],si64>
    %int3_669 = torch.constant.int 3
    %int2_670 = torch.constant.int 2
    %int32_671 = torch.constant.int 32
    %int4_672 = torch.constant.int 4
    %int32_673 = torch.constant.int 32
    %564 = torch.prim.ListConstruct %122, %int3_669, %int2_670, %int32_671, %int4_672, %int32_673 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %565 = torch.aten.view %394, %564 : !torch.vtensor<[?,24576],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,4,32],f16>
    torch.bind_symbolic_shape %565, [%38], affine_map<()[s0] -> (s0, 3, 2, 32, 4, 32)> : !torch.vtensor<[?,3,2,32,4,32],f16>
    %int3_674 = torch.constant.int 3
    %566 = torch.aten.mul.int %122, %int3_674 : !torch.int, !torch.int -> !torch.int
    %int2_675 = torch.constant.int 2
    %567 = torch.aten.mul.int %566, %int2_675 : !torch.int, !torch.int -> !torch.int
    %int32_676 = torch.constant.int 32
    %568 = torch.aten.mul.int %567, %int32_676 : !torch.int, !torch.int -> !torch.int
    %int4_677 = torch.constant.int 4
    %int32_678 = torch.constant.int 32
    %569 = torch.prim.ListConstruct %568, %int4_677, %int32_678 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %570 = torch.aten.view %565, %569 : !torch.vtensor<[?,3,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,4,32],f16>
    torch.bind_symbolic_shape %570, [%38], affine_map<()[s0] -> (s0 * 192, 4, 32)> : !torch.vtensor<[?,4,32],f16>
    %571 = torch.prim.ListConstruct %563 : (!torch.vtensor<[4,1],si64>) -> !torch.list<optional<vtensor>>
    %false_679 = torch.constant.bool false
    %572 = torch.aten.index_put %570, %571, %544, %false_679 : !torch.vtensor<[?,4,32],f16>, !torch.list<optional<vtensor>>, !torch.vtensor<[4,1,4,32],f16>, !torch.bool -> !torch.vtensor<[?,4,32],f16>
    torch.bind_symbolic_shape %572, [%38], affine_map<()[s0] -> (s0 * 192, 4, 32)> : !torch.vtensor<[?,4,32],f16>
    %int3_680 = torch.constant.int 3
    %int2_681 = torch.constant.int 2
    %int32_682 = torch.constant.int 32
    %int4_683 = torch.constant.int 4
    %int32_684 = torch.constant.int 32
    %573 = torch.prim.ListConstruct %122, %int3_680, %int2_681, %int32_682, %int4_683, %int32_684 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %574 = torch.aten.view %572, %573 : !torch.vtensor<[?,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,4,32],f16>
    torch.bind_symbolic_shape %574, [%38], affine_map<()[s0] -> (s0, 3, 2, 32, 4, 32)> : !torch.vtensor<[?,3,2,32,4,32],f16>
    %int24576_685 = torch.constant.int 24576
    %575 = torch.prim.ListConstruct %122, %int24576_685 : (!torch.int, !torch.int) -> !torch.list<int>
    %576 = torch.aten.view %574, %575 : !torch.vtensor<[?,3,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,24576],f16>
    torch.bind_symbolic_shape %576, [%38], affine_map<()[s0] -> (s0, 24576)> : !torch.vtensor<[?,24576],f16>
    %int3_686 = torch.constant.int 3
    %int2_687 = torch.constant.int 2
    %int32_688 = torch.constant.int 32
    %int4_689 = torch.constant.int 4
    %int32_690 = torch.constant.int 32
    %577 = torch.prim.ListConstruct %122, %int3_686, %int2_687, %int32_688, %int4_689, %int32_690 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %578 = torch.aten.view %576, %577 : !torch.vtensor<[?,24576],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,4,32],f16>
    torch.bind_symbolic_shape %578, [%38], affine_map<()[s0] -> (s0, 3, 2, 32, 4, 32)> : !torch.vtensor<[?,3,2,32,4,32],f16>
    %int4_691 = torch.constant.int 4
    %int32_692 = torch.constant.int 32
    %579 = torch.prim.ListConstruct %568, %int4_691, %int32_692 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %580 = torch.aten.view %578, %579 : !torch.vtensor<[?,3,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,4,32],f16>
    torch.bind_symbolic_shape %580, [%38], affine_map<()[s0] -> (s0 * 192, 4, 32)> : !torch.vtensor<[?,4,32],f16>
    %int32_693 = torch.constant.int 32
    %581 = torch.aten.floor_divide.Scalar %arg2, %int32_693 : !torch.vtensor<[4],si64>, !torch.int -> !torch.vtensor<[4],si64>
    %int1_694 = torch.constant.int 1
    %582 = torch.aten.unsqueeze %581, %int1_694 : !torch.vtensor<[4],si64>, !torch.int -> !torch.vtensor<[4,1],si64>
    %int1_695 = torch.constant.int 1
    %false_696 = torch.constant.bool false
    %583 = torch.aten.gather %arg3, %int1_695, %582, %false_696 : !torch.vtensor<[4,?],si64>, !torch.int, !torch.vtensor<[4,1],si64>, !torch.bool -> !torch.vtensor<[4,1],si64>
    %int32_697 = torch.constant.int 32
    %584 = torch.aten.remainder.Scalar %arg2, %int32_697 : !torch.vtensor<[4],si64>, !torch.int -> !torch.vtensor<[4],si64>
    %int1_698 = torch.constant.int 1
    %585 = torch.aten.unsqueeze %584, %int1_698 : !torch.vtensor<[4],si64>, !torch.int -> !torch.vtensor<[4,1],si64>
    %none_699 = torch.constant.none
    %586 = torch.aten.clone %28, %none_699 : !torch.vtensor<[],si64>, !torch.none -> !torch.vtensor<[],si64>
    %int0_700 = torch.constant.int 0
    %587 = torch.aten.unsqueeze %586, %int0_700 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int4_701 = torch.constant.int 4
    %int1_702 = torch.constant.int 1
    %588 = torch.prim.ListConstruct %int4_701, %int1_702 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_703 = torch.constant.int 1
    %int1_704 = torch.constant.int 1
    %589 = torch.prim.ListConstruct %int1_703, %int1_704 : (!torch.int, !torch.int) -> !torch.list<int>
    %int4_705 = torch.constant.int 4
    %int0_706 = torch.constant.int 0
    %cpu_707 = torch.constant.device "cpu"
    %false_708 = torch.constant.bool false
    %590 = torch.aten.empty_strided %588, %589, %int4_705, %int0_706, %cpu_707, %false_708 : !torch.list<int>, !torch.list<int>, !torch.int, !torch.int, !torch.Device, !torch.bool -> !torch.vtensor<[4,1],si64>
    %int2_709 = torch.constant.int 2
    %591 = torch.aten.fill.Scalar %590, %int2_709 : !torch.vtensor<[4,1],si64>, !torch.int -> !torch.vtensor<[4,1],si64>
    %int4_710 = torch.constant.int 4
    %int1_711 = torch.constant.int 1
    %592 = torch.prim.ListConstruct %int4_710, %int1_711 : (!torch.int, !torch.int) -> !torch.list<int>
    %593 = torch.aten.repeat %587, %592 : !torch.vtensor<[1],si64>, !torch.list<int> -> !torch.vtensor<[4,1],si64>
    %int3_712 = torch.constant.int 3
    %594 = torch.aten.mul.Scalar %583, %int3_712 : !torch.vtensor<[4,1],si64>, !torch.int -> !torch.vtensor<[4,1],si64>
    %int1_713 = torch.constant.int 1
    %595 = torch.aten.add.Tensor %594, %591, %int1_713 : !torch.vtensor<[4,1],si64>, !torch.vtensor<[4,1],si64>, !torch.int -> !torch.vtensor<[4,1],si64>
    %int2_714 = torch.constant.int 2
    %596 = torch.aten.mul.Scalar %595, %int2_714 : !torch.vtensor<[4,1],si64>, !torch.int -> !torch.vtensor<[4,1],si64>
    %int1_715 = torch.constant.int 1
    %597 = torch.aten.add.Tensor %596, %593, %int1_715 : !torch.vtensor<[4,1],si64>, !torch.vtensor<[4,1],si64>, !torch.int -> !torch.vtensor<[4,1],si64>
    %int32_716 = torch.constant.int 32
    %598 = torch.aten.mul.Scalar %597, %int32_716 : !torch.vtensor<[4,1],si64>, !torch.int -> !torch.vtensor<[4,1],si64>
    %int1_717 = torch.constant.int 1
    %599 = torch.aten.add.Tensor %598, %585, %int1_717 : !torch.vtensor<[4,1],si64>, !torch.vtensor<[4,1],si64>, !torch.int -> !torch.vtensor<[4,1],si64>
    %600 = torch.prim.ListConstruct %599 : (!torch.vtensor<[4,1],si64>) -> !torch.list<optional<vtensor>>
    %false_718 = torch.constant.bool false
    %601 = torch.aten.index_put %580, %600, %532, %false_718 : !torch.vtensor<[?,4,32],f16>, !torch.list<optional<vtensor>>, !torch.vtensor<[4,1,4,32],f16>, !torch.bool -> !torch.vtensor<[?,4,32],f16>
    torch.bind_symbolic_shape %601, [%38], affine_map<()[s0] -> (s0 * 192, 4, 32)> : !torch.vtensor<[?,4,32],f16>
    %int3_719 = torch.constant.int 3
    %int2_720 = torch.constant.int 2
    %int32_721 = torch.constant.int 32
    %int4_722 = torch.constant.int 4
    %int32_723 = torch.constant.int 32
    %602 = torch.prim.ListConstruct %122, %int3_719, %int2_720, %int32_721, %int4_722, %int32_723 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %603 = torch.aten.view %601, %602 : !torch.vtensor<[?,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,4,32],f16>
    torch.bind_symbolic_shape %603, [%38], affine_map<()[s0] -> (s0, 3, 2, 32, 4, 32)> : !torch.vtensor<[?,3,2,32,4,32],f16>
    %int24576_724 = torch.constant.int 24576
    %604 = torch.prim.ListConstruct %122, %int24576_724 : (!torch.int, !torch.int) -> !torch.list<int>
    %605 = torch.aten.view %603, %604 : !torch.vtensor<[?,3,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,24576],f16>
    torch.overwrite.tensor.contents %605 overwrites %arg4 : !torch.vtensor<[?,24576],f16>, !torch.tensor<[?,24576],f16>
    torch.bind_symbolic_shape %605, [%38], affine_map<()[s0] -> (s0, 24576)> : !torch.vtensor<[?,24576],f16>
    %int4_725 = torch.constant.int 4
    %606 = torch.prim.ListConstruct %int4_725, %39 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_726 = torch.constant.int 1
    %607 = torch.prim.ListConstruct %39, %int1_726 : (!torch.int, !torch.int) -> !torch.list<int>
    %int4_727 = torch.constant.int 4
    %int0_728 = torch.constant.int 0
    %cpu_729 = torch.constant.device "cpu"
    %false_730 = torch.constant.bool false
    %608 = torch.aten.empty_strided %606, %607, %int4_727, %int0_728, %cpu_729, %false_730 : !torch.list<int>, !torch.list<int>, !torch.int, !torch.int, !torch.Device, !torch.bool -> !torch.vtensor<[4,?],si64>
    torch.bind_symbolic_shape %608, [%37], affine_map<()[s0] -> (4, s0)> : !torch.vtensor<[4,?],si64>
    %int2_731 = torch.constant.int 2
    %609 = torch.aten.fill.Scalar %608, %int2_731 : !torch.vtensor<[4,?],si64>, !torch.int -> !torch.vtensor<[4,?],si64>
    torch.bind_symbolic_shape %609, [%37], affine_map<()[s0] -> (4, s0)> : !torch.vtensor<[4,?],si64>
    %int3_732 = torch.constant.int 3
    %610 = torch.aten.mul.Scalar %arg3, %int3_732 : !torch.vtensor<[4,?],si64>, !torch.int -> !torch.vtensor<[4,?],si64>
    torch.bind_symbolic_shape %610, [%37], affine_map<()[s0] -> (4, s0)> : !torch.vtensor<[4,?],si64>
    %int1_733 = torch.constant.int 1
    %611 = torch.aten.add.Tensor %610, %609, %int1_733 : !torch.vtensor<[4,?],si64>, !torch.vtensor<[4,?],si64>, !torch.int -> !torch.vtensor<[4,?],si64>
    torch.bind_symbolic_shape %611, [%37], affine_map<()[s0] -> (4, s0)> : !torch.vtensor<[4,?],si64>
    %int4_734 = torch.constant.int 4
    %612 = torch.aten.mul.int %int4_734, %39 : !torch.int, !torch.int -> !torch.int
    %613 = torch.prim.ListConstruct %612 : (!torch.int) -> !torch.list<int>
    %614 = torch.aten.view %611, %613 : !torch.vtensor<[4,?],si64>, !torch.list<int> -> !torch.vtensor<[?],si64>
    torch.bind_symbolic_shape %614, [%37], affine_map<()[s0] -> (s0 * 4)> : !torch.vtensor<[?],si64>
    %int3_735 = torch.constant.int 3
    %int2_736 = torch.constant.int 2
    %int32_737 = torch.constant.int 32
    %int4_738 = torch.constant.int 4
    %int32_739 = torch.constant.int 32
    %615 = torch.prim.ListConstruct %122, %int3_735, %int2_736, %int32_737, %int4_738, %int32_739 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %616 = torch.aten.view %605, %615 : !torch.vtensor<[?,24576],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,4,32],f16>
    torch.bind_symbolic_shape %616, [%38], affine_map<()[s0] -> (s0, 3, 2, 32, 4, 32)> : !torch.vtensor<[?,3,2,32,4,32],f16>
    %int3_740 = torch.constant.int 3
    %617 = torch.aten.mul.int %122, %int3_740 : !torch.int, !torch.int -> !torch.int
    %int2_741 = torch.constant.int 2
    %int32_742 = torch.constant.int 32
    %int4_743 = torch.constant.int 4
    %int32_744 = torch.constant.int 32
    %618 = torch.prim.ListConstruct %617, %int2_741, %int32_742, %int4_743, %int32_744 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %619 = torch.aten.view %616, %618 : !torch.vtensor<[?,3,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,2,32,4,32],f16>
    torch.bind_symbolic_shape %619, [%38], affine_map<()[s0] -> (s0 * 3, 2, 32, 4, 32)> : !torch.vtensor<[?,2,32,4,32],f16>
    %int0_745 = torch.constant.int 0
    %620 = torch.aten.index_select %619, %int0_745, %614 : !torch.vtensor<[?,2,32,4,32],f16>, !torch.int, !torch.vtensor<[?],si64> -> !torch.vtensor<[?,2,32,4,32],f16>
    torch.bind_symbolic_shape %620, [%37], affine_map<()[s0] -> (s0 * 4, 2, 32, 4, 32)> : !torch.vtensor<[?,2,32,4,32],f16>
    %int4_746 = torch.constant.int 4
    %int2_747 = torch.constant.int 2
    %int32_748 = torch.constant.int 32
    %int4_749 = torch.constant.int 4
    %int32_750 = torch.constant.int 32
    %621 = torch.prim.ListConstruct %int4_746, %39, %int2_747, %int32_748, %int4_749, %int32_750 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %622 = torch.aten.view %620, %621 : !torch.vtensor<[?,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[4,?,2,32,4,32],f16>
    torch.bind_symbolic_shape %622, [%37], affine_map<()[s0] -> (4, s0, 2, 32, 4, 32)> : !torch.vtensor<[4,?,2,32,4,32],f16>
    %int0_751 = torch.constant.int 0
    %int0_752 = torch.constant.int 0
    %int9223372036854775807_753 = torch.constant.int 9223372036854775807
    %int1_754 = torch.constant.int 1
    %623 = torch.aten.slice.Tensor %622, %int0_751, %int0_752, %int9223372036854775807_753, %int1_754 : !torch.vtensor<[4,?,2,32,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,?,2,32,4,32],f16>
    torch.bind_symbolic_shape %623, [%37], affine_map<()[s0] -> (4, s0, 2, 32, 4, 32)> : !torch.vtensor<[4,?,2,32,4,32],f16>
    %int1_755 = torch.constant.int 1
    %int0_756 = torch.constant.int 0
    %int9223372036854775807_757 = torch.constant.int 9223372036854775807
    %int1_758 = torch.constant.int 1
    %624 = torch.aten.slice.Tensor %623, %int1_755, %int0_756, %int9223372036854775807_757, %int1_758 : !torch.vtensor<[4,?,2,32,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,?,2,32,4,32],f16>
    torch.bind_symbolic_shape %624, [%37], affine_map<()[s0] -> (4, s0, 2, 32, 4, 32)> : !torch.vtensor<[4,?,2,32,4,32],f16>
    %int2_759 = torch.constant.int 2
    %int0_760 = torch.constant.int 0
    %625 = torch.aten.select.int %624, %int2_759, %int0_760 : !torch.vtensor<[4,?,2,32,4,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[4,?,32,4,32],f16>
    torch.bind_symbolic_shape %625, [%37], affine_map<()[s0] -> (4, s0, 32, 4, 32)> : !torch.vtensor<[4,?,32,4,32],f16>
    %int32_761 = torch.constant.int 32
    %626 = torch.aten.mul.int %39, %int32_761 : !torch.int, !torch.int -> !torch.int
    %int2_762 = torch.constant.int 2
    %int0_763 = torch.constant.int 0
    %int1_764 = torch.constant.int 1
    %627 = torch.aten.slice.Tensor %625, %int2_762, %int0_763, %626, %int1_764 : !torch.vtensor<[4,?,32,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,?,32,4,32],f16>
    torch.bind_symbolic_shape %627, [%37], affine_map<()[s0] -> (4, s0, 32, 4, 32)> : !torch.vtensor<[4,?,32,4,32],f16>
    %int0_765 = torch.constant.int 0
    %628 = torch.aten.clone %627, %int0_765 : !torch.vtensor<[4,?,32,4,32],f16>, !torch.int -> !torch.vtensor<[4,?,32,4,32],f16>
    torch.bind_symbolic_shape %628, [%37], affine_map<()[s0] -> (4, s0, 32, 4, 32)> : !torch.vtensor<[4,?,32,4,32],f16>
    %int1_766 = torch.constant.int 1
    %629 = torch.aten.size.int %624, %int1_766 : !torch.vtensor<[4,?,2,32,4,32],f16>, !torch.int -> !torch.int
    %int32_767 = torch.constant.int 32
    %630 = torch.aten.mul.int %629, %int32_767 : !torch.int, !torch.int -> !torch.int
    %int4_768 = torch.constant.int 4
    %int4_769 = torch.constant.int 4
    %int32_770 = torch.constant.int 32
    %631 = torch.prim.ListConstruct %int4_768, %630, %int4_769, %int32_770 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %632 = torch.aten._unsafe_view %628, %631 : !torch.vtensor<[4,?,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[4,?,4,32],f16>
    torch.bind_symbolic_shape %632, [%37], affine_map<()[s0] -> (4, s0 * 32, 4, 32)> : !torch.vtensor<[4,?,4,32],f16>
    %int0_771 = torch.constant.int 0
    %int0_772 = torch.constant.int 0
    %int9223372036854775807_773 = torch.constant.int 9223372036854775807
    %int1_774 = torch.constant.int 1
    %633 = torch.aten.slice.Tensor %632, %int0_771, %int0_772, %int9223372036854775807_773, %int1_774 : !torch.vtensor<[4,?,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,?,4,32],f16>
    torch.bind_symbolic_shape %633, [%37], affine_map<()[s0] -> (4, s0 * 32, 4, 32)> : !torch.vtensor<[4,?,4,32],f16>
    %int0_775 = torch.constant.int 0
    %int0_776 = torch.constant.int 0
    %int9223372036854775807_777 = torch.constant.int 9223372036854775807
    %int1_778 = torch.constant.int 1
    %634 = torch.aten.slice.Tensor %622, %int0_775, %int0_776, %int9223372036854775807_777, %int1_778 : !torch.vtensor<[4,?,2,32,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,?,2,32,4,32],f16>
    torch.bind_symbolic_shape %634, [%37], affine_map<()[s0] -> (4, s0, 2, 32, 4, 32)> : !torch.vtensor<[4,?,2,32,4,32],f16>
    %int1_779 = torch.constant.int 1
    %int0_780 = torch.constant.int 0
    %int9223372036854775807_781 = torch.constant.int 9223372036854775807
    %int1_782 = torch.constant.int 1
    %635 = torch.aten.slice.Tensor %634, %int1_779, %int0_780, %int9223372036854775807_781, %int1_782 : !torch.vtensor<[4,?,2,32,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,?,2,32,4,32],f16>
    torch.bind_symbolic_shape %635, [%37], affine_map<()[s0] -> (4, s0, 2, 32, 4, 32)> : !torch.vtensor<[4,?,2,32,4,32],f16>
    %int2_783 = torch.constant.int 2
    %int1_784 = torch.constant.int 1
    %636 = torch.aten.select.int %635, %int2_783, %int1_784 : !torch.vtensor<[4,?,2,32,4,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[4,?,32,4,32],f16>
    torch.bind_symbolic_shape %636, [%37], affine_map<()[s0] -> (4, s0, 32, 4, 32)> : !torch.vtensor<[4,?,32,4,32],f16>
    %int2_785 = torch.constant.int 2
    %int0_786 = torch.constant.int 0
    %int1_787 = torch.constant.int 1
    %637 = torch.aten.slice.Tensor %636, %int2_785, %int0_786, %626, %int1_787 : !torch.vtensor<[4,?,32,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,?,32,4,32],f16>
    torch.bind_symbolic_shape %637, [%37], affine_map<()[s0] -> (4, s0, 32, 4, 32)> : !torch.vtensor<[4,?,32,4,32],f16>
    %int0_788 = torch.constant.int 0
    %638 = torch.aten.clone %637, %int0_788 : !torch.vtensor<[4,?,32,4,32],f16>, !torch.int -> !torch.vtensor<[4,?,32,4,32],f16>
    torch.bind_symbolic_shape %638, [%37], affine_map<()[s0] -> (4, s0, 32, 4, 32)> : !torch.vtensor<[4,?,32,4,32],f16>
    %int1_789 = torch.constant.int 1
    %639 = torch.aten.size.int %635, %int1_789 : !torch.vtensor<[4,?,2,32,4,32],f16>, !torch.int -> !torch.int
    %int32_790 = torch.constant.int 32
    %640 = torch.aten.mul.int %639, %int32_790 : !torch.int, !torch.int -> !torch.int
    %int4_791 = torch.constant.int 4
    %int4_792 = torch.constant.int 4
    %int32_793 = torch.constant.int 32
    %641 = torch.prim.ListConstruct %int4_791, %640, %int4_792, %int32_793 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %642 = torch.aten._unsafe_view %638, %641 : !torch.vtensor<[4,?,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[4,?,4,32],f16>
    torch.bind_symbolic_shape %642, [%37], affine_map<()[s0] -> (4, s0 * 32, 4, 32)> : !torch.vtensor<[4,?,4,32],f16>
    %int0_794 = torch.constant.int 0
    %int0_795 = torch.constant.int 0
    %int9223372036854775807_796 = torch.constant.int 9223372036854775807
    %int1_797 = torch.constant.int 1
    %643 = torch.aten.slice.Tensor %642, %int0_794, %int0_795, %int9223372036854775807_796, %int1_797 : !torch.vtensor<[4,?,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[4,?,4,32],f16>
    torch.bind_symbolic_shape %643, [%37], affine_map<()[s0] -> (4, s0 * 32, 4, 32)> : !torch.vtensor<[4,?,4,32],f16>
    %int-2_798 = torch.constant.int -2
    %644 = torch.aten.unsqueeze %633, %int-2_798 : !torch.vtensor<[4,?,4,32],f16>, !torch.int -> !torch.vtensor<[4,?,4,1,32],f16>
    torch.bind_symbolic_shape %644, [%37], affine_map<()[s0] -> (4, s0 * 32, 4, 1, 32)> : !torch.vtensor<[4,?,4,1,32],f16>
    %int1_799 = torch.constant.int 1
    %645 = torch.aten.size.int %632, %int1_799 : !torch.vtensor<[4,?,4,32],f16>, !torch.int -> !torch.int
    %int4_800 = torch.constant.int 4
    %int4_801 = torch.constant.int 4
    %int2_802 = torch.constant.int 2
    %int32_803 = torch.constant.int 32
    %646 = torch.prim.ListConstruct %int4_800, %645, %int4_801, %int2_802, %int32_803 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_804 = torch.constant.bool false
    %647 = torch.aten.expand %644, %646, %false_804 : !torch.vtensor<[4,?,4,1,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[4,?,4,2,32],f16>
    torch.bind_symbolic_shape %647, [%37], affine_map<()[s0] -> (4, s0 * 32, 4, 2, 32)> : !torch.vtensor<[4,?,4,2,32],f16>
    %int0_805 = torch.constant.int 0
    %648 = torch.aten.clone %647, %int0_805 : !torch.vtensor<[4,?,4,2,32],f16>, !torch.int -> !torch.vtensor<[4,?,4,2,32],f16>
    torch.bind_symbolic_shape %648, [%37], affine_map<()[s0] -> (4, s0 * 32, 4, 2, 32)> : !torch.vtensor<[4,?,4,2,32],f16>
    %int4_806 = torch.constant.int 4
    %int8_807 = torch.constant.int 8
    %int32_808 = torch.constant.int 32
    %649 = torch.prim.ListConstruct %int4_806, %645, %int8_807, %int32_808 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %650 = torch.aten._unsafe_view %648, %649 : !torch.vtensor<[4,?,4,2,32],f16>, !torch.list<int> -> !torch.vtensor<[4,?,8,32],f16>
    torch.bind_symbolic_shape %650, [%37], affine_map<()[s0] -> (4, s0 * 32, 8, 32)> : !torch.vtensor<[4,?,8,32],f16>
    %int-2_809 = torch.constant.int -2
    %651 = torch.aten.unsqueeze %643, %int-2_809 : !torch.vtensor<[4,?,4,32],f16>, !torch.int -> !torch.vtensor<[4,?,4,1,32],f16>
    torch.bind_symbolic_shape %651, [%37], affine_map<()[s0] -> (4, s0 * 32, 4, 1, 32)> : !torch.vtensor<[4,?,4,1,32],f16>
    %int1_810 = torch.constant.int 1
    %652 = torch.aten.size.int %642, %int1_810 : !torch.vtensor<[4,?,4,32],f16>, !torch.int -> !torch.int
    %int4_811 = torch.constant.int 4
    %int4_812 = torch.constant.int 4
    %int2_813 = torch.constant.int 2
    %int32_814 = torch.constant.int 32
    %653 = torch.prim.ListConstruct %int4_811, %652, %int4_812, %int2_813, %int32_814 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_815 = torch.constant.bool false
    %654 = torch.aten.expand %651, %653, %false_815 : !torch.vtensor<[4,?,4,1,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[4,?,4,2,32],f16>
    torch.bind_symbolic_shape %654, [%37], affine_map<()[s0] -> (4, s0 * 32, 4, 2, 32)> : !torch.vtensor<[4,?,4,2,32],f16>
    %int0_816 = torch.constant.int 0
    %655 = torch.aten.clone %654, %int0_816 : !torch.vtensor<[4,?,4,2,32],f16>, !torch.int -> !torch.vtensor<[4,?,4,2,32],f16>
    torch.bind_symbolic_shape %655, [%37], affine_map<()[s0] -> (4, s0 * 32, 4, 2, 32)> : !torch.vtensor<[4,?,4,2,32],f16>
    %int4_817 = torch.constant.int 4
    %int8_818 = torch.constant.int 8
    %int32_819 = torch.constant.int 32
    %656 = torch.prim.ListConstruct %int4_817, %652, %int8_818, %int32_819 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %657 = torch.aten._unsafe_view %655, %656 : !torch.vtensor<[4,?,4,2,32],f16>, !torch.list<int> -> !torch.vtensor<[4,?,8,32],f16>
    torch.bind_symbolic_shape %657, [%37], affine_map<()[s0] -> (4, s0 * 32, 8, 32)> : !torch.vtensor<[4,?,8,32],f16>
    %int1_820 = torch.constant.int 1
    %int2_821 = torch.constant.int 2
    %658 = torch.aten.transpose.int %538, %int1_820, %int2_821 : !torch.vtensor<[4,1,8,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[4,8,1,32],f16>
    %int1_822 = torch.constant.int 1
    %int2_823 = torch.constant.int 2
    %659 = torch.aten.transpose.int %650, %int1_822, %int2_823 : !torch.vtensor<[4,?,8,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[4,8,?,32],f16>
    torch.bind_symbolic_shape %659, [%37], affine_map<()[s0] -> (4, 8, s0 * 32, 32)> : !torch.vtensor<[4,8,?,32],f16>
    %int1_824 = torch.constant.int 1
    %int2_825 = torch.constant.int 2
    %660 = torch.aten.transpose.int %657, %int1_824, %int2_825 : !torch.vtensor<[4,?,8,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[4,8,?,32],f16>
    torch.bind_symbolic_shape %660, [%37], affine_map<()[s0] -> (4, 8, s0 * 32, 32)> : !torch.vtensor<[4,8,?,32],f16>
    %float0.000000e00_826 = torch.constant.float 0.000000e+00
    %false_827 = torch.constant.bool false
    %none_828 = torch.constant.none
    %661:2 = torch.operator "torch.aten._scaled_dot_product_flash_attention_for_cpu"(%658, %659, %660, %float0.000000e00_826, %false_827, %49, %none_828) : (!torch.vtensor<[4,8,1,32],f16>, !torch.vtensor<[4,8,?,32],f16>, !torch.vtensor<[4,8,?,32],f16>, !torch.float, !torch.bool, !torch.vtensor<[4,1,1,?],f16>, !torch.none) -> (!torch.vtensor<[4,8,1,32],f16>, !torch.vtensor<[4,8,1],f32>) 
    %int1_829 = torch.constant.int 1
    %int2_830 = torch.constant.int 2
    %662 = torch.aten.transpose.int %661#0, %int1_829, %int2_830 : !torch.vtensor<[4,8,1,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[4,1,8,32],f16>
    %int4_831 = torch.constant.int 4
    %int1_832 = torch.constant.int 1
    %int256_833 = torch.constant.int 256
    %663 = torch.prim.ListConstruct %int4_831, %int1_832, %int256_833 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %664 = torch.aten.view %662, %663 : !torch.vtensor<[4,1,8,32],f16>, !torch.list<int> -> !torch.vtensor<[4,1,256],f16>
    %int5_834 = torch.constant.int 5
    %665 = torch.prims.convert_element_type %29, %int5_834 : !torch.vtensor<[256,256],f32>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int-2_835 = torch.constant.int -2
    %int-1_836 = torch.constant.int -1
    %666 = torch.aten.transpose.int %665, %int-2_835, %int-1_836 : !torch.vtensor<[256,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,256],f16>
    %int4_837 = torch.constant.int 4
    %int256_838 = torch.constant.int 256
    %667 = torch.prim.ListConstruct %int4_837, %int256_838 : (!torch.int, !torch.int) -> !torch.list<int>
    %668 = torch.aten.view %664, %667 : !torch.vtensor<[4,1,256],f16>, !torch.list<int> -> !torch.vtensor<[4,256],f16>
    %669 = torch.aten.mm %668, %666 : !torch.vtensor<[4,256],f16>, !torch.vtensor<[256,256],f16> -> !torch.vtensor<[4,256],f16>
    %int4_839 = torch.constant.int 4
    %int1_840 = torch.constant.int 1
    %int256_841 = torch.constant.int 256
    %670 = torch.prim.ListConstruct %int4_839, %int1_840, %int256_841 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %671 = torch.aten.view %669, %670 : !torch.vtensor<[4,256],f16>, !torch.list<int> -> !torch.vtensor<[4,1,256],f16>
    %int1_842 = torch.constant.int 1
    %672 = torch.aten.add.Tensor %495, %671, %int1_842 : !torch.vtensor<[4,1,256],f16>, !torch.vtensor<[4,1,256],f16>, !torch.int -> !torch.vtensor<[4,1,256],f16>
    %int6_843 = torch.constant.int 6
    %673 = torch.prims.convert_element_type %672, %int6_843 : !torch.vtensor<[4,1,256],f16>, !torch.int -> !torch.vtensor<[4,1,256],f32>
    %int2_844 = torch.constant.int 2
    %674 = torch.aten.pow.Tensor_Scalar %673, %int2_844 : !torch.vtensor<[4,1,256],f32>, !torch.int -> !torch.vtensor<[4,1,256],f32>
    %int-1_845 = torch.constant.int -1
    %675 = torch.prim.ListConstruct %int-1_845 : (!torch.int) -> !torch.list<int>
    %true_846 = torch.constant.bool true
    %none_847 = torch.constant.none
    %676 = torch.aten.mean.dim %674, %675, %true_846, %none_847 : !torch.vtensor<[4,1,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[4,1,1],f32>
    %float1.000000e-02_848 = torch.constant.float 1.000000e-02
    %int1_849 = torch.constant.int 1
    %677 = torch.aten.add.Scalar %676, %float1.000000e-02_848, %int1_849 : !torch.vtensor<[4,1,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[4,1,1],f32>
    %678 = torch.aten.rsqrt %677 : !torch.vtensor<[4,1,1],f32> -> !torch.vtensor<[4,1,1],f32>
    %679 = torch.aten.mul.Tensor %673, %678 : !torch.vtensor<[4,1,256],f32>, !torch.vtensor<[4,1,1],f32> -> !torch.vtensor<[4,1,256],f32>
    %int5_850 = torch.constant.int 5
    %680 = torch.prims.convert_element_type %679, %int5_850 : !torch.vtensor<[4,1,256],f32>, !torch.int -> !torch.vtensor<[4,1,256],f16>
    %681 = torch.aten.mul.Tensor %30, %680 : !torch.vtensor<[256],f32>, !torch.vtensor<[4,1,256],f16> -> !torch.vtensor<[4,1,256],f32>
    %int5_851 = torch.constant.int 5
    %682 = torch.prims.convert_element_type %681, %int5_851 : !torch.vtensor<[4,1,256],f32>, !torch.int -> !torch.vtensor<[4,1,256],f16>
    %int5_852 = torch.constant.int 5
    %683 = torch.prims.convert_element_type %31, %int5_852 : !torch.vtensor<[23,256],f32>, !torch.int -> !torch.vtensor<[23,256],f16>
    %int-2_853 = torch.constant.int -2
    %int-1_854 = torch.constant.int -1
    %684 = torch.aten.transpose.int %683, %int-2_853, %int-1_854 : !torch.vtensor<[23,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,23],f16>
    %int4_855 = torch.constant.int 4
    %int256_856 = torch.constant.int 256
    %685 = torch.prim.ListConstruct %int4_855, %int256_856 : (!torch.int, !torch.int) -> !torch.list<int>
    %686 = torch.aten.view %682, %685 : !torch.vtensor<[4,1,256],f16>, !torch.list<int> -> !torch.vtensor<[4,256],f16>
    %687 = torch.aten.mm %686, %684 : !torch.vtensor<[4,256],f16>, !torch.vtensor<[256,23],f16> -> !torch.vtensor<[4,23],f16>
    %int4_857 = torch.constant.int 4
    %int1_858 = torch.constant.int 1
    %int23_859 = torch.constant.int 23
    %688 = torch.prim.ListConstruct %int4_857, %int1_858, %int23_859 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %689 = torch.aten.view %687, %688 : !torch.vtensor<[4,23],f16>, !torch.list<int> -> !torch.vtensor<[4,1,23],f16>
    %690 = torch.aten.silu %689 : !torch.vtensor<[4,1,23],f16> -> !torch.vtensor<[4,1,23],f16>
    %int5_860 = torch.constant.int 5
    %691 = torch.prims.convert_element_type %32, %int5_860 : !torch.vtensor<[23,256],f32>, !torch.int -> !torch.vtensor<[23,256],f16>
    %int-2_861 = torch.constant.int -2
    %int-1_862 = torch.constant.int -1
    %692 = torch.aten.transpose.int %691, %int-2_861, %int-1_862 : !torch.vtensor<[23,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,23],f16>
    %int4_863 = torch.constant.int 4
    %int256_864 = torch.constant.int 256
    %693 = torch.prim.ListConstruct %int4_863, %int256_864 : (!torch.int, !torch.int) -> !torch.list<int>
    %694 = torch.aten.view %682, %693 : !torch.vtensor<[4,1,256],f16>, !torch.list<int> -> !torch.vtensor<[4,256],f16>
    %695 = torch.aten.mm %694, %692 : !torch.vtensor<[4,256],f16>, !torch.vtensor<[256,23],f16> -> !torch.vtensor<[4,23],f16>
    %int4_865 = torch.constant.int 4
    %int1_866 = torch.constant.int 1
    %int23_867 = torch.constant.int 23
    %696 = torch.prim.ListConstruct %int4_865, %int1_866, %int23_867 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %697 = torch.aten.view %695, %696 : !torch.vtensor<[4,23],f16>, !torch.list<int> -> !torch.vtensor<[4,1,23],f16>
    %698 = torch.aten.mul.Tensor %690, %697 : !torch.vtensor<[4,1,23],f16>, !torch.vtensor<[4,1,23],f16> -> !torch.vtensor<[4,1,23],f16>
    %int5_868 = torch.constant.int 5
    %699 = torch.prims.convert_element_type %33, %int5_868 : !torch.vtensor<[256,23],f32>, !torch.int -> !torch.vtensor<[256,23],f16>
    %int-2_869 = torch.constant.int -2
    %int-1_870 = torch.constant.int -1
    %700 = torch.aten.transpose.int %699, %int-2_869, %int-1_870 : !torch.vtensor<[256,23],f16>, !torch.int, !torch.int -> !torch.vtensor<[23,256],f16>
    %int4_871 = torch.constant.int 4
    %int23_872 = torch.constant.int 23
    %701 = torch.prim.ListConstruct %int4_871, %int23_872 : (!torch.int, !torch.int) -> !torch.list<int>
    %702 = torch.aten.view %698, %701 : !torch.vtensor<[4,1,23],f16>, !torch.list<int> -> !torch.vtensor<[4,23],f16>
    %703 = torch.aten.mm %702, %700 : !torch.vtensor<[4,23],f16>, !torch.vtensor<[23,256],f16> -> !torch.vtensor<[4,256],f16>
    %int4_873 = torch.constant.int 4
    %int1_874 = torch.constant.int 1
    %int256_875 = torch.constant.int 256
    %704 = torch.prim.ListConstruct %int4_873, %int1_874, %int256_875 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %705 = torch.aten.view %703, %704 : !torch.vtensor<[4,256],f16>, !torch.list<int> -> !torch.vtensor<[4,1,256],f16>
    %int1_876 = torch.constant.int 1
    %706 = torch.aten.add.Tensor %672, %705, %int1_876 : !torch.vtensor<[4,1,256],f16>, !torch.vtensor<[4,1,256],f16>, !torch.int -> !torch.vtensor<[4,1,256],f16>
    %int6_877 = torch.constant.int 6
    %707 = torch.prims.convert_element_type %706, %int6_877 : !torch.vtensor<[4,1,256],f16>, !torch.int -> !torch.vtensor<[4,1,256],f32>
    %int2_878 = torch.constant.int 2
    %708 = torch.aten.pow.Tensor_Scalar %707, %int2_878 : !torch.vtensor<[4,1,256],f32>, !torch.int -> !torch.vtensor<[4,1,256],f32>
    %int-1_879 = torch.constant.int -1
    %709 = torch.prim.ListConstruct %int-1_879 : (!torch.int) -> !torch.list<int>
    %true_880 = torch.constant.bool true
    %none_881 = torch.constant.none
    %710 = torch.aten.mean.dim %708, %709, %true_880, %none_881 : !torch.vtensor<[4,1,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[4,1,1],f32>
    %float1.000000e-02_882 = torch.constant.float 1.000000e-02
    %int1_883 = torch.constant.int 1
    %711 = torch.aten.add.Scalar %710, %float1.000000e-02_882, %int1_883 : !torch.vtensor<[4,1,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[4,1,1],f32>
    %712 = torch.aten.rsqrt %711 : !torch.vtensor<[4,1,1],f32> -> !torch.vtensor<[4,1,1],f32>
    %713 = torch.aten.mul.Tensor %707, %712 : !torch.vtensor<[4,1,256],f32>, !torch.vtensor<[4,1,1],f32> -> !torch.vtensor<[4,1,256],f32>
    %int5_884 = torch.constant.int 5
    %714 = torch.prims.convert_element_type %713, %int5_884 : !torch.vtensor<[4,1,256],f32>, !torch.int -> !torch.vtensor<[4,1,256],f16>
    %715 = torch.aten.mul.Tensor %34, %714 : !torch.vtensor<[1,256],f32>, !torch.vtensor<[4,1,256],f16> -> !torch.vtensor<[4,1,256],f32>
    %int5_885 = torch.constant.int 5
    %716 = torch.prims.convert_element_type %715, %int5_885 : !torch.vtensor<[4,1,256],f32>, !torch.int -> !torch.vtensor<[4,1,256],f16>
    %int5_886 = torch.constant.int 5
    %717 = torch.prims.convert_element_type %35, %int5_886 : !torch.vtensor<[256,256],f32>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int-2_887 = torch.constant.int -2
    %int-1_888 = torch.constant.int -1
    %718 = torch.aten.transpose.int %717, %int-2_887, %int-1_888 : !torch.vtensor<[256,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,256],f16>
    %int4_889 = torch.constant.int 4
    %int256_890 = torch.constant.int 256
    %719 = torch.prim.ListConstruct %int4_889, %int256_890 : (!torch.int, !torch.int) -> !torch.list<int>
    %720 = torch.aten.view %716, %719 : !torch.vtensor<[4,1,256],f16>, !torch.list<int> -> !torch.vtensor<[4,256],f16>
    %721 = torch.aten.mm %720, %718 : !torch.vtensor<[4,256],f16>, !torch.vtensor<[256,256],f16> -> !torch.vtensor<[4,256],f16>
    %int4_891 = torch.constant.int 4
    %int1_892 = torch.constant.int 1
    %int256_893 = torch.constant.int 256
    %722 = torch.prim.ListConstruct %int4_891, %int1_892, %int256_893 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %723 = torch.aten.view %721, %722 : !torch.vtensor<[4,256],f16>, !torch.list<int> -> !torch.vtensor<[4,1,256],f16>
    return %723 : !torch.vtensor<[4,1,256],f16>
  }
  util.func private @sharktank_rotary_embedding_4_D_8_32_f32(%arg0: tensor<4x?x8x32xf32>, %arg1: tensor<4x?x32xf32>) -> tensor<4x?x8x32xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %dim = tensor.dim %arg0, %c0 : tensor<4x?x8x32xf32>
    %dim_0 = tensor.dim %arg0, %c1 : tensor<4x?x8x32xf32>
    %dim_1 = tensor.dim %arg0, %c2 : tensor<4x?x8x32xf32>
    %dim_2 = tensor.dim %arg0, %c3 : tensor<4x?x8x32xf32>
    %0 = tensor.empty(%dim, %dim_0, %dim_1, %dim_2) : tensor<?x?x?x?xf32>
    %cast = tensor.cast %0 : tensor<?x?x?x?xf32> to tensor<4x?x8x32xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg1 : tensor<4x?x32xf32>) outs(%cast : tensor<4x?x8x32xf32>) {
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
      %extracted = tensor.extract %arg0[%2, %3, %4, %10] : tensor<4x?x8x32xf32>
      %extracted_3 = tensor.extract %arg0[%2, %3, %4, %11] : tensor<4x?x8x32xf32>
      %12 = arith.cmpi eq, %7, %c0 : index
      %13 = arith.mulf %extracted, %8 : f32
      %14 = arith.mulf %extracted_3, %9 : f32
      %15 = arith.subf %13, %14 : f32
      %16 = arith.mulf %extracted_3, %8 : f32
      %17 = arith.mulf %extracted, %9 : f32
      %18 = arith.addf %16, %17 : f32
      %19 = arith.select %12, %15, %18 : f32
      linalg.yield %19 : f32
    } -> tensor<4x?x8x32xf32>
    util.return %1 : tensor<4x?x8x32xf32>
  }
  util.func private @sharktank_rotary_embedding_4_D_4_32_f32(%arg0: tensor<4x?x4x32xf32>, %arg1: tensor<4x?x32xf32>) -> tensor<4x?x4x32xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %dim = tensor.dim %arg0, %c0 : tensor<4x?x4x32xf32>
    %dim_0 = tensor.dim %arg0, %c1 : tensor<4x?x4x32xf32>
    %dim_1 = tensor.dim %arg0, %c2 : tensor<4x?x4x32xf32>
    %dim_2 = tensor.dim %arg0, %c3 : tensor<4x?x4x32xf32>
    %0 = tensor.empty(%dim, %dim_0, %dim_1, %dim_2) : tensor<?x?x?x?xf32>
    %cast = tensor.cast %0 : tensor<?x?x?x?xf32> to tensor<4x?x4x32xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg1 : tensor<4x?x32xf32>) outs(%cast : tensor<4x?x4x32xf32>) {
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
      %extracted = tensor.extract %arg0[%2, %3, %4, %10] : tensor<4x?x4x32xf32>
      %extracted_3 = tensor.extract %arg0[%2, %3, %4, %11] : tensor<4x?x4x32xf32>
      %12 = arith.cmpi eq, %7, %c0 : index
      %13 = arith.mulf %extracted, %8 : f32
      %14 = arith.mulf %extracted_3, %9 : f32
      %15 = arith.subf %13, %14 : f32
      %16 = arith.mulf %extracted_3, %8 : f32
      %17 = arith.mulf %extracted, %9 : f32
      %18 = arith.addf %16, %17 : f32
      %19 = arith.select %12, %15, %18 : f32
      linalg.yield %19 : f32
    } -> tensor<4x?x4x32xf32>
    util.return %1 : tensor<4x?x4x32xf32>
  }
  util.func private @sharktank_rotary_embedding_4_1_8_32_f32(%arg0: tensor<4x1x8x32xf32>, %arg1: tensor<4x1x32xf32>) -> tensor<4x1x8x32xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %dim = tensor.dim %arg0, %c0 : tensor<4x1x8x32xf32>
    %dim_0 = tensor.dim %arg0, %c1 : tensor<4x1x8x32xf32>
    %dim_1 = tensor.dim %arg0, %c2 : tensor<4x1x8x32xf32>
    %dim_2 = tensor.dim %arg0, %c3 : tensor<4x1x8x32xf32>
    %0 = tensor.empty(%dim, %dim_0, %dim_1, %dim_2) : tensor<?x?x?x?xf32>
    %cast = tensor.cast %0 : tensor<?x?x?x?xf32> to tensor<4x1x8x32xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg1 : tensor<4x1x32xf32>) outs(%cast : tensor<4x1x8x32xf32>) {
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
      %extracted = tensor.extract %arg0[%2, %3, %4, %10] : tensor<4x1x8x32xf32>
      %extracted_3 = tensor.extract %arg0[%2, %3, %4, %11] : tensor<4x1x8x32xf32>
      %12 = arith.cmpi eq, %7, %c0 : index
      %13 = arith.mulf %extracted, %8 : f32
      %14 = arith.mulf %extracted_3, %9 : f32
      %15 = arith.subf %13, %14 : f32
      %16 = arith.mulf %extracted_3, %8 : f32
      %17 = arith.mulf %extracted, %9 : f32
      %18 = arith.addf %16, %17 : f32
      %19 = arith.select %12, %15, %18 : f32
      linalg.yield %19 : f32
    } -> tensor<4x1x8x32xf32>
    util.return %1 : tensor<4x1x8x32xf32>
  }
  util.func private @sharktank_rotary_embedding_4_1_4_32_f32(%arg0: tensor<4x1x4x32xf32>, %arg1: tensor<4x1x32xf32>) -> tensor<4x1x4x32xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %dim = tensor.dim %arg0, %c0 : tensor<4x1x4x32xf32>
    %dim_0 = tensor.dim %arg0, %c1 : tensor<4x1x4x32xf32>
    %dim_1 = tensor.dim %arg0, %c2 : tensor<4x1x4x32xf32>
    %dim_2 = tensor.dim %arg0, %c3 : tensor<4x1x4x32xf32>
    %0 = tensor.empty(%dim, %dim_0, %dim_1, %dim_2) : tensor<?x?x?x?xf32>
    %cast = tensor.cast %0 : tensor<?x?x?x?xf32> to tensor<4x1x4x32xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg1 : tensor<4x1x32xf32>) outs(%cast : tensor<4x1x4x32xf32>) {
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
      %extracted = tensor.extract %arg0[%2, %3, %4, %10] : tensor<4x1x4x32xf32>
      %extracted_3 = tensor.extract %arg0[%2, %3, %4, %11] : tensor<4x1x4x32xf32>
      %12 = arith.cmpi eq, %7, %c0 : index
      %13 = arith.mulf %extracted, %8 : f32
      %14 = arith.mulf %extracted_3, %9 : f32
      %15 = arith.subf %13, %14 : f32
      %16 = arith.mulf %extracted_3, %8 : f32
      %17 = arith.mulf %extracted, %9 : f32
      %18 = arith.addf %16, %17 : f32
      %19 = arith.select %12, %15, %18 : f32
      linalg.yield %19 : f32
    } -> tensor<4x1x4x32xf32>
    util.return %1 : tensor<4x1x4x32xf32>
  }
}
