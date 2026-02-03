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
    %31 = torch.symbolic_int "32*s1" {min_val = 64, max_val = 96} : !torch.int
    %32 = torch.symbolic_int "s1" {min_val = 2, max_val = 3} : !torch.int
    %33 = torch.symbolic_int "s2" {min_val = 0, max_val = 9223372036854775807} : !torch.int
    torch.bind_symbolic_shape %arg0, [%32], affine_map<()[s0] -> (1, s0 * 32)> : !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %arg2, [%32], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %30, [%33], affine_map<()[s0] -> (s0, 24576)> : !torch.vtensor<[?,24576],f16>
    %int1 = torch.constant.int 1
    %34 = torch.aten.size.int %arg2, %int1 : !torch.vtensor<[1,?],si64>, !torch.int -> !torch.int
    %int0 = torch.constant.int 0
    %35 = torch.aten.size.int %30, %int0 : !torch.vtensor<[?,24576],f16>, !torch.int -> !torch.int
    %int5 = torch.constant.int 5
    %36 = torch.prims.convert_element_type %0, %int5 : !torch.vtensor<[256,256],f32>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int-1 = torch.constant.int -1
    %false = torch.constant.bool false
    %false_0 = torch.constant.bool false
    %37 = torch.aten.embedding %36, %arg0, %int-1, %false, %false_0 : !torch.vtensor<[256,256],f16>, !torch.vtensor<[1,?],si64>, !torch.int, !torch.bool, !torch.bool -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %37, [%32], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int1_1 = torch.constant.int 1
    %38 = torch.aten.size.int %arg0, %int1_1 : !torch.vtensor<[1,?],si64>, !torch.int -> !torch.int
    %int6 = torch.constant.int 6
    %39 = torch.prims.convert_element_type %37, %int6 : !torch.vtensor<[1,?,256],f16>, !torch.int -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %39, [%32], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int2 = torch.constant.int 2
    %40 = torch.aten.pow.Tensor_Scalar %39, %int2 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %40, [%32], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int-1_2 = torch.constant.int -1
    %41 = torch.prim.ListConstruct %int-1_2 : (!torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %none = torch.constant.none
    %42 = torch.aten.mean.dim %40, %41, %true, %none : !torch.vtensor<[1,?,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %42, [%32], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %float1.000000e-02 = torch.constant.float 1.000000e-02
    %int1_3 = torch.constant.int 1
    %43 = torch.aten.add.Scalar %42, %float1.000000e-02, %int1_3 : !torch.vtensor<[1,?,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %43, [%32], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %44 = torch.aten.rsqrt %43 : !torch.vtensor<[1,?,1],f32> -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %44, [%32], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %45 = torch.aten.mul.Tensor %39, %44 : !torch.vtensor<[1,?,256],f32>, !torch.vtensor<[1,?,1],f32> -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %45, [%32], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int5_4 = torch.constant.int 5
    %46 = torch.prims.convert_element_type %45, %int5_4 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %46, [%32], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %47 = torch.aten.mul.Tensor %1, %46 : !torch.vtensor<[256],f32>, !torch.vtensor<[1,?,256],f16> -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %47, [%32], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int5_5 = torch.constant.int 5
    %48 = torch.prims.convert_element_type %47, %int5_5 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %48, [%32], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int5_6 = torch.constant.int 5
    %49 = torch.prims.convert_element_type %2, %int5_6 : !torch.vtensor<[256,256],f32>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int-2 = torch.constant.int -2
    %int-1_7 = torch.constant.int -1
    %50 = torch.aten.transpose.int %49, %int-2, %int-1_7 : !torch.vtensor<[256,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,256],f16>
    %int5_8 = torch.constant.int 5
    %51 = torch.prims.convert_element_type %50, %int5_8 : !torch.vtensor<[256,256],f16>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int256 = torch.constant.int 256
    %52 = torch.prim.ListConstruct %38, %int256 : (!torch.int, !torch.int) -> !torch.list<int>
    %53 = torch.aten.view %48, %52 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %53, [%32], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %54 = torch.aten.mm %53, %51 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,256],f16> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %54, [%32], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %int1_9 = torch.constant.int 1
    %int256_10 = torch.constant.int 256
    %55 = torch.prim.ListConstruct %int1_9, %38, %int256_10 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %56 = torch.aten.view %54, %55 : !torch.vtensor<[?,256],f16>, !torch.list<int> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %56, [%32], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int5_11 = torch.constant.int 5
    %57 = torch.prims.convert_element_type %3, %int5_11 : !torch.vtensor<[128,256],f32>, !torch.int -> !torch.vtensor<[128,256],f16>
    %int-2_12 = torch.constant.int -2
    %int-1_13 = torch.constant.int -1
    %58 = torch.aten.transpose.int %57, %int-2_12, %int-1_13 : !torch.vtensor<[128,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,128],f16>
    %int5_14 = torch.constant.int 5
    %59 = torch.prims.convert_element_type %58, %int5_14 : !torch.vtensor<[256,128],f16>, !torch.int -> !torch.vtensor<[256,128],f16>
    %int256_15 = torch.constant.int 256
    %60 = torch.prim.ListConstruct %38, %int256_15 : (!torch.int, !torch.int) -> !torch.list<int>
    %61 = torch.aten.view %48, %60 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %61, [%32], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %62 = torch.aten.mm %61, %59 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,128],f16> -> !torch.vtensor<[?,128],f16>
    torch.bind_symbolic_shape %62, [%32], affine_map<()[s0] -> (s0 * 32, 128)> : !torch.vtensor<[?,128],f16>
    %int1_16 = torch.constant.int 1
    %int128 = torch.constant.int 128
    %63 = torch.prim.ListConstruct %int1_16, %38, %int128 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %64 = torch.aten.view %62, %63 : !torch.vtensor<[?,128],f16>, !torch.list<int> -> !torch.vtensor<[1,?,128],f16>
    torch.bind_symbolic_shape %64, [%32], affine_map<()[s0] -> (1, s0 * 32, 128)> : !torch.vtensor<[1,?,128],f16>
    %int5_17 = torch.constant.int 5
    %65 = torch.prims.convert_element_type %4, %int5_17 : !torch.vtensor<[128,256],f32>, !torch.int -> !torch.vtensor<[128,256],f16>
    %int-2_18 = torch.constant.int -2
    %int-1_19 = torch.constant.int -1
    %66 = torch.aten.transpose.int %65, %int-2_18, %int-1_19 : !torch.vtensor<[128,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,128],f16>
    %int5_20 = torch.constant.int 5
    %67 = torch.prims.convert_element_type %66, %int5_20 : !torch.vtensor<[256,128],f16>, !torch.int -> !torch.vtensor<[256,128],f16>
    %int256_21 = torch.constant.int 256
    %68 = torch.prim.ListConstruct %38, %int256_21 : (!torch.int, !torch.int) -> !torch.list<int>
    %69 = torch.aten.view %48, %68 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %69, [%32], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %70 = torch.aten.mm %69, %67 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,128],f16> -> !torch.vtensor<[?,128],f16>
    torch.bind_symbolic_shape %70, [%32], affine_map<()[s0] -> (s0 * 32, 128)> : !torch.vtensor<[?,128],f16>
    %int1_22 = torch.constant.int 1
    %int128_23 = torch.constant.int 128
    %71 = torch.prim.ListConstruct %int1_22, %38, %int128_23 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %72 = torch.aten.view %70, %71 : !torch.vtensor<[?,128],f16>, !torch.list<int> -> !torch.vtensor<[1,?,128],f16>
    torch.bind_symbolic_shape %72, [%32], affine_map<()[s0] -> (1, s0 * 32, 128)> : !torch.vtensor<[1,?,128],f16>
    %int1_24 = torch.constant.int 1
    %int8 = torch.constant.int 8
    %int32 = torch.constant.int 32
    %73 = torch.prim.ListConstruct %int1_24, %38, %int8, %int32 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %74 = torch.aten.view %56, %73 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[1,?,8,32],f16>
    torch.bind_symbolic_shape %74, [%32], affine_map<()[s0] -> (1, s0 * 32, 8, 32)> : !torch.vtensor<[1,?,8,32],f16>
    %int1_25 = torch.constant.int 1
    %int4 = torch.constant.int 4
    %int32_26 = torch.constant.int 32
    %75 = torch.prim.ListConstruct %int1_25, %38, %int4, %int32_26 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %76 = torch.aten.view %64, %75 : !torch.vtensor<[1,?,128],f16>, !torch.list<int> -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %76, [%32], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int1_27 = torch.constant.int 1
    %int4_28 = torch.constant.int 4
    %int32_29 = torch.constant.int 32
    %77 = torch.prim.ListConstruct %int1_27, %38, %int4_28, %int32_29 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %78 = torch.aten.view %72, %77 : !torch.vtensor<[1,?,128],f16>, !torch.list<int> -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %78, [%32], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int128_30 = torch.constant.int 128
    %none_31 = torch.constant.none
    %none_32 = torch.constant.none
    %cpu = torch.constant.device "cpu"
    %false_33 = torch.constant.bool false
    %79 = torch.aten.arange %int128_30, %none_31, %none_32, %cpu, %false_33 : !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[128],si64>
    %int0_34 = torch.constant.int 0
    %int32_35 = torch.constant.int 32
    %none_36 = torch.constant.none
    %none_37 = torch.constant.none
    %cpu_38 = torch.constant.device "cpu"
    %false_39 = torch.constant.bool false
    %80 = torch.aten.arange.start %int0_34, %int32_35, %none_36, %none_37, %cpu_38, %false_39 : !torch.int, !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[32],si64>
    %int2_40 = torch.constant.int 2
    %81 = torch.aten.floor_divide.Scalar %80, %int2_40 : !torch.vtensor<[32],si64>, !torch.int -> !torch.vtensor<[32],si64>
    %int6_41 = torch.constant.int 6
    %82 = torch.prims.convert_element_type %81, %int6_41 : !torch.vtensor<[32],si64>, !torch.int -> !torch.vtensor<[32],f32>
    %int32_42 = torch.constant.int 32
    %83 = torch.aten.div.Scalar %82, %int32_42 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[32],f32>
    %float2.000000e00 = torch.constant.float 2.000000e+00
    %84 = torch.aten.mul.Scalar %83, %float2.000000e00 : !torch.vtensor<[32],f32>, !torch.float -> !torch.vtensor<[32],f32>
    %float5.000000e05 = torch.constant.float 5.000000e+05
    %85 = torch.aten.pow.Scalar %float5.000000e05, %84 : !torch.float, !torch.vtensor<[32],f32> -> !torch.vtensor<[32],f32>
    %86 = torch.aten.reciprocal %85 : !torch.vtensor<[32],f32> -> !torch.vtensor<[32],f32>
    %float1.000000e00 = torch.constant.float 1.000000e+00
    %87 = torch.aten.mul.Scalar %86, %float1.000000e00 : !torch.vtensor<[32],f32>, !torch.float -> !torch.vtensor<[32],f32>
    %int1_43 = torch.constant.int 1
    %88 = torch.aten.unsqueeze %79, %int1_43 : !torch.vtensor<[128],si64>, !torch.int -> !torch.vtensor<[128,1],si64>
    %int0_44 = torch.constant.int 0
    %89 = torch.aten.unsqueeze %87, %int0_44 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[1,32],f32>
    %90 = torch.aten.mul.Tensor %88, %89 : !torch.vtensor<[128,1],si64>, !torch.vtensor<[1,32],f32> -> !torch.vtensor<[128,32],f32>
    %int6_45 = torch.constant.int 6
    %91 = torch.prims.convert_element_type %90, %int6_45 : !torch.vtensor<[128,32],f32>, !torch.int -> !torch.vtensor<[128,32],f32>
    %int0_46 = torch.constant.int 0
    %int0_47 = torch.constant.int 0
    %int1_48 = torch.constant.int 1
    %92 = torch.aten.slice.Tensor %91, %int0_46, %int0_47, %38, %int1_48 : !torch.vtensor<[128,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %92, [%32], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int1_49 = torch.constant.int 1
    %int0_50 = torch.constant.int 0
    %int9223372036854775807 = torch.constant.int 9223372036854775807
    %int1_51 = torch.constant.int 1
    %93 = torch.aten.slice.Tensor %92, %int1_49, %int0_50, %int9223372036854775807, %int1_51 : !torch.vtensor<[?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %93, [%32], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int1_52 = torch.constant.int 1
    %int0_53 = torch.constant.int 0
    %int9223372036854775807_54 = torch.constant.int 9223372036854775807
    %int1_55 = torch.constant.int 1
    %94 = torch.aten.slice.Tensor %93, %int1_52, %int0_53, %int9223372036854775807_54, %int1_55 : !torch.vtensor<[?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %94, [%32], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int0_56 = torch.constant.int 0
    %95 = torch.aten.unsqueeze %94, %int0_56 : !torch.vtensor<[?,32],f32>, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %95, [%32], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int1_57 = torch.constant.int 1
    %int0_58 = torch.constant.int 0
    %int9223372036854775807_59 = torch.constant.int 9223372036854775807
    %int1_60 = torch.constant.int 1
    %96 = torch.aten.slice.Tensor %95, %int1_57, %int0_58, %int9223372036854775807_59, %int1_60 : !torch.vtensor<[1,?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %96, [%32], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int2_61 = torch.constant.int 2
    %int0_62 = torch.constant.int 0
    %int9223372036854775807_63 = torch.constant.int 9223372036854775807
    %int1_64 = torch.constant.int 1
    %97 = torch.aten.slice.Tensor %96, %int2_61, %int0_62, %int9223372036854775807_63, %int1_64 : !torch.vtensor<[1,?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %97, [%32], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int1_65 = torch.constant.int 1
    %int1_66 = torch.constant.int 1
    %int1_67 = torch.constant.int 1
    %98 = torch.prim.ListConstruct %int1_65, %int1_66, %int1_67 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %99 = torch.aten.repeat %97, %98 : !torch.vtensor<[1,?,32],f32>, !torch.list<int> -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %99, [%32], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int6_68 = torch.constant.int 6
    %100 = torch.prims.convert_element_type %74, %int6_68 : !torch.vtensor<[1,?,8,32],f16>, !torch.int -> !torch.vtensor<[1,?,8,32],f32>
    torch.bind_symbolic_shape %100, [%32], affine_map<()[s0] -> (1, s0 * 32, 8, 32)> : !torch.vtensor<[1,?,8,32],f32>
    %101 = torch_c.to_builtin_tensor %100 : !torch.vtensor<[1,?,8,32],f32> -> tensor<1x?x8x32xf32>
    %102 = torch_c.to_builtin_tensor %99 : !torch.vtensor<[1,?,32],f32> -> tensor<1x?x32xf32>
    %103 = util.call @sharktank_rotary_embedding_1_D_8_32_f32(%101, %102) : (tensor<1x?x8x32xf32>, tensor<1x?x32xf32>) -> tensor<1x?x8x32xf32>
    %104 = torch_c.from_builtin_tensor %103 : tensor<1x?x8x32xf32> -> !torch.vtensor<[1,?,8,32],f32>
    torch.bind_symbolic_shape %104, [%32], affine_map<()[s0] -> (1, s0 * 32, 8, 32)> : !torch.vtensor<[1,?,8,32],f32>
    %int5_69 = torch.constant.int 5
    %105 = torch.prims.convert_element_type %104, %int5_69 : !torch.vtensor<[1,?,8,32],f32>, !torch.int -> !torch.vtensor<[1,?,8,32],f16>
    torch.bind_symbolic_shape %105, [%32], affine_map<()[s0] -> (1, s0 * 32, 8, 32)> : !torch.vtensor<[1,?,8,32],f16>
    %int128_70 = torch.constant.int 128
    %none_71 = torch.constant.none
    %none_72 = torch.constant.none
    %cpu_73 = torch.constant.device "cpu"
    %false_74 = torch.constant.bool false
    %106 = torch.aten.arange %int128_70, %none_71, %none_72, %cpu_73, %false_74 : !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[128],si64>
    %int0_75 = torch.constant.int 0
    %int32_76 = torch.constant.int 32
    %none_77 = torch.constant.none
    %none_78 = torch.constant.none
    %cpu_79 = torch.constant.device "cpu"
    %false_80 = torch.constant.bool false
    %107 = torch.aten.arange.start %int0_75, %int32_76, %none_77, %none_78, %cpu_79, %false_80 : !torch.int, !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[32],si64>
    %int2_81 = torch.constant.int 2
    %108 = torch.aten.floor_divide.Scalar %107, %int2_81 : !torch.vtensor<[32],si64>, !torch.int -> !torch.vtensor<[32],si64>
    %int6_82 = torch.constant.int 6
    %109 = torch.prims.convert_element_type %108, %int6_82 : !torch.vtensor<[32],si64>, !torch.int -> !torch.vtensor<[32],f32>
    %int32_83 = torch.constant.int 32
    %110 = torch.aten.div.Scalar %109, %int32_83 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[32],f32>
    %float2.000000e00_84 = torch.constant.float 2.000000e+00
    %111 = torch.aten.mul.Scalar %110, %float2.000000e00_84 : !torch.vtensor<[32],f32>, !torch.float -> !torch.vtensor<[32],f32>
    %float5.000000e05_85 = torch.constant.float 5.000000e+05
    %112 = torch.aten.pow.Scalar %float5.000000e05_85, %111 : !torch.float, !torch.vtensor<[32],f32> -> !torch.vtensor<[32],f32>
    %113 = torch.aten.reciprocal %112 : !torch.vtensor<[32],f32> -> !torch.vtensor<[32],f32>
    %float1.000000e00_86 = torch.constant.float 1.000000e+00
    %114 = torch.aten.mul.Scalar %113, %float1.000000e00_86 : !torch.vtensor<[32],f32>, !torch.float -> !torch.vtensor<[32],f32>
    %int1_87 = torch.constant.int 1
    %115 = torch.aten.unsqueeze %106, %int1_87 : !torch.vtensor<[128],si64>, !torch.int -> !torch.vtensor<[128,1],si64>
    %int0_88 = torch.constant.int 0
    %116 = torch.aten.unsqueeze %114, %int0_88 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[1,32],f32>
    %117 = torch.aten.mul.Tensor %115, %116 : !torch.vtensor<[128,1],si64>, !torch.vtensor<[1,32],f32> -> !torch.vtensor<[128,32],f32>
    %int6_89 = torch.constant.int 6
    %118 = torch.prims.convert_element_type %117, %int6_89 : !torch.vtensor<[128,32],f32>, !torch.int -> !torch.vtensor<[128,32],f32>
    %int0_90 = torch.constant.int 0
    %int0_91 = torch.constant.int 0
    %int1_92 = torch.constant.int 1
    %119 = torch.aten.slice.Tensor %118, %int0_90, %int0_91, %38, %int1_92 : !torch.vtensor<[128,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %119, [%32], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int1_93 = torch.constant.int 1
    %int0_94 = torch.constant.int 0
    %int9223372036854775807_95 = torch.constant.int 9223372036854775807
    %int1_96 = torch.constant.int 1
    %120 = torch.aten.slice.Tensor %119, %int1_93, %int0_94, %int9223372036854775807_95, %int1_96 : !torch.vtensor<[?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %120, [%32], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int1_97 = torch.constant.int 1
    %int0_98 = torch.constant.int 0
    %int9223372036854775807_99 = torch.constant.int 9223372036854775807
    %int1_100 = torch.constant.int 1
    %121 = torch.aten.slice.Tensor %120, %int1_97, %int0_98, %int9223372036854775807_99, %int1_100 : !torch.vtensor<[?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %121, [%32], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int0_101 = torch.constant.int 0
    %122 = torch.aten.unsqueeze %121, %int0_101 : !torch.vtensor<[?,32],f32>, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %122, [%32], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int1_102 = torch.constant.int 1
    %int0_103 = torch.constant.int 0
    %int9223372036854775807_104 = torch.constant.int 9223372036854775807
    %int1_105 = torch.constant.int 1
    %123 = torch.aten.slice.Tensor %122, %int1_102, %int0_103, %int9223372036854775807_104, %int1_105 : !torch.vtensor<[1,?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %123, [%32], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int2_106 = torch.constant.int 2
    %int0_107 = torch.constant.int 0
    %int9223372036854775807_108 = torch.constant.int 9223372036854775807
    %int1_109 = torch.constant.int 1
    %124 = torch.aten.slice.Tensor %123, %int2_106, %int0_107, %int9223372036854775807_108, %int1_109 : !torch.vtensor<[1,?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %124, [%32], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int1_110 = torch.constant.int 1
    %int1_111 = torch.constant.int 1
    %int1_112 = torch.constant.int 1
    %125 = torch.prim.ListConstruct %int1_110, %int1_111, %int1_112 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %126 = torch.aten.repeat %124, %125 : !torch.vtensor<[1,?,32],f32>, !torch.list<int> -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %126, [%32], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int6_113 = torch.constant.int 6
    %127 = torch.prims.convert_element_type %76, %int6_113 : !torch.vtensor<[1,?,4,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,32],f32>
    torch.bind_symbolic_shape %127, [%32], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f32>
    %128 = torch_c.to_builtin_tensor %127 : !torch.vtensor<[1,?,4,32],f32> -> tensor<1x?x4x32xf32>
    %129 = torch_c.to_builtin_tensor %126 : !torch.vtensor<[1,?,32],f32> -> tensor<1x?x32xf32>
    %130 = util.call @sharktank_rotary_embedding_1_D_4_32_f32(%128, %129) : (tensor<1x?x4x32xf32>, tensor<1x?x32xf32>) -> tensor<1x?x4x32xf32>
    %131 = torch_c.from_builtin_tensor %130 : tensor<1x?x4x32xf32> -> !torch.vtensor<[1,?,4,32],f32>
    torch.bind_symbolic_shape %131, [%32], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f32>
    %int5_114 = torch.constant.int 5
    %132 = torch.prims.convert_element_type %131, %int5_114 : !torch.vtensor<[1,?,4,32],f32>, !torch.int -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %132, [%32], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int3 = torch.constant.int 3
    %int2_115 = torch.constant.int 2
    %int32_116 = torch.constant.int 32
    %int4_117 = torch.constant.int 4
    %int32_118 = torch.constant.int 32
    %133 = torch.prim.ListConstruct %35, %int3, %int2_115, %int32_116, %int4_117, %int32_118 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %134 = torch.aten.view %30, %133 : !torch.vtensor<[?,24576],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,4,32],f16>
    torch.bind_symbolic_shape %134, [%33], affine_map<()[s0] -> (s0, 3, 2, 32, 4, 32)> : !torch.vtensor<[?,3,2,32,4,32],f16>
    %int3_119 = torch.constant.int 3
    %135 = torch.aten.mul.int %35, %int3_119 : !torch.int, !torch.int -> !torch.int
    %int2_120 = torch.constant.int 2
    %136 = torch.aten.mul.int %135, %int2_120 : !torch.int, !torch.int -> !torch.int
    %int32_121 = torch.constant.int 32
    %int4_122 = torch.constant.int 4
    %int32_123 = torch.constant.int 32
    %137 = torch.prim.ListConstruct %136, %int32_121, %int4_122, %int32_123 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %138 = torch.aten.view %134, %137 : !torch.vtensor<[?,3,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,32,4,32],f16>
    torch.bind_symbolic_shape %138, [%33], affine_map<()[s0] -> (s0 * 6, 32, 4, 32)> : !torch.vtensor<[?,32,4,32],f16>
    %int6_124 = torch.constant.int 6
    %139 = torch.aten.mul.Scalar %arg2, %int6_124 : !torch.vtensor<[1,?],si64>, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %139, [%32], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %int0_125 = torch.constant.int 0
    %int1_126 = torch.constant.int 1
    %140 = torch.aten.add.Scalar %139, %int0_125, %int1_126 : !torch.vtensor<[1,?],si64>, !torch.int, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %140, [%32], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %int1_127 = torch.constant.int 1
    %int32_128 = torch.constant.int 32
    %int4_129 = torch.constant.int 4
    %int32_130 = torch.constant.int 32
    %141 = torch.prim.ListConstruct %int1_127, %34, %int32_128, %int4_129, %int32_130 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %142 = torch.aten.view %132, %141 : !torch.vtensor<[1,?,4,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,32,4,32],f16>
    torch.bind_symbolic_shape %142, [%32], affine_map<()[s0] -> (1, s0, 32, 4, 32)> : !torch.vtensor<[1,?,32,4,32],f16>
    %int32_131 = torch.constant.int 32
    %int4_132 = torch.constant.int 4
    %int32_133 = torch.constant.int 32
    %143 = torch.prim.ListConstruct %34, %int32_131, %int4_132, %int32_133 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %144 = torch.aten.view %142, %143 : !torch.vtensor<[1,?,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,32,4,32],f16>
    torch.bind_symbolic_shape %144, [%32], affine_map<()[s0] -> (s0, 32, 4, 32)> : !torch.vtensor<[?,32,4,32],f16>
    %145 = torch.prim.ListConstruct %34 : (!torch.int) -> !torch.list<int>
    %146 = torch.aten.view %140, %145 : !torch.vtensor<[1,?],si64>, !torch.list<int> -> !torch.vtensor<[?],si64>
    torch.bind_symbolic_shape %146, [%32], affine_map<()[s0] -> (s0)> : !torch.vtensor<[?],si64>
    %int5_134 = torch.constant.int 5
    %147 = torch.prims.convert_element_type %144, %int5_134 : !torch.vtensor<[?,32,4,32],f16>, !torch.int -> !torch.vtensor<[?,32,4,32],f16>
    torch.bind_symbolic_shape %147, [%32], affine_map<()[s0] -> (s0, 32, 4, 32)> : !torch.vtensor<[?,32,4,32],f16>
    %148 = torch.prim.ListConstruct %146 : (!torch.vtensor<[?],si64>) -> !torch.list<optional<vtensor>>
    %false_135 = torch.constant.bool false
    %149 = torch.aten.index_put %138, %148, %147, %false_135 : !torch.vtensor<[?,32,4,32],f16>, !torch.list<optional<vtensor>>, !torch.vtensor<[?,32,4,32],f16>, !torch.bool -> !torch.vtensor<[?,32,4,32],f16>
    torch.bind_symbolic_shape %149, [%33], affine_map<()[s0] -> (s0 * 6, 32, 4, 32)> : !torch.vtensor<[?,32,4,32],f16>
    %int3_136 = torch.constant.int 3
    %int2_137 = torch.constant.int 2
    %int32_138 = torch.constant.int 32
    %int4_139 = torch.constant.int 4
    %int32_140 = torch.constant.int 32
    %150 = torch.prim.ListConstruct %35, %int3_136, %int2_137, %int32_138, %int4_139, %int32_140 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %151 = torch.aten.view %149, %150 : !torch.vtensor<[?,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,4,32],f16>
    torch.bind_symbolic_shape %151, [%33], affine_map<()[s0] -> (s0, 3, 2, 32, 4, 32)> : !torch.vtensor<[?,3,2,32,4,32],f16>
    %int24576 = torch.constant.int 24576
    %152 = torch.prim.ListConstruct %35, %int24576 : (!torch.int, !torch.int) -> !torch.list<int>
    %153 = torch.aten.view %151, %152 : !torch.vtensor<[?,3,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,24576],f16>
    torch.bind_symbolic_shape %153, [%33], affine_map<()[s0] -> (s0, 24576)> : !torch.vtensor<[?,24576],f16>
    %int3_141 = torch.constant.int 3
    %int2_142 = torch.constant.int 2
    %int32_143 = torch.constant.int 32
    %int4_144 = torch.constant.int 4
    %int32_145 = torch.constant.int 32
    %154 = torch.prim.ListConstruct %35, %int3_141, %int2_142, %int32_143, %int4_144, %int32_145 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %155 = torch.aten.view %153, %154 : !torch.vtensor<[?,24576],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,4,32],f16>
    torch.bind_symbolic_shape %155, [%33], affine_map<()[s0] -> (s0, 3, 2, 32, 4, 32)> : !torch.vtensor<[?,3,2,32,4,32],f16>
    %int32_146 = torch.constant.int 32
    %int4_147 = torch.constant.int 4
    %int32_148 = torch.constant.int 32
    %156 = torch.prim.ListConstruct %136, %int32_146, %int4_147, %int32_148 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %157 = torch.aten.view %155, %156 : !torch.vtensor<[?,3,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,32,4,32],f16>
    torch.bind_symbolic_shape %157, [%33], affine_map<()[s0] -> (s0 * 6, 32, 4, 32)> : !torch.vtensor<[?,32,4,32],f16>
    %int1_149 = torch.constant.int 1
    %int32_150 = torch.constant.int 32
    %int4_151 = torch.constant.int 4
    %int32_152 = torch.constant.int 32
    %158 = torch.prim.ListConstruct %int1_149, %34, %int32_150, %int4_151, %int32_152 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %159 = torch.aten.view %78, %158 : !torch.vtensor<[1,?,4,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,32,4,32],f16>
    torch.bind_symbolic_shape %159, [%32], affine_map<()[s0] -> (1, s0, 32, 4, 32)> : !torch.vtensor<[1,?,32,4,32],f16>
    %int32_153 = torch.constant.int 32
    %int4_154 = torch.constant.int 4
    %int32_155 = torch.constant.int 32
    %160 = torch.prim.ListConstruct %34, %int32_153, %int4_154, %int32_155 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %161 = torch.aten.view %159, %160 : !torch.vtensor<[1,?,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,32,4,32],f16>
    torch.bind_symbolic_shape %161, [%32], affine_map<()[s0] -> (s0, 32, 4, 32)> : !torch.vtensor<[?,32,4,32],f16>
    %int1_156 = torch.constant.int 1
    %int1_157 = torch.constant.int 1
    %162 = torch.aten.add.Scalar %140, %int1_156, %int1_157 : !torch.vtensor<[1,?],si64>, !torch.int, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %162, [%32], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %163 = torch.prim.ListConstruct %34 : (!torch.int) -> !torch.list<int>
    %164 = torch.aten.view %162, %163 : !torch.vtensor<[1,?],si64>, !torch.list<int> -> !torch.vtensor<[?],si64>
    torch.bind_symbolic_shape %164, [%32], affine_map<()[s0] -> (s0)> : !torch.vtensor<[?],si64>
    %int5_158 = torch.constant.int 5
    %165 = torch.prims.convert_element_type %161, %int5_158 : !torch.vtensor<[?,32,4,32],f16>, !torch.int -> !torch.vtensor<[?,32,4,32],f16>
    torch.bind_symbolic_shape %165, [%32], affine_map<()[s0] -> (s0, 32, 4, 32)> : !torch.vtensor<[?,32,4,32],f16>
    %166 = torch.prim.ListConstruct %164 : (!torch.vtensor<[?],si64>) -> !torch.list<optional<vtensor>>
    %false_159 = torch.constant.bool false
    %167 = torch.aten.index_put %157, %166, %165, %false_159 : !torch.vtensor<[?,32,4,32],f16>, !torch.list<optional<vtensor>>, !torch.vtensor<[?,32,4,32],f16>, !torch.bool -> !torch.vtensor<[?,32,4,32],f16>
    torch.bind_symbolic_shape %167, [%33], affine_map<()[s0] -> (s0 * 6, 32, 4, 32)> : !torch.vtensor<[?,32,4,32],f16>
    %int3_160 = torch.constant.int 3
    %int2_161 = torch.constant.int 2
    %int32_162 = torch.constant.int 32
    %int4_163 = torch.constant.int 4
    %int32_164 = torch.constant.int 32
    %168 = torch.prim.ListConstruct %35, %int3_160, %int2_161, %int32_162, %int4_163, %int32_164 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %169 = torch.aten.view %167, %168 : !torch.vtensor<[?,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,4,32],f16>
    torch.bind_symbolic_shape %169, [%33], affine_map<()[s0] -> (s0, 3, 2, 32, 4, 32)> : !torch.vtensor<[?,3,2,32,4,32],f16>
    %int24576_165 = torch.constant.int 24576
    %170 = torch.prim.ListConstruct %35, %int24576_165 : (!torch.int, !torch.int) -> !torch.list<int>
    %171 = torch.aten.view %169, %170 : !torch.vtensor<[?,3,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,24576],f16>
    torch.bind_symbolic_shape %171, [%33], affine_map<()[s0] -> (s0, 24576)> : !torch.vtensor<[?,24576],f16>
    %int-2_166 = torch.constant.int -2
    %172 = torch.aten.unsqueeze %132, %int-2_166 : !torch.vtensor<[1,?,4,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,1,32],f16>
    torch.bind_symbolic_shape %172, [%32], affine_map<()[s0] -> (1, s0 * 32, 4, 1, 32)> : !torch.vtensor<[1,?,4,1,32],f16>
    %int1_167 = torch.constant.int 1
    %int4_168 = torch.constant.int 4
    %int2_169 = torch.constant.int 2
    %int32_170 = torch.constant.int 32
    %173 = torch.prim.ListConstruct %int1_167, %38, %int4_168, %int2_169, %int32_170 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_171 = torch.constant.bool false
    %174 = torch.aten.expand %172, %173, %false_171 : !torch.vtensor<[1,?,4,1,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,?,4,2,32],f16>
    torch.bind_symbolic_shape %174, [%32], affine_map<()[s0] -> (1, s0 * 32, 4, 2, 32)> : !torch.vtensor<[1,?,4,2,32],f16>
    %int0_172 = torch.constant.int 0
    %175 = torch.aten.clone %174, %int0_172 : !torch.vtensor<[1,?,4,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,2,32],f16>
    torch.bind_symbolic_shape %175, [%32], affine_map<()[s0] -> (1, s0 * 32, 4, 2, 32)> : !torch.vtensor<[1,?,4,2,32],f16>
    %int1_173 = torch.constant.int 1
    %int8_174 = torch.constant.int 8
    %int32_175 = torch.constant.int 32
    %176 = torch.prim.ListConstruct %int1_173, %38, %int8_174, %int32_175 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %177 = torch.aten._unsafe_view %175, %176 : !torch.vtensor<[1,?,4,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,8,32],f16>
    torch.bind_symbolic_shape %177, [%32], affine_map<()[s0] -> (1, s0 * 32, 8, 32)> : !torch.vtensor<[1,?,8,32],f16>
    %int-2_176 = torch.constant.int -2
    %178 = torch.aten.unsqueeze %78, %int-2_176 : !torch.vtensor<[1,?,4,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,1,32],f16>
    torch.bind_symbolic_shape %178, [%32], affine_map<()[s0] -> (1, s0 * 32, 4, 1, 32)> : !torch.vtensor<[1,?,4,1,32],f16>
    %int1_177 = torch.constant.int 1
    %int4_178 = torch.constant.int 4
    %int2_179 = torch.constant.int 2
    %int32_180 = torch.constant.int 32
    %179 = torch.prim.ListConstruct %int1_177, %38, %int4_178, %int2_179, %int32_180 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_181 = torch.constant.bool false
    %180 = torch.aten.expand %178, %179, %false_181 : !torch.vtensor<[1,?,4,1,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,?,4,2,32],f16>
    torch.bind_symbolic_shape %180, [%32], affine_map<()[s0] -> (1, s0 * 32, 4, 2, 32)> : !torch.vtensor<[1,?,4,2,32],f16>
    %int0_182 = torch.constant.int 0
    %181 = torch.aten.clone %180, %int0_182 : !torch.vtensor<[1,?,4,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,2,32],f16>
    torch.bind_symbolic_shape %181, [%32], affine_map<()[s0] -> (1, s0 * 32, 4, 2, 32)> : !torch.vtensor<[1,?,4,2,32],f16>
    %int1_183 = torch.constant.int 1
    %int8_184 = torch.constant.int 8
    %int32_185 = torch.constant.int 32
    %182 = torch.prim.ListConstruct %int1_183, %38, %int8_184, %int32_185 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %183 = torch.aten._unsafe_view %181, %182 : !torch.vtensor<[1,?,4,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,8,32],f16>
    torch.bind_symbolic_shape %183, [%32], affine_map<()[s0] -> (1, s0 * 32, 8, 32)> : !torch.vtensor<[1,?,8,32],f16>
    %int1_186 = torch.constant.int 1
    %int2_187 = torch.constant.int 2
    %184 = torch.aten.transpose.int %105, %int1_186, %int2_187 : !torch.vtensor<[1,?,8,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,8,?,32],f16>
    torch.bind_symbolic_shape %184, [%32], affine_map<()[s0] -> (1, 8, s0 * 32, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %int1_188 = torch.constant.int 1
    %int2_189 = torch.constant.int 2
    %185 = torch.aten.transpose.int %177, %int1_188, %int2_189 : !torch.vtensor<[1,?,8,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,8,?,32],f16>
    torch.bind_symbolic_shape %185, [%32], affine_map<()[s0] -> (1, 8, s0 * 32, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %int1_190 = torch.constant.int 1
    %int2_191 = torch.constant.int 2
    %186 = torch.aten.transpose.int %183, %int1_190, %int2_191 : !torch.vtensor<[1,?,8,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,8,?,32],f16>
    torch.bind_symbolic_shape %186, [%32], affine_map<()[s0] -> (1, 8, s0 * 32, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %int5_192 = torch.constant.int 5
    %187 = torch.prims.convert_element_type %184, %int5_192 : !torch.vtensor<[1,8,?,32],f16>, !torch.int -> !torch.vtensor<[1,8,?,32],f16>
    torch.bind_symbolic_shape %187, [%32], affine_map<()[s0] -> (1, 8, s0 * 32, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %int5_193 = torch.constant.int 5
    %188 = torch.prims.convert_element_type %185, %int5_193 : !torch.vtensor<[1,8,?,32],f16>, !torch.int -> !torch.vtensor<[1,8,?,32],f16>
    torch.bind_symbolic_shape %188, [%32], affine_map<()[s0] -> (1, 8, s0 * 32, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %int5_194 = torch.constant.int 5
    %189 = torch.prims.convert_element_type %186, %int5_194 : !torch.vtensor<[1,8,?,32],f16>, !torch.int -> !torch.vtensor<[1,8,?,32],f16>
    torch.bind_symbolic_shape %189, [%32], affine_map<()[s0] -> (1, 8, s0 * 32, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %float0.000000e00 = torch.constant.float 0.000000e+00
    %true_195 = torch.constant.bool true
    %none_196 = torch.constant.none
    %none_197 = torch.constant.none
    %190:2 = torch.operator "torch.aten._scaled_dot_product_flash_attention_for_cpu"(%187, %188, %189, %float0.000000e00, %true_195, %none_196, %none_197) : (!torch.vtensor<[1,8,?,32],f16>, !torch.vtensor<[1,8,?,32],f16>, !torch.vtensor<[1,8,?,32],f16>, !torch.float, !torch.bool, !torch.none, !torch.none) -> (!torch.vtensor<[1,8,?,32],f16>, !torch.vtensor<[1,8,?],f32>) 
    torch.bind_symbolic_shape %190#0, [%32], affine_map<()[s0] -> (1, 8, s0 * 32, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %int1_198 = torch.constant.int 1
    %int2_199 = torch.constant.int 2
    %191 = torch.aten.transpose.int %190#0, %int1_198, %int2_199 : !torch.vtensor<[1,8,?,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,?,8,32],f16>
    torch.bind_symbolic_shape %191, [%32], affine_map<()[s0] -> (1, s0 * 32, 8, 32)> : !torch.vtensor<[1,?,8,32],f16>
    %int1_200 = torch.constant.int 1
    %int256_201 = torch.constant.int 256
    %192 = torch.prim.ListConstruct %int1_200, %38, %int256_201 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %193 = torch.aten.view %191, %192 : !torch.vtensor<[1,?,8,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %193, [%32], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int5_202 = torch.constant.int 5
    %194 = torch.prims.convert_element_type %5, %int5_202 : !torch.vtensor<[256,256],f32>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int-2_203 = torch.constant.int -2
    %int-1_204 = torch.constant.int -1
    %195 = torch.aten.transpose.int %194, %int-2_203, %int-1_204 : !torch.vtensor<[256,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,256],f16>
    %int5_205 = torch.constant.int 5
    %196 = torch.prims.convert_element_type %195, %int5_205 : !torch.vtensor<[256,256],f16>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int256_206 = torch.constant.int 256
    %197 = torch.prim.ListConstruct %38, %int256_206 : (!torch.int, !torch.int) -> !torch.list<int>
    %198 = torch.aten.view %193, %197 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %198, [%32], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %199 = torch.aten.mm %198, %196 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,256],f16> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %199, [%32], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %int1_207 = torch.constant.int 1
    %int256_208 = torch.constant.int 256
    %200 = torch.prim.ListConstruct %int1_207, %38, %int256_208 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %201 = torch.aten.view %199, %200 : !torch.vtensor<[?,256],f16>, !torch.list<int> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %201, [%32], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int1_209 = torch.constant.int 1
    %202 = torch.aten.add.Tensor %37, %201, %int1_209 : !torch.vtensor<[1,?,256],f16>, !torch.vtensor<[1,?,256],f16>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %202, [%32], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int6_210 = torch.constant.int 6
    %203 = torch.prims.convert_element_type %202, %int6_210 : !torch.vtensor<[1,?,256],f16>, !torch.int -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %203, [%32], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int2_211 = torch.constant.int 2
    %204 = torch.aten.pow.Tensor_Scalar %203, %int2_211 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %204, [%32], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int-1_212 = torch.constant.int -1
    %205 = torch.prim.ListConstruct %int-1_212 : (!torch.int) -> !torch.list<int>
    %true_213 = torch.constant.bool true
    %none_214 = torch.constant.none
    %206 = torch.aten.mean.dim %204, %205, %true_213, %none_214 : !torch.vtensor<[1,?,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %206, [%32], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %float1.000000e-02_215 = torch.constant.float 1.000000e-02
    %int1_216 = torch.constant.int 1
    %207 = torch.aten.add.Scalar %206, %float1.000000e-02_215, %int1_216 : !torch.vtensor<[1,?,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %207, [%32], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %208 = torch.aten.rsqrt %207 : !torch.vtensor<[1,?,1],f32> -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %208, [%32], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %209 = torch.aten.mul.Tensor %203, %208 : !torch.vtensor<[1,?,256],f32>, !torch.vtensor<[1,?,1],f32> -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %209, [%32], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int5_217 = torch.constant.int 5
    %210 = torch.prims.convert_element_type %209, %int5_217 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %210, [%32], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %211 = torch.aten.mul.Tensor %6, %210 : !torch.vtensor<[256],f32>, !torch.vtensor<[1,?,256],f16> -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %211, [%32], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int5_218 = torch.constant.int 5
    %212 = torch.prims.convert_element_type %211, %int5_218 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %212, [%32], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int5_219 = torch.constant.int 5
    %213 = torch.prims.convert_element_type %7, %int5_219 : !torch.vtensor<[23,256],f32>, !torch.int -> !torch.vtensor<[23,256],f16>
    %int-2_220 = torch.constant.int -2
    %int-1_221 = torch.constant.int -1
    %214 = torch.aten.transpose.int %213, %int-2_220, %int-1_221 : !torch.vtensor<[23,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,23],f16>
    %int5_222 = torch.constant.int 5
    %215 = torch.prims.convert_element_type %214, %int5_222 : !torch.vtensor<[256,23],f16>, !torch.int -> !torch.vtensor<[256,23],f16>
    %int256_223 = torch.constant.int 256
    %216 = torch.prim.ListConstruct %38, %int256_223 : (!torch.int, !torch.int) -> !torch.list<int>
    %217 = torch.aten.view %212, %216 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %217, [%32], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %218 = torch.aten.mm %217, %215 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,23],f16> -> !torch.vtensor<[?,23],f16>
    torch.bind_symbolic_shape %218, [%32], affine_map<()[s0] -> (s0 * 32, 23)> : !torch.vtensor<[?,23],f16>
    %int1_224 = torch.constant.int 1
    %int23 = torch.constant.int 23
    %219 = torch.prim.ListConstruct %int1_224, %38, %int23 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %220 = torch.aten.view %218, %219 : !torch.vtensor<[?,23],f16>, !torch.list<int> -> !torch.vtensor<[1,?,23],f16>
    torch.bind_symbolic_shape %220, [%32], affine_map<()[s0] -> (1, s0 * 32, 23)> : !torch.vtensor<[1,?,23],f16>
    %221 = torch.aten.silu %220 : !torch.vtensor<[1,?,23],f16> -> !torch.vtensor<[1,?,23],f16>
    torch.bind_symbolic_shape %221, [%32], affine_map<()[s0] -> (1, s0 * 32, 23)> : !torch.vtensor<[1,?,23],f16>
    %int5_225 = torch.constant.int 5
    %222 = torch.prims.convert_element_type %8, %int5_225 : !torch.vtensor<[23,256],f32>, !torch.int -> !torch.vtensor<[23,256],f16>
    %int-2_226 = torch.constant.int -2
    %int-1_227 = torch.constant.int -1
    %223 = torch.aten.transpose.int %222, %int-2_226, %int-1_227 : !torch.vtensor<[23,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,23],f16>
    %int5_228 = torch.constant.int 5
    %224 = torch.prims.convert_element_type %223, %int5_228 : !torch.vtensor<[256,23],f16>, !torch.int -> !torch.vtensor<[256,23],f16>
    %int256_229 = torch.constant.int 256
    %225 = torch.prim.ListConstruct %38, %int256_229 : (!torch.int, !torch.int) -> !torch.list<int>
    %226 = torch.aten.view %212, %225 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %226, [%32], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %227 = torch.aten.mm %226, %224 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,23],f16> -> !torch.vtensor<[?,23],f16>
    torch.bind_symbolic_shape %227, [%32], affine_map<()[s0] -> (s0 * 32, 23)> : !torch.vtensor<[?,23],f16>
    %int1_230 = torch.constant.int 1
    %int23_231 = torch.constant.int 23
    %228 = torch.prim.ListConstruct %int1_230, %38, %int23_231 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %229 = torch.aten.view %227, %228 : !torch.vtensor<[?,23],f16>, !torch.list<int> -> !torch.vtensor<[1,?,23],f16>
    torch.bind_symbolic_shape %229, [%32], affine_map<()[s0] -> (1, s0 * 32, 23)> : !torch.vtensor<[1,?,23],f16>
    %230 = torch.aten.mul.Tensor %221, %229 : !torch.vtensor<[1,?,23],f16>, !torch.vtensor<[1,?,23],f16> -> !torch.vtensor<[1,?,23],f16>
    torch.bind_symbolic_shape %230, [%32], affine_map<()[s0] -> (1, s0 * 32, 23)> : !torch.vtensor<[1,?,23],f16>
    %int5_232 = torch.constant.int 5
    %231 = torch.prims.convert_element_type %9, %int5_232 : !torch.vtensor<[256,23],f32>, !torch.int -> !torch.vtensor<[256,23],f16>
    %int-2_233 = torch.constant.int -2
    %int-1_234 = torch.constant.int -1
    %232 = torch.aten.transpose.int %231, %int-2_233, %int-1_234 : !torch.vtensor<[256,23],f16>, !torch.int, !torch.int -> !torch.vtensor<[23,256],f16>
    %int5_235 = torch.constant.int 5
    %233 = torch.prims.convert_element_type %232, %int5_235 : !torch.vtensor<[23,256],f16>, !torch.int -> !torch.vtensor<[23,256],f16>
    %int23_236 = torch.constant.int 23
    %234 = torch.prim.ListConstruct %38, %int23_236 : (!torch.int, !torch.int) -> !torch.list<int>
    %235 = torch.aten.view %230, %234 : !torch.vtensor<[1,?,23],f16>, !torch.list<int> -> !torch.vtensor<[?,23],f16>
    torch.bind_symbolic_shape %235, [%32], affine_map<()[s0] -> (s0 * 32, 23)> : !torch.vtensor<[?,23],f16>
    %236 = torch.aten.mm %235, %233 : !torch.vtensor<[?,23],f16>, !torch.vtensor<[23,256],f16> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %236, [%32], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %int1_237 = torch.constant.int 1
    %int256_238 = torch.constant.int 256
    %237 = torch.prim.ListConstruct %int1_237, %38, %int256_238 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %238 = torch.aten.view %236, %237 : !torch.vtensor<[?,256],f16>, !torch.list<int> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %238, [%32], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int1_239 = torch.constant.int 1
    %239 = torch.aten.add.Tensor %202, %238, %int1_239 : !torch.vtensor<[1,?,256],f16>, !torch.vtensor<[1,?,256],f16>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %239, [%32], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int6_240 = torch.constant.int 6
    %240 = torch.prims.convert_element_type %239, %int6_240 : !torch.vtensor<[1,?,256],f16>, !torch.int -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %240, [%32], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int2_241 = torch.constant.int 2
    %241 = torch.aten.pow.Tensor_Scalar %240, %int2_241 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %241, [%32], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int-1_242 = torch.constant.int -1
    %242 = torch.prim.ListConstruct %int-1_242 : (!torch.int) -> !torch.list<int>
    %true_243 = torch.constant.bool true
    %none_244 = torch.constant.none
    %243 = torch.aten.mean.dim %241, %242, %true_243, %none_244 : !torch.vtensor<[1,?,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %243, [%32], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %float1.000000e-02_245 = torch.constant.float 1.000000e-02
    %int1_246 = torch.constant.int 1
    %244 = torch.aten.add.Scalar %243, %float1.000000e-02_245, %int1_246 : !torch.vtensor<[1,?,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %244, [%32], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %245 = torch.aten.rsqrt %244 : !torch.vtensor<[1,?,1],f32> -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %245, [%32], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %246 = torch.aten.mul.Tensor %240, %245 : !torch.vtensor<[1,?,256],f32>, !torch.vtensor<[1,?,1],f32> -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %246, [%32], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int5_247 = torch.constant.int 5
    %247 = torch.prims.convert_element_type %246, %int5_247 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %247, [%32], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %248 = torch.aten.mul.Tensor %10, %247 : !torch.vtensor<[256],f32>, !torch.vtensor<[1,?,256],f16> -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %248, [%32], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int5_248 = torch.constant.int 5
    %249 = torch.prims.convert_element_type %248, %int5_248 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %249, [%32], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int5_249 = torch.constant.int 5
    %250 = torch.prims.convert_element_type %11, %int5_249 : !torch.vtensor<[256,256],f32>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int-2_250 = torch.constant.int -2
    %int-1_251 = torch.constant.int -1
    %251 = torch.aten.transpose.int %250, %int-2_250, %int-1_251 : !torch.vtensor<[256,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,256],f16>
    %int5_252 = torch.constant.int 5
    %252 = torch.prims.convert_element_type %251, %int5_252 : !torch.vtensor<[256,256],f16>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int256_253 = torch.constant.int 256
    %253 = torch.prim.ListConstruct %38, %int256_253 : (!torch.int, !torch.int) -> !torch.list<int>
    %254 = torch.aten.view %249, %253 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %254, [%32], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %255 = torch.aten.mm %254, %252 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,256],f16> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %255, [%32], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %int1_254 = torch.constant.int 1
    %int256_255 = torch.constant.int 256
    %256 = torch.prim.ListConstruct %int1_254, %38, %int256_255 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %257 = torch.aten.view %255, %256 : !torch.vtensor<[?,256],f16>, !torch.list<int> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %257, [%32], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int5_256 = torch.constant.int 5
    %258 = torch.prims.convert_element_type %12, %int5_256 : !torch.vtensor<[128,256],f32>, !torch.int -> !torch.vtensor<[128,256],f16>
    %int-2_257 = torch.constant.int -2
    %int-1_258 = torch.constant.int -1
    %259 = torch.aten.transpose.int %258, %int-2_257, %int-1_258 : !torch.vtensor<[128,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,128],f16>
    %int5_259 = torch.constant.int 5
    %260 = torch.prims.convert_element_type %259, %int5_259 : !torch.vtensor<[256,128],f16>, !torch.int -> !torch.vtensor<[256,128],f16>
    %int256_260 = torch.constant.int 256
    %261 = torch.prim.ListConstruct %38, %int256_260 : (!torch.int, !torch.int) -> !torch.list<int>
    %262 = torch.aten.view %249, %261 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %262, [%32], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %263 = torch.aten.mm %262, %260 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,128],f16> -> !torch.vtensor<[?,128],f16>
    torch.bind_symbolic_shape %263, [%32], affine_map<()[s0] -> (s0 * 32, 128)> : !torch.vtensor<[?,128],f16>
    %int1_261 = torch.constant.int 1
    %int128_262 = torch.constant.int 128
    %264 = torch.prim.ListConstruct %int1_261, %38, %int128_262 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %265 = torch.aten.view %263, %264 : !torch.vtensor<[?,128],f16>, !torch.list<int> -> !torch.vtensor<[1,?,128],f16>
    torch.bind_symbolic_shape %265, [%32], affine_map<()[s0] -> (1, s0 * 32, 128)> : !torch.vtensor<[1,?,128],f16>
    %int5_263 = torch.constant.int 5
    %266 = torch.prims.convert_element_type %13, %int5_263 : !torch.vtensor<[128,256],f32>, !torch.int -> !torch.vtensor<[128,256],f16>
    %int-2_264 = torch.constant.int -2
    %int-1_265 = torch.constant.int -1
    %267 = torch.aten.transpose.int %266, %int-2_264, %int-1_265 : !torch.vtensor<[128,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,128],f16>
    %int5_266 = torch.constant.int 5
    %268 = torch.prims.convert_element_type %267, %int5_266 : !torch.vtensor<[256,128],f16>, !torch.int -> !torch.vtensor<[256,128],f16>
    %int256_267 = torch.constant.int 256
    %269 = torch.prim.ListConstruct %38, %int256_267 : (!torch.int, !torch.int) -> !torch.list<int>
    %270 = torch.aten.view %249, %269 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %270, [%32], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %271 = torch.aten.mm %270, %268 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,128],f16> -> !torch.vtensor<[?,128],f16>
    torch.bind_symbolic_shape %271, [%32], affine_map<()[s0] -> (s0 * 32, 128)> : !torch.vtensor<[?,128],f16>
    %int1_268 = torch.constant.int 1
    %int128_269 = torch.constant.int 128
    %272 = torch.prim.ListConstruct %int1_268, %38, %int128_269 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %273 = torch.aten.view %271, %272 : !torch.vtensor<[?,128],f16>, !torch.list<int> -> !torch.vtensor<[1,?,128],f16>
    torch.bind_symbolic_shape %273, [%32], affine_map<()[s0] -> (1, s0 * 32, 128)> : !torch.vtensor<[1,?,128],f16>
    %int1_270 = torch.constant.int 1
    %int8_271 = torch.constant.int 8
    %int32_272 = torch.constant.int 32
    %274 = torch.prim.ListConstruct %int1_270, %38, %int8_271, %int32_272 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %275 = torch.aten.view %257, %274 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[1,?,8,32],f16>
    torch.bind_symbolic_shape %275, [%32], affine_map<()[s0] -> (1, s0 * 32, 8, 32)> : !torch.vtensor<[1,?,8,32],f16>
    %int1_273 = torch.constant.int 1
    %int4_274 = torch.constant.int 4
    %int32_275 = torch.constant.int 32
    %276 = torch.prim.ListConstruct %int1_273, %38, %int4_274, %int32_275 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %277 = torch.aten.view %265, %276 : !torch.vtensor<[1,?,128],f16>, !torch.list<int> -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %277, [%32], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int1_276 = torch.constant.int 1
    %int4_277 = torch.constant.int 4
    %int32_278 = torch.constant.int 32
    %278 = torch.prim.ListConstruct %int1_276, %38, %int4_277, %int32_278 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %279 = torch.aten.view %273, %278 : !torch.vtensor<[1,?,128],f16>, !torch.list<int> -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %279, [%32], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int128_279 = torch.constant.int 128
    %none_280 = torch.constant.none
    %none_281 = torch.constant.none
    %cpu_282 = torch.constant.device "cpu"
    %false_283 = torch.constant.bool false
    %280 = torch.aten.arange %int128_279, %none_280, %none_281, %cpu_282, %false_283 : !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[128],si64>
    %int0_284 = torch.constant.int 0
    %int32_285 = torch.constant.int 32
    %none_286 = torch.constant.none
    %none_287 = torch.constant.none
    %cpu_288 = torch.constant.device "cpu"
    %false_289 = torch.constant.bool false
    %281 = torch.aten.arange.start %int0_284, %int32_285, %none_286, %none_287, %cpu_288, %false_289 : !torch.int, !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[32],si64>
    %int2_290 = torch.constant.int 2
    %282 = torch.aten.floor_divide.Scalar %281, %int2_290 : !torch.vtensor<[32],si64>, !torch.int -> !torch.vtensor<[32],si64>
    %int6_291 = torch.constant.int 6
    %283 = torch.prims.convert_element_type %282, %int6_291 : !torch.vtensor<[32],si64>, !torch.int -> !torch.vtensor<[32],f32>
    %int32_292 = torch.constant.int 32
    %284 = torch.aten.div.Scalar %283, %int32_292 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[32],f32>
    %float2.000000e00_293 = torch.constant.float 2.000000e+00
    %285 = torch.aten.mul.Scalar %284, %float2.000000e00_293 : !torch.vtensor<[32],f32>, !torch.float -> !torch.vtensor<[32],f32>
    %float5.000000e05_294 = torch.constant.float 5.000000e+05
    %286 = torch.aten.pow.Scalar %float5.000000e05_294, %285 : !torch.float, !torch.vtensor<[32],f32> -> !torch.vtensor<[32],f32>
    %287 = torch.aten.reciprocal %286 : !torch.vtensor<[32],f32> -> !torch.vtensor<[32],f32>
    %float1.000000e00_295 = torch.constant.float 1.000000e+00
    %288 = torch.aten.mul.Scalar %287, %float1.000000e00_295 : !torch.vtensor<[32],f32>, !torch.float -> !torch.vtensor<[32],f32>
    %int1_296 = torch.constant.int 1
    %289 = torch.aten.unsqueeze %280, %int1_296 : !torch.vtensor<[128],si64>, !torch.int -> !torch.vtensor<[128,1],si64>
    %int0_297 = torch.constant.int 0
    %290 = torch.aten.unsqueeze %288, %int0_297 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[1,32],f32>
    %291 = torch.aten.mul.Tensor %289, %290 : !torch.vtensor<[128,1],si64>, !torch.vtensor<[1,32],f32> -> !torch.vtensor<[128,32],f32>
    %int6_298 = torch.constant.int 6
    %292 = torch.prims.convert_element_type %291, %int6_298 : !torch.vtensor<[128,32],f32>, !torch.int -> !torch.vtensor<[128,32],f32>
    %int0_299 = torch.constant.int 0
    %int0_300 = torch.constant.int 0
    %int1_301 = torch.constant.int 1
    %293 = torch.aten.slice.Tensor %292, %int0_299, %int0_300, %38, %int1_301 : !torch.vtensor<[128,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %293, [%32], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int1_302 = torch.constant.int 1
    %int0_303 = torch.constant.int 0
    %int9223372036854775807_304 = torch.constant.int 9223372036854775807
    %int1_305 = torch.constant.int 1
    %294 = torch.aten.slice.Tensor %293, %int1_302, %int0_303, %int9223372036854775807_304, %int1_305 : !torch.vtensor<[?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %294, [%32], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int1_306 = torch.constant.int 1
    %int0_307 = torch.constant.int 0
    %int9223372036854775807_308 = torch.constant.int 9223372036854775807
    %int1_309 = torch.constant.int 1
    %295 = torch.aten.slice.Tensor %294, %int1_306, %int0_307, %int9223372036854775807_308, %int1_309 : !torch.vtensor<[?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %295, [%32], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int0_310 = torch.constant.int 0
    %296 = torch.aten.unsqueeze %295, %int0_310 : !torch.vtensor<[?,32],f32>, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %296, [%32], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int1_311 = torch.constant.int 1
    %int0_312 = torch.constant.int 0
    %int9223372036854775807_313 = torch.constant.int 9223372036854775807
    %int1_314 = torch.constant.int 1
    %297 = torch.aten.slice.Tensor %296, %int1_311, %int0_312, %int9223372036854775807_313, %int1_314 : !torch.vtensor<[1,?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %297, [%32], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int2_315 = torch.constant.int 2
    %int0_316 = torch.constant.int 0
    %int9223372036854775807_317 = torch.constant.int 9223372036854775807
    %int1_318 = torch.constant.int 1
    %298 = torch.aten.slice.Tensor %297, %int2_315, %int0_316, %int9223372036854775807_317, %int1_318 : !torch.vtensor<[1,?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %298, [%32], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int1_319 = torch.constant.int 1
    %int1_320 = torch.constant.int 1
    %int1_321 = torch.constant.int 1
    %299 = torch.prim.ListConstruct %int1_319, %int1_320, %int1_321 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %300 = torch.aten.repeat %298, %299 : !torch.vtensor<[1,?,32],f32>, !torch.list<int> -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %300, [%32], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int6_322 = torch.constant.int 6
    %301 = torch.prims.convert_element_type %275, %int6_322 : !torch.vtensor<[1,?,8,32],f16>, !torch.int -> !torch.vtensor<[1,?,8,32],f32>
    torch.bind_symbolic_shape %301, [%32], affine_map<()[s0] -> (1, s0 * 32, 8, 32)> : !torch.vtensor<[1,?,8,32],f32>
    %302 = torch_c.to_builtin_tensor %301 : !torch.vtensor<[1,?,8,32],f32> -> tensor<1x?x8x32xf32>
    %303 = torch_c.to_builtin_tensor %300 : !torch.vtensor<[1,?,32],f32> -> tensor<1x?x32xf32>
    %304 = util.call @sharktank_rotary_embedding_1_D_8_32_f32(%302, %303) : (tensor<1x?x8x32xf32>, tensor<1x?x32xf32>) -> tensor<1x?x8x32xf32>
    %305 = torch_c.from_builtin_tensor %304 : tensor<1x?x8x32xf32> -> !torch.vtensor<[1,?,8,32],f32>
    torch.bind_symbolic_shape %305, [%32], affine_map<()[s0] -> (1, s0 * 32, 8, 32)> : !torch.vtensor<[1,?,8,32],f32>
    %int5_323 = torch.constant.int 5
    %306 = torch.prims.convert_element_type %305, %int5_323 : !torch.vtensor<[1,?,8,32],f32>, !torch.int -> !torch.vtensor<[1,?,8,32],f16>
    torch.bind_symbolic_shape %306, [%32], affine_map<()[s0] -> (1, s0 * 32, 8, 32)> : !torch.vtensor<[1,?,8,32],f16>
    %int128_324 = torch.constant.int 128
    %none_325 = torch.constant.none
    %none_326 = torch.constant.none
    %cpu_327 = torch.constant.device "cpu"
    %false_328 = torch.constant.bool false
    %307 = torch.aten.arange %int128_324, %none_325, %none_326, %cpu_327, %false_328 : !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[128],si64>
    %int0_329 = torch.constant.int 0
    %int32_330 = torch.constant.int 32
    %none_331 = torch.constant.none
    %none_332 = torch.constant.none
    %cpu_333 = torch.constant.device "cpu"
    %false_334 = torch.constant.bool false
    %308 = torch.aten.arange.start %int0_329, %int32_330, %none_331, %none_332, %cpu_333, %false_334 : !torch.int, !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[32],si64>
    %int2_335 = torch.constant.int 2
    %309 = torch.aten.floor_divide.Scalar %308, %int2_335 : !torch.vtensor<[32],si64>, !torch.int -> !torch.vtensor<[32],si64>
    %int6_336 = torch.constant.int 6
    %310 = torch.prims.convert_element_type %309, %int6_336 : !torch.vtensor<[32],si64>, !torch.int -> !torch.vtensor<[32],f32>
    %int32_337 = torch.constant.int 32
    %311 = torch.aten.div.Scalar %310, %int32_337 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[32],f32>
    %float2.000000e00_338 = torch.constant.float 2.000000e+00
    %312 = torch.aten.mul.Scalar %311, %float2.000000e00_338 : !torch.vtensor<[32],f32>, !torch.float -> !torch.vtensor<[32],f32>
    %float5.000000e05_339 = torch.constant.float 5.000000e+05
    %313 = torch.aten.pow.Scalar %float5.000000e05_339, %312 : !torch.float, !torch.vtensor<[32],f32> -> !torch.vtensor<[32],f32>
    %314 = torch.aten.reciprocal %313 : !torch.vtensor<[32],f32> -> !torch.vtensor<[32],f32>
    %float1.000000e00_340 = torch.constant.float 1.000000e+00
    %315 = torch.aten.mul.Scalar %314, %float1.000000e00_340 : !torch.vtensor<[32],f32>, !torch.float -> !torch.vtensor<[32],f32>
    %int1_341 = torch.constant.int 1
    %316 = torch.aten.unsqueeze %307, %int1_341 : !torch.vtensor<[128],si64>, !torch.int -> !torch.vtensor<[128,1],si64>
    %int0_342 = torch.constant.int 0
    %317 = torch.aten.unsqueeze %315, %int0_342 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[1,32],f32>
    %318 = torch.aten.mul.Tensor %316, %317 : !torch.vtensor<[128,1],si64>, !torch.vtensor<[1,32],f32> -> !torch.vtensor<[128,32],f32>
    %int6_343 = torch.constant.int 6
    %319 = torch.prims.convert_element_type %318, %int6_343 : !torch.vtensor<[128,32],f32>, !torch.int -> !torch.vtensor<[128,32],f32>
    %int0_344 = torch.constant.int 0
    %int0_345 = torch.constant.int 0
    %int1_346 = torch.constant.int 1
    %320 = torch.aten.slice.Tensor %319, %int0_344, %int0_345, %38, %int1_346 : !torch.vtensor<[128,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %320, [%32], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int1_347 = torch.constant.int 1
    %int0_348 = torch.constant.int 0
    %int9223372036854775807_349 = torch.constant.int 9223372036854775807
    %int1_350 = torch.constant.int 1
    %321 = torch.aten.slice.Tensor %320, %int1_347, %int0_348, %int9223372036854775807_349, %int1_350 : !torch.vtensor<[?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %321, [%32], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int1_351 = torch.constant.int 1
    %int0_352 = torch.constant.int 0
    %int9223372036854775807_353 = torch.constant.int 9223372036854775807
    %int1_354 = torch.constant.int 1
    %322 = torch.aten.slice.Tensor %321, %int1_351, %int0_352, %int9223372036854775807_353, %int1_354 : !torch.vtensor<[?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %322, [%32], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int0_355 = torch.constant.int 0
    %323 = torch.aten.unsqueeze %322, %int0_355 : !torch.vtensor<[?,32],f32>, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %323, [%32], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int1_356 = torch.constant.int 1
    %int0_357 = torch.constant.int 0
    %int9223372036854775807_358 = torch.constant.int 9223372036854775807
    %int1_359 = torch.constant.int 1
    %324 = torch.aten.slice.Tensor %323, %int1_356, %int0_357, %int9223372036854775807_358, %int1_359 : !torch.vtensor<[1,?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %324, [%32], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int2_360 = torch.constant.int 2
    %int0_361 = torch.constant.int 0
    %int9223372036854775807_362 = torch.constant.int 9223372036854775807
    %int1_363 = torch.constant.int 1
    %325 = torch.aten.slice.Tensor %324, %int2_360, %int0_361, %int9223372036854775807_362, %int1_363 : !torch.vtensor<[1,?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %325, [%32], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int1_364 = torch.constant.int 1
    %int1_365 = torch.constant.int 1
    %int1_366 = torch.constant.int 1
    %326 = torch.prim.ListConstruct %int1_364, %int1_365, %int1_366 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %327 = torch.aten.repeat %325, %326 : !torch.vtensor<[1,?,32],f32>, !torch.list<int> -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %327, [%32], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int6_367 = torch.constant.int 6
    %328 = torch.prims.convert_element_type %277, %int6_367 : !torch.vtensor<[1,?,4,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,32],f32>
    torch.bind_symbolic_shape %328, [%32], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f32>
    %329 = torch_c.to_builtin_tensor %328 : !torch.vtensor<[1,?,4,32],f32> -> tensor<1x?x4x32xf32>
    %330 = torch_c.to_builtin_tensor %327 : !torch.vtensor<[1,?,32],f32> -> tensor<1x?x32xf32>
    %331 = util.call @sharktank_rotary_embedding_1_D_4_32_f32(%329, %330) : (tensor<1x?x4x32xf32>, tensor<1x?x32xf32>) -> tensor<1x?x4x32xf32>
    %332 = torch_c.from_builtin_tensor %331 : tensor<1x?x4x32xf32> -> !torch.vtensor<[1,?,4,32],f32>
    torch.bind_symbolic_shape %332, [%32], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f32>
    %int5_368 = torch.constant.int 5
    %333 = torch.prims.convert_element_type %332, %int5_368 : !torch.vtensor<[1,?,4,32],f32>, !torch.int -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %333, [%32], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int6_369 = torch.constant.int 6
    %334 = torch.aten.mul.Scalar %arg2, %int6_369 : !torch.vtensor<[1,?],si64>, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %334, [%32], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %int2_370 = torch.constant.int 2
    %int1_371 = torch.constant.int 1
    %335 = torch.aten.add.Scalar %334, %int2_370, %int1_371 : !torch.vtensor<[1,?],si64>, !torch.int, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %335, [%32], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %int1_372 = torch.constant.int 1
    %int32_373 = torch.constant.int 32
    %int4_374 = torch.constant.int 4
    %int32_375 = torch.constant.int 32
    %336 = torch.prim.ListConstruct %int1_372, %34, %int32_373, %int4_374, %int32_375 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %337 = torch.aten.view %333, %336 : !torch.vtensor<[1,?,4,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,32,4,32],f16>
    torch.bind_symbolic_shape %337, [%32], affine_map<()[s0] -> (1, s0, 32, 4, 32)> : !torch.vtensor<[1,?,32,4,32],f16>
    %int32_376 = torch.constant.int 32
    %int4_377 = torch.constant.int 4
    %int32_378 = torch.constant.int 32
    %338 = torch.prim.ListConstruct %34, %int32_376, %int4_377, %int32_378 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %339 = torch.aten.view %337, %338 : !torch.vtensor<[1,?,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,32,4,32],f16>
    torch.bind_symbolic_shape %339, [%32], affine_map<()[s0] -> (s0, 32, 4, 32)> : !torch.vtensor<[?,32,4,32],f16>
    %340 = torch.prim.ListConstruct %34 : (!torch.int) -> !torch.list<int>
    %341 = torch.aten.view %335, %340 : !torch.vtensor<[1,?],si64>, !torch.list<int> -> !torch.vtensor<[?],si64>
    torch.bind_symbolic_shape %341, [%32], affine_map<()[s0] -> (s0)> : !torch.vtensor<[?],si64>
    %int5_379 = torch.constant.int 5
    %342 = torch.prims.convert_element_type %339, %int5_379 : !torch.vtensor<[?,32,4,32],f16>, !torch.int -> !torch.vtensor<[?,32,4,32],f16>
    torch.bind_symbolic_shape %342, [%32], affine_map<()[s0] -> (s0, 32, 4, 32)> : !torch.vtensor<[?,32,4,32],f16>
    %int3_380 = torch.constant.int 3
    %int2_381 = torch.constant.int 2
    %int32_382 = torch.constant.int 32
    %int4_383 = torch.constant.int 4
    %int32_384 = torch.constant.int 32
    %343 = torch.prim.ListConstruct %35, %int3_380, %int2_381, %int32_382, %int4_383, %int32_384 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %344 = torch.aten.view %171, %343 : !torch.vtensor<[?,24576],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,4,32],f16>
    torch.bind_symbolic_shape %344, [%33], affine_map<()[s0] -> (s0, 3, 2, 32, 4, 32)> : !torch.vtensor<[?,3,2,32,4,32],f16>
    %int32_385 = torch.constant.int 32
    %int4_386 = torch.constant.int 4
    %int32_387 = torch.constant.int 32
    %345 = torch.prim.ListConstruct %136, %int32_385, %int4_386, %int32_387 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %346 = torch.aten.view %344, %345 : !torch.vtensor<[?,3,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,32,4,32],f16>
    torch.bind_symbolic_shape %346, [%33], affine_map<()[s0] -> (s0 * 6, 32, 4, 32)> : !torch.vtensor<[?,32,4,32],f16>
    %347 = torch.prim.ListConstruct %341 : (!torch.vtensor<[?],si64>) -> !torch.list<optional<vtensor>>
    %false_388 = torch.constant.bool false
    %348 = torch.aten.index_put %346, %347, %342, %false_388 : !torch.vtensor<[?,32,4,32],f16>, !torch.list<optional<vtensor>>, !torch.vtensor<[?,32,4,32],f16>, !torch.bool -> !torch.vtensor<[?,32,4,32],f16>
    torch.bind_symbolic_shape %348, [%33], affine_map<()[s0] -> (s0 * 6, 32, 4, 32)> : !torch.vtensor<[?,32,4,32],f16>
    %int3_389 = torch.constant.int 3
    %int2_390 = torch.constant.int 2
    %int32_391 = torch.constant.int 32
    %int4_392 = torch.constant.int 4
    %int32_393 = torch.constant.int 32
    %349 = torch.prim.ListConstruct %35, %int3_389, %int2_390, %int32_391, %int4_392, %int32_393 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %350 = torch.aten.view %348, %349 : !torch.vtensor<[?,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,4,32],f16>
    torch.bind_symbolic_shape %350, [%33], affine_map<()[s0] -> (s0, 3, 2, 32, 4, 32)> : !torch.vtensor<[?,3,2,32,4,32],f16>
    %int24576_394 = torch.constant.int 24576
    %351 = torch.prim.ListConstruct %35, %int24576_394 : (!torch.int, !torch.int) -> !torch.list<int>
    %352 = torch.aten.view %350, %351 : !torch.vtensor<[?,3,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,24576],f16>
    torch.bind_symbolic_shape %352, [%33], affine_map<()[s0] -> (s0, 24576)> : !torch.vtensor<[?,24576],f16>
    %int3_395 = torch.constant.int 3
    %int2_396 = torch.constant.int 2
    %int32_397 = torch.constant.int 32
    %int4_398 = torch.constant.int 4
    %int32_399 = torch.constant.int 32
    %353 = torch.prim.ListConstruct %35, %int3_395, %int2_396, %int32_397, %int4_398, %int32_399 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %354 = torch.aten.view %352, %353 : !torch.vtensor<[?,24576],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,4,32],f16>
    torch.bind_symbolic_shape %354, [%33], affine_map<()[s0] -> (s0, 3, 2, 32, 4, 32)> : !torch.vtensor<[?,3,2,32,4,32],f16>
    %int32_400 = torch.constant.int 32
    %int4_401 = torch.constant.int 4
    %int32_402 = torch.constant.int 32
    %355 = torch.prim.ListConstruct %136, %int32_400, %int4_401, %int32_402 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %356 = torch.aten.view %354, %355 : !torch.vtensor<[?,3,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,32,4,32],f16>
    torch.bind_symbolic_shape %356, [%33], affine_map<()[s0] -> (s0 * 6, 32, 4, 32)> : !torch.vtensor<[?,32,4,32],f16>
    %int1_403 = torch.constant.int 1
    %int32_404 = torch.constant.int 32
    %int4_405 = torch.constant.int 4
    %int32_406 = torch.constant.int 32
    %357 = torch.prim.ListConstruct %int1_403, %34, %int32_404, %int4_405, %int32_406 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %358 = torch.aten.view %279, %357 : !torch.vtensor<[1,?,4,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,32,4,32],f16>
    torch.bind_symbolic_shape %358, [%32], affine_map<()[s0] -> (1, s0, 32, 4, 32)> : !torch.vtensor<[1,?,32,4,32],f16>
    %int32_407 = torch.constant.int 32
    %int4_408 = torch.constant.int 4
    %int32_409 = torch.constant.int 32
    %359 = torch.prim.ListConstruct %34, %int32_407, %int4_408, %int32_409 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %360 = torch.aten.view %358, %359 : !torch.vtensor<[1,?,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,32,4,32],f16>
    torch.bind_symbolic_shape %360, [%32], affine_map<()[s0] -> (s0, 32, 4, 32)> : !torch.vtensor<[?,32,4,32],f16>
    %int1_410 = torch.constant.int 1
    %int1_411 = torch.constant.int 1
    %361 = torch.aten.add.Scalar %335, %int1_410, %int1_411 : !torch.vtensor<[1,?],si64>, !torch.int, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %361, [%32], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %362 = torch.prim.ListConstruct %34 : (!torch.int) -> !torch.list<int>
    %363 = torch.aten.view %361, %362 : !torch.vtensor<[1,?],si64>, !torch.list<int> -> !torch.vtensor<[?],si64>
    torch.bind_symbolic_shape %363, [%32], affine_map<()[s0] -> (s0)> : !torch.vtensor<[?],si64>
    %int5_412 = torch.constant.int 5
    %364 = torch.prims.convert_element_type %360, %int5_412 : !torch.vtensor<[?,32,4,32],f16>, !torch.int -> !torch.vtensor<[?,32,4,32],f16>
    torch.bind_symbolic_shape %364, [%32], affine_map<()[s0] -> (s0, 32, 4, 32)> : !torch.vtensor<[?,32,4,32],f16>
    %365 = torch.prim.ListConstruct %363 : (!torch.vtensor<[?],si64>) -> !torch.list<optional<vtensor>>
    %false_413 = torch.constant.bool false
    %366 = torch.aten.index_put %356, %365, %364, %false_413 : !torch.vtensor<[?,32,4,32],f16>, !torch.list<optional<vtensor>>, !torch.vtensor<[?,32,4,32],f16>, !torch.bool -> !torch.vtensor<[?,32,4,32],f16>
    torch.bind_symbolic_shape %366, [%33], affine_map<()[s0] -> (s0 * 6, 32, 4, 32)> : !torch.vtensor<[?,32,4,32],f16>
    %int3_414 = torch.constant.int 3
    %int2_415 = torch.constant.int 2
    %int32_416 = torch.constant.int 32
    %int4_417 = torch.constant.int 4
    %int32_418 = torch.constant.int 32
    %367 = torch.prim.ListConstruct %35, %int3_414, %int2_415, %int32_416, %int4_417, %int32_418 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %368 = torch.aten.view %366, %367 : !torch.vtensor<[?,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,4,32],f16>
    torch.bind_symbolic_shape %368, [%33], affine_map<()[s0] -> (s0, 3, 2, 32, 4, 32)> : !torch.vtensor<[?,3,2,32,4,32],f16>
    %int24576_419 = torch.constant.int 24576
    %369 = torch.prim.ListConstruct %35, %int24576_419 : (!torch.int, !torch.int) -> !torch.list<int>
    %370 = torch.aten.view %368, %369 : !torch.vtensor<[?,3,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,24576],f16>
    torch.bind_symbolic_shape %370, [%33], affine_map<()[s0] -> (s0, 24576)> : !torch.vtensor<[?,24576],f16>
    %int-2_420 = torch.constant.int -2
    %371 = torch.aten.unsqueeze %333, %int-2_420 : !torch.vtensor<[1,?,4,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,1,32],f16>
    torch.bind_symbolic_shape %371, [%32], affine_map<()[s0] -> (1, s0 * 32, 4, 1, 32)> : !torch.vtensor<[1,?,4,1,32],f16>
    %int1_421 = torch.constant.int 1
    %int4_422 = torch.constant.int 4
    %int2_423 = torch.constant.int 2
    %int32_424 = torch.constant.int 32
    %372 = torch.prim.ListConstruct %int1_421, %38, %int4_422, %int2_423, %int32_424 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_425 = torch.constant.bool false
    %373 = torch.aten.expand %371, %372, %false_425 : !torch.vtensor<[1,?,4,1,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,?,4,2,32],f16>
    torch.bind_symbolic_shape %373, [%32], affine_map<()[s0] -> (1, s0 * 32, 4, 2, 32)> : !torch.vtensor<[1,?,4,2,32],f16>
    %int0_426 = torch.constant.int 0
    %374 = torch.aten.clone %373, %int0_426 : !torch.vtensor<[1,?,4,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,2,32],f16>
    torch.bind_symbolic_shape %374, [%32], affine_map<()[s0] -> (1, s0 * 32, 4, 2, 32)> : !torch.vtensor<[1,?,4,2,32],f16>
    %int1_427 = torch.constant.int 1
    %int8_428 = torch.constant.int 8
    %int32_429 = torch.constant.int 32
    %375 = torch.prim.ListConstruct %int1_427, %38, %int8_428, %int32_429 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %376 = torch.aten._unsafe_view %374, %375 : !torch.vtensor<[1,?,4,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,8,32],f16>
    torch.bind_symbolic_shape %376, [%32], affine_map<()[s0] -> (1, s0 * 32, 8, 32)> : !torch.vtensor<[1,?,8,32],f16>
    %int-2_430 = torch.constant.int -2
    %377 = torch.aten.unsqueeze %279, %int-2_430 : !torch.vtensor<[1,?,4,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,1,32],f16>
    torch.bind_symbolic_shape %377, [%32], affine_map<()[s0] -> (1, s0 * 32, 4, 1, 32)> : !torch.vtensor<[1,?,4,1,32],f16>
    %int1_431 = torch.constant.int 1
    %int4_432 = torch.constant.int 4
    %int2_433 = torch.constant.int 2
    %int32_434 = torch.constant.int 32
    %378 = torch.prim.ListConstruct %int1_431, %38, %int4_432, %int2_433, %int32_434 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_435 = torch.constant.bool false
    %379 = torch.aten.expand %377, %378, %false_435 : !torch.vtensor<[1,?,4,1,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,?,4,2,32],f16>
    torch.bind_symbolic_shape %379, [%32], affine_map<()[s0] -> (1, s0 * 32, 4, 2, 32)> : !torch.vtensor<[1,?,4,2,32],f16>
    %int0_436 = torch.constant.int 0
    %380 = torch.aten.clone %379, %int0_436 : !torch.vtensor<[1,?,4,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,2,32],f16>
    torch.bind_symbolic_shape %380, [%32], affine_map<()[s0] -> (1, s0 * 32, 4, 2, 32)> : !torch.vtensor<[1,?,4,2,32],f16>
    %int1_437 = torch.constant.int 1
    %int8_438 = torch.constant.int 8
    %int32_439 = torch.constant.int 32
    %381 = torch.prim.ListConstruct %int1_437, %38, %int8_438, %int32_439 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %382 = torch.aten._unsafe_view %380, %381 : !torch.vtensor<[1,?,4,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,8,32],f16>
    torch.bind_symbolic_shape %382, [%32], affine_map<()[s0] -> (1, s0 * 32, 8, 32)> : !torch.vtensor<[1,?,8,32],f16>
    %int1_440 = torch.constant.int 1
    %int2_441 = torch.constant.int 2
    %383 = torch.aten.transpose.int %306, %int1_440, %int2_441 : !torch.vtensor<[1,?,8,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,8,?,32],f16>
    torch.bind_symbolic_shape %383, [%32], affine_map<()[s0] -> (1, 8, s0 * 32, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %int1_442 = torch.constant.int 1
    %int2_443 = torch.constant.int 2
    %384 = torch.aten.transpose.int %376, %int1_442, %int2_443 : !torch.vtensor<[1,?,8,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,8,?,32],f16>
    torch.bind_symbolic_shape %384, [%32], affine_map<()[s0] -> (1, 8, s0 * 32, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %int1_444 = torch.constant.int 1
    %int2_445 = torch.constant.int 2
    %385 = torch.aten.transpose.int %382, %int1_444, %int2_445 : !torch.vtensor<[1,?,8,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,8,?,32],f16>
    torch.bind_symbolic_shape %385, [%32], affine_map<()[s0] -> (1, 8, s0 * 32, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %int5_446 = torch.constant.int 5
    %386 = torch.prims.convert_element_type %383, %int5_446 : !torch.vtensor<[1,8,?,32],f16>, !torch.int -> !torch.vtensor<[1,8,?,32],f16>
    torch.bind_symbolic_shape %386, [%32], affine_map<()[s0] -> (1, 8, s0 * 32, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %int5_447 = torch.constant.int 5
    %387 = torch.prims.convert_element_type %384, %int5_447 : !torch.vtensor<[1,8,?,32],f16>, !torch.int -> !torch.vtensor<[1,8,?,32],f16>
    torch.bind_symbolic_shape %387, [%32], affine_map<()[s0] -> (1, 8, s0 * 32, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %int5_448 = torch.constant.int 5
    %388 = torch.prims.convert_element_type %385, %int5_448 : !torch.vtensor<[1,8,?,32],f16>, !torch.int -> !torch.vtensor<[1,8,?,32],f16>
    torch.bind_symbolic_shape %388, [%32], affine_map<()[s0] -> (1, 8, s0 * 32, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %float0.000000e00_449 = torch.constant.float 0.000000e+00
    %true_450 = torch.constant.bool true
    %none_451 = torch.constant.none
    %none_452 = torch.constant.none
    %389:2 = torch.operator "torch.aten._scaled_dot_product_flash_attention_for_cpu"(%386, %387, %388, %float0.000000e00_449, %true_450, %none_451, %none_452) : (!torch.vtensor<[1,8,?,32],f16>, !torch.vtensor<[1,8,?,32],f16>, !torch.vtensor<[1,8,?,32],f16>, !torch.float, !torch.bool, !torch.none, !torch.none) -> (!torch.vtensor<[1,8,?,32],f16>, !torch.vtensor<[1,8,?],f32>) 
    torch.bind_symbolic_shape %389#0, [%32], affine_map<()[s0] -> (1, 8, s0 * 32, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %int1_453 = torch.constant.int 1
    %int2_454 = torch.constant.int 2
    %390 = torch.aten.transpose.int %389#0, %int1_453, %int2_454 : !torch.vtensor<[1,8,?,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,?,8,32],f16>
    torch.bind_symbolic_shape %390, [%32], affine_map<()[s0] -> (1, s0 * 32, 8, 32)> : !torch.vtensor<[1,?,8,32],f16>
    %int1_455 = torch.constant.int 1
    %int256_456 = torch.constant.int 256
    %391 = torch.prim.ListConstruct %int1_455, %38, %int256_456 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %392 = torch.aten.view %390, %391 : !torch.vtensor<[1,?,8,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %392, [%32], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int5_457 = torch.constant.int 5
    %393 = torch.prims.convert_element_type %14, %int5_457 : !torch.vtensor<[256,256],f32>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int-2_458 = torch.constant.int -2
    %int-1_459 = torch.constant.int -1
    %394 = torch.aten.transpose.int %393, %int-2_458, %int-1_459 : !torch.vtensor<[256,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,256],f16>
    %int5_460 = torch.constant.int 5
    %395 = torch.prims.convert_element_type %394, %int5_460 : !torch.vtensor<[256,256],f16>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int256_461 = torch.constant.int 256
    %396 = torch.prim.ListConstruct %38, %int256_461 : (!torch.int, !torch.int) -> !torch.list<int>
    %397 = torch.aten.view %392, %396 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %397, [%32], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %398 = torch.aten.mm %397, %395 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,256],f16> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %398, [%32], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %int1_462 = torch.constant.int 1
    %int256_463 = torch.constant.int 256
    %399 = torch.prim.ListConstruct %int1_462, %38, %int256_463 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %400 = torch.aten.view %398, %399 : !torch.vtensor<[?,256],f16>, !torch.list<int> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %400, [%32], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int1_464 = torch.constant.int 1
    %401 = torch.aten.add.Tensor %239, %400, %int1_464 : !torch.vtensor<[1,?,256],f16>, !torch.vtensor<[1,?,256],f16>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %401, [%32], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int6_465 = torch.constant.int 6
    %402 = torch.prims.convert_element_type %401, %int6_465 : !torch.vtensor<[1,?,256],f16>, !torch.int -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %402, [%32], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int2_466 = torch.constant.int 2
    %403 = torch.aten.pow.Tensor_Scalar %402, %int2_466 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %403, [%32], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int-1_467 = torch.constant.int -1
    %404 = torch.prim.ListConstruct %int-1_467 : (!torch.int) -> !torch.list<int>
    %true_468 = torch.constant.bool true
    %none_469 = torch.constant.none
    %405 = torch.aten.mean.dim %403, %404, %true_468, %none_469 : !torch.vtensor<[1,?,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %405, [%32], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %float1.000000e-02_470 = torch.constant.float 1.000000e-02
    %int1_471 = torch.constant.int 1
    %406 = torch.aten.add.Scalar %405, %float1.000000e-02_470, %int1_471 : !torch.vtensor<[1,?,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %406, [%32], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %407 = torch.aten.rsqrt %406 : !torch.vtensor<[1,?,1],f32> -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %407, [%32], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %408 = torch.aten.mul.Tensor %402, %407 : !torch.vtensor<[1,?,256],f32>, !torch.vtensor<[1,?,1],f32> -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %408, [%32], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int5_472 = torch.constant.int 5
    %409 = torch.prims.convert_element_type %408, %int5_472 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %409, [%32], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %410 = torch.aten.mul.Tensor %15, %409 : !torch.vtensor<[256],f32>, !torch.vtensor<[1,?,256],f16> -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %410, [%32], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int5_473 = torch.constant.int 5
    %411 = torch.prims.convert_element_type %410, %int5_473 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %411, [%32], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int5_474 = torch.constant.int 5
    %412 = torch.prims.convert_element_type %16, %int5_474 : !torch.vtensor<[23,256],f32>, !torch.int -> !torch.vtensor<[23,256],f16>
    %int-2_475 = torch.constant.int -2
    %int-1_476 = torch.constant.int -1
    %413 = torch.aten.transpose.int %412, %int-2_475, %int-1_476 : !torch.vtensor<[23,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,23],f16>
    %int5_477 = torch.constant.int 5
    %414 = torch.prims.convert_element_type %413, %int5_477 : !torch.vtensor<[256,23],f16>, !torch.int -> !torch.vtensor<[256,23],f16>
    %int256_478 = torch.constant.int 256
    %415 = torch.prim.ListConstruct %38, %int256_478 : (!torch.int, !torch.int) -> !torch.list<int>
    %416 = torch.aten.view %411, %415 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %416, [%32], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %417 = torch.aten.mm %416, %414 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,23],f16> -> !torch.vtensor<[?,23],f16>
    torch.bind_symbolic_shape %417, [%32], affine_map<()[s0] -> (s0 * 32, 23)> : !torch.vtensor<[?,23],f16>
    %int1_479 = torch.constant.int 1
    %int23_480 = torch.constant.int 23
    %418 = torch.prim.ListConstruct %int1_479, %38, %int23_480 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %419 = torch.aten.view %417, %418 : !torch.vtensor<[?,23],f16>, !torch.list<int> -> !torch.vtensor<[1,?,23],f16>
    torch.bind_symbolic_shape %419, [%32], affine_map<()[s0] -> (1, s0 * 32, 23)> : !torch.vtensor<[1,?,23],f16>
    %420 = torch.aten.silu %419 : !torch.vtensor<[1,?,23],f16> -> !torch.vtensor<[1,?,23],f16>
    torch.bind_symbolic_shape %420, [%32], affine_map<()[s0] -> (1, s0 * 32, 23)> : !torch.vtensor<[1,?,23],f16>
    %int5_481 = torch.constant.int 5
    %421 = torch.prims.convert_element_type %17, %int5_481 : !torch.vtensor<[23,256],f32>, !torch.int -> !torch.vtensor<[23,256],f16>
    %int-2_482 = torch.constant.int -2
    %int-1_483 = torch.constant.int -1
    %422 = torch.aten.transpose.int %421, %int-2_482, %int-1_483 : !torch.vtensor<[23,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,23],f16>
    %int5_484 = torch.constant.int 5
    %423 = torch.prims.convert_element_type %422, %int5_484 : !torch.vtensor<[256,23],f16>, !torch.int -> !torch.vtensor<[256,23],f16>
    %int256_485 = torch.constant.int 256
    %424 = torch.prim.ListConstruct %38, %int256_485 : (!torch.int, !torch.int) -> !torch.list<int>
    %425 = torch.aten.view %411, %424 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %425, [%32], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %426 = torch.aten.mm %425, %423 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,23],f16> -> !torch.vtensor<[?,23],f16>
    torch.bind_symbolic_shape %426, [%32], affine_map<()[s0] -> (s0 * 32, 23)> : !torch.vtensor<[?,23],f16>
    %int1_486 = torch.constant.int 1
    %int23_487 = torch.constant.int 23
    %427 = torch.prim.ListConstruct %int1_486, %38, %int23_487 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %428 = torch.aten.view %426, %427 : !torch.vtensor<[?,23],f16>, !torch.list<int> -> !torch.vtensor<[1,?,23],f16>
    torch.bind_symbolic_shape %428, [%32], affine_map<()[s0] -> (1, s0 * 32, 23)> : !torch.vtensor<[1,?,23],f16>
    %429 = torch.aten.mul.Tensor %420, %428 : !torch.vtensor<[1,?,23],f16>, !torch.vtensor<[1,?,23],f16> -> !torch.vtensor<[1,?,23],f16>
    torch.bind_symbolic_shape %429, [%32], affine_map<()[s0] -> (1, s0 * 32, 23)> : !torch.vtensor<[1,?,23],f16>
    %int5_488 = torch.constant.int 5
    %430 = torch.prims.convert_element_type %18, %int5_488 : !torch.vtensor<[256,23],f32>, !torch.int -> !torch.vtensor<[256,23],f16>
    %int-2_489 = torch.constant.int -2
    %int-1_490 = torch.constant.int -1
    %431 = torch.aten.transpose.int %430, %int-2_489, %int-1_490 : !torch.vtensor<[256,23],f16>, !torch.int, !torch.int -> !torch.vtensor<[23,256],f16>
    %int5_491 = torch.constant.int 5
    %432 = torch.prims.convert_element_type %431, %int5_491 : !torch.vtensor<[23,256],f16>, !torch.int -> !torch.vtensor<[23,256],f16>
    %int23_492 = torch.constant.int 23
    %433 = torch.prim.ListConstruct %38, %int23_492 : (!torch.int, !torch.int) -> !torch.list<int>
    %434 = torch.aten.view %429, %433 : !torch.vtensor<[1,?,23],f16>, !torch.list<int> -> !torch.vtensor<[?,23],f16>
    torch.bind_symbolic_shape %434, [%32], affine_map<()[s0] -> (s0 * 32, 23)> : !torch.vtensor<[?,23],f16>
    %435 = torch.aten.mm %434, %432 : !torch.vtensor<[?,23],f16>, !torch.vtensor<[23,256],f16> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %435, [%32], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %int1_493 = torch.constant.int 1
    %int256_494 = torch.constant.int 256
    %436 = torch.prim.ListConstruct %int1_493, %38, %int256_494 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %437 = torch.aten.view %435, %436 : !torch.vtensor<[?,256],f16>, !torch.list<int> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %437, [%32], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int1_495 = torch.constant.int 1
    %438 = torch.aten.add.Tensor %401, %437, %int1_495 : !torch.vtensor<[1,?,256],f16>, !torch.vtensor<[1,?,256],f16>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %438, [%32], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int6_496 = torch.constant.int 6
    %439 = torch.prims.convert_element_type %438, %int6_496 : !torch.vtensor<[1,?,256],f16>, !torch.int -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %439, [%32], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int2_497 = torch.constant.int 2
    %440 = torch.aten.pow.Tensor_Scalar %439, %int2_497 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %440, [%32], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int-1_498 = torch.constant.int -1
    %441 = torch.prim.ListConstruct %int-1_498 : (!torch.int) -> !torch.list<int>
    %true_499 = torch.constant.bool true
    %none_500 = torch.constant.none
    %442 = torch.aten.mean.dim %440, %441, %true_499, %none_500 : !torch.vtensor<[1,?,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %442, [%32], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %float1.000000e-02_501 = torch.constant.float 1.000000e-02
    %int1_502 = torch.constant.int 1
    %443 = torch.aten.add.Scalar %442, %float1.000000e-02_501, %int1_502 : !torch.vtensor<[1,?,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %443, [%32], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %444 = torch.aten.rsqrt %443 : !torch.vtensor<[1,?,1],f32> -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %444, [%32], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %445 = torch.aten.mul.Tensor %439, %444 : !torch.vtensor<[1,?,256],f32>, !torch.vtensor<[1,?,1],f32> -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %445, [%32], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int5_503 = torch.constant.int 5
    %446 = torch.prims.convert_element_type %445, %int5_503 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %446, [%32], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %447 = torch.aten.mul.Tensor %19, %446 : !torch.vtensor<[256],f32>, !torch.vtensor<[1,?,256],f16> -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %447, [%32], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int5_504 = torch.constant.int 5
    %448 = torch.prims.convert_element_type %447, %int5_504 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %448, [%32], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int5_505 = torch.constant.int 5
    %449 = torch.prims.convert_element_type %20, %int5_505 : !torch.vtensor<[256,256],f32>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int-2_506 = torch.constant.int -2
    %int-1_507 = torch.constant.int -1
    %450 = torch.aten.transpose.int %449, %int-2_506, %int-1_507 : !torch.vtensor<[256,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,256],f16>
    %int5_508 = torch.constant.int 5
    %451 = torch.prims.convert_element_type %450, %int5_508 : !torch.vtensor<[256,256],f16>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int256_509 = torch.constant.int 256
    %452 = torch.prim.ListConstruct %38, %int256_509 : (!torch.int, !torch.int) -> !torch.list<int>
    %453 = torch.aten.view %448, %452 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %453, [%32], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %454 = torch.aten.mm %453, %451 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,256],f16> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %454, [%32], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %int1_510 = torch.constant.int 1
    %int256_511 = torch.constant.int 256
    %455 = torch.prim.ListConstruct %int1_510, %38, %int256_511 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %456 = torch.aten.view %454, %455 : !torch.vtensor<[?,256],f16>, !torch.list<int> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %456, [%32], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int5_512 = torch.constant.int 5
    %457 = torch.prims.convert_element_type %21, %int5_512 : !torch.vtensor<[128,256],f32>, !torch.int -> !torch.vtensor<[128,256],f16>
    %int-2_513 = torch.constant.int -2
    %int-1_514 = torch.constant.int -1
    %458 = torch.aten.transpose.int %457, %int-2_513, %int-1_514 : !torch.vtensor<[128,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,128],f16>
    %int5_515 = torch.constant.int 5
    %459 = torch.prims.convert_element_type %458, %int5_515 : !torch.vtensor<[256,128],f16>, !torch.int -> !torch.vtensor<[256,128],f16>
    %int256_516 = torch.constant.int 256
    %460 = torch.prim.ListConstruct %38, %int256_516 : (!torch.int, !torch.int) -> !torch.list<int>
    %461 = torch.aten.view %448, %460 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %461, [%32], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %462 = torch.aten.mm %461, %459 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,128],f16> -> !torch.vtensor<[?,128],f16>
    torch.bind_symbolic_shape %462, [%32], affine_map<()[s0] -> (s0 * 32, 128)> : !torch.vtensor<[?,128],f16>
    %int1_517 = torch.constant.int 1
    %int128_518 = torch.constant.int 128
    %463 = torch.prim.ListConstruct %int1_517, %38, %int128_518 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %464 = torch.aten.view %462, %463 : !torch.vtensor<[?,128],f16>, !torch.list<int> -> !torch.vtensor<[1,?,128],f16>
    torch.bind_symbolic_shape %464, [%32], affine_map<()[s0] -> (1, s0 * 32, 128)> : !torch.vtensor<[1,?,128],f16>
    %int5_519 = torch.constant.int 5
    %465 = torch.prims.convert_element_type %22, %int5_519 : !torch.vtensor<[128,256],f32>, !torch.int -> !torch.vtensor<[128,256],f16>
    %int-2_520 = torch.constant.int -2
    %int-1_521 = torch.constant.int -1
    %466 = torch.aten.transpose.int %465, %int-2_520, %int-1_521 : !torch.vtensor<[128,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,128],f16>
    %int5_522 = torch.constant.int 5
    %467 = torch.prims.convert_element_type %466, %int5_522 : !torch.vtensor<[256,128],f16>, !torch.int -> !torch.vtensor<[256,128],f16>
    %int256_523 = torch.constant.int 256
    %468 = torch.prim.ListConstruct %38, %int256_523 : (!torch.int, !torch.int) -> !torch.list<int>
    %469 = torch.aten.view %448, %468 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %469, [%32], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %470 = torch.aten.mm %469, %467 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,128],f16> -> !torch.vtensor<[?,128],f16>
    torch.bind_symbolic_shape %470, [%32], affine_map<()[s0] -> (s0 * 32, 128)> : !torch.vtensor<[?,128],f16>
    %int1_524 = torch.constant.int 1
    %int128_525 = torch.constant.int 128
    %471 = torch.prim.ListConstruct %int1_524, %38, %int128_525 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %472 = torch.aten.view %470, %471 : !torch.vtensor<[?,128],f16>, !torch.list<int> -> !torch.vtensor<[1,?,128],f16>
    torch.bind_symbolic_shape %472, [%32], affine_map<()[s0] -> (1, s0 * 32, 128)> : !torch.vtensor<[1,?,128],f16>
    %int1_526 = torch.constant.int 1
    %int8_527 = torch.constant.int 8
    %int32_528 = torch.constant.int 32
    %473 = torch.prim.ListConstruct %int1_526, %38, %int8_527, %int32_528 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %474 = torch.aten.view %456, %473 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[1,?,8,32],f16>
    torch.bind_symbolic_shape %474, [%32], affine_map<()[s0] -> (1, s0 * 32, 8, 32)> : !torch.vtensor<[1,?,8,32],f16>
    %int1_529 = torch.constant.int 1
    %int4_530 = torch.constant.int 4
    %int32_531 = torch.constant.int 32
    %475 = torch.prim.ListConstruct %int1_529, %38, %int4_530, %int32_531 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %476 = torch.aten.view %464, %475 : !torch.vtensor<[1,?,128],f16>, !torch.list<int> -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %476, [%32], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int1_532 = torch.constant.int 1
    %int4_533 = torch.constant.int 4
    %int32_534 = torch.constant.int 32
    %477 = torch.prim.ListConstruct %int1_532, %38, %int4_533, %int32_534 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %478 = torch.aten.view %472, %477 : !torch.vtensor<[1,?,128],f16>, !torch.list<int> -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %478, [%32], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int128_535 = torch.constant.int 128
    %none_536 = torch.constant.none
    %none_537 = torch.constant.none
    %cpu_538 = torch.constant.device "cpu"
    %false_539 = torch.constant.bool false
    %479 = torch.aten.arange %int128_535, %none_536, %none_537, %cpu_538, %false_539 : !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[128],si64>
    %int0_540 = torch.constant.int 0
    %int32_541 = torch.constant.int 32
    %none_542 = torch.constant.none
    %none_543 = torch.constant.none
    %cpu_544 = torch.constant.device "cpu"
    %false_545 = torch.constant.bool false
    %480 = torch.aten.arange.start %int0_540, %int32_541, %none_542, %none_543, %cpu_544, %false_545 : !torch.int, !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[32],si64>
    %int2_546 = torch.constant.int 2
    %481 = torch.aten.floor_divide.Scalar %480, %int2_546 : !torch.vtensor<[32],si64>, !torch.int -> !torch.vtensor<[32],si64>
    %int6_547 = torch.constant.int 6
    %482 = torch.prims.convert_element_type %481, %int6_547 : !torch.vtensor<[32],si64>, !torch.int -> !torch.vtensor<[32],f32>
    %int32_548 = torch.constant.int 32
    %483 = torch.aten.div.Scalar %482, %int32_548 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[32],f32>
    %float2.000000e00_549 = torch.constant.float 2.000000e+00
    %484 = torch.aten.mul.Scalar %483, %float2.000000e00_549 : !torch.vtensor<[32],f32>, !torch.float -> !torch.vtensor<[32],f32>
    %float5.000000e05_550 = torch.constant.float 5.000000e+05
    %485 = torch.aten.pow.Scalar %float5.000000e05_550, %484 : !torch.float, !torch.vtensor<[32],f32> -> !torch.vtensor<[32],f32>
    %486 = torch.aten.reciprocal %485 : !torch.vtensor<[32],f32> -> !torch.vtensor<[32],f32>
    %float1.000000e00_551 = torch.constant.float 1.000000e+00
    %487 = torch.aten.mul.Scalar %486, %float1.000000e00_551 : !torch.vtensor<[32],f32>, !torch.float -> !torch.vtensor<[32],f32>
    %int1_552 = torch.constant.int 1
    %488 = torch.aten.unsqueeze %479, %int1_552 : !torch.vtensor<[128],si64>, !torch.int -> !torch.vtensor<[128,1],si64>
    %int0_553 = torch.constant.int 0
    %489 = torch.aten.unsqueeze %487, %int0_553 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[1,32],f32>
    %490 = torch.aten.mul.Tensor %488, %489 : !torch.vtensor<[128,1],si64>, !torch.vtensor<[1,32],f32> -> !torch.vtensor<[128,32],f32>
    %int6_554 = torch.constant.int 6
    %491 = torch.prims.convert_element_type %490, %int6_554 : !torch.vtensor<[128,32],f32>, !torch.int -> !torch.vtensor<[128,32],f32>
    %int0_555 = torch.constant.int 0
    %int0_556 = torch.constant.int 0
    %int1_557 = torch.constant.int 1
    %492 = torch.aten.slice.Tensor %491, %int0_555, %int0_556, %38, %int1_557 : !torch.vtensor<[128,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %492, [%32], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int1_558 = torch.constant.int 1
    %int0_559 = torch.constant.int 0
    %int9223372036854775807_560 = torch.constant.int 9223372036854775807
    %int1_561 = torch.constant.int 1
    %493 = torch.aten.slice.Tensor %492, %int1_558, %int0_559, %int9223372036854775807_560, %int1_561 : !torch.vtensor<[?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %493, [%32], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int1_562 = torch.constant.int 1
    %int0_563 = torch.constant.int 0
    %int9223372036854775807_564 = torch.constant.int 9223372036854775807
    %int1_565 = torch.constant.int 1
    %494 = torch.aten.slice.Tensor %493, %int1_562, %int0_563, %int9223372036854775807_564, %int1_565 : !torch.vtensor<[?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %494, [%32], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int0_566 = torch.constant.int 0
    %495 = torch.aten.unsqueeze %494, %int0_566 : !torch.vtensor<[?,32],f32>, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %495, [%32], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int1_567 = torch.constant.int 1
    %int0_568 = torch.constant.int 0
    %int9223372036854775807_569 = torch.constant.int 9223372036854775807
    %int1_570 = torch.constant.int 1
    %496 = torch.aten.slice.Tensor %495, %int1_567, %int0_568, %int9223372036854775807_569, %int1_570 : !torch.vtensor<[1,?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %496, [%32], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int2_571 = torch.constant.int 2
    %int0_572 = torch.constant.int 0
    %int9223372036854775807_573 = torch.constant.int 9223372036854775807
    %int1_574 = torch.constant.int 1
    %497 = torch.aten.slice.Tensor %496, %int2_571, %int0_572, %int9223372036854775807_573, %int1_574 : !torch.vtensor<[1,?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %497, [%32], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int1_575 = torch.constant.int 1
    %int1_576 = torch.constant.int 1
    %int1_577 = torch.constant.int 1
    %498 = torch.prim.ListConstruct %int1_575, %int1_576, %int1_577 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %499 = torch.aten.repeat %497, %498 : !torch.vtensor<[1,?,32],f32>, !torch.list<int> -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %499, [%32], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int6_578 = torch.constant.int 6
    %500 = torch.prims.convert_element_type %474, %int6_578 : !torch.vtensor<[1,?,8,32],f16>, !torch.int -> !torch.vtensor<[1,?,8,32],f32>
    torch.bind_symbolic_shape %500, [%32], affine_map<()[s0] -> (1, s0 * 32, 8, 32)> : !torch.vtensor<[1,?,8,32],f32>
    %501 = torch_c.to_builtin_tensor %500 : !torch.vtensor<[1,?,8,32],f32> -> tensor<1x?x8x32xf32>
    %502 = torch_c.to_builtin_tensor %499 : !torch.vtensor<[1,?,32],f32> -> tensor<1x?x32xf32>
    %503 = util.call @sharktank_rotary_embedding_1_D_8_32_f32(%501, %502) : (tensor<1x?x8x32xf32>, tensor<1x?x32xf32>) -> tensor<1x?x8x32xf32>
    %504 = torch_c.from_builtin_tensor %503 : tensor<1x?x8x32xf32> -> !torch.vtensor<[1,?,8,32],f32>
    torch.bind_symbolic_shape %504, [%32], affine_map<()[s0] -> (1, s0 * 32, 8, 32)> : !torch.vtensor<[1,?,8,32],f32>
    %int5_579 = torch.constant.int 5
    %505 = torch.prims.convert_element_type %504, %int5_579 : !torch.vtensor<[1,?,8,32],f32>, !torch.int -> !torch.vtensor<[1,?,8,32],f16>
    torch.bind_symbolic_shape %505, [%32], affine_map<()[s0] -> (1, s0 * 32, 8, 32)> : !torch.vtensor<[1,?,8,32],f16>
    %int128_580 = torch.constant.int 128
    %none_581 = torch.constant.none
    %none_582 = torch.constant.none
    %cpu_583 = torch.constant.device "cpu"
    %false_584 = torch.constant.bool false
    %506 = torch.aten.arange %int128_580, %none_581, %none_582, %cpu_583, %false_584 : !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[128],si64>
    %int0_585 = torch.constant.int 0
    %int32_586 = torch.constant.int 32
    %none_587 = torch.constant.none
    %none_588 = torch.constant.none
    %cpu_589 = torch.constant.device "cpu"
    %false_590 = torch.constant.bool false
    %507 = torch.aten.arange.start %int0_585, %int32_586, %none_587, %none_588, %cpu_589, %false_590 : !torch.int, !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[32],si64>
    %int2_591 = torch.constant.int 2
    %508 = torch.aten.floor_divide.Scalar %507, %int2_591 : !torch.vtensor<[32],si64>, !torch.int -> !torch.vtensor<[32],si64>
    %int6_592 = torch.constant.int 6
    %509 = torch.prims.convert_element_type %508, %int6_592 : !torch.vtensor<[32],si64>, !torch.int -> !torch.vtensor<[32],f32>
    %int32_593 = torch.constant.int 32
    %510 = torch.aten.div.Scalar %509, %int32_593 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[32],f32>
    %float2.000000e00_594 = torch.constant.float 2.000000e+00
    %511 = torch.aten.mul.Scalar %510, %float2.000000e00_594 : !torch.vtensor<[32],f32>, !torch.float -> !torch.vtensor<[32],f32>
    %float5.000000e05_595 = torch.constant.float 5.000000e+05
    %512 = torch.aten.pow.Scalar %float5.000000e05_595, %511 : !torch.float, !torch.vtensor<[32],f32> -> !torch.vtensor<[32],f32>
    %513 = torch.aten.reciprocal %512 : !torch.vtensor<[32],f32> -> !torch.vtensor<[32],f32>
    %float1.000000e00_596 = torch.constant.float 1.000000e+00
    %514 = torch.aten.mul.Scalar %513, %float1.000000e00_596 : !torch.vtensor<[32],f32>, !torch.float -> !torch.vtensor<[32],f32>
    %int1_597 = torch.constant.int 1
    %515 = torch.aten.unsqueeze %506, %int1_597 : !torch.vtensor<[128],si64>, !torch.int -> !torch.vtensor<[128,1],si64>
    %int0_598 = torch.constant.int 0
    %516 = torch.aten.unsqueeze %514, %int0_598 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[1,32],f32>
    %517 = torch.aten.mul.Tensor %515, %516 : !torch.vtensor<[128,1],si64>, !torch.vtensor<[1,32],f32> -> !torch.vtensor<[128,32],f32>
    %int6_599 = torch.constant.int 6
    %518 = torch.prims.convert_element_type %517, %int6_599 : !torch.vtensor<[128,32],f32>, !torch.int -> !torch.vtensor<[128,32],f32>
    %int0_600 = torch.constant.int 0
    %int0_601 = torch.constant.int 0
    %int1_602 = torch.constant.int 1
    %519 = torch.aten.slice.Tensor %518, %int0_600, %int0_601, %38, %int1_602 : !torch.vtensor<[128,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %519, [%32], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int1_603 = torch.constant.int 1
    %int0_604 = torch.constant.int 0
    %int9223372036854775807_605 = torch.constant.int 9223372036854775807
    %int1_606 = torch.constant.int 1
    %520 = torch.aten.slice.Tensor %519, %int1_603, %int0_604, %int9223372036854775807_605, %int1_606 : !torch.vtensor<[?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %520, [%32], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int1_607 = torch.constant.int 1
    %int0_608 = torch.constant.int 0
    %int9223372036854775807_609 = torch.constant.int 9223372036854775807
    %int1_610 = torch.constant.int 1
    %521 = torch.aten.slice.Tensor %520, %int1_607, %int0_608, %int9223372036854775807_609, %int1_610 : !torch.vtensor<[?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %521, [%32], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int0_611 = torch.constant.int 0
    %522 = torch.aten.unsqueeze %521, %int0_611 : !torch.vtensor<[?,32],f32>, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %522, [%32], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int1_612 = torch.constant.int 1
    %int0_613 = torch.constant.int 0
    %int9223372036854775807_614 = torch.constant.int 9223372036854775807
    %int1_615 = torch.constant.int 1
    %523 = torch.aten.slice.Tensor %522, %int1_612, %int0_613, %int9223372036854775807_614, %int1_615 : !torch.vtensor<[1,?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %523, [%32], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int2_616 = torch.constant.int 2
    %int0_617 = torch.constant.int 0
    %int9223372036854775807_618 = torch.constant.int 9223372036854775807
    %int1_619 = torch.constant.int 1
    %524 = torch.aten.slice.Tensor %523, %int2_616, %int0_617, %int9223372036854775807_618, %int1_619 : !torch.vtensor<[1,?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %524, [%32], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int1_620 = torch.constant.int 1
    %int1_621 = torch.constant.int 1
    %int1_622 = torch.constant.int 1
    %525 = torch.prim.ListConstruct %int1_620, %int1_621, %int1_622 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %526 = torch.aten.repeat %524, %525 : !torch.vtensor<[1,?,32],f32>, !torch.list<int> -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %526, [%32], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int6_623 = torch.constant.int 6
    %527 = torch.prims.convert_element_type %476, %int6_623 : !torch.vtensor<[1,?,4,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,32],f32>
    torch.bind_symbolic_shape %527, [%32], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f32>
    %528 = torch_c.to_builtin_tensor %527 : !torch.vtensor<[1,?,4,32],f32> -> tensor<1x?x4x32xf32>
    %529 = torch_c.to_builtin_tensor %526 : !torch.vtensor<[1,?,32],f32> -> tensor<1x?x32xf32>
    %530 = util.call @sharktank_rotary_embedding_1_D_4_32_f32(%528, %529) : (tensor<1x?x4x32xf32>, tensor<1x?x32xf32>) -> tensor<1x?x4x32xf32>
    %531 = torch_c.from_builtin_tensor %530 : tensor<1x?x4x32xf32> -> !torch.vtensor<[1,?,4,32],f32>
    torch.bind_symbolic_shape %531, [%32], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f32>
    %int5_624 = torch.constant.int 5
    %532 = torch.prims.convert_element_type %531, %int5_624 : !torch.vtensor<[1,?,4,32],f32>, !torch.int -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %532, [%32], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int6_625 = torch.constant.int 6
    %533 = torch.aten.mul.Scalar %arg2, %int6_625 : !torch.vtensor<[1,?],si64>, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %533, [%32], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %int4_626 = torch.constant.int 4
    %int1_627 = torch.constant.int 1
    %534 = torch.aten.add.Scalar %533, %int4_626, %int1_627 : !torch.vtensor<[1,?],si64>, !torch.int, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %534, [%32], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %int1_628 = torch.constant.int 1
    %int32_629 = torch.constant.int 32
    %int4_630 = torch.constant.int 4
    %int32_631 = torch.constant.int 32
    %535 = torch.prim.ListConstruct %int1_628, %34, %int32_629, %int4_630, %int32_631 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %536 = torch.aten.view %532, %535 : !torch.vtensor<[1,?,4,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,32,4,32],f16>
    torch.bind_symbolic_shape %536, [%32], affine_map<()[s0] -> (1, s0, 32, 4, 32)> : !torch.vtensor<[1,?,32,4,32],f16>
    %int32_632 = torch.constant.int 32
    %int4_633 = torch.constant.int 4
    %int32_634 = torch.constant.int 32
    %537 = torch.prim.ListConstruct %34, %int32_632, %int4_633, %int32_634 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %538 = torch.aten.view %536, %537 : !torch.vtensor<[1,?,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,32,4,32],f16>
    torch.bind_symbolic_shape %538, [%32], affine_map<()[s0] -> (s0, 32, 4, 32)> : !torch.vtensor<[?,32,4,32],f16>
    %539 = torch.prim.ListConstruct %34 : (!torch.int) -> !torch.list<int>
    %540 = torch.aten.view %534, %539 : !torch.vtensor<[1,?],si64>, !torch.list<int> -> !torch.vtensor<[?],si64>
    torch.bind_symbolic_shape %540, [%32], affine_map<()[s0] -> (s0)> : !torch.vtensor<[?],si64>
    %int5_635 = torch.constant.int 5
    %541 = torch.prims.convert_element_type %538, %int5_635 : !torch.vtensor<[?,32,4,32],f16>, !torch.int -> !torch.vtensor<[?,32,4,32],f16>
    torch.bind_symbolic_shape %541, [%32], affine_map<()[s0] -> (s0, 32, 4, 32)> : !torch.vtensor<[?,32,4,32],f16>
    %int3_636 = torch.constant.int 3
    %int2_637 = torch.constant.int 2
    %int32_638 = torch.constant.int 32
    %int4_639 = torch.constant.int 4
    %int32_640 = torch.constant.int 32
    %542 = torch.prim.ListConstruct %35, %int3_636, %int2_637, %int32_638, %int4_639, %int32_640 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %543 = torch.aten.view %370, %542 : !torch.vtensor<[?,24576],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,4,32],f16>
    torch.bind_symbolic_shape %543, [%33], affine_map<()[s0] -> (s0, 3, 2, 32, 4, 32)> : !torch.vtensor<[?,3,2,32,4,32],f16>
    %int32_641 = torch.constant.int 32
    %int4_642 = torch.constant.int 4
    %int32_643 = torch.constant.int 32
    %544 = torch.prim.ListConstruct %136, %int32_641, %int4_642, %int32_643 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %545 = torch.aten.view %543, %544 : !torch.vtensor<[?,3,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,32,4,32],f16>
    torch.bind_symbolic_shape %545, [%33], affine_map<()[s0] -> (s0 * 6, 32, 4, 32)> : !torch.vtensor<[?,32,4,32],f16>
    %546 = torch.prim.ListConstruct %540 : (!torch.vtensor<[?],si64>) -> !torch.list<optional<vtensor>>
    %false_644 = torch.constant.bool false
    %547 = torch.aten.index_put %545, %546, %541, %false_644 : !torch.vtensor<[?,32,4,32],f16>, !torch.list<optional<vtensor>>, !torch.vtensor<[?,32,4,32],f16>, !torch.bool -> !torch.vtensor<[?,32,4,32],f16>
    torch.bind_symbolic_shape %547, [%33], affine_map<()[s0] -> (s0 * 6, 32, 4, 32)> : !torch.vtensor<[?,32,4,32],f16>
    %int3_645 = torch.constant.int 3
    %int2_646 = torch.constant.int 2
    %int32_647 = torch.constant.int 32
    %int4_648 = torch.constant.int 4
    %int32_649 = torch.constant.int 32
    %548 = torch.prim.ListConstruct %35, %int3_645, %int2_646, %int32_647, %int4_648, %int32_649 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %549 = torch.aten.view %547, %548 : !torch.vtensor<[?,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,4,32],f16>
    torch.bind_symbolic_shape %549, [%33], affine_map<()[s0] -> (s0, 3, 2, 32, 4, 32)> : !torch.vtensor<[?,3,2,32,4,32],f16>
    %int24576_650 = torch.constant.int 24576
    %550 = torch.prim.ListConstruct %35, %int24576_650 : (!torch.int, !torch.int) -> !torch.list<int>
    %551 = torch.aten.view %549, %550 : !torch.vtensor<[?,3,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,24576],f16>
    torch.bind_symbolic_shape %551, [%33], affine_map<()[s0] -> (s0, 24576)> : !torch.vtensor<[?,24576],f16>
    %int3_651 = torch.constant.int 3
    %int2_652 = torch.constant.int 2
    %int32_653 = torch.constant.int 32
    %int4_654 = torch.constant.int 4
    %int32_655 = torch.constant.int 32
    %552 = torch.prim.ListConstruct %35, %int3_651, %int2_652, %int32_653, %int4_654, %int32_655 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %553 = torch.aten.view %551, %552 : !torch.vtensor<[?,24576],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,4,32],f16>
    torch.bind_symbolic_shape %553, [%33], affine_map<()[s0] -> (s0, 3, 2, 32, 4, 32)> : !torch.vtensor<[?,3,2,32,4,32],f16>
    %int32_656 = torch.constant.int 32
    %int4_657 = torch.constant.int 4
    %int32_658 = torch.constant.int 32
    %554 = torch.prim.ListConstruct %136, %int32_656, %int4_657, %int32_658 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %555 = torch.aten.view %553, %554 : !torch.vtensor<[?,3,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,32,4,32],f16>
    torch.bind_symbolic_shape %555, [%33], affine_map<()[s0] -> (s0 * 6, 32, 4, 32)> : !torch.vtensor<[?,32,4,32],f16>
    %int1_659 = torch.constant.int 1
    %int32_660 = torch.constant.int 32
    %int4_661 = torch.constant.int 4
    %int32_662 = torch.constant.int 32
    %556 = torch.prim.ListConstruct %int1_659, %34, %int32_660, %int4_661, %int32_662 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %557 = torch.aten.view %478, %556 : !torch.vtensor<[1,?,4,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,32,4,32],f16>
    torch.bind_symbolic_shape %557, [%32], affine_map<()[s0] -> (1, s0, 32, 4, 32)> : !torch.vtensor<[1,?,32,4,32],f16>
    %int32_663 = torch.constant.int 32
    %int4_664 = torch.constant.int 4
    %int32_665 = torch.constant.int 32
    %558 = torch.prim.ListConstruct %34, %int32_663, %int4_664, %int32_665 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %559 = torch.aten.view %557, %558 : !torch.vtensor<[1,?,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,32,4,32],f16>
    torch.bind_symbolic_shape %559, [%32], affine_map<()[s0] -> (s0, 32, 4, 32)> : !torch.vtensor<[?,32,4,32],f16>
    %int1_666 = torch.constant.int 1
    %int1_667 = torch.constant.int 1
    %560 = torch.aten.add.Scalar %534, %int1_666, %int1_667 : !torch.vtensor<[1,?],si64>, !torch.int, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %560, [%32], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %561 = torch.prim.ListConstruct %34 : (!torch.int) -> !torch.list<int>
    %562 = torch.aten.view %560, %561 : !torch.vtensor<[1,?],si64>, !torch.list<int> -> !torch.vtensor<[?],si64>
    torch.bind_symbolic_shape %562, [%32], affine_map<()[s0] -> (s0)> : !torch.vtensor<[?],si64>
    %int5_668 = torch.constant.int 5
    %563 = torch.prims.convert_element_type %559, %int5_668 : !torch.vtensor<[?,32,4,32],f16>, !torch.int -> !torch.vtensor<[?,32,4,32],f16>
    torch.bind_symbolic_shape %563, [%32], affine_map<()[s0] -> (s0, 32, 4, 32)> : !torch.vtensor<[?,32,4,32],f16>
    %564 = torch.prim.ListConstruct %562 : (!torch.vtensor<[?],si64>) -> !torch.list<optional<vtensor>>
    %false_669 = torch.constant.bool false
    %565 = torch.aten.index_put %555, %564, %563, %false_669 : !torch.vtensor<[?,32,4,32],f16>, !torch.list<optional<vtensor>>, !torch.vtensor<[?,32,4,32],f16>, !torch.bool -> !torch.vtensor<[?,32,4,32],f16>
    torch.bind_symbolic_shape %565, [%33], affine_map<()[s0] -> (s0 * 6, 32, 4, 32)> : !torch.vtensor<[?,32,4,32],f16>
    %int3_670 = torch.constant.int 3
    %int2_671 = torch.constant.int 2
    %int32_672 = torch.constant.int 32
    %int4_673 = torch.constant.int 4
    %int32_674 = torch.constant.int 32
    %566 = torch.prim.ListConstruct %35, %int3_670, %int2_671, %int32_672, %int4_673, %int32_674 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %567 = torch.aten.view %565, %566 : !torch.vtensor<[?,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,4,32],f16>
    torch.bind_symbolic_shape %567, [%33], affine_map<()[s0] -> (s0, 3, 2, 32, 4, 32)> : !torch.vtensor<[?,3,2,32,4,32],f16>
    %int24576_675 = torch.constant.int 24576
    %568 = torch.prim.ListConstruct %35, %int24576_675 : (!torch.int, !torch.int) -> !torch.list<int>
    %569 = torch.aten.view %567, %568 : !torch.vtensor<[?,3,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,24576],f16>
    torch.overwrite.tensor.contents %569 overwrites %arg3 : !torch.vtensor<[?,24576],f16>, !torch.tensor<[?,24576],f16>
    torch.bind_symbolic_shape %569, [%33], affine_map<()[s0] -> (s0, 24576)> : !torch.vtensor<[?,24576],f16>
    %int-2_676 = torch.constant.int -2
    %570 = torch.aten.unsqueeze %532, %int-2_676 : !torch.vtensor<[1,?,4,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,1,32],f16>
    torch.bind_symbolic_shape %570, [%32], affine_map<()[s0] -> (1, s0 * 32, 4, 1, 32)> : !torch.vtensor<[1,?,4,1,32],f16>
    %int1_677 = torch.constant.int 1
    %int4_678 = torch.constant.int 4
    %int2_679 = torch.constant.int 2
    %int32_680 = torch.constant.int 32
    %571 = torch.prim.ListConstruct %int1_677, %38, %int4_678, %int2_679, %int32_680 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_681 = torch.constant.bool false
    %572 = torch.aten.expand %570, %571, %false_681 : !torch.vtensor<[1,?,4,1,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,?,4,2,32],f16>
    torch.bind_symbolic_shape %572, [%32], affine_map<()[s0] -> (1, s0 * 32, 4, 2, 32)> : !torch.vtensor<[1,?,4,2,32],f16>
    %int0_682 = torch.constant.int 0
    %573 = torch.aten.clone %572, %int0_682 : !torch.vtensor<[1,?,4,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,2,32],f16>
    torch.bind_symbolic_shape %573, [%32], affine_map<()[s0] -> (1, s0 * 32, 4, 2, 32)> : !torch.vtensor<[1,?,4,2,32],f16>
    %int1_683 = torch.constant.int 1
    %int8_684 = torch.constant.int 8
    %int32_685 = torch.constant.int 32
    %574 = torch.prim.ListConstruct %int1_683, %38, %int8_684, %int32_685 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %575 = torch.aten._unsafe_view %573, %574 : !torch.vtensor<[1,?,4,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,8,32],f16>
    torch.bind_symbolic_shape %575, [%32], affine_map<()[s0] -> (1, s0 * 32, 8, 32)> : !torch.vtensor<[1,?,8,32],f16>
    %int-2_686 = torch.constant.int -2
    %576 = torch.aten.unsqueeze %478, %int-2_686 : !torch.vtensor<[1,?,4,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,1,32],f16>
    torch.bind_symbolic_shape %576, [%32], affine_map<()[s0] -> (1, s0 * 32, 4, 1, 32)> : !torch.vtensor<[1,?,4,1,32],f16>
    %int1_687 = torch.constant.int 1
    %int4_688 = torch.constant.int 4
    %int2_689 = torch.constant.int 2
    %int32_690 = torch.constant.int 32
    %577 = torch.prim.ListConstruct %int1_687, %38, %int4_688, %int2_689, %int32_690 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_691 = torch.constant.bool false
    %578 = torch.aten.expand %576, %577, %false_691 : !torch.vtensor<[1,?,4,1,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,?,4,2,32],f16>
    torch.bind_symbolic_shape %578, [%32], affine_map<()[s0] -> (1, s0 * 32, 4, 2, 32)> : !torch.vtensor<[1,?,4,2,32],f16>
    %int0_692 = torch.constant.int 0
    %579 = torch.aten.clone %578, %int0_692 : !torch.vtensor<[1,?,4,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,2,32],f16>
    torch.bind_symbolic_shape %579, [%32], affine_map<()[s0] -> (1, s0 * 32, 4, 2, 32)> : !torch.vtensor<[1,?,4,2,32],f16>
    %int1_693 = torch.constant.int 1
    %int8_694 = torch.constant.int 8
    %int32_695 = torch.constant.int 32
    %580 = torch.prim.ListConstruct %int1_693, %38, %int8_694, %int32_695 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %581 = torch.aten._unsafe_view %579, %580 : !torch.vtensor<[1,?,4,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,8,32],f16>
    torch.bind_symbolic_shape %581, [%32], affine_map<()[s0] -> (1, s0 * 32, 8, 32)> : !torch.vtensor<[1,?,8,32],f16>
    %int1_696 = torch.constant.int 1
    %int2_697 = torch.constant.int 2
    %582 = torch.aten.transpose.int %505, %int1_696, %int2_697 : !torch.vtensor<[1,?,8,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,8,?,32],f16>
    torch.bind_symbolic_shape %582, [%32], affine_map<()[s0] -> (1, 8, s0 * 32, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %int1_698 = torch.constant.int 1
    %int2_699 = torch.constant.int 2
    %583 = torch.aten.transpose.int %575, %int1_698, %int2_699 : !torch.vtensor<[1,?,8,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,8,?,32],f16>
    torch.bind_symbolic_shape %583, [%32], affine_map<()[s0] -> (1, 8, s0 * 32, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %int1_700 = torch.constant.int 1
    %int2_701 = torch.constant.int 2
    %584 = torch.aten.transpose.int %581, %int1_700, %int2_701 : !torch.vtensor<[1,?,8,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,8,?,32],f16>
    torch.bind_symbolic_shape %584, [%32], affine_map<()[s0] -> (1, 8, s0 * 32, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %int5_702 = torch.constant.int 5
    %585 = torch.prims.convert_element_type %582, %int5_702 : !torch.vtensor<[1,8,?,32],f16>, !torch.int -> !torch.vtensor<[1,8,?,32],f16>
    torch.bind_symbolic_shape %585, [%32], affine_map<()[s0] -> (1, 8, s0 * 32, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %int5_703 = torch.constant.int 5
    %586 = torch.prims.convert_element_type %583, %int5_703 : !torch.vtensor<[1,8,?,32],f16>, !torch.int -> !torch.vtensor<[1,8,?,32],f16>
    torch.bind_symbolic_shape %586, [%32], affine_map<()[s0] -> (1, 8, s0 * 32, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %int5_704 = torch.constant.int 5
    %587 = torch.prims.convert_element_type %584, %int5_704 : !torch.vtensor<[1,8,?,32],f16>, !torch.int -> !torch.vtensor<[1,8,?,32],f16>
    torch.bind_symbolic_shape %587, [%32], affine_map<()[s0] -> (1, 8, s0 * 32, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %float0.000000e00_705 = torch.constant.float 0.000000e+00
    %true_706 = torch.constant.bool true
    %none_707 = torch.constant.none
    %none_708 = torch.constant.none
    %588:2 = torch.operator "torch.aten._scaled_dot_product_flash_attention_for_cpu"(%585, %586, %587, %float0.000000e00_705, %true_706, %none_707, %none_708) : (!torch.vtensor<[1,8,?,32],f16>, !torch.vtensor<[1,8,?,32],f16>, !torch.vtensor<[1,8,?,32],f16>, !torch.float, !torch.bool, !torch.none, !torch.none) -> (!torch.vtensor<[1,8,?,32],f16>, !torch.vtensor<[1,8,?],f32>) 
    torch.bind_symbolic_shape %588#0, [%32], affine_map<()[s0] -> (1, 8, s0 * 32, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %int1_709 = torch.constant.int 1
    %int2_710 = torch.constant.int 2
    %589 = torch.aten.transpose.int %588#0, %int1_709, %int2_710 : !torch.vtensor<[1,8,?,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,?,8,32],f16>
    torch.bind_symbolic_shape %589, [%32], affine_map<()[s0] -> (1, s0 * 32, 8, 32)> : !torch.vtensor<[1,?,8,32],f16>
    %int1_711 = torch.constant.int 1
    %int256_712 = torch.constant.int 256
    %590 = torch.prim.ListConstruct %int1_711, %38, %int256_712 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %591 = torch.aten.view %589, %590 : !torch.vtensor<[1,?,8,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %591, [%32], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int5_713 = torch.constant.int 5
    %592 = torch.prims.convert_element_type %23, %int5_713 : !torch.vtensor<[256,256],f32>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int-2_714 = torch.constant.int -2
    %int-1_715 = torch.constant.int -1
    %593 = torch.aten.transpose.int %592, %int-2_714, %int-1_715 : !torch.vtensor<[256,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,256],f16>
    %int5_716 = torch.constant.int 5
    %594 = torch.prims.convert_element_type %593, %int5_716 : !torch.vtensor<[256,256],f16>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int256_717 = torch.constant.int 256
    %595 = torch.prim.ListConstruct %38, %int256_717 : (!torch.int, !torch.int) -> !torch.list<int>
    %596 = torch.aten.view %591, %595 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %596, [%32], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %597 = torch.aten.mm %596, %594 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,256],f16> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %597, [%32], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %int1_718 = torch.constant.int 1
    %int256_719 = torch.constant.int 256
    %598 = torch.prim.ListConstruct %int1_718, %38, %int256_719 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %599 = torch.aten.view %597, %598 : !torch.vtensor<[?,256],f16>, !torch.list<int> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %599, [%32], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int1_720 = torch.constant.int 1
    %600 = torch.aten.add.Tensor %438, %599, %int1_720 : !torch.vtensor<[1,?,256],f16>, !torch.vtensor<[1,?,256],f16>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %600, [%32], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int6_721 = torch.constant.int 6
    %601 = torch.prims.convert_element_type %600, %int6_721 : !torch.vtensor<[1,?,256],f16>, !torch.int -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %601, [%32], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int2_722 = torch.constant.int 2
    %602 = torch.aten.pow.Tensor_Scalar %601, %int2_722 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %602, [%32], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int-1_723 = torch.constant.int -1
    %603 = torch.prim.ListConstruct %int-1_723 : (!torch.int) -> !torch.list<int>
    %true_724 = torch.constant.bool true
    %none_725 = torch.constant.none
    %604 = torch.aten.mean.dim %602, %603, %true_724, %none_725 : !torch.vtensor<[1,?,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %604, [%32], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %float1.000000e-02_726 = torch.constant.float 1.000000e-02
    %int1_727 = torch.constant.int 1
    %605 = torch.aten.add.Scalar %604, %float1.000000e-02_726, %int1_727 : !torch.vtensor<[1,?,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %605, [%32], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %606 = torch.aten.rsqrt %605 : !torch.vtensor<[1,?,1],f32> -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %606, [%32], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %607 = torch.aten.mul.Tensor %601, %606 : !torch.vtensor<[1,?,256],f32>, !torch.vtensor<[1,?,1],f32> -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %607, [%32], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int5_728 = torch.constant.int 5
    %608 = torch.prims.convert_element_type %607, %int5_728 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %608, [%32], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %609 = torch.aten.mul.Tensor %24, %608 : !torch.vtensor<[256],f32>, !torch.vtensor<[1,?,256],f16> -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %609, [%32], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int5_729 = torch.constant.int 5
    %610 = torch.prims.convert_element_type %609, %int5_729 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %610, [%32], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int5_730 = torch.constant.int 5
    %611 = torch.prims.convert_element_type %25, %int5_730 : !torch.vtensor<[23,256],f32>, !torch.int -> !torch.vtensor<[23,256],f16>
    %int-2_731 = torch.constant.int -2
    %int-1_732 = torch.constant.int -1
    %612 = torch.aten.transpose.int %611, %int-2_731, %int-1_732 : !torch.vtensor<[23,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,23],f16>
    %int5_733 = torch.constant.int 5
    %613 = torch.prims.convert_element_type %612, %int5_733 : !torch.vtensor<[256,23],f16>, !torch.int -> !torch.vtensor<[256,23],f16>
    %int256_734 = torch.constant.int 256
    %614 = torch.prim.ListConstruct %38, %int256_734 : (!torch.int, !torch.int) -> !torch.list<int>
    %615 = torch.aten.view %610, %614 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %615, [%32], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %616 = torch.aten.mm %615, %613 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,23],f16> -> !torch.vtensor<[?,23],f16>
    torch.bind_symbolic_shape %616, [%32], affine_map<()[s0] -> (s0 * 32, 23)> : !torch.vtensor<[?,23],f16>
    %int1_735 = torch.constant.int 1
    %int23_736 = torch.constant.int 23
    %617 = torch.prim.ListConstruct %int1_735, %38, %int23_736 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %618 = torch.aten.view %616, %617 : !torch.vtensor<[?,23],f16>, !torch.list<int> -> !torch.vtensor<[1,?,23],f16>
    torch.bind_symbolic_shape %618, [%32], affine_map<()[s0] -> (1, s0 * 32, 23)> : !torch.vtensor<[1,?,23],f16>
    %619 = torch.aten.silu %618 : !torch.vtensor<[1,?,23],f16> -> !torch.vtensor<[1,?,23],f16>
    torch.bind_symbolic_shape %619, [%32], affine_map<()[s0] -> (1, s0 * 32, 23)> : !torch.vtensor<[1,?,23],f16>
    %int5_737 = torch.constant.int 5
    %620 = torch.prims.convert_element_type %26, %int5_737 : !torch.vtensor<[23,256],f32>, !torch.int -> !torch.vtensor<[23,256],f16>
    %int-2_738 = torch.constant.int -2
    %int-1_739 = torch.constant.int -1
    %621 = torch.aten.transpose.int %620, %int-2_738, %int-1_739 : !torch.vtensor<[23,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,23],f16>
    %int5_740 = torch.constant.int 5
    %622 = torch.prims.convert_element_type %621, %int5_740 : !torch.vtensor<[256,23],f16>, !torch.int -> !torch.vtensor<[256,23],f16>
    %int256_741 = torch.constant.int 256
    %623 = torch.prim.ListConstruct %38, %int256_741 : (!torch.int, !torch.int) -> !torch.list<int>
    %624 = torch.aten.view %610, %623 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %624, [%32], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %625 = torch.aten.mm %624, %622 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,23],f16> -> !torch.vtensor<[?,23],f16>
    torch.bind_symbolic_shape %625, [%32], affine_map<()[s0] -> (s0 * 32, 23)> : !torch.vtensor<[?,23],f16>
    %int1_742 = torch.constant.int 1
    %int23_743 = torch.constant.int 23
    %626 = torch.prim.ListConstruct %int1_742, %38, %int23_743 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %627 = torch.aten.view %625, %626 : !torch.vtensor<[?,23],f16>, !torch.list<int> -> !torch.vtensor<[1,?,23],f16>
    torch.bind_symbolic_shape %627, [%32], affine_map<()[s0] -> (1, s0 * 32, 23)> : !torch.vtensor<[1,?,23],f16>
    %628 = torch.aten.mul.Tensor %619, %627 : !torch.vtensor<[1,?,23],f16>, !torch.vtensor<[1,?,23],f16> -> !torch.vtensor<[1,?,23],f16>
    torch.bind_symbolic_shape %628, [%32], affine_map<()[s0] -> (1, s0 * 32, 23)> : !torch.vtensor<[1,?,23],f16>
    %int5_744 = torch.constant.int 5
    %629 = torch.prims.convert_element_type %27, %int5_744 : !torch.vtensor<[256,23],f32>, !torch.int -> !torch.vtensor<[256,23],f16>
    %int-2_745 = torch.constant.int -2
    %int-1_746 = torch.constant.int -1
    %630 = torch.aten.transpose.int %629, %int-2_745, %int-1_746 : !torch.vtensor<[256,23],f16>, !torch.int, !torch.int -> !torch.vtensor<[23,256],f16>
    %int5_747 = torch.constant.int 5
    %631 = torch.prims.convert_element_type %630, %int5_747 : !torch.vtensor<[23,256],f16>, !torch.int -> !torch.vtensor<[23,256],f16>
    %int23_748 = torch.constant.int 23
    %632 = torch.prim.ListConstruct %38, %int23_748 : (!torch.int, !torch.int) -> !torch.list<int>
    %633 = torch.aten.view %628, %632 : !torch.vtensor<[1,?,23],f16>, !torch.list<int> -> !torch.vtensor<[?,23],f16>
    torch.bind_symbolic_shape %633, [%32], affine_map<()[s0] -> (s0 * 32, 23)> : !torch.vtensor<[?,23],f16>
    %634 = torch.aten.mm %633, %631 : !torch.vtensor<[?,23],f16>, !torch.vtensor<[23,256],f16> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %634, [%32], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %int1_749 = torch.constant.int 1
    %int256_750 = torch.constant.int 256
    %635 = torch.prim.ListConstruct %int1_749, %38, %int256_750 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %636 = torch.aten.view %634, %635 : !torch.vtensor<[?,256],f16>, !torch.list<int> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %636, [%32], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int1_751 = torch.constant.int 1
    %637 = torch.aten.add.Tensor %600, %636, %int1_751 : !torch.vtensor<[1,?,256],f16>, !torch.vtensor<[1,?,256],f16>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %637, [%32], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int6_752 = torch.constant.int 6
    %638 = torch.prims.convert_element_type %637, %int6_752 : !torch.vtensor<[1,?,256],f16>, !torch.int -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %638, [%32], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int2_753 = torch.constant.int 2
    %639 = torch.aten.pow.Tensor_Scalar %638, %int2_753 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %639, [%32], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int-1_754 = torch.constant.int -1
    %640 = torch.prim.ListConstruct %int-1_754 : (!torch.int) -> !torch.list<int>
    %true_755 = torch.constant.bool true
    %none_756 = torch.constant.none
    %641 = torch.aten.mean.dim %639, %640, %true_755, %none_756 : !torch.vtensor<[1,?,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %641, [%32], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %float1.000000e-02_757 = torch.constant.float 1.000000e-02
    %int1_758 = torch.constant.int 1
    %642 = torch.aten.add.Scalar %641, %float1.000000e-02_757, %int1_758 : !torch.vtensor<[1,?,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %642, [%32], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %643 = torch.aten.rsqrt %642 : !torch.vtensor<[1,?,1],f32> -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %643, [%32], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %644 = torch.aten.mul.Tensor %638, %643 : !torch.vtensor<[1,?,256],f32>, !torch.vtensor<[1,?,1],f32> -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %644, [%32], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int5_759 = torch.constant.int 5
    %645 = torch.prims.convert_element_type %644, %int5_759 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %645, [%32], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %646 = torch.aten.mul.Tensor %28, %645 : !torch.vtensor<[1,256],f32>, !torch.vtensor<[1,?,256],f16> -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %646, [%32], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int5_760 = torch.constant.int 5
    %647 = torch.prims.convert_element_type %646, %int5_760 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %647, [%32], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int5_761 = torch.constant.int 5
    %648 = torch.prims.convert_element_type %29, %int5_761 : !torch.vtensor<[256,256],f32>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int-2_762 = torch.constant.int -2
    %int-1_763 = torch.constant.int -1
    %649 = torch.aten.transpose.int %648, %int-2_762, %int-1_763 : !torch.vtensor<[256,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,256],f16>
    %int5_764 = torch.constant.int 5
    %650 = torch.prims.convert_element_type %649, %int5_764 : !torch.vtensor<[256,256],f16>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int256_765 = torch.constant.int 256
    %651 = torch.prim.ListConstruct %38, %int256_765 : (!torch.int, !torch.int) -> !torch.list<int>
    %652 = torch.aten.view %647, %651 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %652, [%32], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %653 = torch.aten.mm %652, %650 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,256],f16> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %653, [%32], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %int1_766 = torch.constant.int 1
    %int256_767 = torch.constant.int 256
    %654 = torch.prim.ListConstruct %int1_766, %38, %int256_767 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %655 = torch.aten.view %653, %654 : !torch.vtensor<[?,256],f16>, !torch.list<int> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %655, [%32], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    return %655 : !torch.vtensor<[1,?,256],f16>
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
    %38 = torch.symbolic_int "s1" {min_val = 0, max_val = 9223372036854775807} : !torch.int
    torch.bind_symbolic_shape %arg3, [%37], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %36, [%38], affine_map<()[s0] -> (s0, 24576)> : !torch.vtensor<[?,24576],f16>
    %int1 = torch.constant.int 1
    %39 = torch.aten.size.int %arg3, %int1 : !torch.vtensor<[1,?],si64>, !torch.int -> !torch.int
    %int0 = torch.constant.int 0
    %40 = torch.aten.size.int %36, %int0 : !torch.vtensor<[?,24576],f16>, !torch.int -> !torch.int
    %int32 = torch.constant.int 32
    %41 = torch.aten.mul.int %39, %int32 : !torch.int, !torch.int -> !torch.int
    %int0_0 = torch.constant.int 0
    %int1_1 = torch.constant.int 1
    %none = torch.constant.none
    %none_2 = torch.constant.none
    %cpu = torch.constant.device "cpu"
    %false = torch.constant.bool false
    %42 = torch.aten.arange.start_step %int0_0, %41, %int1_1, %none, %none_2, %cpu, %false : !torch.int, !torch.int, !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[?],si64>
    torch.bind_symbolic_shape %42, [%37], affine_map<()[s0] -> (s0 * 32)> : !torch.vtensor<[?],si64>
    %int-1 = torch.constant.int -1
    %43 = torch.aten.unsqueeze %arg1, %int-1 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %44 = torch.aten.ge.Tensor %42, %43 : !torch.vtensor<[?],si64>, !torch.vtensor<[1,1],si64> -> !torch.vtensor<[1,?],i1>
    torch.bind_symbolic_shape %44, [%37], affine_map<()[s0] -> (1, s0 * 32)> : !torch.vtensor<[1,?],i1>
    %int0_3 = torch.constant.int 0
    %int6 = torch.constant.int 6
    %int0_4 = torch.constant.int 0
    %cpu_5 = torch.constant.device "cpu"
    %none_6 = torch.constant.none
    %45 = torch.aten.scalar_tensor %int0_3, %int6, %int0_4, %cpu_5, %none_6 : !torch.int, !torch.int, !torch.int, !torch.Device, !torch.none -> !torch.vtensor<[],f32>
    %float-Inf = torch.constant.float 0xFFF0000000000000
    %int6_7 = torch.constant.int 6
    %int0_8 = torch.constant.int 0
    %cpu_9 = torch.constant.device "cpu"
    %none_10 = torch.constant.none
    %46 = torch.aten.scalar_tensor %float-Inf, %int6_7, %int0_8, %cpu_9, %none_10 : !torch.float, !torch.int, !torch.int, !torch.Device, !torch.none -> !torch.vtensor<[],f32>
    %47 = torch.aten.where.self %44, %46, %45 : !torch.vtensor<[1,?],i1>, !torch.vtensor<[],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[1,?],f32>
    torch.bind_symbolic_shape %47, [%37], affine_map<()[s0] -> (1, s0 * 32)> : !torch.vtensor<[1,?],f32>
    %int5 = torch.constant.int 5
    %48 = torch.prims.convert_element_type %47, %int5 : !torch.vtensor<[1,?],f32>, !torch.int -> !torch.vtensor<[1,?],f16>
    torch.bind_symbolic_shape %48, [%37], affine_map<()[s0] -> (1, s0 * 32)> : !torch.vtensor<[1,?],f16>
    %int1_11 = torch.constant.int 1
    %49 = torch.aten.unsqueeze %48, %int1_11 : !torch.vtensor<[1,?],f16>, !torch.int -> !torch.vtensor<[1,1,?],f16>
    torch.bind_symbolic_shape %49, [%37], affine_map<()[s0] -> (1, 1, s0 * 32)> : !torch.vtensor<[1,1,?],f16>
    %int1_12 = torch.constant.int 1
    %50 = torch.aten.unsqueeze %49, %int1_12 : !torch.vtensor<[1,1,?],f16>, !torch.int -> !torch.vtensor<[1,1,1,?],f16>
    torch.bind_symbolic_shape %50, [%37], affine_map<()[s0] -> (1, 1, 1, s0 * 32)> : !torch.vtensor<[1,1,1,?],f16>
    %int5_13 = torch.constant.int 5
    %51 = torch.prims.convert_element_type %50, %int5_13 : !torch.vtensor<[1,1,1,?],f16>, !torch.int -> !torch.vtensor<[1,1,1,?],f16>
    torch.bind_symbolic_shape %51, [%37], affine_map<()[s0] -> (1, 1, 1, s0 * 32)> : !torch.vtensor<[1,1,1,?],f16>
    %int0_14 = torch.constant.int 0
    %int1_15 = torch.constant.int 1
    %none_16 = torch.constant.none
    %none_17 = torch.constant.none
    %cpu_18 = torch.constant.device "cpu"
    %false_19 = torch.constant.bool false
    %52 = torch.aten.arange.start %int0_14, %int1_15, %none_16, %none_17, %cpu_18, %false_19 : !torch.int, !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[1],si64>
    %int0_20 = torch.constant.int 0
    %53 = torch.aten.unsqueeze %52, %int0_20 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_21 = torch.constant.int 1
    %54 = torch.aten.unsqueeze %arg2, %int1_21 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_22 = torch.constant.int 1
    %55 = torch.aten.add.Tensor %53, %54, %int1_22 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int128 = torch.constant.int 128
    %none_23 = torch.constant.none
    %none_24 = torch.constant.none
    %cpu_25 = torch.constant.device "cpu"
    %false_26 = torch.constant.bool false
    %56 = torch.aten.arange %int128, %none_23, %none_24, %cpu_25, %false_26 : !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[128],si64>
    %int0_27 = torch.constant.int 0
    %int32_28 = torch.constant.int 32
    %none_29 = torch.constant.none
    %none_30 = torch.constant.none
    %cpu_31 = torch.constant.device "cpu"
    %false_32 = torch.constant.bool false
    %57 = torch.aten.arange.start %int0_27, %int32_28, %none_29, %none_30, %cpu_31, %false_32 : !torch.int, !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[32],si64>
    %int2 = torch.constant.int 2
    %58 = torch.aten.floor_divide.Scalar %57, %int2 : !torch.vtensor<[32],si64>, !torch.int -> !torch.vtensor<[32],si64>
    %int6_33 = torch.constant.int 6
    %59 = torch.prims.convert_element_type %58, %int6_33 : !torch.vtensor<[32],si64>, !torch.int -> !torch.vtensor<[32],f32>
    %int32_34 = torch.constant.int 32
    %60 = torch.aten.div.Scalar %59, %int32_34 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[32],f32>
    %float2.000000e00 = torch.constant.float 2.000000e+00
    %61 = torch.aten.mul.Scalar %60, %float2.000000e00 : !torch.vtensor<[32],f32>, !torch.float -> !torch.vtensor<[32],f32>
    %float5.000000e05 = torch.constant.float 5.000000e+05
    %62 = torch.aten.pow.Scalar %float5.000000e05, %61 : !torch.float, !torch.vtensor<[32],f32> -> !torch.vtensor<[32],f32>
    %63 = torch.aten.reciprocal %62 : !torch.vtensor<[32],f32> -> !torch.vtensor<[32],f32>
    %float1.000000e00 = torch.constant.float 1.000000e+00
    %64 = torch.aten.mul.Scalar %63, %float1.000000e00 : !torch.vtensor<[32],f32>, !torch.float -> !torch.vtensor<[32],f32>
    %int1_35 = torch.constant.int 1
    %65 = torch.aten.unsqueeze %56, %int1_35 : !torch.vtensor<[128],si64>, !torch.int -> !torch.vtensor<[128,1],si64>
    %int0_36 = torch.constant.int 0
    %66 = torch.aten.unsqueeze %64, %int0_36 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[1,32],f32>
    %67 = torch.aten.mul.Tensor %65, %66 : !torch.vtensor<[128,1],si64>, !torch.vtensor<[1,32],f32> -> !torch.vtensor<[128,32],f32>
    %int6_37 = torch.constant.int 6
    %68 = torch.prims.convert_element_type %67, %int6_37 : !torch.vtensor<[128,32],f32>, !torch.int -> !torch.vtensor<[128,32],f32>
    %int1_38 = torch.constant.int 1
    %69 = torch.prim.ListConstruct %int1_38 : (!torch.int) -> !torch.list<int>
    %70 = torch.aten.view %55, %69 : !torch.vtensor<[1,1],si64>, !torch.list<int> -> !torch.vtensor<[1],si64>
    %71 = torch.prim.ListConstruct %70 : (!torch.vtensor<[1],si64>) -> !torch.list<optional<vtensor>>
    %72 = torch.aten.index.Tensor %68, %71 : !torch.vtensor<[128,32],f32>, !torch.list<optional<vtensor>> -> !torch.vtensor<[1,32],f32>
    %int1_39 = torch.constant.int 1
    %73 = torch.aten.unsqueeze %72, %int1_39 : !torch.vtensor<[1,32],f32>, !torch.int -> !torch.vtensor<[1,1,32],f32>
    %int5_40 = torch.constant.int 5
    %74 = torch.prims.convert_element_type %0, %int5_40 : !torch.vtensor<[256,256],f32>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int-1_41 = torch.constant.int -1
    %false_42 = torch.constant.bool false
    %false_43 = torch.constant.bool false
    %75 = torch.aten.embedding %74, %arg0, %int-1_41, %false_42, %false_43 : !torch.vtensor<[256,256],f16>, !torch.vtensor<[1,1],si64>, !torch.int, !torch.bool, !torch.bool -> !torch.vtensor<[1,1,256],f16>
    %int6_44 = torch.constant.int 6
    %76 = torch.prims.convert_element_type %75, %int6_44 : !torch.vtensor<[1,1,256],f16>, !torch.int -> !torch.vtensor<[1,1,256],f32>
    %int2_45 = torch.constant.int 2
    %77 = torch.aten.pow.Tensor_Scalar %76, %int2_45 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f32>
    %int-1_46 = torch.constant.int -1
    %78 = torch.prim.ListConstruct %int-1_46 : (!torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %none_47 = torch.constant.none
    %79 = torch.aten.mean.dim %77, %78, %true, %none_47 : !torch.vtensor<[1,1,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1],f32>
    %float1.000000e-02 = torch.constant.float 1.000000e-02
    %int1_48 = torch.constant.int 1
    %80 = torch.aten.add.Scalar %79, %float1.000000e-02, %int1_48 : !torch.vtensor<[1,1,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,1,1],f32>
    %81 = torch.aten.rsqrt %80 : !torch.vtensor<[1,1,1],f32> -> !torch.vtensor<[1,1,1],f32>
    %82 = torch.aten.mul.Tensor %76, %81 : !torch.vtensor<[1,1,256],f32>, !torch.vtensor<[1,1,1],f32> -> !torch.vtensor<[1,1,256],f32>
    %int5_49 = torch.constant.int 5
    %83 = torch.prims.convert_element_type %82, %int5_49 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %84 = torch.aten.mul.Tensor %1, %83 : !torch.vtensor<[256],f32>, !torch.vtensor<[1,1,256],f16> -> !torch.vtensor<[1,1,256],f32>
    %int5_50 = torch.constant.int 5
    %85 = torch.prims.convert_element_type %84, %int5_50 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %int5_51 = torch.constant.int 5
    %86 = torch.prims.convert_element_type %2, %int5_51 : !torch.vtensor<[256,256],f32>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int-2 = torch.constant.int -2
    %int-1_52 = torch.constant.int -1
    %87 = torch.aten.transpose.int %86, %int-2, %int-1_52 : !torch.vtensor<[256,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,256],f16>
    %int5_53 = torch.constant.int 5
    %88 = torch.prims.convert_element_type %87, %int5_53 : !torch.vtensor<[256,256],f16>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int1_54 = torch.constant.int 1
    %int256 = torch.constant.int 256
    %89 = torch.prim.ListConstruct %int1_54, %int256 : (!torch.int, !torch.int) -> !torch.list<int>
    %90 = torch.aten.view %85, %89 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %91 = torch.aten.mm %90, %88 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,256],f16> -> !torch.vtensor<[1,256],f16>
    %int1_55 = torch.constant.int 1
    %int1_56 = torch.constant.int 1
    %int256_57 = torch.constant.int 256
    %92 = torch.prim.ListConstruct %int1_55, %int1_56, %int256_57 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %93 = torch.aten.view %91, %92 : !torch.vtensor<[1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,1,256],f16>
    %int5_58 = torch.constant.int 5
    %94 = torch.prims.convert_element_type %3, %int5_58 : !torch.vtensor<[128,256],f32>, !torch.int -> !torch.vtensor<[128,256],f16>
    %int-2_59 = torch.constant.int -2
    %int-1_60 = torch.constant.int -1
    %95 = torch.aten.transpose.int %94, %int-2_59, %int-1_60 : !torch.vtensor<[128,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,128],f16>
    %int5_61 = torch.constant.int 5
    %96 = torch.prims.convert_element_type %95, %int5_61 : !torch.vtensor<[256,128],f16>, !torch.int -> !torch.vtensor<[256,128],f16>
    %int1_62 = torch.constant.int 1
    %int256_63 = torch.constant.int 256
    %97 = torch.prim.ListConstruct %int1_62, %int256_63 : (!torch.int, !torch.int) -> !torch.list<int>
    %98 = torch.aten.view %85, %97 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %99 = torch.aten.mm %98, %96 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,128],f16> -> !torch.vtensor<[1,128],f16>
    %int1_64 = torch.constant.int 1
    %int1_65 = torch.constant.int 1
    %int128_66 = torch.constant.int 128
    %100 = torch.prim.ListConstruct %int1_64, %int1_65, %int128_66 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %101 = torch.aten.view %99, %100 : !torch.vtensor<[1,128],f16>, !torch.list<int> -> !torch.vtensor<[1,1,128],f16>
    %int5_67 = torch.constant.int 5
    %102 = torch.prims.convert_element_type %4, %int5_67 : !torch.vtensor<[128,256],f32>, !torch.int -> !torch.vtensor<[128,256],f16>
    %int-2_68 = torch.constant.int -2
    %int-1_69 = torch.constant.int -1
    %103 = torch.aten.transpose.int %102, %int-2_68, %int-1_69 : !torch.vtensor<[128,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,128],f16>
    %int5_70 = torch.constant.int 5
    %104 = torch.prims.convert_element_type %103, %int5_70 : !torch.vtensor<[256,128],f16>, !torch.int -> !torch.vtensor<[256,128],f16>
    %int1_71 = torch.constant.int 1
    %int256_72 = torch.constant.int 256
    %105 = torch.prim.ListConstruct %int1_71, %int256_72 : (!torch.int, !torch.int) -> !torch.list<int>
    %106 = torch.aten.view %85, %105 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %107 = torch.aten.mm %106, %104 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,128],f16> -> !torch.vtensor<[1,128],f16>
    %int1_73 = torch.constant.int 1
    %int1_74 = torch.constant.int 1
    %int128_75 = torch.constant.int 128
    %108 = torch.prim.ListConstruct %int1_73, %int1_74, %int128_75 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %109 = torch.aten.view %107, %108 : !torch.vtensor<[1,128],f16>, !torch.list<int> -> !torch.vtensor<[1,1,128],f16>
    %int1_76 = torch.constant.int 1
    %int1_77 = torch.constant.int 1
    %int8 = torch.constant.int 8
    %int32_78 = torch.constant.int 32
    %110 = torch.prim.ListConstruct %int1_76, %int1_77, %int8, %int32_78 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %111 = torch.aten.view %93, %110 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,1,8,32],f16>
    %int1_79 = torch.constant.int 1
    %int1_80 = torch.constant.int 1
    %int4 = torch.constant.int 4
    %int32_81 = torch.constant.int 32
    %112 = torch.prim.ListConstruct %int1_79, %int1_80, %int4, %int32_81 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %113 = torch.aten.view %101, %112 : !torch.vtensor<[1,1,128],f16>, !torch.list<int> -> !torch.vtensor<[1,1,4,32],f16>
    %int1_82 = torch.constant.int 1
    %int1_83 = torch.constant.int 1
    %int4_84 = torch.constant.int 4
    %int32_85 = torch.constant.int 32
    %114 = torch.prim.ListConstruct %int1_82, %int1_83, %int4_84, %int32_85 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %115 = torch.aten.view %109, %114 : !torch.vtensor<[1,1,128],f16>, !torch.list<int> -> !torch.vtensor<[1,1,4,32],f16>
    %int6_86 = torch.constant.int 6
    %116 = torch.prims.convert_element_type %111, %int6_86 : !torch.vtensor<[1,1,8,32],f16>, !torch.int -> !torch.vtensor<[1,1,8,32],f32>
    %117 = torch_c.to_builtin_tensor %116 : !torch.vtensor<[1,1,8,32],f32> -> tensor<1x1x8x32xf32>
    %118 = torch_c.to_builtin_tensor %73 : !torch.vtensor<[1,1,32],f32> -> tensor<1x1x32xf32>
    %119 = util.call @sharktank_rotary_embedding_1_1_8_32_f32(%117, %118) : (tensor<1x1x8x32xf32>, tensor<1x1x32xf32>) -> tensor<1x1x8x32xf32>
    %120 = torch_c.from_builtin_tensor %119 : tensor<1x1x8x32xf32> -> !torch.vtensor<[1,1,8,32],f32>
    %int5_87 = torch.constant.int 5
    %121 = torch.prims.convert_element_type %120, %int5_87 : !torch.vtensor<[1,1,8,32],f32>, !torch.int -> !torch.vtensor<[1,1,8,32],f16>
    %int6_88 = torch.constant.int 6
    %122 = torch.prims.convert_element_type %113, %int6_88 : !torch.vtensor<[1,1,4,32],f16>, !torch.int -> !torch.vtensor<[1,1,4,32],f32>
    %123 = torch_c.to_builtin_tensor %122 : !torch.vtensor<[1,1,4,32],f32> -> tensor<1x1x4x32xf32>
    %124 = torch_c.to_builtin_tensor %73 : !torch.vtensor<[1,1,32],f32> -> tensor<1x1x32xf32>
    %125 = util.call @sharktank_rotary_embedding_1_1_4_32_f32(%123, %124) : (tensor<1x1x4x32xf32>, tensor<1x1x32xf32>) -> tensor<1x1x4x32xf32>
    %126 = torch_c.from_builtin_tensor %125 : tensor<1x1x4x32xf32> -> !torch.vtensor<[1,1,4,32],f32>
    %int5_89 = torch.constant.int 5
    %127 = torch.prims.convert_element_type %126, %int5_89 : !torch.vtensor<[1,1,4,32],f32>, !torch.int -> !torch.vtensor<[1,1,4,32],f16>
    %int3 = torch.constant.int 3
    %int2_90 = torch.constant.int 2
    %int32_91 = torch.constant.int 32
    %int4_92 = torch.constant.int 4
    %int32_93 = torch.constant.int 32
    %128 = torch.prim.ListConstruct %40, %int3, %int2_90, %int32_91, %int4_92, %int32_93 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %129 = torch.aten.view %36, %128 : !torch.vtensor<[?,24576],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,4,32],f16>
    torch.bind_symbolic_shape %129, [%38], affine_map<()[s0] -> (s0, 3, 2, 32, 4, 32)> : !torch.vtensor<[?,3,2,32,4,32],f16>
    %int3_94 = torch.constant.int 3
    %130 = torch.aten.mul.int %40, %int3_94 : !torch.int, !torch.int -> !torch.int
    %int2_95 = torch.constant.int 2
    %131 = torch.aten.mul.int %130, %int2_95 : !torch.int, !torch.int -> !torch.int
    %int32_96 = torch.constant.int 32
    %132 = torch.aten.mul.int %131, %int32_96 : !torch.int, !torch.int -> !torch.int
    %int4_97 = torch.constant.int 4
    %int32_98 = torch.constant.int 32
    %133 = torch.prim.ListConstruct %132, %int4_97, %int32_98 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %134 = torch.aten.view %129, %133 : !torch.vtensor<[?,3,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,4,32],f16>
    torch.bind_symbolic_shape %134, [%38], affine_map<()[s0] -> (s0 * 192, 4, 32)> : !torch.vtensor<[?,4,32],f16>
    %int32_99 = torch.constant.int 32
    %135 = torch.aten.floor_divide.Scalar %arg2, %int32_99 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int1_100 = torch.constant.int 1
    %136 = torch.aten.unsqueeze %135, %int1_100 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_101 = torch.constant.int 1
    %false_102 = torch.constant.bool false
    %137 = torch.aten.gather %arg3, %int1_101, %136, %false_102 : !torch.vtensor<[1,?],si64>, !torch.int, !torch.vtensor<[1,1],si64>, !torch.bool -> !torch.vtensor<[1,1],si64>
    %int32_103 = torch.constant.int 32
    %138 = torch.aten.remainder.Scalar %arg2, %int32_103 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int1_104 = torch.constant.int 1
    %139 = torch.aten.unsqueeze %138, %int1_104 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %none_105 = torch.constant.none
    %140 = torch.aten.clone %5, %none_105 : !torch.vtensor<[],si64>, !torch.none -> !torch.vtensor<[],si64>
    %141 = torch.aten.detach %140 : !torch.vtensor<[],si64> -> !torch.vtensor<[],si64>
    %142 = torch.aten.detach %141 : !torch.vtensor<[],si64> -> !torch.vtensor<[],si64>
    %143 = torch.aten.detach %142 : !torch.vtensor<[],si64> -> !torch.vtensor<[],si64>
    %int0_106 = torch.constant.int 0
    %144 = torch.aten.unsqueeze %143, %int0_106 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int1_107 = torch.constant.int 1
    %int1_108 = torch.constant.int 1
    %145 = torch.prim.ListConstruct %int1_107, %int1_108 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_109 = torch.constant.int 1
    %int1_110 = torch.constant.int 1
    %146 = torch.prim.ListConstruct %int1_109, %int1_110 : (!torch.int, !torch.int) -> !torch.list<int>
    %int4_111 = torch.constant.int 4
    %int0_112 = torch.constant.int 0
    %cpu_113 = torch.constant.device "cpu"
    %false_114 = torch.constant.bool false
    %147 = torch.aten.empty_strided %145, %146, %int4_111, %int0_112, %cpu_113, %false_114 : !torch.list<int>, !torch.list<int>, !torch.int, !torch.int, !torch.Device, !torch.bool -> !torch.vtensor<[1,1],si64>
    %int0_115 = torch.constant.int 0
    %148 = torch.aten.fill.Scalar %147, %int0_115 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_116 = torch.constant.int 1
    %int1_117 = torch.constant.int 1
    %149 = torch.prim.ListConstruct %int1_116, %int1_117 : (!torch.int, !torch.int) -> !torch.list<int>
    %150 = torch.aten.repeat %144, %149 : !torch.vtensor<[1],si64>, !torch.list<int> -> !torch.vtensor<[1,1],si64>
    %int3_118 = torch.constant.int 3
    %151 = torch.aten.mul.Scalar %137, %int3_118 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_119 = torch.constant.int 1
    %152 = torch.aten.add.Tensor %151, %148, %int1_119 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int2_120 = torch.constant.int 2
    %153 = torch.aten.mul.Scalar %152, %int2_120 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_121 = torch.constant.int 1
    %154 = torch.aten.add.Tensor %153, %150, %int1_121 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int32_122 = torch.constant.int 32
    %155 = torch.aten.mul.Scalar %154, %int32_122 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_123 = torch.constant.int 1
    %156 = torch.aten.add.Tensor %155, %139, %int1_123 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int5_124 = torch.constant.int 5
    %157 = torch.prims.convert_element_type %127, %int5_124 : !torch.vtensor<[1,1,4,32],f16>, !torch.int -> !torch.vtensor<[1,1,4,32],f16>
    %158 = torch.prim.ListConstruct %156 : (!torch.vtensor<[1,1],si64>) -> !torch.list<optional<vtensor>>
    %false_125 = torch.constant.bool false
    %159 = torch.aten.index_put %134, %158, %157, %false_125 : !torch.vtensor<[?,4,32],f16>, !torch.list<optional<vtensor>>, !torch.vtensor<[1,1,4,32],f16>, !torch.bool -> !torch.vtensor<[?,4,32],f16>
    torch.bind_symbolic_shape %159, [%38], affine_map<()[s0] -> (s0 * 192, 4, 32)> : !torch.vtensor<[?,4,32],f16>
    %int3_126 = torch.constant.int 3
    %int2_127 = torch.constant.int 2
    %int32_128 = torch.constant.int 32
    %int4_129 = torch.constant.int 4
    %int32_130 = torch.constant.int 32
    %160 = torch.prim.ListConstruct %40, %int3_126, %int2_127, %int32_128, %int4_129, %int32_130 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %161 = torch.aten.view %159, %160 : !torch.vtensor<[?,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,4,32],f16>
    torch.bind_symbolic_shape %161, [%38], affine_map<()[s0] -> (s0, 3, 2, 32, 4, 32)> : !torch.vtensor<[?,3,2,32,4,32],f16>
    %int24576 = torch.constant.int 24576
    %162 = torch.prim.ListConstruct %40, %int24576 : (!torch.int, !torch.int) -> !torch.list<int>
    %163 = torch.aten.view %161, %162 : !torch.vtensor<[?,3,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,24576],f16>
    torch.bind_symbolic_shape %163, [%38], affine_map<()[s0] -> (s0, 24576)> : !torch.vtensor<[?,24576],f16>
    %int3_131 = torch.constant.int 3
    %int2_132 = torch.constant.int 2
    %int32_133 = torch.constant.int 32
    %int4_134 = torch.constant.int 4
    %int32_135 = torch.constant.int 32
    %164 = torch.prim.ListConstruct %40, %int3_131, %int2_132, %int32_133, %int4_134, %int32_135 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %165 = torch.aten.view %163, %164 : !torch.vtensor<[?,24576],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,4,32],f16>
    torch.bind_symbolic_shape %165, [%38], affine_map<()[s0] -> (s0, 3, 2, 32, 4, 32)> : !torch.vtensor<[?,3,2,32,4,32],f16>
    %int4_136 = torch.constant.int 4
    %int32_137 = torch.constant.int 32
    %166 = torch.prim.ListConstruct %132, %int4_136, %int32_137 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %167 = torch.aten.view %165, %166 : !torch.vtensor<[?,3,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,4,32],f16>
    torch.bind_symbolic_shape %167, [%38], affine_map<()[s0] -> (s0 * 192, 4, 32)> : !torch.vtensor<[?,4,32],f16>
    %int32_138 = torch.constant.int 32
    %168 = torch.aten.floor_divide.Scalar %arg2, %int32_138 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int1_139 = torch.constant.int 1
    %169 = torch.aten.unsqueeze %168, %int1_139 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_140 = torch.constant.int 1
    %false_141 = torch.constant.bool false
    %170 = torch.aten.gather %arg3, %int1_140, %169, %false_141 : !torch.vtensor<[1,?],si64>, !torch.int, !torch.vtensor<[1,1],si64>, !torch.bool -> !torch.vtensor<[1,1],si64>
    %int32_142 = torch.constant.int 32
    %171 = torch.aten.remainder.Scalar %arg2, %int32_142 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int1_143 = torch.constant.int 1
    %172 = torch.aten.unsqueeze %171, %int1_143 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %none_144 = torch.constant.none
    %173 = torch.aten.clone %6, %none_144 : !torch.vtensor<[],si64>, !torch.none -> !torch.vtensor<[],si64>
    %174 = torch.aten.detach %173 : !torch.vtensor<[],si64> -> !torch.vtensor<[],si64>
    %175 = torch.aten.detach %174 : !torch.vtensor<[],si64> -> !torch.vtensor<[],si64>
    %176 = torch.aten.detach %175 : !torch.vtensor<[],si64> -> !torch.vtensor<[],si64>
    %int0_145 = torch.constant.int 0
    %177 = torch.aten.unsqueeze %176, %int0_145 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int1_146 = torch.constant.int 1
    %int1_147 = torch.constant.int 1
    %178 = torch.prim.ListConstruct %int1_146, %int1_147 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_148 = torch.constant.int 1
    %int1_149 = torch.constant.int 1
    %179 = torch.prim.ListConstruct %int1_148, %int1_149 : (!torch.int, !torch.int) -> !torch.list<int>
    %int4_150 = torch.constant.int 4
    %int0_151 = torch.constant.int 0
    %cpu_152 = torch.constant.device "cpu"
    %false_153 = torch.constant.bool false
    %180 = torch.aten.empty_strided %178, %179, %int4_150, %int0_151, %cpu_152, %false_153 : !torch.list<int>, !torch.list<int>, !torch.int, !torch.int, !torch.Device, !torch.bool -> !torch.vtensor<[1,1],si64>
    %int0_154 = torch.constant.int 0
    %181 = torch.aten.fill.Scalar %180, %int0_154 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_155 = torch.constant.int 1
    %int1_156 = torch.constant.int 1
    %182 = torch.prim.ListConstruct %int1_155, %int1_156 : (!torch.int, !torch.int) -> !torch.list<int>
    %183 = torch.aten.repeat %177, %182 : !torch.vtensor<[1],si64>, !torch.list<int> -> !torch.vtensor<[1,1],si64>
    %int3_157 = torch.constant.int 3
    %184 = torch.aten.mul.Scalar %170, %int3_157 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_158 = torch.constant.int 1
    %185 = torch.aten.add.Tensor %184, %181, %int1_158 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int2_159 = torch.constant.int 2
    %186 = torch.aten.mul.Scalar %185, %int2_159 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_160 = torch.constant.int 1
    %187 = torch.aten.add.Tensor %186, %183, %int1_160 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int32_161 = torch.constant.int 32
    %188 = torch.aten.mul.Scalar %187, %int32_161 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_162 = torch.constant.int 1
    %189 = torch.aten.add.Tensor %188, %172, %int1_162 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int5_163 = torch.constant.int 5
    %190 = torch.prims.convert_element_type %115, %int5_163 : !torch.vtensor<[1,1,4,32],f16>, !torch.int -> !torch.vtensor<[1,1,4,32],f16>
    %191 = torch.prim.ListConstruct %189 : (!torch.vtensor<[1,1],si64>) -> !torch.list<optional<vtensor>>
    %false_164 = torch.constant.bool false
    %192 = torch.aten.index_put %167, %191, %190, %false_164 : !torch.vtensor<[?,4,32],f16>, !torch.list<optional<vtensor>>, !torch.vtensor<[1,1,4,32],f16>, !torch.bool -> !torch.vtensor<[?,4,32],f16>
    torch.bind_symbolic_shape %192, [%38], affine_map<()[s0] -> (s0 * 192, 4, 32)> : !torch.vtensor<[?,4,32],f16>
    %int3_165 = torch.constant.int 3
    %int2_166 = torch.constant.int 2
    %int32_167 = torch.constant.int 32
    %int4_168 = torch.constant.int 4
    %int32_169 = torch.constant.int 32
    %193 = torch.prim.ListConstruct %40, %int3_165, %int2_166, %int32_167, %int4_168, %int32_169 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %194 = torch.aten.view %192, %193 : !torch.vtensor<[?,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,4,32],f16>
    torch.bind_symbolic_shape %194, [%38], affine_map<()[s0] -> (s0, 3, 2, 32, 4, 32)> : !torch.vtensor<[?,3,2,32,4,32],f16>
    %int24576_170 = torch.constant.int 24576
    %195 = torch.prim.ListConstruct %40, %int24576_170 : (!torch.int, !torch.int) -> !torch.list<int>
    %196 = torch.aten.view %194, %195 : !torch.vtensor<[?,3,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,24576],f16>
    torch.bind_symbolic_shape %196, [%38], affine_map<()[s0] -> (s0, 24576)> : !torch.vtensor<[?,24576],f16>
    %int1_171 = torch.constant.int 1
    %197 = torch.prim.ListConstruct %int1_171, %39 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_172 = torch.constant.int 1
    %198 = torch.prim.ListConstruct %39, %int1_172 : (!torch.int, !torch.int) -> !torch.list<int>
    %int4_173 = torch.constant.int 4
    %int0_174 = torch.constant.int 0
    %cpu_175 = torch.constant.device "cpu"
    %false_176 = torch.constant.bool false
    %199 = torch.aten.empty_strided %197, %198, %int4_173, %int0_174, %cpu_175, %false_176 : !torch.list<int>, !torch.list<int>, !torch.int, !torch.int, !torch.Device, !torch.bool -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %199, [%37], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %int0_177 = torch.constant.int 0
    %200 = torch.aten.fill.Scalar %199, %int0_177 : !torch.vtensor<[1,?],si64>, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %200, [%37], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %int3_178 = torch.constant.int 3
    %201 = torch.aten.mul.Scalar %arg3, %int3_178 : !torch.vtensor<[1,?],si64>, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %201, [%37], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %int1_179 = torch.constant.int 1
    %202 = torch.aten.add.Tensor %201, %200, %int1_179 : !torch.vtensor<[1,?],si64>, !torch.vtensor<[1,?],si64>, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %202, [%37], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %203 = torch.prim.ListConstruct %39 : (!torch.int) -> !torch.list<int>
    %204 = torch.aten.view %202, %203 : !torch.vtensor<[1,?],si64>, !torch.list<int> -> !torch.vtensor<[?],si64>
    torch.bind_symbolic_shape %204, [%37], affine_map<()[s0] -> (s0)> : !torch.vtensor<[?],si64>
    %int3_180 = torch.constant.int 3
    %int2_181 = torch.constant.int 2
    %int32_182 = torch.constant.int 32
    %int4_183 = torch.constant.int 4
    %int32_184 = torch.constant.int 32
    %205 = torch.prim.ListConstruct %40, %int3_180, %int2_181, %int32_182, %int4_183, %int32_184 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %206 = torch.aten.view %196, %205 : !torch.vtensor<[?,24576],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,4,32],f16>
    torch.bind_symbolic_shape %206, [%38], affine_map<()[s0] -> (s0, 3, 2, 32, 4, 32)> : !torch.vtensor<[?,3,2,32,4,32],f16>
    %int2_185 = torch.constant.int 2
    %int32_186 = torch.constant.int 32
    %int4_187 = torch.constant.int 4
    %int32_188 = torch.constant.int 32
    %207 = torch.prim.ListConstruct %130, %int2_185, %int32_186, %int4_187, %int32_188 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %208 = torch.aten.view %206, %207 : !torch.vtensor<[?,3,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,2,32,4,32],f16>
    torch.bind_symbolic_shape %208, [%38], affine_map<()[s0] -> (s0 * 3, 2, 32, 4, 32)> : !torch.vtensor<[?,2,32,4,32],f16>
    %int0_189 = torch.constant.int 0
    %209 = torch.aten.index_select %208, %int0_189, %204 : !torch.vtensor<[?,2,32,4,32],f16>, !torch.int, !torch.vtensor<[?],si64> -> !torch.vtensor<[?,2,32,4,32],f16>
    torch.bind_symbolic_shape %209, [%37], affine_map<()[s0] -> (s0, 2, 32, 4, 32)> : !torch.vtensor<[?,2,32,4,32],f16>
    %int1_190 = torch.constant.int 1
    %int2_191 = torch.constant.int 2
    %int32_192 = torch.constant.int 32
    %int4_193 = torch.constant.int 4
    %int32_194 = torch.constant.int 32
    %210 = torch.prim.ListConstruct %int1_190, %39, %int2_191, %int32_192, %int4_193, %int32_194 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %211 = torch.aten.view %209, %210 : !torch.vtensor<[?,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,2,32,4,32],f16>
    torch.bind_symbolic_shape %211, [%37], affine_map<()[s0] -> (1, s0, 2, 32, 4, 32)> : !torch.vtensor<[1,?,2,32,4,32],f16>
    %int0_195 = torch.constant.int 0
    %int0_196 = torch.constant.int 0
    %int9223372036854775807 = torch.constant.int 9223372036854775807
    %int1_197 = torch.constant.int 1
    %212 = torch.aten.slice.Tensor %211, %int0_195, %int0_196, %int9223372036854775807, %int1_197 : !torch.vtensor<[1,?,2,32,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,2,32,4,32],f16>
    torch.bind_symbolic_shape %212, [%37], affine_map<()[s0] -> (1, s0, 2, 32, 4, 32)> : !torch.vtensor<[1,?,2,32,4,32],f16>
    %int1_198 = torch.constant.int 1
    %int0_199 = torch.constant.int 0
    %int9223372036854775807_200 = torch.constant.int 9223372036854775807
    %int1_201 = torch.constant.int 1
    %213 = torch.aten.slice.Tensor %212, %int1_198, %int0_199, %int9223372036854775807_200, %int1_201 : !torch.vtensor<[1,?,2,32,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,2,32,4,32],f16>
    torch.bind_symbolic_shape %213, [%37], affine_map<()[s0] -> (1, s0, 2, 32, 4, 32)> : !torch.vtensor<[1,?,2,32,4,32],f16>
    %int2_202 = torch.constant.int 2
    %int0_203 = torch.constant.int 0
    %214 = torch.aten.select.int %213, %int2_202, %int0_203 : !torch.vtensor<[1,?,2,32,4,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,?,32,4,32],f16>
    torch.bind_symbolic_shape %214, [%37], affine_map<()[s0] -> (1, s0, 32, 4, 32)> : !torch.vtensor<[1,?,32,4,32],f16>
    %int2_204 = torch.constant.int 2
    %int0_205 = torch.constant.int 0
    %int1_206 = torch.constant.int 1
    %215 = torch.aten.slice.Tensor %214, %int2_204, %int0_205, %41, %int1_206 : !torch.vtensor<[1,?,32,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32,4,32],f16>
    torch.bind_symbolic_shape %215, [%37], affine_map<()[s0] -> (1, s0, 32, 4, 32)> : !torch.vtensor<[1,?,32,4,32],f16>
    %int0_207 = torch.constant.int 0
    %216 = torch.aten.clone %215, %int0_207 : !torch.vtensor<[1,?,32,4,32],f16>, !torch.int -> !torch.vtensor<[1,?,32,4,32],f16>
    torch.bind_symbolic_shape %216, [%37], affine_map<()[s0] -> (1, s0, 32, 4, 32)> : !torch.vtensor<[1,?,32,4,32],f16>
    %int1_208 = torch.constant.int 1
    %int4_209 = torch.constant.int 4
    %int32_210 = torch.constant.int 32
    %217 = torch.prim.ListConstruct %int1_208, %41, %int4_209, %int32_210 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %218 = torch.aten._unsafe_view %216, %217 : !torch.vtensor<[1,?,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %218, [%37], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int0_211 = torch.constant.int 0
    %int0_212 = torch.constant.int 0
    %int9223372036854775807_213 = torch.constant.int 9223372036854775807
    %int1_214 = torch.constant.int 1
    %219 = torch.aten.slice.Tensor %218, %int0_211, %int0_212, %int9223372036854775807_213, %int1_214 : !torch.vtensor<[1,?,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %219, [%37], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int0_215 = torch.constant.int 0
    %int0_216 = torch.constant.int 0
    %int9223372036854775807_217 = torch.constant.int 9223372036854775807
    %int1_218 = torch.constant.int 1
    %220 = torch.aten.slice.Tensor %211, %int0_215, %int0_216, %int9223372036854775807_217, %int1_218 : !torch.vtensor<[1,?,2,32,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,2,32,4,32],f16>
    torch.bind_symbolic_shape %220, [%37], affine_map<()[s0] -> (1, s0, 2, 32, 4, 32)> : !torch.vtensor<[1,?,2,32,4,32],f16>
    %int1_219 = torch.constant.int 1
    %int0_220 = torch.constant.int 0
    %int9223372036854775807_221 = torch.constant.int 9223372036854775807
    %int1_222 = torch.constant.int 1
    %221 = torch.aten.slice.Tensor %220, %int1_219, %int0_220, %int9223372036854775807_221, %int1_222 : !torch.vtensor<[1,?,2,32,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,2,32,4,32],f16>
    torch.bind_symbolic_shape %221, [%37], affine_map<()[s0] -> (1, s0, 2, 32, 4, 32)> : !torch.vtensor<[1,?,2,32,4,32],f16>
    %int2_223 = torch.constant.int 2
    %int1_224 = torch.constant.int 1
    %222 = torch.aten.select.int %221, %int2_223, %int1_224 : !torch.vtensor<[1,?,2,32,4,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,?,32,4,32],f16>
    torch.bind_symbolic_shape %222, [%37], affine_map<()[s0] -> (1, s0, 32, 4, 32)> : !torch.vtensor<[1,?,32,4,32],f16>
    %int2_225 = torch.constant.int 2
    %int0_226 = torch.constant.int 0
    %int1_227 = torch.constant.int 1
    %223 = torch.aten.slice.Tensor %222, %int2_225, %int0_226, %41, %int1_227 : !torch.vtensor<[1,?,32,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32,4,32],f16>
    torch.bind_symbolic_shape %223, [%37], affine_map<()[s0] -> (1, s0, 32, 4, 32)> : !torch.vtensor<[1,?,32,4,32],f16>
    %int0_228 = torch.constant.int 0
    %224 = torch.aten.clone %223, %int0_228 : !torch.vtensor<[1,?,32,4,32],f16>, !torch.int -> !torch.vtensor<[1,?,32,4,32],f16>
    torch.bind_symbolic_shape %224, [%37], affine_map<()[s0] -> (1, s0, 32, 4, 32)> : !torch.vtensor<[1,?,32,4,32],f16>
    %int1_229 = torch.constant.int 1
    %int4_230 = torch.constant.int 4
    %int32_231 = torch.constant.int 32
    %225 = torch.prim.ListConstruct %int1_229, %41, %int4_230, %int32_231 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %226 = torch.aten._unsafe_view %224, %225 : !torch.vtensor<[1,?,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %226, [%37], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int0_232 = torch.constant.int 0
    %int0_233 = torch.constant.int 0
    %int9223372036854775807_234 = torch.constant.int 9223372036854775807
    %int1_235 = torch.constant.int 1
    %227 = torch.aten.slice.Tensor %226, %int0_232, %int0_233, %int9223372036854775807_234, %int1_235 : !torch.vtensor<[1,?,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %227, [%37], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int-2_236 = torch.constant.int -2
    %228 = torch.aten.unsqueeze %219, %int-2_236 : !torch.vtensor<[1,?,4,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,1,32],f16>
    torch.bind_symbolic_shape %228, [%37], affine_map<()[s0] -> (1, s0 * 32, 4, 1, 32)> : !torch.vtensor<[1,?,4,1,32],f16>
    %int1_237 = torch.constant.int 1
    %int4_238 = torch.constant.int 4
    %int2_239 = torch.constant.int 2
    %int32_240 = torch.constant.int 32
    %229 = torch.prim.ListConstruct %int1_237, %41, %int4_238, %int2_239, %int32_240 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_241 = torch.constant.bool false
    %230 = torch.aten.expand %228, %229, %false_241 : !torch.vtensor<[1,?,4,1,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,?,4,2,32],f16>
    torch.bind_symbolic_shape %230, [%37], affine_map<()[s0] -> (1, s0 * 32, 4, 2, 32)> : !torch.vtensor<[1,?,4,2,32],f16>
    %int0_242 = torch.constant.int 0
    %231 = torch.aten.clone %230, %int0_242 : !torch.vtensor<[1,?,4,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,2,32],f16>
    torch.bind_symbolic_shape %231, [%37], affine_map<()[s0] -> (1, s0 * 32, 4, 2, 32)> : !torch.vtensor<[1,?,4,2,32],f16>
    %int1_243 = torch.constant.int 1
    %int8_244 = torch.constant.int 8
    %int32_245 = torch.constant.int 32
    %232 = torch.prim.ListConstruct %int1_243, %41, %int8_244, %int32_245 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %233 = torch.aten._unsafe_view %231, %232 : !torch.vtensor<[1,?,4,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,8,32],f16>
    torch.bind_symbolic_shape %233, [%37], affine_map<()[s0] -> (1, s0 * 32, 8, 32)> : !torch.vtensor<[1,?,8,32],f16>
    %int-2_246 = torch.constant.int -2
    %234 = torch.aten.unsqueeze %227, %int-2_246 : !torch.vtensor<[1,?,4,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,1,32],f16>
    torch.bind_symbolic_shape %234, [%37], affine_map<()[s0] -> (1, s0 * 32, 4, 1, 32)> : !torch.vtensor<[1,?,4,1,32],f16>
    %int1_247 = torch.constant.int 1
    %int4_248 = torch.constant.int 4
    %int2_249 = torch.constant.int 2
    %int32_250 = torch.constant.int 32
    %235 = torch.prim.ListConstruct %int1_247, %41, %int4_248, %int2_249, %int32_250 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_251 = torch.constant.bool false
    %236 = torch.aten.expand %234, %235, %false_251 : !torch.vtensor<[1,?,4,1,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,?,4,2,32],f16>
    torch.bind_symbolic_shape %236, [%37], affine_map<()[s0] -> (1, s0 * 32, 4, 2, 32)> : !torch.vtensor<[1,?,4,2,32],f16>
    %int0_252 = torch.constant.int 0
    %237 = torch.aten.clone %236, %int0_252 : !torch.vtensor<[1,?,4,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,2,32],f16>
    torch.bind_symbolic_shape %237, [%37], affine_map<()[s0] -> (1, s0 * 32, 4, 2, 32)> : !torch.vtensor<[1,?,4,2,32],f16>
    %int1_253 = torch.constant.int 1
    %int8_254 = torch.constant.int 8
    %int32_255 = torch.constant.int 32
    %238 = torch.prim.ListConstruct %int1_253, %41, %int8_254, %int32_255 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %239 = torch.aten._unsafe_view %237, %238 : !torch.vtensor<[1,?,4,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,8,32],f16>
    torch.bind_symbolic_shape %239, [%37], affine_map<()[s0] -> (1, s0 * 32, 8, 32)> : !torch.vtensor<[1,?,8,32],f16>
    %int1_256 = torch.constant.int 1
    %int2_257 = torch.constant.int 2
    %240 = torch.aten.transpose.int %121, %int1_256, %int2_257 : !torch.vtensor<[1,1,8,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,8,1,32],f16>
    %int1_258 = torch.constant.int 1
    %int2_259 = torch.constant.int 2
    %241 = torch.aten.transpose.int %233, %int1_258, %int2_259 : !torch.vtensor<[1,?,8,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,8,?,32],f16>
    torch.bind_symbolic_shape %241, [%37], affine_map<()[s0] -> (1, 8, s0 * 32, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %int1_260 = torch.constant.int 1
    %int2_261 = torch.constant.int 2
    %242 = torch.aten.transpose.int %239, %int1_260, %int2_261 : !torch.vtensor<[1,?,8,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,8,?,32],f16>
    torch.bind_symbolic_shape %242, [%37], affine_map<()[s0] -> (1, 8, s0 * 32, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %int5_262 = torch.constant.int 5
    %243 = torch.prims.convert_element_type %240, %int5_262 : !torch.vtensor<[1,8,1,32],f16>, !torch.int -> !torch.vtensor<[1,8,1,32],f16>
    %int5_263 = torch.constant.int 5
    %244 = torch.prims.convert_element_type %241, %int5_263 : !torch.vtensor<[1,8,?,32],f16>, !torch.int -> !torch.vtensor<[1,8,?,32],f16>
    torch.bind_symbolic_shape %244, [%37], affine_map<()[s0] -> (1, 8, s0 * 32, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %int5_264 = torch.constant.int 5
    %245 = torch.prims.convert_element_type %242, %int5_264 : !torch.vtensor<[1,8,?,32],f16>, !torch.int -> !torch.vtensor<[1,8,?,32],f16>
    torch.bind_symbolic_shape %245, [%37], affine_map<()[s0] -> (1, 8, s0 * 32, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %int5_265 = torch.constant.int 5
    %246 = torch.prims.convert_element_type %51, %int5_265 : !torch.vtensor<[1,1,1,?],f16>, !torch.int -> !torch.vtensor<[1,1,1,?],f16>
    torch.bind_symbolic_shape %246, [%37], affine_map<()[s0] -> (1, 1, 1, s0 * 32)> : !torch.vtensor<[1,1,1,?],f16>
    %float0.000000e00 = torch.constant.float 0.000000e+00
    %false_266 = torch.constant.bool false
    %none_267 = torch.constant.none
    %247:2 = torch.operator "torch.aten._scaled_dot_product_flash_attention_for_cpu"(%243, %244, %245, %float0.000000e00, %false_266, %246, %none_267) : (!torch.vtensor<[1,8,1,32],f16>, !torch.vtensor<[1,8,?,32],f16>, !torch.vtensor<[1,8,?,32],f16>, !torch.float, !torch.bool, !torch.vtensor<[1,1,1,?],f16>, !torch.none) -> (!torch.vtensor<[1,8,1,32],f16>, !torch.vtensor<[1,8,1],f32>) 
    %int1_268 = torch.constant.int 1
    %int2_269 = torch.constant.int 2
    %248 = torch.aten.transpose.int %247#0, %int1_268, %int2_269 : !torch.vtensor<[1,8,1,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,1,8,32],f16>
    %int1_270 = torch.constant.int 1
    %int1_271 = torch.constant.int 1
    %int256_272 = torch.constant.int 256
    %249 = torch.prim.ListConstruct %int1_270, %int1_271, %int256_272 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %250 = torch.aten.view %248, %249 : !torch.vtensor<[1,1,8,32],f16>, !torch.list<int> -> !torch.vtensor<[1,1,256],f16>
    %int5_273 = torch.constant.int 5
    %251 = torch.prims.convert_element_type %7, %int5_273 : !torch.vtensor<[256,256],f32>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int-2_274 = torch.constant.int -2
    %int-1_275 = torch.constant.int -1
    %252 = torch.aten.transpose.int %251, %int-2_274, %int-1_275 : !torch.vtensor<[256,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,256],f16>
    %int5_276 = torch.constant.int 5
    %253 = torch.prims.convert_element_type %252, %int5_276 : !torch.vtensor<[256,256],f16>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int1_277 = torch.constant.int 1
    %int256_278 = torch.constant.int 256
    %254 = torch.prim.ListConstruct %int1_277, %int256_278 : (!torch.int, !torch.int) -> !torch.list<int>
    %255 = torch.aten.view %250, %254 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %256 = torch.aten.mm %255, %253 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,256],f16> -> !torch.vtensor<[1,256],f16>
    %int1_279 = torch.constant.int 1
    %int1_280 = torch.constant.int 1
    %int256_281 = torch.constant.int 256
    %257 = torch.prim.ListConstruct %int1_279, %int1_280, %int256_281 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %258 = torch.aten.view %256, %257 : !torch.vtensor<[1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,1,256],f16>
    %int1_282 = torch.constant.int 1
    %259 = torch.aten.add.Tensor %75, %258, %int1_282 : !torch.vtensor<[1,1,256],f16>, !torch.vtensor<[1,1,256],f16>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %int6_283 = torch.constant.int 6
    %260 = torch.prims.convert_element_type %259, %int6_283 : !torch.vtensor<[1,1,256],f16>, !torch.int -> !torch.vtensor<[1,1,256],f32>
    %int2_284 = torch.constant.int 2
    %261 = torch.aten.pow.Tensor_Scalar %260, %int2_284 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f32>
    %int-1_285 = torch.constant.int -1
    %262 = torch.prim.ListConstruct %int-1_285 : (!torch.int) -> !torch.list<int>
    %true_286 = torch.constant.bool true
    %none_287 = torch.constant.none
    %263 = torch.aten.mean.dim %261, %262, %true_286, %none_287 : !torch.vtensor<[1,1,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1],f32>
    %float1.000000e-02_288 = torch.constant.float 1.000000e-02
    %int1_289 = torch.constant.int 1
    %264 = torch.aten.add.Scalar %263, %float1.000000e-02_288, %int1_289 : !torch.vtensor<[1,1,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,1,1],f32>
    %265 = torch.aten.rsqrt %264 : !torch.vtensor<[1,1,1],f32> -> !torch.vtensor<[1,1,1],f32>
    %266 = torch.aten.mul.Tensor %260, %265 : !torch.vtensor<[1,1,256],f32>, !torch.vtensor<[1,1,1],f32> -> !torch.vtensor<[1,1,256],f32>
    %int5_290 = torch.constant.int 5
    %267 = torch.prims.convert_element_type %266, %int5_290 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %268 = torch.aten.mul.Tensor %8, %267 : !torch.vtensor<[256],f32>, !torch.vtensor<[1,1,256],f16> -> !torch.vtensor<[1,1,256],f32>
    %int5_291 = torch.constant.int 5
    %269 = torch.prims.convert_element_type %268, %int5_291 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %int5_292 = torch.constant.int 5
    %270 = torch.prims.convert_element_type %9, %int5_292 : !torch.vtensor<[23,256],f32>, !torch.int -> !torch.vtensor<[23,256],f16>
    %int-2_293 = torch.constant.int -2
    %int-1_294 = torch.constant.int -1
    %271 = torch.aten.transpose.int %270, %int-2_293, %int-1_294 : !torch.vtensor<[23,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,23],f16>
    %int5_295 = torch.constant.int 5
    %272 = torch.prims.convert_element_type %271, %int5_295 : !torch.vtensor<[256,23],f16>, !torch.int -> !torch.vtensor<[256,23],f16>
    %int1_296 = torch.constant.int 1
    %int256_297 = torch.constant.int 256
    %273 = torch.prim.ListConstruct %int1_296, %int256_297 : (!torch.int, !torch.int) -> !torch.list<int>
    %274 = torch.aten.view %269, %273 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %275 = torch.aten.mm %274, %272 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,23],f16> -> !torch.vtensor<[1,23],f16>
    %int1_298 = torch.constant.int 1
    %int1_299 = torch.constant.int 1
    %int23 = torch.constant.int 23
    %276 = torch.prim.ListConstruct %int1_298, %int1_299, %int23 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %277 = torch.aten.view %275, %276 : !torch.vtensor<[1,23],f16>, !torch.list<int> -> !torch.vtensor<[1,1,23],f16>
    %278 = torch.aten.silu %277 : !torch.vtensor<[1,1,23],f16> -> !torch.vtensor<[1,1,23],f16>
    %int5_300 = torch.constant.int 5
    %279 = torch.prims.convert_element_type %10, %int5_300 : !torch.vtensor<[23,256],f32>, !torch.int -> !torch.vtensor<[23,256],f16>
    %int-2_301 = torch.constant.int -2
    %int-1_302 = torch.constant.int -1
    %280 = torch.aten.transpose.int %279, %int-2_301, %int-1_302 : !torch.vtensor<[23,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,23],f16>
    %int5_303 = torch.constant.int 5
    %281 = torch.prims.convert_element_type %280, %int5_303 : !torch.vtensor<[256,23],f16>, !torch.int -> !torch.vtensor<[256,23],f16>
    %int1_304 = torch.constant.int 1
    %int256_305 = torch.constant.int 256
    %282 = torch.prim.ListConstruct %int1_304, %int256_305 : (!torch.int, !torch.int) -> !torch.list<int>
    %283 = torch.aten.view %269, %282 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %284 = torch.aten.mm %283, %281 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,23],f16> -> !torch.vtensor<[1,23],f16>
    %int1_306 = torch.constant.int 1
    %int1_307 = torch.constant.int 1
    %int23_308 = torch.constant.int 23
    %285 = torch.prim.ListConstruct %int1_306, %int1_307, %int23_308 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %286 = torch.aten.view %284, %285 : !torch.vtensor<[1,23],f16>, !torch.list<int> -> !torch.vtensor<[1,1,23],f16>
    %287 = torch.aten.mul.Tensor %278, %286 : !torch.vtensor<[1,1,23],f16>, !torch.vtensor<[1,1,23],f16> -> !torch.vtensor<[1,1,23],f16>
    %int5_309 = torch.constant.int 5
    %288 = torch.prims.convert_element_type %11, %int5_309 : !torch.vtensor<[256,23],f32>, !torch.int -> !torch.vtensor<[256,23],f16>
    %int-2_310 = torch.constant.int -2
    %int-1_311 = torch.constant.int -1
    %289 = torch.aten.transpose.int %288, %int-2_310, %int-1_311 : !torch.vtensor<[256,23],f16>, !torch.int, !torch.int -> !torch.vtensor<[23,256],f16>
    %int5_312 = torch.constant.int 5
    %290 = torch.prims.convert_element_type %289, %int5_312 : !torch.vtensor<[23,256],f16>, !torch.int -> !torch.vtensor<[23,256],f16>
    %int1_313 = torch.constant.int 1
    %int23_314 = torch.constant.int 23
    %291 = torch.prim.ListConstruct %int1_313, %int23_314 : (!torch.int, !torch.int) -> !torch.list<int>
    %292 = torch.aten.view %287, %291 : !torch.vtensor<[1,1,23],f16>, !torch.list<int> -> !torch.vtensor<[1,23],f16>
    %293 = torch.aten.mm %292, %290 : !torch.vtensor<[1,23],f16>, !torch.vtensor<[23,256],f16> -> !torch.vtensor<[1,256],f16>
    %int1_315 = torch.constant.int 1
    %int1_316 = torch.constant.int 1
    %int256_317 = torch.constant.int 256
    %294 = torch.prim.ListConstruct %int1_315, %int1_316, %int256_317 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %295 = torch.aten.view %293, %294 : !torch.vtensor<[1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,1,256],f16>
    %int1_318 = torch.constant.int 1
    %296 = torch.aten.add.Tensor %259, %295, %int1_318 : !torch.vtensor<[1,1,256],f16>, !torch.vtensor<[1,1,256],f16>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %int6_319 = torch.constant.int 6
    %297 = torch.prims.convert_element_type %296, %int6_319 : !torch.vtensor<[1,1,256],f16>, !torch.int -> !torch.vtensor<[1,1,256],f32>
    %int2_320 = torch.constant.int 2
    %298 = torch.aten.pow.Tensor_Scalar %297, %int2_320 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f32>
    %int-1_321 = torch.constant.int -1
    %299 = torch.prim.ListConstruct %int-1_321 : (!torch.int) -> !torch.list<int>
    %true_322 = torch.constant.bool true
    %none_323 = torch.constant.none
    %300 = torch.aten.mean.dim %298, %299, %true_322, %none_323 : !torch.vtensor<[1,1,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1],f32>
    %float1.000000e-02_324 = torch.constant.float 1.000000e-02
    %int1_325 = torch.constant.int 1
    %301 = torch.aten.add.Scalar %300, %float1.000000e-02_324, %int1_325 : !torch.vtensor<[1,1,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,1,1],f32>
    %302 = torch.aten.rsqrt %301 : !torch.vtensor<[1,1,1],f32> -> !torch.vtensor<[1,1,1],f32>
    %303 = torch.aten.mul.Tensor %297, %302 : !torch.vtensor<[1,1,256],f32>, !torch.vtensor<[1,1,1],f32> -> !torch.vtensor<[1,1,256],f32>
    %int5_326 = torch.constant.int 5
    %304 = torch.prims.convert_element_type %303, %int5_326 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %305 = torch.aten.mul.Tensor %12, %304 : !torch.vtensor<[256],f32>, !torch.vtensor<[1,1,256],f16> -> !torch.vtensor<[1,1,256],f32>
    %int5_327 = torch.constant.int 5
    %306 = torch.prims.convert_element_type %305, %int5_327 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %int5_328 = torch.constant.int 5
    %307 = torch.prims.convert_element_type %13, %int5_328 : !torch.vtensor<[256,256],f32>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int-2_329 = torch.constant.int -2
    %int-1_330 = torch.constant.int -1
    %308 = torch.aten.transpose.int %307, %int-2_329, %int-1_330 : !torch.vtensor<[256,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,256],f16>
    %int5_331 = torch.constant.int 5
    %309 = torch.prims.convert_element_type %308, %int5_331 : !torch.vtensor<[256,256],f16>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int1_332 = torch.constant.int 1
    %int256_333 = torch.constant.int 256
    %310 = torch.prim.ListConstruct %int1_332, %int256_333 : (!torch.int, !torch.int) -> !torch.list<int>
    %311 = torch.aten.view %306, %310 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %312 = torch.aten.mm %311, %309 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,256],f16> -> !torch.vtensor<[1,256],f16>
    %int1_334 = torch.constant.int 1
    %int1_335 = torch.constant.int 1
    %int256_336 = torch.constant.int 256
    %313 = torch.prim.ListConstruct %int1_334, %int1_335, %int256_336 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %314 = torch.aten.view %312, %313 : !torch.vtensor<[1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,1,256],f16>
    %int5_337 = torch.constant.int 5
    %315 = torch.prims.convert_element_type %14, %int5_337 : !torch.vtensor<[128,256],f32>, !torch.int -> !torch.vtensor<[128,256],f16>
    %int-2_338 = torch.constant.int -2
    %int-1_339 = torch.constant.int -1
    %316 = torch.aten.transpose.int %315, %int-2_338, %int-1_339 : !torch.vtensor<[128,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,128],f16>
    %int5_340 = torch.constant.int 5
    %317 = torch.prims.convert_element_type %316, %int5_340 : !torch.vtensor<[256,128],f16>, !torch.int -> !torch.vtensor<[256,128],f16>
    %int1_341 = torch.constant.int 1
    %int256_342 = torch.constant.int 256
    %318 = torch.prim.ListConstruct %int1_341, %int256_342 : (!torch.int, !torch.int) -> !torch.list<int>
    %319 = torch.aten.view %306, %318 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %320 = torch.aten.mm %319, %317 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,128],f16> -> !torch.vtensor<[1,128],f16>
    %int1_343 = torch.constant.int 1
    %int1_344 = torch.constant.int 1
    %int128_345 = torch.constant.int 128
    %321 = torch.prim.ListConstruct %int1_343, %int1_344, %int128_345 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %322 = torch.aten.view %320, %321 : !torch.vtensor<[1,128],f16>, !torch.list<int> -> !torch.vtensor<[1,1,128],f16>
    %int5_346 = torch.constant.int 5
    %323 = torch.prims.convert_element_type %15, %int5_346 : !torch.vtensor<[128,256],f32>, !torch.int -> !torch.vtensor<[128,256],f16>
    %int-2_347 = torch.constant.int -2
    %int-1_348 = torch.constant.int -1
    %324 = torch.aten.transpose.int %323, %int-2_347, %int-1_348 : !torch.vtensor<[128,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,128],f16>
    %int5_349 = torch.constant.int 5
    %325 = torch.prims.convert_element_type %324, %int5_349 : !torch.vtensor<[256,128],f16>, !torch.int -> !torch.vtensor<[256,128],f16>
    %int1_350 = torch.constant.int 1
    %int256_351 = torch.constant.int 256
    %326 = torch.prim.ListConstruct %int1_350, %int256_351 : (!torch.int, !torch.int) -> !torch.list<int>
    %327 = torch.aten.view %306, %326 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %328 = torch.aten.mm %327, %325 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,128],f16> -> !torch.vtensor<[1,128],f16>
    %int1_352 = torch.constant.int 1
    %int1_353 = torch.constant.int 1
    %int128_354 = torch.constant.int 128
    %329 = torch.prim.ListConstruct %int1_352, %int1_353, %int128_354 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %330 = torch.aten.view %328, %329 : !torch.vtensor<[1,128],f16>, !torch.list<int> -> !torch.vtensor<[1,1,128],f16>
    %int1_355 = torch.constant.int 1
    %int1_356 = torch.constant.int 1
    %int8_357 = torch.constant.int 8
    %int32_358 = torch.constant.int 32
    %331 = torch.prim.ListConstruct %int1_355, %int1_356, %int8_357, %int32_358 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %332 = torch.aten.view %314, %331 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,1,8,32],f16>
    %int1_359 = torch.constant.int 1
    %int1_360 = torch.constant.int 1
    %int4_361 = torch.constant.int 4
    %int32_362 = torch.constant.int 32
    %333 = torch.prim.ListConstruct %int1_359, %int1_360, %int4_361, %int32_362 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %334 = torch.aten.view %322, %333 : !torch.vtensor<[1,1,128],f16>, !torch.list<int> -> !torch.vtensor<[1,1,4,32],f16>
    %int1_363 = torch.constant.int 1
    %int1_364 = torch.constant.int 1
    %int4_365 = torch.constant.int 4
    %int32_366 = torch.constant.int 32
    %335 = torch.prim.ListConstruct %int1_363, %int1_364, %int4_365, %int32_366 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %336 = torch.aten.view %330, %335 : !torch.vtensor<[1,1,128],f16>, !torch.list<int> -> !torch.vtensor<[1,1,4,32],f16>
    %int6_367 = torch.constant.int 6
    %337 = torch.prims.convert_element_type %332, %int6_367 : !torch.vtensor<[1,1,8,32],f16>, !torch.int -> !torch.vtensor<[1,1,8,32],f32>
    %338 = torch_c.to_builtin_tensor %337 : !torch.vtensor<[1,1,8,32],f32> -> tensor<1x1x8x32xf32>
    %339 = torch_c.to_builtin_tensor %73 : !torch.vtensor<[1,1,32],f32> -> tensor<1x1x32xf32>
    %340 = util.call @sharktank_rotary_embedding_1_1_8_32_f32(%338, %339) : (tensor<1x1x8x32xf32>, tensor<1x1x32xf32>) -> tensor<1x1x8x32xf32>
    %341 = torch_c.from_builtin_tensor %340 : tensor<1x1x8x32xf32> -> !torch.vtensor<[1,1,8,32],f32>
    %int5_368 = torch.constant.int 5
    %342 = torch.prims.convert_element_type %341, %int5_368 : !torch.vtensor<[1,1,8,32],f32>, !torch.int -> !torch.vtensor<[1,1,8,32],f16>
    %int6_369 = torch.constant.int 6
    %343 = torch.prims.convert_element_type %334, %int6_369 : !torch.vtensor<[1,1,4,32],f16>, !torch.int -> !torch.vtensor<[1,1,4,32],f32>
    %344 = torch_c.to_builtin_tensor %343 : !torch.vtensor<[1,1,4,32],f32> -> tensor<1x1x4x32xf32>
    %345 = torch_c.to_builtin_tensor %73 : !torch.vtensor<[1,1,32],f32> -> tensor<1x1x32xf32>
    %346 = util.call @sharktank_rotary_embedding_1_1_4_32_f32(%344, %345) : (tensor<1x1x4x32xf32>, tensor<1x1x32xf32>) -> tensor<1x1x4x32xf32>
    %347 = torch_c.from_builtin_tensor %346 : tensor<1x1x4x32xf32> -> !torch.vtensor<[1,1,4,32],f32>
    %int5_370 = torch.constant.int 5
    %348 = torch.prims.convert_element_type %347, %int5_370 : !torch.vtensor<[1,1,4,32],f32>, !torch.int -> !torch.vtensor<[1,1,4,32],f16>
    %int32_371 = torch.constant.int 32
    %349 = torch.aten.floor_divide.Scalar %arg2, %int32_371 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int1_372 = torch.constant.int 1
    %350 = torch.aten.unsqueeze %349, %int1_372 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_373 = torch.constant.int 1
    %false_374 = torch.constant.bool false
    %351 = torch.aten.gather %arg3, %int1_373, %350, %false_374 : !torch.vtensor<[1,?],si64>, !torch.int, !torch.vtensor<[1,1],si64>, !torch.bool -> !torch.vtensor<[1,1],si64>
    %int32_375 = torch.constant.int 32
    %352 = torch.aten.remainder.Scalar %arg2, %int32_375 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int1_376 = torch.constant.int 1
    %353 = torch.aten.unsqueeze %352, %int1_376 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %none_377 = torch.constant.none
    %354 = torch.aten.clone %16, %none_377 : !torch.vtensor<[],si64>, !torch.none -> !torch.vtensor<[],si64>
    %355 = torch.aten.detach %354 : !torch.vtensor<[],si64> -> !torch.vtensor<[],si64>
    %356 = torch.aten.detach %355 : !torch.vtensor<[],si64> -> !torch.vtensor<[],si64>
    %357 = torch.aten.detach %356 : !torch.vtensor<[],si64> -> !torch.vtensor<[],si64>
    %int0_378 = torch.constant.int 0
    %358 = torch.aten.unsqueeze %357, %int0_378 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int1_379 = torch.constant.int 1
    %int1_380 = torch.constant.int 1
    %359 = torch.prim.ListConstruct %int1_379, %int1_380 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_381 = torch.constant.int 1
    %int1_382 = torch.constant.int 1
    %360 = torch.prim.ListConstruct %int1_381, %int1_382 : (!torch.int, !torch.int) -> !torch.list<int>
    %int4_383 = torch.constant.int 4
    %int0_384 = torch.constant.int 0
    %cpu_385 = torch.constant.device "cpu"
    %false_386 = torch.constant.bool false
    %361 = torch.aten.empty_strided %359, %360, %int4_383, %int0_384, %cpu_385, %false_386 : !torch.list<int>, !torch.list<int>, !torch.int, !torch.int, !torch.Device, !torch.bool -> !torch.vtensor<[1,1],si64>
    %int1_387 = torch.constant.int 1
    %362 = torch.aten.fill.Scalar %361, %int1_387 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_388 = torch.constant.int 1
    %int1_389 = torch.constant.int 1
    %363 = torch.prim.ListConstruct %int1_388, %int1_389 : (!torch.int, !torch.int) -> !torch.list<int>
    %364 = torch.aten.repeat %358, %363 : !torch.vtensor<[1],si64>, !torch.list<int> -> !torch.vtensor<[1,1],si64>
    %int3_390 = torch.constant.int 3
    %365 = torch.aten.mul.Scalar %351, %int3_390 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_391 = torch.constant.int 1
    %366 = torch.aten.add.Tensor %365, %362, %int1_391 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int2_392 = torch.constant.int 2
    %367 = torch.aten.mul.Scalar %366, %int2_392 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_393 = torch.constant.int 1
    %368 = torch.aten.add.Tensor %367, %364, %int1_393 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int32_394 = torch.constant.int 32
    %369 = torch.aten.mul.Scalar %368, %int32_394 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_395 = torch.constant.int 1
    %370 = torch.aten.add.Tensor %369, %353, %int1_395 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int5_396 = torch.constant.int 5
    %371 = torch.prims.convert_element_type %348, %int5_396 : !torch.vtensor<[1,1,4,32],f16>, !torch.int -> !torch.vtensor<[1,1,4,32],f16>
    %int3_397 = torch.constant.int 3
    %int2_398 = torch.constant.int 2
    %int32_399 = torch.constant.int 32
    %int4_400 = torch.constant.int 4
    %int32_401 = torch.constant.int 32
    %372 = torch.prim.ListConstruct %40, %int3_397, %int2_398, %int32_399, %int4_400, %int32_401 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %373 = torch.aten.view %196, %372 : !torch.vtensor<[?,24576],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,4,32],f16>
    torch.bind_symbolic_shape %373, [%38], affine_map<()[s0] -> (s0, 3, 2, 32, 4, 32)> : !torch.vtensor<[?,3,2,32,4,32],f16>
    %int4_402 = torch.constant.int 4
    %int32_403 = torch.constant.int 32
    %374 = torch.prim.ListConstruct %132, %int4_402, %int32_403 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %375 = torch.aten.view %373, %374 : !torch.vtensor<[?,3,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,4,32],f16>
    torch.bind_symbolic_shape %375, [%38], affine_map<()[s0] -> (s0 * 192, 4, 32)> : !torch.vtensor<[?,4,32],f16>
    %376 = torch.prim.ListConstruct %370 : (!torch.vtensor<[1,1],si64>) -> !torch.list<optional<vtensor>>
    %false_404 = torch.constant.bool false
    %377 = torch.aten.index_put %375, %376, %371, %false_404 : !torch.vtensor<[?,4,32],f16>, !torch.list<optional<vtensor>>, !torch.vtensor<[1,1,4,32],f16>, !torch.bool -> !torch.vtensor<[?,4,32],f16>
    torch.bind_symbolic_shape %377, [%38], affine_map<()[s0] -> (s0 * 192, 4, 32)> : !torch.vtensor<[?,4,32],f16>
    %int3_405 = torch.constant.int 3
    %int2_406 = torch.constant.int 2
    %int32_407 = torch.constant.int 32
    %int4_408 = torch.constant.int 4
    %int32_409 = torch.constant.int 32
    %378 = torch.prim.ListConstruct %40, %int3_405, %int2_406, %int32_407, %int4_408, %int32_409 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %379 = torch.aten.view %377, %378 : !torch.vtensor<[?,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,4,32],f16>
    torch.bind_symbolic_shape %379, [%38], affine_map<()[s0] -> (s0, 3, 2, 32, 4, 32)> : !torch.vtensor<[?,3,2,32,4,32],f16>
    %int24576_410 = torch.constant.int 24576
    %380 = torch.prim.ListConstruct %40, %int24576_410 : (!torch.int, !torch.int) -> !torch.list<int>
    %381 = torch.aten.view %379, %380 : !torch.vtensor<[?,3,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,24576],f16>
    torch.bind_symbolic_shape %381, [%38], affine_map<()[s0] -> (s0, 24576)> : !torch.vtensor<[?,24576],f16>
    %int3_411 = torch.constant.int 3
    %int2_412 = torch.constant.int 2
    %int32_413 = torch.constant.int 32
    %int4_414 = torch.constant.int 4
    %int32_415 = torch.constant.int 32
    %382 = torch.prim.ListConstruct %40, %int3_411, %int2_412, %int32_413, %int4_414, %int32_415 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %383 = torch.aten.view %381, %382 : !torch.vtensor<[?,24576],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,4,32],f16>
    torch.bind_symbolic_shape %383, [%38], affine_map<()[s0] -> (s0, 3, 2, 32, 4, 32)> : !torch.vtensor<[?,3,2,32,4,32],f16>
    %int4_416 = torch.constant.int 4
    %int32_417 = torch.constant.int 32
    %384 = torch.prim.ListConstruct %132, %int4_416, %int32_417 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %385 = torch.aten.view %383, %384 : !torch.vtensor<[?,3,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,4,32],f16>
    torch.bind_symbolic_shape %385, [%38], affine_map<()[s0] -> (s0 * 192, 4, 32)> : !torch.vtensor<[?,4,32],f16>
    %int32_418 = torch.constant.int 32
    %386 = torch.aten.floor_divide.Scalar %arg2, %int32_418 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int1_419 = torch.constant.int 1
    %387 = torch.aten.unsqueeze %386, %int1_419 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_420 = torch.constant.int 1
    %false_421 = torch.constant.bool false
    %388 = torch.aten.gather %arg3, %int1_420, %387, %false_421 : !torch.vtensor<[1,?],si64>, !torch.int, !torch.vtensor<[1,1],si64>, !torch.bool -> !torch.vtensor<[1,1],si64>
    %int32_422 = torch.constant.int 32
    %389 = torch.aten.remainder.Scalar %arg2, %int32_422 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int1_423 = torch.constant.int 1
    %390 = torch.aten.unsqueeze %389, %int1_423 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %none_424 = torch.constant.none
    %391 = torch.aten.clone %17, %none_424 : !torch.vtensor<[],si64>, !torch.none -> !torch.vtensor<[],si64>
    %392 = torch.aten.detach %391 : !torch.vtensor<[],si64> -> !torch.vtensor<[],si64>
    %393 = torch.aten.detach %392 : !torch.vtensor<[],si64> -> !torch.vtensor<[],si64>
    %394 = torch.aten.detach %393 : !torch.vtensor<[],si64> -> !torch.vtensor<[],si64>
    %int0_425 = torch.constant.int 0
    %395 = torch.aten.unsqueeze %394, %int0_425 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int1_426 = torch.constant.int 1
    %int1_427 = torch.constant.int 1
    %396 = torch.prim.ListConstruct %int1_426, %int1_427 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_428 = torch.constant.int 1
    %int1_429 = torch.constant.int 1
    %397 = torch.prim.ListConstruct %int1_428, %int1_429 : (!torch.int, !torch.int) -> !torch.list<int>
    %int4_430 = torch.constant.int 4
    %int0_431 = torch.constant.int 0
    %cpu_432 = torch.constant.device "cpu"
    %false_433 = torch.constant.bool false
    %398 = torch.aten.empty_strided %396, %397, %int4_430, %int0_431, %cpu_432, %false_433 : !torch.list<int>, !torch.list<int>, !torch.int, !torch.int, !torch.Device, !torch.bool -> !torch.vtensor<[1,1],si64>
    %int1_434 = torch.constant.int 1
    %399 = torch.aten.fill.Scalar %398, %int1_434 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_435 = torch.constant.int 1
    %int1_436 = torch.constant.int 1
    %400 = torch.prim.ListConstruct %int1_435, %int1_436 : (!torch.int, !torch.int) -> !torch.list<int>
    %401 = torch.aten.repeat %395, %400 : !torch.vtensor<[1],si64>, !torch.list<int> -> !torch.vtensor<[1,1],si64>
    %int3_437 = torch.constant.int 3
    %402 = torch.aten.mul.Scalar %388, %int3_437 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_438 = torch.constant.int 1
    %403 = torch.aten.add.Tensor %402, %399, %int1_438 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int2_439 = torch.constant.int 2
    %404 = torch.aten.mul.Scalar %403, %int2_439 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_440 = torch.constant.int 1
    %405 = torch.aten.add.Tensor %404, %401, %int1_440 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int32_441 = torch.constant.int 32
    %406 = torch.aten.mul.Scalar %405, %int32_441 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_442 = torch.constant.int 1
    %407 = torch.aten.add.Tensor %406, %390, %int1_442 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int5_443 = torch.constant.int 5
    %408 = torch.prims.convert_element_type %336, %int5_443 : !torch.vtensor<[1,1,4,32],f16>, !torch.int -> !torch.vtensor<[1,1,4,32],f16>
    %409 = torch.prim.ListConstruct %407 : (!torch.vtensor<[1,1],si64>) -> !torch.list<optional<vtensor>>
    %false_444 = torch.constant.bool false
    %410 = torch.aten.index_put %385, %409, %408, %false_444 : !torch.vtensor<[?,4,32],f16>, !torch.list<optional<vtensor>>, !torch.vtensor<[1,1,4,32],f16>, !torch.bool -> !torch.vtensor<[?,4,32],f16>
    torch.bind_symbolic_shape %410, [%38], affine_map<()[s0] -> (s0 * 192, 4, 32)> : !torch.vtensor<[?,4,32],f16>
    %int3_445 = torch.constant.int 3
    %int2_446 = torch.constant.int 2
    %int32_447 = torch.constant.int 32
    %int4_448 = torch.constant.int 4
    %int32_449 = torch.constant.int 32
    %411 = torch.prim.ListConstruct %40, %int3_445, %int2_446, %int32_447, %int4_448, %int32_449 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %412 = torch.aten.view %410, %411 : !torch.vtensor<[?,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,4,32],f16>
    torch.bind_symbolic_shape %412, [%38], affine_map<()[s0] -> (s0, 3, 2, 32, 4, 32)> : !torch.vtensor<[?,3,2,32,4,32],f16>
    %int24576_450 = torch.constant.int 24576
    %413 = torch.prim.ListConstruct %40, %int24576_450 : (!torch.int, !torch.int) -> !torch.list<int>
    %414 = torch.aten.view %412, %413 : !torch.vtensor<[?,3,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,24576],f16>
    torch.bind_symbolic_shape %414, [%38], affine_map<()[s0] -> (s0, 24576)> : !torch.vtensor<[?,24576],f16>
    %int1_451 = torch.constant.int 1
    %415 = torch.prim.ListConstruct %int1_451, %39 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_452 = torch.constant.int 1
    %416 = torch.prim.ListConstruct %39, %int1_452 : (!torch.int, !torch.int) -> !torch.list<int>
    %int4_453 = torch.constant.int 4
    %int0_454 = torch.constant.int 0
    %cpu_455 = torch.constant.device "cpu"
    %false_456 = torch.constant.bool false
    %417 = torch.aten.empty_strided %415, %416, %int4_453, %int0_454, %cpu_455, %false_456 : !torch.list<int>, !torch.list<int>, !torch.int, !torch.int, !torch.Device, !torch.bool -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %417, [%37], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %int1_457 = torch.constant.int 1
    %418 = torch.aten.fill.Scalar %417, %int1_457 : !torch.vtensor<[1,?],si64>, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %418, [%37], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %int3_458 = torch.constant.int 3
    %419 = torch.aten.mul.Scalar %arg3, %int3_458 : !torch.vtensor<[1,?],si64>, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %419, [%37], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %int1_459 = torch.constant.int 1
    %420 = torch.aten.add.Tensor %419, %418, %int1_459 : !torch.vtensor<[1,?],si64>, !torch.vtensor<[1,?],si64>, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %420, [%37], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %421 = torch.prim.ListConstruct %39 : (!torch.int) -> !torch.list<int>
    %422 = torch.aten.view %420, %421 : !torch.vtensor<[1,?],si64>, !torch.list<int> -> !torch.vtensor<[?],si64>
    torch.bind_symbolic_shape %422, [%37], affine_map<()[s0] -> (s0)> : !torch.vtensor<[?],si64>
    %int3_460 = torch.constant.int 3
    %int2_461 = torch.constant.int 2
    %int32_462 = torch.constant.int 32
    %int4_463 = torch.constant.int 4
    %int32_464 = torch.constant.int 32
    %423 = torch.prim.ListConstruct %40, %int3_460, %int2_461, %int32_462, %int4_463, %int32_464 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %424 = torch.aten.view %414, %423 : !torch.vtensor<[?,24576],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,4,32],f16>
    torch.bind_symbolic_shape %424, [%38], affine_map<()[s0] -> (s0, 3, 2, 32, 4, 32)> : !torch.vtensor<[?,3,2,32,4,32],f16>
    %int2_465 = torch.constant.int 2
    %int32_466 = torch.constant.int 32
    %int4_467 = torch.constant.int 4
    %int32_468 = torch.constant.int 32
    %425 = torch.prim.ListConstruct %130, %int2_465, %int32_466, %int4_467, %int32_468 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %426 = torch.aten.view %424, %425 : !torch.vtensor<[?,3,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,2,32,4,32],f16>
    torch.bind_symbolic_shape %426, [%38], affine_map<()[s0] -> (s0 * 3, 2, 32, 4, 32)> : !torch.vtensor<[?,2,32,4,32],f16>
    %int0_469 = torch.constant.int 0
    %427 = torch.aten.index_select %426, %int0_469, %422 : !torch.vtensor<[?,2,32,4,32],f16>, !torch.int, !torch.vtensor<[?],si64> -> !torch.vtensor<[?,2,32,4,32],f16>
    torch.bind_symbolic_shape %427, [%37], affine_map<()[s0] -> (s0, 2, 32, 4, 32)> : !torch.vtensor<[?,2,32,4,32],f16>
    %int1_470 = torch.constant.int 1
    %int2_471 = torch.constant.int 2
    %int32_472 = torch.constant.int 32
    %int4_473 = torch.constant.int 4
    %int32_474 = torch.constant.int 32
    %428 = torch.prim.ListConstruct %int1_470, %39, %int2_471, %int32_472, %int4_473, %int32_474 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %429 = torch.aten.view %427, %428 : !torch.vtensor<[?,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,2,32,4,32],f16>
    torch.bind_symbolic_shape %429, [%37], affine_map<()[s0] -> (1, s0, 2, 32, 4, 32)> : !torch.vtensor<[1,?,2,32,4,32],f16>
    %int0_475 = torch.constant.int 0
    %int0_476 = torch.constant.int 0
    %int9223372036854775807_477 = torch.constant.int 9223372036854775807
    %int1_478 = torch.constant.int 1
    %430 = torch.aten.slice.Tensor %429, %int0_475, %int0_476, %int9223372036854775807_477, %int1_478 : !torch.vtensor<[1,?,2,32,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,2,32,4,32],f16>
    torch.bind_symbolic_shape %430, [%37], affine_map<()[s0] -> (1, s0, 2, 32, 4, 32)> : !torch.vtensor<[1,?,2,32,4,32],f16>
    %int1_479 = torch.constant.int 1
    %int0_480 = torch.constant.int 0
    %int9223372036854775807_481 = torch.constant.int 9223372036854775807
    %int1_482 = torch.constant.int 1
    %431 = torch.aten.slice.Tensor %430, %int1_479, %int0_480, %int9223372036854775807_481, %int1_482 : !torch.vtensor<[1,?,2,32,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,2,32,4,32],f16>
    torch.bind_symbolic_shape %431, [%37], affine_map<()[s0] -> (1, s0, 2, 32, 4, 32)> : !torch.vtensor<[1,?,2,32,4,32],f16>
    %int2_483 = torch.constant.int 2
    %int0_484 = torch.constant.int 0
    %432 = torch.aten.select.int %431, %int2_483, %int0_484 : !torch.vtensor<[1,?,2,32,4,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,?,32,4,32],f16>
    torch.bind_symbolic_shape %432, [%37], affine_map<()[s0] -> (1, s0, 32, 4, 32)> : !torch.vtensor<[1,?,32,4,32],f16>
    %int2_485 = torch.constant.int 2
    %int0_486 = torch.constant.int 0
    %int1_487 = torch.constant.int 1
    %433 = torch.aten.slice.Tensor %432, %int2_485, %int0_486, %41, %int1_487 : !torch.vtensor<[1,?,32,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32,4,32],f16>
    torch.bind_symbolic_shape %433, [%37], affine_map<()[s0] -> (1, s0, 32, 4, 32)> : !torch.vtensor<[1,?,32,4,32],f16>
    %int0_488 = torch.constant.int 0
    %434 = torch.aten.clone %433, %int0_488 : !torch.vtensor<[1,?,32,4,32],f16>, !torch.int -> !torch.vtensor<[1,?,32,4,32],f16>
    torch.bind_symbolic_shape %434, [%37], affine_map<()[s0] -> (1, s0, 32, 4, 32)> : !torch.vtensor<[1,?,32,4,32],f16>
    %int1_489 = torch.constant.int 1
    %int4_490 = torch.constant.int 4
    %int32_491 = torch.constant.int 32
    %435 = torch.prim.ListConstruct %int1_489, %41, %int4_490, %int32_491 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %436 = torch.aten._unsafe_view %434, %435 : !torch.vtensor<[1,?,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %436, [%37], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int0_492 = torch.constant.int 0
    %int0_493 = torch.constant.int 0
    %int9223372036854775807_494 = torch.constant.int 9223372036854775807
    %int1_495 = torch.constant.int 1
    %437 = torch.aten.slice.Tensor %436, %int0_492, %int0_493, %int9223372036854775807_494, %int1_495 : !torch.vtensor<[1,?,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %437, [%37], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int0_496 = torch.constant.int 0
    %int0_497 = torch.constant.int 0
    %int9223372036854775807_498 = torch.constant.int 9223372036854775807
    %int1_499 = torch.constant.int 1
    %438 = torch.aten.slice.Tensor %429, %int0_496, %int0_497, %int9223372036854775807_498, %int1_499 : !torch.vtensor<[1,?,2,32,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,2,32,4,32],f16>
    torch.bind_symbolic_shape %438, [%37], affine_map<()[s0] -> (1, s0, 2, 32, 4, 32)> : !torch.vtensor<[1,?,2,32,4,32],f16>
    %int1_500 = torch.constant.int 1
    %int0_501 = torch.constant.int 0
    %int9223372036854775807_502 = torch.constant.int 9223372036854775807
    %int1_503 = torch.constant.int 1
    %439 = torch.aten.slice.Tensor %438, %int1_500, %int0_501, %int9223372036854775807_502, %int1_503 : !torch.vtensor<[1,?,2,32,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,2,32,4,32],f16>
    torch.bind_symbolic_shape %439, [%37], affine_map<()[s0] -> (1, s0, 2, 32, 4, 32)> : !torch.vtensor<[1,?,2,32,4,32],f16>
    %int2_504 = torch.constant.int 2
    %int1_505 = torch.constant.int 1
    %440 = torch.aten.select.int %439, %int2_504, %int1_505 : !torch.vtensor<[1,?,2,32,4,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,?,32,4,32],f16>
    torch.bind_symbolic_shape %440, [%37], affine_map<()[s0] -> (1, s0, 32, 4, 32)> : !torch.vtensor<[1,?,32,4,32],f16>
    %int2_506 = torch.constant.int 2
    %int0_507 = torch.constant.int 0
    %int1_508 = torch.constant.int 1
    %441 = torch.aten.slice.Tensor %440, %int2_506, %int0_507, %41, %int1_508 : !torch.vtensor<[1,?,32,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32,4,32],f16>
    torch.bind_symbolic_shape %441, [%37], affine_map<()[s0] -> (1, s0, 32, 4, 32)> : !torch.vtensor<[1,?,32,4,32],f16>
    %int0_509 = torch.constant.int 0
    %442 = torch.aten.clone %441, %int0_509 : !torch.vtensor<[1,?,32,4,32],f16>, !torch.int -> !torch.vtensor<[1,?,32,4,32],f16>
    torch.bind_symbolic_shape %442, [%37], affine_map<()[s0] -> (1, s0, 32, 4, 32)> : !torch.vtensor<[1,?,32,4,32],f16>
    %int1_510 = torch.constant.int 1
    %int4_511 = torch.constant.int 4
    %int32_512 = torch.constant.int 32
    %443 = torch.prim.ListConstruct %int1_510, %41, %int4_511, %int32_512 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %444 = torch.aten._unsafe_view %442, %443 : !torch.vtensor<[1,?,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %444, [%37], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int0_513 = torch.constant.int 0
    %int0_514 = torch.constant.int 0
    %int9223372036854775807_515 = torch.constant.int 9223372036854775807
    %int1_516 = torch.constant.int 1
    %445 = torch.aten.slice.Tensor %444, %int0_513, %int0_514, %int9223372036854775807_515, %int1_516 : !torch.vtensor<[1,?,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %445, [%37], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int-2_517 = torch.constant.int -2
    %446 = torch.aten.unsqueeze %437, %int-2_517 : !torch.vtensor<[1,?,4,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,1,32],f16>
    torch.bind_symbolic_shape %446, [%37], affine_map<()[s0] -> (1, s0 * 32, 4, 1, 32)> : !torch.vtensor<[1,?,4,1,32],f16>
    %int1_518 = torch.constant.int 1
    %int4_519 = torch.constant.int 4
    %int2_520 = torch.constant.int 2
    %int32_521 = torch.constant.int 32
    %447 = torch.prim.ListConstruct %int1_518, %41, %int4_519, %int2_520, %int32_521 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_522 = torch.constant.bool false
    %448 = torch.aten.expand %446, %447, %false_522 : !torch.vtensor<[1,?,4,1,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,?,4,2,32],f16>
    torch.bind_symbolic_shape %448, [%37], affine_map<()[s0] -> (1, s0 * 32, 4, 2, 32)> : !torch.vtensor<[1,?,4,2,32],f16>
    %int0_523 = torch.constant.int 0
    %449 = torch.aten.clone %448, %int0_523 : !torch.vtensor<[1,?,4,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,2,32],f16>
    torch.bind_symbolic_shape %449, [%37], affine_map<()[s0] -> (1, s0 * 32, 4, 2, 32)> : !torch.vtensor<[1,?,4,2,32],f16>
    %int1_524 = torch.constant.int 1
    %int8_525 = torch.constant.int 8
    %int32_526 = torch.constant.int 32
    %450 = torch.prim.ListConstruct %int1_524, %41, %int8_525, %int32_526 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %451 = torch.aten._unsafe_view %449, %450 : !torch.vtensor<[1,?,4,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,8,32],f16>
    torch.bind_symbolic_shape %451, [%37], affine_map<()[s0] -> (1, s0 * 32, 8, 32)> : !torch.vtensor<[1,?,8,32],f16>
    %int-2_527 = torch.constant.int -2
    %452 = torch.aten.unsqueeze %445, %int-2_527 : !torch.vtensor<[1,?,4,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,1,32],f16>
    torch.bind_symbolic_shape %452, [%37], affine_map<()[s0] -> (1, s0 * 32, 4, 1, 32)> : !torch.vtensor<[1,?,4,1,32],f16>
    %int1_528 = torch.constant.int 1
    %int4_529 = torch.constant.int 4
    %int2_530 = torch.constant.int 2
    %int32_531 = torch.constant.int 32
    %453 = torch.prim.ListConstruct %int1_528, %41, %int4_529, %int2_530, %int32_531 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_532 = torch.constant.bool false
    %454 = torch.aten.expand %452, %453, %false_532 : !torch.vtensor<[1,?,4,1,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,?,4,2,32],f16>
    torch.bind_symbolic_shape %454, [%37], affine_map<()[s0] -> (1, s0 * 32, 4, 2, 32)> : !torch.vtensor<[1,?,4,2,32],f16>
    %int0_533 = torch.constant.int 0
    %455 = torch.aten.clone %454, %int0_533 : !torch.vtensor<[1,?,4,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,2,32],f16>
    torch.bind_symbolic_shape %455, [%37], affine_map<()[s0] -> (1, s0 * 32, 4, 2, 32)> : !torch.vtensor<[1,?,4,2,32],f16>
    %int1_534 = torch.constant.int 1
    %int8_535 = torch.constant.int 8
    %int32_536 = torch.constant.int 32
    %456 = torch.prim.ListConstruct %int1_534, %41, %int8_535, %int32_536 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %457 = torch.aten._unsafe_view %455, %456 : !torch.vtensor<[1,?,4,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,8,32],f16>
    torch.bind_symbolic_shape %457, [%37], affine_map<()[s0] -> (1, s0 * 32, 8, 32)> : !torch.vtensor<[1,?,8,32],f16>
    %int1_537 = torch.constant.int 1
    %int2_538 = torch.constant.int 2
    %458 = torch.aten.transpose.int %342, %int1_537, %int2_538 : !torch.vtensor<[1,1,8,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,8,1,32],f16>
    %int1_539 = torch.constant.int 1
    %int2_540 = torch.constant.int 2
    %459 = torch.aten.transpose.int %451, %int1_539, %int2_540 : !torch.vtensor<[1,?,8,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,8,?,32],f16>
    torch.bind_symbolic_shape %459, [%37], affine_map<()[s0] -> (1, 8, s0 * 32, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %int1_541 = torch.constant.int 1
    %int2_542 = torch.constant.int 2
    %460 = torch.aten.transpose.int %457, %int1_541, %int2_542 : !torch.vtensor<[1,?,8,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,8,?,32],f16>
    torch.bind_symbolic_shape %460, [%37], affine_map<()[s0] -> (1, 8, s0 * 32, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %int5_543 = torch.constant.int 5
    %461 = torch.prims.convert_element_type %458, %int5_543 : !torch.vtensor<[1,8,1,32],f16>, !torch.int -> !torch.vtensor<[1,8,1,32],f16>
    %int5_544 = torch.constant.int 5
    %462 = torch.prims.convert_element_type %459, %int5_544 : !torch.vtensor<[1,8,?,32],f16>, !torch.int -> !torch.vtensor<[1,8,?,32],f16>
    torch.bind_symbolic_shape %462, [%37], affine_map<()[s0] -> (1, 8, s0 * 32, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %int5_545 = torch.constant.int 5
    %463 = torch.prims.convert_element_type %460, %int5_545 : !torch.vtensor<[1,8,?,32],f16>, !torch.int -> !torch.vtensor<[1,8,?,32],f16>
    torch.bind_symbolic_shape %463, [%37], affine_map<()[s0] -> (1, 8, s0 * 32, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %int5_546 = torch.constant.int 5
    %464 = torch.prims.convert_element_type %51, %int5_546 : !torch.vtensor<[1,1,1,?],f16>, !torch.int -> !torch.vtensor<[1,1,1,?],f16>
    torch.bind_symbolic_shape %464, [%37], affine_map<()[s0] -> (1, 1, 1, s0 * 32)> : !torch.vtensor<[1,1,1,?],f16>
    %float0.000000e00_547 = torch.constant.float 0.000000e+00
    %false_548 = torch.constant.bool false
    %none_549 = torch.constant.none
    %465:2 = torch.operator "torch.aten._scaled_dot_product_flash_attention_for_cpu"(%461, %462, %463, %float0.000000e00_547, %false_548, %464, %none_549) : (!torch.vtensor<[1,8,1,32],f16>, !torch.vtensor<[1,8,?,32],f16>, !torch.vtensor<[1,8,?,32],f16>, !torch.float, !torch.bool, !torch.vtensor<[1,1,1,?],f16>, !torch.none) -> (!torch.vtensor<[1,8,1,32],f16>, !torch.vtensor<[1,8,1],f32>) 
    %int1_550 = torch.constant.int 1
    %int2_551 = torch.constant.int 2
    %466 = torch.aten.transpose.int %465#0, %int1_550, %int2_551 : !torch.vtensor<[1,8,1,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,1,8,32],f16>
    %int1_552 = torch.constant.int 1
    %int1_553 = torch.constant.int 1
    %int256_554 = torch.constant.int 256
    %467 = torch.prim.ListConstruct %int1_552, %int1_553, %int256_554 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %468 = torch.aten.view %466, %467 : !torch.vtensor<[1,1,8,32],f16>, !torch.list<int> -> !torch.vtensor<[1,1,256],f16>
    %int5_555 = torch.constant.int 5
    %469 = torch.prims.convert_element_type %18, %int5_555 : !torch.vtensor<[256,256],f32>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int-2_556 = torch.constant.int -2
    %int-1_557 = torch.constant.int -1
    %470 = torch.aten.transpose.int %469, %int-2_556, %int-1_557 : !torch.vtensor<[256,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,256],f16>
    %int5_558 = torch.constant.int 5
    %471 = torch.prims.convert_element_type %470, %int5_558 : !torch.vtensor<[256,256],f16>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int1_559 = torch.constant.int 1
    %int256_560 = torch.constant.int 256
    %472 = torch.prim.ListConstruct %int1_559, %int256_560 : (!torch.int, !torch.int) -> !torch.list<int>
    %473 = torch.aten.view %468, %472 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %474 = torch.aten.mm %473, %471 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,256],f16> -> !torch.vtensor<[1,256],f16>
    %int1_561 = torch.constant.int 1
    %int1_562 = torch.constant.int 1
    %int256_563 = torch.constant.int 256
    %475 = torch.prim.ListConstruct %int1_561, %int1_562, %int256_563 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %476 = torch.aten.view %474, %475 : !torch.vtensor<[1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,1,256],f16>
    %int1_564 = torch.constant.int 1
    %477 = torch.aten.add.Tensor %296, %476, %int1_564 : !torch.vtensor<[1,1,256],f16>, !torch.vtensor<[1,1,256],f16>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %int6_565 = torch.constant.int 6
    %478 = torch.prims.convert_element_type %477, %int6_565 : !torch.vtensor<[1,1,256],f16>, !torch.int -> !torch.vtensor<[1,1,256],f32>
    %int2_566 = torch.constant.int 2
    %479 = torch.aten.pow.Tensor_Scalar %478, %int2_566 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f32>
    %int-1_567 = torch.constant.int -1
    %480 = torch.prim.ListConstruct %int-1_567 : (!torch.int) -> !torch.list<int>
    %true_568 = torch.constant.bool true
    %none_569 = torch.constant.none
    %481 = torch.aten.mean.dim %479, %480, %true_568, %none_569 : !torch.vtensor<[1,1,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1],f32>
    %float1.000000e-02_570 = torch.constant.float 1.000000e-02
    %int1_571 = torch.constant.int 1
    %482 = torch.aten.add.Scalar %481, %float1.000000e-02_570, %int1_571 : !torch.vtensor<[1,1,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,1,1],f32>
    %483 = torch.aten.rsqrt %482 : !torch.vtensor<[1,1,1],f32> -> !torch.vtensor<[1,1,1],f32>
    %484 = torch.aten.mul.Tensor %478, %483 : !torch.vtensor<[1,1,256],f32>, !torch.vtensor<[1,1,1],f32> -> !torch.vtensor<[1,1,256],f32>
    %int5_572 = torch.constant.int 5
    %485 = torch.prims.convert_element_type %484, %int5_572 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %486 = torch.aten.mul.Tensor %19, %485 : !torch.vtensor<[256],f32>, !torch.vtensor<[1,1,256],f16> -> !torch.vtensor<[1,1,256],f32>
    %int5_573 = torch.constant.int 5
    %487 = torch.prims.convert_element_type %486, %int5_573 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %int5_574 = torch.constant.int 5
    %488 = torch.prims.convert_element_type %20, %int5_574 : !torch.vtensor<[23,256],f32>, !torch.int -> !torch.vtensor<[23,256],f16>
    %int-2_575 = torch.constant.int -2
    %int-1_576 = torch.constant.int -1
    %489 = torch.aten.transpose.int %488, %int-2_575, %int-1_576 : !torch.vtensor<[23,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,23],f16>
    %int5_577 = torch.constant.int 5
    %490 = torch.prims.convert_element_type %489, %int5_577 : !torch.vtensor<[256,23],f16>, !torch.int -> !torch.vtensor<[256,23],f16>
    %int1_578 = torch.constant.int 1
    %int256_579 = torch.constant.int 256
    %491 = torch.prim.ListConstruct %int1_578, %int256_579 : (!torch.int, !torch.int) -> !torch.list<int>
    %492 = torch.aten.view %487, %491 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %493 = torch.aten.mm %492, %490 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,23],f16> -> !torch.vtensor<[1,23],f16>
    %int1_580 = torch.constant.int 1
    %int1_581 = torch.constant.int 1
    %int23_582 = torch.constant.int 23
    %494 = torch.prim.ListConstruct %int1_580, %int1_581, %int23_582 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %495 = torch.aten.view %493, %494 : !torch.vtensor<[1,23],f16>, !torch.list<int> -> !torch.vtensor<[1,1,23],f16>
    %496 = torch.aten.silu %495 : !torch.vtensor<[1,1,23],f16> -> !torch.vtensor<[1,1,23],f16>
    %int5_583 = torch.constant.int 5
    %497 = torch.prims.convert_element_type %21, %int5_583 : !torch.vtensor<[23,256],f32>, !torch.int -> !torch.vtensor<[23,256],f16>
    %int-2_584 = torch.constant.int -2
    %int-1_585 = torch.constant.int -1
    %498 = torch.aten.transpose.int %497, %int-2_584, %int-1_585 : !torch.vtensor<[23,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,23],f16>
    %int5_586 = torch.constant.int 5
    %499 = torch.prims.convert_element_type %498, %int5_586 : !torch.vtensor<[256,23],f16>, !torch.int -> !torch.vtensor<[256,23],f16>
    %int1_587 = torch.constant.int 1
    %int256_588 = torch.constant.int 256
    %500 = torch.prim.ListConstruct %int1_587, %int256_588 : (!torch.int, !torch.int) -> !torch.list<int>
    %501 = torch.aten.view %487, %500 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %502 = torch.aten.mm %501, %499 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,23],f16> -> !torch.vtensor<[1,23],f16>
    %int1_589 = torch.constant.int 1
    %int1_590 = torch.constant.int 1
    %int23_591 = torch.constant.int 23
    %503 = torch.prim.ListConstruct %int1_589, %int1_590, %int23_591 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %504 = torch.aten.view %502, %503 : !torch.vtensor<[1,23],f16>, !torch.list<int> -> !torch.vtensor<[1,1,23],f16>
    %505 = torch.aten.mul.Tensor %496, %504 : !torch.vtensor<[1,1,23],f16>, !torch.vtensor<[1,1,23],f16> -> !torch.vtensor<[1,1,23],f16>
    %int5_592 = torch.constant.int 5
    %506 = torch.prims.convert_element_type %22, %int5_592 : !torch.vtensor<[256,23],f32>, !torch.int -> !torch.vtensor<[256,23],f16>
    %int-2_593 = torch.constant.int -2
    %int-1_594 = torch.constant.int -1
    %507 = torch.aten.transpose.int %506, %int-2_593, %int-1_594 : !torch.vtensor<[256,23],f16>, !torch.int, !torch.int -> !torch.vtensor<[23,256],f16>
    %int5_595 = torch.constant.int 5
    %508 = torch.prims.convert_element_type %507, %int5_595 : !torch.vtensor<[23,256],f16>, !torch.int -> !torch.vtensor<[23,256],f16>
    %int1_596 = torch.constant.int 1
    %int23_597 = torch.constant.int 23
    %509 = torch.prim.ListConstruct %int1_596, %int23_597 : (!torch.int, !torch.int) -> !torch.list<int>
    %510 = torch.aten.view %505, %509 : !torch.vtensor<[1,1,23],f16>, !torch.list<int> -> !torch.vtensor<[1,23],f16>
    %511 = torch.aten.mm %510, %508 : !torch.vtensor<[1,23],f16>, !torch.vtensor<[23,256],f16> -> !torch.vtensor<[1,256],f16>
    %int1_598 = torch.constant.int 1
    %int1_599 = torch.constant.int 1
    %int256_600 = torch.constant.int 256
    %512 = torch.prim.ListConstruct %int1_598, %int1_599, %int256_600 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %513 = torch.aten.view %511, %512 : !torch.vtensor<[1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,1,256],f16>
    %int1_601 = torch.constant.int 1
    %514 = torch.aten.add.Tensor %477, %513, %int1_601 : !torch.vtensor<[1,1,256],f16>, !torch.vtensor<[1,1,256],f16>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %int6_602 = torch.constant.int 6
    %515 = torch.prims.convert_element_type %514, %int6_602 : !torch.vtensor<[1,1,256],f16>, !torch.int -> !torch.vtensor<[1,1,256],f32>
    %int2_603 = torch.constant.int 2
    %516 = torch.aten.pow.Tensor_Scalar %515, %int2_603 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f32>
    %int-1_604 = torch.constant.int -1
    %517 = torch.prim.ListConstruct %int-1_604 : (!torch.int) -> !torch.list<int>
    %true_605 = torch.constant.bool true
    %none_606 = torch.constant.none
    %518 = torch.aten.mean.dim %516, %517, %true_605, %none_606 : !torch.vtensor<[1,1,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1],f32>
    %float1.000000e-02_607 = torch.constant.float 1.000000e-02
    %int1_608 = torch.constant.int 1
    %519 = torch.aten.add.Scalar %518, %float1.000000e-02_607, %int1_608 : !torch.vtensor<[1,1,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,1,1],f32>
    %520 = torch.aten.rsqrt %519 : !torch.vtensor<[1,1,1],f32> -> !torch.vtensor<[1,1,1],f32>
    %521 = torch.aten.mul.Tensor %515, %520 : !torch.vtensor<[1,1,256],f32>, !torch.vtensor<[1,1,1],f32> -> !torch.vtensor<[1,1,256],f32>
    %int5_609 = torch.constant.int 5
    %522 = torch.prims.convert_element_type %521, %int5_609 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %523 = torch.aten.mul.Tensor %23, %522 : !torch.vtensor<[256],f32>, !torch.vtensor<[1,1,256],f16> -> !torch.vtensor<[1,1,256],f32>
    %int5_610 = torch.constant.int 5
    %524 = torch.prims.convert_element_type %523, %int5_610 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %int5_611 = torch.constant.int 5
    %525 = torch.prims.convert_element_type %24, %int5_611 : !torch.vtensor<[256,256],f32>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int-2_612 = torch.constant.int -2
    %int-1_613 = torch.constant.int -1
    %526 = torch.aten.transpose.int %525, %int-2_612, %int-1_613 : !torch.vtensor<[256,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,256],f16>
    %int5_614 = torch.constant.int 5
    %527 = torch.prims.convert_element_type %526, %int5_614 : !torch.vtensor<[256,256],f16>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int1_615 = torch.constant.int 1
    %int256_616 = torch.constant.int 256
    %528 = torch.prim.ListConstruct %int1_615, %int256_616 : (!torch.int, !torch.int) -> !torch.list<int>
    %529 = torch.aten.view %524, %528 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %530 = torch.aten.mm %529, %527 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,256],f16> -> !torch.vtensor<[1,256],f16>
    %int1_617 = torch.constant.int 1
    %int1_618 = torch.constant.int 1
    %int256_619 = torch.constant.int 256
    %531 = torch.prim.ListConstruct %int1_617, %int1_618, %int256_619 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %532 = torch.aten.view %530, %531 : !torch.vtensor<[1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,1,256],f16>
    %int5_620 = torch.constant.int 5
    %533 = torch.prims.convert_element_type %25, %int5_620 : !torch.vtensor<[128,256],f32>, !torch.int -> !torch.vtensor<[128,256],f16>
    %int-2_621 = torch.constant.int -2
    %int-1_622 = torch.constant.int -1
    %534 = torch.aten.transpose.int %533, %int-2_621, %int-1_622 : !torch.vtensor<[128,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,128],f16>
    %int5_623 = torch.constant.int 5
    %535 = torch.prims.convert_element_type %534, %int5_623 : !torch.vtensor<[256,128],f16>, !torch.int -> !torch.vtensor<[256,128],f16>
    %int1_624 = torch.constant.int 1
    %int256_625 = torch.constant.int 256
    %536 = torch.prim.ListConstruct %int1_624, %int256_625 : (!torch.int, !torch.int) -> !torch.list<int>
    %537 = torch.aten.view %524, %536 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %538 = torch.aten.mm %537, %535 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,128],f16> -> !torch.vtensor<[1,128],f16>
    %int1_626 = torch.constant.int 1
    %int1_627 = torch.constant.int 1
    %int128_628 = torch.constant.int 128
    %539 = torch.prim.ListConstruct %int1_626, %int1_627, %int128_628 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %540 = torch.aten.view %538, %539 : !torch.vtensor<[1,128],f16>, !torch.list<int> -> !torch.vtensor<[1,1,128],f16>
    %int5_629 = torch.constant.int 5
    %541 = torch.prims.convert_element_type %26, %int5_629 : !torch.vtensor<[128,256],f32>, !torch.int -> !torch.vtensor<[128,256],f16>
    %int-2_630 = torch.constant.int -2
    %int-1_631 = torch.constant.int -1
    %542 = torch.aten.transpose.int %541, %int-2_630, %int-1_631 : !torch.vtensor<[128,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,128],f16>
    %int5_632 = torch.constant.int 5
    %543 = torch.prims.convert_element_type %542, %int5_632 : !torch.vtensor<[256,128],f16>, !torch.int -> !torch.vtensor<[256,128],f16>
    %int1_633 = torch.constant.int 1
    %int256_634 = torch.constant.int 256
    %544 = torch.prim.ListConstruct %int1_633, %int256_634 : (!torch.int, !torch.int) -> !torch.list<int>
    %545 = torch.aten.view %524, %544 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %546 = torch.aten.mm %545, %543 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,128],f16> -> !torch.vtensor<[1,128],f16>
    %int1_635 = torch.constant.int 1
    %int1_636 = torch.constant.int 1
    %int128_637 = torch.constant.int 128
    %547 = torch.prim.ListConstruct %int1_635, %int1_636, %int128_637 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %548 = torch.aten.view %546, %547 : !torch.vtensor<[1,128],f16>, !torch.list<int> -> !torch.vtensor<[1,1,128],f16>
    %int1_638 = torch.constant.int 1
    %int1_639 = torch.constant.int 1
    %int8_640 = torch.constant.int 8
    %int32_641 = torch.constant.int 32
    %549 = torch.prim.ListConstruct %int1_638, %int1_639, %int8_640, %int32_641 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %550 = torch.aten.view %532, %549 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,1,8,32],f16>
    %int1_642 = torch.constant.int 1
    %int1_643 = torch.constant.int 1
    %int4_644 = torch.constant.int 4
    %int32_645 = torch.constant.int 32
    %551 = torch.prim.ListConstruct %int1_642, %int1_643, %int4_644, %int32_645 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %552 = torch.aten.view %540, %551 : !torch.vtensor<[1,1,128],f16>, !torch.list<int> -> !torch.vtensor<[1,1,4,32],f16>
    %int1_646 = torch.constant.int 1
    %int1_647 = torch.constant.int 1
    %int4_648 = torch.constant.int 4
    %int32_649 = torch.constant.int 32
    %553 = torch.prim.ListConstruct %int1_646, %int1_647, %int4_648, %int32_649 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %554 = torch.aten.view %548, %553 : !torch.vtensor<[1,1,128],f16>, !torch.list<int> -> !torch.vtensor<[1,1,4,32],f16>
    %int6_650 = torch.constant.int 6
    %555 = torch.prims.convert_element_type %550, %int6_650 : !torch.vtensor<[1,1,8,32],f16>, !torch.int -> !torch.vtensor<[1,1,8,32],f32>
    %556 = torch_c.to_builtin_tensor %555 : !torch.vtensor<[1,1,8,32],f32> -> tensor<1x1x8x32xf32>
    %557 = torch_c.to_builtin_tensor %73 : !torch.vtensor<[1,1,32],f32> -> tensor<1x1x32xf32>
    %558 = util.call @sharktank_rotary_embedding_1_1_8_32_f32(%556, %557) : (tensor<1x1x8x32xf32>, tensor<1x1x32xf32>) -> tensor<1x1x8x32xf32>
    %559 = torch_c.from_builtin_tensor %558 : tensor<1x1x8x32xf32> -> !torch.vtensor<[1,1,8,32],f32>
    %int5_651 = torch.constant.int 5
    %560 = torch.prims.convert_element_type %559, %int5_651 : !torch.vtensor<[1,1,8,32],f32>, !torch.int -> !torch.vtensor<[1,1,8,32],f16>
    %int6_652 = torch.constant.int 6
    %561 = torch.prims.convert_element_type %552, %int6_652 : !torch.vtensor<[1,1,4,32],f16>, !torch.int -> !torch.vtensor<[1,1,4,32],f32>
    %562 = torch_c.to_builtin_tensor %561 : !torch.vtensor<[1,1,4,32],f32> -> tensor<1x1x4x32xf32>
    %563 = torch_c.to_builtin_tensor %73 : !torch.vtensor<[1,1,32],f32> -> tensor<1x1x32xf32>
    %564 = util.call @sharktank_rotary_embedding_1_1_4_32_f32(%562, %563) : (tensor<1x1x4x32xf32>, tensor<1x1x32xf32>) -> tensor<1x1x4x32xf32>
    %565 = torch_c.from_builtin_tensor %564 : tensor<1x1x4x32xf32> -> !torch.vtensor<[1,1,4,32],f32>
    %int5_653 = torch.constant.int 5
    %566 = torch.prims.convert_element_type %565, %int5_653 : !torch.vtensor<[1,1,4,32],f32>, !torch.int -> !torch.vtensor<[1,1,4,32],f16>
    %int32_654 = torch.constant.int 32
    %567 = torch.aten.floor_divide.Scalar %arg2, %int32_654 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int1_655 = torch.constant.int 1
    %568 = torch.aten.unsqueeze %567, %int1_655 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_656 = torch.constant.int 1
    %false_657 = torch.constant.bool false
    %569 = torch.aten.gather %arg3, %int1_656, %568, %false_657 : !torch.vtensor<[1,?],si64>, !torch.int, !torch.vtensor<[1,1],si64>, !torch.bool -> !torch.vtensor<[1,1],si64>
    %int32_658 = torch.constant.int 32
    %570 = torch.aten.remainder.Scalar %arg2, %int32_658 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int1_659 = torch.constant.int 1
    %571 = torch.aten.unsqueeze %570, %int1_659 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %none_660 = torch.constant.none
    %572 = torch.aten.clone %27, %none_660 : !torch.vtensor<[],si64>, !torch.none -> !torch.vtensor<[],si64>
    %573 = torch.aten.detach %572 : !torch.vtensor<[],si64> -> !torch.vtensor<[],si64>
    %574 = torch.aten.detach %573 : !torch.vtensor<[],si64> -> !torch.vtensor<[],si64>
    %575 = torch.aten.detach %574 : !torch.vtensor<[],si64> -> !torch.vtensor<[],si64>
    %int0_661 = torch.constant.int 0
    %576 = torch.aten.unsqueeze %575, %int0_661 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int1_662 = torch.constant.int 1
    %int1_663 = torch.constant.int 1
    %577 = torch.prim.ListConstruct %int1_662, %int1_663 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_664 = torch.constant.int 1
    %int1_665 = torch.constant.int 1
    %578 = torch.prim.ListConstruct %int1_664, %int1_665 : (!torch.int, !torch.int) -> !torch.list<int>
    %int4_666 = torch.constant.int 4
    %int0_667 = torch.constant.int 0
    %cpu_668 = torch.constant.device "cpu"
    %false_669 = torch.constant.bool false
    %579 = torch.aten.empty_strided %577, %578, %int4_666, %int0_667, %cpu_668, %false_669 : !torch.list<int>, !torch.list<int>, !torch.int, !torch.int, !torch.Device, !torch.bool -> !torch.vtensor<[1,1],si64>
    %int2_670 = torch.constant.int 2
    %580 = torch.aten.fill.Scalar %579, %int2_670 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_671 = torch.constant.int 1
    %int1_672 = torch.constant.int 1
    %581 = torch.prim.ListConstruct %int1_671, %int1_672 : (!torch.int, !torch.int) -> !torch.list<int>
    %582 = torch.aten.repeat %576, %581 : !torch.vtensor<[1],si64>, !torch.list<int> -> !torch.vtensor<[1,1],si64>
    %int3_673 = torch.constant.int 3
    %583 = torch.aten.mul.Scalar %569, %int3_673 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_674 = torch.constant.int 1
    %584 = torch.aten.add.Tensor %583, %580, %int1_674 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int2_675 = torch.constant.int 2
    %585 = torch.aten.mul.Scalar %584, %int2_675 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_676 = torch.constant.int 1
    %586 = torch.aten.add.Tensor %585, %582, %int1_676 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int32_677 = torch.constant.int 32
    %587 = torch.aten.mul.Scalar %586, %int32_677 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_678 = torch.constant.int 1
    %588 = torch.aten.add.Tensor %587, %571, %int1_678 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int5_679 = torch.constant.int 5
    %589 = torch.prims.convert_element_type %566, %int5_679 : !torch.vtensor<[1,1,4,32],f16>, !torch.int -> !torch.vtensor<[1,1,4,32],f16>
    %int3_680 = torch.constant.int 3
    %int2_681 = torch.constant.int 2
    %int32_682 = torch.constant.int 32
    %int4_683 = torch.constant.int 4
    %int32_684 = torch.constant.int 32
    %590 = torch.prim.ListConstruct %40, %int3_680, %int2_681, %int32_682, %int4_683, %int32_684 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %591 = torch.aten.view %414, %590 : !torch.vtensor<[?,24576],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,4,32],f16>
    torch.bind_symbolic_shape %591, [%38], affine_map<()[s0] -> (s0, 3, 2, 32, 4, 32)> : !torch.vtensor<[?,3,2,32,4,32],f16>
    %int4_685 = torch.constant.int 4
    %int32_686 = torch.constant.int 32
    %592 = torch.prim.ListConstruct %132, %int4_685, %int32_686 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %593 = torch.aten.view %591, %592 : !torch.vtensor<[?,3,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,4,32],f16>
    torch.bind_symbolic_shape %593, [%38], affine_map<()[s0] -> (s0 * 192, 4, 32)> : !torch.vtensor<[?,4,32],f16>
    %594 = torch.prim.ListConstruct %588 : (!torch.vtensor<[1,1],si64>) -> !torch.list<optional<vtensor>>
    %false_687 = torch.constant.bool false
    %595 = torch.aten.index_put %593, %594, %589, %false_687 : !torch.vtensor<[?,4,32],f16>, !torch.list<optional<vtensor>>, !torch.vtensor<[1,1,4,32],f16>, !torch.bool -> !torch.vtensor<[?,4,32],f16>
    torch.bind_symbolic_shape %595, [%38], affine_map<()[s0] -> (s0 * 192, 4, 32)> : !torch.vtensor<[?,4,32],f16>
    %int3_688 = torch.constant.int 3
    %int2_689 = torch.constant.int 2
    %int32_690 = torch.constant.int 32
    %int4_691 = torch.constant.int 4
    %int32_692 = torch.constant.int 32
    %596 = torch.prim.ListConstruct %40, %int3_688, %int2_689, %int32_690, %int4_691, %int32_692 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %597 = torch.aten.view %595, %596 : !torch.vtensor<[?,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,4,32],f16>
    torch.bind_symbolic_shape %597, [%38], affine_map<()[s0] -> (s0, 3, 2, 32, 4, 32)> : !torch.vtensor<[?,3,2,32,4,32],f16>
    %int24576_693 = torch.constant.int 24576
    %598 = torch.prim.ListConstruct %40, %int24576_693 : (!torch.int, !torch.int) -> !torch.list<int>
    %599 = torch.aten.view %597, %598 : !torch.vtensor<[?,3,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,24576],f16>
    torch.bind_symbolic_shape %599, [%38], affine_map<()[s0] -> (s0, 24576)> : !torch.vtensor<[?,24576],f16>
    %int3_694 = torch.constant.int 3
    %int2_695 = torch.constant.int 2
    %int32_696 = torch.constant.int 32
    %int4_697 = torch.constant.int 4
    %int32_698 = torch.constant.int 32
    %600 = torch.prim.ListConstruct %40, %int3_694, %int2_695, %int32_696, %int4_697, %int32_698 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %601 = torch.aten.view %599, %600 : !torch.vtensor<[?,24576],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,4,32],f16>
    torch.bind_symbolic_shape %601, [%38], affine_map<()[s0] -> (s0, 3, 2, 32, 4, 32)> : !torch.vtensor<[?,3,2,32,4,32],f16>
    %int4_699 = torch.constant.int 4
    %int32_700 = torch.constant.int 32
    %602 = torch.prim.ListConstruct %132, %int4_699, %int32_700 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %603 = torch.aten.view %601, %602 : !torch.vtensor<[?,3,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,4,32],f16>
    torch.bind_symbolic_shape %603, [%38], affine_map<()[s0] -> (s0 * 192, 4, 32)> : !torch.vtensor<[?,4,32],f16>
    %int32_701 = torch.constant.int 32
    %604 = torch.aten.floor_divide.Scalar %arg2, %int32_701 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int1_702 = torch.constant.int 1
    %605 = torch.aten.unsqueeze %604, %int1_702 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_703 = torch.constant.int 1
    %false_704 = torch.constant.bool false
    %606 = torch.aten.gather %arg3, %int1_703, %605, %false_704 : !torch.vtensor<[1,?],si64>, !torch.int, !torch.vtensor<[1,1],si64>, !torch.bool -> !torch.vtensor<[1,1],si64>
    %int32_705 = torch.constant.int 32
    %607 = torch.aten.remainder.Scalar %arg2, %int32_705 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int1_706 = torch.constant.int 1
    %608 = torch.aten.unsqueeze %607, %int1_706 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %none_707 = torch.constant.none
    %609 = torch.aten.clone %28, %none_707 : !torch.vtensor<[],si64>, !torch.none -> !torch.vtensor<[],si64>
    %610 = torch.aten.detach %609 : !torch.vtensor<[],si64> -> !torch.vtensor<[],si64>
    %611 = torch.aten.detach %610 : !torch.vtensor<[],si64> -> !torch.vtensor<[],si64>
    %612 = torch.aten.detach %611 : !torch.vtensor<[],si64> -> !torch.vtensor<[],si64>
    %int0_708 = torch.constant.int 0
    %613 = torch.aten.unsqueeze %612, %int0_708 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int1_709 = torch.constant.int 1
    %int1_710 = torch.constant.int 1
    %614 = torch.prim.ListConstruct %int1_709, %int1_710 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_711 = torch.constant.int 1
    %int1_712 = torch.constant.int 1
    %615 = torch.prim.ListConstruct %int1_711, %int1_712 : (!torch.int, !torch.int) -> !torch.list<int>
    %int4_713 = torch.constant.int 4
    %int0_714 = torch.constant.int 0
    %cpu_715 = torch.constant.device "cpu"
    %false_716 = torch.constant.bool false
    %616 = torch.aten.empty_strided %614, %615, %int4_713, %int0_714, %cpu_715, %false_716 : !torch.list<int>, !torch.list<int>, !torch.int, !torch.int, !torch.Device, !torch.bool -> !torch.vtensor<[1,1],si64>
    %int2_717 = torch.constant.int 2
    %617 = torch.aten.fill.Scalar %616, %int2_717 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_718 = torch.constant.int 1
    %int1_719 = torch.constant.int 1
    %618 = torch.prim.ListConstruct %int1_718, %int1_719 : (!torch.int, !torch.int) -> !torch.list<int>
    %619 = torch.aten.repeat %613, %618 : !torch.vtensor<[1],si64>, !torch.list<int> -> !torch.vtensor<[1,1],si64>
    %int3_720 = torch.constant.int 3
    %620 = torch.aten.mul.Scalar %606, %int3_720 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_721 = torch.constant.int 1
    %621 = torch.aten.add.Tensor %620, %617, %int1_721 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int2_722 = torch.constant.int 2
    %622 = torch.aten.mul.Scalar %621, %int2_722 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_723 = torch.constant.int 1
    %623 = torch.aten.add.Tensor %622, %619, %int1_723 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int32_724 = torch.constant.int 32
    %624 = torch.aten.mul.Scalar %623, %int32_724 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_725 = torch.constant.int 1
    %625 = torch.aten.add.Tensor %624, %608, %int1_725 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int5_726 = torch.constant.int 5
    %626 = torch.prims.convert_element_type %554, %int5_726 : !torch.vtensor<[1,1,4,32],f16>, !torch.int -> !torch.vtensor<[1,1,4,32],f16>
    %627 = torch.prim.ListConstruct %625 : (!torch.vtensor<[1,1],si64>) -> !torch.list<optional<vtensor>>
    %false_727 = torch.constant.bool false
    %628 = torch.aten.index_put %603, %627, %626, %false_727 : !torch.vtensor<[?,4,32],f16>, !torch.list<optional<vtensor>>, !torch.vtensor<[1,1,4,32],f16>, !torch.bool -> !torch.vtensor<[?,4,32],f16>
    torch.bind_symbolic_shape %628, [%38], affine_map<()[s0] -> (s0 * 192, 4, 32)> : !torch.vtensor<[?,4,32],f16>
    %int3_728 = torch.constant.int 3
    %int2_729 = torch.constant.int 2
    %int32_730 = torch.constant.int 32
    %int4_731 = torch.constant.int 4
    %int32_732 = torch.constant.int 32
    %629 = torch.prim.ListConstruct %40, %int3_728, %int2_729, %int32_730, %int4_731, %int32_732 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %630 = torch.aten.view %628, %629 : !torch.vtensor<[?,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,4,32],f16>
    torch.bind_symbolic_shape %630, [%38], affine_map<()[s0] -> (s0, 3, 2, 32, 4, 32)> : !torch.vtensor<[?,3,2,32,4,32],f16>
    %int24576_733 = torch.constant.int 24576
    %631 = torch.prim.ListConstruct %40, %int24576_733 : (!torch.int, !torch.int) -> !torch.list<int>
    %632 = torch.aten.view %630, %631 : !torch.vtensor<[?,3,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,24576],f16>
    torch.overwrite.tensor.contents %632 overwrites %arg4 : !torch.vtensor<[?,24576],f16>, !torch.tensor<[?,24576],f16>
    torch.bind_symbolic_shape %632, [%38], affine_map<()[s0] -> (s0, 24576)> : !torch.vtensor<[?,24576],f16>
    %int1_734 = torch.constant.int 1
    %633 = torch.prim.ListConstruct %int1_734, %39 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_735 = torch.constant.int 1
    %634 = torch.prim.ListConstruct %39, %int1_735 : (!torch.int, !torch.int) -> !torch.list<int>
    %int4_736 = torch.constant.int 4
    %int0_737 = torch.constant.int 0
    %cpu_738 = torch.constant.device "cpu"
    %false_739 = torch.constant.bool false
    %635 = torch.aten.empty_strided %633, %634, %int4_736, %int0_737, %cpu_738, %false_739 : !torch.list<int>, !torch.list<int>, !torch.int, !torch.int, !torch.Device, !torch.bool -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %635, [%37], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %int2_740 = torch.constant.int 2
    %636 = torch.aten.fill.Scalar %635, %int2_740 : !torch.vtensor<[1,?],si64>, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %636, [%37], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %int3_741 = torch.constant.int 3
    %637 = torch.aten.mul.Scalar %arg3, %int3_741 : !torch.vtensor<[1,?],si64>, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %637, [%37], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %int1_742 = torch.constant.int 1
    %638 = torch.aten.add.Tensor %637, %636, %int1_742 : !torch.vtensor<[1,?],si64>, !torch.vtensor<[1,?],si64>, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %638, [%37], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %639 = torch.prim.ListConstruct %39 : (!torch.int) -> !torch.list<int>
    %640 = torch.aten.view %638, %639 : !torch.vtensor<[1,?],si64>, !torch.list<int> -> !torch.vtensor<[?],si64>
    torch.bind_symbolic_shape %640, [%37], affine_map<()[s0] -> (s0)> : !torch.vtensor<[?],si64>
    %int3_743 = torch.constant.int 3
    %int2_744 = torch.constant.int 2
    %int32_745 = torch.constant.int 32
    %int4_746 = torch.constant.int 4
    %int32_747 = torch.constant.int 32
    %641 = torch.prim.ListConstruct %40, %int3_743, %int2_744, %int32_745, %int4_746, %int32_747 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %642 = torch.aten.view %632, %641 : !torch.vtensor<[?,24576],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,4,32],f16>
    torch.bind_symbolic_shape %642, [%38], affine_map<()[s0] -> (s0, 3, 2, 32, 4, 32)> : !torch.vtensor<[?,3,2,32,4,32],f16>
    %int2_748 = torch.constant.int 2
    %int32_749 = torch.constant.int 32
    %int4_750 = torch.constant.int 4
    %int32_751 = torch.constant.int 32
    %643 = torch.prim.ListConstruct %130, %int2_748, %int32_749, %int4_750, %int32_751 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %644 = torch.aten.view %642, %643 : !torch.vtensor<[?,3,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,2,32,4,32],f16>
    torch.bind_symbolic_shape %644, [%38], affine_map<()[s0] -> (s0 * 3, 2, 32, 4, 32)> : !torch.vtensor<[?,2,32,4,32],f16>
    %int0_752 = torch.constant.int 0
    %645 = torch.aten.index_select %644, %int0_752, %640 : !torch.vtensor<[?,2,32,4,32],f16>, !torch.int, !torch.vtensor<[?],si64> -> !torch.vtensor<[?,2,32,4,32],f16>
    torch.bind_symbolic_shape %645, [%37], affine_map<()[s0] -> (s0, 2, 32, 4, 32)> : !torch.vtensor<[?,2,32,4,32],f16>
    %int1_753 = torch.constant.int 1
    %int2_754 = torch.constant.int 2
    %int32_755 = torch.constant.int 32
    %int4_756 = torch.constant.int 4
    %int32_757 = torch.constant.int 32
    %646 = torch.prim.ListConstruct %int1_753, %39, %int2_754, %int32_755, %int4_756, %int32_757 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %647 = torch.aten.view %645, %646 : !torch.vtensor<[?,2,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,2,32,4,32],f16>
    torch.bind_symbolic_shape %647, [%37], affine_map<()[s0] -> (1, s0, 2, 32, 4, 32)> : !torch.vtensor<[1,?,2,32,4,32],f16>
    %int0_758 = torch.constant.int 0
    %int0_759 = torch.constant.int 0
    %int9223372036854775807_760 = torch.constant.int 9223372036854775807
    %int1_761 = torch.constant.int 1
    %648 = torch.aten.slice.Tensor %647, %int0_758, %int0_759, %int9223372036854775807_760, %int1_761 : !torch.vtensor<[1,?,2,32,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,2,32,4,32],f16>
    torch.bind_symbolic_shape %648, [%37], affine_map<()[s0] -> (1, s0, 2, 32, 4, 32)> : !torch.vtensor<[1,?,2,32,4,32],f16>
    %int1_762 = torch.constant.int 1
    %int0_763 = torch.constant.int 0
    %int9223372036854775807_764 = torch.constant.int 9223372036854775807
    %int1_765 = torch.constant.int 1
    %649 = torch.aten.slice.Tensor %648, %int1_762, %int0_763, %int9223372036854775807_764, %int1_765 : !torch.vtensor<[1,?,2,32,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,2,32,4,32],f16>
    torch.bind_symbolic_shape %649, [%37], affine_map<()[s0] -> (1, s0, 2, 32, 4, 32)> : !torch.vtensor<[1,?,2,32,4,32],f16>
    %int2_766 = torch.constant.int 2
    %int0_767 = torch.constant.int 0
    %650 = torch.aten.select.int %649, %int2_766, %int0_767 : !torch.vtensor<[1,?,2,32,4,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,?,32,4,32],f16>
    torch.bind_symbolic_shape %650, [%37], affine_map<()[s0] -> (1, s0, 32, 4, 32)> : !torch.vtensor<[1,?,32,4,32],f16>
    %int2_768 = torch.constant.int 2
    %int0_769 = torch.constant.int 0
    %int1_770 = torch.constant.int 1
    %651 = torch.aten.slice.Tensor %650, %int2_768, %int0_769, %41, %int1_770 : !torch.vtensor<[1,?,32,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32,4,32],f16>
    torch.bind_symbolic_shape %651, [%37], affine_map<()[s0] -> (1, s0, 32, 4, 32)> : !torch.vtensor<[1,?,32,4,32],f16>
    %int0_771 = torch.constant.int 0
    %652 = torch.aten.clone %651, %int0_771 : !torch.vtensor<[1,?,32,4,32],f16>, !torch.int -> !torch.vtensor<[1,?,32,4,32],f16>
    torch.bind_symbolic_shape %652, [%37], affine_map<()[s0] -> (1, s0, 32, 4, 32)> : !torch.vtensor<[1,?,32,4,32],f16>
    %int1_772 = torch.constant.int 1
    %int4_773 = torch.constant.int 4
    %int32_774 = torch.constant.int 32
    %653 = torch.prim.ListConstruct %int1_772, %41, %int4_773, %int32_774 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %654 = torch.aten._unsafe_view %652, %653 : !torch.vtensor<[1,?,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %654, [%37], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int0_775 = torch.constant.int 0
    %int0_776 = torch.constant.int 0
    %int9223372036854775807_777 = torch.constant.int 9223372036854775807
    %int1_778 = torch.constant.int 1
    %655 = torch.aten.slice.Tensor %654, %int0_775, %int0_776, %int9223372036854775807_777, %int1_778 : !torch.vtensor<[1,?,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %655, [%37], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int0_779 = torch.constant.int 0
    %int0_780 = torch.constant.int 0
    %int9223372036854775807_781 = torch.constant.int 9223372036854775807
    %int1_782 = torch.constant.int 1
    %656 = torch.aten.slice.Tensor %647, %int0_779, %int0_780, %int9223372036854775807_781, %int1_782 : !torch.vtensor<[1,?,2,32,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,2,32,4,32],f16>
    torch.bind_symbolic_shape %656, [%37], affine_map<()[s0] -> (1, s0, 2, 32, 4, 32)> : !torch.vtensor<[1,?,2,32,4,32],f16>
    %int1_783 = torch.constant.int 1
    %int0_784 = torch.constant.int 0
    %int9223372036854775807_785 = torch.constant.int 9223372036854775807
    %int1_786 = torch.constant.int 1
    %657 = torch.aten.slice.Tensor %656, %int1_783, %int0_784, %int9223372036854775807_785, %int1_786 : !torch.vtensor<[1,?,2,32,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,2,32,4,32],f16>
    torch.bind_symbolic_shape %657, [%37], affine_map<()[s0] -> (1, s0, 2, 32, 4, 32)> : !torch.vtensor<[1,?,2,32,4,32],f16>
    %int2_787 = torch.constant.int 2
    %int1_788 = torch.constant.int 1
    %658 = torch.aten.select.int %657, %int2_787, %int1_788 : !torch.vtensor<[1,?,2,32,4,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,?,32,4,32],f16>
    torch.bind_symbolic_shape %658, [%37], affine_map<()[s0] -> (1, s0, 32, 4, 32)> : !torch.vtensor<[1,?,32,4,32],f16>
    %int2_789 = torch.constant.int 2
    %int0_790 = torch.constant.int 0
    %int1_791 = torch.constant.int 1
    %659 = torch.aten.slice.Tensor %658, %int2_789, %int0_790, %41, %int1_791 : !torch.vtensor<[1,?,32,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32,4,32],f16>
    torch.bind_symbolic_shape %659, [%37], affine_map<()[s0] -> (1, s0, 32, 4, 32)> : !torch.vtensor<[1,?,32,4,32],f16>
    %int0_792 = torch.constant.int 0
    %660 = torch.aten.clone %659, %int0_792 : !torch.vtensor<[1,?,32,4,32],f16>, !torch.int -> !torch.vtensor<[1,?,32,4,32],f16>
    torch.bind_symbolic_shape %660, [%37], affine_map<()[s0] -> (1, s0, 32, 4, 32)> : !torch.vtensor<[1,?,32,4,32],f16>
    %int1_793 = torch.constant.int 1
    %int4_794 = torch.constant.int 4
    %int32_795 = torch.constant.int 32
    %661 = torch.prim.ListConstruct %int1_793, %41, %int4_794, %int32_795 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %662 = torch.aten._unsafe_view %660, %661 : !torch.vtensor<[1,?,32,4,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %662, [%37], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int0_796 = torch.constant.int 0
    %int0_797 = torch.constant.int 0
    %int9223372036854775807_798 = torch.constant.int 9223372036854775807
    %int1_799 = torch.constant.int 1
    %663 = torch.aten.slice.Tensor %662, %int0_796, %int0_797, %int9223372036854775807_798, %int1_799 : !torch.vtensor<[1,?,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %663, [%37], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int-2_800 = torch.constant.int -2
    %664 = torch.aten.unsqueeze %655, %int-2_800 : !torch.vtensor<[1,?,4,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,1,32],f16>
    torch.bind_symbolic_shape %664, [%37], affine_map<()[s0] -> (1, s0 * 32, 4, 1, 32)> : !torch.vtensor<[1,?,4,1,32],f16>
    %int1_801 = torch.constant.int 1
    %int4_802 = torch.constant.int 4
    %int2_803 = torch.constant.int 2
    %int32_804 = torch.constant.int 32
    %665 = torch.prim.ListConstruct %int1_801, %41, %int4_802, %int2_803, %int32_804 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_805 = torch.constant.bool false
    %666 = torch.aten.expand %664, %665, %false_805 : !torch.vtensor<[1,?,4,1,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,?,4,2,32],f16>
    torch.bind_symbolic_shape %666, [%37], affine_map<()[s0] -> (1, s0 * 32, 4, 2, 32)> : !torch.vtensor<[1,?,4,2,32],f16>
    %int0_806 = torch.constant.int 0
    %667 = torch.aten.clone %666, %int0_806 : !torch.vtensor<[1,?,4,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,2,32],f16>
    torch.bind_symbolic_shape %667, [%37], affine_map<()[s0] -> (1, s0 * 32, 4, 2, 32)> : !torch.vtensor<[1,?,4,2,32],f16>
    %int1_807 = torch.constant.int 1
    %int8_808 = torch.constant.int 8
    %int32_809 = torch.constant.int 32
    %668 = torch.prim.ListConstruct %int1_807, %41, %int8_808, %int32_809 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %669 = torch.aten._unsafe_view %667, %668 : !torch.vtensor<[1,?,4,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,8,32],f16>
    torch.bind_symbolic_shape %669, [%37], affine_map<()[s0] -> (1, s0 * 32, 8, 32)> : !torch.vtensor<[1,?,8,32],f16>
    %int-2_810 = torch.constant.int -2
    %670 = torch.aten.unsqueeze %663, %int-2_810 : !torch.vtensor<[1,?,4,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,1,32],f16>
    torch.bind_symbolic_shape %670, [%37], affine_map<()[s0] -> (1, s0 * 32, 4, 1, 32)> : !torch.vtensor<[1,?,4,1,32],f16>
    %int1_811 = torch.constant.int 1
    %int4_812 = torch.constant.int 4
    %int2_813 = torch.constant.int 2
    %int32_814 = torch.constant.int 32
    %671 = torch.prim.ListConstruct %int1_811, %41, %int4_812, %int2_813, %int32_814 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_815 = torch.constant.bool false
    %672 = torch.aten.expand %670, %671, %false_815 : !torch.vtensor<[1,?,4,1,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,?,4,2,32],f16>
    torch.bind_symbolic_shape %672, [%37], affine_map<()[s0] -> (1, s0 * 32, 4, 2, 32)> : !torch.vtensor<[1,?,4,2,32],f16>
    %int0_816 = torch.constant.int 0
    %673 = torch.aten.clone %672, %int0_816 : !torch.vtensor<[1,?,4,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,2,32],f16>
    torch.bind_symbolic_shape %673, [%37], affine_map<()[s0] -> (1, s0 * 32, 4, 2, 32)> : !torch.vtensor<[1,?,4,2,32],f16>
    %int1_817 = torch.constant.int 1
    %int8_818 = torch.constant.int 8
    %int32_819 = torch.constant.int 32
    %674 = torch.prim.ListConstruct %int1_817, %41, %int8_818, %int32_819 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %675 = torch.aten._unsafe_view %673, %674 : !torch.vtensor<[1,?,4,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,8,32],f16>
    torch.bind_symbolic_shape %675, [%37], affine_map<()[s0] -> (1, s0 * 32, 8, 32)> : !torch.vtensor<[1,?,8,32],f16>
    %int1_820 = torch.constant.int 1
    %int2_821 = torch.constant.int 2
    %676 = torch.aten.transpose.int %560, %int1_820, %int2_821 : !torch.vtensor<[1,1,8,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,8,1,32],f16>
    %int1_822 = torch.constant.int 1
    %int2_823 = torch.constant.int 2
    %677 = torch.aten.transpose.int %669, %int1_822, %int2_823 : !torch.vtensor<[1,?,8,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,8,?,32],f16>
    torch.bind_symbolic_shape %677, [%37], affine_map<()[s0] -> (1, 8, s0 * 32, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %int1_824 = torch.constant.int 1
    %int2_825 = torch.constant.int 2
    %678 = torch.aten.transpose.int %675, %int1_824, %int2_825 : !torch.vtensor<[1,?,8,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,8,?,32],f16>
    torch.bind_symbolic_shape %678, [%37], affine_map<()[s0] -> (1, 8, s0 * 32, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %int5_826 = torch.constant.int 5
    %679 = torch.prims.convert_element_type %676, %int5_826 : !torch.vtensor<[1,8,1,32],f16>, !torch.int -> !torch.vtensor<[1,8,1,32],f16>
    %int5_827 = torch.constant.int 5
    %680 = torch.prims.convert_element_type %677, %int5_827 : !torch.vtensor<[1,8,?,32],f16>, !torch.int -> !torch.vtensor<[1,8,?,32],f16>
    torch.bind_symbolic_shape %680, [%37], affine_map<()[s0] -> (1, 8, s0 * 32, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %int5_828 = torch.constant.int 5
    %681 = torch.prims.convert_element_type %678, %int5_828 : !torch.vtensor<[1,8,?,32],f16>, !torch.int -> !torch.vtensor<[1,8,?,32],f16>
    torch.bind_symbolic_shape %681, [%37], affine_map<()[s0] -> (1, 8, s0 * 32, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %int5_829 = torch.constant.int 5
    %682 = torch.prims.convert_element_type %51, %int5_829 : !torch.vtensor<[1,1,1,?],f16>, !torch.int -> !torch.vtensor<[1,1,1,?],f16>
    torch.bind_symbolic_shape %682, [%37], affine_map<()[s0] -> (1, 1, 1, s0 * 32)> : !torch.vtensor<[1,1,1,?],f16>
    %float0.000000e00_830 = torch.constant.float 0.000000e+00
    %false_831 = torch.constant.bool false
    %none_832 = torch.constant.none
    %683:2 = torch.operator "torch.aten._scaled_dot_product_flash_attention_for_cpu"(%679, %680, %681, %float0.000000e00_830, %false_831, %682, %none_832) : (!torch.vtensor<[1,8,1,32],f16>, !torch.vtensor<[1,8,?,32],f16>, !torch.vtensor<[1,8,?,32],f16>, !torch.float, !torch.bool, !torch.vtensor<[1,1,1,?],f16>, !torch.none) -> (!torch.vtensor<[1,8,1,32],f16>, !torch.vtensor<[1,8,1],f32>) 
    %int1_833 = torch.constant.int 1
    %int2_834 = torch.constant.int 2
    %684 = torch.aten.transpose.int %683#0, %int1_833, %int2_834 : !torch.vtensor<[1,8,1,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,1,8,32],f16>
    %int1_835 = torch.constant.int 1
    %int1_836 = torch.constant.int 1
    %int256_837 = torch.constant.int 256
    %685 = torch.prim.ListConstruct %int1_835, %int1_836, %int256_837 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %686 = torch.aten.view %684, %685 : !torch.vtensor<[1,1,8,32],f16>, !torch.list<int> -> !torch.vtensor<[1,1,256],f16>
    %int5_838 = torch.constant.int 5
    %687 = torch.prims.convert_element_type %29, %int5_838 : !torch.vtensor<[256,256],f32>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int-2_839 = torch.constant.int -2
    %int-1_840 = torch.constant.int -1
    %688 = torch.aten.transpose.int %687, %int-2_839, %int-1_840 : !torch.vtensor<[256,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,256],f16>
    %int5_841 = torch.constant.int 5
    %689 = torch.prims.convert_element_type %688, %int5_841 : !torch.vtensor<[256,256],f16>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int1_842 = torch.constant.int 1
    %int256_843 = torch.constant.int 256
    %690 = torch.prim.ListConstruct %int1_842, %int256_843 : (!torch.int, !torch.int) -> !torch.list<int>
    %691 = torch.aten.view %686, %690 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %692 = torch.aten.mm %691, %689 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,256],f16> -> !torch.vtensor<[1,256],f16>
    %int1_844 = torch.constant.int 1
    %int1_845 = torch.constant.int 1
    %int256_846 = torch.constant.int 256
    %693 = torch.prim.ListConstruct %int1_844, %int1_845, %int256_846 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %694 = torch.aten.view %692, %693 : !torch.vtensor<[1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,1,256],f16>
    %int1_847 = torch.constant.int 1
    %695 = torch.aten.add.Tensor %514, %694, %int1_847 : !torch.vtensor<[1,1,256],f16>, !torch.vtensor<[1,1,256],f16>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %int6_848 = torch.constant.int 6
    %696 = torch.prims.convert_element_type %695, %int6_848 : !torch.vtensor<[1,1,256],f16>, !torch.int -> !torch.vtensor<[1,1,256],f32>
    %int2_849 = torch.constant.int 2
    %697 = torch.aten.pow.Tensor_Scalar %696, %int2_849 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f32>
    %int-1_850 = torch.constant.int -1
    %698 = torch.prim.ListConstruct %int-1_850 : (!torch.int) -> !torch.list<int>
    %true_851 = torch.constant.bool true
    %none_852 = torch.constant.none
    %699 = torch.aten.mean.dim %697, %698, %true_851, %none_852 : !torch.vtensor<[1,1,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1],f32>
    %float1.000000e-02_853 = torch.constant.float 1.000000e-02
    %int1_854 = torch.constant.int 1
    %700 = torch.aten.add.Scalar %699, %float1.000000e-02_853, %int1_854 : !torch.vtensor<[1,1,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,1,1],f32>
    %701 = torch.aten.rsqrt %700 : !torch.vtensor<[1,1,1],f32> -> !torch.vtensor<[1,1,1],f32>
    %702 = torch.aten.mul.Tensor %696, %701 : !torch.vtensor<[1,1,256],f32>, !torch.vtensor<[1,1,1],f32> -> !torch.vtensor<[1,1,256],f32>
    %int5_855 = torch.constant.int 5
    %703 = torch.prims.convert_element_type %702, %int5_855 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %704 = torch.aten.mul.Tensor %30, %703 : !torch.vtensor<[256],f32>, !torch.vtensor<[1,1,256],f16> -> !torch.vtensor<[1,1,256],f32>
    %int5_856 = torch.constant.int 5
    %705 = torch.prims.convert_element_type %704, %int5_856 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %int5_857 = torch.constant.int 5
    %706 = torch.prims.convert_element_type %31, %int5_857 : !torch.vtensor<[23,256],f32>, !torch.int -> !torch.vtensor<[23,256],f16>
    %int-2_858 = torch.constant.int -2
    %int-1_859 = torch.constant.int -1
    %707 = torch.aten.transpose.int %706, %int-2_858, %int-1_859 : !torch.vtensor<[23,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,23],f16>
    %int5_860 = torch.constant.int 5
    %708 = torch.prims.convert_element_type %707, %int5_860 : !torch.vtensor<[256,23],f16>, !torch.int -> !torch.vtensor<[256,23],f16>
    %int1_861 = torch.constant.int 1
    %int256_862 = torch.constant.int 256
    %709 = torch.prim.ListConstruct %int1_861, %int256_862 : (!torch.int, !torch.int) -> !torch.list<int>
    %710 = torch.aten.view %705, %709 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %711 = torch.aten.mm %710, %708 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,23],f16> -> !torch.vtensor<[1,23],f16>
    %int1_863 = torch.constant.int 1
    %int1_864 = torch.constant.int 1
    %int23_865 = torch.constant.int 23
    %712 = torch.prim.ListConstruct %int1_863, %int1_864, %int23_865 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %713 = torch.aten.view %711, %712 : !torch.vtensor<[1,23],f16>, !torch.list<int> -> !torch.vtensor<[1,1,23],f16>
    %714 = torch.aten.silu %713 : !torch.vtensor<[1,1,23],f16> -> !torch.vtensor<[1,1,23],f16>
    %int5_866 = torch.constant.int 5
    %715 = torch.prims.convert_element_type %32, %int5_866 : !torch.vtensor<[23,256],f32>, !torch.int -> !torch.vtensor<[23,256],f16>
    %int-2_867 = torch.constant.int -2
    %int-1_868 = torch.constant.int -1
    %716 = torch.aten.transpose.int %715, %int-2_867, %int-1_868 : !torch.vtensor<[23,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,23],f16>
    %int5_869 = torch.constant.int 5
    %717 = torch.prims.convert_element_type %716, %int5_869 : !torch.vtensor<[256,23],f16>, !torch.int -> !torch.vtensor<[256,23],f16>
    %int1_870 = torch.constant.int 1
    %int256_871 = torch.constant.int 256
    %718 = torch.prim.ListConstruct %int1_870, %int256_871 : (!torch.int, !torch.int) -> !torch.list<int>
    %719 = torch.aten.view %705, %718 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %720 = torch.aten.mm %719, %717 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,23],f16> -> !torch.vtensor<[1,23],f16>
    %int1_872 = torch.constant.int 1
    %int1_873 = torch.constant.int 1
    %int23_874 = torch.constant.int 23
    %721 = torch.prim.ListConstruct %int1_872, %int1_873, %int23_874 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %722 = torch.aten.view %720, %721 : !torch.vtensor<[1,23],f16>, !torch.list<int> -> !torch.vtensor<[1,1,23],f16>
    %723 = torch.aten.mul.Tensor %714, %722 : !torch.vtensor<[1,1,23],f16>, !torch.vtensor<[1,1,23],f16> -> !torch.vtensor<[1,1,23],f16>
    %int5_875 = torch.constant.int 5
    %724 = torch.prims.convert_element_type %33, %int5_875 : !torch.vtensor<[256,23],f32>, !torch.int -> !torch.vtensor<[256,23],f16>
    %int-2_876 = torch.constant.int -2
    %int-1_877 = torch.constant.int -1
    %725 = torch.aten.transpose.int %724, %int-2_876, %int-1_877 : !torch.vtensor<[256,23],f16>, !torch.int, !torch.int -> !torch.vtensor<[23,256],f16>
    %int5_878 = torch.constant.int 5
    %726 = torch.prims.convert_element_type %725, %int5_878 : !torch.vtensor<[23,256],f16>, !torch.int -> !torch.vtensor<[23,256],f16>
    %int1_879 = torch.constant.int 1
    %int23_880 = torch.constant.int 23
    %727 = torch.prim.ListConstruct %int1_879, %int23_880 : (!torch.int, !torch.int) -> !torch.list<int>
    %728 = torch.aten.view %723, %727 : !torch.vtensor<[1,1,23],f16>, !torch.list<int> -> !torch.vtensor<[1,23],f16>
    %729 = torch.aten.mm %728, %726 : !torch.vtensor<[1,23],f16>, !torch.vtensor<[23,256],f16> -> !torch.vtensor<[1,256],f16>
    %int1_881 = torch.constant.int 1
    %int1_882 = torch.constant.int 1
    %int256_883 = torch.constant.int 256
    %730 = torch.prim.ListConstruct %int1_881, %int1_882, %int256_883 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %731 = torch.aten.view %729, %730 : !torch.vtensor<[1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,1,256],f16>
    %int1_884 = torch.constant.int 1
    %732 = torch.aten.add.Tensor %695, %731, %int1_884 : !torch.vtensor<[1,1,256],f16>, !torch.vtensor<[1,1,256],f16>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %int6_885 = torch.constant.int 6
    %733 = torch.prims.convert_element_type %732, %int6_885 : !torch.vtensor<[1,1,256],f16>, !torch.int -> !torch.vtensor<[1,1,256],f32>
    %int2_886 = torch.constant.int 2
    %734 = torch.aten.pow.Tensor_Scalar %733, %int2_886 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f32>
    %int-1_887 = torch.constant.int -1
    %735 = torch.prim.ListConstruct %int-1_887 : (!torch.int) -> !torch.list<int>
    %true_888 = torch.constant.bool true
    %none_889 = torch.constant.none
    %736 = torch.aten.mean.dim %734, %735, %true_888, %none_889 : !torch.vtensor<[1,1,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1],f32>
    %float1.000000e-02_890 = torch.constant.float 1.000000e-02
    %int1_891 = torch.constant.int 1
    %737 = torch.aten.add.Scalar %736, %float1.000000e-02_890, %int1_891 : !torch.vtensor<[1,1,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,1,1],f32>
    %738 = torch.aten.rsqrt %737 : !torch.vtensor<[1,1,1],f32> -> !torch.vtensor<[1,1,1],f32>
    %739 = torch.aten.mul.Tensor %733, %738 : !torch.vtensor<[1,1,256],f32>, !torch.vtensor<[1,1,1],f32> -> !torch.vtensor<[1,1,256],f32>
    %int5_892 = torch.constant.int 5
    %740 = torch.prims.convert_element_type %739, %int5_892 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %741 = torch.aten.mul.Tensor %34, %740 : !torch.vtensor<[1,256],f32>, !torch.vtensor<[1,1,256],f16> -> !torch.vtensor<[1,1,256],f32>
    %int5_893 = torch.constant.int 5
    %742 = torch.prims.convert_element_type %741, %int5_893 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %int5_894 = torch.constant.int 5
    %743 = torch.prims.convert_element_type %35, %int5_894 : !torch.vtensor<[256,256],f32>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int-2_895 = torch.constant.int -2
    %int-1_896 = torch.constant.int -1
    %744 = torch.aten.transpose.int %743, %int-2_895, %int-1_896 : !torch.vtensor<[256,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,256],f16>
    %int5_897 = torch.constant.int 5
    %745 = torch.prims.convert_element_type %744, %int5_897 : !torch.vtensor<[256,256],f16>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int1_898 = torch.constant.int 1
    %int256_899 = torch.constant.int 256
    %746 = torch.prim.ListConstruct %int1_898, %int256_899 : (!torch.int, !torch.int) -> !torch.list<int>
    %747 = torch.aten.view %742, %746 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %748 = torch.aten.mm %747, %745 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,256],f16> -> !torch.vtensor<[1,256],f16>
    %int1_900 = torch.constant.int 1
    %int1_901 = torch.constant.int 1
    %int256_902 = torch.constant.int 256
    %749 = torch.prim.ListConstruct %int1_900, %int1_901, %int256_902 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %750 = torch.aten.view %748, %749 : !torch.vtensor<[1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,1,256],f16>
    return %750 : !torch.vtensor<[1,1,256],f16>
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
