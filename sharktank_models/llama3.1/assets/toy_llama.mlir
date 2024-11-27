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
  func.func @prefill_bs1(%arg0: !torch.vtensor<[1,?],si64>, %arg1: !torch.vtensor<[1],si64>, %arg2: !torch.vtensor<[1,?],si64>, %arg3: !torch.tensor<[?,12288],f16>) -> !torch.vtensor<[1,?,256],f16> attributes {torch.assume_strict_symbolic_shapes} {
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
    %30 = torch.copy.to_vtensor %arg3 : !torch.vtensor<[?,12288],f16>
    %31 = torch.symbolic_int "16*s1" {min_val = 32, max_val = 112} : !torch.int
    %32 = torch.symbolic_int "s1" {min_val = 2, max_val = 7} : !torch.int
    %33 = torch.symbolic_int "s2" {min_val = 0, max_val = 9223372036854775807} : !torch.int
    torch.bind_symbolic_shape %arg0, [%32], affine_map<()[s0] -> (1, s0 * 16)> : !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %arg2, [%32], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %30, [%33], affine_map<()[s0] -> (s0, 12288)> : !torch.vtensor<[?,12288],f16>
    %int1 = torch.constant.int 1
    %34 = torch.aten.size.int %arg2, %int1 : !torch.vtensor<[1,?],si64>, !torch.int -> !torch.int
    %int0 = torch.constant.int 0
    %35 = torch.aten.size.int %30, %int0 : !torch.vtensor<[?,12288],f16>, !torch.int -> !torch.int
    %int1_0 = torch.constant.int 1
    %36 = torch.aten.size.int %arg0, %int1_0 : !torch.vtensor<[1,?],si64>, !torch.int -> !torch.int
    %int0_1 = torch.constant.int 0
    %int1_2 = torch.constant.int 1
    %none = torch.constant.none
    %none_3 = torch.constant.none
    %cpu = torch.constant.device "cpu"
    %false = torch.constant.bool false
    %37 = torch.aten.arange.start_step %int0_1, %36, %int1_2, %none, %none_3, %cpu, %false : !torch.int, !torch.int, !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[?],si64>
    torch.bind_symbolic_shape %37, [%32], affine_map<()[s0] -> (s0 * 16)> : !torch.vtensor<[?],si64>
    %int-1 = torch.constant.int -1
    %38 = torch.aten.unsqueeze %arg1, %int-1 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %39 = torch.aten.ge.Tensor %37, %38 : !torch.vtensor<[?],si64>, !torch.vtensor<[1,1],si64> -> !torch.vtensor<[1,?],i1>
    torch.bind_symbolic_shape %39, [%32], affine_map<()[s0] -> (1, s0 * 16)> : !torch.vtensor<[1,?],i1>
    %int1_4 = torch.constant.int 1
    %int1_5 = torch.constant.int 1
    %40 = torch.prim.ListConstruct %int1_4, %int1_5 : (!torch.int, !torch.int) -> !torch.list<int>
    %int11 = torch.constant.int 11
    %none_6 = torch.constant.none
    %cpu_7 = torch.constant.device "cpu"
    %false_8 = torch.constant.bool false
    %41 = torch.aten.ones %40, %int11, %none_6, %cpu_7, %false_8 : !torch.list<int>, !torch.int, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[1,1],i1>
    %int128 = torch.constant.int 128
    %int128_9 = torch.constant.int 128
    %42 = torch.prim.ListConstruct %int128, %int128_9 : (!torch.int, !torch.int) -> !torch.list<int>
    %false_10 = torch.constant.bool false
    %43 = torch.aten.expand %41, %42, %false_10 : !torch.vtensor<[1,1],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[128,128],i1>
    %int1_11 = torch.constant.int 1
    %44 = torch.aten.triu %43, %int1_11 : !torch.vtensor<[128,128],i1>, !torch.int -> !torch.vtensor<[128,128],i1>
    %int0_12 = torch.constant.int 0
    %45 = torch.aten.unsqueeze %44, %int0_12 : !torch.vtensor<[128,128],i1>, !torch.int -> !torch.vtensor<[1,128,128],i1>
    %int1_13 = torch.constant.int 1
    %46 = torch.aten.unsqueeze %45, %int1_13 : !torch.vtensor<[1,128,128],i1>, !torch.int -> !torch.vtensor<[1,1,128,128],i1>
    %int2 = torch.constant.int 2
    %int0_14 = torch.constant.int 0
    %int9223372036854775807 = torch.constant.int 9223372036854775807
    %int1_15 = torch.constant.int 1
    %47 = torch.aten.slice.Tensor %46, %int2, %int0_14, %int9223372036854775807, %int1_15 : !torch.vtensor<[1,1,128,128],i1>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,1,128,128],i1>
    %int3 = torch.constant.int 3
    %int0_16 = torch.constant.int 0
    %int9223372036854775807_17 = torch.constant.int 9223372036854775807
    %int1_18 = torch.constant.int 1
    %48 = torch.aten.slice.Tensor %47, %int3, %int0_16, %int9223372036854775807_17, %int1_18 : !torch.vtensor<[1,1,128,128],i1>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,1,128,128],i1>
    %int0_19 = torch.constant.int 0
    %int0_20 = torch.constant.int 0
    %int9223372036854775807_21 = torch.constant.int 9223372036854775807
    %int1_22 = torch.constant.int 1
    %49 = torch.aten.slice.Tensor %48, %int0_19, %int0_20, %int9223372036854775807_21, %int1_22 : !torch.vtensor<[1,1,128,128],i1>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,1,128,128],i1>
    %int1_23 = torch.constant.int 1
    %int0_24 = torch.constant.int 0
    %int9223372036854775807_25 = torch.constant.int 9223372036854775807
    %int1_26 = torch.constant.int 1
    %50 = torch.aten.slice.Tensor %49, %int1_23, %int0_24, %int9223372036854775807_25, %int1_26 : !torch.vtensor<[1,1,128,128],i1>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,1,128,128],i1>
    %int2_27 = torch.constant.int 2
    %int0_28 = torch.constant.int 0
    %int1_29 = torch.constant.int 1
    %51 = torch.aten.slice.Tensor %50, %int2_27, %int0_28, %36, %int1_29 : !torch.vtensor<[1,1,128,128],i1>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,1,?,128],i1>
    torch.bind_symbolic_shape %51, [%32], affine_map<()[s0] -> (1, 1, s0 * 16, 128)> : !torch.vtensor<[1,1,?,128],i1>
    %int3_30 = torch.constant.int 3
    %int0_31 = torch.constant.int 0
    %int1_32 = torch.constant.int 1
    %52 = torch.aten.slice.Tensor %51, %int3_30, %int0_31, %36, %int1_32 : !torch.vtensor<[1,1,?,128],i1>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,1,?,?],i1>
    torch.bind_symbolic_shape %52, [%32], affine_map<()[s0] -> (1, 1, s0 * 16, s0 * 16)> : !torch.vtensor<[1,1,?,?],i1>
    %int0_33 = torch.constant.int 0
    %int0_34 = torch.constant.int 0
    %int9223372036854775807_35 = torch.constant.int 9223372036854775807
    %int1_36 = torch.constant.int 1
    %53 = torch.aten.slice.Tensor %39, %int0_33, %int0_34, %int9223372036854775807_35, %int1_36 : !torch.vtensor<[1,?],i1>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?],i1>
    torch.bind_symbolic_shape %53, [%32], affine_map<()[s0] -> (1, s0 * 16)> : !torch.vtensor<[1,?],i1>
    %int1_37 = torch.constant.int 1
    %54 = torch.aten.unsqueeze %53, %int1_37 : !torch.vtensor<[1,?],i1>, !torch.int -> !torch.vtensor<[1,1,?],i1>
    torch.bind_symbolic_shape %54, [%32], affine_map<()[s0] -> (1, 1, s0 * 16)> : !torch.vtensor<[1,1,?],i1>
    %int2_38 = torch.constant.int 2
    %55 = torch.aten.unsqueeze %54, %int2_38 : !torch.vtensor<[1,1,?],i1>, !torch.int -> !torch.vtensor<[1,1,1,?],i1>
    torch.bind_symbolic_shape %55, [%32], affine_map<()[s0] -> (1, 1, 1, s0 * 16)> : !torch.vtensor<[1,1,1,?],i1>
    %int3_39 = torch.constant.int 3
    %int0_40 = torch.constant.int 0
    %int9223372036854775807_41 = torch.constant.int 9223372036854775807
    %int1_42 = torch.constant.int 1
    %56 = torch.aten.slice.Tensor %55, %int3_39, %int0_40, %int9223372036854775807_41, %int1_42 : !torch.vtensor<[1,1,1,?],i1>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,1,1,?],i1>
    torch.bind_symbolic_shape %56, [%32], affine_map<()[s0] -> (1, 1, 1, s0 * 16)> : !torch.vtensor<[1,1,1,?],i1>
    %57 = torch.aten.logical_or %52, %56 : !torch.vtensor<[1,1,?,?],i1>, !torch.vtensor<[1,1,1,?],i1> -> !torch.vtensor<[1,1,?,?],i1>
    torch.bind_symbolic_shape %57, [%32], affine_map<()[s0] -> (1, 1, s0 * 16, s0 * 16)> : !torch.vtensor<[1,1,?,?],i1>
    %int0_43 = torch.constant.int 0
    %int6 = torch.constant.int 6
    %int0_44 = torch.constant.int 0
    %cpu_45 = torch.constant.device "cpu"
    %none_46 = torch.constant.none
    %58 = torch.aten.scalar_tensor %int0_43, %int6, %int0_44, %cpu_45, %none_46 : !torch.int, !torch.int, !torch.int, !torch.Device, !torch.none -> !torch.vtensor<[],f32>
    %float-Inf = torch.constant.float 0xFFF0000000000000
    %int6_47 = torch.constant.int 6
    %int0_48 = torch.constant.int 0
    %cpu_49 = torch.constant.device "cpu"
    %none_50 = torch.constant.none
    %59 = torch.aten.scalar_tensor %float-Inf, %int6_47, %int0_48, %cpu_49, %none_50 : !torch.float, !torch.int, !torch.int, !torch.Device, !torch.none -> !torch.vtensor<[],f32>
    %60 = torch.aten.where.self %57, %59, %58 : !torch.vtensor<[1,1,?,?],i1>, !torch.vtensor<[],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[1,1,?,?],f32>
    torch.bind_symbolic_shape %60, [%32], affine_map<()[s0] -> (1, 1, s0 * 16, s0 * 16)> : !torch.vtensor<[1,1,?,?],f32>
    %int5 = torch.constant.int 5
    %61 = torch.prims.convert_element_type %60, %int5 : !torch.vtensor<[1,1,?,?],f32>, !torch.int -> !torch.vtensor<[1,1,?,?],f16>
    torch.bind_symbolic_shape %61, [%32], affine_map<()[s0] -> (1, 1, s0 * 16, s0 * 16)> : !torch.vtensor<[1,1,?,?],f16>
    %int5_51 = torch.constant.int 5
    %62 = torch.prims.convert_element_type %61, %int5_51 : !torch.vtensor<[1,1,?,?],f16>, !torch.int -> !torch.vtensor<[1,1,?,?],f16>
    torch.bind_symbolic_shape %62, [%32], affine_map<()[s0] -> (1, 1, s0 * 16, s0 * 16)> : !torch.vtensor<[1,1,?,?],f16>
    %int5_52 = torch.constant.int 5
    %63 = torch.prims.convert_element_type %0, %int5_52 : !torch.vtensor<[256,256],f32>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int-1_53 = torch.constant.int -1
    %false_54 = torch.constant.bool false
    %false_55 = torch.constant.bool false
    %64 = torch.aten.embedding %63, %arg0, %int-1_53, %false_54, %false_55 : !torch.vtensor<[256,256],f16>, !torch.vtensor<[1,?],si64>, !torch.int, !torch.bool, !torch.bool -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %64, [%32], affine_map<()[s0] -> (1, s0 * 16, 256)> : !torch.vtensor<[1,?,256],f16>
    %int6_56 = torch.constant.int 6
    %65 = torch.prims.convert_element_type %64, %int6_56 : !torch.vtensor<[1,?,256],f16>, !torch.int -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %65, [%32], affine_map<()[s0] -> (1, s0 * 16, 256)> : !torch.vtensor<[1,?,256],f32>
    %int2_57 = torch.constant.int 2
    %66 = torch.aten.pow.Tensor_Scalar %65, %int2_57 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %66, [%32], affine_map<()[s0] -> (1, s0 * 16, 256)> : !torch.vtensor<[1,?,256],f32>
    %int-1_58 = torch.constant.int -1
    %67 = torch.prim.ListConstruct %int-1_58 : (!torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %none_59 = torch.constant.none
    %68 = torch.aten.mean.dim %66, %67, %true, %none_59 : !torch.vtensor<[1,?,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %68, [%32], affine_map<()[s0] -> (1, s0 * 16, 1)> : !torch.vtensor<[1,?,1],f32>
    %float1.000000e-02 = torch.constant.float 1.000000e-02
    %int1_60 = torch.constant.int 1
    %69 = torch.aten.add.Scalar %68, %float1.000000e-02, %int1_60 : !torch.vtensor<[1,?,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %69, [%32], affine_map<()[s0] -> (1, s0 * 16, 1)> : !torch.vtensor<[1,?,1],f32>
    %70 = torch.aten.rsqrt %69 : !torch.vtensor<[1,?,1],f32> -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %70, [%32], affine_map<()[s0] -> (1, s0 * 16, 1)> : !torch.vtensor<[1,?,1],f32>
    %71 = torch.aten.mul.Tensor %65, %70 : !torch.vtensor<[1,?,256],f32>, !torch.vtensor<[1,?,1],f32> -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %71, [%32], affine_map<()[s0] -> (1, s0 * 16, 256)> : !torch.vtensor<[1,?,256],f32>
    %int6_61 = torch.constant.int 6
    %72 = torch.prims.convert_element_type %71, %int6_61 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %72, [%32], affine_map<()[s0] -> (1, s0 * 16, 256)> : !torch.vtensor<[1,?,256],f32>
    %73 = torch.aten.mul.Tensor %1, %72 : !torch.vtensor<[256],f32>, !torch.vtensor<[1,?,256],f32> -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %73, [%32], affine_map<()[s0] -> (1, s0 * 16, 256)> : !torch.vtensor<[1,?,256],f32>
    %int5_62 = torch.constant.int 5
    %74 = torch.prims.convert_element_type %73, %int5_62 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %74, [%32], affine_map<()[s0] -> (1, s0 * 16, 256)> : !torch.vtensor<[1,?,256],f16>
    %int5_63 = torch.constant.int 5
    %75 = torch.prims.convert_element_type %2, %int5_63 : !torch.vtensor<[256,256],f32>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int-2 = torch.constant.int -2
    %int-1_64 = torch.constant.int -1
    %76 = torch.aten.transpose.int %75, %int-2, %int-1_64 : !torch.vtensor<[256,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,256],f16>
    %int5_65 = torch.constant.int 5
    %77 = torch.prims.convert_element_type %76, %int5_65 : !torch.vtensor<[256,256],f16>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int256 = torch.constant.int 256
    %78 = torch.prim.ListConstruct %36, %int256 : (!torch.int, !torch.int) -> !torch.list<int>
    %79 = torch.aten.view %74, %78 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %79, [%32], affine_map<()[s0] -> (s0 * 16, 256)> : !torch.vtensor<[?,256],f16>
    %80 = torch.aten.mm %79, %77 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,256],f16> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %80, [%32], affine_map<()[s0] -> (s0 * 16, 256)> : !torch.vtensor<[?,256],f16>
    %int1_66 = torch.constant.int 1
    %int256_67 = torch.constant.int 256
    %81 = torch.prim.ListConstruct %int1_66, %36, %int256_67 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %82 = torch.aten.view %80, %81 : !torch.vtensor<[?,256],f16>, !torch.list<int> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %82, [%32], affine_map<()[s0] -> (1, s0 * 16, 256)> : !torch.vtensor<[1,?,256],f16>
    %int5_68 = torch.constant.int 5
    %83 = torch.prims.convert_element_type %3, %int5_68 : !torch.vtensor<[128,256],f32>, !torch.int -> !torch.vtensor<[128,256],f16>
    %int-2_69 = torch.constant.int -2
    %int-1_70 = torch.constant.int -1
    %84 = torch.aten.transpose.int %83, %int-2_69, %int-1_70 : !torch.vtensor<[128,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,128],f16>
    %int5_71 = torch.constant.int 5
    %85 = torch.prims.convert_element_type %84, %int5_71 : !torch.vtensor<[256,128],f16>, !torch.int -> !torch.vtensor<[256,128],f16>
    %int256_72 = torch.constant.int 256
    %86 = torch.prim.ListConstruct %36, %int256_72 : (!torch.int, !torch.int) -> !torch.list<int>
    %87 = torch.aten.view %74, %86 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %87, [%32], affine_map<()[s0] -> (s0 * 16, 256)> : !torch.vtensor<[?,256],f16>
    %88 = torch.aten.mm %87, %85 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,128],f16> -> !torch.vtensor<[?,128],f16>
    torch.bind_symbolic_shape %88, [%32], affine_map<()[s0] -> (s0 * 16, 128)> : !torch.vtensor<[?,128],f16>
    %int1_73 = torch.constant.int 1
    %int128_74 = torch.constant.int 128
    %89 = torch.prim.ListConstruct %int1_73, %36, %int128_74 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %90 = torch.aten.view %88, %89 : !torch.vtensor<[?,128],f16>, !torch.list<int> -> !torch.vtensor<[1,?,128],f16>
    torch.bind_symbolic_shape %90, [%32], affine_map<()[s0] -> (1, s0 * 16, 128)> : !torch.vtensor<[1,?,128],f16>
    %int5_75 = torch.constant.int 5
    %91 = torch.prims.convert_element_type %4, %int5_75 : !torch.vtensor<[128,256],f32>, !torch.int -> !torch.vtensor<[128,256],f16>
    %int-2_76 = torch.constant.int -2
    %int-1_77 = torch.constant.int -1
    %92 = torch.aten.transpose.int %91, %int-2_76, %int-1_77 : !torch.vtensor<[128,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,128],f16>
    %int5_78 = torch.constant.int 5
    %93 = torch.prims.convert_element_type %92, %int5_78 : !torch.vtensor<[256,128],f16>, !torch.int -> !torch.vtensor<[256,128],f16>
    %int256_79 = torch.constant.int 256
    %94 = torch.prim.ListConstruct %36, %int256_79 : (!torch.int, !torch.int) -> !torch.list<int>
    %95 = torch.aten.view %74, %94 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %95, [%32], affine_map<()[s0] -> (s0 * 16, 256)> : !torch.vtensor<[?,256],f16>
    %96 = torch.aten.mm %95, %93 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,128],f16> -> !torch.vtensor<[?,128],f16>
    torch.bind_symbolic_shape %96, [%32], affine_map<()[s0] -> (s0 * 16, 128)> : !torch.vtensor<[?,128],f16>
    %int1_80 = torch.constant.int 1
    %int128_81 = torch.constant.int 128
    %97 = torch.prim.ListConstruct %int1_80, %36, %int128_81 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %98 = torch.aten.view %96, %97 : !torch.vtensor<[?,128],f16>, !torch.list<int> -> !torch.vtensor<[1,?,128],f16>
    torch.bind_symbolic_shape %98, [%32], affine_map<()[s0] -> (1, s0 * 16, 128)> : !torch.vtensor<[1,?,128],f16>
    %int1_82 = torch.constant.int 1
    %int8 = torch.constant.int 8
    %int32 = torch.constant.int 32
    %99 = torch.prim.ListConstruct %int1_82, %36, %int8, %int32 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %100 = torch.aten.view %82, %99 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[1,?,8,32],f16>
    torch.bind_symbolic_shape %100, [%32], affine_map<()[s0] -> (1, s0 * 16, 8, 32)> : !torch.vtensor<[1,?,8,32],f16>
    %int1_83 = torch.constant.int 1
    %int4 = torch.constant.int 4
    %int32_84 = torch.constant.int 32
    %101 = torch.prim.ListConstruct %int1_83, %36, %int4, %int32_84 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %102 = torch.aten.view %90, %101 : !torch.vtensor<[1,?,128],f16>, !torch.list<int> -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %102, [%32], affine_map<()[s0] -> (1, s0 * 16, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int1_85 = torch.constant.int 1
    %int4_86 = torch.constant.int 4
    %int32_87 = torch.constant.int 32
    %103 = torch.prim.ListConstruct %int1_85, %36, %int4_86, %int32_87 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %104 = torch.aten.view %98, %103 : !torch.vtensor<[1,?,128],f16>, !torch.list<int> -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %104, [%32], affine_map<()[s0] -> (1, s0 * 16, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int128_88 = torch.constant.int 128
    %none_89 = torch.constant.none
    %none_90 = torch.constant.none
    %cpu_91 = torch.constant.device "cpu"
    %false_92 = torch.constant.bool false
    %105 = torch.aten.arange %int128_88, %none_89, %none_90, %cpu_91, %false_92 : !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[128],si64>
    %int0_93 = torch.constant.int 0
    %int32_94 = torch.constant.int 32
    %int2_95 = torch.constant.int 2
    %none_96 = torch.constant.none
    %none_97 = torch.constant.none
    %cpu_98 = torch.constant.device "cpu"
    %false_99 = torch.constant.bool false
    %106 = torch.aten.arange.start_step %int0_93, %int32_94, %int2_95, %none_96, %none_97, %cpu_98, %false_99 : !torch.int, !torch.int, !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[16],si64>
    %int0_100 = torch.constant.int 0
    %int0_101 = torch.constant.int 0
    %int16 = torch.constant.int 16
    %int1_102 = torch.constant.int 1
    %107 = torch.aten.slice.Tensor %106, %int0_100, %int0_101, %int16, %int1_102 : !torch.vtensor<[16],si64>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[16],si64>
    %int6_103 = torch.constant.int 6
    %108 = torch.prims.convert_element_type %107, %int6_103 : !torch.vtensor<[16],si64>, !torch.int -> !torch.vtensor<[16],f32>
    %int32_104 = torch.constant.int 32
    %109 = torch.aten.div.Scalar %108, %int32_104 : !torch.vtensor<[16],f32>, !torch.int -> !torch.vtensor<[16],f32>
    %float5.000000e05 = torch.constant.float 5.000000e+05
    %110 = torch.aten.pow.Scalar %float5.000000e05, %109 : !torch.float, !torch.vtensor<[16],f32> -> !torch.vtensor<[16],f32>
    %111 = torch.aten.reciprocal %110 : !torch.vtensor<[16],f32> -> !torch.vtensor<[16],f32>
    %float1.000000e00 = torch.constant.float 1.000000e+00
    %112 = torch.aten.mul.Scalar %111, %float1.000000e00 : !torch.vtensor<[16],f32>, !torch.float -> !torch.vtensor<[16],f32>
    %int128_105 = torch.constant.int 128
    %int1_106 = torch.constant.int 1
    %113 = torch.prim.ListConstruct %int128_105, %int1_106 : (!torch.int, !torch.int) -> !torch.list<int>
    %114 = torch.aten.view %105, %113 : !torch.vtensor<[128],si64>, !torch.list<int> -> !torch.vtensor<[128,1],si64>
    %115 = torch.aten.mul.Tensor %114, %112 : !torch.vtensor<[128,1],si64>, !torch.vtensor<[16],f32> -> !torch.vtensor<[128,16],f32>
    %int6_107 = torch.constant.int 6
    %116 = torch.prims.convert_element_type %115, %int6_107 : !torch.vtensor<[128,16],f32>, !torch.int -> !torch.vtensor<[128,16],f32>
    %117 = torch.aten.cos %116 : !torch.vtensor<[128,16],f32> -> !torch.vtensor<[128,16],f32>
    %118 = torch.aten.sin %116 : !torch.vtensor<[128,16],f32> -> !torch.vtensor<[128,16],f32>
    %119 = torch.aten.complex %117, %118 : !torch.vtensor<[128,16],f32>, !torch.vtensor<[128,16],f32> -> !torch.vtensor<[128,16],complex<f32>>
    %int0_108 = torch.constant.int 0
    %int0_109 = torch.constant.int 0
    %int1_110 = torch.constant.int 1
    %120 = torch.aten.slice.Tensor %119, %int0_108, %int0_109, %36, %int1_110 : !torch.vtensor<[128,16],complex<f32>>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,16],complex<f32>>
    torch.bind_symbolic_shape %120, [%32], affine_map<()[s0] -> (s0 * 16, 16)> : !torch.vtensor<[?,16],complex<f32>>
    %int1_111 = torch.constant.int 1
    %int0_112 = torch.constant.int 0
    %int9223372036854775807_113 = torch.constant.int 9223372036854775807
    %int1_114 = torch.constant.int 1
    %121 = torch.aten.slice.Tensor %120, %int1_111, %int0_112, %int9223372036854775807_113, %int1_114 : !torch.vtensor<[?,16],complex<f32>>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,16],complex<f32>>
    torch.bind_symbolic_shape %121, [%32], affine_map<()[s0] -> (s0 * 16, 16)> : !torch.vtensor<[?,16],complex<f32>>
    %int0_115 = torch.constant.int 0
    %122 = torch.aten.unsqueeze %121, %int0_115 : !torch.vtensor<[?,16],complex<f32>>, !torch.int -> !torch.vtensor<[1,?,16],complex<f32>>
    torch.bind_symbolic_shape %122, [%32], affine_map<()[s0] -> (1, s0 * 16, 16)> : !torch.vtensor<[1,?,16],complex<f32>>
    %int2_116 = torch.constant.int 2
    %123 = torch.aten.unsqueeze %122, %int2_116 : !torch.vtensor<[1,?,16],complex<f32>>, !torch.int -> !torch.vtensor<[1,?,1,16],complex<f32>>
    torch.bind_symbolic_shape %123, [%32], affine_map<()[s0] -> (1, s0 * 16, 1, 16)> : !torch.vtensor<[1,?,1,16],complex<f32>>
    %int3_117 = torch.constant.int 3
    %int0_118 = torch.constant.int 0
    %int9223372036854775807_119 = torch.constant.int 9223372036854775807
    %int1_120 = torch.constant.int 1
    %124 = torch.aten.slice.Tensor %123, %int3_117, %int0_118, %int9223372036854775807_119, %int1_120 : !torch.vtensor<[1,?,1,16],complex<f32>>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,1,16],complex<f32>>
    torch.bind_symbolic_shape %124, [%32], affine_map<()[s0] -> (1, s0 * 16, 1, 16)> : !torch.vtensor<[1,?,1,16],complex<f32>>
    %125 = torch_c.to_builtin_tensor %100 : !torch.vtensor<[1,?,8,32],f16> -> tensor<1x?x8x32xf16>
    %c1 = arith.constant 1 : index
    %dim = tensor.dim %125, %c1 : tensor<1x?x8x32xf16>
    %126 = flow.tensor.bitcast %125 : tensor<1x?x8x32xf16>{%dim} -> tensor<1x?x8x16xcomplex<f16>>{%dim}
    %127 = torch_c.from_builtin_tensor %126 : tensor<1x?x8x16xcomplex<f16>> -> !torch.vtensor<[1,?,8,16],complex<f16>>
    torch.bind_symbolic_shape %127, [%32], affine_map<()[s0] -> (1, s0 * 16, 8, 16)> : !torch.vtensor<[1,?,8,16],complex<f16>>
    %128 = torch.aten.mul.Tensor %127, %124 : !torch.vtensor<[1,?,8,16],complex<f16>>, !torch.vtensor<[1,?,1,16],complex<f32>> -> !torch.vtensor<[1,?,8,16],complex<f32>>
    torch.bind_symbolic_shape %128, [%32], affine_map<()[s0] -> (1, s0 * 16, 8, 16)> : !torch.vtensor<[1,?,8,16],complex<f32>>
    %129 = torch_c.to_builtin_tensor %128 : !torch.vtensor<[1,?,8,16],complex<f32>> -> tensor<1x?x8x16xcomplex<f32>>
    %c1_121 = arith.constant 1 : index
    %dim_122 = tensor.dim %129, %c1_121 : tensor<1x?x8x16xcomplex<f32>>
    %130 = flow.tensor.bitcast %129 : tensor<1x?x8x16xcomplex<f32>>{%dim_122} -> tensor<1x?x8x32xf32>{%dim_122}
    %131 = torch_c.from_builtin_tensor %130 : tensor<1x?x8x32xf32> -> !torch.vtensor<[1,?,8,32],f32>
    torch.bind_symbolic_shape %131, [%32], affine_map<()[s0] -> (1, s0 * 16, 8, 32)> : !torch.vtensor<[1,?,8,32],f32>
    %int5_123 = torch.constant.int 5
    %132 = torch.prims.convert_element_type %131, %int5_123 : !torch.vtensor<[1,?,8,32],f32>, !torch.int -> !torch.vtensor<[1,?,8,32],f16>
    torch.bind_symbolic_shape %132, [%32], affine_map<()[s0] -> (1, s0 * 16, 8, 32)> : !torch.vtensor<[1,?,8,32],f16>
    %int128_124 = torch.constant.int 128
    %none_125 = torch.constant.none
    %none_126 = torch.constant.none
    %cpu_127 = torch.constant.device "cpu"
    %false_128 = torch.constant.bool false
    %133 = torch.aten.arange %int128_124, %none_125, %none_126, %cpu_127, %false_128 : !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[128],si64>
    %int0_129 = torch.constant.int 0
    %int32_130 = torch.constant.int 32
    %int2_131 = torch.constant.int 2
    %none_132 = torch.constant.none
    %none_133 = torch.constant.none
    %cpu_134 = torch.constant.device "cpu"
    %false_135 = torch.constant.bool false
    %134 = torch.aten.arange.start_step %int0_129, %int32_130, %int2_131, %none_132, %none_133, %cpu_134, %false_135 : !torch.int, !torch.int, !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[16],si64>
    %int0_136 = torch.constant.int 0
    %int0_137 = torch.constant.int 0
    %int16_138 = torch.constant.int 16
    %int1_139 = torch.constant.int 1
    %135 = torch.aten.slice.Tensor %134, %int0_136, %int0_137, %int16_138, %int1_139 : !torch.vtensor<[16],si64>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[16],si64>
    %int6_140 = torch.constant.int 6
    %136 = torch.prims.convert_element_type %135, %int6_140 : !torch.vtensor<[16],si64>, !torch.int -> !torch.vtensor<[16],f32>
    %int32_141 = torch.constant.int 32
    %137 = torch.aten.div.Scalar %136, %int32_141 : !torch.vtensor<[16],f32>, !torch.int -> !torch.vtensor<[16],f32>
    %float5.000000e05_142 = torch.constant.float 5.000000e+05
    %138 = torch.aten.pow.Scalar %float5.000000e05_142, %137 : !torch.float, !torch.vtensor<[16],f32> -> !torch.vtensor<[16],f32>
    %139 = torch.aten.reciprocal %138 : !torch.vtensor<[16],f32> -> !torch.vtensor<[16],f32>
    %float1.000000e00_143 = torch.constant.float 1.000000e+00
    %140 = torch.aten.mul.Scalar %139, %float1.000000e00_143 : !torch.vtensor<[16],f32>, !torch.float -> !torch.vtensor<[16],f32>
    %int128_144 = torch.constant.int 128
    %int1_145 = torch.constant.int 1
    %141 = torch.prim.ListConstruct %int128_144, %int1_145 : (!torch.int, !torch.int) -> !torch.list<int>
    %142 = torch.aten.view %133, %141 : !torch.vtensor<[128],si64>, !torch.list<int> -> !torch.vtensor<[128,1],si64>
    %143 = torch.aten.mul.Tensor %142, %140 : !torch.vtensor<[128,1],si64>, !torch.vtensor<[16],f32> -> !torch.vtensor<[128,16],f32>
    %int6_146 = torch.constant.int 6
    %144 = torch.prims.convert_element_type %143, %int6_146 : !torch.vtensor<[128,16],f32>, !torch.int -> !torch.vtensor<[128,16],f32>
    %145 = torch.aten.cos %144 : !torch.vtensor<[128,16],f32> -> !torch.vtensor<[128,16],f32>
    %146 = torch.aten.sin %144 : !torch.vtensor<[128,16],f32> -> !torch.vtensor<[128,16],f32>
    %147 = torch.aten.complex %145, %146 : !torch.vtensor<[128,16],f32>, !torch.vtensor<[128,16],f32> -> !torch.vtensor<[128,16],complex<f32>>
    %int0_147 = torch.constant.int 0
    %int0_148 = torch.constant.int 0
    %int1_149 = torch.constant.int 1
    %148 = torch.aten.slice.Tensor %147, %int0_147, %int0_148, %36, %int1_149 : !torch.vtensor<[128,16],complex<f32>>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,16],complex<f32>>
    torch.bind_symbolic_shape %148, [%32], affine_map<()[s0] -> (s0 * 16, 16)> : !torch.vtensor<[?,16],complex<f32>>
    %int1_150 = torch.constant.int 1
    %int0_151 = torch.constant.int 0
    %int9223372036854775807_152 = torch.constant.int 9223372036854775807
    %int1_153 = torch.constant.int 1
    %149 = torch.aten.slice.Tensor %148, %int1_150, %int0_151, %int9223372036854775807_152, %int1_153 : !torch.vtensor<[?,16],complex<f32>>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,16],complex<f32>>
    torch.bind_symbolic_shape %149, [%32], affine_map<()[s0] -> (s0 * 16, 16)> : !torch.vtensor<[?,16],complex<f32>>
    %int0_154 = torch.constant.int 0
    %150 = torch.aten.unsqueeze %149, %int0_154 : !torch.vtensor<[?,16],complex<f32>>, !torch.int -> !torch.vtensor<[1,?,16],complex<f32>>
    torch.bind_symbolic_shape %150, [%32], affine_map<()[s0] -> (1, s0 * 16, 16)> : !torch.vtensor<[1,?,16],complex<f32>>
    %int2_155 = torch.constant.int 2
    %151 = torch.aten.unsqueeze %150, %int2_155 : !torch.vtensor<[1,?,16],complex<f32>>, !torch.int -> !torch.vtensor<[1,?,1,16],complex<f32>>
    torch.bind_symbolic_shape %151, [%32], affine_map<()[s0] -> (1, s0 * 16, 1, 16)> : !torch.vtensor<[1,?,1,16],complex<f32>>
    %int3_156 = torch.constant.int 3
    %int0_157 = torch.constant.int 0
    %int9223372036854775807_158 = torch.constant.int 9223372036854775807
    %int1_159 = torch.constant.int 1
    %152 = torch.aten.slice.Tensor %151, %int3_156, %int0_157, %int9223372036854775807_158, %int1_159 : !torch.vtensor<[1,?,1,16],complex<f32>>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,1,16],complex<f32>>
    torch.bind_symbolic_shape %152, [%32], affine_map<()[s0] -> (1, s0 * 16, 1, 16)> : !torch.vtensor<[1,?,1,16],complex<f32>>
    %153 = torch_c.to_builtin_tensor %102 : !torch.vtensor<[1,?,4,32],f16> -> tensor<1x?x4x32xf16>
    %c1_160 = arith.constant 1 : index
    %dim_161 = tensor.dim %153, %c1_160 : tensor<1x?x4x32xf16>
    %154 = flow.tensor.bitcast %153 : tensor<1x?x4x32xf16>{%dim_161} -> tensor<1x?x4x16xcomplex<f16>>{%dim_161}
    %155 = torch_c.from_builtin_tensor %154 : tensor<1x?x4x16xcomplex<f16>> -> !torch.vtensor<[1,?,4,16],complex<f16>>
    torch.bind_symbolic_shape %155, [%32], affine_map<()[s0] -> (1, s0 * 16, 4, 16)> : !torch.vtensor<[1,?,4,16],complex<f16>>
    %156 = torch.aten.mul.Tensor %155, %152 : !torch.vtensor<[1,?,4,16],complex<f16>>, !torch.vtensor<[1,?,1,16],complex<f32>> -> !torch.vtensor<[1,?,4,16],complex<f32>>
    torch.bind_symbolic_shape %156, [%32], affine_map<()[s0] -> (1, s0 * 16, 4, 16)> : !torch.vtensor<[1,?,4,16],complex<f32>>
    %157 = torch_c.to_builtin_tensor %156 : !torch.vtensor<[1,?,4,16],complex<f32>> -> tensor<1x?x4x16xcomplex<f32>>
    %c1_162 = arith.constant 1 : index
    %dim_163 = tensor.dim %157, %c1_162 : tensor<1x?x4x16xcomplex<f32>>
    %158 = flow.tensor.bitcast %157 : tensor<1x?x4x16xcomplex<f32>>{%dim_163} -> tensor<1x?x4x32xf32>{%dim_163}
    %159 = torch_c.from_builtin_tensor %158 : tensor<1x?x4x32xf32> -> !torch.vtensor<[1,?,4,32],f32>
    torch.bind_symbolic_shape %159, [%32], affine_map<()[s0] -> (1, s0 * 16, 4, 32)> : !torch.vtensor<[1,?,4,32],f32>
    %int5_164 = torch.constant.int 5
    %160 = torch.prims.convert_element_type %159, %int5_164 : !torch.vtensor<[1,?,4,32],f32>, !torch.int -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %160, [%32], affine_map<()[s0] -> (1, s0 * 16, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int3_165 = torch.constant.int 3
    %int2_166 = torch.constant.int 2
    %int16_167 = torch.constant.int 16
    %int4_168 = torch.constant.int 4
    %int32_169 = torch.constant.int 32
    %161 = torch.prim.ListConstruct %35, %int3_165, %int2_166, %int16_167, %int4_168, %int32_169 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %162 = torch.aten.view %30, %161 : !torch.vtensor<[?,12288],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,16,4,32],f16>
    torch.bind_symbolic_shape %162, [%33], affine_map<()[s0] -> (s0, 3, 2, 16, 4, 32)> : !torch.vtensor<[?,3,2,16,4,32],f16>
    %int3_170 = torch.constant.int 3
    %163 = torch.aten.mul.int %35, %int3_170 : !torch.int, !torch.int -> !torch.int
    %int2_171 = torch.constant.int 2
    %164 = torch.aten.mul.int %163, %int2_171 : !torch.int, !torch.int -> !torch.int
    %int16_172 = torch.constant.int 16
    %int4_173 = torch.constant.int 4
    %int32_174 = torch.constant.int 32
    %165 = torch.prim.ListConstruct %164, %int16_172, %int4_173, %int32_174 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %166 = torch.aten.view %162, %165 : !torch.vtensor<[?,3,2,16,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,16,4,32],f16>
    torch.bind_symbolic_shape %166, [%33], affine_map<()[s0] -> (s0 * 6, 16, 4, 32)> : !torch.vtensor<[?,16,4,32],f16>
    %int6_175 = torch.constant.int 6
    %167 = torch.aten.mul.Scalar %arg2, %int6_175 : !torch.vtensor<[1,?],si64>, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %167, [%32], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %int0_176 = torch.constant.int 0
    %int1_177 = torch.constant.int 1
    %168 = torch.aten.add.Scalar %167, %int0_176, %int1_177 : !torch.vtensor<[1,?],si64>, !torch.int, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %168, [%32], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %int1_178 = torch.constant.int 1
    %int16_179 = torch.constant.int 16
    %int4_180 = torch.constant.int 4
    %int32_181 = torch.constant.int 32
    %169 = torch.prim.ListConstruct %int1_178, %34, %int16_179, %int4_180, %int32_181 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %170 = torch.aten.view %160, %169 : !torch.vtensor<[1,?,4,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,16,4,32],f16>
    torch.bind_symbolic_shape %170, [%32], affine_map<()[s0] -> (1, s0, 16, 4, 32)> : !torch.vtensor<[1,?,16,4,32],f16>
    %int16_182 = torch.constant.int 16
    %int4_183 = torch.constant.int 4
    %int32_184 = torch.constant.int 32
    %171 = torch.prim.ListConstruct %34, %int16_182, %int4_183, %int32_184 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %172 = torch.aten.view %170, %171 : !torch.vtensor<[1,?,16,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,16,4,32],f16>
    torch.bind_symbolic_shape %172, [%32], affine_map<()[s0] -> (s0, 16, 4, 32)> : !torch.vtensor<[?,16,4,32],f16>
    %173 = torch.prim.ListConstruct %34 : (!torch.int) -> !torch.list<int>
    %174 = torch.aten.view %168, %173 : !torch.vtensor<[1,?],si64>, !torch.list<int> -> !torch.vtensor<[?],si64>
    torch.bind_symbolic_shape %174, [%32], affine_map<()[s0] -> (s0)> : !torch.vtensor<[?],si64>
    %int1_185 = torch.constant.int 1
    %int16_186 = torch.constant.int 16
    %int4_187 = torch.constant.int 4
    %int32_188 = torch.constant.int 32
    %175 = torch.prim.ListConstruct %int1_185, %34, %int16_186, %int4_187, %int32_188 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %176 = torch.aten.view %104, %175 : !torch.vtensor<[1,?,4,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,16,4,32],f16>
    torch.bind_symbolic_shape %176, [%32], affine_map<()[s0] -> (1, s0, 16, 4, 32)> : !torch.vtensor<[1,?,16,4,32],f16>
    %int16_189 = torch.constant.int 16
    %int4_190 = torch.constant.int 4
    %int32_191 = torch.constant.int 32
    %177 = torch.prim.ListConstruct %34, %int16_189, %int4_190, %int32_191 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %178 = torch.aten.view %176, %177 : !torch.vtensor<[1,?,16,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,16,4,32],f16>
    torch.bind_symbolic_shape %178, [%32], affine_map<()[s0] -> (s0, 16, 4, 32)> : !torch.vtensor<[?,16,4,32],f16>
    %int1_192 = torch.constant.int 1
    %int1_193 = torch.constant.int 1
    %179 = torch.aten.add.Scalar %168, %int1_192, %int1_193 : !torch.vtensor<[1,?],si64>, !torch.int, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %179, [%32], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %180 = torch.prim.ListConstruct %34 : (!torch.int) -> !torch.list<int>
    %181 = torch.aten.view %179, %180 : !torch.vtensor<[1,?],si64>, !torch.list<int> -> !torch.vtensor<[?],si64>
    torch.bind_symbolic_shape %181, [%32], affine_map<()[s0] -> (s0)> : !torch.vtensor<[?],si64>
    %182 = torch.prim.ListConstruct %174, %181 : (!torch.vtensor<[?],si64>, !torch.vtensor<[?],si64>) -> !torch.list<vtensor>
    %int0_194 = torch.constant.int 0
    %183 = torch.aten.cat %182, %int0_194 : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[?],si64>
    torch.bind_symbolic_shape %183, [%32], affine_map<()[s0] -> (s0 * 2)> : !torch.vtensor<[?],si64>
    %184 = torch.prim.ListConstruct %172, %178 : (!torch.vtensor<[?,16,4,32],f16>, !torch.vtensor<[?,16,4,32],f16>) -> !torch.list<vtensor>
    %int0_195 = torch.constant.int 0
    %185 = torch.aten.cat %184, %int0_195 : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[?,16,4,32],f16>
    torch.bind_symbolic_shape %185, [%32], affine_map<()[s0] -> (s0 * 2, 16, 4, 32)> : !torch.vtensor<[?,16,4,32],f16>
    %186 = torch.prim.ListConstruct %183 : (!torch.vtensor<[?],si64>) -> !torch.list<optional<vtensor>>
    %false_196 = torch.constant.bool false
    %187 = torch.aten.index_put %166, %186, %185, %false_196 : !torch.vtensor<[?,16,4,32],f16>, !torch.list<optional<vtensor>>, !torch.vtensor<[?,16,4,32],f16>, !torch.bool -> !torch.vtensor<[?,16,4,32],f16>
    torch.bind_symbolic_shape %187, [%33], affine_map<()[s0] -> (s0 * 6, 16, 4, 32)> : !torch.vtensor<[?,16,4,32],f16>
    %int3_197 = torch.constant.int 3
    %int2_198 = torch.constant.int 2
    %int16_199 = torch.constant.int 16
    %int4_200 = torch.constant.int 4
    %int32_201 = torch.constant.int 32
    %188 = torch.prim.ListConstruct %35, %int3_197, %int2_198, %int16_199, %int4_200, %int32_201 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %189 = torch.aten.view %187, %188 : !torch.vtensor<[?,16,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,16,4,32],f16>
    torch.bind_symbolic_shape %189, [%33], affine_map<()[s0] -> (s0, 3, 2, 16, 4, 32)> : !torch.vtensor<[?,3,2,16,4,32],f16>
    %int12288 = torch.constant.int 12288
    %190 = torch.prim.ListConstruct %35, %int12288 : (!torch.int, !torch.int) -> !torch.list<int>
    %191 = torch.aten.view %189, %190 : !torch.vtensor<[?,3,2,16,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,12288],f16>
    torch.bind_symbolic_shape %191, [%33], affine_map<()[s0] -> (s0, 12288)> : !torch.vtensor<[?,12288],f16>
    %int-2_202 = torch.constant.int -2
    %192 = torch.aten.unsqueeze %160, %int-2_202 : !torch.vtensor<[1,?,4,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,1,32],f16>
    torch.bind_symbolic_shape %192, [%32], affine_map<()[s0] -> (1, s0 * 16, 4, 1, 32)> : !torch.vtensor<[1,?,4,1,32],f16>
    %int1_203 = torch.constant.int 1
    %int4_204 = torch.constant.int 4
    %int2_205 = torch.constant.int 2
    %int32_206 = torch.constant.int 32
    %193 = torch.prim.ListConstruct %int1_203, %36, %int4_204, %int2_205, %int32_206 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_207 = torch.constant.bool false
    %194 = torch.aten.expand %192, %193, %false_207 : !torch.vtensor<[1,?,4,1,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,?,4,2,32],f16>
    torch.bind_symbolic_shape %194, [%32], affine_map<()[s0] -> (1, s0 * 16, 4, 2, 32)> : !torch.vtensor<[1,?,4,2,32],f16>
    %int0_208 = torch.constant.int 0
    %195 = torch.aten.clone %194, %int0_208 : !torch.vtensor<[1,?,4,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,2,32],f16>
    torch.bind_symbolic_shape %195, [%32], affine_map<()[s0] -> (1, s0 * 16, 4, 2, 32)> : !torch.vtensor<[1,?,4,2,32],f16>
    %int1_209 = torch.constant.int 1
    %int8_210 = torch.constant.int 8
    %int32_211 = torch.constant.int 32
    %196 = torch.prim.ListConstruct %int1_209, %36, %int8_210, %int32_211 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %197 = torch.aten._unsafe_view %195, %196 : !torch.vtensor<[1,?,4,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,8,32],f16>
    torch.bind_symbolic_shape %197, [%32], affine_map<()[s0] -> (1, s0 * 16, 8, 32)> : !torch.vtensor<[1,?,8,32],f16>
    %int-2_212 = torch.constant.int -2
    %198 = torch.aten.unsqueeze %104, %int-2_212 : !torch.vtensor<[1,?,4,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,1,32],f16>
    torch.bind_symbolic_shape %198, [%32], affine_map<()[s0] -> (1, s0 * 16, 4, 1, 32)> : !torch.vtensor<[1,?,4,1,32],f16>
    %int1_213 = torch.constant.int 1
    %int4_214 = torch.constant.int 4
    %int2_215 = torch.constant.int 2
    %int32_216 = torch.constant.int 32
    %199 = torch.prim.ListConstruct %int1_213, %36, %int4_214, %int2_215, %int32_216 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_217 = torch.constant.bool false
    %200 = torch.aten.expand %198, %199, %false_217 : !torch.vtensor<[1,?,4,1,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,?,4,2,32],f16>
    torch.bind_symbolic_shape %200, [%32], affine_map<()[s0] -> (1, s0 * 16, 4, 2, 32)> : !torch.vtensor<[1,?,4,2,32],f16>
    %int0_218 = torch.constant.int 0
    %201 = torch.aten.clone %200, %int0_218 : !torch.vtensor<[1,?,4,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,2,32],f16>
    torch.bind_symbolic_shape %201, [%32], affine_map<()[s0] -> (1, s0 * 16, 4, 2, 32)> : !torch.vtensor<[1,?,4,2,32],f16>
    %int1_219 = torch.constant.int 1
    %int8_220 = torch.constant.int 8
    %int32_221 = torch.constant.int 32
    %202 = torch.prim.ListConstruct %int1_219, %36, %int8_220, %int32_221 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %203 = torch.aten._unsafe_view %201, %202 : !torch.vtensor<[1,?,4,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,8,32],f16>
    torch.bind_symbolic_shape %203, [%32], affine_map<()[s0] -> (1, s0 * 16, 8, 32)> : !torch.vtensor<[1,?,8,32],f16>
    %int1_222 = torch.constant.int 1
    %int2_223 = torch.constant.int 2
    %204 = torch.aten.transpose.int %132, %int1_222, %int2_223 : !torch.vtensor<[1,?,8,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,8,?,32],f16>
    torch.bind_symbolic_shape %204, [%32], affine_map<()[s0] -> (1, 8, s0 * 16, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %int1_224 = torch.constant.int 1
    %int2_225 = torch.constant.int 2
    %205 = torch.aten.transpose.int %197, %int1_224, %int2_225 : !torch.vtensor<[1,?,8,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,8,?,32],f16>
    torch.bind_symbolic_shape %205, [%32], affine_map<()[s0] -> (1, 8, s0 * 16, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %int1_226 = torch.constant.int 1
    %int2_227 = torch.constant.int 2
    %206 = torch.aten.transpose.int %203, %int1_226, %int2_227 : !torch.vtensor<[1,?,8,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,8,?,32],f16>
    torch.bind_symbolic_shape %206, [%32], affine_map<()[s0] -> (1, 8, s0 * 16, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %int2_228 = torch.constant.int 2
    %int3_229 = torch.constant.int 3
    %207 = torch.aten.transpose.int %205, %int2_228, %int3_229 : !torch.vtensor<[1,8,?,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,8,32,?],f16>
    torch.bind_symbolic_shape %207, [%32], affine_map<()[s0] -> (1, 8, 32, s0 * 16)> : !torch.vtensor<[1,8,32,?],f16>
    %int5_230 = torch.constant.int 5
    %208 = torch.prims.convert_element_type %207, %int5_230 : !torch.vtensor<[1,8,32,?],f16>, !torch.int -> !torch.vtensor<[1,8,32,?],f16>
    torch.bind_symbolic_shape %208, [%32], affine_map<()[s0] -> (1, 8, 32, s0 * 16)> : !torch.vtensor<[1,8,32,?],f16>
    %int1_231 = torch.constant.int 1
    %int8_232 = torch.constant.int 8
    %int32_233 = torch.constant.int 32
    %209 = torch.prim.ListConstruct %int1_231, %int8_232, %36, %int32_233 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_234 = torch.constant.bool false
    %210 = torch.aten.expand %204, %209, %false_234 : !torch.vtensor<[1,8,?,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,8,?,32],f16>
    torch.bind_symbolic_shape %210, [%32], affine_map<()[s0] -> (1, 8, s0 * 16, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %int8_235 = torch.constant.int 8
    %int32_236 = torch.constant.int 32
    %211 = torch.prim.ListConstruct %int8_235, %36, %int32_236 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %212 = torch.aten.view %210, %211 : !torch.vtensor<[1,8,?,32],f16>, !torch.list<int> -> !torch.vtensor<[8,?,32],f16>
    torch.bind_symbolic_shape %212, [%32], affine_map<()[s0] -> (8, s0 * 16, 32)> : !torch.vtensor<[8,?,32],f16>
    %int1_237 = torch.constant.int 1
    %int8_238 = torch.constant.int 8
    %int32_239 = torch.constant.int 32
    %213 = torch.prim.ListConstruct %int1_237, %int8_238, %int32_239, %36 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_240 = torch.constant.bool false
    %214 = torch.aten.expand %208, %213, %false_240 : !torch.vtensor<[1,8,32,?],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,8,32,?],f16>
    torch.bind_symbolic_shape %214, [%32], affine_map<()[s0] -> (1, 8, 32, s0 * 16)> : !torch.vtensor<[1,8,32,?],f16>
    %int8_241 = torch.constant.int 8
    %int32_242 = torch.constant.int 32
    %215 = torch.prim.ListConstruct %int8_241, %int32_242, %36 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %216 = torch.aten.view %214, %215 : !torch.vtensor<[1,8,32,?],f16>, !torch.list<int> -> !torch.vtensor<[8,32,?],f16>
    torch.bind_symbolic_shape %216, [%32], affine_map<()[s0] -> (8, 32, s0 * 16)> : !torch.vtensor<[8,32,?],f16>
    %217 = torch.aten.bmm %212, %216 : !torch.vtensor<[8,?,32],f16>, !torch.vtensor<[8,32,?],f16> -> !torch.vtensor<[8,?,?],f16>
    torch.bind_symbolic_shape %217, [%32], affine_map<()[s0] -> (8, s0 * 16, s0 * 16)> : !torch.vtensor<[8,?,?],f16>
    %int1_243 = torch.constant.int 1
    %int8_244 = torch.constant.int 8
    %218 = torch.prim.ListConstruct %int1_243, %int8_244, %36, %36 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %219 = torch.aten.view %217, %218 : !torch.vtensor<[8,?,?],f16>, !torch.list<int> -> !torch.vtensor<[1,8,?,?],f16>
    torch.bind_symbolic_shape %219, [%32], affine_map<()[s0] -> (1, 8, s0 * 16, s0 * 16)> : !torch.vtensor<[1,8,?,?],f16>
    %float5.656850e00 = torch.constant.float 5.6568542494923806
    %220 = torch.aten.div.Scalar %219, %float5.656850e00 : !torch.vtensor<[1,8,?,?],f16>, !torch.float -> !torch.vtensor<[1,8,?,?],f16>
    torch.bind_symbolic_shape %220, [%32], affine_map<()[s0] -> (1, 8, s0 * 16, s0 * 16)> : !torch.vtensor<[1,8,?,?],f16>
    %int1_245 = torch.constant.int 1
    %221 = torch.aten.add.Tensor %220, %62, %int1_245 : !torch.vtensor<[1,8,?,?],f16>, !torch.vtensor<[1,1,?,?],f16>, !torch.int -> !torch.vtensor<[1,8,?,?],f16>
    torch.bind_symbolic_shape %221, [%32], affine_map<()[s0] -> (1, 8, s0 * 16, s0 * 16)> : !torch.vtensor<[1,8,?,?],f16>
    %int6_246 = torch.constant.int 6
    %222 = torch.prims.convert_element_type %221, %int6_246 : !torch.vtensor<[1,8,?,?],f16>, !torch.int -> !torch.vtensor<[1,8,?,?],f32>
    torch.bind_symbolic_shape %222, [%32], affine_map<()[s0] -> (1, 8, s0 * 16, s0 * 16)> : !torch.vtensor<[1,8,?,?],f32>
    %int-1_247 = torch.constant.int -1
    %false_248 = torch.constant.bool false
    %223 = torch.aten._softmax %222, %int-1_247, %false_248 : !torch.vtensor<[1,8,?,?],f32>, !torch.int, !torch.bool -> !torch.vtensor<[1,8,?,?],f32>
    torch.bind_symbolic_shape %223, [%32], affine_map<()[s0] -> (1, 8, s0 * 16, s0 * 16)> : !torch.vtensor<[1,8,?,?],f32>
    %int5_249 = torch.constant.int 5
    %224 = torch.prims.convert_element_type %223, %int5_249 : !torch.vtensor<[1,8,?,?],f32>, !torch.int -> !torch.vtensor<[1,8,?,?],f16>
    torch.bind_symbolic_shape %224, [%32], affine_map<()[s0] -> (1, 8, s0 * 16, s0 * 16)> : !torch.vtensor<[1,8,?,?],f16>
    %int5_250 = torch.constant.int 5
    %225 = torch.prims.convert_element_type %206, %int5_250 : !torch.vtensor<[1,8,?,32],f16>, !torch.int -> !torch.vtensor<[1,8,?,32],f16>
    torch.bind_symbolic_shape %225, [%32], affine_map<()[s0] -> (1, 8, s0 * 16, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %int1_251 = torch.constant.int 1
    %int8_252 = torch.constant.int 8
    %226 = torch.prim.ListConstruct %int1_251, %int8_252, %36, %36 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_253 = torch.constant.bool false
    %227 = torch.aten.expand %224, %226, %false_253 : !torch.vtensor<[1,8,?,?],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,8,?,?],f16>
    torch.bind_symbolic_shape %227, [%32], affine_map<()[s0] -> (1, 8, s0 * 16, s0 * 16)> : !torch.vtensor<[1,8,?,?],f16>
    %int8_254 = torch.constant.int 8
    %228 = torch.prim.ListConstruct %int8_254, %36, %36 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %229 = torch.aten.view %227, %228 : !torch.vtensor<[1,8,?,?],f16>, !torch.list<int> -> !torch.vtensor<[8,?,?],f16>
    torch.bind_symbolic_shape %229, [%32], affine_map<()[s0] -> (8, s0 * 16, s0 * 16)> : !torch.vtensor<[8,?,?],f16>
    %int1_255 = torch.constant.int 1
    %int8_256 = torch.constant.int 8
    %int32_257 = torch.constant.int 32
    %230 = torch.prim.ListConstruct %int1_255, %int8_256, %36, %int32_257 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_258 = torch.constant.bool false
    %231 = torch.aten.expand %225, %230, %false_258 : !torch.vtensor<[1,8,?,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,8,?,32],f16>
    torch.bind_symbolic_shape %231, [%32], affine_map<()[s0] -> (1, 8, s0 * 16, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %int8_259 = torch.constant.int 8
    %int32_260 = torch.constant.int 32
    %232 = torch.prim.ListConstruct %int8_259, %36, %int32_260 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %233 = torch.aten.view %231, %232 : !torch.vtensor<[1,8,?,32],f16>, !torch.list<int> -> !torch.vtensor<[8,?,32],f16>
    torch.bind_symbolic_shape %233, [%32], affine_map<()[s0] -> (8, s0 * 16, 32)> : !torch.vtensor<[8,?,32],f16>
    %234 = torch.aten.bmm %229, %233 : !torch.vtensor<[8,?,?],f16>, !torch.vtensor<[8,?,32],f16> -> !torch.vtensor<[8,?,32],f16>
    torch.bind_symbolic_shape %234, [%32], affine_map<()[s0] -> (8, s0 * 16, 32)> : !torch.vtensor<[8,?,32],f16>
    %int1_261 = torch.constant.int 1
    %int8_262 = torch.constant.int 8
    %int32_263 = torch.constant.int 32
    %235 = torch.prim.ListConstruct %int1_261, %int8_262, %36, %int32_263 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %236 = torch.aten.view %234, %235 : !torch.vtensor<[8,?,32],f16>, !torch.list<int> -> !torch.vtensor<[1,8,?,32],f16>
    torch.bind_symbolic_shape %236, [%32], affine_map<()[s0] -> (1, 8, s0 * 16, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %int1_264 = torch.constant.int 1
    %int2_265 = torch.constant.int 2
    %237 = torch.aten.transpose.int %236, %int1_264, %int2_265 : !torch.vtensor<[1,8,?,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,?,8,32],f16>
    torch.bind_symbolic_shape %237, [%32], affine_map<()[s0] -> (1, s0 * 16, 8, 32)> : !torch.vtensor<[1,?,8,32],f16>
    %int0_266 = torch.constant.int 0
    %238 = torch.aten.clone %237, %int0_266 : !torch.vtensor<[1,?,8,32],f16>, !torch.int -> !torch.vtensor<[1,?,8,32],f16>
    torch.bind_symbolic_shape %238, [%32], affine_map<()[s0] -> (1, s0 * 16, 8, 32)> : !torch.vtensor<[1,?,8,32],f16>
    %int1_267 = torch.constant.int 1
    %int256_268 = torch.constant.int 256
    %239 = torch.prim.ListConstruct %int1_267, %36, %int256_268 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %240 = torch.aten._unsafe_view %238, %239 : !torch.vtensor<[1,?,8,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %240, [%32], affine_map<()[s0] -> (1, s0 * 16, 256)> : !torch.vtensor<[1,?,256],f16>
    %int5_269 = torch.constant.int 5
    %241 = torch.prims.convert_element_type %5, %int5_269 : !torch.vtensor<[256,256],f32>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int-2_270 = torch.constant.int -2
    %int-1_271 = torch.constant.int -1
    %242 = torch.aten.transpose.int %241, %int-2_270, %int-1_271 : !torch.vtensor<[256,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,256],f16>
    %int5_272 = torch.constant.int 5
    %243 = torch.prims.convert_element_type %242, %int5_272 : !torch.vtensor<[256,256],f16>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int256_273 = torch.constant.int 256
    %244 = torch.prim.ListConstruct %36, %int256_273 : (!torch.int, !torch.int) -> !torch.list<int>
    %245 = torch.aten.view %240, %244 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %245, [%32], affine_map<()[s0] -> (s0 * 16, 256)> : !torch.vtensor<[?,256],f16>
    %246 = torch.aten.mm %245, %243 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,256],f16> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %246, [%32], affine_map<()[s0] -> (s0 * 16, 256)> : !torch.vtensor<[?,256],f16>
    %int1_274 = torch.constant.int 1
    %int256_275 = torch.constant.int 256
    %247 = torch.prim.ListConstruct %int1_274, %36, %int256_275 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %248 = torch.aten.view %246, %247 : !torch.vtensor<[?,256],f16>, !torch.list<int> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %248, [%32], affine_map<()[s0] -> (1, s0 * 16, 256)> : !torch.vtensor<[1,?,256],f16>
    %int1_276 = torch.constant.int 1
    %249 = torch.aten.add.Tensor %64, %248, %int1_276 : !torch.vtensor<[1,?,256],f16>, !torch.vtensor<[1,?,256],f16>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %249, [%32], affine_map<()[s0] -> (1, s0 * 16, 256)> : !torch.vtensor<[1,?,256],f16>
    %int6_277 = torch.constant.int 6
    %250 = torch.prims.convert_element_type %249, %int6_277 : !torch.vtensor<[1,?,256],f16>, !torch.int -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %250, [%32], affine_map<()[s0] -> (1, s0 * 16, 256)> : !torch.vtensor<[1,?,256],f32>
    %int2_278 = torch.constant.int 2
    %251 = torch.aten.pow.Tensor_Scalar %250, %int2_278 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %251, [%32], affine_map<()[s0] -> (1, s0 * 16, 256)> : !torch.vtensor<[1,?,256],f32>
    %int-1_279 = torch.constant.int -1
    %252 = torch.prim.ListConstruct %int-1_279 : (!torch.int) -> !torch.list<int>
    %true_280 = torch.constant.bool true
    %none_281 = torch.constant.none
    %253 = torch.aten.mean.dim %251, %252, %true_280, %none_281 : !torch.vtensor<[1,?,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %253, [%32], affine_map<()[s0] -> (1, s0 * 16, 1)> : !torch.vtensor<[1,?,1],f32>
    %float1.000000e-02_282 = torch.constant.float 1.000000e-02
    %int1_283 = torch.constant.int 1
    %254 = torch.aten.add.Scalar %253, %float1.000000e-02_282, %int1_283 : !torch.vtensor<[1,?,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %254, [%32], affine_map<()[s0] -> (1, s0 * 16, 1)> : !torch.vtensor<[1,?,1],f32>
    %255 = torch.aten.rsqrt %254 : !torch.vtensor<[1,?,1],f32> -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %255, [%32], affine_map<()[s0] -> (1, s0 * 16, 1)> : !torch.vtensor<[1,?,1],f32>
    %256 = torch.aten.mul.Tensor %250, %255 : !torch.vtensor<[1,?,256],f32>, !torch.vtensor<[1,?,1],f32> -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %256, [%32], affine_map<()[s0] -> (1, s0 * 16, 256)> : !torch.vtensor<[1,?,256],f32>
    %int6_284 = torch.constant.int 6
    %257 = torch.prims.convert_element_type %256, %int6_284 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %257, [%32], affine_map<()[s0] -> (1, s0 * 16, 256)> : !torch.vtensor<[1,?,256],f32>
    %258 = torch.aten.mul.Tensor %6, %257 : !torch.vtensor<[256],f32>, !torch.vtensor<[1,?,256],f32> -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %258, [%32], affine_map<()[s0] -> (1, s0 * 16, 256)> : !torch.vtensor<[1,?,256],f32>
    %int5_285 = torch.constant.int 5
    %259 = torch.prims.convert_element_type %258, %int5_285 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %259, [%32], affine_map<()[s0] -> (1, s0 * 16, 256)> : !torch.vtensor<[1,?,256],f16>
    %int5_286 = torch.constant.int 5
    %260 = torch.prims.convert_element_type %7, %int5_286 : !torch.vtensor<[23,256],f32>, !torch.int -> !torch.vtensor<[23,256],f16>
    %int-2_287 = torch.constant.int -2
    %int-1_288 = torch.constant.int -1
    %261 = torch.aten.transpose.int %260, %int-2_287, %int-1_288 : !torch.vtensor<[23,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,23],f16>
    %int5_289 = torch.constant.int 5
    %262 = torch.prims.convert_element_type %261, %int5_289 : !torch.vtensor<[256,23],f16>, !torch.int -> !torch.vtensor<[256,23],f16>
    %int256_290 = torch.constant.int 256
    %263 = torch.prim.ListConstruct %36, %int256_290 : (!torch.int, !torch.int) -> !torch.list<int>
    %264 = torch.aten.view %259, %263 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %264, [%32], affine_map<()[s0] -> (s0 * 16, 256)> : !torch.vtensor<[?,256],f16>
    %265 = torch.aten.mm %264, %262 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,23],f16> -> !torch.vtensor<[?,23],f16>
    torch.bind_symbolic_shape %265, [%32], affine_map<()[s0] -> (s0 * 16, 23)> : !torch.vtensor<[?,23],f16>
    %int1_291 = torch.constant.int 1
    %int23 = torch.constant.int 23
    %266 = torch.prim.ListConstruct %int1_291, %36, %int23 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %267 = torch.aten.view %265, %266 : !torch.vtensor<[?,23],f16>, !torch.list<int> -> !torch.vtensor<[1,?,23],f16>
    torch.bind_symbolic_shape %267, [%32], affine_map<()[s0] -> (1, s0 * 16, 23)> : !torch.vtensor<[1,?,23],f16>
    %268 = torch.aten.silu %267 : !torch.vtensor<[1,?,23],f16> -> !torch.vtensor<[1,?,23],f16>
    torch.bind_symbolic_shape %268, [%32], affine_map<()[s0] -> (1, s0 * 16, 23)> : !torch.vtensor<[1,?,23],f16>
    %int5_292 = torch.constant.int 5
    %269 = torch.prims.convert_element_type %8, %int5_292 : !torch.vtensor<[23,256],f32>, !torch.int -> !torch.vtensor<[23,256],f16>
    %int-2_293 = torch.constant.int -2
    %int-1_294 = torch.constant.int -1
    %270 = torch.aten.transpose.int %269, %int-2_293, %int-1_294 : !torch.vtensor<[23,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,23],f16>
    %int5_295 = torch.constant.int 5
    %271 = torch.prims.convert_element_type %270, %int5_295 : !torch.vtensor<[256,23],f16>, !torch.int -> !torch.vtensor<[256,23],f16>
    %int256_296 = torch.constant.int 256
    %272 = torch.prim.ListConstruct %36, %int256_296 : (!torch.int, !torch.int) -> !torch.list<int>
    %273 = torch.aten.view %259, %272 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %273, [%32], affine_map<()[s0] -> (s0 * 16, 256)> : !torch.vtensor<[?,256],f16>
    %274 = torch.aten.mm %273, %271 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,23],f16> -> !torch.vtensor<[?,23],f16>
    torch.bind_symbolic_shape %274, [%32], affine_map<()[s0] -> (s0 * 16, 23)> : !torch.vtensor<[?,23],f16>
    %int1_297 = torch.constant.int 1
    %int23_298 = torch.constant.int 23
    %275 = torch.prim.ListConstruct %int1_297, %36, %int23_298 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %276 = torch.aten.view %274, %275 : !torch.vtensor<[?,23],f16>, !torch.list<int> -> !torch.vtensor<[1,?,23],f16>
    torch.bind_symbolic_shape %276, [%32], affine_map<()[s0] -> (1, s0 * 16, 23)> : !torch.vtensor<[1,?,23],f16>
    %277 = torch.aten.mul.Tensor %268, %276 : !torch.vtensor<[1,?,23],f16>, !torch.vtensor<[1,?,23],f16> -> !torch.vtensor<[1,?,23],f16>
    torch.bind_symbolic_shape %277, [%32], affine_map<()[s0] -> (1, s0 * 16, 23)> : !torch.vtensor<[1,?,23],f16>
    %int5_299 = torch.constant.int 5
    %278 = torch.prims.convert_element_type %9, %int5_299 : !torch.vtensor<[256,23],f32>, !torch.int -> !torch.vtensor<[256,23],f16>
    %int-2_300 = torch.constant.int -2
    %int-1_301 = torch.constant.int -1
    %279 = torch.aten.transpose.int %278, %int-2_300, %int-1_301 : !torch.vtensor<[256,23],f16>, !torch.int, !torch.int -> !torch.vtensor<[23,256],f16>
    %int5_302 = torch.constant.int 5
    %280 = torch.prims.convert_element_type %279, %int5_302 : !torch.vtensor<[23,256],f16>, !torch.int -> !torch.vtensor<[23,256],f16>
    %int23_303 = torch.constant.int 23
    %281 = torch.prim.ListConstruct %36, %int23_303 : (!torch.int, !torch.int) -> !torch.list<int>
    %282 = torch.aten.view %277, %281 : !torch.vtensor<[1,?,23],f16>, !torch.list<int> -> !torch.vtensor<[?,23],f16>
    torch.bind_symbolic_shape %282, [%32], affine_map<()[s0] -> (s0 * 16, 23)> : !torch.vtensor<[?,23],f16>
    %283 = torch.aten.mm %282, %280 : !torch.vtensor<[?,23],f16>, !torch.vtensor<[23,256],f16> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %283, [%32], affine_map<()[s0] -> (s0 * 16, 256)> : !torch.vtensor<[?,256],f16>
    %int1_304 = torch.constant.int 1
    %int256_305 = torch.constant.int 256
    %284 = torch.prim.ListConstruct %int1_304, %36, %int256_305 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %285 = torch.aten.view %283, %284 : !torch.vtensor<[?,256],f16>, !torch.list<int> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %285, [%32], affine_map<()[s0] -> (1, s0 * 16, 256)> : !torch.vtensor<[1,?,256],f16>
    %int1_306 = torch.constant.int 1
    %286 = torch.aten.add.Tensor %249, %285, %int1_306 : !torch.vtensor<[1,?,256],f16>, !torch.vtensor<[1,?,256],f16>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %286, [%32], affine_map<()[s0] -> (1, s0 * 16, 256)> : !torch.vtensor<[1,?,256],f16>
    %int6_307 = torch.constant.int 6
    %287 = torch.prims.convert_element_type %286, %int6_307 : !torch.vtensor<[1,?,256],f16>, !torch.int -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %287, [%32], affine_map<()[s0] -> (1, s0 * 16, 256)> : !torch.vtensor<[1,?,256],f32>
    %int2_308 = torch.constant.int 2
    %288 = torch.aten.pow.Tensor_Scalar %287, %int2_308 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %288, [%32], affine_map<()[s0] -> (1, s0 * 16, 256)> : !torch.vtensor<[1,?,256],f32>
    %int-1_309 = torch.constant.int -1
    %289 = torch.prim.ListConstruct %int-1_309 : (!torch.int) -> !torch.list<int>
    %true_310 = torch.constant.bool true
    %none_311 = torch.constant.none
    %290 = torch.aten.mean.dim %288, %289, %true_310, %none_311 : !torch.vtensor<[1,?,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %290, [%32], affine_map<()[s0] -> (1, s0 * 16, 1)> : !torch.vtensor<[1,?,1],f32>
    %float1.000000e-02_312 = torch.constant.float 1.000000e-02
    %int1_313 = torch.constant.int 1
    %291 = torch.aten.add.Scalar %290, %float1.000000e-02_312, %int1_313 : !torch.vtensor<[1,?,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %291, [%32], affine_map<()[s0] -> (1, s0 * 16, 1)> : !torch.vtensor<[1,?,1],f32>
    %292 = torch.aten.rsqrt %291 : !torch.vtensor<[1,?,1],f32> -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %292, [%32], affine_map<()[s0] -> (1, s0 * 16, 1)> : !torch.vtensor<[1,?,1],f32>
    %293 = torch.aten.mul.Tensor %287, %292 : !torch.vtensor<[1,?,256],f32>, !torch.vtensor<[1,?,1],f32> -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %293, [%32], affine_map<()[s0] -> (1, s0 * 16, 256)> : !torch.vtensor<[1,?,256],f32>
    %int6_314 = torch.constant.int 6
    %294 = torch.prims.convert_element_type %293, %int6_314 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %294, [%32], affine_map<()[s0] -> (1, s0 * 16, 256)> : !torch.vtensor<[1,?,256],f32>
    %295 = torch.aten.mul.Tensor %10, %294 : !torch.vtensor<[256],f32>, !torch.vtensor<[1,?,256],f32> -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %295, [%32], affine_map<()[s0] -> (1, s0 * 16, 256)> : !torch.vtensor<[1,?,256],f32>
    %int5_315 = torch.constant.int 5
    %296 = torch.prims.convert_element_type %295, %int5_315 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %296, [%32], affine_map<()[s0] -> (1, s0 * 16, 256)> : !torch.vtensor<[1,?,256],f16>
    %int5_316 = torch.constant.int 5
    %297 = torch.prims.convert_element_type %11, %int5_316 : !torch.vtensor<[256,256],f32>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int-2_317 = torch.constant.int -2
    %int-1_318 = torch.constant.int -1
    %298 = torch.aten.transpose.int %297, %int-2_317, %int-1_318 : !torch.vtensor<[256,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,256],f16>
    %int5_319 = torch.constant.int 5
    %299 = torch.prims.convert_element_type %298, %int5_319 : !torch.vtensor<[256,256],f16>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int256_320 = torch.constant.int 256
    %300 = torch.prim.ListConstruct %36, %int256_320 : (!torch.int, !torch.int) -> !torch.list<int>
    %301 = torch.aten.view %296, %300 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %301, [%32], affine_map<()[s0] -> (s0 * 16, 256)> : !torch.vtensor<[?,256],f16>
    %302 = torch.aten.mm %301, %299 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,256],f16> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %302, [%32], affine_map<()[s0] -> (s0 * 16, 256)> : !torch.vtensor<[?,256],f16>
    %int1_321 = torch.constant.int 1
    %int256_322 = torch.constant.int 256
    %303 = torch.prim.ListConstruct %int1_321, %36, %int256_322 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %304 = torch.aten.view %302, %303 : !torch.vtensor<[?,256],f16>, !torch.list<int> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %304, [%32], affine_map<()[s0] -> (1, s0 * 16, 256)> : !torch.vtensor<[1,?,256],f16>
    %int5_323 = torch.constant.int 5
    %305 = torch.prims.convert_element_type %12, %int5_323 : !torch.vtensor<[128,256],f32>, !torch.int -> !torch.vtensor<[128,256],f16>
    %int-2_324 = torch.constant.int -2
    %int-1_325 = torch.constant.int -1
    %306 = torch.aten.transpose.int %305, %int-2_324, %int-1_325 : !torch.vtensor<[128,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,128],f16>
    %int5_326 = torch.constant.int 5
    %307 = torch.prims.convert_element_type %306, %int5_326 : !torch.vtensor<[256,128],f16>, !torch.int -> !torch.vtensor<[256,128],f16>
    %int256_327 = torch.constant.int 256
    %308 = torch.prim.ListConstruct %36, %int256_327 : (!torch.int, !torch.int) -> !torch.list<int>
    %309 = torch.aten.view %296, %308 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %309, [%32], affine_map<()[s0] -> (s0 * 16, 256)> : !torch.vtensor<[?,256],f16>
    %310 = torch.aten.mm %309, %307 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,128],f16> -> !torch.vtensor<[?,128],f16>
    torch.bind_symbolic_shape %310, [%32], affine_map<()[s0] -> (s0 * 16, 128)> : !torch.vtensor<[?,128],f16>
    %int1_328 = torch.constant.int 1
    %int128_329 = torch.constant.int 128
    %311 = torch.prim.ListConstruct %int1_328, %36, %int128_329 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %312 = torch.aten.view %310, %311 : !torch.vtensor<[?,128],f16>, !torch.list<int> -> !torch.vtensor<[1,?,128],f16>
    torch.bind_symbolic_shape %312, [%32], affine_map<()[s0] -> (1, s0 * 16, 128)> : !torch.vtensor<[1,?,128],f16>
    %int5_330 = torch.constant.int 5
    %313 = torch.prims.convert_element_type %13, %int5_330 : !torch.vtensor<[128,256],f32>, !torch.int -> !torch.vtensor<[128,256],f16>
    %int-2_331 = torch.constant.int -2
    %int-1_332 = torch.constant.int -1
    %314 = torch.aten.transpose.int %313, %int-2_331, %int-1_332 : !torch.vtensor<[128,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,128],f16>
    %int5_333 = torch.constant.int 5
    %315 = torch.prims.convert_element_type %314, %int5_333 : !torch.vtensor<[256,128],f16>, !torch.int -> !torch.vtensor<[256,128],f16>
    %int256_334 = torch.constant.int 256
    %316 = torch.prim.ListConstruct %36, %int256_334 : (!torch.int, !torch.int) -> !torch.list<int>
    %317 = torch.aten.view %296, %316 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %317, [%32], affine_map<()[s0] -> (s0 * 16, 256)> : !torch.vtensor<[?,256],f16>
    %318 = torch.aten.mm %317, %315 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,128],f16> -> !torch.vtensor<[?,128],f16>
    torch.bind_symbolic_shape %318, [%32], affine_map<()[s0] -> (s0 * 16, 128)> : !torch.vtensor<[?,128],f16>
    %int1_335 = torch.constant.int 1
    %int128_336 = torch.constant.int 128
    %319 = torch.prim.ListConstruct %int1_335, %36, %int128_336 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %320 = torch.aten.view %318, %319 : !torch.vtensor<[?,128],f16>, !torch.list<int> -> !torch.vtensor<[1,?,128],f16>
    torch.bind_symbolic_shape %320, [%32], affine_map<()[s0] -> (1, s0 * 16, 128)> : !torch.vtensor<[1,?,128],f16>
    %int1_337 = torch.constant.int 1
    %int8_338 = torch.constant.int 8
    %int32_339 = torch.constant.int 32
    %321 = torch.prim.ListConstruct %int1_337, %36, %int8_338, %int32_339 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %322 = torch.aten.view %304, %321 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[1,?,8,32],f16>
    torch.bind_symbolic_shape %322, [%32], affine_map<()[s0] -> (1, s0 * 16, 8, 32)> : !torch.vtensor<[1,?,8,32],f16>
    %int1_340 = torch.constant.int 1
    %int4_341 = torch.constant.int 4
    %int32_342 = torch.constant.int 32
    %323 = torch.prim.ListConstruct %int1_340, %36, %int4_341, %int32_342 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %324 = torch.aten.view %312, %323 : !torch.vtensor<[1,?,128],f16>, !torch.list<int> -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %324, [%32], affine_map<()[s0] -> (1, s0 * 16, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int1_343 = torch.constant.int 1
    %int4_344 = torch.constant.int 4
    %int32_345 = torch.constant.int 32
    %325 = torch.prim.ListConstruct %int1_343, %36, %int4_344, %int32_345 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %326 = torch.aten.view %320, %325 : !torch.vtensor<[1,?,128],f16>, !torch.list<int> -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %326, [%32], affine_map<()[s0] -> (1, s0 * 16, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int128_346 = torch.constant.int 128
    %none_347 = torch.constant.none
    %none_348 = torch.constant.none
    %cpu_349 = torch.constant.device "cpu"
    %false_350 = torch.constant.bool false
    %327 = torch.aten.arange %int128_346, %none_347, %none_348, %cpu_349, %false_350 : !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[128],si64>
    %int0_351 = torch.constant.int 0
    %int32_352 = torch.constant.int 32
    %int2_353 = torch.constant.int 2
    %none_354 = torch.constant.none
    %none_355 = torch.constant.none
    %cpu_356 = torch.constant.device "cpu"
    %false_357 = torch.constant.bool false
    %328 = torch.aten.arange.start_step %int0_351, %int32_352, %int2_353, %none_354, %none_355, %cpu_356, %false_357 : !torch.int, !torch.int, !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[16],si64>
    %int0_358 = torch.constant.int 0
    %int0_359 = torch.constant.int 0
    %int16_360 = torch.constant.int 16
    %int1_361 = torch.constant.int 1
    %329 = torch.aten.slice.Tensor %328, %int0_358, %int0_359, %int16_360, %int1_361 : !torch.vtensor<[16],si64>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[16],si64>
    %int6_362 = torch.constant.int 6
    %330 = torch.prims.convert_element_type %329, %int6_362 : !torch.vtensor<[16],si64>, !torch.int -> !torch.vtensor<[16],f32>
    %int32_363 = torch.constant.int 32
    %331 = torch.aten.div.Scalar %330, %int32_363 : !torch.vtensor<[16],f32>, !torch.int -> !torch.vtensor<[16],f32>
    %float5.000000e05_364 = torch.constant.float 5.000000e+05
    %332 = torch.aten.pow.Scalar %float5.000000e05_364, %331 : !torch.float, !torch.vtensor<[16],f32> -> !torch.vtensor<[16],f32>
    %333 = torch.aten.reciprocal %332 : !torch.vtensor<[16],f32> -> !torch.vtensor<[16],f32>
    %float1.000000e00_365 = torch.constant.float 1.000000e+00
    %334 = torch.aten.mul.Scalar %333, %float1.000000e00_365 : !torch.vtensor<[16],f32>, !torch.float -> !torch.vtensor<[16],f32>
    %int128_366 = torch.constant.int 128
    %int1_367 = torch.constant.int 1
    %335 = torch.prim.ListConstruct %int128_366, %int1_367 : (!torch.int, !torch.int) -> !torch.list<int>
    %336 = torch.aten.view %327, %335 : !torch.vtensor<[128],si64>, !torch.list<int> -> !torch.vtensor<[128,1],si64>
    %337 = torch.aten.mul.Tensor %336, %334 : !torch.vtensor<[128,1],si64>, !torch.vtensor<[16],f32> -> !torch.vtensor<[128,16],f32>
    %int6_368 = torch.constant.int 6
    %338 = torch.prims.convert_element_type %337, %int6_368 : !torch.vtensor<[128,16],f32>, !torch.int -> !torch.vtensor<[128,16],f32>
    %339 = torch.aten.cos %338 : !torch.vtensor<[128,16],f32> -> !torch.vtensor<[128,16],f32>
    %340 = torch.aten.sin %338 : !torch.vtensor<[128,16],f32> -> !torch.vtensor<[128,16],f32>
    %341 = torch.aten.complex %339, %340 : !torch.vtensor<[128,16],f32>, !torch.vtensor<[128,16],f32> -> !torch.vtensor<[128,16],complex<f32>>
    %int0_369 = torch.constant.int 0
    %int0_370 = torch.constant.int 0
    %int1_371 = torch.constant.int 1
    %342 = torch.aten.slice.Tensor %341, %int0_369, %int0_370, %36, %int1_371 : !torch.vtensor<[128,16],complex<f32>>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,16],complex<f32>>
    torch.bind_symbolic_shape %342, [%32], affine_map<()[s0] -> (s0 * 16, 16)> : !torch.vtensor<[?,16],complex<f32>>
    %int1_372 = torch.constant.int 1
    %int0_373 = torch.constant.int 0
    %int9223372036854775807_374 = torch.constant.int 9223372036854775807
    %int1_375 = torch.constant.int 1
    %343 = torch.aten.slice.Tensor %342, %int1_372, %int0_373, %int9223372036854775807_374, %int1_375 : !torch.vtensor<[?,16],complex<f32>>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,16],complex<f32>>
    torch.bind_symbolic_shape %343, [%32], affine_map<()[s0] -> (s0 * 16, 16)> : !torch.vtensor<[?,16],complex<f32>>
    %int0_376 = torch.constant.int 0
    %344 = torch.aten.unsqueeze %343, %int0_376 : !torch.vtensor<[?,16],complex<f32>>, !torch.int -> !torch.vtensor<[1,?,16],complex<f32>>
    torch.bind_symbolic_shape %344, [%32], affine_map<()[s0] -> (1, s0 * 16, 16)> : !torch.vtensor<[1,?,16],complex<f32>>
    %int2_377 = torch.constant.int 2
    %345 = torch.aten.unsqueeze %344, %int2_377 : !torch.vtensor<[1,?,16],complex<f32>>, !torch.int -> !torch.vtensor<[1,?,1,16],complex<f32>>
    torch.bind_symbolic_shape %345, [%32], affine_map<()[s0] -> (1, s0 * 16, 1, 16)> : !torch.vtensor<[1,?,1,16],complex<f32>>
    %int3_378 = torch.constant.int 3
    %int0_379 = torch.constant.int 0
    %int9223372036854775807_380 = torch.constant.int 9223372036854775807
    %int1_381 = torch.constant.int 1
    %346 = torch.aten.slice.Tensor %345, %int3_378, %int0_379, %int9223372036854775807_380, %int1_381 : !torch.vtensor<[1,?,1,16],complex<f32>>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,1,16],complex<f32>>
    torch.bind_symbolic_shape %346, [%32], affine_map<()[s0] -> (1, s0 * 16, 1, 16)> : !torch.vtensor<[1,?,1,16],complex<f32>>
    %347 = torch_c.to_builtin_tensor %322 : !torch.vtensor<[1,?,8,32],f16> -> tensor<1x?x8x32xf16>
    %c1_382 = arith.constant 1 : index
    %dim_383 = tensor.dim %347, %c1_382 : tensor<1x?x8x32xf16>
    %348 = flow.tensor.bitcast %347 : tensor<1x?x8x32xf16>{%dim_383} -> tensor<1x?x8x16xcomplex<f16>>{%dim_383}
    %349 = torch_c.from_builtin_tensor %348 : tensor<1x?x8x16xcomplex<f16>> -> !torch.vtensor<[1,?,8,16],complex<f16>>
    torch.bind_symbolic_shape %349, [%32], affine_map<()[s0] -> (1, s0 * 16, 8, 16)> : !torch.vtensor<[1,?,8,16],complex<f16>>
    %350 = torch.aten.mul.Tensor %349, %346 : !torch.vtensor<[1,?,8,16],complex<f16>>, !torch.vtensor<[1,?,1,16],complex<f32>> -> !torch.vtensor<[1,?,8,16],complex<f32>>
    torch.bind_symbolic_shape %350, [%32], affine_map<()[s0] -> (1, s0 * 16, 8, 16)> : !torch.vtensor<[1,?,8,16],complex<f32>>
    %351 = torch_c.to_builtin_tensor %350 : !torch.vtensor<[1,?,8,16],complex<f32>> -> tensor<1x?x8x16xcomplex<f32>>
    %c1_384 = arith.constant 1 : index
    %dim_385 = tensor.dim %351, %c1_384 : tensor<1x?x8x16xcomplex<f32>>
    %352 = flow.tensor.bitcast %351 : tensor<1x?x8x16xcomplex<f32>>{%dim_385} -> tensor<1x?x8x32xf32>{%dim_385}
    %353 = torch_c.from_builtin_tensor %352 : tensor<1x?x8x32xf32> -> !torch.vtensor<[1,?,8,32],f32>
    torch.bind_symbolic_shape %353, [%32], affine_map<()[s0] -> (1, s0 * 16, 8, 32)> : !torch.vtensor<[1,?,8,32],f32>
    %int5_386 = torch.constant.int 5
    %354 = torch.prims.convert_element_type %353, %int5_386 : !torch.vtensor<[1,?,8,32],f32>, !torch.int -> !torch.vtensor<[1,?,8,32],f16>
    torch.bind_symbolic_shape %354, [%32], affine_map<()[s0] -> (1, s0 * 16, 8, 32)> : !torch.vtensor<[1,?,8,32],f16>
    %int128_387 = torch.constant.int 128
    %none_388 = torch.constant.none
    %none_389 = torch.constant.none
    %cpu_390 = torch.constant.device "cpu"
    %false_391 = torch.constant.bool false
    %355 = torch.aten.arange %int128_387, %none_388, %none_389, %cpu_390, %false_391 : !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[128],si64>
    %int0_392 = torch.constant.int 0
    %int32_393 = torch.constant.int 32
    %int2_394 = torch.constant.int 2
    %none_395 = torch.constant.none
    %none_396 = torch.constant.none
    %cpu_397 = torch.constant.device "cpu"
    %false_398 = torch.constant.bool false
    %356 = torch.aten.arange.start_step %int0_392, %int32_393, %int2_394, %none_395, %none_396, %cpu_397, %false_398 : !torch.int, !torch.int, !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[16],si64>
    %int0_399 = torch.constant.int 0
    %int0_400 = torch.constant.int 0
    %int16_401 = torch.constant.int 16
    %int1_402 = torch.constant.int 1
    %357 = torch.aten.slice.Tensor %356, %int0_399, %int0_400, %int16_401, %int1_402 : !torch.vtensor<[16],si64>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[16],si64>
    %int6_403 = torch.constant.int 6
    %358 = torch.prims.convert_element_type %357, %int6_403 : !torch.vtensor<[16],si64>, !torch.int -> !torch.vtensor<[16],f32>
    %int32_404 = torch.constant.int 32
    %359 = torch.aten.div.Scalar %358, %int32_404 : !torch.vtensor<[16],f32>, !torch.int -> !torch.vtensor<[16],f32>
    %float5.000000e05_405 = torch.constant.float 5.000000e+05
    %360 = torch.aten.pow.Scalar %float5.000000e05_405, %359 : !torch.float, !torch.vtensor<[16],f32> -> !torch.vtensor<[16],f32>
    %361 = torch.aten.reciprocal %360 : !torch.vtensor<[16],f32> -> !torch.vtensor<[16],f32>
    %float1.000000e00_406 = torch.constant.float 1.000000e+00
    %362 = torch.aten.mul.Scalar %361, %float1.000000e00_406 : !torch.vtensor<[16],f32>, !torch.float -> !torch.vtensor<[16],f32>
    %int128_407 = torch.constant.int 128
    %int1_408 = torch.constant.int 1
    %363 = torch.prim.ListConstruct %int128_407, %int1_408 : (!torch.int, !torch.int) -> !torch.list<int>
    %364 = torch.aten.view %355, %363 : !torch.vtensor<[128],si64>, !torch.list<int> -> !torch.vtensor<[128,1],si64>
    %365 = torch.aten.mul.Tensor %364, %362 : !torch.vtensor<[128,1],si64>, !torch.vtensor<[16],f32> -> !torch.vtensor<[128,16],f32>
    %int6_409 = torch.constant.int 6
    %366 = torch.prims.convert_element_type %365, %int6_409 : !torch.vtensor<[128,16],f32>, !torch.int -> !torch.vtensor<[128,16],f32>
    %367 = torch.aten.cos %366 : !torch.vtensor<[128,16],f32> -> !torch.vtensor<[128,16],f32>
    %368 = torch.aten.sin %366 : !torch.vtensor<[128,16],f32> -> !torch.vtensor<[128,16],f32>
    %369 = torch.aten.complex %367, %368 : !torch.vtensor<[128,16],f32>, !torch.vtensor<[128,16],f32> -> !torch.vtensor<[128,16],complex<f32>>
    %int0_410 = torch.constant.int 0
    %int0_411 = torch.constant.int 0
    %int1_412 = torch.constant.int 1
    %370 = torch.aten.slice.Tensor %369, %int0_410, %int0_411, %36, %int1_412 : !torch.vtensor<[128,16],complex<f32>>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,16],complex<f32>>
    torch.bind_symbolic_shape %370, [%32], affine_map<()[s0] -> (s0 * 16, 16)> : !torch.vtensor<[?,16],complex<f32>>
    %int1_413 = torch.constant.int 1
    %int0_414 = torch.constant.int 0
    %int9223372036854775807_415 = torch.constant.int 9223372036854775807
    %int1_416 = torch.constant.int 1
    %371 = torch.aten.slice.Tensor %370, %int1_413, %int0_414, %int9223372036854775807_415, %int1_416 : !torch.vtensor<[?,16],complex<f32>>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,16],complex<f32>>
    torch.bind_symbolic_shape %371, [%32], affine_map<()[s0] -> (s0 * 16, 16)> : !torch.vtensor<[?,16],complex<f32>>
    %int0_417 = torch.constant.int 0
    %372 = torch.aten.unsqueeze %371, %int0_417 : !torch.vtensor<[?,16],complex<f32>>, !torch.int -> !torch.vtensor<[1,?,16],complex<f32>>
    torch.bind_symbolic_shape %372, [%32], affine_map<()[s0] -> (1, s0 * 16, 16)> : !torch.vtensor<[1,?,16],complex<f32>>
    %int2_418 = torch.constant.int 2
    %373 = torch.aten.unsqueeze %372, %int2_418 : !torch.vtensor<[1,?,16],complex<f32>>, !torch.int -> !torch.vtensor<[1,?,1,16],complex<f32>>
    torch.bind_symbolic_shape %373, [%32], affine_map<()[s0] -> (1, s0 * 16, 1, 16)> : !torch.vtensor<[1,?,1,16],complex<f32>>
    %int3_419 = torch.constant.int 3
    %int0_420 = torch.constant.int 0
    %int9223372036854775807_421 = torch.constant.int 9223372036854775807
    %int1_422 = torch.constant.int 1
    %374 = torch.aten.slice.Tensor %373, %int3_419, %int0_420, %int9223372036854775807_421, %int1_422 : !torch.vtensor<[1,?,1,16],complex<f32>>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,1,16],complex<f32>>
    torch.bind_symbolic_shape %374, [%32], affine_map<()[s0] -> (1, s0 * 16, 1, 16)> : !torch.vtensor<[1,?,1,16],complex<f32>>
    %375 = torch_c.to_builtin_tensor %324 : !torch.vtensor<[1,?,4,32],f16> -> tensor<1x?x4x32xf16>
    %c1_423 = arith.constant 1 : index
    %dim_424 = tensor.dim %375, %c1_423 : tensor<1x?x4x32xf16>
    %376 = flow.tensor.bitcast %375 : tensor<1x?x4x32xf16>{%dim_424} -> tensor<1x?x4x16xcomplex<f16>>{%dim_424}
    %377 = torch_c.from_builtin_tensor %376 : tensor<1x?x4x16xcomplex<f16>> -> !torch.vtensor<[1,?,4,16],complex<f16>>
    torch.bind_symbolic_shape %377, [%32], affine_map<()[s0] -> (1, s0 * 16, 4, 16)> : !torch.vtensor<[1,?,4,16],complex<f16>>
    %378 = torch.aten.mul.Tensor %377, %374 : !torch.vtensor<[1,?,4,16],complex<f16>>, !torch.vtensor<[1,?,1,16],complex<f32>> -> !torch.vtensor<[1,?,4,16],complex<f32>>
    torch.bind_symbolic_shape %378, [%32], affine_map<()[s0] -> (1, s0 * 16, 4, 16)> : !torch.vtensor<[1,?,4,16],complex<f32>>
    %379 = torch_c.to_builtin_tensor %378 : !torch.vtensor<[1,?,4,16],complex<f32>> -> tensor<1x?x4x16xcomplex<f32>>
    %c1_425 = arith.constant 1 : index
    %dim_426 = tensor.dim %379, %c1_425 : tensor<1x?x4x16xcomplex<f32>>
    %380 = flow.tensor.bitcast %379 : tensor<1x?x4x16xcomplex<f32>>{%dim_426} -> tensor<1x?x4x32xf32>{%dim_426}
    %381 = torch_c.from_builtin_tensor %380 : tensor<1x?x4x32xf32> -> !torch.vtensor<[1,?,4,32],f32>
    torch.bind_symbolic_shape %381, [%32], affine_map<()[s0] -> (1, s0 * 16, 4, 32)> : !torch.vtensor<[1,?,4,32],f32>
    %int5_427 = torch.constant.int 5
    %382 = torch.prims.convert_element_type %381, %int5_427 : !torch.vtensor<[1,?,4,32],f32>, !torch.int -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %382, [%32], affine_map<()[s0] -> (1, s0 * 16, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int6_428 = torch.constant.int 6
    %383 = torch.aten.mul.Scalar %arg2, %int6_428 : !torch.vtensor<[1,?],si64>, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %383, [%32], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %int2_429 = torch.constant.int 2
    %int1_430 = torch.constant.int 1
    %384 = torch.aten.add.Scalar %383, %int2_429, %int1_430 : !torch.vtensor<[1,?],si64>, !torch.int, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %384, [%32], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %int1_431 = torch.constant.int 1
    %int16_432 = torch.constant.int 16
    %int4_433 = torch.constant.int 4
    %int32_434 = torch.constant.int 32
    %385 = torch.prim.ListConstruct %int1_431, %34, %int16_432, %int4_433, %int32_434 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %386 = torch.aten.view %382, %385 : !torch.vtensor<[1,?,4,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,16,4,32],f16>
    torch.bind_symbolic_shape %386, [%32], affine_map<()[s0] -> (1, s0, 16, 4, 32)> : !torch.vtensor<[1,?,16,4,32],f16>
    %int16_435 = torch.constant.int 16
    %int4_436 = torch.constant.int 4
    %int32_437 = torch.constant.int 32
    %387 = torch.prim.ListConstruct %34, %int16_435, %int4_436, %int32_437 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %388 = torch.aten.view %386, %387 : !torch.vtensor<[1,?,16,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,16,4,32],f16>
    torch.bind_symbolic_shape %388, [%32], affine_map<()[s0] -> (s0, 16, 4, 32)> : !torch.vtensor<[?,16,4,32],f16>
    %389 = torch.prim.ListConstruct %34 : (!torch.int) -> !torch.list<int>
    %390 = torch.aten.view %384, %389 : !torch.vtensor<[1,?],si64>, !torch.list<int> -> !torch.vtensor<[?],si64>
    torch.bind_symbolic_shape %390, [%32], affine_map<()[s0] -> (s0)> : !torch.vtensor<[?],si64>
    %int1_438 = torch.constant.int 1
    %int16_439 = torch.constant.int 16
    %int4_440 = torch.constant.int 4
    %int32_441 = torch.constant.int 32
    %391 = torch.prim.ListConstruct %int1_438, %34, %int16_439, %int4_440, %int32_441 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %392 = torch.aten.view %326, %391 : !torch.vtensor<[1,?,4,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,16,4,32],f16>
    torch.bind_symbolic_shape %392, [%32], affine_map<()[s0] -> (1, s0, 16, 4, 32)> : !torch.vtensor<[1,?,16,4,32],f16>
    %int16_442 = torch.constant.int 16
    %int4_443 = torch.constant.int 4
    %int32_444 = torch.constant.int 32
    %393 = torch.prim.ListConstruct %34, %int16_442, %int4_443, %int32_444 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %394 = torch.aten.view %392, %393 : !torch.vtensor<[1,?,16,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,16,4,32],f16>
    torch.bind_symbolic_shape %394, [%32], affine_map<()[s0] -> (s0, 16, 4, 32)> : !torch.vtensor<[?,16,4,32],f16>
    %int1_445 = torch.constant.int 1
    %int1_446 = torch.constant.int 1
    %395 = torch.aten.add.Scalar %384, %int1_445, %int1_446 : !torch.vtensor<[1,?],si64>, !torch.int, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %395, [%32], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %396 = torch.prim.ListConstruct %34 : (!torch.int) -> !torch.list<int>
    %397 = torch.aten.view %395, %396 : !torch.vtensor<[1,?],si64>, !torch.list<int> -> !torch.vtensor<[?],si64>
    torch.bind_symbolic_shape %397, [%32], affine_map<()[s0] -> (s0)> : !torch.vtensor<[?],si64>
    %398 = torch.prim.ListConstruct %390, %397 : (!torch.vtensor<[?],si64>, !torch.vtensor<[?],si64>) -> !torch.list<vtensor>
    %int0_447 = torch.constant.int 0
    %399 = torch.aten.cat %398, %int0_447 : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[?],si64>
    torch.bind_symbolic_shape %399, [%32], affine_map<()[s0] -> (s0 * 2)> : !torch.vtensor<[?],si64>
    %400 = torch.prim.ListConstruct %388, %394 : (!torch.vtensor<[?,16,4,32],f16>, !torch.vtensor<[?,16,4,32],f16>) -> !torch.list<vtensor>
    %int0_448 = torch.constant.int 0
    %401 = torch.aten.cat %400, %int0_448 : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[?,16,4,32],f16>
    torch.bind_symbolic_shape %401, [%32], affine_map<()[s0] -> (s0 * 2, 16, 4, 32)> : !torch.vtensor<[?,16,4,32],f16>
    %int3_449 = torch.constant.int 3
    %int2_450 = torch.constant.int 2
    %int16_451 = torch.constant.int 16
    %int4_452 = torch.constant.int 4
    %int32_453 = torch.constant.int 32
    %402 = torch.prim.ListConstruct %35, %int3_449, %int2_450, %int16_451, %int4_452, %int32_453 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %403 = torch.aten.view %191, %402 : !torch.vtensor<[?,12288],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,16,4,32],f16>
    torch.bind_symbolic_shape %403, [%33], affine_map<()[s0] -> (s0, 3, 2, 16, 4, 32)> : !torch.vtensor<[?,3,2,16,4,32],f16>
    %int16_454 = torch.constant.int 16
    %int4_455 = torch.constant.int 4
    %int32_456 = torch.constant.int 32
    %404 = torch.prim.ListConstruct %164, %int16_454, %int4_455, %int32_456 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %405 = torch.aten.view %403, %404 : !torch.vtensor<[?,3,2,16,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,16,4,32],f16>
    torch.bind_symbolic_shape %405, [%33], affine_map<()[s0] -> (s0 * 6, 16, 4, 32)> : !torch.vtensor<[?,16,4,32],f16>
    %406 = torch.prim.ListConstruct %399 : (!torch.vtensor<[?],si64>) -> !torch.list<optional<vtensor>>
    %false_457 = torch.constant.bool false
    %407 = torch.aten.index_put %405, %406, %401, %false_457 : !torch.vtensor<[?,16,4,32],f16>, !torch.list<optional<vtensor>>, !torch.vtensor<[?,16,4,32],f16>, !torch.bool -> !torch.vtensor<[?,16,4,32],f16>
    torch.bind_symbolic_shape %407, [%33], affine_map<()[s0] -> (s0 * 6, 16, 4, 32)> : !torch.vtensor<[?,16,4,32],f16>
    %int3_458 = torch.constant.int 3
    %int2_459 = torch.constant.int 2
    %int16_460 = torch.constant.int 16
    %int4_461 = torch.constant.int 4
    %int32_462 = torch.constant.int 32
    %408 = torch.prim.ListConstruct %35, %int3_458, %int2_459, %int16_460, %int4_461, %int32_462 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %409 = torch.aten.view %407, %408 : !torch.vtensor<[?,16,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,16,4,32],f16>
    torch.bind_symbolic_shape %409, [%33], affine_map<()[s0] -> (s0, 3, 2, 16, 4, 32)> : !torch.vtensor<[?,3,2,16,4,32],f16>
    %int12288_463 = torch.constant.int 12288
    %410 = torch.prim.ListConstruct %35, %int12288_463 : (!torch.int, !torch.int) -> !torch.list<int>
    %411 = torch.aten.view %409, %410 : !torch.vtensor<[?,3,2,16,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,12288],f16>
    torch.bind_symbolic_shape %411, [%33], affine_map<()[s0] -> (s0, 12288)> : !torch.vtensor<[?,12288],f16>
    %int-2_464 = torch.constant.int -2
    %412 = torch.aten.unsqueeze %382, %int-2_464 : !torch.vtensor<[1,?,4,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,1,32],f16>
    torch.bind_symbolic_shape %412, [%32], affine_map<()[s0] -> (1, s0 * 16, 4, 1, 32)> : !torch.vtensor<[1,?,4,1,32],f16>
    %int1_465 = torch.constant.int 1
    %int4_466 = torch.constant.int 4
    %int2_467 = torch.constant.int 2
    %int32_468 = torch.constant.int 32
    %413 = torch.prim.ListConstruct %int1_465, %36, %int4_466, %int2_467, %int32_468 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_469 = torch.constant.bool false
    %414 = torch.aten.expand %412, %413, %false_469 : !torch.vtensor<[1,?,4,1,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,?,4,2,32],f16>
    torch.bind_symbolic_shape %414, [%32], affine_map<()[s0] -> (1, s0 * 16, 4, 2, 32)> : !torch.vtensor<[1,?,4,2,32],f16>
    %int0_470 = torch.constant.int 0
    %415 = torch.aten.clone %414, %int0_470 : !torch.vtensor<[1,?,4,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,2,32],f16>
    torch.bind_symbolic_shape %415, [%32], affine_map<()[s0] -> (1, s0 * 16, 4, 2, 32)> : !torch.vtensor<[1,?,4,2,32],f16>
    %int1_471 = torch.constant.int 1
    %int8_472 = torch.constant.int 8
    %int32_473 = torch.constant.int 32
    %416 = torch.prim.ListConstruct %int1_471, %36, %int8_472, %int32_473 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %417 = torch.aten._unsafe_view %415, %416 : !torch.vtensor<[1,?,4,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,8,32],f16>
    torch.bind_symbolic_shape %417, [%32], affine_map<()[s0] -> (1, s0 * 16, 8, 32)> : !torch.vtensor<[1,?,8,32],f16>
    %int-2_474 = torch.constant.int -2
    %418 = torch.aten.unsqueeze %326, %int-2_474 : !torch.vtensor<[1,?,4,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,1,32],f16>
    torch.bind_symbolic_shape %418, [%32], affine_map<()[s0] -> (1, s0 * 16, 4, 1, 32)> : !torch.vtensor<[1,?,4,1,32],f16>
    %int1_475 = torch.constant.int 1
    %int4_476 = torch.constant.int 4
    %int2_477 = torch.constant.int 2
    %int32_478 = torch.constant.int 32
    %419 = torch.prim.ListConstruct %int1_475, %36, %int4_476, %int2_477, %int32_478 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_479 = torch.constant.bool false
    %420 = torch.aten.expand %418, %419, %false_479 : !torch.vtensor<[1,?,4,1,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,?,4,2,32],f16>
    torch.bind_symbolic_shape %420, [%32], affine_map<()[s0] -> (1, s0 * 16, 4, 2, 32)> : !torch.vtensor<[1,?,4,2,32],f16>
    %int0_480 = torch.constant.int 0
    %421 = torch.aten.clone %420, %int0_480 : !torch.vtensor<[1,?,4,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,2,32],f16>
    torch.bind_symbolic_shape %421, [%32], affine_map<()[s0] -> (1, s0 * 16, 4, 2, 32)> : !torch.vtensor<[1,?,4,2,32],f16>
    %int1_481 = torch.constant.int 1
    %int8_482 = torch.constant.int 8
    %int32_483 = torch.constant.int 32
    %422 = torch.prim.ListConstruct %int1_481, %36, %int8_482, %int32_483 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %423 = torch.aten._unsafe_view %421, %422 : !torch.vtensor<[1,?,4,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,8,32],f16>
    torch.bind_symbolic_shape %423, [%32], affine_map<()[s0] -> (1, s0 * 16, 8, 32)> : !torch.vtensor<[1,?,8,32],f16>
    %int1_484 = torch.constant.int 1
    %int2_485 = torch.constant.int 2
    %424 = torch.aten.transpose.int %354, %int1_484, %int2_485 : !torch.vtensor<[1,?,8,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,8,?,32],f16>
    torch.bind_symbolic_shape %424, [%32], affine_map<()[s0] -> (1, 8, s0 * 16, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %int1_486 = torch.constant.int 1
    %int2_487 = torch.constant.int 2
    %425 = torch.aten.transpose.int %417, %int1_486, %int2_487 : !torch.vtensor<[1,?,8,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,8,?,32],f16>
    torch.bind_symbolic_shape %425, [%32], affine_map<()[s0] -> (1, 8, s0 * 16, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %int1_488 = torch.constant.int 1
    %int2_489 = torch.constant.int 2
    %426 = torch.aten.transpose.int %423, %int1_488, %int2_489 : !torch.vtensor<[1,?,8,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,8,?,32],f16>
    torch.bind_symbolic_shape %426, [%32], affine_map<()[s0] -> (1, 8, s0 * 16, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %int2_490 = torch.constant.int 2
    %int3_491 = torch.constant.int 3
    %427 = torch.aten.transpose.int %425, %int2_490, %int3_491 : !torch.vtensor<[1,8,?,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,8,32,?],f16>
    torch.bind_symbolic_shape %427, [%32], affine_map<()[s0] -> (1, 8, 32, s0 * 16)> : !torch.vtensor<[1,8,32,?],f16>
    %int5_492 = torch.constant.int 5
    %428 = torch.prims.convert_element_type %427, %int5_492 : !torch.vtensor<[1,8,32,?],f16>, !torch.int -> !torch.vtensor<[1,8,32,?],f16>
    torch.bind_symbolic_shape %428, [%32], affine_map<()[s0] -> (1, 8, 32, s0 * 16)> : !torch.vtensor<[1,8,32,?],f16>
    %int1_493 = torch.constant.int 1
    %int8_494 = torch.constant.int 8
    %int32_495 = torch.constant.int 32
    %429 = torch.prim.ListConstruct %int1_493, %int8_494, %36, %int32_495 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_496 = torch.constant.bool false
    %430 = torch.aten.expand %424, %429, %false_496 : !torch.vtensor<[1,8,?,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,8,?,32],f16>
    torch.bind_symbolic_shape %430, [%32], affine_map<()[s0] -> (1, 8, s0 * 16, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %int8_497 = torch.constant.int 8
    %int32_498 = torch.constant.int 32
    %431 = torch.prim.ListConstruct %int8_497, %36, %int32_498 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %432 = torch.aten.view %430, %431 : !torch.vtensor<[1,8,?,32],f16>, !torch.list<int> -> !torch.vtensor<[8,?,32],f16>
    torch.bind_symbolic_shape %432, [%32], affine_map<()[s0] -> (8, s0 * 16, 32)> : !torch.vtensor<[8,?,32],f16>
    %int1_499 = torch.constant.int 1
    %int8_500 = torch.constant.int 8
    %int32_501 = torch.constant.int 32
    %433 = torch.prim.ListConstruct %int1_499, %int8_500, %int32_501, %36 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_502 = torch.constant.bool false
    %434 = torch.aten.expand %428, %433, %false_502 : !torch.vtensor<[1,8,32,?],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,8,32,?],f16>
    torch.bind_symbolic_shape %434, [%32], affine_map<()[s0] -> (1, 8, 32, s0 * 16)> : !torch.vtensor<[1,8,32,?],f16>
    %int8_503 = torch.constant.int 8
    %int32_504 = torch.constant.int 32
    %435 = torch.prim.ListConstruct %int8_503, %int32_504, %36 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %436 = torch.aten.view %434, %435 : !torch.vtensor<[1,8,32,?],f16>, !torch.list<int> -> !torch.vtensor<[8,32,?],f16>
    torch.bind_symbolic_shape %436, [%32], affine_map<()[s0] -> (8, 32, s0 * 16)> : !torch.vtensor<[8,32,?],f16>
    %437 = torch.aten.bmm %432, %436 : !torch.vtensor<[8,?,32],f16>, !torch.vtensor<[8,32,?],f16> -> !torch.vtensor<[8,?,?],f16>
    torch.bind_symbolic_shape %437, [%32], affine_map<()[s0] -> (8, s0 * 16, s0 * 16)> : !torch.vtensor<[8,?,?],f16>
    %int1_505 = torch.constant.int 1
    %int8_506 = torch.constant.int 8
    %438 = torch.prim.ListConstruct %int1_505, %int8_506, %36, %36 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %439 = torch.aten.view %437, %438 : !torch.vtensor<[8,?,?],f16>, !torch.list<int> -> !torch.vtensor<[1,8,?,?],f16>
    torch.bind_symbolic_shape %439, [%32], affine_map<()[s0] -> (1, 8, s0 * 16, s0 * 16)> : !torch.vtensor<[1,8,?,?],f16>
    %float5.656850e00_507 = torch.constant.float 5.6568542494923806
    %440 = torch.aten.div.Scalar %439, %float5.656850e00_507 : !torch.vtensor<[1,8,?,?],f16>, !torch.float -> !torch.vtensor<[1,8,?,?],f16>
    torch.bind_symbolic_shape %440, [%32], affine_map<()[s0] -> (1, 8, s0 * 16, s0 * 16)> : !torch.vtensor<[1,8,?,?],f16>
    %int1_508 = torch.constant.int 1
    %441 = torch.aten.add.Tensor %440, %62, %int1_508 : !torch.vtensor<[1,8,?,?],f16>, !torch.vtensor<[1,1,?,?],f16>, !torch.int -> !torch.vtensor<[1,8,?,?],f16>
    torch.bind_symbolic_shape %441, [%32], affine_map<()[s0] -> (1, 8, s0 * 16, s0 * 16)> : !torch.vtensor<[1,8,?,?],f16>
    %int6_509 = torch.constant.int 6
    %442 = torch.prims.convert_element_type %441, %int6_509 : !torch.vtensor<[1,8,?,?],f16>, !torch.int -> !torch.vtensor<[1,8,?,?],f32>
    torch.bind_symbolic_shape %442, [%32], affine_map<()[s0] -> (1, 8, s0 * 16, s0 * 16)> : !torch.vtensor<[1,8,?,?],f32>
    %int-1_510 = torch.constant.int -1
    %false_511 = torch.constant.bool false
    %443 = torch.aten._softmax %442, %int-1_510, %false_511 : !torch.vtensor<[1,8,?,?],f32>, !torch.int, !torch.bool -> !torch.vtensor<[1,8,?,?],f32>
    torch.bind_symbolic_shape %443, [%32], affine_map<()[s0] -> (1, 8, s0 * 16, s0 * 16)> : !torch.vtensor<[1,8,?,?],f32>
    %int5_512 = torch.constant.int 5
    %444 = torch.prims.convert_element_type %443, %int5_512 : !torch.vtensor<[1,8,?,?],f32>, !torch.int -> !torch.vtensor<[1,8,?,?],f16>
    torch.bind_symbolic_shape %444, [%32], affine_map<()[s0] -> (1, 8, s0 * 16, s0 * 16)> : !torch.vtensor<[1,8,?,?],f16>
    %int5_513 = torch.constant.int 5
    %445 = torch.prims.convert_element_type %426, %int5_513 : !torch.vtensor<[1,8,?,32],f16>, !torch.int -> !torch.vtensor<[1,8,?,32],f16>
    torch.bind_symbolic_shape %445, [%32], affine_map<()[s0] -> (1, 8, s0 * 16, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %int1_514 = torch.constant.int 1
    %int8_515 = torch.constant.int 8
    %446 = torch.prim.ListConstruct %int1_514, %int8_515, %36, %36 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_516 = torch.constant.bool false
    %447 = torch.aten.expand %444, %446, %false_516 : !torch.vtensor<[1,8,?,?],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,8,?,?],f16>
    torch.bind_symbolic_shape %447, [%32], affine_map<()[s0] -> (1, 8, s0 * 16, s0 * 16)> : !torch.vtensor<[1,8,?,?],f16>
    %int8_517 = torch.constant.int 8
    %448 = torch.prim.ListConstruct %int8_517, %36, %36 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %449 = torch.aten.view %447, %448 : !torch.vtensor<[1,8,?,?],f16>, !torch.list<int> -> !torch.vtensor<[8,?,?],f16>
    torch.bind_symbolic_shape %449, [%32], affine_map<()[s0] -> (8, s0 * 16, s0 * 16)> : !torch.vtensor<[8,?,?],f16>
    %int1_518 = torch.constant.int 1
    %int8_519 = torch.constant.int 8
    %int32_520 = torch.constant.int 32
    %450 = torch.prim.ListConstruct %int1_518, %int8_519, %36, %int32_520 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_521 = torch.constant.bool false
    %451 = torch.aten.expand %445, %450, %false_521 : !torch.vtensor<[1,8,?,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,8,?,32],f16>
    torch.bind_symbolic_shape %451, [%32], affine_map<()[s0] -> (1, 8, s0 * 16, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %int8_522 = torch.constant.int 8
    %int32_523 = torch.constant.int 32
    %452 = torch.prim.ListConstruct %int8_522, %36, %int32_523 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %453 = torch.aten.view %451, %452 : !torch.vtensor<[1,8,?,32],f16>, !torch.list<int> -> !torch.vtensor<[8,?,32],f16>
    torch.bind_symbolic_shape %453, [%32], affine_map<()[s0] -> (8, s0 * 16, 32)> : !torch.vtensor<[8,?,32],f16>
    %454 = torch.aten.bmm %449, %453 : !torch.vtensor<[8,?,?],f16>, !torch.vtensor<[8,?,32],f16> -> !torch.vtensor<[8,?,32],f16>
    torch.bind_symbolic_shape %454, [%32], affine_map<()[s0] -> (8, s0 * 16, 32)> : !torch.vtensor<[8,?,32],f16>
    %int1_524 = torch.constant.int 1
    %int8_525 = torch.constant.int 8
    %int32_526 = torch.constant.int 32
    %455 = torch.prim.ListConstruct %int1_524, %int8_525, %36, %int32_526 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %456 = torch.aten.view %454, %455 : !torch.vtensor<[8,?,32],f16>, !torch.list<int> -> !torch.vtensor<[1,8,?,32],f16>
    torch.bind_symbolic_shape %456, [%32], affine_map<()[s0] -> (1, 8, s0 * 16, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %int1_527 = torch.constant.int 1
    %int2_528 = torch.constant.int 2
    %457 = torch.aten.transpose.int %456, %int1_527, %int2_528 : !torch.vtensor<[1,8,?,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,?,8,32],f16>
    torch.bind_symbolic_shape %457, [%32], affine_map<()[s0] -> (1, s0 * 16, 8, 32)> : !torch.vtensor<[1,?,8,32],f16>
    %int0_529 = torch.constant.int 0
    %458 = torch.aten.clone %457, %int0_529 : !torch.vtensor<[1,?,8,32],f16>, !torch.int -> !torch.vtensor<[1,?,8,32],f16>
    torch.bind_symbolic_shape %458, [%32], affine_map<()[s0] -> (1, s0 * 16, 8, 32)> : !torch.vtensor<[1,?,8,32],f16>
    %int1_530 = torch.constant.int 1
    %int256_531 = torch.constant.int 256
    %459 = torch.prim.ListConstruct %int1_530, %36, %int256_531 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %460 = torch.aten._unsafe_view %458, %459 : !torch.vtensor<[1,?,8,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %460, [%32], affine_map<()[s0] -> (1, s0 * 16, 256)> : !torch.vtensor<[1,?,256],f16>
    %int5_532 = torch.constant.int 5
    %461 = torch.prims.convert_element_type %14, %int5_532 : !torch.vtensor<[256,256],f32>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int-2_533 = torch.constant.int -2
    %int-1_534 = torch.constant.int -1
    %462 = torch.aten.transpose.int %461, %int-2_533, %int-1_534 : !torch.vtensor<[256,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,256],f16>
    %int5_535 = torch.constant.int 5
    %463 = torch.prims.convert_element_type %462, %int5_535 : !torch.vtensor<[256,256],f16>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int256_536 = torch.constant.int 256
    %464 = torch.prim.ListConstruct %36, %int256_536 : (!torch.int, !torch.int) -> !torch.list<int>
    %465 = torch.aten.view %460, %464 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %465, [%32], affine_map<()[s0] -> (s0 * 16, 256)> : !torch.vtensor<[?,256],f16>
    %466 = torch.aten.mm %465, %463 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,256],f16> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %466, [%32], affine_map<()[s0] -> (s0 * 16, 256)> : !torch.vtensor<[?,256],f16>
    %int1_537 = torch.constant.int 1
    %int256_538 = torch.constant.int 256
    %467 = torch.prim.ListConstruct %int1_537, %36, %int256_538 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %468 = torch.aten.view %466, %467 : !torch.vtensor<[?,256],f16>, !torch.list<int> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %468, [%32], affine_map<()[s0] -> (1, s0 * 16, 256)> : !torch.vtensor<[1,?,256],f16>
    %int1_539 = torch.constant.int 1
    %469 = torch.aten.add.Tensor %286, %468, %int1_539 : !torch.vtensor<[1,?,256],f16>, !torch.vtensor<[1,?,256],f16>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %469, [%32], affine_map<()[s0] -> (1, s0 * 16, 256)> : !torch.vtensor<[1,?,256],f16>
    %int6_540 = torch.constant.int 6
    %470 = torch.prims.convert_element_type %469, %int6_540 : !torch.vtensor<[1,?,256],f16>, !torch.int -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %470, [%32], affine_map<()[s0] -> (1, s0 * 16, 256)> : !torch.vtensor<[1,?,256],f32>
    %int2_541 = torch.constant.int 2
    %471 = torch.aten.pow.Tensor_Scalar %470, %int2_541 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %471, [%32], affine_map<()[s0] -> (1, s0 * 16, 256)> : !torch.vtensor<[1,?,256],f32>
    %int-1_542 = torch.constant.int -1
    %472 = torch.prim.ListConstruct %int-1_542 : (!torch.int) -> !torch.list<int>
    %true_543 = torch.constant.bool true
    %none_544 = torch.constant.none
    %473 = torch.aten.mean.dim %471, %472, %true_543, %none_544 : !torch.vtensor<[1,?,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %473, [%32], affine_map<()[s0] -> (1, s0 * 16, 1)> : !torch.vtensor<[1,?,1],f32>
    %float1.000000e-02_545 = torch.constant.float 1.000000e-02
    %int1_546 = torch.constant.int 1
    %474 = torch.aten.add.Scalar %473, %float1.000000e-02_545, %int1_546 : !torch.vtensor<[1,?,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %474, [%32], affine_map<()[s0] -> (1, s0 * 16, 1)> : !torch.vtensor<[1,?,1],f32>
    %475 = torch.aten.rsqrt %474 : !torch.vtensor<[1,?,1],f32> -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %475, [%32], affine_map<()[s0] -> (1, s0 * 16, 1)> : !torch.vtensor<[1,?,1],f32>
    %476 = torch.aten.mul.Tensor %470, %475 : !torch.vtensor<[1,?,256],f32>, !torch.vtensor<[1,?,1],f32> -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %476, [%32], affine_map<()[s0] -> (1, s0 * 16, 256)> : !torch.vtensor<[1,?,256],f32>
    %int6_547 = torch.constant.int 6
    %477 = torch.prims.convert_element_type %476, %int6_547 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %477, [%32], affine_map<()[s0] -> (1, s0 * 16, 256)> : !torch.vtensor<[1,?,256],f32>
    %478 = torch.aten.mul.Tensor %15, %477 : !torch.vtensor<[256],f32>, !torch.vtensor<[1,?,256],f32> -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %478, [%32], affine_map<()[s0] -> (1, s0 * 16, 256)> : !torch.vtensor<[1,?,256],f32>
    %int5_548 = torch.constant.int 5
    %479 = torch.prims.convert_element_type %478, %int5_548 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %479, [%32], affine_map<()[s0] -> (1, s0 * 16, 256)> : !torch.vtensor<[1,?,256],f16>
    %int5_549 = torch.constant.int 5
    %480 = torch.prims.convert_element_type %16, %int5_549 : !torch.vtensor<[23,256],f32>, !torch.int -> !torch.vtensor<[23,256],f16>
    %int-2_550 = torch.constant.int -2
    %int-1_551 = torch.constant.int -1
    %481 = torch.aten.transpose.int %480, %int-2_550, %int-1_551 : !torch.vtensor<[23,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,23],f16>
    %int5_552 = torch.constant.int 5
    %482 = torch.prims.convert_element_type %481, %int5_552 : !torch.vtensor<[256,23],f16>, !torch.int -> !torch.vtensor<[256,23],f16>
    %int256_553 = torch.constant.int 256
    %483 = torch.prim.ListConstruct %36, %int256_553 : (!torch.int, !torch.int) -> !torch.list<int>
    %484 = torch.aten.view %479, %483 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %484, [%32], affine_map<()[s0] -> (s0 * 16, 256)> : !torch.vtensor<[?,256],f16>
    %485 = torch.aten.mm %484, %482 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,23],f16> -> !torch.vtensor<[?,23],f16>
    torch.bind_symbolic_shape %485, [%32], affine_map<()[s0] -> (s0 * 16, 23)> : !torch.vtensor<[?,23],f16>
    %int1_554 = torch.constant.int 1
    %int23_555 = torch.constant.int 23
    %486 = torch.prim.ListConstruct %int1_554, %36, %int23_555 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %487 = torch.aten.view %485, %486 : !torch.vtensor<[?,23],f16>, !torch.list<int> -> !torch.vtensor<[1,?,23],f16>
    torch.bind_symbolic_shape %487, [%32], affine_map<()[s0] -> (1, s0 * 16, 23)> : !torch.vtensor<[1,?,23],f16>
    %488 = torch.aten.silu %487 : !torch.vtensor<[1,?,23],f16> -> !torch.vtensor<[1,?,23],f16>
    torch.bind_symbolic_shape %488, [%32], affine_map<()[s0] -> (1, s0 * 16, 23)> : !torch.vtensor<[1,?,23],f16>
    %int5_556 = torch.constant.int 5
    %489 = torch.prims.convert_element_type %17, %int5_556 : !torch.vtensor<[23,256],f32>, !torch.int -> !torch.vtensor<[23,256],f16>
    %int-2_557 = torch.constant.int -2
    %int-1_558 = torch.constant.int -1
    %490 = torch.aten.transpose.int %489, %int-2_557, %int-1_558 : !torch.vtensor<[23,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,23],f16>
    %int5_559 = torch.constant.int 5
    %491 = torch.prims.convert_element_type %490, %int5_559 : !torch.vtensor<[256,23],f16>, !torch.int -> !torch.vtensor<[256,23],f16>
    %int256_560 = torch.constant.int 256
    %492 = torch.prim.ListConstruct %36, %int256_560 : (!torch.int, !torch.int) -> !torch.list<int>
    %493 = torch.aten.view %479, %492 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %493, [%32], affine_map<()[s0] -> (s0 * 16, 256)> : !torch.vtensor<[?,256],f16>
    %494 = torch.aten.mm %493, %491 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,23],f16> -> !torch.vtensor<[?,23],f16>
    torch.bind_symbolic_shape %494, [%32], affine_map<()[s0] -> (s0 * 16, 23)> : !torch.vtensor<[?,23],f16>
    %int1_561 = torch.constant.int 1
    %int23_562 = torch.constant.int 23
    %495 = torch.prim.ListConstruct %int1_561, %36, %int23_562 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %496 = torch.aten.view %494, %495 : !torch.vtensor<[?,23],f16>, !torch.list<int> -> !torch.vtensor<[1,?,23],f16>
    torch.bind_symbolic_shape %496, [%32], affine_map<()[s0] -> (1, s0 * 16, 23)> : !torch.vtensor<[1,?,23],f16>
    %497 = torch.aten.mul.Tensor %488, %496 : !torch.vtensor<[1,?,23],f16>, !torch.vtensor<[1,?,23],f16> -> !torch.vtensor<[1,?,23],f16>
    torch.bind_symbolic_shape %497, [%32], affine_map<()[s0] -> (1, s0 * 16, 23)> : !torch.vtensor<[1,?,23],f16>
    %int5_563 = torch.constant.int 5
    %498 = torch.prims.convert_element_type %18, %int5_563 : !torch.vtensor<[256,23],f32>, !torch.int -> !torch.vtensor<[256,23],f16>
    %int-2_564 = torch.constant.int -2
    %int-1_565 = torch.constant.int -1
    %499 = torch.aten.transpose.int %498, %int-2_564, %int-1_565 : !torch.vtensor<[256,23],f16>, !torch.int, !torch.int -> !torch.vtensor<[23,256],f16>
    %int5_566 = torch.constant.int 5
    %500 = torch.prims.convert_element_type %499, %int5_566 : !torch.vtensor<[23,256],f16>, !torch.int -> !torch.vtensor<[23,256],f16>
    %int23_567 = torch.constant.int 23
    %501 = torch.prim.ListConstruct %36, %int23_567 : (!torch.int, !torch.int) -> !torch.list<int>
    %502 = torch.aten.view %497, %501 : !torch.vtensor<[1,?,23],f16>, !torch.list<int> -> !torch.vtensor<[?,23],f16>
    torch.bind_symbolic_shape %502, [%32], affine_map<()[s0] -> (s0 * 16, 23)> : !torch.vtensor<[?,23],f16>
    %503 = torch.aten.mm %502, %500 : !torch.vtensor<[?,23],f16>, !torch.vtensor<[23,256],f16> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %503, [%32], affine_map<()[s0] -> (s0 * 16, 256)> : !torch.vtensor<[?,256],f16>
    %int1_568 = torch.constant.int 1
    %int256_569 = torch.constant.int 256
    %504 = torch.prim.ListConstruct %int1_568, %36, %int256_569 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %505 = torch.aten.view %503, %504 : !torch.vtensor<[?,256],f16>, !torch.list<int> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %505, [%32], affine_map<()[s0] -> (1, s0 * 16, 256)> : !torch.vtensor<[1,?,256],f16>
    %int1_570 = torch.constant.int 1
    %506 = torch.aten.add.Tensor %469, %505, %int1_570 : !torch.vtensor<[1,?,256],f16>, !torch.vtensor<[1,?,256],f16>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %506, [%32], affine_map<()[s0] -> (1, s0 * 16, 256)> : !torch.vtensor<[1,?,256],f16>
    %int6_571 = torch.constant.int 6
    %507 = torch.prims.convert_element_type %506, %int6_571 : !torch.vtensor<[1,?,256],f16>, !torch.int -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %507, [%32], affine_map<()[s0] -> (1, s0 * 16, 256)> : !torch.vtensor<[1,?,256],f32>
    %int2_572 = torch.constant.int 2
    %508 = torch.aten.pow.Tensor_Scalar %507, %int2_572 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %508, [%32], affine_map<()[s0] -> (1, s0 * 16, 256)> : !torch.vtensor<[1,?,256],f32>
    %int-1_573 = torch.constant.int -1
    %509 = torch.prim.ListConstruct %int-1_573 : (!torch.int) -> !torch.list<int>
    %true_574 = torch.constant.bool true
    %none_575 = torch.constant.none
    %510 = torch.aten.mean.dim %508, %509, %true_574, %none_575 : !torch.vtensor<[1,?,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %510, [%32], affine_map<()[s0] -> (1, s0 * 16, 1)> : !torch.vtensor<[1,?,1],f32>
    %float1.000000e-02_576 = torch.constant.float 1.000000e-02
    %int1_577 = torch.constant.int 1
    %511 = torch.aten.add.Scalar %510, %float1.000000e-02_576, %int1_577 : !torch.vtensor<[1,?,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %511, [%32], affine_map<()[s0] -> (1, s0 * 16, 1)> : !torch.vtensor<[1,?,1],f32>
    %512 = torch.aten.rsqrt %511 : !torch.vtensor<[1,?,1],f32> -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %512, [%32], affine_map<()[s0] -> (1, s0 * 16, 1)> : !torch.vtensor<[1,?,1],f32>
    %513 = torch.aten.mul.Tensor %507, %512 : !torch.vtensor<[1,?,256],f32>, !torch.vtensor<[1,?,1],f32> -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %513, [%32], affine_map<()[s0] -> (1, s0 * 16, 256)> : !torch.vtensor<[1,?,256],f32>
    %int6_578 = torch.constant.int 6
    %514 = torch.prims.convert_element_type %513, %int6_578 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %514, [%32], affine_map<()[s0] -> (1, s0 * 16, 256)> : !torch.vtensor<[1,?,256],f32>
    %515 = torch.aten.mul.Tensor %19, %514 : !torch.vtensor<[256],f32>, !torch.vtensor<[1,?,256],f32> -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %515, [%32], affine_map<()[s0] -> (1, s0 * 16, 256)> : !torch.vtensor<[1,?,256],f32>
    %int5_579 = torch.constant.int 5
    %516 = torch.prims.convert_element_type %515, %int5_579 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %516, [%32], affine_map<()[s0] -> (1, s0 * 16, 256)> : !torch.vtensor<[1,?,256],f16>
    %int5_580 = torch.constant.int 5
    %517 = torch.prims.convert_element_type %20, %int5_580 : !torch.vtensor<[256,256],f32>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int-2_581 = torch.constant.int -2
    %int-1_582 = torch.constant.int -1
    %518 = torch.aten.transpose.int %517, %int-2_581, %int-1_582 : !torch.vtensor<[256,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,256],f16>
    %int5_583 = torch.constant.int 5
    %519 = torch.prims.convert_element_type %518, %int5_583 : !torch.vtensor<[256,256],f16>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int256_584 = torch.constant.int 256
    %520 = torch.prim.ListConstruct %36, %int256_584 : (!torch.int, !torch.int) -> !torch.list<int>
    %521 = torch.aten.view %516, %520 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %521, [%32], affine_map<()[s0] -> (s0 * 16, 256)> : !torch.vtensor<[?,256],f16>
    %522 = torch.aten.mm %521, %519 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,256],f16> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %522, [%32], affine_map<()[s0] -> (s0 * 16, 256)> : !torch.vtensor<[?,256],f16>
    %int1_585 = torch.constant.int 1
    %int256_586 = torch.constant.int 256
    %523 = torch.prim.ListConstruct %int1_585, %36, %int256_586 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %524 = torch.aten.view %522, %523 : !torch.vtensor<[?,256],f16>, !torch.list<int> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %524, [%32], affine_map<()[s0] -> (1, s0 * 16, 256)> : !torch.vtensor<[1,?,256],f16>
    %int5_587 = torch.constant.int 5
    %525 = torch.prims.convert_element_type %21, %int5_587 : !torch.vtensor<[128,256],f32>, !torch.int -> !torch.vtensor<[128,256],f16>
    %int-2_588 = torch.constant.int -2
    %int-1_589 = torch.constant.int -1
    %526 = torch.aten.transpose.int %525, %int-2_588, %int-1_589 : !torch.vtensor<[128,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,128],f16>
    %int5_590 = torch.constant.int 5
    %527 = torch.prims.convert_element_type %526, %int5_590 : !torch.vtensor<[256,128],f16>, !torch.int -> !torch.vtensor<[256,128],f16>
    %int256_591 = torch.constant.int 256
    %528 = torch.prim.ListConstruct %36, %int256_591 : (!torch.int, !torch.int) -> !torch.list<int>
    %529 = torch.aten.view %516, %528 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %529, [%32], affine_map<()[s0] -> (s0 * 16, 256)> : !torch.vtensor<[?,256],f16>
    %530 = torch.aten.mm %529, %527 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,128],f16> -> !torch.vtensor<[?,128],f16>
    torch.bind_symbolic_shape %530, [%32], affine_map<()[s0] -> (s0 * 16, 128)> : !torch.vtensor<[?,128],f16>
    %int1_592 = torch.constant.int 1
    %int128_593 = torch.constant.int 128
    %531 = torch.prim.ListConstruct %int1_592, %36, %int128_593 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %532 = torch.aten.view %530, %531 : !torch.vtensor<[?,128],f16>, !torch.list<int> -> !torch.vtensor<[1,?,128],f16>
    torch.bind_symbolic_shape %532, [%32], affine_map<()[s0] -> (1, s0 * 16, 128)> : !torch.vtensor<[1,?,128],f16>
    %int5_594 = torch.constant.int 5
    %533 = torch.prims.convert_element_type %22, %int5_594 : !torch.vtensor<[128,256],f32>, !torch.int -> !torch.vtensor<[128,256],f16>
    %int-2_595 = torch.constant.int -2
    %int-1_596 = torch.constant.int -1
    %534 = torch.aten.transpose.int %533, %int-2_595, %int-1_596 : !torch.vtensor<[128,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,128],f16>
    %int5_597 = torch.constant.int 5
    %535 = torch.prims.convert_element_type %534, %int5_597 : !torch.vtensor<[256,128],f16>, !torch.int -> !torch.vtensor<[256,128],f16>
    %int256_598 = torch.constant.int 256
    %536 = torch.prim.ListConstruct %36, %int256_598 : (!torch.int, !torch.int) -> !torch.list<int>
    %537 = torch.aten.view %516, %536 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %537, [%32], affine_map<()[s0] -> (s0 * 16, 256)> : !torch.vtensor<[?,256],f16>
    %538 = torch.aten.mm %537, %535 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,128],f16> -> !torch.vtensor<[?,128],f16>
    torch.bind_symbolic_shape %538, [%32], affine_map<()[s0] -> (s0 * 16, 128)> : !torch.vtensor<[?,128],f16>
    %int1_599 = torch.constant.int 1
    %int128_600 = torch.constant.int 128
    %539 = torch.prim.ListConstruct %int1_599, %36, %int128_600 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %540 = torch.aten.view %538, %539 : !torch.vtensor<[?,128],f16>, !torch.list<int> -> !torch.vtensor<[1,?,128],f16>
    torch.bind_symbolic_shape %540, [%32], affine_map<()[s0] -> (1, s0 * 16, 128)> : !torch.vtensor<[1,?,128],f16>
    %int1_601 = torch.constant.int 1
    %int8_602 = torch.constant.int 8
    %int32_603 = torch.constant.int 32
    %541 = torch.prim.ListConstruct %int1_601, %36, %int8_602, %int32_603 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %542 = torch.aten.view %524, %541 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[1,?,8,32],f16>
    torch.bind_symbolic_shape %542, [%32], affine_map<()[s0] -> (1, s0 * 16, 8, 32)> : !torch.vtensor<[1,?,8,32],f16>
    %int1_604 = torch.constant.int 1
    %int4_605 = torch.constant.int 4
    %int32_606 = torch.constant.int 32
    %543 = torch.prim.ListConstruct %int1_604, %36, %int4_605, %int32_606 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %544 = torch.aten.view %532, %543 : !torch.vtensor<[1,?,128],f16>, !torch.list<int> -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %544, [%32], affine_map<()[s0] -> (1, s0 * 16, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int1_607 = torch.constant.int 1
    %int4_608 = torch.constant.int 4
    %int32_609 = torch.constant.int 32
    %545 = torch.prim.ListConstruct %int1_607, %36, %int4_608, %int32_609 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %546 = torch.aten.view %540, %545 : !torch.vtensor<[1,?,128],f16>, !torch.list<int> -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %546, [%32], affine_map<()[s0] -> (1, s0 * 16, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int128_610 = torch.constant.int 128
    %none_611 = torch.constant.none
    %none_612 = torch.constant.none
    %cpu_613 = torch.constant.device "cpu"
    %false_614 = torch.constant.bool false
    %547 = torch.aten.arange %int128_610, %none_611, %none_612, %cpu_613, %false_614 : !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[128],si64>
    %int0_615 = torch.constant.int 0
    %int32_616 = torch.constant.int 32
    %int2_617 = torch.constant.int 2
    %none_618 = torch.constant.none
    %none_619 = torch.constant.none
    %cpu_620 = torch.constant.device "cpu"
    %false_621 = torch.constant.bool false
    %548 = torch.aten.arange.start_step %int0_615, %int32_616, %int2_617, %none_618, %none_619, %cpu_620, %false_621 : !torch.int, !torch.int, !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[16],si64>
    %int0_622 = torch.constant.int 0
    %int0_623 = torch.constant.int 0
    %int16_624 = torch.constant.int 16
    %int1_625 = torch.constant.int 1
    %549 = torch.aten.slice.Tensor %548, %int0_622, %int0_623, %int16_624, %int1_625 : !torch.vtensor<[16],si64>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[16],si64>
    %int6_626 = torch.constant.int 6
    %550 = torch.prims.convert_element_type %549, %int6_626 : !torch.vtensor<[16],si64>, !torch.int -> !torch.vtensor<[16],f32>
    %int32_627 = torch.constant.int 32
    %551 = torch.aten.div.Scalar %550, %int32_627 : !torch.vtensor<[16],f32>, !torch.int -> !torch.vtensor<[16],f32>
    %float5.000000e05_628 = torch.constant.float 5.000000e+05
    %552 = torch.aten.pow.Scalar %float5.000000e05_628, %551 : !torch.float, !torch.vtensor<[16],f32> -> !torch.vtensor<[16],f32>
    %553 = torch.aten.reciprocal %552 : !torch.vtensor<[16],f32> -> !torch.vtensor<[16],f32>
    %float1.000000e00_629 = torch.constant.float 1.000000e+00
    %554 = torch.aten.mul.Scalar %553, %float1.000000e00_629 : !torch.vtensor<[16],f32>, !torch.float -> !torch.vtensor<[16],f32>
    %int128_630 = torch.constant.int 128
    %int1_631 = torch.constant.int 1
    %555 = torch.prim.ListConstruct %int128_630, %int1_631 : (!torch.int, !torch.int) -> !torch.list<int>
    %556 = torch.aten.view %547, %555 : !torch.vtensor<[128],si64>, !torch.list<int> -> !torch.vtensor<[128,1],si64>
    %557 = torch.aten.mul.Tensor %556, %554 : !torch.vtensor<[128,1],si64>, !torch.vtensor<[16],f32> -> !torch.vtensor<[128,16],f32>
    %int6_632 = torch.constant.int 6
    %558 = torch.prims.convert_element_type %557, %int6_632 : !torch.vtensor<[128,16],f32>, !torch.int -> !torch.vtensor<[128,16],f32>
    %559 = torch.aten.cos %558 : !torch.vtensor<[128,16],f32> -> !torch.vtensor<[128,16],f32>
    %560 = torch.aten.sin %558 : !torch.vtensor<[128,16],f32> -> !torch.vtensor<[128,16],f32>
    %561 = torch.aten.complex %559, %560 : !torch.vtensor<[128,16],f32>, !torch.vtensor<[128,16],f32> -> !torch.vtensor<[128,16],complex<f32>>
    %int0_633 = torch.constant.int 0
    %int0_634 = torch.constant.int 0
    %int1_635 = torch.constant.int 1
    %562 = torch.aten.slice.Tensor %561, %int0_633, %int0_634, %36, %int1_635 : !torch.vtensor<[128,16],complex<f32>>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,16],complex<f32>>
    torch.bind_symbolic_shape %562, [%32], affine_map<()[s0] -> (s0 * 16, 16)> : !torch.vtensor<[?,16],complex<f32>>
    %int1_636 = torch.constant.int 1
    %int0_637 = torch.constant.int 0
    %int9223372036854775807_638 = torch.constant.int 9223372036854775807
    %int1_639 = torch.constant.int 1
    %563 = torch.aten.slice.Tensor %562, %int1_636, %int0_637, %int9223372036854775807_638, %int1_639 : !torch.vtensor<[?,16],complex<f32>>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,16],complex<f32>>
    torch.bind_symbolic_shape %563, [%32], affine_map<()[s0] -> (s0 * 16, 16)> : !torch.vtensor<[?,16],complex<f32>>
    %int0_640 = torch.constant.int 0
    %564 = torch.aten.unsqueeze %563, %int0_640 : !torch.vtensor<[?,16],complex<f32>>, !torch.int -> !torch.vtensor<[1,?,16],complex<f32>>
    torch.bind_symbolic_shape %564, [%32], affine_map<()[s0] -> (1, s0 * 16, 16)> : !torch.vtensor<[1,?,16],complex<f32>>
    %int2_641 = torch.constant.int 2
    %565 = torch.aten.unsqueeze %564, %int2_641 : !torch.vtensor<[1,?,16],complex<f32>>, !torch.int -> !torch.vtensor<[1,?,1,16],complex<f32>>
    torch.bind_symbolic_shape %565, [%32], affine_map<()[s0] -> (1, s0 * 16, 1, 16)> : !torch.vtensor<[1,?,1,16],complex<f32>>
    %int3_642 = torch.constant.int 3
    %int0_643 = torch.constant.int 0
    %int9223372036854775807_644 = torch.constant.int 9223372036854775807
    %int1_645 = torch.constant.int 1
    %566 = torch.aten.slice.Tensor %565, %int3_642, %int0_643, %int9223372036854775807_644, %int1_645 : !torch.vtensor<[1,?,1,16],complex<f32>>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,1,16],complex<f32>>
    torch.bind_symbolic_shape %566, [%32], affine_map<()[s0] -> (1, s0 * 16, 1, 16)> : !torch.vtensor<[1,?,1,16],complex<f32>>
    %567 = torch_c.to_builtin_tensor %542 : !torch.vtensor<[1,?,8,32],f16> -> tensor<1x?x8x32xf16>
    %c1_646 = arith.constant 1 : index
    %dim_647 = tensor.dim %567, %c1_646 : tensor<1x?x8x32xf16>
    %568 = flow.tensor.bitcast %567 : tensor<1x?x8x32xf16>{%dim_647} -> tensor<1x?x8x16xcomplex<f16>>{%dim_647}
    %569 = torch_c.from_builtin_tensor %568 : tensor<1x?x8x16xcomplex<f16>> -> !torch.vtensor<[1,?,8,16],complex<f16>>
    torch.bind_symbolic_shape %569, [%32], affine_map<()[s0] -> (1, s0 * 16, 8, 16)> : !torch.vtensor<[1,?,8,16],complex<f16>>
    %570 = torch.aten.mul.Tensor %569, %566 : !torch.vtensor<[1,?,8,16],complex<f16>>, !torch.vtensor<[1,?,1,16],complex<f32>> -> !torch.vtensor<[1,?,8,16],complex<f32>>
    torch.bind_symbolic_shape %570, [%32], affine_map<()[s0] -> (1, s0 * 16, 8, 16)> : !torch.vtensor<[1,?,8,16],complex<f32>>
    %571 = torch_c.to_builtin_tensor %570 : !torch.vtensor<[1,?,8,16],complex<f32>> -> tensor<1x?x8x16xcomplex<f32>>
    %c1_648 = arith.constant 1 : index
    %dim_649 = tensor.dim %571, %c1_648 : tensor<1x?x8x16xcomplex<f32>>
    %572 = flow.tensor.bitcast %571 : tensor<1x?x8x16xcomplex<f32>>{%dim_649} -> tensor<1x?x8x32xf32>{%dim_649}
    %573 = torch_c.from_builtin_tensor %572 : tensor<1x?x8x32xf32> -> !torch.vtensor<[1,?,8,32],f32>
    torch.bind_symbolic_shape %573, [%32], affine_map<()[s0] -> (1, s0 * 16, 8, 32)> : !torch.vtensor<[1,?,8,32],f32>
    %int5_650 = torch.constant.int 5
    %574 = torch.prims.convert_element_type %573, %int5_650 : !torch.vtensor<[1,?,8,32],f32>, !torch.int -> !torch.vtensor<[1,?,8,32],f16>
    torch.bind_symbolic_shape %574, [%32], affine_map<()[s0] -> (1, s0 * 16, 8, 32)> : !torch.vtensor<[1,?,8,32],f16>
    %int128_651 = torch.constant.int 128
    %none_652 = torch.constant.none
    %none_653 = torch.constant.none
    %cpu_654 = torch.constant.device "cpu"
    %false_655 = torch.constant.bool false
    %575 = torch.aten.arange %int128_651, %none_652, %none_653, %cpu_654, %false_655 : !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[128],si64>
    %int0_656 = torch.constant.int 0
    %int32_657 = torch.constant.int 32
    %int2_658 = torch.constant.int 2
    %none_659 = torch.constant.none
    %none_660 = torch.constant.none
    %cpu_661 = torch.constant.device "cpu"
    %false_662 = torch.constant.bool false
    %576 = torch.aten.arange.start_step %int0_656, %int32_657, %int2_658, %none_659, %none_660, %cpu_661, %false_662 : !torch.int, !torch.int, !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[16],si64>
    %int0_663 = torch.constant.int 0
    %int0_664 = torch.constant.int 0
    %int16_665 = torch.constant.int 16
    %int1_666 = torch.constant.int 1
    %577 = torch.aten.slice.Tensor %576, %int0_663, %int0_664, %int16_665, %int1_666 : !torch.vtensor<[16],si64>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[16],si64>
    %int6_667 = torch.constant.int 6
    %578 = torch.prims.convert_element_type %577, %int6_667 : !torch.vtensor<[16],si64>, !torch.int -> !torch.vtensor<[16],f32>
    %int32_668 = torch.constant.int 32
    %579 = torch.aten.div.Scalar %578, %int32_668 : !torch.vtensor<[16],f32>, !torch.int -> !torch.vtensor<[16],f32>
    %float5.000000e05_669 = torch.constant.float 5.000000e+05
    %580 = torch.aten.pow.Scalar %float5.000000e05_669, %579 : !torch.float, !torch.vtensor<[16],f32> -> !torch.vtensor<[16],f32>
    %581 = torch.aten.reciprocal %580 : !torch.vtensor<[16],f32> -> !torch.vtensor<[16],f32>
    %float1.000000e00_670 = torch.constant.float 1.000000e+00
    %582 = torch.aten.mul.Scalar %581, %float1.000000e00_670 : !torch.vtensor<[16],f32>, !torch.float -> !torch.vtensor<[16],f32>
    %int128_671 = torch.constant.int 128
    %int1_672 = torch.constant.int 1
    %583 = torch.prim.ListConstruct %int128_671, %int1_672 : (!torch.int, !torch.int) -> !torch.list<int>
    %584 = torch.aten.view %575, %583 : !torch.vtensor<[128],si64>, !torch.list<int> -> !torch.vtensor<[128,1],si64>
    %585 = torch.aten.mul.Tensor %584, %582 : !torch.vtensor<[128,1],si64>, !torch.vtensor<[16],f32> -> !torch.vtensor<[128,16],f32>
    %int6_673 = torch.constant.int 6
    %586 = torch.prims.convert_element_type %585, %int6_673 : !torch.vtensor<[128,16],f32>, !torch.int -> !torch.vtensor<[128,16],f32>
    %587 = torch.aten.cos %586 : !torch.vtensor<[128,16],f32> -> !torch.vtensor<[128,16],f32>
    %588 = torch.aten.sin %586 : !torch.vtensor<[128,16],f32> -> !torch.vtensor<[128,16],f32>
    %589 = torch.aten.complex %587, %588 : !torch.vtensor<[128,16],f32>, !torch.vtensor<[128,16],f32> -> !torch.vtensor<[128,16],complex<f32>>
    %int0_674 = torch.constant.int 0
    %int0_675 = torch.constant.int 0
    %int1_676 = torch.constant.int 1
    %590 = torch.aten.slice.Tensor %589, %int0_674, %int0_675, %36, %int1_676 : !torch.vtensor<[128,16],complex<f32>>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,16],complex<f32>>
    torch.bind_symbolic_shape %590, [%32], affine_map<()[s0] -> (s0 * 16, 16)> : !torch.vtensor<[?,16],complex<f32>>
    %int1_677 = torch.constant.int 1
    %int0_678 = torch.constant.int 0
    %int9223372036854775807_679 = torch.constant.int 9223372036854775807
    %int1_680 = torch.constant.int 1
    %591 = torch.aten.slice.Tensor %590, %int1_677, %int0_678, %int9223372036854775807_679, %int1_680 : !torch.vtensor<[?,16],complex<f32>>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,16],complex<f32>>
    torch.bind_symbolic_shape %591, [%32], affine_map<()[s0] -> (s0 * 16, 16)> : !torch.vtensor<[?,16],complex<f32>>
    %int0_681 = torch.constant.int 0
    %592 = torch.aten.unsqueeze %591, %int0_681 : !torch.vtensor<[?,16],complex<f32>>, !torch.int -> !torch.vtensor<[1,?,16],complex<f32>>
    torch.bind_symbolic_shape %592, [%32], affine_map<()[s0] -> (1, s0 * 16, 16)> : !torch.vtensor<[1,?,16],complex<f32>>
    %int2_682 = torch.constant.int 2
    %593 = torch.aten.unsqueeze %592, %int2_682 : !torch.vtensor<[1,?,16],complex<f32>>, !torch.int -> !torch.vtensor<[1,?,1,16],complex<f32>>
    torch.bind_symbolic_shape %593, [%32], affine_map<()[s0] -> (1, s0 * 16, 1, 16)> : !torch.vtensor<[1,?,1,16],complex<f32>>
    %int3_683 = torch.constant.int 3
    %int0_684 = torch.constant.int 0
    %int9223372036854775807_685 = torch.constant.int 9223372036854775807
    %int1_686 = torch.constant.int 1
    %594 = torch.aten.slice.Tensor %593, %int3_683, %int0_684, %int9223372036854775807_685, %int1_686 : !torch.vtensor<[1,?,1,16],complex<f32>>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,1,16],complex<f32>>
    torch.bind_symbolic_shape %594, [%32], affine_map<()[s0] -> (1, s0 * 16, 1, 16)> : !torch.vtensor<[1,?,1,16],complex<f32>>
    %595 = torch_c.to_builtin_tensor %544 : !torch.vtensor<[1,?,4,32],f16> -> tensor<1x?x4x32xf16>
    %c1_687 = arith.constant 1 : index
    %dim_688 = tensor.dim %595, %c1_687 : tensor<1x?x4x32xf16>
    %596 = flow.tensor.bitcast %595 : tensor<1x?x4x32xf16>{%dim_688} -> tensor<1x?x4x16xcomplex<f16>>{%dim_688}
    %597 = torch_c.from_builtin_tensor %596 : tensor<1x?x4x16xcomplex<f16>> -> !torch.vtensor<[1,?,4,16],complex<f16>>
    torch.bind_symbolic_shape %597, [%32], affine_map<()[s0] -> (1, s0 * 16, 4, 16)> : !torch.vtensor<[1,?,4,16],complex<f16>>
    %598 = torch.aten.mul.Tensor %597, %594 : !torch.vtensor<[1,?,4,16],complex<f16>>, !torch.vtensor<[1,?,1,16],complex<f32>> -> !torch.vtensor<[1,?,4,16],complex<f32>>
    torch.bind_symbolic_shape %598, [%32], affine_map<()[s0] -> (1, s0 * 16, 4, 16)> : !torch.vtensor<[1,?,4,16],complex<f32>>
    %599 = torch_c.to_builtin_tensor %598 : !torch.vtensor<[1,?,4,16],complex<f32>> -> tensor<1x?x4x16xcomplex<f32>>
    %c1_689 = arith.constant 1 : index
    %dim_690 = tensor.dim %599, %c1_689 : tensor<1x?x4x16xcomplex<f32>>
    %600 = flow.tensor.bitcast %599 : tensor<1x?x4x16xcomplex<f32>>{%dim_690} -> tensor<1x?x4x32xf32>{%dim_690}
    %601 = torch_c.from_builtin_tensor %600 : tensor<1x?x4x32xf32> -> !torch.vtensor<[1,?,4,32],f32>
    torch.bind_symbolic_shape %601, [%32], affine_map<()[s0] -> (1, s0 * 16, 4, 32)> : !torch.vtensor<[1,?,4,32],f32>
    %int5_691 = torch.constant.int 5
    %602 = torch.prims.convert_element_type %601, %int5_691 : !torch.vtensor<[1,?,4,32],f32>, !torch.int -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %602, [%32], affine_map<()[s0] -> (1, s0 * 16, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int6_692 = torch.constant.int 6
    %603 = torch.aten.mul.Scalar %arg2, %int6_692 : !torch.vtensor<[1,?],si64>, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %603, [%32], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %int4_693 = torch.constant.int 4
    %int1_694 = torch.constant.int 1
    %604 = torch.aten.add.Scalar %603, %int4_693, %int1_694 : !torch.vtensor<[1,?],si64>, !torch.int, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %604, [%32], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %int1_695 = torch.constant.int 1
    %int16_696 = torch.constant.int 16
    %int4_697 = torch.constant.int 4
    %int32_698 = torch.constant.int 32
    %605 = torch.prim.ListConstruct %int1_695, %34, %int16_696, %int4_697, %int32_698 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %606 = torch.aten.view %602, %605 : !torch.vtensor<[1,?,4,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,16,4,32],f16>
    torch.bind_symbolic_shape %606, [%32], affine_map<()[s0] -> (1, s0, 16, 4, 32)> : !torch.vtensor<[1,?,16,4,32],f16>
    %int16_699 = torch.constant.int 16
    %int4_700 = torch.constant.int 4
    %int32_701 = torch.constant.int 32
    %607 = torch.prim.ListConstruct %34, %int16_699, %int4_700, %int32_701 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %608 = torch.aten.view %606, %607 : !torch.vtensor<[1,?,16,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,16,4,32],f16>
    torch.bind_symbolic_shape %608, [%32], affine_map<()[s0] -> (s0, 16, 4, 32)> : !torch.vtensor<[?,16,4,32],f16>
    %609 = torch.prim.ListConstruct %34 : (!torch.int) -> !torch.list<int>
    %610 = torch.aten.view %604, %609 : !torch.vtensor<[1,?],si64>, !torch.list<int> -> !torch.vtensor<[?],si64>
    torch.bind_symbolic_shape %610, [%32], affine_map<()[s0] -> (s0)> : !torch.vtensor<[?],si64>
    %int1_702 = torch.constant.int 1
    %int16_703 = torch.constant.int 16
    %int4_704 = torch.constant.int 4
    %int32_705 = torch.constant.int 32
    %611 = torch.prim.ListConstruct %int1_702, %34, %int16_703, %int4_704, %int32_705 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %612 = torch.aten.view %546, %611 : !torch.vtensor<[1,?,4,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,16,4,32],f16>
    torch.bind_symbolic_shape %612, [%32], affine_map<()[s0] -> (1, s0, 16, 4, 32)> : !torch.vtensor<[1,?,16,4,32],f16>
    %int16_706 = torch.constant.int 16
    %int4_707 = torch.constant.int 4
    %int32_708 = torch.constant.int 32
    %613 = torch.prim.ListConstruct %34, %int16_706, %int4_707, %int32_708 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %614 = torch.aten.view %612, %613 : !torch.vtensor<[1,?,16,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,16,4,32],f16>
    torch.bind_symbolic_shape %614, [%32], affine_map<()[s0] -> (s0, 16, 4, 32)> : !torch.vtensor<[?,16,4,32],f16>
    %int1_709 = torch.constant.int 1
    %int1_710 = torch.constant.int 1
    %615 = torch.aten.add.Scalar %604, %int1_709, %int1_710 : !torch.vtensor<[1,?],si64>, !torch.int, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %615, [%32], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %616 = torch.prim.ListConstruct %34 : (!torch.int) -> !torch.list<int>
    %617 = torch.aten.view %615, %616 : !torch.vtensor<[1,?],si64>, !torch.list<int> -> !torch.vtensor<[?],si64>
    torch.bind_symbolic_shape %617, [%32], affine_map<()[s0] -> (s0)> : !torch.vtensor<[?],si64>
    %618 = torch.prim.ListConstruct %610, %617 : (!torch.vtensor<[?],si64>, !torch.vtensor<[?],si64>) -> !torch.list<vtensor>
    %int0_711 = torch.constant.int 0
    %619 = torch.aten.cat %618, %int0_711 : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[?],si64>
    torch.bind_symbolic_shape %619, [%32], affine_map<()[s0] -> (s0 * 2)> : !torch.vtensor<[?],si64>
    %620 = torch.prim.ListConstruct %608, %614 : (!torch.vtensor<[?,16,4,32],f16>, !torch.vtensor<[?,16,4,32],f16>) -> !torch.list<vtensor>
    %int0_712 = torch.constant.int 0
    %621 = torch.aten.cat %620, %int0_712 : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[?,16,4,32],f16>
    torch.bind_symbolic_shape %621, [%32], affine_map<()[s0] -> (s0 * 2, 16, 4, 32)> : !torch.vtensor<[?,16,4,32],f16>
    %int3_713 = torch.constant.int 3
    %int2_714 = torch.constant.int 2
    %int16_715 = torch.constant.int 16
    %int4_716 = torch.constant.int 4
    %int32_717 = torch.constant.int 32
    %622 = torch.prim.ListConstruct %35, %int3_713, %int2_714, %int16_715, %int4_716, %int32_717 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %623 = torch.aten.view %411, %622 : !torch.vtensor<[?,12288],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,16,4,32],f16>
    torch.bind_symbolic_shape %623, [%33], affine_map<()[s0] -> (s0, 3, 2, 16, 4, 32)> : !torch.vtensor<[?,3,2,16,4,32],f16>
    %int16_718 = torch.constant.int 16
    %int4_719 = torch.constant.int 4
    %int32_720 = torch.constant.int 32
    %624 = torch.prim.ListConstruct %164, %int16_718, %int4_719, %int32_720 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %625 = torch.aten.view %623, %624 : !torch.vtensor<[?,3,2,16,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,16,4,32],f16>
    torch.bind_symbolic_shape %625, [%33], affine_map<()[s0] -> (s0 * 6, 16, 4, 32)> : !torch.vtensor<[?,16,4,32],f16>
    %626 = torch.prim.ListConstruct %619 : (!torch.vtensor<[?],si64>) -> !torch.list<optional<vtensor>>
    %false_721 = torch.constant.bool false
    %627 = torch.aten.index_put %625, %626, %621, %false_721 : !torch.vtensor<[?,16,4,32],f16>, !torch.list<optional<vtensor>>, !torch.vtensor<[?,16,4,32],f16>, !torch.bool -> !torch.vtensor<[?,16,4,32],f16>
    torch.bind_symbolic_shape %627, [%33], affine_map<()[s0] -> (s0 * 6, 16, 4, 32)> : !torch.vtensor<[?,16,4,32],f16>
    %int3_722 = torch.constant.int 3
    %int2_723 = torch.constant.int 2
    %int16_724 = torch.constant.int 16
    %int4_725 = torch.constant.int 4
    %int32_726 = torch.constant.int 32
    %628 = torch.prim.ListConstruct %35, %int3_722, %int2_723, %int16_724, %int4_725, %int32_726 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %629 = torch.aten.view %627, %628 : !torch.vtensor<[?,16,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,16,4,32],f16>
    torch.bind_symbolic_shape %629, [%33], affine_map<()[s0] -> (s0, 3, 2, 16, 4, 32)> : !torch.vtensor<[?,3,2,16,4,32],f16>
    %int12288_727 = torch.constant.int 12288
    %630 = torch.prim.ListConstruct %35, %int12288_727 : (!torch.int, !torch.int) -> !torch.list<int>
    %631 = torch.aten.view %629, %630 : !torch.vtensor<[?,3,2,16,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,12288],f16>
    torch.overwrite.tensor.contents %631 overwrites %arg3 : !torch.vtensor<[?,12288],f16>, !torch.tensor<[?,12288],f16>
    torch.bind_symbolic_shape %631, [%33], affine_map<()[s0] -> (s0, 12288)> : !torch.vtensor<[?,12288],f16>
    %int-2_728 = torch.constant.int -2
    %632 = torch.aten.unsqueeze %602, %int-2_728 : !torch.vtensor<[1,?,4,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,1,32],f16>
    torch.bind_symbolic_shape %632, [%32], affine_map<()[s0] -> (1, s0 * 16, 4, 1, 32)> : !torch.vtensor<[1,?,4,1,32],f16>
    %int1_729 = torch.constant.int 1
    %int4_730 = torch.constant.int 4
    %int2_731 = torch.constant.int 2
    %int32_732 = torch.constant.int 32
    %633 = torch.prim.ListConstruct %int1_729, %36, %int4_730, %int2_731, %int32_732 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_733 = torch.constant.bool false
    %634 = torch.aten.expand %632, %633, %false_733 : !torch.vtensor<[1,?,4,1,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,?,4,2,32],f16>
    torch.bind_symbolic_shape %634, [%32], affine_map<()[s0] -> (1, s0 * 16, 4, 2, 32)> : !torch.vtensor<[1,?,4,2,32],f16>
    %int0_734 = torch.constant.int 0
    %635 = torch.aten.clone %634, %int0_734 : !torch.vtensor<[1,?,4,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,2,32],f16>
    torch.bind_symbolic_shape %635, [%32], affine_map<()[s0] -> (1, s0 * 16, 4, 2, 32)> : !torch.vtensor<[1,?,4,2,32],f16>
    %int1_735 = torch.constant.int 1
    %int8_736 = torch.constant.int 8
    %int32_737 = torch.constant.int 32
    %636 = torch.prim.ListConstruct %int1_735, %36, %int8_736, %int32_737 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %637 = torch.aten._unsafe_view %635, %636 : !torch.vtensor<[1,?,4,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,8,32],f16>
    torch.bind_symbolic_shape %637, [%32], affine_map<()[s0] -> (1, s0 * 16, 8, 32)> : !torch.vtensor<[1,?,8,32],f16>
    %int-2_738 = torch.constant.int -2
    %638 = torch.aten.unsqueeze %546, %int-2_738 : !torch.vtensor<[1,?,4,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,1,32],f16>
    torch.bind_symbolic_shape %638, [%32], affine_map<()[s0] -> (1, s0 * 16, 4, 1, 32)> : !torch.vtensor<[1,?,4,1,32],f16>
    %int1_739 = torch.constant.int 1
    %int4_740 = torch.constant.int 4
    %int2_741 = torch.constant.int 2
    %int32_742 = torch.constant.int 32
    %639 = torch.prim.ListConstruct %int1_739, %36, %int4_740, %int2_741, %int32_742 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_743 = torch.constant.bool false
    %640 = torch.aten.expand %638, %639, %false_743 : !torch.vtensor<[1,?,4,1,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,?,4,2,32],f16>
    torch.bind_symbolic_shape %640, [%32], affine_map<()[s0] -> (1, s0 * 16, 4, 2, 32)> : !torch.vtensor<[1,?,4,2,32],f16>
    %int0_744 = torch.constant.int 0
    %641 = torch.aten.clone %640, %int0_744 : !torch.vtensor<[1,?,4,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,2,32],f16>
    torch.bind_symbolic_shape %641, [%32], affine_map<()[s0] -> (1, s0 * 16, 4, 2, 32)> : !torch.vtensor<[1,?,4,2,32],f16>
    %int1_745 = torch.constant.int 1
    %int8_746 = torch.constant.int 8
    %int32_747 = torch.constant.int 32
    %642 = torch.prim.ListConstruct %int1_745, %36, %int8_746, %int32_747 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %643 = torch.aten._unsafe_view %641, %642 : !torch.vtensor<[1,?,4,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,8,32],f16>
    torch.bind_symbolic_shape %643, [%32], affine_map<()[s0] -> (1, s0 * 16, 8, 32)> : !torch.vtensor<[1,?,8,32],f16>
    %int1_748 = torch.constant.int 1
    %int2_749 = torch.constant.int 2
    %644 = torch.aten.transpose.int %574, %int1_748, %int2_749 : !torch.vtensor<[1,?,8,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,8,?,32],f16>
    torch.bind_symbolic_shape %644, [%32], affine_map<()[s0] -> (1, 8, s0 * 16, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %int1_750 = torch.constant.int 1
    %int2_751 = torch.constant.int 2
    %645 = torch.aten.transpose.int %637, %int1_750, %int2_751 : !torch.vtensor<[1,?,8,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,8,?,32],f16>
    torch.bind_symbolic_shape %645, [%32], affine_map<()[s0] -> (1, 8, s0 * 16, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %int1_752 = torch.constant.int 1
    %int2_753 = torch.constant.int 2
    %646 = torch.aten.transpose.int %643, %int1_752, %int2_753 : !torch.vtensor<[1,?,8,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,8,?,32],f16>
    torch.bind_symbolic_shape %646, [%32], affine_map<()[s0] -> (1, 8, s0 * 16, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %int2_754 = torch.constant.int 2
    %int3_755 = torch.constant.int 3
    %647 = torch.aten.transpose.int %645, %int2_754, %int3_755 : !torch.vtensor<[1,8,?,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,8,32,?],f16>
    torch.bind_symbolic_shape %647, [%32], affine_map<()[s0] -> (1, 8, 32, s0 * 16)> : !torch.vtensor<[1,8,32,?],f16>
    %int5_756 = torch.constant.int 5
    %648 = torch.prims.convert_element_type %647, %int5_756 : !torch.vtensor<[1,8,32,?],f16>, !torch.int -> !torch.vtensor<[1,8,32,?],f16>
    torch.bind_symbolic_shape %648, [%32], affine_map<()[s0] -> (1, 8, 32, s0 * 16)> : !torch.vtensor<[1,8,32,?],f16>
    %int1_757 = torch.constant.int 1
    %int8_758 = torch.constant.int 8
    %int32_759 = torch.constant.int 32
    %649 = torch.prim.ListConstruct %int1_757, %int8_758, %36, %int32_759 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_760 = torch.constant.bool false
    %650 = torch.aten.expand %644, %649, %false_760 : !torch.vtensor<[1,8,?,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,8,?,32],f16>
    torch.bind_symbolic_shape %650, [%32], affine_map<()[s0] -> (1, 8, s0 * 16, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %int8_761 = torch.constant.int 8
    %int32_762 = torch.constant.int 32
    %651 = torch.prim.ListConstruct %int8_761, %36, %int32_762 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %652 = torch.aten.view %650, %651 : !torch.vtensor<[1,8,?,32],f16>, !torch.list<int> -> !torch.vtensor<[8,?,32],f16>
    torch.bind_symbolic_shape %652, [%32], affine_map<()[s0] -> (8, s0 * 16, 32)> : !torch.vtensor<[8,?,32],f16>
    %int1_763 = torch.constant.int 1
    %int8_764 = torch.constant.int 8
    %int32_765 = torch.constant.int 32
    %653 = torch.prim.ListConstruct %int1_763, %int8_764, %int32_765, %36 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_766 = torch.constant.bool false
    %654 = torch.aten.expand %648, %653, %false_766 : !torch.vtensor<[1,8,32,?],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,8,32,?],f16>
    torch.bind_symbolic_shape %654, [%32], affine_map<()[s0] -> (1, 8, 32, s0 * 16)> : !torch.vtensor<[1,8,32,?],f16>
    %int8_767 = torch.constant.int 8
    %int32_768 = torch.constant.int 32
    %655 = torch.prim.ListConstruct %int8_767, %int32_768, %36 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %656 = torch.aten.view %654, %655 : !torch.vtensor<[1,8,32,?],f16>, !torch.list<int> -> !torch.vtensor<[8,32,?],f16>
    torch.bind_symbolic_shape %656, [%32], affine_map<()[s0] -> (8, 32, s0 * 16)> : !torch.vtensor<[8,32,?],f16>
    %657 = torch.aten.bmm %652, %656 : !torch.vtensor<[8,?,32],f16>, !torch.vtensor<[8,32,?],f16> -> !torch.vtensor<[8,?,?],f16>
    torch.bind_symbolic_shape %657, [%32], affine_map<()[s0] -> (8, s0 * 16, s0 * 16)> : !torch.vtensor<[8,?,?],f16>
    %int1_769 = torch.constant.int 1
    %int8_770 = torch.constant.int 8
    %658 = torch.prim.ListConstruct %int1_769, %int8_770, %36, %36 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %659 = torch.aten.view %657, %658 : !torch.vtensor<[8,?,?],f16>, !torch.list<int> -> !torch.vtensor<[1,8,?,?],f16>
    torch.bind_symbolic_shape %659, [%32], affine_map<()[s0] -> (1, 8, s0 * 16, s0 * 16)> : !torch.vtensor<[1,8,?,?],f16>
    %float5.656850e00_771 = torch.constant.float 5.6568542494923806
    %660 = torch.aten.div.Scalar %659, %float5.656850e00_771 : !torch.vtensor<[1,8,?,?],f16>, !torch.float -> !torch.vtensor<[1,8,?,?],f16>
    torch.bind_symbolic_shape %660, [%32], affine_map<()[s0] -> (1, 8, s0 * 16, s0 * 16)> : !torch.vtensor<[1,8,?,?],f16>
    %int1_772 = torch.constant.int 1
    %661 = torch.aten.add.Tensor %660, %62, %int1_772 : !torch.vtensor<[1,8,?,?],f16>, !torch.vtensor<[1,1,?,?],f16>, !torch.int -> !torch.vtensor<[1,8,?,?],f16>
    torch.bind_symbolic_shape %661, [%32], affine_map<()[s0] -> (1, 8, s0 * 16, s0 * 16)> : !torch.vtensor<[1,8,?,?],f16>
    %int6_773 = torch.constant.int 6
    %662 = torch.prims.convert_element_type %661, %int6_773 : !torch.vtensor<[1,8,?,?],f16>, !torch.int -> !torch.vtensor<[1,8,?,?],f32>
    torch.bind_symbolic_shape %662, [%32], affine_map<()[s0] -> (1, 8, s0 * 16, s0 * 16)> : !torch.vtensor<[1,8,?,?],f32>
    %int-1_774 = torch.constant.int -1
    %false_775 = torch.constant.bool false
    %663 = torch.aten._softmax %662, %int-1_774, %false_775 : !torch.vtensor<[1,8,?,?],f32>, !torch.int, !torch.bool -> !torch.vtensor<[1,8,?,?],f32>
    torch.bind_symbolic_shape %663, [%32], affine_map<()[s0] -> (1, 8, s0 * 16, s0 * 16)> : !torch.vtensor<[1,8,?,?],f32>
    %int5_776 = torch.constant.int 5
    %664 = torch.prims.convert_element_type %663, %int5_776 : !torch.vtensor<[1,8,?,?],f32>, !torch.int -> !torch.vtensor<[1,8,?,?],f16>
    torch.bind_symbolic_shape %664, [%32], affine_map<()[s0] -> (1, 8, s0 * 16, s0 * 16)> : !torch.vtensor<[1,8,?,?],f16>
    %int5_777 = torch.constant.int 5
    %665 = torch.prims.convert_element_type %646, %int5_777 : !torch.vtensor<[1,8,?,32],f16>, !torch.int -> !torch.vtensor<[1,8,?,32],f16>
    torch.bind_symbolic_shape %665, [%32], affine_map<()[s0] -> (1, 8, s0 * 16, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %int1_778 = torch.constant.int 1
    %int8_779 = torch.constant.int 8
    %666 = torch.prim.ListConstruct %int1_778, %int8_779, %36, %36 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_780 = torch.constant.bool false
    %667 = torch.aten.expand %664, %666, %false_780 : !torch.vtensor<[1,8,?,?],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,8,?,?],f16>
    torch.bind_symbolic_shape %667, [%32], affine_map<()[s0] -> (1, 8, s0 * 16, s0 * 16)> : !torch.vtensor<[1,8,?,?],f16>
    %int8_781 = torch.constant.int 8
    %668 = torch.prim.ListConstruct %int8_781, %36, %36 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %669 = torch.aten.view %667, %668 : !torch.vtensor<[1,8,?,?],f16>, !torch.list<int> -> !torch.vtensor<[8,?,?],f16>
    torch.bind_symbolic_shape %669, [%32], affine_map<()[s0] -> (8, s0 * 16, s0 * 16)> : !torch.vtensor<[8,?,?],f16>
    %int1_782 = torch.constant.int 1
    %int8_783 = torch.constant.int 8
    %int32_784 = torch.constant.int 32
    %670 = torch.prim.ListConstruct %int1_782, %int8_783, %36, %int32_784 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_785 = torch.constant.bool false
    %671 = torch.aten.expand %665, %670, %false_785 : !torch.vtensor<[1,8,?,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,8,?,32],f16>
    torch.bind_symbolic_shape %671, [%32], affine_map<()[s0] -> (1, 8, s0 * 16, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %int8_786 = torch.constant.int 8
    %int32_787 = torch.constant.int 32
    %672 = torch.prim.ListConstruct %int8_786, %36, %int32_787 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %673 = torch.aten.view %671, %672 : !torch.vtensor<[1,8,?,32],f16>, !torch.list<int> -> !torch.vtensor<[8,?,32],f16>
    torch.bind_symbolic_shape %673, [%32], affine_map<()[s0] -> (8, s0 * 16, 32)> : !torch.vtensor<[8,?,32],f16>
    %674 = torch.aten.bmm %669, %673 : !torch.vtensor<[8,?,?],f16>, !torch.vtensor<[8,?,32],f16> -> !torch.vtensor<[8,?,32],f16>
    torch.bind_symbolic_shape %674, [%32], affine_map<()[s0] -> (8, s0 * 16, 32)> : !torch.vtensor<[8,?,32],f16>
    %int1_788 = torch.constant.int 1
    %int8_789 = torch.constant.int 8
    %int32_790 = torch.constant.int 32
    %675 = torch.prim.ListConstruct %int1_788, %int8_789, %36, %int32_790 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %676 = torch.aten.view %674, %675 : !torch.vtensor<[8,?,32],f16>, !torch.list<int> -> !torch.vtensor<[1,8,?,32],f16>
    torch.bind_symbolic_shape %676, [%32], affine_map<()[s0] -> (1, 8, s0 * 16, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %int1_791 = torch.constant.int 1
    %int2_792 = torch.constant.int 2
    %677 = torch.aten.transpose.int %676, %int1_791, %int2_792 : !torch.vtensor<[1,8,?,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,?,8,32],f16>
    torch.bind_symbolic_shape %677, [%32], affine_map<()[s0] -> (1, s0 * 16, 8, 32)> : !torch.vtensor<[1,?,8,32],f16>
    %int0_793 = torch.constant.int 0
    %678 = torch.aten.clone %677, %int0_793 : !torch.vtensor<[1,?,8,32],f16>, !torch.int -> !torch.vtensor<[1,?,8,32],f16>
    torch.bind_symbolic_shape %678, [%32], affine_map<()[s0] -> (1, s0 * 16, 8, 32)> : !torch.vtensor<[1,?,8,32],f16>
    %int1_794 = torch.constant.int 1
    %int256_795 = torch.constant.int 256
    %679 = torch.prim.ListConstruct %int1_794, %36, %int256_795 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %680 = torch.aten._unsafe_view %678, %679 : !torch.vtensor<[1,?,8,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %680, [%32], affine_map<()[s0] -> (1, s0 * 16, 256)> : !torch.vtensor<[1,?,256],f16>
    %int5_796 = torch.constant.int 5
    %681 = torch.prims.convert_element_type %23, %int5_796 : !torch.vtensor<[256,256],f32>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int-2_797 = torch.constant.int -2
    %int-1_798 = torch.constant.int -1
    %682 = torch.aten.transpose.int %681, %int-2_797, %int-1_798 : !torch.vtensor<[256,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,256],f16>
    %int5_799 = torch.constant.int 5
    %683 = torch.prims.convert_element_type %682, %int5_799 : !torch.vtensor<[256,256],f16>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int256_800 = torch.constant.int 256
    %684 = torch.prim.ListConstruct %36, %int256_800 : (!torch.int, !torch.int) -> !torch.list<int>
    %685 = torch.aten.view %680, %684 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %685, [%32], affine_map<()[s0] -> (s0 * 16, 256)> : !torch.vtensor<[?,256],f16>
    %686 = torch.aten.mm %685, %683 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,256],f16> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %686, [%32], affine_map<()[s0] -> (s0 * 16, 256)> : !torch.vtensor<[?,256],f16>
    %int1_801 = torch.constant.int 1
    %int256_802 = torch.constant.int 256
    %687 = torch.prim.ListConstruct %int1_801, %36, %int256_802 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %688 = torch.aten.view %686, %687 : !torch.vtensor<[?,256],f16>, !torch.list<int> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %688, [%32], affine_map<()[s0] -> (1, s0 * 16, 256)> : !torch.vtensor<[1,?,256],f16>
    %int1_803 = torch.constant.int 1
    %689 = torch.aten.add.Tensor %506, %688, %int1_803 : !torch.vtensor<[1,?,256],f16>, !torch.vtensor<[1,?,256],f16>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %689, [%32], affine_map<()[s0] -> (1, s0 * 16, 256)> : !torch.vtensor<[1,?,256],f16>
    %int6_804 = torch.constant.int 6
    %690 = torch.prims.convert_element_type %689, %int6_804 : !torch.vtensor<[1,?,256],f16>, !torch.int -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %690, [%32], affine_map<()[s0] -> (1, s0 * 16, 256)> : !torch.vtensor<[1,?,256],f32>
    %int2_805 = torch.constant.int 2
    %691 = torch.aten.pow.Tensor_Scalar %690, %int2_805 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %691, [%32], affine_map<()[s0] -> (1, s0 * 16, 256)> : !torch.vtensor<[1,?,256],f32>
    %int-1_806 = torch.constant.int -1
    %692 = torch.prim.ListConstruct %int-1_806 : (!torch.int) -> !torch.list<int>
    %true_807 = torch.constant.bool true
    %none_808 = torch.constant.none
    %693 = torch.aten.mean.dim %691, %692, %true_807, %none_808 : !torch.vtensor<[1,?,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %693, [%32], affine_map<()[s0] -> (1, s0 * 16, 1)> : !torch.vtensor<[1,?,1],f32>
    %float1.000000e-02_809 = torch.constant.float 1.000000e-02
    %int1_810 = torch.constant.int 1
    %694 = torch.aten.add.Scalar %693, %float1.000000e-02_809, %int1_810 : !torch.vtensor<[1,?,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %694, [%32], affine_map<()[s0] -> (1, s0 * 16, 1)> : !torch.vtensor<[1,?,1],f32>
    %695 = torch.aten.rsqrt %694 : !torch.vtensor<[1,?,1],f32> -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %695, [%32], affine_map<()[s0] -> (1, s0 * 16, 1)> : !torch.vtensor<[1,?,1],f32>
    %696 = torch.aten.mul.Tensor %690, %695 : !torch.vtensor<[1,?,256],f32>, !torch.vtensor<[1,?,1],f32> -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %696, [%32], affine_map<()[s0] -> (1, s0 * 16, 256)> : !torch.vtensor<[1,?,256],f32>
    %int6_811 = torch.constant.int 6
    %697 = torch.prims.convert_element_type %696, %int6_811 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %697, [%32], affine_map<()[s0] -> (1, s0 * 16, 256)> : !torch.vtensor<[1,?,256],f32>
    %698 = torch.aten.mul.Tensor %24, %697 : !torch.vtensor<[256],f32>, !torch.vtensor<[1,?,256],f32> -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %698, [%32], affine_map<()[s0] -> (1, s0 * 16, 256)> : !torch.vtensor<[1,?,256],f32>
    %int5_812 = torch.constant.int 5
    %699 = torch.prims.convert_element_type %698, %int5_812 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %699, [%32], affine_map<()[s0] -> (1, s0 * 16, 256)> : !torch.vtensor<[1,?,256],f16>
    %int5_813 = torch.constant.int 5
    %700 = torch.prims.convert_element_type %25, %int5_813 : !torch.vtensor<[23,256],f32>, !torch.int -> !torch.vtensor<[23,256],f16>
    %int-2_814 = torch.constant.int -2
    %int-1_815 = torch.constant.int -1
    %701 = torch.aten.transpose.int %700, %int-2_814, %int-1_815 : !torch.vtensor<[23,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,23],f16>
    %int5_816 = torch.constant.int 5
    %702 = torch.prims.convert_element_type %701, %int5_816 : !torch.vtensor<[256,23],f16>, !torch.int -> !torch.vtensor<[256,23],f16>
    %int256_817 = torch.constant.int 256
    %703 = torch.prim.ListConstruct %36, %int256_817 : (!torch.int, !torch.int) -> !torch.list<int>
    %704 = torch.aten.view %699, %703 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %704, [%32], affine_map<()[s0] -> (s0 * 16, 256)> : !torch.vtensor<[?,256],f16>
    %705 = torch.aten.mm %704, %702 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,23],f16> -> !torch.vtensor<[?,23],f16>
    torch.bind_symbolic_shape %705, [%32], affine_map<()[s0] -> (s0 * 16, 23)> : !torch.vtensor<[?,23],f16>
    %int1_818 = torch.constant.int 1
    %int23_819 = torch.constant.int 23
    %706 = torch.prim.ListConstruct %int1_818, %36, %int23_819 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %707 = torch.aten.view %705, %706 : !torch.vtensor<[?,23],f16>, !torch.list<int> -> !torch.vtensor<[1,?,23],f16>
    torch.bind_symbolic_shape %707, [%32], affine_map<()[s0] -> (1, s0 * 16, 23)> : !torch.vtensor<[1,?,23],f16>
    %708 = torch.aten.silu %707 : !torch.vtensor<[1,?,23],f16> -> !torch.vtensor<[1,?,23],f16>
    torch.bind_symbolic_shape %708, [%32], affine_map<()[s0] -> (1, s0 * 16, 23)> : !torch.vtensor<[1,?,23],f16>
    %int5_820 = torch.constant.int 5
    %709 = torch.prims.convert_element_type %26, %int5_820 : !torch.vtensor<[23,256],f32>, !torch.int -> !torch.vtensor<[23,256],f16>
    %int-2_821 = torch.constant.int -2
    %int-1_822 = torch.constant.int -1
    %710 = torch.aten.transpose.int %709, %int-2_821, %int-1_822 : !torch.vtensor<[23,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,23],f16>
    %int5_823 = torch.constant.int 5
    %711 = torch.prims.convert_element_type %710, %int5_823 : !torch.vtensor<[256,23],f16>, !torch.int -> !torch.vtensor<[256,23],f16>
    %int256_824 = torch.constant.int 256
    %712 = torch.prim.ListConstruct %36, %int256_824 : (!torch.int, !torch.int) -> !torch.list<int>
    %713 = torch.aten.view %699, %712 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %713, [%32], affine_map<()[s0] -> (s0 * 16, 256)> : !torch.vtensor<[?,256],f16>
    %714 = torch.aten.mm %713, %711 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,23],f16> -> !torch.vtensor<[?,23],f16>
    torch.bind_symbolic_shape %714, [%32], affine_map<()[s0] -> (s0 * 16, 23)> : !torch.vtensor<[?,23],f16>
    %int1_825 = torch.constant.int 1
    %int23_826 = torch.constant.int 23
    %715 = torch.prim.ListConstruct %int1_825, %36, %int23_826 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %716 = torch.aten.view %714, %715 : !torch.vtensor<[?,23],f16>, !torch.list<int> -> !torch.vtensor<[1,?,23],f16>
    torch.bind_symbolic_shape %716, [%32], affine_map<()[s0] -> (1, s0 * 16, 23)> : !torch.vtensor<[1,?,23],f16>
    %717 = torch.aten.mul.Tensor %708, %716 : !torch.vtensor<[1,?,23],f16>, !torch.vtensor<[1,?,23],f16> -> !torch.vtensor<[1,?,23],f16>
    torch.bind_symbolic_shape %717, [%32], affine_map<()[s0] -> (1, s0 * 16, 23)> : !torch.vtensor<[1,?,23],f16>
    %int5_827 = torch.constant.int 5
    %718 = torch.prims.convert_element_type %27, %int5_827 : !torch.vtensor<[256,23],f32>, !torch.int -> !torch.vtensor<[256,23],f16>
    %int-2_828 = torch.constant.int -2
    %int-1_829 = torch.constant.int -1
    %719 = torch.aten.transpose.int %718, %int-2_828, %int-1_829 : !torch.vtensor<[256,23],f16>, !torch.int, !torch.int -> !torch.vtensor<[23,256],f16>
    %int5_830 = torch.constant.int 5
    %720 = torch.prims.convert_element_type %719, %int5_830 : !torch.vtensor<[23,256],f16>, !torch.int -> !torch.vtensor<[23,256],f16>
    %int23_831 = torch.constant.int 23
    %721 = torch.prim.ListConstruct %36, %int23_831 : (!torch.int, !torch.int) -> !torch.list<int>
    %722 = torch.aten.view %717, %721 : !torch.vtensor<[1,?,23],f16>, !torch.list<int> -> !torch.vtensor<[?,23],f16>
    torch.bind_symbolic_shape %722, [%32], affine_map<()[s0] -> (s0 * 16, 23)> : !torch.vtensor<[?,23],f16>
    %723 = torch.aten.mm %722, %720 : !torch.vtensor<[?,23],f16>, !torch.vtensor<[23,256],f16> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %723, [%32], affine_map<()[s0] -> (s0 * 16, 256)> : !torch.vtensor<[?,256],f16>
    %int1_832 = torch.constant.int 1
    %int256_833 = torch.constant.int 256
    %724 = torch.prim.ListConstruct %int1_832, %36, %int256_833 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %725 = torch.aten.view %723, %724 : !torch.vtensor<[?,256],f16>, !torch.list<int> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %725, [%32], affine_map<()[s0] -> (1, s0 * 16, 256)> : !torch.vtensor<[1,?,256],f16>
    %int1_834 = torch.constant.int 1
    %726 = torch.aten.add.Tensor %689, %725, %int1_834 : !torch.vtensor<[1,?,256],f16>, !torch.vtensor<[1,?,256],f16>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %726, [%32], affine_map<()[s0] -> (1, s0 * 16, 256)> : !torch.vtensor<[1,?,256],f16>
    %int6_835 = torch.constant.int 6
    %727 = torch.prims.convert_element_type %726, %int6_835 : !torch.vtensor<[1,?,256],f16>, !torch.int -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %727, [%32], affine_map<()[s0] -> (1, s0 * 16, 256)> : !torch.vtensor<[1,?,256],f32>
    %int2_836 = torch.constant.int 2
    %728 = torch.aten.pow.Tensor_Scalar %727, %int2_836 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %728, [%32], affine_map<()[s0] -> (1, s0 * 16, 256)> : !torch.vtensor<[1,?,256],f32>
    %int-1_837 = torch.constant.int -1
    %729 = torch.prim.ListConstruct %int-1_837 : (!torch.int) -> !torch.list<int>
    %true_838 = torch.constant.bool true
    %none_839 = torch.constant.none
    %730 = torch.aten.mean.dim %728, %729, %true_838, %none_839 : !torch.vtensor<[1,?,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %730, [%32], affine_map<()[s0] -> (1, s0 * 16, 1)> : !torch.vtensor<[1,?,1],f32>
    %float1.000000e-02_840 = torch.constant.float 1.000000e-02
    %int1_841 = torch.constant.int 1
    %731 = torch.aten.add.Scalar %730, %float1.000000e-02_840, %int1_841 : !torch.vtensor<[1,?,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %731, [%32], affine_map<()[s0] -> (1, s0 * 16, 1)> : !torch.vtensor<[1,?,1],f32>
    %732 = torch.aten.rsqrt %731 : !torch.vtensor<[1,?,1],f32> -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %732, [%32], affine_map<()[s0] -> (1, s0 * 16, 1)> : !torch.vtensor<[1,?,1],f32>
    %733 = torch.aten.mul.Tensor %727, %732 : !torch.vtensor<[1,?,256],f32>, !torch.vtensor<[1,?,1],f32> -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %733, [%32], affine_map<()[s0] -> (1, s0 * 16, 256)> : !torch.vtensor<[1,?,256],f32>
    %int6_842 = torch.constant.int 6
    %734 = torch.prims.convert_element_type %733, %int6_842 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %734, [%32], affine_map<()[s0] -> (1, s0 * 16, 256)> : !torch.vtensor<[1,?,256],f32>
    %735 = torch.aten.mul.Tensor %28, %734 : !torch.vtensor<[1,256],f32>, !torch.vtensor<[1,?,256],f32> -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %735, [%32], affine_map<()[s0] -> (1, s0 * 16, 256)> : !torch.vtensor<[1,?,256],f32>
    %int5_843 = torch.constant.int 5
    %736 = torch.prims.convert_element_type %735, %int5_843 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %736, [%32], affine_map<()[s0] -> (1, s0 * 16, 256)> : !torch.vtensor<[1,?,256],f16>
    %int5_844 = torch.constant.int 5
    %737 = torch.prims.convert_element_type %29, %int5_844 : !torch.vtensor<[256,256],f32>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int-2_845 = torch.constant.int -2
    %int-1_846 = torch.constant.int -1
    %738 = torch.aten.transpose.int %737, %int-2_845, %int-1_846 : !torch.vtensor<[256,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,256],f16>
    %int5_847 = torch.constant.int 5
    %739 = torch.prims.convert_element_type %738, %int5_847 : !torch.vtensor<[256,256],f16>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int256_848 = torch.constant.int 256
    %740 = torch.prim.ListConstruct %36, %int256_848 : (!torch.int, !torch.int) -> !torch.list<int>
    %741 = torch.aten.view %736, %740 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %741, [%32], affine_map<()[s0] -> (s0 * 16, 256)> : !torch.vtensor<[?,256],f16>
    %742 = torch.aten.mm %741, %739 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,256],f16> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %742, [%32], affine_map<()[s0] -> (s0 * 16, 256)> : !torch.vtensor<[?,256],f16>
    %int1_849 = torch.constant.int 1
    %int256_850 = torch.constant.int 256
    %743 = torch.prim.ListConstruct %int1_849, %36, %int256_850 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %744 = torch.aten.view %742, %743 : !torch.vtensor<[?,256],f16>, !torch.list<int> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %744, [%32], affine_map<()[s0] -> (1, s0 * 16, 256)> : !torch.vtensor<[1,?,256],f16>
    return %744 : !torch.vtensor<[1,?,256],f16>
  }
  func.func @decode_bs1(%arg0: !torch.vtensor<[1,1],si64>, %arg1: !torch.vtensor<[1],si64>, %arg2: !torch.vtensor<[1],si64>, %arg3: !torch.vtensor<[1,?],si64>, %arg4: !torch.tensor<[?,12288],f16>) -> !torch.vtensor<[1,1,256],f16> attributes {torch.assume_strict_symbolic_shapes} {
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
    %30 = torch.copy.to_vtensor %arg4 : !torch.vtensor<[?,12288],f16>
    %31 = torch.symbolic_int "s0" {min_val = 2, max_val = 7} : !torch.int
    %32 = torch.symbolic_int "s1" {min_val = 0, max_val = 9223372036854775807} : !torch.int
    torch.bind_symbolic_shape %arg3, [%31], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %30, [%32], affine_map<()[s0] -> (s0, 12288)> : !torch.vtensor<[?,12288],f16>
    %int1 = torch.constant.int 1
    %33 = torch.aten.size.int %arg3, %int1 : !torch.vtensor<[1,?],si64>, !torch.int -> !torch.int
    %int0 = torch.constant.int 0
    %34 = torch.aten.size.int %30, %int0 : !torch.vtensor<[?,12288],f16>, !torch.int -> !torch.int
    %int16 = torch.constant.int 16
    %35 = torch.aten.mul.int %33, %int16 : !torch.int, !torch.int -> !torch.int
    %int0_0 = torch.constant.int 0
    %int1_1 = torch.constant.int 1
    %none = torch.constant.none
    %none_2 = torch.constant.none
    %cpu = torch.constant.device "cpu"
    %false = torch.constant.bool false
    %36 = torch.aten.arange.start_step %int0_0, %35, %int1_1, %none, %none_2, %cpu, %false : !torch.int, !torch.int, !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[?],si64>
    torch.bind_symbolic_shape %36, [%31], affine_map<()[s0] -> (s0 * 16)> : !torch.vtensor<[?],si64>
    %int-1 = torch.constant.int -1
    %37 = torch.aten.unsqueeze %arg1, %int-1 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %38 = torch.aten.ge.Tensor %36, %37 : !torch.vtensor<[?],si64>, !torch.vtensor<[1,1],si64> -> !torch.vtensor<[1,?],i1>
    torch.bind_symbolic_shape %38, [%31], affine_map<()[s0] -> (1, s0 * 16)> : !torch.vtensor<[1,?],i1>
    %int0_3 = torch.constant.int 0
    %int6 = torch.constant.int 6
    %int0_4 = torch.constant.int 0
    %cpu_5 = torch.constant.device "cpu"
    %none_6 = torch.constant.none
    %39 = torch.aten.scalar_tensor %int0_3, %int6, %int0_4, %cpu_5, %none_6 : !torch.int, !torch.int, !torch.int, !torch.Device, !torch.none -> !torch.vtensor<[],f32>
    %float-Inf = torch.constant.float 0xFFF0000000000000
    %int6_7 = torch.constant.int 6
    %int0_8 = torch.constant.int 0
    %cpu_9 = torch.constant.device "cpu"
    %none_10 = torch.constant.none
    %40 = torch.aten.scalar_tensor %float-Inf, %int6_7, %int0_8, %cpu_9, %none_10 : !torch.float, !torch.int, !torch.int, !torch.Device, !torch.none -> !torch.vtensor<[],f32>
    %41 = torch.aten.where.self %38, %40, %39 : !torch.vtensor<[1,?],i1>, !torch.vtensor<[],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[1,?],f32>
    torch.bind_symbolic_shape %41, [%31], affine_map<()[s0] -> (1, s0 * 16)> : !torch.vtensor<[1,?],f32>
    %int5 = torch.constant.int 5
    %42 = torch.prims.convert_element_type %41, %int5 : !torch.vtensor<[1,?],f32>, !torch.int -> !torch.vtensor<[1,?],f16>
    torch.bind_symbolic_shape %42, [%31], affine_map<()[s0] -> (1, s0 * 16)> : !torch.vtensor<[1,?],f16>
    %int1_11 = torch.constant.int 1
    %43 = torch.aten.unsqueeze %42, %int1_11 : !torch.vtensor<[1,?],f16>, !torch.int -> !torch.vtensor<[1,1,?],f16>
    torch.bind_symbolic_shape %43, [%31], affine_map<()[s0] -> (1, 1, s0 * 16)> : !torch.vtensor<[1,1,?],f16>
    %int1_12 = torch.constant.int 1
    %44 = torch.aten.unsqueeze %43, %int1_12 : !torch.vtensor<[1,1,?],f16>, !torch.int -> !torch.vtensor<[1,1,1,?],f16>
    torch.bind_symbolic_shape %44, [%31], affine_map<()[s0] -> (1, 1, 1, s0 * 16)> : !torch.vtensor<[1,1,1,?],f16>
    %int5_13 = torch.constant.int 5
    %45 = torch.prims.convert_element_type %44, %int5_13 : !torch.vtensor<[1,1,1,?],f16>, !torch.int -> !torch.vtensor<[1,1,1,?],f16>
    torch.bind_symbolic_shape %45, [%31], affine_map<()[s0] -> (1, 1, 1, s0 * 16)> : !torch.vtensor<[1,1,1,?],f16>
    %int0_14 = torch.constant.int 0
    %int1_15 = torch.constant.int 1
    %none_16 = torch.constant.none
    %none_17 = torch.constant.none
    %cpu_18 = torch.constant.device "cpu"
    %false_19 = torch.constant.bool false
    %46 = torch.aten.arange.start %int0_14, %int1_15, %none_16, %none_17, %cpu_18, %false_19 : !torch.int, !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[1],si64>
    %int0_20 = torch.constant.int 0
    %47 = torch.aten.unsqueeze %46, %int0_20 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_21 = torch.constant.int 1
    %48 = torch.aten.unsqueeze %arg2, %int1_21 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_22 = torch.constant.int 1
    %49 = torch.aten.add.Tensor %47, %48, %int1_22 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int128 = torch.constant.int 128
    %none_23 = torch.constant.none
    %none_24 = torch.constant.none
    %cpu_25 = torch.constant.device "cpu"
    %false_26 = torch.constant.bool false
    %50 = torch.aten.arange %int128, %none_23, %none_24, %cpu_25, %false_26 : !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[128],si64>
    %int0_27 = torch.constant.int 0
    %int32 = torch.constant.int 32
    %int2 = torch.constant.int 2
    %none_28 = torch.constant.none
    %none_29 = torch.constant.none
    %cpu_30 = torch.constant.device "cpu"
    %false_31 = torch.constant.bool false
    %51 = torch.aten.arange.start_step %int0_27, %int32, %int2, %none_28, %none_29, %cpu_30, %false_31 : !torch.int, !torch.int, !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[16],si64>
    %int0_32 = torch.constant.int 0
    %int0_33 = torch.constant.int 0
    %int16_34 = torch.constant.int 16
    %int1_35 = torch.constant.int 1
    %52 = torch.aten.slice.Tensor %51, %int0_32, %int0_33, %int16_34, %int1_35 : !torch.vtensor<[16],si64>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[16],si64>
    %int6_36 = torch.constant.int 6
    %53 = torch.prims.convert_element_type %52, %int6_36 : !torch.vtensor<[16],si64>, !torch.int -> !torch.vtensor<[16],f32>
    %int32_37 = torch.constant.int 32
    %54 = torch.aten.div.Scalar %53, %int32_37 : !torch.vtensor<[16],f32>, !torch.int -> !torch.vtensor<[16],f32>
    %float5.000000e05 = torch.constant.float 5.000000e+05
    %55 = torch.aten.pow.Scalar %float5.000000e05, %54 : !torch.float, !torch.vtensor<[16],f32> -> !torch.vtensor<[16],f32>
    %56 = torch.aten.reciprocal %55 : !torch.vtensor<[16],f32> -> !torch.vtensor<[16],f32>
    %float1.000000e00 = torch.constant.float 1.000000e+00
    %57 = torch.aten.mul.Scalar %56, %float1.000000e00 : !torch.vtensor<[16],f32>, !torch.float -> !torch.vtensor<[16],f32>
    %int128_38 = torch.constant.int 128
    %int1_39 = torch.constant.int 1
    %58 = torch.prim.ListConstruct %int128_38, %int1_39 : (!torch.int, !torch.int) -> !torch.list<int>
    %59 = torch.aten.view %50, %58 : !torch.vtensor<[128],si64>, !torch.list<int> -> !torch.vtensor<[128,1],si64>
    %60 = torch.aten.mul.Tensor %59, %57 : !torch.vtensor<[128,1],si64>, !torch.vtensor<[16],f32> -> !torch.vtensor<[128,16],f32>
    %int6_40 = torch.constant.int 6
    %61 = torch.prims.convert_element_type %60, %int6_40 : !torch.vtensor<[128,16],f32>, !torch.int -> !torch.vtensor<[128,16],f32>
    %62 = torch.aten.cos %61 : !torch.vtensor<[128,16],f32> -> !torch.vtensor<[128,16],f32>
    %63 = torch.aten.sin %61 : !torch.vtensor<[128,16],f32> -> !torch.vtensor<[128,16],f32>
    %64 = torch.aten.complex %62, %63 : !torch.vtensor<[128,16],f32>, !torch.vtensor<[128,16],f32> -> !torch.vtensor<[128,16],complex<f32>>
    %65 = torch.prim.ListConstruct %49 : (!torch.vtensor<[1,1],si64>) -> !torch.list<optional<vtensor>>
    %66 = torch.aten.index.Tensor %64, %65 : !torch.vtensor<[128,16],complex<f32>>, !torch.list<optional<vtensor>> -> !torch.vtensor<[1,1,16],complex<f32>>
    %int2_41 = torch.constant.int 2
    %67 = torch.aten.unsqueeze %66, %int2_41 : !torch.vtensor<[1,1,16],complex<f32>>, !torch.int -> !torch.vtensor<[1,1,1,16],complex<f32>>
    %int1_42 = torch.constant.int 1
    %int128_43 = torch.constant.int 128
    %int4 = torch.constant.int 4
    %int32_44 = torch.constant.int 32
    %68 = torch.prim.ListConstruct %int1_42, %int128_43, %int4, %int32_44 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %int5_45 = torch.constant.int 5
    %none_46 = torch.constant.none
    %cpu_47 = torch.constant.device "cpu"
    %false_48 = torch.constant.bool false
    %none_49 = torch.constant.none
    %69 = torch.aten.empty.memory_format %68, %int5_45, %none_46, %cpu_47, %false_48, %none_49 : !torch.list<int>, !torch.int, !torch.none, !torch.Device, !torch.bool, !torch.none -> !torch.vtensor<[1,128,4,32],f16>
    %int1_50 = torch.constant.int 1
    %int128_51 = torch.constant.int 128
    %int4_52 = torch.constant.int 4
    %int32_53 = torch.constant.int 32
    %70 = torch.prim.ListConstruct %int1_50, %int128_51, %int4_52, %int32_53 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %int5_54 = torch.constant.int 5
    %none_55 = torch.constant.none
    %cpu_56 = torch.constant.device "cpu"
    %false_57 = torch.constant.bool false
    %none_58 = torch.constant.none
    %71 = torch.aten.empty.memory_format %70, %int5_54, %none_55, %cpu_56, %false_57, %none_58 : !torch.list<int>, !torch.int, !torch.none, !torch.Device, !torch.bool, !torch.none -> !torch.vtensor<[1,128,4,32],f16>
    %int5_59 = torch.constant.int 5
    %72 = torch.prims.convert_element_type %0, %int5_59 : !torch.vtensor<[256,256],f32>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int-1_60 = torch.constant.int -1
    %false_61 = torch.constant.bool false
    %false_62 = torch.constant.bool false
    %73 = torch.aten.embedding %72, %arg0, %int-1_60, %false_61, %false_62 : !torch.vtensor<[256,256],f16>, !torch.vtensor<[1,1],si64>, !torch.int, !torch.bool, !torch.bool -> !torch.vtensor<[1,1,256],f16>
    %int6_63 = torch.constant.int 6
    %74 = torch.prims.convert_element_type %73, %int6_63 : !torch.vtensor<[1,1,256],f16>, !torch.int -> !torch.vtensor<[1,1,256],f32>
    %int2_64 = torch.constant.int 2
    %75 = torch.aten.pow.Tensor_Scalar %74, %int2_64 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f32>
    %int-1_65 = torch.constant.int -1
    %76 = torch.prim.ListConstruct %int-1_65 : (!torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %none_66 = torch.constant.none
    %77 = torch.aten.mean.dim %75, %76, %true, %none_66 : !torch.vtensor<[1,1,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1],f32>
    %float1.000000e-02 = torch.constant.float 1.000000e-02
    %int1_67 = torch.constant.int 1
    %78 = torch.aten.add.Scalar %77, %float1.000000e-02, %int1_67 : !torch.vtensor<[1,1,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,1,1],f32>
    %79 = torch.aten.rsqrt %78 : !torch.vtensor<[1,1,1],f32> -> !torch.vtensor<[1,1,1],f32>
    %80 = torch.aten.mul.Tensor %74, %79 : !torch.vtensor<[1,1,256],f32>, !torch.vtensor<[1,1,1],f32> -> !torch.vtensor<[1,1,256],f32>
    %int6_68 = torch.constant.int 6
    %81 = torch.prims.convert_element_type %80, %int6_68 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f32>
    %82 = torch.aten.mul.Tensor %1, %81 : !torch.vtensor<[256],f32>, !torch.vtensor<[1,1,256],f32> -> !torch.vtensor<[1,1,256],f32>
    %int5_69 = torch.constant.int 5
    %83 = torch.prims.convert_element_type %82, %int5_69 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %int5_70 = torch.constant.int 5
    %84 = torch.prims.convert_element_type %2, %int5_70 : !torch.vtensor<[256,256],f32>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int-2 = torch.constant.int -2
    %int-1_71 = torch.constant.int -1
    %85 = torch.aten.transpose.int %84, %int-2, %int-1_71 : !torch.vtensor<[256,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,256],f16>
    %int5_72 = torch.constant.int 5
    %86 = torch.prims.convert_element_type %85, %int5_72 : !torch.vtensor<[256,256],f16>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int1_73 = torch.constant.int 1
    %int256 = torch.constant.int 256
    %87 = torch.prim.ListConstruct %int1_73, %int256 : (!torch.int, !torch.int) -> !torch.list<int>
    %88 = torch.aten.view %83, %87 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %89 = torch.aten.mm %88, %86 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,256],f16> -> !torch.vtensor<[1,256],f16>
    %int1_74 = torch.constant.int 1
    %int1_75 = torch.constant.int 1
    %int256_76 = torch.constant.int 256
    %90 = torch.prim.ListConstruct %int1_74, %int1_75, %int256_76 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %91 = torch.aten.view %89, %90 : !torch.vtensor<[1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,1,256],f16>
    %int5_77 = torch.constant.int 5
    %92 = torch.prims.convert_element_type %3, %int5_77 : !torch.vtensor<[128,256],f32>, !torch.int -> !torch.vtensor<[128,256],f16>
    %int-2_78 = torch.constant.int -2
    %int-1_79 = torch.constant.int -1
    %93 = torch.aten.transpose.int %92, %int-2_78, %int-1_79 : !torch.vtensor<[128,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,128],f16>
    %int5_80 = torch.constant.int 5
    %94 = torch.prims.convert_element_type %93, %int5_80 : !torch.vtensor<[256,128],f16>, !torch.int -> !torch.vtensor<[256,128],f16>
    %int1_81 = torch.constant.int 1
    %int256_82 = torch.constant.int 256
    %95 = torch.prim.ListConstruct %int1_81, %int256_82 : (!torch.int, !torch.int) -> !torch.list<int>
    %96 = torch.aten.view %83, %95 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %97 = torch.aten.mm %96, %94 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,128],f16> -> !torch.vtensor<[1,128],f16>
    %int1_83 = torch.constant.int 1
    %int1_84 = torch.constant.int 1
    %int128_85 = torch.constant.int 128
    %98 = torch.prim.ListConstruct %int1_83, %int1_84, %int128_85 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %99 = torch.aten.view %97, %98 : !torch.vtensor<[1,128],f16>, !torch.list<int> -> !torch.vtensor<[1,1,128],f16>
    %int5_86 = torch.constant.int 5
    %100 = torch.prims.convert_element_type %4, %int5_86 : !torch.vtensor<[128,256],f32>, !torch.int -> !torch.vtensor<[128,256],f16>
    %int-2_87 = torch.constant.int -2
    %int-1_88 = torch.constant.int -1
    %101 = torch.aten.transpose.int %100, %int-2_87, %int-1_88 : !torch.vtensor<[128,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,128],f16>
    %int5_89 = torch.constant.int 5
    %102 = torch.prims.convert_element_type %101, %int5_89 : !torch.vtensor<[256,128],f16>, !torch.int -> !torch.vtensor<[256,128],f16>
    %int1_90 = torch.constant.int 1
    %int256_91 = torch.constant.int 256
    %103 = torch.prim.ListConstruct %int1_90, %int256_91 : (!torch.int, !torch.int) -> !torch.list<int>
    %104 = torch.aten.view %83, %103 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %105 = torch.aten.mm %104, %102 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,128],f16> -> !torch.vtensor<[1,128],f16>
    %int1_92 = torch.constant.int 1
    %int1_93 = torch.constant.int 1
    %int128_94 = torch.constant.int 128
    %106 = torch.prim.ListConstruct %int1_92, %int1_93, %int128_94 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %107 = torch.aten.view %105, %106 : !torch.vtensor<[1,128],f16>, !torch.list<int> -> !torch.vtensor<[1,1,128],f16>
    %int1_95 = torch.constant.int 1
    %int1_96 = torch.constant.int 1
    %int8 = torch.constant.int 8
    %int32_97 = torch.constant.int 32
    %108 = torch.prim.ListConstruct %int1_95, %int1_96, %int8, %int32_97 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %109 = torch.aten.view %91, %108 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,1,8,32],f16>
    %int1_98 = torch.constant.int 1
    %int1_99 = torch.constant.int 1
    %int4_100 = torch.constant.int 4
    %int32_101 = torch.constant.int 32
    %110 = torch.prim.ListConstruct %int1_98, %int1_99, %int4_100, %int32_101 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %111 = torch.aten.view %99, %110 : !torch.vtensor<[1,1,128],f16>, !torch.list<int> -> !torch.vtensor<[1,1,4,32],f16>
    %int1_102 = torch.constant.int 1
    %int1_103 = torch.constant.int 1
    %int4_104 = torch.constant.int 4
    %int32_105 = torch.constant.int 32
    %112 = torch.prim.ListConstruct %int1_102, %int1_103, %int4_104, %int32_105 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %113 = torch.aten.view %107, %112 : !torch.vtensor<[1,1,128],f16>, !torch.list<int> -> !torch.vtensor<[1,1,4,32],f16>
    %114 = torch_c.to_builtin_tensor %109 : !torch.vtensor<[1,1,8,32],f16> -> tensor<1x1x8x32xf16>
    %115 = flow.tensor.bitcast %114 : tensor<1x1x8x32xf16> -> tensor<1x1x8x16xcomplex<f16>>
    %116 = torch_c.from_builtin_tensor %115 : tensor<1x1x8x16xcomplex<f16>> -> !torch.vtensor<[1,1,8,16],complex<f16>>
    %117 = torch.aten.mul.Tensor %116, %67 : !torch.vtensor<[1,1,8,16],complex<f16>>, !torch.vtensor<[1,1,1,16],complex<f32>> -> !torch.vtensor<[1,1,8,16],complex<f32>>
    %118 = torch_c.to_builtin_tensor %117 : !torch.vtensor<[1,1,8,16],complex<f32>> -> tensor<1x1x8x16xcomplex<f32>>
    %119 = flow.tensor.bitcast %118 : tensor<1x1x8x16xcomplex<f32>> -> tensor<1x1x8x32xf32>
    %120 = torch_c.from_builtin_tensor %119 : tensor<1x1x8x32xf32> -> !torch.vtensor<[1,1,8,32],f32>
    %int5_106 = torch.constant.int 5
    %121 = torch.prims.convert_element_type %120, %int5_106 : !torch.vtensor<[1,1,8,32],f32>, !torch.int -> !torch.vtensor<[1,1,8,32],f16>
    %122 = torch_c.to_builtin_tensor %111 : !torch.vtensor<[1,1,4,32],f16> -> tensor<1x1x4x32xf16>
    %123 = flow.tensor.bitcast %122 : tensor<1x1x4x32xf16> -> tensor<1x1x4x16xcomplex<f16>>
    %124 = torch_c.from_builtin_tensor %123 : tensor<1x1x4x16xcomplex<f16>> -> !torch.vtensor<[1,1,4,16],complex<f16>>
    %125 = torch.aten.mul.Tensor %124, %67 : !torch.vtensor<[1,1,4,16],complex<f16>>, !torch.vtensor<[1,1,1,16],complex<f32>> -> !torch.vtensor<[1,1,4,16],complex<f32>>
    %126 = torch_c.to_builtin_tensor %125 : !torch.vtensor<[1,1,4,16],complex<f32>> -> tensor<1x1x4x16xcomplex<f32>>
    %127 = flow.tensor.bitcast %126 : tensor<1x1x4x16xcomplex<f32>> -> tensor<1x1x4x32xf32>
    %128 = torch_c.from_builtin_tensor %127 : tensor<1x1x4x32xf32> -> !torch.vtensor<[1,1,4,32],f32>
    %int5_107 = torch.constant.int 5
    %129 = torch.prims.convert_element_type %128, %int5_107 : !torch.vtensor<[1,1,4,32],f32>, !torch.int -> !torch.vtensor<[1,1,4,32],f16>
    %int3 = torch.constant.int 3
    %int2_108 = torch.constant.int 2
    %int16_109 = torch.constant.int 16
    %int4_110 = torch.constant.int 4
    %int32_111 = torch.constant.int 32
    %130 = torch.prim.ListConstruct %34, %int3, %int2_108, %int16_109, %int4_110, %int32_111 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %131 = torch.aten.view %30, %130 : !torch.vtensor<[?,12288],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,16,4,32],f16>
    torch.bind_symbolic_shape %131, [%32], affine_map<()[s0] -> (s0, 3, 2, 16, 4, 32)> : !torch.vtensor<[?,3,2,16,4,32],f16>
    %132 = torch.prim.ListConstruct %129, %113 : (!torch.vtensor<[1,1,4,32],f16>, !torch.vtensor<[1,1,4,32],f16>) -> !torch.list<vtensor>
    %int1_112 = torch.constant.int 1
    %133 = torch.aten.cat %132, %int1_112 : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[1,2,4,32],f16>
    %int16_113 = torch.constant.int 16
    %134 = torch.aten.floor_divide.Scalar %arg2, %int16_113 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int1_114 = torch.constant.int 1
    %135 = torch.aten.unsqueeze %134, %int1_114 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_115 = torch.constant.int 1
    %false_116 = torch.constant.bool false
    %136 = torch.aten.gather %arg3, %int1_115, %135, %false_116 : !torch.vtensor<[1,?],si64>, !torch.int, !torch.vtensor<[1,1],si64>, !torch.bool -> !torch.vtensor<[1,1],si64>
    %int16_117 = torch.constant.int 16
    %137 = torch.aten.remainder.Scalar %arg2, %int16_117 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int1_118 = torch.constant.int 1
    %138 = torch.aten.unsqueeze %137, %int1_118 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int0_119 = torch.constant.int 0
    %int2_120 = torch.constant.int 2
    %none_121 = torch.constant.none
    %none_122 = torch.constant.none
    %cpu_123 = torch.constant.device "cpu"
    %false_124 = torch.constant.bool false
    %139 = torch.aten.arange.start %int0_119, %int2_120, %none_121, %none_122, %cpu_123, %false_124 : !torch.int, !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[2],si64>
    %int0_125 = torch.constant.int 0
    %140 = torch.aten.unsqueeze %139, %int0_125 : !torch.vtensor<[2],si64>, !torch.int -> !torch.vtensor<[1,2],si64>
    %int1_126 = torch.constant.int 1
    %int2_127 = torch.constant.int 2
    %141 = torch.prim.ListConstruct %int1_126, %int2_127 : (!torch.int, !torch.int) -> !torch.list<int>
    %142 = torch.aten.repeat %136, %141 : !torch.vtensor<[1,1],si64>, !torch.list<int> -> !torch.vtensor<[1,2],si64>
    %int1_128 = torch.constant.int 1
    %int2_129 = torch.constant.int 2
    %143 = torch.prim.ListConstruct %int1_128, %int2_129 : (!torch.int, !torch.int) -> !torch.list<int>
    %int2_130 = torch.constant.int 2
    %int1_131 = torch.constant.int 1
    %144 = torch.prim.ListConstruct %int2_130, %int1_131 : (!torch.int, !torch.int) -> !torch.list<int>
    %int4_132 = torch.constant.int 4
    %int0_133 = torch.constant.int 0
    %cpu_134 = torch.constant.device "cpu"
    %false_135 = torch.constant.bool false
    %145 = torch.aten.empty_strided %143, %144, %int4_132, %int0_133, %cpu_134, %false_135 : !torch.list<int>, !torch.list<int>, !torch.int, !torch.int, !torch.Device, !torch.bool -> !torch.vtensor<[1,2],si64>
    %int0_136 = torch.constant.int 0
    %146 = torch.aten.fill.Scalar %145, %int0_136 : !torch.vtensor<[1,2],si64>, !torch.int -> !torch.vtensor<[1,2],si64>
    %int1_137 = torch.constant.int 1
    %int2_138 = torch.constant.int 2
    %147 = torch.prim.ListConstruct %int1_137, %int2_138 : (!torch.int, !torch.int) -> !torch.list<int>
    %148 = torch.aten.repeat %138, %147 : !torch.vtensor<[1,1],si64>, !torch.list<int> -> !torch.vtensor<[1,2],si64>
    %int1_139 = torch.constant.int 1
    %int1_140 = torch.constant.int 1
    %149 = torch.prim.ListConstruct %int1_139, %int1_140 : (!torch.int, !torch.int) -> !torch.list<int>
    %150 = torch.aten.repeat %140, %149 : !torch.vtensor<[1,2],si64>, !torch.list<int> -> !torch.vtensor<[1,2],si64>
    %151 = torch.prim.ListConstruct %142, %146, %150, %148 : (!torch.vtensor<[1,2],si64>, !torch.vtensor<[1,2],si64>, !torch.vtensor<[1,2],si64>, !torch.vtensor<[1,2],si64>) -> !torch.list<optional<vtensor>>
    %false_141 = torch.constant.bool false
    %152 = torch.aten.index_put %131, %151, %133, %false_141 : !torch.vtensor<[?,3,2,16,4,32],f16>, !torch.list<optional<vtensor>>, !torch.vtensor<[1,2,4,32],f16>, !torch.bool -> !torch.vtensor<[?,3,2,16,4,32],f16>
    torch.bind_symbolic_shape %152, [%32], affine_map<()[s0] -> (s0, 3, 2, 16, 4, 32)> : !torch.vtensor<[?,3,2,16,4,32],f16>
    %int12288 = torch.constant.int 12288
    %153 = torch.prim.ListConstruct %34, %int12288 : (!torch.int, !torch.int) -> !torch.list<int>
    %154 = torch.aten.view %152, %153 : !torch.vtensor<[?,3,2,16,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,12288],f16>
    torch.bind_symbolic_shape %154, [%32], affine_map<()[s0] -> (s0, 12288)> : !torch.vtensor<[?,12288],f16>
    %int0_142 = torch.constant.int 0
    %int0_143 = torch.constant.int 0
    %int9223372036854775807 = torch.constant.int 9223372036854775807
    %int1_144 = torch.constant.int 1
    %155 = torch.aten.slice.Tensor %69, %int0_142, %int0_143, %int9223372036854775807, %int1_144 : !torch.vtensor<[1,128,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,128,4,32],f16>
    %int1_145 = torch.constant.int 1
    %int0_146 = torch.constant.int 0
    %int1_147 = torch.constant.int 1
    %156 = torch.aten.slice.Tensor %155, %int1_145, %int0_146, %35, %int1_147 : !torch.vtensor<[1,128,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %156, [%31], affine_map<()[s0] -> (1, s0 * 16, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int0_148 = torch.constant.int 0
    %int0_149 = torch.constant.int 0
    %int9223372036854775807_150 = torch.constant.int 9223372036854775807
    %int1_151 = torch.constant.int 1
    %157 = torch.aten.slice.Tensor %71, %int0_148, %int0_149, %int9223372036854775807_150, %int1_151 : !torch.vtensor<[1,128,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,128,4,32],f16>
    %int1_152 = torch.constant.int 1
    %int0_153 = torch.constant.int 0
    %int1_154 = torch.constant.int 1
    %158 = torch.aten.slice.Tensor %157, %int1_152, %int0_153, %35, %int1_154 : !torch.vtensor<[1,128,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %158, [%31], affine_map<()[s0] -> (1, s0 * 16, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int6_155 = torch.constant.int 6
    %159 = torch.aten.mul.Scalar %arg3, %int6_155 : !torch.vtensor<[1,?],si64>, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %159, [%31], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %int0_156 = torch.constant.int 0
    %int1_157 = torch.constant.int 1
    %160 = torch.aten.add.Scalar %159, %int0_156, %int1_157 : !torch.vtensor<[1,?],si64>, !torch.int, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %160, [%31], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %161 = torch.prim.ListConstruct %33 : (!torch.int) -> !torch.list<int>
    %162 = torch.aten.view %160, %161 : !torch.vtensor<[1,?],si64>, !torch.list<int> -> !torch.vtensor<[?],si64>
    torch.bind_symbolic_shape %162, [%31], affine_map<()[s0] -> (s0)> : !torch.vtensor<[?],si64>
    %int3_158 = torch.constant.int 3
    %int2_159 = torch.constant.int 2
    %int16_160 = torch.constant.int 16
    %int4_161 = torch.constant.int 4
    %int32_162 = torch.constant.int 32
    %163 = torch.prim.ListConstruct %34, %int3_158, %int2_159, %int16_160, %int4_161, %int32_162 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %164 = torch.aten.view %154, %163 : !torch.vtensor<[?,12288],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,16,4,32],f16>
    torch.bind_symbolic_shape %164, [%32], affine_map<()[s0] -> (s0, 3, 2, 16, 4, 32)> : !torch.vtensor<[?,3,2,16,4,32],f16>
    %int3_163 = torch.constant.int 3
    %165 = torch.aten.mul.int %34, %int3_163 : !torch.int, !torch.int -> !torch.int
    %int2_164 = torch.constant.int 2
    %166 = torch.aten.mul.int %165, %int2_164 : !torch.int, !torch.int -> !torch.int
    %int16_165 = torch.constant.int 16
    %int4_166 = torch.constant.int 4
    %int32_167 = torch.constant.int 32
    %167 = torch.prim.ListConstruct %166, %int16_165, %int4_166, %int32_167 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %168 = torch.aten.view %164, %167 : !torch.vtensor<[?,3,2,16,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,16,4,32],f16>
    torch.bind_symbolic_shape %168, [%32], affine_map<()[s0] -> (s0 * 6, 16, 4, 32)> : !torch.vtensor<[?,16,4,32],f16>
    %int0_168 = torch.constant.int 0
    %169 = torch.aten.index_select %168, %int0_168, %162 : !torch.vtensor<[?,16,4,32],f16>, !torch.int, !torch.vtensor<[?],si64> -> !torch.vtensor<[?,16,4,32],f16>
    torch.bind_symbolic_shape %169, [%31], affine_map<()[s0] -> (s0, 16, 4, 32)> : !torch.vtensor<[?,16,4,32],f16>
    %int1_169 = torch.constant.int 1
    %int16_170 = torch.constant.int 16
    %int4_171 = torch.constant.int 4
    %int32_172 = torch.constant.int 32
    %170 = torch.prim.ListConstruct %int1_169, %33, %int16_170, %int4_171, %int32_172 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %171 = torch.aten.view %169, %170 : !torch.vtensor<[?,16,4,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,16,4,32],f16>
    torch.bind_symbolic_shape %171, [%31], affine_map<()[s0] -> (1, s0, 16, 4, 32)> : !torch.vtensor<[1,?,16,4,32],f16>
    %int1_173 = torch.constant.int 1
    %int4_174 = torch.constant.int 4
    %int32_175 = torch.constant.int 32
    %172 = torch.prim.ListConstruct %int1_173, %35, %int4_174, %int32_175 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %173 = torch.aten.view %171, %172 : !torch.vtensor<[1,?,16,4,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %173, [%31], affine_map<()[s0] -> (1, s0 * 16, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %false_176 = torch.constant.bool false
    %174 = torch.aten.copy %156, %173, %false_176 : !torch.vtensor<[1,?,4,32],f16>, !torch.vtensor<[1,?,4,32],f16>, !torch.bool -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %174, [%31], affine_map<()[s0] -> (1, s0 * 16, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int0_177 = torch.constant.int 0
    %int0_178 = torch.constant.int 0
    %int9223372036854775807_179 = torch.constant.int 9223372036854775807
    %int1_180 = torch.constant.int 1
    %175 = torch.aten.slice.Tensor %69, %int0_177, %int0_178, %int9223372036854775807_179, %int1_180 : !torch.vtensor<[1,128,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,128,4,32],f16>
    %int1_181 = torch.constant.int 1
    %int0_182 = torch.constant.int 0
    %int1_183 = torch.constant.int 1
    %176 = torch.aten.slice_scatter %175, %174, %int1_181, %int0_182, %35, %int1_183 : !torch.vtensor<[1,128,4,32],f16>, !torch.vtensor<[1,?,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,128,4,32],f16>
    %int0_184 = torch.constant.int 0
    %int0_185 = torch.constant.int 0
    %int9223372036854775807_186 = torch.constant.int 9223372036854775807
    %int1_187 = torch.constant.int 1
    %177 = torch.aten.slice_scatter %69, %176, %int0_184, %int0_185, %int9223372036854775807_186, %int1_187 : !torch.vtensor<[1,128,4,32],f16>, !torch.vtensor<[1,128,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,128,4,32],f16>
    %int1_188 = torch.constant.int 1
    %int1_189 = torch.constant.int 1
    %178 = torch.aten.add.Scalar %160, %int1_188, %int1_189 : !torch.vtensor<[1,?],si64>, !torch.int, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %178, [%31], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %179 = torch.prim.ListConstruct %33 : (!torch.int) -> !torch.list<int>
    %180 = torch.aten.view %178, %179 : !torch.vtensor<[1,?],si64>, !torch.list<int> -> !torch.vtensor<[?],si64>
    torch.bind_symbolic_shape %180, [%31], affine_map<()[s0] -> (s0)> : !torch.vtensor<[?],si64>
    %int0_190 = torch.constant.int 0
    %181 = torch.aten.index_select %168, %int0_190, %180 : !torch.vtensor<[?,16,4,32],f16>, !torch.int, !torch.vtensor<[?],si64> -> !torch.vtensor<[?,16,4,32],f16>
    torch.bind_symbolic_shape %181, [%31], affine_map<()[s0] -> (s0, 16, 4, 32)> : !torch.vtensor<[?,16,4,32],f16>
    %int1_191 = torch.constant.int 1
    %int16_192 = torch.constant.int 16
    %int4_193 = torch.constant.int 4
    %int32_194 = torch.constant.int 32
    %182 = torch.prim.ListConstruct %int1_191, %33, %int16_192, %int4_193, %int32_194 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %183 = torch.aten.view %181, %182 : !torch.vtensor<[?,16,4,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,16,4,32],f16>
    torch.bind_symbolic_shape %183, [%31], affine_map<()[s0] -> (1, s0, 16, 4, 32)> : !torch.vtensor<[1,?,16,4,32],f16>
    %int1_195 = torch.constant.int 1
    %int4_196 = torch.constant.int 4
    %int32_197 = torch.constant.int 32
    %184 = torch.prim.ListConstruct %int1_195, %35, %int4_196, %int32_197 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %185 = torch.aten.view %183, %184 : !torch.vtensor<[1,?,16,4,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %185, [%31], affine_map<()[s0] -> (1, s0 * 16, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %false_198 = torch.constant.bool false
    %186 = torch.aten.copy %158, %185, %false_198 : !torch.vtensor<[1,?,4,32],f16>, !torch.vtensor<[1,?,4,32],f16>, !torch.bool -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %186, [%31], affine_map<()[s0] -> (1, s0 * 16, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int0_199 = torch.constant.int 0
    %int0_200 = torch.constant.int 0
    %int9223372036854775807_201 = torch.constant.int 9223372036854775807
    %int1_202 = torch.constant.int 1
    %187 = torch.aten.slice.Tensor %71, %int0_199, %int0_200, %int9223372036854775807_201, %int1_202 : !torch.vtensor<[1,128,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,128,4,32],f16>
    %int1_203 = torch.constant.int 1
    %int0_204 = torch.constant.int 0
    %int1_205 = torch.constant.int 1
    %188 = torch.aten.slice_scatter %187, %186, %int1_203, %int0_204, %35, %int1_205 : !torch.vtensor<[1,128,4,32],f16>, !torch.vtensor<[1,?,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,128,4,32],f16>
    %int0_206 = torch.constant.int 0
    %int0_207 = torch.constant.int 0
    %int9223372036854775807_208 = torch.constant.int 9223372036854775807
    %int1_209 = torch.constant.int 1
    %189 = torch.aten.slice_scatter %71, %188, %int0_206, %int0_207, %int9223372036854775807_208, %int1_209 : !torch.vtensor<[1,128,4,32],f16>, !torch.vtensor<[1,128,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,128,4,32],f16>
    %int0_210 = torch.constant.int 0
    %int0_211 = torch.constant.int 0
    %int9223372036854775807_212 = torch.constant.int 9223372036854775807
    %int1_213 = torch.constant.int 1
    %190 = torch.aten.slice.Tensor %177, %int0_210, %int0_211, %int9223372036854775807_212, %int1_213 : !torch.vtensor<[1,128,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,128,4,32],f16>
    %int1_214 = torch.constant.int 1
    %int0_215 = torch.constant.int 0
    %int1_216 = torch.constant.int 1
    %191 = torch.aten.slice.Tensor %190, %int1_214, %int0_215, %35, %int1_216 : !torch.vtensor<[1,128,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %191, [%31], affine_map<()[s0] -> (1, s0 * 16, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int0_217 = torch.constant.int 0
    %int0_218 = torch.constant.int 0
    %int9223372036854775807_219 = torch.constant.int 9223372036854775807
    %int1_220 = torch.constant.int 1
    %192 = torch.aten.slice.Tensor %191, %int0_217, %int0_218, %int9223372036854775807_219, %int1_220 : !torch.vtensor<[1,?,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %192, [%31], affine_map<()[s0] -> (1, s0 * 16, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int2_221 = torch.constant.int 2
    %int0_222 = torch.constant.int 0
    %int9223372036854775807_223 = torch.constant.int 9223372036854775807
    %int1_224 = torch.constant.int 1
    %193 = torch.aten.slice.Tensor %192, %int2_221, %int0_222, %int9223372036854775807_223, %int1_224 : !torch.vtensor<[1,?,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %193, [%31], affine_map<()[s0] -> (1, s0 * 16, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int-2_225 = torch.constant.int -2
    %194 = torch.aten.unsqueeze %193, %int-2_225 : !torch.vtensor<[1,?,4,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,1,32],f16>
    torch.bind_symbolic_shape %194, [%31], affine_map<()[s0] -> (1, s0 * 16, 4, 1, 32)> : !torch.vtensor<[1,?,4,1,32],f16>
    %int1_226 = torch.constant.int 1
    %int4_227 = torch.constant.int 4
    %int2_228 = torch.constant.int 2
    %int32_229 = torch.constant.int 32
    %195 = torch.prim.ListConstruct %int1_226, %35, %int4_227, %int2_228, %int32_229 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_230 = torch.constant.bool false
    %196 = torch.aten.expand %194, %195, %false_230 : !torch.vtensor<[1,?,4,1,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,?,4,2,32],f16>
    torch.bind_symbolic_shape %196, [%31], affine_map<()[s0] -> (1, s0 * 16, 4, 2, 32)> : !torch.vtensor<[1,?,4,2,32],f16>
    %int0_231 = torch.constant.int 0
    %197 = torch.aten.clone %196, %int0_231 : !torch.vtensor<[1,?,4,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,2,32],f16>
    torch.bind_symbolic_shape %197, [%31], affine_map<()[s0] -> (1, s0 * 16, 4, 2, 32)> : !torch.vtensor<[1,?,4,2,32],f16>
    %int1_232 = torch.constant.int 1
    %int8_233 = torch.constant.int 8
    %int32_234 = torch.constant.int 32
    %198 = torch.prim.ListConstruct %int1_232, %35, %int8_233, %int32_234 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %199 = torch.aten._unsafe_view %197, %198 : !torch.vtensor<[1,?,4,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,8,32],f16>
    torch.bind_symbolic_shape %199, [%31], affine_map<()[s0] -> (1, s0 * 16, 8, 32)> : !torch.vtensor<[1,?,8,32],f16>
    %int0_235 = torch.constant.int 0
    %int0_236 = torch.constant.int 0
    %int9223372036854775807_237 = torch.constant.int 9223372036854775807
    %int1_238 = torch.constant.int 1
    %200 = torch.aten.slice.Tensor %189, %int0_235, %int0_236, %int9223372036854775807_237, %int1_238 : !torch.vtensor<[1,128,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,128,4,32],f16>
    %int1_239 = torch.constant.int 1
    %int0_240 = torch.constant.int 0
    %int1_241 = torch.constant.int 1
    %201 = torch.aten.slice.Tensor %200, %int1_239, %int0_240, %35, %int1_241 : !torch.vtensor<[1,128,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %201, [%31], affine_map<()[s0] -> (1, s0 * 16, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int0_242 = torch.constant.int 0
    %int0_243 = torch.constant.int 0
    %int9223372036854775807_244 = torch.constant.int 9223372036854775807
    %int1_245 = torch.constant.int 1
    %202 = torch.aten.slice.Tensor %201, %int0_242, %int0_243, %int9223372036854775807_244, %int1_245 : !torch.vtensor<[1,?,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %202, [%31], affine_map<()[s0] -> (1, s0 * 16, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int2_246 = torch.constant.int 2
    %int0_247 = torch.constant.int 0
    %int9223372036854775807_248 = torch.constant.int 9223372036854775807
    %int1_249 = torch.constant.int 1
    %203 = torch.aten.slice.Tensor %202, %int2_246, %int0_247, %int9223372036854775807_248, %int1_249 : !torch.vtensor<[1,?,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %203, [%31], affine_map<()[s0] -> (1, s0 * 16, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int-2_250 = torch.constant.int -2
    %204 = torch.aten.unsqueeze %203, %int-2_250 : !torch.vtensor<[1,?,4,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,1,32],f16>
    torch.bind_symbolic_shape %204, [%31], affine_map<()[s0] -> (1, s0 * 16, 4, 1, 32)> : !torch.vtensor<[1,?,4,1,32],f16>
    %int1_251 = torch.constant.int 1
    %int4_252 = torch.constant.int 4
    %int2_253 = torch.constant.int 2
    %int32_254 = torch.constant.int 32
    %205 = torch.prim.ListConstruct %int1_251, %35, %int4_252, %int2_253, %int32_254 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_255 = torch.constant.bool false
    %206 = torch.aten.expand %204, %205, %false_255 : !torch.vtensor<[1,?,4,1,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,?,4,2,32],f16>
    torch.bind_symbolic_shape %206, [%31], affine_map<()[s0] -> (1, s0 * 16, 4, 2, 32)> : !torch.vtensor<[1,?,4,2,32],f16>
    %int0_256 = torch.constant.int 0
    %207 = torch.aten.clone %206, %int0_256 : !torch.vtensor<[1,?,4,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,2,32],f16>
    torch.bind_symbolic_shape %207, [%31], affine_map<()[s0] -> (1, s0 * 16, 4, 2, 32)> : !torch.vtensor<[1,?,4,2,32],f16>
    %int1_257 = torch.constant.int 1
    %int8_258 = torch.constant.int 8
    %int32_259 = torch.constant.int 32
    %208 = torch.prim.ListConstruct %int1_257, %35, %int8_258, %int32_259 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %209 = torch.aten._unsafe_view %207, %208 : !torch.vtensor<[1,?,4,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,8,32],f16>
    torch.bind_symbolic_shape %209, [%31], affine_map<()[s0] -> (1, s0 * 16, 8, 32)> : !torch.vtensor<[1,?,8,32],f16>
    %int1_260 = torch.constant.int 1
    %int2_261 = torch.constant.int 2
    %210 = torch.aten.transpose.int %121, %int1_260, %int2_261 : !torch.vtensor<[1,1,8,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,8,1,32],f16>
    %int1_262 = torch.constant.int 1
    %int2_263 = torch.constant.int 2
    %211 = torch.aten.transpose.int %199, %int1_262, %int2_263 : !torch.vtensor<[1,?,8,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,8,?,32],f16>
    torch.bind_symbolic_shape %211, [%31], affine_map<()[s0] -> (1, 8, s0 * 16, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %int1_264 = torch.constant.int 1
    %int2_265 = torch.constant.int 2
    %212 = torch.aten.transpose.int %209, %int1_264, %int2_265 : !torch.vtensor<[1,?,8,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,8,?,32],f16>
    torch.bind_symbolic_shape %212, [%31], affine_map<()[s0] -> (1, 8, s0 * 16, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %int2_266 = torch.constant.int 2
    %int3_267 = torch.constant.int 3
    %213 = torch.aten.transpose.int %211, %int2_266, %int3_267 : !torch.vtensor<[1,8,?,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,8,32,?],f16>
    torch.bind_symbolic_shape %213, [%31], affine_map<()[s0] -> (1, 8, 32, s0 * 16)> : !torch.vtensor<[1,8,32,?],f16>
    %int5_268 = torch.constant.int 5
    %214 = torch.prims.convert_element_type %213, %int5_268 : !torch.vtensor<[1,8,32,?],f16>, !torch.int -> !torch.vtensor<[1,8,32,?],f16>
    torch.bind_symbolic_shape %214, [%31], affine_map<()[s0] -> (1, 8, 32, s0 * 16)> : !torch.vtensor<[1,8,32,?],f16>
    %int1_269 = torch.constant.int 1
    %int8_270 = torch.constant.int 8
    %int1_271 = torch.constant.int 1
    %int32_272 = torch.constant.int 32
    %215 = torch.prim.ListConstruct %int1_269, %int8_270, %int1_271, %int32_272 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_273 = torch.constant.bool false
    %216 = torch.aten.expand %210, %215, %false_273 : !torch.vtensor<[1,8,1,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,8,1,32],f16>
    %int8_274 = torch.constant.int 8
    %int1_275 = torch.constant.int 1
    %int32_276 = torch.constant.int 32
    %217 = torch.prim.ListConstruct %int8_274, %int1_275, %int32_276 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %218 = torch.aten.view %216, %217 : !torch.vtensor<[1,8,1,32],f16>, !torch.list<int> -> !torch.vtensor<[8,1,32],f16>
    %int1_277 = torch.constant.int 1
    %int8_278 = torch.constant.int 8
    %int32_279 = torch.constant.int 32
    %219 = torch.prim.ListConstruct %int1_277, %int8_278, %int32_279, %35 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_280 = torch.constant.bool false
    %220 = torch.aten.expand %214, %219, %false_280 : !torch.vtensor<[1,8,32,?],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,8,32,?],f16>
    torch.bind_symbolic_shape %220, [%31], affine_map<()[s0] -> (1, 8, 32, s0 * 16)> : !torch.vtensor<[1,8,32,?],f16>
    %int8_281 = torch.constant.int 8
    %int32_282 = torch.constant.int 32
    %221 = torch.prim.ListConstruct %int8_281, %int32_282, %35 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %222 = torch.aten.view %220, %221 : !torch.vtensor<[1,8,32,?],f16>, !torch.list<int> -> !torch.vtensor<[8,32,?],f16>
    torch.bind_symbolic_shape %222, [%31], affine_map<()[s0] -> (8, 32, s0 * 16)> : !torch.vtensor<[8,32,?],f16>
    %223 = torch.aten.bmm %218, %222 : !torch.vtensor<[8,1,32],f16>, !torch.vtensor<[8,32,?],f16> -> !torch.vtensor<[8,1,?],f16>
    torch.bind_symbolic_shape %223, [%31], affine_map<()[s0] -> (8, 1, s0 * 16)> : !torch.vtensor<[8,1,?],f16>
    %int1_283 = torch.constant.int 1
    %int8_284 = torch.constant.int 8
    %int1_285 = torch.constant.int 1
    %224 = torch.prim.ListConstruct %int1_283, %int8_284, %int1_285, %35 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %225 = torch.aten.view %223, %224 : !torch.vtensor<[8,1,?],f16>, !torch.list<int> -> !torch.vtensor<[1,8,1,?],f16>
    torch.bind_symbolic_shape %225, [%31], affine_map<()[s0] -> (1, 8, 1, s0 * 16)> : !torch.vtensor<[1,8,1,?],f16>
    %float5.656850e00 = torch.constant.float 5.6568542494923806
    %226 = torch.aten.div.Scalar %225, %float5.656850e00 : !torch.vtensor<[1,8,1,?],f16>, !torch.float -> !torch.vtensor<[1,8,1,?],f16>
    torch.bind_symbolic_shape %226, [%31], affine_map<()[s0] -> (1, 8, 1, s0 * 16)> : !torch.vtensor<[1,8,1,?],f16>
    %int1_286 = torch.constant.int 1
    %227 = torch.aten.add.Tensor %226, %45, %int1_286 : !torch.vtensor<[1,8,1,?],f16>, !torch.vtensor<[1,1,1,?],f16>, !torch.int -> !torch.vtensor<[1,8,1,?],f16>
    torch.bind_symbolic_shape %227, [%31], affine_map<()[s0] -> (1, 8, 1, s0 * 16)> : !torch.vtensor<[1,8,1,?],f16>
    %int6_287 = torch.constant.int 6
    %228 = torch.prims.convert_element_type %227, %int6_287 : !torch.vtensor<[1,8,1,?],f16>, !torch.int -> !torch.vtensor<[1,8,1,?],f32>
    torch.bind_symbolic_shape %228, [%31], affine_map<()[s0] -> (1, 8, 1, s0 * 16)> : !torch.vtensor<[1,8,1,?],f32>
    %int-1_288 = torch.constant.int -1
    %false_289 = torch.constant.bool false
    %229 = torch.aten._softmax %228, %int-1_288, %false_289 : !torch.vtensor<[1,8,1,?],f32>, !torch.int, !torch.bool -> !torch.vtensor<[1,8,1,?],f32>
    torch.bind_symbolic_shape %229, [%31], affine_map<()[s0] -> (1, 8, 1, s0 * 16)> : !torch.vtensor<[1,8,1,?],f32>
    %int5_290 = torch.constant.int 5
    %230 = torch.prims.convert_element_type %229, %int5_290 : !torch.vtensor<[1,8,1,?],f32>, !torch.int -> !torch.vtensor<[1,8,1,?],f16>
    torch.bind_symbolic_shape %230, [%31], affine_map<()[s0] -> (1, 8, 1, s0 * 16)> : !torch.vtensor<[1,8,1,?],f16>
    %int5_291 = torch.constant.int 5
    %231 = torch.prims.convert_element_type %212, %int5_291 : !torch.vtensor<[1,8,?,32],f16>, !torch.int -> !torch.vtensor<[1,8,?,32],f16>
    torch.bind_symbolic_shape %231, [%31], affine_map<()[s0] -> (1, 8, s0 * 16, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %int1_292 = torch.constant.int 1
    %int8_293 = torch.constant.int 8
    %int1_294 = torch.constant.int 1
    %232 = torch.prim.ListConstruct %int1_292, %int8_293, %int1_294, %35 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_295 = torch.constant.bool false
    %233 = torch.aten.expand %230, %232, %false_295 : !torch.vtensor<[1,8,1,?],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,8,1,?],f16>
    torch.bind_symbolic_shape %233, [%31], affine_map<()[s0] -> (1, 8, 1, s0 * 16)> : !torch.vtensor<[1,8,1,?],f16>
    %int8_296 = torch.constant.int 8
    %int1_297 = torch.constant.int 1
    %234 = torch.prim.ListConstruct %int8_296, %int1_297, %35 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %235 = torch.aten.view %233, %234 : !torch.vtensor<[1,8,1,?],f16>, !torch.list<int> -> !torch.vtensor<[8,1,?],f16>
    torch.bind_symbolic_shape %235, [%31], affine_map<()[s0] -> (8, 1, s0 * 16)> : !torch.vtensor<[8,1,?],f16>
    %int1_298 = torch.constant.int 1
    %int8_299 = torch.constant.int 8
    %int32_300 = torch.constant.int 32
    %236 = torch.prim.ListConstruct %int1_298, %int8_299, %35, %int32_300 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_301 = torch.constant.bool false
    %237 = torch.aten.expand %231, %236, %false_301 : !torch.vtensor<[1,8,?,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,8,?,32],f16>
    torch.bind_symbolic_shape %237, [%31], affine_map<()[s0] -> (1, 8, s0 * 16, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %int8_302 = torch.constant.int 8
    %int32_303 = torch.constant.int 32
    %238 = torch.prim.ListConstruct %int8_302, %35, %int32_303 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %239 = torch.aten.view %237, %238 : !torch.vtensor<[1,8,?,32],f16>, !torch.list<int> -> !torch.vtensor<[8,?,32],f16>
    torch.bind_symbolic_shape %239, [%31], affine_map<()[s0] -> (8, s0 * 16, 32)> : !torch.vtensor<[8,?,32],f16>
    %240 = torch.aten.bmm %235, %239 : !torch.vtensor<[8,1,?],f16>, !torch.vtensor<[8,?,32],f16> -> !torch.vtensor<[8,1,32],f16>
    %int1_304 = torch.constant.int 1
    %int8_305 = torch.constant.int 8
    %int1_306 = torch.constant.int 1
    %int32_307 = torch.constant.int 32
    %241 = torch.prim.ListConstruct %int1_304, %int8_305, %int1_306, %int32_307 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %242 = torch.aten.view %240, %241 : !torch.vtensor<[8,1,32],f16>, !torch.list<int> -> !torch.vtensor<[1,8,1,32],f16>
    %int1_308 = torch.constant.int 1
    %int2_309 = torch.constant.int 2
    %243 = torch.aten.transpose.int %242, %int1_308, %int2_309 : !torch.vtensor<[1,8,1,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,1,8,32],f16>
    %int1_310 = torch.constant.int 1
    %int1_311 = torch.constant.int 1
    %int256_312 = torch.constant.int 256
    %244 = torch.prim.ListConstruct %int1_310, %int1_311, %int256_312 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %245 = torch.aten.view %243, %244 : !torch.vtensor<[1,1,8,32],f16>, !torch.list<int> -> !torch.vtensor<[1,1,256],f16>
    %int5_313 = torch.constant.int 5
    %246 = torch.prims.convert_element_type %5, %int5_313 : !torch.vtensor<[256,256],f32>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int-2_314 = torch.constant.int -2
    %int-1_315 = torch.constant.int -1
    %247 = torch.aten.transpose.int %246, %int-2_314, %int-1_315 : !torch.vtensor<[256,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,256],f16>
    %int5_316 = torch.constant.int 5
    %248 = torch.prims.convert_element_type %247, %int5_316 : !torch.vtensor<[256,256],f16>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int1_317 = torch.constant.int 1
    %int256_318 = torch.constant.int 256
    %249 = torch.prim.ListConstruct %int1_317, %int256_318 : (!torch.int, !torch.int) -> !torch.list<int>
    %250 = torch.aten.view %245, %249 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %251 = torch.aten.mm %250, %248 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,256],f16> -> !torch.vtensor<[1,256],f16>
    %int1_319 = torch.constant.int 1
    %int1_320 = torch.constant.int 1
    %int256_321 = torch.constant.int 256
    %252 = torch.prim.ListConstruct %int1_319, %int1_320, %int256_321 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %253 = torch.aten.view %251, %252 : !torch.vtensor<[1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,1,256],f16>
    %int1_322 = torch.constant.int 1
    %254 = torch.aten.add.Tensor %73, %253, %int1_322 : !torch.vtensor<[1,1,256],f16>, !torch.vtensor<[1,1,256],f16>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %int6_323 = torch.constant.int 6
    %255 = torch.prims.convert_element_type %254, %int6_323 : !torch.vtensor<[1,1,256],f16>, !torch.int -> !torch.vtensor<[1,1,256],f32>
    %int2_324 = torch.constant.int 2
    %256 = torch.aten.pow.Tensor_Scalar %255, %int2_324 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f32>
    %int-1_325 = torch.constant.int -1
    %257 = torch.prim.ListConstruct %int-1_325 : (!torch.int) -> !torch.list<int>
    %true_326 = torch.constant.bool true
    %none_327 = torch.constant.none
    %258 = torch.aten.mean.dim %256, %257, %true_326, %none_327 : !torch.vtensor<[1,1,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1],f32>
    %float1.000000e-02_328 = torch.constant.float 1.000000e-02
    %int1_329 = torch.constant.int 1
    %259 = torch.aten.add.Scalar %258, %float1.000000e-02_328, %int1_329 : !torch.vtensor<[1,1,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,1,1],f32>
    %260 = torch.aten.rsqrt %259 : !torch.vtensor<[1,1,1],f32> -> !torch.vtensor<[1,1,1],f32>
    %261 = torch.aten.mul.Tensor %255, %260 : !torch.vtensor<[1,1,256],f32>, !torch.vtensor<[1,1,1],f32> -> !torch.vtensor<[1,1,256],f32>
    %int6_330 = torch.constant.int 6
    %262 = torch.prims.convert_element_type %261, %int6_330 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f32>
    %263 = torch.aten.mul.Tensor %6, %262 : !torch.vtensor<[256],f32>, !torch.vtensor<[1,1,256],f32> -> !torch.vtensor<[1,1,256],f32>
    %int5_331 = torch.constant.int 5
    %264 = torch.prims.convert_element_type %263, %int5_331 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %int5_332 = torch.constant.int 5
    %265 = torch.prims.convert_element_type %7, %int5_332 : !torch.vtensor<[23,256],f32>, !torch.int -> !torch.vtensor<[23,256],f16>
    %int-2_333 = torch.constant.int -2
    %int-1_334 = torch.constant.int -1
    %266 = torch.aten.transpose.int %265, %int-2_333, %int-1_334 : !torch.vtensor<[23,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,23],f16>
    %int5_335 = torch.constant.int 5
    %267 = torch.prims.convert_element_type %266, %int5_335 : !torch.vtensor<[256,23],f16>, !torch.int -> !torch.vtensor<[256,23],f16>
    %int1_336 = torch.constant.int 1
    %int256_337 = torch.constant.int 256
    %268 = torch.prim.ListConstruct %int1_336, %int256_337 : (!torch.int, !torch.int) -> !torch.list<int>
    %269 = torch.aten.view %264, %268 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %270 = torch.aten.mm %269, %267 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,23],f16> -> !torch.vtensor<[1,23],f16>
    %int1_338 = torch.constant.int 1
    %int1_339 = torch.constant.int 1
    %int23 = torch.constant.int 23
    %271 = torch.prim.ListConstruct %int1_338, %int1_339, %int23 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %272 = torch.aten.view %270, %271 : !torch.vtensor<[1,23],f16>, !torch.list<int> -> !torch.vtensor<[1,1,23],f16>
    %273 = torch.aten.silu %272 : !torch.vtensor<[1,1,23],f16> -> !torch.vtensor<[1,1,23],f16>
    %int5_340 = torch.constant.int 5
    %274 = torch.prims.convert_element_type %8, %int5_340 : !torch.vtensor<[23,256],f32>, !torch.int -> !torch.vtensor<[23,256],f16>
    %int-2_341 = torch.constant.int -2
    %int-1_342 = torch.constant.int -1
    %275 = torch.aten.transpose.int %274, %int-2_341, %int-1_342 : !torch.vtensor<[23,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,23],f16>
    %int5_343 = torch.constant.int 5
    %276 = torch.prims.convert_element_type %275, %int5_343 : !torch.vtensor<[256,23],f16>, !torch.int -> !torch.vtensor<[256,23],f16>
    %int1_344 = torch.constant.int 1
    %int256_345 = torch.constant.int 256
    %277 = torch.prim.ListConstruct %int1_344, %int256_345 : (!torch.int, !torch.int) -> !torch.list<int>
    %278 = torch.aten.view %264, %277 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %279 = torch.aten.mm %278, %276 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,23],f16> -> !torch.vtensor<[1,23],f16>
    %int1_346 = torch.constant.int 1
    %int1_347 = torch.constant.int 1
    %int23_348 = torch.constant.int 23
    %280 = torch.prim.ListConstruct %int1_346, %int1_347, %int23_348 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %281 = torch.aten.view %279, %280 : !torch.vtensor<[1,23],f16>, !torch.list<int> -> !torch.vtensor<[1,1,23],f16>
    %282 = torch.aten.mul.Tensor %273, %281 : !torch.vtensor<[1,1,23],f16>, !torch.vtensor<[1,1,23],f16> -> !torch.vtensor<[1,1,23],f16>
    %int5_349 = torch.constant.int 5
    %283 = torch.prims.convert_element_type %9, %int5_349 : !torch.vtensor<[256,23],f32>, !torch.int -> !torch.vtensor<[256,23],f16>
    %int-2_350 = torch.constant.int -2
    %int-1_351 = torch.constant.int -1
    %284 = torch.aten.transpose.int %283, %int-2_350, %int-1_351 : !torch.vtensor<[256,23],f16>, !torch.int, !torch.int -> !torch.vtensor<[23,256],f16>
    %int5_352 = torch.constant.int 5
    %285 = torch.prims.convert_element_type %284, %int5_352 : !torch.vtensor<[23,256],f16>, !torch.int -> !torch.vtensor<[23,256],f16>
    %int1_353 = torch.constant.int 1
    %int23_354 = torch.constant.int 23
    %286 = torch.prim.ListConstruct %int1_353, %int23_354 : (!torch.int, !torch.int) -> !torch.list<int>
    %287 = torch.aten.view %282, %286 : !torch.vtensor<[1,1,23],f16>, !torch.list<int> -> !torch.vtensor<[1,23],f16>
    %288 = torch.aten.mm %287, %285 : !torch.vtensor<[1,23],f16>, !torch.vtensor<[23,256],f16> -> !torch.vtensor<[1,256],f16>
    %int1_355 = torch.constant.int 1
    %int1_356 = torch.constant.int 1
    %int256_357 = torch.constant.int 256
    %289 = torch.prim.ListConstruct %int1_355, %int1_356, %int256_357 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %290 = torch.aten.view %288, %289 : !torch.vtensor<[1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,1,256],f16>
    %int1_358 = torch.constant.int 1
    %291 = torch.aten.add.Tensor %254, %290, %int1_358 : !torch.vtensor<[1,1,256],f16>, !torch.vtensor<[1,1,256],f16>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %int6_359 = torch.constant.int 6
    %292 = torch.prims.convert_element_type %291, %int6_359 : !torch.vtensor<[1,1,256],f16>, !torch.int -> !torch.vtensor<[1,1,256],f32>
    %int2_360 = torch.constant.int 2
    %293 = torch.aten.pow.Tensor_Scalar %292, %int2_360 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f32>
    %int-1_361 = torch.constant.int -1
    %294 = torch.prim.ListConstruct %int-1_361 : (!torch.int) -> !torch.list<int>
    %true_362 = torch.constant.bool true
    %none_363 = torch.constant.none
    %295 = torch.aten.mean.dim %293, %294, %true_362, %none_363 : !torch.vtensor<[1,1,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1],f32>
    %float1.000000e-02_364 = torch.constant.float 1.000000e-02
    %int1_365 = torch.constant.int 1
    %296 = torch.aten.add.Scalar %295, %float1.000000e-02_364, %int1_365 : !torch.vtensor<[1,1,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,1,1],f32>
    %297 = torch.aten.rsqrt %296 : !torch.vtensor<[1,1,1],f32> -> !torch.vtensor<[1,1,1],f32>
    %298 = torch.aten.mul.Tensor %292, %297 : !torch.vtensor<[1,1,256],f32>, !torch.vtensor<[1,1,1],f32> -> !torch.vtensor<[1,1,256],f32>
    %int6_366 = torch.constant.int 6
    %299 = torch.prims.convert_element_type %298, %int6_366 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f32>
    %300 = torch.aten.mul.Tensor %10, %299 : !torch.vtensor<[256],f32>, !torch.vtensor<[1,1,256],f32> -> !torch.vtensor<[1,1,256],f32>
    %int5_367 = torch.constant.int 5
    %301 = torch.prims.convert_element_type %300, %int5_367 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %int5_368 = torch.constant.int 5
    %302 = torch.prims.convert_element_type %11, %int5_368 : !torch.vtensor<[256,256],f32>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int-2_369 = torch.constant.int -2
    %int-1_370 = torch.constant.int -1
    %303 = torch.aten.transpose.int %302, %int-2_369, %int-1_370 : !torch.vtensor<[256,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,256],f16>
    %int5_371 = torch.constant.int 5
    %304 = torch.prims.convert_element_type %303, %int5_371 : !torch.vtensor<[256,256],f16>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int1_372 = torch.constant.int 1
    %int256_373 = torch.constant.int 256
    %305 = torch.prim.ListConstruct %int1_372, %int256_373 : (!torch.int, !torch.int) -> !torch.list<int>
    %306 = torch.aten.view %301, %305 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %307 = torch.aten.mm %306, %304 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,256],f16> -> !torch.vtensor<[1,256],f16>
    %int1_374 = torch.constant.int 1
    %int1_375 = torch.constant.int 1
    %int256_376 = torch.constant.int 256
    %308 = torch.prim.ListConstruct %int1_374, %int1_375, %int256_376 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %309 = torch.aten.view %307, %308 : !torch.vtensor<[1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,1,256],f16>
    %int5_377 = torch.constant.int 5
    %310 = torch.prims.convert_element_type %12, %int5_377 : !torch.vtensor<[128,256],f32>, !torch.int -> !torch.vtensor<[128,256],f16>
    %int-2_378 = torch.constant.int -2
    %int-1_379 = torch.constant.int -1
    %311 = torch.aten.transpose.int %310, %int-2_378, %int-1_379 : !torch.vtensor<[128,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,128],f16>
    %int5_380 = torch.constant.int 5
    %312 = torch.prims.convert_element_type %311, %int5_380 : !torch.vtensor<[256,128],f16>, !torch.int -> !torch.vtensor<[256,128],f16>
    %int1_381 = torch.constant.int 1
    %int256_382 = torch.constant.int 256
    %313 = torch.prim.ListConstruct %int1_381, %int256_382 : (!torch.int, !torch.int) -> !torch.list<int>
    %314 = torch.aten.view %301, %313 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %315 = torch.aten.mm %314, %312 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,128],f16> -> !torch.vtensor<[1,128],f16>
    %int1_383 = torch.constant.int 1
    %int1_384 = torch.constant.int 1
    %int128_385 = torch.constant.int 128
    %316 = torch.prim.ListConstruct %int1_383, %int1_384, %int128_385 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %317 = torch.aten.view %315, %316 : !torch.vtensor<[1,128],f16>, !torch.list<int> -> !torch.vtensor<[1,1,128],f16>
    %int5_386 = torch.constant.int 5
    %318 = torch.prims.convert_element_type %13, %int5_386 : !torch.vtensor<[128,256],f32>, !torch.int -> !torch.vtensor<[128,256],f16>
    %int-2_387 = torch.constant.int -2
    %int-1_388 = torch.constant.int -1
    %319 = torch.aten.transpose.int %318, %int-2_387, %int-1_388 : !torch.vtensor<[128,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,128],f16>
    %int5_389 = torch.constant.int 5
    %320 = torch.prims.convert_element_type %319, %int5_389 : !torch.vtensor<[256,128],f16>, !torch.int -> !torch.vtensor<[256,128],f16>
    %int1_390 = torch.constant.int 1
    %int256_391 = torch.constant.int 256
    %321 = torch.prim.ListConstruct %int1_390, %int256_391 : (!torch.int, !torch.int) -> !torch.list<int>
    %322 = torch.aten.view %301, %321 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %323 = torch.aten.mm %322, %320 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,128],f16> -> !torch.vtensor<[1,128],f16>
    %int1_392 = torch.constant.int 1
    %int1_393 = torch.constant.int 1
    %int128_394 = torch.constant.int 128
    %324 = torch.prim.ListConstruct %int1_392, %int1_393, %int128_394 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %325 = torch.aten.view %323, %324 : !torch.vtensor<[1,128],f16>, !torch.list<int> -> !torch.vtensor<[1,1,128],f16>
    %int1_395 = torch.constant.int 1
    %int1_396 = torch.constant.int 1
    %int8_397 = torch.constant.int 8
    %int32_398 = torch.constant.int 32
    %326 = torch.prim.ListConstruct %int1_395, %int1_396, %int8_397, %int32_398 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %327 = torch.aten.view %309, %326 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,1,8,32],f16>
    %int1_399 = torch.constant.int 1
    %int1_400 = torch.constant.int 1
    %int4_401 = torch.constant.int 4
    %int32_402 = torch.constant.int 32
    %328 = torch.prim.ListConstruct %int1_399, %int1_400, %int4_401, %int32_402 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %329 = torch.aten.view %317, %328 : !torch.vtensor<[1,1,128],f16>, !torch.list<int> -> !torch.vtensor<[1,1,4,32],f16>
    %int1_403 = torch.constant.int 1
    %int1_404 = torch.constant.int 1
    %int4_405 = torch.constant.int 4
    %int32_406 = torch.constant.int 32
    %330 = torch.prim.ListConstruct %int1_403, %int1_404, %int4_405, %int32_406 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %331 = torch.aten.view %325, %330 : !torch.vtensor<[1,1,128],f16>, !torch.list<int> -> !torch.vtensor<[1,1,4,32],f16>
    %332 = torch_c.to_builtin_tensor %327 : !torch.vtensor<[1,1,8,32],f16> -> tensor<1x1x8x32xf16>
    %333 = flow.tensor.bitcast %332 : tensor<1x1x8x32xf16> -> tensor<1x1x8x16xcomplex<f16>>
    %334 = torch_c.from_builtin_tensor %333 : tensor<1x1x8x16xcomplex<f16>> -> !torch.vtensor<[1,1,8,16],complex<f16>>
    %335 = torch.aten.mul.Tensor %334, %67 : !torch.vtensor<[1,1,8,16],complex<f16>>, !torch.vtensor<[1,1,1,16],complex<f32>> -> !torch.vtensor<[1,1,8,16],complex<f32>>
    %336 = torch_c.to_builtin_tensor %335 : !torch.vtensor<[1,1,8,16],complex<f32>> -> tensor<1x1x8x16xcomplex<f32>>
    %337 = flow.tensor.bitcast %336 : tensor<1x1x8x16xcomplex<f32>> -> tensor<1x1x8x32xf32>
    %338 = torch_c.from_builtin_tensor %337 : tensor<1x1x8x32xf32> -> !torch.vtensor<[1,1,8,32],f32>
    %int5_407 = torch.constant.int 5
    %339 = torch.prims.convert_element_type %338, %int5_407 : !torch.vtensor<[1,1,8,32],f32>, !torch.int -> !torch.vtensor<[1,1,8,32],f16>
    %340 = torch_c.to_builtin_tensor %329 : !torch.vtensor<[1,1,4,32],f16> -> tensor<1x1x4x32xf16>
    %341 = flow.tensor.bitcast %340 : tensor<1x1x4x32xf16> -> tensor<1x1x4x16xcomplex<f16>>
    %342 = torch_c.from_builtin_tensor %341 : tensor<1x1x4x16xcomplex<f16>> -> !torch.vtensor<[1,1,4,16],complex<f16>>
    %343 = torch.aten.mul.Tensor %342, %67 : !torch.vtensor<[1,1,4,16],complex<f16>>, !torch.vtensor<[1,1,1,16],complex<f32>> -> !torch.vtensor<[1,1,4,16],complex<f32>>
    %344 = torch_c.to_builtin_tensor %343 : !torch.vtensor<[1,1,4,16],complex<f32>> -> tensor<1x1x4x16xcomplex<f32>>
    %345 = flow.tensor.bitcast %344 : tensor<1x1x4x16xcomplex<f32>> -> tensor<1x1x4x32xf32>
    %346 = torch_c.from_builtin_tensor %345 : tensor<1x1x4x32xf32> -> !torch.vtensor<[1,1,4,32],f32>
    %int5_408 = torch.constant.int 5
    %347 = torch.prims.convert_element_type %346, %int5_408 : !torch.vtensor<[1,1,4,32],f32>, !torch.int -> !torch.vtensor<[1,1,4,32],f16>
    %348 = torch.prim.ListConstruct %347, %331 : (!torch.vtensor<[1,1,4,32],f16>, !torch.vtensor<[1,1,4,32],f16>) -> !torch.list<vtensor>
    %int1_409 = torch.constant.int 1
    %349 = torch.aten.cat %348, %int1_409 : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[1,2,4,32],f16>
    %int16_410 = torch.constant.int 16
    %350 = torch.aten.floor_divide.Scalar %arg2, %int16_410 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int1_411 = torch.constant.int 1
    %351 = torch.aten.unsqueeze %350, %int1_411 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_412 = torch.constant.int 1
    %false_413 = torch.constant.bool false
    %352 = torch.aten.gather %arg3, %int1_412, %351, %false_413 : !torch.vtensor<[1,?],si64>, !torch.int, !torch.vtensor<[1,1],si64>, !torch.bool -> !torch.vtensor<[1,1],si64>
    %int16_414 = torch.constant.int 16
    %353 = torch.aten.remainder.Scalar %arg2, %int16_414 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int1_415 = torch.constant.int 1
    %354 = torch.aten.unsqueeze %353, %int1_415 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int0_416 = torch.constant.int 0
    %int2_417 = torch.constant.int 2
    %none_418 = torch.constant.none
    %none_419 = torch.constant.none
    %cpu_420 = torch.constant.device "cpu"
    %false_421 = torch.constant.bool false
    %355 = torch.aten.arange.start %int0_416, %int2_417, %none_418, %none_419, %cpu_420, %false_421 : !torch.int, !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[2],si64>
    %int0_422 = torch.constant.int 0
    %356 = torch.aten.unsqueeze %355, %int0_422 : !torch.vtensor<[2],si64>, !torch.int -> !torch.vtensor<[1,2],si64>
    %int1_423 = torch.constant.int 1
    %int2_424 = torch.constant.int 2
    %357 = torch.prim.ListConstruct %int1_423, %int2_424 : (!torch.int, !torch.int) -> !torch.list<int>
    %358 = torch.aten.repeat %352, %357 : !torch.vtensor<[1,1],si64>, !torch.list<int> -> !torch.vtensor<[1,2],si64>
    %int1_425 = torch.constant.int 1
    %int2_426 = torch.constant.int 2
    %359 = torch.prim.ListConstruct %int1_425, %int2_426 : (!torch.int, !torch.int) -> !torch.list<int>
    %int2_427 = torch.constant.int 2
    %int1_428 = torch.constant.int 1
    %360 = torch.prim.ListConstruct %int2_427, %int1_428 : (!torch.int, !torch.int) -> !torch.list<int>
    %int4_429 = torch.constant.int 4
    %int0_430 = torch.constant.int 0
    %cpu_431 = torch.constant.device "cpu"
    %false_432 = torch.constant.bool false
    %361 = torch.aten.empty_strided %359, %360, %int4_429, %int0_430, %cpu_431, %false_432 : !torch.list<int>, !torch.list<int>, !torch.int, !torch.int, !torch.Device, !torch.bool -> !torch.vtensor<[1,2],si64>
    %int1_433 = torch.constant.int 1
    %362 = torch.aten.fill.Scalar %361, %int1_433 : !torch.vtensor<[1,2],si64>, !torch.int -> !torch.vtensor<[1,2],si64>
    %int1_434 = torch.constant.int 1
    %int2_435 = torch.constant.int 2
    %363 = torch.prim.ListConstruct %int1_434, %int2_435 : (!torch.int, !torch.int) -> !torch.list<int>
    %364 = torch.aten.repeat %354, %363 : !torch.vtensor<[1,1],si64>, !torch.list<int> -> !torch.vtensor<[1,2],si64>
    %int1_436 = torch.constant.int 1
    %int1_437 = torch.constant.int 1
    %365 = torch.prim.ListConstruct %int1_436, %int1_437 : (!torch.int, !torch.int) -> !torch.list<int>
    %366 = torch.aten.repeat %356, %365 : !torch.vtensor<[1,2],si64>, !torch.list<int> -> !torch.vtensor<[1,2],si64>
    %int3_438 = torch.constant.int 3
    %int2_439 = torch.constant.int 2
    %int16_440 = torch.constant.int 16
    %int4_441 = torch.constant.int 4
    %int32_442 = torch.constant.int 32
    %367 = torch.prim.ListConstruct %34, %int3_438, %int2_439, %int16_440, %int4_441, %int32_442 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %368 = torch.aten.view %154, %367 : !torch.vtensor<[?,12288],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,16,4,32],f16>
    torch.bind_symbolic_shape %368, [%32], affine_map<()[s0] -> (s0, 3, 2, 16, 4, 32)> : !torch.vtensor<[?,3,2,16,4,32],f16>
    %369 = torch.prim.ListConstruct %358, %362, %366, %364 : (!torch.vtensor<[1,2],si64>, !torch.vtensor<[1,2],si64>, !torch.vtensor<[1,2],si64>, !torch.vtensor<[1,2],si64>) -> !torch.list<optional<vtensor>>
    %false_443 = torch.constant.bool false
    %370 = torch.aten.index_put %368, %369, %349, %false_443 : !torch.vtensor<[?,3,2,16,4,32],f16>, !torch.list<optional<vtensor>>, !torch.vtensor<[1,2,4,32],f16>, !torch.bool -> !torch.vtensor<[?,3,2,16,4,32],f16>
    torch.bind_symbolic_shape %370, [%32], affine_map<()[s0] -> (s0, 3, 2, 16, 4, 32)> : !torch.vtensor<[?,3,2,16,4,32],f16>
    %int12288_444 = torch.constant.int 12288
    %371 = torch.prim.ListConstruct %34, %int12288_444 : (!torch.int, !torch.int) -> !torch.list<int>
    %372 = torch.aten.view %370, %371 : !torch.vtensor<[?,3,2,16,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,12288],f16>
    torch.bind_symbolic_shape %372, [%32], affine_map<()[s0] -> (s0, 12288)> : !torch.vtensor<[?,12288],f16>
    %int6_445 = torch.constant.int 6
    %373 = torch.aten.mul.Scalar %arg3, %int6_445 : !torch.vtensor<[1,?],si64>, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %373, [%31], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %int2_446 = torch.constant.int 2
    %int1_447 = torch.constant.int 1
    %374 = torch.aten.add.Scalar %373, %int2_446, %int1_447 : !torch.vtensor<[1,?],si64>, !torch.int, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %374, [%31], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %375 = torch.prim.ListConstruct %33 : (!torch.int) -> !torch.list<int>
    %376 = torch.aten.view %374, %375 : !torch.vtensor<[1,?],si64>, !torch.list<int> -> !torch.vtensor<[?],si64>
    torch.bind_symbolic_shape %376, [%31], affine_map<()[s0] -> (s0)> : !torch.vtensor<[?],si64>
    %int3_448 = torch.constant.int 3
    %int2_449 = torch.constant.int 2
    %int16_450 = torch.constant.int 16
    %int4_451 = torch.constant.int 4
    %int32_452 = torch.constant.int 32
    %377 = torch.prim.ListConstruct %34, %int3_448, %int2_449, %int16_450, %int4_451, %int32_452 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %378 = torch.aten.view %372, %377 : !torch.vtensor<[?,12288],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,16,4,32],f16>
    torch.bind_symbolic_shape %378, [%32], affine_map<()[s0] -> (s0, 3, 2, 16, 4, 32)> : !torch.vtensor<[?,3,2,16,4,32],f16>
    %int16_453 = torch.constant.int 16
    %int4_454 = torch.constant.int 4
    %int32_455 = torch.constant.int 32
    %379 = torch.prim.ListConstruct %166, %int16_453, %int4_454, %int32_455 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %380 = torch.aten.view %378, %379 : !torch.vtensor<[?,3,2,16,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,16,4,32],f16>
    torch.bind_symbolic_shape %380, [%32], affine_map<()[s0] -> (s0 * 6, 16, 4, 32)> : !torch.vtensor<[?,16,4,32],f16>
    %int0_456 = torch.constant.int 0
    %381 = torch.aten.index_select %380, %int0_456, %376 : !torch.vtensor<[?,16,4,32],f16>, !torch.int, !torch.vtensor<[?],si64> -> !torch.vtensor<[?,16,4,32],f16>
    torch.bind_symbolic_shape %381, [%31], affine_map<()[s0] -> (s0, 16, 4, 32)> : !torch.vtensor<[?,16,4,32],f16>
    %int1_457 = torch.constant.int 1
    %int16_458 = torch.constant.int 16
    %int4_459 = torch.constant.int 4
    %int32_460 = torch.constant.int 32
    %382 = torch.prim.ListConstruct %int1_457, %33, %int16_458, %int4_459, %int32_460 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %383 = torch.aten.view %381, %382 : !torch.vtensor<[?,16,4,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,16,4,32],f16>
    torch.bind_symbolic_shape %383, [%31], affine_map<()[s0] -> (1, s0, 16, 4, 32)> : !torch.vtensor<[1,?,16,4,32],f16>
    %int1_461 = torch.constant.int 1
    %int4_462 = torch.constant.int 4
    %int32_463 = torch.constant.int 32
    %384 = torch.prim.ListConstruct %int1_461, %35, %int4_462, %int32_463 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %385 = torch.aten.view %383, %384 : !torch.vtensor<[1,?,16,4,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %385, [%31], affine_map<()[s0] -> (1, s0 * 16, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int0_464 = torch.constant.int 0
    %int0_465 = torch.constant.int 0
    %int9223372036854775807_466 = torch.constant.int 9223372036854775807
    %int1_467 = torch.constant.int 1
    %386 = torch.aten.slice.Tensor %177, %int0_464, %int0_465, %int9223372036854775807_466, %int1_467 : !torch.vtensor<[1,128,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,128,4,32],f16>
    %int1_468 = torch.constant.int 1
    %int0_469 = torch.constant.int 0
    %int1_470 = torch.constant.int 1
    %387 = torch.aten.slice.Tensor %386, %int1_468, %int0_469, %35, %int1_470 : !torch.vtensor<[1,128,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %387, [%31], affine_map<()[s0] -> (1, s0 * 16, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %false_471 = torch.constant.bool false
    %388 = torch.aten.copy %387, %385, %false_471 : !torch.vtensor<[1,?,4,32],f16>, !torch.vtensor<[1,?,4,32],f16>, !torch.bool -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %388, [%31], affine_map<()[s0] -> (1, s0 * 16, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int0_472 = torch.constant.int 0
    %int0_473 = torch.constant.int 0
    %int9223372036854775807_474 = torch.constant.int 9223372036854775807
    %int1_475 = torch.constant.int 1
    %389 = torch.aten.slice.Tensor %177, %int0_472, %int0_473, %int9223372036854775807_474, %int1_475 : !torch.vtensor<[1,128,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,128,4,32],f16>
    %int1_476 = torch.constant.int 1
    %int0_477 = torch.constant.int 0
    %int1_478 = torch.constant.int 1
    %390 = torch.aten.slice_scatter %389, %388, %int1_476, %int0_477, %35, %int1_478 : !torch.vtensor<[1,128,4,32],f16>, !torch.vtensor<[1,?,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,128,4,32],f16>
    %int0_479 = torch.constant.int 0
    %int0_480 = torch.constant.int 0
    %int9223372036854775807_481 = torch.constant.int 9223372036854775807
    %int1_482 = torch.constant.int 1
    %391 = torch.aten.slice_scatter %177, %390, %int0_479, %int0_480, %int9223372036854775807_481, %int1_482 : !torch.vtensor<[1,128,4,32],f16>, !torch.vtensor<[1,128,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,128,4,32],f16>
    %int1_483 = torch.constant.int 1
    %int1_484 = torch.constant.int 1
    %392 = torch.aten.add.Scalar %374, %int1_483, %int1_484 : !torch.vtensor<[1,?],si64>, !torch.int, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %392, [%31], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %393 = torch.prim.ListConstruct %33 : (!torch.int) -> !torch.list<int>
    %394 = torch.aten.view %392, %393 : !torch.vtensor<[1,?],si64>, !torch.list<int> -> !torch.vtensor<[?],si64>
    torch.bind_symbolic_shape %394, [%31], affine_map<()[s0] -> (s0)> : !torch.vtensor<[?],si64>
    %int0_485 = torch.constant.int 0
    %395 = torch.aten.index_select %380, %int0_485, %394 : !torch.vtensor<[?,16,4,32],f16>, !torch.int, !torch.vtensor<[?],si64> -> !torch.vtensor<[?,16,4,32],f16>
    torch.bind_symbolic_shape %395, [%31], affine_map<()[s0] -> (s0, 16, 4, 32)> : !torch.vtensor<[?,16,4,32],f16>
    %int1_486 = torch.constant.int 1
    %int16_487 = torch.constant.int 16
    %int4_488 = torch.constant.int 4
    %int32_489 = torch.constant.int 32
    %396 = torch.prim.ListConstruct %int1_486, %33, %int16_487, %int4_488, %int32_489 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %397 = torch.aten.view %395, %396 : !torch.vtensor<[?,16,4,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,16,4,32],f16>
    torch.bind_symbolic_shape %397, [%31], affine_map<()[s0] -> (1, s0, 16, 4, 32)> : !torch.vtensor<[1,?,16,4,32],f16>
    %int1_490 = torch.constant.int 1
    %int4_491 = torch.constant.int 4
    %int32_492 = torch.constant.int 32
    %398 = torch.prim.ListConstruct %int1_490, %35, %int4_491, %int32_492 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %399 = torch.aten.view %397, %398 : !torch.vtensor<[1,?,16,4,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %399, [%31], affine_map<()[s0] -> (1, s0 * 16, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int0_493 = torch.constant.int 0
    %int0_494 = torch.constant.int 0
    %int9223372036854775807_495 = torch.constant.int 9223372036854775807
    %int1_496 = torch.constant.int 1
    %400 = torch.aten.slice.Tensor %189, %int0_493, %int0_494, %int9223372036854775807_495, %int1_496 : !torch.vtensor<[1,128,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,128,4,32],f16>
    %int1_497 = torch.constant.int 1
    %int0_498 = torch.constant.int 0
    %int1_499 = torch.constant.int 1
    %401 = torch.aten.slice.Tensor %400, %int1_497, %int0_498, %35, %int1_499 : !torch.vtensor<[1,128,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %401, [%31], affine_map<()[s0] -> (1, s0 * 16, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %false_500 = torch.constant.bool false
    %402 = torch.aten.copy %401, %399, %false_500 : !torch.vtensor<[1,?,4,32],f16>, !torch.vtensor<[1,?,4,32],f16>, !torch.bool -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %402, [%31], affine_map<()[s0] -> (1, s0 * 16, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int0_501 = torch.constant.int 0
    %int0_502 = torch.constant.int 0
    %int9223372036854775807_503 = torch.constant.int 9223372036854775807
    %int1_504 = torch.constant.int 1
    %403 = torch.aten.slice.Tensor %189, %int0_501, %int0_502, %int9223372036854775807_503, %int1_504 : !torch.vtensor<[1,128,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,128,4,32],f16>
    %int1_505 = torch.constant.int 1
    %int0_506 = torch.constant.int 0
    %int1_507 = torch.constant.int 1
    %404 = torch.aten.slice_scatter %403, %402, %int1_505, %int0_506, %35, %int1_507 : !torch.vtensor<[1,128,4,32],f16>, !torch.vtensor<[1,?,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,128,4,32],f16>
    %int0_508 = torch.constant.int 0
    %int0_509 = torch.constant.int 0
    %int9223372036854775807_510 = torch.constant.int 9223372036854775807
    %int1_511 = torch.constant.int 1
    %405 = torch.aten.slice_scatter %189, %404, %int0_508, %int0_509, %int9223372036854775807_510, %int1_511 : !torch.vtensor<[1,128,4,32],f16>, !torch.vtensor<[1,128,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,128,4,32],f16>
    %int0_512 = torch.constant.int 0
    %int0_513 = torch.constant.int 0
    %int9223372036854775807_514 = torch.constant.int 9223372036854775807
    %int1_515 = torch.constant.int 1
    %406 = torch.aten.slice.Tensor %391, %int0_512, %int0_513, %int9223372036854775807_514, %int1_515 : !torch.vtensor<[1,128,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,128,4,32],f16>
    %int1_516 = torch.constant.int 1
    %int0_517 = torch.constant.int 0
    %int1_518 = torch.constant.int 1
    %407 = torch.aten.slice.Tensor %406, %int1_516, %int0_517, %35, %int1_518 : !torch.vtensor<[1,128,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %407, [%31], affine_map<()[s0] -> (1, s0 * 16, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int0_519 = torch.constant.int 0
    %int0_520 = torch.constant.int 0
    %int9223372036854775807_521 = torch.constant.int 9223372036854775807
    %int1_522 = torch.constant.int 1
    %408 = torch.aten.slice.Tensor %407, %int0_519, %int0_520, %int9223372036854775807_521, %int1_522 : !torch.vtensor<[1,?,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %408, [%31], affine_map<()[s0] -> (1, s0 * 16, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int2_523 = torch.constant.int 2
    %int0_524 = torch.constant.int 0
    %int9223372036854775807_525 = torch.constant.int 9223372036854775807
    %int1_526 = torch.constant.int 1
    %409 = torch.aten.slice.Tensor %408, %int2_523, %int0_524, %int9223372036854775807_525, %int1_526 : !torch.vtensor<[1,?,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %409, [%31], affine_map<()[s0] -> (1, s0 * 16, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int-2_527 = torch.constant.int -2
    %410 = torch.aten.unsqueeze %409, %int-2_527 : !torch.vtensor<[1,?,4,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,1,32],f16>
    torch.bind_symbolic_shape %410, [%31], affine_map<()[s0] -> (1, s0 * 16, 4, 1, 32)> : !torch.vtensor<[1,?,4,1,32],f16>
    %int1_528 = torch.constant.int 1
    %int4_529 = torch.constant.int 4
    %int2_530 = torch.constant.int 2
    %int32_531 = torch.constant.int 32
    %411 = torch.prim.ListConstruct %int1_528, %35, %int4_529, %int2_530, %int32_531 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_532 = torch.constant.bool false
    %412 = torch.aten.expand %410, %411, %false_532 : !torch.vtensor<[1,?,4,1,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,?,4,2,32],f16>
    torch.bind_symbolic_shape %412, [%31], affine_map<()[s0] -> (1, s0 * 16, 4, 2, 32)> : !torch.vtensor<[1,?,4,2,32],f16>
    %int0_533 = torch.constant.int 0
    %413 = torch.aten.clone %412, %int0_533 : !torch.vtensor<[1,?,4,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,2,32],f16>
    torch.bind_symbolic_shape %413, [%31], affine_map<()[s0] -> (1, s0 * 16, 4, 2, 32)> : !torch.vtensor<[1,?,4,2,32],f16>
    %int1_534 = torch.constant.int 1
    %int8_535 = torch.constant.int 8
    %int32_536 = torch.constant.int 32
    %414 = torch.prim.ListConstruct %int1_534, %35, %int8_535, %int32_536 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %415 = torch.aten._unsafe_view %413, %414 : !torch.vtensor<[1,?,4,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,8,32],f16>
    torch.bind_symbolic_shape %415, [%31], affine_map<()[s0] -> (1, s0 * 16, 8, 32)> : !torch.vtensor<[1,?,8,32],f16>
    %int0_537 = torch.constant.int 0
    %int0_538 = torch.constant.int 0
    %int9223372036854775807_539 = torch.constant.int 9223372036854775807
    %int1_540 = torch.constant.int 1
    %416 = torch.aten.slice.Tensor %405, %int0_537, %int0_538, %int9223372036854775807_539, %int1_540 : !torch.vtensor<[1,128,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,128,4,32],f16>
    %int1_541 = torch.constant.int 1
    %int0_542 = torch.constant.int 0
    %int1_543 = torch.constant.int 1
    %417 = torch.aten.slice.Tensor %416, %int1_541, %int0_542, %35, %int1_543 : !torch.vtensor<[1,128,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %417, [%31], affine_map<()[s0] -> (1, s0 * 16, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int0_544 = torch.constant.int 0
    %int0_545 = torch.constant.int 0
    %int9223372036854775807_546 = torch.constant.int 9223372036854775807
    %int1_547 = torch.constant.int 1
    %418 = torch.aten.slice.Tensor %417, %int0_544, %int0_545, %int9223372036854775807_546, %int1_547 : !torch.vtensor<[1,?,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %418, [%31], affine_map<()[s0] -> (1, s0 * 16, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int2_548 = torch.constant.int 2
    %int0_549 = torch.constant.int 0
    %int9223372036854775807_550 = torch.constant.int 9223372036854775807
    %int1_551 = torch.constant.int 1
    %419 = torch.aten.slice.Tensor %418, %int2_548, %int0_549, %int9223372036854775807_550, %int1_551 : !torch.vtensor<[1,?,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %419, [%31], affine_map<()[s0] -> (1, s0 * 16, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int-2_552 = torch.constant.int -2
    %420 = torch.aten.unsqueeze %419, %int-2_552 : !torch.vtensor<[1,?,4,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,1,32],f16>
    torch.bind_symbolic_shape %420, [%31], affine_map<()[s0] -> (1, s0 * 16, 4, 1, 32)> : !torch.vtensor<[1,?,4,1,32],f16>
    %int1_553 = torch.constant.int 1
    %int4_554 = torch.constant.int 4
    %int2_555 = torch.constant.int 2
    %int32_556 = torch.constant.int 32
    %421 = torch.prim.ListConstruct %int1_553, %35, %int4_554, %int2_555, %int32_556 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_557 = torch.constant.bool false
    %422 = torch.aten.expand %420, %421, %false_557 : !torch.vtensor<[1,?,4,1,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,?,4,2,32],f16>
    torch.bind_symbolic_shape %422, [%31], affine_map<()[s0] -> (1, s0 * 16, 4, 2, 32)> : !torch.vtensor<[1,?,4,2,32],f16>
    %int0_558 = torch.constant.int 0
    %423 = torch.aten.clone %422, %int0_558 : !torch.vtensor<[1,?,4,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,2,32],f16>
    torch.bind_symbolic_shape %423, [%31], affine_map<()[s0] -> (1, s0 * 16, 4, 2, 32)> : !torch.vtensor<[1,?,4,2,32],f16>
    %int1_559 = torch.constant.int 1
    %int8_560 = torch.constant.int 8
    %int32_561 = torch.constant.int 32
    %424 = torch.prim.ListConstruct %int1_559, %35, %int8_560, %int32_561 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %425 = torch.aten._unsafe_view %423, %424 : !torch.vtensor<[1,?,4,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,8,32],f16>
    torch.bind_symbolic_shape %425, [%31], affine_map<()[s0] -> (1, s0 * 16, 8, 32)> : !torch.vtensor<[1,?,8,32],f16>
    %int1_562 = torch.constant.int 1
    %int2_563 = torch.constant.int 2
    %426 = torch.aten.transpose.int %339, %int1_562, %int2_563 : !torch.vtensor<[1,1,8,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,8,1,32],f16>
    %int1_564 = torch.constant.int 1
    %int2_565 = torch.constant.int 2
    %427 = torch.aten.transpose.int %415, %int1_564, %int2_565 : !torch.vtensor<[1,?,8,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,8,?,32],f16>
    torch.bind_symbolic_shape %427, [%31], affine_map<()[s0] -> (1, 8, s0 * 16, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %int1_566 = torch.constant.int 1
    %int2_567 = torch.constant.int 2
    %428 = torch.aten.transpose.int %425, %int1_566, %int2_567 : !torch.vtensor<[1,?,8,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,8,?,32],f16>
    torch.bind_symbolic_shape %428, [%31], affine_map<()[s0] -> (1, 8, s0 * 16, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %int2_568 = torch.constant.int 2
    %int3_569 = torch.constant.int 3
    %429 = torch.aten.transpose.int %427, %int2_568, %int3_569 : !torch.vtensor<[1,8,?,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,8,32,?],f16>
    torch.bind_symbolic_shape %429, [%31], affine_map<()[s0] -> (1, 8, 32, s0 * 16)> : !torch.vtensor<[1,8,32,?],f16>
    %int5_570 = torch.constant.int 5
    %430 = torch.prims.convert_element_type %429, %int5_570 : !torch.vtensor<[1,8,32,?],f16>, !torch.int -> !torch.vtensor<[1,8,32,?],f16>
    torch.bind_symbolic_shape %430, [%31], affine_map<()[s0] -> (1, 8, 32, s0 * 16)> : !torch.vtensor<[1,8,32,?],f16>
    %int1_571 = torch.constant.int 1
    %int8_572 = torch.constant.int 8
    %int1_573 = torch.constant.int 1
    %int32_574 = torch.constant.int 32
    %431 = torch.prim.ListConstruct %int1_571, %int8_572, %int1_573, %int32_574 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_575 = torch.constant.bool false
    %432 = torch.aten.expand %426, %431, %false_575 : !torch.vtensor<[1,8,1,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,8,1,32],f16>
    %int8_576 = torch.constant.int 8
    %int1_577 = torch.constant.int 1
    %int32_578 = torch.constant.int 32
    %433 = torch.prim.ListConstruct %int8_576, %int1_577, %int32_578 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %434 = torch.aten.view %432, %433 : !torch.vtensor<[1,8,1,32],f16>, !torch.list<int> -> !torch.vtensor<[8,1,32],f16>
    %int1_579 = torch.constant.int 1
    %int8_580 = torch.constant.int 8
    %int32_581 = torch.constant.int 32
    %435 = torch.prim.ListConstruct %int1_579, %int8_580, %int32_581, %35 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_582 = torch.constant.bool false
    %436 = torch.aten.expand %430, %435, %false_582 : !torch.vtensor<[1,8,32,?],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,8,32,?],f16>
    torch.bind_symbolic_shape %436, [%31], affine_map<()[s0] -> (1, 8, 32, s0 * 16)> : !torch.vtensor<[1,8,32,?],f16>
    %int8_583 = torch.constant.int 8
    %int32_584 = torch.constant.int 32
    %437 = torch.prim.ListConstruct %int8_583, %int32_584, %35 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %438 = torch.aten.view %436, %437 : !torch.vtensor<[1,8,32,?],f16>, !torch.list<int> -> !torch.vtensor<[8,32,?],f16>
    torch.bind_symbolic_shape %438, [%31], affine_map<()[s0] -> (8, 32, s0 * 16)> : !torch.vtensor<[8,32,?],f16>
    %439 = torch.aten.bmm %434, %438 : !torch.vtensor<[8,1,32],f16>, !torch.vtensor<[8,32,?],f16> -> !torch.vtensor<[8,1,?],f16>
    torch.bind_symbolic_shape %439, [%31], affine_map<()[s0] -> (8, 1, s0 * 16)> : !torch.vtensor<[8,1,?],f16>
    %int1_585 = torch.constant.int 1
    %int8_586 = torch.constant.int 8
    %int1_587 = torch.constant.int 1
    %440 = torch.prim.ListConstruct %int1_585, %int8_586, %int1_587, %35 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %441 = torch.aten.view %439, %440 : !torch.vtensor<[8,1,?],f16>, !torch.list<int> -> !torch.vtensor<[1,8,1,?],f16>
    torch.bind_symbolic_shape %441, [%31], affine_map<()[s0] -> (1, 8, 1, s0 * 16)> : !torch.vtensor<[1,8,1,?],f16>
    %float5.656850e00_588 = torch.constant.float 5.6568542494923806
    %442 = torch.aten.div.Scalar %441, %float5.656850e00_588 : !torch.vtensor<[1,8,1,?],f16>, !torch.float -> !torch.vtensor<[1,8,1,?],f16>
    torch.bind_symbolic_shape %442, [%31], affine_map<()[s0] -> (1, 8, 1, s0 * 16)> : !torch.vtensor<[1,8,1,?],f16>
    %int1_589 = torch.constant.int 1
    %443 = torch.aten.add.Tensor %442, %45, %int1_589 : !torch.vtensor<[1,8,1,?],f16>, !torch.vtensor<[1,1,1,?],f16>, !torch.int -> !torch.vtensor<[1,8,1,?],f16>
    torch.bind_symbolic_shape %443, [%31], affine_map<()[s0] -> (1, 8, 1, s0 * 16)> : !torch.vtensor<[1,8,1,?],f16>
    %int6_590 = torch.constant.int 6
    %444 = torch.prims.convert_element_type %443, %int6_590 : !torch.vtensor<[1,8,1,?],f16>, !torch.int -> !torch.vtensor<[1,8,1,?],f32>
    torch.bind_symbolic_shape %444, [%31], affine_map<()[s0] -> (1, 8, 1, s0 * 16)> : !torch.vtensor<[1,8,1,?],f32>
    %int-1_591 = torch.constant.int -1
    %false_592 = torch.constant.bool false
    %445 = torch.aten._softmax %444, %int-1_591, %false_592 : !torch.vtensor<[1,8,1,?],f32>, !torch.int, !torch.bool -> !torch.vtensor<[1,8,1,?],f32>
    torch.bind_symbolic_shape %445, [%31], affine_map<()[s0] -> (1, 8, 1, s0 * 16)> : !torch.vtensor<[1,8,1,?],f32>
    %int5_593 = torch.constant.int 5
    %446 = torch.prims.convert_element_type %445, %int5_593 : !torch.vtensor<[1,8,1,?],f32>, !torch.int -> !torch.vtensor<[1,8,1,?],f16>
    torch.bind_symbolic_shape %446, [%31], affine_map<()[s0] -> (1, 8, 1, s0 * 16)> : !torch.vtensor<[1,8,1,?],f16>
    %int5_594 = torch.constant.int 5
    %447 = torch.prims.convert_element_type %428, %int5_594 : !torch.vtensor<[1,8,?,32],f16>, !torch.int -> !torch.vtensor<[1,8,?,32],f16>
    torch.bind_symbolic_shape %447, [%31], affine_map<()[s0] -> (1, 8, s0 * 16, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %int1_595 = torch.constant.int 1
    %int8_596 = torch.constant.int 8
    %int1_597 = torch.constant.int 1
    %448 = torch.prim.ListConstruct %int1_595, %int8_596, %int1_597, %35 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_598 = torch.constant.bool false
    %449 = torch.aten.expand %446, %448, %false_598 : !torch.vtensor<[1,8,1,?],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,8,1,?],f16>
    torch.bind_symbolic_shape %449, [%31], affine_map<()[s0] -> (1, 8, 1, s0 * 16)> : !torch.vtensor<[1,8,1,?],f16>
    %int8_599 = torch.constant.int 8
    %int1_600 = torch.constant.int 1
    %450 = torch.prim.ListConstruct %int8_599, %int1_600, %35 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %451 = torch.aten.view %449, %450 : !torch.vtensor<[1,8,1,?],f16>, !torch.list<int> -> !torch.vtensor<[8,1,?],f16>
    torch.bind_symbolic_shape %451, [%31], affine_map<()[s0] -> (8, 1, s0 * 16)> : !torch.vtensor<[8,1,?],f16>
    %int1_601 = torch.constant.int 1
    %int8_602 = torch.constant.int 8
    %int32_603 = torch.constant.int 32
    %452 = torch.prim.ListConstruct %int1_601, %int8_602, %35, %int32_603 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_604 = torch.constant.bool false
    %453 = torch.aten.expand %447, %452, %false_604 : !torch.vtensor<[1,8,?,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,8,?,32],f16>
    torch.bind_symbolic_shape %453, [%31], affine_map<()[s0] -> (1, 8, s0 * 16, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %int8_605 = torch.constant.int 8
    %int32_606 = torch.constant.int 32
    %454 = torch.prim.ListConstruct %int8_605, %35, %int32_606 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %455 = torch.aten.view %453, %454 : !torch.vtensor<[1,8,?,32],f16>, !torch.list<int> -> !torch.vtensor<[8,?,32],f16>
    torch.bind_symbolic_shape %455, [%31], affine_map<()[s0] -> (8, s0 * 16, 32)> : !torch.vtensor<[8,?,32],f16>
    %456 = torch.aten.bmm %451, %455 : !torch.vtensor<[8,1,?],f16>, !torch.vtensor<[8,?,32],f16> -> !torch.vtensor<[8,1,32],f16>
    %int1_607 = torch.constant.int 1
    %int8_608 = torch.constant.int 8
    %int1_609 = torch.constant.int 1
    %int32_610 = torch.constant.int 32
    %457 = torch.prim.ListConstruct %int1_607, %int8_608, %int1_609, %int32_610 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %458 = torch.aten.view %456, %457 : !torch.vtensor<[8,1,32],f16>, !torch.list<int> -> !torch.vtensor<[1,8,1,32],f16>
    %int1_611 = torch.constant.int 1
    %int2_612 = torch.constant.int 2
    %459 = torch.aten.transpose.int %458, %int1_611, %int2_612 : !torch.vtensor<[1,8,1,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,1,8,32],f16>
    %int1_613 = torch.constant.int 1
    %int1_614 = torch.constant.int 1
    %int256_615 = torch.constant.int 256
    %460 = torch.prim.ListConstruct %int1_613, %int1_614, %int256_615 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %461 = torch.aten.view %459, %460 : !torch.vtensor<[1,1,8,32],f16>, !torch.list<int> -> !torch.vtensor<[1,1,256],f16>
    %int5_616 = torch.constant.int 5
    %462 = torch.prims.convert_element_type %14, %int5_616 : !torch.vtensor<[256,256],f32>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int-2_617 = torch.constant.int -2
    %int-1_618 = torch.constant.int -1
    %463 = torch.aten.transpose.int %462, %int-2_617, %int-1_618 : !torch.vtensor<[256,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,256],f16>
    %int5_619 = torch.constant.int 5
    %464 = torch.prims.convert_element_type %463, %int5_619 : !torch.vtensor<[256,256],f16>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int1_620 = torch.constant.int 1
    %int256_621 = torch.constant.int 256
    %465 = torch.prim.ListConstruct %int1_620, %int256_621 : (!torch.int, !torch.int) -> !torch.list<int>
    %466 = torch.aten.view %461, %465 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %467 = torch.aten.mm %466, %464 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,256],f16> -> !torch.vtensor<[1,256],f16>
    %int1_622 = torch.constant.int 1
    %int1_623 = torch.constant.int 1
    %int256_624 = torch.constant.int 256
    %468 = torch.prim.ListConstruct %int1_622, %int1_623, %int256_624 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %469 = torch.aten.view %467, %468 : !torch.vtensor<[1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,1,256],f16>
    %int1_625 = torch.constant.int 1
    %470 = torch.aten.add.Tensor %291, %469, %int1_625 : !torch.vtensor<[1,1,256],f16>, !torch.vtensor<[1,1,256],f16>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %int6_626 = torch.constant.int 6
    %471 = torch.prims.convert_element_type %470, %int6_626 : !torch.vtensor<[1,1,256],f16>, !torch.int -> !torch.vtensor<[1,1,256],f32>
    %int2_627 = torch.constant.int 2
    %472 = torch.aten.pow.Tensor_Scalar %471, %int2_627 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f32>
    %int-1_628 = torch.constant.int -1
    %473 = torch.prim.ListConstruct %int-1_628 : (!torch.int) -> !torch.list<int>
    %true_629 = torch.constant.bool true
    %none_630 = torch.constant.none
    %474 = torch.aten.mean.dim %472, %473, %true_629, %none_630 : !torch.vtensor<[1,1,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1],f32>
    %float1.000000e-02_631 = torch.constant.float 1.000000e-02
    %int1_632 = torch.constant.int 1
    %475 = torch.aten.add.Scalar %474, %float1.000000e-02_631, %int1_632 : !torch.vtensor<[1,1,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,1,1],f32>
    %476 = torch.aten.rsqrt %475 : !torch.vtensor<[1,1,1],f32> -> !torch.vtensor<[1,1,1],f32>
    %477 = torch.aten.mul.Tensor %471, %476 : !torch.vtensor<[1,1,256],f32>, !torch.vtensor<[1,1,1],f32> -> !torch.vtensor<[1,1,256],f32>
    %int6_633 = torch.constant.int 6
    %478 = torch.prims.convert_element_type %477, %int6_633 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f32>
    %479 = torch.aten.mul.Tensor %15, %478 : !torch.vtensor<[256],f32>, !torch.vtensor<[1,1,256],f32> -> !torch.vtensor<[1,1,256],f32>
    %int5_634 = torch.constant.int 5
    %480 = torch.prims.convert_element_type %479, %int5_634 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %int5_635 = torch.constant.int 5
    %481 = torch.prims.convert_element_type %16, %int5_635 : !torch.vtensor<[23,256],f32>, !torch.int -> !torch.vtensor<[23,256],f16>
    %int-2_636 = torch.constant.int -2
    %int-1_637 = torch.constant.int -1
    %482 = torch.aten.transpose.int %481, %int-2_636, %int-1_637 : !torch.vtensor<[23,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,23],f16>
    %int5_638 = torch.constant.int 5
    %483 = torch.prims.convert_element_type %482, %int5_638 : !torch.vtensor<[256,23],f16>, !torch.int -> !torch.vtensor<[256,23],f16>
    %int1_639 = torch.constant.int 1
    %int256_640 = torch.constant.int 256
    %484 = torch.prim.ListConstruct %int1_639, %int256_640 : (!torch.int, !torch.int) -> !torch.list<int>
    %485 = torch.aten.view %480, %484 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %486 = torch.aten.mm %485, %483 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,23],f16> -> !torch.vtensor<[1,23],f16>
    %int1_641 = torch.constant.int 1
    %int1_642 = torch.constant.int 1
    %int23_643 = torch.constant.int 23
    %487 = torch.prim.ListConstruct %int1_641, %int1_642, %int23_643 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %488 = torch.aten.view %486, %487 : !torch.vtensor<[1,23],f16>, !torch.list<int> -> !torch.vtensor<[1,1,23],f16>
    %489 = torch.aten.silu %488 : !torch.vtensor<[1,1,23],f16> -> !torch.vtensor<[1,1,23],f16>
    %int5_644 = torch.constant.int 5
    %490 = torch.prims.convert_element_type %17, %int5_644 : !torch.vtensor<[23,256],f32>, !torch.int -> !torch.vtensor<[23,256],f16>
    %int-2_645 = torch.constant.int -2
    %int-1_646 = torch.constant.int -1
    %491 = torch.aten.transpose.int %490, %int-2_645, %int-1_646 : !torch.vtensor<[23,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,23],f16>
    %int5_647 = torch.constant.int 5
    %492 = torch.prims.convert_element_type %491, %int5_647 : !torch.vtensor<[256,23],f16>, !torch.int -> !torch.vtensor<[256,23],f16>
    %int1_648 = torch.constant.int 1
    %int256_649 = torch.constant.int 256
    %493 = torch.prim.ListConstruct %int1_648, %int256_649 : (!torch.int, !torch.int) -> !torch.list<int>
    %494 = torch.aten.view %480, %493 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %495 = torch.aten.mm %494, %492 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,23],f16> -> !torch.vtensor<[1,23],f16>
    %int1_650 = torch.constant.int 1
    %int1_651 = torch.constant.int 1
    %int23_652 = torch.constant.int 23
    %496 = torch.prim.ListConstruct %int1_650, %int1_651, %int23_652 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %497 = torch.aten.view %495, %496 : !torch.vtensor<[1,23],f16>, !torch.list<int> -> !torch.vtensor<[1,1,23],f16>
    %498 = torch.aten.mul.Tensor %489, %497 : !torch.vtensor<[1,1,23],f16>, !torch.vtensor<[1,1,23],f16> -> !torch.vtensor<[1,1,23],f16>
    %int5_653 = torch.constant.int 5
    %499 = torch.prims.convert_element_type %18, %int5_653 : !torch.vtensor<[256,23],f32>, !torch.int -> !torch.vtensor<[256,23],f16>
    %int-2_654 = torch.constant.int -2
    %int-1_655 = torch.constant.int -1
    %500 = torch.aten.transpose.int %499, %int-2_654, %int-1_655 : !torch.vtensor<[256,23],f16>, !torch.int, !torch.int -> !torch.vtensor<[23,256],f16>
    %int5_656 = torch.constant.int 5
    %501 = torch.prims.convert_element_type %500, %int5_656 : !torch.vtensor<[23,256],f16>, !torch.int -> !torch.vtensor<[23,256],f16>
    %int1_657 = torch.constant.int 1
    %int23_658 = torch.constant.int 23
    %502 = torch.prim.ListConstruct %int1_657, %int23_658 : (!torch.int, !torch.int) -> !torch.list<int>
    %503 = torch.aten.view %498, %502 : !torch.vtensor<[1,1,23],f16>, !torch.list<int> -> !torch.vtensor<[1,23],f16>
    %504 = torch.aten.mm %503, %501 : !torch.vtensor<[1,23],f16>, !torch.vtensor<[23,256],f16> -> !torch.vtensor<[1,256],f16>
    %int1_659 = torch.constant.int 1
    %int1_660 = torch.constant.int 1
    %int256_661 = torch.constant.int 256
    %505 = torch.prim.ListConstruct %int1_659, %int1_660, %int256_661 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %506 = torch.aten.view %504, %505 : !torch.vtensor<[1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,1,256],f16>
    %int1_662 = torch.constant.int 1
    %507 = torch.aten.add.Tensor %470, %506, %int1_662 : !torch.vtensor<[1,1,256],f16>, !torch.vtensor<[1,1,256],f16>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %int6_663 = torch.constant.int 6
    %508 = torch.prims.convert_element_type %507, %int6_663 : !torch.vtensor<[1,1,256],f16>, !torch.int -> !torch.vtensor<[1,1,256],f32>
    %int2_664 = torch.constant.int 2
    %509 = torch.aten.pow.Tensor_Scalar %508, %int2_664 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f32>
    %int-1_665 = torch.constant.int -1
    %510 = torch.prim.ListConstruct %int-1_665 : (!torch.int) -> !torch.list<int>
    %true_666 = torch.constant.bool true
    %none_667 = torch.constant.none
    %511 = torch.aten.mean.dim %509, %510, %true_666, %none_667 : !torch.vtensor<[1,1,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1],f32>
    %float1.000000e-02_668 = torch.constant.float 1.000000e-02
    %int1_669 = torch.constant.int 1
    %512 = torch.aten.add.Scalar %511, %float1.000000e-02_668, %int1_669 : !torch.vtensor<[1,1,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,1,1],f32>
    %513 = torch.aten.rsqrt %512 : !torch.vtensor<[1,1,1],f32> -> !torch.vtensor<[1,1,1],f32>
    %514 = torch.aten.mul.Tensor %508, %513 : !torch.vtensor<[1,1,256],f32>, !torch.vtensor<[1,1,1],f32> -> !torch.vtensor<[1,1,256],f32>
    %int6_670 = torch.constant.int 6
    %515 = torch.prims.convert_element_type %514, %int6_670 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f32>
    %516 = torch.aten.mul.Tensor %19, %515 : !torch.vtensor<[256],f32>, !torch.vtensor<[1,1,256],f32> -> !torch.vtensor<[1,1,256],f32>
    %int5_671 = torch.constant.int 5
    %517 = torch.prims.convert_element_type %516, %int5_671 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %int5_672 = torch.constant.int 5
    %518 = torch.prims.convert_element_type %20, %int5_672 : !torch.vtensor<[256,256],f32>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int-2_673 = torch.constant.int -2
    %int-1_674 = torch.constant.int -1
    %519 = torch.aten.transpose.int %518, %int-2_673, %int-1_674 : !torch.vtensor<[256,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,256],f16>
    %int5_675 = torch.constant.int 5
    %520 = torch.prims.convert_element_type %519, %int5_675 : !torch.vtensor<[256,256],f16>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int1_676 = torch.constant.int 1
    %int256_677 = torch.constant.int 256
    %521 = torch.prim.ListConstruct %int1_676, %int256_677 : (!torch.int, !torch.int) -> !torch.list<int>
    %522 = torch.aten.view %517, %521 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %523 = torch.aten.mm %522, %520 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,256],f16> -> !torch.vtensor<[1,256],f16>
    %int1_678 = torch.constant.int 1
    %int1_679 = torch.constant.int 1
    %int256_680 = torch.constant.int 256
    %524 = torch.prim.ListConstruct %int1_678, %int1_679, %int256_680 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %525 = torch.aten.view %523, %524 : !torch.vtensor<[1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,1,256],f16>
    %int5_681 = torch.constant.int 5
    %526 = torch.prims.convert_element_type %21, %int5_681 : !torch.vtensor<[128,256],f32>, !torch.int -> !torch.vtensor<[128,256],f16>
    %int-2_682 = torch.constant.int -2
    %int-1_683 = torch.constant.int -1
    %527 = torch.aten.transpose.int %526, %int-2_682, %int-1_683 : !torch.vtensor<[128,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,128],f16>
    %int5_684 = torch.constant.int 5
    %528 = torch.prims.convert_element_type %527, %int5_684 : !torch.vtensor<[256,128],f16>, !torch.int -> !torch.vtensor<[256,128],f16>
    %int1_685 = torch.constant.int 1
    %int256_686 = torch.constant.int 256
    %529 = torch.prim.ListConstruct %int1_685, %int256_686 : (!torch.int, !torch.int) -> !torch.list<int>
    %530 = torch.aten.view %517, %529 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %531 = torch.aten.mm %530, %528 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,128],f16> -> !torch.vtensor<[1,128],f16>
    %int1_687 = torch.constant.int 1
    %int1_688 = torch.constant.int 1
    %int128_689 = torch.constant.int 128
    %532 = torch.prim.ListConstruct %int1_687, %int1_688, %int128_689 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %533 = torch.aten.view %531, %532 : !torch.vtensor<[1,128],f16>, !torch.list<int> -> !torch.vtensor<[1,1,128],f16>
    %int5_690 = torch.constant.int 5
    %534 = torch.prims.convert_element_type %22, %int5_690 : !torch.vtensor<[128,256],f32>, !torch.int -> !torch.vtensor<[128,256],f16>
    %int-2_691 = torch.constant.int -2
    %int-1_692 = torch.constant.int -1
    %535 = torch.aten.transpose.int %534, %int-2_691, %int-1_692 : !torch.vtensor<[128,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,128],f16>
    %int5_693 = torch.constant.int 5
    %536 = torch.prims.convert_element_type %535, %int5_693 : !torch.vtensor<[256,128],f16>, !torch.int -> !torch.vtensor<[256,128],f16>
    %int1_694 = torch.constant.int 1
    %int256_695 = torch.constant.int 256
    %537 = torch.prim.ListConstruct %int1_694, %int256_695 : (!torch.int, !torch.int) -> !torch.list<int>
    %538 = torch.aten.view %517, %537 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %539 = torch.aten.mm %538, %536 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,128],f16> -> !torch.vtensor<[1,128],f16>
    %int1_696 = torch.constant.int 1
    %int1_697 = torch.constant.int 1
    %int128_698 = torch.constant.int 128
    %540 = torch.prim.ListConstruct %int1_696, %int1_697, %int128_698 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %541 = torch.aten.view %539, %540 : !torch.vtensor<[1,128],f16>, !torch.list<int> -> !torch.vtensor<[1,1,128],f16>
    %int1_699 = torch.constant.int 1
    %int1_700 = torch.constant.int 1
    %int8_701 = torch.constant.int 8
    %int32_702 = torch.constant.int 32
    %542 = torch.prim.ListConstruct %int1_699, %int1_700, %int8_701, %int32_702 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %543 = torch.aten.view %525, %542 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,1,8,32],f16>
    %int1_703 = torch.constant.int 1
    %int1_704 = torch.constant.int 1
    %int4_705 = torch.constant.int 4
    %int32_706 = torch.constant.int 32
    %544 = torch.prim.ListConstruct %int1_703, %int1_704, %int4_705, %int32_706 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %545 = torch.aten.view %533, %544 : !torch.vtensor<[1,1,128],f16>, !torch.list<int> -> !torch.vtensor<[1,1,4,32],f16>
    %int1_707 = torch.constant.int 1
    %int1_708 = torch.constant.int 1
    %int4_709 = torch.constant.int 4
    %int32_710 = torch.constant.int 32
    %546 = torch.prim.ListConstruct %int1_707, %int1_708, %int4_709, %int32_710 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %547 = torch.aten.view %541, %546 : !torch.vtensor<[1,1,128],f16>, !torch.list<int> -> !torch.vtensor<[1,1,4,32],f16>
    %548 = torch_c.to_builtin_tensor %543 : !torch.vtensor<[1,1,8,32],f16> -> tensor<1x1x8x32xf16>
    %549 = flow.tensor.bitcast %548 : tensor<1x1x8x32xf16> -> tensor<1x1x8x16xcomplex<f16>>
    %550 = torch_c.from_builtin_tensor %549 : tensor<1x1x8x16xcomplex<f16>> -> !torch.vtensor<[1,1,8,16],complex<f16>>
    %551 = torch.aten.mul.Tensor %550, %67 : !torch.vtensor<[1,1,8,16],complex<f16>>, !torch.vtensor<[1,1,1,16],complex<f32>> -> !torch.vtensor<[1,1,8,16],complex<f32>>
    %552 = torch_c.to_builtin_tensor %551 : !torch.vtensor<[1,1,8,16],complex<f32>> -> tensor<1x1x8x16xcomplex<f32>>
    %553 = flow.tensor.bitcast %552 : tensor<1x1x8x16xcomplex<f32>> -> tensor<1x1x8x32xf32>
    %554 = torch_c.from_builtin_tensor %553 : tensor<1x1x8x32xf32> -> !torch.vtensor<[1,1,8,32],f32>
    %int5_711 = torch.constant.int 5
    %555 = torch.prims.convert_element_type %554, %int5_711 : !torch.vtensor<[1,1,8,32],f32>, !torch.int -> !torch.vtensor<[1,1,8,32],f16>
    %556 = torch_c.to_builtin_tensor %545 : !torch.vtensor<[1,1,4,32],f16> -> tensor<1x1x4x32xf16>
    %557 = flow.tensor.bitcast %556 : tensor<1x1x4x32xf16> -> tensor<1x1x4x16xcomplex<f16>>
    %558 = torch_c.from_builtin_tensor %557 : tensor<1x1x4x16xcomplex<f16>> -> !torch.vtensor<[1,1,4,16],complex<f16>>
    %559 = torch.aten.mul.Tensor %558, %67 : !torch.vtensor<[1,1,4,16],complex<f16>>, !torch.vtensor<[1,1,1,16],complex<f32>> -> !torch.vtensor<[1,1,4,16],complex<f32>>
    %560 = torch_c.to_builtin_tensor %559 : !torch.vtensor<[1,1,4,16],complex<f32>> -> tensor<1x1x4x16xcomplex<f32>>
    %561 = flow.tensor.bitcast %560 : tensor<1x1x4x16xcomplex<f32>> -> tensor<1x1x4x32xf32>
    %562 = torch_c.from_builtin_tensor %561 : tensor<1x1x4x32xf32> -> !torch.vtensor<[1,1,4,32],f32>
    %int5_712 = torch.constant.int 5
    %563 = torch.prims.convert_element_type %562, %int5_712 : !torch.vtensor<[1,1,4,32],f32>, !torch.int -> !torch.vtensor<[1,1,4,32],f16>
    %564 = torch.prim.ListConstruct %563, %547 : (!torch.vtensor<[1,1,4,32],f16>, !torch.vtensor<[1,1,4,32],f16>) -> !torch.list<vtensor>
    %int1_713 = torch.constant.int 1
    %565 = torch.aten.cat %564, %int1_713 : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[1,2,4,32],f16>
    %int16_714 = torch.constant.int 16
    %566 = torch.aten.floor_divide.Scalar %arg2, %int16_714 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int1_715 = torch.constant.int 1
    %567 = torch.aten.unsqueeze %566, %int1_715 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_716 = torch.constant.int 1
    %false_717 = torch.constant.bool false
    %568 = torch.aten.gather %arg3, %int1_716, %567, %false_717 : !torch.vtensor<[1,?],si64>, !torch.int, !torch.vtensor<[1,1],si64>, !torch.bool -> !torch.vtensor<[1,1],si64>
    %int16_718 = torch.constant.int 16
    %569 = torch.aten.remainder.Scalar %arg2, %int16_718 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int1_719 = torch.constant.int 1
    %570 = torch.aten.unsqueeze %569, %int1_719 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int0_720 = torch.constant.int 0
    %int2_721 = torch.constant.int 2
    %none_722 = torch.constant.none
    %none_723 = torch.constant.none
    %cpu_724 = torch.constant.device "cpu"
    %false_725 = torch.constant.bool false
    %571 = torch.aten.arange.start %int0_720, %int2_721, %none_722, %none_723, %cpu_724, %false_725 : !torch.int, !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[2],si64>
    %int0_726 = torch.constant.int 0
    %572 = torch.aten.unsqueeze %571, %int0_726 : !torch.vtensor<[2],si64>, !torch.int -> !torch.vtensor<[1,2],si64>
    %int1_727 = torch.constant.int 1
    %int2_728 = torch.constant.int 2
    %573 = torch.prim.ListConstruct %int1_727, %int2_728 : (!torch.int, !torch.int) -> !torch.list<int>
    %574 = torch.aten.repeat %568, %573 : !torch.vtensor<[1,1],si64>, !torch.list<int> -> !torch.vtensor<[1,2],si64>
    %int1_729 = torch.constant.int 1
    %int2_730 = torch.constant.int 2
    %575 = torch.prim.ListConstruct %int1_729, %int2_730 : (!torch.int, !torch.int) -> !torch.list<int>
    %int2_731 = torch.constant.int 2
    %int1_732 = torch.constant.int 1
    %576 = torch.prim.ListConstruct %int2_731, %int1_732 : (!torch.int, !torch.int) -> !torch.list<int>
    %int4_733 = torch.constant.int 4
    %int0_734 = torch.constant.int 0
    %cpu_735 = torch.constant.device "cpu"
    %false_736 = torch.constant.bool false
    %577 = torch.aten.empty_strided %575, %576, %int4_733, %int0_734, %cpu_735, %false_736 : !torch.list<int>, !torch.list<int>, !torch.int, !torch.int, !torch.Device, !torch.bool -> !torch.vtensor<[1,2],si64>
    %int2_737 = torch.constant.int 2
    %578 = torch.aten.fill.Scalar %577, %int2_737 : !torch.vtensor<[1,2],si64>, !torch.int -> !torch.vtensor<[1,2],si64>
    %int1_738 = torch.constant.int 1
    %int2_739 = torch.constant.int 2
    %579 = torch.prim.ListConstruct %int1_738, %int2_739 : (!torch.int, !torch.int) -> !torch.list<int>
    %580 = torch.aten.repeat %570, %579 : !torch.vtensor<[1,1],si64>, !torch.list<int> -> !torch.vtensor<[1,2],si64>
    %int1_740 = torch.constant.int 1
    %int1_741 = torch.constant.int 1
    %581 = torch.prim.ListConstruct %int1_740, %int1_741 : (!torch.int, !torch.int) -> !torch.list<int>
    %582 = torch.aten.repeat %572, %581 : !torch.vtensor<[1,2],si64>, !torch.list<int> -> !torch.vtensor<[1,2],si64>
    %int3_742 = torch.constant.int 3
    %int2_743 = torch.constant.int 2
    %int16_744 = torch.constant.int 16
    %int4_745 = torch.constant.int 4
    %int32_746 = torch.constant.int 32
    %583 = torch.prim.ListConstruct %34, %int3_742, %int2_743, %int16_744, %int4_745, %int32_746 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %584 = torch.aten.view %372, %583 : !torch.vtensor<[?,12288],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,16,4,32],f16>
    torch.bind_symbolic_shape %584, [%32], affine_map<()[s0] -> (s0, 3, 2, 16, 4, 32)> : !torch.vtensor<[?,3,2,16,4,32],f16>
    %585 = torch.prim.ListConstruct %574, %578, %582, %580 : (!torch.vtensor<[1,2],si64>, !torch.vtensor<[1,2],si64>, !torch.vtensor<[1,2],si64>, !torch.vtensor<[1,2],si64>) -> !torch.list<optional<vtensor>>
    %false_747 = torch.constant.bool false
    %586 = torch.aten.index_put %584, %585, %565, %false_747 : !torch.vtensor<[?,3,2,16,4,32],f16>, !torch.list<optional<vtensor>>, !torch.vtensor<[1,2,4,32],f16>, !torch.bool -> !torch.vtensor<[?,3,2,16,4,32],f16>
    torch.bind_symbolic_shape %586, [%32], affine_map<()[s0] -> (s0, 3, 2, 16, 4, 32)> : !torch.vtensor<[?,3,2,16,4,32],f16>
    %int12288_748 = torch.constant.int 12288
    %587 = torch.prim.ListConstruct %34, %int12288_748 : (!torch.int, !torch.int) -> !torch.list<int>
    %588 = torch.aten.view %586, %587 : !torch.vtensor<[?,3,2,16,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,12288],f16>
    torch.overwrite.tensor.contents %588 overwrites %arg4 : !torch.vtensor<[?,12288],f16>, !torch.tensor<[?,12288],f16>
    torch.bind_symbolic_shape %588, [%32], affine_map<()[s0] -> (s0, 12288)> : !torch.vtensor<[?,12288],f16>
    %int6_749 = torch.constant.int 6
    %589 = torch.aten.mul.Scalar %arg3, %int6_749 : !torch.vtensor<[1,?],si64>, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %589, [%31], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %int4_750 = torch.constant.int 4
    %int1_751 = torch.constant.int 1
    %590 = torch.aten.add.Scalar %589, %int4_750, %int1_751 : !torch.vtensor<[1,?],si64>, !torch.int, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %590, [%31], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %591 = torch.prim.ListConstruct %33 : (!torch.int) -> !torch.list<int>
    %592 = torch.aten.view %590, %591 : !torch.vtensor<[1,?],si64>, !torch.list<int> -> !torch.vtensor<[?],si64>
    torch.bind_symbolic_shape %592, [%31], affine_map<()[s0] -> (s0)> : !torch.vtensor<[?],si64>
    %int3_752 = torch.constant.int 3
    %int2_753 = torch.constant.int 2
    %int16_754 = torch.constant.int 16
    %int4_755 = torch.constant.int 4
    %int32_756 = torch.constant.int 32
    %593 = torch.prim.ListConstruct %34, %int3_752, %int2_753, %int16_754, %int4_755, %int32_756 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %594 = torch.aten.view %588, %593 : !torch.vtensor<[?,12288],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,16,4,32],f16>
    torch.bind_symbolic_shape %594, [%32], affine_map<()[s0] -> (s0, 3, 2, 16, 4, 32)> : !torch.vtensor<[?,3,2,16,4,32],f16>
    %int16_757 = torch.constant.int 16
    %int4_758 = torch.constant.int 4
    %int32_759 = torch.constant.int 32
    %595 = torch.prim.ListConstruct %166, %int16_757, %int4_758, %int32_759 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %596 = torch.aten.view %594, %595 : !torch.vtensor<[?,3,2,16,4,32],f16>, !torch.list<int> -> !torch.vtensor<[?,16,4,32],f16>
    torch.bind_symbolic_shape %596, [%32], affine_map<()[s0] -> (s0 * 6, 16, 4, 32)> : !torch.vtensor<[?,16,4,32],f16>
    %int0_760 = torch.constant.int 0
    %597 = torch.aten.index_select %596, %int0_760, %592 : !torch.vtensor<[?,16,4,32],f16>, !torch.int, !torch.vtensor<[?],si64> -> !torch.vtensor<[?,16,4,32],f16>
    torch.bind_symbolic_shape %597, [%31], affine_map<()[s0] -> (s0, 16, 4, 32)> : !torch.vtensor<[?,16,4,32],f16>
    %int1_761 = torch.constant.int 1
    %int16_762 = torch.constant.int 16
    %int4_763 = torch.constant.int 4
    %int32_764 = torch.constant.int 32
    %598 = torch.prim.ListConstruct %int1_761, %33, %int16_762, %int4_763, %int32_764 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %599 = torch.aten.view %597, %598 : !torch.vtensor<[?,16,4,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,16,4,32],f16>
    torch.bind_symbolic_shape %599, [%31], affine_map<()[s0] -> (1, s0, 16, 4, 32)> : !torch.vtensor<[1,?,16,4,32],f16>
    %int1_765 = torch.constant.int 1
    %int4_766 = torch.constant.int 4
    %int32_767 = torch.constant.int 32
    %600 = torch.prim.ListConstruct %int1_765, %35, %int4_766, %int32_767 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %601 = torch.aten.view %599, %600 : !torch.vtensor<[1,?,16,4,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %601, [%31], affine_map<()[s0] -> (1, s0 * 16, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int0_768 = torch.constant.int 0
    %int0_769 = torch.constant.int 0
    %int9223372036854775807_770 = torch.constant.int 9223372036854775807
    %int1_771 = torch.constant.int 1
    %602 = torch.aten.slice.Tensor %391, %int0_768, %int0_769, %int9223372036854775807_770, %int1_771 : !torch.vtensor<[1,128,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,128,4,32],f16>
    %int1_772 = torch.constant.int 1
    %int0_773 = torch.constant.int 0
    %int1_774 = torch.constant.int 1
    %603 = torch.aten.slice.Tensor %602, %int1_772, %int0_773, %35, %int1_774 : !torch.vtensor<[1,128,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %603, [%31], affine_map<()[s0] -> (1, s0 * 16, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %false_775 = torch.constant.bool false
    %604 = torch.aten.copy %603, %601, %false_775 : !torch.vtensor<[1,?,4,32],f16>, !torch.vtensor<[1,?,4,32],f16>, !torch.bool -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %604, [%31], affine_map<()[s0] -> (1, s0 * 16, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int0_776 = torch.constant.int 0
    %int0_777 = torch.constant.int 0
    %int9223372036854775807_778 = torch.constant.int 9223372036854775807
    %int1_779 = torch.constant.int 1
    %605 = torch.aten.slice.Tensor %391, %int0_776, %int0_777, %int9223372036854775807_778, %int1_779 : !torch.vtensor<[1,128,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,128,4,32],f16>
    %int1_780 = torch.constant.int 1
    %int0_781 = torch.constant.int 0
    %int1_782 = torch.constant.int 1
    %606 = torch.aten.slice_scatter %605, %604, %int1_780, %int0_781, %35, %int1_782 : !torch.vtensor<[1,128,4,32],f16>, !torch.vtensor<[1,?,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,128,4,32],f16>
    %int0_783 = torch.constant.int 0
    %int0_784 = torch.constant.int 0
    %int9223372036854775807_785 = torch.constant.int 9223372036854775807
    %int1_786 = torch.constant.int 1
    %607 = torch.aten.slice_scatter %391, %606, %int0_783, %int0_784, %int9223372036854775807_785, %int1_786 : !torch.vtensor<[1,128,4,32],f16>, !torch.vtensor<[1,128,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,128,4,32],f16>
    %int1_787 = torch.constant.int 1
    %int1_788 = torch.constant.int 1
    %608 = torch.aten.add.Scalar %590, %int1_787, %int1_788 : !torch.vtensor<[1,?],si64>, !torch.int, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %608, [%31], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %609 = torch.prim.ListConstruct %33 : (!torch.int) -> !torch.list<int>
    %610 = torch.aten.view %608, %609 : !torch.vtensor<[1,?],si64>, !torch.list<int> -> !torch.vtensor<[?],si64>
    torch.bind_symbolic_shape %610, [%31], affine_map<()[s0] -> (s0)> : !torch.vtensor<[?],si64>
    %int0_789 = torch.constant.int 0
    %611 = torch.aten.index_select %596, %int0_789, %610 : !torch.vtensor<[?,16,4,32],f16>, !torch.int, !torch.vtensor<[?],si64> -> !torch.vtensor<[?,16,4,32],f16>
    torch.bind_symbolic_shape %611, [%31], affine_map<()[s0] -> (s0, 16, 4, 32)> : !torch.vtensor<[?,16,4,32],f16>
    %int1_790 = torch.constant.int 1
    %int16_791 = torch.constant.int 16
    %int4_792 = torch.constant.int 4
    %int32_793 = torch.constant.int 32
    %612 = torch.prim.ListConstruct %int1_790, %33, %int16_791, %int4_792, %int32_793 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %613 = torch.aten.view %611, %612 : !torch.vtensor<[?,16,4,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,16,4,32],f16>
    torch.bind_symbolic_shape %613, [%31], affine_map<()[s0] -> (1, s0, 16, 4, 32)> : !torch.vtensor<[1,?,16,4,32],f16>
    %int1_794 = torch.constant.int 1
    %int4_795 = torch.constant.int 4
    %int32_796 = torch.constant.int 32
    %614 = torch.prim.ListConstruct %int1_794, %35, %int4_795, %int32_796 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %615 = torch.aten.view %613, %614 : !torch.vtensor<[1,?,16,4,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %615, [%31], affine_map<()[s0] -> (1, s0 * 16, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int0_797 = torch.constant.int 0
    %int0_798 = torch.constant.int 0
    %int9223372036854775807_799 = torch.constant.int 9223372036854775807
    %int1_800 = torch.constant.int 1
    %616 = torch.aten.slice.Tensor %405, %int0_797, %int0_798, %int9223372036854775807_799, %int1_800 : !torch.vtensor<[1,128,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,128,4,32],f16>
    %int1_801 = torch.constant.int 1
    %int0_802 = torch.constant.int 0
    %int1_803 = torch.constant.int 1
    %617 = torch.aten.slice.Tensor %616, %int1_801, %int0_802, %35, %int1_803 : !torch.vtensor<[1,128,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %617, [%31], affine_map<()[s0] -> (1, s0 * 16, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %false_804 = torch.constant.bool false
    %618 = torch.aten.copy %617, %615, %false_804 : !torch.vtensor<[1,?,4,32],f16>, !torch.vtensor<[1,?,4,32],f16>, !torch.bool -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %618, [%31], affine_map<()[s0] -> (1, s0 * 16, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int0_805 = torch.constant.int 0
    %int0_806 = torch.constant.int 0
    %int9223372036854775807_807 = torch.constant.int 9223372036854775807
    %int1_808 = torch.constant.int 1
    %619 = torch.aten.slice.Tensor %405, %int0_805, %int0_806, %int9223372036854775807_807, %int1_808 : !torch.vtensor<[1,128,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,128,4,32],f16>
    %int1_809 = torch.constant.int 1
    %int0_810 = torch.constant.int 0
    %int1_811 = torch.constant.int 1
    %620 = torch.aten.slice_scatter %619, %618, %int1_809, %int0_810, %35, %int1_811 : !torch.vtensor<[1,128,4,32],f16>, !torch.vtensor<[1,?,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,128,4,32],f16>
    %int0_812 = torch.constant.int 0
    %int0_813 = torch.constant.int 0
    %int9223372036854775807_814 = torch.constant.int 9223372036854775807
    %int1_815 = torch.constant.int 1
    %621 = torch.aten.slice_scatter %405, %620, %int0_812, %int0_813, %int9223372036854775807_814, %int1_815 : !torch.vtensor<[1,128,4,32],f16>, !torch.vtensor<[1,128,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,128,4,32],f16>
    %int0_816 = torch.constant.int 0
    %int0_817 = torch.constant.int 0
    %int9223372036854775807_818 = torch.constant.int 9223372036854775807
    %int1_819 = torch.constant.int 1
    %622 = torch.aten.slice.Tensor %607, %int0_816, %int0_817, %int9223372036854775807_818, %int1_819 : !torch.vtensor<[1,128,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,128,4,32],f16>
    %int1_820 = torch.constant.int 1
    %int0_821 = torch.constant.int 0
    %int1_822 = torch.constant.int 1
    %623 = torch.aten.slice.Tensor %622, %int1_820, %int0_821, %35, %int1_822 : !torch.vtensor<[1,128,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %623, [%31], affine_map<()[s0] -> (1, s0 * 16, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int0_823 = torch.constant.int 0
    %int0_824 = torch.constant.int 0
    %int9223372036854775807_825 = torch.constant.int 9223372036854775807
    %int1_826 = torch.constant.int 1
    %624 = torch.aten.slice.Tensor %623, %int0_823, %int0_824, %int9223372036854775807_825, %int1_826 : !torch.vtensor<[1,?,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %624, [%31], affine_map<()[s0] -> (1, s0 * 16, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int2_827 = torch.constant.int 2
    %int0_828 = torch.constant.int 0
    %int9223372036854775807_829 = torch.constant.int 9223372036854775807
    %int1_830 = torch.constant.int 1
    %625 = torch.aten.slice.Tensor %624, %int2_827, %int0_828, %int9223372036854775807_829, %int1_830 : !torch.vtensor<[1,?,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %625, [%31], affine_map<()[s0] -> (1, s0 * 16, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int-2_831 = torch.constant.int -2
    %626 = torch.aten.unsqueeze %625, %int-2_831 : !torch.vtensor<[1,?,4,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,1,32],f16>
    torch.bind_symbolic_shape %626, [%31], affine_map<()[s0] -> (1, s0 * 16, 4, 1, 32)> : !torch.vtensor<[1,?,4,1,32],f16>
    %int1_832 = torch.constant.int 1
    %int4_833 = torch.constant.int 4
    %int2_834 = torch.constant.int 2
    %int32_835 = torch.constant.int 32
    %627 = torch.prim.ListConstruct %int1_832, %35, %int4_833, %int2_834, %int32_835 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_836 = torch.constant.bool false
    %628 = torch.aten.expand %626, %627, %false_836 : !torch.vtensor<[1,?,4,1,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,?,4,2,32],f16>
    torch.bind_symbolic_shape %628, [%31], affine_map<()[s0] -> (1, s0 * 16, 4, 2, 32)> : !torch.vtensor<[1,?,4,2,32],f16>
    %int0_837 = torch.constant.int 0
    %629 = torch.aten.clone %628, %int0_837 : !torch.vtensor<[1,?,4,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,2,32],f16>
    torch.bind_symbolic_shape %629, [%31], affine_map<()[s0] -> (1, s0 * 16, 4, 2, 32)> : !torch.vtensor<[1,?,4,2,32],f16>
    %int1_838 = torch.constant.int 1
    %int8_839 = torch.constant.int 8
    %int32_840 = torch.constant.int 32
    %630 = torch.prim.ListConstruct %int1_838, %35, %int8_839, %int32_840 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %631 = torch.aten._unsafe_view %629, %630 : !torch.vtensor<[1,?,4,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,8,32],f16>
    torch.bind_symbolic_shape %631, [%31], affine_map<()[s0] -> (1, s0 * 16, 8, 32)> : !torch.vtensor<[1,?,8,32],f16>
    %int0_841 = torch.constant.int 0
    %int0_842 = torch.constant.int 0
    %int9223372036854775807_843 = torch.constant.int 9223372036854775807
    %int1_844 = torch.constant.int 1
    %632 = torch.aten.slice.Tensor %621, %int0_841, %int0_842, %int9223372036854775807_843, %int1_844 : !torch.vtensor<[1,128,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,128,4,32],f16>
    %int1_845 = torch.constant.int 1
    %int0_846 = torch.constant.int 0
    %int1_847 = torch.constant.int 1
    %633 = torch.aten.slice.Tensor %632, %int1_845, %int0_846, %35, %int1_847 : !torch.vtensor<[1,128,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %633, [%31], affine_map<()[s0] -> (1, s0 * 16, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int0_848 = torch.constant.int 0
    %int0_849 = torch.constant.int 0
    %int9223372036854775807_850 = torch.constant.int 9223372036854775807
    %int1_851 = torch.constant.int 1
    %634 = torch.aten.slice.Tensor %633, %int0_848, %int0_849, %int9223372036854775807_850, %int1_851 : !torch.vtensor<[1,?,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %634, [%31], affine_map<()[s0] -> (1, s0 * 16, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int2_852 = torch.constant.int 2
    %int0_853 = torch.constant.int 0
    %int9223372036854775807_854 = torch.constant.int 9223372036854775807
    %int1_855 = torch.constant.int 1
    %635 = torch.aten.slice.Tensor %634, %int2_852, %int0_853, %int9223372036854775807_854, %int1_855 : !torch.vtensor<[1,?,4,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %635, [%31], affine_map<()[s0] -> (1, s0 * 16, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int-2_856 = torch.constant.int -2
    %636 = torch.aten.unsqueeze %635, %int-2_856 : !torch.vtensor<[1,?,4,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,1,32],f16>
    torch.bind_symbolic_shape %636, [%31], affine_map<()[s0] -> (1, s0 * 16, 4, 1, 32)> : !torch.vtensor<[1,?,4,1,32],f16>
    %int1_857 = torch.constant.int 1
    %int4_858 = torch.constant.int 4
    %int2_859 = torch.constant.int 2
    %int32_860 = torch.constant.int 32
    %637 = torch.prim.ListConstruct %int1_857, %35, %int4_858, %int2_859, %int32_860 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_861 = torch.constant.bool false
    %638 = torch.aten.expand %636, %637, %false_861 : !torch.vtensor<[1,?,4,1,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,?,4,2,32],f16>
    torch.bind_symbolic_shape %638, [%31], affine_map<()[s0] -> (1, s0 * 16, 4, 2, 32)> : !torch.vtensor<[1,?,4,2,32],f16>
    %int0_862 = torch.constant.int 0
    %639 = torch.aten.clone %638, %int0_862 : !torch.vtensor<[1,?,4,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,2,32],f16>
    torch.bind_symbolic_shape %639, [%31], affine_map<()[s0] -> (1, s0 * 16, 4, 2, 32)> : !torch.vtensor<[1,?,4,2,32],f16>
    %int1_863 = torch.constant.int 1
    %int8_864 = torch.constant.int 8
    %int32_865 = torch.constant.int 32
    %640 = torch.prim.ListConstruct %int1_863, %35, %int8_864, %int32_865 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %641 = torch.aten._unsafe_view %639, %640 : !torch.vtensor<[1,?,4,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,8,32],f16>
    torch.bind_symbolic_shape %641, [%31], affine_map<()[s0] -> (1, s0 * 16, 8, 32)> : !torch.vtensor<[1,?,8,32],f16>
    %int1_866 = torch.constant.int 1
    %int2_867 = torch.constant.int 2
    %642 = torch.aten.transpose.int %555, %int1_866, %int2_867 : !torch.vtensor<[1,1,8,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,8,1,32],f16>
    %int1_868 = torch.constant.int 1
    %int2_869 = torch.constant.int 2
    %643 = torch.aten.transpose.int %631, %int1_868, %int2_869 : !torch.vtensor<[1,?,8,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,8,?,32],f16>
    torch.bind_symbolic_shape %643, [%31], affine_map<()[s0] -> (1, 8, s0 * 16, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %int1_870 = torch.constant.int 1
    %int2_871 = torch.constant.int 2
    %644 = torch.aten.transpose.int %641, %int1_870, %int2_871 : !torch.vtensor<[1,?,8,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,8,?,32],f16>
    torch.bind_symbolic_shape %644, [%31], affine_map<()[s0] -> (1, 8, s0 * 16, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %int2_872 = torch.constant.int 2
    %int3_873 = torch.constant.int 3
    %645 = torch.aten.transpose.int %643, %int2_872, %int3_873 : !torch.vtensor<[1,8,?,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,8,32,?],f16>
    torch.bind_symbolic_shape %645, [%31], affine_map<()[s0] -> (1, 8, 32, s0 * 16)> : !torch.vtensor<[1,8,32,?],f16>
    %int5_874 = torch.constant.int 5
    %646 = torch.prims.convert_element_type %645, %int5_874 : !torch.vtensor<[1,8,32,?],f16>, !torch.int -> !torch.vtensor<[1,8,32,?],f16>
    torch.bind_symbolic_shape %646, [%31], affine_map<()[s0] -> (1, 8, 32, s0 * 16)> : !torch.vtensor<[1,8,32,?],f16>
    %int1_875 = torch.constant.int 1
    %int8_876 = torch.constant.int 8
    %int1_877 = torch.constant.int 1
    %int32_878 = torch.constant.int 32
    %647 = torch.prim.ListConstruct %int1_875, %int8_876, %int1_877, %int32_878 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_879 = torch.constant.bool false
    %648 = torch.aten.expand %642, %647, %false_879 : !torch.vtensor<[1,8,1,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,8,1,32],f16>
    %int8_880 = torch.constant.int 8
    %int1_881 = torch.constant.int 1
    %int32_882 = torch.constant.int 32
    %649 = torch.prim.ListConstruct %int8_880, %int1_881, %int32_882 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %650 = torch.aten.view %648, %649 : !torch.vtensor<[1,8,1,32],f16>, !torch.list<int> -> !torch.vtensor<[8,1,32],f16>
    %int1_883 = torch.constant.int 1
    %int8_884 = torch.constant.int 8
    %int32_885 = torch.constant.int 32
    %651 = torch.prim.ListConstruct %int1_883, %int8_884, %int32_885, %35 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_886 = torch.constant.bool false
    %652 = torch.aten.expand %646, %651, %false_886 : !torch.vtensor<[1,8,32,?],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,8,32,?],f16>
    torch.bind_symbolic_shape %652, [%31], affine_map<()[s0] -> (1, 8, 32, s0 * 16)> : !torch.vtensor<[1,8,32,?],f16>
    %int8_887 = torch.constant.int 8
    %int32_888 = torch.constant.int 32
    %653 = torch.prim.ListConstruct %int8_887, %int32_888, %35 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %654 = torch.aten.view %652, %653 : !torch.vtensor<[1,8,32,?],f16>, !torch.list<int> -> !torch.vtensor<[8,32,?],f16>
    torch.bind_symbolic_shape %654, [%31], affine_map<()[s0] -> (8, 32, s0 * 16)> : !torch.vtensor<[8,32,?],f16>
    %655 = torch.aten.bmm %650, %654 : !torch.vtensor<[8,1,32],f16>, !torch.vtensor<[8,32,?],f16> -> !torch.vtensor<[8,1,?],f16>
    torch.bind_symbolic_shape %655, [%31], affine_map<()[s0] -> (8, 1, s0 * 16)> : !torch.vtensor<[8,1,?],f16>
    %int1_889 = torch.constant.int 1
    %int8_890 = torch.constant.int 8
    %int1_891 = torch.constant.int 1
    %656 = torch.prim.ListConstruct %int1_889, %int8_890, %int1_891, %35 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %657 = torch.aten.view %655, %656 : !torch.vtensor<[8,1,?],f16>, !torch.list<int> -> !torch.vtensor<[1,8,1,?],f16>
    torch.bind_symbolic_shape %657, [%31], affine_map<()[s0] -> (1, 8, 1, s0 * 16)> : !torch.vtensor<[1,8,1,?],f16>
    %float5.656850e00_892 = torch.constant.float 5.6568542494923806
    %658 = torch.aten.div.Scalar %657, %float5.656850e00_892 : !torch.vtensor<[1,8,1,?],f16>, !torch.float -> !torch.vtensor<[1,8,1,?],f16>
    torch.bind_symbolic_shape %658, [%31], affine_map<()[s0] -> (1, 8, 1, s0 * 16)> : !torch.vtensor<[1,8,1,?],f16>
    %int1_893 = torch.constant.int 1
    %659 = torch.aten.add.Tensor %658, %45, %int1_893 : !torch.vtensor<[1,8,1,?],f16>, !torch.vtensor<[1,1,1,?],f16>, !torch.int -> !torch.vtensor<[1,8,1,?],f16>
    torch.bind_symbolic_shape %659, [%31], affine_map<()[s0] -> (1, 8, 1, s0 * 16)> : !torch.vtensor<[1,8,1,?],f16>
    %int6_894 = torch.constant.int 6
    %660 = torch.prims.convert_element_type %659, %int6_894 : !torch.vtensor<[1,8,1,?],f16>, !torch.int -> !torch.vtensor<[1,8,1,?],f32>
    torch.bind_symbolic_shape %660, [%31], affine_map<()[s0] -> (1, 8, 1, s0 * 16)> : !torch.vtensor<[1,8,1,?],f32>
    %int-1_895 = torch.constant.int -1
    %false_896 = torch.constant.bool false
    %661 = torch.aten._softmax %660, %int-1_895, %false_896 : !torch.vtensor<[1,8,1,?],f32>, !torch.int, !torch.bool -> !torch.vtensor<[1,8,1,?],f32>
    torch.bind_symbolic_shape %661, [%31], affine_map<()[s0] -> (1, 8, 1, s0 * 16)> : !torch.vtensor<[1,8,1,?],f32>
    %int5_897 = torch.constant.int 5
    %662 = torch.prims.convert_element_type %661, %int5_897 : !torch.vtensor<[1,8,1,?],f32>, !torch.int -> !torch.vtensor<[1,8,1,?],f16>
    torch.bind_symbolic_shape %662, [%31], affine_map<()[s0] -> (1, 8, 1, s0 * 16)> : !torch.vtensor<[1,8,1,?],f16>
    %int5_898 = torch.constant.int 5
    %663 = torch.prims.convert_element_type %644, %int5_898 : !torch.vtensor<[1,8,?,32],f16>, !torch.int -> !torch.vtensor<[1,8,?,32],f16>
    torch.bind_symbolic_shape %663, [%31], affine_map<()[s0] -> (1, 8, s0 * 16, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %int1_899 = torch.constant.int 1
    %int8_900 = torch.constant.int 8
    %int1_901 = torch.constant.int 1
    %664 = torch.prim.ListConstruct %int1_899, %int8_900, %int1_901, %35 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_902 = torch.constant.bool false
    %665 = torch.aten.expand %662, %664, %false_902 : !torch.vtensor<[1,8,1,?],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,8,1,?],f16>
    torch.bind_symbolic_shape %665, [%31], affine_map<()[s0] -> (1, 8, 1, s0 * 16)> : !torch.vtensor<[1,8,1,?],f16>
    %int8_903 = torch.constant.int 8
    %int1_904 = torch.constant.int 1
    %666 = torch.prim.ListConstruct %int8_903, %int1_904, %35 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %667 = torch.aten.view %665, %666 : !torch.vtensor<[1,8,1,?],f16>, !torch.list<int> -> !torch.vtensor<[8,1,?],f16>
    torch.bind_symbolic_shape %667, [%31], affine_map<()[s0] -> (8, 1, s0 * 16)> : !torch.vtensor<[8,1,?],f16>
    %int1_905 = torch.constant.int 1
    %int8_906 = torch.constant.int 8
    %int32_907 = torch.constant.int 32
    %668 = torch.prim.ListConstruct %int1_905, %int8_906, %35, %int32_907 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_908 = torch.constant.bool false
    %669 = torch.aten.expand %663, %668, %false_908 : !torch.vtensor<[1,8,?,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,8,?,32],f16>
    torch.bind_symbolic_shape %669, [%31], affine_map<()[s0] -> (1, 8, s0 * 16, 32)> : !torch.vtensor<[1,8,?,32],f16>
    %int8_909 = torch.constant.int 8
    %int32_910 = torch.constant.int 32
    %670 = torch.prim.ListConstruct %int8_909, %35, %int32_910 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %671 = torch.aten.view %669, %670 : !torch.vtensor<[1,8,?,32],f16>, !torch.list<int> -> !torch.vtensor<[8,?,32],f16>
    torch.bind_symbolic_shape %671, [%31], affine_map<()[s0] -> (8, s0 * 16, 32)> : !torch.vtensor<[8,?,32],f16>
    %672 = torch.aten.bmm %667, %671 : !torch.vtensor<[8,1,?],f16>, !torch.vtensor<[8,?,32],f16> -> !torch.vtensor<[8,1,32],f16>
    %int1_911 = torch.constant.int 1
    %int8_912 = torch.constant.int 8
    %int1_913 = torch.constant.int 1
    %int32_914 = torch.constant.int 32
    %673 = torch.prim.ListConstruct %int1_911, %int8_912, %int1_913, %int32_914 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %674 = torch.aten.view %672, %673 : !torch.vtensor<[8,1,32],f16>, !torch.list<int> -> !torch.vtensor<[1,8,1,32],f16>
    %int1_915 = torch.constant.int 1
    %int2_916 = torch.constant.int 2
    %675 = torch.aten.transpose.int %674, %int1_915, %int2_916 : !torch.vtensor<[1,8,1,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,1,8,32],f16>
    %int1_917 = torch.constant.int 1
    %int1_918 = torch.constant.int 1
    %int256_919 = torch.constant.int 256
    %676 = torch.prim.ListConstruct %int1_917, %int1_918, %int256_919 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %677 = torch.aten.view %675, %676 : !torch.vtensor<[1,1,8,32],f16>, !torch.list<int> -> !torch.vtensor<[1,1,256],f16>
    %int5_920 = torch.constant.int 5
    %678 = torch.prims.convert_element_type %23, %int5_920 : !torch.vtensor<[256,256],f32>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int-2_921 = torch.constant.int -2
    %int-1_922 = torch.constant.int -1
    %679 = torch.aten.transpose.int %678, %int-2_921, %int-1_922 : !torch.vtensor<[256,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,256],f16>
    %int5_923 = torch.constant.int 5
    %680 = torch.prims.convert_element_type %679, %int5_923 : !torch.vtensor<[256,256],f16>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int1_924 = torch.constant.int 1
    %int256_925 = torch.constant.int 256
    %681 = torch.prim.ListConstruct %int1_924, %int256_925 : (!torch.int, !torch.int) -> !torch.list<int>
    %682 = torch.aten.view %677, %681 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %683 = torch.aten.mm %682, %680 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,256],f16> -> !torch.vtensor<[1,256],f16>
    %int1_926 = torch.constant.int 1
    %int1_927 = torch.constant.int 1
    %int256_928 = torch.constant.int 256
    %684 = torch.prim.ListConstruct %int1_926, %int1_927, %int256_928 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %685 = torch.aten.view %683, %684 : !torch.vtensor<[1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,1,256],f16>
    %int1_929 = torch.constant.int 1
    %686 = torch.aten.add.Tensor %507, %685, %int1_929 : !torch.vtensor<[1,1,256],f16>, !torch.vtensor<[1,1,256],f16>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %int6_930 = torch.constant.int 6
    %687 = torch.prims.convert_element_type %686, %int6_930 : !torch.vtensor<[1,1,256],f16>, !torch.int -> !torch.vtensor<[1,1,256],f32>
    %int2_931 = torch.constant.int 2
    %688 = torch.aten.pow.Tensor_Scalar %687, %int2_931 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f32>
    %int-1_932 = torch.constant.int -1
    %689 = torch.prim.ListConstruct %int-1_932 : (!torch.int) -> !torch.list<int>
    %true_933 = torch.constant.bool true
    %none_934 = torch.constant.none
    %690 = torch.aten.mean.dim %688, %689, %true_933, %none_934 : !torch.vtensor<[1,1,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1],f32>
    %float1.000000e-02_935 = torch.constant.float 1.000000e-02
    %int1_936 = torch.constant.int 1
    %691 = torch.aten.add.Scalar %690, %float1.000000e-02_935, %int1_936 : !torch.vtensor<[1,1,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,1,1],f32>
    %692 = torch.aten.rsqrt %691 : !torch.vtensor<[1,1,1],f32> -> !torch.vtensor<[1,1,1],f32>
    %693 = torch.aten.mul.Tensor %687, %692 : !torch.vtensor<[1,1,256],f32>, !torch.vtensor<[1,1,1],f32> -> !torch.vtensor<[1,1,256],f32>
    %int6_937 = torch.constant.int 6
    %694 = torch.prims.convert_element_type %693, %int6_937 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f32>
    %695 = torch.aten.mul.Tensor %24, %694 : !torch.vtensor<[256],f32>, !torch.vtensor<[1,1,256],f32> -> !torch.vtensor<[1,1,256],f32>
    %int5_938 = torch.constant.int 5
    %696 = torch.prims.convert_element_type %695, %int5_938 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %int5_939 = torch.constant.int 5
    %697 = torch.prims.convert_element_type %25, %int5_939 : !torch.vtensor<[23,256],f32>, !torch.int -> !torch.vtensor<[23,256],f16>
    %int-2_940 = torch.constant.int -2
    %int-1_941 = torch.constant.int -1
    %698 = torch.aten.transpose.int %697, %int-2_940, %int-1_941 : !torch.vtensor<[23,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,23],f16>
    %int5_942 = torch.constant.int 5
    %699 = torch.prims.convert_element_type %698, %int5_942 : !torch.vtensor<[256,23],f16>, !torch.int -> !torch.vtensor<[256,23],f16>
    %int1_943 = torch.constant.int 1
    %int256_944 = torch.constant.int 256
    %700 = torch.prim.ListConstruct %int1_943, %int256_944 : (!torch.int, !torch.int) -> !torch.list<int>
    %701 = torch.aten.view %696, %700 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %702 = torch.aten.mm %701, %699 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,23],f16> -> !torch.vtensor<[1,23],f16>
    %int1_945 = torch.constant.int 1
    %int1_946 = torch.constant.int 1
    %int23_947 = torch.constant.int 23
    %703 = torch.prim.ListConstruct %int1_945, %int1_946, %int23_947 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %704 = torch.aten.view %702, %703 : !torch.vtensor<[1,23],f16>, !torch.list<int> -> !torch.vtensor<[1,1,23],f16>
    %705 = torch.aten.silu %704 : !torch.vtensor<[1,1,23],f16> -> !torch.vtensor<[1,1,23],f16>
    %int5_948 = torch.constant.int 5
    %706 = torch.prims.convert_element_type %26, %int5_948 : !torch.vtensor<[23,256],f32>, !torch.int -> !torch.vtensor<[23,256],f16>
    %int-2_949 = torch.constant.int -2
    %int-1_950 = torch.constant.int -1
    %707 = torch.aten.transpose.int %706, %int-2_949, %int-1_950 : !torch.vtensor<[23,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,23],f16>
    %int5_951 = torch.constant.int 5
    %708 = torch.prims.convert_element_type %707, %int5_951 : !torch.vtensor<[256,23],f16>, !torch.int -> !torch.vtensor<[256,23],f16>
    %int1_952 = torch.constant.int 1
    %int256_953 = torch.constant.int 256
    %709 = torch.prim.ListConstruct %int1_952, %int256_953 : (!torch.int, !torch.int) -> !torch.list<int>
    %710 = torch.aten.view %696, %709 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %711 = torch.aten.mm %710, %708 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,23],f16> -> !torch.vtensor<[1,23],f16>
    %int1_954 = torch.constant.int 1
    %int1_955 = torch.constant.int 1
    %int23_956 = torch.constant.int 23
    %712 = torch.prim.ListConstruct %int1_954, %int1_955, %int23_956 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %713 = torch.aten.view %711, %712 : !torch.vtensor<[1,23],f16>, !torch.list<int> -> !torch.vtensor<[1,1,23],f16>
    %714 = torch.aten.mul.Tensor %705, %713 : !torch.vtensor<[1,1,23],f16>, !torch.vtensor<[1,1,23],f16> -> !torch.vtensor<[1,1,23],f16>
    %int5_957 = torch.constant.int 5
    %715 = torch.prims.convert_element_type %27, %int5_957 : !torch.vtensor<[256,23],f32>, !torch.int -> !torch.vtensor<[256,23],f16>
    %int-2_958 = torch.constant.int -2
    %int-1_959 = torch.constant.int -1
    %716 = torch.aten.transpose.int %715, %int-2_958, %int-1_959 : !torch.vtensor<[256,23],f16>, !torch.int, !torch.int -> !torch.vtensor<[23,256],f16>
    %int5_960 = torch.constant.int 5
    %717 = torch.prims.convert_element_type %716, %int5_960 : !torch.vtensor<[23,256],f16>, !torch.int -> !torch.vtensor<[23,256],f16>
    %int1_961 = torch.constant.int 1
    %int23_962 = torch.constant.int 23
    %718 = torch.prim.ListConstruct %int1_961, %int23_962 : (!torch.int, !torch.int) -> !torch.list<int>
    %719 = torch.aten.view %714, %718 : !torch.vtensor<[1,1,23],f16>, !torch.list<int> -> !torch.vtensor<[1,23],f16>
    %720 = torch.aten.mm %719, %717 : !torch.vtensor<[1,23],f16>, !torch.vtensor<[23,256],f16> -> !torch.vtensor<[1,256],f16>
    %int1_963 = torch.constant.int 1
    %int1_964 = torch.constant.int 1
    %int256_965 = torch.constant.int 256
    %721 = torch.prim.ListConstruct %int1_963, %int1_964, %int256_965 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %722 = torch.aten.view %720, %721 : !torch.vtensor<[1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,1,256],f16>
    %int1_966 = torch.constant.int 1
    %723 = torch.aten.add.Tensor %686, %722, %int1_966 : !torch.vtensor<[1,1,256],f16>, !torch.vtensor<[1,1,256],f16>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %int6_967 = torch.constant.int 6
    %724 = torch.prims.convert_element_type %723, %int6_967 : !torch.vtensor<[1,1,256],f16>, !torch.int -> !torch.vtensor<[1,1,256],f32>
    %int2_968 = torch.constant.int 2
    %725 = torch.aten.pow.Tensor_Scalar %724, %int2_968 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f32>
    %int-1_969 = torch.constant.int -1
    %726 = torch.prim.ListConstruct %int-1_969 : (!torch.int) -> !torch.list<int>
    %true_970 = torch.constant.bool true
    %none_971 = torch.constant.none
    %727 = torch.aten.mean.dim %725, %726, %true_970, %none_971 : !torch.vtensor<[1,1,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1],f32>
    %float1.000000e-02_972 = torch.constant.float 1.000000e-02
    %int1_973 = torch.constant.int 1
    %728 = torch.aten.add.Scalar %727, %float1.000000e-02_972, %int1_973 : !torch.vtensor<[1,1,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,1,1],f32>
    %729 = torch.aten.rsqrt %728 : !torch.vtensor<[1,1,1],f32> -> !torch.vtensor<[1,1,1],f32>
    %730 = torch.aten.mul.Tensor %724, %729 : !torch.vtensor<[1,1,256],f32>, !torch.vtensor<[1,1,1],f32> -> !torch.vtensor<[1,1,256],f32>
    %int6_974 = torch.constant.int 6
    %731 = torch.prims.convert_element_type %730, %int6_974 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f32>
    %732 = torch.aten.mul.Tensor %28, %731 : !torch.vtensor<[1,256],f32>, !torch.vtensor<[1,1,256],f32> -> !torch.vtensor<[1,1,256],f32>
    %int5_975 = torch.constant.int 5
    %733 = torch.prims.convert_element_type %732, %int5_975 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %int5_976 = torch.constant.int 5
    %734 = torch.prims.convert_element_type %29, %int5_976 : !torch.vtensor<[256,256],f32>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int-2_977 = torch.constant.int -2
    %int-1_978 = torch.constant.int -1
    %735 = torch.aten.transpose.int %734, %int-2_977, %int-1_978 : !torch.vtensor<[256,256],f16>, !torch.int, !torch.int -> !torch.vtensor<[256,256],f16>
    %int5_979 = torch.constant.int 5
    %736 = torch.prims.convert_element_type %735, %int5_979 : !torch.vtensor<[256,256],f16>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int1_980 = torch.constant.int 1
    %int256_981 = torch.constant.int 256
    %737 = torch.prim.ListConstruct %int1_980, %int256_981 : (!torch.int, !torch.int) -> !torch.list<int>
    %738 = torch.aten.view %733, %737 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %739 = torch.aten.mm %738, %736 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,256],f16> -> !torch.vtensor<[1,256],f16>
    %int1_982 = torch.constant.int 1
    %int1_983 = torch.constant.int 1
    %int256_984 = torch.constant.int 256
    %740 = torch.prim.ListConstruct %int1_982, %int1_983, %int256_984 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %741 = torch.aten.view %739, %740 : !torch.vtensor<[1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,1,256],f16>
    return %741 : !torch.vtensor<[1,1,256],f16>
  }
}
