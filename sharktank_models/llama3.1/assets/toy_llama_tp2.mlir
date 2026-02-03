#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module @module {
  util.global private @__auto.token_embd.weight {stream.affinity = #hal.device.promise<@__device_0>} = #stream.parameter.named<"model"::"token_embd.weight"> : tensor<256x256xf32>
  util.global private @__auto.token_embd.weight$1 {stream.affinity = #hal.device.promise<@__device_1>} = #stream.parameter.named<"model"::"token_embd.weight"> : tensor<256x256xf32>
  util.global private @__auto.blk.0.attn_norm.weight {stream.affinity = #hal.device.promise<@__device_0>} = #stream.parameter.named<"model"::"blk.0.attn_norm.weight"> : tensor<256xf32>
  util.global private @__auto.blk.0.attn_norm.weight$1 {stream.affinity = #hal.device.promise<@__device_1>} = #stream.parameter.named<"model"::"blk.0.attn_norm.weight"> : tensor<256xf32>
  util.global private @__auto.blk.0.attn_q.weight.shard.0 {stream.affinity = #hal.device.promise<@__device_0>} = #stream.parameter.named<"model"::"blk.0.attn_q.weight.shard.0"> : tensor<128x256xf32>
  util.global private @__auto.blk.0.attn_q.weight.shard.1 {stream.affinity = #hal.device.promise<@__device_1>} = #stream.parameter.named<"model"::"blk.0.attn_q.weight.shard.1"> : tensor<128x256xf32>
  util.global private @__auto.blk.0.attn_k.weight.shard.0 {stream.affinity = #hal.device.promise<@__device_0>} = #stream.parameter.named<"model"::"blk.0.attn_k.weight.shard.0"> : tensor<64x256xf32>
  util.global private @__auto.blk.0.attn_k.weight.shard.1 {stream.affinity = #hal.device.promise<@__device_1>} = #stream.parameter.named<"model"::"blk.0.attn_k.weight.shard.1"> : tensor<64x256xf32>
  util.global private @__auto.blk.0.attn_v.weight.shard.0 {stream.affinity = #hal.device.promise<@__device_0>} = #stream.parameter.named<"model"::"blk.0.attn_v.weight.shard.0"> : tensor<64x256xf32>
  util.global private @__auto.blk.0.attn_v.weight.shard.1 {stream.affinity = #hal.device.promise<@__device_1>} = #stream.parameter.named<"model"::"blk.0.attn_v.weight.shard.1"> : tensor<64x256xf32>
  util.global private @__auto.blk.0.attn_output.weight.shard.0 {stream.affinity = #hal.device.promise<@__device_0>} = #stream.parameter.named<"model"::"blk.0.attn_output.weight.shard.0"> : tensor<256x128xf32>
  util.global private @__auto.blk.0.attn_output.weight.shard.1 {stream.affinity = #hal.device.promise<@__device_1>} = #stream.parameter.named<"model"::"blk.0.attn_output.weight.shard.1"> : tensor<256x128xf32>
  util.global private @__auto.blk.0.ffn_norm.weight {stream.affinity = #hal.device.promise<@__device_0>} = #stream.parameter.named<"model"::"blk.0.ffn_norm.weight"> : tensor<256xf32>
  util.global private @__auto.blk.0.ffn_norm.weight$1 {stream.affinity = #hal.device.promise<@__device_1>} = #stream.parameter.named<"model"::"blk.0.ffn_norm.weight"> : tensor<256xf32>
  util.global private @__auto.blk.0.ffn_gate.weight.shard.0 {stream.affinity = #hal.device.promise<@__device_0>} = #stream.parameter.named<"model"::"blk.0.ffn_gate.weight.shard.0"> : tensor<12x256xf32>
  util.global private @__auto.blk.0.ffn_gate.weight.shard.1 {stream.affinity = #hal.device.promise<@__device_1>} = #stream.parameter.named<"model"::"blk.0.ffn_gate.weight.shard.1"> : tensor<11x256xf32>
  util.global private @__auto.blk.0.ffn_up.weight.shard.0 {stream.affinity = #hal.device.promise<@__device_0>} = #stream.parameter.named<"model"::"blk.0.ffn_up.weight.shard.0"> : tensor<12x256xf32>
  util.global private @__auto.blk.0.ffn_up.weight.shard.1 {stream.affinity = #hal.device.promise<@__device_1>} = #stream.parameter.named<"model"::"blk.0.ffn_up.weight.shard.1"> : tensor<11x256xf32>
  util.global private @__auto.blk.0.ffn_down.weight.shard.0 {stream.affinity = #hal.device.promise<@__device_0>} = #stream.parameter.named<"model"::"blk.0.ffn_down.weight.shard.0"> : tensor<256x12xf32>
  util.global private @__auto.blk.0.ffn_down.weight.shard.1 {stream.affinity = #hal.device.promise<@__device_1>} = #stream.parameter.named<"model"::"blk.0.ffn_down.weight.shard.1"> : tensor<256x11xf32>
  util.global private @__auto.blk.1.attn_norm.weight {stream.affinity = #hal.device.promise<@__device_0>} = #stream.parameter.named<"model"::"blk.1.attn_norm.weight"> : tensor<256xf32>
  util.global private @__auto.blk.1.attn_norm.weight$1 {stream.affinity = #hal.device.promise<@__device_1>} = #stream.parameter.named<"model"::"blk.1.attn_norm.weight"> : tensor<256xf32>
  util.global private @__auto.blk.1.attn_q.weight.shard.0 {stream.affinity = #hal.device.promise<@__device_0>} = #stream.parameter.named<"model"::"blk.1.attn_q.weight.shard.0"> : tensor<128x256xf32>
  util.global private @__auto.blk.1.attn_q.weight.shard.1 {stream.affinity = #hal.device.promise<@__device_1>} = #stream.parameter.named<"model"::"blk.1.attn_q.weight.shard.1"> : tensor<128x256xf32>
  util.global private @__auto.blk.1.attn_k.weight.shard.0 {stream.affinity = #hal.device.promise<@__device_0>} = #stream.parameter.named<"model"::"blk.1.attn_k.weight.shard.0"> : tensor<64x256xf32>
  util.global private @__auto.blk.1.attn_k.weight.shard.1 {stream.affinity = #hal.device.promise<@__device_1>} = #stream.parameter.named<"model"::"blk.1.attn_k.weight.shard.1"> : tensor<64x256xf32>
  util.global private @__auto.blk.1.attn_v.weight.shard.0 {stream.affinity = #hal.device.promise<@__device_0>} = #stream.parameter.named<"model"::"blk.1.attn_v.weight.shard.0"> : tensor<64x256xf32>
  util.global private @__auto.blk.1.attn_v.weight.shard.1 {stream.affinity = #hal.device.promise<@__device_1>} = #stream.parameter.named<"model"::"blk.1.attn_v.weight.shard.1"> : tensor<64x256xf32>
  util.global private @__auto.blk.1.attn_output.weight.shard.0 {stream.affinity = #hal.device.promise<@__device_0>} = #stream.parameter.named<"model"::"blk.1.attn_output.weight.shard.0"> : tensor<256x128xf32>
  util.global private @__auto.blk.1.attn_output.weight.shard.1 {stream.affinity = #hal.device.promise<@__device_1>} = #stream.parameter.named<"model"::"blk.1.attn_output.weight.shard.1"> : tensor<256x128xf32>
  util.global private @__auto.blk.1.ffn_norm.weight {stream.affinity = #hal.device.promise<@__device_0>} = #stream.parameter.named<"model"::"blk.1.ffn_norm.weight"> : tensor<256xf32>
  util.global private @__auto.blk.1.ffn_norm.weight$1 {stream.affinity = #hal.device.promise<@__device_1>} = #stream.parameter.named<"model"::"blk.1.ffn_norm.weight"> : tensor<256xf32>
  util.global private @__auto.blk.1.ffn_gate.weight.shard.0 {stream.affinity = #hal.device.promise<@__device_0>} = #stream.parameter.named<"model"::"blk.1.ffn_gate.weight.shard.0"> : tensor<12x256xf32>
  util.global private @__auto.blk.1.ffn_gate.weight.shard.1 {stream.affinity = #hal.device.promise<@__device_1>} = #stream.parameter.named<"model"::"blk.1.ffn_gate.weight.shard.1"> : tensor<11x256xf32>
  util.global private @__auto.blk.1.ffn_up.weight.shard.0 {stream.affinity = #hal.device.promise<@__device_0>} = #stream.parameter.named<"model"::"blk.1.ffn_up.weight.shard.0"> : tensor<12x256xf32>
  util.global private @__auto.blk.1.ffn_up.weight.shard.1 {stream.affinity = #hal.device.promise<@__device_1>} = #stream.parameter.named<"model"::"blk.1.ffn_up.weight.shard.1"> : tensor<11x256xf32>
  util.global private @__auto.blk.1.ffn_down.weight.shard.0 {stream.affinity = #hal.device.promise<@__device_0>} = #stream.parameter.named<"model"::"blk.1.ffn_down.weight.shard.0"> : tensor<256x12xf32>
  util.global private @__auto.blk.1.ffn_down.weight.shard.1 {stream.affinity = #hal.device.promise<@__device_1>} = #stream.parameter.named<"model"::"blk.1.ffn_down.weight.shard.1"> : tensor<256x11xf32>
  util.global private @__auto.blk.2.attn_norm.weight {stream.affinity = #hal.device.promise<@__device_0>} = #stream.parameter.named<"model"::"blk.2.attn_norm.weight"> : tensor<256xf32>
  util.global private @__auto.blk.2.attn_norm.weight$1 {stream.affinity = #hal.device.promise<@__device_1>} = #stream.parameter.named<"model"::"blk.2.attn_norm.weight"> : tensor<256xf32>
  util.global private @__auto.blk.2.attn_q.weight.shard.0 {stream.affinity = #hal.device.promise<@__device_0>} = #stream.parameter.named<"model"::"blk.2.attn_q.weight.shard.0"> : tensor<128x256xf32>
  util.global private @__auto.blk.2.attn_q.weight.shard.1 {stream.affinity = #hal.device.promise<@__device_1>} = #stream.parameter.named<"model"::"blk.2.attn_q.weight.shard.1"> : tensor<128x256xf32>
  util.global private @__auto.blk.2.attn_k.weight.shard.0 {stream.affinity = #hal.device.promise<@__device_0>} = #stream.parameter.named<"model"::"blk.2.attn_k.weight.shard.0"> : tensor<64x256xf32>
  util.global private @__auto.blk.2.attn_k.weight.shard.1 {stream.affinity = #hal.device.promise<@__device_1>} = #stream.parameter.named<"model"::"blk.2.attn_k.weight.shard.1"> : tensor<64x256xf32>
  util.global private @__auto.blk.2.attn_v.weight.shard.0 {stream.affinity = #hal.device.promise<@__device_0>} = #stream.parameter.named<"model"::"blk.2.attn_v.weight.shard.0"> : tensor<64x256xf32>
  util.global private @__auto.blk.2.attn_v.weight.shard.1 {stream.affinity = #hal.device.promise<@__device_1>} = #stream.parameter.named<"model"::"blk.2.attn_v.weight.shard.1"> : tensor<64x256xf32>
  util.global private @__auto.blk.2.attn_output.weight.shard.0 {stream.affinity = #hal.device.promise<@__device_0>} = #stream.parameter.named<"model"::"blk.2.attn_output.weight.shard.0"> : tensor<256x128xf32>
  util.global private @__auto.blk.2.attn_output.weight.shard.1 {stream.affinity = #hal.device.promise<@__device_1>} = #stream.parameter.named<"model"::"blk.2.attn_output.weight.shard.1"> : tensor<256x128xf32>
  util.global private @__auto.blk.2.ffn_norm.weight {stream.affinity = #hal.device.promise<@__device_0>} = #stream.parameter.named<"model"::"blk.2.ffn_norm.weight"> : tensor<256xf32>
  util.global private @__auto.blk.2.ffn_norm.weight$1 {stream.affinity = #hal.device.promise<@__device_1>} = #stream.parameter.named<"model"::"blk.2.ffn_norm.weight"> : tensor<256xf32>
  util.global private @__auto.blk.2.ffn_gate.weight.shard.0 {stream.affinity = #hal.device.promise<@__device_0>} = #stream.parameter.named<"model"::"blk.2.ffn_gate.weight.shard.0"> : tensor<12x256xf32>
  util.global private @__auto.blk.2.ffn_gate.weight.shard.1 {stream.affinity = #hal.device.promise<@__device_1>} = #stream.parameter.named<"model"::"blk.2.ffn_gate.weight.shard.1"> : tensor<11x256xf32>
  util.global private @__auto.blk.2.ffn_up.weight.shard.0 {stream.affinity = #hal.device.promise<@__device_0>} = #stream.parameter.named<"model"::"blk.2.ffn_up.weight.shard.0"> : tensor<12x256xf32>
  util.global private @__auto.blk.2.ffn_up.weight.shard.1 {stream.affinity = #hal.device.promise<@__device_1>} = #stream.parameter.named<"model"::"blk.2.ffn_up.weight.shard.1"> : tensor<11x256xf32>
  util.global private @__auto.blk.2.ffn_down.weight.shard.0 {stream.affinity = #hal.device.promise<@__device_0>} = #stream.parameter.named<"model"::"blk.2.ffn_down.weight.shard.0"> : tensor<256x12xf32>
  util.global private @__auto.blk.2.ffn_down.weight.shard.1 {stream.affinity = #hal.device.promise<@__device_1>} = #stream.parameter.named<"model"::"blk.2.ffn_down.weight.shard.1"> : tensor<256x11xf32>
  util.global private @__auto.output_norm.weight {stream.affinity = #hal.device.promise<@__device_0>} = #stream.parameter.named<"model"::"output_norm.weight"> : tensor<1x256xf32>
  util.global private @__auto.output_norm.weight$1 {stream.affinity = #hal.device.promise<@__device_1>} = #stream.parameter.named<"model"::"output_norm.weight"> : tensor<1x256xf32>
  util.global private @__auto.output.weight.shard.0 {stream.affinity = #hal.device.promise<@__device_0>} = #stream.parameter.named<"model"::"output.weight.shard.0"> : tensor<256x128xf32>
  util.global private @__auto.output.weight.shard.1 {stream.affinity = #hal.device.promise<@__device_1>} = #stream.parameter.named<"model"::"output.weight.shard.1"> : tensor<256x128xf32>
  func.func @prefill_bs1(%arg0: !torch.vtensor<[1,?],si64> {iree.abi.affinity = #hal.device.promise<@__device_0>}, %arg1: !torch.vtensor<[1],si64> {iree.abi.affinity = #hal.device.promise<@__device_0>}, %arg2: !torch.vtensor<[1,?],si64> {iree.abi.affinity = #hal.device.promise<@__device_0>}, %arg3: !torch.tensor<[?,12288],f16> {iree.abi.affinity = #hal.device.promise<@__device_0>}, %arg4: !torch.tensor<[?,12288],f16> {iree.abi.affinity = #hal.device.promise<@__device_1>}) -> !torch.vtensor<[1,?,256],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %__auto.token_embd.weight = util.global.load @__auto.token_embd.weight : tensor<256x256xf32>
    %0 = torch_c.from_builtin_tensor %__auto.token_embd.weight : tensor<256x256xf32> -> !torch.vtensor<[256,256],f32>
    %__auto.token_embd.weight$1 = util.global.load @__auto.token_embd.weight$1 : tensor<256x256xf32>
    %1 = torch_c.from_builtin_tensor %__auto.token_embd.weight$1 : tensor<256x256xf32> -> !torch.vtensor<[256,256],f32>
    %__auto.blk.0.attn_norm.weight = util.global.load @__auto.blk.0.attn_norm.weight : tensor<256xf32>
    %2 = torch_c.from_builtin_tensor %__auto.blk.0.attn_norm.weight : tensor<256xf32> -> !torch.vtensor<[256],f32>
    %__auto.blk.0.attn_norm.weight$1 = util.global.load @__auto.blk.0.attn_norm.weight$1 : tensor<256xf32>
    %3 = torch_c.from_builtin_tensor %__auto.blk.0.attn_norm.weight$1 : tensor<256xf32> -> !torch.vtensor<[256],f32>
    %__auto.blk.0.attn_q.weight.shard.0 = util.global.load @__auto.blk.0.attn_q.weight.shard.0 : tensor<128x256xf32>
    %4 = torch_c.from_builtin_tensor %__auto.blk.0.attn_q.weight.shard.0 : tensor<128x256xf32> -> !torch.vtensor<[128,256],f32>
    %__auto.blk.0.attn_q.weight.shard.1 = util.global.load @__auto.blk.0.attn_q.weight.shard.1 : tensor<128x256xf32>
    %5 = torch_c.from_builtin_tensor %__auto.blk.0.attn_q.weight.shard.1 : tensor<128x256xf32> -> !torch.vtensor<[128,256],f32>
    %__auto.blk.0.attn_k.weight.shard.0 = util.global.load @__auto.blk.0.attn_k.weight.shard.0 : tensor<64x256xf32>
    %6 = torch_c.from_builtin_tensor %__auto.blk.0.attn_k.weight.shard.0 : tensor<64x256xf32> -> !torch.vtensor<[64,256],f32>
    %__auto.blk.0.attn_k.weight.shard.1 = util.global.load @__auto.blk.0.attn_k.weight.shard.1 : tensor<64x256xf32>
    %7 = torch_c.from_builtin_tensor %__auto.blk.0.attn_k.weight.shard.1 : tensor<64x256xf32> -> !torch.vtensor<[64,256],f32>
    %__auto.blk.0.attn_v.weight.shard.0 = util.global.load @__auto.blk.0.attn_v.weight.shard.0 : tensor<64x256xf32>
    %8 = torch_c.from_builtin_tensor %__auto.blk.0.attn_v.weight.shard.0 : tensor<64x256xf32> -> !torch.vtensor<[64,256],f32>
    %__auto.blk.0.attn_v.weight.shard.1 = util.global.load @__auto.blk.0.attn_v.weight.shard.1 : tensor<64x256xf32>
    %9 = torch_c.from_builtin_tensor %__auto.blk.0.attn_v.weight.shard.1 : tensor<64x256xf32> -> !torch.vtensor<[64,256],f32>
    %__auto.blk.0.attn_output.weight.shard.0 = util.global.load @__auto.blk.0.attn_output.weight.shard.0 : tensor<256x128xf32>
    %10 = torch_c.from_builtin_tensor %__auto.blk.0.attn_output.weight.shard.0 : tensor<256x128xf32> -> !torch.vtensor<[256,128],f32>
    %__auto.blk.0.attn_output.weight.shard.1 = util.global.load @__auto.blk.0.attn_output.weight.shard.1 : tensor<256x128xf32>
    %11 = torch_c.from_builtin_tensor %__auto.blk.0.attn_output.weight.shard.1 : tensor<256x128xf32> -> !torch.vtensor<[256,128],f32>
    %__auto.blk.0.ffn_norm.weight = util.global.load @__auto.blk.0.ffn_norm.weight : tensor<256xf32>
    %12 = torch_c.from_builtin_tensor %__auto.blk.0.ffn_norm.weight : tensor<256xf32> -> !torch.vtensor<[256],f32>
    %__auto.blk.0.ffn_norm.weight$1 = util.global.load @__auto.blk.0.ffn_norm.weight$1 : tensor<256xf32>
    %13 = torch_c.from_builtin_tensor %__auto.blk.0.ffn_norm.weight$1 : tensor<256xf32> -> !torch.vtensor<[256],f32>
    %__auto.blk.0.ffn_gate.weight.shard.0 = util.global.load @__auto.blk.0.ffn_gate.weight.shard.0 : tensor<12x256xf32>
    %14 = torch_c.from_builtin_tensor %__auto.blk.0.ffn_gate.weight.shard.0 : tensor<12x256xf32> -> !torch.vtensor<[12,256],f32>
    %__auto.blk.0.ffn_gate.weight.shard.1 = util.global.load @__auto.blk.0.ffn_gate.weight.shard.1 : tensor<11x256xf32>
    %15 = torch_c.from_builtin_tensor %__auto.blk.0.ffn_gate.weight.shard.1 : tensor<11x256xf32> -> !torch.vtensor<[11,256],f32>
    %__auto.blk.0.ffn_up.weight.shard.0 = util.global.load @__auto.blk.0.ffn_up.weight.shard.0 : tensor<12x256xf32>
    %16 = torch_c.from_builtin_tensor %__auto.blk.0.ffn_up.weight.shard.0 : tensor<12x256xf32> -> !torch.vtensor<[12,256],f32>
    %__auto.blk.0.ffn_up.weight.shard.1 = util.global.load @__auto.blk.0.ffn_up.weight.shard.1 : tensor<11x256xf32>
    %17 = torch_c.from_builtin_tensor %__auto.blk.0.ffn_up.weight.shard.1 : tensor<11x256xf32> -> !torch.vtensor<[11,256],f32>
    %__auto.blk.0.ffn_down.weight.shard.0 = util.global.load @__auto.blk.0.ffn_down.weight.shard.0 : tensor<256x12xf32>
    %18 = torch_c.from_builtin_tensor %__auto.blk.0.ffn_down.weight.shard.0 : tensor<256x12xf32> -> !torch.vtensor<[256,12],f32>
    %__auto.blk.0.ffn_down.weight.shard.1 = util.global.load @__auto.blk.0.ffn_down.weight.shard.1 : tensor<256x11xf32>
    %19 = torch_c.from_builtin_tensor %__auto.blk.0.ffn_down.weight.shard.1 : tensor<256x11xf32> -> !torch.vtensor<[256,11],f32>
    %__auto.blk.1.attn_norm.weight = util.global.load @__auto.blk.1.attn_norm.weight : tensor<256xf32>
    %20 = torch_c.from_builtin_tensor %__auto.blk.1.attn_norm.weight : tensor<256xf32> -> !torch.vtensor<[256],f32>
    %__auto.blk.1.attn_norm.weight$1 = util.global.load @__auto.blk.1.attn_norm.weight$1 : tensor<256xf32>
    %21 = torch_c.from_builtin_tensor %__auto.blk.1.attn_norm.weight$1 : tensor<256xf32> -> !torch.vtensor<[256],f32>
    %__auto.blk.1.attn_q.weight.shard.0 = util.global.load @__auto.blk.1.attn_q.weight.shard.0 : tensor<128x256xf32>
    %22 = torch_c.from_builtin_tensor %__auto.blk.1.attn_q.weight.shard.0 : tensor<128x256xf32> -> !torch.vtensor<[128,256],f32>
    %__auto.blk.1.attn_q.weight.shard.1 = util.global.load @__auto.blk.1.attn_q.weight.shard.1 : tensor<128x256xf32>
    %23 = torch_c.from_builtin_tensor %__auto.blk.1.attn_q.weight.shard.1 : tensor<128x256xf32> -> !torch.vtensor<[128,256],f32>
    %__auto.blk.1.attn_k.weight.shard.0 = util.global.load @__auto.blk.1.attn_k.weight.shard.0 : tensor<64x256xf32>
    %24 = torch_c.from_builtin_tensor %__auto.blk.1.attn_k.weight.shard.0 : tensor<64x256xf32> -> !torch.vtensor<[64,256],f32>
    %__auto.blk.1.attn_k.weight.shard.1 = util.global.load @__auto.blk.1.attn_k.weight.shard.1 : tensor<64x256xf32>
    %25 = torch_c.from_builtin_tensor %__auto.blk.1.attn_k.weight.shard.1 : tensor<64x256xf32> -> !torch.vtensor<[64,256],f32>
    %__auto.blk.1.attn_v.weight.shard.0 = util.global.load @__auto.blk.1.attn_v.weight.shard.0 : tensor<64x256xf32>
    %26 = torch_c.from_builtin_tensor %__auto.blk.1.attn_v.weight.shard.0 : tensor<64x256xf32> -> !torch.vtensor<[64,256],f32>
    %__auto.blk.1.attn_v.weight.shard.1 = util.global.load @__auto.blk.1.attn_v.weight.shard.1 : tensor<64x256xf32>
    %27 = torch_c.from_builtin_tensor %__auto.blk.1.attn_v.weight.shard.1 : tensor<64x256xf32> -> !torch.vtensor<[64,256],f32>
    %__auto.blk.1.attn_output.weight.shard.0 = util.global.load @__auto.blk.1.attn_output.weight.shard.0 : tensor<256x128xf32>
    %28 = torch_c.from_builtin_tensor %__auto.blk.1.attn_output.weight.shard.0 : tensor<256x128xf32> -> !torch.vtensor<[256,128],f32>
    %__auto.blk.1.attn_output.weight.shard.1 = util.global.load @__auto.blk.1.attn_output.weight.shard.1 : tensor<256x128xf32>
    %29 = torch_c.from_builtin_tensor %__auto.blk.1.attn_output.weight.shard.1 : tensor<256x128xf32> -> !torch.vtensor<[256,128],f32>
    %__auto.blk.1.ffn_norm.weight = util.global.load @__auto.blk.1.ffn_norm.weight : tensor<256xf32>
    %30 = torch_c.from_builtin_tensor %__auto.blk.1.ffn_norm.weight : tensor<256xf32> -> !torch.vtensor<[256],f32>
    %__auto.blk.1.ffn_norm.weight$1 = util.global.load @__auto.blk.1.ffn_norm.weight$1 : tensor<256xf32>
    %31 = torch_c.from_builtin_tensor %__auto.blk.1.ffn_norm.weight$1 : tensor<256xf32> -> !torch.vtensor<[256],f32>
    %__auto.blk.1.ffn_gate.weight.shard.0 = util.global.load @__auto.blk.1.ffn_gate.weight.shard.0 : tensor<12x256xf32>
    %32 = torch_c.from_builtin_tensor %__auto.blk.1.ffn_gate.weight.shard.0 : tensor<12x256xf32> -> !torch.vtensor<[12,256],f32>
    %__auto.blk.1.ffn_gate.weight.shard.1 = util.global.load @__auto.blk.1.ffn_gate.weight.shard.1 : tensor<11x256xf32>
    %33 = torch_c.from_builtin_tensor %__auto.blk.1.ffn_gate.weight.shard.1 : tensor<11x256xf32> -> !torch.vtensor<[11,256],f32>
    %__auto.blk.1.ffn_up.weight.shard.0 = util.global.load @__auto.blk.1.ffn_up.weight.shard.0 : tensor<12x256xf32>
    %34 = torch_c.from_builtin_tensor %__auto.blk.1.ffn_up.weight.shard.0 : tensor<12x256xf32> -> !torch.vtensor<[12,256],f32>
    %__auto.blk.1.ffn_up.weight.shard.1 = util.global.load @__auto.blk.1.ffn_up.weight.shard.1 : tensor<11x256xf32>
    %35 = torch_c.from_builtin_tensor %__auto.blk.1.ffn_up.weight.shard.1 : tensor<11x256xf32> -> !torch.vtensor<[11,256],f32>
    %__auto.blk.1.ffn_down.weight.shard.0 = util.global.load @__auto.blk.1.ffn_down.weight.shard.0 : tensor<256x12xf32>
    %36 = torch_c.from_builtin_tensor %__auto.blk.1.ffn_down.weight.shard.0 : tensor<256x12xf32> -> !torch.vtensor<[256,12],f32>
    %__auto.blk.1.ffn_down.weight.shard.1 = util.global.load @__auto.blk.1.ffn_down.weight.shard.1 : tensor<256x11xf32>
    %37 = torch_c.from_builtin_tensor %__auto.blk.1.ffn_down.weight.shard.1 : tensor<256x11xf32> -> !torch.vtensor<[256,11],f32>
    %__auto.blk.2.attn_norm.weight = util.global.load @__auto.blk.2.attn_norm.weight : tensor<256xf32>
    %38 = torch_c.from_builtin_tensor %__auto.blk.2.attn_norm.weight : tensor<256xf32> -> !torch.vtensor<[256],f32>
    %__auto.blk.2.attn_norm.weight$1 = util.global.load @__auto.blk.2.attn_norm.weight$1 : tensor<256xf32>
    %39 = torch_c.from_builtin_tensor %__auto.blk.2.attn_norm.weight$1 : tensor<256xf32> -> !torch.vtensor<[256],f32>
    %__auto.blk.2.attn_q.weight.shard.0 = util.global.load @__auto.blk.2.attn_q.weight.shard.0 : tensor<128x256xf32>
    %40 = torch_c.from_builtin_tensor %__auto.blk.2.attn_q.weight.shard.0 : tensor<128x256xf32> -> !torch.vtensor<[128,256],f32>
    %__auto.blk.2.attn_q.weight.shard.1 = util.global.load @__auto.blk.2.attn_q.weight.shard.1 : tensor<128x256xf32>
    %41 = torch_c.from_builtin_tensor %__auto.blk.2.attn_q.weight.shard.1 : tensor<128x256xf32> -> !torch.vtensor<[128,256],f32>
    %__auto.blk.2.attn_k.weight.shard.0 = util.global.load @__auto.blk.2.attn_k.weight.shard.0 : tensor<64x256xf32>
    %42 = torch_c.from_builtin_tensor %__auto.blk.2.attn_k.weight.shard.0 : tensor<64x256xf32> -> !torch.vtensor<[64,256],f32>
    %__auto.blk.2.attn_k.weight.shard.1 = util.global.load @__auto.blk.2.attn_k.weight.shard.1 : tensor<64x256xf32>
    %43 = torch_c.from_builtin_tensor %__auto.blk.2.attn_k.weight.shard.1 : tensor<64x256xf32> -> !torch.vtensor<[64,256],f32>
    %__auto.blk.2.attn_v.weight.shard.0 = util.global.load @__auto.blk.2.attn_v.weight.shard.0 : tensor<64x256xf32>
    %44 = torch_c.from_builtin_tensor %__auto.blk.2.attn_v.weight.shard.0 : tensor<64x256xf32> -> !torch.vtensor<[64,256],f32>
    %__auto.blk.2.attn_v.weight.shard.1 = util.global.load @__auto.blk.2.attn_v.weight.shard.1 : tensor<64x256xf32>
    %45 = torch_c.from_builtin_tensor %__auto.blk.2.attn_v.weight.shard.1 : tensor<64x256xf32> -> !torch.vtensor<[64,256],f32>
    %__auto.blk.2.attn_output.weight.shard.0 = util.global.load @__auto.blk.2.attn_output.weight.shard.0 : tensor<256x128xf32>
    %46 = torch_c.from_builtin_tensor %__auto.blk.2.attn_output.weight.shard.0 : tensor<256x128xf32> -> !torch.vtensor<[256,128],f32>
    %__auto.blk.2.attn_output.weight.shard.1 = util.global.load @__auto.blk.2.attn_output.weight.shard.1 : tensor<256x128xf32>
    %47 = torch_c.from_builtin_tensor %__auto.blk.2.attn_output.weight.shard.1 : tensor<256x128xf32> -> !torch.vtensor<[256,128],f32>
    %__auto.blk.2.ffn_norm.weight = util.global.load @__auto.blk.2.ffn_norm.weight : tensor<256xf32>
    %48 = torch_c.from_builtin_tensor %__auto.blk.2.ffn_norm.weight : tensor<256xf32> -> !torch.vtensor<[256],f32>
    %__auto.blk.2.ffn_norm.weight$1 = util.global.load @__auto.blk.2.ffn_norm.weight$1 : tensor<256xf32>
    %49 = torch_c.from_builtin_tensor %__auto.blk.2.ffn_norm.weight$1 : tensor<256xf32> -> !torch.vtensor<[256],f32>
    %__auto.blk.2.ffn_gate.weight.shard.0 = util.global.load @__auto.blk.2.ffn_gate.weight.shard.0 : tensor<12x256xf32>
    %50 = torch_c.from_builtin_tensor %__auto.blk.2.ffn_gate.weight.shard.0 : tensor<12x256xf32> -> !torch.vtensor<[12,256],f32>
    %__auto.blk.2.ffn_gate.weight.shard.1 = util.global.load @__auto.blk.2.ffn_gate.weight.shard.1 : tensor<11x256xf32>
    %51 = torch_c.from_builtin_tensor %__auto.blk.2.ffn_gate.weight.shard.1 : tensor<11x256xf32> -> !torch.vtensor<[11,256],f32>
    %__auto.blk.2.ffn_up.weight.shard.0 = util.global.load @__auto.blk.2.ffn_up.weight.shard.0 : tensor<12x256xf32>
    %52 = torch_c.from_builtin_tensor %__auto.blk.2.ffn_up.weight.shard.0 : tensor<12x256xf32> -> !torch.vtensor<[12,256],f32>
    %__auto.blk.2.ffn_up.weight.shard.1 = util.global.load @__auto.blk.2.ffn_up.weight.shard.1 : tensor<11x256xf32>
    %53 = torch_c.from_builtin_tensor %__auto.blk.2.ffn_up.weight.shard.1 : tensor<11x256xf32> -> !torch.vtensor<[11,256],f32>
    %__auto.blk.2.ffn_down.weight.shard.0 = util.global.load @__auto.blk.2.ffn_down.weight.shard.0 : tensor<256x12xf32>
    %54 = torch_c.from_builtin_tensor %__auto.blk.2.ffn_down.weight.shard.0 : tensor<256x12xf32> -> !torch.vtensor<[256,12],f32>
    %__auto.blk.2.ffn_down.weight.shard.1 = util.global.load @__auto.blk.2.ffn_down.weight.shard.1 : tensor<256x11xf32>
    %55 = torch_c.from_builtin_tensor %__auto.blk.2.ffn_down.weight.shard.1 : tensor<256x11xf32> -> !torch.vtensor<[256,11],f32>
    %__auto.output_norm.weight = util.global.load @__auto.output_norm.weight : tensor<1x256xf32>
    %56 = torch_c.from_builtin_tensor %__auto.output_norm.weight : tensor<1x256xf32> -> !torch.vtensor<[1,256],f32>
    %__auto.output_norm.weight$1 = util.global.load @__auto.output_norm.weight$1 : tensor<1x256xf32>
    %57 = torch_c.from_builtin_tensor %__auto.output_norm.weight$1 : tensor<1x256xf32> -> !torch.vtensor<[1,256],f32>
    %__auto.output.weight.shard.0 = util.global.load @__auto.output.weight.shard.0 : tensor<256x128xf32>
    %58 = torch_c.from_builtin_tensor %__auto.output.weight.shard.0 : tensor<256x128xf32> -> !torch.vtensor<[256,128],f32>
    %__auto.output.weight.shard.1 = util.global.load @__auto.output.weight.shard.1 : tensor<256x128xf32>
    %59 = torch_c.from_builtin_tensor %__auto.output.weight.shard.1 : tensor<256x128xf32> -> !torch.vtensor<[256,128],f32>
    %60 = torch.copy.to_vtensor %arg3 : !torch.vtensor<[?,12288],f16>
    %61 = torch.copy.to_vtensor %arg4 : !torch.vtensor<[?,12288],f16>
    %62 = torch.symbolic_int "32*s1" {min_val = 64, max_val = 96} : !torch.int
    %63 = torch.symbolic_int "s1" {min_val = 2, max_val = 3} : !torch.int
    %64 = torch.symbolic_int "s2" {min_val = 0, max_val = 9223372036854775807} : !torch.int
    torch.bind_symbolic_shape %arg0, [%63], affine_map<()[s0] -> (1, s0 * 32)> : !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %arg2, [%63], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %60, [%64], affine_map<()[s0] -> (s0, 12288)> : !torch.vtensor<[?,12288],f16>
    torch.bind_symbolic_shape %61, [%64], affine_map<()[s0] -> (s0, 12288)> : !torch.vtensor<[?,12288],f16>
    %int1 = torch.constant.int 1
    %65 = torch.aten.size.int %arg2, %int1 : !torch.vtensor<[1,?],si64>, !torch.int -> !torch.int
    %int0 = torch.constant.int 0
    %66 = torch.aten.size.int %60, %int0 : !torch.vtensor<[?,12288],f16>, !torch.int -> !torch.int
    %int1_0 = torch.constant.int 1
    %67 = torch.aten.size.int %arg0, %int1_0 : !torch.vtensor<[1,?],si64>, !torch.int -> !torch.int
    %int0_1 = torch.constant.int 0
    %int1_2 = torch.constant.int 1
    %none = torch.constant.none
    %none_3 = torch.constant.none
    %cpu = torch.constant.device "cpu"
    %false = torch.constant.bool false
    %68 = torch.aten.arange.start_step %int0_1, %67, %int1_2, %none, %none_3, %cpu, %false : !torch.int, !torch.int, !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[?],si64>
    torch.bind_symbolic_shape %68, [%63], affine_map<()[s0] -> (s0 * 32)> : !torch.vtensor<[?],si64>
    %int-1 = torch.constant.int -1
    %69 = torch.aten.unsqueeze %arg1, %int-1 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %70 = torch.aten.ge.Tensor %68, %69 : !torch.vtensor<[?],si64>, !torch.vtensor<[1,1],si64> -> !torch.vtensor<[1,?],i1>
    torch.bind_symbolic_shape %70, [%63], affine_map<()[s0] -> (1, s0 * 32)> : !torch.vtensor<[1,?],i1>
    %int1_4 = torch.constant.int 1
    %int1_5 = torch.constant.int 1
    %71 = torch.prim.ListConstruct %int1_4, %int1_5 : (!torch.int, !torch.int) -> !torch.list<int>
    %int11 = torch.constant.int 11
    %none_6 = torch.constant.none
    %cpu_7 = torch.constant.device "cpu"
    %false_8 = torch.constant.bool false
    %72 = torch.aten.ones %71, %int11, %none_6, %cpu_7, %false_8 : !torch.list<int>, !torch.int, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[1,1],i1>
    %int128 = torch.constant.int 128
    %int128_9 = torch.constant.int 128
    %73 = torch.prim.ListConstruct %int128, %int128_9 : (!torch.int, !torch.int) -> !torch.list<int>
    %false_10 = torch.constant.bool false
    %74 = torch.aten.expand %72, %73, %false_10 : !torch.vtensor<[1,1],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[128,128],i1>
    %int1_11 = torch.constant.int 1
    %75 = torch.aten.triu %74, %int1_11 : !torch.vtensor<[128,128],i1>, !torch.int -> !torch.vtensor<[128,128],i1>
    %int0_12 = torch.constant.int 0
    %76 = torch.aten.unsqueeze %75, %int0_12 : !torch.vtensor<[128,128],i1>, !torch.int -> !torch.vtensor<[1,128,128],i1>
    %int1_13 = torch.constant.int 1
    %77 = torch.aten.unsqueeze %76, %int1_13 : !torch.vtensor<[1,128,128],i1>, !torch.int -> !torch.vtensor<[1,1,128,128],i1>
    %int2 = torch.constant.int 2
    %int0_14 = torch.constant.int 0
    %int9223372036854775807 = torch.constant.int 9223372036854775807
    %int1_15 = torch.constant.int 1
    %78 = torch.aten.slice.Tensor %77, %int2, %int0_14, %int9223372036854775807, %int1_15 : !torch.vtensor<[1,1,128,128],i1>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,1,128,128],i1>
    %int3 = torch.constant.int 3
    %int0_16 = torch.constant.int 0
    %int9223372036854775807_17 = torch.constant.int 9223372036854775807
    %int1_18 = torch.constant.int 1
    %79 = torch.aten.slice.Tensor %78, %int3, %int0_16, %int9223372036854775807_17, %int1_18 : !torch.vtensor<[1,1,128,128],i1>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,1,128,128],i1>
    %int0_19 = torch.constant.int 0
    %int0_20 = torch.constant.int 0
    %int9223372036854775807_21 = torch.constant.int 9223372036854775807
    %int1_22 = torch.constant.int 1
    %80 = torch.aten.slice.Tensor %79, %int0_19, %int0_20, %int9223372036854775807_21, %int1_22 : !torch.vtensor<[1,1,128,128],i1>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,1,128,128],i1>
    %int1_23 = torch.constant.int 1
    %int0_24 = torch.constant.int 0
    %int9223372036854775807_25 = torch.constant.int 9223372036854775807
    %int1_26 = torch.constant.int 1
    %81 = torch.aten.slice.Tensor %80, %int1_23, %int0_24, %int9223372036854775807_25, %int1_26 : !torch.vtensor<[1,1,128,128],i1>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,1,128,128],i1>
    %int2_27 = torch.constant.int 2
    %int0_28 = torch.constant.int 0
    %int1_29 = torch.constant.int 1
    %82 = torch.aten.slice.Tensor %81, %int2_27, %int0_28, %67, %int1_29 : !torch.vtensor<[1,1,128,128],i1>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,1,?,128],i1>
    torch.bind_symbolic_shape %82, [%63], affine_map<()[s0] -> (1, 1, s0 * 32, 128)> : !torch.vtensor<[1,1,?,128],i1>
    %int3_30 = torch.constant.int 3
    %int0_31 = torch.constant.int 0
    %int1_32 = torch.constant.int 1
    %83 = torch.aten.slice.Tensor %82, %int3_30, %int0_31, %67, %int1_32 : !torch.vtensor<[1,1,?,128],i1>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,1,?,?],i1>
    torch.bind_symbolic_shape %83, [%63], affine_map<()[s0] -> (1, 1, s0 * 32, s0 * 32)> : !torch.vtensor<[1,1,?,?],i1>
    %int0_33 = torch.constant.int 0
    %int0_34 = torch.constant.int 0
    %int9223372036854775807_35 = torch.constant.int 9223372036854775807
    %int1_36 = torch.constant.int 1
    %84 = torch.aten.slice.Tensor %70, %int0_33, %int0_34, %int9223372036854775807_35, %int1_36 : !torch.vtensor<[1,?],i1>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?],i1>
    torch.bind_symbolic_shape %84, [%63], affine_map<()[s0] -> (1, s0 * 32)> : !torch.vtensor<[1,?],i1>
    %int1_37 = torch.constant.int 1
    %85 = torch.aten.unsqueeze %84, %int1_37 : !torch.vtensor<[1,?],i1>, !torch.int -> !torch.vtensor<[1,1,?],i1>
    torch.bind_symbolic_shape %85, [%63], affine_map<()[s0] -> (1, 1, s0 * 32)> : !torch.vtensor<[1,1,?],i1>
    %int2_38 = torch.constant.int 2
    %86 = torch.aten.unsqueeze %85, %int2_38 : !torch.vtensor<[1,1,?],i1>, !torch.int -> !torch.vtensor<[1,1,1,?],i1>
    torch.bind_symbolic_shape %86, [%63], affine_map<()[s0] -> (1, 1, 1, s0 * 32)> : !torch.vtensor<[1,1,1,?],i1>
    %int3_39 = torch.constant.int 3
    %int0_40 = torch.constant.int 0
    %int9223372036854775807_41 = torch.constant.int 9223372036854775807
    %int1_42 = torch.constant.int 1
    %87 = torch.aten.slice.Tensor %86, %int3_39, %int0_40, %int9223372036854775807_41, %int1_42 : !torch.vtensor<[1,1,1,?],i1>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,1,1,?],i1>
    torch.bind_symbolic_shape %87, [%63], affine_map<()[s0] -> (1, 1, 1, s0 * 32)> : !torch.vtensor<[1,1,1,?],i1>
    %88 = torch.aten.logical_or %83, %87 : !torch.vtensor<[1,1,?,?],i1>, !torch.vtensor<[1,1,1,?],i1> -> !torch.vtensor<[1,1,?,?],i1>
    torch.bind_symbolic_shape %88, [%63], affine_map<()[s0] -> (1, 1, s0 * 32, s0 * 32)> : !torch.vtensor<[1,1,?,?],i1>
    %int0_43 = torch.constant.int 0
    %int6 = torch.constant.int 6
    %int0_44 = torch.constant.int 0
    %cpu_45 = torch.constant.device "cpu"
    %none_46 = torch.constant.none
    %89 = torch.aten.scalar_tensor %int0_43, %int6, %int0_44, %cpu_45, %none_46 : !torch.int, !torch.int, !torch.int, !torch.Device, !torch.none -> !torch.vtensor<[],f32>
    %float-Inf = torch.constant.float 0xFFF0000000000000
    %int6_47 = torch.constant.int 6
    %int0_48 = torch.constant.int 0
    %cpu_49 = torch.constant.device "cpu"
    %none_50 = torch.constant.none
    %90 = torch.aten.scalar_tensor %float-Inf, %int6_47, %int0_48, %cpu_49, %none_50 : !torch.float, !torch.int, !torch.int, !torch.Device, !torch.none -> !torch.vtensor<[],f32>
    %91 = torch.aten.where.self %88, %90, %89 : !torch.vtensor<[1,1,?,?],i1>, !torch.vtensor<[],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[1,1,?,?],f32>
    torch.bind_symbolic_shape %91, [%63], affine_map<()[s0] -> (1, 1, s0 * 32, s0 * 32)> : !torch.vtensor<[1,1,?,?],f32>
    %int5 = torch.constant.int 5
    %92 = torch.prims.convert_element_type %91, %int5 : !torch.vtensor<[1,1,?,?],f32>, !torch.int -> !torch.vtensor<[1,1,?,?],f16>
    torch.bind_symbolic_shape %92, [%63], affine_map<()[s0] -> (1, 1, s0 * 32, s0 * 32)> : !torch.vtensor<[1,1,?,?],f16>
    %int5_51 = torch.constant.int 5
    %93 = torch.prims.convert_element_type %92, %int5_51 : !torch.vtensor<[1,1,?,?],f16>, !torch.int -> !torch.vtensor<[1,1,?,?],f16>
    torch.bind_symbolic_shape %93, [%63], affine_map<()[s0] -> (1, 1, s0 * 32, s0 * 32)> : !torch.vtensor<[1,1,?,?],f16>
    %94 = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[1,?],si64> -> tensor<1x?xi64>
    %c1 = arith.constant 1 : index
    %dim = tensor.dim %94, %c1 : tensor<1x?xi64>
    %95 = flow.tensor.transfer %94 : tensor<1x?xi64>{%dim} to #hal.device.promise<@__device_0>
    %96 = torch_c.from_builtin_tensor %95 : tensor<1x?xi64> -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %96, [%63], affine_map<()[s0] -> (1, s0 * 32)> : !torch.vtensor<[1,?],si64>
    %97 = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[1,?],si64> -> tensor<1x?xi64>
    %c1_52 = arith.constant 1 : index
    %dim_53 = tensor.dim %97, %c1_52 : tensor<1x?xi64>
    %98 = flow.tensor.transfer %97 : tensor<1x?xi64>{%dim_53} to #hal.device.promise<@__device_1>
    %99 = torch_c.from_builtin_tensor %98 : tensor<1x?xi64> -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %99, [%63], affine_map<()[s0] -> (1, s0 * 32)> : !torch.vtensor<[1,?],si64>
    %100 = torch_c.to_builtin_tensor %93 : !torch.vtensor<[1,1,?,?],f16> -> tensor<1x1x?x?xf16>
    %c2 = arith.constant 2 : index
    %dim_54 = tensor.dim %100, %c2 : tensor<1x1x?x?xf16>
    %c3 = arith.constant 3 : index
    %dim_55 = tensor.dim %100, %c3 : tensor<1x1x?x?xf16>
    %101 = flow.tensor.transfer %100 : tensor<1x1x?x?xf16>{%dim_54, %dim_55} to #hal.device.promise<@__device_0>
    %102 = torch_c.from_builtin_tensor %101 : tensor<1x1x?x?xf16> -> !torch.vtensor<[1,1,?,?],f16>
    torch.bind_symbolic_shape %102, [%63], affine_map<()[s0] -> (1, 1, s0 * 32, s0 * 32)> : !torch.vtensor<[1,1,?,?],f16>
    %103 = torch_c.to_builtin_tensor %93 : !torch.vtensor<[1,1,?,?],f16> -> tensor<1x1x?x?xf16>
    %c2_56 = arith.constant 2 : index
    %dim_57 = tensor.dim %103, %c2_56 : tensor<1x1x?x?xf16>
    %c3_58 = arith.constant 3 : index
    %dim_59 = tensor.dim %103, %c3_58 : tensor<1x1x?x?xf16>
    %104 = flow.tensor.transfer %103 : tensor<1x1x?x?xf16>{%dim_57, %dim_59} to #hal.device.promise<@__device_1>
    %105 = torch_c.from_builtin_tensor %104 : tensor<1x1x?x?xf16> -> !torch.vtensor<[1,1,?,?],f16>
    torch.bind_symbolic_shape %105, [%63], affine_map<()[s0] -> (1, 1, s0 * 32, s0 * 32)> : !torch.vtensor<[1,1,?,?],f16>
    %106 = torch_c.to_builtin_tensor %arg2 : !torch.vtensor<[1,?],si64> -> tensor<1x?xi64>
    %c1_60 = arith.constant 1 : index
    %dim_61 = tensor.dim %106, %c1_60 : tensor<1x?xi64>
    %107 = flow.tensor.transfer %106 : tensor<1x?xi64>{%dim_61} to #hal.device.promise<@__device_0>
    %108 = torch_c.from_builtin_tensor %107 : tensor<1x?xi64> -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %108, [%63], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %109 = torch_c.to_builtin_tensor %arg2 : !torch.vtensor<[1,?],si64> -> tensor<1x?xi64>
    %c1_62 = arith.constant 1 : index
    %dim_63 = tensor.dim %109, %c1_62 : tensor<1x?xi64>
    %110 = flow.tensor.transfer %109 : tensor<1x?xi64>{%dim_63} to #hal.device.promise<@__device_1>
    %111 = torch_c.from_builtin_tensor %110 : tensor<1x?xi64> -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %111, [%63], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %int5_64 = torch.constant.int 5
    %112 = torch.prims.convert_element_type %0, %int5_64 : !torch.vtensor<[256,256],f32>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int-1_65 = torch.constant.int -1
    %false_66 = torch.constant.bool false
    %false_67 = torch.constant.bool false
    %113 = torch.aten.embedding %112, %96, %int-1_65, %false_66, %false_67 : !torch.vtensor<[256,256],f16>, !torch.vtensor<[1,?],si64>, !torch.int, !torch.bool, !torch.bool -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %113, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int5_68 = torch.constant.int 5
    %114 = torch.prims.convert_element_type %1, %int5_68 : !torch.vtensor<[256,256],f32>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int-1_69 = torch.constant.int -1
    %false_70 = torch.constant.bool false
    %false_71 = torch.constant.bool false
    %115 = torch.aten.embedding %114, %99, %int-1_69, %false_70, %false_71 : !torch.vtensor<[256,256],f16>, !torch.vtensor<[1,?],si64>, !torch.int, !torch.bool, !torch.bool -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %115, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int6_72 = torch.constant.int 6
    %116 = torch.prims.convert_element_type %113, %int6_72 : !torch.vtensor<[1,?,256],f16>, !torch.int -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %116, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int6_73 = torch.constant.int 6
    %117 = torch.prims.convert_element_type %115, %int6_73 : !torch.vtensor<[1,?,256],f16>, !torch.int -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %117, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int2_74 = torch.constant.int 2
    %118 = torch.aten.pow.Tensor_Scalar %116, %int2_74 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %118, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int2_75 = torch.constant.int 2
    %119 = torch.aten.pow.Tensor_Scalar %117, %int2_75 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %119, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int-1_76 = torch.constant.int -1
    %120 = torch.prim.ListConstruct %int-1_76 : (!torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %none_77 = torch.constant.none
    %121 = torch.aten.mean.dim %118, %120, %true, %none_77 : !torch.vtensor<[1,?,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %121, [%63], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %int-1_78 = torch.constant.int -1
    %122 = torch.prim.ListConstruct %int-1_78 : (!torch.int) -> !torch.list<int>
    %true_79 = torch.constant.bool true
    %none_80 = torch.constant.none
    %123 = torch.aten.mean.dim %119, %122, %true_79, %none_80 : !torch.vtensor<[1,?,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %123, [%63], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %float1.000000e-02 = torch.constant.float 1.000000e-02
    %int1_81 = torch.constant.int 1
    %124 = torch.aten.add.Scalar %121, %float1.000000e-02, %int1_81 : !torch.vtensor<[1,?,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %124, [%63], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %float1.000000e-02_82 = torch.constant.float 1.000000e-02
    %int1_83 = torch.constant.int 1
    %125 = torch.aten.add.Scalar %123, %float1.000000e-02_82, %int1_83 : !torch.vtensor<[1,?,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %125, [%63], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %126 = torch.aten.rsqrt %124 : !torch.vtensor<[1,?,1],f32> -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %126, [%63], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %127 = torch.aten.rsqrt %125 : !torch.vtensor<[1,?,1],f32> -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %127, [%63], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %128 = torch.aten.mul.Tensor %116, %126 : !torch.vtensor<[1,?,256],f32>, !torch.vtensor<[1,?,1],f32> -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %128, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %129 = torch.aten.mul.Tensor %117, %127 : !torch.vtensor<[1,?,256],f32>, !torch.vtensor<[1,?,1],f32> -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %129, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int5_84 = torch.constant.int 5
    %130 = torch.prims.convert_element_type %128, %int5_84 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %130, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int5_85 = torch.constant.int 5
    %131 = torch.prims.convert_element_type %129, %int5_85 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %131, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %132 = torch.aten.mul.Tensor %2, %130 : !torch.vtensor<[256],f32>, !torch.vtensor<[1,?,256],f16> -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %132, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %133 = torch.aten.mul.Tensor %3, %131 : !torch.vtensor<[256],f32>, !torch.vtensor<[1,?,256],f16> -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %133, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int5_86 = torch.constant.int 5
    %134 = torch.prims.convert_element_type %132, %int5_86 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %134, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int5_87 = torch.constant.int 5
    %135 = torch.prims.convert_element_type %133, %int5_87 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %135, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int1_88 = torch.constant.int 1
    %int0_89 = torch.constant.int 0
    %136 = torch.prim.ListConstruct %int1_88, %int0_89 : (!torch.int, !torch.int) -> !torch.list<int>
    %137 = torch.aten.permute %4, %136 : !torch.vtensor<[128,256],f32>, !torch.list<int> -> !torch.vtensor<[256,128],f32>
    %int1_90 = torch.constant.int 1
    %int0_91 = torch.constant.int 0
    %138 = torch.prim.ListConstruct %int1_90, %int0_91 : (!torch.int, !torch.int) -> !torch.list<int>
    %139 = torch.aten.permute %5, %138 : !torch.vtensor<[128,256],f32>, !torch.list<int> -> !torch.vtensor<[256,128],f32>
    %int5_92 = torch.constant.int 5
    %140 = torch.prims.convert_element_type %137, %int5_92 : !torch.vtensor<[256,128],f32>, !torch.int -> !torch.vtensor<[256,128],f16>
    %int256 = torch.constant.int 256
    %141 = torch.prim.ListConstruct %67, %int256 : (!torch.int, !torch.int) -> !torch.list<int>
    %142 = torch.aten.view %134, %141 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %142, [%63], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %143 = torch.aten.mm %142, %140 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,128],f16> -> !torch.vtensor<[?,128],f16>
    torch.bind_symbolic_shape %143, [%63], affine_map<()[s0] -> (s0 * 32, 128)> : !torch.vtensor<[?,128],f16>
    %int1_93 = torch.constant.int 1
    %int128_94 = torch.constant.int 128
    %144 = torch.prim.ListConstruct %int1_93, %67, %int128_94 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %145 = torch.aten.view %143, %144 : !torch.vtensor<[?,128],f16>, !torch.list<int> -> !torch.vtensor<[1,?,128],f16>
    torch.bind_symbolic_shape %145, [%63], affine_map<()[s0] -> (1, s0 * 32, 128)> : !torch.vtensor<[1,?,128],f16>
    %int5_95 = torch.constant.int 5
    %146 = torch.prims.convert_element_type %139, %int5_95 : !torch.vtensor<[256,128],f32>, !torch.int -> !torch.vtensor<[256,128],f16>
    %int256_96 = torch.constant.int 256
    %147 = torch.prim.ListConstruct %67, %int256_96 : (!torch.int, !torch.int) -> !torch.list<int>
    %148 = torch.aten.view %135, %147 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %148, [%63], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %149 = torch.aten.mm %148, %146 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,128],f16> -> !torch.vtensor<[?,128],f16>
    torch.bind_symbolic_shape %149, [%63], affine_map<()[s0] -> (s0 * 32, 128)> : !torch.vtensor<[?,128],f16>
    %int1_97 = torch.constant.int 1
    %int128_98 = torch.constant.int 128
    %150 = torch.prim.ListConstruct %int1_97, %67, %int128_98 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %151 = torch.aten.view %149, %150 : !torch.vtensor<[?,128],f16>, !torch.list<int> -> !torch.vtensor<[1,?,128],f16>
    torch.bind_symbolic_shape %151, [%63], affine_map<()[s0] -> (1, s0 * 32, 128)> : !torch.vtensor<[1,?,128],f16>
    %int1_99 = torch.constant.int 1
    %int0_100 = torch.constant.int 0
    %152 = torch.prim.ListConstruct %int1_99, %int0_100 : (!torch.int, !torch.int) -> !torch.list<int>
    %153 = torch.aten.permute %6, %152 : !torch.vtensor<[64,256],f32>, !torch.list<int> -> !torch.vtensor<[256,64],f32>
    %int1_101 = torch.constant.int 1
    %int0_102 = torch.constant.int 0
    %154 = torch.prim.ListConstruct %int1_101, %int0_102 : (!torch.int, !torch.int) -> !torch.list<int>
    %155 = torch.aten.permute %7, %154 : !torch.vtensor<[64,256],f32>, !torch.list<int> -> !torch.vtensor<[256,64],f32>
    %int5_103 = torch.constant.int 5
    %156 = torch.prims.convert_element_type %153, %int5_103 : !torch.vtensor<[256,64],f32>, !torch.int -> !torch.vtensor<[256,64],f16>
    %int256_104 = torch.constant.int 256
    %157 = torch.prim.ListConstruct %67, %int256_104 : (!torch.int, !torch.int) -> !torch.list<int>
    %158 = torch.aten.view %134, %157 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %158, [%63], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %159 = torch.aten.mm %158, %156 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,64],f16> -> !torch.vtensor<[?,64],f16>
    torch.bind_symbolic_shape %159, [%63], affine_map<()[s0] -> (s0 * 32, 64)> : !torch.vtensor<[?,64],f16>
    %int1_105 = torch.constant.int 1
    %int64 = torch.constant.int 64
    %160 = torch.prim.ListConstruct %int1_105, %67, %int64 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %161 = torch.aten.view %159, %160 : !torch.vtensor<[?,64],f16>, !torch.list<int> -> !torch.vtensor<[1,?,64],f16>
    torch.bind_symbolic_shape %161, [%63], affine_map<()[s0] -> (1, s0 * 32, 64)> : !torch.vtensor<[1,?,64],f16>
    %int5_106 = torch.constant.int 5
    %162 = torch.prims.convert_element_type %155, %int5_106 : !torch.vtensor<[256,64],f32>, !torch.int -> !torch.vtensor<[256,64],f16>
    %int256_107 = torch.constant.int 256
    %163 = torch.prim.ListConstruct %67, %int256_107 : (!torch.int, !torch.int) -> !torch.list<int>
    %164 = torch.aten.view %135, %163 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %164, [%63], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %165 = torch.aten.mm %164, %162 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,64],f16> -> !torch.vtensor<[?,64],f16>
    torch.bind_symbolic_shape %165, [%63], affine_map<()[s0] -> (s0 * 32, 64)> : !torch.vtensor<[?,64],f16>
    %int1_108 = torch.constant.int 1
    %int64_109 = torch.constant.int 64
    %166 = torch.prim.ListConstruct %int1_108, %67, %int64_109 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %167 = torch.aten.view %165, %166 : !torch.vtensor<[?,64],f16>, !torch.list<int> -> !torch.vtensor<[1,?,64],f16>
    torch.bind_symbolic_shape %167, [%63], affine_map<()[s0] -> (1, s0 * 32, 64)> : !torch.vtensor<[1,?,64],f16>
    %int1_110 = torch.constant.int 1
    %int0_111 = torch.constant.int 0
    %168 = torch.prim.ListConstruct %int1_110, %int0_111 : (!torch.int, !torch.int) -> !torch.list<int>
    %169 = torch.aten.permute %8, %168 : !torch.vtensor<[64,256],f32>, !torch.list<int> -> !torch.vtensor<[256,64],f32>
    %int1_112 = torch.constant.int 1
    %int0_113 = torch.constant.int 0
    %170 = torch.prim.ListConstruct %int1_112, %int0_113 : (!torch.int, !torch.int) -> !torch.list<int>
    %171 = torch.aten.permute %9, %170 : !torch.vtensor<[64,256],f32>, !torch.list<int> -> !torch.vtensor<[256,64],f32>
    %int5_114 = torch.constant.int 5
    %172 = torch.prims.convert_element_type %169, %int5_114 : !torch.vtensor<[256,64],f32>, !torch.int -> !torch.vtensor<[256,64],f16>
    %int256_115 = torch.constant.int 256
    %173 = torch.prim.ListConstruct %67, %int256_115 : (!torch.int, !torch.int) -> !torch.list<int>
    %174 = torch.aten.view %134, %173 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %174, [%63], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %175 = torch.aten.mm %174, %172 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,64],f16> -> !torch.vtensor<[?,64],f16>
    torch.bind_symbolic_shape %175, [%63], affine_map<()[s0] -> (s0 * 32, 64)> : !torch.vtensor<[?,64],f16>
    %int1_116 = torch.constant.int 1
    %int64_117 = torch.constant.int 64
    %176 = torch.prim.ListConstruct %int1_116, %67, %int64_117 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %177 = torch.aten.view %175, %176 : !torch.vtensor<[?,64],f16>, !torch.list<int> -> !torch.vtensor<[1,?,64],f16>
    torch.bind_symbolic_shape %177, [%63], affine_map<()[s0] -> (1, s0 * 32, 64)> : !torch.vtensor<[1,?,64],f16>
    %int5_118 = torch.constant.int 5
    %178 = torch.prims.convert_element_type %171, %int5_118 : !torch.vtensor<[256,64],f32>, !torch.int -> !torch.vtensor<[256,64],f16>
    %int256_119 = torch.constant.int 256
    %179 = torch.prim.ListConstruct %67, %int256_119 : (!torch.int, !torch.int) -> !torch.list<int>
    %180 = torch.aten.view %135, %179 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %180, [%63], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %181 = torch.aten.mm %180, %178 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,64],f16> -> !torch.vtensor<[?,64],f16>
    torch.bind_symbolic_shape %181, [%63], affine_map<()[s0] -> (s0 * 32, 64)> : !torch.vtensor<[?,64],f16>
    %int1_120 = torch.constant.int 1
    %int64_121 = torch.constant.int 64
    %182 = torch.prim.ListConstruct %int1_120, %67, %int64_121 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %183 = torch.aten.view %181, %182 : !torch.vtensor<[?,64],f16>, !torch.list<int> -> !torch.vtensor<[1,?,64],f16>
    torch.bind_symbolic_shape %183, [%63], affine_map<()[s0] -> (1, s0 * 32, 64)> : !torch.vtensor<[1,?,64],f16>
    %int1_122 = torch.constant.int 1
    %int4 = torch.constant.int 4
    %int32 = torch.constant.int 32
    %184 = torch.prim.ListConstruct %int1_122, %67, %int4, %int32 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %185 = torch.aten.view %145, %184 : !torch.vtensor<[1,?,128],f16>, !torch.list<int> -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %185, [%63], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int1_123 = torch.constant.int 1
    %int4_124 = torch.constant.int 4
    %int32_125 = torch.constant.int 32
    %186 = torch.prim.ListConstruct %int1_123, %67, %int4_124, %int32_125 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %187 = torch.aten.view %151, %186 : !torch.vtensor<[1,?,128],f16>, !torch.list<int> -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %187, [%63], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int1_126 = torch.constant.int 1
    %int2_127 = torch.constant.int 2
    %int32_128 = torch.constant.int 32
    %188 = torch.prim.ListConstruct %int1_126, %67, %int2_127, %int32_128 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %189 = torch.aten.view %161, %188 : !torch.vtensor<[1,?,64],f16>, !torch.list<int> -> !torch.vtensor<[1,?,2,32],f16>
    torch.bind_symbolic_shape %189, [%63], affine_map<()[s0] -> (1, s0 * 32, 2, 32)> : !torch.vtensor<[1,?,2,32],f16>
    %int1_129 = torch.constant.int 1
    %int2_130 = torch.constant.int 2
    %int32_131 = torch.constant.int 32
    %190 = torch.prim.ListConstruct %int1_129, %67, %int2_130, %int32_131 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %191 = torch.aten.view %167, %190 : !torch.vtensor<[1,?,64],f16>, !torch.list<int> -> !torch.vtensor<[1,?,2,32],f16>
    torch.bind_symbolic_shape %191, [%63], affine_map<()[s0] -> (1, s0 * 32, 2, 32)> : !torch.vtensor<[1,?,2,32],f16>
    %int1_132 = torch.constant.int 1
    %int2_133 = torch.constant.int 2
    %int32_134 = torch.constant.int 32
    %192 = torch.prim.ListConstruct %int1_132, %67, %int2_133, %int32_134 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %193 = torch.aten.view %177, %192 : !torch.vtensor<[1,?,64],f16>, !torch.list<int> -> !torch.vtensor<[1,?,2,32],f16>
    torch.bind_symbolic_shape %193, [%63], affine_map<()[s0] -> (1, s0 * 32, 2, 32)> : !torch.vtensor<[1,?,2,32],f16>
    %int1_135 = torch.constant.int 1
    %int2_136 = torch.constant.int 2
    %int32_137 = torch.constant.int 32
    %194 = torch.prim.ListConstruct %int1_135, %67, %int2_136, %int32_137 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %195 = torch.aten.view %183, %194 : !torch.vtensor<[1,?,64],f16>, !torch.list<int> -> !torch.vtensor<[1,?,2,32],f16>
    torch.bind_symbolic_shape %195, [%63], affine_map<()[s0] -> (1, s0 * 32, 2, 32)> : !torch.vtensor<[1,?,2,32],f16>
    %int128_138 = torch.constant.int 128
    %none_139 = torch.constant.none
    %none_140 = torch.constant.none
    %cpu_141 = torch.constant.device "cpu"
    %false_142 = torch.constant.bool false
    %196 = torch.aten.arange %int128_138, %none_139, %none_140, %cpu_141, %false_142 : !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[128],si64>
    %int0_143 = torch.constant.int 0
    %int32_144 = torch.constant.int 32
    %none_145 = torch.constant.none
    %none_146 = torch.constant.none
    %cpu_147 = torch.constant.device "cpu"
    %false_148 = torch.constant.bool false
    %197 = torch.aten.arange.start %int0_143, %int32_144, %none_145, %none_146, %cpu_147, %false_148 : !torch.int, !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[32],si64>
    %int2_149 = torch.constant.int 2
    %198 = torch.aten.floor_divide.Scalar %197, %int2_149 : !torch.vtensor<[32],si64>, !torch.int -> !torch.vtensor<[32],si64>
    %int6_150 = torch.constant.int 6
    %199 = torch.prims.convert_element_type %198, %int6_150 : !torch.vtensor<[32],si64>, !torch.int -> !torch.vtensor<[32],f32>
    %int32_151 = torch.constant.int 32
    %200 = torch.aten.div.Scalar %199, %int32_151 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[32],f32>
    %float2.000000e00 = torch.constant.float 2.000000e+00
    %201 = torch.aten.mul.Scalar %200, %float2.000000e00 : !torch.vtensor<[32],f32>, !torch.float -> !torch.vtensor<[32],f32>
    %float5.000000e05 = torch.constant.float 5.000000e+05
    %202 = torch.aten.pow.Scalar %float5.000000e05, %201 : !torch.float, !torch.vtensor<[32],f32> -> !torch.vtensor<[32],f32>
    %203 = torch.aten.reciprocal %202 : !torch.vtensor<[32],f32> -> !torch.vtensor<[32],f32>
    %float1.000000e00 = torch.constant.float 1.000000e+00
    %204 = torch.aten.mul.Scalar %203, %float1.000000e00 : !torch.vtensor<[32],f32>, !torch.float -> !torch.vtensor<[32],f32>
    %int1_152 = torch.constant.int 1
    %205 = torch.aten.unsqueeze %196, %int1_152 : !torch.vtensor<[128],si64>, !torch.int -> !torch.vtensor<[128,1],si64>
    %int0_153 = torch.constant.int 0
    %206 = torch.aten.unsqueeze %204, %int0_153 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[1,32],f32>
    %207 = torch.aten.mul.Tensor %205, %206 : !torch.vtensor<[128,1],si64>, !torch.vtensor<[1,32],f32> -> !torch.vtensor<[128,32],f32>
    %int6_154 = torch.constant.int 6
    %208 = torch.prims.convert_element_type %207, %int6_154 : !torch.vtensor<[128,32],f32>, !torch.int -> !torch.vtensor<[128,32],f32>
    %209 = torch_c.to_builtin_tensor %208 : !torch.vtensor<[128,32],f32> -> tensor<128x32xf32>
    %210 = flow.tensor.transfer %209 : tensor<128x32xf32> to #hal.device.promise<@__device_0>
    %211 = torch_c.from_builtin_tensor %210 : tensor<128x32xf32> -> !torch.vtensor<[128,32],f32>
    %212 = torch_c.to_builtin_tensor %208 : !torch.vtensor<[128,32],f32> -> tensor<128x32xf32>
    %213 = flow.tensor.transfer %212 : tensor<128x32xf32> to #hal.device.promise<@__device_1>
    %214 = torch_c.from_builtin_tensor %213 : tensor<128x32xf32> -> !torch.vtensor<[128,32],f32>
    %int0_155 = torch.constant.int 0
    %int0_156 = torch.constant.int 0
    %int1_157 = torch.constant.int 1
    %215 = torch.aten.slice.Tensor %211, %int0_155, %int0_156, %67, %int1_157 : !torch.vtensor<[128,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %215, [%63], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int1_158 = torch.constant.int 1
    %int0_159 = torch.constant.int 0
    %int9223372036854775807_160 = torch.constant.int 9223372036854775807
    %int1_161 = torch.constant.int 1
    %216 = torch.aten.slice.Tensor %215, %int1_158, %int0_159, %int9223372036854775807_160, %int1_161 : !torch.vtensor<[?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %216, [%63], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int1_162 = torch.constant.int 1
    %int0_163 = torch.constant.int 0
    %int9223372036854775807_164 = torch.constant.int 9223372036854775807
    %int1_165 = torch.constant.int 1
    %217 = torch.aten.slice.Tensor %216, %int1_162, %int0_163, %int9223372036854775807_164, %int1_165 : !torch.vtensor<[?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %217, [%63], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int0_166 = torch.constant.int 0
    %218 = torch.aten.unsqueeze %217, %int0_166 : !torch.vtensor<[?,32],f32>, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %218, [%63], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int1_167 = torch.constant.int 1
    %int0_168 = torch.constant.int 0
    %int9223372036854775807_169 = torch.constant.int 9223372036854775807
    %int1_170 = torch.constant.int 1
    %219 = torch.aten.slice.Tensor %218, %int1_167, %int0_168, %int9223372036854775807_169, %int1_170 : !torch.vtensor<[1,?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %219, [%63], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int2_171 = torch.constant.int 2
    %int0_172 = torch.constant.int 0
    %int9223372036854775807_173 = torch.constant.int 9223372036854775807
    %int1_174 = torch.constant.int 1
    %220 = torch.aten.slice.Tensor %219, %int2_171, %int0_172, %int9223372036854775807_173, %int1_174 : !torch.vtensor<[1,?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %220, [%63], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int1_175 = torch.constant.int 1
    %int1_176 = torch.constant.int 1
    %int1_177 = torch.constant.int 1
    %221 = torch.prim.ListConstruct %int1_175, %int1_176, %int1_177 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %222 = torch.aten.repeat %220, %221 : !torch.vtensor<[1,?,32],f32>, !torch.list<int> -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %222, [%63], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int6_178 = torch.constant.int 6
    %223 = torch.prims.convert_element_type %185, %int6_178 : !torch.vtensor<[1,?,4,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,32],f32>
    torch.bind_symbolic_shape %223, [%63], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f32>
    %224 = torch_c.to_builtin_tensor %223 : !torch.vtensor<[1,?,4,32],f32> -> tensor<1x?x4x32xf32>
    %225 = torch_c.to_builtin_tensor %222 : !torch.vtensor<[1,?,32],f32> -> tensor<1x?x32xf32>
    %226 = util.call @sharktank_rotary_embedding_1_D_4_32_f32(%224, %225) : (tensor<1x?x4x32xf32>, tensor<1x?x32xf32>) -> tensor<1x?x4x32xf32>
    %227 = torch_c.from_builtin_tensor %226 : tensor<1x?x4x32xf32> -> !torch.vtensor<[1,?,4,32],f32>
    torch.bind_symbolic_shape %227, [%63], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f32>
    %int5_179 = torch.constant.int 5
    %228 = torch.prims.convert_element_type %227, %int5_179 : !torch.vtensor<[1,?,4,32],f32>, !torch.int -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %228, [%63], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int0_180 = torch.constant.int 0
    %int0_181 = torch.constant.int 0
    %int1_182 = torch.constant.int 1
    %229 = torch.aten.slice.Tensor %214, %int0_180, %int0_181, %67, %int1_182 : !torch.vtensor<[128,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %229, [%63], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int1_183 = torch.constant.int 1
    %int0_184 = torch.constant.int 0
    %int9223372036854775807_185 = torch.constant.int 9223372036854775807
    %int1_186 = torch.constant.int 1
    %230 = torch.aten.slice.Tensor %229, %int1_183, %int0_184, %int9223372036854775807_185, %int1_186 : !torch.vtensor<[?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %230, [%63], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int1_187 = torch.constant.int 1
    %int0_188 = torch.constant.int 0
    %int9223372036854775807_189 = torch.constant.int 9223372036854775807
    %int1_190 = torch.constant.int 1
    %231 = torch.aten.slice.Tensor %230, %int1_187, %int0_188, %int9223372036854775807_189, %int1_190 : !torch.vtensor<[?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %231, [%63], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int0_191 = torch.constant.int 0
    %232 = torch.aten.unsqueeze %231, %int0_191 : !torch.vtensor<[?,32],f32>, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %232, [%63], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int1_192 = torch.constant.int 1
    %int0_193 = torch.constant.int 0
    %int9223372036854775807_194 = torch.constant.int 9223372036854775807
    %int1_195 = torch.constant.int 1
    %233 = torch.aten.slice.Tensor %232, %int1_192, %int0_193, %int9223372036854775807_194, %int1_195 : !torch.vtensor<[1,?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %233, [%63], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int2_196 = torch.constant.int 2
    %int0_197 = torch.constant.int 0
    %int9223372036854775807_198 = torch.constant.int 9223372036854775807
    %int1_199 = torch.constant.int 1
    %234 = torch.aten.slice.Tensor %233, %int2_196, %int0_197, %int9223372036854775807_198, %int1_199 : !torch.vtensor<[1,?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %234, [%63], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int1_200 = torch.constant.int 1
    %int1_201 = torch.constant.int 1
    %int1_202 = torch.constant.int 1
    %235 = torch.prim.ListConstruct %int1_200, %int1_201, %int1_202 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %236 = torch.aten.repeat %234, %235 : !torch.vtensor<[1,?,32],f32>, !torch.list<int> -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %236, [%63], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int6_203 = torch.constant.int 6
    %237 = torch.prims.convert_element_type %187, %int6_203 : !torch.vtensor<[1,?,4,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,32],f32>
    torch.bind_symbolic_shape %237, [%63], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f32>
    %238 = torch_c.to_builtin_tensor %237 : !torch.vtensor<[1,?,4,32],f32> -> tensor<1x?x4x32xf32>
    %239 = torch_c.to_builtin_tensor %236 : !torch.vtensor<[1,?,32],f32> -> tensor<1x?x32xf32>
    %240 = util.call @sharktank_rotary_embedding_1_D_4_32_f32(%238, %239) : (tensor<1x?x4x32xf32>, tensor<1x?x32xf32>) -> tensor<1x?x4x32xf32>
    %241 = torch_c.from_builtin_tensor %240 : tensor<1x?x4x32xf32> -> !torch.vtensor<[1,?,4,32],f32>
    torch.bind_symbolic_shape %241, [%63], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f32>
    %int5_204 = torch.constant.int 5
    %242 = torch.prims.convert_element_type %241, %int5_204 : !torch.vtensor<[1,?,4,32],f32>, !torch.int -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %242, [%63], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int128_205 = torch.constant.int 128
    %none_206 = torch.constant.none
    %none_207 = torch.constant.none
    %cpu_208 = torch.constant.device "cpu"
    %false_209 = torch.constant.bool false
    %243 = torch.aten.arange %int128_205, %none_206, %none_207, %cpu_208, %false_209 : !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[128],si64>
    %int0_210 = torch.constant.int 0
    %int32_211 = torch.constant.int 32
    %none_212 = torch.constant.none
    %none_213 = torch.constant.none
    %cpu_214 = torch.constant.device "cpu"
    %false_215 = torch.constant.bool false
    %244 = torch.aten.arange.start %int0_210, %int32_211, %none_212, %none_213, %cpu_214, %false_215 : !torch.int, !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[32],si64>
    %int2_216 = torch.constant.int 2
    %245 = torch.aten.floor_divide.Scalar %244, %int2_216 : !torch.vtensor<[32],si64>, !torch.int -> !torch.vtensor<[32],si64>
    %int6_217 = torch.constant.int 6
    %246 = torch.prims.convert_element_type %245, %int6_217 : !torch.vtensor<[32],si64>, !torch.int -> !torch.vtensor<[32],f32>
    %int32_218 = torch.constant.int 32
    %247 = torch.aten.div.Scalar %246, %int32_218 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[32],f32>
    %float2.000000e00_219 = torch.constant.float 2.000000e+00
    %248 = torch.aten.mul.Scalar %247, %float2.000000e00_219 : !torch.vtensor<[32],f32>, !torch.float -> !torch.vtensor<[32],f32>
    %float5.000000e05_220 = torch.constant.float 5.000000e+05
    %249 = torch.aten.pow.Scalar %float5.000000e05_220, %248 : !torch.float, !torch.vtensor<[32],f32> -> !torch.vtensor<[32],f32>
    %250 = torch.aten.reciprocal %249 : !torch.vtensor<[32],f32> -> !torch.vtensor<[32],f32>
    %float1.000000e00_221 = torch.constant.float 1.000000e+00
    %251 = torch.aten.mul.Scalar %250, %float1.000000e00_221 : !torch.vtensor<[32],f32>, !torch.float -> !torch.vtensor<[32],f32>
    %int1_222 = torch.constant.int 1
    %252 = torch.aten.unsqueeze %243, %int1_222 : !torch.vtensor<[128],si64>, !torch.int -> !torch.vtensor<[128,1],si64>
    %int0_223 = torch.constant.int 0
    %253 = torch.aten.unsqueeze %251, %int0_223 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[1,32],f32>
    %254 = torch.aten.mul.Tensor %252, %253 : !torch.vtensor<[128,1],si64>, !torch.vtensor<[1,32],f32> -> !torch.vtensor<[128,32],f32>
    %int6_224 = torch.constant.int 6
    %255 = torch.prims.convert_element_type %254, %int6_224 : !torch.vtensor<[128,32],f32>, !torch.int -> !torch.vtensor<[128,32],f32>
    %256 = torch_c.to_builtin_tensor %255 : !torch.vtensor<[128,32],f32> -> tensor<128x32xf32>
    %257 = flow.tensor.transfer %256 : tensor<128x32xf32> to #hal.device.promise<@__device_0>
    %258 = torch_c.from_builtin_tensor %257 : tensor<128x32xf32> -> !torch.vtensor<[128,32],f32>
    %259 = torch_c.to_builtin_tensor %255 : !torch.vtensor<[128,32],f32> -> tensor<128x32xf32>
    %260 = flow.tensor.transfer %259 : tensor<128x32xf32> to #hal.device.promise<@__device_1>
    %261 = torch_c.from_builtin_tensor %260 : tensor<128x32xf32> -> !torch.vtensor<[128,32],f32>
    %int0_225 = torch.constant.int 0
    %int0_226 = torch.constant.int 0
    %int1_227 = torch.constant.int 1
    %262 = torch.aten.slice.Tensor %258, %int0_225, %int0_226, %67, %int1_227 : !torch.vtensor<[128,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %262, [%63], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int1_228 = torch.constant.int 1
    %int0_229 = torch.constant.int 0
    %int9223372036854775807_230 = torch.constant.int 9223372036854775807
    %int1_231 = torch.constant.int 1
    %263 = torch.aten.slice.Tensor %262, %int1_228, %int0_229, %int9223372036854775807_230, %int1_231 : !torch.vtensor<[?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %263, [%63], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int1_232 = torch.constant.int 1
    %int0_233 = torch.constant.int 0
    %int9223372036854775807_234 = torch.constant.int 9223372036854775807
    %int1_235 = torch.constant.int 1
    %264 = torch.aten.slice.Tensor %263, %int1_232, %int0_233, %int9223372036854775807_234, %int1_235 : !torch.vtensor<[?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %264, [%63], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int0_236 = torch.constant.int 0
    %265 = torch.aten.unsqueeze %264, %int0_236 : !torch.vtensor<[?,32],f32>, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %265, [%63], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int1_237 = torch.constant.int 1
    %int0_238 = torch.constant.int 0
    %int9223372036854775807_239 = torch.constant.int 9223372036854775807
    %int1_240 = torch.constant.int 1
    %266 = torch.aten.slice.Tensor %265, %int1_237, %int0_238, %int9223372036854775807_239, %int1_240 : !torch.vtensor<[1,?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %266, [%63], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int2_241 = torch.constant.int 2
    %int0_242 = torch.constant.int 0
    %int9223372036854775807_243 = torch.constant.int 9223372036854775807
    %int1_244 = torch.constant.int 1
    %267 = torch.aten.slice.Tensor %266, %int2_241, %int0_242, %int9223372036854775807_243, %int1_244 : !torch.vtensor<[1,?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %267, [%63], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int1_245 = torch.constant.int 1
    %int1_246 = torch.constant.int 1
    %int1_247 = torch.constant.int 1
    %268 = torch.prim.ListConstruct %int1_245, %int1_246, %int1_247 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %269 = torch.aten.repeat %267, %268 : !torch.vtensor<[1,?,32],f32>, !torch.list<int> -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %269, [%63], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int6_248 = torch.constant.int 6
    %270 = torch.prims.convert_element_type %189, %int6_248 : !torch.vtensor<[1,?,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,2,32],f32>
    torch.bind_symbolic_shape %270, [%63], affine_map<()[s0] -> (1, s0 * 32, 2, 32)> : !torch.vtensor<[1,?,2,32],f32>
    %271 = torch_c.to_builtin_tensor %270 : !torch.vtensor<[1,?,2,32],f32> -> tensor<1x?x2x32xf32>
    %272 = torch_c.to_builtin_tensor %269 : !torch.vtensor<[1,?,32],f32> -> tensor<1x?x32xf32>
    %273 = util.call @sharktank_rotary_embedding_1_D_2_32_f32(%271, %272) : (tensor<1x?x2x32xf32>, tensor<1x?x32xf32>) -> tensor<1x?x2x32xf32>
    %274 = torch_c.from_builtin_tensor %273 : tensor<1x?x2x32xf32> -> !torch.vtensor<[1,?,2,32],f32>
    torch.bind_symbolic_shape %274, [%63], affine_map<()[s0] -> (1, s0 * 32, 2, 32)> : !torch.vtensor<[1,?,2,32],f32>
    %int5_249 = torch.constant.int 5
    %275 = torch.prims.convert_element_type %274, %int5_249 : !torch.vtensor<[1,?,2,32],f32>, !torch.int -> !torch.vtensor<[1,?,2,32],f16>
    torch.bind_symbolic_shape %275, [%63], affine_map<()[s0] -> (1, s0 * 32, 2, 32)> : !torch.vtensor<[1,?,2,32],f16>
    %int0_250 = torch.constant.int 0
    %int0_251 = torch.constant.int 0
    %int1_252 = torch.constant.int 1
    %276 = torch.aten.slice.Tensor %261, %int0_250, %int0_251, %67, %int1_252 : !torch.vtensor<[128,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %276, [%63], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int1_253 = torch.constant.int 1
    %int0_254 = torch.constant.int 0
    %int9223372036854775807_255 = torch.constant.int 9223372036854775807
    %int1_256 = torch.constant.int 1
    %277 = torch.aten.slice.Tensor %276, %int1_253, %int0_254, %int9223372036854775807_255, %int1_256 : !torch.vtensor<[?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %277, [%63], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int1_257 = torch.constant.int 1
    %int0_258 = torch.constant.int 0
    %int9223372036854775807_259 = torch.constant.int 9223372036854775807
    %int1_260 = torch.constant.int 1
    %278 = torch.aten.slice.Tensor %277, %int1_257, %int0_258, %int9223372036854775807_259, %int1_260 : !torch.vtensor<[?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %278, [%63], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int0_261 = torch.constant.int 0
    %279 = torch.aten.unsqueeze %278, %int0_261 : !torch.vtensor<[?,32],f32>, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %279, [%63], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int1_262 = torch.constant.int 1
    %int0_263 = torch.constant.int 0
    %int9223372036854775807_264 = torch.constant.int 9223372036854775807
    %int1_265 = torch.constant.int 1
    %280 = torch.aten.slice.Tensor %279, %int1_262, %int0_263, %int9223372036854775807_264, %int1_265 : !torch.vtensor<[1,?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %280, [%63], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int2_266 = torch.constant.int 2
    %int0_267 = torch.constant.int 0
    %int9223372036854775807_268 = torch.constant.int 9223372036854775807
    %int1_269 = torch.constant.int 1
    %281 = torch.aten.slice.Tensor %280, %int2_266, %int0_267, %int9223372036854775807_268, %int1_269 : !torch.vtensor<[1,?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %281, [%63], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int1_270 = torch.constant.int 1
    %int1_271 = torch.constant.int 1
    %int1_272 = torch.constant.int 1
    %282 = torch.prim.ListConstruct %int1_270, %int1_271, %int1_272 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %283 = torch.aten.repeat %281, %282 : !torch.vtensor<[1,?,32],f32>, !torch.list<int> -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %283, [%63], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int6_273 = torch.constant.int 6
    %284 = torch.prims.convert_element_type %191, %int6_273 : !torch.vtensor<[1,?,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,2,32],f32>
    torch.bind_symbolic_shape %284, [%63], affine_map<()[s0] -> (1, s0 * 32, 2, 32)> : !torch.vtensor<[1,?,2,32],f32>
    %285 = torch_c.to_builtin_tensor %284 : !torch.vtensor<[1,?,2,32],f32> -> tensor<1x?x2x32xf32>
    %286 = torch_c.to_builtin_tensor %283 : !torch.vtensor<[1,?,32],f32> -> tensor<1x?x32xf32>
    %287 = util.call @sharktank_rotary_embedding_1_D_2_32_f32(%285, %286) : (tensor<1x?x2x32xf32>, tensor<1x?x32xf32>) -> tensor<1x?x2x32xf32>
    %288 = torch_c.from_builtin_tensor %287 : tensor<1x?x2x32xf32> -> !torch.vtensor<[1,?,2,32],f32>
    torch.bind_symbolic_shape %288, [%63], affine_map<()[s0] -> (1, s0 * 32, 2, 32)> : !torch.vtensor<[1,?,2,32],f32>
    %int5_274 = torch.constant.int 5
    %289 = torch.prims.convert_element_type %288, %int5_274 : !torch.vtensor<[1,?,2,32],f32>, !torch.int -> !torch.vtensor<[1,?,2,32],f16>
    torch.bind_symbolic_shape %289, [%63], affine_map<()[s0] -> (1, s0 * 32, 2, 32)> : !torch.vtensor<[1,?,2,32],f16>
    %int3_275 = torch.constant.int 3
    %int2_276 = torch.constant.int 2
    %int32_277 = torch.constant.int 32
    %int2_278 = torch.constant.int 2
    %int32_279 = torch.constant.int 32
    %290 = torch.prim.ListConstruct %66, %int3_275, %int2_276, %int32_277, %int2_278, %int32_279 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %291 = torch.aten.view %60, %290 : !torch.vtensor<[?,12288],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,2,32],f16>
    torch.bind_symbolic_shape %291, [%64], affine_map<()[s0] -> (s0, 3, 2, 32, 2, 32)> : !torch.vtensor<[?,3,2,32,2,32],f16>
    %int3_280 = torch.constant.int 3
    %int2_281 = torch.constant.int 2
    %int32_282 = torch.constant.int 32
    %int2_283 = torch.constant.int 2
    %int32_284 = torch.constant.int 32
    %292 = torch.prim.ListConstruct %66, %int3_280, %int2_281, %int32_282, %int2_283, %int32_284 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %293 = torch.aten.view %61, %292 : !torch.vtensor<[?,12288],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,2,32],f16>
    torch.bind_symbolic_shape %293, [%64], affine_map<()[s0] -> (s0, 3, 2, 32, 2, 32)> : !torch.vtensor<[?,3,2,32,2,32],f16>
    %int3_285 = torch.constant.int 3
    %294 = torch.aten.mul.int %66, %int3_285 : !torch.int, !torch.int -> !torch.int
    %int2_286 = torch.constant.int 2
    %295 = torch.aten.mul.int %294, %int2_286 : !torch.int, !torch.int -> !torch.int
    %int32_287 = torch.constant.int 32
    %int2_288 = torch.constant.int 2
    %int32_289 = torch.constant.int 32
    %296 = torch.prim.ListConstruct %295, %int32_287, %int2_288, %int32_289 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %297 = torch.aten.view %291, %296 : !torch.vtensor<[?,3,2,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,32,2,32],f16>
    torch.bind_symbolic_shape %297, [%64], affine_map<()[s0] -> (s0 * 6, 32, 2, 32)> : !torch.vtensor<[?,32,2,32],f16>
    %int32_290 = torch.constant.int 32
    %int2_291 = torch.constant.int 2
    %int32_292 = torch.constant.int 32
    %298 = torch.prim.ListConstruct %295, %int32_290, %int2_291, %int32_292 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %299 = torch.aten.view %293, %298 : !torch.vtensor<[?,3,2,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,32,2,32],f16>
    torch.bind_symbolic_shape %299, [%64], affine_map<()[s0] -> (s0 * 6, 32, 2, 32)> : !torch.vtensor<[?,32,2,32],f16>
    %int6_293 = torch.constant.int 6
    %300 = torch.aten.mul.Scalar %108, %int6_293 : !torch.vtensor<[1,?],si64>, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %300, [%63], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %int6_294 = torch.constant.int 6
    %301 = torch.aten.mul.Scalar %111, %int6_294 : !torch.vtensor<[1,?],si64>, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %301, [%63], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %int0_295 = torch.constant.int 0
    %int1_296 = torch.constant.int 1
    %302 = torch.aten.add.Scalar %300, %int0_295, %int1_296 : !torch.vtensor<[1,?],si64>, !torch.int, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %302, [%63], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %int0_297 = torch.constant.int 0
    %int1_298 = torch.constant.int 1
    %303 = torch.aten.add.Scalar %301, %int0_297, %int1_298 : !torch.vtensor<[1,?],si64>, !torch.int, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %303, [%63], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %int1_299 = torch.constant.int 1
    %int32_300 = torch.constant.int 32
    %int2_301 = torch.constant.int 2
    %int32_302 = torch.constant.int 32
    %304 = torch.prim.ListConstruct %int1_299, %65, %int32_300, %int2_301, %int32_302 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %305 = torch.aten.view %275, %304 : !torch.vtensor<[1,?,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,32,2,32],f16>
    torch.bind_symbolic_shape %305, [%63], affine_map<()[s0] -> (1, s0, 32, 2, 32)> : !torch.vtensor<[1,?,32,2,32],f16>
    %int1_303 = torch.constant.int 1
    %int32_304 = torch.constant.int 32
    %int2_305 = torch.constant.int 2
    %int32_306 = torch.constant.int 32
    %306 = torch.prim.ListConstruct %int1_303, %65, %int32_304, %int2_305, %int32_306 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %307 = torch.aten.view %289, %306 : !torch.vtensor<[1,?,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,32,2,32],f16>
    torch.bind_symbolic_shape %307, [%63], affine_map<()[s0] -> (1, s0, 32, 2, 32)> : !torch.vtensor<[1,?,32,2,32],f16>
    %int32_307 = torch.constant.int 32
    %int2_308 = torch.constant.int 2
    %int32_309 = torch.constant.int 32
    %308 = torch.prim.ListConstruct %65, %int32_307, %int2_308, %int32_309 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %309 = torch.aten.view %305, %308 : !torch.vtensor<[1,?,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,32,2,32],f16>
    torch.bind_symbolic_shape %309, [%63], affine_map<()[s0] -> (s0, 32, 2, 32)> : !torch.vtensor<[?,32,2,32],f16>
    %int32_310 = torch.constant.int 32
    %int2_311 = torch.constant.int 2
    %int32_312 = torch.constant.int 32
    %310 = torch.prim.ListConstruct %65, %int32_310, %int2_311, %int32_312 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %311 = torch.aten.view %307, %310 : !torch.vtensor<[1,?,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,32,2,32],f16>
    torch.bind_symbolic_shape %311, [%63], affine_map<()[s0] -> (s0, 32, 2, 32)> : !torch.vtensor<[?,32,2,32],f16>
    %312 = torch.prim.ListConstruct %65 : (!torch.int) -> !torch.list<int>
    %313 = torch.aten.view %302, %312 : !torch.vtensor<[1,?],si64>, !torch.list<int> -> !torch.vtensor<[?],si64>
    torch.bind_symbolic_shape %313, [%63], affine_map<()[s0] -> (s0)> : !torch.vtensor<[?],si64>
    %314 = torch.prim.ListConstruct %65 : (!torch.int) -> !torch.list<int>
    %315 = torch.aten.view %303, %314 : !torch.vtensor<[1,?],si64>, !torch.list<int> -> !torch.vtensor<[?],si64>
    torch.bind_symbolic_shape %315, [%63], affine_map<()[s0] -> (s0)> : !torch.vtensor<[?],si64>
    %int5_313 = torch.constant.int 5
    %316 = torch.prims.convert_element_type %309, %int5_313 : !torch.vtensor<[?,32,2,32],f16>, !torch.int -> !torch.vtensor<[?,32,2,32],f16>
    torch.bind_symbolic_shape %316, [%63], affine_map<()[s0] -> (s0, 32, 2, 32)> : !torch.vtensor<[?,32,2,32],f16>
    %int5_314 = torch.constant.int 5
    %317 = torch.prims.convert_element_type %311, %int5_314 : !torch.vtensor<[?,32,2,32],f16>, !torch.int -> !torch.vtensor<[?,32,2,32],f16>
    torch.bind_symbolic_shape %317, [%63], affine_map<()[s0] -> (s0, 32, 2, 32)> : !torch.vtensor<[?,32,2,32],f16>
    %318 = torch.prim.ListConstruct %313 : (!torch.vtensor<[?],si64>) -> !torch.list<optional<vtensor>>
    %false_315 = torch.constant.bool false
    %319 = torch.aten.index_put %297, %318, %316, %false_315 : !torch.vtensor<[?,32,2,32],f16>, !torch.list<optional<vtensor>>, !torch.vtensor<[?,32,2,32],f16>, !torch.bool -> !torch.vtensor<[?,32,2,32],f16>
    torch.bind_symbolic_shape %319, [%64], affine_map<()[s0] -> (s0 * 6, 32, 2, 32)> : !torch.vtensor<[?,32,2,32],f16>
    %int3_316 = torch.constant.int 3
    %int2_317 = torch.constant.int 2
    %int32_318 = torch.constant.int 32
    %int2_319 = torch.constant.int 2
    %int32_320 = torch.constant.int 32
    %320 = torch.prim.ListConstruct %66, %int3_316, %int2_317, %int32_318, %int2_319, %int32_320 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %321 = torch.aten.view %319, %320 : !torch.vtensor<[?,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,2,32],f16>
    torch.bind_symbolic_shape %321, [%64], affine_map<()[s0] -> (s0, 3, 2, 32, 2, 32)> : !torch.vtensor<[?,3,2,32,2,32],f16>
    %int12288 = torch.constant.int 12288
    %322 = torch.prim.ListConstruct %66, %int12288 : (!torch.int, !torch.int) -> !torch.list<int>
    %323 = torch.aten.view %321, %322 : !torch.vtensor<[?,3,2,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,12288],f16>
    torch.bind_symbolic_shape %323, [%64], affine_map<()[s0] -> (s0, 12288)> : !torch.vtensor<[?,12288],f16>
    %int3_321 = torch.constant.int 3
    %int2_322 = torch.constant.int 2
    %int32_323 = torch.constant.int 32
    %int2_324 = torch.constant.int 2
    %int32_325 = torch.constant.int 32
    %324 = torch.prim.ListConstruct %66, %int3_321, %int2_322, %int32_323, %int2_324, %int32_325 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %325 = torch.aten.view %323, %324 : !torch.vtensor<[?,12288],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,2,32],f16>
    torch.bind_symbolic_shape %325, [%64], affine_map<()[s0] -> (s0, 3, 2, 32, 2, 32)> : !torch.vtensor<[?,3,2,32,2,32],f16>
    %int32_326 = torch.constant.int 32
    %int2_327 = torch.constant.int 2
    %int32_328 = torch.constant.int 32
    %326 = torch.prim.ListConstruct %295, %int32_326, %int2_327, %int32_328 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %327 = torch.aten.view %325, %326 : !torch.vtensor<[?,3,2,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,32,2,32],f16>
    torch.bind_symbolic_shape %327, [%64], affine_map<()[s0] -> (s0 * 6, 32, 2, 32)> : !torch.vtensor<[?,32,2,32],f16>
    %328 = torch.prim.ListConstruct %315 : (!torch.vtensor<[?],si64>) -> !torch.list<optional<vtensor>>
    %false_329 = torch.constant.bool false
    %329 = torch.aten.index_put %299, %328, %317, %false_329 : !torch.vtensor<[?,32,2,32],f16>, !torch.list<optional<vtensor>>, !torch.vtensor<[?,32,2,32],f16>, !torch.bool -> !torch.vtensor<[?,32,2,32],f16>
    torch.bind_symbolic_shape %329, [%64], affine_map<()[s0] -> (s0 * 6, 32, 2, 32)> : !torch.vtensor<[?,32,2,32],f16>
    %int3_330 = torch.constant.int 3
    %int2_331 = torch.constant.int 2
    %int32_332 = torch.constant.int 32
    %int2_333 = torch.constant.int 2
    %int32_334 = torch.constant.int 32
    %330 = torch.prim.ListConstruct %66, %int3_330, %int2_331, %int32_332, %int2_333, %int32_334 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %331 = torch.aten.view %329, %330 : !torch.vtensor<[?,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,2,32],f16>
    torch.bind_symbolic_shape %331, [%64], affine_map<()[s0] -> (s0, 3, 2, 32, 2, 32)> : !torch.vtensor<[?,3,2,32,2,32],f16>
    %int12288_335 = torch.constant.int 12288
    %332 = torch.prim.ListConstruct %66, %int12288_335 : (!torch.int, !torch.int) -> !torch.list<int>
    %333 = torch.aten.view %331, %332 : !torch.vtensor<[?,3,2,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,12288],f16>
    torch.bind_symbolic_shape %333, [%64], affine_map<()[s0] -> (s0, 12288)> : !torch.vtensor<[?,12288],f16>
    %int3_336 = torch.constant.int 3
    %int2_337 = torch.constant.int 2
    %int32_338 = torch.constant.int 32
    %int2_339 = torch.constant.int 2
    %int32_340 = torch.constant.int 32
    %334 = torch.prim.ListConstruct %66, %int3_336, %int2_337, %int32_338, %int2_339, %int32_340 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %335 = torch.aten.view %333, %334 : !torch.vtensor<[?,12288],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,2,32],f16>
    torch.bind_symbolic_shape %335, [%64], affine_map<()[s0] -> (s0, 3, 2, 32, 2, 32)> : !torch.vtensor<[?,3,2,32,2,32],f16>
    %int32_341 = torch.constant.int 32
    %int2_342 = torch.constant.int 2
    %int32_343 = torch.constant.int 32
    %336 = torch.prim.ListConstruct %295, %int32_341, %int2_342, %int32_343 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %337 = torch.aten.view %335, %336 : !torch.vtensor<[?,3,2,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,32,2,32],f16>
    torch.bind_symbolic_shape %337, [%64], affine_map<()[s0] -> (s0 * 6, 32, 2, 32)> : !torch.vtensor<[?,32,2,32],f16>
    %int1_344 = torch.constant.int 1
    %int32_345 = torch.constant.int 32
    %int2_346 = torch.constant.int 2
    %int32_347 = torch.constant.int 32
    %338 = torch.prim.ListConstruct %int1_344, %65, %int32_345, %int2_346, %int32_347 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %339 = torch.aten.view %193, %338 : !torch.vtensor<[1,?,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,32,2,32],f16>
    torch.bind_symbolic_shape %339, [%63], affine_map<()[s0] -> (1, s0, 32, 2, 32)> : !torch.vtensor<[1,?,32,2,32],f16>
    %int1_348 = torch.constant.int 1
    %int32_349 = torch.constant.int 32
    %int2_350 = torch.constant.int 2
    %int32_351 = torch.constant.int 32
    %340 = torch.prim.ListConstruct %int1_348, %65, %int32_349, %int2_350, %int32_351 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %341 = torch.aten.view %195, %340 : !torch.vtensor<[1,?,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,32,2,32],f16>
    torch.bind_symbolic_shape %341, [%63], affine_map<()[s0] -> (1, s0, 32, 2, 32)> : !torch.vtensor<[1,?,32,2,32],f16>
    %int32_352 = torch.constant.int 32
    %int2_353 = torch.constant.int 2
    %int32_354 = torch.constant.int 32
    %342 = torch.prim.ListConstruct %65, %int32_352, %int2_353, %int32_354 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %343 = torch.aten.view %339, %342 : !torch.vtensor<[1,?,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,32,2,32],f16>
    torch.bind_symbolic_shape %343, [%63], affine_map<()[s0] -> (s0, 32, 2, 32)> : !torch.vtensor<[?,32,2,32],f16>
    %int32_355 = torch.constant.int 32
    %int2_356 = torch.constant.int 2
    %int32_357 = torch.constant.int 32
    %344 = torch.prim.ListConstruct %65, %int32_355, %int2_356, %int32_357 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %345 = torch.aten.view %341, %344 : !torch.vtensor<[1,?,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,32,2,32],f16>
    torch.bind_symbolic_shape %345, [%63], affine_map<()[s0] -> (s0, 32, 2, 32)> : !torch.vtensor<[?,32,2,32],f16>
    %int1_358 = torch.constant.int 1
    %int1_359 = torch.constant.int 1
    %346 = torch.aten.add.Scalar %302, %int1_358, %int1_359 : !torch.vtensor<[1,?],si64>, !torch.int, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %346, [%63], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %int1_360 = torch.constant.int 1
    %int1_361 = torch.constant.int 1
    %347 = torch.aten.add.Scalar %303, %int1_360, %int1_361 : !torch.vtensor<[1,?],si64>, !torch.int, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %347, [%63], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %348 = torch.prim.ListConstruct %65 : (!torch.int) -> !torch.list<int>
    %349 = torch.aten.view %346, %348 : !torch.vtensor<[1,?],si64>, !torch.list<int> -> !torch.vtensor<[?],si64>
    torch.bind_symbolic_shape %349, [%63], affine_map<()[s0] -> (s0)> : !torch.vtensor<[?],si64>
    %350 = torch.prim.ListConstruct %65 : (!torch.int) -> !torch.list<int>
    %351 = torch.aten.view %347, %350 : !torch.vtensor<[1,?],si64>, !torch.list<int> -> !torch.vtensor<[?],si64>
    torch.bind_symbolic_shape %351, [%63], affine_map<()[s0] -> (s0)> : !torch.vtensor<[?],si64>
    %int5_362 = torch.constant.int 5
    %352 = torch.prims.convert_element_type %343, %int5_362 : !torch.vtensor<[?,32,2,32],f16>, !torch.int -> !torch.vtensor<[?,32,2,32],f16>
    torch.bind_symbolic_shape %352, [%63], affine_map<()[s0] -> (s0, 32, 2, 32)> : !torch.vtensor<[?,32,2,32],f16>
    %int5_363 = torch.constant.int 5
    %353 = torch.prims.convert_element_type %345, %int5_363 : !torch.vtensor<[?,32,2,32],f16>, !torch.int -> !torch.vtensor<[?,32,2,32],f16>
    torch.bind_symbolic_shape %353, [%63], affine_map<()[s0] -> (s0, 32, 2, 32)> : !torch.vtensor<[?,32,2,32],f16>
    %354 = torch.prim.ListConstruct %349 : (!torch.vtensor<[?],si64>) -> !torch.list<optional<vtensor>>
    %false_364 = torch.constant.bool false
    %355 = torch.aten.index_put %327, %354, %352, %false_364 : !torch.vtensor<[?,32,2,32],f16>, !torch.list<optional<vtensor>>, !torch.vtensor<[?,32,2,32],f16>, !torch.bool -> !torch.vtensor<[?,32,2,32],f16>
    torch.bind_symbolic_shape %355, [%64], affine_map<()[s0] -> (s0 * 6, 32, 2, 32)> : !torch.vtensor<[?,32,2,32],f16>
    %int3_365 = torch.constant.int 3
    %int2_366 = torch.constant.int 2
    %int32_367 = torch.constant.int 32
    %int2_368 = torch.constant.int 2
    %int32_369 = torch.constant.int 32
    %356 = torch.prim.ListConstruct %66, %int3_365, %int2_366, %int32_367, %int2_368, %int32_369 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %357 = torch.aten.view %355, %356 : !torch.vtensor<[?,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,2,32],f16>
    torch.bind_symbolic_shape %357, [%64], affine_map<()[s0] -> (s0, 3, 2, 32, 2, 32)> : !torch.vtensor<[?,3,2,32,2,32],f16>
    %int12288_370 = torch.constant.int 12288
    %358 = torch.prim.ListConstruct %66, %int12288_370 : (!torch.int, !torch.int) -> !torch.list<int>
    %359 = torch.aten.view %357, %358 : !torch.vtensor<[?,3,2,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,12288],f16>
    torch.bind_symbolic_shape %359, [%64], affine_map<()[s0] -> (s0, 12288)> : !torch.vtensor<[?,12288],f16>
    %360 = torch.prim.ListConstruct %351 : (!torch.vtensor<[?],si64>) -> !torch.list<optional<vtensor>>
    %false_371 = torch.constant.bool false
    %361 = torch.aten.index_put %337, %360, %353, %false_371 : !torch.vtensor<[?,32,2,32],f16>, !torch.list<optional<vtensor>>, !torch.vtensor<[?,32,2,32],f16>, !torch.bool -> !torch.vtensor<[?,32,2,32],f16>
    torch.bind_symbolic_shape %361, [%64], affine_map<()[s0] -> (s0 * 6, 32, 2, 32)> : !torch.vtensor<[?,32,2,32],f16>
    %int3_372 = torch.constant.int 3
    %int2_373 = torch.constant.int 2
    %int32_374 = torch.constant.int 32
    %int2_375 = torch.constant.int 2
    %int32_376 = torch.constant.int 32
    %362 = torch.prim.ListConstruct %66, %int3_372, %int2_373, %int32_374, %int2_375, %int32_376 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %363 = torch.aten.view %361, %362 : !torch.vtensor<[?,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,2,32],f16>
    torch.bind_symbolic_shape %363, [%64], affine_map<()[s0] -> (s0, 3, 2, 32, 2, 32)> : !torch.vtensor<[?,3,2,32,2,32],f16>
    %int12288_377 = torch.constant.int 12288
    %364 = torch.prim.ListConstruct %66, %int12288_377 : (!torch.int, !torch.int) -> !torch.list<int>
    %365 = torch.aten.view %363, %364 : !torch.vtensor<[?,3,2,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,12288],f16>
    torch.bind_symbolic_shape %365, [%64], affine_map<()[s0] -> (s0, 12288)> : !torch.vtensor<[?,12288],f16>
    %int-2 = torch.constant.int -2
    %366 = torch.aten.unsqueeze %275, %int-2 : !torch.vtensor<[1,?,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,2,1,32],f16>
    torch.bind_symbolic_shape %366, [%63], affine_map<()[s0] -> (1, s0 * 32, 2, 1, 32)> : !torch.vtensor<[1,?,2,1,32],f16>
    %int-2_378 = torch.constant.int -2
    %367 = torch.aten.unsqueeze %289, %int-2_378 : !torch.vtensor<[1,?,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,2,1,32],f16>
    torch.bind_symbolic_shape %367, [%63], affine_map<()[s0] -> (1, s0 * 32, 2, 1, 32)> : !torch.vtensor<[1,?,2,1,32],f16>
    %int1_379 = torch.constant.int 1
    %int2_380 = torch.constant.int 2
    %int2_381 = torch.constant.int 2
    %int32_382 = torch.constant.int 32
    %368 = torch.prim.ListConstruct %int1_379, %67, %int2_380, %int2_381, %int32_382 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_383 = torch.constant.bool false
    %369 = torch.aten.expand %366, %368, %false_383 : !torch.vtensor<[1,?,2,1,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,?,2,2,32],f16>
    torch.bind_symbolic_shape %369, [%63], affine_map<()[s0] -> (1, s0 * 32, 2, 2, 32)> : !torch.vtensor<[1,?,2,2,32],f16>
    %int1_384 = torch.constant.int 1
    %int2_385 = torch.constant.int 2
    %int2_386 = torch.constant.int 2
    %int32_387 = torch.constant.int 32
    %370 = torch.prim.ListConstruct %int1_384, %67, %int2_385, %int2_386, %int32_387 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_388 = torch.constant.bool false
    %371 = torch.aten.expand %367, %370, %false_388 : !torch.vtensor<[1,?,2,1,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,?,2,2,32],f16>
    torch.bind_symbolic_shape %371, [%63], affine_map<()[s0] -> (1, s0 * 32, 2, 2, 32)> : !torch.vtensor<[1,?,2,2,32],f16>
    %int0_389 = torch.constant.int 0
    %372 = torch.aten.clone %369, %int0_389 : !torch.vtensor<[1,?,2,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,2,2,32],f16>
    torch.bind_symbolic_shape %372, [%63], affine_map<()[s0] -> (1, s0 * 32, 2, 2, 32)> : !torch.vtensor<[1,?,2,2,32],f16>
    %int1_390 = torch.constant.int 1
    %int4_391 = torch.constant.int 4
    %int32_392 = torch.constant.int 32
    %373 = torch.prim.ListConstruct %int1_390, %67, %int4_391, %int32_392 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %374 = torch.aten._unsafe_view %372, %373 : !torch.vtensor<[1,?,2,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %374, [%63], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int0_393 = torch.constant.int 0
    %375 = torch.aten.clone %371, %int0_393 : !torch.vtensor<[1,?,2,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,2,2,32],f16>
    torch.bind_symbolic_shape %375, [%63], affine_map<()[s0] -> (1, s0 * 32, 2, 2, 32)> : !torch.vtensor<[1,?,2,2,32],f16>
    %int1_394 = torch.constant.int 1
    %int4_395 = torch.constant.int 4
    %int32_396 = torch.constant.int 32
    %376 = torch.prim.ListConstruct %int1_394, %67, %int4_395, %int32_396 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %377 = torch.aten._unsafe_view %375, %376 : !torch.vtensor<[1,?,2,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %377, [%63], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int-2_397 = torch.constant.int -2
    %378 = torch.aten.unsqueeze %193, %int-2_397 : !torch.vtensor<[1,?,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,2,1,32],f16>
    torch.bind_symbolic_shape %378, [%63], affine_map<()[s0] -> (1, s0 * 32, 2, 1, 32)> : !torch.vtensor<[1,?,2,1,32],f16>
    %int-2_398 = torch.constant.int -2
    %379 = torch.aten.unsqueeze %195, %int-2_398 : !torch.vtensor<[1,?,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,2,1,32],f16>
    torch.bind_symbolic_shape %379, [%63], affine_map<()[s0] -> (1, s0 * 32, 2, 1, 32)> : !torch.vtensor<[1,?,2,1,32],f16>
    %int1_399 = torch.constant.int 1
    %int2_400 = torch.constant.int 2
    %int2_401 = torch.constant.int 2
    %int32_402 = torch.constant.int 32
    %380 = torch.prim.ListConstruct %int1_399, %67, %int2_400, %int2_401, %int32_402 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_403 = torch.constant.bool false
    %381 = torch.aten.expand %378, %380, %false_403 : !torch.vtensor<[1,?,2,1,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,?,2,2,32],f16>
    torch.bind_symbolic_shape %381, [%63], affine_map<()[s0] -> (1, s0 * 32, 2, 2, 32)> : !torch.vtensor<[1,?,2,2,32],f16>
    %int1_404 = torch.constant.int 1
    %int2_405 = torch.constant.int 2
    %int2_406 = torch.constant.int 2
    %int32_407 = torch.constant.int 32
    %382 = torch.prim.ListConstruct %int1_404, %67, %int2_405, %int2_406, %int32_407 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_408 = torch.constant.bool false
    %383 = torch.aten.expand %379, %382, %false_408 : !torch.vtensor<[1,?,2,1,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,?,2,2,32],f16>
    torch.bind_symbolic_shape %383, [%63], affine_map<()[s0] -> (1, s0 * 32, 2, 2, 32)> : !torch.vtensor<[1,?,2,2,32],f16>
    %int0_409 = torch.constant.int 0
    %384 = torch.aten.clone %381, %int0_409 : !torch.vtensor<[1,?,2,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,2,2,32],f16>
    torch.bind_symbolic_shape %384, [%63], affine_map<()[s0] -> (1, s0 * 32, 2, 2, 32)> : !torch.vtensor<[1,?,2,2,32],f16>
    %int1_410 = torch.constant.int 1
    %int4_411 = torch.constant.int 4
    %int32_412 = torch.constant.int 32
    %385 = torch.prim.ListConstruct %int1_410, %67, %int4_411, %int32_412 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %386 = torch.aten._unsafe_view %384, %385 : !torch.vtensor<[1,?,2,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %386, [%63], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int0_413 = torch.constant.int 0
    %387 = torch.aten.clone %383, %int0_413 : !torch.vtensor<[1,?,2,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,2,2,32],f16>
    torch.bind_symbolic_shape %387, [%63], affine_map<()[s0] -> (1, s0 * 32, 2, 2, 32)> : !torch.vtensor<[1,?,2,2,32],f16>
    %int1_414 = torch.constant.int 1
    %int4_415 = torch.constant.int 4
    %int32_416 = torch.constant.int 32
    %388 = torch.prim.ListConstruct %int1_414, %67, %int4_415, %int32_416 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %389 = torch.aten._unsafe_view %387, %388 : !torch.vtensor<[1,?,2,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %389, [%63], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int1_417 = torch.constant.int 1
    %int2_418 = torch.constant.int 2
    %390 = torch.aten.transpose.int %228, %int1_417, %int2_418 : !torch.vtensor<[1,?,4,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,4,?,32],f16>
    torch.bind_symbolic_shape %390, [%63], affine_map<()[s0] -> (1, 4, s0 * 32, 32)> : !torch.vtensor<[1,4,?,32],f16>
    %int1_419 = torch.constant.int 1
    %int2_420 = torch.constant.int 2
    %391 = torch.aten.transpose.int %242, %int1_419, %int2_420 : !torch.vtensor<[1,?,4,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,4,?,32],f16>
    torch.bind_symbolic_shape %391, [%63], affine_map<()[s0] -> (1, 4, s0 * 32, 32)> : !torch.vtensor<[1,4,?,32],f16>
    %int1_421 = torch.constant.int 1
    %int2_422 = torch.constant.int 2
    %392 = torch.aten.transpose.int %374, %int1_421, %int2_422 : !torch.vtensor<[1,?,4,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,4,?,32],f16>
    torch.bind_symbolic_shape %392, [%63], affine_map<()[s0] -> (1, 4, s0 * 32, 32)> : !torch.vtensor<[1,4,?,32],f16>
    %int1_423 = torch.constant.int 1
    %int2_424 = torch.constant.int 2
    %393 = torch.aten.transpose.int %377, %int1_423, %int2_424 : !torch.vtensor<[1,?,4,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,4,?,32],f16>
    torch.bind_symbolic_shape %393, [%63], affine_map<()[s0] -> (1, 4, s0 * 32, 32)> : !torch.vtensor<[1,4,?,32],f16>
    %int1_425 = torch.constant.int 1
    %int2_426 = torch.constant.int 2
    %394 = torch.aten.transpose.int %386, %int1_425, %int2_426 : !torch.vtensor<[1,?,4,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,4,?,32],f16>
    torch.bind_symbolic_shape %394, [%63], affine_map<()[s0] -> (1, 4, s0 * 32, 32)> : !torch.vtensor<[1,4,?,32],f16>
    %int1_427 = torch.constant.int 1
    %int2_428 = torch.constant.int 2
    %395 = torch.aten.transpose.int %389, %int1_427, %int2_428 : !torch.vtensor<[1,?,4,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,4,?,32],f16>
    torch.bind_symbolic_shape %395, [%63], affine_map<()[s0] -> (1, 4, s0 * 32, 32)> : !torch.vtensor<[1,4,?,32],f16>
    %int5_429 = torch.constant.int 5
    %396 = torch.prims.convert_element_type %390, %int5_429 : !torch.vtensor<[1,4,?,32],f16>, !torch.int -> !torch.vtensor<[1,4,?,32],f16>
    torch.bind_symbolic_shape %396, [%63], affine_map<()[s0] -> (1, 4, s0 * 32, 32)> : !torch.vtensor<[1,4,?,32],f16>
    %int5_430 = torch.constant.int 5
    %397 = torch.prims.convert_element_type %391, %int5_430 : !torch.vtensor<[1,4,?,32],f16>, !torch.int -> !torch.vtensor<[1,4,?,32],f16>
    torch.bind_symbolic_shape %397, [%63], affine_map<()[s0] -> (1, 4, s0 * 32, 32)> : !torch.vtensor<[1,4,?,32],f16>
    %int5_431 = torch.constant.int 5
    %398 = torch.prims.convert_element_type %392, %int5_431 : !torch.vtensor<[1,4,?,32],f16>, !torch.int -> !torch.vtensor<[1,4,?,32],f16>
    torch.bind_symbolic_shape %398, [%63], affine_map<()[s0] -> (1, 4, s0 * 32, 32)> : !torch.vtensor<[1,4,?,32],f16>
    %int5_432 = torch.constant.int 5
    %399 = torch.prims.convert_element_type %393, %int5_432 : !torch.vtensor<[1,4,?,32],f16>, !torch.int -> !torch.vtensor<[1,4,?,32],f16>
    torch.bind_symbolic_shape %399, [%63], affine_map<()[s0] -> (1, 4, s0 * 32, 32)> : !torch.vtensor<[1,4,?,32],f16>
    %int5_433 = torch.constant.int 5
    %400 = torch.prims.convert_element_type %394, %int5_433 : !torch.vtensor<[1,4,?,32],f16>, !torch.int -> !torch.vtensor<[1,4,?,32],f16>
    torch.bind_symbolic_shape %400, [%63], affine_map<()[s0] -> (1, 4, s0 * 32, 32)> : !torch.vtensor<[1,4,?,32],f16>
    %int5_434 = torch.constant.int 5
    %401 = torch.prims.convert_element_type %395, %int5_434 : !torch.vtensor<[1,4,?,32],f16>, !torch.int -> !torch.vtensor<[1,4,?,32],f16>
    torch.bind_symbolic_shape %401, [%63], affine_map<()[s0] -> (1, 4, s0 * 32, 32)> : !torch.vtensor<[1,4,?,32],f16>
    %int5_435 = torch.constant.int 5
    %402 = torch.prims.convert_element_type %102, %int5_435 : !torch.vtensor<[1,1,?,?],f16>, !torch.int -> !torch.vtensor<[1,1,?,?],f16>
    torch.bind_symbolic_shape %402, [%63], affine_map<()[s0] -> (1, 1, s0 * 32, s0 * 32)> : !torch.vtensor<[1,1,?,?],f16>
    %int5_436 = torch.constant.int 5
    %403 = torch.prims.convert_element_type %105, %int5_436 : !torch.vtensor<[1,1,?,?],f16>, !torch.int -> !torch.vtensor<[1,1,?,?],f16>
    torch.bind_symbolic_shape %403, [%63], affine_map<()[s0] -> (1, 1, s0 * 32, s0 * 32)> : !torch.vtensor<[1,1,?,?],f16>
    %float0.000000e00 = torch.constant.float 0.000000e+00
    %false_437 = torch.constant.bool false
    %none_438 = torch.constant.none
    %404:2 = torch.operator "torch.aten._scaled_dot_product_flash_attention_for_cpu"(%396, %398, %400, %float0.000000e00, %false_437, %402, %none_438) : (!torch.vtensor<[1,4,?,32],f16>, !torch.vtensor<[1,4,?,32],f16>, !torch.vtensor<[1,4,?,32],f16>, !torch.float, !torch.bool, !torch.vtensor<[1,1,?,?],f16>, !torch.none) -> (!torch.vtensor<[1,4,?,32],f16>, !torch.vtensor<[1,4,?],f32>) 
    torch.bind_symbolic_shape %404#0, [%63], affine_map<()[s0] -> (1, 4, s0 * 32, 32)> : !torch.vtensor<[1,4,?,32],f16>
    %float0.000000e00_439 = torch.constant.float 0.000000e+00
    %false_440 = torch.constant.bool false
    %none_441 = torch.constant.none
    %405:2 = torch.operator "torch.aten._scaled_dot_product_flash_attention_for_cpu"(%397, %399, %401, %float0.000000e00_439, %false_440, %403, %none_441) : (!torch.vtensor<[1,4,?,32],f16>, !torch.vtensor<[1,4,?,32],f16>, !torch.vtensor<[1,4,?,32],f16>, !torch.float, !torch.bool, !torch.vtensor<[1,1,?,?],f16>, !torch.none) -> (!torch.vtensor<[1,4,?,32],f16>, !torch.vtensor<[1,4,?],f32>) 
    torch.bind_symbolic_shape %405#0, [%63], affine_map<()[s0] -> (1, 4, s0 * 32, 32)> : !torch.vtensor<[1,4,?,32],f16>
    %int1_442 = torch.constant.int 1
    %int2_443 = torch.constant.int 2
    %406 = torch.aten.transpose.int %404#0, %int1_442, %int2_443 : !torch.vtensor<[1,4,?,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %406, [%63], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int1_444 = torch.constant.int 1
    %int2_445 = torch.constant.int 2
    %407 = torch.aten.transpose.int %405#0, %int1_444, %int2_445 : !torch.vtensor<[1,4,?,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %407, [%63], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int1_446 = torch.constant.int 1
    %int128_447 = torch.constant.int 128
    %408 = torch.prim.ListConstruct %int1_446, %67, %int128_447 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %409 = torch.aten.view %406, %408 : !torch.vtensor<[1,?,4,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,128],f16>
    torch.bind_symbolic_shape %409, [%63], affine_map<()[s0] -> (1, s0 * 32, 128)> : !torch.vtensor<[1,?,128],f16>
    %int1_448 = torch.constant.int 1
    %int128_449 = torch.constant.int 128
    %410 = torch.prim.ListConstruct %int1_448, %67, %int128_449 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %411 = torch.aten.view %407, %410 : !torch.vtensor<[1,?,4,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,128],f16>
    torch.bind_symbolic_shape %411, [%63], affine_map<()[s0] -> (1, s0 * 32, 128)> : !torch.vtensor<[1,?,128],f16>
    %int1_450 = torch.constant.int 1
    %int0_451 = torch.constant.int 0
    %412 = torch.prim.ListConstruct %int1_450, %int0_451 : (!torch.int, !torch.int) -> !torch.list<int>
    %413 = torch.aten.permute %10, %412 : !torch.vtensor<[256,128],f32>, !torch.list<int> -> !torch.vtensor<[128,256],f32>
    %int1_452 = torch.constant.int 1
    %int0_453 = torch.constant.int 0
    %414 = torch.prim.ListConstruct %int1_452, %int0_453 : (!torch.int, !torch.int) -> !torch.list<int>
    %415 = torch.aten.permute %11, %414 : !torch.vtensor<[256,128],f32>, !torch.list<int> -> !torch.vtensor<[128,256],f32>
    %int5_454 = torch.constant.int 5
    %416 = torch.prims.convert_element_type %413, %int5_454 : !torch.vtensor<[128,256],f32>, !torch.int -> !torch.vtensor<[128,256],f16>
    %int128_455 = torch.constant.int 128
    %417 = torch.prim.ListConstruct %67, %int128_455 : (!torch.int, !torch.int) -> !torch.list<int>
    %418 = torch.aten.view %409, %417 : !torch.vtensor<[1,?,128],f16>, !torch.list<int> -> !torch.vtensor<[?,128],f16>
    torch.bind_symbolic_shape %418, [%63], affine_map<()[s0] -> (s0 * 32, 128)> : !torch.vtensor<[?,128],f16>
    %419 = torch.aten.mm %418, %416 : !torch.vtensor<[?,128],f16>, !torch.vtensor<[128,256],f16> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %419, [%63], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %int1_456 = torch.constant.int 1
    %int256_457 = torch.constant.int 256
    %420 = torch.prim.ListConstruct %int1_456, %67, %int256_457 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %421 = torch.aten.view %419, %420 : !torch.vtensor<[?,256],f16>, !torch.list<int> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %421, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int5_458 = torch.constant.int 5
    %422 = torch.prims.convert_element_type %415, %int5_458 : !torch.vtensor<[128,256],f32>, !torch.int -> !torch.vtensor<[128,256],f16>
    %int128_459 = torch.constant.int 128
    %423 = torch.prim.ListConstruct %67, %int128_459 : (!torch.int, !torch.int) -> !torch.list<int>
    %424 = torch.aten.view %411, %423 : !torch.vtensor<[1,?,128],f16>, !torch.list<int> -> !torch.vtensor<[?,128],f16>
    torch.bind_symbolic_shape %424, [%63], affine_map<()[s0] -> (s0 * 32, 128)> : !torch.vtensor<[?,128],f16>
    %425 = torch.aten.mm %424, %422 : !torch.vtensor<[?,128],f16>, !torch.vtensor<[128,256],f16> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %425, [%63], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %int1_460 = torch.constant.int 1
    %int256_461 = torch.constant.int 256
    %426 = torch.prim.ListConstruct %int1_460, %67, %int256_461 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %427 = torch.aten.view %425, %426 : !torch.vtensor<[?,256],f16>, !torch.list<int> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %427, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %428 = torch_c.to_builtin_tensor %421 : !torch.vtensor<[1,?,256],f16> -> tensor<1x?x256xf16>
    %c1_462 = arith.constant 1 : index
    %dim_463 = tensor.dim %428, %c1_462 : tensor<1x?x256xf16>
    %429 = flow.tensor.barrier %428 : tensor<1x?x256xf16>{%dim_463} on #hal.device.promise<@__device_0>
    %430 = torch_c.from_builtin_tensor %429 : tensor<1x?x256xf16> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %430, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %431 = torch_c.to_builtin_tensor %427 : !torch.vtensor<[1,?,256],f16> -> tensor<1x?x256xf16>
    %c1_464 = arith.constant 1 : index
    %dim_465 = tensor.dim %431, %c1_464 : tensor<1x?x256xf16>
    %432 = flow.tensor.transfer %431 : tensor<1x?x256xf16>{%dim_465} to #hal.device.promise<@__device_0>
    %433 = torch_c.from_builtin_tensor %432 : tensor<1x?x256xf16> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %433, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int1_466 = torch.constant.int 1
    %434 = torch.aten.add.Tensor %430, %433, %int1_466 : !torch.vtensor<[1,?,256],f16>, !torch.vtensor<[1,?,256],f16>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %434, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %435 = torch_c.to_builtin_tensor %434 : !torch.vtensor<[1,?,256],f16> -> tensor<1x?x256xf16>
    %c1_467 = arith.constant 1 : index
    %dim_468 = tensor.dim %435, %c1_467 : tensor<1x?x256xf16>
    %436 = flow.tensor.barrier %435 : tensor<1x?x256xf16>{%dim_468} on #hal.device.promise<@__device_0>
    %437 = torch_c.from_builtin_tensor %436 : tensor<1x?x256xf16> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %437, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %438 = torch_c.to_builtin_tensor %434 : !torch.vtensor<[1,?,256],f16> -> tensor<1x?x256xf16>
    %c1_469 = arith.constant 1 : index
    %dim_470 = tensor.dim %438, %c1_469 : tensor<1x?x256xf16>
    %439 = flow.tensor.transfer %438 : tensor<1x?x256xf16>{%dim_470} to #hal.device.promise<@__device_1>
    %440 = torch_c.from_builtin_tensor %439 : tensor<1x?x256xf16> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %440, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int1_471 = torch.constant.int 1
    %441 = torch.aten.add.Tensor %113, %437, %int1_471 : !torch.vtensor<[1,?,256],f16>, !torch.vtensor<[1,?,256],f16>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %441, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int1_472 = torch.constant.int 1
    %442 = torch.aten.add.Tensor %115, %440, %int1_472 : !torch.vtensor<[1,?,256],f16>, !torch.vtensor<[1,?,256],f16>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %442, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int6_473 = torch.constant.int 6
    %443 = torch.prims.convert_element_type %441, %int6_473 : !torch.vtensor<[1,?,256],f16>, !torch.int -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %443, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int6_474 = torch.constant.int 6
    %444 = torch.prims.convert_element_type %442, %int6_474 : !torch.vtensor<[1,?,256],f16>, !torch.int -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %444, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int2_475 = torch.constant.int 2
    %445 = torch.aten.pow.Tensor_Scalar %443, %int2_475 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %445, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int2_476 = torch.constant.int 2
    %446 = torch.aten.pow.Tensor_Scalar %444, %int2_476 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %446, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int-1_477 = torch.constant.int -1
    %447 = torch.prim.ListConstruct %int-1_477 : (!torch.int) -> !torch.list<int>
    %true_478 = torch.constant.bool true
    %none_479 = torch.constant.none
    %448 = torch.aten.mean.dim %445, %447, %true_478, %none_479 : !torch.vtensor<[1,?,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %448, [%63], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %int-1_480 = torch.constant.int -1
    %449 = torch.prim.ListConstruct %int-1_480 : (!torch.int) -> !torch.list<int>
    %true_481 = torch.constant.bool true
    %none_482 = torch.constant.none
    %450 = torch.aten.mean.dim %446, %449, %true_481, %none_482 : !torch.vtensor<[1,?,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %450, [%63], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %float1.000000e-02_483 = torch.constant.float 1.000000e-02
    %int1_484 = torch.constant.int 1
    %451 = torch.aten.add.Scalar %448, %float1.000000e-02_483, %int1_484 : !torch.vtensor<[1,?,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %451, [%63], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %float1.000000e-02_485 = torch.constant.float 1.000000e-02
    %int1_486 = torch.constant.int 1
    %452 = torch.aten.add.Scalar %450, %float1.000000e-02_485, %int1_486 : !torch.vtensor<[1,?,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %452, [%63], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %453 = torch.aten.rsqrt %451 : !torch.vtensor<[1,?,1],f32> -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %453, [%63], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %454 = torch.aten.rsqrt %452 : !torch.vtensor<[1,?,1],f32> -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %454, [%63], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %455 = torch.aten.mul.Tensor %443, %453 : !torch.vtensor<[1,?,256],f32>, !torch.vtensor<[1,?,1],f32> -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %455, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %456 = torch.aten.mul.Tensor %444, %454 : !torch.vtensor<[1,?,256],f32>, !torch.vtensor<[1,?,1],f32> -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %456, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int5_487 = torch.constant.int 5
    %457 = torch.prims.convert_element_type %455, %int5_487 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %457, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int5_488 = torch.constant.int 5
    %458 = torch.prims.convert_element_type %456, %int5_488 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %458, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %459 = torch.aten.mul.Tensor %12, %457 : !torch.vtensor<[256],f32>, !torch.vtensor<[1,?,256],f16> -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %459, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %460 = torch.aten.mul.Tensor %13, %458 : !torch.vtensor<[256],f32>, !torch.vtensor<[1,?,256],f16> -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %460, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int5_489 = torch.constant.int 5
    %461 = torch.prims.convert_element_type %459, %int5_489 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %461, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int5_490 = torch.constant.int 5
    %462 = torch.prims.convert_element_type %460, %int5_490 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %462, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int1_491 = torch.constant.int 1
    %int0_492 = torch.constant.int 0
    %463 = torch.prim.ListConstruct %int1_491, %int0_492 : (!torch.int, !torch.int) -> !torch.list<int>
    %464 = torch.aten.permute %14, %463 : !torch.vtensor<[12,256],f32>, !torch.list<int> -> !torch.vtensor<[256,12],f32>
    %int1_493 = torch.constant.int 1
    %int0_494 = torch.constant.int 0
    %465 = torch.prim.ListConstruct %int1_493, %int0_494 : (!torch.int, !torch.int) -> !torch.list<int>
    %466 = torch.aten.permute %15, %465 : !torch.vtensor<[11,256],f32>, !torch.list<int> -> !torch.vtensor<[256,11],f32>
    %int5_495 = torch.constant.int 5
    %467 = torch.prims.convert_element_type %464, %int5_495 : !torch.vtensor<[256,12],f32>, !torch.int -> !torch.vtensor<[256,12],f16>
    %int256_496 = torch.constant.int 256
    %468 = torch.prim.ListConstruct %67, %int256_496 : (!torch.int, !torch.int) -> !torch.list<int>
    %469 = torch.aten.view %461, %468 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %469, [%63], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %470 = torch.aten.mm %469, %467 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,12],f16> -> !torch.vtensor<[?,12],f16>
    torch.bind_symbolic_shape %470, [%63], affine_map<()[s0] -> (s0 * 32, 12)> : !torch.vtensor<[?,12],f16>
    %int1_497 = torch.constant.int 1
    %int12 = torch.constant.int 12
    %471 = torch.prim.ListConstruct %int1_497, %67, %int12 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %472 = torch.aten.view %470, %471 : !torch.vtensor<[?,12],f16>, !torch.list<int> -> !torch.vtensor<[1,?,12],f16>
    torch.bind_symbolic_shape %472, [%63], affine_map<()[s0] -> (1, s0 * 32, 12)> : !torch.vtensor<[1,?,12],f16>
    %int5_498 = torch.constant.int 5
    %473 = torch.prims.convert_element_type %466, %int5_498 : !torch.vtensor<[256,11],f32>, !torch.int -> !torch.vtensor<[256,11],f16>
    %int256_499 = torch.constant.int 256
    %474 = torch.prim.ListConstruct %67, %int256_499 : (!torch.int, !torch.int) -> !torch.list<int>
    %475 = torch.aten.view %462, %474 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %475, [%63], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %476 = torch.aten.mm %475, %473 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,11],f16> -> !torch.vtensor<[?,11],f16>
    torch.bind_symbolic_shape %476, [%63], affine_map<()[s0] -> (s0 * 32, 11)> : !torch.vtensor<[?,11],f16>
    %int1_500 = torch.constant.int 1
    %int11_501 = torch.constant.int 11
    %477 = torch.prim.ListConstruct %int1_500, %67, %int11_501 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %478 = torch.aten.view %476, %477 : !torch.vtensor<[?,11],f16>, !torch.list<int> -> !torch.vtensor<[1,?,11],f16>
    torch.bind_symbolic_shape %478, [%63], affine_map<()[s0] -> (1, s0 * 32, 11)> : !torch.vtensor<[1,?,11],f16>
    %479 = torch.aten.silu %472 : !torch.vtensor<[1,?,12],f16> -> !torch.vtensor<[1,?,12],f16>
    torch.bind_symbolic_shape %479, [%63], affine_map<()[s0] -> (1, s0 * 32, 12)> : !torch.vtensor<[1,?,12],f16>
    %480 = torch.aten.silu %478 : !torch.vtensor<[1,?,11],f16> -> !torch.vtensor<[1,?,11],f16>
    torch.bind_symbolic_shape %480, [%63], affine_map<()[s0] -> (1, s0 * 32, 11)> : !torch.vtensor<[1,?,11],f16>
    %int1_502 = torch.constant.int 1
    %int0_503 = torch.constant.int 0
    %481 = torch.prim.ListConstruct %int1_502, %int0_503 : (!torch.int, !torch.int) -> !torch.list<int>
    %482 = torch.aten.permute %16, %481 : !torch.vtensor<[12,256],f32>, !torch.list<int> -> !torch.vtensor<[256,12],f32>
    %int1_504 = torch.constant.int 1
    %int0_505 = torch.constant.int 0
    %483 = torch.prim.ListConstruct %int1_504, %int0_505 : (!torch.int, !torch.int) -> !torch.list<int>
    %484 = torch.aten.permute %17, %483 : !torch.vtensor<[11,256],f32>, !torch.list<int> -> !torch.vtensor<[256,11],f32>
    %int5_506 = torch.constant.int 5
    %485 = torch.prims.convert_element_type %482, %int5_506 : !torch.vtensor<[256,12],f32>, !torch.int -> !torch.vtensor<[256,12],f16>
    %int256_507 = torch.constant.int 256
    %486 = torch.prim.ListConstruct %67, %int256_507 : (!torch.int, !torch.int) -> !torch.list<int>
    %487 = torch.aten.view %461, %486 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %487, [%63], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %488 = torch.aten.mm %487, %485 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,12],f16> -> !torch.vtensor<[?,12],f16>
    torch.bind_symbolic_shape %488, [%63], affine_map<()[s0] -> (s0 * 32, 12)> : !torch.vtensor<[?,12],f16>
    %int1_508 = torch.constant.int 1
    %int12_509 = torch.constant.int 12
    %489 = torch.prim.ListConstruct %int1_508, %67, %int12_509 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %490 = torch.aten.view %488, %489 : !torch.vtensor<[?,12],f16>, !torch.list<int> -> !torch.vtensor<[1,?,12],f16>
    torch.bind_symbolic_shape %490, [%63], affine_map<()[s0] -> (1, s0 * 32, 12)> : !torch.vtensor<[1,?,12],f16>
    %int5_510 = torch.constant.int 5
    %491 = torch.prims.convert_element_type %484, %int5_510 : !torch.vtensor<[256,11],f32>, !torch.int -> !torch.vtensor<[256,11],f16>
    %int256_511 = torch.constant.int 256
    %492 = torch.prim.ListConstruct %67, %int256_511 : (!torch.int, !torch.int) -> !torch.list<int>
    %493 = torch.aten.view %462, %492 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %493, [%63], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %494 = torch.aten.mm %493, %491 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,11],f16> -> !torch.vtensor<[?,11],f16>
    torch.bind_symbolic_shape %494, [%63], affine_map<()[s0] -> (s0 * 32, 11)> : !torch.vtensor<[?,11],f16>
    %int1_512 = torch.constant.int 1
    %int11_513 = torch.constant.int 11
    %495 = torch.prim.ListConstruct %int1_512, %67, %int11_513 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %496 = torch.aten.view %494, %495 : !torch.vtensor<[?,11],f16>, !torch.list<int> -> !torch.vtensor<[1,?,11],f16>
    torch.bind_symbolic_shape %496, [%63], affine_map<()[s0] -> (1, s0 * 32, 11)> : !torch.vtensor<[1,?,11],f16>
    %497 = torch.aten.mul.Tensor %479, %490 : !torch.vtensor<[1,?,12],f16>, !torch.vtensor<[1,?,12],f16> -> !torch.vtensor<[1,?,12],f16>
    torch.bind_symbolic_shape %497, [%63], affine_map<()[s0] -> (1, s0 * 32, 12)> : !torch.vtensor<[1,?,12],f16>
    %498 = torch.aten.mul.Tensor %480, %496 : !torch.vtensor<[1,?,11],f16>, !torch.vtensor<[1,?,11],f16> -> !torch.vtensor<[1,?,11],f16>
    torch.bind_symbolic_shape %498, [%63], affine_map<()[s0] -> (1, s0 * 32, 11)> : !torch.vtensor<[1,?,11],f16>
    %int1_514 = torch.constant.int 1
    %int0_515 = torch.constant.int 0
    %499 = torch.prim.ListConstruct %int1_514, %int0_515 : (!torch.int, !torch.int) -> !torch.list<int>
    %500 = torch.aten.permute %18, %499 : !torch.vtensor<[256,12],f32>, !torch.list<int> -> !torch.vtensor<[12,256],f32>
    %int1_516 = torch.constant.int 1
    %int0_517 = torch.constant.int 0
    %501 = torch.prim.ListConstruct %int1_516, %int0_517 : (!torch.int, !torch.int) -> !torch.list<int>
    %502 = torch.aten.permute %19, %501 : !torch.vtensor<[256,11],f32>, !torch.list<int> -> !torch.vtensor<[11,256],f32>
    %int5_518 = torch.constant.int 5
    %503 = torch.prims.convert_element_type %500, %int5_518 : !torch.vtensor<[12,256],f32>, !torch.int -> !torch.vtensor<[12,256],f16>
    %int12_519 = torch.constant.int 12
    %504 = torch.prim.ListConstruct %67, %int12_519 : (!torch.int, !torch.int) -> !torch.list<int>
    %505 = torch.aten.view %497, %504 : !torch.vtensor<[1,?,12],f16>, !torch.list<int> -> !torch.vtensor<[?,12],f16>
    torch.bind_symbolic_shape %505, [%63], affine_map<()[s0] -> (s0 * 32, 12)> : !torch.vtensor<[?,12],f16>
    %506 = torch.aten.mm %505, %503 : !torch.vtensor<[?,12],f16>, !torch.vtensor<[12,256],f16> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %506, [%63], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %int1_520 = torch.constant.int 1
    %int256_521 = torch.constant.int 256
    %507 = torch.prim.ListConstruct %int1_520, %67, %int256_521 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %508 = torch.aten.view %506, %507 : !torch.vtensor<[?,256],f16>, !torch.list<int> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %508, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int5_522 = torch.constant.int 5
    %509 = torch.prims.convert_element_type %502, %int5_522 : !torch.vtensor<[11,256],f32>, !torch.int -> !torch.vtensor<[11,256],f16>
    %int11_523 = torch.constant.int 11
    %510 = torch.prim.ListConstruct %67, %int11_523 : (!torch.int, !torch.int) -> !torch.list<int>
    %511 = torch.aten.view %498, %510 : !torch.vtensor<[1,?,11],f16>, !torch.list<int> -> !torch.vtensor<[?,11],f16>
    torch.bind_symbolic_shape %511, [%63], affine_map<()[s0] -> (s0 * 32, 11)> : !torch.vtensor<[?,11],f16>
    %512 = torch.aten.mm %511, %509 : !torch.vtensor<[?,11],f16>, !torch.vtensor<[11,256],f16> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %512, [%63], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %int1_524 = torch.constant.int 1
    %int256_525 = torch.constant.int 256
    %513 = torch.prim.ListConstruct %int1_524, %67, %int256_525 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %514 = torch.aten.view %512, %513 : !torch.vtensor<[?,256],f16>, !torch.list<int> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %514, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %515 = torch_c.to_builtin_tensor %508 : !torch.vtensor<[1,?,256],f16> -> tensor<1x?x256xf16>
    %c1_526 = arith.constant 1 : index
    %dim_527 = tensor.dim %515, %c1_526 : tensor<1x?x256xf16>
    %516 = flow.tensor.barrier %515 : tensor<1x?x256xf16>{%dim_527} on #hal.device.promise<@__device_0>
    %517 = torch_c.from_builtin_tensor %516 : tensor<1x?x256xf16> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %517, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %518 = torch_c.to_builtin_tensor %514 : !torch.vtensor<[1,?,256],f16> -> tensor<1x?x256xf16>
    %c1_528 = arith.constant 1 : index
    %dim_529 = tensor.dim %518, %c1_528 : tensor<1x?x256xf16>
    %519 = flow.tensor.transfer %518 : tensor<1x?x256xf16>{%dim_529} to #hal.device.promise<@__device_0>
    %520 = torch_c.from_builtin_tensor %519 : tensor<1x?x256xf16> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %520, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int1_530 = torch.constant.int 1
    %521 = torch.aten.add.Tensor %517, %520, %int1_530 : !torch.vtensor<[1,?,256],f16>, !torch.vtensor<[1,?,256],f16>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %521, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %522 = torch_c.to_builtin_tensor %521 : !torch.vtensor<[1,?,256],f16> -> tensor<1x?x256xf16>
    %c1_531 = arith.constant 1 : index
    %dim_532 = tensor.dim %522, %c1_531 : tensor<1x?x256xf16>
    %523 = flow.tensor.barrier %522 : tensor<1x?x256xf16>{%dim_532} on #hal.device.promise<@__device_0>
    %524 = torch_c.from_builtin_tensor %523 : tensor<1x?x256xf16> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %524, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %525 = torch_c.to_builtin_tensor %521 : !torch.vtensor<[1,?,256],f16> -> tensor<1x?x256xf16>
    %c1_533 = arith.constant 1 : index
    %dim_534 = tensor.dim %525, %c1_533 : tensor<1x?x256xf16>
    %526 = flow.tensor.transfer %525 : tensor<1x?x256xf16>{%dim_534} to #hal.device.promise<@__device_1>
    %527 = torch_c.from_builtin_tensor %526 : tensor<1x?x256xf16> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %527, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int1_535 = torch.constant.int 1
    %528 = torch.aten.add.Tensor %441, %524, %int1_535 : !torch.vtensor<[1,?,256],f16>, !torch.vtensor<[1,?,256],f16>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %528, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int1_536 = torch.constant.int 1
    %529 = torch.aten.add.Tensor %442, %527, %int1_536 : !torch.vtensor<[1,?,256],f16>, !torch.vtensor<[1,?,256],f16>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %529, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int6_537 = torch.constant.int 6
    %530 = torch.prims.convert_element_type %528, %int6_537 : !torch.vtensor<[1,?,256],f16>, !torch.int -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %530, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int6_538 = torch.constant.int 6
    %531 = torch.prims.convert_element_type %529, %int6_538 : !torch.vtensor<[1,?,256],f16>, !torch.int -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %531, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int2_539 = torch.constant.int 2
    %532 = torch.aten.pow.Tensor_Scalar %530, %int2_539 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %532, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int2_540 = torch.constant.int 2
    %533 = torch.aten.pow.Tensor_Scalar %531, %int2_540 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %533, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int-1_541 = torch.constant.int -1
    %534 = torch.prim.ListConstruct %int-1_541 : (!torch.int) -> !torch.list<int>
    %true_542 = torch.constant.bool true
    %none_543 = torch.constant.none
    %535 = torch.aten.mean.dim %532, %534, %true_542, %none_543 : !torch.vtensor<[1,?,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %535, [%63], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %int-1_544 = torch.constant.int -1
    %536 = torch.prim.ListConstruct %int-1_544 : (!torch.int) -> !torch.list<int>
    %true_545 = torch.constant.bool true
    %none_546 = torch.constant.none
    %537 = torch.aten.mean.dim %533, %536, %true_545, %none_546 : !torch.vtensor<[1,?,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %537, [%63], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %float1.000000e-02_547 = torch.constant.float 1.000000e-02
    %int1_548 = torch.constant.int 1
    %538 = torch.aten.add.Scalar %535, %float1.000000e-02_547, %int1_548 : !torch.vtensor<[1,?,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %538, [%63], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %float1.000000e-02_549 = torch.constant.float 1.000000e-02
    %int1_550 = torch.constant.int 1
    %539 = torch.aten.add.Scalar %537, %float1.000000e-02_549, %int1_550 : !torch.vtensor<[1,?,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %539, [%63], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %540 = torch.aten.rsqrt %538 : !torch.vtensor<[1,?,1],f32> -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %540, [%63], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %541 = torch.aten.rsqrt %539 : !torch.vtensor<[1,?,1],f32> -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %541, [%63], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %542 = torch.aten.mul.Tensor %530, %540 : !torch.vtensor<[1,?,256],f32>, !torch.vtensor<[1,?,1],f32> -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %542, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %543 = torch.aten.mul.Tensor %531, %541 : !torch.vtensor<[1,?,256],f32>, !torch.vtensor<[1,?,1],f32> -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %543, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int5_551 = torch.constant.int 5
    %544 = torch.prims.convert_element_type %542, %int5_551 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %544, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int5_552 = torch.constant.int 5
    %545 = torch.prims.convert_element_type %543, %int5_552 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %545, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %546 = torch.aten.mul.Tensor %20, %544 : !torch.vtensor<[256],f32>, !torch.vtensor<[1,?,256],f16> -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %546, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %547 = torch.aten.mul.Tensor %21, %545 : !torch.vtensor<[256],f32>, !torch.vtensor<[1,?,256],f16> -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %547, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int5_553 = torch.constant.int 5
    %548 = torch.prims.convert_element_type %546, %int5_553 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %548, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int5_554 = torch.constant.int 5
    %549 = torch.prims.convert_element_type %547, %int5_554 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %549, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int1_555 = torch.constant.int 1
    %int0_556 = torch.constant.int 0
    %550 = torch.prim.ListConstruct %int1_555, %int0_556 : (!torch.int, !torch.int) -> !torch.list<int>
    %551 = torch.aten.permute %22, %550 : !torch.vtensor<[128,256],f32>, !torch.list<int> -> !torch.vtensor<[256,128],f32>
    %int1_557 = torch.constant.int 1
    %int0_558 = torch.constant.int 0
    %552 = torch.prim.ListConstruct %int1_557, %int0_558 : (!torch.int, !torch.int) -> !torch.list<int>
    %553 = torch.aten.permute %23, %552 : !torch.vtensor<[128,256],f32>, !torch.list<int> -> !torch.vtensor<[256,128],f32>
    %int5_559 = torch.constant.int 5
    %554 = torch.prims.convert_element_type %551, %int5_559 : !torch.vtensor<[256,128],f32>, !torch.int -> !torch.vtensor<[256,128],f16>
    %int256_560 = torch.constant.int 256
    %555 = torch.prim.ListConstruct %67, %int256_560 : (!torch.int, !torch.int) -> !torch.list<int>
    %556 = torch.aten.view %548, %555 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %556, [%63], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %557 = torch.aten.mm %556, %554 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,128],f16> -> !torch.vtensor<[?,128],f16>
    torch.bind_symbolic_shape %557, [%63], affine_map<()[s0] -> (s0 * 32, 128)> : !torch.vtensor<[?,128],f16>
    %int1_561 = torch.constant.int 1
    %int128_562 = torch.constant.int 128
    %558 = torch.prim.ListConstruct %int1_561, %67, %int128_562 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %559 = torch.aten.view %557, %558 : !torch.vtensor<[?,128],f16>, !torch.list<int> -> !torch.vtensor<[1,?,128],f16>
    torch.bind_symbolic_shape %559, [%63], affine_map<()[s0] -> (1, s0 * 32, 128)> : !torch.vtensor<[1,?,128],f16>
    %int5_563 = torch.constant.int 5
    %560 = torch.prims.convert_element_type %553, %int5_563 : !torch.vtensor<[256,128],f32>, !torch.int -> !torch.vtensor<[256,128],f16>
    %int256_564 = torch.constant.int 256
    %561 = torch.prim.ListConstruct %67, %int256_564 : (!torch.int, !torch.int) -> !torch.list<int>
    %562 = torch.aten.view %549, %561 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %562, [%63], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %563 = torch.aten.mm %562, %560 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,128],f16> -> !torch.vtensor<[?,128],f16>
    torch.bind_symbolic_shape %563, [%63], affine_map<()[s0] -> (s0 * 32, 128)> : !torch.vtensor<[?,128],f16>
    %int1_565 = torch.constant.int 1
    %int128_566 = torch.constant.int 128
    %564 = torch.prim.ListConstruct %int1_565, %67, %int128_566 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %565 = torch.aten.view %563, %564 : !torch.vtensor<[?,128],f16>, !torch.list<int> -> !torch.vtensor<[1,?,128],f16>
    torch.bind_symbolic_shape %565, [%63], affine_map<()[s0] -> (1, s0 * 32, 128)> : !torch.vtensor<[1,?,128],f16>
    %int1_567 = torch.constant.int 1
    %int0_568 = torch.constant.int 0
    %566 = torch.prim.ListConstruct %int1_567, %int0_568 : (!torch.int, !torch.int) -> !torch.list<int>
    %567 = torch.aten.permute %24, %566 : !torch.vtensor<[64,256],f32>, !torch.list<int> -> !torch.vtensor<[256,64],f32>
    %int1_569 = torch.constant.int 1
    %int0_570 = torch.constant.int 0
    %568 = torch.prim.ListConstruct %int1_569, %int0_570 : (!torch.int, !torch.int) -> !torch.list<int>
    %569 = torch.aten.permute %25, %568 : !torch.vtensor<[64,256],f32>, !torch.list<int> -> !torch.vtensor<[256,64],f32>
    %int5_571 = torch.constant.int 5
    %570 = torch.prims.convert_element_type %567, %int5_571 : !torch.vtensor<[256,64],f32>, !torch.int -> !torch.vtensor<[256,64],f16>
    %int256_572 = torch.constant.int 256
    %571 = torch.prim.ListConstruct %67, %int256_572 : (!torch.int, !torch.int) -> !torch.list<int>
    %572 = torch.aten.view %548, %571 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %572, [%63], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %573 = torch.aten.mm %572, %570 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,64],f16> -> !torch.vtensor<[?,64],f16>
    torch.bind_symbolic_shape %573, [%63], affine_map<()[s0] -> (s0 * 32, 64)> : !torch.vtensor<[?,64],f16>
    %int1_573 = torch.constant.int 1
    %int64_574 = torch.constant.int 64
    %574 = torch.prim.ListConstruct %int1_573, %67, %int64_574 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %575 = torch.aten.view %573, %574 : !torch.vtensor<[?,64],f16>, !torch.list<int> -> !torch.vtensor<[1,?,64],f16>
    torch.bind_symbolic_shape %575, [%63], affine_map<()[s0] -> (1, s0 * 32, 64)> : !torch.vtensor<[1,?,64],f16>
    %int5_575 = torch.constant.int 5
    %576 = torch.prims.convert_element_type %569, %int5_575 : !torch.vtensor<[256,64],f32>, !torch.int -> !torch.vtensor<[256,64],f16>
    %int256_576 = torch.constant.int 256
    %577 = torch.prim.ListConstruct %67, %int256_576 : (!torch.int, !torch.int) -> !torch.list<int>
    %578 = torch.aten.view %549, %577 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %578, [%63], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %579 = torch.aten.mm %578, %576 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,64],f16> -> !torch.vtensor<[?,64],f16>
    torch.bind_symbolic_shape %579, [%63], affine_map<()[s0] -> (s0 * 32, 64)> : !torch.vtensor<[?,64],f16>
    %int1_577 = torch.constant.int 1
    %int64_578 = torch.constant.int 64
    %580 = torch.prim.ListConstruct %int1_577, %67, %int64_578 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %581 = torch.aten.view %579, %580 : !torch.vtensor<[?,64],f16>, !torch.list<int> -> !torch.vtensor<[1,?,64],f16>
    torch.bind_symbolic_shape %581, [%63], affine_map<()[s0] -> (1, s0 * 32, 64)> : !torch.vtensor<[1,?,64],f16>
    %int1_579 = torch.constant.int 1
    %int0_580 = torch.constant.int 0
    %582 = torch.prim.ListConstruct %int1_579, %int0_580 : (!torch.int, !torch.int) -> !torch.list<int>
    %583 = torch.aten.permute %26, %582 : !torch.vtensor<[64,256],f32>, !torch.list<int> -> !torch.vtensor<[256,64],f32>
    %int1_581 = torch.constant.int 1
    %int0_582 = torch.constant.int 0
    %584 = torch.prim.ListConstruct %int1_581, %int0_582 : (!torch.int, !torch.int) -> !torch.list<int>
    %585 = torch.aten.permute %27, %584 : !torch.vtensor<[64,256],f32>, !torch.list<int> -> !torch.vtensor<[256,64],f32>
    %int5_583 = torch.constant.int 5
    %586 = torch.prims.convert_element_type %583, %int5_583 : !torch.vtensor<[256,64],f32>, !torch.int -> !torch.vtensor<[256,64],f16>
    %int256_584 = torch.constant.int 256
    %587 = torch.prim.ListConstruct %67, %int256_584 : (!torch.int, !torch.int) -> !torch.list<int>
    %588 = torch.aten.view %548, %587 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %588, [%63], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %589 = torch.aten.mm %588, %586 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,64],f16> -> !torch.vtensor<[?,64],f16>
    torch.bind_symbolic_shape %589, [%63], affine_map<()[s0] -> (s0 * 32, 64)> : !torch.vtensor<[?,64],f16>
    %int1_585 = torch.constant.int 1
    %int64_586 = torch.constant.int 64
    %590 = torch.prim.ListConstruct %int1_585, %67, %int64_586 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %591 = torch.aten.view %589, %590 : !torch.vtensor<[?,64],f16>, !torch.list<int> -> !torch.vtensor<[1,?,64],f16>
    torch.bind_symbolic_shape %591, [%63], affine_map<()[s0] -> (1, s0 * 32, 64)> : !torch.vtensor<[1,?,64],f16>
    %int5_587 = torch.constant.int 5
    %592 = torch.prims.convert_element_type %585, %int5_587 : !torch.vtensor<[256,64],f32>, !torch.int -> !torch.vtensor<[256,64],f16>
    %int256_588 = torch.constant.int 256
    %593 = torch.prim.ListConstruct %67, %int256_588 : (!torch.int, !torch.int) -> !torch.list<int>
    %594 = torch.aten.view %549, %593 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %594, [%63], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %595 = torch.aten.mm %594, %592 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,64],f16> -> !torch.vtensor<[?,64],f16>
    torch.bind_symbolic_shape %595, [%63], affine_map<()[s0] -> (s0 * 32, 64)> : !torch.vtensor<[?,64],f16>
    %int1_589 = torch.constant.int 1
    %int64_590 = torch.constant.int 64
    %596 = torch.prim.ListConstruct %int1_589, %67, %int64_590 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %597 = torch.aten.view %595, %596 : !torch.vtensor<[?,64],f16>, !torch.list<int> -> !torch.vtensor<[1,?,64],f16>
    torch.bind_symbolic_shape %597, [%63], affine_map<()[s0] -> (1, s0 * 32, 64)> : !torch.vtensor<[1,?,64],f16>
    %int1_591 = torch.constant.int 1
    %int4_592 = torch.constant.int 4
    %int32_593 = torch.constant.int 32
    %598 = torch.prim.ListConstruct %int1_591, %67, %int4_592, %int32_593 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %599 = torch.aten.view %559, %598 : !torch.vtensor<[1,?,128],f16>, !torch.list<int> -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %599, [%63], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int1_594 = torch.constant.int 1
    %int4_595 = torch.constant.int 4
    %int32_596 = torch.constant.int 32
    %600 = torch.prim.ListConstruct %int1_594, %67, %int4_595, %int32_596 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %601 = torch.aten.view %565, %600 : !torch.vtensor<[1,?,128],f16>, !torch.list<int> -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %601, [%63], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int1_597 = torch.constant.int 1
    %int2_598 = torch.constant.int 2
    %int32_599 = torch.constant.int 32
    %602 = torch.prim.ListConstruct %int1_597, %67, %int2_598, %int32_599 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %603 = torch.aten.view %575, %602 : !torch.vtensor<[1,?,64],f16>, !torch.list<int> -> !torch.vtensor<[1,?,2,32],f16>
    torch.bind_symbolic_shape %603, [%63], affine_map<()[s0] -> (1, s0 * 32, 2, 32)> : !torch.vtensor<[1,?,2,32],f16>
    %int1_600 = torch.constant.int 1
    %int2_601 = torch.constant.int 2
    %int32_602 = torch.constant.int 32
    %604 = torch.prim.ListConstruct %int1_600, %67, %int2_601, %int32_602 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %605 = torch.aten.view %581, %604 : !torch.vtensor<[1,?,64],f16>, !torch.list<int> -> !torch.vtensor<[1,?,2,32],f16>
    torch.bind_symbolic_shape %605, [%63], affine_map<()[s0] -> (1, s0 * 32, 2, 32)> : !torch.vtensor<[1,?,2,32],f16>
    %int1_603 = torch.constant.int 1
    %int2_604 = torch.constant.int 2
    %int32_605 = torch.constant.int 32
    %606 = torch.prim.ListConstruct %int1_603, %67, %int2_604, %int32_605 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %607 = torch.aten.view %591, %606 : !torch.vtensor<[1,?,64],f16>, !torch.list<int> -> !torch.vtensor<[1,?,2,32],f16>
    torch.bind_symbolic_shape %607, [%63], affine_map<()[s0] -> (1, s0 * 32, 2, 32)> : !torch.vtensor<[1,?,2,32],f16>
    %int1_606 = torch.constant.int 1
    %int2_607 = torch.constant.int 2
    %int32_608 = torch.constant.int 32
    %608 = torch.prim.ListConstruct %int1_606, %67, %int2_607, %int32_608 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %609 = torch.aten.view %597, %608 : !torch.vtensor<[1,?,64],f16>, !torch.list<int> -> !torch.vtensor<[1,?,2,32],f16>
    torch.bind_symbolic_shape %609, [%63], affine_map<()[s0] -> (1, s0 * 32, 2, 32)> : !torch.vtensor<[1,?,2,32],f16>
    %int128_609 = torch.constant.int 128
    %none_610 = torch.constant.none
    %none_611 = torch.constant.none
    %cpu_612 = torch.constant.device "cpu"
    %false_613 = torch.constant.bool false
    %610 = torch.aten.arange %int128_609, %none_610, %none_611, %cpu_612, %false_613 : !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[128],si64>
    %int0_614 = torch.constant.int 0
    %int32_615 = torch.constant.int 32
    %none_616 = torch.constant.none
    %none_617 = torch.constant.none
    %cpu_618 = torch.constant.device "cpu"
    %false_619 = torch.constant.bool false
    %611 = torch.aten.arange.start %int0_614, %int32_615, %none_616, %none_617, %cpu_618, %false_619 : !torch.int, !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[32],si64>
    %int2_620 = torch.constant.int 2
    %612 = torch.aten.floor_divide.Scalar %611, %int2_620 : !torch.vtensor<[32],si64>, !torch.int -> !torch.vtensor<[32],si64>
    %int6_621 = torch.constant.int 6
    %613 = torch.prims.convert_element_type %612, %int6_621 : !torch.vtensor<[32],si64>, !torch.int -> !torch.vtensor<[32],f32>
    %int32_622 = torch.constant.int 32
    %614 = torch.aten.div.Scalar %613, %int32_622 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[32],f32>
    %float2.000000e00_623 = torch.constant.float 2.000000e+00
    %615 = torch.aten.mul.Scalar %614, %float2.000000e00_623 : !torch.vtensor<[32],f32>, !torch.float -> !torch.vtensor<[32],f32>
    %float5.000000e05_624 = torch.constant.float 5.000000e+05
    %616 = torch.aten.pow.Scalar %float5.000000e05_624, %615 : !torch.float, !torch.vtensor<[32],f32> -> !torch.vtensor<[32],f32>
    %617 = torch.aten.reciprocal %616 : !torch.vtensor<[32],f32> -> !torch.vtensor<[32],f32>
    %float1.000000e00_625 = torch.constant.float 1.000000e+00
    %618 = torch.aten.mul.Scalar %617, %float1.000000e00_625 : !torch.vtensor<[32],f32>, !torch.float -> !torch.vtensor<[32],f32>
    %int1_626 = torch.constant.int 1
    %619 = torch.aten.unsqueeze %610, %int1_626 : !torch.vtensor<[128],si64>, !torch.int -> !torch.vtensor<[128,1],si64>
    %int0_627 = torch.constant.int 0
    %620 = torch.aten.unsqueeze %618, %int0_627 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[1,32],f32>
    %621 = torch.aten.mul.Tensor %619, %620 : !torch.vtensor<[128,1],si64>, !torch.vtensor<[1,32],f32> -> !torch.vtensor<[128,32],f32>
    %int6_628 = torch.constant.int 6
    %622 = torch.prims.convert_element_type %621, %int6_628 : !torch.vtensor<[128,32],f32>, !torch.int -> !torch.vtensor<[128,32],f32>
    %623 = torch_c.to_builtin_tensor %622 : !torch.vtensor<[128,32],f32> -> tensor<128x32xf32>
    %624 = flow.tensor.transfer %623 : tensor<128x32xf32> to #hal.device.promise<@__device_0>
    %625 = torch_c.from_builtin_tensor %624 : tensor<128x32xf32> -> !torch.vtensor<[128,32],f32>
    %626 = torch_c.to_builtin_tensor %622 : !torch.vtensor<[128,32],f32> -> tensor<128x32xf32>
    %627 = flow.tensor.transfer %626 : tensor<128x32xf32> to #hal.device.promise<@__device_1>
    %628 = torch_c.from_builtin_tensor %627 : tensor<128x32xf32> -> !torch.vtensor<[128,32],f32>
    %int0_629 = torch.constant.int 0
    %int0_630 = torch.constant.int 0
    %int1_631 = torch.constant.int 1
    %629 = torch.aten.slice.Tensor %625, %int0_629, %int0_630, %67, %int1_631 : !torch.vtensor<[128,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %629, [%63], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int1_632 = torch.constant.int 1
    %int0_633 = torch.constant.int 0
    %int9223372036854775807_634 = torch.constant.int 9223372036854775807
    %int1_635 = torch.constant.int 1
    %630 = torch.aten.slice.Tensor %629, %int1_632, %int0_633, %int9223372036854775807_634, %int1_635 : !torch.vtensor<[?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %630, [%63], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int1_636 = torch.constant.int 1
    %int0_637 = torch.constant.int 0
    %int9223372036854775807_638 = torch.constant.int 9223372036854775807
    %int1_639 = torch.constant.int 1
    %631 = torch.aten.slice.Tensor %630, %int1_636, %int0_637, %int9223372036854775807_638, %int1_639 : !torch.vtensor<[?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %631, [%63], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int0_640 = torch.constant.int 0
    %632 = torch.aten.unsqueeze %631, %int0_640 : !torch.vtensor<[?,32],f32>, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %632, [%63], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int1_641 = torch.constant.int 1
    %int0_642 = torch.constant.int 0
    %int9223372036854775807_643 = torch.constant.int 9223372036854775807
    %int1_644 = torch.constant.int 1
    %633 = torch.aten.slice.Tensor %632, %int1_641, %int0_642, %int9223372036854775807_643, %int1_644 : !torch.vtensor<[1,?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %633, [%63], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int2_645 = torch.constant.int 2
    %int0_646 = torch.constant.int 0
    %int9223372036854775807_647 = torch.constant.int 9223372036854775807
    %int1_648 = torch.constant.int 1
    %634 = torch.aten.slice.Tensor %633, %int2_645, %int0_646, %int9223372036854775807_647, %int1_648 : !torch.vtensor<[1,?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %634, [%63], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int1_649 = torch.constant.int 1
    %int1_650 = torch.constant.int 1
    %int1_651 = torch.constant.int 1
    %635 = torch.prim.ListConstruct %int1_649, %int1_650, %int1_651 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %636 = torch.aten.repeat %634, %635 : !torch.vtensor<[1,?,32],f32>, !torch.list<int> -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %636, [%63], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int6_652 = torch.constant.int 6
    %637 = torch.prims.convert_element_type %599, %int6_652 : !torch.vtensor<[1,?,4,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,32],f32>
    torch.bind_symbolic_shape %637, [%63], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f32>
    %638 = torch_c.to_builtin_tensor %637 : !torch.vtensor<[1,?,4,32],f32> -> tensor<1x?x4x32xf32>
    %639 = torch_c.to_builtin_tensor %636 : !torch.vtensor<[1,?,32],f32> -> tensor<1x?x32xf32>
    %640 = util.call @sharktank_rotary_embedding_1_D_4_32_f32(%638, %639) : (tensor<1x?x4x32xf32>, tensor<1x?x32xf32>) -> tensor<1x?x4x32xf32>
    %641 = torch_c.from_builtin_tensor %640 : tensor<1x?x4x32xf32> -> !torch.vtensor<[1,?,4,32],f32>
    torch.bind_symbolic_shape %641, [%63], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f32>
    %int5_653 = torch.constant.int 5
    %642 = torch.prims.convert_element_type %641, %int5_653 : !torch.vtensor<[1,?,4,32],f32>, !torch.int -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %642, [%63], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int0_654 = torch.constant.int 0
    %int0_655 = torch.constant.int 0
    %int1_656 = torch.constant.int 1
    %643 = torch.aten.slice.Tensor %628, %int0_654, %int0_655, %67, %int1_656 : !torch.vtensor<[128,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %643, [%63], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int1_657 = torch.constant.int 1
    %int0_658 = torch.constant.int 0
    %int9223372036854775807_659 = torch.constant.int 9223372036854775807
    %int1_660 = torch.constant.int 1
    %644 = torch.aten.slice.Tensor %643, %int1_657, %int0_658, %int9223372036854775807_659, %int1_660 : !torch.vtensor<[?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %644, [%63], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int1_661 = torch.constant.int 1
    %int0_662 = torch.constant.int 0
    %int9223372036854775807_663 = torch.constant.int 9223372036854775807
    %int1_664 = torch.constant.int 1
    %645 = torch.aten.slice.Tensor %644, %int1_661, %int0_662, %int9223372036854775807_663, %int1_664 : !torch.vtensor<[?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %645, [%63], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int0_665 = torch.constant.int 0
    %646 = torch.aten.unsqueeze %645, %int0_665 : !torch.vtensor<[?,32],f32>, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %646, [%63], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int1_666 = torch.constant.int 1
    %int0_667 = torch.constant.int 0
    %int9223372036854775807_668 = torch.constant.int 9223372036854775807
    %int1_669 = torch.constant.int 1
    %647 = torch.aten.slice.Tensor %646, %int1_666, %int0_667, %int9223372036854775807_668, %int1_669 : !torch.vtensor<[1,?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %647, [%63], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int2_670 = torch.constant.int 2
    %int0_671 = torch.constant.int 0
    %int9223372036854775807_672 = torch.constant.int 9223372036854775807
    %int1_673 = torch.constant.int 1
    %648 = torch.aten.slice.Tensor %647, %int2_670, %int0_671, %int9223372036854775807_672, %int1_673 : !torch.vtensor<[1,?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %648, [%63], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int1_674 = torch.constant.int 1
    %int1_675 = torch.constant.int 1
    %int1_676 = torch.constant.int 1
    %649 = torch.prim.ListConstruct %int1_674, %int1_675, %int1_676 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %650 = torch.aten.repeat %648, %649 : !torch.vtensor<[1,?,32],f32>, !torch.list<int> -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %650, [%63], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int6_677 = torch.constant.int 6
    %651 = torch.prims.convert_element_type %601, %int6_677 : !torch.vtensor<[1,?,4,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,32],f32>
    torch.bind_symbolic_shape %651, [%63], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f32>
    %652 = torch_c.to_builtin_tensor %651 : !torch.vtensor<[1,?,4,32],f32> -> tensor<1x?x4x32xf32>
    %653 = torch_c.to_builtin_tensor %650 : !torch.vtensor<[1,?,32],f32> -> tensor<1x?x32xf32>
    %654 = util.call @sharktank_rotary_embedding_1_D_4_32_f32(%652, %653) : (tensor<1x?x4x32xf32>, tensor<1x?x32xf32>) -> tensor<1x?x4x32xf32>
    %655 = torch_c.from_builtin_tensor %654 : tensor<1x?x4x32xf32> -> !torch.vtensor<[1,?,4,32],f32>
    torch.bind_symbolic_shape %655, [%63], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f32>
    %int5_678 = torch.constant.int 5
    %656 = torch.prims.convert_element_type %655, %int5_678 : !torch.vtensor<[1,?,4,32],f32>, !torch.int -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %656, [%63], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int128_679 = torch.constant.int 128
    %none_680 = torch.constant.none
    %none_681 = torch.constant.none
    %cpu_682 = torch.constant.device "cpu"
    %false_683 = torch.constant.bool false
    %657 = torch.aten.arange %int128_679, %none_680, %none_681, %cpu_682, %false_683 : !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[128],si64>
    %int0_684 = torch.constant.int 0
    %int32_685 = torch.constant.int 32
    %none_686 = torch.constant.none
    %none_687 = torch.constant.none
    %cpu_688 = torch.constant.device "cpu"
    %false_689 = torch.constant.bool false
    %658 = torch.aten.arange.start %int0_684, %int32_685, %none_686, %none_687, %cpu_688, %false_689 : !torch.int, !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[32],si64>
    %int2_690 = torch.constant.int 2
    %659 = torch.aten.floor_divide.Scalar %658, %int2_690 : !torch.vtensor<[32],si64>, !torch.int -> !torch.vtensor<[32],si64>
    %int6_691 = torch.constant.int 6
    %660 = torch.prims.convert_element_type %659, %int6_691 : !torch.vtensor<[32],si64>, !torch.int -> !torch.vtensor<[32],f32>
    %int32_692 = torch.constant.int 32
    %661 = torch.aten.div.Scalar %660, %int32_692 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[32],f32>
    %float2.000000e00_693 = torch.constant.float 2.000000e+00
    %662 = torch.aten.mul.Scalar %661, %float2.000000e00_693 : !torch.vtensor<[32],f32>, !torch.float -> !torch.vtensor<[32],f32>
    %float5.000000e05_694 = torch.constant.float 5.000000e+05
    %663 = torch.aten.pow.Scalar %float5.000000e05_694, %662 : !torch.float, !torch.vtensor<[32],f32> -> !torch.vtensor<[32],f32>
    %664 = torch.aten.reciprocal %663 : !torch.vtensor<[32],f32> -> !torch.vtensor<[32],f32>
    %float1.000000e00_695 = torch.constant.float 1.000000e+00
    %665 = torch.aten.mul.Scalar %664, %float1.000000e00_695 : !torch.vtensor<[32],f32>, !torch.float -> !torch.vtensor<[32],f32>
    %int1_696 = torch.constant.int 1
    %666 = torch.aten.unsqueeze %657, %int1_696 : !torch.vtensor<[128],si64>, !torch.int -> !torch.vtensor<[128,1],si64>
    %int0_697 = torch.constant.int 0
    %667 = torch.aten.unsqueeze %665, %int0_697 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[1,32],f32>
    %668 = torch.aten.mul.Tensor %666, %667 : !torch.vtensor<[128,1],si64>, !torch.vtensor<[1,32],f32> -> !torch.vtensor<[128,32],f32>
    %int6_698 = torch.constant.int 6
    %669 = torch.prims.convert_element_type %668, %int6_698 : !torch.vtensor<[128,32],f32>, !torch.int -> !torch.vtensor<[128,32],f32>
    %670 = torch_c.to_builtin_tensor %669 : !torch.vtensor<[128,32],f32> -> tensor<128x32xf32>
    %671 = flow.tensor.transfer %670 : tensor<128x32xf32> to #hal.device.promise<@__device_0>
    %672 = torch_c.from_builtin_tensor %671 : tensor<128x32xf32> -> !torch.vtensor<[128,32],f32>
    %673 = torch_c.to_builtin_tensor %669 : !torch.vtensor<[128,32],f32> -> tensor<128x32xf32>
    %674 = flow.tensor.transfer %673 : tensor<128x32xf32> to #hal.device.promise<@__device_1>
    %675 = torch_c.from_builtin_tensor %674 : tensor<128x32xf32> -> !torch.vtensor<[128,32],f32>
    %int0_699 = torch.constant.int 0
    %int0_700 = torch.constant.int 0
    %int1_701 = torch.constant.int 1
    %676 = torch.aten.slice.Tensor %672, %int0_699, %int0_700, %67, %int1_701 : !torch.vtensor<[128,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %676, [%63], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int1_702 = torch.constant.int 1
    %int0_703 = torch.constant.int 0
    %int9223372036854775807_704 = torch.constant.int 9223372036854775807
    %int1_705 = torch.constant.int 1
    %677 = torch.aten.slice.Tensor %676, %int1_702, %int0_703, %int9223372036854775807_704, %int1_705 : !torch.vtensor<[?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %677, [%63], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int1_706 = torch.constant.int 1
    %int0_707 = torch.constant.int 0
    %int9223372036854775807_708 = torch.constant.int 9223372036854775807
    %int1_709 = torch.constant.int 1
    %678 = torch.aten.slice.Tensor %677, %int1_706, %int0_707, %int9223372036854775807_708, %int1_709 : !torch.vtensor<[?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %678, [%63], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int0_710 = torch.constant.int 0
    %679 = torch.aten.unsqueeze %678, %int0_710 : !torch.vtensor<[?,32],f32>, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %679, [%63], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int1_711 = torch.constant.int 1
    %int0_712 = torch.constant.int 0
    %int9223372036854775807_713 = torch.constant.int 9223372036854775807
    %int1_714 = torch.constant.int 1
    %680 = torch.aten.slice.Tensor %679, %int1_711, %int0_712, %int9223372036854775807_713, %int1_714 : !torch.vtensor<[1,?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %680, [%63], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int2_715 = torch.constant.int 2
    %int0_716 = torch.constant.int 0
    %int9223372036854775807_717 = torch.constant.int 9223372036854775807
    %int1_718 = torch.constant.int 1
    %681 = torch.aten.slice.Tensor %680, %int2_715, %int0_716, %int9223372036854775807_717, %int1_718 : !torch.vtensor<[1,?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %681, [%63], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int1_719 = torch.constant.int 1
    %int1_720 = torch.constant.int 1
    %int1_721 = torch.constant.int 1
    %682 = torch.prim.ListConstruct %int1_719, %int1_720, %int1_721 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %683 = torch.aten.repeat %681, %682 : !torch.vtensor<[1,?,32],f32>, !torch.list<int> -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %683, [%63], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int6_722 = torch.constant.int 6
    %684 = torch.prims.convert_element_type %603, %int6_722 : !torch.vtensor<[1,?,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,2,32],f32>
    torch.bind_symbolic_shape %684, [%63], affine_map<()[s0] -> (1, s0 * 32, 2, 32)> : !torch.vtensor<[1,?,2,32],f32>
    %685 = torch_c.to_builtin_tensor %684 : !torch.vtensor<[1,?,2,32],f32> -> tensor<1x?x2x32xf32>
    %686 = torch_c.to_builtin_tensor %683 : !torch.vtensor<[1,?,32],f32> -> tensor<1x?x32xf32>
    %687 = util.call @sharktank_rotary_embedding_1_D_2_32_f32(%685, %686) : (tensor<1x?x2x32xf32>, tensor<1x?x32xf32>) -> tensor<1x?x2x32xf32>
    %688 = torch_c.from_builtin_tensor %687 : tensor<1x?x2x32xf32> -> !torch.vtensor<[1,?,2,32],f32>
    torch.bind_symbolic_shape %688, [%63], affine_map<()[s0] -> (1, s0 * 32, 2, 32)> : !torch.vtensor<[1,?,2,32],f32>
    %int5_723 = torch.constant.int 5
    %689 = torch.prims.convert_element_type %688, %int5_723 : !torch.vtensor<[1,?,2,32],f32>, !torch.int -> !torch.vtensor<[1,?,2,32],f16>
    torch.bind_symbolic_shape %689, [%63], affine_map<()[s0] -> (1, s0 * 32, 2, 32)> : !torch.vtensor<[1,?,2,32],f16>
    %int0_724 = torch.constant.int 0
    %int0_725 = torch.constant.int 0
    %int1_726 = torch.constant.int 1
    %690 = torch.aten.slice.Tensor %675, %int0_724, %int0_725, %67, %int1_726 : !torch.vtensor<[128,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %690, [%63], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int1_727 = torch.constant.int 1
    %int0_728 = torch.constant.int 0
    %int9223372036854775807_729 = torch.constant.int 9223372036854775807
    %int1_730 = torch.constant.int 1
    %691 = torch.aten.slice.Tensor %690, %int1_727, %int0_728, %int9223372036854775807_729, %int1_730 : !torch.vtensor<[?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %691, [%63], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int1_731 = torch.constant.int 1
    %int0_732 = torch.constant.int 0
    %int9223372036854775807_733 = torch.constant.int 9223372036854775807
    %int1_734 = torch.constant.int 1
    %692 = torch.aten.slice.Tensor %691, %int1_731, %int0_732, %int9223372036854775807_733, %int1_734 : !torch.vtensor<[?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %692, [%63], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int0_735 = torch.constant.int 0
    %693 = torch.aten.unsqueeze %692, %int0_735 : !torch.vtensor<[?,32],f32>, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %693, [%63], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int1_736 = torch.constant.int 1
    %int0_737 = torch.constant.int 0
    %int9223372036854775807_738 = torch.constant.int 9223372036854775807
    %int1_739 = torch.constant.int 1
    %694 = torch.aten.slice.Tensor %693, %int1_736, %int0_737, %int9223372036854775807_738, %int1_739 : !torch.vtensor<[1,?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %694, [%63], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int2_740 = torch.constant.int 2
    %int0_741 = torch.constant.int 0
    %int9223372036854775807_742 = torch.constant.int 9223372036854775807
    %int1_743 = torch.constant.int 1
    %695 = torch.aten.slice.Tensor %694, %int2_740, %int0_741, %int9223372036854775807_742, %int1_743 : !torch.vtensor<[1,?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %695, [%63], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int1_744 = torch.constant.int 1
    %int1_745 = torch.constant.int 1
    %int1_746 = torch.constant.int 1
    %696 = torch.prim.ListConstruct %int1_744, %int1_745, %int1_746 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %697 = torch.aten.repeat %695, %696 : !torch.vtensor<[1,?,32],f32>, !torch.list<int> -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %697, [%63], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int6_747 = torch.constant.int 6
    %698 = torch.prims.convert_element_type %605, %int6_747 : !torch.vtensor<[1,?,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,2,32],f32>
    torch.bind_symbolic_shape %698, [%63], affine_map<()[s0] -> (1, s0 * 32, 2, 32)> : !torch.vtensor<[1,?,2,32],f32>
    %699 = torch_c.to_builtin_tensor %698 : !torch.vtensor<[1,?,2,32],f32> -> tensor<1x?x2x32xf32>
    %700 = torch_c.to_builtin_tensor %697 : !torch.vtensor<[1,?,32],f32> -> tensor<1x?x32xf32>
    %701 = util.call @sharktank_rotary_embedding_1_D_2_32_f32(%699, %700) : (tensor<1x?x2x32xf32>, tensor<1x?x32xf32>) -> tensor<1x?x2x32xf32>
    %702 = torch_c.from_builtin_tensor %701 : tensor<1x?x2x32xf32> -> !torch.vtensor<[1,?,2,32],f32>
    torch.bind_symbolic_shape %702, [%63], affine_map<()[s0] -> (1, s0 * 32, 2, 32)> : !torch.vtensor<[1,?,2,32],f32>
    %int5_748 = torch.constant.int 5
    %703 = torch.prims.convert_element_type %702, %int5_748 : !torch.vtensor<[1,?,2,32],f32>, !torch.int -> !torch.vtensor<[1,?,2,32],f16>
    torch.bind_symbolic_shape %703, [%63], affine_map<()[s0] -> (1, s0 * 32, 2, 32)> : !torch.vtensor<[1,?,2,32],f16>
    %int6_749 = torch.constant.int 6
    %704 = torch.aten.mul.Scalar %108, %int6_749 : !torch.vtensor<[1,?],si64>, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %704, [%63], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %int6_750 = torch.constant.int 6
    %705 = torch.aten.mul.Scalar %111, %int6_750 : !torch.vtensor<[1,?],si64>, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %705, [%63], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %int2_751 = torch.constant.int 2
    %int1_752 = torch.constant.int 1
    %706 = torch.aten.add.Scalar %704, %int2_751, %int1_752 : !torch.vtensor<[1,?],si64>, !torch.int, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %706, [%63], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %int2_753 = torch.constant.int 2
    %int1_754 = torch.constant.int 1
    %707 = torch.aten.add.Scalar %705, %int2_753, %int1_754 : !torch.vtensor<[1,?],si64>, !torch.int, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %707, [%63], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %int1_755 = torch.constant.int 1
    %int32_756 = torch.constant.int 32
    %int2_757 = torch.constant.int 2
    %int32_758 = torch.constant.int 32
    %708 = torch.prim.ListConstruct %int1_755, %65, %int32_756, %int2_757, %int32_758 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %709 = torch.aten.view %689, %708 : !torch.vtensor<[1,?,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,32,2,32],f16>
    torch.bind_symbolic_shape %709, [%63], affine_map<()[s0] -> (1, s0, 32, 2, 32)> : !torch.vtensor<[1,?,32,2,32],f16>
    %int1_759 = torch.constant.int 1
    %int32_760 = torch.constant.int 32
    %int2_761 = torch.constant.int 2
    %int32_762 = torch.constant.int 32
    %710 = torch.prim.ListConstruct %int1_759, %65, %int32_760, %int2_761, %int32_762 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %711 = torch.aten.view %703, %710 : !torch.vtensor<[1,?,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,32,2,32],f16>
    torch.bind_symbolic_shape %711, [%63], affine_map<()[s0] -> (1, s0, 32, 2, 32)> : !torch.vtensor<[1,?,32,2,32],f16>
    %int32_763 = torch.constant.int 32
    %int2_764 = torch.constant.int 2
    %int32_765 = torch.constant.int 32
    %712 = torch.prim.ListConstruct %65, %int32_763, %int2_764, %int32_765 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %713 = torch.aten.view %709, %712 : !torch.vtensor<[1,?,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,32,2,32],f16>
    torch.bind_symbolic_shape %713, [%63], affine_map<()[s0] -> (s0, 32, 2, 32)> : !torch.vtensor<[?,32,2,32],f16>
    %int32_766 = torch.constant.int 32
    %int2_767 = torch.constant.int 2
    %int32_768 = torch.constant.int 32
    %714 = torch.prim.ListConstruct %65, %int32_766, %int2_767, %int32_768 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %715 = torch.aten.view %711, %714 : !torch.vtensor<[1,?,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,32,2,32],f16>
    torch.bind_symbolic_shape %715, [%63], affine_map<()[s0] -> (s0, 32, 2, 32)> : !torch.vtensor<[?,32,2,32],f16>
    %716 = torch.prim.ListConstruct %65 : (!torch.int) -> !torch.list<int>
    %717 = torch.aten.view %706, %716 : !torch.vtensor<[1,?],si64>, !torch.list<int> -> !torch.vtensor<[?],si64>
    torch.bind_symbolic_shape %717, [%63], affine_map<()[s0] -> (s0)> : !torch.vtensor<[?],si64>
    %718 = torch.prim.ListConstruct %65 : (!torch.int) -> !torch.list<int>
    %719 = torch.aten.view %707, %718 : !torch.vtensor<[1,?],si64>, !torch.list<int> -> !torch.vtensor<[?],si64>
    torch.bind_symbolic_shape %719, [%63], affine_map<()[s0] -> (s0)> : !torch.vtensor<[?],si64>
    %int5_769 = torch.constant.int 5
    %720 = torch.prims.convert_element_type %713, %int5_769 : !torch.vtensor<[?,32,2,32],f16>, !torch.int -> !torch.vtensor<[?,32,2,32],f16>
    torch.bind_symbolic_shape %720, [%63], affine_map<()[s0] -> (s0, 32, 2, 32)> : !torch.vtensor<[?,32,2,32],f16>
    %int5_770 = torch.constant.int 5
    %721 = torch.prims.convert_element_type %715, %int5_770 : !torch.vtensor<[?,32,2,32],f16>, !torch.int -> !torch.vtensor<[?,32,2,32],f16>
    torch.bind_symbolic_shape %721, [%63], affine_map<()[s0] -> (s0, 32, 2, 32)> : !torch.vtensor<[?,32,2,32],f16>
    %int3_771 = torch.constant.int 3
    %int2_772 = torch.constant.int 2
    %int32_773 = torch.constant.int 32
    %int2_774 = torch.constant.int 2
    %int32_775 = torch.constant.int 32
    %722 = torch.prim.ListConstruct %66, %int3_771, %int2_772, %int32_773, %int2_774, %int32_775 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %723 = torch.aten.view %359, %722 : !torch.vtensor<[?,12288],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,2,32],f16>
    torch.bind_symbolic_shape %723, [%64], affine_map<()[s0] -> (s0, 3, 2, 32, 2, 32)> : !torch.vtensor<[?,3,2,32,2,32],f16>
    %int32_776 = torch.constant.int 32
    %int2_777 = torch.constant.int 2
    %int32_778 = torch.constant.int 32
    %724 = torch.prim.ListConstruct %295, %int32_776, %int2_777, %int32_778 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %725 = torch.aten.view %723, %724 : !torch.vtensor<[?,3,2,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,32,2,32],f16>
    torch.bind_symbolic_shape %725, [%64], affine_map<()[s0] -> (s0 * 6, 32, 2, 32)> : !torch.vtensor<[?,32,2,32],f16>
    %726 = torch.prim.ListConstruct %717 : (!torch.vtensor<[?],si64>) -> !torch.list<optional<vtensor>>
    %false_779 = torch.constant.bool false
    %727 = torch.aten.index_put %725, %726, %720, %false_779 : !torch.vtensor<[?,32,2,32],f16>, !torch.list<optional<vtensor>>, !torch.vtensor<[?,32,2,32],f16>, !torch.bool -> !torch.vtensor<[?,32,2,32],f16>
    torch.bind_symbolic_shape %727, [%64], affine_map<()[s0] -> (s0 * 6, 32, 2, 32)> : !torch.vtensor<[?,32,2,32],f16>
    %int3_780 = torch.constant.int 3
    %int2_781 = torch.constant.int 2
    %int32_782 = torch.constant.int 32
    %int2_783 = torch.constant.int 2
    %int32_784 = torch.constant.int 32
    %728 = torch.prim.ListConstruct %66, %int3_780, %int2_781, %int32_782, %int2_783, %int32_784 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %729 = torch.aten.view %727, %728 : !torch.vtensor<[?,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,2,32],f16>
    torch.bind_symbolic_shape %729, [%64], affine_map<()[s0] -> (s0, 3, 2, 32, 2, 32)> : !torch.vtensor<[?,3,2,32,2,32],f16>
    %int12288_785 = torch.constant.int 12288
    %730 = torch.prim.ListConstruct %66, %int12288_785 : (!torch.int, !torch.int) -> !torch.list<int>
    %731 = torch.aten.view %729, %730 : !torch.vtensor<[?,3,2,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,12288],f16>
    torch.bind_symbolic_shape %731, [%64], affine_map<()[s0] -> (s0, 12288)> : !torch.vtensor<[?,12288],f16>
    %int3_786 = torch.constant.int 3
    %int2_787 = torch.constant.int 2
    %int32_788 = torch.constant.int 32
    %int2_789 = torch.constant.int 2
    %int32_790 = torch.constant.int 32
    %732 = torch.prim.ListConstruct %66, %int3_786, %int2_787, %int32_788, %int2_789, %int32_790 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %733 = torch.aten.view %731, %732 : !torch.vtensor<[?,12288],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,2,32],f16>
    torch.bind_symbolic_shape %733, [%64], affine_map<()[s0] -> (s0, 3, 2, 32, 2, 32)> : !torch.vtensor<[?,3,2,32,2,32],f16>
    %int32_791 = torch.constant.int 32
    %int2_792 = torch.constant.int 2
    %int32_793 = torch.constant.int 32
    %734 = torch.prim.ListConstruct %295, %int32_791, %int2_792, %int32_793 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %735 = torch.aten.view %733, %734 : !torch.vtensor<[?,3,2,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,32,2,32],f16>
    torch.bind_symbolic_shape %735, [%64], affine_map<()[s0] -> (s0 * 6, 32, 2, 32)> : !torch.vtensor<[?,32,2,32],f16>
    %int3_794 = torch.constant.int 3
    %int2_795 = torch.constant.int 2
    %int32_796 = torch.constant.int 32
    %int2_797 = torch.constant.int 2
    %int32_798 = torch.constant.int 32
    %736 = torch.prim.ListConstruct %66, %int3_794, %int2_795, %int32_796, %int2_797, %int32_798 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %737 = torch.aten.view %365, %736 : !torch.vtensor<[?,12288],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,2,32],f16>
    torch.bind_symbolic_shape %737, [%64], affine_map<()[s0] -> (s0, 3, 2, 32, 2, 32)> : !torch.vtensor<[?,3,2,32,2,32],f16>
    %int32_799 = torch.constant.int 32
    %int2_800 = torch.constant.int 2
    %int32_801 = torch.constant.int 32
    %738 = torch.prim.ListConstruct %295, %int32_799, %int2_800, %int32_801 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %739 = torch.aten.view %737, %738 : !torch.vtensor<[?,3,2,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,32,2,32],f16>
    torch.bind_symbolic_shape %739, [%64], affine_map<()[s0] -> (s0 * 6, 32, 2, 32)> : !torch.vtensor<[?,32,2,32],f16>
    %740 = torch.prim.ListConstruct %719 : (!torch.vtensor<[?],si64>) -> !torch.list<optional<vtensor>>
    %false_802 = torch.constant.bool false
    %741 = torch.aten.index_put %739, %740, %721, %false_802 : !torch.vtensor<[?,32,2,32],f16>, !torch.list<optional<vtensor>>, !torch.vtensor<[?,32,2,32],f16>, !torch.bool -> !torch.vtensor<[?,32,2,32],f16>
    torch.bind_symbolic_shape %741, [%64], affine_map<()[s0] -> (s0 * 6, 32, 2, 32)> : !torch.vtensor<[?,32,2,32],f16>
    %int3_803 = torch.constant.int 3
    %int2_804 = torch.constant.int 2
    %int32_805 = torch.constant.int 32
    %int2_806 = torch.constant.int 2
    %int32_807 = torch.constant.int 32
    %742 = torch.prim.ListConstruct %66, %int3_803, %int2_804, %int32_805, %int2_806, %int32_807 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %743 = torch.aten.view %741, %742 : !torch.vtensor<[?,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,2,32],f16>
    torch.bind_symbolic_shape %743, [%64], affine_map<()[s0] -> (s0, 3, 2, 32, 2, 32)> : !torch.vtensor<[?,3,2,32,2,32],f16>
    %int12288_808 = torch.constant.int 12288
    %744 = torch.prim.ListConstruct %66, %int12288_808 : (!torch.int, !torch.int) -> !torch.list<int>
    %745 = torch.aten.view %743, %744 : !torch.vtensor<[?,3,2,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,12288],f16>
    torch.bind_symbolic_shape %745, [%64], affine_map<()[s0] -> (s0, 12288)> : !torch.vtensor<[?,12288],f16>
    %int3_809 = torch.constant.int 3
    %int2_810 = torch.constant.int 2
    %int32_811 = torch.constant.int 32
    %int2_812 = torch.constant.int 2
    %int32_813 = torch.constant.int 32
    %746 = torch.prim.ListConstruct %66, %int3_809, %int2_810, %int32_811, %int2_812, %int32_813 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %747 = torch.aten.view %745, %746 : !torch.vtensor<[?,12288],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,2,32],f16>
    torch.bind_symbolic_shape %747, [%64], affine_map<()[s0] -> (s0, 3, 2, 32, 2, 32)> : !torch.vtensor<[?,3,2,32,2,32],f16>
    %int32_814 = torch.constant.int 32
    %int2_815 = torch.constant.int 2
    %int32_816 = torch.constant.int 32
    %748 = torch.prim.ListConstruct %295, %int32_814, %int2_815, %int32_816 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %749 = torch.aten.view %747, %748 : !torch.vtensor<[?,3,2,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,32,2,32],f16>
    torch.bind_symbolic_shape %749, [%64], affine_map<()[s0] -> (s0 * 6, 32, 2, 32)> : !torch.vtensor<[?,32,2,32],f16>
    %int1_817 = torch.constant.int 1
    %int32_818 = torch.constant.int 32
    %int2_819 = torch.constant.int 2
    %int32_820 = torch.constant.int 32
    %750 = torch.prim.ListConstruct %int1_817, %65, %int32_818, %int2_819, %int32_820 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %751 = torch.aten.view %607, %750 : !torch.vtensor<[1,?,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,32,2,32],f16>
    torch.bind_symbolic_shape %751, [%63], affine_map<()[s0] -> (1, s0, 32, 2, 32)> : !torch.vtensor<[1,?,32,2,32],f16>
    %int1_821 = torch.constant.int 1
    %int32_822 = torch.constant.int 32
    %int2_823 = torch.constant.int 2
    %int32_824 = torch.constant.int 32
    %752 = torch.prim.ListConstruct %int1_821, %65, %int32_822, %int2_823, %int32_824 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %753 = torch.aten.view %609, %752 : !torch.vtensor<[1,?,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,32,2,32],f16>
    torch.bind_symbolic_shape %753, [%63], affine_map<()[s0] -> (1, s0, 32, 2, 32)> : !torch.vtensor<[1,?,32,2,32],f16>
    %int32_825 = torch.constant.int 32
    %int2_826 = torch.constant.int 2
    %int32_827 = torch.constant.int 32
    %754 = torch.prim.ListConstruct %65, %int32_825, %int2_826, %int32_827 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %755 = torch.aten.view %751, %754 : !torch.vtensor<[1,?,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,32,2,32],f16>
    torch.bind_symbolic_shape %755, [%63], affine_map<()[s0] -> (s0, 32, 2, 32)> : !torch.vtensor<[?,32,2,32],f16>
    %int32_828 = torch.constant.int 32
    %int2_829 = torch.constant.int 2
    %int32_830 = torch.constant.int 32
    %756 = torch.prim.ListConstruct %65, %int32_828, %int2_829, %int32_830 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %757 = torch.aten.view %753, %756 : !torch.vtensor<[1,?,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,32,2,32],f16>
    torch.bind_symbolic_shape %757, [%63], affine_map<()[s0] -> (s0, 32, 2, 32)> : !torch.vtensor<[?,32,2,32],f16>
    %int1_831 = torch.constant.int 1
    %int1_832 = torch.constant.int 1
    %758 = torch.aten.add.Scalar %706, %int1_831, %int1_832 : !torch.vtensor<[1,?],si64>, !torch.int, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %758, [%63], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %int1_833 = torch.constant.int 1
    %int1_834 = torch.constant.int 1
    %759 = torch.aten.add.Scalar %707, %int1_833, %int1_834 : !torch.vtensor<[1,?],si64>, !torch.int, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %759, [%63], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %760 = torch.prim.ListConstruct %65 : (!torch.int) -> !torch.list<int>
    %761 = torch.aten.view %758, %760 : !torch.vtensor<[1,?],si64>, !torch.list<int> -> !torch.vtensor<[?],si64>
    torch.bind_symbolic_shape %761, [%63], affine_map<()[s0] -> (s0)> : !torch.vtensor<[?],si64>
    %762 = torch.prim.ListConstruct %65 : (!torch.int) -> !torch.list<int>
    %763 = torch.aten.view %759, %762 : !torch.vtensor<[1,?],si64>, !torch.list<int> -> !torch.vtensor<[?],si64>
    torch.bind_symbolic_shape %763, [%63], affine_map<()[s0] -> (s0)> : !torch.vtensor<[?],si64>
    %int5_835 = torch.constant.int 5
    %764 = torch.prims.convert_element_type %755, %int5_835 : !torch.vtensor<[?,32,2,32],f16>, !torch.int -> !torch.vtensor<[?,32,2,32],f16>
    torch.bind_symbolic_shape %764, [%63], affine_map<()[s0] -> (s0, 32, 2, 32)> : !torch.vtensor<[?,32,2,32],f16>
    %int5_836 = torch.constant.int 5
    %765 = torch.prims.convert_element_type %757, %int5_836 : !torch.vtensor<[?,32,2,32],f16>, !torch.int -> !torch.vtensor<[?,32,2,32],f16>
    torch.bind_symbolic_shape %765, [%63], affine_map<()[s0] -> (s0, 32, 2, 32)> : !torch.vtensor<[?,32,2,32],f16>
    %766 = torch.prim.ListConstruct %761 : (!torch.vtensor<[?],si64>) -> !torch.list<optional<vtensor>>
    %false_837 = torch.constant.bool false
    %767 = torch.aten.index_put %735, %766, %764, %false_837 : !torch.vtensor<[?,32,2,32],f16>, !torch.list<optional<vtensor>>, !torch.vtensor<[?,32,2,32],f16>, !torch.bool -> !torch.vtensor<[?,32,2,32],f16>
    torch.bind_symbolic_shape %767, [%64], affine_map<()[s0] -> (s0 * 6, 32, 2, 32)> : !torch.vtensor<[?,32,2,32],f16>
    %int3_838 = torch.constant.int 3
    %int2_839 = torch.constant.int 2
    %int32_840 = torch.constant.int 32
    %int2_841 = torch.constant.int 2
    %int32_842 = torch.constant.int 32
    %768 = torch.prim.ListConstruct %66, %int3_838, %int2_839, %int32_840, %int2_841, %int32_842 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %769 = torch.aten.view %767, %768 : !torch.vtensor<[?,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,2,32],f16>
    torch.bind_symbolic_shape %769, [%64], affine_map<()[s0] -> (s0, 3, 2, 32, 2, 32)> : !torch.vtensor<[?,3,2,32,2,32],f16>
    %int12288_843 = torch.constant.int 12288
    %770 = torch.prim.ListConstruct %66, %int12288_843 : (!torch.int, !torch.int) -> !torch.list<int>
    %771 = torch.aten.view %769, %770 : !torch.vtensor<[?,3,2,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,12288],f16>
    torch.bind_symbolic_shape %771, [%64], affine_map<()[s0] -> (s0, 12288)> : !torch.vtensor<[?,12288],f16>
    %772 = torch.prim.ListConstruct %763 : (!torch.vtensor<[?],si64>) -> !torch.list<optional<vtensor>>
    %false_844 = torch.constant.bool false
    %773 = torch.aten.index_put %749, %772, %765, %false_844 : !torch.vtensor<[?,32,2,32],f16>, !torch.list<optional<vtensor>>, !torch.vtensor<[?,32,2,32],f16>, !torch.bool -> !torch.vtensor<[?,32,2,32],f16>
    torch.bind_symbolic_shape %773, [%64], affine_map<()[s0] -> (s0 * 6, 32, 2, 32)> : !torch.vtensor<[?,32,2,32],f16>
    %int3_845 = torch.constant.int 3
    %int2_846 = torch.constant.int 2
    %int32_847 = torch.constant.int 32
    %int2_848 = torch.constant.int 2
    %int32_849 = torch.constant.int 32
    %774 = torch.prim.ListConstruct %66, %int3_845, %int2_846, %int32_847, %int2_848, %int32_849 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %775 = torch.aten.view %773, %774 : !torch.vtensor<[?,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,2,32],f16>
    torch.bind_symbolic_shape %775, [%64], affine_map<()[s0] -> (s0, 3, 2, 32, 2, 32)> : !torch.vtensor<[?,3,2,32,2,32],f16>
    %int12288_850 = torch.constant.int 12288
    %776 = torch.prim.ListConstruct %66, %int12288_850 : (!torch.int, !torch.int) -> !torch.list<int>
    %777 = torch.aten.view %775, %776 : !torch.vtensor<[?,3,2,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,12288],f16>
    torch.bind_symbolic_shape %777, [%64], affine_map<()[s0] -> (s0, 12288)> : !torch.vtensor<[?,12288],f16>
    %int-2_851 = torch.constant.int -2
    %778 = torch.aten.unsqueeze %689, %int-2_851 : !torch.vtensor<[1,?,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,2,1,32],f16>
    torch.bind_symbolic_shape %778, [%63], affine_map<()[s0] -> (1, s0 * 32, 2, 1, 32)> : !torch.vtensor<[1,?,2,1,32],f16>
    %int-2_852 = torch.constant.int -2
    %779 = torch.aten.unsqueeze %703, %int-2_852 : !torch.vtensor<[1,?,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,2,1,32],f16>
    torch.bind_symbolic_shape %779, [%63], affine_map<()[s0] -> (1, s0 * 32, 2, 1, 32)> : !torch.vtensor<[1,?,2,1,32],f16>
    %int1_853 = torch.constant.int 1
    %int2_854 = torch.constant.int 2
    %int2_855 = torch.constant.int 2
    %int32_856 = torch.constant.int 32
    %780 = torch.prim.ListConstruct %int1_853, %67, %int2_854, %int2_855, %int32_856 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_857 = torch.constant.bool false
    %781 = torch.aten.expand %778, %780, %false_857 : !torch.vtensor<[1,?,2,1,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,?,2,2,32],f16>
    torch.bind_symbolic_shape %781, [%63], affine_map<()[s0] -> (1, s0 * 32, 2, 2, 32)> : !torch.vtensor<[1,?,2,2,32],f16>
    %int1_858 = torch.constant.int 1
    %int2_859 = torch.constant.int 2
    %int2_860 = torch.constant.int 2
    %int32_861 = torch.constant.int 32
    %782 = torch.prim.ListConstruct %int1_858, %67, %int2_859, %int2_860, %int32_861 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_862 = torch.constant.bool false
    %783 = torch.aten.expand %779, %782, %false_862 : !torch.vtensor<[1,?,2,1,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,?,2,2,32],f16>
    torch.bind_symbolic_shape %783, [%63], affine_map<()[s0] -> (1, s0 * 32, 2, 2, 32)> : !torch.vtensor<[1,?,2,2,32],f16>
    %int0_863 = torch.constant.int 0
    %784 = torch.aten.clone %781, %int0_863 : !torch.vtensor<[1,?,2,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,2,2,32],f16>
    torch.bind_symbolic_shape %784, [%63], affine_map<()[s0] -> (1, s0 * 32, 2, 2, 32)> : !torch.vtensor<[1,?,2,2,32],f16>
    %int1_864 = torch.constant.int 1
    %int4_865 = torch.constant.int 4
    %int32_866 = torch.constant.int 32
    %785 = torch.prim.ListConstruct %int1_864, %67, %int4_865, %int32_866 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %786 = torch.aten._unsafe_view %784, %785 : !torch.vtensor<[1,?,2,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %786, [%63], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int0_867 = torch.constant.int 0
    %787 = torch.aten.clone %783, %int0_867 : !torch.vtensor<[1,?,2,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,2,2,32],f16>
    torch.bind_symbolic_shape %787, [%63], affine_map<()[s0] -> (1, s0 * 32, 2, 2, 32)> : !torch.vtensor<[1,?,2,2,32],f16>
    %int1_868 = torch.constant.int 1
    %int4_869 = torch.constant.int 4
    %int32_870 = torch.constant.int 32
    %788 = torch.prim.ListConstruct %int1_868, %67, %int4_869, %int32_870 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %789 = torch.aten._unsafe_view %787, %788 : !torch.vtensor<[1,?,2,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %789, [%63], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int-2_871 = torch.constant.int -2
    %790 = torch.aten.unsqueeze %607, %int-2_871 : !torch.vtensor<[1,?,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,2,1,32],f16>
    torch.bind_symbolic_shape %790, [%63], affine_map<()[s0] -> (1, s0 * 32, 2, 1, 32)> : !torch.vtensor<[1,?,2,1,32],f16>
    %int-2_872 = torch.constant.int -2
    %791 = torch.aten.unsqueeze %609, %int-2_872 : !torch.vtensor<[1,?,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,2,1,32],f16>
    torch.bind_symbolic_shape %791, [%63], affine_map<()[s0] -> (1, s0 * 32, 2, 1, 32)> : !torch.vtensor<[1,?,2,1,32],f16>
    %int1_873 = torch.constant.int 1
    %int2_874 = torch.constant.int 2
    %int2_875 = torch.constant.int 2
    %int32_876 = torch.constant.int 32
    %792 = torch.prim.ListConstruct %int1_873, %67, %int2_874, %int2_875, %int32_876 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_877 = torch.constant.bool false
    %793 = torch.aten.expand %790, %792, %false_877 : !torch.vtensor<[1,?,2,1,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,?,2,2,32],f16>
    torch.bind_symbolic_shape %793, [%63], affine_map<()[s0] -> (1, s0 * 32, 2, 2, 32)> : !torch.vtensor<[1,?,2,2,32],f16>
    %int1_878 = torch.constant.int 1
    %int2_879 = torch.constant.int 2
    %int2_880 = torch.constant.int 2
    %int32_881 = torch.constant.int 32
    %794 = torch.prim.ListConstruct %int1_878, %67, %int2_879, %int2_880, %int32_881 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_882 = torch.constant.bool false
    %795 = torch.aten.expand %791, %794, %false_882 : !torch.vtensor<[1,?,2,1,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,?,2,2,32],f16>
    torch.bind_symbolic_shape %795, [%63], affine_map<()[s0] -> (1, s0 * 32, 2, 2, 32)> : !torch.vtensor<[1,?,2,2,32],f16>
    %int0_883 = torch.constant.int 0
    %796 = torch.aten.clone %793, %int0_883 : !torch.vtensor<[1,?,2,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,2,2,32],f16>
    torch.bind_symbolic_shape %796, [%63], affine_map<()[s0] -> (1, s0 * 32, 2, 2, 32)> : !torch.vtensor<[1,?,2,2,32],f16>
    %int1_884 = torch.constant.int 1
    %int4_885 = torch.constant.int 4
    %int32_886 = torch.constant.int 32
    %797 = torch.prim.ListConstruct %int1_884, %67, %int4_885, %int32_886 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %798 = torch.aten._unsafe_view %796, %797 : !torch.vtensor<[1,?,2,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %798, [%63], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int0_887 = torch.constant.int 0
    %799 = torch.aten.clone %795, %int0_887 : !torch.vtensor<[1,?,2,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,2,2,32],f16>
    torch.bind_symbolic_shape %799, [%63], affine_map<()[s0] -> (1, s0 * 32, 2, 2, 32)> : !torch.vtensor<[1,?,2,2,32],f16>
    %int1_888 = torch.constant.int 1
    %int4_889 = torch.constant.int 4
    %int32_890 = torch.constant.int 32
    %800 = torch.prim.ListConstruct %int1_888, %67, %int4_889, %int32_890 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %801 = torch.aten._unsafe_view %799, %800 : !torch.vtensor<[1,?,2,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %801, [%63], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int1_891 = torch.constant.int 1
    %int2_892 = torch.constant.int 2
    %802 = torch.aten.transpose.int %642, %int1_891, %int2_892 : !torch.vtensor<[1,?,4,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,4,?,32],f16>
    torch.bind_symbolic_shape %802, [%63], affine_map<()[s0] -> (1, 4, s0 * 32, 32)> : !torch.vtensor<[1,4,?,32],f16>
    %int1_893 = torch.constant.int 1
    %int2_894 = torch.constant.int 2
    %803 = torch.aten.transpose.int %656, %int1_893, %int2_894 : !torch.vtensor<[1,?,4,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,4,?,32],f16>
    torch.bind_symbolic_shape %803, [%63], affine_map<()[s0] -> (1, 4, s0 * 32, 32)> : !torch.vtensor<[1,4,?,32],f16>
    %int1_895 = torch.constant.int 1
    %int2_896 = torch.constant.int 2
    %804 = torch.aten.transpose.int %786, %int1_895, %int2_896 : !torch.vtensor<[1,?,4,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,4,?,32],f16>
    torch.bind_symbolic_shape %804, [%63], affine_map<()[s0] -> (1, 4, s0 * 32, 32)> : !torch.vtensor<[1,4,?,32],f16>
    %int1_897 = torch.constant.int 1
    %int2_898 = torch.constant.int 2
    %805 = torch.aten.transpose.int %789, %int1_897, %int2_898 : !torch.vtensor<[1,?,4,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,4,?,32],f16>
    torch.bind_symbolic_shape %805, [%63], affine_map<()[s0] -> (1, 4, s0 * 32, 32)> : !torch.vtensor<[1,4,?,32],f16>
    %int1_899 = torch.constant.int 1
    %int2_900 = torch.constant.int 2
    %806 = torch.aten.transpose.int %798, %int1_899, %int2_900 : !torch.vtensor<[1,?,4,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,4,?,32],f16>
    torch.bind_symbolic_shape %806, [%63], affine_map<()[s0] -> (1, 4, s0 * 32, 32)> : !torch.vtensor<[1,4,?,32],f16>
    %int1_901 = torch.constant.int 1
    %int2_902 = torch.constant.int 2
    %807 = torch.aten.transpose.int %801, %int1_901, %int2_902 : !torch.vtensor<[1,?,4,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,4,?,32],f16>
    torch.bind_symbolic_shape %807, [%63], affine_map<()[s0] -> (1, 4, s0 * 32, 32)> : !torch.vtensor<[1,4,?,32],f16>
    %int5_903 = torch.constant.int 5
    %808 = torch.prims.convert_element_type %802, %int5_903 : !torch.vtensor<[1,4,?,32],f16>, !torch.int -> !torch.vtensor<[1,4,?,32],f16>
    torch.bind_symbolic_shape %808, [%63], affine_map<()[s0] -> (1, 4, s0 * 32, 32)> : !torch.vtensor<[1,4,?,32],f16>
    %int5_904 = torch.constant.int 5
    %809 = torch.prims.convert_element_type %803, %int5_904 : !torch.vtensor<[1,4,?,32],f16>, !torch.int -> !torch.vtensor<[1,4,?,32],f16>
    torch.bind_symbolic_shape %809, [%63], affine_map<()[s0] -> (1, 4, s0 * 32, 32)> : !torch.vtensor<[1,4,?,32],f16>
    %int5_905 = torch.constant.int 5
    %810 = torch.prims.convert_element_type %804, %int5_905 : !torch.vtensor<[1,4,?,32],f16>, !torch.int -> !torch.vtensor<[1,4,?,32],f16>
    torch.bind_symbolic_shape %810, [%63], affine_map<()[s0] -> (1, 4, s0 * 32, 32)> : !torch.vtensor<[1,4,?,32],f16>
    %int5_906 = torch.constant.int 5
    %811 = torch.prims.convert_element_type %805, %int5_906 : !torch.vtensor<[1,4,?,32],f16>, !torch.int -> !torch.vtensor<[1,4,?,32],f16>
    torch.bind_symbolic_shape %811, [%63], affine_map<()[s0] -> (1, 4, s0 * 32, 32)> : !torch.vtensor<[1,4,?,32],f16>
    %int5_907 = torch.constant.int 5
    %812 = torch.prims.convert_element_type %806, %int5_907 : !torch.vtensor<[1,4,?,32],f16>, !torch.int -> !torch.vtensor<[1,4,?,32],f16>
    torch.bind_symbolic_shape %812, [%63], affine_map<()[s0] -> (1, 4, s0 * 32, 32)> : !torch.vtensor<[1,4,?,32],f16>
    %int5_908 = torch.constant.int 5
    %813 = torch.prims.convert_element_type %807, %int5_908 : !torch.vtensor<[1,4,?,32],f16>, !torch.int -> !torch.vtensor<[1,4,?,32],f16>
    torch.bind_symbolic_shape %813, [%63], affine_map<()[s0] -> (1, 4, s0 * 32, 32)> : !torch.vtensor<[1,4,?,32],f16>
    %int5_909 = torch.constant.int 5
    %814 = torch.prims.convert_element_type %102, %int5_909 : !torch.vtensor<[1,1,?,?],f16>, !torch.int -> !torch.vtensor<[1,1,?,?],f16>
    torch.bind_symbolic_shape %814, [%63], affine_map<()[s0] -> (1, 1, s0 * 32, s0 * 32)> : !torch.vtensor<[1,1,?,?],f16>
    %int5_910 = torch.constant.int 5
    %815 = torch.prims.convert_element_type %105, %int5_910 : !torch.vtensor<[1,1,?,?],f16>, !torch.int -> !torch.vtensor<[1,1,?,?],f16>
    torch.bind_symbolic_shape %815, [%63], affine_map<()[s0] -> (1, 1, s0 * 32, s0 * 32)> : !torch.vtensor<[1,1,?,?],f16>
    %float0.000000e00_911 = torch.constant.float 0.000000e+00
    %false_912 = torch.constant.bool false
    %none_913 = torch.constant.none
    %816:2 = torch.operator "torch.aten._scaled_dot_product_flash_attention_for_cpu"(%808, %810, %812, %float0.000000e00_911, %false_912, %814, %none_913) : (!torch.vtensor<[1,4,?,32],f16>, !torch.vtensor<[1,4,?,32],f16>, !torch.vtensor<[1,4,?,32],f16>, !torch.float, !torch.bool, !torch.vtensor<[1,1,?,?],f16>, !torch.none) -> (!torch.vtensor<[1,4,?,32],f16>, !torch.vtensor<[1,4,?],f32>) 
    torch.bind_symbolic_shape %816#0, [%63], affine_map<()[s0] -> (1, 4, s0 * 32, 32)> : !torch.vtensor<[1,4,?,32],f16>
    %float0.000000e00_914 = torch.constant.float 0.000000e+00
    %false_915 = torch.constant.bool false
    %none_916 = torch.constant.none
    %817:2 = torch.operator "torch.aten._scaled_dot_product_flash_attention_for_cpu"(%809, %811, %813, %float0.000000e00_914, %false_915, %815, %none_916) : (!torch.vtensor<[1,4,?,32],f16>, !torch.vtensor<[1,4,?,32],f16>, !torch.vtensor<[1,4,?,32],f16>, !torch.float, !torch.bool, !torch.vtensor<[1,1,?,?],f16>, !torch.none) -> (!torch.vtensor<[1,4,?,32],f16>, !torch.vtensor<[1,4,?],f32>) 
    torch.bind_symbolic_shape %817#0, [%63], affine_map<()[s0] -> (1, 4, s0 * 32, 32)> : !torch.vtensor<[1,4,?,32],f16>
    %int1_917 = torch.constant.int 1
    %int2_918 = torch.constant.int 2
    %818 = torch.aten.transpose.int %816#0, %int1_917, %int2_918 : !torch.vtensor<[1,4,?,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %818, [%63], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int1_919 = torch.constant.int 1
    %int2_920 = torch.constant.int 2
    %819 = torch.aten.transpose.int %817#0, %int1_919, %int2_920 : !torch.vtensor<[1,4,?,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %819, [%63], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int1_921 = torch.constant.int 1
    %int128_922 = torch.constant.int 128
    %820 = torch.prim.ListConstruct %int1_921, %67, %int128_922 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %821 = torch.aten.view %818, %820 : !torch.vtensor<[1,?,4,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,128],f16>
    torch.bind_symbolic_shape %821, [%63], affine_map<()[s0] -> (1, s0 * 32, 128)> : !torch.vtensor<[1,?,128],f16>
    %int1_923 = torch.constant.int 1
    %int128_924 = torch.constant.int 128
    %822 = torch.prim.ListConstruct %int1_923, %67, %int128_924 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %823 = torch.aten.view %819, %822 : !torch.vtensor<[1,?,4,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,128],f16>
    torch.bind_symbolic_shape %823, [%63], affine_map<()[s0] -> (1, s0 * 32, 128)> : !torch.vtensor<[1,?,128],f16>
    %int1_925 = torch.constant.int 1
    %int0_926 = torch.constant.int 0
    %824 = torch.prim.ListConstruct %int1_925, %int0_926 : (!torch.int, !torch.int) -> !torch.list<int>
    %825 = torch.aten.permute %28, %824 : !torch.vtensor<[256,128],f32>, !torch.list<int> -> !torch.vtensor<[128,256],f32>
    %int1_927 = torch.constant.int 1
    %int0_928 = torch.constant.int 0
    %826 = torch.prim.ListConstruct %int1_927, %int0_928 : (!torch.int, !torch.int) -> !torch.list<int>
    %827 = torch.aten.permute %29, %826 : !torch.vtensor<[256,128],f32>, !torch.list<int> -> !torch.vtensor<[128,256],f32>
    %int5_929 = torch.constant.int 5
    %828 = torch.prims.convert_element_type %825, %int5_929 : !torch.vtensor<[128,256],f32>, !torch.int -> !torch.vtensor<[128,256],f16>
    %int128_930 = torch.constant.int 128
    %829 = torch.prim.ListConstruct %67, %int128_930 : (!torch.int, !torch.int) -> !torch.list<int>
    %830 = torch.aten.view %821, %829 : !torch.vtensor<[1,?,128],f16>, !torch.list<int> -> !torch.vtensor<[?,128],f16>
    torch.bind_symbolic_shape %830, [%63], affine_map<()[s0] -> (s0 * 32, 128)> : !torch.vtensor<[?,128],f16>
    %831 = torch.aten.mm %830, %828 : !torch.vtensor<[?,128],f16>, !torch.vtensor<[128,256],f16> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %831, [%63], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %int1_931 = torch.constant.int 1
    %int256_932 = torch.constant.int 256
    %832 = torch.prim.ListConstruct %int1_931, %67, %int256_932 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %833 = torch.aten.view %831, %832 : !torch.vtensor<[?,256],f16>, !torch.list<int> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %833, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int5_933 = torch.constant.int 5
    %834 = torch.prims.convert_element_type %827, %int5_933 : !torch.vtensor<[128,256],f32>, !torch.int -> !torch.vtensor<[128,256],f16>
    %int128_934 = torch.constant.int 128
    %835 = torch.prim.ListConstruct %67, %int128_934 : (!torch.int, !torch.int) -> !torch.list<int>
    %836 = torch.aten.view %823, %835 : !torch.vtensor<[1,?,128],f16>, !torch.list<int> -> !torch.vtensor<[?,128],f16>
    torch.bind_symbolic_shape %836, [%63], affine_map<()[s0] -> (s0 * 32, 128)> : !torch.vtensor<[?,128],f16>
    %837 = torch.aten.mm %836, %834 : !torch.vtensor<[?,128],f16>, !torch.vtensor<[128,256],f16> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %837, [%63], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %int1_935 = torch.constant.int 1
    %int256_936 = torch.constant.int 256
    %838 = torch.prim.ListConstruct %int1_935, %67, %int256_936 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %839 = torch.aten.view %837, %838 : !torch.vtensor<[?,256],f16>, !torch.list<int> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %839, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %840 = torch_c.to_builtin_tensor %833 : !torch.vtensor<[1,?,256],f16> -> tensor<1x?x256xf16>
    %c1_937 = arith.constant 1 : index
    %dim_938 = tensor.dim %840, %c1_937 : tensor<1x?x256xf16>
    %841 = flow.tensor.barrier %840 : tensor<1x?x256xf16>{%dim_938} on #hal.device.promise<@__device_0>
    %842 = torch_c.from_builtin_tensor %841 : tensor<1x?x256xf16> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %842, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %843 = torch_c.to_builtin_tensor %839 : !torch.vtensor<[1,?,256],f16> -> tensor<1x?x256xf16>
    %c1_939 = arith.constant 1 : index
    %dim_940 = tensor.dim %843, %c1_939 : tensor<1x?x256xf16>
    %844 = flow.tensor.transfer %843 : tensor<1x?x256xf16>{%dim_940} to #hal.device.promise<@__device_0>
    %845 = torch_c.from_builtin_tensor %844 : tensor<1x?x256xf16> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %845, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int1_941 = torch.constant.int 1
    %846 = torch.aten.add.Tensor %842, %845, %int1_941 : !torch.vtensor<[1,?,256],f16>, !torch.vtensor<[1,?,256],f16>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %846, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %847 = torch_c.to_builtin_tensor %846 : !torch.vtensor<[1,?,256],f16> -> tensor<1x?x256xf16>
    %c1_942 = arith.constant 1 : index
    %dim_943 = tensor.dim %847, %c1_942 : tensor<1x?x256xf16>
    %848 = flow.tensor.barrier %847 : tensor<1x?x256xf16>{%dim_943} on #hal.device.promise<@__device_0>
    %849 = torch_c.from_builtin_tensor %848 : tensor<1x?x256xf16> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %849, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %850 = torch_c.to_builtin_tensor %846 : !torch.vtensor<[1,?,256],f16> -> tensor<1x?x256xf16>
    %c1_944 = arith.constant 1 : index
    %dim_945 = tensor.dim %850, %c1_944 : tensor<1x?x256xf16>
    %851 = flow.tensor.transfer %850 : tensor<1x?x256xf16>{%dim_945} to #hal.device.promise<@__device_1>
    %852 = torch_c.from_builtin_tensor %851 : tensor<1x?x256xf16> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %852, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int1_946 = torch.constant.int 1
    %853 = torch.aten.add.Tensor %528, %849, %int1_946 : !torch.vtensor<[1,?,256],f16>, !torch.vtensor<[1,?,256],f16>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %853, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int1_947 = torch.constant.int 1
    %854 = torch.aten.add.Tensor %529, %852, %int1_947 : !torch.vtensor<[1,?,256],f16>, !torch.vtensor<[1,?,256],f16>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %854, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int6_948 = torch.constant.int 6
    %855 = torch.prims.convert_element_type %853, %int6_948 : !torch.vtensor<[1,?,256],f16>, !torch.int -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %855, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int6_949 = torch.constant.int 6
    %856 = torch.prims.convert_element_type %854, %int6_949 : !torch.vtensor<[1,?,256],f16>, !torch.int -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %856, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int2_950 = torch.constant.int 2
    %857 = torch.aten.pow.Tensor_Scalar %855, %int2_950 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %857, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int2_951 = torch.constant.int 2
    %858 = torch.aten.pow.Tensor_Scalar %856, %int2_951 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %858, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int-1_952 = torch.constant.int -1
    %859 = torch.prim.ListConstruct %int-1_952 : (!torch.int) -> !torch.list<int>
    %true_953 = torch.constant.bool true
    %none_954 = torch.constant.none
    %860 = torch.aten.mean.dim %857, %859, %true_953, %none_954 : !torch.vtensor<[1,?,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %860, [%63], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %int-1_955 = torch.constant.int -1
    %861 = torch.prim.ListConstruct %int-1_955 : (!torch.int) -> !torch.list<int>
    %true_956 = torch.constant.bool true
    %none_957 = torch.constant.none
    %862 = torch.aten.mean.dim %858, %861, %true_956, %none_957 : !torch.vtensor<[1,?,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %862, [%63], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %float1.000000e-02_958 = torch.constant.float 1.000000e-02
    %int1_959 = torch.constant.int 1
    %863 = torch.aten.add.Scalar %860, %float1.000000e-02_958, %int1_959 : !torch.vtensor<[1,?,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %863, [%63], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %float1.000000e-02_960 = torch.constant.float 1.000000e-02
    %int1_961 = torch.constant.int 1
    %864 = torch.aten.add.Scalar %862, %float1.000000e-02_960, %int1_961 : !torch.vtensor<[1,?,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %864, [%63], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %865 = torch.aten.rsqrt %863 : !torch.vtensor<[1,?,1],f32> -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %865, [%63], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %866 = torch.aten.rsqrt %864 : !torch.vtensor<[1,?,1],f32> -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %866, [%63], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %867 = torch.aten.mul.Tensor %855, %865 : !torch.vtensor<[1,?,256],f32>, !torch.vtensor<[1,?,1],f32> -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %867, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %868 = torch.aten.mul.Tensor %856, %866 : !torch.vtensor<[1,?,256],f32>, !torch.vtensor<[1,?,1],f32> -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %868, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int5_962 = torch.constant.int 5
    %869 = torch.prims.convert_element_type %867, %int5_962 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %869, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int5_963 = torch.constant.int 5
    %870 = torch.prims.convert_element_type %868, %int5_963 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %870, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %871 = torch.aten.mul.Tensor %30, %869 : !torch.vtensor<[256],f32>, !torch.vtensor<[1,?,256],f16> -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %871, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %872 = torch.aten.mul.Tensor %31, %870 : !torch.vtensor<[256],f32>, !torch.vtensor<[1,?,256],f16> -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %872, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int5_964 = torch.constant.int 5
    %873 = torch.prims.convert_element_type %871, %int5_964 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %873, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int5_965 = torch.constant.int 5
    %874 = torch.prims.convert_element_type %872, %int5_965 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %874, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int1_966 = torch.constant.int 1
    %int0_967 = torch.constant.int 0
    %875 = torch.prim.ListConstruct %int1_966, %int0_967 : (!torch.int, !torch.int) -> !torch.list<int>
    %876 = torch.aten.permute %32, %875 : !torch.vtensor<[12,256],f32>, !torch.list<int> -> !torch.vtensor<[256,12],f32>
    %int1_968 = torch.constant.int 1
    %int0_969 = torch.constant.int 0
    %877 = torch.prim.ListConstruct %int1_968, %int0_969 : (!torch.int, !torch.int) -> !torch.list<int>
    %878 = torch.aten.permute %33, %877 : !torch.vtensor<[11,256],f32>, !torch.list<int> -> !torch.vtensor<[256,11],f32>
    %int5_970 = torch.constant.int 5
    %879 = torch.prims.convert_element_type %876, %int5_970 : !torch.vtensor<[256,12],f32>, !torch.int -> !torch.vtensor<[256,12],f16>
    %int256_971 = torch.constant.int 256
    %880 = torch.prim.ListConstruct %67, %int256_971 : (!torch.int, !torch.int) -> !torch.list<int>
    %881 = torch.aten.view %873, %880 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %881, [%63], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %882 = torch.aten.mm %881, %879 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,12],f16> -> !torch.vtensor<[?,12],f16>
    torch.bind_symbolic_shape %882, [%63], affine_map<()[s0] -> (s0 * 32, 12)> : !torch.vtensor<[?,12],f16>
    %int1_972 = torch.constant.int 1
    %int12_973 = torch.constant.int 12
    %883 = torch.prim.ListConstruct %int1_972, %67, %int12_973 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %884 = torch.aten.view %882, %883 : !torch.vtensor<[?,12],f16>, !torch.list<int> -> !torch.vtensor<[1,?,12],f16>
    torch.bind_symbolic_shape %884, [%63], affine_map<()[s0] -> (1, s0 * 32, 12)> : !torch.vtensor<[1,?,12],f16>
    %int5_974 = torch.constant.int 5
    %885 = torch.prims.convert_element_type %878, %int5_974 : !torch.vtensor<[256,11],f32>, !torch.int -> !torch.vtensor<[256,11],f16>
    %int256_975 = torch.constant.int 256
    %886 = torch.prim.ListConstruct %67, %int256_975 : (!torch.int, !torch.int) -> !torch.list<int>
    %887 = torch.aten.view %874, %886 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %887, [%63], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %888 = torch.aten.mm %887, %885 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,11],f16> -> !torch.vtensor<[?,11],f16>
    torch.bind_symbolic_shape %888, [%63], affine_map<()[s0] -> (s0 * 32, 11)> : !torch.vtensor<[?,11],f16>
    %int1_976 = torch.constant.int 1
    %int11_977 = torch.constant.int 11
    %889 = torch.prim.ListConstruct %int1_976, %67, %int11_977 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %890 = torch.aten.view %888, %889 : !torch.vtensor<[?,11],f16>, !torch.list<int> -> !torch.vtensor<[1,?,11],f16>
    torch.bind_symbolic_shape %890, [%63], affine_map<()[s0] -> (1, s0 * 32, 11)> : !torch.vtensor<[1,?,11],f16>
    %891 = torch.aten.silu %884 : !torch.vtensor<[1,?,12],f16> -> !torch.vtensor<[1,?,12],f16>
    torch.bind_symbolic_shape %891, [%63], affine_map<()[s0] -> (1, s0 * 32, 12)> : !torch.vtensor<[1,?,12],f16>
    %892 = torch.aten.silu %890 : !torch.vtensor<[1,?,11],f16> -> !torch.vtensor<[1,?,11],f16>
    torch.bind_symbolic_shape %892, [%63], affine_map<()[s0] -> (1, s0 * 32, 11)> : !torch.vtensor<[1,?,11],f16>
    %int1_978 = torch.constant.int 1
    %int0_979 = torch.constant.int 0
    %893 = torch.prim.ListConstruct %int1_978, %int0_979 : (!torch.int, !torch.int) -> !torch.list<int>
    %894 = torch.aten.permute %34, %893 : !torch.vtensor<[12,256],f32>, !torch.list<int> -> !torch.vtensor<[256,12],f32>
    %int1_980 = torch.constant.int 1
    %int0_981 = torch.constant.int 0
    %895 = torch.prim.ListConstruct %int1_980, %int0_981 : (!torch.int, !torch.int) -> !torch.list<int>
    %896 = torch.aten.permute %35, %895 : !torch.vtensor<[11,256],f32>, !torch.list<int> -> !torch.vtensor<[256,11],f32>
    %int5_982 = torch.constant.int 5
    %897 = torch.prims.convert_element_type %894, %int5_982 : !torch.vtensor<[256,12],f32>, !torch.int -> !torch.vtensor<[256,12],f16>
    %int256_983 = torch.constant.int 256
    %898 = torch.prim.ListConstruct %67, %int256_983 : (!torch.int, !torch.int) -> !torch.list<int>
    %899 = torch.aten.view %873, %898 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %899, [%63], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %900 = torch.aten.mm %899, %897 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,12],f16> -> !torch.vtensor<[?,12],f16>
    torch.bind_symbolic_shape %900, [%63], affine_map<()[s0] -> (s0 * 32, 12)> : !torch.vtensor<[?,12],f16>
    %int1_984 = torch.constant.int 1
    %int12_985 = torch.constant.int 12
    %901 = torch.prim.ListConstruct %int1_984, %67, %int12_985 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %902 = torch.aten.view %900, %901 : !torch.vtensor<[?,12],f16>, !torch.list<int> -> !torch.vtensor<[1,?,12],f16>
    torch.bind_symbolic_shape %902, [%63], affine_map<()[s0] -> (1, s0 * 32, 12)> : !torch.vtensor<[1,?,12],f16>
    %int5_986 = torch.constant.int 5
    %903 = torch.prims.convert_element_type %896, %int5_986 : !torch.vtensor<[256,11],f32>, !torch.int -> !torch.vtensor<[256,11],f16>
    %int256_987 = torch.constant.int 256
    %904 = torch.prim.ListConstruct %67, %int256_987 : (!torch.int, !torch.int) -> !torch.list<int>
    %905 = torch.aten.view %874, %904 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %905, [%63], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %906 = torch.aten.mm %905, %903 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,11],f16> -> !torch.vtensor<[?,11],f16>
    torch.bind_symbolic_shape %906, [%63], affine_map<()[s0] -> (s0 * 32, 11)> : !torch.vtensor<[?,11],f16>
    %int1_988 = torch.constant.int 1
    %int11_989 = torch.constant.int 11
    %907 = torch.prim.ListConstruct %int1_988, %67, %int11_989 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %908 = torch.aten.view %906, %907 : !torch.vtensor<[?,11],f16>, !torch.list<int> -> !torch.vtensor<[1,?,11],f16>
    torch.bind_symbolic_shape %908, [%63], affine_map<()[s0] -> (1, s0 * 32, 11)> : !torch.vtensor<[1,?,11],f16>
    %909 = torch.aten.mul.Tensor %891, %902 : !torch.vtensor<[1,?,12],f16>, !torch.vtensor<[1,?,12],f16> -> !torch.vtensor<[1,?,12],f16>
    torch.bind_symbolic_shape %909, [%63], affine_map<()[s0] -> (1, s0 * 32, 12)> : !torch.vtensor<[1,?,12],f16>
    %910 = torch.aten.mul.Tensor %892, %908 : !torch.vtensor<[1,?,11],f16>, !torch.vtensor<[1,?,11],f16> -> !torch.vtensor<[1,?,11],f16>
    torch.bind_symbolic_shape %910, [%63], affine_map<()[s0] -> (1, s0 * 32, 11)> : !torch.vtensor<[1,?,11],f16>
    %int1_990 = torch.constant.int 1
    %int0_991 = torch.constant.int 0
    %911 = torch.prim.ListConstruct %int1_990, %int0_991 : (!torch.int, !torch.int) -> !torch.list<int>
    %912 = torch.aten.permute %36, %911 : !torch.vtensor<[256,12],f32>, !torch.list<int> -> !torch.vtensor<[12,256],f32>
    %int1_992 = torch.constant.int 1
    %int0_993 = torch.constant.int 0
    %913 = torch.prim.ListConstruct %int1_992, %int0_993 : (!torch.int, !torch.int) -> !torch.list<int>
    %914 = torch.aten.permute %37, %913 : !torch.vtensor<[256,11],f32>, !torch.list<int> -> !torch.vtensor<[11,256],f32>
    %int5_994 = torch.constant.int 5
    %915 = torch.prims.convert_element_type %912, %int5_994 : !torch.vtensor<[12,256],f32>, !torch.int -> !torch.vtensor<[12,256],f16>
    %int12_995 = torch.constant.int 12
    %916 = torch.prim.ListConstruct %67, %int12_995 : (!torch.int, !torch.int) -> !torch.list<int>
    %917 = torch.aten.view %909, %916 : !torch.vtensor<[1,?,12],f16>, !torch.list<int> -> !torch.vtensor<[?,12],f16>
    torch.bind_symbolic_shape %917, [%63], affine_map<()[s0] -> (s0 * 32, 12)> : !torch.vtensor<[?,12],f16>
    %918 = torch.aten.mm %917, %915 : !torch.vtensor<[?,12],f16>, !torch.vtensor<[12,256],f16> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %918, [%63], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %int1_996 = torch.constant.int 1
    %int256_997 = torch.constant.int 256
    %919 = torch.prim.ListConstruct %int1_996, %67, %int256_997 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %920 = torch.aten.view %918, %919 : !torch.vtensor<[?,256],f16>, !torch.list<int> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %920, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int5_998 = torch.constant.int 5
    %921 = torch.prims.convert_element_type %914, %int5_998 : !torch.vtensor<[11,256],f32>, !torch.int -> !torch.vtensor<[11,256],f16>
    %int11_999 = torch.constant.int 11
    %922 = torch.prim.ListConstruct %67, %int11_999 : (!torch.int, !torch.int) -> !torch.list<int>
    %923 = torch.aten.view %910, %922 : !torch.vtensor<[1,?,11],f16>, !torch.list<int> -> !torch.vtensor<[?,11],f16>
    torch.bind_symbolic_shape %923, [%63], affine_map<()[s0] -> (s0 * 32, 11)> : !torch.vtensor<[?,11],f16>
    %924 = torch.aten.mm %923, %921 : !torch.vtensor<[?,11],f16>, !torch.vtensor<[11,256],f16> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %924, [%63], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %int1_1000 = torch.constant.int 1
    %int256_1001 = torch.constant.int 256
    %925 = torch.prim.ListConstruct %int1_1000, %67, %int256_1001 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %926 = torch.aten.view %924, %925 : !torch.vtensor<[?,256],f16>, !torch.list<int> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %926, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %927 = torch_c.to_builtin_tensor %920 : !torch.vtensor<[1,?,256],f16> -> tensor<1x?x256xf16>
    %c1_1002 = arith.constant 1 : index
    %dim_1003 = tensor.dim %927, %c1_1002 : tensor<1x?x256xf16>
    %928 = flow.tensor.barrier %927 : tensor<1x?x256xf16>{%dim_1003} on #hal.device.promise<@__device_0>
    %929 = torch_c.from_builtin_tensor %928 : tensor<1x?x256xf16> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %929, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %930 = torch_c.to_builtin_tensor %926 : !torch.vtensor<[1,?,256],f16> -> tensor<1x?x256xf16>
    %c1_1004 = arith.constant 1 : index
    %dim_1005 = tensor.dim %930, %c1_1004 : tensor<1x?x256xf16>
    %931 = flow.tensor.transfer %930 : tensor<1x?x256xf16>{%dim_1005} to #hal.device.promise<@__device_0>
    %932 = torch_c.from_builtin_tensor %931 : tensor<1x?x256xf16> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %932, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int1_1006 = torch.constant.int 1
    %933 = torch.aten.add.Tensor %929, %932, %int1_1006 : !torch.vtensor<[1,?,256],f16>, !torch.vtensor<[1,?,256],f16>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %933, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %934 = torch_c.to_builtin_tensor %933 : !torch.vtensor<[1,?,256],f16> -> tensor<1x?x256xf16>
    %c1_1007 = arith.constant 1 : index
    %dim_1008 = tensor.dim %934, %c1_1007 : tensor<1x?x256xf16>
    %935 = flow.tensor.barrier %934 : tensor<1x?x256xf16>{%dim_1008} on #hal.device.promise<@__device_0>
    %936 = torch_c.from_builtin_tensor %935 : tensor<1x?x256xf16> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %936, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %937 = torch_c.to_builtin_tensor %933 : !torch.vtensor<[1,?,256],f16> -> tensor<1x?x256xf16>
    %c1_1009 = arith.constant 1 : index
    %dim_1010 = tensor.dim %937, %c1_1009 : tensor<1x?x256xf16>
    %938 = flow.tensor.transfer %937 : tensor<1x?x256xf16>{%dim_1010} to #hal.device.promise<@__device_1>
    %939 = torch_c.from_builtin_tensor %938 : tensor<1x?x256xf16> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %939, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int1_1011 = torch.constant.int 1
    %940 = torch.aten.add.Tensor %853, %936, %int1_1011 : !torch.vtensor<[1,?,256],f16>, !torch.vtensor<[1,?,256],f16>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %940, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int1_1012 = torch.constant.int 1
    %941 = torch.aten.add.Tensor %854, %939, %int1_1012 : !torch.vtensor<[1,?,256],f16>, !torch.vtensor<[1,?,256],f16>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %941, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int6_1013 = torch.constant.int 6
    %942 = torch.prims.convert_element_type %940, %int6_1013 : !torch.vtensor<[1,?,256],f16>, !torch.int -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %942, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int6_1014 = torch.constant.int 6
    %943 = torch.prims.convert_element_type %941, %int6_1014 : !torch.vtensor<[1,?,256],f16>, !torch.int -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %943, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int2_1015 = torch.constant.int 2
    %944 = torch.aten.pow.Tensor_Scalar %942, %int2_1015 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %944, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int2_1016 = torch.constant.int 2
    %945 = torch.aten.pow.Tensor_Scalar %943, %int2_1016 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %945, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int-1_1017 = torch.constant.int -1
    %946 = torch.prim.ListConstruct %int-1_1017 : (!torch.int) -> !torch.list<int>
    %true_1018 = torch.constant.bool true
    %none_1019 = torch.constant.none
    %947 = torch.aten.mean.dim %944, %946, %true_1018, %none_1019 : !torch.vtensor<[1,?,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %947, [%63], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %int-1_1020 = torch.constant.int -1
    %948 = torch.prim.ListConstruct %int-1_1020 : (!torch.int) -> !torch.list<int>
    %true_1021 = torch.constant.bool true
    %none_1022 = torch.constant.none
    %949 = torch.aten.mean.dim %945, %948, %true_1021, %none_1022 : !torch.vtensor<[1,?,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %949, [%63], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %float1.000000e-02_1023 = torch.constant.float 1.000000e-02
    %int1_1024 = torch.constant.int 1
    %950 = torch.aten.add.Scalar %947, %float1.000000e-02_1023, %int1_1024 : !torch.vtensor<[1,?,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %950, [%63], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %float1.000000e-02_1025 = torch.constant.float 1.000000e-02
    %int1_1026 = torch.constant.int 1
    %951 = torch.aten.add.Scalar %949, %float1.000000e-02_1025, %int1_1026 : !torch.vtensor<[1,?,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %951, [%63], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %952 = torch.aten.rsqrt %950 : !torch.vtensor<[1,?,1],f32> -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %952, [%63], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %953 = torch.aten.rsqrt %951 : !torch.vtensor<[1,?,1],f32> -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %953, [%63], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %954 = torch.aten.mul.Tensor %942, %952 : !torch.vtensor<[1,?,256],f32>, !torch.vtensor<[1,?,1],f32> -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %954, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %955 = torch.aten.mul.Tensor %943, %953 : !torch.vtensor<[1,?,256],f32>, !torch.vtensor<[1,?,1],f32> -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %955, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int5_1027 = torch.constant.int 5
    %956 = torch.prims.convert_element_type %954, %int5_1027 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %956, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int5_1028 = torch.constant.int 5
    %957 = torch.prims.convert_element_type %955, %int5_1028 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %957, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %958 = torch.aten.mul.Tensor %38, %956 : !torch.vtensor<[256],f32>, !torch.vtensor<[1,?,256],f16> -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %958, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %959 = torch.aten.mul.Tensor %39, %957 : !torch.vtensor<[256],f32>, !torch.vtensor<[1,?,256],f16> -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %959, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int5_1029 = torch.constant.int 5
    %960 = torch.prims.convert_element_type %958, %int5_1029 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %960, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int5_1030 = torch.constant.int 5
    %961 = torch.prims.convert_element_type %959, %int5_1030 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %961, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int1_1031 = torch.constant.int 1
    %int0_1032 = torch.constant.int 0
    %962 = torch.prim.ListConstruct %int1_1031, %int0_1032 : (!torch.int, !torch.int) -> !torch.list<int>
    %963 = torch.aten.permute %40, %962 : !torch.vtensor<[128,256],f32>, !torch.list<int> -> !torch.vtensor<[256,128],f32>
    %int1_1033 = torch.constant.int 1
    %int0_1034 = torch.constant.int 0
    %964 = torch.prim.ListConstruct %int1_1033, %int0_1034 : (!torch.int, !torch.int) -> !torch.list<int>
    %965 = torch.aten.permute %41, %964 : !torch.vtensor<[128,256],f32>, !torch.list<int> -> !torch.vtensor<[256,128],f32>
    %int5_1035 = torch.constant.int 5
    %966 = torch.prims.convert_element_type %963, %int5_1035 : !torch.vtensor<[256,128],f32>, !torch.int -> !torch.vtensor<[256,128],f16>
    %int256_1036 = torch.constant.int 256
    %967 = torch.prim.ListConstruct %67, %int256_1036 : (!torch.int, !torch.int) -> !torch.list<int>
    %968 = torch.aten.view %960, %967 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %968, [%63], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %969 = torch.aten.mm %968, %966 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,128],f16> -> !torch.vtensor<[?,128],f16>
    torch.bind_symbolic_shape %969, [%63], affine_map<()[s0] -> (s0 * 32, 128)> : !torch.vtensor<[?,128],f16>
    %int1_1037 = torch.constant.int 1
    %int128_1038 = torch.constant.int 128
    %970 = torch.prim.ListConstruct %int1_1037, %67, %int128_1038 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %971 = torch.aten.view %969, %970 : !torch.vtensor<[?,128],f16>, !torch.list<int> -> !torch.vtensor<[1,?,128],f16>
    torch.bind_symbolic_shape %971, [%63], affine_map<()[s0] -> (1, s0 * 32, 128)> : !torch.vtensor<[1,?,128],f16>
    %int5_1039 = torch.constant.int 5
    %972 = torch.prims.convert_element_type %965, %int5_1039 : !torch.vtensor<[256,128],f32>, !torch.int -> !torch.vtensor<[256,128],f16>
    %int256_1040 = torch.constant.int 256
    %973 = torch.prim.ListConstruct %67, %int256_1040 : (!torch.int, !torch.int) -> !torch.list<int>
    %974 = torch.aten.view %961, %973 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %974, [%63], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %975 = torch.aten.mm %974, %972 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,128],f16> -> !torch.vtensor<[?,128],f16>
    torch.bind_symbolic_shape %975, [%63], affine_map<()[s0] -> (s0 * 32, 128)> : !torch.vtensor<[?,128],f16>
    %int1_1041 = torch.constant.int 1
    %int128_1042 = torch.constant.int 128
    %976 = torch.prim.ListConstruct %int1_1041, %67, %int128_1042 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %977 = torch.aten.view %975, %976 : !torch.vtensor<[?,128],f16>, !torch.list<int> -> !torch.vtensor<[1,?,128],f16>
    torch.bind_symbolic_shape %977, [%63], affine_map<()[s0] -> (1, s0 * 32, 128)> : !torch.vtensor<[1,?,128],f16>
    %int1_1043 = torch.constant.int 1
    %int0_1044 = torch.constant.int 0
    %978 = torch.prim.ListConstruct %int1_1043, %int0_1044 : (!torch.int, !torch.int) -> !torch.list<int>
    %979 = torch.aten.permute %42, %978 : !torch.vtensor<[64,256],f32>, !torch.list<int> -> !torch.vtensor<[256,64],f32>
    %int1_1045 = torch.constant.int 1
    %int0_1046 = torch.constant.int 0
    %980 = torch.prim.ListConstruct %int1_1045, %int0_1046 : (!torch.int, !torch.int) -> !torch.list<int>
    %981 = torch.aten.permute %43, %980 : !torch.vtensor<[64,256],f32>, !torch.list<int> -> !torch.vtensor<[256,64],f32>
    %int5_1047 = torch.constant.int 5
    %982 = torch.prims.convert_element_type %979, %int5_1047 : !torch.vtensor<[256,64],f32>, !torch.int -> !torch.vtensor<[256,64],f16>
    %int256_1048 = torch.constant.int 256
    %983 = torch.prim.ListConstruct %67, %int256_1048 : (!torch.int, !torch.int) -> !torch.list<int>
    %984 = torch.aten.view %960, %983 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %984, [%63], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %985 = torch.aten.mm %984, %982 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,64],f16> -> !torch.vtensor<[?,64],f16>
    torch.bind_symbolic_shape %985, [%63], affine_map<()[s0] -> (s0 * 32, 64)> : !torch.vtensor<[?,64],f16>
    %int1_1049 = torch.constant.int 1
    %int64_1050 = torch.constant.int 64
    %986 = torch.prim.ListConstruct %int1_1049, %67, %int64_1050 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %987 = torch.aten.view %985, %986 : !torch.vtensor<[?,64],f16>, !torch.list<int> -> !torch.vtensor<[1,?,64],f16>
    torch.bind_symbolic_shape %987, [%63], affine_map<()[s0] -> (1, s0 * 32, 64)> : !torch.vtensor<[1,?,64],f16>
    %int5_1051 = torch.constant.int 5
    %988 = torch.prims.convert_element_type %981, %int5_1051 : !torch.vtensor<[256,64],f32>, !torch.int -> !torch.vtensor<[256,64],f16>
    %int256_1052 = torch.constant.int 256
    %989 = torch.prim.ListConstruct %67, %int256_1052 : (!torch.int, !torch.int) -> !torch.list<int>
    %990 = torch.aten.view %961, %989 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %990, [%63], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %991 = torch.aten.mm %990, %988 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,64],f16> -> !torch.vtensor<[?,64],f16>
    torch.bind_symbolic_shape %991, [%63], affine_map<()[s0] -> (s0 * 32, 64)> : !torch.vtensor<[?,64],f16>
    %int1_1053 = torch.constant.int 1
    %int64_1054 = torch.constant.int 64
    %992 = torch.prim.ListConstruct %int1_1053, %67, %int64_1054 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %993 = torch.aten.view %991, %992 : !torch.vtensor<[?,64],f16>, !torch.list<int> -> !torch.vtensor<[1,?,64],f16>
    torch.bind_symbolic_shape %993, [%63], affine_map<()[s0] -> (1, s0 * 32, 64)> : !torch.vtensor<[1,?,64],f16>
    %int1_1055 = torch.constant.int 1
    %int0_1056 = torch.constant.int 0
    %994 = torch.prim.ListConstruct %int1_1055, %int0_1056 : (!torch.int, !torch.int) -> !torch.list<int>
    %995 = torch.aten.permute %44, %994 : !torch.vtensor<[64,256],f32>, !torch.list<int> -> !torch.vtensor<[256,64],f32>
    %int1_1057 = torch.constant.int 1
    %int0_1058 = torch.constant.int 0
    %996 = torch.prim.ListConstruct %int1_1057, %int0_1058 : (!torch.int, !torch.int) -> !torch.list<int>
    %997 = torch.aten.permute %45, %996 : !torch.vtensor<[64,256],f32>, !torch.list<int> -> !torch.vtensor<[256,64],f32>
    %int5_1059 = torch.constant.int 5
    %998 = torch.prims.convert_element_type %995, %int5_1059 : !torch.vtensor<[256,64],f32>, !torch.int -> !torch.vtensor<[256,64],f16>
    %int256_1060 = torch.constant.int 256
    %999 = torch.prim.ListConstruct %67, %int256_1060 : (!torch.int, !torch.int) -> !torch.list<int>
    %1000 = torch.aten.view %960, %999 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %1000, [%63], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %1001 = torch.aten.mm %1000, %998 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,64],f16> -> !torch.vtensor<[?,64],f16>
    torch.bind_symbolic_shape %1001, [%63], affine_map<()[s0] -> (s0 * 32, 64)> : !torch.vtensor<[?,64],f16>
    %int1_1061 = torch.constant.int 1
    %int64_1062 = torch.constant.int 64
    %1002 = torch.prim.ListConstruct %int1_1061, %67, %int64_1062 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1003 = torch.aten.view %1001, %1002 : !torch.vtensor<[?,64],f16>, !torch.list<int> -> !torch.vtensor<[1,?,64],f16>
    torch.bind_symbolic_shape %1003, [%63], affine_map<()[s0] -> (1, s0 * 32, 64)> : !torch.vtensor<[1,?,64],f16>
    %int5_1063 = torch.constant.int 5
    %1004 = torch.prims.convert_element_type %997, %int5_1063 : !torch.vtensor<[256,64],f32>, !torch.int -> !torch.vtensor<[256,64],f16>
    %int256_1064 = torch.constant.int 256
    %1005 = torch.prim.ListConstruct %67, %int256_1064 : (!torch.int, !torch.int) -> !torch.list<int>
    %1006 = torch.aten.view %961, %1005 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %1006, [%63], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %1007 = torch.aten.mm %1006, %1004 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,64],f16> -> !torch.vtensor<[?,64],f16>
    torch.bind_symbolic_shape %1007, [%63], affine_map<()[s0] -> (s0 * 32, 64)> : !torch.vtensor<[?,64],f16>
    %int1_1065 = torch.constant.int 1
    %int64_1066 = torch.constant.int 64
    %1008 = torch.prim.ListConstruct %int1_1065, %67, %int64_1066 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1009 = torch.aten.view %1007, %1008 : !torch.vtensor<[?,64],f16>, !torch.list<int> -> !torch.vtensor<[1,?,64],f16>
    torch.bind_symbolic_shape %1009, [%63], affine_map<()[s0] -> (1, s0 * 32, 64)> : !torch.vtensor<[1,?,64],f16>
    %int1_1067 = torch.constant.int 1
    %int4_1068 = torch.constant.int 4
    %int32_1069 = torch.constant.int 32
    %1010 = torch.prim.ListConstruct %int1_1067, %67, %int4_1068, %int32_1069 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1011 = torch.aten.view %971, %1010 : !torch.vtensor<[1,?,128],f16>, !torch.list<int> -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %1011, [%63], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int1_1070 = torch.constant.int 1
    %int4_1071 = torch.constant.int 4
    %int32_1072 = torch.constant.int 32
    %1012 = torch.prim.ListConstruct %int1_1070, %67, %int4_1071, %int32_1072 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1013 = torch.aten.view %977, %1012 : !torch.vtensor<[1,?,128],f16>, !torch.list<int> -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %1013, [%63], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int1_1073 = torch.constant.int 1
    %int2_1074 = torch.constant.int 2
    %int32_1075 = torch.constant.int 32
    %1014 = torch.prim.ListConstruct %int1_1073, %67, %int2_1074, %int32_1075 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1015 = torch.aten.view %987, %1014 : !torch.vtensor<[1,?,64],f16>, !torch.list<int> -> !torch.vtensor<[1,?,2,32],f16>
    torch.bind_symbolic_shape %1015, [%63], affine_map<()[s0] -> (1, s0 * 32, 2, 32)> : !torch.vtensor<[1,?,2,32],f16>
    %int1_1076 = torch.constant.int 1
    %int2_1077 = torch.constant.int 2
    %int32_1078 = torch.constant.int 32
    %1016 = torch.prim.ListConstruct %int1_1076, %67, %int2_1077, %int32_1078 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1017 = torch.aten.view %993, %1016 : !torch.vtensor<[1,?,64],f16>, !torch.list<int> -> !torch.vtensor<[1,?,2,32],f16>
    torch.bind_symbolic_shape %1017, [%63], affine_map<()[s0] -> (1, s0 * 32, 2, 32)> : !torch.vtensor<[1,?,2,32],f16>
    %int1_1079 = torch.constant.int 1
    %int2_1080 = torch.constant.int 2
    %int32_1081 = torch.constant.int 32
    %1018 = torch.prim.ListConstruct %int1_1079, %67, %int2_1080, %int32_1081 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1019 = torch.aten.view %1003, %1018 : !torch.vtensor<[1,?,64],f16>, !torch.list<int> -> !torch.vtensor<[1,?,2,32],f16>
    torch.bind_symbolic_shape %1019, [%63], affine_map<()[s0] -> (1, s0 * 32, 2, 32)> : !torch.vtensor<[1,?,2,32],f16>
    %int1_1082 = torch.constant.int 1
    %int2_1083 = torch.constant.int 2
    %int32_1084 = torch.constant.int 32
    %1020 = torch.prim.ListConstruct %int1_1082, %67, %int2_1083, %int32_1084 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1021 = torch.aten.view %1009, %1020 : !torch.vtensor<[1,?,64],f16>, !torch.list<int> -> !torch.vtensor<[1,?,2,32],f16>
    torch.bind_symbolic_shape %1021, [%63], affine_map<()[s0] -> (1, s0 * 32, 2, 32)> : !torch.vtensor<[1,?,2,32],f16>
    %int128_1085 = torch.constant.int 128
    %none_1086 = torch.constant.none
    %none_1087 = torch.constant.none
    %cpu_1088 = torch.constant.device "cpu"
    %false_1089 = torch.constant.bool false
    %1022 = torch.aten.arange %int128_1085, %none_1086, %none_1087, %cpu_1088, %false_1089 : !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[128],si64>
    %int0_1090 = torch.constant.int 0
    %int32_1091 = torch.constant.int 32
    %none_1092 = torch.constant.none
    %none_1093 = torch.constant.none
    %cpu_1094 = torch.constant.device "cpu"
    %false_1095 = torch.constant.bool false
    %1023 = torch.aten.arange.start %int0_1090, %int32_1091, %none_1092, %none_1093, %cpu_1094, %false_1095 : !torch.int, !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[32],si64>
    %int2_1096 = torch.constant.int 2
    %1024 = torch.aten.floor_divide.Scalar %1023, %int2_1096 : !torch.vtensor<[32],si64>, !torch.int -> !torch.vtensor<[32],si64>
    %int6_1097 = torch.constant.int 6
    %1025 = torch.prims.convert_element_type %1024, %int6_1097 : !torch.vtensor<[32],si64>, !torch.int -> !torch.vtensor<[32],f32>
    %int32_1098 = torch.constant.int 32
    %1026 = torch.aten.div.Scalar %1025, %int32_1098 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[32],f32>
    %float2.000000e00_1099 = torch.constant.float 2.000000e+00
    %1027 = torch.aten.mul.Scalar %1026, %float2.000000e00_1099 : !torch.vtensor<[32],f32>, !torch.float -> !torch.vtensor<[32],f32>
    %float5.000000e05_1100 = torch.constant.float 5.000000e+05
    %1028 = torch.aten.pow.Scalar %float5.000000e05_1100, %1027 : !torch.float, !torch.vtensor<[32],f32> -> !torch.vtensor<[32],f32>
    %1029 = torch.aten.reciprocal %1028 : !torch.vtensor<[32],f32> -> !torch.vtensor<[32],f32>
    %float1.000000e00_1101 = torch.constant.float 1.000000e+00
    %1030 = torch.aten.mul.Scalar %1029, %float1.000000e00_1101 : !torch.vtensor<[32],f32>, !torch.float -> !torch.vtensor<[32],f32>
    %int1_1102 = torch.constant.int 1
    %1031 = torch.aten.unsqueeze %1022, %int1_1102 : !torch.vtensor<[128],si64>, !torch.int -> !torch.vtensor<[128,1],si64>
    %int0_1103 = torch.constant.int 0
    %1032 = torch.aten.unsqueeze %1030, %int0_1103 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[1,32],f32>
    %1033 = torch.aten.mul.Tensor %1031, %1032 : !torch.vtensor<[128,1],si64>, !torch.vtensor<[1,32],f32> -> !torch.vtensor<[128,32],f32>
    %int6_1104 = torch.constant.int 6
    %1034 = torch.prims.convert_element_type %1033, %int6_1104 : !torch.vtensor<[128,32],f32>, !torch.int -> !torch.vtensor<[128,32],f32>
    %1035 = torch_c.to_builtin_tensor %1034 : !torch.vtensor<[128,32],f32> -> tensor<128x32xf32>
    %1036 = flow.tensor.transfer %1035 : tensor<128x32xf32> to #hal.device.promise<@__device_0>
    %1037 = torch_c.from_builtin_tensor %1036 : tensor<128x32xf32> -> !torch.vtensor<[128,32],f32>
    %1038 = torch_c.to_builtin_tensor %1034 : !torch.vtensor<[128,32],f32> -> tensor<128x32xf32>
    %1039 = flow.tensor.transfer %1038 : tensor<128x32xf32> to #hal.device.promise<@__device_1>
    %1040 = torch_c.from_builtin_tensor %1039 : tensor<128x32xf32> -> !torch.vtensor<[128,32],f32>
    %int0_1105 = torch.constant.int 0
    %int0_1106 = torch.constant.int 0
    %int1_1107 = torch.constant.int 1
    %1041 = torch.aten.slice.Tensor %1037, %int0_1105, %int0_1106, %67, %int1_1107 : !torch.vtensor<[128,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %1041, [%63], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int1_1108 = torch.constant.int 1
    %int0_1109 = torch.constant.int 0
    %int9223372036854775807_1110 = torch.constant.int 9223372036854775807
    %int1_1111 = torch.constant.int 1
    %1042 = torch.aten.slice.Tensor %1041, %int1_1108, %int0_1109, %int9223372036854775807_1110, %int1_1111 : !torch.vtensor<[?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %1042, [%63], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int1_1112 = torch.constant.int 1
    %int0_1113 = torch.constant.int 0
    %int9223372036854775807_1114 = torch.constant.int 9223372036854775807
    %int1_1115 = torch.constant.int 1
    %1043 = torch.aten.slice.Tensor %1042, %int1_1112, %int0_1113, %int9223372036854775807_1114, %int1_1115 : !torch.vtensor<[?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %1043, [%63], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int0_1116 = torch.constant.int 0
    %1044 = torch.aten.unsqueeze %1043, %int0_1116 : !torch.vtensor<[?,32],f32>, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %1044, [%63], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int1_1117 = torch.constant.int 1
    %int0_1118 = torch.constant.int 0
    %int9223372036854775807_1119 = torch.constant.int 9223372036854775807
    %int1_1120 = torch.constant.int 1
    %1045 = torch.aten.slice.Tensor %1044, %int1_1117, %int0_1118, %int9223372036854775807_1119, %int1_1120 : !torch.vtensor<[1,?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %1045, [%63], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int2_1121 = torch.constant.int 2
    %int0_1122 = torch.constant.int 0
    %int9223372036854775807_1123 = torch.constant.int 9223372036854775807
    %int1_1124 = torch.constant.int 1
    %1046 = torch.aten.slice.Tensor %1045, %int2_1121, %int0_1122, %int9223372036854775807_1123, %int1_1124 : !torch.vtensor<[1,?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %1046, [%63], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int1_1125 = torch.constant.int 1
    %int1_1126 = torch.constant.int 1
    %int1_1127 = torch.constant.int 1
    %1047 = torch.prim.ListConstruct %int1_1125, %int1_1126, %int1_1127 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1048 = torch.aten.repeat %1046, %1047 : !torch.vtensor<[1,?,32],f32>, !torch.list<int> -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %1048, [%63], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int6_1128 = torch.constant.int 6
    %1049 = torch.prims.convert_element_type %1011, %int6_1128 : !torch.vtensor<[1,?,4,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,32],f32>
    torch.bind_symbolic_shape %1049, [%63], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f32>
    %1050 = torch_c.to_builtin_tensor %1049 : !torch.vtensor<[1,?,4,32],f32> -> tensor<1x?x4x32xf32>
    %1051 = torch_c.to_builtin_tensor %1048 : !torch.vtensor<[1,?,32],f32> -> tensor<1x?x32xf32>
    %1052 = util.call @sharktank_rotary_embedding_1_D_4_32_f32(%1050, %1051) : (tensor<1x?x4x32xf32>, tensor<1x?x32xf32>) -> tensor<1x?x4x32xf32>
    %1053 = torch_c.from_builtin_tensor %1052 : tensor<1x?x4x32xf32> -> !torch.vtensor<[1,?,4,32],f32>
    torch.bind_symbolic_shape %1053, [%63], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f32>
    %int5_1129 = torch.constant.int 5
    %1054 = torch.prims.convert_element_type %1053, %int5_1129 : !torch.vtensor<[1,?,4,32],f32>, !torch.int -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %1054, [%63], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int0_1130 = torch.constant.int 0
    %int0_1131 = torch.constant.int 0
    %int1_1132 = torch.constant.int 1
    %1055 = torch.aten.slice.Tensor %1040, %int0_1130, %int0_1131, %67, %int1_1132 : !torch.vtensor<[128,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %1055, [%63], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int1_1133 = torch.constant.int 1
    %int0_1134 = torch.constant.int 0
    %int9223372036854775807_1135 = torch.constant.int 9223372036854775807
    %int1_1136 = torch.constant.int 1
    %1056 = torch.aten.slice.Tensor %1055, %int1_1133, %int0_1134, %int9223372036854775807_1135, %int1_1136 : !torch.vtensor<[?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %1056, [%63], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int1_1137 = torch.constant.int 1
    %int0_1138 = torch.constant.int 0
    %int9223372036854775807_1139 = torch.constant.int 9223372036854775807
    %int1_1140 = torch.constant.int 1
    %1057 = torch.aten.slice.Tensor %1056, %int1_1137, %int0_1138, %int9223372036854775807_1139, %int1_1140 : !torch.vtensor<[?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %1057, [%63], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int0_1141 = torch.constant.int 0
    %1058 = torch.aten.unsqueeze %1057, %int0_1141 : !torch.vtensor<[?,32],f32>, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %1058, [%63], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int1_1142 = torch.constant.int 1
    %int0_1143 = torch.constant.int 0
    %int9223372036854775807_1144 = torch.constant.int 9223372036854775807
    %int1_1145 = torch.constant.int 1
    %1059 = torch.aten.slice.Tensor %1058, %int1_1142, %int0_1143, %int9223372036854775807_1144, %int1_1145 : !torch.vtensor<[1,?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %1059, [%63], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int2_1146 = torch.constant.int 2
    %int0_1147 = torch.constant.int 0
    %int9223372036854775807_1148 = torch.constant.int 9223372036854775807
    %int1_1149 = torch.constant.int 1
    %1060 = torch.aten.slice.Tensor %1059, %int2_1146, %int0_1147, %int9223372036854775807_1148, %int1_1149 : !torch.vtensor<[1,?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %1060, [%63], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int1_1150 = torch.constant.int 1
    %int1_1151 = torch.constant.int 1
    %int1_1152 = torch.constant.int 1
    %1061 = torch.prim.ListConstruct %int1_1150, %int1_1151, %int1_1152 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1062 = torch.aten.repeat %1060, %1061 : !torch.vtensor<[1,?,32],f32>, !torch.list<int> -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %1062, [%63], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int6_1153 = torch.constant.int 6
    %1063 = torch.prims.convert_element_type %1013, %int6_1153 : !torch.vtensor<[1,?,4,32],f16>, !torch.int -> !torch.vtensor<[1,?,4,32],f32>
    torch.bind_symbolic_shape %1063, [%63], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f32>
    %1064 = torch_c.to_builtin_tensor %1063 : !torch.vtensor<[1,?,4,32],f32> -> tensor<1x?x4x32xf32>
    %1065 = torch_c.to_builtin_tensor %1062 : !torch.vtensor<[1,?,32],f32> -> tensor<1x?x32xf32>
    %1066 = util.call @sharktank_rotary_embedding_1_D_4_32_f32(%1064, %1065) : (tensor<1x?x4x32xf32>, tensor<1x?x32xf32>) -> tensor<1x?x4x32xf32>
    %1067 = torch_c.from_builtin_tensor %1066 : tensor<1x?x4x32xf32> -> !torch.vtensor<[1,?,4,32],f32>
    torch.bind_symbolic_shape %1067, [%63], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f32>
    %int5_1154 = torch.constant.int 5
    %1068 = torch.prims.convert_element_type %1067, %int5_1154 : !torch.vtensor<[1,?,4,32],f32>, !torch.int -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %1068, [%63], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int128_1155 = torch.constant.int 128
    %none_1156 = torch.constant.none
    %none_1157 = torch.constant.none
    %cpu_1158 = torch.constant.device "cpu"
    %false_1159 = torch.constant.bool false
    %1069 = torch.aten.arange %int128_1155, %none_1156, %none_1157, %cpu_1158, %false_1159 : !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[128],si64>
    %int0_1160 = torch.constant.int 0
    %int32_1161 = torch.constant.int 32
    %none_1162 = torch.constant.none
    %none_1163 = torch.constant.none
    %cpu_1164 = torch.constant.device "cpu"
    %false_1165 = torch.constant.bool false
    %1070 = torch.aten.arange.start %int0_1160, %int32_1161, %none_1162, %none_1163, %cpu_1164, %false_1165 : !torch.int, !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[32],si64>
    %int2_1166 = torch.constant.int 2
    %1071 = torch.aten.floor_divide.Scalar %1070, %int2_1166 : !torch.vtensor<[32],si64>, !torch.int -> !torch.vtensor<[32],si64>
    %int6_1167 = torch.constant.int 6
    %1072 = torch.prims.convert_element_type %1071, %int6_1167 : !torch.vtensor<[32],si64>, !torch.int -> !torch.vtensor<[32],f32>
    %int32_1168 = torch.constant.int 32
    %1073 = torch.aten.div.Scalar %1072, %int32_1168 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[32],f32>
    %float2.000000e00_1169 = torch.constant.float 2.000000e+00
    %1074 = torch.aten.mul.Scalar %1073, %float2.000000e00_1169 : !torch.vtensor<[32],f32>, !torch.float -> !torch.vtensor<[32],f32>
    %float5.000000e05_1170 = torch.constant.float 5.000000e+05
    %1075 = torch.aten.pow.Scalar %float5.000000e05_1170, %1074 : !torch.float, !torch.vtensor<[32],f32> -> !torch.vtensor<[32],f32>
    %1076 = torch.aten.reciprocal %1075 : !torch.vtensor<[32],f32> -> !torch.vtensor<[32],f32>
    %float1.000000e00_1171 = torch.constant.float 1.000000e+00
    %1077 = torch.aten.mul.Scalar %1076, %float1.000000e00_1171 : !torch.vtensor<[32],f32>, !torch.float -> !torch.vtensor<[32],f32>
    %int1_1172 = torch.constant.int 1
    %1078 = torch.aten.unsqueeze %1069, %int1_1172 : !torch.vtensor<[128],si64>, !torch.int -> !torch.vtensor<[128,1],si64>
    %int0_1173 = torch.constant.int 0
    %1079 = torch.aten.unsqueeze %1077, %int0_1173 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[1,32],f32>
    %1080 = torch.aten.mul.Tensor %1078, %1079 : !torch.vtensor<[128,1],si64>, !torch.vtensor<[1,32],f32> -> !torch.vtensor<[128,32],f32>
    %int6_1174 = torch.constant.int 6
    %1081 = torch.prims.convert_element_type %1080, %int6_1174 : !torch.vtensor<[128,32],f32>, !torch.int -> !torch.vtensor<[128,32],f32>
    %1082 = torch_c.to_builtin_tensor %1081 : !torch.vtensor<[128,32],f32> -> tensor<128x32xf32>
    %1083 = flow.tensor.transfer %1082 : tensor<128x32xf32> to #hal.device.promise<@__device_0>
    %1084 = torch_c.from_builtin_tensor %1083 : tensor<128x32xf32> -> !torch.vtensor<[128,32],f32>
    %1085 = torch_c.to_builtin_tensor %1081 : !torch.vtensor<[128,32],f32> -> tensor<128x32xf32>
    %1086 = flow.tensor.transfer %1085 : tensor<128x32xf32> to #hal.device.promise<@__device_1>
    %1087 = torch_c.from_builtin_tensor %1086 : tensor<128x32xf32> -> !torch.vtensor<[128,32],f32>
    %int0_1175 = torch.constant.int 0
    %int0_1176 = torch.constant.int 0
    %int1_1177 = torch.constant.int 1
    %1088 = torch.aten.slice.Tensor %1084, %int0_1175, %int0_1176, %67, %int1_1177 : !torch.vtensor<[128,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %1088, [%63], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int1_1178 = torch.constant.int 1
    %int0_1179 = torch.constant.int 0
    %int9223372036854775807_1180 = torch.constant.int 9223372036854775807
    %int1_1181 = torch.constant.int 1
    %1089 = torch.aten.slice.Tensor %1088, %int1_1178, %int0_1179, %int9223372036854775807_1180, %int1_1181 : !torch.vtensor<[?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %1089, [%63], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int1_1182 = torch.constant.int 1
    %int0_1183 = torch.constant.int 0
    %int9223372036854775807_1184 = torch.constant.int 9223372036854775807
    %int1_1185 = torch.constant.int 1
    %1090 = torch.aten.slice.Tensor %1089, %int1_1182, %int0_1183, %int9223372036854775807_1184, %int1_1185 : !torch.vtensor<[?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %1090, [%63], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int0_1186 = torch.constant.int 0
    %1091 = torch.aten.unsqueeze %1090, %int0_1186 : !torch.vtensor<[?,32],f32>, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %1091, [%63], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int1_1187 = torch.constant.int 1
    %int0_1188 = torch.constant.int 0
    %int9223372036854775807_1189 = torch.constant.int 9223372036854775807
    %int1_1190 = torch.constant.int 1
    %1092 = torch.aten.slice.Tensor %1091, %int1_1187, %int0_1188, %int9223372036854775807_1189, %int1_1190 : !torch.vtensor<[1,?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %1092, [%63], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int2_1191 = torch.constant.int 2
    %int0_1192 = torch.constant.int 0
    %int9223372036854775807_1193 = torch.constant.int 9223372036854775807
    %int1_1194 = torch.constant.int 1
    %1093 = torch.aten.slice.Tensor %1092, %int2_1191, %int0_1192, %int9223372036854775807_1193, %int1_1194 : !torch.vtensor<[1,?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %1093, [%63], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int1_1195 = torch.constant.int 1
    %int1_1196 = torch.constant.int 1
    %int1_1197 = torch.constant.int 1
    %1094 = torch.prim.ListConstruct %int1_1195, %int1_1196, %int1_1197 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1095 = torch.aten.repeat %1093, %1094 : !torch.vtensor<[1,?,32],f32>, !torch.list<int> -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %1095, [%63], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int6_1198 = torch.constant.int 6
    %1096 = torch.prims.convert_element_type %1015, %int6_1198 : !torch.vtensor<[1,?,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,2,32],f32>
    torch.bind_symbolic_shape %1096, [%63], affine_map<()[s0] -> (1, s0 * 32, 2, 32)> : !torch.vtensor<[1,?,2,32],f32>
    %1097 = torch_c.to_builtin_tensor %1096 : !torch.vtensor<[1,?,2,32],f32> -> tensor<1x?x2x32xf32>
    %1098 = torch_c.to_builtin_tensor %1095 : !torch.vtensor<[1,?,32],f32> -> tensor<1x?x32xf32>
    %1099 = util.call @sharktank_rotary_embedding_1_D_2_32_f32(%1097, %1098) : (tensor<1x?x2x32xf32>, tensor<1x?x32xf32>) -> tensor<1x?x2x32xf32>
    %1100 = torch_c.from_builtin_tensor %1099 : tensor<1x?x2x32xf32> -> !torch.vtensor<[1,?,2,32],f32>
    torch.bind_symbolic_shape %1100, [%63], affine_map<()[s0] -> (1, s0 * 32, 2, 32)> : !torch.vtensor<[1,?,2,32],f32>
    %int5_1199 = torch.constant.int 5
    %1101 = torch.prims.convert_element_type %1100, %int5_1199 : !torch.vtensor<[1,?,2,32],f32>, !torch.int -> !torch.vtensor<[1,?,2,32],f16>
    torch.bind_symbolic_shape %1101, [%63], affine_map<()[s0] -> (1, s0 * 32, 2, 32)> : !torch.vtensor<[1,?,2,32],f16>
    %int0_1200 = torch.constant.int 0
    %int0_1201 = torch.constant.int 0
    %int1_1202 = torch.constant.int 1
    %1102 = torch.aten.slice.Tensor %1087, %int0_1200, %int0_1201, %67, %int1_1202 : !torch.vtensor<[128,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %1102, [%63], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int1_1203 = torch.constant.int 1
    %int0_1204 = torch.constant.int 0
    %int9223372036854775807_1205 = torch.constant.int 9223372036854775807
    %int1_1206 = torch.constant.int 1
    %1103 = torch.aten.slice.Tensor %1102, %int1_1203, %int0_1204, %int9223372036854775807_1205, %int1_1206 : !torch.vtensor<[?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %1103, [%63], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int1_1207 = torch.constant.int 1
    %int0_1208 = torch.constant.int 0
    %int9223372036854775807_1209 = torch.constant.int 9223372036854775807
    %int1_1210 = torch.constant.int 1
    %1104 = torch.aten.slice.Tensor %1103, %int1_1207, %int0_1208, %int9223372036854775807_1209, %int1_1210 : !torch.vtensor<[?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,32],f32>
    torch.bind_symbolic_shape %1104, [%63], affine_map<()[s0] -> (s0 * 32, 32)> : !torch.vtensor<[?,32],f32>
    %int0_1211 = torch.constant.int 0
    %1105 = torch.aten.unsqueeze %1104, %int0_1211 : !torch.vtensor<[?,32],f32>, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %1105, [%63], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int1_1212 = torch.constant.int 1
    %int0_1213 = torch.constant.int 0
    %int9223372036854775807_1214 = torch.constant.int 9223372036854775807
    %int1_1215 = torch.constant.int 1
    %1106 = torch.aten.slice.Tensor %1105, %int1_1212, %int0_1213, %int9223372036854775807_1214, %int1_1215 : !torch.vtensor<[1,?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %1106, [%63], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int2_1216 = torch.constant.int 2
    %int0_1217 = torch.constant.int 0
    %int9223372036854775807_1218 = torch.constant.int 9223372036854775807
    %int1_1219 = torch.constant.int 1
    %1107 = torch.aten.slice.Tensor %1106, %int2_1216, %int0_1217, %int9223372036854775807_1218, %int1_1219 : !torch.vtensor<[1,?,32],f32>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %1107, [%63], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int1_1220 = torch.constant.int 1
    %int1_1221 = torch.constant.int 1
    %int1_1222 = torch.constant.int 1
    %1108 = torch.prim.ListConstruct %int1_1220, %int1_1221, %int1_1222 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1109 = torch.aten.repeat %1107, %1108 : !torch.vtensor<[1,?,32],f32>, !torch.list<int> -> !torch.vtensor<[1,?,32],f32>
    torch.bind_symbolic_shape %1109, [%63], affine_map<()[s0] -> (1, s0 * 32, 32)> : !torch.vtensor<[1,?,32],f32>
    %int6_1223 = torch.constant.int 6
    %1110 = torch.prims.convert_element_type %1017, %int6_1223 : !torch.vtensor<[1,?,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,2,32],f32>
    torch.bind_symbolic_shape %1110, [%63], affine_map<()[s0] -> (1, s0 * 32, 2, 32)> : !torch.vtensor<[1,?,2,32],f32>
    %1111 = torch_c.to_builtin_tensor %1110 : !torch.vtensor<[1,?,2,32],f32> -> tensor<1x?x2x32xf32>
    %1112 = torch_c.to_builtin_tensor %1109 : !torch.vtensor<[1,?,32],f32> -> tensor<1x?x32xf32>
    %1113 = util.call @sharktank_rotary_embedding_1_D_2_32_f32(%1111, %1112) : (tensor<1x?x2x32xf32>, tensor<1x?x32xf32>) -> tensor<1x?x2x32xf32>
    %1114 = torch_c.from_builtin_tensor %1113 : tensor<1x?x2x32xf32> -> !torch.vtensor<[1,?,2,32],f32>
    torch.bind_symbolic_shape %1114, [%63], affine_map<()[s0] -> (1, s0 * 32, 2, 32)> : !torch.vtensor<[1,?,2,32],f32>
    %int5_1224 = torch.constant.int 5
    %1115 = torch.prims.convert_element_type %1114, %int5_1224 : !torch.vtensor<[1,?,2,32],f32>, !torch.int -> !torch.vtensor<[1,?,2,32],f16>
    torch.bind_symbolic_shape %1115, [%63], affine_map<()[s0] -> (1, s0 * 32, 2, 32)> : !torch.vtensor<[1,?,2,32],f16>
    %int6_1225 = torch.constant.int 6
    %1116 = torch.aten.mul.Scalar %108, %int6_1225 : !torch.vtensor<[1,?],si64>, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %1116, [%63], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %int6_1226 = torch.constant.int 6
    %1117 = torch.aten.mul.Scalar %111, %int6_1226 : !torch.vtensor<[1,?],si64>, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %1117, [%63], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %int4_1227 = torch.constant.int 4
    %int1_1228 = torch.constant.int 1
    %1118 = torch.aten.add.Scalar %1116, %int4_1227, %int1_1228 : !torch.vtensor<[1,?],si64>, !torch.int, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %1118, [%63], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %int4_1229 = torch.constant.int 4
    %int1_1230 = torch.constant.int 1
    %1119 = torch.aten.add.Scalar %1117, %int4_1229, %int1_1230 : !torch.vtensor<[1,?],si64>, !torch.int, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %1119, [%63], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %int1_1231 = torch.constant.int 1
    %int32_1232 = torch.constant.int 32
    %int2_1233 = torch.constant.int 2
    %int32_1234 = torch.constant.int 32
    %1120 = torch.prim.ListConstruct %int1_1231, %65, %int32_1232, %int2_1233, %int32_1234 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1121 = torch.aten.view %1101, %1120 : !torch.vtensor<[1,?,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,32,2,32],f16>
    torch.bind_symbolic_shape %1121, [%63], affine_map<()[s0] -> (1, s0, 32, 2, 32)> : !torch.vtensor<[1,?,32,2,32],f16>
    %int1_1235 = torch.constant.int 1
    %int32_1236 = torch.constant.int 32
    %int2_1237 = torch.constant.int 2
    %int32_1238 = torch.constant.int 32
    %1122 = torch.prim.ListConstruct %int1_1235, %65, %int32_1236, %int2_1237, %int32_1238 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1123 = torch.aten.view %1115, %1122 : !torch.vtensor<[1,?,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,32,2,32],f16>
    torch.bind_symbolic_shape %1123, [%63], affine_map<()[s0] -> (1, s0, 32, 2, 32)> : !torch.vtensor<[1,?,32,2,32],f16>
    %int32_1239 = torch.constant.int 32
    %int2_1240 = torch.constant.int 2
    %int32_1241 = torch.constant.int 32
    %1124 = torch.prim.ListConstruct %65, %int32_1239, %int2_1240, %int32_1241 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1125 = torch.aten.view %1121, %1124 : !torch.vtensor<[1,?,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,32,2,32],f16>
    torch.bind_symbolic_shape %1125, [%63], affine_map<()[s0] -> (s0, 32, 2, 32)> : !torch.vtensor<[?,32,2,32],f16>
    %int32_1242 = torch.constant.int 32
    %int2_1243 = torch.constant.int 2
    %int32_1244 = torch.constant.int 32
    %1126 = torch.prim.ListConstruct %65, %int32_1242, %int2_1243, %int32_1244 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1127 = torch.aten.view %1123, %1126 : !torch.vtensor<[1,?,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,32,2,32],f16>
    torch.bind_symbolic_shape %1127, [%63], affine_map<()[s0] -> (s0, 32, 2, 32)> : !torch.vtensor<[?,32,2,32],f16>
    %1128 = torch.prim.ListConstruct %65 : (!torch.int) -> !torch.list<int>
    %1129 = torch.aten.view %1118, %1128 : !torch.vtensor<[1,?],si64>, !torch.list<int> -> !torch.vtensor<[?],si64>
    torch.bind_symbolic_shape %1129, [%63], affine_map<()[s0] -> (s0)> : !torch.vtensor<[?],si64>
    %1130 = torch.prim.ListConstruct %65 : (!torch.int) -> !torch.list<int>
    %1131 = torch.aten.view %1119, %1130 : !torch.vtensor<[1,?],si64>, !torch.list<int> -> !torch.vtensor<[?],si64>
    torch.bind_symbolic_shape %1131, [%63], affine_map<()[s0] -> (s0)> : !torch.vtensor<[?],si64>
    %int5_1245 = torch.constant.int 5
    %1132 = torch.prims.convert_element_type %1125, %int5_1245 : !torch.vtensor<[?,32,2,32],f16>, !torch.int -> !torch.vtensor<[?,32,2,32],f16>
    torch.bind_symbolic_shape %1132, [%63], affine_map<()[s0] -> (s0, 32, 2, 32)> : !torch.vtensor<[?,32,2,32],f16>
    %int5_1246 = torch.constant.int 5
    %1133 = torch.prims.convert_element_type %1127, %int5_1246 : !torch.vtensor<[?,32,2,32],f16>, !torch.int -> !torch.vtensor<[?,32,2,32],f16>
    torch.bind_symbolic_shape %1133, [%63], affine_map<()[s0] -> (s0, 32, 2, 32)> : !torch.vtensor<[?,32,2,32],f16>
    %int3_1247 = torch.constant.int 3
    %int2_1248 = torch.constant.int 2
    %int32_1249 = torch.constant.int 32
    %int2_1250 = torch.constant.int 2
    %int32_1251 = torch.constant.int 32
    %1134 = torch.prim.ListConstruct %66, %int3_1247, %int2_1248, %int32_1249, %int2_1250, %int32_1251 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1135 = torch.aten.view %771, %1134 : !torch.vtensor<[?,12288],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,2,32],f16>
    torch.bind_symbolic_shape %1135, [%64], affine_map<()[s0] -> (s0, 3, 2, 32, 2, 32)> : !torch.vtensor<[?,3,2,32,2,32],f16>
    %int32_1252 = torch.constant.int 32
    %int2_1253 = torch.constant.int 2
    %int32_1254 = torch.constant.int 32
    %1136 = torch.prim.ListConstruct %295, %int32_1252, %int2_1253, %int32_1254 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1137 = torch.aten.view %1135, %1136 : !torch.vtensor<[?,3,2,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,32,2,32],f16>
    torch.bind_symbolic_shape %1137, [%64], affine_map<()[s0] -> (s0 * 6, 32, 2, 32)> : !torch.vtensor<[?,32,2,32],f16>
    %1138 = torch.prim.ListConstruct %1129 : (!torch.vtensor<[?],si64>) -> !torch.list<optional<vtensor>>
    %false_1255 = torch.constant.bool false
    %1139 = torch.aten.index_put %1137, %1138, %1132, %false_1255 : !torch.vtensor<[?,32,2,32],f16>, !torch.list<optional<vtensor>>, !torch.vtensor<[?,32,2,32],f16>, !torch.bool -> !torch.vtensor<[?,32,2,32],f16>
    torch.bind_symbolic_shape %1139, [%64], affine_map<()[s0] -> (s0 * 6, 32, 2, 32)> : !torch.vtensor<[?,32,2,32],f16>
    %int3_1256 = torch.constant.int 3
    %int2_1257 = torch.constant.int 2
    %int32_1258 = torch.constant.int 32
    %int2_1259 = torch.constant.int 2
    %int32_1260 = torch.constant.int 32
    %1140 = torch.prim.ListConstruct %66, %int3_1256, %int2_1257, %int32_1258, %int2_1259, %int32_1260 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1141 = torch.aten.view %1139, %1140 : !torch.vtensor<[?,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,2,32],f16>
    torch.bind_symbolic_shape %1141, [%64], affine_map<()[s0] -> (s0, 3, 2, 32, 2, 32)> : !torch.vtensor<[?,3,2,32,2,32],f16>
    %int12288_1261 = torch.constant.int 12288
    %1142 = torch.prim.ListConstruct %66, %int12288_1261 : (!torch.int, !torch.int) -> !torch.list<int>
    %1143 = torch.aten.view %1141, %1142 : !torch.vtensor<[?,3,2,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,12288],f16>
    torch.bind_symbolic_shape %1143, [%64], affine_map<()[s0] -> (s0, 12288)> : !torch.vtensor<[?,12288],f16>
    %int3_1262 = torch.constant.int 3
    %int2_1263 = torch.constant.int 2
    %int32_1264 = torch.constant.int 32
    %int2_1265 = torch.constant.int 2
    %int32_1266 = torch.constant.int 32
    %1144 = torch.prim.ListConstruct %66, %int3_1262, %int2_1263, %int32_1264, %int2_1265, %int32_1266 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1145 = torch.aten.view %1143, %1144 : !torch.vtensor<[?,12288],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,2,32],f16>
    torch.bind_symbolic_shape %1145, [%64], affine_map<()[s0] -> (s0, 3, 2, 32, 2, 32)> : !torch.vtensor<[?,3,2,32,2,32],f16>
    %int32_1267 = torch.constant.int 32
    %int2_1268 = torch.constant.int 2
    %int32_1269 = torch.constant.int 32
    %1146 = torch.prim.ListConstruct %295, %int32_1267, %int2_1268, %int32_1269 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1147 = torch.aten.view %1145, %1146 : !torch.vtensor<[?,3,2,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,32,2,32],f16>
    torch.bind_symbolic_shape %1147, [%64], affine_map<()[s0] -> (s0 * 6, 32, 2, 32)> : !torch.vtensor<[?,32,2,32],f16>
    %int3_1270 = torch.constant.int 3
    %int2_1271 = torch.constant.int 2
    %int32_1272 = torch.constant.int 32
    %int2_1273 = torch.constant.int 2
    %int32_1274 = torch.constant.int 32
    %1148 = torch.prim.ListConstruct %66, %int3_1270, %int2_1271, %int32_1272, %int2_1273, %int32_1274 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1149 = torch.aten.view %777, %1148 : !torch.vtensor<[?,12288],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,2,32],f16>
    torch.bind_symbolic_shape %1149, [%64], affine_map<()[s0] -> (s0, 3, 2, 32, 2, 32)> : !torch.vtensor<[?,3,2,32,2,32],f16>
    %int32_1275 = torch.constant.int 32
    %int2_1276 = torch.constant.int 2
    %int32_1277 = torch.constant.int 32
    %1150 = torch.prim.ListConstruct %295, %int32_1275, %int2_1276, %int32_1277 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1151 = torch.aten.view %1149, %1150 : !torch.vtensor<[?,3,2,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,32,2,32],f16>
    torch.bind_symbolic_shape %1151, [%64], affine_map<()[s0] -> (s0 * 6, 32, 2, 32)> : !torch.vtensor<[?,32,2,32],f16>
    %1152 = torch.prim.ListConstruct %1131 : (!torch.vtensor<[?],si64>) -> !torch.list<optional<vtensor>>
    %false_1278 = torch.constant.bool false
    %1153 = torch.aten.index_put %1151, %1152, %1133, %false_1278 : !torch.vtensor<[?,32,2,32],f16>, !torch.list<optional<vtensor>>, !torch.vtensor<[?,32,2,32],f16>, !torch.bool -> !torch.vtensor<[?,32,2,32],f16>
    torch.bind_symbolic_shape %1153, [%64], affine_map<()[s0] -> (s0 * 6, 32, 2, 32)> : !torch.vtensor<[?,32,2,32],f16>
    %int3_1279 = torch.constant.int 3
    %int2_1280 = torch.constant.int 2
    %int32_1281 = torch.constant.int 32
    %int2_1282 = torch.constant.int 2
    %int32_1283 = torch.constant.int 32
    %1154 = torch.prim.ListConstruct %66, %int3_1279, %int2_1280, %int32_1281, %int2_1282, %int32_1283 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1155 = torch.aten.view %1153, %1154 : !torch.vtensor<[?,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,2,32],f16>
    torch.bind_symbolic_shape %1155, [%64], affine_map<()[s0] -> (s0, 3, 2, 32, 2, 32)> : !torch.vtensor<[?,3,2,32,2,32],f16>
    %int12288_1284 = torch.constant.int 12288
    %1156 = torch.prim.ListConstruct %66, %int12288_1284 : (!torch.int, !torch.int) -> !torch.list<int>
    %1157 = torch.aten.view %1155, %1156 : !torch.vtensor<[?,3,2,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,12288],f16>
    torch.bind_symbolic_shape %1157, [%64], affine_map<()[s0] -> (s0, 12288)> : !torch.vtensor<[?,12288],f16>
    %int3_1285 = torch.constant.int 3
    %int2_1286 = torch.constant.int 2
    %int32_1287 = torch.constant.int 32
    %int2_1288 = torch.constant.int 2
    %int32_1289 = torch.constant.int 32
    %1158 = torch.prim.ListConstruct %66, %int3_1285, %int2_1286, %int32_1287, %int2_1288, %int32_1289 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1159 = torch.aten.view %1157, %1158 : !torch.vtensor<[?,12288],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,2,32],f16>
    torch.bind_symbolic_shape %1159, [%64], affine_map<()[s0] -> (s0, 3, 2, 32, 2, 32)> : !torch.vtensor<[?,3,2,32,2,32],f16>
    %int32_1290 = torch.constant.int 32
    %int2_1291 = torch.constant.int 2
    %int32_1292 = torch.constant.int 32
    %1160 = torch.prim.ListConstruct %295, %int32_1290, %int2_1291, %int32_1292 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1161 = torch.aten.view %1159, %1160 : !torch.vtensor<[?,3,2,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,32,2,32],f16>
    torch.bind_symbolic_shape %1161, [%64], affine_map<()[s0] -> (s0 * 6, 32, 2, 32)> : !torch.vtensor<[?,32,2,32],f16>
    %int1_1293 = torch.constant.int 1
    %int32_1294 = torch.constant.int 32
    %int2_1295 = torch.constant.int 2
    %int32_1296 = torch.constant.int 32
    %1162 = torch.prim.ListConstruct %int1_1293, %65, %int32_1294, %int2_1295, %int32_1296 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1163 = torch.aten.view %1019, %1162 : !torch.vtensor<[1,?,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,32,2,32],f16>
    torch.bind_symbolic_shape %1163, [%63], affine_map<()[s0] -> (1, s0, 32, 2, 32)> : !torch.vtensor<[1,?,32,2,32],f16>
    %int1_1297 = torch.constant.int 1
    %int32_1298 = torch.constant.int 32
    %int2_1299 = torch.constant.int 2
    %int32_1300 = torch.constant.int 32
    %1164 = torch.prim.ListConstruct %int1_1297, %65, %int32_1298, %int2_1299, %int32_1300 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1165 = torch.aten.view %1021, %1164 : !torch.vtensor<[1,?,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,32,2,32],f16>
    torch.bind_symbolic_shape %1165, [%63], affine_map<()[s0] -> (1, s0, 32, 2, 32)> : !torch.vtensor<[1,?,32,2,32],f16>
    %int32_1301 = torch.constant.int 32
    %int2_1302 = torch.constant.int 2
    %int32_1303 = torch.constant.int 32
    %1166 = torch.prim.ListConstruct %65, %int32_1301, %int2_1302, %int32_1303 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1167 = torch.aten.view %1163, %1166 : !torch.vtensor<[1,?,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,32,2,32],f16>
    torch.bind_symbolic_shape %1167, [%63], affine_map<()[s0] -> (s0, 32, 2, 32)> : !torch.vtensor<[?,32,2,32],f16>
    %int32_1304 = torch.constant.int 32
    %int2_1305 = torch.constant.int 2
    %int32_1306 = torch.constant.int 32
    %1168 = torch.prim.ListConstruct %65, %int32_1304, %int2_1305, %int32_1306 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1169 = torch.aten.view %1165, %1168 : !torch.vtensor<[1,?,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,32,2,32],f16>
    torch.bind_symbolic_shape %1169, [%63], affine_map<()[s0] -> (s0, 32, 2, 32)> : !torch.vtensor<[?,32,2,32],f16>
    %int1_1307 = torch.constant.int 1
    %int1_1308 = torch.constant.int 1
    %1170 = torch.aten.add.Scalar %1118, %int1_1307, %int1_1308 : !torch.vtensor<[1,?],si64>, !torch.int, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %1170, [%63], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %int1_1309 = torch.constant.int 1
    %int1_1310 = torch.constant.int 1
    %1171 = torch.aten.add.Scalar %1119, %int1_1309, %int1_1310 : !torch.vtensor<[1,?],si64>, !torch.int, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %1171, [%63], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %1172 = torch.prim.ListConstruct %65 : (!torch.int) -> !torch.list<int>
    %1173 = torch.aten.view %1170, %1172 : !torch.vtensor<[1,?],si64>, !torch.list<int> -> !torch.vtensor<[?],si64>
    torch.bind_symbolic_shape %1173, [%63], affine_map<()[s0] -> (s0)> : !torch.vtensor<[?],si64>
    %1174 = torch.prim.ListConstruct %65 : (!torch.int) -> !torch.list<int>
    %1175 = torch.aten.view %1171, %1174 : !torch.vtensor<[1,?],si64>, !torch.list<int> -> !torch.vtensor<[?],si64>
    torch.bind_symbolic_shape %1175, [%63], affine_map<()[s0] -> (s0)> : !torch.vtensor<[?],si64>
    %int5_1311 = torch.constant.int 5
    %1176 = torch.prims.convert_element_type %1167, %int5_1311 : !torch.vtensor<[?,32,2,32],f16>, !torch.int -> !torch.vtensor<[?,32,2,32],f16>
    torch.bind_symbolic_shape %1176, [%63], affine_map<()[s0] -> (s0, 32, 2, 32)> : !torch.vtensor<[?,32,2,32],f16>
    %int5_1312 = torch.constant.int 5
    %1177 = torch.prims.convert_element_type %1169, %int5_1312 : !torch.vtensor<[?,32,2,32],f16>, !torch.int -> !torch.vtensor<[?,32,2,32],f16>
    torch.bind_symbolic_shape %1177, [%63], affine_map<()[s0] -> (s0, 32, 2, 32)> : !torch.vtensor<[?,32,2,32],f16>
    %1178 = torch.prim.ListConstruct %1173 : (!torch.vtensor<[?],si64>) -> !torch.list<optional<vtensor>>
    %false_1313 = torch.constant.bool false
    %1179 = torch.aten.index_put %1147, %1178, %1176, %false_1313 : !torch.vtensor<[?,32,2,32],f16>, !torch.list<optional<vtensor>>, !torch.vtensor<[?,32,2,32],f16>, !torch.bool -> !torch.vtensor<[?,32,2,32],f16>
    torch.bind_symbolic_shape %1179, [%64], affine_map<()[s0] -> (s0 * 6, 32, 2, 32)> : !torch.vtensor<[?,32,2,32],f16>
    %int3_1314 = torch.constant.int 3
    %int2_1315 = torch.constant.int 2
    %int32_1316 = torch.constant.int 32
    %int2_1317 = torch.constant.int 2
    %int32_1318 = torch.constant.int 32
    %1180 = torch.prim.ListConstruct %66, %int3_1314, %int2_1315, %int32_1316, %int2_1317, %int32_1318 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1181 = torch.aten.view %1179, %1180 : !torch.vtensor<[?,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,2,32],f16>
    torch.bind_symbolic_shape %1181, [%64], affine_map<()[s0] -> (s0, 3, 2, 32, 2, 32)> : !torch.vtensor<[?,3,2,32,2,32],f16>
    %int12288_1319 = torch.constant.int 12288
    %1182 = torch.prim.ListConstruct %66, %int12288_1319 : (!torch.int, !torch.int) -> !torch.list<int>
    %1183 = torch.aten.view %1181, %1182 : !torch.vtensor<[?,3,2,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,12288],f16>
    torch.overwrite.tensor.contents %1183 overwrites %arg3 : !torch.vtensor<[?,12288],f16>, !torch.tensor<[?,12288],f16>
    torch.bind_symbolic_shape %1183, [%64], affine_map<()[s0] -> (s0, 12288)> : !torch.vtensor<[?,12288],f16>
    %1184 = torch.prim.ListConstruct %1175 : (!torch.vtensor<[?],si64>) -> !torch.list<optional<vtensor>>
    %false_1320 = torch.constant.bool false
    %1185 = torch.aten.index_put %1161, %1184, %1177, %false_1320 : !torch.vtensor<[?,32,2,32],f16>, !torch.list<optional<vtensor>>, !torch.vtensor<[?,32,2,32],f16>, !torch.bool -> !torch.vtensor<[?,32,2,32],f16>
    torch.bind_symbolic_shape %1185, [%64], affine_map<()[s0] -> (s0 * 6, 32, 2, 32)> : !torch.vtensor<[?,32,2,32],f16>
    %int3_1321 = torch.constant.int 3
    %int2_1322 = torch.constant.int 2
    %int32_1323 = torch.constant.int 32
    %int2_1324 = torch.constant.int 2
    %int32_1325 = torch.constant.int 32
    %1186 = torch.prim.ListConstruct %66, %int3_1321, %int2_1322, %int32_1323, %int2_1324, %int32_1325 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1187 = torch.aten.view %1185, %1186 : !torch.vtensor<[?,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,2,32],f16>
    torch.bind_symbolic_shape %1187, [%64], affine_map<()[s0] -> (s0, 3, 2, 32, 2, 32)> : !torch.vtensor<[?,3,2,32,2,32],f16>
    %int12288_1326 = torch.constant.int 12288
    %1188 = torch.prim.ListConstruct %66, %int12288_1326 : (!torch.int, !torch.int) -> !torch.list<int>
    %1189 = torch.aten.view %1187, %1188 : !torch.vtensor<[?,3,2,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,12288],f16>
    torch.overwrite.tensor.contents %1189 overwrites %arg4 : !torch.vtensor<[?,12288],f16>, !torch.tensor<[?,12288],f16>
    torch.bind_symbolic_shape %1189, [%64], affine_map<()[s0] -> (s0, 12288)> : !torch.vtensor<[?,12288],f16>
    %int-2_1327 = torch.constant.int -2
    %1190 = torch.aten.unsqueeze %1101, %int-2_1327 : !torch.vtensor<[1,?,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,2,1,32],f16>
    torch.bind_symbolic_shape %1190, [%63], affine_map<()[s0] -> (1, s0 * 32, 2, 1, 32)> : !torch.vtensor<[1,?,2,1,32],f16>
    %int-2_1328 = torch.constant.int -2
    %1191 = torch.aten.unsqueeze %1115, %int-2_1328 : !torch.vtensor<[1,?,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,2,1,32],f16>
    torch.bind_symbolic_shape %1191, [%63], affine_map<()[s0] -> (1, s0 * 32, 2, 1, 32)> : !torch.vtensor<[1,?,2,1,32],f16>
    %int1_1329 = torch.constant.int 1
    %int2_1330 = torch.constant.int 2
    %int2_1331 = torch.constant.int 2
    %int32_1332 = torch.constant.int 32
    %1192 = torch.prim.ListConstruct %int1_1329, %67, %int2_1330, %int2_1331, %int32_1332 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_1333 = torch.constant.bool false
    %1193 = torch.aten.expand %1190, %1192, %false_1333 : !torch.vtensor<[1,?,2,1,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,?,2,2,32],f16>
    torch.bind_symbolic_shape %1193, [%63], affine_map<()[s0] -> (1, s0 * 32, 2, 2, 32)> : !torch.vtensor<[1,?,2,2,32],f16>
    %int1_1334 = torch.constant.int 1
    %int2_1335 = torch.constant.int 2
    %int2_1336 = torch.constant.int 2
    %int32_1337 = torch.constant.int 32
    %1194 = torch.prim.ListConstruct %int1_1334, %67, %int2_1335, %int2_1336, %int32_1337 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_1338 = torch.constant.bool false
    %1195 = torch.aten.expand %1191, %1194, %false_1338 : !torch.vtensor<[1,?,2,1,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,?,2,2,32],f16>
    torch.bind_symbolic_shape %1195, [%63], affine_map<()[s0] -> (1, s0 * 32, 2, 2, 32)> : !torch.vtensor<[1,?,2,2,32],f16>
    %int0_1339 = torch.constant.int 0
    %1196 = torch.aten.clone %1193, %int0_1339 : !torch.vtensor<[1,?,2,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,2,2,32],f16>
    torch.bind_symbolic_shape %1196, [%63], affine_map<()[s0] -> (1, s0 * 32, 2, 2, 32)> : !torch.vtensor<[1,?,2,2,32],f16>
    %int1_1340 = torch.constant.int 1
    %int4_1341 = torch.constant.int 4
    %int32_1342 = torch.constant.int 32
    %1197 = torch.prim.ListConstruct %int1_1340, %67, %int4_1341, %int32_1342 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1198 = torch.aten._unsafe_view %1196, %1197 : !torch.vtensor<[1,?,2,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %1198, [%63], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int0_1343 = torch.constant.int 0
    %1199 = torch.aten.clone %1195, %int0_1343 : !torch.vtensor<[1,?,2,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,2,2,32],f16>
    torch.bind_symbolic_shape %1199, [%63], affine_map<()[s0] -> (1, s0 * 32, 2, 2, 32)> : !torch.vtensor<[1,?,2,2,32],f16>
    %int1_1344 = torch.constant.int 1
    %int4_1345 = torch.constant.int 4
    %int32_1346 = torch.constant.int 32
    %1200 = torch.prim.ListConstruct %int1_1344, %67, %int4_1345, %int32_1346 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1201 = torch.aten._unsafe_view %1199, %1200 : !torch.vtensor<[1,?,2,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %1201, [%63], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int-2_1347 = torch.constant.int -2
    %1202 = torch.aten.unsqueeze %1019, %int-2_1347 : !torch.vtensor<[1,?,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,2,1,32],f16>
    torch.bind_symbolic_shape %1202, [%63], affine_map<()[s0] -> (1, s0 * 32, 2, 1, 32)> : !torch.vtensor<[1,?,2,1,32],f16>
    %int-2_1348 = torch.constant.int -2
    %1203 = torch.aten.unsqueeze %1021, %int-2_1348 : !torch.vtensor<[1,?,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,2,1,32],f16>
    torch.bind_symbolic_shape %1203, [%63], affine_map<()[s0] -> (1, s0 * 32, 2, 1, 32)> : !torch.vtensor<[1,?,2,1,32],f16>
    %int1_1349 = torch.constant.int 1
    %int2_1350 = torch.constant.int 2
    %int2_1351 = torch.constant.int 2
    %int32_1352 = torch.constant.int 32
    %1204 = torch.prim.ListConstruct %int1_1349, %67, %int2_1350, %int2_1351, %int32_1352 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_1353 = torch.constant.bool false
    %1205 = torch.aten.expand %1202, %1204, %false_1353 : !torch.vtensor<[1,?,2,1,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,?,2,2,32],f16>
    torch.bind_symbolic_shape %1205, [%63], affine_map<()[s0] -> (1, s0 * 32, 2, 2, 32)> : !torch.vtensor<[1,?,2,2,32],f16>
    %int1_1354 = torch.constant.int 1
    %int2_1355 = torch.constant.int 2
    %int2_1356 = torch.constant.int 2
    %int32_1357 = torch.constant.int 32
    %1206 = torch.prim.ListConstruct %int1_1354, %67, %int2_1355, %int2_1356, %int32_1357 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_1358 = torch.constant.bool false
    %1207 = torch.aten.expand %1203, %1206, %false_1358 : !torch.vtensor<[1,?,2,1,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,?,2,2,32],f16>
    torch.bind_symbolic_shape %1207, [%63], affine_map<()[s0] -> (1, s0 * 32, 2, 2, 32)> : !torch.vtensor<[1,?,2,2,32],f16>
    %int0_1359 = torch.constant.int 0
    %1208 = torch.aten.clone %1205, %int0_1359 : !torch.vtensor<[1,?,2,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,2,2,32],f16>
    torch.bind_symbolic_shape %1208, [%63], affine_map<()[s0] -> (1, s0 * 32, 2, 2, 32)> : !torch.vtensor<[1,?,2,2,32],f16>
    %int1_1360 = torch.constant.int 1
    %int4_1361 = torch.constant.int 4
    %int32_1362 = torch.constant.int 32
    %1209 = torch.prim.ListConstruct %int1_1360, %67, %int4_1361, %int32_1362 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1210 = torch.aten._unsafe_view %1208, %1209 : !torch.vtensor<[1,?,2,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %1210, [%63], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int0_1363 = torch.constant.int 0
    %1211 = torch.aten.clone %1207, %int0_1363 : !torch.vtensor<[1,?,2,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,2,2,32],f16>
    torch.bind_symbolic_shape %1211, [%63], affine_map<()[s0] -> (1, s0 * 32, 2, 2, 32)> : !torch.vtensor<[1,?,2,2,32],f16>
    %int1_1364 = torch.constant.int 1
    %int4_1365 = torch.constant.int 4
    %int32_1366 = torch.constant.int 32
    %1212 = torch.prim.ListConstruct %int1_1364, %67, %int4_1365, %int32_1366 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1213 = torch.aten._unsafe_view %1211, %1212 : !torch.vtensor<[1,?,2,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %1213, [%63], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int1_1367 = torch.constant.int 1
    %int2_1368 = torch.constant.int 2
    %1214 = torch.aten.transpose.int %1054, %int1_1367, %int2_1368 : !torch.vtensor<[1,?,4,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,4,?,32],f16>
    torch.bind_symbolic_shape %1214, [%63], affine_map<()[s0] -> (1, 4, s0 * 32, 32)> : !torch.vtensor<[1,4,?,32],f16>
    %int1_1369 = torch.constant.int 1
    %int2_1370 = torch.constant.int 2
    %1215 = torch.aten.transpose.int %1068, %int1_1369, %int2_1370 : !torch.vtensor<[1,?,4,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,4,?,32],f16>
    torch.bind_symbolic_shape %1215, [%63], affine_map<()[s0] -> (1, 4, s0 * 32, 32)> : !torch.vtensor<[1,4,?,32],f16>
    %int1_1371 = torch.constant.int 1
    %int2_1372 = torch.constant.int 2
    %1216 = torch.aten.transpose.int %1198, %int1_1371, %int2_1372 : !torch.vtensor<[1,?,4,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,4,?,32],f16>
    torch.bind_symbolic_shape %1216, [%63], affine_map<()[s0] -> (1, 4, s0 * 32, 32)> : !torch.vtensor<[1,4,?,32],f16>
    %int1_1373 = torch.constant.int 1
    %int2_1374 = torch.constant.int 2
    %1217 = torch.aten.transpose.int %1201, %int1_1373, %int2_1374 : !torch.vtensor<[1,?,4,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,4,?,32],f16>
    torch.bind_symbolic_shape %1217, [%63], affine_map<()[s0] -> (1, 4, s0 * 32, 32)> : !torch.vtensor<[1,4,?,32],f16>
    %int1_1375 = torch.constant.int 1
    %int2_1376 = torch.constant.int 2
    %1218 = torch.aten.transpose.int %1210, %int1_1375, %int2_1376 : !torch.vtensor<[1,?,4,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,4,?,32],f16>
    torch.bind_symbolic_shape %1218, [%63], affine_map<()[s0] -> (1, 4, s0 * 32, 32)> : !torch.vtensor<[1,4,?,32],f16>
    %int1_1377 = torch.constant.int 1
    %int2_1378 = torch.constant.int 2
    %1219 = torch.aten.transpose.int %1213, %int1_1377, %int2_1378 : !torch.vtensor<[1,?,4,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,4,?,32],f16>
    torch.bind_symbolic_shape %1219, [%63], affine_map<()[s0] -> (1, 4, s0 * 32, 32)> : !torch.vtensor<[1,4,?,32],f16>
    %int5_1379 = torch.constant.int 5
    %1220 = torch.prims.convert_element_type %1214, %int5_1379 : !torch.vtensor<[1,4,?,32],f16>, !torch.int -> !torch.vtensor<[1,4,?,32],f16>
    torch.bind_symbolic_shape %1220, [%63], affine_map<()[s0] -> (1, 4, s0 * 32, 32)> : !torch.vtensor<[1,4,?,32],f16>
    %int5_1380 = torch.constant.int 5
    %1221 = torch.prims.convert_element_type %1215, %int5_1380 : !torch.vtensor<[1,4,?,32],f16>, !torch.int -> !torch.vtensor<[1,4,?,32],f16>
    torch.bind_symbolic_shape %1221, [%63], affine_map<()[s0] -> (1, 4, s0 * 32, 32)> : !torch.vtensor<[1,4,?,32],f16>
    %int5_1381 = torch.constant.int 5
    %1222 = torch.prims.convert_element_type %1216, %int5_1381 : !torch.vtensor<[1,4,?,32],f16>, !torch.int -> !torch.vtensor<[1,4,?,32],f16>
    torch.bind_symbolic_shape %1222, [%63], affine_map<()[s0] -> (1, 4, s0 * 32, 32)> : !torch.vtensor<[1,4,?,32],f16>
    %int5_1382 = torch.constant.int 5
    %1223 = torch.prims.convert_element_type %1217, %int5_1382 : !torch.vtensor<[1,4,?,32],f16>, !torch.int -> !torch.vtensor<[1,4,?,32],f16>
    torch.bind_symbolic_shape %1223, [%63], affine_map<()[s0] -> (1, 4, s0 * 32, 32)> : !torch.vtensor<[1,4,?,32],f16>
    %int5_1383 = torch.constant.int 5
    %1224 = torch.prims.convert_element_type %1218, %int5_1383 : !torch.vtensor<[1,4,?,32],f16>, !torch.int -> !torch.vtensor<[1,4,?,32],f16>
    torch.bind_symbolic_shape %1224, [%63], affine_map<()[s0] -> (1, 4, s0 * 32, 32)> : !torch.vtensor<[1,4,?,32],f16>
    %int5_1384 = torch.constant.int 5
    %1225 = torch.prims.convert_element_type %1219, %int5_1384 : !torch.vtensor<[1,4,?,32],f16>, !torch.int -> !torch.vtensor<[1,4,?,32],f16>
    torch.bind_symbolic_shape %1225, [%63], affine_map<()[s0] -> (1, 4, s0 * 32, 32)> : !torch.vtensor<[1,4,?,32],f16>
    %int5_1385 = torch.constant.int 5
    %1226 = torch.prims.convert_element_type %102, %int5_1385 : !torch.vtensor<[1,1,?,?],f16>, !torch.int -> !torch.vtensor<[1,1,?,?],f16>
    torch.bind_symbolic_shape %1226, [%63], affine_map<()[s0] -> (1, 1, s0 * 32, s0 * 32)> : !torch.vtensor<[1,1,?,?],f16>
    %int5_1386 = torch.constant.int 5
    %1227 = torch.prims.convert_element_type %105, %int5_1386 : !torch.vtensor<[1,1,?,?],f16>, !torch.int -> !torch.vtensor<[1,1,?,?],f16>
    torch.bind_symbolic_shape %1227, [%63], affine_map<()[s0] -> (1, 1, s0 * 32, s0 * 32)> : !torch.vtensor<[1,1,?,?],f16>
    %float0.000000e00_1387 = torch.constant.float 0.000000e+00
    %false_1388 = torch.constant.bool false
    %none_1389 = torch.constant.none
    %1228:2 = torch.operator "torch.aten._scaled_dot_product_flash_attention_for_cpu"(%1220, %1222, %1224, %float0.000000e00_1387, %false_1388, %1226, %none_1389) : (!torch.vtensor<[1,4,?,32],f16>, !torch.vtensor<[1,4,?,32],f16>, !torch.vtensor<[1,4,?,32],f16>, !torch.float, !torch.bool, !torch.vtensor<[1,1,?,?],f16>, !torch.none) -> (!torch.vtensor<[1,4,?,32],f16>, !torch.vtensor<[1,4,?],f32>) 
    torch.bind_symbolic_shape %1228#0, [%63], affine_map<()[s0] -> (1, 4, s0 * 32, 32)> : !torch.vtensor<[1,4,?,32],f16>
    %float0.000000e00_1390 = torch.constant.float 0.000000e+00
    %false_1391 = torch.constant.bool false
    %none_1392 = torch.constant.none
    %1229:2 = torch.operator "torch.aten._scaled_dot_product_flash_attention_for_cpu"(%1221, %1223, %1225, %float0.000000e00_1390, %false_1391, %1227, %none_1392) : (!torch.vtensor<[1,4,?,32],f16>, !torch.vtensor<[1,4,?,32],f16>, !torch.vtensor<[1,4,?,32],f16>, !torch.float, !torch.bool, !torch.vtensor<[1,1,?,?],f16>, !torch.none) -> (!torch.vtensor<[1,4,?,32],f16>, !torch.vtensor<[1,4,?],f32>) 
    torch.bind_symbolic_shape %1229#0, [%63], affine_map<()[s0] -> (1, 4, s0 * 32, 32)> : !torch.vtensor<[1,4,?,32],f16>
    %int1_1393 = torch.constant.int 1
    %int2_1394 = torch.constant.int 2
    %1230 = torch.aten.transpose.int %1228#0, %int1_1393, %int2_1394 : !torch.vtensor<[1,4,?,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %1230, [%63], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int1_1395 = torch.constant.int 1
    %int2_1396 = torch.constant.int 2
    %1231 = torch.aten.transpose.int %1229#0, %int1_1395, %int2_1396 : !torch.vtensor<[1,4,?,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %1231, [%63], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int1_1397 = torch.constant.int 1
    %int128_1398 = torch.constant.int 128
    %1232 = torch.prim.ListConstruct %int1_1397, %67, %int128_1398 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1233 = torch.aten.view %1230, %1232 : !torch.vtensor<[1,?,4,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,128],f16>
    torch.bind_symbolic_shape %1233, [%63], affine_map<()[s0] -> (1, s0 * 32, 128)> : !torch.vtensor<[1,?,128],f16>
    %int1_1399 = torch.constant.int 1
    %int128_1400 = torch.constant.int 128
    %1234 = torch.prim.ListConstruct %int1_1399, %67, %int128_1400 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1235 = torch.aten.view %1231, %1234 : !torch.vtensor<[1,?,4,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,128],f16>
    torch.bind_symbolic_shape %1235, [%63], affine_map<()[s0] -> (1, s0 * 32, 128)> : !torch.vtensor<[1,?,128],f16>
    %int1_1401 = torch.constant.int 1
    %int0_1402 = torch.constant.int 0
    %1236 = torch.prim.ListConstruct %int1_1401, %int0_1402 : (!torch.int, !torch.int) -> !torch.list<int>
    %1237 = torch.aten.permute %46, %1236 : !torch.vtensor<[256,128],f32>, !torch.list<int> -> !torch.vtensor<[128,256],f32>
    %int1_1403 = torch.constant.int 1
    %int0_1404 = torch.constant.int 0
    %1238 = torch.prim.ListConstruct %int1_1403, %int0_1404 : (!torch.int, !torch.int) -> !torch.list<int>
    %1239 = torch.aten.permute %47, %1238 : !torch.vtensor<[256,128],f32>, !torch.list<int> -> !torch.vtensor<[128,256],f32>
    %int5_1405 = torch.constant.int 5
    %1240 = torch.prims.convert_element_type %1237, %int5_1405 : !torch.vtensor<[128,256],f32>, !torch.int -> !torch.vtensor<[128,256],f16>
    %int128_1406 = torch.constant.int 128
    %1241 = torch.prim.ListConstruct %67, %int128_1406 : (!torch.int, !torch.int) -> !torch.list<int>
    %1242 = torch.aten.view %1233, %1241 : !torch.vtensor<[1,?,128],f16>, !torch.list<int> -> !torch.vtensor<[?,128],f16>
    torch.bind_symbolic_shape %1242, [%63], affine_map<()[s0] -> (s0 * 32, 128)> : !torch.vtensor<[?,128],f16>
    %1243 = torch.aten.mm %1242, %1240 : !torch.vtensor<[?,128],f16>, !torch.vtensor<[128,256],f16> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %1243, [%63], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %int1_1407 = torch.constant.int 1
    %int256_1408 = torch.constant.int 256
    %1244 = torch.prim.ListConstruct %int1_1407, %67, %int256_1408 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1245 = torch.aten.view %1243, %1244 : !torch.vtensor<[?,256],f16>, !torch.list<int> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %1245, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int5_1409 = torch.constant.int 5
    %1246 = torch.prims.convert_element_type %1239, %int5_1409 : !torch.vtensor<[128,256],f32>, !torch.int -> !torch.vtensor<[128,256],f16>
    %int128_1410 = torch.constant.int 128
    %1247 = torch.prim.ListConstruct %67, %int128_1410 : (!torch.int, !torch.int) -> !torch.list<int>
    %1248 = torch.aten.view %1235, %1247 : !torch.vtensor<[1,?,128],f16>, !torch.list<int> -> !torch.vtensor<[?,128],f16>
    torch.bind_symbolic_shape %1248, [%63], affine_map<()[s0] -> (s0 * 32, 128)> : !torch.vtensor<[?,128],f16>
    %1249 = torch.aten.mm %1248, %1246 : !torch.vtensor<[?,128],f16>, !torch.vtensor<[128,256],f16> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %1249, [%63], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %int1_1411 = torch.constant.int 1
    %int256_1412 = torch.constant.int 256
    %1250 = torch.prim.ListConstruct %int1_1411, %67, %int256_1412 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1251 = torch.aten.view %1249, %1250 : !torch.vtensor<[?,256],f16>, !torch.list<int> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %1251, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %1252 = torch_c.to_builtin_tensor %1245 : !torch.vtensor<[1,?,256],f16> -> tensor<1x?x256xf16>
    %c1_1413 = arith.constant 1 : index
    %dim_1414 = tensor.dim %1252, %c1_1413 : tensor<1x?x256xf16>
    %1253 = flow.tensor.barrier %1252 : tensor<1x?x256xf16>{%dim_1414} on #hal.device.promise<@__device_0>
    %1254 = torch_c.from_builtin_tensor %1253 : tensor<1x?x256xf16> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %1254, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %1255 = torch_c.to_builtin_tensor %1251 : !torch.vtensor<[1,?,256],f16> -> tensor<1x?x256xf16>
    %c1_1415 = arith.constant 1 : index
    %dim_1416 = tensor.dim %1255, %c1_1415 : tensor<1x?x256xf16>
    %1256 = flow.tensor.transfer %1255 : tensor<1x?x256xf16>{%dim_1416} to #hal.device.promise<@__device_0>
    %1257 = torch_c.from_builtin_tensor %1256 : tensor<1x?x256xf16> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %1257, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int1_1417 = torch.constant.int 1
    %1258 = torch.aten.add.Tensor %1254, %1257, %int1_1417 : !torch.vtensor<[1,?,256],f16>, !torch.vtensor<[1,?,256],f16>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %1258, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %1259 = torch_c.to_builtin_tensor %1258 : !torch.vtensor<[1,?,256],f16> -> tensor<1x?x256xf16>
    %c1_1418 = arith.constant 1 : index
    %dim_1419 = tensor.dim %1259, %c1_1418 : tensor<1x?x256xf16>
    %1260 = flow.tensor.barrier %1259 : tensor<1x?x256xf16>{%dim_1419} on #hal.device.promise<@__device_0>
    %1261 = torch_c.from_builtin_tensor %1260 : tensor<1x?x256xf16> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %1261, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %1262 = torch_c.to_builtin_tensor %1258 : !torch.vtensor<[1,?,256],f16> -> tensor<1x?x256xf16>
    %c1_1420 = arith.constant 1 : index
    %dim_1421 = tensor.dim %1262, %c1_1420 : tensor<1x?x256xf16>
    %1263 = flow.tensor.transfer %1262 : tensor<1x?x256xf16>{%dim_1421} to #hal.device.promise<@__device_1>
    %1264 = torch_c.from_builtin_tensor %1263 : tensor<1x?x256xf16> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %1264, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int1_1422 = torch.constant.int 1
    %1265 = torch.aten.add.Tensor %940, %1261, %int1_1422 : !torch.vtensor<[1,?,256],f16>, !torch.vtensor<[1,?,256],f16>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %1265, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int1_1423 = torch.constant.int 1
    %1266 = torch.aten.add.Tensor %941, %1264, %int1_1423 : !torch.vtensor<[1,?,256],f16>, !torch.vtensor<[1,?,256],f16>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %1266, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int6_1424 = torch.constant.int 6
    %1267 = torch.prims.convert_element_type %1265, %int6_1424 : !torch.vtensor<[1,?,256],f16>, !torch.int -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %1267, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int6_1425 = torch.constant.int 6
    %1268 = torch.prims.convert_element_type %1266, %int6_1425 : !torch.vtensor<[1,?,256],f16>, !torch.int -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %1268, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int2_1426 = torch.constant.int 2
    %1269 = torch.aten.pow.Tensor_Scalar %1267, %int2_1426 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %1269, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int2_1427 = torch.constant.int 2
    %1270 = torch.aten.pow.Tensor_Scalar %1268, %int2_1427 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %1270, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int-1_1428 = torch.constant.int -1
    %1271 = torch.prim.ListConstruct %int-1_1428 : (!torch.int) -> !torch.list<int>
    %true_1429 = torch.constant.bool true
    %none_1430 = torch.constant.none
    %1272 = torch.aten.mean.dim %1269, %1271, %true_1429, %none_1430 : !torch.vtensor<[1,?,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %1272, [%63], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %int-1_1431 = torch.constant.int -1
    %1273 = torch.prim.ListConstruct %int-1_1431 : (!torch.int) -> !torch.list<int>
    %true_1432 = torch.constant.bool true
    %none_1433 = torch.constant.none
    %1274 = torch.aten.mean.dim %1270, %1273, %true_1432, %none_1433 : !torch.vtensor<[1,?,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %1274, [%63], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %float1.000000e-02_1434 = torch.constant.float 1.000000e-02
    %int1_1435 = torch.constant.int 1
    %1275 = torch.aten.add.Scalar %1272, %float1.000000e-02_1434, %int1_1435 : !torch.vtensor<[1,?,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %1275, [%63], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %float1.000000e-02_1436 = torch.constant.float 1.000000e-02
    %int1_1437 = torch.constant.int 1
    %1276 = torch.aten.add.Scalar %1274, %float1.000000e-02_1436, %int1_1437 : !torch.vtensor<[1,?,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %1276, [%63], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %1277 = torch.aten.rsqrt %1275 : !torch.vtensor<[1,?,1],f32> -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %1277, [%63], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %1278 = torch.aten.rsqrt %1276 : !torch.vtensor<[1,?,1],f32> -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %1278, [%63], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %1279 = torch.aten.mul.Tensor %1267, %1277 : !torch.vtensor<[1,?,256],f32>, !torch.vtensor<[1,?,1],f32> -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %1279, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %1280 = torch.aten.mul.Tensor %1268, %1278 : !torch.vtensor<[1,?,256],f32>, !torch.vtensor<[1,?,1],f32> -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %1280, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int5_1438 = torch.constant.int 5
    %1281 = torch.prims.convert_element_type %1279, %int5_1438 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %1281, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int5_1439 = torch.constant.int 5
    %1282 = torch.prims.convert_element_type %1280, %int5_1439 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %1282, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %1283 = torch.aten.mul.Tensor %48, %1281 : !torch.vtensor<[256],f32>, !torch.vtensor<[1,?,256],f16> -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %1283, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %1284 = torch.aten.mul.Tensor %49, %1282 : !torch.vtensor<[256],f32>, !torch.vtensor<[1,?,256],f16> -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %1284, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int5_1440 = torch.constant.int 5
    %1285 = torch.prims.convert_element_type %1283, %int5_1440 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %1285, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int5_1441 = torch.constant.int 5
    %1286 = torch.prims.convert_element_type %1284, %int5_1441 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %1286, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int1_1442 = torch.constant.int 1
    %int0_1443 = torch.constant.int 0
    %1287 = torch.prim.ListConstruct %int1_1442, %int0_1443 : (!torch.int, !torch.int) -> !torch.list<int>
    %1288 = torch.aten.permute %50, %1287 : !torch.vtensor<[12,256],f32>, !torch.list<int> -> !torch.vtensor<[256,12],f32>
    %int1_1444 = torch.constant.int 1
    %int0_1445 = torch.constant.int 0
    %1289 = torch.prim.ListConstruct %int1_1444, %int0_1445 : (!torch.int, !torch.int) -> !torch.list<int>
    %1290 = torch.aten.permute %51, %1289 : !torch.vtensor<[11,256],f32>, !torch.list<int> -> !torch.vtensor<[256,11],f32>
    %int5_1446 = torch.constant.int 5
    %1291 = torch.prims.convert_element_type %1288, %int5_1446 : !torch.vtensor<[256,12],f32>, !torch.int -> !torch.vtensor<[256,12],f16>
    %int256_1447 = torch.constant.int 256
    %1292 = torch.prim.ListConstruct %67, %int256_1447 : (!torch.int, !torch.int) -> !torch.list<int>
    %1293 = torch.aten.view %1285, %1292 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %1293, [%63], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %1294 = torch.aten.mm %1293, %1291 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,12],f16> -> !torch.vtensor<[?,12],f16>
    torch.bind_symbolic_shape %1294, [%63], affine_map<()[s0] -> (s0 * 32, 12)> : !torch.vtensor<[?,12],f16>
    %int1_1448 = torch.constant.int 1
    %int12_1449 = torch.constant.int 12
    %1295 = torch.prim.ListConstruct %int1_1448, %67, %int12_1449 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1296 = torch.aten.view %1294, %1295 : !torch.vtensor<[?,12],f16>, !torch.list<int> -> !torch.vtensor<[1,?,12],f16>
    torch.bind_symbolic_shape %1296, [%63], affine_map<()[s0] -> (1, s0 * 32, 12)> : !torch.vtensor<[1,?,12],f16>
    %int5_1450 = torch.constant.int 5
    %1297 = torch.prims.convert_element_type %1290, %int5_1450 : !torch.vtensor<[256,11],f32>, !torch.int -> !torch.vtensor<[256,11],f16>
    %int256_1451 = torch.constant.int 256
    %1298 = torch.prim.ListConstruct %67, %int256_1451 : (!torch.int, !torch.int) -> !torch.list<int>
    %1299 = torch.aten.view %1286, %1298 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %1299, [%63], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %1300 = torch.aten.mm %1299, %1297 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,11],f16> -> !torch.vtensor<[?,11],f16>
    torch.bind_symbolic_shape %1300, [%63], affine_map<()[s0] -> (s0 * 32, 11)> : !torch.vtensor<[?,11],f16>
    %int1_1452 = torch.constant.int 1
    %int11_1453 = torch.constant.int 11
    %1301 = torch.prim.ListConstruct %int1_1452, %67, %int11_1453 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1302 = torch.aten.view %1300, %1301 : !torch.vtensor<[?,11],f16>, !torch.list<int> -> !torch.vtensor<[1,?,11],f16>
    torch.bind_symbolic_shape %1302, [%63], affine_map<()[s0] -> (1, s0 * 32, 11)> : !torch.vtensor<[1,?,11],f16>
    %1303 = torch.aten.silu %1296 : !torch.vtensor<[1,?,12],f16> -> !torch.vtensor<[1,?,12],f16>
    torch.bind_symbolic_shape %1303, [%63], affine_map<()[s0] -> (1, s0 * 32, 12)> : !torch.vtensor<[1,?,12],f16>
    %1304 = torch.aten.silu %1302 : !torch.vtensor<[1,?,11],f16> -> !torch.vtensor<[1,?,11],f16>
    torch.bind_symbolic_shape %1304, [%63], affine_map<()[s0] -> (1, s0 * 32, 11)> : !torch.vtensor<[1,?,11],f16>
    %int1_1454 = torch.constant.int 1
    %int0_1455 = torch.constant.int 0
    %1305 = torch.prim.ListConstruct %int1_1454, %int0_1455 : (!torch.int, !torch.int) -> !torch.list<int>
    %1306 = torch.aten.permute %52, %1305 : !torch.vtensor<[12,256],f32>, !torch.list<int> -> !torch.vtensor<[256,12],f32>
    %int1_1456 = torch.constant.int 1
    %int0_1457 = torch.constant.int 0
    %1307 = torch.prim.ListConstruct %int1_1456, %int0_1457 : (!torch.int, !torch.int) -> !torch.list<int>
    %1308 = torch.aten.permute %53, %1307 : !torch.vtensor<[11,256],f32>, !torch.list<int> -> !torch.vtensor<[256,11],f32>
    %int5_1458 = torch.constant.int 5
    %1309 = torch.prims.convert_element_type %1306, %int5_1458 : !torch.vtensor<[256,12],f32>, !torch.int -> !torch.vtensor<[256,12],f16>
    %int256_1459 = torch.constant.int 256
    %1310 = torch.prim.ListConstruct %67, %int256_1459 : (!torch.int, !torch.int) -> !torch.list<int>
    %1311 = torch.aten.view %1285, %1310 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %1311, [%63], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %1312 = torch.aten.mm %1311, %1309 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,12],f16> -> !torch.vtensor<[?,12],f16>
    torch.bind_symbolic_shape %1312, [%63], affine_map<()[s0] -> (s0 * 32, 12)> : !torch.vtensor<[?,12],f16>
    %int1_1460 = torch.constant.int 1
    %int12_1461 = torch.constant.int 12
    %1313 = torch.prim.ListConstruct %int1_1460, %67, %int12_1461 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1314 = torch.aten.view %1312, %1313 : !torch.vtensor<[?,12],f16>, !torch.list<int> -> !torch.vtensor<[1,?,12],f16>
    torch.bind_symbolic_shape %1314, [%63], affine_map<()[s0] -> (1, s0 * 32, 12)> : !torch.vtensor<[1,?,12],f16>
    %int5_1462 = torch.constant.int 5
    %1315 = torch.prims.convert_element_type %1308, %int5_1462 : !torch.vtensor<[256,11],f32>, !torch.int -> !torch.vtensor<[256,11],f16>
    %int256_1463 = torch.constant.int 256
    %1316 = torch.prim.ListConstruct %67, %int256_1463 : (!torch.int, !torch.int) -> !torch.list<int>
    %1317 = torch.aten.view %1286, %1316 : !torch.vtensor<[1,?,256],f16>, !torch.list<int> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %1317, [%63], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %1318 = torch.aten.mm %1317, %1315 : !torch.vtensor<[?,256],f16>, !torch.vtensor<[256,11],f16> -> !torch.vtensor<[?,11],f16>
    torch.bind_symbolic_shape %1318, [%63], affine_map<()[s0] -> (s0 * 32, 11)> : !torch.vtensor<[?,11],f16>
    %int1_1464 = torch.constant.int 1
    %int11_1465 = torch.constant.int 11
    %1319 = torch.prim.ListConstruct %int1_1464, %67, %int11_1465 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1320 = torch.aten.view %1318, %1319 : !torch.vtensor<[?,11],f16>, !torch.list<int> -> !torch.vtensor<[1,?,11],f16>
    torch.bind_symbolic_shape %1320, [%63], affine_map<()[s0] -> (1, s0 * 32, 11)> : !torch.vtensor<[1,?,11],f16>
    %1321 = torch.aten.mul.Tensor %1303, %1314 : !torch.vtensor<[1,?,12],f16>, !torch.vtensor<[1,?,12],f16> -> !torch.vtensor<[1,?,12],f16>
    torch.bind_symbolic_shape %1321, [%63], affine_map<()[s0] -> (1, s0 * 32, 12)> : !torch.vtensor<[1,?,12],f16>
    %1322 = torch.aten.mul.Tensor %1304, %1320 : !torch.vtensor<[1,?,11],f16>, !torch.vtensor<[1,?,11],f16> -> !torch.vtensor<[1,?,11],f16>
    torch.bind_symbolic_shape %1322, [%63], affine_map<()[s0] -> (1, s0 * 32, 11)> : !torch.vtensor<[1,?,11],f16>
    %int1_1466 = torch.constant.int 1
    %int0_1467 = torch.constant.int 0
    %1323 = torch.prim.ListConstruct %int1_1466, %int0_1467 : (!torch.int, !torch.int) -> !torch.list<int>
    %1324 = torch.aten.permute %54, %1323 : !torch.vtensor<[256,12],f32>, !torch.list<int> -> !torch.vtensor<[12,256],f32>
    %int1_1468 = torch.constant.int 1
    %int0_1469 = torch.constant.int 0
    %1325 = torch.prim.ListConstruct %int1_1468, %int0_1469 : (!torch.int, !torch.int) -> !torch.list<int>
    %1326 = torch.aten.permute %55, %1325 : !torch.vtensor<[256,11],f32>, !torch.list<int> -> !torch.vtensor<[11,256],f32>
    %int5_1470 = torch.constant.int 5
    %1327 = torch.prims.convert_element_type %1324, %int5_1470 : !torch.vtensor<[12,256],f32>, !torch.int -> !torch.vtensor<[12,256],f16>
    %int12_1471 = torch.constant.int 12
    %1328 = torch.prim.ListConstruct %67, %int12_1471 : (!torch.int, !torch.int) -> !torch.list<int>
    %1329 = torch.aten.view %1321, %1328 : !torch.vtensor<[1,?,12],f16>, !torch.list<int> -> !torch.vtensor<[?,12],f16>
    torch.bind_symbolic_shape %1329, [%63], affine_map<()[s0] -> (s0 * 32, 12)> : !torch.vtensor<[?,12],f16>
    %1330 = torch.aten.mm %1329, %1327 : !torch.vtensor<[?,12],f16>, !torch.vtensor<[12,256],f16> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %1330, [%63], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %int1_1472 = torch.constant.int 1
    %int256_1473 = torch.constant.int 256
    %1331 = torch.prim.ListConstruct %int1_1472, %67, %int256_1473 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1332 = torch.aten.view %1330, %1331 : !torch.vtensor<[?,256],f16>, !torch.list<int> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %1332, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int5_1474 = torch.constant.int 5
    %1333 = torch.prims.convert_element_type %1326, %int5_1474 : !torch.vtensor<[11,256],f32>, !torch.int -> !torch.vtensor<[11,256],f16>
    %int11_1475 = torch.constant.int 11
    %1334 = torch.prim.ListConstruct %67, %int11_1475 : (!torch.int, !torch.int) -> !torch.list<int>
    %1335 = torch.aten.view %1322, %1334 : !torch.vtensor<[1,?,11],f16>, !torch.list<int> -> !torch.vtensor<[?,11],f16>
    torch.bind_symbolic_shape %1335, [%63], affine_map<()[s0] -> (s0 * 32, 11)> : !torch.vtensor<[?,11],f16>
    %1336 = torch.aten.mm %1335, %1333 : !torch.vtensor<[?,11],f16>, !torch.vtensor<[11,256],f16> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %1336, [%63], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %int1_1476 = torch.constant.int 1
    %int256_1477 = torch.constant.int 256
    %1337 = torch.prim.ListConstruct %int1_1476, %67, %int256_1477 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1338 = torch.aten.view %1336, %1337 : !torch.vtensor<[?,256],f16>, !torch.list<int> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %1338, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %1339 = torch_c.to_builtin_tensor %1332 : !torch.vtensor<[1,?,256],f16> -> tensor<1x?x256xf16>
    %c1_1478 = arith.constant 1 : index
    %dim_1479 = tensor.dim %1339, %c1_1478 : tensor<1x?x256xf16>
    %1340 = flow.tensor.barrier %1339 : tensor<1x?x256xf16>{%dim_1479} on #hal.device.promise<@__device_0>
    %1341 = torch_c.from_builtin_tensor %1340 : tensor<1x?x256xf16> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %1341, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %1342 = torch_c.to_builtin_tensor %1338 : !torch.vtensor<[1,?,256],f16> -> tensor<1x?x256xf16>
    %c1_1480 = arith.constant 1 : index
    %dim_1481 = tensor.dim %1342, %c1_1480 : tensor<1x?x256xf16>
    %1343 = flow.tensor.transfer %1342 : tensor<1x?x256xf16>{%dim_1481} to #hal.device.promise<@__device_0>
    %1344 = torch_c.from_builtin_tensor %1343 : tensor<1x?x256xf16> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %1344, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int1_1482 = torch.constant.int 1
    %1345 = torch.aten.add.Tensor %1341, %1344, %int1_1482 : !torch.vtensor<[1,?,256],f16>, !torch.vtensor<[1,?,256],f16>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %1345, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %1346 = torch_c.to_builtin_tensor %1345 : !torch.vtensor<[1,?,256],f16> -> tensor<1x?x256xf16>
    %c1_1483 = arith.constant 1 : index
    %dim_1484 = tensor.dim %1346, %c1_1483 : tensor<1x?x256xf16>
    %1347 = flow.tensor.barrier %1346 : tensor<1x?x256xf16>{%dim_1484} on #hal.device.promise<@__device_0>
    %1348 = torch_c.from_builtin_tensor %1347 : tensor<1x?x256xf16> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %1348, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %1349 = torch_c.to_builtin_tensor %1345 : !torch.vtensor<[1,?,256],f16> -> tensor<1x?x256xf16>
    %c1_1485 = arith.constant 1 : index
    %dim_1486 = tensor.dim %1349, %c1_1485 : tensor<1x?x256xf16>
    %1350 = flow.tensor.transfer %1349 : tensor<1x?x256xf16>{%dim_1486} to #hal.device.promise<@__device_1>
    %1351 = torch_c.from_builtin_tensor %1350 : tensor<1x?x256xf16> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %1351, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int1_1487 = torch.constant.int 1
    %1352 = torch.aten.add.Tensor %1265, %1348, %int1_1487 : !torch.vtensor<[1,?,256],f16>, !torch.vtensor<[1,?,256],f16>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %1352, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int1_1488 = torch.constant.int 1
    %1353 = torch.aten.add.Tensor %1266, %1351, %int1_1488 : !torch.vtensor<[1,?,256],f16>, !torch.vtensor<[1,?,256],f16>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %1353, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int6_1489 = torch.constant.int 6
    %1354 = torch.prims.convert_element_type %1352, %int6_1489 : !torch.vtensor<[1,?,256],f16>, !torch.int -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %1354, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int6_1490 = torch.constant.int 6
    %1355 = torch.prims.convert_element_type %1353, %int6_1490 : !torch.vtensor<[1,?,256],f16>, !torch.int -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %1355, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int2_1491 = torch.constant.int 2
    %1356 = torch.aten.pow.Tensor_Scalar %1354, %int2_1491 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %1356, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int2_1492 = torch.constant.int 2
    %1357 = torch.aten.pow.Tensor_Scalar %1355, %int2_1492 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %1357, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int-1_1493 = torch.constant.int -1
    %1358 = torch.prim.ListConstruct %int-1_1493 : (!torch.int) -> !torch.list<int>
    %true_1494 = torch.constant.bool true
    %none_1495 = torch.constant.none
    %1359 = torch.aten.mean.dim %1356, %1358, %true_1494, %none_1495 : !torch.vtensor<[1,?,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %1359, [%63], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %int-1_1496 = torch.constant.int -1
    %1360 = torch.prim.ListConstruct %int-1_1496 : (!torch.int) -> !torch.list<int>
    %true_1497 = torch.constant.bool true
    %none_1498 = torch.constant.none
    %1361 = torch.aten.mean.dim %1357, %1360, %true_1497, %none_1498 : !torch.vtensor<[1,?,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %1361, [%63], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %float1.000000e-02_1499 = torch.constant.float 1.000000e-02
    %int1_1500 = torch.constant.int 1
    %1362 = torch.aten.add.Scalar %1359, %float1.000000e-02_1499, %int1_1500 : !torch.vtensor<[1,?,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %1362, [%63], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %float1.000000e-02_1501 = torch.constant.float 1.000000e-02
    %int1_1502 = torch.constant.int 1
    %1363 = torch.aten.add.Scalar %1361, %float1.000000e-02_1501, %int1_1502 : !torch.vtensor<[1,?,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %1363, [%63], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %1364 = torch.aten.rsqrt %1362 : !torch.vtensor<[1,?,1],f32> -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %1364, [%63], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %1365 = torch.aten.rsqrt %1363 : !torch.vtensor<[1,?,1],f32> -> !torch.vtensor<[1,?,1],f32>
    torch.bind_symbolic_shape %1365, [%63], affine_map<()[s0] -> (1, s0 * 32, 1)> : !torch.vtensor<[1,?,1],f32>
    %1366 = torch.aten.mul.Tensor %1354, %1364 : !torch.vtensor<[1,?,256],f32>, !torch.vtensor<[1,?,1],f32> -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %1366, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %1367 = torch.aten.mul.Tensor %1355, %1365 : !torch.vtensor<[1,?,256],f32>, !torch.vtensor<[1,?,1],f32> -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %1367, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int5_1503 = torch.constant.int 5
    %1368 = torch.prims.convert_element_type %1366, %int5_1503 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %1368, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int5_1504 = torch.constant.int 5
    %1369 = torch.prims.convert_element_type %1367, %int5_1504 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %1369, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %1370 = torch.aten.mul.Tensor %56, %1368 : !torch.vtensor<[1,256],f32>, !torch.vtensor<[1,?,256],f16> -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %1370, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %1371 = torch.aten.mul.Tensor %57, %1369 : !torch.vtensor<[1,256],f32>, !torch.vtensor<[1,?,256],f16> -> !torch.vtensor<[1,?,256],f32>
    torch.bind_symbolic_shape %1371, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f32>
    %int5_1505 = torch.constant.int 5
    %1372 = torch.prims.convert_element_type %1370, %int5_1505 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %1372, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int5_1506 = torch.constant.int 5
    %1373 = torch.prims.convert_element_type %1371, %int5_1506 : !torch.vtensor<[1,?,256],f32>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %1373, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int1_1507 = torch.constant.int 1
    %int0_1508 = torch.constant.int 0
    %1374 = torch.prim.ListConstruct %int1_1507, %int0_1508 : (!torch.int, !torch.int) -> !torch.list<int>
    %1375 = torch.aten.permute %58, %1374 : !torch.vtensor<[256,128],f32>, !torch.list<int> -> !torch.vtensor<[128,256],f32>
    %int1_1509 = torch.constant.int 1
    %int0_1510 = torch.constant.int 0
    %1376 = torch.prim.ListConstruct %int1_1509, %int0_1510 : (!torch.int, !torch.int) -> !torch.list<int>
    %1377 = torch.aten.permute %59, %1376 : !torch.vtensor<[256,128],f32>, !torch.list<int> -> !torch.vtensor<[128,256],f32>
    %int0_1511 = torch.constant.int 0
    %int0_1512 = torch.constant.int 0
    %int9223372036854775807_1513 = torch.constant.int 9223372036854775807
    %int1_1514 = torch.constant.int 1
    %1378 = torch.aten.slice.Tensor %1372, %int0_1511, %int0_1512, %int9223372036854775807_1513, %int1_1514 : !torch.vtensor<[1,?,256],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %1378, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int1_1515 = torch.constant.int 1
    %int0_1516 = torch.constant.int 0
    %int9223372036854775807_1517 = torch.constant.int 9223372036854775807
    %int1_1518 = torch.constant.int 1
    %1379 = torch.aten.slice.Tensor %1378, %int1_1515, %int0_1516, %int9223372036854775807_1517, %int1_1518 : !torch.vtensor<[1,?,256],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %1379, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int2_1519 = torch.constant.int 2
    %int0_1520 = torch.constant.int 0
    %int128_1521 = torch.constant.int 128
    %int1_1522 = torch.constant.int 1
    %1380 = torch.aten.slice.Tensor %1379, %int2_1519, %int0_1520, %int128_1521, %int1_1522 : !torch.vtensor<[1,?,256],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,128],f16>
    torch.bind_symbolic_shape %1380, [%63], affine_map<()[s0] -> (1, s0 * 32, 128)> : !torch.vtensor<[1,?,128],f16>
    %int0_1523 = torch.constant.int 0
    %int0_1524 = torch.constant.int 0
    %int9223372036854775807_1525 = torch.constant.int 9223372036854775807
    %int1_1526 = torch.constant.int 1
    %1381 = torch.aten.slice.Tensor %1373, %int0_1523, %int0_1524, %int9223372036854775807_1525, %int1_1526 : !torch.vtensor<[1,?,256],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %1381, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int1_1527 = torch.constant.int 1
    %int0_1528 = torch.constant.int 0
    %int9223372036854775807_1529 = torch.constant.int 9223372036854775807
    %int1_1530 = torch.constant.int 1
    %1382 = torch.aten.slice.Tensor %1381, %int1_1527, %int0_1528, %int9223372036854775807_1529, %int1_1530 : !torch.vtensor<[1,?,256],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %1382, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int2_1531 = torch.constant.int 2
    %int128_1532 = torch.constant.int 128
    %int256_1533 = torch.constant.int 256
    %int1_1534 = torch.constant.int 1
    %1383 = torch.aten.slice.Tensor %1382, %int2_1531, %int128_1532, %int256_1533, %int1_1534 : !torch.vtensor<[1,?,256],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,128],f16>
    torch.bind_symbolic_shape %1383, [%63], affine_map<()[s0] -> (1, s0 * 32, 128)> : !torch.vtensor<[1,?,128],f16>
    %int5_1535 = torch.constant.int 5
    %1384 = torch.prims.convert_element_type %1375, %int5_1535 : !torch.vtensor<[128,256],f32>, !torch.int -> !torch.vtensor<[128,256],f16>
    %int128_1536 = torch.constant.int 128
    %1385 = torch.prim.ListConstruct %67, %int128_1536 : (!torch.int, !torch.int) -> !torch.list<int>
    %1386 = torch.aten.view %1380, %1385 : !torch.vtensor<[1,?,128],f16>, !torch.list<int> -> !torch.vtensor<[?,128],f16>
    torch.bind_symbolic_shape %1386, [%63], affine_map<()[s0] -> (s0 * 32, 128)> : !torch.vtensor<[?,128],f16>
    %1387 = torch.aten.mm %1386, %1384 : !torch.vtensor<[?,128],f16>, !torch.vtensor<[128,256],f16> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %1387, [%63], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %int1_1537 = torch.constant.int 1
    %int256_1538 = torch.constant.int 256
    %1388 = torch.prim.ListConstruct %int1_1537, %67, %int256_1538 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1389 = torch.aten.view %1387, %1388 : !torch.vtensor<[?,256],f16>, !torch.list<int> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %1389, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int5_1539 = torch.constant.int 5
    %1390 = torch.prims.convert_element_type %1377, %int5_1539 : !torch.vtensor<[128,256],f32>, !torch.int -> !torch.vtensor<[128,256],f16>
    %int128_1540 = torch.constant.int 128
    %1391 = torch.prim.ListConstruct %67, %int128_1540 : (!torch.int, !torch.int) -> !torch.list<int>
    %1392 = torch.aten.view %1383, %1391 : !torch.vtensor<[1,?,128],f16>, !torch.list<int> -> !torch.vtensor<[?,128],f16>
    torch.bind_symbolic_shape %1392, [%63], affine_map<()[s0] -> (s0 * 32, 128)> : !torch.vtensor<[?,128],f16>
    %1393 = torch.aten.mm %1392, %1390 : !torch.vtensor<[?,128],f16>, !torch.vtensor<[128,256],f16> -> !torch.vtensor<[?,256],f16>
    torch.bind_symbolic_shape %1393, [%63], affine_map<()[s0] -> (s0 * 32, 256)> : !torch.vtensor<[?,256],f16>
    %int1_1541 = torch.constant.int 1
    %int256_1542 = torch.constant.int 256
    %1394 = torch.prim.ListConstruct %int1_1541, %67, %int256_1542 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1395 = torch.aten.view %1393, %1394 : !torch.vtensor<[?,256],f16>, !torch.list<int> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %1395, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %1396 = torch_c.to_builtin_tensor %1389 : !torch.vtensor<[1,?,256],f16> -> tensor<1x?x256xf16>
    %c1_1543 = arith.constant 1 : index
    %dim_1544 = tensor.dim %1396, %c1_1543 : tensor<1x?x256xf16>
    %1397 = flow.tensor.barrier %1396 : tensor<1x?x256xf16>{%dim_1544} on #hal.device.promise<@__device_0>
    %1398 = torch_c.from_builtin_tensor %1397 : tensor<1x?x256xf16> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %1398, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %1399 = torch_c.to_builtin_tensor %1395 : !torch.vtensor<[1,?,256],f16> -> tensor<1x?x256xf16>
    %c1_1545 = arith.constant 1 : index
    %dim_1546 = tensor.dim %1399, %c1_1545 : tensor<1x?x256xf16>
    %1400 = flow.tensor.transfer %1399 : tensor<1x?x256xf16>{%dim_1546} to #hal.device.promise<@__device_0>
    %1401 = torch_c.from_builtin_tensor %1400 : tensor<1x?x256xf16> -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %1401, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    %int1_1547 = torch.constant.int 1
    %1402 = torch.aten.add.Tensor %1398, %1401, %int1_1547 : !torch.vtensor<[1,?,256],f16>, !torch.vtensor<[1,?,256],f16>, !torch.int -> !torch.vtensor<[1,?,256],f16>
    torch.bind_symbolic_shape %1402, [%63], affine_map<()[s0] -> (1, s0 * 32, 256)> : !torch.vtensor<[1,?,256],f16>
    return %1402 : !torch.vtensor<[1,?,256],f16>
  }
  func.func @decode_bs1(%arg0: !torch.vtensor<[1,1],si64> {iree.abi.affinity = #hal.device.promise<@__device_0>}, %arg1: !torch.vtensor<[1],si64> {iree.abi.affinity = #hal.device.promise<@__device_0>}, %arg2: !torch.vtensor<[1],si64> {iree.abi.affinity = #hal.device.promise<@__device_0>}, %arg3: !torch.vtensor<[1,?],si64> {iree.abi.affinity = #hal.device.promise<@__device_0>}, %arg4: !torch.tensor<[?,12288],f16> {iree.abi.affinity = #hal.device.promise<@__device_0>}, %arg5: !torch.tensor<[?,12288],f16> {iree.abi.affinity = #hal.device.promise<@__device_1>}) -> !torch.vtensor<[1,1,256],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %__auto.token_embd.weight = util.global.load @__auto.token_embd.weight : tensor<256x256xf32>
    %0 = torch_c.from_builtin_tensor %__auto.token_embd.weight : tensor<256x256xf32> -> !torch.vtensor<[256,256],f32>
    %__auto.token_embd.weight$1 = util.global.load @__auto.token_embd.weight$1 : tensor<256x256xf32>
    %1 = torch_c.from_builtin_tensor %__auto.token_embd.weight$1 : tensor<256x256xf32> -> !torch.vtensor<[256,256],f32>
    %__auto.blk.0.attn_norm.weight = util.global.load @__auto.blk.0.attn_norm.weight : tensor<256xf32>
    %2 = torch_c.from_builtin_tensor %__auto.blk.0.attn_norm.weight : tensor<256xf32> -> !torch.vtensor<[256],f32>
    %__auto.blk.0.attn_norm.weight$1 = util.global.load @__auto.blk.0.attn_norm.weight$1 : tensor<256xf32>
    %3 = torch_c.from_builtin_tensor %__auto.blk.0.attn_norm.weight$1 : tensor<256xf32> -> !torch.vtensor<[256],f32>
    %__auto.blk.0.attn_q.weight.shard.0 = util.global.load @__auto.blk.0.attn_q.weight.shard.0 : tensor<128x256xf32>
    %4 = torch_c.from_builtin_tensor %__auto.blk.0.attn_q.weight.shard.0 : tensor<128x256xf32> -> !torch.vtensor<[128,256],f32>
    %__auto.blk.0.attn_q.weight.shard.1 = util.global.load @__auto.blk.0.attn_q.weight.shard.1 : tensor<128x256xf32>
    %5 = torch_c.from_builtin_tensor %__auto.blk.0.attn_q.weight.shard.1 : tensor<128x256xf32> -> !torch.vtensor<[128,256],f32>
    %__auto.blk.0.attn_k.weight.shard.0 = util.global.load @__auto.blk.0.attn_k.weight.shard.0 : tensor<64x256xf32>
    %6 = torch_c.from_builtin_tensor %__auto.blk.0.attn_k.weight.shard.0 : tensor<64x256xf32> -> !torch.vtensor<[64,256],f32>
    %__auto.blk.0.attn_k.weight.shard.1 = util.global.load @__auto.blk.0.attn_k.weight.shard.1 : tensor<64x256xf32>
    %7 = torch_c.from_builtin_tensor %__auto.blk.0.attn_k.weight.shard.1 : tensor<64x256xf32> -> !torch.vtensor<[64,256],f32>
    %__auto.blk.0.attn_v.weight.shard.0 = util.global.load @__auto.blk.0.attn_v.weight.shard.0 : tensor<64x256xf32>
    %8 = torch_c.from_builtin_tensor %__auto.blk.0.attn_v.weight.shard.0 : tensor<64x256xf32> -> !torch.vtensor<[64,256],f32>
    %__auto.blk.0.attn_v.weight.shard.1 = util.global.load @__auto.blk.0.attn_v.weight.shard.1 : tensor<64x256xf32>
    %9 = torch_c.from_builtin_tensor %__auto.blk.0.attn_v.weight.shard.1 : tensor<64x256xf32> -> !torch.vtensor<[64,256],f32>
    %10 = torch.vtensor.literal(dense<0> : tensor<si64>) : !torch.vtensor<[],si64>
    %11 = torch.vtensor.literal(dense<0> : tensor<si64>) : !torch.vtensor<[],si64>
    %12 = torch.vtensor.literal(dense<1> : tensor<si64>) : !torch.vtensor<[],si64>
    %13 = torch.vtensor.literal(dense<1> : tensor<si64>) : !torch.vtensor<[],si64>
    %__auto.blk.0.attn_output.weight.shard.0 = util.global.load @__auto.blk.0.attn_output.weight.shard.0 : tensor<256x128xf32>
    %14 = torch_c.from_builtin_tensor %__auto.blk.0.attn_output.weight.shard.0 : tensor<256x128xf32> -> !torch.vtensor<[256,128],f32>
    %__auto.blk.0.attn_output.weight.shard.1 = util.global.load @__auto.blk.0.attn_output.weight.shard.1 : tensor<256x128xf32>
    %15 = torch_c.from_builtin_tensor %__auto.blk.0.attn_output.weight.shard.1 : tensor<256x128xf32> -> !torch.vtensor<[256,128],f32>
    %__auto.blk.0.ffn_norm.weight = util.global.load @__auto.blk.0.ffn_norm.weight : tensor<256xf32>
    %16 = torch_c.from_builtin_tensor %__auto.blk.0.ffn_norm.weight : tensor<256xf32> -> !torch.vtensor<[256],f32>
    %__auto.blk.0.ffn_norm.weight$1 = util.global.load @__auto.blk.0.ffn_norm.weight$1 : tensor<256xf32>
    %17 = torch_c.from_builtin_tensor %__auto.blk.0.ffn_norm.weight$1 : tensor<256xf32> -> !torch.vtensor<[256],f32>
    %__auto.blk.0.ffn_gate.weight.shard.0 = util.global.load @__auto.blk.0.ffn_gate.weight.shard.0 : tensor<12x256xf32>
    %18 = torch_c.from_builtin_tensor %__auto.blk.0.ffn_gate.weight.shard.0 : tensor<12x256xf32> -> !torch.vtensor<[12,256],f32>
    %__auto.blk.0.ffn_gate.weight.shard.1 = util.global.load @__auto.blk.0.ffn_gate.weight.shard.1 : tensor<11x256xf32>
    %19 = torch_c.from_builtin_tensor %__auto.blk.0.ffn_gate.weight.shard.1 : tensor<11x256xf32> -> !torch.vtensor<[11,256],f32>
    %__auto.blk.0.ffn_up.weight.shard.0 = util.global.load @__auto.blk.0.ffn_up.weight.shard.0 : tensor<12x256xf32>
    %20 = torch_c.from_builtin_tensor %__auto.blk.0.ffn_up.weight.shard.0 : tensor<12x256xf32> -> !torch.vtensor<[12,256],f32>
    %__auto.blk.0.ffn_up.weight.shard.1 = util.global.load @__auto.blk.0.ffn_up.weight.shard.1 : tensor<11x256xf32>
    %21 = torch_c.from_builtin_tensor %__auto.blk.0.ffn_up.weight.shard.1 : tensor<11x256xf32> -> !torch.vtensor<[11,256],f32>
    %__auto.blk.0.ffn_down.weight.shard.0 = util.global.load @__auto.blk.0.ffn_down.weight.shard.0 : tensor<256x12xf32>
    %22 = torch_c.from_builtin_tensor %__auto.blk.0.ffn_down.weight.shard.0 : tensor<256x12xf32> -> !torch.vtensor<[256,12],f32>
    %__auto.blk.0.ffn_down.weight.shard.1 = util.global.load @__auto.blk.0.ffn_down.weight.shard.1 : tensor<256x11xf32>
    %23 = torch_c.from_builtin_tensor %__auto.blk.0.ffn_down.weight.shard.1 : tensor<256x11xf32> -> !torch.vtensor<[256,11],f32>
    %__auto.blk.1.attn_norm.weight = util.global.load @__auto.blk.1.attn_norm.weight : tensor<256xf32>
    %24 = torch_c.from_builtin_tensor %__auto.blk.1.attn_norm.weight : tensor<256xf32> -> !torch.vtensor<[256],f32>
    %__auto.blk.1.attn_norm.weight$1 = util.global.load @__auto.blk.1.attn_norm.weight$1 : tensor<256xf32>
    %25 = torch_c.from_builtin_tensor %__auto.blk.1.attn_norm.weight$1 : tensor<256xf32> -> !torch.vtensor<[256],f32>
    %__auto.blk.1.attn_q.weight.shard.0 = util.global.load @__auto.blk.1.attn_q.weight.shard.0 : tensor<128x256xf32>
    %26 = torch_c.from_builtin_tensor %__auto.blk.1.attn_q.weight.shard.0 : tensor<128x256xf32> -> !torch.vtensor<[128,256],f32>
    %__auto.blk.1.attn_q.weight.shard.1 = util.global.load @__auto.blk.1.attn_q.weight.shard.1 : tensor<128x256xf32>
    %27 = torch_c.from_builtin_tensor %__auto.blk.1.attn_q.weight.shard.1 : tensor<128x256xf32> -> !torch.vtensor<[128,256],f32>
    %__auto.blk.1.attn_k.weight.shard.0 = util.global.load @__auto.blk.1.attn_k.weight.shard.0 : tensor<64x256xf32>
    %28 = torch_c.from_builtin_tensor %__auto.blk.1.attn_k.weight.shard.0 : tensor<64x256xf32> -> !torch.vtensor<[64,256],f32>
    %__auto.blk.1.attn_k.weight.shard.1 = util.global.load @__auto.blk.1.attn_k.weight.shard.1 : tensor<64x256xf32>
    %29 = torch_c.from_builtin_tensor %__auto.blk.1.attn_k.weight.shard.1 : tensor<64x256xf32> -> !torch.vtensor<[64,256],f32>
    %__auto.blk.1.attn_v.weight.shard.0 = util.global.load @__auto.blk.1.attn_v.weight.shard.0 : tensor<64x256xf32>
    %30 = torch_c.from_builtin_tensor %__auto.blk.1.attn_v.weight.shard.0 : tensor<64x256xf32> -> !torch.vtensor<[64,256],f32>
    %__auto.blk.1.attn_v.weight.shard.1 = util.global.load @__auto.blk.1.attn_v.weight.shard.1 : tensor<64x256xf32>
    %31 = torch_c.from_builtin_tensor %__auto.blk.1.attn_v.weight.shard.1 : tensor<64x256xf32> -> !torch.vtensor<[64,256],f32>
    %32 = torch.vtensor.literal(dense<0> : tensor<si64>) : !torch.vtensor<[],si64>
    %33 = torch.vtensor.literal(dense<0> : tensor<si64>) : !torch.vtensor<[],si64>
    %34 = torch.vtensor.literal(dense<1> : tensor<si64>) : !torch.vtensor<[],si64>
    %35 = torch.vtensor.literal(dense<1> : tensor<si64>) : !torch.vtensor<[],si64>
    %__auto.blk.1.attn_output.weight.shard.0 = util.global.load @__auto.blk.1.attn_output.weight.shard.0 : tensor<256x128xf32>
    %36 = torch_c.from_builtin_tensor %__auto.blk.1.attn_output.weight.shard.0 : tensor<256x128xf32> -> !torch.vtensor<[256,128],f32>
    %__auto.blk.1.attn_output.weight.shard.1 = util.global.load @__auto.blk.1.attn_output.weight.shard.1 : tensor<256x128xf32>
    %37 = torch_c.from_builtin_tensor %__auto.blk.1.attn_output.weight.shard.1 : tensor<256x128xf32> -> !torch.vtensor<[256,128],f32>
    %__auto.blk.1.ffn_norm.weight = util.global.load @__auto.blk.1.ffn_norm.weight : tensor<256xf32>
    %38 = torch_c.from_builtin_tensor %__auto.blk.1.ffn_norm.weight : tensor<256xf32> -> !torch.vtensor<[256],f32>
    %__auto.blk.1.ffn_norm.weight$1 = util.global.load @__auto.blk.1.ffn_norm.weight$1 : tensor<256xf32>
    %39 = torch_c.from_builtin_tensor %__auto.blk.1.ffn_norm.weight$1 : tensor<256xf32> -> !torch.vtensor<[256],f32>
    %__auto.blk.1.ffn_gate.weight.shard.0 = util.global.load @__auto.blk.1.ffn_gate.weight.shard.0 : tensor<12x256xf32>
    %40 = torch_c.from_builtin_tensor %__auto.blk.1.ffn_gate.weight.shard.0 : tensor<12x256xf32> -> !torch.vtensor<[12,256],f32>
    %__auto.blk.1.ffn_gate.weight.shard.1 = util.global.load @__auto.blk.1.ffn_gate.weight.shard.1 : tensor<11x256xf32>
    %41 = torch_c.from_builtin_tensor %__auto.blk.1.ffn_gate.weight.shard.1 : tensor<11x256xf32> -> !torch.vtensor<[11,256],f32>
    %__auto.blk.1.ffn_up.weight.shard.0 = util.global.load @__auto.blk.1.ffn_up.weight.shard.0 : tensor<12x256xf32>
    %42 = torch_c.from_builtin_tensor %__auto.blk.1.ffn_up.weight.shard.0 : tensor<12x256xf32> -> !torch.vtensor<[12,256],f32>
    %__auto.blk.1.ffn_up.weight.shard.1 = util.global.load @__auto.blk.1.ffn_up.weight.shard.1 : tensor<11x256xf32>
    %43 = torch_c.from_builtin_tensor %__auto.blk.1.ffn_up.weight.shard.1 : tensor<11x256xf32> -> !torch.vtensor<[11,256],f32>
    %__auto.blk.1.ffn_down.weight.shard.0 = util.global.load @__auto.blk.1.ffn_down.weight.shard.0 : tensor<256x12xf32>
    %44 = torch_c.from_builtin_tensor %__auto.blk.1.ffn_down.weight.shard.0 : tensor<256x12xf32> -> !torch.vtensor<[256,12],f32>
    %__auto.blk.1.ffn_down.weight.shard.1 = util.global.load @__auto.blk.1.ffn_down.weight.shard.1 : tensor<256x11xf32>
    %45 = torch_c.from_builtin_tensor %__auto.blk.1.ffn_down.weight.shard.1 : tensor<256x11xf32> -> !torch.vtensor<[256,11],f32>
    %__auto.blk.2.attn_norm.weight = util.global.load @__auto.blk.2.attn_norm.weight : tensor<256xf32>
    %46 = torch_c.from_builtin_tensor %__auto.blk.2.attn_norm.weight : tensor<256xf32> -> !torch.vtensor<[256],f32>
    %__auto.blk.2.attn_norm.weight$1 = util.global.load @__auto.blk.2.attn_norm.weight$1 : tensor<256xf32>
    %47 = torch_c.from_builtin_tensor %__auto.blk.2.attn_norm.weight$1 : tensor<256xf32> -> !torch.vtensor<[256],f32>
    %__auto.blk.2.attn_q.weight.shard.0 = util.global.load @__auto.blk.2.attn_q.weight.shard.0 : tensor<128x256xf32>
    %48 = torch_c.from_builtin_tensor %__auto.blk.2.attn_q.weight.shard.0 : tensor<128x256xf32> -> !torch.vtensor<[128,256],f32>
    %__auto.blk.2.attn_q.weight.shard.1 = util.global.load @__auto.blk.2.attn_q.weight.shard.1 : tensor<128x256xf32>
    %49 = torch_c.from_builtin_tensor %__auto.blk.2.attn_q.weight.shard.1 : tensor<128x256xf32> -> !torch.vtensor<[128,256],f32>
    %__auto.blk.2.attn_k.weight.shard.0 = util.global.load @__auto.blk.2.attn_k.weight.shard.0 : tensor<64x256xf32>
    %50 = torch_c.from_builtin_tensor %__auto.blk.2.attn_k.weight.shard.0 : tensor<64x256xf32> -> !torch.vtensor<[64,256],f32>
    %__auto.blk.2.attn_k.weight.shard.1 = util.global.load @__auto.blk.2.attn_k.weight.shard.1 : tensor<64x256xf32>
    %51 = torch_c.from_builtin_tensor %__auto.blk.2.attn_k.weight.shard.1 : tensor<64x256xf32> -> !torch.vtensor<[64,256],f32>
    %__auto.blk.2.attn_v.weight.shard.0 = util.global.load @__auto.blk.2.attn_v.weight.shard.0 : tensor<64x256xf32>
    %52 = torch_c.from_builtin_tensor %__auto.blk.2.attn_v.weight.shard.0 : tensor<64x256xf32> -> !torch.vtensor<[64,256],f32>
    %__auto.blk.2.attn_v.weight.shard.1 = util.global.load @__auto.blk.2.attn_v.weight.shard.1 : tensor<64x256xf32>
    %53 = torch_c.from_builtin_tensor %__auto.blk.2.attn_v.weight.shard.1 : tensor<64x256xf32> -> !torch.vtensor<[64,256],f32>
    %54 = torch.vtensor.literal(dense<0> : tensor<si64>) : !torch.vtensor<[],si64>
    %55 = torch.vtensor.literal(dense<0> : tensor<si64>) : !torch.vtensor<[],si64>
    %56 = torch.vtensor.literal(dense<1> : tensor<si64>) : !torch.vtensor<[],si64>
    %57 = torch.vtensor.literal(dense<1> : tensor<si64>) : !torch.vtensor<[],si64>
    %__auto.blk.2.attn_output.weight.shard.0 = util.global.load @__auto.blk.2.attn_output.weight.shard.0 : tensor<256x128xf32>
    %58 = torch_c.from_builtin_tensor %__auto.blk.2.attn_output.weight.shard.0 : tensor<256x128xf32> -> !torch.vtensor<[256,128],f32>
    %__auto.blk.2.attn_output.weight.shard.1 = util.global.load @__auto.blk.2.attn_output.weight.shard.1 : tensor<256x128xf32>
    %59 = torch_c.from_builtin_tensor %__auto.blk.2.attn_output.weight.shard.1 : tensor<256x128xf32> -> !torch.vtensor<[256,128],f32>
    %__auto.blk.2.ffn_norm.weight = util.global.load @__auto.blk.2.ffn_norm.weight : tensor<256xf32>
    %60 = torch_c.from_builtin_tensor %__auto.blk.2.ffn_norm.weight : tensor<256xf32> -> !torch.vtensor<[256],f32>
    %__auto.blk.2.ffn_norm.weight$1 = util.global.load @__auto.blk.2.ffn_norm.weight$1 : tensor<256xf32>
    %61 = torch_c.from_builtin_tensor %__auto.blk.2.ffn_norm.weight$1 : tensor<256xf32> -> !torch.vtensor<[256],f32>
    %__auto.blk.2.ffn_gate.weight.shard.0 = util.global.load @__auto.blk.2.ffn_gate.weight.shard.0 : tensor<12x256xf32>
    %62 = torch_c.from_builtin_tensor %__auto.blk.2.ffn_gate.weight.shard.0 : tensor<12x256xf32> -> !torch.vtensor<[12,256],f32>
    %__auto.blk.2.ffn_gate.weight.shard.1 = util.global.load @__auto.blk.2.ffn_gate.weight.shard.1 : tensor<11x256xf32>
    %63 = torch_c.from_builtin_tensor %__auto.blk.2.ffn_gate.weight.shard.1 : tensor<11x256xf32> -> !torch.vtensor<[11,256],f32>
    %__auto.blk.2.ffn_up.weight.shard.0 = util.global.load @__auto.blk.2.ffn_up.weight.shard.0 : tensor<12x256xf32>
    %64 = torch_c.from_builtin_tensor %__auto.blk.2.ffn_up.weight.shard.0 : tensor<12x256xf32> -> !torch.vtensor<[12,256],f32>
    %__auto.blk.2.ffn_up.weight.shard.1 = util.global.load @__auto.blk.2.ffn_up.weight.shard.1 : tensor<11x256xf32>
    %65 = torch_c.from_builtin_tensor %__auto.blk.2.ffn_up.weight.shard.1 : tensor<11x256xf32> -> !torch.vtensor<[11,256],f32>
    %__auto.blk.2.ffn_down.weight.shard.0 = util.global.load @__auto.blk.2.ffn_down.weight.shard.0 : tensor<256x12xf32>
    %66 = torch_c.from_builtin_tensor %__auto.blk.2.ffn_down.weight.shard.0 : tensor<256x12xf32> -> !torch.vtensor<[256,12],f32>
    %__auto.blk.2.ffn_down.weight.shard.1 = util.global.load @__auto.blk.2.ffn_down.weight.shard.1 : tensor<256x11xf32>
    %67 = torch_c.from_builtin_tensor %__auto.blk.2.ffn_down.weight.shard.1 : tensor<256x11xf32> -> !torch.vtensor<[256,11],f32>
    %__auto.output_norm.weight = util.global.load @__auto.output_norm.weight : tensor<1x256xf32>
    %68 = torch_c.from_builtin_tensor %__auto.output_norm.weight : tensor<1x256xf32> -> !torch.vtensor<[1,256],f32>
    %__auto.output_norm.weight$1 = util.global.load @__auto.output_norm.weight$1 : tensor<1x256xf32>
    %69 = torch_c.from_builtin_tensor %__auto.output_norm.weight$1 : tensor<1x256xf32> -> !torch.vtensor<[1,256],f32>
    %__auto.output.weight.shard.0 = util.global.load @__auto.output.weight.shard.0 : tensor<256x128xf32>
    %70 = torch_c.from_builtin_tensor %__auto.output.weight.shard.0 : tensor<256x128xf32> -> !torch.vtensor<[256,128],f32>
    %__auto.output.weight.shard.1 = util.global.load @__auto.output.weight.shard.1 : tensor<256x128xf32>
    %71 = torch_c.from_builtin_tensor %__auto.output.weight.shard.1 : tensor<256x128xf32> -> !torch.vtensor<[256,128],f32>
    %72 = torch.copy.to_vtensor %arg4 : !torch.vtensor<[?,12288],f16>
    %73 = torch.copy.to_vtensor %arg5 : !torch.vtensor<[?,12288],f16>
    %74 = torch.symbolic_int "s0" {min_val = 2, max_val = 3} : !torch.int
    %75 = torch.symbolic_int "s1" {min_val = 0, max_val = 9223372036854775807} : !torch.int
    torch.bind_symbolic_shape %arg3, [%74], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %72, [%75], affine_map<()[s0] -> (s0, 12288)> : !torch.vtensor<[?,12288],f16>
    torch.bind_symbolic_shape %73, [%75], affine_map<()[s0] -> (s0, 12288)> : !torch.vtensor<[?,12288],f16>
    %int1 = torch.constant.int 1
    %76 = torch.aten.size.int %arg3, %int1 : !torch.vtensor<[1,?],si64>, !torch.int -> !torch.int
    %int0 = torch.constant.int 0
    %77 = torch.aten.size.int %72, %int0 : !torch.vtensor<[?,12288],f16>, !torch.int -> !torch.int
    %int32 = torch.constant.int 32
    %78 = torch.aten.mul.int %76, %int32 : !torch.int, !torch.int -> !torch.int
    %int0_0 = torch.constant.int 0
    %int1_1 = torch.constant.int 1
    %none = torch.constant.none
    %none_2 = torch.constant.none
    %cpu = torch.constant.device "cpu"
    %false = torch.constant.bool false
    %79 = torch.aten.arange.start_step %int0_0, %78, %int1_1, %none, %none_2, %cpu, %false : !torch.int, !torch.int, !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[?],si64>
    torch.bind_symbolic_shape %79, [%74], affine_map<()[s0] -> (s0 * 32)> : !torch.vtensor<[?],si64>
    %int-1 = torch.constant.int -1
    %80 = torch.aten.unsqueeze %arg1, %int-1 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %81 = torch.aten.ge.Tensor %79, %80 : !torch.vtensor<[?],si64>, !torch.vtensor<[1,1],si64> -> !torch.vtensor<[1,?],i1>
    torch.bind_symbolic_shape %81, [%74], affine_map<()[s0] -> (1, s0 * 32)> : !torch.vtensor<[1,?],i1>
    %int0_3 = torch.constant.int 0
    %int6 = torch.constant.int 6
    %int0_4 = torch.constant.int 0
    %cpu_5 = torch.constant.device "cpu"
    %none_6 = torch.constant.none
    %82 = torch.aten.scalar_tensor %int0_3, %int6, %int0_4, %cpu_5, %none_6 : !torch.int, !torch.int, !torch.int, !torch.Device, !torch.none -> !torch.vtensor<[],f32>
    %float-Inf = torch.constant.float 0xFFF0000000000000
    %int6_7 = torch.constant.int 6
    %int0_8 = torch.constant.int 0
    %cpu_9 = torch.constant.device "cpu"
    %none_10 = torch.constant.none
    %83 = torch.aten.scalar_tensor %float-Inf, %int6_7, %int0_8, %cpu_9, %none_10 : !torch.float, !torch.int, !torch.int, !torch.Device, !torch.none -> !torch.vtensor<[],f32>
    %84 = torch.aten.where.self %81, %83, %82 : !torch.vtensor<[1,?],i1>, !torch.vtensor<[],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[1,?],f32>
    torch.bind_symbolic_shape %84, [%74], affine_map<()[s0] -> (1, s0 * 32)> : !torch.vtensor<[1,?],f32>
    %int5 = torch.constant.int 5
    %85 = torch.prims.convert_element_type %84, %int5 : !torch.vtensor<[1,?],f32>, !torch.int -> !torch.vtensor<[1,?],f16>
    torch.bind_symbolic_shape %85, [%74], affine_map<()[s0] -> (1, s0 * 32)> : !torch.vtensor<[1,?],f16>
    %int1_11 = torch.constant.int 1
    %86 = torch.aten.unsqueeze %85, %int1_11 : !torch.vtensor<[1,?],f16>, !torch.int -> !torch.vtensor<[1,1,?],f16>
    torch.bind_symbolic_shape %86, [%74], affine_map<()[s0] -> (1, 1, s0 * 32)> : !torch.vtensor<[1,1,?],f16>
    %int1_12 = torch.constant.int 1
    %87 = torch.aten.unsqueeze %86, %int1_12 : !torch.vtensor<[1,1,?],f16>, !torch.int -> !torch.vtensor<[1,1,1,?],f16>
    torch.bind_symbolic_shape %87, [%74], affine_map<()[s0] -> (1, 1, 1, s0 * 32)> : !torch.vtensor<[1,1,1,?],f16>
    %int5_13 = torch.constant.int 5
    %88 = torch.prims.convert_element_type %87, %int5_13 : !torch.vtensor<[1,1,1,?],f16>, !torch.int -> !torch.vtensor<[1,1,1,?],f16>
    torch.bind_symbolic_shape %88, [%74], affine_map<()[s0] -> (1, 1, 1, s0 * 32)> : !torch.vtensor<[1,1,1,?],f16>
    %89 = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[1,1],si64> -> tensor<1x1xi64>
    %90 = flow.tensor.transfer %89 : tensor<1x1xi64> to #hal.device.promise<@__device_0>
    %91 = torch_c.from_builtin_tensor %90 : tensor<1x1xi64> -> !torch.vtensor<[1,1],si64>
    %92 = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[1,1],si64> -> tensor<1x1xi64>
    %93 = flow.tensor.transfer %92 : tensor<1x1xi64> to #hal.device.promise<@__device_1>
    %94 = torch_c.from_builtin_tensor %93 : tensor<1x1xi64> -> !torch.vtensor<[1,1],si64>
    %95 = torch_c.to_builtin_tensor %88 : !torch.vtensor<[1,1,1,?],f16> -> tensor<1x1x1x?xf16>
    %c3 = arith.constant 3 : index
    %dim = tensor.dim %95, %c3 : tensor<1x1x1x?xf16>
    %96 = flow.tensor.transfer %95 : tensor<1x1x1x?xf16>{%dim} to #hal.device.promise<@__device_0>
    %97 = torch_c.from_builtin_tensor %96 : tensor<1x1x1x?xf16> -> !torch.vtensor<[1,1,1,?],f16>
    torch.bind_symbolic_shape %97, [%74], affine_map<()[s0] -> (1, 1, 1, s0 * 32)> : !torch.vtensor<[1,1,1,?],f16>
    %98 = torch_c.to_builtin_tensor %88 : !torch.vtensor<[1,1,1,?],f16> -> tensor<1x1x1x?xf16>
    %c3_14 = arith.constant 3 : index
    %dim_15 = tensor.dim %98, %c3_14 : tensor<1x1x1x?xf16>
    %99 = flow.tensor.transfer %98 : tensor<1x1x1x?xf16>{%dim_15} to #hal.device.promise<@__device_1>
    %100 = torch_c.from_builtin_tensor %99 : tensor<1x1x1x?xf16> -> !torch.vtensor<[1,1,1,?],f16>
    torch.bind_symbolic_shape %100, [%74], affine_map<()[s0] -> (1, 1, 1, s0 * 32)> : !torch.vtensor<[1,1,1,?],f16>
    %101 = torch_c.to_builtin_tensor %arg2 : !torch.vtensor<[1],si64> -> tensor<1xi64>
    %102 = flow.tensor.transfer %101 : tensor<1xi64> to #hal.device.promise<@__device_0>
    %103 = torch_c.from_builtin_tensor %102 : tensor<1xi64> -> !torch.vtensor<[1],si64>
    %104 = torch_c.to_builtin_tensor %arg2 : !torch.vtensor<[1],si64> -> tensor<1xi64>
    %105 = flow.tensor.transfer %104 : tensor<1xi64> to #hal.device.promise<@__device_1>
    %106 = torch_c.from_builtin_tensor %105 : tensor<1xi64> -> !torch.vtensor<[1],si64>
    %107 = torch_c.to_builtin_tensor %arg3 : !torch.vtensor<[1,?],si64> -> tensor<1x?xi64>
    %c1 = arith.constant 1 : index
    %dim_16 = tensor.dim %107, %c1 : tensor<1x?xi64>
    %108 = flow.tensor.transfer %107 : tensor<1x?xi64>{%dim_16} to #hal.device.promise<@__device_0>
    %109 = torch_c.from_builtin_tensor %108 : tensor<1x?xi64> -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %109, [%74], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %110 = torch_c.to_builtin_tensor %arg3 : !torch.vtensor<[1,?],si64> -> tensor<1x?xi64>
    %c1_17 = arith.constant 1 : index
    %dim_18 = tensor.dim %110, %c1_17 : tensor<1x?xi64>
    %111 = flow.tensor.transfer %110 : tensor<1x?xi64>{%dim_18} to #hal.device.promise<@__device_1>
    %112 = torch_c.from_builtin_tensor %111 : tensor<1x?xi64> -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %112, [%74], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %int0_19 = torch.constant.int 0
    %int1_20 = torch.constant.int 1
    %none_21 = torch.constant.none
    %none_22 = torch.constant.none
    %cpu_23 = torch.constant.device "cpu"
    %false_24 = torch.constant.bool false
    %113 = torch.aten.arange.start %int0_19, %int1_20, %none_21, %none_22, %cpu_23, %false_24 : !torch.int, !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[1],si64>
    %int0_25 = torch.constant.int 0
    %114 = torch.aten.unsqueeze %113, %int0_25 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_26 = torch.constant.int 1
    %115 = torch.aten.unsqueeze %103, %int1_26 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_27 = torch.constant.int 1
    %116 = torch.aten.unsqueeze %106, %int1_27 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %117 = torch_c.to_builtin_tensor %114 : !torch.vtensor<[1,1],si64> -> tensor<1x1xi64>
    %118 = flow.tensor.transfer %117 : tensor<1x1xi64> to #hal.device.promise<@__device_0>
    %119 = torch_c.from_builtin_tensor %118 : tensor<1x1xi64> -> !torch.vtensor<[1,1],si64>
    %120 = torch_c.to_builtin_tensor %114 : !torch.vtensor<[1,1],si64> -> tensor<1x1xi64>
    %121 = flow.tensor.transfer %120 : tensor<1x1xi64> to #hal.device.promise<@__device_1>
    %122 = torch_c.from_builtin_tensor %121 : tensor<1x1xi64> -> !torch.vtensor<[1,1],si64>
    %int1_28 = torch.constant.int 1
    %123 = torch.aten.add.Tensor %115, %119, %int1_28 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_29 = torch.constant.int 1
    %124 = torch.aten.add.Tensor %116, %122, %int1_29 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int128 = torch.constant.int 128
    %none_30 = torch.constant.none
    %none_31 = torch.constant.none
    %cpu_32 = torch.constant.device "cpu"
    %false_33 = torch.constant.bool false
    %125 = torch.aten.arange %int128, %none_30, %none_31, %cpu_32, %false_33 : !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[128],si64>
    %int0_34 = torch.constant.int 0
    %int32_35 = torch.constant.int 32
    %none_36 = torch.constant.none
    %none_37 = torch.constant.none
    %cpu_38 = torch.constant.device "cpu"
    %false_39 = torch.constant.bool false
    %126 = torch.aten.arange.start %int0_34, %int32_35, %none_36, %none_37, %cpu_38, %false_39 : !torch.int, !torch.int, !torch.none, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[32],si64>
    %int2 = torch.constant.int 2
    %127 = torch.aten.floor_divide.Scalar %126, %int2 : !torch.vtensor<[32],si64>, !torch.int -> !torch.vtensor<[32],si64>
    %int6_40 = torch.constant.int 6
    %128 = torch.prims.convert_element_type %127, %int6_40 : !torch.vtensor<[32],si64>, !torch.int -> !torch.vtensor<[32],f32>
    %int32_41 = torch.constant.int 32
    %129 = torch.aten.div.Scalar %128, %int32_41 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[32],f32>
    %float2.000000e00 = torch.constant.float 2.000000e+00
    %130 = torch.aten.mul.Scalar %129, %float2.000000e00 : !torch.vtensor<[32],f32>, !torch.float -> !torch.vtensor<[32],f32>
    %float5.000000e05 = torch.constant.float 5.000000e+05
    %131 = torch.aten.pow.Scalar %float5.000000e05, %130 : !torch.float, !torch.vtensor<[32],f32> -> !torch.vtensor<[32],f32>
    %132 = torch.aten.reciprocal %131 : !torch.vtensor<[32],f32> -> !torch.vtensor<[32],f32>
    %float1.000000e00 = torch.constant.float 1.000000e+00
    %133 = torch.aten.mul.Scalar %132, %float1.000000e00 : !torch.vtensor<[32],f32>, !torch.float -> !torch.vtensor<[32],f32>
    %int1_42 = torch.constant.int 1
    %134 = torch.aten.unsqueeze %125, %int1_42 : !torch.vtensor<[128],si64>, !torch.int -> !torch.vtensor<[128,1],si64>
    %int0_43 = torch.constant.int 0
    %135 = torch.aten.unsqueeze %133, %int0_43 : !torch.vtensor<[32],f32>, !torch.int -> !torch.vtensor<[1,32],f32>
    %136 = torch.aten.mul.Tensor %134, %135 : !torch.vtensor<[128,1],si64>, !torch.vtensor<[1,32],f32> -> !torch.vtensor<[128,32],f32>
    %int6_44 = torch.constant.int 6
    %137 = torch.prims.convert_element_type %136, %int6_44 : !torch.vtensor<[128,32],f32>, !torch.int -> !torch.vtensor<[128,32],f32>
    %138 = torch_c.to_builtin_tensor %137 : !torch.vtensor<[128,32],f32> -> tensor<128x32xf32>
    %139 = flow.tensor.transfer %138 : tensor<128x32xf32> to #hal.device.promise<@__device_0>
    %140 = torch_c.from_builtin_tensor %139 : tensor<128x32xf32> -> !torch.vtensor<[128,32],f32>
    %141 = torch_c.to_builtin_tensor %137 : !torch.vtensor<[128,32],f32> -> tensor<128x32xf32>
    %142 = flow.tensor.transfer %141 : tensor<128x32xf32> to #hal.device.promise<@__device_1>
    %143 = torch_c.from_builtin_tensor %142 : tensor<128x32xf32> -> !torch.vtensor<[128,32],f32>
    %int1_45 = torch.constant.int 1
    %144 = torch.prim.ListConstruct %int1_45 : (!torch.int) -> !torch.list<int>
    %145 = torch.aten.view %123, %144 : !torch.vtensor<[1,1],si64>, !torch.list<int> -> !torch.vtensor<[1],si64>
    %int1_46 = torch.constant.int 1
    %146 = torch.prim.ListConstruct %int1_46 : (!torch.int) -> !torch.list<int>
    %147 = torch.aten.view %124, %146 : !torch.vtensor<[1,1],si64>, !torch.list<int> -> !torch.vtensor<[1],si64>
    %148 = torch.prim.ListConstruct %145 : (!torch.vtensor<[1],si64>) -> !torch.list<optional<vtensor>>
    %149 = torch.aten.index.Tensor %140, %148 : !torch.vtensor<[128,32],f32>, !torch.list<optional<vtensor>> -> !torch.vtensor<[1,32],f32>
    %150 = torch.prim.ListConstruct %147 : (!torch.vtensor<[1],si64>) -> !torch.list<optional<vtensor>>
    %151 = torch.aten.index.Tensor %143, %150 : !torch.vtensor<[128,32],f32>, !torch.list<optional<vtensor>> -> !torch.vtensor<[1,32],f32>
    %int1_47 = torch.constant.int 1
    %152 = torch.aten.unsqueeze %149, %int1_47 : !torch.vtensor<[1,32],f32>, !torch.int -> !torch.vtensor<[1,1,32],f32>
    %int1_48 = torch.constant.int 1
    %153 = torch.aten.unsqueeze %151, %int1_48 : !torch.vtensor<[1,32],f32>, !torch.int -> !torch.vtensor<[1,1,32],f32>
    %int5_49 = torch.constant.int 5
    %154 = torch.prims.convert_element_type %0, %int5_49 : !torch.vtensor<[256,256],f32>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int-1_50 = torch.constant.int -1
    %false_51 = torch.constant.bool false
    %false_52 = torch.constant.bool false
    %155 = torch.aten.embedding %154, %91, %int-1_50, %false_51, %false_52 : !torch.vtensor<[256,256],f16>, !torch.vtensor<[1,1],si64>, !torch.int, !torch.bool, !torch.bool -> !torch.vtensor<[1,1,256],f16>
    %int5_53 = torch.constant.int 5
    %156 = torch.prims.convert_element_type %1, %int5_53 : !torch.vtensor<[256,256],f32>, !torch.int -> !torch.vtensor<[256,256],f16>
    %int-1_54 = torch.constant.int -1
    %false_55 = torch.constant.bool false
    %false_56 = torch.constant.bool false
    %157 = torch.aten.embedding %156, %94, %int-1_54, %false_55, %false_56 : !torch.vtensor<[256,256],f16>, !torch.vtensor<[1,1],si64>, !torch.int, !torch.bool, !torch.bool -> !torch.vtensor<[1,1,256],f16>
    %int6_57 = torch.constant.int 6
    %158 = torch.prims.convert_element_type %155, %int6_57 : !torch.vtensor<[1,1,256],f16>, !torch.int -> !torch.vtensor<[1,1,256],f32>
    %int6_58 = torch.constant.int 6
    %159 = torch.prims.convert_element_type %157, %int6_58 : !torch.vtensor<[1,1,256],f16>, !torch.int -> !torch.vtensor<[1,1,256],f32>
    %int2_59 = torch.constant.int 2
    %160 = torch.aten.pow.Tensor_Scalar %158, %int2_59 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f32>
    %int2_60 = torch.constant.int 2
    %161 = torch.aten.pow.Tensor_Scalar %159, %int2_60 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f32>
    %int-1_61 = torch.constant.int -1
    %162 = torch.prim.ListConstruct %int-1_61 : (!torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %none_62 = torch.constant.none
    %163 = torch.aten.mean.dim %160, %162, %true, %none_62 : !torch.vtensor<[1,1,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1],f32>
    %int-1_63 = torch.constant.int -1
    %164 = torch.prim.ListConstruct %int-1_63 : (!torch.int) -> !torch.list<int>
    %true_64 = torch.constant.bool true
    %none_65 = torch.constant.none
    %165 = torch.aten.mean.dim %161, %164, %true_64, %none_65 : !torch.vtensor<[1,1,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1],f32>
    %float1.000000e-02 = torch.constant.float 1.000000e-02
    %int1_66 = torch.constant.int 1
    %166 = torch.aten.add.Scalar %163, %float1.000000e-02, %int1_66 : !torch.vtensor<[1,1,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,1,1],f32>
    %float1.000000e-02_67 = torch.constant.float 1.000000e-02
    %int1_68 = torch.constant.int 1
    %167 = torch.aten.add.Scalar %165, %float1.000000e-02_67, %int1_68 : !torch.vtensor<[1,1,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,1,1],f32>
    %168 = torch.aten.rsqrt %166 : !torch.vtensor<[1,1,1],f32> -> !torch.vtensor<[1,1,1],f32>
    %169 = torch.aten.rsqrt %167 : !torch.vtensor<[1,1,1],f32> -> !torch.vtensor<[1,1,1],f32>
    %170 = torch.aten.mul.Tensor %158, %168 : !torch.vtensor<[1,1,256],f32>, !torch.vtensor<[1,1,1],f32> -> !torch.vtensor<[1,1,256],f32>
    %171 = torch.aten.mul.Tensor %159, %169 : !torch.vtensor<[1,1,256],f32>, !torch.vtensor<[1,1,1],f32> -> !torch.vtensor<[1,1,256],f32>
    %int5_69 = torch.constant.int 5
    %172 = torch.prims.convert_element_type %170, %int5_69 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %int5_70 = torch.constant.int 5
    %173 = torch.prims.convert_element_type %171, %int5_70 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %174 = torch.aten.mul.Tensor %2, %172 : !torch.vtensor<[256],f32>, !torch.vtensor<[1,1,256],f16> -> !torch.vtensor<[1,1,256],f32>
    %175 = torch.aten.mul.Tensor %3, %173 : !torch.vtensor<[256],f32>, !torch.vtensor<[1,1,256],f16> -> !torch.vtensor<[1,1,256],f32>
    %int5_71 = torch.constant.int 5
    %176 = torch.prims.convert_element_type %174, %int5_71 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %int5_72 = torch.constant.int 5
    %177 = torch.prims.convert_element_type %175, %int5_72 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %int1_73 = torch.constant.int 1
    %int0_74 = torch.constant.int 0
    %178 = torch.prim.ListConstruct %int1_73, %int0_74 : (!torch.int, !torch.int) -> !torch.list<int>
    %179 = torch.aten.permute %4, %178 : !torch.vtensor<[128,256],f32>, !torch.list<int> -> !torch.vtensor<[256,128],f32>
    %int1_75 = torch.constant.int 1
    %int0_76 = torch.constant.int 0
    %180 = torch.prim.ListConstruct %int1_75, %int0_76 : (!torch.int, !torch.int) -> !torch.list<int>
    %181 = torch.aten.permute %5, %180 : !torch.vtensor<[128,256],f32>, !torch.list<int> -> !torch.vtensor<[256,128],f32>
    %int5_77 = torch.constant.int 5
    %182 = torch.prims.convert_element_type %179, %int5_77 : !torch.vtensor<[256,128],f32>, !torch.int -> !torch.vtensor<[256,128],f16>
    %int1_78 = torch.constant.int 1
    %int256 = torch.constant.int 256
    %183 = torch.prim.ListConstruct %int1_78, %int256 : (!torch.int, !torch.int) -> !torch.list<int>
    %184 = torch.aten.view %176, %183 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %185 = torch.aten.mm %184, %182 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,128],f16> -> !torch.vtensor<[1,128],f16>
    %int1_79 = torch.constant.int 1
    %int1_80 = torch.constant.int 1
    %int128_81 = torch.constant.int 128
    %186 = torch.prim.ListConstruct %int1_79, %int1_80, %int128_81 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %187 = torch.aten.view %185, %186 : !torch.vtensor<[1,128],f16>, !torch.list<int> -> !torch.vtensor<[1,1,128],f16>
    %int5_82 = torch.constant.int 5
    %188 = torch.prims.convert_element_type %181, %int5_82 : !torch.vtensor<[256,128],f32>, !torch.int -> !torch.vtensor<[256,128],f16>
    %int1_83 = torch.constant.int 1
    %int256_84 = torch.constant.int 256
    %189 = torch.prim.ListConstruct %int1_83, %int256_84 : (!torch.int, !torch.int) -> !torch.list<int>
    %190 = torch.aten.view %177, %189 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %191 = torch.aten.mm %190, %188 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,128],f16> -> !torch.vtensor<[1,128],f16>
    %int1_85 = torch.constant.int 1
    %int1_86 = torch.constant.int 1
    %int128_87 = torch.constant.int 128
    %192 = torch.prim.ListConstruct %int1_85, %int1_86, %int128_87 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %193 = torch.aten.view %191, %192 : !torch.vtensor<[1,128],f16>, !torch.list<int> -> !torch.vtensor<[1,1,128],f16>
    %int1_88 = torch.constant.int 1
    %int0_89 = torch.constant.int 0
    %194 = torch.prim.ListConstruct %int1_88, %int0_89 : (!torch.int, !torch.int) -> !torch.list<int>
    %195 = torch.aten.permute %6, %194 : !torch.vtensor<[64,256],f32>, !torch.list<int> -> !torch.vtensor<[256,64],f32>
    %int1_90 = torch.constant.int 1
    %int0_91 = torch.constant.int 0
    %196 = torch.prim.ListConstruct %int1_90, %int0_91 : (!torch.int, !torch.int) -> !torch.list<int>
    %197 = torch.aten.permute %7, %196 : !torch.vtensor<[64,256],f32>, !torch.list<int> -> !torch.vtensor<[256,64],f32>
    %int5_92 = torch.constant.int 5
    %198 = torch.prims.convert_element_type %195, %int5_92 : !torch.vtensor<[256,64],f32>, !torch.int -> !torch.vtensor<[256,64],f16>
    %int1_93 = torch.constant.int 1
    %int256_94 = torch.constant.int 256
    %199 = torch.prim.ListConstruct %int1_93, %int256_94 : (!torch.int, !torch.int) -> !torch.list<int>
    %200 = torch.aten.view %176, %199 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %201 = torch.aten.mm %200, %198 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,64],f16> -> !torch.vtensor<[1,64],f16>
    %int1_95 = torch.constant.int 1
    %int1_96 = torch.constant.int 1
    %int64 = torch.constant.int 64
    %202 = torch.prim.ListConstruct %int1_95, %int1_96, %int64 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %203 = torch.aten.view %201, %202 : !torch.vtensor<[1,64],f16>, !torch.list<int> -> !torch.vtensor<[1,1,64],f16>
    %int5_97 = torch.constant.int 5
    %204 = torch.prims.convert_element_type %197, %int5_97 : !torch.vtensor<[256,64],f32>, !torch.int -> !torch.vtensor<[256,64],f16>
    %int1_98 = torch.constant.int 1
    %int256_99 = torch.constant.int 256
    %205 = torch.prim.ListConstruct %int1_98, %int256_99 : (!torch.int, !torch.int) -> !torch.list<int>
    %206 = torch.aten.view %177, %205 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %207 = torch.aten.mm %206, %204 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,64],f16> -> !torch.vtensor<[1,64],f16>
    %int1_100 = torch.constant.int 1
    %int1_101 = torch.constant.int 1
    %int64_102 = torch.constant.int 64
    %208 = torch.prim.ListConstruct %int1_100, %int1_101, %int64_102 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %209 = torch.aten.view %207, %208 : !torch.vtensor<[1,64],f16>, !torch.list<int> -> !torch.vtensor<[1,1,64],f16>
    %int1_103 = torch.constant.int 1
    %int0_104 = torch.constant.int 0
    %210 = torch.prim.ListConstruct %int1_103, %int0_104 : (!torch.int, !torch.int) -> !torch.list<int>
    %211 = torch.aten.permute %8, %210 : !torch.vtensor<[64,256],f32>, !torch.list<int> -> !torch.vtensor<[256,64],f32>
    %int1_105 = torch.constant.int 1
    %int0_106 = torch.constant.int 0
    %212 = torch.prim.ListConstruct %int1_105, %int0_106 : (!torch.int, !torch.int) -> !torch.list<int>
    %213 = torch.aten.permute %9, %212 : !torch.vtensor<[64,256],f32>, !torch.list<int> -> !torch.vtensor<[256,64],f32>
    %int5_107 = torch.constant.int 5
    %214 = torch.prims.convert_element_type %211, %int5_107 : !torch.vtensor<[256,64],f32>, !torch.int -> !torch.vtensor<[256,64],f16>
    %int1_108 = torch.constant.int 1
    %int256_109 = torch.constant.int 256
    %215 = torch.prim.ListConstruct %int1_108, %int256_109 : (!torch.int, !torch.int) -> !torch.list<int>
    %216 = torch.aten.view %176, %215 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %217 = torch.aten.mm %216, %214 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,64],f16> -> !torch.vtensor<[1,64],f16>
    %int1_110 = torch.constant.int 1
    %int1_111 = torch.constant.int 1
    %int64_112 = torch.constant.int 64
    %218 = torch.prim.ListConstruct %int1_110, %int1_111, %int64_112 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %219 = torch.aten.view %217, %218 : !torch.vtensor<[1,64],f16>, !torch.list<int> -> !torch.vtensor<[1,1,64],f16>
    %int5_113 = torch.constant.int 5
    %220 = torch.prims.convert_element_type %213, %int5_113 : !torch.vtensor<[256,64],f32>, !torch.int -> !torch.vtensor<[256,64],f16>
    %int1_114 = torch.constant.int 1
    %int256_115 = torch.constant.int 256
    %221 = torch.prim.ListConstruct %int1_114, %int256_115 : (!torch.int, !torch.int) -> !torch.list<int>
    %222 = torch.aten.view %177, %221 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %223 = torch.aten.mm %222, %220 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,64],f16> -> !torch.vtensor<[1,64],f16>
    %int1_116 = torch.constant.int 1
    %int1_117 = torch.constant.int 1
    %int64_118 = torch.constant.int 64
    %224 = torch.prim.ListConstruct %int1_116, %int1_117, %int64_118 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %225 = torch.aten.view %223, %224 : !torch.vtensor<[1,64],f16>, !torch.list<int> -> !torch.vtensor<[1,1,64],f16>
    %int1_119 = torch.constant.int 1
    %int1_120 = torch.constant.int 1
    %int4 = torch.constant.int 4
    %int32_121 = torch.constant.int 32
    %226 = torch.prim.ListConstruct %int1_119, %int1_120, %int4, %int32_121 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %227 = torch.aten.view %187, %226 : !torch.vtensor<[1,1,128],f16>, !torch.list<int> -> !torch.vtensor<[1,1,4,32],f16>
    %int1_122 = torch.constant.int 1
    %int1_123 = torch.constant.int 1
    %int4_124 = torch.constant.int 4
    %int32_125 = torch.constant.int 32
    %228 = torch.prim.ListConstruct %int1_122, %int1_123, %int4_124, %int32_125 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %229 = torch.aten.view %193, %228 : !torch.vtensor<[1,1,128],f16>, !torch.list<int> -> !torch.vtensor<[1,1,4,32],f16>
    %int1_126 = torch.constant.int 1
    %int1_127 = torch.constant.int 1
    %int2_128 = torch.constant.int 2
    %int32_129 = torch.constant.int 32
    %230 = torch.prim.ListConstruct %int1_126, %int1_127, %int2_128, %int32_129 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %231 = torch.aten.view %203, %230 : !torch.vtensor<[1,1,64],f16>, !torch.list<int> -> !torch.vtensor<[1,1,2,32],f16>
    %int1_130 = torch.constant.int 1
    %int1_131 = torch.constant.int 1
    %int2_132 = torch.constant.int 2
    %int32_133 = torch.constant.int 32
    %232 = torch.prim.ListConstruct %int1_130, %int1_131, %int2_132, %int32_133 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %233 = torch.aten.view %209, %232 : !torch.vtensor<[1,1,64],f16>, !torch.list<int> -> !torch.vtensor<[1,1,2,32],f16>
    %int1_134 = torch.constant.int 1
    %int1_135 = torch.constant.int 1
    %int2_136 = torch.constant.int 2
    %int32_137 = torch.constant.int 32
    %234 = torch.prim.ListConstruct %int1_134, %int1_135, %int2_136, %int32_137 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %235 = torch.aten.view %219, %234 : !torch.vtensor<[1,1,64],f16>, !torch.list<int> -> !torch.vtensor<[1,1,2,32],f16>
    %int1_138 = torch.constant.int 1
    %int1_139 = torch.constant.int 1
    %int2_140 = torch.constant.int 2
    %int32_141 = torch.constant.int 32
    %236 = torch.prim.ListConstruct %int1_138, %int1_139, %int2_140, %int32_141 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %237 = torch.aten.view %225, %236 : !torch.vtensor<[1,1,64],f16>, !torch.list<int> -> !torch.vtensor<[1,1,2,32],f16>
    %int6_142 = torch.constant.int 6
    %238 = torch.prims.convert_element_type %227, %int6_142 : !torch.vtensor<[1,1,4,32],f16>, !torch.int -> !torch.vtensor<[1,1,4,32],f32>
    %239 = torch_c.to_builtin_tensor %238 : !torch.vtensor<[1,1,4,32],f32> -> tensor<1x1x4x32xf32>
    %240 = torch_c.to_builtin_tensor %152 : !torch.vtensor<[1,1,32],f32> -> tensor<1x1x32xf32>
    %241 = util.call @sharktank_rotary_embedding_1_1_4_32_f32(%239, %240) : (tensor<1x1x4x32xf32>, tensor<1x1x32xf32>) -> tensor<1x1x4x32xf32>
    %242 = torch_c.from_builtin_tensor %241 : tensor<1x1x4x32xf32> -> !torch.vtensor<[1,1,4,32],f32>
    %int5_143 = torch.constant.int 5
    %243 = torch.prims.convert_element_type %242, %int5_143 : !torch.vtensor<[1,1,4,32],f32>, !torch.int -> !torch.vtensor<[1,1,4,32],f16>
    %int6_144 = torch.constant.int 6
    %244 = torch.prims.convert_element_type %229, %int6_144 : !torch.vtensor<[1,1,4,32],f16>, !torch.int -> !torch.vtensor<[1,1,4,32],f32>
    %245 = torch_c.to_builtin_tensor %244 : !torch.vtensor<[1,1,4,32],f32> -> tensor<1x1x4x32xf32>
    %246 = torch_c.to_builtin_tensor %153 : !torch.vtensor<[1,1,32],f32> -> tensor<1x1x32xf32>
    %247 = util.call @sharktank_rotary_embedding_1_1_4_32_f32(%245, %246) : (tensor<1x1x4x32xf32>, tensor<1x1x32xf32>) -> tensor<1x1x4x32xf32>
    %248 = torch_c.from_builtin_tensor %247 : tensor<1x1x4x32xf32> -> !torch.vtensor<[1,1,4,32],f32>
    %int5_145 = torch.constant.int 5
    %249 = torch.prims.convert_element_type %248, %int5_145 : !torch.vtensor<[1,1,4,32],f32>, !torch.int -> !torch.vtensor<[1,1,4,32],f16>
    %int6_146 = torch.constant.int 6
    %250 = torch.prims.convert_element_type %231, %int6_146 : !torch.vtensor<[1,1,2,32],f16>, !torch.int -> !torch.vtensor<[1,1,2,32],f32>
    %251 = torch_c.to_builtin_tensor %250 : !torch.vtensor<[1,1,2,32],f32> -> tensor<1x1x2x32xf32>
    %252 = torch_c.to_builtin_tensor %152 : !torch.vtensor<[1,1,32],f32> -> tensor<1x1x32xf32>
    %253 = util.call @sharktank_rotary_embedding_1_1_2_32_f32(%251, %252) : (tensor<1x1x2x32xf32>, tensor<1x1x32xf32>) -> tensor<1x1x2x32xf32>
    %254 = torch_c.from_builtin_tensor %253 : tensor<1x1x2x32xf32> -> !torch.vtensor<[1,1,2,32],f32>
    %int5_147 = torch.constant.int 5
    %255 = torch.prims.convert_element_type %254, %int5_147 : !torch.vtensor<[1,1,2,32],f32>, !torch.int -> !torch.vtensor<[1,1,2,32],f16>
    %int6_148 = torch.constant.int 6
    %256 = torch.prims.convert_element_type %233, %int6_148 : !torch.vtensor<[1,1,2,32],f16>, !torch.int -> !torch.vtensor<[1,1,2,32],f32>
    %257 = torch_c.to_builtin_tensor %256 : !torch.vtensor<[1,1,2,32],f32> -> tensor<1x1x2x32xf32>
    %258 = torch_c.to_builtin_tensor %153 : !torch.vtensor<[1,1,32],f32> -> tensor<1x1x32xf32>
    %259 = util.call @sharktank_rotary_embedding_1_1_2_32_f32(%257, %258) : (tensor<1x1x2x32xf32>, tensor<1x1x32xf32>) -> tensor<1x1x2x32xf32>
    %260 = torch_c.from_builtin_tensor %259 : tensor<1x1x2x32xf32> -> !torch.vtensor<[1,1,2,32],f32>
    %int5_149 = torch.constant.int 5
    %261 = torch.prims.convert_element_type %260, %int5_149 : !torch.vtensor<[1,1,2,32],f32>, !torch.int -> !torch.vtensor<[1,1,2,32],f16>
    %int3 = torch.constant.int 3
    %int2_150 = torch.constant.int 2
    %int32_151 = torch.constant.int 32
    %int2_152 = torch.constant.int 2
    %int32_153 = torch.constant.int 32
    %262 = torch.prim.ListConstruct %77, %int3, %int2_150, %int32_151, %int2_152, %int32_153 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %263 = torch.aten.view %72, %262 : !torch.vtensor<[?,12288],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,2,32],f16>
    torch.bind_symbolic_shape %263, [%75], affine_map<()[s0] -> (s0, 3, 2, 32, 2, 32)> : !torch.vtensor<[?,3,2,32,2,32],f16>
    %int3_154 = torch.constant.int 3
    %int2_155 = torch.constant.int 2
    %int32_156 = torch.constant.int 32
    %int2_157 = torch.constant.int 2
    %int32_158 = torch.constant.int 32
    %264 = torch.prim.ListConstruct %77, %int3_154, %int2_155, %int32_156, %int2_157, %int32_158 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %265 = torch.aten.view %73, %264 : !torch.vtensor<[?,12288],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,2,32],f16>
    torch.bind_symbolic_shape %265, [%75], affine_map<()[s0] -> (s0, 3, 2, 32, 2, 32)> : !torch.vtensor<[?,3,2,32,2,32],f16>
    %int3_159 = torch.constant.int 3
    %266 = torch.aten.mul.int %77, %int3_159 : !torch.int, !torch.int -> !torch.int
    %int2_160 = torch.constant.int 2
    %267 = torch.aten.mul.int %266, %int2_160 : !torch.int, !torch.int -> !torch.int
    %int32_161 = torch.constant.int 32
    %268 = torch.aten.mul.int %267, %int32_161 : !torch.int, !torch.int -> !torch.int
    %int2_162 = torch.constant.int 2
    %int32_163 = torch.constant.int 32
    %269 = torch.prim.ListConstruct %268, %int2_162, %int32_163 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %270 = torch.aten.view %263, %269 : !torch.vtensor<[?,3,2,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,2,32],f16>
    torch.bind_symbolic_shape %270, [%75], affine_map<()[s0] -> (s0 * 192, 2, 32)> : !torch.vtensor<[?,2,32],f16>
    %int2_164 = torch.constant.int 2
    %int32_165 = torch.constant.int 32
    %271 = torch.prim.ListConstruct %268, %int2_164, %int32_165 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %272 = torch.aten.view %265, %271 : !torch.vtensor<[?,3,2,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,2,32],f16>
    torch.bind_symbolic_shape %272, [%75], affine_map<()[s0] -> (s0 * 192, 2, 32)> : !torch.vtensor<[?,2,32],f16>
    %int32_166 = torch.constant.int 32
    %273 = torch.aten.floor_divide.Scalar %103, %int32_166 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int32_167 = torch.constant.int 32
    %274 = torch.aten.floor_divide.Scalar %106, %int32_167 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int1_168 = torch.constant.int 1
    %275 = torch.aten.unsqueeze %273, %int1_168 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_169 = torch.constant.int 1
    %276 = torch.aten.unsqueeze %274, %int1_169 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_170 = torch.constant.int 1
    %false_171 = torch.constant.bool false
    %277 = torch.aten.gather %109, %int1_170, %275, %false_171 : !torch.vtensor<[1,?],si64>, !torch.int, !torch.vtensor<[1,1],si64>, !torch.bool -> !torch.vtensor<[1,1],si64>
    %int1_172 = torch.constant.int 1
    %false_173 = torch.constant.bool false
    %278 = torch.aten.gather %112, %int1_172, %276, %false_173 : !torch.vtensor<[1,?],si64>, !torch.int, !torch.vtensor<[1,1],si64>, !torch.bool -> !torch.vtensor<[1,1],si64>
    %int32_174 = torch.constant.int 32
    %279 = torch.aten.remainder.Scalar %103, %int32_174 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int32_175 = torch.constant.int 32
    %280 = torch.aten.remainder.Scalar %106, %int32_175 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int1_176 = torch.constant.int 1
    %281 = torch.aten.unsqueeze %279, %int1_176 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_177 = torch.constant.int 1
    %282 = torch.aten.unsqueeze %280, %int1_177 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %none_178 = torch.constant.none
    %283 = torch.aten.clone %10, %none_178 : !torch.vtensor<[],si64>, !torch.none -> !torch.vtensor<[],si64>
    %284 = torch.aten.detach %283 : !torch.vtensor<[],si64> -> !torch.vtensor<[],si64>
    %285 = torch.aten.detach %284 : !torch.vtensor<[],si64> -> !torch.vtensor<[],si64>
    %286 = torch.aten.detach %285 : !torch.vtensor<[],si64> -> !torch.vtensor<[],si64>
    %int0_179 = torch.constant.int 0
    %287 = torch.aten.unsqueeze %286, %int0_179 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %none_180 = torch.constant.none
    %288 = torch.aten.clone %11, %none_180 : !torch.vtensor<[],si64>, !torch.none -> !torch.vtensor<[],si64>
    %289 = torch.aten.detach %288 : !torch.vtensor<[],si64> -> !torch.vtensor<[],si64>
    %290 = torch.aten.detach %289 : !torch.vtensor<[],si64> -> !torch.vtensor<[],si64>
    %291 = torch.aten.detach %290 : !torch.vtensor<[],si64> -> !torch.vtensor<[],si64>
    %int0_181 = torch.constant.int 0
    %292 = torch.aten.unsqueeze %291, %int0_181 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int1_182 = torch.constant.int 1
    %int1_183 = torch.constant.int 1
    %293 = torch.prim.ListConstruct %int1_182, %int1_183 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_184 = torch.constant.int 1
    %int1_185 = torch.constant.int 1
    %294 = torch.prim.ListConstruct %int1_184, %int1_185 : (!torch.int, !torch.int) -> !torch.list<int>
    %int4_186 = torch.constant.int 4
    %int0_187 = torch.constant.int 0
    %cpu_188 = torch.constant.device "cpu"
    %false_189 = torch.constant.bool false
    %295 = torch.aten.empty_strided %293, %294, %int4_186, %int0_187, %cpu_188, %false_189 : !torch.list<int>, !torch.list<int>, !torch.int, !torch.int, !torch.Device, !torch.bool -> !torch.vtensor<[1,1],si64>
    %int0_190 = torch.constant.int 0
    %296 = torch.aten.fill.Scalar %295, %int0_190 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_191 = torch.constant.int 1
    %int1_192 = torch.constant.int 1
    %297 = torch.prim.ListConstruct %int1_191, %int1_192 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_193 = torch.constant.int 1
    %int1_194 = torch.constant.int 1
    %298 = torch.prim.ListConstruct %int1_193, %int1_194 : (!torch.int, !torch.int) -> !torch.list<int>
    %int4_195 = torch.constant.int 4
    %int0_196 = torch.constant.int 0
    %cpu_197 = torch.constant.device "cpu"
    %false_198 = torch.constant.bool false
    %299 = torch.aten.empty_strided %297, %298, %int4_195, %int0_196, %cpu_197, %false_198 : !torch.list<int>, !torch.list<int>, !torch.int, !torch.int, !torch.Device, !torch.bool -> !torch.vtensor<[1,1],si64>
    %int0_199 = torch.constant.int 0
    %300 = torch.aten.fill.Scalar %299, %int0_199 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_200 = torch.constant.int 1
    %int1_201 = torch.constant.int 1
    %301 = torch.prim.ListConstruct %int1_200, %int1_201 : (!torch.int, !torch.int) -> !torch.list<int>
    %302 = torch.aten.repeat %287, %301 : !torch.vtensor<[1],si64>, !torch.list<int> -> !torch.vtensor<[1,1],si64>
    %int1_202 = torch.constant.int 1
    %int1_203 = torch.constant.int 1
    %303 = torch.prim.ListConstruct %int1_202, %int1_203 : (!torch.int, !torch.int) -> !torch.list<int>
    %304 = torch.aten.repeat %292, %303 : !torch.vtensor<[1],si64>, !torch.list<int> -> !torch.vtensor<[1,1],si64>
    %int3_204 = torch.constant.int 3
    %305 = torch.aten.mul.Scalar %277, %int3_204 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int3_205 = torch.constant.int 3
    %306 = torch.aten.mul.Scalar %278, %int3_205 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_206 = torch.constant.int 1
    %307 = torch.aten.add.Tensor %305, %296, %int1_206 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_207 = torch.constant.int 1
    %308 = torch.aten.add.Tensor %306, %300, %int1_207 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int2_208 = torch.constant.int 2
    %309 = torch.aten.mul.Scalar %307, %int2_208 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int2_209 = torch.constant.int 2
    %310 = torch.aten.mul.Scalar %308, %int2_209 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_210 = torch.constant.int 1
    %311 = torch.aten.add.Tensor %309, %302, %int1_210 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_211 = torch.constant.int 1
    %312 = torch.aten.add.Tensor %310, %304, %int1_211 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int32_212 = torch.constant.int 32
    %313 = torch.aten.mul.Scalar %311, %int32_212 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int32_213 = torch.constant.int 32
    %314 = torch.aten.mul.Scalar %312, %int32_213 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_214 = torch.constant.int 1
    %315 = torch.aten.add.Tensor %313, %281, %int1_214 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_215 = torch.constant.int 1
    %316 = torch.aten.add.Tensor %314, %282, %int1_215 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int5_216 = torch.constant.int 5
    %317 = torch.prims.convert_element_type %255, %int5_216 : !torch.vtensor<[1,1,2,32],f16>, !torch.int -> !torch.vtensor<[1,1,2,32],f16>
    %int5_217 = torch.constant.int 5
    %318 = torch.prims.convert_element_type %261, %int5_217 : !torch.vtensor<[1,1,2,32],f16>, !torch.int -> !torch.vtensor<[1,1,2,32],f16>
    %319 = torch.prim.ListConstruct %315 : (!torch.vtensor<[1,1],si64>) -> !torch.list<optional<vtensor>>
    %false_218 = torch.constant.bool false
    %320 = torch.aten.index_put %270, %319, %317, %false_218 : !torch.vtensor<[?,2,32],f16>, !torch.list<optional<vtensor>>, !torch.vtensor<[1,1,2,32],f16>, !torch.bool -> !torch.vtensor<[?,2,32],f16>
    torch.bind_symbolic_shape %320, [%75], affine_map<()[s0] -> (s0 * 192, 2, 32)> : !torch.vtensor<[?,2,32],f16>
    %int3_219 = torch.constant.int 3
    %int2_220 = torch.constant.int 2
    %int32_221 = torch.constant.int 32
    %int2_222 = torch.constant.int 2
    %int32_223 = torch.constant.int 32
    %321 = torch.prim.ListConstruct %77, %int3_219, %int2_220, %int32_221, %int2_222, %int32_223 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %322 = torch.aten.view %320, %321 : !torch.vtensor<[?,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,2,32],f16>
    torch.bind_symbolic_shape %322, [%75], affine_map<()[s0] -> (s0, 3, 2, 32, 2, 32)> : !torch.vtensor<[?,3,2,32,2,32],f16>
    %int12288 = torch.constant.int 12288
    %323 = torch.prim.ListConstruct %77, %int12288 : (!torch.int, !torch.int) -> !torch.list<int>
    %324 = torch.aten.view %322, %323 : !torch.vtensor<[?,3,2,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,12288],f16>
    torch.bind_symbolic_shape %324, [%75], affine_map<()[s0] -> (s0, 12288)> : !torch.vtensor<[?,12288],f16>
    %int3_224 = torch.constant.int 3
    %int2_225 = torch.constant.int 2
    %int32_226 = torch.constant.int 32
    %int2_227 = torch.constant.int 2
    %int32_228 = torch.constant.int 32
    %325 = torch.prim.ListConstruct %77, %int3_224, %int2_225, %int32_226, %int2_227, %int32_228 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %326 = torch.aten.view %324, %325 : !torch.vtensor<[?,12288],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,2,32],f16>
    torch.bind_symbolic_shape %326, [%75], affine_map<()[s0] -> (s0, 3, 2, 32, 2, 32)> : !torch.vtensor<[?,3,2,32,2,32],f16>
    %int2_229 = torch.constant.int 2
    %int32_230 = torch.constant.int 32
    %327 = torch.prim.ListConstruct %268, %int2_229, %int32_230 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %328 = torch.aten.view %326, %327 : !torch.vtensor<[?,3,2,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,2,32],f16>
    torch.bind_symbolic_shape %328, [%75], affine_map<()[s0] -> (s0 * 192, 2, 32)> : !torch.vtensor<[?,2,32],f16>
    %329 = torch.prim.ListConstruct %316 : (!torch.vtensor<[1,1],si64>) -> !torch.list<optional<vtensor>>
    %false_231 = torch.constant.bool false
    %330 = torch.aten.index_put %272, %329, %318, %false_231 : !torch.vtensor<[?,2,32],f16>, !torch.list<optional<vtensor>>, !torch.vtensor<[1,1,2,32],f16>, !torch.bool -> !torch.vtensor<[?,2,32],f16>
    torch.bind_symbolic_shape %330, [%75], affine_map<()[s0] -> (s0 * 192, 2, 32)> : !torch.vtensor<[?,2,32],f16>
    %int3_232 = torch.constant.int 3
    %int2_233 = torch.constant.int 2
    %int32_234 = torch.constant.int 32
    %int2_235 = torch.constant.int 2
    %int32_236 = torch.constant.int 32
    %331 = torch.prim.ListConstruct %77, %int3_232, %int2_233, %int32_234, %int2_235, %int32_236 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %332 = torch.aten.view %330, %331 : !torch.vtensor<[?,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,2,32],f16>
    torch.bind_symbolic_shape %332, [%75], affine_map<()[s0] -> (s0, 3, 2, 32, 2, 32)> : !torch.vtensor<[?,3,2,32,2,32],f16>
    %int12288_237 = torch.constant.int 12288
    %333 = torch.prim.ListConstruct %77, %int12288_237 : (!torch.int, !torch.int) -> !torch.list<int>
    %334 = torch.aten.view %332, %333 : !torch.vtensor<[?,3,2,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,12288],f16>
    torch.bind_symbolic_shape %334, [%75], affine_map<()[s0] -> (s0, 12288)> : !torch.vtensor<[?,12288],f16>
    %int3_238 = torch.constant.int 3
    %int2_239 = torch.constant.int 2
    %int32_240 = torch.constant.int 32
    %int2_241 = torch.constant.int 2
    %int32_242 = torch.constant.int 32
    %335 = torch.prim.ListConstruct %77, %int3_238, %int2_239, %int32_240, %int2_241, %int32_242 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %336 = torch.aten.view %334, %335 : !torch.vtensor<[?,12288],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,2,32],f16>
    torch.bind_symbolic_shape %336, [%75], affine_map<()[s0] -> (s0, 3, 2, 32, 2, 32)> : !torch.vtensor<[?,3,2,32,2,32],f16>
    %int2_243 = torch.constant.int 2
    %int32_244 = torch.constant.int 32
    %337 = torch.prim.ListConstruct %268, %int2_243, %int32_244 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %338 = torch.aten.view %336, %337 : !torch.vtensor<[?,3,2,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,2,32],f16>
    torch.bind_symbolic_shape %338, [%75], affine_map<()[s0] -> (s0 * 192, 2, 32)> : !torch.vtensor<[?,2,32],f16>
    %int32_245 = torch.constant.int 32
    %339 = torch.aten.floor_divide.Scalar %103, %int32_245 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int32_246 = torch.constant.int 32
    %340 = torch.aten.floor_divide.Scalar %106, %int32_246 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int1_247 = torch.constant.int 1
    %341 = torch.aten.unsqueeze %339, %int1_247 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_248 = torch.constant.int 1
    %342 = torch.aten.unsqueeze %340, %int1_248 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_249 = torch.constant.int 1
    %false_250 = torch.constant.bool false
    %343 = torch.aten.gather %109, %int1_249, %341, %false_250 : !torch.vtensor<[1,?],si64>, !torch.int, !torch.vtensor<[1,1],si64>, !torch.bool -> !torch.vtensor<[1,1],si64>
    %int1_251 = torch.constant.int 1
    %false_252 = torch.constant.bool false
    %344 = torch.aten.gather %112, %int1_251, %342, %false_252 : !torch.vtensor<[1,?],si64>, !torch.int, !torch.vtensor<[1,1],si64>, !torch.bool -> !torch.vtensor<[1,1],si64>
    %int32_253 = torch.constant.int 32
    %345 = torch.aten.remainder.Scalar %103, %int32_253 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int32_254 = torch.constant.int 32
    %346 = torch.aten.remainder.Scalar %106, %int32_254 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int1_255 = torch.constant.int 1
    %347 = torch.aten.unsqueeze %345, %int1_255 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_256 = torch.constant.int 1
    %348 = torch.aten.unsqueeze %346, %int1_256 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %none_257 = torch.constant.none
    %349 = torch.aten.clone %12, %none_257 : !torch.vtensor<[],si64>, !torch.none -> !torch.vtensor<[],si64>
    %350 = torch.aten.detach %349 : !torch.vtensor<[],si64> -> !torch.vtensor<[],si64>
    %351 = torch.aten.detach %350 : !torch.vtensor<[],si64> -> !torch.vtensor<[],si64>
    %352 = torch.aten.detach %351 : !torch.vtensor<[],si64> -> !torch.vtensor<[],si64>
    %int0_258 = torch.constant.int 0
    %353 = torch.aten.unsqueeze %352, %int0_258 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %none_259 = torch.constant.none
    %354 = torch.aten.clone %13, %none_259 : !torch.vtensor<[],si64>, !torch.none -> !torch.vtensor<[],si64>
    %355 = torch.aten.detach %354 : !torch.vtensor<[],si64> -> !torch.vtensor<[],si64>
    %356 = torch.aten.detach %355 : !torch.vtensor<[],si64> -> !torch.vtensor<[],si64>
    %357 = torch.aten.detach %356 : !torch.vtensor<[],si64> -> !torch.vtensor<[],si64>
    %int0_260 = torch.constant.int 0
    %358 = torch.aten.unsqueeze %357, %int0_260 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int1_261 = torch.constant.int 1
    %int1_262 = torch.constant.int 1
    %359 = torch.prim.ListConstruct %int1_261, %int1_262 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_263 = torch.constant.int 1
    %int1_264 = torch.constant.int 1
    %360 = torch.prim.ListConstruct %int1_263, %int1_264 : (!torch.int, !torch.int) -> !torch.list<int>
    %int4_265 = torch.constant.int 4
    %int0_266 = torch.constant.int 0
    %cpu_267 = torch.constant.device "cpu"
    %false_268 = torch.constant.bool false
    %361 = torch.aten.empty_strided %359, %360, %int4_265, %int0_266, %cpu_267, %false_268 : !torch.list<int>, !torch.list<int>, !torch.int, !torch.int, !torch.Device, !torch.bool -> !torch.vtensor<[1,1],si64>
    %int0_269 = torch.constant.int 0
    %362 = torch.aten.fill.Scalar %361, %int0_269 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_270 = torch.constant.int 1
    %int1_271 = torch.constant.int 1
    %363 = torch.prim.ListConstruct %int1_270, %int1_271 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_272 = torch.constant.int 1
    %int1_273 = torch.constant.int 1
    %364 = torch.prim.ListConstruct %int1_272, %int1_273 : (!torch.int, !torch.int) -> !torch.list<int>
    %int4_274 = torch.constant.int 4
    %int0_275 = torch.constant.int 0
    %cpu_276 = torch.constant.device "cpu"
    %false_277 = torch.constant.bool false
    %365 = torch.aten.empty_strided %363, %364, %int4_274, %int0_275, %cpu_276, %false_277 : !torch.list<int>, !torch.list<int>, !torch.int, !torch.int, !torch.Device, !torch.bool -> !torch.vtensor<[1,1],si64>
    %int0_278 = torch.constant.int 0
    %366 = torch.aten.fill.Scalar %365, %int0_278 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_279 = torch.constant.int 1
    %int1_280 = torch.constant.int 1
    %367 = torch.prim.ListConstruct %int1_279, %int1_280 : (!torch.int, !torch.int) -> !torch.list<int>
    %368 = torch.aten.repeat %353, %367 : !torch.vtensor<[1],si64>, !torch.list<int> -> !torch.vtensor<[1,1],si64>
    %int1_281 = torch.constant.int 1
    %int1_282 = torch.constant.int 1
    %369 = torch.prim.ListConstruct %int1_281, %int1_282 : (!torch.int, !torch.int) -> !torch.list<int>
    %370 = torch.aten.repeat %358, %369 : !torch.vtensor<[1],si64>, !torch.list<int> -> !torch.vtensor<[1,1],si64>
    %int3_283 = torch.constant.int 3
    %371 = torch.aten.mul.Scalar %343, %int3_283 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int3_284 = torch.constant.int 3
    %372 = torch.aten.mul.Scalar %344, %int3_284 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_285 = torch.constant.int 1
    %373 = torch.aten.add.Tensor %371, %362, %int1_285 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_286 = torch.constant.int 1
    %374 = torch.aten.add.Tensor %372, %366, %int1_286 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int2_287 = torch.constant.int 2
    %375 = torch.aten.mul.Scalar %373, %int2_287 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int2_288 = torch.constant.int 2
    %376 = torch.aten.mul.Scalar %374, %int2_288 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_289 = torch.constant.int 1
    %377 = torch.aten.add.Tensor %375, %368, %int1_289 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_290 = torch.constant.int 1
    %378 = torch.aten.add.Tensor %376, %370, %int1_290 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int32_291 = torch.constant.int 32
    %379 = torch.aten.mul.Scalar %377, %int32_291 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int32_292 = torch.constant.int 32
    %380 = torch.aten.mul.Scalar %378, %int32_292 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_293 = torch.constant.int 1
    %381 = torch.aten.add.Tensor %379, %347, %int1_293 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_294 = torch.constant.int 1
    %382 = torch.aten.add.Tensor %380, %348, %int1_294 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int5_295 = torch.constant.int 5
    %383 = torch.prims.convert_element_type %235, %int5_295 : !torch.vtensor<[1,1,2,32],f16>, !torch.int -> !torch.vtensor<[1,1,2,32],f16>
    %int5_296 = torch.constant.int 5
    %384 = torch.prims.convert_element_type %237, %int5_296 : !torch.vtensor<[1,1,2,32],f16>, !torch.int -> !torch.vtensor<[1,1,2,32],f16>
    %385 = torch.prim.ListConstruct %381 : (!torch.vtensor<[1,1],si64>) -> !torch.list<optional<vtensor>>
    %false_297 = torch.constant.bool false
    %386 = torch.aten.index_put %328, %385, %383, %false_297 : !torch.vtensor<[?,2,32],f16>, !torch.list<optional<vtensor>>, !torch.vtensor<[1,1,2,32],f16>, !torch.bool -> !torch.vtensor<[?,2,32],f16>
    torch.bind_symbolic_shape %386, [%75], affine_map<()[s0] -> (s0 * 192, 2, 32)> : !torch.vtensor<[?,2,32],f16>
    %int3_298 = torch.constant.int 3
    %int2_299 = torch.constant.int 2
    %int32_300 = torch.constant.int 32
    %int2_301 = torch.constant.int 2
    %int32_302 = torch.constant.int 32
    %387 = torch.prim.ListConstruct %77, %int3_298, %int2_299, %int32_300, %int2_301, %int32_302 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %388 = torch.aten.view %386, %387 : !torch.vtensor<[?,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,2,32],f16>
    torch.bind_symbolic_shape %388, [%75], affine_map<()[s0] -> (s0, 3, 2, 32, 2, 32)> : !torch.vtensor<[?,3,2,32,2,32],f16>
    %int12288_303 = torch.constant.int 12288
    %389 = torch.prim.ListConstruct %77, %int12288_303 : (!torch.int, !torch.int) -> !torch.list<int>
    %390 = torch.aten.view %388, %389 : !torch.vtensor<[?,3,2,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,12288],f16>
    torch.bind_symbolic_shape %390, [%75], affine_map<()[s0] -> (s0, 12288)> : !torch.vtensor<[?,12288],f16>
    %391 = torch.prim.ListConstruct %382 : (!torch.vtensor<[1,1],si64>) -> !torch.list<optional<vtensor>>
    %false_304 = torch.constant.bool false
    %392 = torch.aten.index_put %338, %391, %384, %false_304 : !torch.vtensor<[?,2,32],f16>, !torch.list<optional<vtensor>>, !torch.vtensor<[1,1,2,32],f16>, !torch.bool -> !torch.vtensor<[?,2,32],f16>
    torch.bind_symbolic_shape %392, [%75], affine_map<()[s0] -> (s0 * 192, 2, 32)> : !torch.vtensor<[?,2,32],f16>
    %int3_305 = torch.constant.int 3
    %int2_306 = torch.constant.int 2
    %int32_307 = torch.constant.int 32
    %int2_308 = torch.constant.int 2
    %int32_309 = torch.constant.int 32
    %393 = torch.prim.ListConstruct %77, %int3_305, %int2_306, %int32_307, %int2_308, %int32_309 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %394 = torch.aten.view %392, %393 : !torch.vtensor<[?,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,2,32],f16>
    torch.bind_symbolic_shape %394, [%75], affine_map<()[s0] -> (s0, 3, 2, 32, 2, 32)> : !torch.vtensor<[?,3,2,32,2,32],f16>
    %int12288_310 = torch.constant.int 12288
    %395 = torch.prim.ListConstruct %77, %int12288_310 : (!torch.int, !torch.int) -> !torch.list<int>
    %396 = torch.aten.view %394, %395 : !torch.vtensor<[?,3,2,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,12288],f16>
    torch.bind_symbolic_shape %396, [%75], affine_map<()[s0] -> (s0, 12288)> : !torch.vtensor<[?,12288],f16>
    %int1_311 = torch.constant.int 1
    %397 = torch.prim.ListConstruct %int1_311, %76 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_312 = torch.constant.int 1
    %398 = torch.prim.ListConstruct %76, %int1_312 : (!torch.int, !torch.int) -> !torch.list<int>
    %int4_313 = torch.constant.int 4
    %int0_314 = torch.constant.int 0
    %cpu_315 = torch.constant.device "cpu"
    %false_316 = torch.constant.bool false
    %399 = torch.aten.empty_strided %397, %398, %int4_313, %int0_314, %cpu_315, %false_316 : !torch.list<int>, !torch.list<int>, !torch.int, !torch.int, !torch.Device, !torch.bool -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %399, [%74], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %int0_317 = torch.constant.int 0
    %400 = torch.aten.fill.Scalar %399, %int0_317 : !torch.vtensor<[1,?],si64>, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %400, [%74], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %int3_318 = torch.constant.int 3
    %401 = torch.aten.mul.Scalar %109, %int3_318 : !torch.vtensor<[1,?],si64>, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %401, [%74], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %int3_319 = torch.constant.int 3
    %402 = torch.aten.mul.Scalar %112, %int3_319 : !torch.vtensor<[1,?],si64>, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %402, [%74], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %403 = torch_c.to_builtin_tensor %400 : !torch.vtensor<[1,?],si64> -> tensor<1x?xi64>
    %c1_320 = arith.constant 1 : index
    %dim_321 = tensor.dim %403, %c1_320 : tensor<1x?xi64>
    %404 = flow.tensor.transfer %403 : tensor<1x?xi64>{%dim_321} to #hal.device.promise<@__device_0>
    %405 = torch_c.from_builtin_tensor %404 : tensor<1x?xi64> -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %405, [%74], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %406 = torch_c.to_builtin_tensor %400 : !torch.vtensor<[1,?],si64> -> tensor<1x?xi64>
    %c1_322 = arith.constant 1 : index
    %dim_323 = tensor.dim %406, %c1_322 : tensor<1x?xi64>
    %407 = flow.tensor.transfer %406 : tensor<1x?xi64>{%dim_323} to #hal.device.promise<@__device_1>
    %408 = torch_c.from_builtin_tensor %407 : tensor<1x?xi64> -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %408, [%74], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %int1_324 = torch.constant.int 1
    %409 = torch.aten.add.Tensor %401, %405, %int1_324 : !torch.vtensor<[1,?],si64>, !torch.vtensor<[1,?],si64>, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %409, [%74], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %int1_325 = torch.constant.int 1
    %410 = torch.aten.add.Tensor %402, %408, %int1_325 : !torch.vtensor<[1,?],si64>, !torch.vtensor<[1,?],si64>, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %410, [%74], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %411 = torch.prim.ListConstruct %76 : (!torch.int) -> !torch.list<int>
    %412 = torch.aten.view %409, %411 : !torch.vtensor<[1,?],si64>, !torch.list<int> -> !torch.vtensor<[?],si64>
    torch.bind_symbolic_shape %412, [%74], affine_map<()[s0] -> (s0)> : !torch.vtensor<[?],si64>
    %413 = torch.prim.ListConstruct %76 : (!torch.int) -> !torch.list<int>
    %414 = torch.aten.view %410, %413 : !torch.vtensor<[1,?],si64>, !torch.list<int> -> !torch.vtensor<[?],si64>
    torch.bind_symbolic_shape %414, [%74], affine_map<()[s0] -> (s0)> : !torch.vtensor<[?],si64>
    %int3_326 = torch.constant.int 3
    %int2_327 = torch.constant.int 2
    %int32_328 = torch.constant.int 32
    %int2_329 = torch.constant.int 2
    %int32_330 = torch.constant.int 32
    %415 = torch.prim.ListConstruct %77, %int3_326, %int2_327, %int32_328, %int2_329, %int32_330 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %416 = torch.aten.view %390, %415 : !torch.vtensor<[?,12288],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,2,32],f16>
    torch.bind_symbolic_shape %416, [%75], affine_map<()[s0] -> (s0, 3, 2, 32, 2, 32)> : !torch.vtensor<[?,3,2,32,2,32],f16>
    %int2_331 = torch.constant.int 2
    %int32_332 = torch.constant.int 32
    %int2_333 = torch.constant.int 2
    %int32_334 = torch.constant.int 32
    %417 = torch.prim.ListConstruct %266, %int2_331, %int32_332, %int2_333, %int32_334 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %418 = torch.aten.view %416, %417 : !torch.vtensor<[?,3,2,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,2,32,2,32],f16>
    torch.bind_symbolic_shape %418, [%75], affine_map<()[s0] -> (s0 * 3, 2, 32, 2, 32)> : !torch.vtensor<[?,2,32,2,32],f16>
    %int0_335 = torch.constant.int 0
    %419 = torch.aten.index_select %418, %int0_335, %412 : !torch.vtensor<[?,2,32,2,32],f16>, !torch.int, !torch.vtensor<[?],si64> -> !torch.vtensor<[?,2,32,2,32],f16>
    torch.bind_symbolic_shape %419, [%74], affine_map<()[s0] -> (s0, 2, 32, 2, 32)> : !torch.vtensor<[?,2,32,2,32],f16>
    %int3_336 = torch.constant.int 3
    %int2_337 = torch.constant.int 2
    %int32_338 = torch.constant.int 32
    %int2_339 = torch.constant.int 2
    %int32_340 = torch.constant.int 32
    %420 = torch.prim.ListConstruct %77, %int3_336, %int2_337, %int32_338, %int2_339, %int32_340 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %421 = torch.aten.view %396, %420 : !torch.vtensor<[?,12288],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,2,32],f16>
    torch.bind_symbolic_shape %421, [%75], affine_map<()[s0] -> (s0, 3, 2, 32, 2, 32)> : !torch.vtensor<[?,3,2,32,2,32],f16>
    %int2_341 = torch.constant.int 2
    %int32_342 = torch.constant.int 32
    %int2_343 = torch.constant.int 2
    %int32_344 = torch.constant.int 32
    %422 = torch.prim.ListConstruct %266, %int2_341, %int32_342, %int2_343, %int32_344 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %423 = torch.aten.view %421, %422 : !torch.vtensor<[?,3,2,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,2,32,2,32],f16>
    torch.bind_symbolic_shape %423, [%75], affine_map<()[s0] -> (s0 * 3, 2, 32, 2, 32)> : !torch.vtensor<[?,2,32,2,32],f16>
    %int0_345 = torch.constant.int 0
    %424 = torch.aten.index_select %423, %int0_345, %414 : !torch.vtensor<[?,2,32,2,32],f16>, !torch.int, !torch.vtensor<[?],si64> -> !torch.vtensor<[?,2,32,2,32],f16>
    torch.bind_symbolic_shape %424, [%74], affine_map<()[s0] -> (s0, 2, 32, 2, 32)> : !torch.vtensor<[?,2,32,2,32],f16>
    %int1_346 = torch.constant.int 1
    %int2_347 = torch.constant.int 2
    %int32_348 = torch.constant.int 32
    %int2_349 = torch.constant.int 2
    %int32_350 = torch.constant.int 32
    %425 = torch.prim.ListConstruct %int1_346, %76, %int2_347, %int32_348, %int2_349, %int32_350 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %426 = torch.aten.view %419, %425 : !torch.vtensor<[?,2,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,2,32,2,32],f16>
    torch.bind_symbolic_shape %426, [%74], affine_map<()[s0] -> (1, s0, 2, 32, 2, 32)> : !torch.vtensor<[1,?,2,32,2,32],f16>
    %int1_351 = torch.constant.int 1
    %int2_352 = torch.constant.int 2
    %int32_353 = torch.constant.int 32
    %int2_354 = torch.constant.int 2
    %int32_355 = torch.constant.int 32
    %427 = torch.prim.ListConstruct %int1_351, %76, %int2_352, %int32_353, %int2_354, %int32_355 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %428 = torch.aten.view %424, %427 : !torch.vtensor<[?,2,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,2,32,2,32],f16>
    torch.bind_symbolic_shape %428, [%74], affine_map<()[s0] -> (1, s0, 2, 32, 2, 32)> : !torch.vtensor<[1,?,2,32,2,32],f16>
    %int0_356 = torch.constant.int 0
    %int0_357 = torch.constant.int 0
    %int9223372036854775807 = torch.constant.int 9223372036854775807
    %int1_358 = torch.constant.int 1
    %429 = torch.aten.slice.Tensor %426, %int0_356, %int0_357, %int9223372036854775807, %int1_358 : !torch.vtensor<[1,?,2,32,2,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,2,32,2,32],f16>
    torch.bind_symbolic_shape %429, [%74], affine_map<()[s0] -> (1, s0, 2, 32, 2, 32)> : !torch.vtensor<[1,?,2,32,2,32],f16>
    %int1_359 = torch.constant.int 1
    %int0_360 = torch.constant.int 0
    %int9223372036854775807_361 = torch.constant.int 9223372036854775807
    %int1_362 = torch.constant.int 1
    %430 = torch.aten.slice.Tensor %429, %int1_359, %int0_360, %int9223372036854775807_361, %int1_362 : !torch.vtensor<[1,?,2,32,2,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,2,32,2,32],f16>
    torch.bind_symbolic_shape %430, [%74], affine_map<()[s0] -> (1, s0, 2, 32, 2, 32)> : !torch.vtensor<[1,?,2,32,2,32],f16>
    %int2_363 = torch.constant.int 2
    %int0_364 = torch.constant.int 0
    %431 = torch.aten.select.int %430, %int2_363, %int0_364 : !torch.vtensor<[1,?,2,32,2,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,?,32,2,32],f16>
    torch.bind_symbolic_shape %431, [%74], affine_map<()[s0] -> (1, s0, 32, 2, 32)> : !torch.vtensor<[1,?,32,2,32],f16>
    %int2_365 = torch.constant.int 2
    %int0_366 = torch.constant.int 0
    %int1_367 = torch.constant.int 1
    %432 = torch.aten.slice.Tensor %431, %int2_365, %int0_366, %78, %int1_367 : !torch.vtensor<[1,?,32,2,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32,2,32],f16>
    torch.bind_symbolic_shape %432, [%74], affine_map<()[s0] -> (1, s0, 32, 2, 32)> : !torch.vtensor<[1,?,32,2,32],f16>
    %int0_368 = torch.constant.int 0
    %int0_369 = torch.constant.int 0
    %int9223372036854775807_370 = torch.constant.int 9223372036854775807
    %int1_371 = torch.constant.int 1
    %433 = torch.aten.slice.Tensor %428, %int0_368, %int0_369, %int9223372036854775807_370, %int1_371 : !torch.vtensor<[1,?,2,32,2,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,2,32,2,32],f16>
    torch.bind_symbolic_shape %433, [%74], affine_map<()[s0] -> (1, s0, 2, 32, 2, 32)> : !torch.vtensor<[1,?,2,32,2,32],f16>
    %int1_372 = torch.constant.int 1
    %int0_373 = torch.constant.int 0
    %int9223372036854775807_374 = torch.constant.int 9223372036854775807
    %int1_375 = torch.constant.int 1
    %434 = torch.aten.slice.Tensor %433, %int1_372, %int0_373, %int9223372036854775807_374, %int1_375 : !torch.vtensor<[1,?,2,32,2,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,2,32,2,32],f16>
    torch.bind_symbolic_shape %434, [%74], affine_map<()[s0] -> (1, s0, 2, 32, 2, 32)> : !torch.vtensor<[1,?,2,32,2,32],f16>
    %int2_376 = torch.constant.int 2
    %int0_377 = torch.constant.int 0
    %435 = torch.aten.select.int %434, %int2_376, %int0_377 : !torch.vtensor<[1,?,2,32,2,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,?,32,2,32],f16>
    torch.bind_symbolic_shape %435, [%74], affine_map<()[s0] -> (1, s0, 32, 2, 32)> : !torch.vtensor<[1,?,32,2,32],f16>
    %int2_378 = torch.constant.int 2
    %int0_379 = torch.constant.int 0
    %int1_380 = torch.constant.int 1
    %436 = torch.aten.slice.Tensor %435, %int2_378, %int0_379, %78, %int1_380 : !torch.vtensor<[1,?,32,2,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32,2,32],f16>
    torch.bind_symbolic_shape %436, [%74], affine_map<()[s0] -> (1, s0, 32, 2, 32)> : !torch.vtensor<[1,?,32,2,32],f16>
    %int0_381 = torch.constant.int 0
    %437 = torch.aten.clone %432, %int0_381 : !torch.vtensor<[1,?,32,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,32,2,32],f16>
    torch.bind_symbolic_shape %437, [%74], affine_map<()[s0] -> (1, s0, 32, 2, 32)> : !torch.vtensor<[1,?,32,2,32],f16>
    %int1_382 = torch.constant.int 1
    %int2_383 = torch.constant.int 2
    %int32_384 = torch.constant.int 32
    %438 = torch.prim.ListConstruct %int1_382, %78, %int2_383, %int32_384 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %439 = torch.aten._unsafe_view %437, %438 : !torch.vtensor<[1,?,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,2,32],f16>
    torch.bind_symbolic_shape %439, [%74], affine_map<()[s0] -> (1, s0 * 32, 2, 32)> : !torch.vtensor<[1,?,2,32],f16>
    %int0_385 = torch.constant.int 0
    %440 = torch.aten.clone %436, %int0_385 : !torch.vtensor<[1,?,32,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,32,2,32],f16>
    torch.bind_symbolic_shape %440, [%74], affine_map<()[s0] -> (1, s0, 32, 2, 32)> : !torch.vtensor<[1,?,32,2,32],f16>
    %int1_386 = torch.constant.int 1
    %int2_387 = torch.constant.int 2
    %int32_388 = torch.constant.int 32
    %441 = torch.prim.ListConstruct %int1_386, %78, %int2_387, %int32_388 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %442 = torch.aten._unsafe_view %440, %441 : !torch.vtensor<[1,?,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,2,32],f16>
    torch.bind_symbolic_shape %442, [%74], affine_map<()[s0] -> (1, s0 * 32, 2, 32)> : !torch.vtensor<[1,?,2,32],f16>
    %int0_389 = torch.constant.int 0
    %int0_390 = torch.constant.int 0
    %int9223372036854775807_391 = torch.constant.int 9223372036854775807
    %int1_392 = torch.constant.int 1
    %443 = torch.aten.slice.Tensor %439, %int0_389, %int0_390, %int9223372036854775807_391, %int1_392 : !torch.vtensor<[1,?,2,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,2,32],f16>
    torch.bind_symbolic_shape %443, [%74], affine_map<()[s0] -> (1, s0 * 32, 2, 32)> : !torch.vtensor<[1,?,2,32],f16>
    %int0_393 = torch.constant.int 0
    %int0_394 = torch.constant.int 0
    %int9223372036854775807_395 = torch.constant.int 9223372036854775807
    %int1_396 = torch.constant.int 1
    %444 = torch.aten.slice.Tensor %442, %int0_393, %int0_394, %int9223372036854775807_395, %int1_396 : !torch.vtensor<[1,?,2,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,2,32],f16>
    torch.bind_symbolic_shape %444, [%74], affine_map<()[s0] -> (1, s0 * 32, 2, 32)> : !torch.vtensor<[1,?,2,32],f16>
    %int0_397 = torch.constant.int 0
    %int0_398 = torch.constant.int 0
    %int9223372036854775807_399 = torch.constant.int 9223372036854775807
    %int1_400 = torch.constant.int 1
    %445 = torch.aten.slice.Tensor %426, %int0_397, %int0_398, %int9223372036854775807_399, %int1_400 : !torch.vtensor<[1,?,2,32,2,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,2,32,2,32],f16>
    torch.bind_symbolic_shape %445, [%74], affine_map<()[s0] -> (1, s0, 2, 32, 2, 32)> : !torch.vtensor<[1,?,2,32,2,32],f16>
    %int1_401 = torch.constant.int 1
    %int0_402 = torch.constant.int 0
    %int9223372036854775807_403 = torch.constant.int 9223372036854775807
    %int1_404 = torch.constant.int 1
    %446 = torch.aten.slice.Tensor %445, %int1_401, %int0_402, %int9223372036854775807_403, %int1_404 : !torch.vtensor<[1,?,2,32,2,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,2,32,2,32],f16>
    torch.bind_symbolic_shape %446, [%74], affine_map<()[s0] -> (1, s0, 2, 32, 2, 32)> : !torch.vtensor<[1,?,2,32,2,32],f16>
    %int2_405 = torch.constant.int 2
    %int1_406 = torch.constant.int 1
    %447 = torch.aten.select.int %446, %int2_405, %int1_406 : !torch.vtensor<[1,?,2,32,2,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,?,32,2,32],f16>
    torch.bind_symbolic_shape %447, [%74], affine_map<()[s0] -> (1, s0, 32, 2, 32)> : !torch.vtensor<[1,?,32,2,32],f16>
    %int2_407 = torch.constant.int 2
    %int0_408 = torch.constant.int 0
    %int1_409 = torch.constant.int 1
    %448 = torch.aten.slice.Tensor %447, %int2_407, %int0_408, %78, %int1_409 : !torch.vtensor<[1,?,32,2,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32,2,32],f16>
    torch.bind_symbolic_shape %448, [%74], affine_map<()[s0] -> (1, s0, 32, 2, 32)> : !torch.vtensor<[1,?,32,2,32],f16>
    %int0_410 = torch.constant.int 0
    %int0_411 = torch.constant.int 0
    %int9223372036854775807_412 = torch.constant.int 9223372036854775807
    %int1_413 = torch.constant.int 1
    %449 = torch.aten.slice.Tensor %428, %int0_410, %int0_411, %int9223372036854775807_412, %int1_413 : !torch.vtensor<[1,?,2,32,2,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,2,32,2,32],f16>
    torch.bind_symbolic_shape %449, [%74], affine_map<()[s0] -> (1, s0, 2, 32, 2, 32)> : !torch.vtensor<[1,?,2,32,2,32],f16>
    %int1_414 = torch.constant.int 1
    %int0_415 = torch.constant.int 0
    %int9223372036854775807_416 = torch.constant.int 9223372036854775807
    %int1_417 = torch.constant.int 1
    %450 = torch.aten.slice.Tensor %449, %int1_414, %int0_415, %int9223372036854775807_416, %int1_417 : !torch.vtensor<[1,?,2,32,2,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,2,32,2,32],f16>
    torch.bind_symbolic_shape %450, [%74], affine_map<()[s0] -> (1, s0, 2, 32, 2, 32)> : !torch.vtensor<[1,?,2,32,2,32],f16>
    %int2_418 = torch.constant.int 2
    %int1_419 = torch.constant.int 1
    %451 = torch.aten.select.int %450, %int2_418, %int1_419 : !torch.vtensor<[1,?,2,32,2,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,?,32,2,32],f16>
    torch.bind_symbolic_shape %451, [%74], affine_map<()[s0] -> (1, s0, 32, 2, 32)> : !torch.vtensor<[1,?,32,2,32],f16>
    %int2_420 = torch.constant.int 2
    %int0_421 = torch.constant.int 0
    %int1_422 = torch.constant.int 1
    %452 = torch.aten.slice.Tensor %451, %int2_420, %int0_421, %78, %int1_422 : !torch.vtensor<[1,?,32,2,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32,2,32],f16>
    torch.bind_symbolic_shape %452, [%74], affine_map<()[s0] -> (1, s0, 32, 2, 32)> : !torch.vtensor<[1,?,32,2,32],f16>
    %int0_423 = torch.constant.int 0
    %453 = torch.aten.clone %448, %int0_423 : !torch.vtensor<[1,?,32,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,32,2,32],f16>
    torch.bind_symbolic_shape %453, [%74], affine_map<()[s0] -> (1, s0, 32, 2, 32)> : !torch.vtensor<[1,?,32,2,32],f16>
    %int1_424 = torch.constant.int 1
    %int2_425 = torch.constant.int 2
    %int32_426 = torch.constant.int 32
    %454 = torch.prim.ListConstruct %int1_424, %78, %int2_425, %int32_426 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %455 = torch.aten._unsafe_view %453, %454 : !torch.vtensor<[1,?,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,2,32],f16>
    torch.bind_symbolic_shape %455, [%74], affine_map<()[s0] -> (1, s0 * 32, 2, 32)> : !torch.vtensor<[1,?,2,32],f16>
    %int0_427 = torch.constant.int 0
    %456 = torch.aten.clone %452, %int0_427 : !torch.vtensor<[1,?,32,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,32,2,32],f16>
    torch.bind_symbolic_shape %456, [%74], affine_map<()[s0] -> (1, s0, 32, 2, 32)> : !torch.vtensor<[1,?,32,2,32],f16>
    %int1_428 = torch.constant.int 1
    %int2_429 = torch.constant.int 2
    %int32_430 = torch.constant.int 32
    %457 = torch.prim.ListConstruct %int1_428, %78, %int2_429, %int32_430 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %458 = torch.aten._unsafe_view %456, %457 : !torch.vtensor<[1,?,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,2,32],f16>
    torch.bind_symbolic_shape %458, [%74], affine_map<()[s0] -> (1, s0 * 32, 2, 32)> : !torch.vtensor<[1,?,2,32],f16>
    %int0_431 = torch.constant.int 0
    %int0_432 = torch.constant.int 0
    %int9223372036854775807_433 = torch.constant.int 9223372036854775807
    %int1_434 = torch.constant.int 1
    %459 = torch.aten.slice.Tensor %455, %int0_431, %int0_432, %int9223372036854775807_433, %int1_434 : !torch.vtensor<[1,?,2,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,2,32],f16>
    torch.bind_symbolic_shape %459, [%74], affine_map<()[s0] -> (1, s0 * 32, 2, 32)> : !torch.vtensor<[1,?,2,32],f16>
    %int0_435 = torch.constant.int 0
    %int0_436 = torch.constant.int 0
    %int9223372036854775807_437 = torch.constant.int 9223372036854775807
    %int1_438 = torch.constant.int 1
    %460 = torch.aten.slice.Tensor %458, %int0_435, %int0_436, %int9223372036854775807_437, %int1_438 : !torch.vtensor<[1,?,2,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,2,32],f16>
    torch.bind_symbolic_shape %460, [%74], affine_map<()[s0] -> (1, s0 * 32, 2, 32)> : !torch.vtensor<[1,?,2,32],f16>
    %int-2 = torch.constant.int -2
    %461 = torch.aten.unsqueeze %443, %int-2 : !torch.vtensor<[1,?,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,2,1,32],f16>
    torch.bind_symbolic_shape %461, [%74], affine_map<()[s0] -> (1, s0 * 32, 2, 1, 32)> : !torch.vtensor<[1,?,2,1,32],f16>
    %int-2_439 = torch.constant.int -2
    %462 = torch.aten.unsqueeze %444, %int-2_439 : !torch.vtensor<[1,?,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,2,1,32],f16>
    torch.bind_symbolic_shape %462, [%74], affine_map<()[s0] -> (1, s0 * 32, 2, 1, 32)> : !torch.vtensor<[1,?,2,1,32],f16>
    %int1_440 = torch.constant.int 1
    %int2_441 = torch.constant.int 2
    %int2_442 = torch.constant.int 2
    %int32_443 = torch.constant.int 32
    %463 = torch.prim.ListConstruct %int1_440, %78, %int2_441, %int2_442, %int32_443 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_444 = torch.constant.bool false
    %464 = torch.aten.expand %461, %463, %false_444 : !torch.vtensor<[1,?,2,1,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,?,2,2,32],f16>
    torch.bind_symbolic_shape %464, [%74], affine_map<()[s0] -> (1, s0 * 32, 2, 2, 32)> : !torch.vtensor<[1,?,2,2,32],f16>
    %int1_445 = torch.constant.int 1
    %int2_446 = torch.constant.int 2
    %int2_447 = torch.constant.int 2
    %int32_448 = torch.constant.int 32
    %465 = torch.prim.ListConstruct %int1_445, %78, %int2_446, %int2_447, %int32_448 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_449 = torch.constant.bool false
    %466 = torch.aten.expand %462, %465, %false_449 : !torch.vtensor<[1,?,2,1,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,?,2,2,32],f16>
    torch.bind_symbolic_shape %466, [%74], affine_map<()[s0] -> (1, s0 * 32, 2, 2, 32)> : !torch.vtensor<[1,?,2,2,32],f16>
    %int0_450 = torch.constant.int 0
    %467 = torch.aten.clone %464, %int0_450 : !torch.vtensor<[1,?,2,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,2,2,32],f16>
    torch.bind_symbolic_shape %467, [%74], affine_map<()[s0] -> (1, s0 * 32, 2, 2, 32)> : !torch.vtensor<[1,?,2,2,32],f16>
    %int1_451 = torch.constant.int 1
    %int4_452 = torch.constant.int 4
    %int32_453 = torch.constant.int 32
    %468 = torch.prim.ListConstruct %int1_451, %78, %int4_452, %int32_453 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %469 = torch.aten._unsafe_view %467, %468 : !torch.vtensor<[1,?,2,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %469, [%74], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int0_454 = torch.constant.int 0
    %470 = torch.aten.clone %466, %int0_454 : !torch.vtensor<[1,?,2,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,2,2,32],f16>
    torch.bind_symbolic_shape %470, [%74], affine_map<()[s0] -> (1, s0 * 32, 2, 2, 32)> : !torch.vtensor<[1,?,2,2,32],f16>
    %int1_455 = torch.constant.int 1
    %int4_456 = torch.constant.int 4
    %int32_457 = torch.constant.int 32
    %471 = torch.prim.ListConstruct %int1_455, %78, %int4_456, %int32_457 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %472 = torch.aten._unsafe_view %470, %471 : !torch.vtensor<[1,?,2,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %472, [%74], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int-2_458 = torch.constant.int -2
    %473 = torch.aten.unsqueeze %459, %int-2_458 : !torch.vtensor<[1,?,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,2,1,32],f16>
    torch.bind_symbolic_shape %473, [%74], affine_map<()[s0] -> (1, s0 * 32, 2, 1, 32)> : !torch.vtensor<[1,?,2,1,32],f16>
    %int-2_459 = torch.constant.int -2
    %474 = torch.aten.unsqueeze %460, %int-2_459 : !torch.vtensor<[1,?,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,2,1,32],f16>
    torch.bind_symbolic_shape %474, [%74], affine_map<()[s0] -> (1, s0 * 32, 2, 1, 32)> : !torch.vtensor<[1,?,2,1,32],f16>
    %int1_460 = torch.constant.int 1
    %int2_461 = torch.constant.int 2
    %int2_462 = torch.constant.int 2
    %int32_463 = torch.constant.int 32
    %475 = torch.prim.ListConstruct %int1_460, %78, %int2_461, %int2_462, %int32_463 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_464 = torch.constant.bool false
    %476 = torch.aten.expand %473, %475, %false_464 : !torch.vtensor<[1,?,2,1,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,?,2,2,32],f16>
    torch.bind_symbolic_shape %476, [%74], affine_map<()[s0] -> (1, s0 * 32, 2, 2, 32)> : !torch.vtensor<[1,?,2,2,32],f16>
    %int1_465 = torch.constant.int 1
    %int2_466 = torch.constant.int 2
    %int2_467 = torch.constant.int 2
    %int32_468 = torch.constant.int 32
    %477 = torch.prim.ListConstruct %int1_465, %78, %int2_466, %int2_467, %int32_468 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_469 = torch.constant.bool false
    %478 = torch.aten.expand %474, %477, %false_469 : !torch.vtensor<[1,?,2,1,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,?,2,2,32],f16>
    torch.bind_symbolic_shape %478, [%74], affine_map<()[s0] -> (1, s0 * 32, 2, 2, 32)> : !torch.vtensor<[1,?,2,2,32],f16>
    %int0_470 = torch.constant.int 0
    %479 = torch.aten.clone %476, %int0_470 : !torch.vtensor<[1,?,2,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,2,2,32],f16>
    torch.bind_symbolic_shape %479, [%74], affine_map<()[s0] -> (1, s0 * 32, 2, 2, 32)> : !torch.vtensor<[1,?,2,2,32],f16>
    %int1_471 = torch.constant.int 1
    %int4_472 = torch.constant.int 4
    %int32_473 = torch.constant.int 32
    %480 = torch.prim.ListConstruct %int1_471, %78, %int4_472, %int32_473 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %481 = torch.aten._unsafe_view %479, %480 : !torch.vtensor<[1,?,2,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %481, [%74], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int0_474 = torch.constant.int 0
    %482 = torch.aten.clone %478, %int0_474 : !torch.vtensor<[1,?,2,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,2,2,32],f16>
    torch.bind_symbolic_shape %482, [%74], affine_map<()[s0] -> (1, s0 * 32, 2, 2, 32)> : !torch.vtensor<[1,?,2,2,32],f16>
    %int1_475 = torch.constant.int 1
    %int4_476 = torch.constant.int 4
    %int32_477 = torch.constant.int 32
    %483 = torch.prim.ListConstruct %int1_475, %78, %int4_476, %int32_477 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %484 = torch.aten._unsafe_view %482, %483 : !torch.vtensor<[1,?,2,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %484, [%74], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int1_478 = torch.constant.int 1
    %int2_479 = torch.constant.int 2
    %485 = torch.aten.transpose.int %243, %int1_478, %int2_479 : !torch.vtensor<[1,1,4,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,4,1,32],f16>
    %int1_480 = torch.constant.int 1
    %int2_481 = torch.constant.int 2
    %486 = torch.aten.transpose.int %249, %int1_480, %int2_481 : !torch.vtensor<[1,1,4,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,4,1,32],f16>
    %int1_482 = torch.constant.int 1
    %int2_483 = torch.constant.int 2
    %487 = torch.aten.transpose.int %469, %int1_482, %int2_483 : !torch.vtensor<[1,?,4,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,4,?,32],f16>
    torch.bind_symbolic_shape %487, [%74], affine_map<()[s0] -> (1, 4, s0 * 32, 32)> : !torch.vtensor<[1,4,?,32],f16>
    %int1_484 = torch.constant.int 1
    %int2_485 = torch.constant.int 2
    %488 = torch.aten.transpose.int %472, %int1_484, %int2_485 : !torch.vtensor<[1,?,4,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,4,?,32],f16>
    torch.bind_symbolic_shape %488, [%74], affine_map<()[s0] -> (1, 4, s0 * 32, 32)> : !torch.vtensor<[1,4,?,32],f16>
    %int1_486 = torch.constant.int 1
    %int2_487 = torch.constant.int 2
    %489 = torch.aten.transpose.int %481, %int1_486, %int2_487 : !torch.vtensor<[1,?,4,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,4,?,32],f16>
    torch.bind_symbolic_shape %489, [%74], affine_map<()[s0] -> (1, 4, s0 * 32, 32)> : !torch.vtensor<[1,4,?,32],f16>
    %int1_488 = torch.constant.int 1
    %int2_489 = torch.constant.int 2
    %490 = torch.aten.transpose.int %484, %int1_488, %int2_489 : !torch.vtensor<[1,?,4,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,4,?,32],f16>
    torch.bind_symbolic_shape %490, [%74], affine_map<()[s0] -> (1, 4, s0 * 32, 32)> : !torch.vtensor<[1,4,?,32],f16>
    %int5_490 = torch.constant.int 5
    %491 = torch.prims.convert_element_type %485, %int5_490 : !torch.vtensor<[1,4,1,32],f16>, !torch.int -> !torch.vtensor<[1,4,1,32],f16>
    %int5_491 = torch.constant.int 5
    %492 = torch.prims.convert_element_type %486, %int5_491 : !torch.vtensor<[1,4,1,32],f16>, !torch.int -> !torch.vtensor<[1,4,1,32],f16>
    %int5_492 = torch.constant.int 5
    %493 = torch.prims.convert_element_type %487, %int5_492 : !torch.vtensor<[1,4,?,32],f16>, !torch.int -> !torch.vtensor<[1,4,?,32],f16>
    torch.bind_symbolic_shape %493, [%74], affine_map<()[s0] -> (1, 4, s0 * 32, 32)> : !torch.vtensor<[1,4,?,32],f16>
    %int5_493 = torch.constant.int 5
    %494 = torch.prims.convert_element_type %488, %int5_493 : !torch.vtensor<[1,4,?,32],f16>, !torch.int -> !torch.vtensor<[1,4,?,32],f16>
    torch.bind_symbolic_shape %494, [%74], affine_map<()[s0] -> (1, 4, s0 * 32, 32)> : !torch.vtensor<[1,4,?,32],f16>
    %int5_494 = torch.constant.int 5
    %495 = torch.prims.convert_element_type %489, %int5_494 : !torch.vtensor<[1,4,?,32],f16>, !torch.int -> !torch.vtensor<[1,4,?,32],f16>
    torch.bind_symbolic_shape %495, [%74], affine_map<()[s0] -> (1, 4, s0 * 32, 32)> : !torch.vtensor<[1,4,?,32],f16>
    %int5_495 = torch.constant.int 5
    %496 = torch.prims.convert_element_type %490, %int5_495 : !torch.vtensor<[1,4,?,32],f16>, !torch.int -> !torch.vtensor<[1,4,?,32],f16>
    torch.bind_symbolic_shape %496, [%74], affine_map<()[s0] -> (1, 4, s0 * 32, 32)> : !torch.vtensor<[1,4,?,32],f16>
    %int5_496 = torch.constant.int 5
    %497 = torch.prims.convert_element_type %97, %int5_496 : !torch.vtensor<[1,1,1,?],f16>, !torch.int -> !torch.vtensor<[1,1,1,?],f16>
    torch.bind_symbolic_shape %497, [%74], affine_map<()[s0] -> (1, 1, 1, s0 * 32)> : !torch.vtensor<[1,1,1,?],f16>
    %int5_497 = torch.constant.int 5
    %498 = torch.prims.convert_element_type %100, %int5_497 : !torch.vtensor<[1,1,1,?],f16>, !torch.int -> !torch.vtensor<[1,1,1,?],f16>
    torch.bind_symbolic_shape %498, [%74], affine_map<()[s0] -> (1, 1, 1, s0 * 32)> : !torch.vtensor<[1,1,1,?],f16>
    %float0.000000e00 = torch.constant.float 0.000000e+00
    %false_498 = torch.constant.bool false
    %none_499 = torch.constant.none
    %499:2 = torch.operator "torch.aten._scaled_dot_product_flash_attention_for_cpu"(%491, %493, %495, %float0.000000e00, %false_498, %497, %none_499) : (!torch.vtensor<[1,4,1,32],f16>, !torch.vtensor<[1,4,?,32],f16>, !torch.vtensor<[1,4,?,32],f16>, !torch.float, !torch.bool, !torch.vtensor<[1,1,1,?],f16>, !torch.none) -> (!torch.vtensor<[1,4,1,32],f16>, !torch.vtensor<[1,4,1],f32>) 
    %float0.000000e00_500 = torch.constant.float 0.000000e+00
    %false_501 = torch.constant.bool false
    %none_502 = torch.constant.none
    %500:2 = torch.operator "torch.aten._scaled_dot_product_flash_attention_for_cpu"(%492, %494, %496, %float0.000000e00_500, %false_501, %498, %none_502) : (!torch.vtensor<[1,4,1,32],f16>, !torch.vtensor<[1,4,?,32],f16>, !torch.vtensor<[1,4,?,32],f16>, !torch.float, !torch.bool, !torch.vtensor<[1,1,1,?],f16>, !torch.none) -> (!torch.vtensor<[1,4,1,32],f16>, !torch.vtensor<[1,4,1],f32>) 
    %int1_503 = torch.constant.int 1
    %int2_504 = torch.constant.int 2
    %501 = torch.aten.transpose.int %499#0, %int1_503, %int2_504 : !torch.vtensor<[1,4,1,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,1,4,32],f16>
    %int1_505 = torch.constant.int 1
    %int2_506 = torch.constant.int 2
    %502 = torch.aten.transpose.int %500#0, %int1_505, %int2_506 : !torch.vtensor<[1,4,1,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,1,4,32],f16>
    %int1_507 = torch.constant.int 1
    %int1_508 = torch.constant.int 1
    %int128_509 = torch.constant.int 128
    %503 = torch.prim.ListConstruct %int1_507, %int1_508, %int128_509 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %504 = torch.aten.view %501, %503 : !torch.vtensor<[1,1,4,32],f16>, !torch.list<int> -> !torch.vtensor<[1,1,128],f16>
    %int1_510 = torch.constant.int 1
    %int1_511 = torch.constant.int 1
    %int128_512 = torch.constant.int 128
    %505 = torch.prim.ListConstruct %int1_510, %int1_511, %int128_512 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %506 = torch.aten.view %502, %505 : !torch.vtensor<[1,1,4,32],f16>, !torch.list<int> -> !torch.vtensor<[1,1,128],f16>
    %int1_513 = torch.constant.int 1
    %int0_514 = torch.constant.int 0
    %507 = torch.prim.ListConstruct %int1_513, %int0_514 : (!torch.int, !torch.int) -> !torch.list<int>
    %508 = torch.aten.permute %14, %507 : !torch.vtensor<[256,128],f32>, !torch.list<int> -> !torch.vtensor<[128,256],f32>
    %int1_515 = torch.constant.int 1
    %int0_516 = torch.constant.int 0
    %509 = torch.prim.ListConstruct %int1_515, %int0_516 : (!torch.int, !torch.int) -> !torch.list<int>
    %510 = torch.aten.permute %15, %509 : !torch.vtensor<[256,128],f32>, !torch.list<int> -> !torch.vtensor<[128,256],f32>
    %int5_517 = torch.constant.int 5
    %511 = torch.prims.convert_element_type %508, %int5_517 : !torch.vtensor<[128,256],f32>, !torch.int -> !torch.vtensor<[128,256],f16>
    %int1_518 = torch.constant.int 1
    %int128_519 = torch.constant.int 128
    %512 = torch.prim.ListConstruct %int1_518, %int128_519 : (!torch.int, !torch.int) -> !torch.list<int>
    %513 = torch.aten.view %504, %512 : !torch.vtensor<[1,1,128],f16>, !torch.list<int> -> !torch.vtensor<[1,128],f16>
    %514 = torch.aten.mm %513, %511 : !torch.vtensor<[1,128],f16>, !torch.vtensor<[128,256],f16> -> !torch.vtensor<[1,256],f16>
    %int1_520 = torch.constant.int 1
    %int1_521 = torch.constant.int 1
    %int256_522 = torch.constant.int 256
    %515 = torch.prim.ListConstruct %int1_520, %int1_521, %int256_522 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %516 = torch.aten.view %514, %515 : !torch.vtensor<[1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,1,256],f16>
    %int5_523 = torch.constant.int 5
    %517 = torch.prims.convert_element_type %510, %int5_523 : !torch.vtensor<[128,256],f32>, !torch.int -> !torch.vtensor<[128,256],f16>
    %int1_524 = torch.constant.int 1
    %int128_525 = torch.constant.int 128
    %518 = torch.prim.ListConstruct %int1_524, %int128_525 : (!torch.int, !torch.int) -> !torch.list<int>
    %519 = torch.aten.view %506, %518 : !torch.vtensor<[1,1,128],f16>, !torch.list<int> -> !torch.vtensor<[1,128],f16>
    %520 = torch.aten.mm %519, %517 : !torch.vtensor<[1,128],f16>, !torch.vtensor<[128,256],f16> -> !torch.vtensor<[1,256],f16>
    %int1_526 = torch.constant.int 1
    %int1_527 = torch.constant.int 1
    %int256_528 = torch.constant.int 256
    %521 = torch.prim.ListConstruct %int1_526, %int1_527, %int256_528 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %522 = torch.aten.view %520, %521 : !torch.vtensor<[1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,1,256],f16>
    %523 = torch_c.to_builtin_tensor %516 : !torch.vtensor<[1,1,256],f16> -> tensor<1x1x256xf16>
    %524 = flow.tensor.barrier %523 : tensor<1x1x256xf16> on #hal.device.promise<@__device_0>
    %525 = torch_c.from_builtin_tensor %524 : tensor<1x1x256xf16> -> !torch.vtensor<[1,1,256],f16>
    %526 = torch_c.to_builtin_tensor %522 : !torch.vtensor<[1,1,256],f16> -> tensor<1x1x256xf16>
    %527 = flow.tensor.transfer %526 : tensor<1x1x256xf16> to #hal.device.promise<@__device_0>
    %528 = torch_c.from_builtin_tensor %527 : tensor<1x1x256xf16> -> !torch.vtensor<[1,1,256],f16>
    %int1_529 = torch.constant.int 1
    %529 = torch.aten.add.Tensor %525, %528, %int1_529 : !torch.vtensor<[1,1,256],f16>, !torch.vtensor<[1,1,256],f16>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %530 = torch_c.to_builtin_tensor %529 : !torch.vtensor<[1,1,256],f16> -> tensor<1x1x256xf16>
    %531 = flow.tensor.barrier %530 : tensor<1x1x256xf16> on #hal.device.promise<@__device_0>
    %532 = torch_c.from_builtin_tensor %531 : tensor<1x1x256xf16> -> !torch.vtensor<[1,1,256],f16>
    %533 = torch_c.to_builtin_tensor %529 : !torch.vtensor<[1,1,256],f16> -> tensor<1x1x256xf16>
    %534 = flow.tensor.transfer %533 : tensor<1x1x256xf16> to #hal.device.promise<@__device_1>
    %535 = torch_c.from_builtin_tensor %534 : tensor<1x1x256xf16> -> !torch.vtensor<[1,1,256],f16>
    %int1_530 = torch.constant.int 1
    %536 = torch.aten.add.Tensor %155, %532, %int1_530 : !torch.vtensor<[1,1,256],f16>, !torch.vtensor<[1,1,256],f16>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %int1_531 = torch.constant.int 1
    %537 = torch.aten.add.Tensor %157, %535, %int1_531 : !torch.vtensor<[1,1,256],f16>, !torch.vtensor<[1,1,256],f16>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %int6_532 = torch.constant.int 6
    %538 = torch.prims.convert_element_type %536, %int6_532 : !torch.vtensor<[1,1,256],f16>, !torch.int -> !torch.vtensor<[1,1,256],f32>
    %int6_533 = torch.constant.int 6
    %539 = torch.prims.convert_element_type %537, %int6_533 : !torch.vtensor<[1,1,256],f16>, !torch.int -> !torch.vtensor<[1,1,256],f32>
    %int2_534 = torch.constant.int 2
    %540 = torch.aten.pow.Tensor_Scalar %538, %int2_534 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f32>
    %int2_535 = torch.constant.int 2
    %541 = torch.aten.pow.Tensor_Scalar %539, %int2_535 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f32>
    %int-1_536 = torch.constant.int -1
    %542 = torch.prim.ListConstruct %int-1_536 : (!torch.int) -> !torch.list<int>
    %true_537 = torch.constant.bool true
    %none_538 = torch.constant.none
    %543 = torch.aten.mean.dim %540, %542, %true_537, %none_538 : !torch.vtensor<[1,1,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1],f32>
    %int-1_539 = torch.constant.int -1
    %544 = torch.prim.ListConstruct %int-1_539 : (!torch.int) -> !torch.list<int>
    %true_540 = torch.constant.bool true
    %none_541 = torch.constant.none
    %545 = torch.aten.mean.dim %541, %544, %true_540, %none_541 : !torch.vtensor<[1,1,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1],f32>
    %float1.000000e-02_542 = torch.constant.float 1.000000e-02
    %int1_543 = torch.constant.int 1
    %546 = torch.aten.add.Scalar %543, %float1.000000e-02_542, %int1_543 : !torch.vtensor<[1,1,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,1,1],f32>
    %float1.000000e-02_544 = torch.constant.float 1.000000e-02
    %int1_545 = torch.constant.int 1
    %547 = torch.aten.add.Scalar %545, %float1.000000e-02_544, %int1_545 : !torch.vtensor<[1,1,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,1,1],f32>
    %548 = torch.aten.rsqrt %546 : !torch.vtensor<[1,1,1],f32> -> !torch.vtensor<[1,1,1],f32>
    %549 = torch.aten.rsqrt %547 : !torch.vtensor<[1,1,1],f32> -> !torch.vtensor<[1,1,1],f32>
    %550 = torch.aten.mul.Tensor %538, %548 : !torch.vtensor<[1,1,256],f32>, !torch.vtensor<[1,1,1],f32> -> !torch.vtensor<[1,1,256],f32>
    %551 = torch.aten.mul.Tensor %539, %549 : !torch.vtensor<[1,1,256],f32>, !torch.vtensor<[1,1,1],f32> -> !torch.vtensor<[1,1,256],f32>
    %int5_546 = torch.constant.int 5
    %552 = torch.prims.convert_element_type %550, %int5_546 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %int5_547 = torch.constant.int 5
    %553 = torch.prims.convert_element_type %551, %int5_547 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %554 = torch.aten.mul.Tensor %16, %552 : !torch.vtensor<[256],f32>, !torch.vtensor<[1,1,256],f16> -> !torch.vtensor<[1,1,256],f32>
    %555 = torch.aten.mul.Tensor %17, %553 : !torch.vtensor<[256],f32>, !torch.vtensor<[1,1,256],f16> -> !torch.vtensor<[1,1,256],f32>
    %int5_548 = torch.constant.int 5
    %556 = torch.prims.convert_element_type %554, %int5_548 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %int5_549 = torch.constant.int 5
    %557 = torch.prims.convert_element_type %555, %int5_549 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %int1_550 = torch.constant.int 1
    %int0_551 = torch.constant.int 0
    %558 = torch.prim.ListConstruct %int1_550, %int0_551 : (!torch.int, !torch.int) -> !torch.list<int>
    %559 = torch.aten.permute %18, %558 : !torch.vtensor<[12,256],f32>, !torch.list<int> -> !torch.vtensor<[256,12],f32>
    %int1_552 = torch.constant.int 1
    %int0_553 = torch.constant.int 0
    %560 = torch.prim.ListConstruct %int1_552, %int0_553 : (!torch.int, !torch.int) -> !torch.list<int>
    %561 = torch.aten.permute %19, %560 : !torch.vtensor<[11,256],f32>, !torch.list<int> -> !torch.vtensor<[256,11],f32>
    %int5_554 = torch.constant.int 5
    %562 = torch.prims.convert_element_type %559, %int5_554 : !torch.vtensor<[256,12],f32>, !torch.int -> !torch.vtensor<[256,12],f16>
    %int1_555 = torch.constant.int 1
    %int256_556 = torch.constant.int 256
    %563 = torch.prim.ListConstruct %int1_555, %int256_556 : (!torch.int, !torch.int) -> !torch.list<int>
    %564 = torch.aten.view %556, %563 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %565 = torch.aten.mm %564, %562 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,12],f16> -> !torch.vtensor<[1,12],f16>
    %int1_557 = torch.constant.int 1
    %int1_558 = torch.constant.int 1
    %int12 = torch.constant.int 12
    %566 = torch.prim.ListConstruct %int1_557, %int1_558, %int12 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %567 = torch.aten.view %565, %566 : !torch.vtensor<[1,12],f16>, !torch.list<int> -> !torch.vtensor<[1,1,12],f16>
    %int5_559 = torch.constant.int 5
    %568 = torch.prims.convert_element_type %561, %int5_559 : !torch.vtensor<[256,11],f32>, !torch.int -> !torch.vtensor<[256,11],f16>
    %int1_560 = torch.constant.int 1
    %int256_561 = torch.constant.int 256
    %569 = torch.prim.ListConstruct %int1_560, %int256_561 : (!torch.int, !torch.int) -> !torch.list<int>
    %570 = torch.aten.view %557, %569 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %571 = torch.aten.mm %570, %568 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,11],f16> -> !torch.vtensor<[1,11],f16>
    %int1_562 = torch.constant.int 1
    %int1_563 = torch.constant.int 1
    %int11 = torch.constant.int 11
    %572 = torch.prim.ListConstruct %int1_562, %int1_563, %int11 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %573 = torch.aten.view %571, %572 : !torch.vtensor<[1,11],f16>, !torch.list<int> -> !torch.vtensor<[1,1,11],f16>
    %574 = torch.aten.silu %567 : !torch.vtensor<[1,1,12],f16> -> !torch.vtensor<[1,1,12],f16>
    %575 = torch.aten.silu %573 : !torch.vtensor<[1,1,11],f16> -> !torch.vtensor<[1,1,11],f16>
    %int1_564 = torch.constant.int 1
    %int0_565 = torch.constant.int 0
    %576 = torch.prim.ListConstruct %int1_564, %int0_565 : (!torch.int, !torch.int) -> !torch.list<int>
    %577 = torch.aten.permute %20, %576 : !torch.vtensor<[12,256],f32>, !torch.list<int> -> !torch.vtensor<[256,12],f32>
    %int1_566 = torch.constant.int 1
    %int0_567 = torch.constant.int 0
    %578 = torch.prim.ListConstruct %int1_566, %int0_567 : (!torch.int, !torch.int) -> !torch.list<int>
    %579 = torch.aten.permute %21, %578 : !torch.vtensor<[11,256],f32>, !torch.list<int> -> !torch.vtensor<[256,11],f32>
    %int5_568 = torch.constant.int 5
    %580 = torch.prims.convert_element_type %577, %int5_568 : !torch.vtensor<[256,12],f32>, !torch.int -> !torch.vtensor<[256,12],f16>
    %int1_569 = torch.constant.int 1
    %int256_570 = torch.constant.int 256
    %581 = torch.prim.ListConstruct %int1_569, %int256_570 : (!torch.int, !torch.int) -> !torch.list<int>
    %582 = torch.aten.view %556, %581 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %583 = torch.aten.mm %582, %580 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,12],f16> -> !torch.vtensor<[1,12],f16>
    %int1_571 = torch.constant.int 1
    %int1_572 = torch.constant.int 1
    %int12_573 = torch.constant.int 12
    %584 = torch.prim.ListConstruct %int1_571, %int1_572, %int12_573 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %585 = torch.aten.view %583, %584 : !torch.vtensor<[1,12],f16>, !torch.list<int> -> !torch.vtensor<[1,1,12],f16>
    %int5_574 = torch.constant.int 5
    %586 = torch.prims.convert_element_type %579, %int5_574 : !torch.vtensor<[256,11],f32>, !torch.int -> !torch.vtensor<[256,11],f16>
    %int1_575 = torch.constant.int 1
    %int256_576 = torch.constant.int 256
    %587 = torch.prim.ListConstruct %int1_575, %int256_576 : (!torch.int, !torch.int) -> !torch.list<int>
    %588 = torch.aten.view %557, %587 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %589 = torch.aten.mm %588, %586 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,11],f16> -> !torch.vtensor<[1,11],f16>
    %int1_577 = torch.constant.int 1
    %int1_578 = torch.constant.int 1
    %int11_579 = torch.constant.int 11
    %590 = torch.prim.ListConstruct %int1_577, %int1_578, %int11_579 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %591 = torch.aten.view %589, %590 : !torch.vtensor<[1,11],f16>, !torch.list<int> -> !torch.vtensor<[1,1,11],f16>
    %592 = torch.aten.mul.Tensor %574, %585 : !torch.vtensor<[1,1,12],f16>, !torch.vtensor<[1,1,12],f16> -> !torch.vtensor<[1,1,12],f16>
    %593 = torch.aten.mul.Tensor %575, %591 : !torch.vtensor<[1,1,11],f16>, !torch.vtensor<[1,1,11],f16> -> !torch.vtensor<[1,1,11],f16>
    %int1_580 = torch.constant.int 1
    %int0_581 = torch.constant.int 0
    %594 = torch.prim.ListConstruct %int1_580, %int0_581 : (!torch.int, !torch.int) -> !torch.list<int>
    %595 = torch.aten.permute %22, %594 : !torch.vtensor<[256,12],f32>, !torch.list<int> -> !torch.vtensor<[12,256],f32>
    %int1_582 = torch.constant.int 1
    %int0_583 = torch.constant.int 0
    %596 = torch.prim.ListConstruct %int1_582, %int0_583 : (!torch.int, !torch.int) -> !torch.list<int>
    %597 = torch.aten.permute %23, %596 : !torch.vtensor<[256,11],f32>, !torch.list<int> -> !torch.vtensor<[11,256],f32>
    %int5_584 = torch.constant.int 5
    %598 = torch.prims.convert_element_type %595, %int5_584 : !torch.vtensor<[12,256],f32>, !torch.int -> !torch.vtensor<[12,256],f16>
    %int1_585 = torch.constant.int 1
    %int12_586 = torch.constant.int 12
    %599 = torch.prim.ListConstruct %int1_585, %int12_586 : (!torch.int, !torch.int) -> !torch.list<int>
    %600 = torch.aten.view %592, %599 : !torch.vtensor<[1,1,12],f16>, !torch.list<int> -> !torch.vtensor<[1,12],f16>
    %601 = torch.aten.mm %600, %598 : !torch.vtensor<[1,12],f16>, !torch.vtensor<[12,256],f16> -> !torch.vtensor<[1,256],f16>
    %int1_587 = torch.constant.int 1
    %int1_588 = torch.constant.int 1
    %int256_589 = torch.constant.int 256
    %602 = torch.prim.ListConstruct %int1_587, %int1_588, %int256_589 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %603 = torch.aten.view %601, %602 : !torch.vtensor<[1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,1,256],f16>
    %int5_590 = torch.constant.int 5
    %604 = torch.prims.convert_element_type %597, %int5_590 : !torch.vtensor<[11,256],f32>, !torch.int -> !torch.vtensor<[11,256],f16>
    %int1_591 = torch.constant.int 1
    %int11_592 = torch.constant.int 11
    %605 = torch.prim.ListConstruct %int1_591, %int11_592 : (!torch.int, !torch.int) -> !torch.list<int>
    %606 = torch.aten.view %593, %605 : !torch.vtensor<[1,1,11],f16>, !torch.list<int> -> !torch.vtensor<[1,11],f16>
    %607 = torch.aten.mm %606, %604 : !torch.vtensor<[1,11],f16>, !torch.vtensor<[11,256],f16> -> !torch.vtensor<[1,256],f16>
    %int1_593 = torch.constant.int 1
    %int1_594 = torch.constant.int 1
    %int256_595 = torch.constant.int 256
    %608 = torch.prim.ListConstruct %int1_593, %int1_594, %int256_595 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %609 = torch.aten.view %607, %608 : !torch.vtensor<[1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,1,256],f16>
    %610 = torch_c.to_builtin_tensor %603 : !torch.vtensor<[1,1,256],f16> -> tensor<1x1x256xf16>
    %611 = flow.tensor.barrier %610 : tensor<1x1x256xf16> on #hal.device.promise<@__device_0>
    %612 = torch_c.from_builtin_tensor %611 : tensor<1x1x256xf16> -> !torch.vtensor<[1,1,256],f16>
    %613 = torch_c.to_builtin_tensor %609 : !torch.vtensor<[1,1,256],f16> -> tensor<1x1x256xf16>
    %614 = flow.tensor.transfer %613 : tensor<1x1x256xf16> to #hal.device.promise<@__device_0>
    %615 = torch_c.from_builtin_tensor %614 : tensor<1x1x256xf16> -> !torch.vtensor<[1,1,256],f16>
    %int1_596 = torch.constant.int 1
    %616 = torch.aten.add.Tensor %612, %615, %int1_596 : !torch.vtensor<[1,1,256],f16>, !torch.vtensor<[1,1,256],f16>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %617 = torch_c.to_builtin_tensor %616 : !torch.vtensor<[1,1,256],f16> -> tensor<1x1x256xf16>
    %618 = flow.tensor.barrier %617 : tensor<1x1x256xf16> on #hal.device.promise<@__device_0>
    %619 = torch_c.from_builtin_tensor %618 : tensor<1x1x256xf16> -> !torch.vtensor<[1,1,256],f16>
    %620 = torch_c.to_builtin_tensor %616 : !torch.vtensor<[1,1,256],f16> -> tensor<1x1x256xf16>
    %621 = flow.tensor.transfer %620 : tensor<1x1x256xf16> to #hal.device.promise<@__device_1>
    %622 = torch_c.from_builtin_tensor %621 : tensor<1x1x256xf16> -> !torch.vtensor<[1,1,256],f16>
    %int1_597 = torch.constant.int 1
    %623 = torch.aten.add.Tensor %536, %619, %int1_597 : !torch.vtensor<[1,1,256],f16>, !torch.vtensor<[1,1,256],f16>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %int1_598 = torch.constant.int 1
    %624 = torch.aten.add.Tensor %537, %622, %int1_598 : !torch.vtensor<[1,1,256],f16>, !torch.vtensor<[1,1,256],f16>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %int6_599 = torch.constant.int 6
    %625 = torch.prims.convert_element_type %623, %int6_599 : !torch.vtensor<[1,1,256],f16>, !torch.int -> !torch.vtensor<[1,1,256],f32>
    %int6_600 = torch.constant.int 6
    %626 = torch.prims.convert_element_type %624, %int6_600 : !torch.vtensor<[1,1,256],f16>, !torch.int -> !torch.vtensor<[1,1,256],f32>
    %int2_601 = torch.constant.int 2
    %627 = torch.aten.pow.Tensor_Scalar %625, %int2_601 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f32>
    %int2_602 = torch.constant.int 2
    %628 = torch.aten.pow.Tensor_Scalar %626, %int2_602 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f32>
    %int-1_603 = torch.constant.int -1
    %629 = torch.prim.ListConstruct %int-1_603 : (!torch.int) -> !torch.list<int>
    %true_604 = torch.constant.bool true
    %none_605 = torch.constant.none
    %630 = torch.aten.mean.dim %627, %629, %true_604, %none_605 : !torch.vtensor<[1,1,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1],f32>
    %int-1_606 = torch.constant.int -1
    %631 = torch.prim.ListConstruct %int-1_606 : (!torch.int) -> !torch.list<int>
    %true_607 = torch.constant.bool true
    %none_608 = torch.constant.none
    %632 = torch.aten.mean.dim %628, %631, %true_607, %none_608 : !torch.vtensor<[1,1,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1],f32>
    %float1.000000e-02_609 = torch.constant.float 1.000000e-02
    %int1_610 = torch.constant.int 1
    %633 = torch.aten.add.Scalar %630, %float1.000000e-02_609, %int1_610 : !torch.vtensor<[1,1,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,1,1],f32>
    %float1.000000e-02_611 = torch.constant.float 1.000000e-02
    %int1_612 = torch.constant.int 1
    %634 = torch.aten.add.Scalar %632, %float1.000000e-02_611, %int1_612 : !torch.vtensor<[1,1,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,1,1],f32>
    %635 = torch.aten.rsqrt %633 : !torch.vtensor<[1,1,1],f32> -> !torch.vtensor<[1,1,1],f32>
    %636 = torch.aten.rsqrt %634 : !torch.vtensor<[1,1,1],f32> -> !torch.vtensor<[1,1,1],f32>
    %637 = torch.aten.mul.Tensor %625, %635 : !torch.vtensor<[1,1,256],f32>, !torch.vtensor<[1,1,1],f32> -> !torch.vtensor<[1,1,256],f32>
    %638 = torch.aten.mul.Tensor %626, %636 : !torch.vtensor<[1,1,256],f32>, !torch.vtensor<[1,1,1],f32> -> !torch.vtensor<[1,1,256],f32>
    %int5_613 = torch.constant.int 5
    %639 = torch.prims.convert_element_type %637, %int5_613 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %int5_614 = torch.constant.int 5
    %640 = torch.prims.convert_element_type %638, %int5_614 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %641 = torch.aten.mul.Tensor %24, %639 : !torch.vtensor<[256],f32>, !torch.vtensor<[1,1,256],f16> -> !torch.vtensor<[1,1,256],f32>
    %642 = torch.aten.mul.Tensor %25, %640 : !torch.vtensor<[256],f32>, !torch.vtensor<[1,1,256],f16> -> !torch.vtensor<[1,1,256],f32>
    %int5_615 = torch.constant.int 5
    %643 = torch.prims.convert_element_type %641, %int5_615 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %int5_616 = torch.constant.int 5
    %644 = torch.prims.convert_element_type %642, %int5_616 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %int1_617 = torch.constant.int 1
    %int0_618 = torch.constant.int 0
    %645 = torch.prim.ListConstruct %int1_617, %int0_618 : (!torch.int, !torch.int) -> !torch.list<int>
    %646 = torch.aten.permute %26, %645 : !torch.vtensor<[128,256],f32>, !torch.list<int> -> !torch.vtensor<[256,128],f32>
    %int1_619 = torch.constant.int 1
    %int0_620 = torch.constant.int 0
    %647 = torch.prim.ListConstruct %int1_619, %int0_620 : (!torch.int, !torch.int) -> !torch.list<int>
    %648 = torch.aten.permute %27, %647 : !torch.vtensor<[128,256],f32>, !torch.list<int> -> !torch.vtensor<[256,128],f32>
    %int5_621 = torch.constant.int 5
    %649 = torch.prims.convert_element_type %646, %int5_621 : !torch.vtensor<[256,128],f32>, !torch.int -> !torch.vtensor<[256,128],f16>
    %int1_622 = torch.constant.int 1
    %int256_623 = torch.constant.int 256
    %650 = torch.prim.ListConstruct %int1_622, %int256_623 : (!torch.int, !torch.int) -> !torch.list<int>
    %651 = torch.aten.view %643, %650 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %652 = torch.aten.mm %651, %649 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,128],f16> -> !torch.vtensor<[1,128],f16>
    %int1_624 = torch.constant.int 1
    %int1_625 = torch.constant.int 1
    %int128_626 = torch.constant.int 128
    %653 = torch.prim.ListConstruct %int1_624, %int1_625, %int128_626 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %654 = torch.aten.view %652, %653 : !torch.vtensor<[1,128],f16>, !torch.list<int> -> !torch.vtensor<[1,1,128],f16>
    %int5_627 = torch.constant.int 5
    %655 = torch.prims.convert_element_type %648, %int5_627 : !torch.vtensor<[256,128],f32>, !torch.int -> !torch.vtensor<[256,128],f16>
    %int1_628 = torch.constant.int 1
    %int256_629 = torch.constant.int 256
    %656 = torch.prim.ListConstruct %int1_628, %int256_629 : (!torch.int, !torch.int) -> !torch.list<int>
    %657 = torch.aten.view %644, %656 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %658 = torch.aten.mm %657, %655 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,128],f16> -> !torch.vtensor<[1,128],f16>
    %int1_630 = torch.constant.int 1
    %int1_631 = torch.constant.int 1
    %int128_632 = torch.constant.int 128
    %659 = torch.prim.ListConstruct %int1_630, %int1_631, %int128_632 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %660 = torch.aten.view %658, %659 : !torch.vtensor<[1,128],f16>, !torch.list<int> -> !torch.vtensor<[1,1,128],f16>
    %int1_633 = torch.constant.int 1
    %int0_634 = torch.constant.int 0
    %661 = torch.prim.ListConstruct %int1_633, %int0_634 : (!torch.int, !torch.int) -> !torch.list<int>
    %662 = torch.aten.permute %28, %661 : !torch.vtensor<[64,256],f32>, !torch.list<int> -> !torch.vtensor<[256,64],f32>
    %int1_635 = torch.constant.int 1
    %int0_636 = torch.constant.int 0
    %663 = torch.prim.ListConstruct %int1_635, %int0_636 : (!torch.int, !torch.int) -> !torch.list<int>
    %664 = torch.aten.permute %29, %663 : !torch.vtensor<[64,256],f32>, !torch.list<int> -> !torch.vtensor<[256,64],f32>
    %int5_637 = torch.constant.int 5
    %665 = torch.prims.convert_element_type %662, %int5_637 : !torch.vtensor<[256,64],f32>, !torch.int -> !torch.vtensor<[256,64],f16>
    %int1_638 = torch.constant.int 1
    %int256_639 = torch.constant.int 256
    %666 = torch.prim.ListConstruct %int1_638, %int256_639 : (!torch.int, !torch.int) -> !torch.list<int>
    %667 = torch.aten.view %643, %666 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %668 = torch.aten.mm %667, %665 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,64],f16> -> !torch.vtensor<[1,64],f16>
    %int1_640 = torch.constant.int 1
    %int1_641 = torch.constant.int 1
    %int64_642 = torch.constant.int 64
    %669 = torch.prim.ListConstruct %int1_640, %int1_641, %int64_642 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %670 = torch.aten.view %668, %669 : !torch.vtensor<[1,64],f16>, !torch.list<int> -> !torch.vtensor<[1,1,64],f16>
    %int5_643 = torch.constant.int 5
    %671 = torch.prims.convert_element_type %664, %int5_643 : !torch.vtensor<[256,64],f32>, !torch.int -> !torch.vtensor<[256,64],f16>
    %int1_644 = torch.constant.int 1
    %int256_645 = torch.constant.int 256
    %672 = torch.prim.ListConstruct %int1_644, %int256_645 : (!torch.int, !torch.int) -> !torch.list<int>
    %673 = torch.aten.view %644, %672 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %674 = torch.aten.mm %673, %671 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,64],f16> -> !torch.vtensor<[1,64],f16>
    %int1_646 = torch.constant.int 1
    %int1_647 = torch.constant.int 1
    %int64_648 = torch.constant.int 64
    %675 = torch.prim.ListConstruct %int1_646, %int1_647, %int64_648 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %676 = torch.aten.view %674, %675 : !torch.vtensor<[1,64],f16>, !torch.list<int> -> !torch.vtensor<[1,1,64],f16>
    %int1_649 = torch.constant.int 1
    %int0_650 = torch.constant.int 0
    %677 = torch.prim.ListConstruct %int1_649, %int0_650 : (!torch.int, !torch.int) -> !torch.list<int>
    %678 = torch.aten.permute %30, %677 : !torch.vtensor<[64,256],f32>, !torch.list<int> -> !torch.vtensor<[256,64],f32>
    %int1_651 = torch.constant.int 1
    %int0_652 = torch.constant.int 0
    %679 = torch.prim.ListConstruct %int1_651, %int0_652 : (!torch.int, !torch.int) -> !torch.list<int>
    %680 = torch.aten.permute %31, %679 : !torch.vtensor<[64,256],f32>, !torch.list<int> -> !torch.vtensor<[256,64],f32>
    %int5_653 = torch.constant.int 5
    %681 = torch.prims.convert_element_type %678, %int5_653 : !torch.vtensor<[256,64],f32>, !torch.int -> !torch.vtensor<[256,64],f16>
    %int1_654 = torch.constant.int 1
    %int256_655 = torch.constant.int 256
    %682 = torch.prim.ListConstruct %int1_654, %int256_655 : (!torch.int, !torch.int) -> !torch.list<int>
    %683 = torch.aten.view %643, %682 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %684 = torch.aten.mm %683, %681 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,64],f16> -> !torch.vtensor<[1,64],f16>
    %int1_656 = torch.constant.int 1
    %int1_657 = torch.constant.int 1
    %int64_658 = torch.constant.int 64
    %685 = torch.prim.ListConstruct %int1_656, %int1_657, %int64_658 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %686 = torch.aten.view %684, %685 : !torch.vtensor<[1,64],f16>, !torch.list<int> -> !torch.vtensor<[1,1,64],f16>
    %int5_659 = torch.constant.int 5
    %687 = torch.prims.convert_element_type %680, %int5_659 : !torch.vtensor<[256,64],f32>, !torch.int -> !torch.vtensor<[256,64],f16>
    %int1_660 = torch.constant.int 1
    %int256_661 = torch.constant.int 256
    %688 = torch.prim.ListConstruct %int1_660, %int256_661 : (!torch.int, !torch.int) -> !torch.list<int>
    %689 = torch.aten.view %644, %688 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %690 = torch.aten.mm %689, %687 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,64],f16> -> !torch.vtensor<[1,64],f16>
    %int1_662 = torch.constant.int 1
    %int1_663 = torch.constant.int 1
    %int64_664 = torch.constant.int 64
    %691 = torch.prim.ListConstruct %int1_662, %int1_663, %int64_664 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %692 = torch.aten.view %690, %691 : !torch.vtensor<[1,64],f16>, !torch.list<int> -> !torch.vtensor<[1,1,64],f16>
    %int1_665 = torch.constant.int 1
    %int1_666 = torch.constant.int 1
    %int4_667 = torch.constant.int 4
    %int32_668 = torch.constant.int 32
    %693 = torch.prim.ListConstruct %int1_665, %int1_666, %int4_667, %int32_668 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %694 = torch.aten.view %654, %693 : !torch.vtensor<[1,1,128],f16>, !torch.list<int> -> !torch.vtensor<[1,1,4,32],f16>
    %int1_669 = torch.constant.int 1
    %int1_670 = torch.constant.int 1
    %int4_671 = torch.constant.int 4
    %int32_672 = torch.constant.int 32
    %695 = torch.prim.ListConstruct %int1_669, %int1_670, %int4_671, %int32_672 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %696 = torch.aten.view %660, %695 : !torch.vtensor<[1,1,128],f16>, !torch.list<int> -> !torch.vtensor<[1,1,4,32],f16>
    %int1_673 = torch.constant.int 1
    %int1_674 = torch.constant.int 1
    %int2_675 = torch.constant.int 2
    %int32_676 = torch.constant.int 32
    %697 = torch.prim.ListConstruct %int1_673, %int1_674, %int2_675, %int32_676 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %698 = torch.aten.view %670, %697 : !torch.vtensor<[1,1,64],f16>, !torch.list<int> -> !torch.vtensor<[1,1,2,32],f16>
    %int1_677 = torch.constant.int 1
    %int1_678 = torch.constant.int 1
    %int2_679 = torch.constant.int 2
    %int32_680 = torch.constant.int 32
    %699 = torch.prim.ListConstruct %int1_677, %int1_678, %int2_679, %int32_680 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %700 = torch.aten.view %676, %699 : !torch.vtensor<[1,1,64],f16>, !torch.list<int> -> !torch.vtensor<[1,1,2,32],f16>
    %int1_681 = torch.constant.int 1
    %int1_682 = torch.constant.int 1
    %int2_683 = torch.constant.int 2
    %int32_684 = torch.constant.int 32
    %701 = torch.prim.ListConstruct %int1_681, %int1_682, %int2_683, %int32_684 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %702 = torch.aten.view %686, %701 : !torch.vtensor<[1,1,64],f16>, !torch.list<int> -> !torch.vtensor<[1,1,2,32],f16>
    %int1_685 = torch.constant.int 1
    %int1_686 = torch.constant.int 1
    %int2_687 = torch.constant.int 2
    %int32_688 = torch.constant.int 32
    %703 = torch.prim.ListConstruct %int1_685, %int1_686, %int2_687, %int32_688 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %704 = torch.aten.view %692, %703 : !torch.vtensor<[1,1,64],f16>, !torch.list<int> -> !torch.vtensor<[1,1,2,32],f16>
    %int6_689 = torch.constant.int 6
    %705 = torch.prims.convert_element_type %694, %int6_689 : !torch.vtensor<[1,1,4,32],f16>, !torch.int -> !torch.vtensor<[1,1,4,32],f32>
    %706 = torch_c.to_builtin_tensor %705 : !torch.vtensor<[1,1,4,32],f32> -> tensor<1x1x4x32xf32>
    %707 = torch_c.to_builtin_tensor %152 : !torch.vtensor<[1,1,32],f32> -> tensor<1x1x32xf32>
    %708 = util.call @sharktank_rotary_embedding_1_1_4_32_f32(%706, %707) : (tensor<1x1x4x32xf32>, tensor<1x1x32xf32>) -> tensor<1x1x4x32xf32>
    %709 = torch_c.from_builtin_tensor %708 : tensor<1x1x4x32xf32> -> !torch.vtensor<[1,1,4,32],f32>
    %int5_690 = torch.constant.int 5
    %710 = torch.prims.convert_element_type %709, %int5_690 : !torch.vtensor<[1,1,4,32],f32>, !torch.int -> !torch.vtensor<[1,1,4,32],f16>
    %int6_691 = torch.constant.int 6
    %711 = torch.prims.convert_element_type %696, %int6_691 : !torch.vtensor<[1,1,4,32],f16>, !torch.int -> !torch.vtensor<[1,1,4,32],f32>
    %712 = torch_c.to_builtin_tensor %711 : !torch.vtensor<[1,1,4,32],f32> -> tensor<1x1x4x32xf32>
    %713 = torch_c.to_builtin_tensor %153 : !torch.vtensor<[1,1,32],f32> -> tensor<1x1x32xf32>
    %714 = util.call @sharktank_rotary_embedding_1_1_4_32_f32(%712, %713) : (tensor<1x1x4x32xf32>, tensor<1x1x32xf32>) -> tensor<1x1x4x32xf32>
    %715 = torch_c.from_builtin_tensor %714 : tensor<1x1x4x32xf32> -> !torch.vtensor<[1,1,4,32],f32>
    %int5_692 = torch.constant.int 5
    %716 = torch.prims.convert_element_type %715, %int5_692 : !torch.vtensor<[1,1,4,32],f32>, !torch.int -> !torch.vtensor<[1,1,4,32],f16>
    %int6_693 = torch.constant.int 6
    %717 = torch.prims.convert_element_type %698, %int6_693 : !torch.vtensor<[1,1,2,32],f16>, !torch.int -> !torch.vtensor<[1,1,2,32],f32>
    %718 = torch_c.to_builtin_tensor %717 : !torch.vtensor<[1,1,2,32],f32> -> tensor<1x1x2x32xf32>
    %719 = torch_c.to_builtin_tensor %152 : !torch.vtensor<[1,1,32],f32> -> tensor<1x1x32xf32>
    %720 = util.call @sharktank_rotary_embedding_1_1_2_32_f32(%718, %719) : (tensor<1x1x2x32xf32>, tensor<1x1x32xf32>) -> tensor<1x1x2x32xf32>
    %721 = torch_c.from_builtin_tensor %720 : tensor<1x1x2x32xf32> -> !torch.vtensor<[1,1,2,32],f32>
    %int5_694 = torch.constant.int 5
    %722 = torch.prims.convert_element_type %721, %int5_694 : !torch.vtensor<[1,1,2,32],f32>, !torch.int -> !torch.vtensor<[1,1,2,32],f16>
    %int6_695 = torch.constant.int 6
    %723 = torch.prims.convert_element_type %700, %int6_695 : !torch.vtensor<[1,1,2,32],f16>, !torch.int -> !torch.vtensor<[1,1,2,32],f32>
    %724 = torch_c.to_builtin_tensor %723 : !torch.vtensor<[1,1,2,32],f32> -> tensor<1x1x2x32xf32>
    %725 = torch_c.to_builtin_tensor %153 : !torch.vtensor<[1,1,32],f32> -> tensor<1x1x32xf32>
    %726 = util.call @sharktank_rotary_embedding_1_1_2_32_f32(%724, %725) : (tensor<1x1x2x32xf32>, tensor<1x1x32xf32>) -> tensor<1x1x2x32xf32>
    %727 = torch_c.from_builtin_tensor %726 : tensor<1x1x2x32xf32> -> !torch.vtensor<[1,1,2,32],f32>
    %int5_696 = torch.constant.int 5
    %728 = torch.prims.convert_element_type %727, %int5_696 : !torch.vtensor<[1,1,2,32],f32>, !torch.int -> !torch.vtensor<[1,1,2,32],f16>
    %int32_697 = torch.constant.int 32
    %729 = torch.aten.floor_divide.Scalar %103, %int32_697 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int32_698 = torch.constant.int 32
    %730 = torch.aten.floor_divide.Scalar %106, %int32_698 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int1_699 = torch.constant.int 1
    %731 = torch.aten.unsqueeze %729, %int1_699 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_700 = torch.constant.int 1
    %732 = torch.aten.unsqueeze %730, %int1_700 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_701 = torch.constant.int 1
    %false_702 = torch.constant.bool false
    %733 = torch.aten.gather %109, %int1_701, %731, %false_702 : !torch.vtensor<[1,?],si64>, !torch.int, !torch.vtensor<[1,1],si64>, !torch.bool -> !torch.vtensor<[1,1],si64>
    %int1_703 = torch.constant.int 1
    %false_704 = torch.constant.bool false
    %734 = torch.aten.gather %112, %int1_703, %732, %false_704 : !torch.vtensor<[1,?],si64>, !torch.int, !torch.vtensor<[1,1],si64>, !torch.bool -> !torch.vtensor<[1,1],si64>
    %int32_705 = torch.constant.int 32
    %735 = torch.aten.remainder.Scalar %103, %int32_705 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int32_706 = torch.constant.int 32
    %736 = torch.aten.remainder.Scalar %106, %int32_706 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int1_707 = torch.constant.int 1
    %737 = torch.aten.unsqueeze %735, %int1_707 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_708 = torch.constant.int 1
    %738 = torch.aten.unsqueeze %736, %int1_708 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %none_709 = torch.constant.none
    %739 = torch.aten.clone %32, %none_709 : !torch.vtensor<[],si64>, !torch.none -> !torch.vtensor<[],si64>
    %740 = torch.aten.detach %739 : !torch.vtensor<[],si64> -> !torch.vtensor<[],si64>
    %741 = torch.aten.detach %740 : !torch.vtensor<[],si64> -> !torch.vtensor<[],si64>
    %742 = torch.aten.detach %741 : !torch.vtensor<[],si64> -> !torch.vtensor<[],si64>
    %int0_710 = torch.constant.int 0
    %743 = torch.aten.unsqueeze %742, %int0_710 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %none_711 = torch.constant.none
    %744 = torch.aten.clone %33, %none_711 : !torch.vtensor<[],si64>, !torch.none -> !torch.vtensor<[],si64>
    %745 = torch.aten.detach %744 : !torch.vtensor<[],si64> -> !torch.vtensor<[],si64>
    %746 = torch.aten.detach %745 : !torch.vtensor<[],si64> -> !torch.vtensor<[],si64>
    %747 = torch.aten.detach %746 : !torch.vtensor<[],si64> -> !torch.vtensor<[],si64>
    %int0_712 = torch.constant.int 0
    %748 = torch.aten.unsqueeze %747, %int0_712 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int1_713 = torch.constant.int 1
    %int1_714 = torch.constant.int 1
    %749 = torch.prim.ListConstruct %int1_713, %int1_714 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_715 = torch.constant.int 1
    %int1_716 = torch.constant.int 1
    %750 = torch.prim.ListConstruct %int1_715, %int1_716 : (!torch.int, !torch.int) -> !torch.list<int>
    %int4_717 = torch.constant.int 4
    %int0_718 = torch.constant.int 0
    %cpu_719 = torch.constant.device "cpu"
    %false_720 = torch.constant.bool false
    %751 = torch.aten.empty_strided %749, %750, %int4_717, %int0_718, %cpu_719, %false_720 : !torch.list<int>, !torch.list<int>, !torch.int, !torch.int, !torch.Device, !torch.bool -> !torch.vtensor<[1,1],si64>
    %int1_721 = torch.constant.int 1
    %752 = torch.aten.fill.Scalar %751, %int1_721 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_722 = torch.constant.int 1
    %int1_723 = torch.constant.int 1
    %753 = torch.prim.ListConstruct %int1_722, %int1_723 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_724 = torch.constant.int 1
    %int1_725 = torch.constant.int 1
    %754 = torch.prim.ListConstruct %int1_724, %int1_725 : (!torch.int, !torch.int) -> !torch.list<int>
    %int4_726 = torch.constant.int 4
    %int0_727 = torch.constant.int 0
    %cpu_728 = torch.constant.device "cpu"
    %false_729 = torch.constant.bool false
    %755 = torch.aten.empty_strided %753, %754, %int4_726, %int0_727, %cpu_728, %false_729 : !torch.list<int>, !torch.list<int>, !torch.int, !torch.int, !torch.Device, !torch.bool -> !torch.vtensor<[1,1],si64>
    %int1_730 = torch.constant.int 1
    %756 = torch.aten.fill.Scalar %755, %int1_730 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_731 = torch.constant.int 1
    %int1_732 = torch.constant.int 1
    %757 = torch.prim.ListConstruct %int1_731, %int1_732 : (!torch.int, !torch.int) -> !torch.list<int>
    %758 = torch.aten.repeat %743, %757 : !torch.vtensor<[1],si64>, !torch.list<int> -> !torch.vtensor<[1,1],si64>
    %int1_733 = torch.constant.int 1
    %int1_734 = torch.constant.int 1
    %759 = torch.prim.ListConstruct %int1_733, %int1_734 : (!torch.int, !torch.int) -> !torch.list<int>
    %760 = torch.aten.repeat %748, %759 : !torch.vtensor<[1],si64>, !torch.list<int> -> !torch.vtensor<[1,1],si64>
    %int3_735 = torch.constant.int 3
    %761 = torch.aten.mul.Scalar %733, %int3_735 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int3_736 = torch.constant.int 3
    %762 = torch.aten.mul.Scalar %734, %int3_736 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_737 = torch.constant.int 1
    %763 = torch.aten.add.Tensor %761, %752, %int1_737 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_738 = torch.constant.int 1
    %764 = torch.aten.add.Tensor %762, %756, %int1_738 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int2_739 = torch.constant.int 2
    %765 = torch.aten.mul.Scalar %763, %int2_739 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int2_740 = torch.constant.int 2
    %766 = torch.aten.mul.Scalar %764, %int2_740 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_741 = torch.constant.int 1
    %767 = torch.aten.add.Tensor %765, %758, %int1_741 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_742 = torch.constant.int 1
    %768 = torch.aten.add.Tensor %766, %760, %int1_742 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int32_743 = torch.constant.int 32
    %769 = torch.aten.mul.Scalar %767, %int32_743 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int32_744 = torch.constant.int 32
    %770 = torch.aten.mul.Scalar %768, %int32_744 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_745 = torch.constant.int 1
    %771 = torch.aten.add.Tensor %769, %737, %int1_745 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_746 = torch.constant.int 1
    %772 = torch.aten.add.Tensor %770, %738, %int1_746 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int5_747 = torch.constant.int 5
    %773 = torch.prims.convert_element_type %722, %int5_747 : !torch.vtensor<[1,1,2,32],f16>, !torch.int -> !torch.vtensor<[1,1,2,32],f16>
    %int5_748 = torch.constant.int 5
    %774 = torch.prims.convert_element_type %728, %int5_748 : !torch.vtensor<[1,1,2,32],f16>, !torch.int -> !torch.vtensor<[1,1,2,32],f16>
    %int3_749 = torch.constant.int 3
    %int2_750 = torch.constant.int 2
    %int32_751 = torch.constant.int 32
    %int2_752 = torch.constant.int 2
    %int32_753 = torch.constant.int 32
    %775 = torch.prim.ListConstruct %77, %int3_749, %int2_750, %int32_751, %int2_752, %int32_753 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %776 = torch.aten.view %390, %775 : !torch.vtensor<[?,12288],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,2,32],f16>
    torch.bind_symbolic_shape %776, [%75], affine_map<()[s0] -> (s0, 3, 2, 32, 2, 32)> : !torch.vtensor<[?,3,2,32,2,32],f16>
    %int2_754 = torch.constant.int 2
    %int32_755 = torch.constant.int 32
    %777 = torch.prim.ListConstruct %268, %int2_754, %int32_755 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %778 = torch.aten.view %776, %777 : !torch.vtensor<[?,3,2,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,2,32],f16>
    torch.bind_symbolic_shape %778, [%75], affine_map<()[s0] -> (s0 * 192, 2, 32)> : !torch.vtensor<[?,2,32],f16>
    %779 = torch.prim.ListConstruct %771 : (!torch.vtensor<[1,1],si64>) -> !torch.list<optional<vtensor>>
    %false_756 = torch.constant.bool false
    %780 = torch.aten.index_put %778, %779, %773, %false_756 : !torch.vtensor<[?,2,32],f16>, !torch.list<optional<vtensor>>, !torch.vtensor<[1,1,2,32],f16>, !torch.bool -> !torch.vtensor<[?,2,32],f16>
    torch.bind_symbolic_shape %780, [%75], affine_map<()[s0] -> (s0 * 192, 2, 32)> : !torch.vtensor<[?,2,32],f16>
    %int3_757 = torch.constant.int 3
    %int2_758 = torch.constant.int 2
    %int32_759 = torch.constant.int 32
    %int2_760 = torch.constant.int 2
    %int32_761 = torch.constant.int 32
    %781 = torch.prim.ListConstruct %77, %int3_757, %int2_758, %int32_759, %int2_760, %int32_761 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %782 = torch.aten.view %780, %781 : !torch.vtensor<[?,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,2,32],f16>
    torch.bind_symbolic_shape %782, [%75], affine_map<()[s0] -> (s0, 3, 2, 32, 2, 32)> : !torch.vtensor<[?,3,2,32,2,32],f16>
    %int12288_762 = torch.constant.int 12288
    %783 = torch.prim.ListConstruct %77, %int12288_762 : (!torch.int, !torch.int) -> !torch.list<int>
    %784 = torch.aten.view %782, %783 : !torch.vtensor<[?,3,2,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,12288],f16>
    torch.bind_symbolic_shape %784, [%75], affine_map<()[s0] -> (s0, 12288)> : !torch.vtensor<[?,12288],f16>
    %int3_763 = torch.constant.int 3
    %int2_764 = torch.constant.int 2
    %int32_765 = torch.constant.int 32
    %int2_766 = torch.constant.int 2
    %int32_767 = torch.constant.int 32
    %785 = torch.prim.ListConstruct %77, %int3_763, %int2_764, %int32_765, %int2_766, %int32_767 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %786 = torch.aten.view %784, %785 : !torch.vtensor<[?,12288],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,2,32],f16>
    torch.bind_symbolic_shape %786, [%75], affine_map<()[s0] -> (s0, 3, 2, 32, 2, 32)> : !torch.vtensor<[?,3,2,32,2,32],f16>
    %int2_768 = torch.constant.int 2
    %int32_769 = torch.constant.int 32
    %787 = torch.prim.ListConstruct %268, %int2_768, %int32_769 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %788 = torch.aten.view %786, %787 : !torch.vtensor<[?,3,2,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,2,32],f16>
    torch.bind_symbolic_shape %788, [%75], affine_map<()[s0] -> (s0 * 192, 2, 32)> : !torch.vtensor<[?,2,32],f16>
    %int3_770 = torch.constant.int 3
    %int2_771 = torch.constant.int 2
    %int32_772 = torch.constant.int 32
    %int2_773 = torch.constant.int 2
    %int32_774 = torch.constant.int 32
    %789 = torch.prim.ListConstruct %77, %int3_770, %int2_771, %int32_772, %int2_773, %int32_774 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %790 = torch.aten.view %396, %789 : !torch.vtensor<[?,12288],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,2,32],f16>
    torch.bind_symbolic_shape %790, [%75], affine_map<()[s0] -> (s0, 3, 2, 32, 2, 32)> : !torch.vtensor<[?,3,2,32,2,32],f16>
    %int2_775 = torch.constant.int 2
    %int32_776 = torch.constant.int 32
    %791 = torch.prim.ListConstruct %268, %int2_775, %int32_776 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %792 = torch.aten.view %790, %791 : !torch.vtensor<[?,3,2,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,2,32],f16>
    torch.bind_symbolic_shape %792, [%75], affine_map<()[s0] -> (s0 * 192, 2, 32)> : !torch.vtensor<[?,2,32],f16>
    %793 = torch.prim.ListConstruct %772 : (!torch.vtensor<[1,1],si64>) -> !torch.list<optional<vtensor>>
    %false_777 = torch.constant.bool false
    %794 = torch.aten.index_put %792, %793, %774, %false_777 : !torch.vtensor<[?,2,32],f16>, !torch.list<optional<vtensor>>, !torch.vtensor<[1,1,2,32],f16>, !torch.bool -> !torch.vtensor<[?,2,32],f16>
    torch.bind_symbolic_shape %794, [%75], affine_map<()[s0] -> (s0 * 192, 2, 32)> : !torch.vtensor<[?,2,32],f16>
    %int3_778 = torch.constant.int 3
    %int2_779 = torch.constant.int 2
    %int32_780 = torch.constant.int 32
    %int2_781 = torch.constant.int 2
    %int32_782 = torch.constant.int 32
    %795 = torch.prim.ListConstruct %77, %int3_778, %int2_779, %int32_780, %int2_781, %int32_782 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %796 = torch.aten.view %794, %795 : !torch.vtensor<[?,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,2,32],f16>
    torch.bind_symbolic_shape %796, [%75], affine_map<()[s0] -> (s0, 3, 2, 32, 2, 32)> : !torch.vtensor<[?,3,2,32,2,32],f16>
    %int12288_783 = torch.constant.int 12288
    %797 = torch.prim.ListConstruct %77, %int12288_783 : (!torch.int, !torch.int) -> !torch.list<int>
    %798 = torch.aten.view %796, %797 : !torch.vtensor<[?,3,2,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,12288],f16>
    torch.bind_symbolic_shape %798, [%75], affine_map<()[s0] -> (s0, 12288)> : !torch.vtensor<[?,12288],f16>
    %int3_784 = torch.constant.int 3
    %int2_785 = torch.constant.int 2
    %int32_786 = torch.constant.int 32
    %int2_787 = torch.constant.int 2
    %int32_788 = torch.constant.int 32
    %799 = torch.prim.ListConstruct %77, %int3_784, %int2_785, %int32_786, %int2_787, %int32_788 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %800 = torch.aten.view %798, %799 : !torch.vtensor<[?,12288],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,2,32],f16>
    torch.bind_symbolic_shape %800, [%75], affine_map<()[s0] -> (s0, 3, 2, 32, 2, 32)> : !torch.vtensor<[?,3,2,32,2,32],f16>
    %int2_789 = torch.constant.int 2
    %int32_790 = torch.constant.int 32
    %801 = torch.prim.ListConstruct %268, %int2_789, %int32_790 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %802 = torch.aten.view %800, %801 : !torch.vtensor<[?,3,2,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,2,32],f16>
    torch.bind_symbolic_shape %802, [%75], affine_map<()[s0] -> (s0 * 192, 2, 32)> : !torch.vtensor<[?,2,32],f16>
    %int32_791 = torch.constant.int 32
    %803 = torch.aten.floor_divide.Scalar %103, %int32_791 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int32_792 = torch.constant.int 32
    %804 = torch.aten.floor_divide.Scalar %106, %int32_792 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int1_793 = torch.constant.int 1
    %805 = torch.aten.unsqueeze %803, %int1_793 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_794 = torch.constant.int 1
    %806 = torch.aten.unsqueeze %804, %int1_794 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_795 = torch.constant.int 1
    %false_796 = torch.constant.bool false
    %807 = torch.aten.gather %109, %int1_795, %805, %false_796 : !torch.vtensor<[1,?],si64>, !torch.int, !torch.vtensor<[1,1],si64>, !torch.bool -> !torch.vtensor<[1,1],si64>
    %int1_797 = torch.constant.int 1
    %false_798 = torch.constant.bool false
    %808 = torch.aten.gather %112, %int1_797, %806, %false_798 : !torch.vtensor<[1,?],si64>, !torch.int, !torch.vtensor<[1,1],si64>, !torch.bool -> !torch.vtensor<[1,1],si64>
    %int32_799 = torch.constant.int 32
    %809 = torch.aten.remainder.Scalar %103, %int32_799 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int32_800 = torch.constant.int 32
    %810 = torch.aten.remainder.Scalar %106, %int32_800 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int1_801 = torch.constant.int 1
    %811 = torch.aten.unsqueeze %809, %int1_801 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_802 = torch.constant.int 1
    %812 = torch.aten.unsqueeze %810, %int1_802 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %none_803 = torch.constant.none
    %813 = torch.aten.clone %34, %none_803 : !torch.vtensor<[],si64>, !torch.none -> !torch.vtensor<[],si64>
    %814 = torch.aten.detach %813 : !torch.vtensor<[],si64> -> !torch.vtensor<[],si64>
    %815 = torch.aten.detach %814 : !torch.vtensor<[],si64> -> !torch.vtensor<[],si64>
    %816 = torch.aten.detach %815 : !torch.vtensor<[],si64> -> !torch.vtensor<[],si64>
    %int0_804 = torch.constant.int 0
    %817 = torch.aten.unsqueeze %816, %int0_804 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %none_805 = torch.constant.none
    %818 = torch.aten.clone %35, %none_805 : !torch.vtensor<[],si64>, !torch.none -> !torch.vtensor<[],si64>
    %819 = torch.aten.detach %818 : !torch.vtensor<[],si64> -> !torch.vtensor<[],si64>
    %820 = torch.aten.detach %819 : !torch.vtensor<[],si64> -> !torch.vtensor<[],si64>
    %821 = torch.aten.detach %820 : !torch.vtensor<[],si64> -> !torch.vtensor<[],si64>
    %int0_806 = torch.constant.int 0
    %822 = torch.aten.unsqueeze %821, %int0_806 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int1_807 = torch.constant.int 1
    %int1_808 = torch.constant.int 1
    %823 = torch.prim.ListConstruct %int1_807, %int1_808 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_809 = torch.constant.int 1
    %int1_810 = torch.constant.int 1
    %824 = torch.prim.ListConstruct %int1_809, %int1_810 : (!torch.int, !torch.int) -> !torch.list<int>
    %int4_811 = torch.constant.int 4
    %int0_812 = torch.constant.int 0
    %cpu_813 = torch.constant.device "cpu"
    %false_814 = torch.constant.bool false
    %825 = torch.aten.empty_strided %823, %824, %int4_811, %int0_812, %cpu_813, %false_814 : !torch.list<int>, !torch.list<int>, !torch.int, !torch.int, !torch.Device, !torch.bool -> !torch.vtensor<[1,1],si64>
    %int1_815 = torch.constant.int 1
    %826 = torch.aten.fill.Scalar %825, %int1_815 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_816 = torch.constant.int 1
    %int1_817 = torch.constant.int 1
    %827 = torch.prim.ListConstruct %int1_816, %int1_817 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_818 = torch.constant.int 1
    %int1_819 = torch.constant.int 1
    %828 = torch.prim.ListConstruct %int1_818, %int1_819 : (!torch.int, !torch.int) -> !torch.list<int>
    %int4_820 = torch.constant.int 4
    %int0_821 = torch.constant.int 0
    %cpu_822 = torch.constant.device "cpu"
    %false_823 = torch.constant.bool false
    %829 = torch.aten.empty_strided %827, %828, %int4_820, %int0_821, %cpu_822, %false_823 : !torch.list<int>, !torch.list<int>, !torch.int, !torch.int, !torch.Device, !torch.bool -> !torch.vtensor<[1,1],si64>
    %int1_824 = torch.constant.int 1
    %830 = torch.aten.fill.Scalar %829, %int1_824 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_825 = torch.constant.int 1
    %int1_826 = torch.constant.int 1
    %831 = torch.prim.ListConstruct %int1_825, %int1_826 : (!torch.int, !torch.int) -> !torch.list<int>
    %832 = torch.aten.repeat %817, %831 : !torch.vtensor<[1],si64>, !torch.list<int> -> !torch.vtensor<[1,1],si64>
    %int1_827 = torch.constant.int 1
    %int1_828 = torch.constant.int 1
    %833 = torch.prim.ListConstruct %int1_827, %int1_828 : (!torch.int, !torch.int) -> !torch.list<int>
    %834 = torch.aten.repeat %822, %833 : !torch.vtensor<[1],si64>, !torch.list<int> -> !torch.vtensor<[1,1],si64>
    %int3_829 = torch.constant.int 3
    %835 = torch.aten.mul.Scalar %807, %int3_829 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int3_830 = torch.constant.int 3
    %836 = torch.aten.mul.Scalar %808, %int3_830 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_831 = torch.constant.int 1
    %837 = torch.aten.add.Tensor %835, %826, %int1_831 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_832 = torch.constant.int 1
    %838 = torch.aten.add.Tensor %836, %830, %int1_832 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int2_833 = torch.constant.int 2
    %839 = torch.aten.mul.Scalar %837, %int2_833 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int2_834 = torch.constant.int 2
    %840 = torch.aten.mul.Scalar %838, %int2_834 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_835 = torch.constant.int 1
    %841 = torch.aten.add.Tensor %839, %832, %int1_835 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_836 = torch.constant.int 1
    %842 = torch.aten.add.Tensor %840, %834, %int1_836 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int32_837 = torch.constant.int 32
    %843 = torch.aten.mul.Scalar %841, %int32_837 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int32_838 = torch.constant.int 32
    %844 = torch.aten.mul.Scalar %842, %int32_838 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_839 = torch.constant.int 1
    %845 = torch.aten.add.Tensor %843, %811, %int1_839 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_840 = torch.constant.int 1
    %846 = torch.aten.add.Tensor %844, %812, %int1_840 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int5_841 = torch.constant.int 5
    %847 = torch.prims.convert_element_type %702, %int5_841 : !torch.vtensor<[1,1,2,32],f16>, !torch.int -> !torch.vtensor<[1,1,2,32],f16>
    %int5_842 = torch.constant.int 5
    %848 = torch.prims.convert_element_type %704, %int5_842 : !torch.vtensor<[1,1,2,32],f16>, !torch.int -> !torch.vtensor<[1,1,2,32],f16>
    %849 = torch.prim.ListConstruct %845 : (!torch.vtensor<[1,1],si64>) -> !torch.list<optional<vtensor>>
    %false_843 = torch.constant.bool false
    %850 = torch.aten.index_put %788, %849, %847, %false_843 : !torch.vtensor<[?,2,32],f16>, !torch.list<optional<vtensor>>, !torch.vtensor<[1,1,2,32],f16>, !torch.bool -> !torch.vtensor<[?,2,32],f16>
    torch.bind_symbolic_shape %850, [%75], affine_map<()[s0] -> (s0 * 192, 2, 32)> : !torch.vtensor<[?,2,32],f16>
    %int3_844 = torch.constant.int 3
    %int2_845 = torch.constant.int 2
    %int32_846 = torch.constant.int 32
    %int2_847 = torch.constant.int 2
    %int32_848 = torch.constant.int 32
    %851 = torch.prim.ListConstruct %77, %int3_844, %int2_845, %int32_846, %int2_847, %int32_848 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %852 = torch.aten.view %850, %851 : !torch.vtensor<[?,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,2,32],f16>
    torch.bind_symbolic_shape %852, [%75], affine_map<()[s0] -> (s0, 3, 2, 32, 2, 32)> : !torch.vtensor<[?,3,2,32,2,32],f16>
    %int12288_849 = torch.constant.int 12288
    %853 = torch.prim.ListConstruct %77, %int12288_849 : (!torch.int, !torch.int) -> !torch.list<int>
    %854 = torch.aten.view %852, %853 : !torch.vtensor<[?,3,2,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,12288],f16>
    torch.bind_symbolic_shape %854, [%75], affine_map<()[s0] -> (s0, 12288)> : !torch.vtensor<[?,12288],f16>
    %855 = torch.prim.ListConstruct %846 : (!torch.vtensor<[1,1],si64>) -> !torch.list<optional<vtensor>>
    %false_850 = torch.constant.bool false
    %856 = torch.aten.index_put %802, %855, %848, %false_850 : !torch.vtensor<[?,2,32],f16>, !torch.list<optional<vtensor>>, !torch.vtensor<[1,1,2,32],f16>, !torch.bool -> !torch.vtensor<[?,2,32],f16>
    torch.bind_symbolic_shape %856, [%75], affine_map<()[s0] -> (s0 * 192, 2, 32)> : !torch.vtensor<[?,2,32],f16>
    %int3_851 = torch.constant.int 3
    %int2_852 = torch.constant.int 2
    %int32_853 = torch.constant.int 32
    %int2_854 = torch.constant.int 2
    %int32_855 = torch.constant.int 32
    %857 = torch.prim.ListConstruct %77, %int3_851, %int2_852, %int32_853, %int2_854, %int32_855 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %858 = torch.aten.view %856, %857 : !torch.vtensor<[?,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,2,32],f16>
    torch.bind_symbolic_shape %858, [%75], affine_map<()[s0] -> (s0, 3, 2, 32, 2, 32)> : !torch.vtensor<[?,3,2,32,2,32],f16>
    %int12288_856 = torch.constant.int 12288
    %859 = torch.prim.ListConstruct %77, %int12288_856 : (!torch.int, !torch.int) -> !torch.list<int>
    %860 = torch.aten.view %858, %859 : !torch.vtensor<[?,3,2,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,12288],f16>
    torch.bind_symbolic_shape %860, [%75], affine_map<()[s0] -> (s0, 12288)> : !torch.vtensor<[?,12288],f16>
    %int1_857 = torch.constant.int 1
    %861 = torch.prim.ListConstruct %int1_857, %76 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_858 = torch.constant.int 1
    %862 = torch.prim.ListConstruct %76, %int1_858 : (!torch.int, !torch.int) -> !torch.list<int>
    %int4_859 = torch.constant.int 4
    %int0_860 = torch.constant.int 0
    %cpu_861 = torch.constant.device "cpu"
    %false_862 = torch.constant.bool false
    %863 = torch.aten.empty_strided %861, %862, %int4_859, %int0_860, %cpu_861, %false_862 : !torch.list<int>, !torch.list<int>, !torch.int, !torch.int, !torch.Device, !torch.bool -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %863, [%74], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %int1_863 = torch.constant.int 1
    %864 = torch.aten.fill.Scalar %863, %int1_863 : !torch.vtensor<[1,?],si64>, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %864, [%74], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %int3_864 = torch.constant.int 3
    %865 = torch.aten.mul.Scalar %109, %int3_864 : !torch.vtensor<[1,?],si64>, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %865, [%74], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %int3_865 = torch.constant.int 3
    %866 = torch.aten.mul.Scalar %112, %int3_865 : !torch.vtensor<[1,?],si64>, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %866, [%74], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %867 = torch_c.to_builtin_tensor %864 : !torch.vtensor<[1,?],si64> -> tensor<1x?xi64>
    %c1_866 = arith.constant 1 : index
    %dim_867 = tensor.dim %867, %c1_866 : tensor<1x?xi64>
    %868 = flow.tensor.transfer %867 : tensor<1x?xi64>{%dim_867} to #hal.device.promise<@__device_0>
    %869 = torch_c.from_builtin_tensor %868 : tensor<1x?xi64> -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %869, [%74], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %870 = torch_c.to_builtin_tensor %864 : !torch.vtensor<[1,?],si64> -> tensor<1x?xi64>
    %c1_868 = arith.constant 1 : index
    %dim_869 = tensor.dim %870, %c1_868 : tensor<1x?xi64>
    %871 = flow.tensor.transfer %870 : tensor<1x?xi64>{%dim_869} to #hal.device.promise<@__device_1>
    %872 = torch_c.from_builtin_tensor %871 : tensor<1x?xi64> -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %872, [%74], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %int1_870 = torch.constant.int 1
    %873 = torch.aten.add.Tensor %865, %869, %int1_870 : !torch.vtensor<[1,?],si64>, !torch.vtensor<[1,?],si64>, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %873, [%74], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %int1_871 = torch.constant.int 1
    %874 = torch.aten.add.Tensor %866, %872, %int1_871 : !torch.vtensor<[1,?],si64>, !torch.vtensor<[1,?],si64>, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %874, [%74], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %875 = torch.prim.ListConstruct %76 : (!torch.int) -> !torch.list<int>
    %876 = torch.aten.view %873, %875 : !torch.vtensor<[1,?],si64>, !torch.list<int> -> !torch.vtensor<[?],si64>
    torch.bind_symbolic_shape %876, [%74], affine_map<()[s0] -> (s0)> : !torch.vtensor<[?],si64>
    %877 = torch.prim.ListConstruct %76 : (!torch.int) -> !torch.list<int>
    %878 = torch.aten.view %874, %877 : !torch.vtensor<[1,?],si64>, !torch.list<int> -> !torch.vtensor<[?],si64>
    torch.bind_symbolic_shape %878, [%74], affine_map<()[s0] -> (s0)> : !torch.vtensor<[?],si64>
    %int3_872 = torch.constant.int 3
    %int2_873 = torch.constant.int 2
    %int32_874 = torch.constant.int 32
    %int2_875 = torch.constant.int 2
    %int32_876 = torch.constant.int 32
    %879 = torch.prim.ListConstruct %77, %int3_872, %int2_873, %int32_874, %int2_875, %int32_876 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %880 = torch.aten.view %854, %879 : !torch.vtensor<[?,12288],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,2,32],f16>
    torch.bind_symbolic_shape %880, [%75], affine_map<()[s0] -> (s0, 3, 2, 32, 2, 32)> : !torch.vtensor<[?,3,2,32,2,32],f16>
    %int2_877 = torch.constant.int 2
    %int32_878 = torch.constant.int 32
    %int2_879 = torch.constant.int 2
    %int32_880 = torch.constant.int 32
    %881 = torch.prim.ListConstruct %266, %int2_877, %int32_878, %int2_879, %int32_880 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %882 = torch.aten.view %880, %881 : !torch.vtensor<[?,3,2,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,2,32,2,32],f16>
    torch.bind_symbolic_shape %882, [%75], affine_map<()[s0] -> (s0 * 3, 2, 32, 2, 32)> : !torch.vtensor<[?,2,32,2,32],f16>
    %int0_881 = torch.constant.int 0
    %883 = torch.aten.index_select %882, %int0_881, %876 : !torch.vtensor<[?,2,32,2,32],f16>, !torch.int, !torch.vtensor<[?],si64> -> !torch.vtensor<[?,2,32,2,32],f16>
    torch.bind_symbolic_shape %883, [%74], affine_map<()[s0] -> (s0, 2, 32, 2, 32)> : !torch.vtensor<[?,2,32,2,32],f16>
    %int3_882 = torch.constant.int 3
    %int2_883 = torch.constant.int 2
    %int32_884 = torch.constant.int 32
    %int2_885 = torch.constant.int 2
    %int32_886 = torch.constant.int 32
    %884 = torch.prim.ListConstruct %77, %int3_882, %int2_883, %int32_884, %int2_885, %int32_886 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %885 = torch.aten.view %860, %884 : !torch.vtensor<[?,12288],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,2,32],f16>
    torch.bind_symbolic_shape %885, [%75], affine_map<()[s0] -> (s0, 3, 2, 32, 2, 32)> : !torch.vtensor<[?,3,2,32,2,32],f16>
    %int2_887 = torch.constant.int 2
    %int32_888 = torch.constant.int 32
    %int2_889 = torch.constant.int 2
    %int32_890 = torch.constant.int 32
    %886 = torch.prim.ListConstruct %266, %int2_887, %int32_888, %int2_889, %int32_890 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %887 = torch.aten.view %885, %886 : !torch.vtensor<[?,3,2,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,2,32,2,32],f16>
    torch.bind_symbolic_shape %887, [%75], affine_map<()[s0] -> (s0 * 3, 2, 32, 2, 32)> : !torch.vtensor<[?,2,32,2,32],f16>
    %int0_891 = torch.constant.int 0
    %888 = torch.aten.index_select %887, %int0_891, %878 : !torch.vtensor<[?,2,32,2,32],f16>, !torch.int, !torch.vtensor<[?],si64> -> !torch.vtensor<[?,2,32,2,32],f16>
    torch.bind_symbolic_shape %888, [%74], affine_map<()[s0] -> (s0, 2, 32, 2, 32)> : !torch.vtensor<[?,2,32,2,32],f16>
    %int1_892 = torch.constant.int 1
    %int2_893 = torch.constant.int 2
    %int32_894 = torch.constant.int 32
    %int2_895 = torch.constant.int 2
    %int32_896 = torch.constant.int 32
    %889 = torch.prim.ListConstruct %int1_892, %76, %int2_893, %int32_894, %int2_895, %int32_896 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %890 = torch.aten.view %883, %889 : !torch.vtensor<[?,2,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,2,32,2,32],f16>
    torch.bind_symbolic_shape %890, [%74], affine_map<()[s0] -> (1, s0, 2, 32, 2, 32)> : !torch.vtensor<[1,?,2,32,2,32],f16>
    %int1_897 = torch.constant.int 1
    %int2_898 = torch.constant.int 2
    %int32_899 = torch.constant.int 32
    %int2_900 = torch.constant.int 2
    %int32_901 = torch.constant.int 32
    %891 = torch.prim.ListConstruct %int1_897, %76, %int2_898, %int32_899, %int2_900, %int32_901 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %892 = torch.aten.view %888, %891 : !torch.vtensor<[?,2,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,2,32,2,32],f16>
    torch.bind_symbolic_shape %892, [%74], affine_map<()[s0] -> (1, s0, 2, 32, 2, 32)> : !torch.vtensor<[1,?,2,32,2,32],f16>
    %int0_902 = torch.constant.int 0
    %int0_903 = torch.constant.int 0
    %int9223372036854775807_904 = torch.constant.int 9223372036854775807
    %int1_905 = torch.constant.int 1
    %893 = torch.aten.slice.Tensor %890, %int0_902, %int0_903, %int9223372036854775807_904, %int1_905 : !torch.vtensor<[1,?,2,32,2,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,2,32,2,32],f16>
    torch.bind_symbolic_shape %893, [%74], affine_map<()[s0] -> (1, s0, 2, 32, 2, 32)> : !torch.vtensor<[1,?,2,32,2,32],f16>
    %int1_906 = torch.constant.int 1
    %int0_907 = torch.constant.int 0
    %int9223372036854775807_908 = torch.constant.int 9223372036854775807
    %int1_909 = torch.constant.int 1
    %894 = torch.aten.slice.Tensor %893, %int1_906, %int0_907, %int9223372036854775807_908, %int1_909 : !torch.vtensor<[1,?,2,32,2,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,2,32,2,32],f16>
    torch.bind_symbolic_shape %894, [%74], affine_map<()[s0] -> (1, s0, 2, 32, 2, 32)> : !torch.vtensor<[1,?,2,32,2,32],f16>
    %int2_910 = torch.constant.int 2
    %int0_911 = torch.constant.int 0
    %895 = torch.aten.select.int %894, %int2_910, %int0_911 : !torch.vtensor<[1,?,2,32,2,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,?,32,2,32],f16>
    torch.bind_symbolic_shape %895, [%74], affine_map<()[s0] -> (1, s0, 32, 2, 32)> : !torch.vtensor<[1,?,32,2,32],f16>
    %int2_912 = torch.constant.int 2
    %int0_913 = torch.constant.int 0
    %int1_914 = torch.constant.int 1
    %896 = torch.aten.slice.Tensor %895, %int2_912, %int0_913, %78, %int1_914 : !torch.vtensor<[1,?,32,2,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32,2,32],f16>
    torch.bind_symbolic_shape %896, [%74], affine_map<()[s0] -> (1, s0, 32, 2, 32)> : !torch.vtensor<[1,?,32,2,32],f16>
    %int0_915 = torch.constant.int 0
    %int0_916 = torch.constant.int 0
    %int9223372036854775807_917 = torch.constant.int 9223372036854775807
    %int1_918 = torch.constant.int 1
    %897 = torch.aten.slice.Tensor %892, %int0_915, %int0_916, %int9223372036854775807_917, %int1_918 : !torch.vtensor<[1,?,2,32,2,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,2,32,2,32],f16>
    torch.bind_symbolic_shape %897, [%74], affine_map<()[s0] -> (1, s0, 2, 32, 2, 32)> : !torch.vtensor<[1,?,2,32,2,32],f16>
    %int1_919 = torch.constant.int 1
    %int0_920 = torch.constant.int 0
    %int9223372036854775807_921 = torch.constant.int 9223372036854775807
    %int1_922 = torch.constant.int 1
    %898 = torch.aten.slice.Tensor %897, %int1_919, %int0_920, %int9223372036854775807_921, %int1_922 : !torch.vtensor<[1,?,2,32,2,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,2,32,2,32],f16>
    torch.bind_symbolic_shape %898, [%74], affine_map<()[s0] -> (1, s0, 2, 32, 2, 32)> : !torch.vtensor<[1,?,2,32,2,32],f16>
    %int2_923 = torch.constant.int 2
    %int0_924 = torch.constant.int 0
    %899 = torch.aten.select.int %898, %int2_923, %int0_924 : !torch.vtensor<[1,?,2,32,2,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,?,32,2,32],f16>
    torch.bind_symbolic_shape %899, [%74], affine_map<()[s0] -> (1, s0, 32, 2, 32)> : !torch.vtensor<[1,?,32,2,32],f16>
    %int2_925 = torch.constant.int 2
    %int0_926 = torch.constant.int 0
    %int1_927 = torch.constant.int 1
    %900 = torch.aten.slice.Tensor %899, %int2_925, %int0_926, %78, %int1_927 : !torch.vtensor<[1,?,32,2,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32,2,32],f16>
    torch.bind_symbolic_shape %900, [%74], affine_map<()[s0] -> (1, s0, 32, 2, 32)> : !torch.vtensor<[1,?,32,2,32],f16>
    %int0_928 = torch.constant.int 0
    %901 = torch.aten.clone %896, %int0_928 : !torch.vtensor<[1,?,32,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,32,2,32],f16>
    torch.bind_symbolic_shape %901, [%74], affine_map<()[s0] -> (1, s0, 32, 2, 32)> : !torch.vtensor<[1,?,32,2,32],f16>
    %int1_929 = torch.constant.int 1
    %int2_930 = torch.constant.int 2
    %int32_931 = torch.constant.int 32
    %902 = torch.prim.ListConstruct %int1_929, %78, %int2_930, %int32_931 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %903 = torch.aten._unsafe_view %901, %902 : !torch.vtensor<[1,?,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,2,32],f16>
    torch.bind_symbolic_shape %903, [%74], affine_map<()[s0] -> (1, s0 * 32, 2, 32)> : !torch.vtensor<[1,?,2,32],f16>
    %int0_932 = torch.constant.int 0
    %904 = torch.aten.clone %900, %int0_932 : !torch.vtensor<[1,?,32,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,32,2,32],f16>
    torch.bind_symbolic_shape %904, [%74], affine_map<()[s0] -> (1, s0, 32, 2, 32)> : !torch.vtensor<[1,?,32,2,32],f16>
    %int1_933 = torch.constant.int 1
    %int2_934 = torch.constant.int 2
    %int32_935 = torch.constant.int 32
    %905 = torch.prim.ListConstruct %int1_933, %78, %int2_934, %int32_935 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %906 = torch.aten._unsafe_view %904, %905 : !torch.vtensor<[1,?,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,2,32],f16>
    torch.bind_symbolic_shape %906, [%74], affine_map<()[s0] -> (1, s0 * 32, 2, 32)> : !torch.vtensor<[1,?,2,32],f16>
    %int0_936 = torch.constant.int 0
    %int0_937 = torch.constant.int 0
    %int9223372036854775807_938 = torch.constant.int 9223372036854775807
    %int1_939 = torch.constant.int 1
    %907 = torch.aten.slice.Tensor %903, %int0_936, %int0_937, %int9223372036854775807_938, %int1_939 : !torch.vtensor<[1,?,2,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,2,32],f16>
    torch.bind_symbolic_shape %907, [%74], affine_map<()[s0] -> (1, s0 * 32, 2, 32)> : !torch.vtensor<[1,?,2,32],f16>
    %int0_940 = torch.constant.int 0
    %int0_941 = torch.constant.int 0
    %int9223372036854775807_942 = torch.constant.int 9223372036854775807
    %int1_943 = torch.constant.int 1
    %908 = torch.aten.slice.Tensor %906, %int0_940, %int0_941, %int9223372036854775807_942, %int1_943 : !torch.vtensor<[1,?,2,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,2,32],f16>
    torch.bind_symbolic_shape %908, [%74], affine_map<()[s0] -> (1, s0 * 32, 2, 32)> : !torch.vtensor<[1,?,2,32],f16>
    %int0_944 = torch.constant.int 0
    %int0_945 = torch.constant.int 0
    %int9223372036854775807_946 = torch.constant.int 9223372036854775807
    %int1_947 = torch.constant.int 1
    %909 = torch.aten.slice.Tensor %890, %int0_944, %int0_945, %int9223372036854775807_946, %int1_947 : !torch.vtensor<[1,?,2,32,2,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,2,32,2,32],f16>
    torch.bind_symbolic_shape %909, [%74], affine_map<()[s0] -> (1, s0, 2, 32, 2, 32)> : !torch.vtensor<[1,?,2,32,2,32],f16>
    %int1_948 = torch.constant.int 1
    %int0_949 = torch.constant.int 0
    %int9223372036854775807_950 = torch.constant.int 9223372036854775807
    %int1_951 = torch.constant.int 1
    %910 = torch.aten.slice.Tensor %909, %int1_948, %int0_949, %int9223372036854775807_950, %int1_951 : !torch.vtensor<[1,?,2,32,2,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,2,32,2,32],f16>
    torch.bind_symbolic_shape %910, [%74], affine_map<()[s0] -> (1, s0, 2, 32, 2, 32)> : !torch.vtensor<[1,?,2,32,2,32],f16>
    %int2_952 = torch.constant.int 2
    %int1_953 = torch.constant.int 1
    %911 = torch.aten.select.int %910, %int2_952, %int1_953 : !torch.vtensor<[1,?,2,32,2,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,?,32,2,32],f16>
    torch.bind_symbolic_shape %911, [%74], affine_map<()[s0] -> (1, s0, 32, 2, 32)> : !torch.vtensor<[1,?,32,2,32],f16>
    %int2_954 = torch.constant.int 2
    %int0_955 = torch.constant.int 0
    %int1_956 = torch.constant.int 1
    %912 = torch.aten.slice.Tensor %911, %int2_954, %int0_955, %78, %int1_956 : !torch.vtensor<[1,?,32,2,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32,2,32],f16>
    torch.bind_symbolic_shape %912, [%74], affine_map<()[s0] -> (1, s0, 32, 2, 32)> : !torch.vtensor<[1,?,32,2,32],f16>
    %int0_957 = torch.constant.int 0
    %int0_958 = torch.constant.int 0
    %int9223372036854775807_959 = torch.constant.int 9223372036854775807
    %int1_960 = torch.constant.int 1
    %913 = torch.aten.slice.Tensor %892, %int0_957, %int0_958, %int9223372036854775807_959, %int1_960 : !torch.vtensor<[1,?,2,32,2,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,2,32,2,32],f16>
    torch.bind_symbolic_shape %913, [%74], affine_map<()[s0] -> (1, s0, 2, 32, 2, 32)> : !torch.vtensor<[1,?,2,32,2,32],f16>
    %int1_961 = torch.constant.int 1
    %int0_962 = torch.constant.int 0
    %int9223372036854775807_963 = torch.constant.int 9223372036854775807
    %int1_964 = torch.constant.int 1
    %914 = torch.aten.slice.Tensor %913, %int1_961, %int0_962, %int9223372036854775807_963, %int1_964 : !torch.vtensor<[1,?,2,32,2,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,2,32,2,32],f16>
    torch.bind_symbolic_shape %914, [%74], affine_map<()[s0] -> (1, s0, 2, 32, 2, 32)> : !torch.vtensor<[1,?,2,32,2,32],f16>
    %int2_965 = torch.constant.int 2
    %int1_966 = torch.constant.int 1
    %915 = torch.aten.select.int %914, %int2_965, %int1_966 : !torch.vtensor<[1,?,2,32,2,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,?,32,2,32],f16>
    torch.bind_symbolic_shape %915, [%74], affine_map<()[s0] -> (1, s0, 32, 2, 32)> : !torch.vtensor<[1,?,32,2,32],f16>
    %int2_967 = torch.constant.int 2
    %int0_968 = torch.constant.int 0
    %int1_969 = torch.constant.int 1
    %916 = torch.aten.slice.Tensor %915, %int2_967, %int0_968, %78, %int1_969 : !torch.vtensor<[1,?,32,2,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32,2,32],f16>
    torch.bind_symbolic_shape %916, [%74], affine_map<()[s0] -> (1, s0, 32, 2, 32)> : !torch.vtensor<[1,?,32,2,32],f16>
    %int0_970 = torch.constant.int 0
    %917 = torch.aten.clone %912, %int0_970 : !torch.vtensor<[1,?,32,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,32,2,32],f16>
    torch.bind_symbolic_shape %917, [%74], affine_map<()[s0] -> (1, s0, 32, 2, 32)> : !torch.vtensor<[1,?,32,2,32],f16>
    %int1_971 = torch.constant.int 1
    %int2_972 = torch.constant.int 2
    %int32_973 = torch.constant.int 32
    %918 = torch.prim.ListConstruct %int1_971, %78, %int2_972, %int32_973 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %919 = torch.aten._unsafe_view %917, %918 : !torch.vtensor<[1,?,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,2,32],f16>
    torch.bind_symbolic_shape %919, [%74], affine_map<()[s0] -> (1, s0 * 32, 2, 32)> : !torch.vtensor<[1,?,2,32],f16>
    %int0_974 = torch.constant.int 0
    %920 = torch.aten.clone %916, %int0_974 : !torch.vtensor<[1,?,32,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,32,2,32],f16>
    torch.bind_symbolic_shape %920, [%74], affine_map<()[s0] -> (1, s0, 32, 2, 32)> : !torch.vtensor<[1,?,32,2,32],f16>
    %int1_975 = torch.constant.int 1
    %int2_976 = torch.constant.int 2
    %int32_977 = torch.constant.int 32
    %921 = torch.prim.ListConstruct %int1_975, %78, %int2_976, %int32_977 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %922 = torch.aten._unsafe_view %920, %921 : !torch.vtensor<[1,?,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,2,32],f16>
    torch.bind_symbolic_shape %922, [%74], affine_map<()[s0] -> (1, s0 * 32, 2, 32)> : !torch.vtensor<[1,?,2,32],f16>
    %int0_978 = torch.constant.int 0
    %int0_979 = torch.constant.int 0
    %int9223372036854775807_980 = torch.constant.int 9223372036854775807
    %int1_981 = torch.constant.int 1
    %923 = torch.aten.slice.Tensor %919, %int0_978, %int0_979, %int9223372036854775807_980, %int1_981 : !torch.vtensor<[1,?,2,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,2,32],f16>
    torch.bind_symbolic_shape %923, [%74], affine_map<()[s0] -> (1, s0 * 32, 2, 32)> : !torch.vtensor<[1,?,2,32],f16>
    %int0_982 = torch.constant.int 0
    %int0_983 = torch.constant.int 0
    %int9223372036854775807_984 = torch.constant.int 9223372036854775807
    %int1_985 = torch.constant.int 1
    %924 = torch.aten.slice.Tensor %922, %int0_982, %int0_983, %int9223372036854775807_984, %int1_985 : !torch.vtensor<[1,?,2,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,2,32],f16>
    torch.bind_symbolic_shape %924, [%74], affine_map<()[s0] -> (1, s0 * 32, 2, 32)> : !torch.vtensor<[1,?,2,32],f16>
    %int-2_986 = torch.constant.int -2
    %925 = torch.aten.unsqueeze %907, %int-2_986 : !torch.vtensor<[1,?,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,2,1,32],f16>
    torch.bind_symbolic_shape %925, [%74], affine_map<()[s0] -> (1, s0 * 32, 2, 1, 32)> : !torch.vtensor<[1,?,2,1,32],f16>
    %int-2_987 = torch.constant.int -2
    %926 = torch.aten.unsqueeze %908, %int-2_987 : !torch.vtensor<[1,?,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,2,1,32],f16>
    torch.bind_symbolic_shape %926, [%74], affine_map<()[s0] -> (1, s0 * 32, 2, 1, 32)> : !torch.vtensor<[1,?,2,1,32],f16>
    %int1_988 = torch.constant.int 1
    %int2_989 = torch.constant.int 2
    %int2_990 = torch.constant.int 2
    %int32_991 = torch.constant.int 32
    %927 = torch.prim.ListConstruct %int1_988, %78, %int2_989, %int2_990, %int32_991 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_992 = torch.constant.bool false
    %928 = torch.aten.expand %925, %927, %false_992 : !torch.vtensor<[1,?,2,1,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,?,2,2,32],f16>
    torch.bind_symbolic_shape %928, [%74], affine_map<()[s0] -> (1, s0 * 32, 2, 2, 32)> : !torch.vtensor<[1,?,2,2,32],f16>
    %int1_993 = torch.constant.int 1
    %int2_994 = torch.constant.int 2
    %int2_995 = torch.constant.int 2
    %int32_996 = torch.constant.int 32
    %929 = torch.prim.ListConstruct %int1_993, %78, %int2_994, %int2_995, %int32_996 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_997 = torch.constant.bool false
    %930 = torch.aten.expand %926, %929, %false_997 : !torch.vtensor<[1,?,2,1,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,?,2,2,32],f16>
    torch.bind_symbolic_shape %930, [%74], affine_map<()[s0] -> (1, s0 * 32, 2, 2, 32)> : !torch.vtensor<[1,?,2,2,32],f16>
    %int0_998 = torch.constant.int 0
    %931 = torch.aten.clone %928, %int0_998 : !torch.vtensor<[1,?,2,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,2,2,32],f16>
    torch.bind_symbolic_shape %931, [%74], affine_map<()[s0] -> (1, s0 * 32, 2, 2, 32)> : !torch.vtensor<[1,?,2,2,32],f16>
    %int1_999 = torch.constant.int 1
    %int4_1000 = torch.constant.int 4
    %int32_1001 = torch.constant.int 32
    %932 = torch.prim.ListConstruct %int1_999, %78, %int4_1000, %int32_1001 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %933 = torch.aten._unsafe_view %931, %932 : !torch.vtensor<[1,?,2,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %933, [%74], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int0_1002 = torch.constant.int 0
    %934 = torch.aten.clone %930, %int0_1002 : !torch.vtensor<[1,?,2,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,2,2,32],f16>
    torch.bind_symbolic_shape %934, [%74], affine_map<()[s0] -> (1, s0 * 32, 2, 2, 32)> : !torch.vtensor<[1,?,2,2,32],f16>
    %int1_1003 = torch.constant.int 1
    %int4_1004 = torch.constant.int 4
    %int32_1005 = torch.constant.int 32
    %935 = torch.prim.ListConstruct %int1_1003, %78, %int4_1004, %int32_1005 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %936 = torch.aten._unsafe_view %934, %935 : !torch.vtensor<[1,?,2,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %936, [%74], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int-2_1006 = torch.constant.int -2
    %937 = torch.aten.unsqueeze %923, %int-2_1006 : !torch.vtensor<[1,?,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,2,1,32],f16>
    torch.bind_symbolic_shape %937, [%74], affine_map<()[s0] -> (1, s0 * 32, 2, 1, 32)> : !torch.vtensor<[1,?,2,1,32],f16>
    %int-2_1007 = torch.constant.int -2
    %938 = torch.aten.unsqueeze %924, %int-2_1007 : !torch.vtensor<[1,?,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,2,1,32],f16>
    torch.bind_symbolic_shape %938, [%74], affine_map<()[s0] -> (1, s0 * 32, 2, 1, 32)> : !torch.vtensor<[1,?,2,1,32],f16>
    %int1_1008 = torch.constant.int 1
    %int2_1009 = torch.constant.int 2
    %int2_1010 = torch.constant.int 2
    %int32_1011 = torch.constant.int 32
    %939 = torch.prim.ListConstruct %int1_1008, %78, %int2_1009, %int2_1010, %int32_1011 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_1012 = torch.constant.bool false
    %940 = torch.aten.expand %937, %939, %false_1012 : !torch.vtensor<[1,?,2,1,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,?,2,2,32],f16>
    torch.bind_symbolic_shape %940, [%74], affine_map<()[s0] -> (1, s0 * 32, 2, 2, 32)> : !torch.vtensor<[1,?,2,2,32],f16>
    %int1_1013 = torch.constant.int 1
    %int2_1014 = torch.constant.int 2
    %int2_1015 = torch.constant.int 2
    %int32_1016 = torch.constant.int 32
    %941 = torch.prim.ListConstruct %int1_1013, %78, %int2_1014, %int2_1015, %int32_1016 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_1017 = torch.constant.bool false
    %942 = torch.aten.expand %938, %941, %false_1017 : !torch.vtensor<[1,?,2,1,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,?,2,2,32],f16>
    torch.bind_symbolic_shape %942, [%74], affine_map<()[s0] -> (1, s0 * 32, 2, 2, 32)> : !torch.vtensor<[1,?,2,2,32],f16>
    %int0_1018 = torch.constant.int 0
    %943 = torch.aten.clone %940, %int0_1018 : !torch.vtensor<[1,?,2,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,2,2,32],f16>
    torch.bind_symbolic_shape %943, [%74], affine_map<()[s0] -> (1, s0 * 32, 2, 2, 32)> : !torch.vtensor<[1,?,2,2,32],f16>
    %int1_1019 = torch.constant.int 1
    %int4_1020 = torch.constant.int 4
    %int32_1021 = torch.constant.int 32
    %944 = torch.prim.ListConstruct %int1_1019, %78, %int4_1020, %int32_1021 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %945 = torch.aten._unsafe_view %943, %944 : !torch.vtensor<[1,?,2,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %945, [%74], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int0_1022 = torch.constant.int 0
    %946 = torch.aten.clone %942, %int0_1022 : !torch.vtensor<[1,?,2,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,2,2,32],f16>
    torch.bind_symbolic_shape %946, [%74], affine_map<()[s0] -> (1, s0 * 32, 2, 2, 32)> : !torch.vtensor<[1,?,2,2,32],f16>
    %int1_1023 = torch.constant.int 1
    %int4_1024 = torch.constant.int 4
    %int32_1025 = torch.constant.int 32
    %947 = torch.prim.ListConstruct %int1_1023, %78, %int4_1024, %int32_1025 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %948 = torch.aten._unsafe_view %946, %947 : !torch.vtensor<[1,?,2,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %948, [%74], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int1_1026 = torch.constant.int 1
    %int2_1027 = torch.constant.int 2
    %949 = torch.aten.transpose.int %710, %int1_1026, %int2_1027 : !torch.vtensor<[1,1,4,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,4,1,32],f16>
    %int1_1028 = torch.constant.int 1
    %int2_1029 = torch.constant.int 2
    %950 = torch.aten.transpose.int %716, %int1_1028, %int2_1029 : !torch.vtensor<[1,1,4,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,4,1,32],f16>
    %int1_1030 = torch.constant.int 1
    %int2_1031 = torch.constant.int 2
    %951 = torch.aten.transpose.int %933, %int1_1030, %int2_1031 : !torch.vtensor<[1,?,4,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,4,?,32],f16>
    torch.bind_symbolic_shape %951, [%74], affine_map<()[s0] -> (1, 4, s0 * 32, 32)> : !torch.vtensor<[1,4,?,32],f16>
    %int1_1032 = torch.constant.int 1
    %int2_1033 = torch.constant.int 2
    %952 = torch.aten.transpose.int %936, %int1_1032, %int2_1033 : !torch.vtensor<[1,?,4,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,4,?,32],f16>
    torch.bind_symbolic_shape %952, [%74], affine_map<()[s0] -> (1, 4, s0 * 32, 32)> : !torch.vtensor<[1,4,?,32],f16>
    %int1_1034 = torch.constant.int 1
    %int2_1035 = torch.constant.int 2
    %953 = torch.aten.transpose.int %945, %int1_1034, %int2_1035 : !torch.vtensor<[1,?,4,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,4,?,32],f16>
    torch.bind_symbolic_shape %953, [%74], affine_map<()[s0] -> (1, 4, s0 * 32, 32)> : !torch.vtensor<[1,4,?,32],f16>
    %int1_1036 = torch.constant.int 1
    %int2_1037 = torch.constant.int 2
    %954 = torch.aten.transpose.int %948, %int1_1036, %int2_1037 : !torch.vtensor<[1,?,4,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,4,?,32],f16>
    torch.bind_symbolic_shape %954, [%74], affine_map<()[s0] -> (1, 4, s0 * 32, 32)> : !torch.vtensor<[1,4,?,32],f16>
    %int5_1038 = torch.constant.int 5
    %955 = torch.prims.convert_element_type %949, %int5_1038 : !torch.vtensor<[1,4,1,32],f16>, !torch.int -> !torch.vtensor<[1,4,1,32],f16>
    %int5_1039 = torch.constant.int 5
    %956 = torch.prims.convert_element_type %950, %int5_1039 : !torch.vtensor<[1,4,1,32],f16>, !torch.int -> !torch.vtensor<[1,4,1,32],f16>
    %int5_1040 = torch.constant.int 5
    %957 = torch.prims.convert_element_type %951, %int5_1040 : !torch.vtensor<[1,4,?,32],f16>, !torch.int -> !torch.vtensor<[1,4,?,32],f16>
    torch.bind_symbolic_shape %957, [%74], affine_map<()[s0] -> (1, 4, s0 * 32, 32)> : !torch.vtensor<[1,4,?,32],f16>
    %int5_1041 = torch.constant.int 5
    %958 = torch.prims.convert_element_type %952, %int5_1041 : !torch.vtensor<[1,4,?,32],f16>, !torch.int -> !torch.vtensor<[1,4,?,32],f16>
    torch.bind_symbolic_shape %958, [%74], affine_map<()[s0] -> (1, 4, s0 * 32, 32)> : !torch.vtensor<[1,4,?,32],f16>
    %int5_1042 = torch.constant.int 5
    %959 = torch.prims.convert_element_type %953, %int5_1042 : !torch.vtensor<[1,4,?,32],f16>, !torch.int -> !torch.vtensor<[1,4,?,32],f16>
    torch.bind_symbolic_shape %959, [%74], affine_map<()[s0] -> (1, 4, s0 * 32, 32)> : !torch.vtensor<[1,4,?,32],f16>
    %int5_1043 = torch.constant.int 5
    %960 = torch.prims.convert_element_type %954, %int5_1043 : !torch.vtensor<[1,4,?,32],f16>, !torch.int -> !torch.vtensor<[1,4,?,32],f16>
    torch.bind_symbolic_shape %960, [%74], affine_map<()[s0] -> (1, 4, s0 * 32, 32)> : !torch.vtensor<[1,4,?,32],f16>
    %int5_1044 = torch.constant.int 5
    %961 = torch.prims.convert_element_type %97, %int5_1044 : !torch.vtensor<[1,1,1,?],f16>, !torch.int -> !torch.vtensor<[1,1,1,?],f16>
    torch.bind_symbolic_shape %961, [%74], affine_map<()[s0] -> (1, 1, 1, s0 * 32)> : !torch.vtensor<[1,1,1,?],f16>
    %int5_1045 = torch.constant.int 5
    %962 = torch.prims.convert_element_type %100, %int5_1045 : !torch.vtensor<[1,1,1,?],f16>, !torch.int -> !torch.vtensor<[1,1,1,?],f16>
    torch.bind_symbolic_shape %962, [%74], affine_map<()[s0] -> (1, 1, 1, s0 * 32)> : !torch.vtensor<[1,1,1,?],f16>
    %float0.000000e00_1046 = torch.constant.float 0.000000e+00
    %false_1047 = torch.constant.bool false
    %none_1048 = torch.constant.none
    %963:2 = torch.operator "torch.aten._scaled_dot_product_flash_attention_for_cpu"(%955, %957, %959, %float0.000000e00_1046, %false_1047, %961, %none_1048) : (!torch.vtensor<[1,4,1,32],f16>, !torch.vtensor<[1,4,?,32],f16>, !torch.vtensor<[1,4,?,32],f16>, !torch.float, !torch.bool, !torch.vtensor<[1,1,1,?],f16>, !torch.none) -> (!torch.vtensor<[1,4,1,32],f16>, !torch.vtensor<[1,4,1],f32>) 
    %float0.000000e00_1049 = torch.constant.float 0.000000e+00
    %false_1050 = torch.constant.bool false
    %none_1051 = torch.constant.none
    %964:2 = torch.operator "torch.aten._scaled_dot_product_flash_attention_for_cpu"(%956, %958, %960, %float0.000000e00_1049, %false_1050, %962, %none_1051) : (!torch.vtensor<[1,4,1,32],f16>, !torch.vtensor<[1,4,?,32],f16>, !torch.vtensor<[1,4,?,32],f16>, !torch.float, !torch.bool, !torch.vtensor<[1,1,1,?],f16>, !torch.none) -> (!torch.vtensor<[1,4,1,32],f16>, !torch.vtensor<[1,4,1],f32>) 
    %int1_1052 = torch.constant.int 1
    %int2_1053 = torch.constant.int 2
    %965 = torch.aten.transpose.int %963#0, %int1_1052, %int2_1053 : !torch.vtensor<[1,4,1,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,1,4,32],f16>
    %int1_1054 = torch.constant.int 1
    %int2_1055 = torch.constant.int 2
    %966 = torch.aten.transpose.int %964#0, %int1_1054, %int2_1055 : !torch.vtensor<[1,4,1,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,1,4,32],f16>
    %int1_1056 = torch.constant.int 1
    %int1_1057 = torch.constant.int 1
    %int128_1058 = torch.constant.int 128
    %967 = torch.prim.ListConstruct %int1_1056, %int1_1057, %int128_1058 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %968 = torch.aten.view %965, %967 : !torch.vtensor<[1,1,4,32],f16>, !torch.list<int> -> !torch.vtensor<[1,1,128],f16>
    %int1_1059 = torch.constant.int 1
    %int1_1060 = torch.constant.int 1
    %int128_1061 = torch.constant.int 128
    %969 = torch.prim.ListConstruct %int1_1059, %int1_1060, %int128_1061 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %970 = torch.aten.view %966, %969 : !torch.vtensor<[1,1,4,32],f16>, !torch.list<int> -> !torch.vtensor<[1,1,128],f16>
    %int1_1062 = torch.constant.int 1
    %int0_1063 = torch.constant.int 0
    %971 = torch.prim.ListConstruct %int1_1062, %int0_1063 : (!torch.int, !torch.int) -> !torch.list<int>
    %972 = torch.aten.permute %36, %971 : !torch.vtensor<[256,128],f32>, !torch.list<int> -> !torch.vtensor<[128,256],f32>
    %int1_1064 = torch.constant.int 1
    %int0_1065 = torch.constant.int 0
    %973 = torch.prim.ListConstruct %int1_1064, %int0_1065 : (!torch.int, !torch.int) -> !torch.list<int>
    %974 = torch.aten.permute %37, %973 : !torch.vtensor<[256,128],f32>, !torch.list<int> -> !torch.vtensor<[128,256],f32>
    %int5_1066 = torch.constant.int 5
    %975 = torch.prims.convert_element_type %972, %int5_1066 : !torch.vtensor<[128,256],f32>, !torch.int -> !torch.vtensor<[128,256],f16>
    %int1_1067 = torch.constant.int 1
    %int128_1068 = torch.constant.int 128
    %976 = torch.prim.ListConstruct %int1_1067, %int128_1068 : (!torch.int, !torch.int) -> !torch.list<int>
    %977 = torch.aten.view %968, %976 : !torch.vtensor<[1,1,128],f16>, !torch.list<int> -> !torch.vtensor<[1,128],f16>
    %978 = torch.aten.mm %977, %975 : !torch.vtensor<[1,128],f16>, !torch.vtensor<[128,256],f16> -> !torch.vtensor<[1,256],f16>
    %int1_1069 = torch.constant.int 1
    %int1_1070 = torch.constant.int 1
    %int256_1071 = torch.constant.int 256
    %979 = torch.prim.ListConstruct %int1_1069, %int1_1070, %int256_1071 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %980 = torch.aten.view %978, %979 : !torch.vtensor<[1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,1,256],f16>
    %int5_1072 = torch.constant.int 5
    %981 = torch.prims.convert_element_type %974, %int5_1072 : !torch.vtensor<[128,256],f32>, !torch.int -> !torch.vtensor<[128,256],f16>
    %int1_1073 = torch.constant.int 1
    %int128_1074 = torch.constant.int 128
    %982 = torch.prim.ListConstruct %int1_1073, %int128_1074 : (!torch.int, !torch.int) -> !torch.list<int>
    %983 = torch.aten.view %970, %982 : !torch.vtensor<[1,1,128],f16>, !torch.list<int> -> !torch.vtensor<[1,128],f16>
    %984 = torch.aten.mm %983, %981 : !torch.vtensor<[1,128],f16>, !torch.vtensor<[128,256],f16> -> !torch.vtensor<[1,256],f16>
    %int1_1075 = torch.constant.int 1
    %int1_1076 = torch.constant.int 1
    %int256_1077 = torch.constant.int 256
    %985 = torch.prim.ListConstruct %int1_1075, %int1_1076, %int256_1077 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %986 = torch.aten.view %984, %985 : !torch.vtensor<[1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,1,256],f16>
    %987 = torch_c.to_builtin_tensor %980 : !torch.vtensor<[1,1,256],f16> -> tensor<1x1x256xf16>
    %988 = flow.tensor.barrier %987 : tensor<1x1x256xf16> on #hal.device.promise<@__device_0>
    %989 = torch_c.from_builtin_tensor %988 : tensor<1x1x256xf16> -> !torch.vtensor<[1,1,256],f16>
    %990 = torch_c.to_builtin_tensor %986 : !torch.vtensor<[1,1,256],f16> -> tensor<1x1x256xf16>
    %991 = flow.tensor.transfer %990 : tensor<1x1x256xf16> to #hal.device.promise<@__device_0>
    %992 = torch_c.from_builtin_tensor %991 : tensor<1x1x256xf16> -> !torch.vtensor<[1,1,256],f16>
    %int1_1078 = torch.constant.int 1
    %993 = torch.aten.add.Tensor %989, %992, %int1_1078 : !torch.vtensor<[1,1,256],f16>, !torch.vtensor<[1,1,256],f16>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %994 = torch_c.to_builtin_tensor %993 : !torch.vtensor<[1,1,256],f16> -> tensor<1x1x256xf16>
    %995 = flow.tensor.barrier %994 : tensor<1x1x256xf16> on #hal.device.promise<@__device_0>
    %996 = torch_c.from_builtin_tensor %995 : tensor<1x1x256xf16> -> !torch.vtensor<[1,1,256],f16>
    %997 = torch_c.to_builtin_tensor %993 : !torch.vtensor<[1,1,256],f16> -> tensor<1x1x256xf16>
    %998 = flow.tensor.transfer %997 : tensor<1x1x256xf16> to #hal.device.promise<@__device_1>
    %999 = torch_c.from_builtin_tensor %998 : tensor<1x1x256xf16> -> !torch.vtensor<[1,1,256],f16>
    %int1_1079 = torch.constant.int 1
    %1000 = torch.aten.add.Tensor %623, %996, %int1_1079 : !torch.vtensor<[1,1,256],f16>, !torch.vtensor<[1,1,256],f16>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %int1_1080 = torch.constant.int 1
    %1001 = torch.aten.add.Tensor %624, %999, %int1_1080 : !torch.vtensor<[1,1,256],f16>, !torch.vtensor<[1,1,256],f16>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %int6_1081 = torch.constant.int 6
    %1002 = torch.prims.convert_element_type %1000, %int6_1081 : !torch.vtensor<[1,1,256],f16>, !torch.int -> !torch.vtensor<[1,1,256],f32>
    %int6_1082 = torch.constant.int 6
    %1003 = torch.prims.convert_element_type %1001, %int6_1082 : !torch.vtensor<[1,1,256],f16>, !torch.int -> !torch.vtensor<[1,1,256],f32>
    %int2_1083 = torch.constant.int 2
    %1004 = torch.aten.pow.Tensor_Scalar %1002, %int2_1083 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f32>
    %int2_1084 = torch.constant.int 2
    %1005 = torch.aten.pow.Tensor_Scalar %1003, %int2_1084 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f32>
    %int-1_1085 = torch.constant.int -1
    %1006 = torch.prim.ListConstruct %int-1_1085 : (!torch.int) -> !torch.list<int>
    %true_1086 = torch.constant.bool true
    %none_1087 = torch.constant.none
    %1007 = torch.aten.mean.dim %1004, %1006, %true_1086, %none_1087 : !torch.vtensor<[1,1,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1],f32>
    %int-1_1088 = torch.constant.int -1
    %1008 = torch.prim.ListConstruct %int-1_1088 : (!torch.int) -> !torch.list<int>
    %true_1089 = torch.constant.bool true
    %none_1090 = torch.constant.none
    %1009 = torch.aten.mean.dim %1005, %1008, %true_1089, %none_1090 : !torch.vtensor<[1,1,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1],f32>
    %float1.000000e-02_1091 = torch.constant.float 1.000000e-02
    %int1_1092 = torch.constant.int 1
    %1010 = torch.aten.add.Scalar %1007, %float1.000000e-02_1091, %int1_1092 : !torch.vtensor<[1,1,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,1,1],f32>
    %float1.000000e-02_1093 = torch.constant.float 1.000000e-02
    %int1_1094 = torch.constant.int 1
    %1011 = torch.aten.add.Scalar %1009, %float1.000000e-02_1093, %int1_1094 : !torch.vtensor<[1,1,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,1,1],f32>
    %1012 = torch.aten.rsqrt %1010 : !torch.vtensor<[1,1,1],f32> -> !torch.vtensor<[1,1,1],f32>
    %1013 = torch.aten.rsqrt %1011 : !torch.vtensor<[1,1,1],f32> -> !torch.vtensor<[1,1,1],f32>
    %1014 = torch.aten.mul.Tensor %1002, %1012 : !torch.vtensor<[1,1,256],f32>, !torch.vtensor<[1,1,1],f32> -> !torch.vtensor<[1,1,256],f32>
    %1015 = torch.aten.mul.Tensor %1003, %1013 : !torch.vtensor<[1,1,256],f32>, !torch.vtensor<[1,1,1],f32> -> !torch.vtensor<[1,1,256],f32>
    %int5_1095 = torch.constant.int 5
    %1016 = torch.prims.convert_element_type %1014, %int5_1095 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %int5_1096 = torch.constant.int 5
    %1017 = torch.prims.convert_element_type %1015, %int5_1096 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %1018 = torch.aten.mul.Tensor %38, %1016 : !torch.vtensor<[256],f32>, !torch.vtensor<[1,1,256],f16> -> !torch.vtensor<[1,1,256],f32>
    %1019 = torch.aten.mul.Tensor %39, %1017 : !torch.vtensor<[256],f32>, !torch.vtensor<[1,1,256],f16> -> !torch.vtensor<[1,1,256],f32>
    %int5_1097 = torch.constant.int 5
    %1020 = torch.prims.convert_element_type %1018, %int5_1097 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %int5_1098 = torch.constant.int 5
    %1021 = torch.prims.convert_element_type %1019, %int5_1098 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %int1_1099 = torch.constant.int 1
    %int0_1100 = torch.constant.int 0
    %1022 = torch.prim.ListConstruct %int1_1099, %int0_1100 : (!torch.int, !torch.int) -> !torch.list<int>
    %1023 = torch.aten.permute %40, %1022 : !torch.vtensor<[12,256],f32>, !torch.list<int> -> !torch.vtensor<[256,12],f32>
    %int1_1101 = torch.constant.int 1
    %int0_1102 = torch.constant.int 0
    %1024 = torch.prim.ListConstruct %int1_1101, %int0_1102 : (!torch.int, !torch.int) -> !torch.list<int>
    %1025 = torch.aten.permute %41, %1024 : !torch.vtensor<[11,256],f32>, !torch.list<int> -> !torch.vtensor<[256,11],f32>
    %int5_1103 = torch.constant.int 5
    %1026 = torch.prims.convert_element_type %1023, %int5_1103 : !torch.vtensor<[256,12],f32>, !torch.int -> !torch.vtensor<[256,12],f16>
    %int1_1104 = torch.constant.int 1
    %int256_1105 = torch.constant.int 256
    %1027 = torch.prim.ListConstruct %int1_1104, %int256_1105 : (!torch.int, !torch.int) -> !torch.list<int>
    %1028 = torch.aten.view %1020, %1027 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %1029 = torch.aten.mm %1028, %1026 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,12],f16> -> !torch.vtensor<[1,12],f16>
    %int1_1106 = torch.constant.int 1
    %int1_1107 = torch.constant.int 1
    %int12_1108 = torch.constant.int 12
    %1030 = torch.prim.ListConstruct %int1_1106, %int1_1107, %int12_1108 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1031 = torch.aten.view %1029, %1030 : !torch.vtensor<[1,12],f16>, !torch.list<int> -> !torch.vtensor<[1,1,12],f16>
    %int5_1109 = torch.constant.int 5
    %1032 = torch.prims.convert_element_type %1025, %int5_1109 : !torch.vtensor<[256,11],f32>, !torch.int -> !torch.vtensor<[256,11],f16>
    %int1_1110 = torch.constant.int 1
    %int256_1111 = torch.constant.int 256
    %1033 = torch.prim.ListConstruct %int1_1110, %int256_1111 : (!torch.int, !torch.int) -> !torch.list<int>
    %1034 = torch.aten.view %1021, %1033 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %1035 = torch.aten.mm %1034, %1032 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,11],f16> -> !torch.vtensor<[1,11],f16>
    %int1_1112 = torch.constant.int 1
    %int1_1113 = torch.constant.int 1
    %int11_1114 = torch.constant.int 11
    %1036 = torch.prim.ListConstruct %int1_1112, %int1_1113, %int11_1114 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1037 = torch.aten.view %1035, %1036 : !torch.vtensor<[1,11],f16>, !torch.list<int> -> !torch.vtensor<[1,1,11],f16>
    %1038 = torch.aten.silu %1031 : !torch.vtensor<[1,1,12],f16> -> !torch.vtensor<[1,1,12],f16>
    %1039 = torch.aten.silu %1037 : !torch.vtensor<[1,1,11],f16> -> !torch.vtensor<[1,1,11],f16>
    %int1_1115 = torch.constant.int 1
    %int0_1116 = torch.constant.int 0
    %1040 = torch.prim.ListConstruct %int1_1115, %int0_1116 : (!torch.int, !torch.int) -> !torch.list<int>
    %1041 = torch.aten.permute %42, %1040 : !torch.vtensor<[12,256],f32>, !torch.list<int> -> !torch.vtensor<[256,12],f32>
    %int1_1117 = torch.constant.int 1
    %int0_1118 = torch.constant.int 0
    %1042 = torch.prim.ListConstruct %int1_1117, %int0_1118 : (!torch.int, !torch.int) -> !torch.list<int>
    %1043 = torch.aten.permute %43, %1042 : !torch.vtensor<[11,256],f32>, !torch.list<int> -> !torch.vtensor<[256,11],f32>
    %int5_1119 = torch.constant.int 5
    %1044 = torch.prims.convert_element_type %1041, %int5_1119 : !torch.vtensor<[256,12],f32>, !torch.int -> !torch.vtensor<[256,12],f16>
    %int1_1120 = torch.constant.int 1
    %int256_1121 = torch.constant.int 256
    %1045 = torch.prim.ListConstruct %int1_1120, %int256_1121 : (!torch.int, !torch.int) -> !torch.list<int>
    %1046 = torch.aten.view %1020, %1045 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %1047 = torch.aten.mm %1046, %1044 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,12],f16> -> !torch.vtensor<[1,12],f16>
    %int1_1122 = torch.constant.int 1
    %int1_1123 = torch.constant.int 1
    %int12_1124 = torch.constant.int 12
    %1048 = torch.prim.ListConstruct %int1_1122, %int1_1123, %int12_1124 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1049 = torch.aten.view %1047, %1048 : !torch.vtensor<[1,12],f16>, !torch.list<int> -> !torch.vtensor<[1,1,12],f16>
    %int5_1125 = torch.constant.int 5
    %1050 = torch.prims.convert_element_type %1043, %int5_1125 : !torch.vtensor<[256,11],f32>, !torch.int -> !torch.vtensor<[256,11],f16>
    %int1_1126 = torch.constant.int 1
    %int256_1127 = torch.constant.int 256
    %1051 = torch.prim.ListConstruct %int1_1126, %int256_1127 : (!torch.int, !torch.int) -> !torch.list<int>
    %1052 = torch.aten.view %1021, %1051 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %1053 = torch.aten.mm %1052, %1050 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,11],f16> -> !torch.vtensor<[1,11],f16>
    %int1_1128 = torch.constant.int 1
    %int1_1129 = torch.constant.int 1
    %int11_1130 = torch.constant.int 11
    %1054 = torch.prim.ListConstruct %int1_1128, %int1_1129, %int11_1130 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1055 = torch.aten.view %1053, %1054 : !torch.vtensor<[1,11],f16>, !torch.list<int> -> !torch.vtensor<[1,1,11],f16>
    %1056 = torch.aten.mul.Tensor %1038, %1049 : !torch.vtensor<[1,1,12],f16>, !torch.vtensor<[1,1,12],f16> -> !torch.vtensor<[1,1,12],f16>
    %1057 = torch.aten.mul.Tensor %1039, %1055 : !torch.vtensor<[1,1,11],f16>, !torch.vtensor<[1,1,11],f16> -> !torch.vtensor<[1,1,11],f16>
    %int1_1131 = torch.constant.int 1
    %int0_1132 = torch.constant.int 0
    %1058 = torch.prim.ListConstruct %int1_1131, %int0_1132 : (!torch.int, !torch.int) -> !torch.list<int>
    %1059 = torch.aten.permute %44, %1058 : !torch.vtensor<[256,12],f32>, !torch.list<int> -> !torch.vtensor<[12,256],f32>
    %int1_1133 = torch.constant.int 1
    %int0_1134 = torch.constant.int 0
    %1060 = torch.prim.ListConstruct %int1_1133, %int0_1134 : (!torch.int, !torch.int) -> !torch.list<int>
    %1061 = torch.aten.permute %45, %1060 : !torch.vtensor<[256,11],f32>, !torch.list<int> -> !torch.vtensor<[11,256],f32>
    %int5_1135 = torch.constant.int 5
    %1062 = torch.prims.convert_element_type %1059, %int5_1135 : !torch.vtensor<[12,256],f32>, !torch.int -> !torch.vtensor<[12,256],f16>
    %int1_1136 = torch.constant.int 1
    %int12_1137 = torch.constant.int 12
    %1063 = torch.prim.ListConstruct %int1_1136, %int12_1137 : (!torch.int, !torch.int) -> !torch.list<int>
    %1064 = torch.aten.view %1056, %1063 : !torch.vtensor<[1,1,12],f16>, !torch.list<int> -> !torch.vtensor<[1,12],f16>
    %1065 = torch.aten.mm %1064, %1062 : !torch.vtensor<[1,12],f16>, !torch.vtensor<[12,256],f16> -> !torch.vtensor<[1,256],f16>
    %int1_1138 = torch.constant.int 1
    %int1_1139 = torch.constant.int 1
    %int256_1140 = torch.constant.int 256
    %1066 = torch.prim.ListConstruct %int1_1138, %int1_1139, %int256_1140 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1067 = torch.aten.view %1065, %1066 : !torch.vtensor<[1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,1,256],f16>
    %int5_1141 = torch.constant.int 5
    %1068 = torch.prims.convert_element_type %1061, %int5_1141 : !torch.vtensor<[11,256],f32>, !torch.int -> !torch.vtensor<[11,256],f16>
    %int1_1142 = torch.constant.int 1
    %int11_1143 = torch.constant.int 11
    %1069 = torch.prim.ListConstruct %int1_1142, %int11_1143 : (!torch.int, !torch.int) -> !torch.list<int>
    %1070 = torch.aten.view %1057, %1069 : !torch.vtensor<[1,1,11],f16>, !torch.list<int> -> !torch.vtensor<[1,11],f16>
    %1071 = torch.aten.mm %1070, %1068 : !torch.vtensor<[1,11],f16>, !torch.vtensor<[11,256],f16> -> !torch.vtensor<[1,256],f16>
    %int1_1144 = torch.constant.int 1
    %int1_1145 = torch.constant.int 1
    %int256_1146 = torch.constant.int 256
    %1072 = torch.prim.ListConstruct %int1_1144, %int1_1145, %int256_1146 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1073 = torch.aten.view %1071, %1072 : !torch.vtensor<[1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,1,256],f16>
    %1074 = torch_c.to_builtin_tensor %1067 : !torch.vtensor<[1,1,256],f16> -> tensor<1x1x256xf16>
    %1075 = flow.tensor.barrier %1074 : tensor<1x1x256xf16> on #hal.device.promise<@__device_0>
    %1076 = torch_c.from_builtin_tensor %1075 : tensor<1x1x256xf16> -> !torch.vtensor<[1,1,256],f16>
    %1077 = torch_c.to_builtin_tensor %1073 : !torch.vtensor<[1,1,256],f16> -> tensor<1x1x256xf16>
    %1078 = flow.tensor.transfer %1077 : tensor<1x1x256xf16> to #hal.device.promise<@__device_0>
    %1079 = torch_c.from_builtin_tensor %1078 : tensor<1x1x256xf16> -> !torch.vtensor<[1,1,256],f16>
    %int1_1147 = torch.constant.int 1
    %1080 = torch.aten.add.Tensor %1076, %1079, %int1_1147 : !torch.vtensor<[1,1,256],f16>, !torch.vtensor<[1,1,256],f16>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %1081 = torch_c.to_builtin_tensor %1080 : !torch.vtensor<[1,1,256],f16> -> tensor<1x1x256xf16>
    %1082 = flow.tensor.barrier %1081 : tensor<1x1x256xf16> on #hal.device.promise<@__device_0>
    %1083 = torch_c.from_builtin_tensor %1082 : tensor<1x1x256xf16> -> !torch.vtensor<[1,1,256],f16>
    %1084 = torch_c.to_builtin_tensor %1080 : !torch.vtensor<[1,1,256],f16> -> tensor<1x1x256xf16>
    %1085 = flow.tensor.transfer %1084 : tensor<1x1x256xf16> to #hal.device.promise<@__device_1>
    %1086 = torch_c.from_builtin_tensor %1085 : tensor<1x1x256xf16> -> !torch.vtensor<[1,1,256],f16>
    %int1_1148 = torch.constant.int 1
    %1087 = torch.aten.add.Tensor %1000, %1083, %int1_1148 : !torch.vtensor<[1,1,256],f16>, !torch.vtensor<[1,1,256],f16>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %int1_1149 = torch.constant.int 1
    %1088 = torch.aten.add.Tensor %1001, %1086, %int1_1149 : !torch.vtensor<[1,1,256],f16>, !torch.vtensor<[1,1,256],f16>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %int6_1150 = torch.constant.int 6
    %1089 = torch.prims.convert_element_type %1087, %int6_1150 : !torch.vtensor<[1,1,256],f16>, !torch.int -> !torch.vtensor<[1,1,256],f32>
    %int6_1151 = torch.constant.int 6
    %1090 = torch.prims.convert_element_type %1088, %int6_1151 : !torch.vtensor<[1,1,256],f16>, !torch.int -> !torch.vtensor<[1,1,256],f32>
    %int2_1152 = torch.constant.int 2
    %1091 = torch.aten.pow.Tensor_Scalar %1089, %int2_1152 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f32>
    %int2_1153 = torch.constant.int 2
    %1092 = torch.aten.pow.Tensor_Scalar %1090, %int2_1153 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f32>
    %int-1_1154 = torch.constant.int -1
    %1093 = torch.prim.ListConstruct %int-1_1154 : (!torch.int) -> !torch.list<int>
    %true_1155 = torch.constant.bool true
    %none_1156 = torch.constant.none
    %1094 = torch.aten.mean.dim %1091, %1093, %true_1155, %none_1156 : !torch.vtensor<[1,1,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1],f32>
    %int-1_1157 = torch.constant.int -1
    %1095 = torch.prim.ListConstruct %int-1_1157 : (!torch.int) -> !torch.list<int>
    %true_1158 = torch.constant.bool true
    %none_1159 = torch.constant.none
    %1096 = torch.aten.mean.dim %1092, %1095, %true_1158, %none_1159 : !torch.vtensor<[1,1,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1],f32>
    %float1.000000e-02_1160 = torch.constant.float 1.000000e-02
    %int1_1161 = torch.constant.int 1
    %1097 = torch.aten.add.Scalar %1094, %float1.000000e-02_1160, %int1_1161 : !torch.vtensor<[1,1,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,1,1],f32>
    %float1.000000e-02_1162 = torch.constant.float 1.000000e-02
    %int1_1163 = torch.constant.int 1
    %1098 = torch.aten.add.Scalar %1096, %float1.000000e-02_1162, %int1_1163 : !torch.vtensor<[1,1,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,1,1],f32>
    %1099 = torch.aten.rsqrt %1097 : !torch.vtensor<[1,1,1],f32> -> !torch.vtensor<[1,1,1],f32>
    %1100 = torch.aten.rsqrt %1098 : !torch.vtensor<[1,1,1],f32> -> !torch.vtensor<[1,1,1],f32>
    %1101 = torch.aten.mul.Tensor %1089, %1099 : !torch.vtensor<[1,1,256],f32>, !torch.vtensor<[1,1,1],f32> -> !torch.vtensor<[1,1,256],f32>
    %1102 = torch.aten.mul.Tensor %1090, %1100 : !torch.vtensor<[1,1,256],f32>, !torch.vtensor<[1,1,1],f32> -> !torch.vtensor<[1,1,256],f32>
    %int5_1164 = torch.constant.int 5
    %1103 = torch.prims.convert_element_type %1101, %int5_1164 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %int5_1165 = torch.constant.int 5
    %1104 = torch.prims.convert_element_type %1102, %int5_1165 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %1105 = torch.aten.mul.Tensor %46, %1103 : !torch.vtensor<[256],f32>, !torch.vtensor<[1,1,256],f16> -> !torch.vtensor<[1,1,256],f32>
    %1106 = torch.aten.mul.Tensor %47, %1104 : !torch.vtensor<[256],f32>, !torch.vtensor<[1,1,256],f16> -> !torch.vtensor<[1,1,256],f32>
    %int5_1166 = torch.constant.int 5
    %1107 = torch.prims.convert_element_type %1105, %int5_1166 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %int5_1167 = torch.constant.int 5
    %1108 = torch.prims.convert_element_type %1106, %int5_1167 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %int1_1168 = torch.constant.int 1
    %int0_1169 = torch.constant.int 0
    %1109 = torch.prim.ListConstruct %int1_1168, %int0_1169 : (!torch.int, !torch.int) -> !torch.list<int>
    %1110 = torch.aten.permute %48, %1109 : !torch.vtensor<[128,256],f32>, !torch.list<int> -> !torch.vtensor<[256,128],f32>
    %int1_1170 = torch.constant.int 1
    %int0_1171 = torch.constant.int 0
    %1111 = torch.prim.ListConstruct %int1_1170, %int0_1171 : (!torch.int, !torch.int) -> !torch.list<int>
    %1112 = torch.aten.permute %49, %1111 : !torch.vtensor<[128,256],f32>, !torch.list<int> -> !torch.vtensor<[256,128],f32>
    %int5_1172 = torch.constant.int 5
    %1113 = torch.prims.convert_element_type %1110, %int5_1172 : !torch.vtensor<[256,128],f32>, !torch.int -> !torch.vtensor<[256,128],f16>
    %int1_1173 = torch.constant.int 1
    %int256_1174 = torch.constant.int 256
    %1114 = torch.prim.ListConstruct %int1_1173, %int256_1174 : (!torch.int, !torch.int) -> !torch.list<int>
    %1115 = torch.aten.view %1107, %1114 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %1116 = torch.aten.mm %1115, %1113 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,128],f16> -> !torch.vtensor<[1,128],f16>
    %int1_1175 = torch.constant.int 1
    %int1_1176 = torch.constant.int 1
    %int128_1177 = torch.constant.int 128
    %1117 = torch.prim.ListConstruct %int1_1175, %int1_1176, %int128_1177 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1118 = torch.aten.view %1116, %1117 : !torch.vtensor<[1,128],f16>, !torch.list<int> -> !torch.vtensor<[1,1,128],f16>
    %int5_1178 = torch.constant.int 5
    %1119 = torch.prims.convert_element_type %1112, %int5_1178 : !torch.vtensor<[256,128],f32>, !torch.int -> !torch.vtensor<[256,128],f16>
    %int1_1179 = torch.constant.int 1
    %int256_1180 = torch.constant.int 256
    %1120 = torch.prim.ListConstruct %int1_1179, %int256_1180 : (!torch.int, !torch.int) -> !torch.list<int>
    %1121 = torch.aten.view %1108, %1120 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %1122 = torch.aten.mm %1121, %1119 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,128],f16> -> !torch.vtensor<[1,128],f16>
    %int1_1181 = torch.constant.int 1
    %int1_1182 = torch.constant.int 1
    %int128_1183 = torch.constant.int 128
    %1123 = torch.prim.ListConstruct %int1_1181, %int1_1182, %int128_1183 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1124 = torch.aten.view %1122, %1123 : !torch.vtensor<[1,128],f16>, !torch.list<int> -> !torch.vtensor<[1,1,128],f16>
    %int1_1184 = torch.constant.int 1
    %int0_1185 = torch.constant.int 0
    %1125 = torch.prim.ListConstruct %int1_1184, %int0_1185 : (!torch.int, !torch.int) -> !torch.list<int>
    %1126 = torch.aten.permute %50, %1125 : !torch.vtensor<[64,256],f32>, !torch.list<int> -> !torch.vtensor<[256,64],f32>
    %int1_1186 = torch.constant.int 1
    %int0_1187 = torch.constant.int 0
    %1127 = torch.prim.ListConstruct %int1_1186, %int0_1187 : (!torch.int, !torch.int) -> !torch.list<int>
    %1128 = torch.aten.permute %51, %1127 : !torch.vtensor<[64,256],f32>, !torch.list<int> -> !torch.vtensor<[256,64],f32>
    %int5_1188 = torch.constant.int 5
    %1129 = torch.prims.convert_element_type %1126, %int5_1188 : !torch.vtensor<[256,64],f32>, !torch.int -> !torch.vtensor<[256,64],f16>
    %int1_1189 = torch.constant.int 1
    %int256_1190 = torch.constant.int 256
    %1130 = torch.prim.ListConstruct %int1_1189, %int256_1190 : (!torch.int, !torch.int) -> !torch.list<int>
    %1131 = torch.aten.view %1107, %1130 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %1132 = torch.aten.mm %1131, %1129 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,64],f16> -> !torch.vtensor<[1,64],f16>
    %int1_1191 = torch.constant.int 1
    %int1_1192 = torch.constant.int 1
    %int64_1193 = torch.constant.int 64
    %1133 = torch.prim.ListConstruct %int1_1191, %int1_1192, %int64_1193 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1134 = torch.aten.view %1132, %1133 : !torch.vtensor<[1,64],f16>, !torch.list<int> -> !torch.vtensor<[1,1,64],f16>
    %int5_1194 = torch.constant.int 5
    %1135 = torch.prims.convert_element_type %1128, %int5_1194 : !torch.vtensor<[256,64],f32>, !torch.int -> !torch.vtensor<[256,64],f16>
    %int1_1195 = torch.constant.int 1
    %int256_1196 = torch.constant.int 256
    %1136 = torch.prim.ListConstruct %int1_1195, %int256_1196 : (!torch.int, !torch.int) -> !torch.list<int>
    %1137 = torch.aten.view %1108, %1136 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %1138 = torch.aten.mm %1137, %1135 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,64],f16> -> !torch.vtensor<[1,64],f16>
    %int1_1197 = torch.constant.int 1
    %int1_1198 = torch.constant.int 1
    %int64_1199 = torch.constant.int 64
    %1139 = torch.prim.ListConstruct %int1_1197, %int1_1198, %int64_1199 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1140 = torch.aten.view %1138, %1139 : !torch.vtensor<[1,64],f16>, !torch.list<int> -> !torch.vtensor<[1,1,64],f16>
    %int1_1200 = torch.constant.int 1
    %int0_1201 = torch.constant.int 0
    %1141 = torch.prim.ListConstruct %int1_1200, %int0_1201 : (!torch.int, !torch.int) -> !torch.list<int>
    %1142 = torch.aten.permute %52, %1141 : !torch.vtensor<[64,256],f32>, !torch.list<int> -> !torch.vtensor<[256,64],f32>
    %int1_1202 = torch.constant.int 1
    %int0_1203 = torch.constant.int 0
    %1143 = torch.prim.ListConstruct %int1_1202, %int0_1203 : (!torch.int, !torch.int) -> !torch.list<int>
    %1144 = torch.aten.permute %53, %1143 : !torch.vtensor<[64,256],f32>, !torch.list<int> -> !torch.vtensor<[256,64],f32>
    %int5_1204 = torch.constant.int 5
    %1145 = torch.prims.convert_element_type %1142, %int5_1204 : !torch.vtensor<[256,64],f32>, !torch.int -> !torch.vtensor<[256,64],f16>
    %int1_1205 = torch.constant.int 1
    %int256_1206 = torch.constant.int 256
    %1146 = torch.prim.ListConstruct %int1_1205, %int256_1206 : (!torch.int, !torch.int) -> !torch.list<int>
    %1147 = torch.aten.view %1107, %1146 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %1148 = torch.aten.mm %1147, %1145 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,64],f16> -> !torch.vtensor<[1,64],f16>
    %int1_1207 = torch.constant.int 1
    %int1_1208 = torch.constant.int 1
    %int64_1209 = torch.constant.int 64
    %1149 = torch.prim.ListConstruct %int1_1207, %int1_1208, %int64_1209 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1150 = torch.aten.view %1148, %1149 : !torch.vtensor<[1,64],f16>, !torch.list<int> -> !torch.vtensor<[1,1,64],f16>
    %int5_1210 = torch.constant.int 5
    %1151 = torch.prims.convert_element_type %1144, %int5_1210 : !torch.vtensor<[256,64],f32>, !torch.int -> !torch.vtensor<[256,64],f16>
    %int1_1211 = torch.constant.int 1
    %int256_1212 = torch.constant.int 256
    %1152 = torch.prim.ListConstruct %int1_1211, %int256_1212 : (!torch.int, !torch.int) -> !torch.list<int>
    %1153 = torch.aten.view %1108, %1152 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %1154 = torch.aten.mm %1153, %1151 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,64],f16> -> !torch.vtensor<[1,64],f16>
    %int1_1213 = torch.constant.int 1
    %int1_1214 = torch.constant.int 1
    %int64_1215 = torch.constant.int 64
    %1155 = torch.prim.ListConstruct %int1_1213, %int1_1214, %int64_1215 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1156 = torch.aten.view %1154, %1155 : !torch.vtensor<[1,64],f16>, !torch.list<int> -> !torch.vtensor<[1,1,64],f16>
    %int1_1216 = torch.constant.int 1
    %int1_1217 = torch.constant.int 1
    %int4_1218 = torch.constant.int 4
    %int32_1219 = torch.constant.int 32
    %1157 = torch.prim.ListConstruct %int1_1216, %int1_1217, %int4_1218, %int32_1219 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1158 = torch.aten.view %1118, %1157 : !torch.vtensor<[1,1,128],f16>, !torch.list<int> -> !torch.vtensor<[1,1,4,32],f16>
    %int1_1220 = torch.constant.int 1
    %int1_1221 = torch.constant.int 1
    %int4_1222 = torch.constant.int 4
    %int32_1223 = torch.constant.int 32
    %1159 = torch.prim.ListConstruct %int1_1220, %int1_1221, %int4_1222, %int32_1223 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1160 = torch.aten.view %1124, %1159 : !torch.vtensor<[1,1,128],f16>, !torch.list<int> -> !torch.vtensor<[1,1,4,32],f16>
    %int1_1224 = torch.constant.int 1
    %int1_1225 = torch.constant.int 1
    %int2_1226 = torch.constant.int 2
    %int32_1227 = torch.constant.int 32
    %1161 = torch.prim.ListConstruct %int1_1224, %int1_1225, %int2_1226, %int32_1227 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1162 = torch.aten.view %1134, %1161 : !torch.vtensor<[1,1,64],f16>, !torch.list<int> -> !torch.vtensor<[1,1,2,32],f16>
    %int1_1228 = torch.constant.int 1
    %int1_1229 = torch.constant.int 1
    %int2_1230 = torch.constant.int 2
    %int32_1231 = torch.constant.int 32
    %1163 = torch.prim.ListConstruct %int1_1228, %int1_1229, %int2_1230, %int32_1231 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1164 = torch.aten.view %1140, %1163 : !torch.vtensor<[1,1,64],f16>, !torch.list<int> -> !torch.vtensor<[1,1,2,32],f16>
    %int1_1232 = torch.constant.int 1
    %int1_1233 = torch.constant.int 1
    %int2_1234 = torch.constant.int 2
    %int32_1235 = torch.constant.int 32
    %1165 = torch.prim.ListConstruct %int1_1232, %int1_1233, %int2_1234, %int32_1235 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1166 = torch.aten.view %1150, %1165 : !torch.vtensor<[1,1,64],f16>, !torch.list<int> -> !torch.vtensor<[1,1,2,32],f16>
    %int1_1236 = torch.constant.int 1
    %int1_1237 = torch.constant.int 1
    %int2_1238 = torch.constant.int 2
    %int32_1239 = torch.constant.int 32
    %1167 = torch.prim.ListConstruct %int1_1236, %int1_1237, %int2_1238, %int32_1239 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1168 = torch.aten.view %1156, %1167 : !torch.vtensor<[1,1,64],f16>, !torch.list<int> -> !torch.vtensor<[1,1,2,32],f16>
    %int6_1240 = torch.constant.int 6
    %1169 = torch.prims.convert_element_type %1158, %int6_1240 : !torch.vtensor<[1,1,4,32],f16>, !torch.int -> !torch.vtensor<[1,1,4,32],f32>
    %1170 = torch_c.to_builtin_tensor %1169 : !torch.vtensor<[1,1,4,32],f32> -> tensor<1x1x4x32xf32>
    %1171 = torch_c.to_builtin_tensor %152 : !torch.vtensor<[1,1,32],f32> -> tensor<1x1x32xf32>
    %1172 = util.call @sharktank_rotary_embedding_1_1_4_32_f32(%1170, %1171) : (tensor<1x1x4x32xf32>, tensor<1x1x32xf32>) -> tensor<1x1x4x32xf32>
    %1173 = torch_c.from_builtin_tensor %1172 : tensor<1x1x4x32xf32> -> !torch.vtensor<[1,1,4,32],f32>
    %int5_1241 = torch.constant.int 5
    %1174 = torch.prims.convert_element_type %1173, %int5_1241 : !torch.vtensor<[1,1,4,32],f32>, !torch.int -> !torch.vtensor<[1,1,4,32],f16>
    %int6_1242 = torch.constant.int 6
    %1175 = torch.prims.convert_element_type %1160, %int6_1242 : !torch.vtensor<[1,1,4,32],f16>, !torch.int -> !torch.vtensor<[1,1,4,32],f32>
    %1176 = torch_c.to_builtin_tensor %1175 : !torch.vtensor<[1,1,4,32],f32> -> tensor<1x1x4x32xf32>
    %1177 = torch_c.to_builtin_tensor %153 : !torch.vtensor<[1,1,32],f32> -> tensor<1x1x32xf32>
    %1178 = util.call @sharktank_rotary_embedding_1_1_4_32_f32(%1176, %1177) : (tensor<1x1x4x32xf32>, tensor<1x1x32xf32>) -> tensor<1x1x4x32xf32>
    %1179 = torch_c.from_builtin_tensor %1178 : tensor<1x1x4x32xf32> -> !torch.vtensor<[1,1,4,32],f32>
    %int5_1243 = torch.constant.int 5
    %1180 = torch.prims.convert_element_type %1179, %int5_1243 : !torch.vtensor<[1,1,4,32],f32>, !torch.int -> !torch.vtensor<[1,1,4,32],f16>
    %int6_1244 = torch.constant.int 6
    %1181 = torch.prims.convert_element_type %1162, %int6_1244 : !torch.vtensor<[1,1,2,32],f16>, !torch.int -> !torch.vtensor<[1,1,2,32],f32>
    %1182 = torch_c.to_builtin_tensor %1181 : !torch.vtensor<[1,1,2,32],f32> -> tensor<1x1x2x32xf32>
    %1183 = torch_c.to_builtin_tensor %152 : !torch.vtensor<[1,1,32],f32> -> tensor<1x1x32xf32>
    %1184 = util.call @sharktank_rotary_embedding_1_1_2_32_f32(%1182, %1183) : (tensor<1x1x2x32xf32>, tensor<1x1x32xf32>) -> tensor<1x1x2x32xf32>
    %1185 = torch_c.from_builtin_tensor %1184 : tensor<1x1x2x32xf32> -> !torch.vtensor<[1,1,2,32],f32>
    %int5_1245 = torch.constant.int 5
    %1186 = torch.prims.convert_element_type %1185, %int5_1245 : !torch.vtensor<[1,1,2,32],f32>, !torch.int -> !torch.vtensor<[1,1,2,32],f16>
    %int6_1246 = torch.constant.int 6
    %1187 = torch.prims.convert_element_type %1164, %int6_1246 : !torch.vtensor<[1,1,2,32],f16>, !torch.int -> !torch.vtensor<[1,1,2,32],f32>
    %1188 = torch_c.to_builtin_tensor %1187 : !torch.vtensor<[1,1,2,32],f32> -> tensor<1x1x2x32xf32>
    %1189 = torch_c.to_builtin_tensor %153 : !torch.vtensor<[1,1,32],f32> -> tensor<1x1x32xf32>
    %1190 = util.call @sharktank_rotary_embedding_1_1_2_32_f32(%1188, %1189) : (tensor<1x1x2x32xf32>, tensor<1x1x32xf32>) -> tensor<1x1x2x32xf32>
    %1191 = torch_c.from_builtin_tensor %1190 : tensor<1x1x2x32xf32> -> !torch.vtensor<[1,1,2,32],f32>
    %int5_1247 = torch.constant.int 5
    %1192 = torch.prims.convert_element_type %1191, %int5_1247 : !torch.vtensor<[1,1,2,32],f32>, !torch.int -> !torch.vtensor<[1,1,2,32],f16>
    %int32_1248 = torch.constant.int 32
    %1193 = torch.aten.floor_divide.Scalar %103, %int32_1248 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int32_1249 = torch.constant.int 32
    %1194 = torch.aten.floor_divide.Scalar %106, %int32_1249 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int1_1250 = torch.constant.int 1
    %1195 = torch.aten.unsqueeze %1193, %int1_1250 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_1251 = torch.constant.int 1
    %1196 = torch.aten.unsqueeze %1194, %int1_1251 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_1252 = torch.constant.int 1
    %false_1253 = torch.constant.bool false
    %1197 = torch.aten.gather %109, %int1_1252, %1195, %false_1253 : !torch.vtensor<[1,?],si64>, !torch.int, !torch.vtensor<[1,1],si64>, !torch.bool -> !torch.vtensor<[1,1],si64>
    %int1_1254 = torch.constant.int 1
    %false_1255 = torch.constant.bool false
    %1198 = torch.aten.gather %112, %int1_1254, %1196, %false_1255 : !torch.vtensor<[1,?],si64>, !torch.int, !torch.vtensor<[1,1],si64>, !torch.bool -> !torch.vtensor<[1,1],si64>
    %int32_1256 = torch.constant.int 32
    %1199 = torch.aten.remainder.Scalar %103, %int32_1256 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int32_1257 = torch.constant.int 32
    %1200 = torch.aten.remainder.Scalar %106, %int32_1257 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int1_1258 = torch.constant.int 1
    %1201 = torch.aten.unsqueeze %1199, %int1_1258 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_1259 = torch.constant.int 1
    %1202 = torch.aten.unsqueeze %1200, %int1_1259 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %none_1260 = torch.constant.none
    %1203 = torch.aten.clone %54, %none_1260 : !torch.vtensor<[],si64>, !torch.none -> !torch.vtensor<[],si64>
    %1204 = torch.aten.detach %1203 : !torch.vtensor<[],si64> -> !torch.vtensor<[],si64>
    %1205 = torch.aten.detach %1204 : !torch.vtensor<[],si64> -> !torch.vtensor<[],si64>
    %1206 = torch.aten.detach %1205 : !torch.vtensor<[],si64> -> !torch.vtensor<[],si64>
    %int0_1261 = torch.constant.int 0
    %1207 = torch.aten.unsqueeze %1206, %int0_1261 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %none_1262 = torch.constant.none
    %1208 = torch.aten.clone %55, %none_1262 : !torch.vtensor<[],si64>, !torch.none -> !torch.vtensor<[],si64>
    %1209 = torch.aten.detach %1208 : !torch.vtensor<[],si64> -> !torch.vtensor<[],si64>
    %1210 = torch.aten.detach %1209 : !torch.vtensor<[],si64> -> !torch.vtensor<[],si64>
    %1211 = torch.aten.detach %1210 : !torch.vtensor<[],si64> -> !torch.vtensor<[],si64>
    %int0_1263 = torch.constant.int 0
    %1212 = torch.aten.unsqueeze %1211, %int0_1263 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int1_1264 = torch.constant.int 1
    %int1_1265 = torch.constant.int 1
    %1213 = torch.prim.ListConstruct %int1_1264, %int1_1265 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_1266 = torch.constant.int 1
    %int1_1267 = torch.constant.int 1
    %1214 = torch.prim.ListConstruct %int1_1266, %int1_1267 : (!torch.int, !torch.int) -> !torch.list<int>
    %int4_1268 = torch.constant.int 4
    %int0_1269 = torch.constant.int 0
    %cpu_1270 = torch.constant.device "cpu"
    %false_1271 = torch.constant.bool false
    %1215 = torch.aten.empty_strided %1213, %1214, %int4_1268, %int0_1269, %cpu_1270, %false_1271 : !torch.list<int>, !torch.list<int>, !torch.int, !torch.int, !torch.Device, !torch.bool -> !torch.vtensor<[1,1],si64>
    %int2_1272 = torch.constant.int 2
    %1216 = torch.aten.fill.Scalar %1215, %int2_1272 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_1273 = torch.constant.int 1
    %int1_1274 = torch.constant.int 1
    %1217 = torch.prim.ListConstruct %int1_1273, %int1_1274 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_1275 = torch.constant.int 1
    %int1_1276 = torch.constant.int 1
    %1218 = torch.prim.ListConstruct %int1_1275, %int1_1276 : (!torch.int, !torch.int) -> !torch.list<int>
    %int4_1277 = torch.constant.int 4
    %int0_1278 = torch.constant.int 0
    %cpu_1279 = torch.constant.device "cpu"
    %false_1280 = torch.constant.bool false
    %1219 = torch.aten.empty_strided %1217, %1218, %int4_1277, %int0_1278, %cpu_1279, %false_1280 : !torch.list<int>, !torch.list<int>, !torch.int, !torch.int, !torch.Device, !torch.bool -> !torch.vtensor<[1,1],si64>
    %int2_1281 = torch.constant.int 2
    %1220 = torch.aten.fill.Scalar %1219, %int2_1281 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_1282 = torch.constant.int 1
    %int1_1283 = torch.constant.int 1
    %1221 = torch.prim.ListConstruct %int1_1282, %int1_1283 : (!torch.int, !torch.int) -> !torch.list<int>
    %1222 = torch.aten.repeat %1207, %1221 : !torch.vtensor<[1],si64>, !torch.list<int> -> !torch.vtensor<[1,1],si64>
    %int1_1284 = torch.constant.int 1
    %int1_1285 = torch.constant.int 1
    %1223 = torch.prim.ListConstruct %int1_1284, %int1_1285 : (!torch.int, !torch.int) -> !torch.list<int>
    %1224 = torch.aten.repeat %1212, %1223 : !torch.vtensor<[1],si64>, !torch.list<int> -> !torch.vtensor<[1,1],si64>
    %int3_1286 = torch.constant.int 3
    %1225 = torch.aten.mul.Scalar %1197, %int3_1286 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int3_1287 = torch.constant.int 3
    %1226 = torch.aten.mul.Scalar %1198, %int3_1287 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_1288 = torch.constant.int 1
    %1227 = torch.aten.add.Tensor %1225, %1216, %int1_1288 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_1289 = torch.constant.int 1
    %1228 = torch.aten.add.Tensor %1226, %1220, %int1_1289 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int2_1290 = torch.constant.int 2
    %1229 = torch.aten.mul.Scalar %1227, %int2_1290 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int2_1291 = torch.constant.int 2
    %1230 = torch.aten.mul.Scalar %1228, %int2_1291 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_1292 = torch.constant.int 1
    %1231 = torch.aten.add.Tensor %1229, %1222, %int1_1292 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_1293 = torch.constant.int 1
    %1232 = torch.aten.add.Tensor %1230, %1224, %int1_1293 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int32_1294 = torch.constant.int 32
    %1233 = torch.aten.mul.Scalar %1231, %int32_1294 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int32_1295 = torch.constant.int 32
    %1234 = torch.aten.mul.Scalar %1232, %int32_1295 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_1296 = torch.constant.int 1
    %1235 = torch.aten.add.Tensor %1233, %1201, %int1_1296 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_1297 = torch.constant.int 1
    %1236 = torch.aten.add.Tensor %1234, %1202, %int1_1297 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int5_1298 = torch.constant.int 5
    %1237 = torch.prims.convert_element_type %1186, %int5_1298 : !torch.vtensor<[1,1,2,32],f16>, !torch.int -> !torch.vtensor<[1,1,2,32],f16>
    %int5_1299 = torch.constant.int 5
    %1238 = torch.prims.convert_element_type %1192, %int5_1299 : !torch.vtensor<[1,1,2,32],f16>, !torch.int -> !torch.vtensor<[1,1,2,32],f16>
    %int3_1300 = torch.constant.int 3
    %int2_1301 = torch.constant.int 2
    %int32_1302 = torch.constant.int 32
    %int2_1303 = torch.constant.int 2
    %int32_1304 = torch.constant.int 32
    %1239 = torch.prim.ListConstruct %77, %int3_1300, %int2_1301, %int32_1302, %int2_1303, %int32_1304 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1240 = torch.aten.view %854, %1239 : !torch.vtensor<[?,12288],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,2,32],f16>
    torch.bind_symbolic_shape %1240, [%75], affine_map<()[s0] -> (s0, 3, 2, 32, 2, 32)> : !torch.vtensor<[?,3,2,32,2,32],f16>
    %int2_1305 = torch.constant.int 2
    %int32_1306 = torch.constant.int 32
    %1241 = torch.prim.ListConstruct %268, %int2_1305, %int32_1306 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1242 = torch.aten.view %1240, %1241 : !torch.vtensor<[?,3,2,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,2,32],f16>
    torch.bind_symbolic_shape %1242, [%75], affine_map<()[s0] -> (s0 * 192, 2, 32)> : !torch.vtensor<[?,2,32],f16>
    %1243 = torch.prim.ListConstruct %1235 : (!torch.vtensor<[1,1],si64>) -> !torch.list<optional<vtensor>>
    %false_1307 = torch.constant.bool false
    %1244 = torch.aten.index_put %1242, %1243, %1237, %false_1307 : !torch.vtensor<[?,2,32],f16>, !torch.list<optional<vtensor>>, !torch.vtensor<[1,1,2,32],f16>, !torch.bool -> !torch.vtensor<[?,2,32],f16>
    torch.bind_symbolic_shape %1244, [%75], affine_map<()[s0] -> (s0 * 192, 2, 32)> : !torch.vtensor<[?,2,32],f16>
    %int3_1308 = torch.constant.int 3
    %int2_1309 = torch.constant.int 2
    %int32_1310 = torch.constant.int 32
    %int2_1311 = torch.constant.int 2
    %int32_1312 = torch.constant.int 32
    %1245 = torch.prim.ListConstruct %77, %int3_1308, %int2_1309, %int32_1310, %int2_1311, %int32_1312 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1246 = torch.aten.view %1244, %1245 : !torch.vtensor<[?,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,2,32],f16>
    torch.bind_symbolic_shape %1246, [%75], affine_map<()[s0] -> (s0, 3, 2, 32, 2, 32)> : !torch.vtensor<[?,3,2,32,2,32],f16>
    %int12288_1313 = torch.constant.int 12288
    %1247 = torch.prim.ListConstruct %77, %int12288_1313 : (!torch.int, !torch.int) -> !torch.list<int>
    %1248 = torch.aten.view %1246, %1247 : !torch.vtensor<[?,3,2,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,12288],f16>
    torch.bind_symbolic_shape %1248, [%75], affine_map<()[s0] -> (s0, 12288)> : !torch.vtensor<[?,12288],f16>
    %int3_1314 = torch.constant.int 3
    %int2_1315 = torch.constant.int 2
    %int32_1316 = torch.constant.int 32
    %int2_1317 = torch.constant.int 2
    %int32_1318 = torch.constant.int 32
    %1249 = torch.prim.ListConstruct %77, %int3_1314, %int2_1315, %int32_1316, %int2_1317, %int32_1318 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1250 = torch.aten.view %1248, %1249 : !torch.vtensor<[?,12288],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,2,32],f16>
    torch.bind_symbolic_shape %1250, [%75], affine_map<()[s0] -> (s0, 3, 2, 32, 2, 32)> : !torch.vtensor<[?,3,2,32,2,32],f16>
    %int2_1319 = torch.constant.int 2
    %int32_1320 = torch.constant.int 32
    %1251 = torch.prim.ListConstruct %268, %int2_1319, %int32_1320 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1252 = torch.aten.view %1250, %1251 : !torch.vtensor<[?,3,2,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,2,32],f16>
    torch.bind_symbolic_shape %1252, [%75], affine_map<()[s0] -> (s0 * 192, 2, 32)> : !torch.vtensor<[?,2,32],f16>
    %int3_1321 = torch.constant.int 3
    %int2_1322 = torch.constant.int 2
    %int32_1323 = torch.constant.int 32
    %int2_1324 = torch.constant.int 2
    %int32_1325 = torch.constant.int 32
    %1253 = torch.prim.ListConstruct %77, %int3_1321, %int2_1322, %int32_1323, %int2_1324, %int32_1325 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1254 = torch.aten.view %860, %1253 : !torch.vtensor<[?,12288],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,2,32],f16>
    torch.bind_symbolic_shape %1254, [%75], affine_map<()[s0] -> (s0, 3, 2, 32, 2, 32)> : !torch.vtensor<[?,3,2,32,2,32],f16>
    %int2_1326 = torch.constant.int 2
    %int32_1327 = torch.constant.int 32
    %1255 = torch.prim.ListConstruct %268, %int2_1326, %int32_1327 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1256 = torch.aten.view %1254, %1255 : !torch.vtensor<[?,3,2,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,2,32],f16>
    torch.bind_symbolic_shape %1256, [%75], affine_map<()[s0] -> (s0 * 192, 2, 32)> : !torch.vtensor<[?,2,32],f16>
    %1257 = torch.prim.ListConstruct %1236 : (!torch.vtensor<[1,1],si64>) -> !torch.list<optional<vtensor>>
    %false_1328 = torch.constant.bool false
    %1258 = torch.aten.index_put %1256, %1257, %1238, %false_1328 : !torch.vtensor<[?,2,32],f16>, !torch.list<optional<vtensor>>, !torch.vtensor<[1,1,2,32],f16>, !torch.bool -> !torch.vtensor<[?,2,32],f16>
    torch.bind_symbolic_shape %1258, [%75], affine_map<()[s0] -> (s0 * 192, 2, 32)> : !torch.vtensor<[?,2,32],f16>
    %int3_1329 = torch.constant.int 3
    %int2_1330 = torch.constant.int 2
    %int32_1331 = torch.constant.int 32
    %int2_1332 = torch.constant.int 2
    %int32_1333 = torch.constant.int 32
    %1259 = torch.prim.ListConstruct %77, %int3_1329, %int2_1330, %int32_1331, %int2_1332, %int32_1333 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1260 = torch.aten.view %1258, %1259 : !torch.vtensor<[?,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,2,32],f16>
    torch.bind_symbolic_shape %1260, [%75], affine_map<()[s0] -> (s0, 3, 2, 32, 2, 32)> : !torch.vtensor<[?,3,2,32,2,32],f16>
    %int12288_1334 = torch.constant.int 12288
    %1261 = torch.prim.ListConstruct %77, %int12288_1334 : (!torch.int, !torch.int) -> !torch.list<int>
    %1262 = torch.aten.view %1260, %1261 : !torch.vtensor<[?,3,2,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,12288],f16>
    torch.bind_symbolic_shape %1262, [%75], affine_map<()[s0] -> (s0, 12288)> : !torch.vtensor<[?,12288],f16>
    %int3_1335 = torch.constant.int 3
    %int2_1336 = torch.constant.int 2
    %int32_1337 = torch.constant.int 32
    %int2_1338 = torch.constant.int 2
    %int32_1339 = torch.constant.int 32
    %1263 = torch.prim.ListConstruct %77, %int3_1335, %int2_1336, %int32_1337, %int2_1338, %int32_1339 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1264 = torch.aten.view %1262, %1263 : !torch.vtensor<[?,12288],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,2,32],f16>
    torch.bind_symbolic_shape %1264, [%75], affine_map<()[s0] -> (s0, 3, 2, 32, 2, 32)> : !torch.vtensor<[?,3,2,32,2,32],f16>
    %int2_1340 = torch.constant.int 2
    %int32_1341 = torch.constant.int 32
    %1265 = torch.prim.ListConstruct %268, %int2_1340, %int32_1341 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1266 = torch.aten.view %1264, %1265 : !torch.vtensor<[?,3,2,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,2,32],f16>
    torch.bind_symbolic_shape %1266, [%75], affine_map<()[s0] -> (s0 * 192, 2, 32)> : !torch.vtensor<[?,2,32],f16>
    %int32_1342 = torch.constant.int 32
    %1267 = torch.aten.floor_divide.Scalar %103, %int32_1342 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int32_1343 = torch.constant.int 32
    %1268 = torch.aten.floor_divide.Scalar %106, %int32_1343 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int1_1344 = torch.constant.int 1
    %1269 = torch.aten.unsqueeze %1267, %int1_1344 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_1345 = torch.constant.int 1
    %1270 = torch.aten.unsqueeze %1268, %int1_1345 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_1346 = torch.constant.int 1
    %false_1347 = torch.constant.bool false
    %1271 = torch.aten.gather %109, %int1_1346, %1269, %false_1347 : !torch.vtensor<[1,?],si64>, !torch.int, !torch.vtensor<[1,1],si64>, !torch.bool -> !torch.vtensor<[1,1],si64>
    %int1_1348 = torch.constant.int 1
    %false_1349 = torch.constant.bool false
    %1272 = torch.aten.gather %112, %int1_1348, %1270, %false_1349 : !torch.vtensor<[1,?],si64>, !torch.int, !torch.vtensor<[1,1],si64>, !torch.bool -> !torch.vtensor<[1,1],si64>
    %int32_1350 = torch.constant.int 32
    %1273 = torch.aten.remainder.Scalar %103, %int32_1350 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int32_1351 = torch.constant.int 32
    %1274 = torch.aten.remainder.Scalar %106, %int32_1351 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int1_1352 = torch.constant.int 1
    %1275 = torch.aten.unsqueeze %1273, %int1_1352 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_1353 = torch.constant.int 1
    %1276 = torch.aten.unsqueeze %1274, %int1_1353 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %none_1354 = torch.constant.none
    %1277 = torch.aten.clone %56, %none_1354 : !torch.vtensor<[],si64>, !torch.none -> !torch.vtensor<[],si64>
    %1278 = torch.aten.detach %1277 : !torch.vtensor<[],si64> -> !torch.vtensor<[],si64>
    %1279 = torch.aten.detach %1278 : !torch.vtensor<[],si64> -> !torch.vtensor<[],si64>
    %1280 = torch.aten.detach %1279 : !torch.vtensor<[],si64> -> !torch.vtensor<[],si64>
    %int0_1355 = torch.constant.int 0
    %1281 = torch.aten.unsqueeze %1280, %int0_1355 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %none_1356 = torch.constant.none
    %1282 = torch.aten.clone %57, %none_1356 : !torch.vtensor<[],si64>, !torch.none -> !torch.vtensor<[],si64>
    %1283 = torch.aten.detach %1282 : !torch.vtensor<[],si64> -> !torch.vtensor<[],si64>
    %1284 = torch.aten.detach %1283 : !torch.vtensor<[],si64> -> !torch.vtensor<[],si64>
    %1285 = torch.aten.detach %1284 : !torch.vtensor<[],si64> -> !torch.vtensor<[],si64>
    %int0_1357 = torch.constant.int 0
    %1286 = torch.aten.unsqueeze %1285, %int0_1357 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int1_1358 = torch.constant.int 1
    %int1_1359 = torch.constant.int 1
    %1287 = torch.prim.ListConstruct %int1_1358, %int1_1359 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_1360 = torch.constant.int 1
    %int1_1361 = torch.constant.int 1
    %1288 = torch.prim.ListConstruct %int1_1360, %int1_1361 : (!torch.int, !torch.int) -> !torch.list<int>
    %int4_1362 = torch.constant.int 4
    %int0_1363 = torch.constant.int 0
    %cpu_1364 = torch.constant.device "cpu"
    %false_1365 = torch.constant.bool false
    %1289 = torch.aten.empty_strided %1287, %1288, %int4_1362, %int0_1363, %cpu_1364, %false_1365 : !torch.list<int>, !torch.list<int>, !torch.int, !torch.int, !torch.Device, !torch.bool -> !torch.vtensor<[1,1],si64>
    %int2_1366 = torch.constant.int 2
    %1290 = torch.aten.fill.Scalar %1289, %int2_1366 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_1367 = torch.constant.int 1
    %int1_1368 = torch.constant.int 1
    %1291 = torch.prim.ListConstruct %int1_1367, %int1_1368 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_1369 = torch.constant.int 1
    %int1_1370 = torch.constant.int 1
    %1292 = torch.prim.ListConstruct %int1_1369, %int1_1370 : (!torch.int, !torch.int) -> !torch.list<int>
    %int4_1371 = torch.constant.int 4
    %int0_1372 = torch.constant.int 0
    %cpu_1373 = torch.constant.device "cpu"
    %false_1374 = torch.constant.bool false
    %1293 = torch.aten.empty_strided %1291, %1292, %int4_1371, %int0_1372, %cpu_1373, %false_1374 : !torch.list<int>, !torch.list<int>, !torch.int, !torch.int, !torch.Device, !torch.bool -> !torch.vtensor<[1,1],si64>
    %int2_1375 = torch.constant.int 2
    %1294 = torch.aten.fill.Scalar %1293, %int2_1375 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_1376 = torch.constant.int 1
    %int1_1377 = torch.constant.int 1
    %1295 = torch.prim.ListConstruct %int1_1376, %int1_1377 : (!torch.int, !torch.int) -> !torch.list<int>
    %1296 = torch.aten.repeat %1281, %1295 : !torch.vtensor<[1],si64>, !torch.list<int> -> !torch.vtensor<[1,1],si64>
    %int1_1378 = torch.constant.int 1
    %int1_1379 = torch.constant.int 1
    %1297 = torch.prim.ListConstruct %int1_1378, %int1_1379 : (!torch.int, !torch.int) -> !torch.list<int>
    %1298 = torch.aten.repeat %1286, %1297 : !torch.vtensor<[1],si64>, !torch.list<int> -> !torch.vtensor<[1,1],si64>
    %int3_1380 = torch.constant.int 3
    %1299 = torch.aten.mul.Scalar %1271, %int3_1380 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int3_1381 = torch.constant.int 3
    %1300 = torch.aten.mul.Scalar %1272, %int3_1381 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_1382 = torch.constant.int 1
    %1301 = torch.aten.add.Tensor %1299, %1290, %int1_1382 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_1383 = torch.constant.int 1
    %1302 = torch.aten.add.Tensor %1300, %1294, %int1_1383 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int2_1384 = torch.constant.int 2
    %1303 = torch.aten.mul.Scalar %1301, %int2_1384 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int2_1385 = torch.constant.int 2
    %1304 = torch.aten.mul.Scalar %1302, %int2_1385 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_1386 = torch.constant.int 1
    %1305 = torch.aten.add.Tensor %1303, %1296, %int1_1386 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_1387 = torch.constant.int 1
    %1306 = torch.aten.add.Tensor %1304, %1298, %int1_1387 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int32_1388 = torch.constant.int 32
    %1307 = torch.aten.mul.Scalar %1305, %int32_1388 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int32_1389 = torch.constant.int 32
    %1308 = torch.aten.mul.Scalar %1306, %int32_1389 : !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_1390 = torch.constant.int 1
    %1309 = torch.aten.add.Tensor %1307, %1275, %int1_1390 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int1_1391 = torch.constant.int 1
    %1310 = torch.aten.add.Tensor %1308, %1276, %int1_1391 : !torch.vtensor<[1,1],si64>, !torch.vtensor<[1,1],si64>, !torch.int -> !torch.vtensor<[1,1],si64>
    %int5_1392 = torch.constant.int 5
    %1311 = torch.prims.convert_element_type %1166, %int5_1392 : !torch.vtensor<[1,1,2,32],f16>, !torch.int -> !torch.vtensor<[1,1,2,32],f16>
    %int5_1393 = torch.constant.int 5
    %1312 = torch.prims.convert_element_type %1168, %int5_1393 : !torch.vtensor<[1,1,2,32],f16>, !torch.int -> !torch.vtensor<[1,1,2,32],f16>
    %1313 = torch.prim.ListConstruct %1309 : (!torch.vtensor<[1,1],si64>) -> !torch.list<optional<vtensor>>
    %false_1394 = torch.constant.bool false
    %1314 = torch.aten.index_put %1252, %1313, %1311, %false_1394 : !torch.vtensor<[?,2,32],f16>, !torch.list<optional<vtensor>>, !torch.vtensor<[1,1,2,32],f16>, !torch.bool -> !torch.vtensor<[?,2,32],f16>
    torch.bind_symbolic_shape %1314, [%75], affine_map<()[s0] -> (s0 * 192, 2, 32)> : !torch.vtensor<[?,2,32],f16>
    %int3_1395 = torch.constant.int 3
    %int2_1396 = torch.constant.int 2
    %int32_1397 = torch.constant.int 32
    %int2_1398 = torch.constant.int 2
    %int32_1399 = torch.constant.int 32
    %1315 = torch.prim.ListConstruct %77, %int3_1395, %int2_1396, %int32_1397, %int2_1398, %int32_1399 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1316 = torch.aten.view %1314, %1315 : !torch.vtensor<[?,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,2,32],f16>
    torch.bind_symbolic_shape %1316, [%75], affine_map<()[s0] -> (s0, 3, 2, 32, 2, 32)> : !torch.vtensor<[?,3,2,32,2,32],f16>
    %int12288_1400 = torch.constant.int 12288
    %1317 = torch.prim.ListConstruct %77, %int12288_1400 : (!torch.int, !torch.int) -> !torch.list<int>
    %1318 = torch.aten.view %1316, %1317 : !torch.vtensor<[?,3,2,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,12288],f16>
    torch.overwrite.tensor.contents %1318 overwrites %arg4 : !torch.vtensor<[?,12288],f16>, !torch.tensor<[?,12288],f16>
    torch.bind_symbolic_shape %1318, [%75], affine_map<()[s0] -> (s0, 12288)> : !torch.vtensor<[?,12288],f16>
    %1319 = torch.prim.ListConstruct %1310 : (!torch.vtensor<[1,1],si64>) -> !torch.list<optional<vtensor>>
    %false_1401 = torch.constant.bool false
    %1320 = torch.aten.index_put %1266, %1319, %1312, %false_1401 : !torch.vtensor<[?,2,32],f16>, !torch.list<optional<vtensor>>, !torch.vtensor<[1,1,2,32],f16>, !torch.bool -> !torch.vtensor<[?,2,32],f16>
    torch.bind_symbolic_shape %1320, [%75], affine_map<()[s0] -> (s0 * 192, 2, 32)> : !torch.vtensor<[?,2,32],f16>
    %int3_1402 = torch.constant.int 3
    %int2_1403 = torch.constant.int 2
    %int32_1404 = torch.constant.int 32
    %int2_1405 = torch.constant.int 2
    %int32_1406 = torch.constant.int 32
    %1321 = torch.prim.ListConstruct %77, %int3_1402, %int2_1403, %int32_1404, %int2_1405, %int32_1406 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1322 = torch.aten.view %1320, %1321 : !torch.vtensor<[?,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,2,32],f16>
    torch.bind_symbolic_shape %1322, [%75], affine_map<()[s0] -> (s0, 3, 2, 32, 2, 32)> : !torch.vtensor<[?,3,2,32,2,32],f16>
    %int12288_1407 = torch.constant.int 12288
    %1323 = torch.prim.ListConstruct %77, %int12288_1407 : (!torch.int, !torch.int) -> !torch.list<int>
    %1324 = torch.aten.view %1322, %1323 : !torch.vtensor<[?,3,2,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,12288],f16>
    torch.overwrite.tensor.contents %1324 overwrites %arg5 : !torch.vtensor<[?,12288],f16>, !torch.tensor<[?,12288],f16>
    torch.bind_symbolic_shape %1324, [%75], affine_map<()[s0] -> (s0, 12288)> : !torch.vtensor<[?,12288],f16>
    %int1_1408 = torch.constant.int 1
    %1325 = torch.prim.ListConstruct %int1_1408, %76 : (!torch.int, !torch.int) -> !torch.list<int>
    %int1_1409 = torch.constant.int 1
    %1326 = torch.prim.ListConstruct %76, %int1_1409 : (!torch.int, !torch.int) -> !torch.list<int>
    %int4_1410 = torch.constant.int 4
    %int0_1411 = torch.constant.int 0
    %cpu_1412 = torch.constant.device "cpu"
    %false_1413 = torch.constant.bool false
    %1327 = torch.aten.empty_strided %1325, %1326, %int4_1410, %int0_1411, %cpu_1412, %false_1413 : !torch.list<int>, !torch.list<int>, !torch.int, !torch.int, !torch.Device, !torch.bool -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %1327, [%74], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %int2_1414 = torch.constant.int 2
    %1328 = torch.aten.fill.Scalar %1327, %int2_1414 : !torch.vtensor<[1,?],si64>, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %1328, [%74], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %int3_1415 = torch.constant.int 3
    %1329 = torch.aten.mul.Scalar %109, %int3_1415 : !torch.vtensor<[1,?],si64>, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %1329, [%74], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %int3_1416 = torch.constant.int 3
    %1330 = torch.aten.mul.Scalar %112, %int3_1416 : !torch.vtensor<[1,?],si64>, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %1330, [%74], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %1331 = torch_c.to_builtin_tensor %1328 : !torch.vtensor<[1,?],si64> -> tensor<1x?xi64>
    %c1_1417 = arith.constant 1 : index
    %dim_1418 = tensor.dim %1331, %c1_1417 : tensor<1x?xi64>
    %1332 = flow.tensor.transfer %1331 : tensor<1x?xi64>{%dim_1418} to #hal.device.promise<@__device_0>
    %1333 = torch_c.from_builtin_tensor %1332 : tensor<1x?xi64> -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %1333, [%74], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %1334 = torch_c.to_builtin_tensor %1328 : !torch.vtensor<[1,?],si64> -> tensor<1x?xi64>
    %c1_1419 = arith.constant 1 : index
    %dim_1420 = tensor.dim %1334, %c1_1419 : tensor<1x?xi64>
    %1335 = flow.tensor.transfer %1334 : tensor<1x?xi64>{%dim_1420} to #hal.device.promise<@__device_1>
    %1336 = torch_c.from_builtin_tensor %1335 : tensor<1x?xi64> -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %1336, [%74], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %int1_1421 = torch.constant.int 1
    %1337 = torch.aten.add.Tensor %1329, %1333, %int1_1421 : !torch.vtensor<[1,?],si64>, !torch.vtensor<[1,?],si64>, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %1337, [%74], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %int1_1422 = torch.constant.int 1
    %1338 = torch.aten.add.Tensor %1330, %1336, %int1_1422 : !torch.vtensor<[1,?],si64>, !torch.vtensor<[1,?],si64>, !torch.int -> !torch.vtensor<[1,?],si64>
    torch.bind_symbolic_shape %1338, [%74], affine_map<()[s0] -> (1, s0)> : !torch.vtensor<[1,?],si64>
    %1339 = torch.prim.ListConstruct %76 : (!torch.int) -> !torch.list<int>
    %1340 = torch.aten.view %1337, %1339 : !torch.vtensor<[1,?],si64>, !torch.list<int> -> !torch.vtensor<[?],si64>
    torch.bind_symbolic_shape %1340, [%74], affine_map<()[s0] -> (s0)> : !torch.vtensor<[?],si64>
    %1341 = torch.prim.ListConstruct %76 : (!torch.int) -> !torch.list<int>
    %1342 = torch.aten.view %1338, %1341 : !torch.vtensor<[1,?],si64>, !torch.list<int> -> !torch.vtensor<[?],si64>
    torch.bind_symbolic_shape %1342, [%74], affine_map<()[s0] -> (s0)> : !torch.vtensor<[?],si64>
    %int3_1423 = torch.constant.int 3
    %int2_1424 = torch.constant.int 2
    %int32_1425 = torch.constant.int 32
    %int2_1426 = torch.constant.int 2
    %int32_1427 = torch.constant.int 32
    %1343 = torch.prim.ListConstruct %77, %int3_1423, %int2_1424, %int32_1425, %int2_1426, %int32_1427 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1344 = torch.aten.view %1318, %1343 : !torch.vtensor<[?,12288],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,2,32],f16>
    torch.bind_symbolic_shape %1344, [%75], affine_map<()[s0] -> (s0, 3, 2, 32, 2, 32)> : !torch.vtensor<[?,3,2,32,2,32],f16>
    %int2_1428 = torch.constant.int 2
    %int32_1429 = torch.constant.int 32
    %int2_1430 = torch.constant.int 2
    %int32_1431 = torch.constant.int 32
    %1345 = torch.prim.ListConstruct %266, %int2_1428, %int32_1429, %int2_1430, %int32_1431 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1346 = torch.aten.view %1344, %1345 : !torch.vtensor<[?,3,2,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,2,32,2,32],f16>
    torch.bind_symbolic_shape %1346, [%75], affine_map<()[s0] -> (s0 * 3, 2, 32, 2, 32)> : !torch.vtensor<[?,2,32,2,32],f16>
    %int0_1432 = torch.constant.int 0
    %1347 = torch.aten.index_select %1346, %int0_1432, %1340 : !torch.vtensor<[?,2,32,2,32],f16>, !torch.int, !torch.vtensor<[?],si64> -> !torch.vtensor<[?,2,32,2,32],f16>
    torch.bind_symbolic_shape %1347, [%74], affine_map<()[s0] -> (s0, 2, 32, 2, 32)> : !torch.vtensor<[?,2,32,2,32],f16>
    %int3_1433 = torch.constant.int 3
    %int2_1434 = torch.constant.int 2
    %int32_1435 = torch.constant.int 32
    %int2_1436 = torch.constant.int 2
    %int32_1437 = torch.constant.int 32
    %1348 = torch.prim.ListConstruct %77, %int3_1433, %int2_1434, %int32_1435, %int2_1436, %int32_1437 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1349 = torch.aten.view %1324, %1348 : !torch.vtensor<[?,12288],f16>, !torch.list<int> -> !torch.vtensor<[?,3,2,32,2,32],f16>
    torch.bind_symbolic_shape %1349, [%75], affine_map<()[s0] -> (s0, 3, 2, 32, 2, 32)> : !torch.vtensor<[?,3,2,32,2,32],f16>
    %int2_1438 = torch.constant.int 2
    %int32_1439 = torch.constant.int 32
    %int2_1440 = torch.constant.int 2
    %int32_1441 = torch.constant.int 32
    %1350 = torch.prim.ListConstruct %266, %int2_1438, %int32_1439, %int2_1440, %int32_1441 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1351 = torch.aten.view %1349, %1350 : !torch.vtensor<[?,3,2,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[?,2,32,2,32],f16>
    torch.bind_symbolic_shape %1351, [%75], affine_map<()[s0] -> (s0 * 3, 2, 32, 2, 32)> : !torch.vtensor<[?,2,32,2,32],f16>
    %int0_1442 = torch.constant.int 0
    %1352 = torch.aten.index_select %1351, %int0_1442, %1342 : !torch.vtensor<[?,2,32,2,32],f16>, !torch.int, !torch.vtensor<[?],si64> -> !torch.vtensor<[?,2,32,2,32],f16>
    torch.bind_symbolic_shape %1352, [%74], affine_map<()[s0] -> (s0, 2, 32, 2, 32)> : !torch.vtensor<[?,2,32,2,32],f16>
    %int1_1443 = torch.constant.int 1
    %int2_1444 = torch.constant.int 2
    %int32_1445 = torch.constant.int 32
    %int2_1446 = torch.constant.int 2
    %int32_1447 = torch.constant.int 32
    %1353 = torch.prim.ListConstruct %int1_1443, %76, %int2_1444, %int32_1445, %int2_1446, %int32_1447 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1354 = torch.aten.view %1347, %1353 : !torch.vtensor<[?,2,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,2,32,2,32],f16>
    torch.bind_symbolic_shape %1354, [%74], affine_map<()[s0] -> (1, s0, 2, 32, 2, 32)> : !torch.vtensor<[1,?,2,32,2,32],f16>
    %int1_1448 = torch.constant.int 1
    %int2_1449 = torch.constant.int 2
    %int32_1450 = torch.constant.int 32
    %int2_1451 = torch.constant.int 2
    %int32_1452 = torch.constant.int 32
    %1355 = torch.prim.ListConstruct %int1_1448, %76, %int2_1449, %int32_1450, %int2_1451, %int32_1452 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1356 = torch.aten.view %1352, %1355 : !torch.vtensor<[?,2,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,2,32,2,32],f16>
    torch.bind_symbolic_shape %1356, [%74], affine_map<()[s0] -> (1, s0, 2, 32, 2, 32)> : !torch.vtensor<[1,?,2,32,2,32],f16>
    %int0_1453 = torch.constant.int 0
    %int0_1454 = torch.constant.int 0
    %int9223372036854775807_1455 = torch.constant.int 9223372036854775807
    %int1_1456 = torch.constant.int 1
    %1357 = torch.aten.slice.Tensor %1354, %int0_1453, %int0_1454, %int9223372036854775807_1455, %int1_1456 : !torch.vtensor<[1,?,2,32,2,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,2,32,2,32],f16>
    torch.bind_symbolic_shape %1357, [%74], affine_map<()[s0] -> (1, s0, 2, 32, 2, 32)> : !torch.vtensor<[1,?,2,32,2,32],f16>
    %int1_1457 = torch.constant.int 1
    %int0_1458 = torch.constant.int 0
    %int9223372036854775807_1459 = torch.constant.int 9223372036854775807
    %int1_1460 = torch.constant.int 1
    %1358 = torch.aten.slice.Tensor %1357, %int1_1457, %int0_1458, %int9223372036854775807_1459, %int1_1460 : !torch.vtensor<[1,?,2,32,2,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,2,32,2,32],f16>
    torch.bind_symbolic_shape %1358, [%74], affine_map<()[s0] -> (1, s0, 2, 32, 2, 32)> : !torch.vtensor<[1,?,2,32,2,32],f16>
    %int2_1461 = torch.constant.int 2
    %int0_1462 = torch.constant.int 0
    %1359 = torch.aten.select.int %1358, %int2_1461, %int0_1462 : !torch.vtensor<[1,?,2,32,2,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,?,32,2,32],f16>
    torch.bind_symbolic_shape %1359, [%74], affine_map<()[s0] -> (1, s0, 32, 2, 32)> : !torch.vtensor<[1,?,32,2,32],f16>
    %int2_1463 = torch.constant.int 2
    %int0_1464 = torch.constant.int 0
    %int1_1465 = torch.constant.int 1
    %1360 = torch.aten.slice.Tensor %1359, %int2_1463, %int0_1464, %78, %int1_1465 : !torch.vtensor<[1,?,32,2,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32,2,32],f16>
    torch.bind_symbolic_shape %1360, [%74], affine_map<()[s0] -> (1, s0, 32, 2, 32)> : !torch.vtensor<[1,?,32,2,32],f16>
    %int0_1466 = torch.constant.int 0
    %int0_1467 = torch.constant.int 0
    %int9223372036854775807_1468 = torch.constant.int 9223372036854775807
    %int1_1469 = torch.constant.int 1
    %1361 = torch.aten.slice.Tensor %1356, %int0_1466, %int0_1467, %int9223372036854775807_1468, %int1_1469 : !torch.vtensor<[1,?,2,32,2,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,2,32,2,32],f16>
    torch.bind_symbolic_shape %1361, [%74], affine_map<()[s0] -> (1, s0, 2, 32, 2, 32)> : !torch.vtensor<[1,?,2,32,2,32],f16>
    %int1_1470 = torch.constant.int 1
    %int0_1471 = torch.constant.int 0
    %int9223372036854775807_1472 = torch.constant.int 9223372036854775807
    %int1_1473 = torch.constant.int 1
    %1362 = torch.aten.slice.Tensor %1361, %int1_1470, %int0_1471, %int9223372036854775807_1472, %int1_1473 : !torch.vtensor<[1,?,2,32,2,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,2,32,2,32],f16>
    torch.bind_symbolic_shape %1362, [%74], affine_map<()[s0] -> (1, s0, 2, 32, 2, 32)> : !torch.vtensor<[1,?,2,32,2,32],f16>
    %int2_1474 = torch.constant.int 2
    %int0_1475 = torch.constant.int 0
    %1363 = torch.aten.select.int %1362, %int2_1474, %int0_1475 : !torch.vtensor<[1,?,2,32,2,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,?,32,2,32],f16>
    torch.bind_symbolic_shape %1363, [%74], affine_map<()[s0] -> (1, s0, 32, 2, 32)> : !torch.vtensor<[1,?,32,2,32],f16>
    %int2_1476 = torch.constant.int 2
    %int0_1477 = torch.constant.int 0
    %int1_1478 = torch.constant.int 1
    %1364 = torch.aten.slice.Tensor %1363, %int2_1476, %int0_1477, %78, %int1_1478 : !torch.vtensor<[1,?,32,2,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32,2,32],f16>
    torch.bind_symbolic_shape %1364, [%74], affine_map<()[s0] -> (1, s0, 32, 2, 32)> : !torch.vtensor<[1,?,32,2,32],f16>
    %int0_1479 = torch.constant.int 0
    %1365 = torch.aten.clone %1360, %int0_1479 : !torch.vtensor<[1,?,32,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,32,2,32],f16>
    torch.bind_symbolic_shape %1365, [%74], affine_map<()[s0] -> (1, s0, 32, 2, 32)> : !torch.vtensor<[1,?,32,2,32],f16>
    %int1_1480 = torch.constant.int 1
    %int2_1481 = torch.constant.int 2
    %int32_1482 = torch.constant.int 32
    %1366 = torch.prim.ListConstruct %int1_1480, %78, %int2_1481, %int32_1482 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1367 = torch.aten._unsafe_view %1365, %1366 : !torch.vtensor<[1,?,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,2,32],f16>
    torch.bind_symbolic_shape %1367, [%74], affine_map<()[s0] -> (1, s0 * 32, 2, 32)> : !torch.vtensor<[1,?,2,32],f16>
    %int0_1483 = torch.constant.int 0
    %1368 = torch.aten.clone %1364, %int0_1483 : !torch.vtensor<[1,?,32,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,32,2,32],f16>
    torch.bind_symbolic_shape %1368, [%74], affine_map<()[s0] -> (1, s0, 32, 2, 32)> : !torch.vtensor<[1,?,32,2,32],f16>
    %int1_1484 = torch.constant.int 1
    %int2_1485 = torch.constant.int 2
    %int32_1486 = torch.constant.int 32
    %1369 = torch.prim.ListConstruct %int1_1484, %78, %int2_1485, %int32_1486 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1370 = torch.aten._unsafe_view %1368, %1369 : !torch.vtensor<[1,?,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,2,32],f16>
    torch.bind_symbolic_shape %1370, [%74], affine_map<()[s0] -> (1, s0 * 32, 2, 32)> : !torch.vtensor<[1,?,2,32],f16>
    %int0_1487 = torch.constant.int 0
    %int0_1488 = torch.constant.int 0
    %int9223372036854775807_1489 = torch.constant.int 9223372036854775807
    %int1_1490 = torch.constant.int 1
    %1371 = torch.aten.slice.Tensor %1367, %int0_1487, %int0_1488, %int9223372036854775807_1489, %int1_1490 : !torch.vtensor<[1,?,2,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,2,32],f16>
    torch.bind_symbolic_shape %1371, [%74], affine_map<()[s0] -> (1, s0 * 32, 2, 32)> : !torch.vtensor<[1,?,2,32],f16>
    %int0_1491 = torch.constant.int 0
    %int0_1492 = torch.constant.int 0
    %int9223372036854775807_1493 = torch.constant.int 9223372036854775807
    %int1_1494 = torch.constant.int 1
    %1372 = torch.aten.slice.Tensor %1370, %int0_1491, %int0_1492, %int9223372036854775807_1493, %int1_1494 : !torch.vtensor<[1,?,2,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,2,32],f16>
    torch.bind_symbolic_shape %1372, [%74], affine_map<()[s0] -> (1, s0 * 32, 2, 32)> : !torch.vtensor<[1,?,2,32],f16>
    %int0_1495 = torch.constant.int 0
    %int0_1496 = torch.constant.int 0
    %int9223372036854775807_1497 = torch.constant.int 9223372036854775807
    %int1_1498 = torch.constant.int 1
    %1373 = torch.aten.slice.Tensor %1354, %int0_1495, %int0_1496, %int9223372036854775807_1497, %int1_1498 : !torch.vtensor<[1,?,2,32,2,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,2,32,2,32],f16>
    torch.bind_symbolic_shape %1373, [%74], affine_map<()[s0] -> (1, s0, 2, 32, 2, 32)> : !torch.vtensor<[1,?,2,32,2,32],f16>
    %int1_1499 = torch.constant.int 1
    %int0_1500 = torch.constant.int 0
    %int9223372036854775807_1501 = torch.constant.int 9223372036854775807
    %int1_1502 = torch.constant.int 1
    %1374 = torch.aten.slice.Tensor %1373, %int1_1499, %int0_1500, %int9223372036854775807_1501, %int1_1502 : !torch.vtensor<[1,?,2,32,2,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,2,32,2,32],f16>
    torch.bind_symbolic_shape %1374, [%74], affine_map<()[s0] -> (1, s0, 2, 32, 2, 32)> : !torch.vtensor<[1,?,2,32,2,32],f16>
    %int2_1503 = torch.constant.int 2
    %int1_1504 = torch.constant.int 1
    %1375 = torch.aten.select.int %1374, %int2_1503, %int1_1504 : !torch.vtensor<[1,?,2,32,2,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,?,32,2,32],f16>
    torch.bind_symbolic_shape %1375, [%74], affine_map<()[s0] -> (1, s0, 32, 2, 32)> : !torch.vtensor<[1,?,32,2,32],f16>
    %int2_1505 = torch.constant.int 2
    %int0_1506 = torch.constant.int 0
    %int1_1507 = torch.constant.int 1
    %1376 = torch.aten.slice.Tensor %1375, %int2_1505, %int0_1506, %78, %int1_1507 : !torch.vtensor<[1,?,32,2,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32,2,32],f16>
    torch.bind_symbolic_shape %1376, [%74], affine_map<()[s0] -> (1, s0, 32, 2, 32)> : !torch.vtensor<[1,?,32,2,32],f16>
    %int0_1508 = torch.constant.int 0
    %int0_1509 = torch.constant.int 0
    %int9223372036854775807_1510 = torch.constant.int 9223372036854775807
    %int1_1511 = torch.constant.int 1
    %1377 = torch.aten.slice.Tensor %1356, %int0_1508, %int0_1509, %int9223372036854775807_1510, %int1_1511 : !torch.vtensor<[1,?,2,32,2,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,2,32,2,32],f16>
    torch.bind_symbolic_shape %1377, [%74], affine_map<()[s0] -> (1, s0, 2, 32, 2, 32)> : !torch.vtensor<[1,?,2,32,2,32],f16>
    %int1_1512 = torch.constant.int 1
    %int0_1513 = torch.constant.int 0
    %int9223372036854775807_1514 = torch.constant.int 9223372036854775807
    %int1_1515 = torch.constant.int 1
    %1378 = torch.aten.slice.Tensor %1377, %int1_1512, %int0_1513, %int9223372036854775807_1514, %int1_1515 : !torch.vtensor<[1,?,2,32,2,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,2,32,2,32],f16>
    torch.bind_symbolic_shape %1378, [%74], affine_map<()[s0] -> (1, s0, 2, 32, 2, 32)> : !torch.vtensor<[1,?,2,32,2,32],f16>
    %int2_1516 = torch.constant.int 2
    %int1_1517 = torch.constant.int 1
    %1379 = torch.aten.select.int %1378, %int2_1516, %int1_1517 : !torch.vtensor<[1,?,2,32,2,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,?,32,2,32],f16>
    torch.bind_symbolic_shape %1379, [%74], affine_map<()[s0] -> (1, s0, 32, 2, 32)> : !torch.vtensor<[1,?,32,2,32],f16>
    %int2_1518 = torch.constant.int 2
    %int0_1519 = torch.constant.int 0
    %int1_1520 = torch.constant.int 1
    %1380 = torch.aten.slice.Tensor %1379, %int2_1518, %int0_1519, %78, %int1_1520 : !torch.vtensor<[1,?,32,2,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,32,2,32],f16>
    torch.bind_symbolic_shape %1380, [%74], affine_map<()[s0] -> (1, s0, 32, 2, 32)> : !torch.vtensor<[1,?,32,2,32],f16>
    %int0_1521 = torch.constant.int 0
    %1381 = torch.aten.clone %1376, %int0_1521 : !torch.vtensor<[1,?,32,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,32,2,32],f16>
    torch.bind_symbolic_shape %1381, [%74], affine_map<()[s0] -> (1, s0, 32, 2, 32)> : !torch.vtensor<[1,?,32,2,32],f16>
    %int1_1522 = torch.constant.int 1
    %int2_1523 = torch.constant.int 2
    %int32_1524 = torch.constant.int 32
    %1382 = torch.prim.ListConstruct %int1_1522, %78, %int2_1523, %int32_1524 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1383 = torch.aten._unsafe_view %1381, %1382 : !torch.vtensor<[1,?,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,2,32],f16>
    torch.bind_symbolic_shape %1383, [%74], affine_map<()[s0] -> (1, s0 * 32, 2, 32)> : !torch.vtensor<[1,?,2,32],f16>
    %int0_1525 = torch.constant.int 0
    %1384 = torch.aten.clone %1380, %int0_1525 : !torch.vtensor<[1,?,32,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,32,2,32],f16>
    torch.bind_symbolic_shape %1384, [%74], affine_map<()[s0] -> (1, s0, 32, 2, 32)> : !torch.vtensor<[1,?,32,2,32],f16>
    %int1_1526 = torch.constant.int 1
    %int2_1527 = torch.constant.int 2
    %int32_1528 = torch.constant.int 32
    %1385 = torch.prim.ListConstruct %int1_1526, %78, %int2_1527, %int32_1528 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1386 = torch.aten._unsafe_view %1384, %1385 : !torch.vtensor<[1,?,32,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,2,32],f16>
    torch.bind_symbolic_shape %1386, [%74], affine_map<()[s0] -> (1, s0 * 32, 2, 32)> : !torch.vtensor<[1,?,2,32],f16>
    %int0_1529 = torch.constant.int 0
    %int0_1530 = torch.constant.int 0
    %int9223372036854775807_1531 = torch.constant.int 9223372036854775807
    %int1_1532 = torch.constant.int 1
    %1387 = torch.aten.slice.Tensor %1383, %int0_1529, %int0_1530, %int9223372036854775807_1531, %int1_1532 : !torch.vtensor<[1,?,2,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,2,32],f16>
    torch.bind_symbolic_shape %1387, [%74], affine_map<()[s0] -> (1, s0 * 32, 2, 32)> : !torch.vtensor<[1,?,2,32],f16>
    %int0_1533 = torch.constant.int 0
    %int0_1534 = torch.constant.int 0
    %int9223372036854775807_1535 = torch.constant.int 9223372036854775807
    %int1_1536 = torch.constant.int 1
    %1388 = torch.aten.slice.Tensor %1386, %int0_1533, %int0_1534, %int9223372036854775807_1535, %int1_1536 : !torch.vtensor<[1,?,2,32],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,?,2,32],f16>
    torch.bind_symbolic_shape %1388, [%74], affine_map<()[s0] -> (1, s0 * 32, 2, 32)> : !torch.vtensor<[1,?,2,32],f16>
    %int-2_1537 = torch.constant.int -2
    %1389 = torch.aten.unsqueeze %1371, %int-2_1537 : !torch.vtensor<[1,?,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,2,1,32],f16>
    torch.bind_symbolic_shape %1389, [%74], affine_map<()[s0] -> (1, s0 * 32, 2, 1, 32)> : !torch.vtensor<[1,?,2,1,32],f16>
    %int-2_1538 = torch.constant.int -2
    %1390 = torch.aten.unsqueeze %1372, %int-2_1538 : !torch.vtensor<[1,?,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,2,1,32],f16>
    torch.bind_symbolic_shape %1390, [%74], affine_map<()[s0] -> (1, s0 * 32, 2, 1, 32)> : !torch.vtensor<[1,?,2,1,32],f16>
    %int1_1539 = torch.constant.int 1
    %int2_1540 = torch.constant.int 2
    %int2_1541 = torch.constant.int 2
    %int32_1542 = torch.constant.int 32
    %1391 = torch.prim.ListConstruct %int1_1539, %78, %int2_1540, %int2_1541, %int32_1542 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_1543 = torch.constant.bool false
    %1392 = torch.aten.expand %1389, %1391, %false_1543 : !torch.vtensor<[1,?,2,1,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,?,2,2,32],f16>
    torch.bind_symbolic_shape %1392, [%74], affine_map<()[s0] -> (1, s0 * 32, 2, 2, 32)> : !torch.vtensor<[1,?,2,2,32],f16>
    %int1_1544 = torch.constant.int 1
    %int2_1545 = torch.constant.int 2
    %int2_1546 = torch.constant.int 2
    %int32_1547 = torch.constant.int 32
    %1393 = torch.prim.ListConstruct %int1_1544, %78, %int2_1545, %int2_1546, %int32_1547 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_1548 = torch.constant.bool false
    %1394 = torch.aten.expand %1390, %1393, %false_1548 : !torch.vtensor<[1,?,2,1,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,?,2,2,32],f16>
    torch.bind_symbolic_shape %1394, [%74], affine_map<()[s0] -> (1, s0 * 32, 2, 2, 32)> : !torch.vtensor<[1,?,2,2,32],f16>
    %int0_1549 = torch.constant.int 0
    %1395 = torch.aten.clone %1392, %int0_1549 : !torch.vtensor<[1,?,2,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,2,2,32],f16>
    torch.bind_symbolic_shape %1395, [%74], affine_map<()[s0] -> (1, s0 * 32, 2, 2, 32)> : !torch.vtensor<[1,?,2,2,32],f16>
    %int1_1550 = torch.constant.int 1
    %int4_1551 = torch.constant.int 4
    %int32_1552 = torch.constant.int 32
    %1396 = torch.prim.ListConstruct %int1_1550, %78, %int4_1551, %int32_1552 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1397 = torch.aten._unsafe_view %1395, %1396 : !torch.vtensor<[1,?,2,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %1397, [%74], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int0_1553 = torch.constant.int 0
    %1398 = torch.aten.clone %1394, %int0_1553 : !torch.vtensor<[1,?,2,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,2,2,32],f16>
    torch.bind_symbolic_shape %1398, [%74], affine_map<()[s0] -> (1, s0 * 32, 2, 2, 32)> : !torch.vtensor<[1,?,2,2,32],f16>
    %int1_1554 = torch.constant.int 1
    %int4_1555 = torch.constant.int 4
    %int32_1556 = torch.constant.int 32
    %1399 = torch.prim.ListConstruct %int1_1554, %78, %int4_1555, %int32_1556 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1400 = torch.aten._unsafe_view %1398, %1399 : !torch.vtensor<[1,?,2,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %1400, [%74], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int-2_1557 = torch.constant.int -2
    %1401 = torch.aten.unsqueeze %1387, %int-2_1557 : !torch.vtensor<[1,?,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,2,1,32],f16>
    torch.bind_symbolic_shape %1401, [%74], affine_map<()[s0] -> (1, s0 * 32, 2, 1, 32)> : !torch.vtensor<[1,?,2,1,32],f16>
    %int-2_1558 = torch.constant.int -2
    %1402 = torch.aten.unsqueeze %1388, %int-2_1558 : !torch.vtensor<[1,?,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,2,1,32],f16>
    torch.bind_symbolic_shape %1402, [%74], affine_map<()[s0] -> (1, s0 * 32, 2, 1, 32)> : !torch.vtensor<[1,?,2,1,32],f16>
    %int1_1559 = torch.constant.int 1
    %int2_1560 = torch.constant.int 2
    %int2_1561 = torch.constant.int 2
    %int32_1562 = torch.constant.int 32
    %1403 = torch.prim.ListConstruct %int1_1559, %78, %int2_1560, %int2_1561, %int32_1562 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_1563 = torch.constant.bool false
    %1404 = torch.aten.expand %1401, %1403, %false_1563 : !torch.vtensor<[1,?,2,1,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,?,2,2,32],f16>
    torch.bind_symbolic_shape %1404, [%74], affine_map<()[s0] -> (1, s0 * 32, 2, 2, 32)> : !torch.vtensor<[1,?,2,2,32],f16>
    %int1_1564 = torch.constant.int 1
    %int2_1565 = torch.constant.int 2
    %int2_1566 = torch.constant.int 2
    %int32_1567 = torch.constant.int 32
    %1405 = torch.prim.ListConstruct %int1_1564, %78, %int2_1565, %int2_1566, %int32_1567 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_1568 = torch.constant.bool false
    %1406 = torch.aten.expand %1402, %1405, %false_1568 : !torch.vtensor<[1,?,2,1,32],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,?,2,2,32],f16>
    torch.bind_symbolic_shape %1406, [%74], affine_map<()[s0] -> (1, s0 * 32, 2, 2, 32)> : !torch.vtensor<[1,?,2,2,32],f16>
    %int0_1569 = torch.constant.int 0
    %1407 = torch.aten.clone %1404, %int0_1569 : !torch.vtensor<[1,?,2,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,2,2,32],f16>
    torch.bind_symbolic_shape %1407, [%74], affine_map<()[s0] -> (1, s0 * 32, 2, 2, 32)> : !torch.vtensor<[1,?,2,2,32],f16>
    %int1_1570 = torch.constant.int 1
    %int4_1571 = torch.constant.int 4
    %int32_1572 = torch.constant.int 32
    %1408 = torch.prim.ListConstruct %int1_1570, %78, %int4_1571, %int32_1572 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1409 = torch.aten._unsafe_view %1407, %1408 : !torch.vtensor<[1,?,2,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %1409, [%74], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int0_1573 = torch.constant.int 0
    %1410 = torch.aten.clone %1406, %int0_1573 : !torch.vtensor<[1,?,2,2,32],f16>, !torch.int -> !torch.vtensor<[1,?,2,2,32],f16>
    torch.bind_symbolic_shape %1410, [%74], affine_map<()[s0] -> (1, s0 * 32, 2, 2, 32)> : !torch.vtensor<[1,?,2,2,32],f16>
    %int1_1574 = torch.constant.int 1
    %int4_1575 = torch.constant.int 4
    %int32_1576 = torch.constant.int 32
    %1411 = torch.prim.ListConstruct %int1_1574, %78, %int4_1575, %int32_1576 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1412 = torch.aten._unsafe_view %1410, %1411 : !torch.vtensor<[1,?,2,2,32],f16>, !torch.list<int> -> !torch.vtensor<[1,?,4,32],f16>
    torch.bind_symbolic_shape %1412, [%74], affine_map<()[s0] -> (1, s0 * 32, 4, 32)> : !torch.vtensor<[1,?,4,32],f16>
    %int1_1577 = torch.constant.int 1
    %int2_1578 = torch.constant.int 2
    %1413 = torch.aten.transpose.int %1174, %int1_1577, %int2_1578 : !torch.vtensor<[1,1,4,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,4,1,32],f16>
    %int1_1579 = torch.constant.int 1
    %int2_1580 = torch.constant.int 2
    %1414 = torch.aten.transpose.int %1180, %int1_1579, %int2_1580 : !torch.vtensor<[1,1,4,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,4,1,32],f16>
    %int1_1581 = torch.constant.int 1
    %int2_1582 = torch.constant.int 2
    %1415 = torch.aten.transpose.int %1397, %int1_1581, %int2_1582 : !torch.vtensor<[1,?,4,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,4,?,32],f16>
    torch.bind_symbolic_shape %1415, [%74], affine_map<()[s0] -> (1, 4, s0 * 32, 32)> : !torch.vtensor<[1,4,?,32],f16>
    %int1_1583 = torch.constant.int 1
    %int2_1584 = torch.constant.int 2
    %1416 = torch.aten.transpose.int %1400, %int1_1583, %int2_1584 : !torch.vtensor<[1,?,4,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,4,?,32],f16>
    torch.bind_symbolic_shape %1416, [%74], affine_map<()[s0] -> (1, 4, s0 * 32, 32)> : !torch.vtensor<[1,4,?,32],f16>
    %int1_1585 = torch.constant.int 1
    %int2_1586 = torch.constant.int 2
    %1417 = torch.aten.transpose.int %1409, %int1_1585, %int2_1586 : !torch.vtensor<[1,?,4,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,4,?,32],f16>
    torch.bind_symbolic_shape %1417, [%74], affine_map<()[s0] -> (1, 4, s0 * 32, 32)> : !torch.vtensor<[1,4,?,32],f16>
    %int1_1587 = torch.constant.int 1
    %int2_1588 = torch.constant.int 2
    %1418 = torch.aten.transpose.int %1412, %int1_1587, %int2_1588 : !torch.vtensor<[1,?,4,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,4,?,32],f16>
    torch.bind_symbolic_shape %1418, [%74], affine_map<()[s0] -> (1, 4, s0 * 32, 32)> : !torch.vtensor<[1,4,?,32],f16>
    %int5_1589 = torch.constant.int 5
    %1419 = torch.prims.convert_element_type %1413, %int5_1589 : !torch.vtensor<[1,4,1,32],f16>, !torch.int -> !torch.vtensor<[1,4,1,32],f16>
    %int5_1590 = torch.constant.int 5
    %1420 = torch.prims.convert_element_type %1414, %int5_1590 : !torch.vtensor<[1,4,1,32],f16>, !torch.int -> !torch.vtensor<[1,4,1,32],f16>
    %int5_1591 = torch.constant.int 5
    %1421 = torch.prims.convert_element_type %1415, %int5_1591 : !torch.vtensor<[1,4,?,32],f16>, !torch.int -> !torch.vtensor<[1,4,?,32],f16>
    torch.bind_symbolic_shape %1421, [%74], affine_map<()[s0] -> (1, 4, s0 * 32, 32)> : !torch.vtensor<[1,4,?,32],f16>
    %int5_1592 = torch.constant.int 5
    %1422 = torch.prims.convert_element_type %1416, %int5_1592 : !torch.vtensor<[1,4,?,32],f16>, !torch.int -> !torch.vtensor<[1,4,?,32],f16>
    torch.bind_symbolic_shape %1422, [%74], affine_map<()[s0] -> (1, 4, s0 * 32, 32)> : !torch.vtensor<[1,4,?,32],f16>
    %int5_1593 = torch.constant.int 5
    %1423 = torch.prims.convert_element_type %1417, %int5_1593 : !torch.vtensor<[1,4,?,32],f16>, !torch.int -> !torch.vtensor<[1,4,?,32],f16>
    torch.bind_symbolic_shape %1423, [%74], affine_map<()[s0] -> (1, 4, s0 * 32, 32)> : !torch.vtensor<[1,4,?,32],f16>
    %int5_1594 = torch.constant.int 5
    %1424 = torch.prims.convert_element_type %1418, %int5_1594 : !torch.vtensor<[1,4,?,32],f16>, !torch.int -> !torch.vtensor<[1,4,?,32],f16>
    torch.bind_symbolic_shape %1424, [%74], affine_map<()[s0] -> (1, 4, s0 * 32, 32)> : !torch.vtensor<[1,4,?,32],f16>
    %int5_1595 = torch.constant.int 5
    %1425 = torch.prims.convert_element_type %97, %int5_1595 : !torch.vtensor<[1,1,1,?],f16>, !torch.int -> !torch.vtensor<[1,1,1,?],f16>
    torch.bind_symbolic_shape %1425, [%74], affine_map<()[s0] -> (1, 1, 1, s0 * 32)> : !torch.vtensor<[1,1,1,?],f16>
    %int5_1596 = torch.constant.int 5
    %1426 = torch.prims.convert_element_type %100, %int5_1596 : !torch.vtensor<[1,1,1,?],f16>, !torch.int -> !torch.vtensor<[1,1,1,?],f16>
    torch.bind_symbolic_shape %1426, [%74], affine_map<()[s0] -> (1, 1, 1, s0 * 32)> : !torch.vtensor<[1,1,1,?],f16>
    %float0.000000e00_1597 = torch.constant.float 0.000000e+00
    %false_1598 = torch.constant.bool false
    %none_1599 = torch.constant.none
    %1427:2 = torch.operator "torch.aten._scaled_dot_product_flash_attention_for_cpu"(%1419, %1421, %1423, %float0.000000e00_1597, %false_1598, %1425, %none_1599) : (!torch.vtensor<[1,4,1,32],f16>, !torch.vtensor<[1,4,?,32],f16>, !torch.vtensor<[1,4,?,32],f16>, !torch.float, !torch.bool, !torch.vtensor<[1,1,1,?],f16>, !torch.none) -> (!torch.vtensor<[1,4,1,32],f16>, !torch.vtensor<[1,4,1],f32>) 
    %float0.000000e00_1600 = torch.constant.float 0.000000e+00
    %false_1601 = torch.constant.bool false
    %none_1602 = torch.constant.none
    %1428:2 = torch.operator "torch.aten._scaled_dot_product_flash_attention_for_cpu"(%1420, %1422, %1424, %float0.000000e00_1600, %false_1601, %1426, %none_1602) : (!torch.vtensor<[1,4,1,32],f16>, !torch.vtensor<[1,4,?,32],f16>, !torch.vtensor<[1,4,?,32],f16>, !torch.float, !torch.bool, !torch.vtensor<[1,1,1,?],f16>, !torch.none) -> (!torch.vtensor<[1,4,1,32],f16>, !torch.vtensor<[1,4,1],f32>) 
    %int1_1603 = torch.constant.int 1
    %int2_1604 = torch.constant.int 2
    %1429 = torch.aten.transpose.int %1427#0, %int1_1603, %int2_1604 : !torch.vtensor<[1,4,1,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,1,4,32],f16>
    %int1_1605 = torch.constant.int 1
    %int2_1606 = torch.constant.int 2
    %1430 = torch.aten.transpose.int %1428#0, %int1_1605, %int2_1606 : !torch.vtensor<[1,4,1,32],f16>, !torch.int, !torch.int -> !torch.vtensor<[1,1,4,32],f16>
    %int1_1607 = torch.constant.int 1
    %int1_1608 = torch.constant.int 1
    %int128_1609 = torch.constant.int 128
    %1431 = torch.prim.ListConstruct %int1_1607, %int1_1608, %int128_1609 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1432 = torch.aten.view %1429, %1431 : !torch.vtensor<[1,1,4,32],f16>, !torch.list<int> -> !torch.vtensor<[1,1,128],f16>
    %int1_1610 = torch.constant.int 1
    %int1_1611 = torch.constant.int 1
    %int128_1612 = torch.constant.int 128
    %1433 = torch.prim.ListConstruct %int1_1610, %int1_1611, %int128_1612 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1434 = torch.aten.view %1430, %1433 : !torch.vtensor<[1,1,4,32],f16>, !torch.list<int> -> !torch.vtensor<[1,1,128],f16>
    %int1_1613 = torch.constant.int 1
    %int0_1614 = torch.constant.int 0
    %1435 = torch.prim.ListConstruct %int1_1613, %int0_1614 : (!torch.int, !torch.int) -> !torch.list<int>
    %1436 = torch.aten.permute %58, %1435 : !torch.vtensor<[256,128],f32>, !torch.list<int> -> !torch.vtensor<[128,256],f32>
    %int1_1615 = torch.constant.int 1
    %int0_1616 = torch.constant.int 0
    %1437 = torch.prim.ListConstruct %int1_1615, %int0_1616 : (!torch.int, !torch.int) -> !torch.list<int>
    %1438 = torch.aten.permute %59, %1437 : !torch.vtensor<[256,128],f32>, !torch.list<int> -> !torch.vtensor<[128,256],f32>
    %int5_1617 = torch.constant.int 5
    %1439 = torch.prims.convert_element_type %1436, %int5_1617 : !torch.vtensor<[128,256],f32>, !torch.int -> !torch.vtensor<[128,256],f16>
    %int1_1618 = torch.constant.int 1
    %int128_1619 = torch.constant.int 128
    %1440 = torch.prim.ListConstruct %int1_1618, %int128_1619 : (!torch.int, !torch.int) -> !torch.list<int>
    %1441 = torch.aten.view %1432, %1440 : !torch.vtensor<[1,1,128],f16>, !torch.list<int> -> !torch.vtensor<[1,128],f16>
    %1442 = torch.aten.mm %1441, %1439 : !torch.vtensor<[1,128],f16>, !torch.vtensor<[128,256],f16> -> !torch.vtensor<[1,256],f16>
    %int1_1620 = torch.constant.int 1
    %int1_1621 = torch.constant.int 1
    %int256_1622 = torch.constant.int 256
    %1443 = torch.prim.ListConstruct %int1_1620, %int1_1621, %int256_1622 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1444 = torch.aten.view %1442, %1443 : !torch.vtensor<[1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,1,256],f16>
    %int5_1623 = torch.constant.int 5
    %1445 = torch.prims.convert_element_type %1438, %int5_1623 : !torch.vtensor<[128,256],f32>, !torch.int -> !torch.vtensor<[128,256],f16>
    %int1_1624 = torch.constant.int 1
    %int128_1625 = torch.constant.int 128
    %1446 = torch.prim.ListConstruct %int1_1624, %int128_1625 : (!torch.int, !torch.int) -> !torch.list<int>
    %1447 = torch.aten.view %1434, %1446 : !torch.vtensor<[1,1,128],f16>, !torch.list<int> -> !torch.vtensor<[1,128],f16>
    %1448 = torch.aten.mm %1447, %1445 : !torch.vtensor<[1,128],f16>, !torch.vtensor<[128,256],f16> -> !torch.vtensor<[1,256],f16>
    %int1_1626 = torch.constant.int 1
    %int1_1627 = torch.constant.int 1
    %int256_1628 = torch.constant.int 256
    %1449 = torch.prim.ListConstruct %int1_1626, %int1_1627, %int256_1628 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1450 = torch.aten.view %1448, %1449 : !torch.vtensor<[1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,1,256],f16>
    %1451 = torch_c.to_builtin_tensor %1444 : !torch.vtensor<[1,1,256],f16> -> tensor<1x1x256xf16>
    %1452 = flow.tensor.barrier %1451 : tensor<1x1x256xf16> on #hal.device.promise<@__device_0>
    %1453 = torch_c.from_builtin_tensor %1452 : tensor<1x1x256xf16> -> !torch.vtensor<[1,1,256],f16>
    %1454 = torch_c.to_builtin_tensor %1450 : !torch.vtensor<[1,1,256],f16> -> tensor<1x1x256xf16>
    %1455 = flow.tensor.transfer %1454 : tensor<1x1x256xf16> to #hal.device.promise<@__device_0>
    %1456 = torch_c.from_builtin_tensor %1455 : tensor<1x1x256xf16> -> !torch.vtensor<[1,1,256],f16>
    %int1_1629 = torch.constant.int 1
    %1457 = torch.aten.add.Tensor %1453, %1456, %int1_1629 : !torch.vtensor<[1,1,256],f16>, !torch.vtensor<[1,1,256],f16>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %1458 = torch_c.to_builtin_tensor %1457 : !torch.vtensor<[1,1,256],f16> -> tensor<1x1x256xf16>
    %1459 = flow.tensor.barrier %1458 : tensor<1x1x256xf16> on #hal.device.promise<@__device_0>
    %1460 = torch_c.from_builtin_tensor %1459 : tensor<1x1x256xf16> -> !torch.vtensor<[1,1,256],f16>
    %1461 = torch_c.to_builtin_tensor %1457 : !torch.vtensor<[1,1,256],f16> -> tensor<1x1x256xf16>
    %1462 = flow.tensor.transfer %1461 : tensor<1x1x256xf16> to #hal.device.promise<@__device_1>
    %1463 = torch_c.from_builtin_tensor %1462 : tensor<1x1x256xf16> -> !torch.vtensor<[1,1,256],f16>
    %int1_1630 = torch.constant.int 1
    %1464 = torch.aten.add.Tensor %1087, %1460, %int1_1630 : !torch.vtensor<[1,1,256],f16>, !torch.vtensor<[1,1,256],f16>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %int1_1631 = torch.constant.int 1
    %1465 = torch.aten.add.Tensor %1088, %1463, %int1_1631 : !torch.vtensor<[1,1,256],f16>, !torch.vtensor<[1,1,256],f16>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %int6_1632 = torch.constant.int 6
    %1466 = torch.prims.convert_element_type %1464, %int6_1632 : !torch.vtensor<[1,1,256],f16>, !torch.int -> !torch.vtensor<[1,1,256],f32>
    %int6_1633 = torch.constant.int 6
    %1467 = torch.prims.convert_element_type %1465, %int6_1633 : !torch.vtensor<[1,1,256],f16>, !torch.int -> !torch.vtensor<[1,1,256],f32>
    %int2_1634 = torch.constant.int 2
    %1468 = torch.aten.pow.Tensor_Scalar %1466, %int2_1634 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f32>
    %int2_1635 = torch.constant.int 2
    %1469 = torch.aten.pow.Tensor_Scalar %1467, %int2_1635 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f32>
    %int-1_1636 = torch.constant.int -1
    %1470 = torch.prim.ListConstruct %int-1_1636 : (!torch.int) -> !torch.list<int>
    %true_1637 = torch.constant.bool true
    %none_1638 = torch.constant.none
    %1471 = torch.aten.mean.dim %1468, %1470, %true_1637, %none_1638 : !torch.vtensor<[1,1,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1],f32>
    %int-1_1639 = torch.constant.int -1
    %1472 = torch.prim.ListConstruct %int-1_1639 : (!torch.int) -> !torch.list<int>
    %true_1640 = torch.constant.bool true
    %none_1641 = torch.constant.none
    %1473 = torch.aten.mean.dim %1469, %1472, %true_1640, %none_1641 : !torch.vtensor<[1,1,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1],f32>
    %float1.000000e-02_1642 = torch.constant.float 1.000000e-02
    %int1_1643 = torch.constant.int 1
    %1474 = torch.aten.add.Scalar %1471, %float1.000000e-02_1642, %int1_1643 : !torch.vtensor<[1,1,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,1,1],f32>
    %float1.000000e-02_1644 = torch.constant.float 1.000000e-02
    %int1_1645 = torch.constant.int 1
    %1475 = torch.aten.add.Scalar %1473, %float1.000000e-02_1644, %int1_1645 : !torch.vtensor<[1,1,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,1,1],f32>
    %1476 = torch.aten.rsqrt %1474 : !torch.vtensor<[1,1,1],f32> -> !torch.vtensor<[1,1,1],f32>
    %1477 = torch.aten.rsqrt %1475 : !torch.vtensor<[1,1,1],f32> -> !torch.vtensor<[1,1,1],f32>
    %1478 = torch.aten.mul.Tensor %1466, %1476 : !torch.vtensor<[1,1,256],f32>, !torch.vtensor<[1,1,1],f32> -> !torch.vtensor<[1,1,256],f32>
    %1479 = torch.aten.mul.Tensor %1467, %1477 : !torch.vtensor<[1,1,256],f32>, !torch.vtensor<[1,1,1],f32> -> !torch.vtensor<[1,1,256],f32>
    %int5_1646 = torch.constant.int 5
    %1480 = torch.prims.convert_element_type %1478, %int5_1646 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %int5_1647 = torch.constant.int 5
    %1481 = torch.prims.convert_element_type %1479, %int5_1647 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %1482 = torch.aten.mul.Tensor %60, %1480 : !torch.vtensor<[256],f32>, !torch.vtensor<[1,1,256],f16> -> !torch.vtensor<[1,1,256],f32>
    %1483 = torch.aten.mul.Tensor %61, %1481 : !torch.vtensor<[256],f32>, !torch.vtensor<[1,1,256],f16> -> !torch.vtensor<[1,1,256],f32>
    %int5_1648 = torch.constant.int 5
    %1484 = torch.prims.convert_element_type %1482, %int5_1648 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %int5_1649 = torch.constant.int 5
    %1485 = torch.prims.convert_element_type %1483, %int5_1649 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %int1_1650 = torch.constant.int 1
    %int0_1651 = torch.constant.int 0
    %1486 = torch.prim.ListConstruct %int1_1650, %int0_1651 : (!torch.int, !torch.int) -> !torch.list<int>
    %1487 = torch.aten.permute %62, %1486 : !torch.vtensor<[12,256],f32>, !torch.list<int> -> !torch.vtensor<[256,12],f32>
    %int1_1652 = torch.constant.int 1
    %int0_1653 = torch.constant.int 0
    %1488 = torch.prim.ListConstruct %int1_1652, %int0_1653 : (!torch.int, !torch.int) -> !torch.list<int>
    %1489 = torch.aten.permute %63, %1488 : !torch.vtensor<[11,256],f32>, !torch.list<int> -> !torch.vtensor<[256,11],f32>
    %int5_1654 = torch.constant.int 5
    %1490 = torch.prims.convert_element_type %1487, %int5_1654 : !torch.vtensor<[256,12],f32>, !torch.int -> !torch.vtensor<[256,12],f16>
    %int1_1655 = torch.constant.int 1
    %int256_1656 = torch.constant.int 256
    %1491 = torch.prim.ListConstruct %int1_1655, %int256_1656 : (!torch.int, !torch.int) -> !torch.list<int>
    %1492 = torch.aten.view %1484, %1491 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %1493 = torch.aten.mm %1492, %1490 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,12],f16> -> !torch.vtensor<[1,12],f16>
    %int1_1657 = torch.constant.int 1
    %int1_1658 = torch.constant.int 1
    %int12_1659 = torch.constant.int 12
    %1494 = torch.prim.ListConstruct %int1_1657, %int1_1658, %int12_1659 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1495 = torch.aten.view %1493, %1494 : !torch.vtensor<[1,12],f16>, !torch.list<int> -> !torch.vtensor<[1,1,12],f16>
    %int5_1660 = torch.constant.int 5
    %1496 = torch.prims.convert_element_type %1489, %int5_1660 : !torch.vtensor<[256,11],f32>, !torch.int -> !torch.vtensor<[256,11],f16>
    %int1_1661 = torch.constant.int 1
    %int256_1662 = torch.constant.int 256
    %1497 = torch.prim.ListConstruct %int1_1661, %int256_1662 : (!torch.int, !torch.int) -> !torch.list<int>
    %1498 = torch.aten.view %1485, %1497 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %1499 = torch.aten.mm %1498, %1496 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,11],f16> -> !torch.vtensor<[1,11],f16>
    %int1_1663 = torch.constant.int 1
    %int1_1664 = torch.constant.int 1
    %int11_1665 = torch.constant.int 11
    %1500 = torch.prim.ListConstruct %int1_1663, %int1_1664, %int11_1665 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1501 = torch.aten.view %1499, %1500 : !torch.vtensor<[1,11],f16>, !torch.list<int> -> !torch.vtensor<[1,1,11],f16>
    %1502 = torch.aten.silu %1495 : !torch.vtensor<[1,1,12],f16> -> !torch.vtensor<[1,1,12],f16>
    %1503 = torch.aten.silu %1501 : !torch.vtensor<[1,1,11],f16> -> !torch.vtensor<[1,1,11],f16>
    %int1_1666 = torch.constant.int 1
    %int0_1667 = torch.constant.int 0
    %1504 = torch.prim.ListConstruct %int1_1666, %int0_1667 : (!torch.int, !torch.int) -> !torch.list<int>
    %1505 = torch.aten.permute %64, %1504 : !torch.vtensor<[12,256],f32>, !torch.list<int> -> !torch.vtensor<[256,12],f32>
    %int1_1668 = torch.constant.int 1
    %int0_1669 = torch.constant.int 0
    %1506 = torch.prim.ListConstruct %int1_1668, %int0_1669 : (!torch.int, !torch.int) -> !torch.list<int>
    %1507 = torch.aten.permute %65, %1506 : !torch.vtensor<[11,256],f32>, !torch.list<int> -> !torch.vtensor<[256,11],f32>
    %int5_1670 = torch.constant.int 5
    %1508 = torch.prims.convert_element_type %1505, %int5_1670 : !torch.vtensor<[256,12],f32>, !torch.int -> !torch.vtensor<[256,12],f16>
    %int1_1671 = torch.constant.int 1
    %int256_1672 = torch.constant.int 256
    %1509 = torch.prim.ListConstruct %int1_1671, %int256_1672 : (!torch.int, !torch.int) -> !torch.list<int>
    %1510 = torch.aten.view %1484, %1509 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %1511 = torch.aten.mm %1510, %1508 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,12],f16> -> !torch.vtensor<[1,12],f16>
    %int1_1673 = torch.constant.int 1
    %int1_1674 = torch.constant.int 1
    %int12_1675 = torch.constant.int 12
    %1512 = torch.prim.ListConstruct %int1_1673, %int1_1674, %int12_1675 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1513 = torch.aten.view %1511, %1512 : !torch.vtensor<[1,12],f16>, !torch.list<int> -> !torch.vtensor<[1,1,12],f16>
    %int5_1676 = torch.constant.int 5
    %1514 = torch.prims.convert_element_type %1507, %int5_1676 : !torch.vtensor<[256,11],f32>, !torch.int -> !torch.vtensor<[256,11],f16>
    %int1_1677 = torch.constant.int 1
    %int256_1678 = torch.constant.int 256
    %1515 = torch.prim.ListConstruct %int1_1677, %int256_1678 : (!torch.int, !torch.int) -> !torch.list<int>
    %1516 = torch.aten.view %1485, %1515 : !torch.vtensor<[1,1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,256],f16>
    %1517 = torch.aten.mm %1516, %1514 : !torch.vtensor<[1,256],f16>, !torch.vtensor<[256,11],f16> -> !torch.vtensor<[1,11],f16>
    %int1_1679 = torch.constant.int 1
    %int1_1680 = torch.constant.int 1
    %int11_1681 = torch.constant.int 11
    %1518 = torch.prim.ListConstruct %int1_1679, %int1_1680, %int11_1681 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1519 = torch.aten.view %1517, %1518 : !torch.vtensor<[1,11],f16>, !torch.list<int> -> !torch.vtensor<[1,1,11],f16>
    %1520 = torch.aten.mul.Tensor %1502, %1513 : !torch.vtensor<[1,1,12],f16>, !torch.vtensor<[1,1,12],f16> -> !torch.vtensor<[1,1,12],f16>
    %1521 = torch.aten.mul.Tensor %1503, %1519 : !torch.vtensor<[1,1,11],f16>, !torch.vtensor<[1,1,11],f16> -> !torch.vtensor<[1,1,11],f16>
    %int1_1682 = torch.constant.int 1
    %int0_1683 = torch.constant.int 0
    %1522 = torch.prim.ListConstruct %int1_1682, %int0_1683 : (!torch.int, !torch.int) -> !torch.list<int>
    %1523 = torch.aten.permute %66, %1522 : !torch.vtensor<[256,12],f32>, !torch.list<int> -> !torch.vtensor<[12,256],f32>
    %int1_1684 = torch.constant.int 1
    %int0_1685 = torch.constant.int 0
    %1524 = torch.prim.ListConstruct %int1_1684, %int0_1685 : (!torch.int, !torch.int) -> !torch.list<int>
    %1525 = torch.aten.permute %67, %1524 : !torch.vtensor<[256,11],f32>, !torch.list<int> -> !torch.vtensor<[11,256],f32>
    %int5_1686 = torch.constant.int 5
    %1526 = torch.prims.convert_element_type %1523, %int5_1686 : !torch.vtensor<[12,256],f32>, !torch.int -> !torch.vtensor<[12,256],f16>
    %int1_1687 = torch.constant.int 1
    %int12_1688 = torch.constant.int 12
    %1527 = torch.prim.ListConstruct %int1_1687, %int12_1688 : (!torch.int, !torch.int) -> !torch.list<int>
    %1528 = torch.aten.view %1520, %1527 : !torch.vtensor<[1,1,12],f16>, !torch.list<int> -> !torch.vtensor<[1,12],f16>
    %1529 = torch.aten.mm %1528, %1526 : !torch.vtensor<[1,12],f16>, !torch.vtensor<[12,256],f16> -> !torch.vtensor<[1,256],f16>
    %int1_1689 = torch.constant.int 1
    %int1_1690 = torch.constant.int 1
    %int256_1691 = torch.constant.int 256
    %1530 = torch.prim.ListConstruct %int1_1689, %int1_1690, %int256_1691 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1531 = torch.aten.view %1529, %1530 : !torch.vtensor<[1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,1,256],f16>
    %int5_1692 = torch.constant.int 5
    %1532 = torch.prims.convert_element_type %1525, %int5_1692 : !torch.vtensor<[11,256],f32>, !torch.int -> !torch.vtensor<[11,256],f16>
    %int1_1693 = torch.constant.int 1
    %int11_1694 = torch.constant.int 11
    %1533 = torch.prim.ListConstruct %int1_1693, %int11_1694 : (!torch.int, !torch.int) -> !torch.list<int>
    %1534 = torch.aten.view %1521, %1533 : !torch.vtensor<[1,1,11],f16>, !torch.list<int> -> !torch.vtensor<[1,11],f16>
    %1535 = torch.aten.mm %1534, %1532 : !torch.vtensor<[1,11],f16>, !torch.vtensor<[11,256],f16> -> !torch.vtensor<[1,256],f16>
    %int1_1695 = torch.constant.int 1
    %int1_1696 = torch.constant.int 1
    %int256_1697 = torch.constant.int 256
    %1536 = torch.prim.ListConstruct %int1_1695, %int1_1696, %int256_1697 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1537 = torch.aten.view %1535, %1536 : !torch.vtensor<[1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,1,256],f16>
    %1538 = torch_c.to_builtin_tensor %1531 : !torch.vtensor<[1,1,256],f16> -> tensor<1x1x256xf16>
    %1539 = flow.tensor.barrier %1538 : tensor<1x1x256xf16> on #hal.device.promise<@__device_0>
    %1540 = torch_c.from_builtin_tensor %1539 : tensor<1x1x256xf16> -> !torch.vtensor<[1,1,256],f16>
    %1541 = torch_c.to_builtin_tensor %1537 : !torch.vtensor<[1,1,256],f16> -> tensor<1x1x256xf16>
    %1542 = flow.tensor.transfer %1541 : tensor<1x1x256xf16> to #hal.device.promise<@__device_0>
    %1543 = torch_c.from_builtin_tensor %1542 : tensor<1x1x256xf16> -> !torch.vtensor<[1,1,256],f16>
    %int1_1698 = torch.constant.int 1
    %1544 = torch.aten.add.Tensor %1540, %1543, %int1_1698 : !torch.vtensor<[1,1,256],f16>, !torch.vtensor<[1,1,256],f16>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %1545 = torch_c.to_builtin_tensor %1544 : !torch.vtensor<[1,1,256],f16> -> tensor<1x1x256xf16>
    %1546 = flow.tensor.barrier %1545 : tensor<1x1x256xf16> on #hal.device.promise<@__device_0>
    %1547 = torch_c.from_builtin_tensor %1546 : tensor<1x1x256xf16> -> !torch.vtensor<[1,1,256],f16>
    %1548 = torch_c.to_builtin_tensor %1544 : !torch.vtensor<[1,1,256],f16> -> tensor<1x1x256xf16>
    %1549 = flow.tensor.transfer %1548 : tensor<1x1x256xf16> to #hal.device.promise<@__device_1>
    %1550 = torch_c.from_builtin_tensor %1549 : tensor<1x1x256xf16> -> !torch.vtensor<[1,1,256],f16>
    %int1_1699 = torch.constant.int 1
    %1551 = torch.aten.add.Tensor %1464, %1547, %int1_1699 : !torch.vtensor<[1,1,256],f16>, !torch.vtensor<[1,1,256],f16>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %int1_1700 = torch.constant.int 1
    %1552 = torch.aten.add.Tensor %1465, %1550, %int1_1700 : !torch.vtensor<[1,1,256],f16>, !torch.vtensor<[1,1,256],f16>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %int6_1701 = torch.constant.int 6
    %1553 = torch.prims.convert_element_type %1551, %int6_1701 : !torch.vtensor<[1,1,256],f16>, !torch.int -> !torch.vtensor<[1,1,256],f32>
    %int6_1702 = torch.constant.int 6
    %1554 = torch.prims.convert_element_type %1552, %int6_1702 : !torch.vtensor<[1,1,256],f16>, !torch.int -> !torch.vtensor<[1,1,256],f32>
    %int2_1703 = torch.constant.int 2
    %1555 = torch.aten.pow.Tensor_Scalar %1553, %int2_1703 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f32>
    %int2_1704 = torch.constant.int 2
    %1556 = torch.aten.pow.Tensor_Scalar %1554, %int2_1704 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f32>
    %int-1_1705 = torch.constant.int -1
    %1557 = torch.prim.ListConstruct %int-1_1705 : (!torch.int) -> !torch.list<int>
    %true_1706 = torch.constant.bool true
    %none_1707 = torch.constant.none
    %1558 = torch.aten.mean.dim %1555, %1557, %true_1706, %none_1707 : !torch.vtensor<[1,1,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1],f32>
    %int-1_1708 = torch.constant.int -1
    %1559 = torch.prim.ListConstruct %int-1_1708 : (!torch.int) -> !torch.list<int>
    %true_1709 = torch.constant.bool true
    %none_1710 = torch.constant.none
    %1560 = torch.aten.mean.dim %1556, %1559, %true_1709, %none_1710 : !torch.vtensor<[1,1,256],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1],f32>
    %float1.000000e-02_1711 = torch.constant.float 1.000000e-02
    %int1_1712 = torch.constant.int 1
    %1561 = torch.aten.add.Scalar %1558, %float1.000000e-02_1711, %int1_1712 : !torch.vtensor<[1,1,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,1,1],f32>
    %float1.000000e-02_1713 = torch.constant.float 1.000000e-02
    %int1_1714 = torch.constant.int 1
    %1562 = torch.aten.add.Scalar %1560, %float1.000000e-02_1713, %int1_1714 : !torch.vtensor<[1,1,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,1,1],f32>
    %1563 = torch.aten.rsqrt %1561 : !torch.vtensor<[1,1,1],f32> -> !torch.vtensor<[1,1,1],f32>
    %1564 = torch.aten.rsqrt %1562 : !torch.vtensor<[1,1,1],f32> -> !torch.vtensor<[1,1,1],f32>
    %1565 = torch.aten.mul.Tensor %1553, %1563 : !torch.vtensor<[1,1,256],f32>, !torch.vtensor<[1,1,1],f32> -> !torch.vtensor<[1,1,256],f32>
    %1566 = torch.aten.mul.Tensor %1554, %1564 : !torch.vtensor<[1,1,256],f32>, !torch.vtensor<[1,1,1],f32> -> !torch.vtensor<[1,1,256],f32>
    %int5_1715 = torch.constant.int 5
    %1567 = torch.prims.convert_element_type %1565, %int5_1715 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %int5_1716 = torch.constant.int 5
    %1568 = torch.prims.convert_element_type %1566, %int5_1716 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %1569 = torch.aten.mul.Tensor %68, %1567 : !torch.vtensor<[1,256],f32>, !torch.vtensor<[1,1,256],f16> -> !torch.vtensor<[1,1,256],f32>
    %1570 = torch.aten.mul.Tensor %69, %1568 : !torch.vtensor<[1,256],f32>, !torch.vtensor<[1,1,256],f16> -> !torch.vtensor<[1,1,256],f32>
    %int5_1717 = torch.constant.int 5
    %1571 = torch.prims.convert_element_type %1569, %int5_1717 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %int5_1718 = torch.constant.int 5
    %1572 = torch.prims.convert_element_type %1570, %int5_1718 : !torch.vtensor<[1,1,256],f32>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %int1_1719 = torch.constant.int 1
    %int0_1720 = torch.constant.int 0
    %1573 = torch.prim.ListConstruct %int1_1719, %int0_1720 : (!torch.int, !torch.int) -> !torch.list<int>
    %1574 = torch.aten.permute %70, %1573 : !torch.vtensor<[256,128],f32>, !torch.list<int> -> !torch.vtensor<[128,256],f32>
    %int1_1721 = torch.constant.int 1
    %int0_1722 = torch.constant.int 0
    %1575 = torch.prim.ListConstruct %int1_1721, %int0_1722 : (!torch.int, !torch.int) -> !torch.list<int>
    %1576 = torch.aten.permute %71, %1575 : !torch.vtensor<[256,128],f32>, !torch.list<int> -> !torch.vtensor<[128,256],f32>
    %int0_1723 = torch.constant.int 0
    %int0_1724 = torch.constant.int 0
    %int9223372036854775807_1725 = torch.constant.int 9223372036854775807
    %int1_1726 = torch.constant.int 1
    %1577 = torch.aten.slice.Tensor %1571, %int0_1723, %int0_1724, %int9223372036854775807_1725, %int1_1726 : !torch.vtensor<[1,1,256],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %int1_1727 = torch.constant.int 1
    %int0_1728 = torch.constant.int 0
    %int9223372036854775807_1729 = torch.constant.int 9223372036854775807
    %int1_1730 = torch.constant.int 1
    %1578 = torch.aten.slice.Tensor %1577, %int1_1727, %int0_1728, %int9223372036854775807_1729, %int1_1730 : !torch.vtensor<[1,1,256],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %int2_1731 = torch.constant.int 2
    %int0_1732 = torch.constant.int 0
    %int128_1733 = torch.constant.int 128
    %int1_1734 = torch.constant.int 1
    %1579 = torch.aten.slice.Tensor %1578, %int2_1731, %int0_1732, %int128_1733, %int1_1734 : !torch.vtensor<[1,1,256],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,1,128],f16>
    %int0_1735 = torch.constant.int 0
    %int0_1736 = torch.constant.int 0
    %int9223372036854775807_1737 = torch.constant.int 9223372036854775807
    %int1_1738 = torch.constant.int 1
    %1580 = torch.aten.slice.Tensor %1572, %int0_1735, %int0_1736, %int9223372036854775807_1737, %int1_1738 : !torch.vtensor<[1,1,256],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %int1_1739 = torch.constant.int 1
    %int0_1740 = torch.constant.int 0
    %int9223372036854775807_1741 = torch.constant.int 9223372036854775807
    %int1_1742 = torch.constant.int 1
    %1581 = torch.aten.slice.Tensor %1580, %int1_1739, %int0_1740, %int9223372036854775807_1741, %int1_1742 : !torch.vtensor<[1,1,256],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,1,256],f16>
    %int2_1743 = torch.constant.int 2
    %int128_1744 = torch.constant.int 128
    %int256_1745 = torch.constant.int 256
    %int1_1746 = torch.constant.int 1
    %1582 = torch.aten.slice.Tensor %1581, %int2_1743, %int128_1744, %int256_1745, %int1_1746 : !torch.vtensor<[1,1,256],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[1,1,128],f16>
    %int5_1747 = torch.constant.int 5
    %1583 = torch.prims.convert_element_type %1574, %int5_1747 : !torch.vtensor<[128,256],f32>, !torch.int -> !torch.vtensor<[128,256],f16>
    %int1_1748 = torch.constant.int 1
    %int128_1749 = torch.constant.int 128
    %1584 = torch.prim.ListConstruct %int1_1748, %int128_1749 : (!torch.int, !torch.int) -> !torch.list<int>
    %1585 = torch.aten.view %1579, %1584 : !torch.vtensor<[1,1,128],f16>, !torch.list<int> -> !torch.vtensor<[1,128],f16>
    %1586 = torch.aten.mm %1585, %1583 : !torch.vtensor<[1,128],f16>, !torch.vtensor<[128,256],f16> -> !torch.vtensor<[1,256],f16>
    %int1_1750 = torch.constant.int 1
    %int1_1751 = torch.constant.int 1
    %int256_1752 = torch.constant.int 256
    %1587 = torch.prim.ListConstruct %int1_1750, %int1_1751, %int256_1752 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1588 = torch.aten.view %1586, %1587 : !torch.vtensor<[1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,1,256],f16>
    %int5_1753 = torch.constant.int 5
    %1589 = torch.prims.convert_element_type %1576, %int5_1753 : !torch.vtensor<[128,256],f32>, !torch.int -> !torch.vtensor<[128,256],f16>
    %int1_1754 = torch.constant.int 1
    %int128_1755 = torch.constant.int 128
    %1590 = torch.prim.ListConstruct %int1_1754, %int128_1755 : (!torch.int, !torch.int) -> !torch.list<int>
    %1591 = torch.aten.view %1582, %1590 : !torch.vtensor<[1,1,128],f16>, !torch.list<int> -> !torch.vtensor<[1,128],f16>
    %1592 = torch.aten.mm %1591, %1589 : !torch.vtensor<[1,128],f16>, !torch.vtensor<[128,256],f16> -> !torch.vtensor<[1,256],f16>
    %int1_1756 = torch.constant.int 1
    %int1_1757 = torch.constant.int 1
    %int256_1758 = torch.constant.int 256
    %1593 = torch.prim.ListConstruct %int1_1756, %int1_1757, %int256_1758 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %1594 = torch.aten.view %1592, %1593 : !torch.vtensor<[1,256],f16>, !torch.list<int> -> !torch.vtensor<[1,1,256],f16>
    %1595 = torch_c.to_builtin_tensor %1588 : !torch.vtensor<[1,1,256],f16> -> tensor<1x1x256xf16>
    %1596 = flow.tensor.barrier %1595 : tensor<1x1x256xf16> on #hal.device.promise<@__device_0>
    %1597 = torch_c.from_builtin_tensor %1596 : tensor<1x1x256xf16> -> !torch.vtensor<[1,1,256],f16>
    %1598 = torch_c.to_builtin_tensor %1594 : !torch.vtensor<[1,1,256],f16> -> tensor<1x1x256xf16>
    %1599 = flow.tensor.transfer %1598 : tensor<1x1x256xf16> to #hal.device.promise<@__device_0>
    %1600 = torch_c.from_builtin_tensor %1599 : tensor<1x1x256xf16> -> !torch.vtensor<[1,1,256],f16>
    %int1_1759 = torch.constant.int 1
    %1601 = torch.aten.add.Tensor %1597, %1600, %int1_1759 : !torch.vtensor<[1,1,256],f16>, !torch.vtensor<[1,1,256],f16>, !torch.int -> !torch.vtensor<[1,1,256],f16>
    return %1601 : !torch.vtensor<[1,1,256],f16>
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
  util.func private @sharktank_rotary_embedding_1_D_2_32_f32(%arg0: tensor<1x?x2x32xf32>, %arg1: tensor<1x?x32xf32>) -> tensor<1x?x2x32xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %dim = tensor.dim %arg0, %c0 : tensor<1x?x2x32xf32>
    %dim_0 = tensor.dim %arg0, %c1 : tensor<1x?x2x32xf32>
    %dim_1 = tensor.dim %arg0, %c2 : tensor<1x?x2x32xf32>
    %dim_2 = tensor.dim %arg0, %c3 : tensor<1x?x2x32xf32>
    %0 = tensor.empty(%dim, %dim_0, %dim_1, %dim_2) : tensor<?x?x?x?xf32>
    %cast = tensor.cast %0 : tensor<?x?x?x?xf32> to tensor<1x?x2x32xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg1 : tensor<1x?x32xf32>) outs(%cast : tensor<1x?x2x32xf32>) {
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
      %extracted = tensor.extract %arg0[%2, %3, %4, %10] : tensor<1x?x2x32xf32>
      %extracted_3 = tensor.extract %arg0[%2, %3, %4, %11] : tensor<1x?x2x32xf32>
      %12 = arith.cmpi eq, %7, %c0 : index
      %13 = arith.mulf %extracted, %8 : f32
      %14 = arith.mulf %extracted_3, %9 : f32
      %15 = arith.subf %13, %14 : f32
      %16 = arith.mulf %extracted_3, %8 : f32
      %17 = arith.mulf %extracted, %9 : f32
      %18 = arith.addf %16, %17 : f32
      %19 = arith.select %12, %15, %18 : f32
      linalg.yield %19 : f32
    } -> tensor<1x?x2x32xf32>
    util.return %1 : tensor<1x?x2x32xf32>
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
  util.func private @sharktank_rotary_embedding_1_1_2_32_f32(%arg0: tensor<1x1x2x32xf32>, %arg1: tensor<1x1x32xf32>) -> tensor<1x1x2x32xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %dim = tensor.dim %arg0, %c0 : tensor<1x1x2x32xf32>
    %dim_0 = tensor.dim %arg0, %c1 : tensor<1x1x2x32xf32>
    %dim_1 = tensor.dim %arg0, %c2 : tensor<1x1x2x32xf32>
    %dim_2 = tensor.dim %arg0, %c3 : tensor<1x1x2x32xf32>
    %0 = tensor.empty(%dim, %dim_0, %dim_1, %dim_2) : tensor<?x?x?x?xf32>
    %cast = tensor.cast %0 : tensor<?x?x?x?xf32> to tensor<1x1x2x32xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg1 : tensor<1x1x32xf32>) outs(%cast : tensor<1x1x2x32xf32>) {
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
      %extracted = tensor.extract %arg0[%2, %3, %4, %10] : tensor<1x1x2x32xf32>
      %extracted_3 = tensor.extract %arg0[%2, %3, %4, %11] : tensor<1x1x2x32xf32>
      %12 = arith.cmpi eq, %7, %c0 : index
      %13 = arith.mulf %extracted, %8 : f32
      %14 = arith.mulf %extracted_3, %9 : f32
      %15 = arith.subf %13, %14 : f32
      %16 = arith.mulf %extracted_3, %8 : f32
      %17 = arith.mulf %extracted, %9 : f32
      %18 = arith.addf %16, %17 : f32
      %19 = arith.select %12, %15, %18 : f32
      linalg.yield %19 : f32
    } -> tensor<1x1x2x32xf32>
    util.return %1 : tensor<1x1x2x32xf32>
  }
}
