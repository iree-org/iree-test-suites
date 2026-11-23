module @module {
  func.func @main(%arg0: !torch.vtensor<[4,8,1024,64],f32>, %arg1: !torch.vtensor<[4,8,1024,64],f32>, %arg2: !torch.vtensor<[4,8,1024,64],f32>) -> !torch.vtensor<[4,8,1024,64],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense<1> : tensor<1x1x1xsi32>) : !torch.vtensor<[1,1,1],si32>
    %1 = torch.vtensor.literal(dense<0> : tensor<1x1x1x1xsi32>) : !torch.vtensor<[1,1,1,1],si32>
    %2 = torch.vtensor.literal(dense<1> : tensor<1x1x1xsi32>) : !torch.vtensor<[1,1,1],si32>
    %3 = torch.vtensor.literal(dense<0> : tensor<1x1x1x1xsi32>) : !torch.vtensor<[1,1,1,1],si32>
    %float1.250000e-01 = torch.constant.float 1.250000e-01
    %false = torch.constant.bool false
    %false_0 = torch.constant.bool false
    %output, %logsumexp, %max_scores = torch.hop_flex_attention %arg0, %arg1, %arg2, %float1.250000e-01, %false, %false_0 {mask_mod_fn = @sdpa_mask0, score_mod_fn = @sdpa_score0} : !torch.vtensor<[4,8,1024,64],f32>, !torch.vtensor<[4,8,1024,64],f32>, !torch.vtensor<[4,8,1024,64],f32>, !torch.float, !torch.bool, !torch.bool -> !torch.vtensor<[4,8,1024,64],f32>, !torch.vtensor<[4,8,1024],f32>, !torch.vtensor<[4,8,1024],f32>
    return %output : !torch.vtensor<[4,8,1024,64],f32>
  }
  func.func private @sdpa_score0(%arg0: !torch.vtensor<[],f32>, %arg1: !torch.vtensor<[],si32>, %arg2: !torch.vtensor<[],si32>, %arg3: !torch.vtensor<[],si32>, %arg4: !torch.vtensor<[],si32>) -> !torch.vtensor<[],f32> {
    %0 = torch.aten.tanh %arg0 : !torch.vtensor<[],f32> -> !torch.vtensor<[],f32>
    return %0 : !torch.vtensor<[],f32>
  }
  func.func private @sdpa_mask0(%arg0: !torch.vtensor<[],si32>, %arg1: !torch.vtensor<[],si32>, %arg2: !torch.vtensor<[],si32>, %arg3: !torch.vtensor<[],si32>) -> !torch.vtensor<[],i1> {
    %0 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %int11 = torch.constant.int 11
    %none = torch.constant.none
    %cpu = torch.constant.device "cpu"
    %false = torch.constant.bool false
    %1 = torch.aten.new_ones %arg0, %0, %int11, %none, %cpu, %false : !torch.vtensor<[],si32>, !torch.list<int>, !torch.int, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[],i1>
    return %1 : !torch.vtensor<[],i1>
  }
}
