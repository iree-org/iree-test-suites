module @module {
  func.func @main(%arg0: !torch.vtensor<[997,997],f32>, %arg1: !torch.vtensor<[997,997],f32>, %arg2: !torch.vtensor<[997,997],f32>, %arg3: !torch.tensor<[997,997],f32>) -> !torch.vtensor<[997,997],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %false = torch.constant.bool false
    %false_0 = torch.constant.bool false
    %true = torch.constant.bool true
    %0 = torch.copy.to_vtensor %arg3 : !torch.vtensor<[997,997],f32>
    %1 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[997,997],f32>, !torch.vtensor<[997,997],f32> -> !torch.vtensor<[997,997],f32>
    %false_1 = torch.constant.bool false
    %2 = torch.aten.copy %0, %1, %false_1 : !torch.vtensor<[997,997],f32>, !torch.vtensor<[997,997],f32>, !torch.bool -> !torch.vtensor<[997,997],f32>
    %int1 = torch.constant.int 1
    %3 = torch.aten.add.Tensor %2, %arg2, %int1 : !torch.vtensor<[997,997],f32>, !torch.vtensor<[997,997],f32>, !torch.int -> !torch.vtensor<[997,997],f32>
    torch.overwrite.tensor.contents %3 overwrites %arg3 : !torch.vtensor<[997,997],f32>, !torch.tensor<[997,997],f32>
    return %3 : !torch.vtensor<[997,997],f32>
  }
}
