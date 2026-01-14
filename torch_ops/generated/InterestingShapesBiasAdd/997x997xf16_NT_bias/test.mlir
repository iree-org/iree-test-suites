module @module {
  func.func @main(%arg0: !torch.vtensor<[997,997],f16>, %arg1: !torch.vtensor<[997,997],f16>, %arg2: !torch.vtensor<[997,997],f16>, %arg3: !torch.tensor<[997,997],f16>) -> !torch.vtensor<[997,997],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %false = torch.constant.bool false
    %true = torch.constant.bool true
    %true_0 = torch.constant.bool true
    %0 = torch.copy.to_vtensor %arg3 : !torch.vtensor<[997,997],f16>
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %1 = torch.aten.transpose.int %arg1, %int0, %int1 : !torch.vtensor<[997,997],f16>, !torch.int, !torch.int -> !torch.vtensor<[997,997],f16>
    %2 = torch.aten.mm %arg0, %1 : !torch.vtensor<[997,997],f16>, !torch.vtensor<[997,997],f16> -> !torch.vtensor<[997,997],f16>
    %false_1 = torch.constant.bool false
    %3 = torch.aten.copy %0, %2, %false_1 : !torch.vtensor<[997,997],f16>, !torch.vtensor<[997,997],f16>, !torch.bool -> !torch.vtensor<[997,997],f16>
    %int1_2 = torch.constant.int 1
    %4 = torch.aten.add.Tensor %3, %arg2, %int1_2 : !torch.vtensor<[997,997],f16>, !torch.vtensor<[997,997],f16>, !torch.int -> !torch.vtensor<[997,997],f16>
    torch.overwrite.tensor.contents %4 overwrites %arg3 : !torch.vtensor<[997,997],f16>, !torch.tensor<[997,997],f16>
    return %4 : !torch.vtensor<[997,997],f16>
  }
}
