module @module {
  func.func @main(%arg0: !torch.vtensor<[1152,999],f16>, %arg1: !torch.vtensor<[999,576],f16>, %arg2: !torch.vtensor<[1152,576],f16>, %arg3: !torch.tensor<[1152,576],f16>) -> !torch.vtensor<[1152,576],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %false = torch.constant.bool false
    %false_0 = torch.constant.bool false
    %false_1 = torch.constant.bool false
    %0 = torch.copy.to_vtensor %arg3 : !torch.vtensor<[1152,576],f16>
    %1 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[1152,999],f16>, !torch.vtensor<[999,576],f16> -> !torch.vtensor<[1152,576],f16>
    %false_2 = torch.constant.bool false
    %2 = torch.aten.copy %0, %1, %false_2 : !torch.vtensor<[1152,576],f16>, !torch.vtensor<[1152,576],f16>, !torch.bool -> !torch.vtensor<[1152,576],f16>
    torch.overwrite.tensor.contents %2 overwrites %arg3 : !torch.vtensor<[1152,576],f16>, !torch.tensor<[1152,576],f16>
    return %2 : !torch.vtensor<[1152,576],f16>
  }
}
