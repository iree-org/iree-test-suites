module @module {
  func.func @main(%arg0: !torch.vtensor<[1536,64],si64>, %arg1: !torch.vtensor<[35,64],si64>, %arg2: !torch.vtensor<[1536,35],si64>, %arg3: !torch.tensor<[1536,35],si64>) -> !torch.vtensor<[1536,35],si64> attributes {torch.assume_strict_symbolic_shapes} {
    %false = torch.constant.bool false
    %true = torch.constant.bool true
    %false_0 = torch.constant.bool false
    %0 = torch.copy.to_vtensor %arg3 : !torch.vtensor<[1536,35],si64>
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %1 = torch.aten.transpose.int %arg1, %int0, %int1 : !torch.vtensor<[35,64],si64>, !torch.int, !torch.int -> !torch.vtensor<[64,35],si64>
    %2 = torch.aten.mm %arg0, %1 : !torch.vtensor<[1536,64],si64>, !torch.vtensor<[64,35],si64> -> !torch.vtensor<[1536,35],si64>
    %false_1 = torch.constant.bool false
    %3 = torch.aten.copy %0, %2, %false_1 : !torch.vtensor<[1536,35],si64>, !torch.vtensor<[1536,35],si64>, !torch.bool -> !torch.vtensor<[1536,35],si64>
    torch.overwrite.tensor.contents %3 overwrites %arg3 : !torch.vtensor<[1536,35],si64>, !torch.tensor<[1536,35],si64>
    return %3 : !torch.vtensor<[1536,35],si64>
  }
}
