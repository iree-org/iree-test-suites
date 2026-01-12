module @module {
  func.func @main(%arg0: !torch.vtensor<[999,999],si64>, %arg1: !torch.vtensor<[999,999],si64>, %arg2: !torch.vtensor<[999,999],si64>, %arg3: !torch.tensor<[999,999],si64>) -> !torch.vtensor<[999,999],si64> attributes {torch.assume_strict_symbolic_shapes} {
    %false = torch.constant.bool false
    %false_0 = torch.constant.bool false
    %true = torch.constant.bool true
    %0 = torch.copy.to_vtensor %arg3 : !torch.vtensor<[999,999],si64>
    %1 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[999,999],si64>, !torch.vtensor<[999,999],si64> -> !torch.vtensor<[999,999],si64>
    %false_1 = torch.constant.bool false
    %2 = torch.aten.copy %0, %1, %false_1 : !torch.vtensor<[999,999],si64>, !torch.vtensor<[999,999],si64>, !torch.bool -> !torch.vtensor<[999,999],si64>
    %int1 = torch.constant.int 1
    %3 = torch.aten.add.Tensor %2, %arg2, %int1 : !torch.vtensor<[999,999],si64>, !torch.vtensor<[999,999],si64>, !torch.int -> !torch.vtensor<[999,999],si64>
    torch.overwrite.tensor.contents %3 overwrites %arg3 : !torch.vtensor<[999,999],si64>, !torch.tensor<[999,999],si64>
    return %3 : !torch.vtensor<[999,999],si64>
  }
}
