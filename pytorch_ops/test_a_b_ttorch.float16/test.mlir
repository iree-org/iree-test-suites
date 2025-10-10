module @module {
  func.func @main(%arg0: !torch.vtensor<[64,64],f16>, %arg1: !torch.vtensor<[64,64],f16>) -> !torch.vtensor<[64,64],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %0 = torch.aten.transpose.int %arg1, %int0, %int1 : !torch.vtensor<[64,64],f16>, !torch.int, !torch.int -> !torch.vtensor<[64,64],f16>
    %1 = torch.aten.matmul %arg0, %0 : !torch.vtensor<[64,64],f16>, !torch.vtensor<[64,64],f16> -> !torch.vtensor<[64,64],f16>
    return %1 : !torch.vtensor<[64,64],f16>
  }
}
