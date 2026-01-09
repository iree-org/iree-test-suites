module @module {
  func.func @main(%arg0: !torch.vtensor<[64,64],f16>, %arg1: !torch.vtensor<[64,64],f16>, %arg2: !torch.vtensor<[64,64],f16>) -> !torch.vtensor<[64,64],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[64,64],f16>, !torch.vtensor<[64,64],f16> -> !torch.vtensor<[64,64],f16>
    %int1 = torch.constant.int 1
    %1 = torch.aten.add.Tensor %0, %arg2, %int1 : !torch.vtensor<[64,64],f16>, !torch.vtensor<[64,64],f16>, !torch.int -> !torch.vtensor<[64,64],f16>
    return %1 : !torch.vtensor<[64,64],f16>
  }
}
