module @module {
  func.func @main(%arg0: !torch.vtensor<[?,64],f16>, %arg1: !torch.vtensor<[64,?],f16>) -> !torch.vtensor<[?,?],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[?,64],f16>, !torch.vtensor<[64,?],f16> -> !torch.vtensor<[?,?],f16>
    return %0 : !torch.vtensor<[?,?],f16>
  }
}
