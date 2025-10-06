module @module {
  func.func @main(%arg0: !torch.vtensor<[?,64],f32>, %arg1: !torch.vtensor<[64,?],f32>) -> !torch.vtensor<[?,?],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[?,64],f32>, !torch.vtensor<[64,?],f32> -> !torch.vtensor<[?,?],f32>
    return %0 : !torch.vtensor<[?,?],f32>
  }
}
