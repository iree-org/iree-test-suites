module @module {
  func.func @float16(%arg0: !torch.vtensor<[64,64],f16>, %arg1: !torch.vtensor<[64,64],f16>) -> !torch.vtensor<[64,64],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[64,64],f16>, !torch.vtensor<[64,64],f16> -> !torch.vtensor<[64,64],f16>
    return %0 : !torch.vtensor<[64,64],f16>
  }
  func.func @float32(%arg0: !torch.vtensor<[64,64],f32>, %arg1: !torch.vtensor<[64,64],f32>) -> !torch.vtensor<[64,64],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[64,64],f32>, !torch.vtensor<[64,64],f32> -> !torch.vtensor<[64,64],f32>
    return %0 : !torch.vtensor<[64,64],f32>
  }
}
