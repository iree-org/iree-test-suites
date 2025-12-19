module @module {
  func.func @main(%arg0: !torch.vtensor<[64,64],f32>, %arg1: !torch.vtensor<[64,64],f32>, %arg2: !torch.vtensor<[64,64],f32>) -> !torch.vtensor<[64,64],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[64,64],f32>, !torch.vtensor<[64,64],f32> -> !torch.vtensor<[64,64],f32>
    %int1 = torch.constant.int 1
    %1 = torch.aten.add.Tensor %0, %arg2, %int1 : !torch.vtensor<[64,64],f32>, !torch.vtensor<[64,64],f32>, !torch.int -> !torch.vtensor<[64,64],f32>
    %2 = torch.aten.relu %1 : !torch.vtensor<[64,64],f32> -> !torch.vtensor<[64,64],f32>
    return %2 : !torch.vtensor<[64,64],f32>
  }
}
