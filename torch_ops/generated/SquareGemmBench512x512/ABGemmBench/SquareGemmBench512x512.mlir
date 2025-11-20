module @module {
  func.func @entry_point(%arg0: !torch.vtensor<[512,512],f32>, %arg1: !torch.vtensor<[512,512],f32>) -> !torch.vtensor<[512,512],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[512,512],f32>, !torch.vtensor<[512,512],f32> -> !torch.vtensor<[512,512],f32>
    return %0 : !torch.vtensor<[512,512],f32>
  }
}
