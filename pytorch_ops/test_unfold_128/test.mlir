module @module {
  func.func @main(%arg0: !torch.vtensor<[128],f32>) -> !torch.vtensor<[127,2],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %int0 = torch.constant.int 0
    %int2 = torch.constant.int 2
    %int1 = torch.constant.int 1
    %0 = torch.aten.unfold %arg0, %int0, %int2, %int1 : !torch.vtensor<[128],f32>, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[127,2],f32>
    return %0 : !torch.vtensor<[127,2],f32>
  }
}
