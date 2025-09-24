module @module {
  func.func @main(%arg0: !torch.vtensor<[64,64],f32>, %arg1: !torch.vtensor<[64,64],f32>, %arg2: !torch.vtensor<[64,64],f32>) -> !torch.vtensor<[64,64],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %1 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %2 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %3 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %int0 = torch.constant.int 0
    %4 = torch.aten._trilinear %arg0, %arg1, %arg2, %0, %1, %2, %3, %int0 : !torch.vtensor<[64,64],f32>, !torch.vtensor<[64,64],f32>, !torch.vtensor<[64,64],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int -> !torch.vtensor<[64,64],f32>
    return %4 : !torch.vtensor<[64,64],f32>
  }
}
