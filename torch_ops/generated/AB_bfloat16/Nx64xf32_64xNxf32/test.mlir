module @module {
  func.func @main(%arg0: !torch.vtensor<[?,64],f32>, %arg1: !torch.vtensor<[64,?],f32>) -> !torch.vtensor<[?,?],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %arg0, %none, %none_0, %int6, %cpu, %int0 : !torch.vtensor<[?,64],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int15 = torch.constant.int 15
    %0 = torch.prims.convert_element_type %arg0, %int15 : !torch.vtensor<[?,64],f32>, !torch.int -> !torch.vtensor<[?,64],bf16>
    %none_1 = torch.constant.none
    %none_2 = torch.constant.none
    %int6_3 = torch.constant.int 6
    %cpu_4 = torch.constant.device "cpu"
    %int0_5 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %arg1, %none_1, %none_2, %int6_3, %cpu_4, %int0_5 : !torch.vtensor<[64,?],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int15_6 = torch.constant.int 15
    %1 = torch.prims.convert_element_type %arg1, %int15_6 : !torch.vtensor<[64,?],f32>, !torch.int -> !torch.vtensor<[64,?],bf16>
    %2 = torch.aten.matmul %0, %1 : !torch.vtensor<[?,64],bf16>, !torch.vtensor<[64,?],bf16> -> !torch.vtensor<[?,?],bf16>
    %none_7 = torch.constant.none
    %none_8 = torch.constant.none
    %int15_9 = torch.constant.int 15
    %cpu_10 = torch.constant.device "cpu"
    %int0_11 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %2, %none_7, %none_8, %int15_9, %cpu_10, %int0_11 : !torch.vtensor<[?,?],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int6_12 = torch.constant.int 6
    %3 = torch.prims.convert_element_type %2, %int6_12 : !torch.vtensor<[?,?],bf16>, !torch.int -> !torch.vtensor<[?,?],f32>
    return %3 : !torch.vtensor<[?,?],f32>
  }
}
