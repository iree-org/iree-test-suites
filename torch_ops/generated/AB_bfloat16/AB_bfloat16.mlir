module @module {
  func.func @from_float32(%arg0: !torch.vtensor<[64,?],f32>, %arg1: !torch.vtensor<[?,64],f32>) -> !torch.vtensor<[64,64],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %int1 = torch.constant.int 1
    %0 = torch.aten.size.int %arg0, %int1 : !torch.vtensor<[64,?],f32>, !torch.int -> !torch.int
    %int0 = torch.constant.int 0
    %1 = torch.aten.size.int %arg1, %int0 : !torch.vtensor<[?,64],f32>, !torch.int -> !torch.int
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu = torch.constant.device "cpu"
    %int0_1 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %arg0, %none, %none_0, %int6, %cpu, %int0_1 : !torch.vtensor<[64,?],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %2 = torch.aten.eq.int %0, %1 : !torch.int, !torch.int -> !torch.bool
    %3 = torch.aten.Int.bool %2 : !torch.bool -> !torch.int
    %str = torch.constant.str "Runtime assertion failed for expression Eq(s56, s96) on node 'eq_3'"
    torch.aten._assert_scalar %3, %str : !torch.int, !torch.str
    %int15 = torch.constant.int 15
    %4 = torch.prims.convert_element_type %arg0, %int15 : !torch.vtensor<[64,?],f32>, !torch.int -> !torch.vtensor<[64,?],bf16>
    %5 = torch.aten.eq.int %0, %0 : !torch.int, !torch.int -> !torch.bool
    %6 = torch.aten.Int.bool %5 : !torch.bool -> !torch.int
    %str_2 = torch.constant.str "Runtime assertion failed for expression Eq(s56, s96) on node 'eq'"
    torch.aten._assert_scalar %6, %str_2 : !torch.int, !torch.str
    %none_3 = torch.constant.none
    %none_4 = torch.constant.none
    %int6_5 = torch.constant.int 6
    %cpu_6 = torch.constant.device "cpu"
    %int0_7 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %arg1, %none_3, %none_4, %int6_5, %cpu_6, %int0_7 : !torch.vtensor<[?,64],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int15_8 = torch.constant.int 15
    %7 = torch.prims.convert_element_type %arg1, %int15_8 : !torch.vtensor<[?,64],f32>, !torch.int -> !torch.vtensor<[?,64],bf16>
    %8 = torch.aten.matmul %4, %7 : !torch.vtensor<[64,?],bf16>, !torch.vtensor<[?,64],bf16> -> !torch.vtensor<[64,64],bf16>
    %none_9 = torch.constant.none
    %none_10 = torch.constant.none
    %int15_11 = torch.constant.int 15
    %cpu_12 = torch.constant.device "cpu"
    %int0_13 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %8, %none_9, %none_10, %int15_11, %cpu_12, %int0_13 : !torch.vtensor<[64,64],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int6_14 = torch.constant.int 6
    %9 = torch.prims.convert_element_type %8, %int6_14 : !torch.vtensor<[64,64],bf16>, !torch.int -> !torch.vtensor<[64,64],f32>
    return %9 : !torch.vtensor<[64,64],f32>
  }
}
