module @module {
  func.func @main(%arg0: !torch.vtensor<[3240000,2],f32>, %arg1: !torch.vtensor<[2,2],f32>, %arg2: !torch.vtensor<[3240000,2],f32>, %arg3: !torch.vtensor<[3240000,2],f32>) -> !torch.vtensor<[3240000,2],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %false = torch.constant.bool false
    %true = torch.constant.bool true
    %false_0 = torch.constant.bool false
    %true_1 = torch.constant.bool true
    %none = torch.constant.none
    %none_2 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %arg0, %none, %none_2, %int6, %cpu, %int0 : !torch.vtensor<[3240000,2],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int15 = torch.constant.int 15
    %0 = torch.prims.convert_element_type %arg0, %int15 : !torch.vtensor<[3240000,2],f32>, !torch.int -> !torch.vtensor<[3240000,2],bf16>
    %none_3 = torch.constant.none
    %none_4 = torch.constant.none
    %int6_5 = torch.constant.int 6
    %cpu_6 = torch.constant.device "cpu"
    %int0_7 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %arg1, %none_3, %none_4, %int6_5, %cpu_6, %int0_7 : !torch.vtensor<[2,2],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int15_8 = torch.constant.int 15
    %1 = torch.prims.convert_element_type %arg1, %int15_8 : !torch.vtensor<[2,2],f32>, !torch.int -> !torch.vtensor<[2,2],bf16>
    %none_9 = torch.constant.none
    %none_10 = torch.constant.none
    %int6_11 = torch.constant.int 6
    %cpu_12 = torch.constant.device "cpu"
    %int0_13 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %arg2, %none_9, %none_10, %int6_11, %cpu_12, %int0_13 : !torch.vtensor<[3240000,2],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_14 = torch.constant.none
    %none_15 = torch.constant.none
    %int6_16 = torch.constant.int 6
    %cpu_17 = torch.constant.device "cpu"
    %int0_18 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %arg3, %none_14, %none_15, %int6_16, %cpu_17, %int0_18 : !torch.vtensor<[3240000,2],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int15_19 = torch.constant.int 15
    %2 = torch.prims.convert_element_type %arg3, %int15_19 : !torch.vtensor<[3240000,2],f32>, !torch.int -> !torch.vtensor<[3240000,2],bf16>
    %int0_20 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %3 = torch.aten.transpose.int %1, %int0_20, %int1 : !torch.vtensor<[2,2],bf16>, !torch.int, !torch.int -> !torch.vtensor<[2,2],bf16>
    %4 = torch.aten.mm %0, %3 : !torch.vtensor<[3240000,2],bf16>, !torch.vtensor<[2,2],bf16> -> !torch.vtensor<[3240000,2],bf16>
    %false_21 = torch.constant.bool false
    %5 = torch.aten.copy %2, %4, %false_21 : !torch.vtensor<[3240000,2],bf16>, !torch.vtensor<[3240000,2],bf16>, !torch.bool -> !torch.vtensor<[3240000,2],bf16>
    %none_22 = torch.constant.none
    %none_23 = torch.constant.none
    %int15_24 = torch.constant.int 15
    %cpu_25 = torch.constant.device "cpu"
    %int0_26 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_22, %none_23, %int15_24, %cpu_25, %int0_26 : !torch.vtensor<[3240000,2],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int6_27 = torch.constant.int 6
    %6 = torch.prims.convert_element_type %5, %int6_27 : !torch.vtensor<[3240000,2],bf16>, !torch.int -> !torch.vtensor<[3240000,2],f32>
    return %6 : !torch.vtensor<[3240000,2],f32>
  }
}
