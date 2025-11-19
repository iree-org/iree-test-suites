module @module {
  func.func @from_float32(%arg0: !torch.vtensor<[64,?],f32>, %arg1: !torch.vtensor<[?,64],f32>) -> !torch.vtensor<[64,64],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %int15 = torch.constant.int 15
    %0 = torch.prims.convert_element_type %arg0, %int15 : !torch.vtensor<[64,?],f32>, !torch.int -> !torch.vtensor<[64,?],bf16>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int6 = torch.constant.int 6
    %none_1 = torch.constant.none
    %none_2 = torch.constant.none
    torch.aten._assert_tensor_metadata %arg0, %none, %none_0, %int6, %none_1, %none_2 : !torch.vtensor<[64,?],f32>, !torch.none, !torch.none, !torch.int, !torch.none, !torch.none
    %int15_3 = torch.constant.int 15
    %1 = torch.prims.convert_element_type %arg1, %int15_3 : !torch.vtensor<[?,64],f32>, !torch.int -> !torch.vtensor<[?,64],bf16>
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int6_6 = torch.constant.int 6
    %none_7 = torch.constant.none
    %none_8 = torch.constant.none
    torch.aten._assert_tensor_metadata %arg1, %none_4, %none_5, %int6_6, %none_7, %none_8 : !torch.vtensor<[?,64],f32>, !torch.none, !torch.none, !torch.int, !torch.none, !torch.none
    %2 = torch.aten.matmul %0, %1 : !torch.vtensor<[64,?],bf16>, !torch.vtensor<[?,64],bf16> -> !torch.vtensor<[64,64],bf16>
    %int6_9 = torch.constant.int 6
    %3 = torch.prims.convert_element_type %2, %int6_9 : !torch.vtensor<[64,64],bf16>, !torch.int -> !torch.vtensor<[64,64],f32>
    %none_10 = torch.constant.none
    %none_11 = torch.constant.none
    %int15_12 = torch.constant.int 15
    %none_13 = torch.constant.none
    %none_14 = torch.constant.none
    torch.aten._assert_tensor_metadata %2, %none_10, %none_11, %int15_12, %none_13, %none_14 : !torch.vtensor<[64,64],bf16>, !torch.none, !torch.none, !torch.int, !torch.none, !torch.none
    return %3 : !torch.vtensor<[64,64],f32>
  }
}
