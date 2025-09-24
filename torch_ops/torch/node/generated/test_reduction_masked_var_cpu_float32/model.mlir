module @module {
  func.func @test_reduction_masked_var_cpu_float32_0(%arg0: !torch.vtensor<[5,5,5],f32>) -> !torch.vtensor<[],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %int5 = torch.constant.int 5
    %int5_0 = torch.constant.int 5
    %int5_1 = torch.constant.int 5
    %0 = torch.prim.ListConstruct %int5, %int5_0, %int5_1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %int4 = torch.constant.int 4
    %none = torch.constant.none
    %cpu = torch.constant.device "cpu"
    %false = torch.constant.bool false
    %1 = torch.aten.ones %0, %int4, %none, %cpu, %false : !torch.list<int>, !torch.int, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[5,5,5],si64>
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %int2 = torch.constant.int 2
    %2 = torch.prim.ListConstruct %int0, %int1, %int2 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %int4_2 = torch.constant.int 4
    %3 = torch.aten.sum.dim_IntList %1, %2, %true, %int4_2 : !torch.vtensor<[5,5,5],si64>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[1,1,1],si64>
    %int0_3 = torch.constant.int 0
    %int1_4 = torch.constant.int 1
    %int2_5 = torch.constant.int 2
    %4 = torch.prim.ListConstruct %int0_3, %int1_4, %int2_5 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %true_6 = torch.constant.bool true
    %int6 = torch.constant.int 6
    %5 = torch.aten.sum.dim_IntList %arg0, %4, %true_6, %int6 : !torch.vtensor<[5,5,5],f32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[1,1,1],f32>
    %6 = torch.operator "torch.aten.divide.Tensor"(%5, %3) : (!torch.vtensor<[1,1,1],f32>, !torch.vtensor<[1,1,1],si64>) -> !torch.vtensor<[1,1,1],f32> 
    %int1_7 = torch.constant.int 1
    %7 = torch.operator "torch.aten.subtract.Tensor"(%arg0, %6, %int1_7) : (!torch.vtensor<[5,5,5],f32>, !torch.vtensor<[1,1,1],f32>, !torch.int) -> !torch.vtensor<[5,5,5],f32> 
    %8 = torch.aten.mul.Tensor %7, %7 : !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[5,5,5],f32> -> !torch.vtensor<[5,5,5],f32>
    %int0_8 = torch.constant.int 0
    %int1_9 = torch.constant.int 1
    %int2_10 = torch.constant.int 2
    %9 = torch.prim.ListConstruct %int0_8, %int1_9, %int2_10 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_11 = torch.constant.bool false
    %int6_12 = torch.constant.int 6
    %10 = torch.aten.sum.dim_IntList %8, %9, %false_11, %int6_12 : !torch.vtensor<[5,5,5],f32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[],f32>
    %11 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %12 = torch.aten.view %3, %11 : !torch.vtensor<[1,1,1],si64>, !torch.list<int> -> !torch.vtensor<[],si64>
    %none_13 = torch.constant.none
    %none_14 = torch.constant.none
    %int4_15 = torch.constant.int 4
    %cpu_16 = torch.constant.device "cpu"
    %int0_17 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %12, %none_13, %none_14, %int4_15, %cpu_16, %int0_17 : !torch.vtensor<[],si64>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int6_18 = torch.constant.int 6
    %13 = torch.prims.convert_element_type %12, %int6_18 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[],f32>
    %float1.000000e00 = torch.constant.float 1.000000e+00
    %14 = torch.prim.NumToTensor.Scalar %float1.000000e00 : !torch.float -> !torch.vtensor<[],f32>
    %int1_19 = torch.constant.int 1
    %15 = torch.operator "torch.aten.subtract.Tensor"(%13, %14, %int1_19) : (!torch.vtensor<[],f32>, !torch.vtensor<[],f32>, !torch.int) -> !torch.vtensor<[],f32> 
    %16 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %none_20 = torch.constant.none
    %none_21 = torch.constant.none
    %none_22 = torch.constant.none
    %false_23 = torch.constant.bool false
    %17 = torch.aten.new_zeros %15, %16, %none_20, %none_21, %none_22, %false_23 : !torch.vtensor<[],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[],f32>
    %18 = torch.aten.maximum %15, %17 : !torch.vtensor<[],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[],f32>
    %19 = torch.operator "torch.aten.divide.Tensor"(%10, %18) : (!torch.vtensor<[],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> 
    %none_24 = torch.constant.none
    %none_25 = torch.constant.none
    %int6_26 = torch.constant.int 6
    %cpu_27 = torch.constant.device "cpu"
    %int0_28 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %19, %none_24, %none_25, %int6_26, %cpu_27, %int0_28 : !torch.vtensor<[],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %19 : !torch.vtensor<[],f32>
  }
  func.func @test_reduction_masked_var_cpu_float32_1(%arg0: !torch.vtensor<[5,5,5],f32>) -> !torch.vtensor<[],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_5_5_5_torch.uint8> : tensor<5x5x5xui8>) : !torch.vtensor<[5,5,5],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[5,5,5],ui8>, !torch.int -> !torch.vtensor<[5,5,5],i1>
    %2 = torch.vtensor.literal(dense<0.000000e+00> : tensor<f32>) : !torch.vtensor<[],f32>
    %3 = torch.vtensor.literal(dense<0.000000e+00> : tensor<f32>) : !torch.vtensor<[],f32>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %true = torch.constant.bool true
    %none_3 = torch.constant.none
    %4 = torch.aten.sum.dim_IntList %1, %none_2, %true, %none_3 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1],si64>
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int11_6 = torch.constant.int 11
    %cpu_7 = torch.constant.device "cpu"
    %int0_8 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none_4, %none_5, %int11_6, %cpu_7, %int0_8 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_9 = torch.constant.none
    %5 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %6 = torch.aten.where.self %1, %arg0, %5 : !torch.vtensor<[5,5,5],i1>, !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[5,5,5],f32>
    %int0_10 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %int2 = torch.constant.int 2
    %7 = torch.prim.ListConstruct %int0_10, %int1, %int2 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %true_11 = torch.constant.bool true
    %int6 = torch.constant.int 6
    %8 = torch.aten.sum.dim_IntList %6, %7, %true_11, %int6 : !torch.vtensor<[5,5,5],f32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[1,1,1],f32>
    %9 = torch.operator "torch.aten.divide.Tensor"(%8, %4) : (!torch.vtensor<[1,1,1],f32>, !torch.vtensor<[1,1,1],si64>) -> !torch.vtensor<[1,1,1],f32> 
    %int1_12 = torch.constant.int 1
    %10 = torch.operator "torch.aten.subtract.Tensor"(%arg0, %9, %int1_12) : (!torch.vtensor<[5,5,5],f32>, !torch.vtensor<[1,1,1],f32>, !torch.int) -> !torch.vtensor<[5,5,5],f32> 
    %11 = torch.aten.mul.Tensor %10, %10 : !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[5,5,5],f32> -> !torch.vtensor<[5,5,5],f32>
    %none_13 = torch.constant.none
    %none_14 = torch.constant.none
    %int11_15 = torch.constant.int 11
    %cpu_16 = torch.constant.device "cpu"
    %int0_17 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none_13, %none_14, %int11_15, %cpu_16, %int0_17 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_18 = torch.constant.none
    %12 = torch.aten.clone %3, %none_18 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %13 = torch.aten.where.self %1, %11, %12 : !torch.vtensor<[5,5,5],i1>, !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[5,5,5],f32>
    %int0_19 = torch.constant.int 0
    %int1_20 = torch.constant.int 1
    %int2_21 = torch.constant.int 2
    %14 = torch.prim.ListConstruct %int0_19, %int1_20, %int2_21 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %int6_22 = torch.constant.int 6
    %15 = torch.aten.sum.dim_IntList %13, %14, %false, %int6_22 : !torch.vtensor<[5,5,5],f32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[],f32>
    %16 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %17 = torch.aten.view %4, %16 : !torch.vtensor<[1,1,1],si64>, !torch.list<int> -> !torch.vtensor<[],si64>
    %none_23 = torch.constant.none
    %none_24 = torch.constant.none
    %int4 = torch.constant.int 4
    %cpu_25 = torch.constant.device "cpu"
    %int0_26 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %17, %none_23, %none_24, %int4, %cpu_25, %int0_26 : !torch.vtensor<[],si64>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int6_27 = torch.constant.int 6
    %18 = torch.prims.convert_element_type %17, %int6_27 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[],f32>
    %float1.000000e00 = torch.constant.float 1.000000e+00
    %19 = torch.prim.NumToTensor.Scalar %float1.000000e00 : !torch.float -> !torch.vtensor<[],f32>
    %int1_28 = torch.constant.int 1
    %20 = torch.operator "torch.aten.subtract.Tensor"(%18, %19, %int1_28) : (!torch.vtensor<[],f32>, !torch.vtensor<[],f32>, !torch.int) -> !torch.vtensor<[],f32> 
    %21 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %none_29 = torch.constant.none
    %none_30 = torch.constant.none
    %none_31 = torch.constant.none
    %false_32 = torch.constant.bool false
    %22 = torch.aten.new_zeros %20, %21, %none_29, %none_30, %none_31, %false_32 : !torch.vtensor<[],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[],f32>
    %23 = torch.aten.maximum %20, %22 : !torch.vtensor<[],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[],f32>
    %24 = torch.operator "torch.aten.divide.Tensor"(%15, %23) : (!torch.vtensor<[],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> 
    %none_33 = torch.constant.none
    %none_34 = torch.constant.none
    %int6_35 = torch.constant.int 6
    %cpu_36 = torch.constant.device "cpu"
    %int0_37 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %24, %none_33, %none_34, %int6_35, %cpu_36, %int0_37 : !torch.vtensor<[],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %24 : !torch.vtensor<[],f32>
  }
  func.func @test_reduction_masked_var_cpu_float32_2(%arg0: !torch.vtensor<[5,5,5],f32>) -> !torch.vtensor<[],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_5_5_1_torch.uint8> : tensor<5x5x1xui8>) : !torch.vtensor<[5,5,1],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[5,5,1],ui8>, !torch.int -> !torch.vtensor<[5,5,1],i1>
    %2 = torch.vtensor.literal(dense<0.000000e+00> : tensor<f32>) : !torch.vtensor<[],f32>
    %3 = torch.vtensor.literal(dense<0.000000e+00> : tensor<f32>) : !torch.vtensor<[],f32>
    %none = torch.constant.none
    %4 = torch.aten.clone %1, %none : !torch.vtensor<[5,5,1],i1>, !torch.none -> !torch.vtensor<[5,5,1],i1>
    %int5 = torch.constant.int 5
    %int5_0 = torch.constant.int 5
    %int5_1 = torch.constant.int 5
    %5 = torch.prim.ListConstruct %int5, %int5_0, %int5_1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %6 = torch.aten.expand %4, %5, %false : !torch.vtensor<[5,5,1],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[5,5,5],i1>
    %none_2 = torch.constant.none
    %none_3 = torch.constant.none
    %int11_4 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_2, %none_3, %int11_4, %cpu, %int0 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_5 = torch.constant.none
    %none_6 = torch.constant.none
    %int11_7 = torch.constant.int 11
    %cpu_8 = torch.constant.device "cpu"
    %int0_9 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_5, %none_6, %int11_7, %cpu_8, %int0_9 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_10 = torch.constant.none
    %true = torch.constant.bool true
    %none_11 = torch.constant.none
    %7 = torch.aten.sum.dim_IntList %6, %none_10, %true, %none_11 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1],si64>
    %none_12 = torch.constant.none
    %none_13 = torch.constant.none
    %int11_14 = torch.constant.int 11
    %cpu_15 = torch.constant.device "cpu"
    %int0_16 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_12, %none_13, %int11_14, %cpu_15, %int0_16 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_17 = torch.constant.none
    %8 = torch.aten.clone %2, %none_17 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %9 = torch.aten.where.self %6, %arg0, %8 : !torch.vtensor<[5,5,5],i1>, !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[5,5,5],f32>
    %int0_18 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %int2 = torch.constant.int 2
    %10 = torch.prim.ListConstruct %int0_18, %int1, %int2 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %true_19 = torch.constant.bool true
    %int6 = torch.constant.int 6
    %11 = torch.aten.sum.dim_IntList %9, %10, %true_19, %int6 : !torch.vtensor<[5,5,5],f32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[1,1,1],f32>
    %12 = torch.operator "torch.aten.divide.Tensor"(%11, %7) : (!torch.vtensor<[1,1,1],f32>, !torch.vtensor<[1,1,1],si64>) -> !torch.vtensor<[1,1,1],f32> 
    %int1_20 = torch.constant.int 1
    %13 = torch.operator "torch.aten.subtract.Tensor"(%arg0, %12, %int1_20) : (!torch.vtensor<[5,5,5],f32>, !torch.vtensor<[1,1,1],f32>, !torch.int) -> !torch.vtensor<[5,5,5],f32> 
    %14 = torch.aten.mul.Tensor %13, %13 : !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[5,5,5],f32> -> !torch.vtensor<[5,5,5],f32>
    %none_21 = torch.constant.none
    %none_22 = torch.constant.none
    %int11_23 = torch.constant.int 11
    %cpu_24 = torch.constant.device "cpu"
    %int0_25 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_21, %none_22, %int11_23, %cpu_24, %int0_25 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_26 = torch.constant.none
    %15 = torch.aten.clone %3, %none_26 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %16 = torch.aten.where.self %6, %14, %15 : !torch.vtensor<[5,5,5],i1>, !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[5,5,5],f32>
    %int0_27 = torch.constant.int 0
    %int1_28 = torch.constant.int 1
    %int2_29 = torch.constant.int 2
    %17 = torch.prim.ListConstruct %int0_27, %int1_28, %int2_29 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_30 = torch.constant.bool false
    %int6_31 = torch.constant.int 6
    %18 = torch.aten.sum.dim_IntList %16, %17, %false_30, %int6_31 : !torch.vtensor<[5,5,5],f32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[],f32>
    %19 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %20 = torch.aten.view %7, %19 : !torch.vtensor<[1,1,1],si64>, !torch.list<int> -> !torch.vtensor<[],si64>
    %none_32 = torch.constant.none
    %none_33 = torch.constant.none
    %int4 = torch.constant.int 4
    %cpu_34 = torch.constant.device "cpu"
    %int0_35 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %20, %none_32, %none_33, %int4, %cpu_34, %int0_35 : !torch.vtensor<[],si64>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int6_36 = torch.constant.int 6
    %21 = torch.prims.convert_element_type %20, %int6_36 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[],f32>
    %float1.000000e00 = torch.constant.float 1.000000e+00
    %22 = torch.prim.NumToTensor.Scalar %float1.000000e00 : !torch.float -> !torch.vtensor<[],f32>
    %int1_37 = torch.constant.int 1
    %23 = torch.operator "torch.aten.subtract.Tensor"(%21, %22, %int1_37) : (!torch.vtensor<[],f32>, !torch.vtensor<[],f32>, !torch.int) -> !torch.vtensor<[],f32> 
    %24 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %none_38 = torch.constant.none
    %none_39 = torch.constant.none
    %none_40 = torch.constant.none
    %false_41 = torch.constant.bool false
    %25 = torch.aten.new_zeros %23, %24, %none_38, %none_39, %none_40, %false_41 : !torch.vtensor<[],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[],f32>
    %26 = torch.aten.maximum %23, %25 : !torch.vtensor<[],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[],f32>
    %27 = torch.operator "torch.aten.divide.Tensor"(%18, %26) : (!torch.vtensor<[],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> 
    %none_42 = torch.constant.none
    %none_43 = torch.constant.none
    %int6_44 = torch.constant.int 6
    %cpu_45 = torch.constant.device "cpu"
    %int0_46 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %27, %none_42, %none_43, %int6_44, %cpu_45, %int0_46 : !torch.vtensor<[],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %27 : !torch.vtensor<[],f32>
  }
  func.func @test_reduction_masked_var_cpu_float32_3(%arg0: !torch.vtensor<[5,5,5],f32>) -> !torch.vtensor<[],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_5_1_5_torch.uint8> : tensor<5x1x5xui8>) : !torch.vtensor<[5,1,5],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[5,1,5],ui8>, !torch.int -> !torch.vtensor<[5,1,5],i1>
    %2 = torch.vtensor.literal(dense<0.000000e+00> : tensor<f32>) : !torch.vtensor<[],f32>
    %3 = torch.vtensor.literal(dense<0.000000e+00> : tensor<f32>) : !torch.vtensor<[],f32>
    %none = torch.constant.none
    %4 = torch.aten.clone %1, %none : !torch.vtensor<[5,1,5],i1>, !torch.none -> !torch.vtensor<[5,1,5],i1>
    %int5 = torch.constant.int 5
    %int5_0 = torch.constant.int 5
    %int5_1 = torch.constant.int 5
    %5 = torch.prim.ListConstruct %int5, %int5_0, %int5_1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %6 = torch.aten.expand %4, %5, %false : !torch.vtensor<[5,1,5],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[5,5,5],i1>
    %none_2 = torch.constant.none
    %none_3 = torch.constant.none
    %int11_4 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_2, %none_3, %int11_4, %cpu, %int0 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_5 = torch.constant.none
    %none_6 = torch.constant.none
    %int11_7 = torch.constant.int 11
    %cpu_8 = torch.constant.device "cpu"
    %int0_9 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_5, %none_6, %int11_7, %cpu_8, %int0_9 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_10 = torch.constant.none
    %true = torch.constant.bool true
    %none_11 = torch.constant.none
    %7 = torch.aten.sum.dim_IntList %6, %none_10, %true, %none_11 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1],si64>
    %none_12 = torch.constant.none
    %none_13 = torch.constant.none
    %int11_14 = torch.constant.int 11
    %cpu_15 = torch.constant.device "cpu"
    %int0_16 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_12, %none_13, %int11_14, %cpu_15, %int0_16 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_17 = torch.constant.none
    %8 = torch.aten.clone %2, %none_17 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %9 = torch.aten.where.self %6, %arg0, %8 : !torch.vtensor<[5,5,5],i1>, !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[5,5,5],f32>
    %int0_18 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %int2 = torch.constant.int 2
    %10 = torch.prim.ListConstruct %int0_18, %int1, %int2 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %true_19 = torch.constant.bool true
    %int6 = torch.constant.int 6
    %11 = torch.aten.sum.dim_IntList %9, %10, %true_19, %int6 : !torch.vtensor<[5,5,5],f32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[1,1,1],f32>
    %12 = torch.operator "torch.aten.divide.Tensor"(%11, %7) : (!torch.vtensor<[1,1,1],f32>, !torch.vtensor<[1,1,1],si64>) -> !torch.vtensor<[1,1,1],f32> 
    %int1_20 = torch.constant.int 1
    %13 = torch.operator "torch.aten.subtract.Tensor"(%arg0, %12, %int1_20) : (!torch.vtensor<[5,5,5],f32>, !torch.vtensor<[1,1,1],f32>, !torch.int) -> !torch.vtensor<[5,5,5],f32> 
    %14 = torch.aten.mul.Tensor %13, %13 : !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[5,5,5],f32> -> !torch.vtensor<[5,5,5],f32>
    %none_21 = torch.constant.none
    %none_22 = torch.constant.none
    %int11_23 = torch.constant.int 11
    %cpu_24 = torch.constant.device "cpu"
    %int0_25 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_21, %none_22, %int11_23, %cpu_24, %int0_25 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_26 = torch.constant.none
    %15 = torch.aten.clone %3, %none_26 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %16 = torch.aten.where.self %6, %14, %15 : !torch.vtensor<[5,5,5],i1>, !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[5,5,5],f32>
    %int0_27 = torch.constant.int 0
    %int1_28 = torch.constant.int 1
    %int2_29 = torch.constant.int 2
    %17 = torch.prim.ListConstruct %int0_27, %int1_28, %int2_29 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_30 = torch.constant.bool false
    %int6_31 = torch.constant.int 6
    %18 = torch.aten.sum.dim_IntList %16, %17, %false_30, %int6_31 : !torch.vtensor<[5,5,5],f32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[],f32>
    %19 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %20 = torch.aten.view %7, %19 : !torch.vtensor<[1,1,1],si64>, !torch.list<int> -> !torch.vtensor<[],si64>
    %none_32 = torch.constant.none
    %none_33 = torch.constant.none
    %int4 = torch.constant.int 4
    %cpu_34 = torch.constant.device "cpu"
    %int0_35 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %20, %none_32, %none_33, %int4, %cpu_34, %int0_35 : !torch.vtensor<[],si64>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int6_36 = torch.constant.int 6
    %21 = torch.prims.convert_element_type %20, %int6_36 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[],f32>
    %float1.000000e00 = torch.constant.float 1.000000e+00
    %22 = torch.prim.NumToTensor.Scalar %float1.000000e00 : !torch.float -> !torch.vtensor<[],f32>
    %int1_37 = torch.constant.int 1
    %23 = torch.operator "torch.aten.subtract.Tensor"(%21, %22, %int1_37) : (!torch.vtensor<[],f32>, !torch.vtensor<[],f32>, !torch.int) -> !torch.vtensor<[],f32> 
    %24 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %none_38 = torch.constant.none
    %none_39 = torch.constant.none
    %none_40 = torch.constant.none
    %false_41 = torch.constant.bool false
    %25 = torch.aten.new_zeros %23, %24, %none_38, %none_39, %none_40, %false_41 : !torch.vtensor<[],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[],f32>
    %26 = torch.aten.maximum %23, %25 : !torch.vtensor<[],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[],f32>
    %27 = torch.operator "torch.aten.divide.Tensor"(%18, %26) : (!torch.vtensor<[],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> 
    %none_42 = torch.constant.none
    %none_43 = torch.constant.none
    %int6_44 = torch.constant.int 6
    %cpu_45 = torch.constant.device "cpu"
    %int0_46 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %27, %none_42, %none_43, %int6_44, %cpu_45, %int0_46 : !torch.vtensor<[],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %27 : !torch.vtensor<[],f32>
  }
  func.func @test_reduction_masked_var_cpu_float32_4(%arg0: !torch.vtensor<[5,5,5],f32>) -> !torch.vtensor<[],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_1_5_5_torch.uint8> : tensor<1x5x5xui8>) : !torch.vtensor<[1,5,5],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[1,5,5],ui8>, !torch.int -> !torch.vtensor<[1,5,5],i1>
    %2 = torch.vtensor.literal(dense<0.000000e+00> : tensor<f32>) : !torch.vtensor<[],f32>
    %3 = torch.vtensor.literal(dense<0.000000e+00> : tensor<f32>) : !torch.vtensor<[],f32>
    %none = torch.constant.none
    %4 = torch.aten.clone %1, %none : !torch.vtensor<[1,5,5],i1>, !torch.none -> !torch.vtensor<[1,5,5],i1>
    %int5 = torch.constant.int 5
    %int5_0 = torch.constant.int 5
    %int5_1 = torch.constant.int 5
    %5 = torch.prim.ListConstruct %int5, %int5_0, %int5_1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %6 = torch.aten.expand %4, %5, %false : !torch.vtensor<[1,5,5],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[5,5,5],i1>
    %none_2 = torch.constant.none
    %none_3 = torch.constant.none
    %int11_4 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_2, %none_3, %int11_4, %cpu, %int0 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_5 = torch.constant.none
    %none_6 = torch.constant.none
    %int11_7 = torch.constant.int 11
    %cpu_8 = torch.constant.device "cpu"
    %int0_9 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_5, %none_6, %int11_7, %cpu_8, %int0_9 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_10 = torch.constant.none
    %true = torch.constant.bool true
    %none_11 = torch.constant.none
    %7 = torch.aten.sum.dim_IntList %6, %none_10, %true, %none_11 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1],si64>
    %none_12 = torch.constant.none
    %none_13 = torch.constant.none
    %int11_14 = torch.constant.int 11
    %cpu_15 = torch.constant.device "cpu"
    %int0_16 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_12, %none_13, %int11_14, %cpu_15, %int0_16 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_17 = torch.constant.none
    %8 = torch.aten.clone %2, %none_17 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %9 = torch.aten.where.self %6, %arg0, %8 : !torch.vtensor<[5,5,5],i1>, !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[5,5,5],f32>
    %int0_18 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %int2 = torch.constant.int 2
    %10 = torch.prim.ListConstruct %int0_18, %int1, %int2 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %true_19 = torch.constant.bool true
    %int6 = torch.constant.int 6
    %11 = torch.aten.sum.dim_IntList %9, %10, %true_19, %int6 : !torch.vtensor<[5,5,5],f32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[1,1,1],f32>
    %12 = torch.operator "torch.aten.divide.Tensor"(%11, %7) : (!torch.vtensor<[1,1,1],f32>, !torch.vtensor<[1,1,1],si64>) -> !torch.vtensor<[1,1,1],f32> 
    %int1_20 = torch.constant.int 1
    %13 = torch.operator "torch.aten.subtract.Tensor"(%arg0, %12, %int1_20) : (!torch.vtensor<[5,5,5],f32>, !torch.vtensor<[1,1,1],f32>, !torch.int) -> !torch.vtensor<[5,5,5],f32> 
    %14 = torch.aten.mul.Tensor %13, %13 : !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[5,5,5],f32> -> !torch.vtensor<[5,5,5],f32>
    %none_21 = torch.constant.none
    %none_22 = torch.constant.none
    %int11_23 = torch.constant.int 11
    %cpu_24 = torch.constant.device "cpu"
    %int0_25 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_21, %none_22, %int11_23, %cpu_24, %int0_25 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_26 = torch.constant.none
    %15 = torch.aten.clone %3, %none_26 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %16 = torch.aten.where.self %6, %14, %15 : !torch.vtensor<[5,5,5],i1>, !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[5,5,5],f32>
    %int0_27 = torch.constant.int 0
    %int1_28 = torch.constant.int 1
    %int2_29 = torch.constant.int 2
    %17 = torch.prim.ListConstruct %int0_27, %int1_28, %int2_29 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_30 = torch.constant.bool false
    %int6_31 = torch.constant.int 6
    %18 = torch.aten.sum.dim_IntList %16, %17, %false_30, %int6_31 : !torch.vtensor<[5,5,5],f32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[],f32>
    %19 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %20 = torch.aten.view %7, %19 : !torch.vtensor<[1,1,1],si64>, !torch.list<int> -> !torch.vtensor<[],si64>
    %none_32 = torch.constant.none
    %none_33 = torch.constant.none
    %int4 = torch.constant.int 4
    %cpu_34 = torch.constant.device "cpu"
    %int0_35 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %20, %none_32, %none_33, %int4, %cpu_34, %int0_35 : !torch.vtensor<[],si64>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int6_36 = torch.constant.int 6
    %21 = torch.prims.convert_element_type %20, %int6_36 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[],f32>
    %float1.000000e00 = torch.constant.float 1.000000e+00
    %22 = torch.prim.NumToTensor.Scalar %float1.000000e00 : !torch.float -> !torch.vtensor<[],f32>
    %int1_37 = torch.constant.int 1
    %23 = torch.operator "torch.aten.subtract.Tensor"(%21, %22, %int1_37) : (!torch.vtensor<[],f32>, !torch.vtensor<[],f32>, !torch.int) -> !torch.vtensor<[],f32> 
    %24 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %none_38 = torch.constant.none
    %none_39 = torch.constant.none
    %none_40 = torch.constant.none
    %false_41 = torch.constant.bool false
    %25 = torch.aten.new_zeros %23, %24, %none_38, %none_39, %none_40, %false_41 : !torch.vtensor<[],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[],f32>
    %26 = torch.aten.maximum %23, %25 : !torch.vtensor<[],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[],f32>
    %27 = torch.operator "torch.aten.divide.Tensor"(%18, %26) : (!torch.vtensor<[],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> 
    %none_42 = torch.constant.none
    %none_43 = torch.constant.none
    %int6_44 = torch.constant.int 6
    %cpu_45 = torch.constant.device "cpu"
    %int0_46 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %27, %none_42, %none_43, %int6_44, %cpu_45, %int0_46 : !torch.vtensor<[],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %27 : !torch.vtensor<[],f32>
  }
  func.func @test_reduction_masked_var_cpu_float32_5(%arg0: !torch.vtensor<[5,5,5],f32>) -> !torch.vtensor<[],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_5_5_torch.uint8> : tensor<5x5xui8>) : !torch.vtensor<[5,5],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[5,5],ui8>, !torch.int -> !torch.vtensor<[5,5],i1>
    %2 = torch.vtensor.literal(dense<0.000000e+00> : tensor<f32>) : !torch.vtensor<[],f32>
    %3 = torch.vtensor.literal(dense<0.000000e+00> : tensor<f32>) : !torch.vtensor<[],f32>
    %none = torch.constant.none
    %4 = torch.aten.clone %1, %none : !torch.vtensor<[5,5],i1>, !torch.none -> !torch.vtensor<[5,5],i1>
    %int5 = torch.constant.int 5
    %int5_0 = torch.constant.int 5
    %int5_1 = torch.constant.int 5
    %5 = torch.prim.ListConstruct %int5, %int5_0, %int5_1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %6 = torch.aten.expand %4, %5, %false : !torch.vtensor<[5,5],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[5,5,5],i1>
    %none_2 = torch.constant.none
    %none_3 = torch.constant.none
    %int11_4 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_2, %none_3, %int11_4, %cpu, %int0 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_5 = torch.constant.none
    %none_6 = torch.constant.none
    %int11_7 = torch.constant.int 11
    %cpu_8 = torch.constant.device "cpu"
    %int0_9 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_5, %none_6, %int11_7, %cpu_8, %int0_9 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_10 = torch.constant.none
    %true = torch.constant.bool true
    %none_11 = torch.constant.none
    %7 = torch.aten.sum.dim_IntList %6, %none_10, %true, %none_11 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1],si64>
    %none_12 = torch.constant.none
    %none_13 = torch.constant.none
    %int11_14 = torch.constant.int 11
    %cpu_15 = torch.constant.device "cpu"
    %int0_16 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_12, %none_13, %int11_14, %cpu_15, %int0_16 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_17 = torch.constant.none
    %8 = torch.aten.clone %2, %none_17 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %9 = torch.aten.where.self %6, %arg0, %8 : !torch.vtensor<[5,5,5],i1>, !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[5,5,5],f32>
    %int0_18 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %int2 = torch.constant.int 2
    %10 = torch.prim.ListConstruct %int0_18, %int1, %int2 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %true_19 = torch.constant.bool true
    %int6 = torch.constant.int 6
    %11 = torch.aten.sum.dim_IntList %9, %10, %true_19, %int6 : !torch.vtensor<[5,5,5],f32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[1,1,1],f32>
    %12 = torch.operator "torch.aten.divide.Tensor"(%11, %7) : (!torch.vtensor<[1,1,1],f32>, !torch.vtensor<[1,1,1],si64>) -> !torch.vtensor<[1,1,1],f32> 
    %int1_20 = torch.constant.int 1
    %13 = torch.operator "torch.aten.subtract.Tensor"(%arg0, %12, %int1_20) : (!torch.vtensor<[5,5,5],f32>, !torch.vtensor<[1,1,1],f32>, !torch.int) -> !torch.vtensor<[5,5,5],f32> 
    %14 = torch.aten.mul.Tensor %13, %13 : !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[5,5,5],f32> -> !torch.vtensor<[5,5,5],f32>
    %none_21 = torch.constant.none
    %none_22 = torch.constant.none
    %int11_23 = torch.constant.int 11
    %cpu_24 = torch.constant.device "cpu"
    %int0_25 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_21, %none_22, %int11_23, %cpu_24, %int0_25 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_26 = torch.constant.none
    %15 = torch.aten.clone %3, %none_26 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %16 = torch.aten.where.self %6, %14, %15 : !torch.vtensor<[5,5,5],i1>, !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[5,5,5],f32>
    %int0_27 = torch.constant.int 0
    %int1_28 = torch.constant.int 1
    %int2_29 = torch.constant.int 2
    %17 = torch.prim.ListConstruct %int0_27, %int1_28, %int2_29 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_30 = torch.constant.bool false
    %int6_31 = torch.constant.int 6
    %18 = torch.aten.sum.dim_IntList %16, %17, %false_30, %int6_31 : !torch.vtensor<[5,5,5],f32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[],f32>
    %19 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %20 = torch.aten.view %7, %19 : !torch.vtensor<[1,1,1],si64>, !torch.list<int> -> !torch.vtensor<[],si64>
    %none_32 = torch.constant.none
    %none_33 = torch.constant.none
    %int4 = torch.constant.int 4
    %cpu_34 = torch.constant.device "cpu"
    %int0_35 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %20, %none_32, %none_33, %int4, %cpu_34, %int0_35 : !torch.vtensor<[],si64>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int6_36 = torch.constant.int 6
    %21 = torch.prims.convert_element_type %20, %int6_36 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[],f32>
    %float1.000000e00 = torch.constant.float 1.000000e+00
    %22 = torch.prim.NumToTensor.Scalar %float1.000000e00 : !torch.float -> !torch.vtensor<[],f32>
    %int1_37 = torch.constant.int 1
    %23 = torch.operator "torch.aten.subtract.Tensor"(%21, %22, %int1_37) : (!torch.vtensor<[],f32>, !torch.vtensor<[],f32>, !torch.int) -> !torch.vtensor<[],f32> 
    %24 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %none_38 = torch.constant.none
    %none_39 = torch.constant.none
    %none_40 = torch.constant.none
    %false_41 = torch.constant.bool false
    %25 = torch.aten.new_zeros %23, %24, %none_38, %none_39, %none_40, %false_41 : !torch.vtensor<[],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[],f32>
    %26 = torch.aten.maximum %23, %25 : !torch.vtensor<[],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[],f32>
    %27 = torch.operator "torch.aten.divide.Tensor"(%18, %26) : (!torch.vtensor<[],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> 
    %none_42 = torch.constant.none
    %none_43 = torch.constant.none
    %int6_44 = torch.constant.int 6
    %cpu_45 = torch.constant.device "cpu"
    %int0_46 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %27, %none_42, %none_43, %int6_44, %cpu_45, %int0_46 : !torch.vtensor<[],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %27 : !torch.vtensor<[],f32>
  }
  func.func @test_reduction_masked_var_cpu_float32_6(%arg0: !torch.vtensor<[5,5,5],f32>) -> !torch.vtensor<[],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_5_torch.uint8> : tensor<5xui8>) : !torch.vtensor<[5],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[5],ui8>, !torch.int -> !torch.vtensor<[5],i1>
    %2 = torch.vtensor.literal(dense<0.000000e+00> : tensor<f32>) : !torch.vtensor<[],f32>
    %3 = torch.vtensor.literal(dense<0.000000e+00> : tensor<f32>) : !torch.vtensor<[],f32>
    %none = torch.constant.none
    %4 = torch.aten.clone %1, %none : !torch.vtensor<[5],i1>, !torch.none -> !torch.vtensor<[5],i1>
    %int5 = torch.constant.int 5
    %int5_0 = torch.constant.int 5
    %int5_1 = torch.constant.int 5
    %5 = torch.prim.ListConstruct %int5, %int5_0, %int5_1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %6 = torch.aten.expand %4, %5, %false : !torch.vtensor<[5],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[5,5,5],i1>
    %none_2 = torch.constant.none
    %none_3 = torch.constant.none
    %int11_4 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_2, %none_3, %int11_4, %cpu, %int0 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_5 = torch.constant.none
    %none_6 = torch.constant.none
    %int11_7 = torch.constant.int 11
    %cpu_8 = torch.constant.device "cpu"
    %int0_9 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_5, %none_6, %int11_7, %cpu_8, %int0_9 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_10 = torch.constant.none
    %true = torch.constant.bool true
    %none_11 = torch.constant.none
    %7 = torch.aten.sum.dim_IntList %6, %none_10, %true, %none_11 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1],si64>
    %none_12 = torch.constant.none
    %none_13 = torch.constant.none
    %int11_14 = torch.constant.int 11
    %cpu_15 = torch.constant.device "cpu"
    %int0_16 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_12, %none_13, %int11_14, %cpu_15, %int0_16 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_17 = torch.constant.none
    %8 = torch.aten.clone %2, %none_17 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %9 = torch.aten.where.self %6, %arg0, %8 : !torch.vtensor<[5,5,5],i1>, !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[5,5,5],f32>
    %int0_18 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %int2 = torch.constant.int 2
    %10 = torch.prim.ListConstruct %int0_18, %int1, %int2 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %true_19 = torch.constant.bool true
    %int6 = torch.constant.int 6
    %11 = torch.aten.sum.dim_IntList %9, %10, %true_19, %int6 : !torch.vtensor<[5,5,5],f32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[1,1,1],f32>
    %12 = torch.operator "torch.aten.divide.Tensor"(%11, %7) : (!torch.vtensor<[1,1,1],f32>, !torch.vtensor<[1,1,1],si64>) -> !torch.vtensor<[1,1,1],f32> 
    %int1_20 = torch.constant.int 1
    %13 = torch.operator "torch.aten.subtract.Tensor"(%arg0, %12, %int1_20) : (!torch.vtensor<[5,5,5],f32>, !torch.vtensor<[1,1,1],f32>, !torch.int) -> !torch.vtensor<[5,5,5],f32> 
    %14 = torch.aten.mul.Tensor %13, %13 : !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[5,5,5],f32> -> !torch.vtensor<[5,5,5],f32>
    %none_21 = torch.constant.none
    %none_22 = torch.constant.none
    %int11_23 = torch.constant.int 11
    %cpu_24 = torch.constant.device "cpu"
    %int0_25 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_21, %none_22, %int11_23, %cpu_24, %int0_25 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_26 = torch.constant.none
    %15 = torch.aten.clone %3, %none_26 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %16 = torch.aten.where.self %6, %14, %15 : !torch.vtensor<[5,5,5],i1>, !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[5,5,5],f32>
    %int0_27 = torch.constant.int 0
    %int1_28 = torch.constant.int 1
    %int2_29 = torch.constant.int 2
    %17 = torch.prim.ListConstruct %int0_27, %int1_28, %int2_29 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_30 = torch.constant.bool false
    %int6_31 = torch.constant.int 6
    %18 = torch.aten.sum.dim_IntList %16, %17, %false_30, %int6_31 : !torch.vtensor<[5,5,5],f32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[],f32>
    %19 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %20 = torch.aten.view %7, %19 : !torch.vtensor<[1,1,1],si64>, !torch.list<int> -> !torch.vtensor<[],si64>
    %none_32 = torch.constant.none
    %none_33 = torch.constant.none
    %int4 = torch.constant.int 4
    %cpu_34 = torch.constant.device "cpu"
    %int0_35 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %20, %none_32, %none_33, %int4, %cpu_34, %int0_35 : !torch.vtensor<[],si64>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int6_36 = torch.constant.int 6
    %21 = torch.prims.convert_element_type %20, %int6_36 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[],f32>
    %float1.000000e00 = torch.constant.float 1.000000e+00
    %22 = torch.prim.NumToTensor.Scalar %float1.000000e00 : !torch.float -> !torch.vtensor<[],f32>
    %int1_37 = torch.constant.int 1
    %23 = torch.operator "torch.aten.subtract.Tensor"(%21, %22, %int1_37) : (!torch.vtensor<[],f32>, !torch.vtensor<[],f32>, !torch.int) -> !torch.vtensor<[],f32> 
    %24 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %none_38 = torch.constant.none
    %none_39 = torch.constant.none
    %none_40 = torch.constant.none
    %false_41 = torch.constant.bool false
    %25 = torch.aten.new_zeros %23, %24, %none_38, %none_39, %none_40, %false_41 : !torch.vtensor<[],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[],f32>
    %26 = torch.aten.maximum %23, %25 : !torch.vtensor<[],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[],f32>
    %27 = torch.operator "torch.aten.divide.Tensor"(%18, %26) : (!torch.vtensor<[],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> 
    %none_42 = torch.constant.none
    %none_43 = torch.constant.none
    %int6_44 = torch.constant.int 6
    %cpu_45 = torch.constant.device "cpu"
    %int0_46 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %27, %none_42, %none_43, %int6_44, %cpu_45, %int0_46 : !torch.vtensor<[],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %27 : !torch.vtensor<[],f32>
  }
  func.func @test_reduction_masked_var_cpu_float32_7(%arg0: !torch.vtensor<[5,5,5],f32>) -> !torch.vtensor<[5,5],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %int5 = torch.constant.int 5
    %int5_0 = torch.constant.int 5
    %int5_1 = torch.constant.int 5
    %0 = torch.prim.ListConstruct %int5, %int5_0, %int5_1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %int4 = torch.constant.int 4
    %none = torch.constant.none
    %cpu = torch.constant.device "cpu"
    %false = torch.constant.bool false
    %1 = torch.aten.ones %0, %int4, %none, %cpu, %false : !torch.list<int>, !torch.int, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[5,5,5],si64>
    %int1 = torch.constant.int 1
    %2 = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %int4_2 = torch.constant.int 4
    %3 = torch.aten.sum.dim_IntList %1, %2, %true, %int4_2 : !torch.vtensor<[5,5,5],si64>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[5,1,5],si64>
    %int1_3 = torch.constant.int 1
    %4 = torch.prim.ListConstruct %int1_3 : (!torch.int) -> !torch.list<int>
    %true_4 = torch.constant.bool true
    %int6 = torch.constant.int 6
    %5 = torch.aten.sum.dim_IntList %arg0, %4, %true_4, %int6 : !torch.vtensor<[5,5,5],f32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[5,1,5],f32>
    %6 = torch.operator "torch.aten.divide.Tensor"(%5, %3) : (!torch.vtensor<[5,1,5],f32>, !torch.vtensor<[5,1,5],si64>) -> !torch.vtensor<[5,1,5],f32> 
    %int1_5 = torch.constant.int 1
    %7 = torch.operator "torch.aten.subtract.Tensor"(%arg0, %6, %int1_5) : (!torch.vtensor<[5,5,5],f32>, !torch.vtensor<[5,1,5],f32>, !torch.int) -> !torch.vtensor<[5,5,5],f32> 
    %8 = torch.aten.mul.Tensor %7, %7 : !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[5,5,5],f32> -> !torch.vtensor<[5,5,5],f32>
    %int1_6 = torch.constant.int 1
    %9 = torch.prim.ListConstruct %int1_6 : (!torch.int) -> !torch.list<int>
    %false_7 = torch.constant.bool false
    %int6_8 = torch.constant.int 6
    %10 = torch.aten.sum.dim_IntList %8, %9, %false_7, %int6_8 : !torch.vtensor<[5,5,5],f32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[5,5],f32>
    %int5_9 = torch.constant.int 5
    %int5_10 = torch.constant.int 5
    %11 = torch.prim.ListConstruct %int5_9, %int5_10 : (!torch.int, !torch.int) -> !torch.list<int>
    %12 = torch.aten.view %3, %11 : !torch.vtensor<[5,1,5],si64>, !torch.list<int> -> !torch.vtensor<[5,5],si64>
    %none_11 = torch.constant.none
    %none_12 = torch.constant.none
    %int4_13 = torch.constant.int 4
    %cpu_14 = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %12, %none_11, %none_12, %int4_13, %cpu_14, %int0 : !torch.vtensor<[5,5],si64>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int6_15 = torch.constant.int 6
    %13 = torch.prims.convert_element_type %12, %int6_15 : !torch.vtensor<[5,5],si64>, !torch.int -> !torch.vtensor<[5,5],f32>
    %float1.000000e00 = torch.constant.float 1.000000e+00
    %14 = torch.prim.NumToTensor.Scalar %float1.000000e00 : !torch.float -> !torch.vtensor<[],f32>
    %int1_16 = torch.constant.int 1
    %15 = torch.operator "torch.aten.subtract.Tensor"(%13, %14, %int1_16) : (!torch.vtensor<[5,5],f32>, !torch.vtensor<[],f32>, !torch.int) -> !torch.vtensor<[5,5],f32> 
    %16 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %none_17 = torch.constant.none
    %none_18 = torch.constant.none
    %none_19 = torch.constant.none
    %false_20 = torch.constant.bool false
    %17 = torch.aten.new_zeros %15, %16, %none_17, %none_18, %none_19, %false_20 : !torch.vtensor<[5,5],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[],f32>
    %18 = torch.aten.maximum %15, %17 : !torch.vtensor<[5,5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[5,5],f32>
    %19 = torch.operator "torch.aten.divide.Tensor"(%10, %18) : (!torch.vtensor<[5,5],f32>, !torch.vtensor<[5,5],f32>) -> !torch.vtensor<[5,5],f32> 
    %none_21 = torch.constant.none
    %none_22 = torch.constant.none
    %int6_23 = torch.constant.int 6
    %cpu_24 = torch.constant.device "cpu"
    %int0_25 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %19, %none_21, %none_22, %int6_23, %cpu_24, %int0_25 : !torch.vtensor<[5,5],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %19 : !torch.vtensor<[5,5],f32>
  }
  func.func @test_reduction_masked_var_cpu_float32_8(%arg0: !torch.vtensor<[5,5,5],f32>) -> !torch.vtensor<[5,1,5],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %int5 = torch.constant.int 5
    %int5_0 = torch.constant.int 5
    %int5_1 = torch.constant.int 5
    %0 = torch.prim.ListConstruct %int5, %int5_0, %int5_1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %int4 = torch.constant.int 4
    %none = torch.constant.none
    %cpu = torch.constant.device "cpu"
    %false = torch.constant.bool false
    %1 = torch.aten.ones %0, %int4, %none, %cpu, %false : !torch.list<int>, !torch.int, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[5,5,5],si64>
    %int1 = torch.constant.int 1
    %2 = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %int4_2 = torch.constant.int 4
    %3 = torch.aten.sum.dim_IntList %1, %2, %true, %int4_2 : !torch.vtensor<[5,5,5],si64>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[5,1,5],si64>
    %int1_3 = torch.constant.int 1
    %4 = torch.prim.ListConstruct %int1_3 : (!torch.int) -> !torch.list<int>
    %true_4 = torch.constant.bool true
    %int6 = torch.constant.int 6
    %5 = torch.aten.sum.dim_IntList %arg0, %4, %true_4, %int6 : !torch.vtensor<[5,5,5],f32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[5,1,5],f32>
    %6 = torch.operator "torch.aten.divide.Tensor"(%5, %3) : (!torch.vtensor<[5,1,5],f32>, !torch.vtensor<[5,1,5],si64>) -> !torch.vtensor<[5,1,5],f32> 
    %int1_5 = torch.constant.int 1
    %7 = torch.operator "torch.aten.subtract.Tensor"(%arg0, %6, %int1_5) : (!torch.vtensor<[5,5,5],f32>, !torch.vtensor<[5,1,5],f32>, !torch.int) -> !torch.vtensor<[5,5,5],f32> 
    %8 = torch.aten.mul.Tensor %7, %7 : !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[5,5,5],f32> -> !torch.vtensor<[5,5,5],f32>
    %int1_6 = torch.constant.int 1
    %9 = torch.prim.ListConstruct %int1_6 : (!torch.int) -> !torch.list<int>
    %true_7 = torch.constant.bool true
    %int6_8 = torch.constant.int 6
    %10 = torch.aten.sum.dim_IntList %8, %9, %true_7, %int6_8 : !torch.vtensor<[5,5,5],f32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[5,1,5],f32>
    %none_9 = torch.constant.none
    %none_10 = torch.constant.none
    %int4_11 = torch.constant.int 4
    %cpu_12 = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %3, %none_9, %none_10, %int4_11, %cpu_12, %int0 : !torch.vtensor<[5,1,5],si64>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int6_13 = torch.constant.int 6
    %11 = torch.prims.convert_element_type %3, %int6_13 : !torch.vtensor<[5,1,5],si64>, !torch.int -> !torch.vtensor<[5,1,5],f32>
    %float1.000000e00 = torch.constant.float 1.000000e+00
    %12 = torch.prim.NumToTensor.Scalar %float1.000000e00 : !torch.float -> !torch.vtensor<[],f32>
    %int1_14 = torch.constant.int 1
    %13 = torch.operator "torch.aten.subtract.Tensor"(%11, %12, %int1_14) : (!torch.vtensor<[5,1,5],f32>, !torch.vtensor<[],f32>, !torch.int) -> !torch.vtensor<[5,1,5],f32> 
    %14 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %none_15 = torch.constant.none
    %none_16 = torch.constant.none
    %none_17 = torch.constant.none
    %false_18 = torch.constant.bool false
    %15 = torch.aten.new_zeros %13, %14, %none_15, %none_16, %none_17, %false_18 : !torch.vtensor<[5,1,5],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[],f32>
    %16 = torch.aten.maximum %13, %15 : !torch.vtensor<[5,1,5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[5,1,5],f32>
    %17 = torch.operator "torch.aten.divide.Tensor"(%10, %16) : (!torch.vtensor<[5,1,5],f32>, !torch.vtensor<[5,1,5],f32>) -> !torch.vtensor<[5,1,5],f32> 
    %none_19 = torch.constant.none
    %none_20 = torch.constant.none
    %int6_21 = torch.constant.int 6
    %cpu_22 = torch.constant.device "cpu"
    %int0_23 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %17, %none_19, %none_20, %int6_21, %cpu_22, %int0_23 : !torch.vtensor<[5,1,5],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %17 : !torch.vtensor<[5,1,5],f32>
  }
  func.func @test_reduction_masked_var_cpu_float32_9(%arg0: !torch.vtensor<[5],f32>) -> !torch.vtensor<[1],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %int5 = torch.constant.int 5
    %0 = torch.prim.ListConstruct %int5 : (!torch.int) -> !torch.list<int>
    %int4 = torch.constant.int 4
    %none = torch.constant.none
    %cpu = torch.constant.device "cpu"
    %false = torch.constant.bool false
    %1 = torch.aten.ones %0, %int4, %none, %cpu, %false : !torch.list<int>, !torch.int, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[5],si64>
    %int0 = torch.constant.int 0
    %2 = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %int4_0 = torch.constant.int 4
    %3 = torch.aten.sum.dim_IntList %1, %2, %true, %int4_0 : !torch.vtensor<[5],si64>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[1],si64>
    %int0_1 = torch.constant.int 0
    %4 = torch.prim.ListConstruct %int0_1 : (!torch.int) -> !torch.list<int>
    %true_2 = torch.constant.bool true
    %int6 = torch.constant.int 6
    %5 = torch.aten.sum.dim_IntList %arg0, %4, %true_2, %int6 : !torch.vtensor<[5],f32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[1],f32>
    %6 = torch.operator "torch.aten.divide.Tensor"(%5, %3) : (!torch.vtensor<[1],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[1],f32> 
    %int1 = torch.constant.int 1
    %7 = torch.operator "torch.aten.subtract.Tensor"(%arg0, %6, %int1) : (!torch.vtensor<[5],f32>, !torch.vtensor<[1],f32>, !torch.int) -> !torch.vtensor<[5],f32> 
    %8 = torch.aten.mul.Tensor %7, %7 : !torch.vtensor<[5],f32>, !torch.vtensor<[5],f32> -> !torch.vtensor<[5],f32>
    %int0_3 = torch.constant.int 0
    %9 = torch.prim.ListConstruct %int0_3 : (!torch.int) -> !torch.list<int>
    %true_4 = torch.constant.bool true
    %int6_5 = torch.constant.int 6
    %10 = torch.aten.sum.dim_IntList %8, %9, %true_4, %int6_5 : !torch.vtensor<[5],f32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[1],f32>
    %none_6 = torch.constant.none
    %none_7 = torch.constant.none
    %int4_8 = torch.constant.int 4
    %cpu_9 = torch.constant.device "cpu"
    %int0_10 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %3, %none_6, %none_7, %int4_8, %cpu_9, %int0_10 : !torch.vtensor<[1],si64>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int6_11 = torch.constant.int 6
    %11 = torch.prims.convert_element_type %3, %int6_11 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1],f32>
    %float1.000000e00 = torch.constant.float 1.000000e+00
    %12 = torch.prim.NumToTensor.Scalar %float1.000000e00 : !torch.float -> !torch.vtensor<[],f32>
    %int1_12 = torch.constant.int 1
    %13 = torch.operator "torch.aten.subtract.Tensor"(%11, %12, %int1_12) : (!torch.vtensor<[1],f32>, !torch.vtensor<[],f32>, !torch.int) -> !torch.vtensor<[1],f32> 
    %14 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %none_13 = torch.constant.none
    %none_14 = torch.constant.none
    %none_15 = torch.constant.none
    %false_16 = torch.constant.bool false
    %15 = torch.aten.new_zeros %13, %14, %none_13, %none_14, %none_15, %false_16 : !torch.vtensor<[1],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[],f32>
    %16 = torch.aten.maximum %13, %15 : !torch.vtensor<[1],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[1],f32>
    %17 = torch.operator "torch.aten.divide.Tensor"(%10, %16) : (!torch.vtensor<[1],f32>, !torch.vtensor<[1],f32>) -> !torch.vtensor<[1],f32> 
    %none_17 = torch.constant.none
    %none_18 = torch.constant.none
    %int6_19 = torch.constant.int 6
    %cpu_20 = torch.constant.device "cpu"
    %int0_21 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %17, %none_17, %none_18, %int6_19, %cpu_20, %int0_21 : !torch.vtensor<[1],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %17 : !torch.vtensor<[1],f32>
  }
  func.func @test_reduction_masked_var_cpu_float32_10(%arg0: !torch.vtensor<[5],f32>) -> !torch.vtensor<[1],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_5_torch.uint8_1> : tensor<5xui8>) : !torch.vtensor<[5],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[5],ui8>, !torch.int -> !torch.vtensor<[5],i1>
    %2 = torch.vtensor.literal(dense<0.000000e+00> : tensor<f32>) : !torch.vtensor<[],f32>
    %3 = torch.vtensor.literal(dense<0.000000e+00> : tensor<f32>) : !torch.vtensor<[],f32>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int0_2 = torch.constant.int 0
    %4 = torch.prim.ListConstruct %int0_2 : (!torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %none_3 = torch.constant.none
    %5 = torch.aten.sum.dim_IntList %1, %4, %true, %none_3 : !torch.vtensor<[5],i1>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1],si64>
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int11_6 = torch.constant.int 11
    %cpu_7 = torch.constant.device "cpu"
    %int0_8 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none_4, %none_5, %int11_6, %cpu_7, %int0_8 : !torch.vtensor<[5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_9 = torch.constant.none
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %7 = torch.aten.where.self %1, %arg0, %6 : !torch.vtensor<[5],i1>, !torch.vtensor<[5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[5],f32>
    %int0_10 = torch.constant.int 0
    %8 = torch.prim.ListConstruct %int0_10 : (!torch.int) -> !torch.list<int>
    %true_11 = torch.constant.bool true
    %int6 = torch.constant.int 6
    %9 = torch.aten.sum.dim_IntList %7, %8, %true_11, %int6 : !torch.vtensor<[5],f32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[1],f32>
    %10 = torch.operator "torch.aten.divide.Tensor"(%9, %5) : (!torch.vtensor<[1],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[1],f32> 
    %int1 = torch.constant.int 1
    %11 = torch.operator "torch.aten.subtract.Tensor"(%arg0, %10, %int1) : (!torch.vtensor<[5],f32>, !torch.vtensor<[1],f32>, !torch.int) -> !torch.vtensor<[5],f32> 
    %12 = torch.aten.mul.Tensor %11, %11 : !torch.vtensor<[5],f32>, !torch.vtensor<[5],f32> -> !torch.vtensor<[5],f32>
    %none_12 = torch.constant.none
    %none_13 = torch.constant.none
    %int11_14 = torch.constant.int 11
    %cpu_15 = torch.constant.device "cpu"
    %int0_16 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none_12, %none_13, %int11_14, %cpu_15, %int0_16 : !torch.vtensor<[5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_17 = torch.constant.none
    %13 = torch.aten.clone %3, %none_17 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %14 = torch.aten.where.self %1, %12, %13 : !torch.vtensor<[5],i1>, !torch.vtensor<[5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[5],f32>
    %int0_18 = torch.constant.int 0
    %15 = torch.prim.ListConstruct %int0_18 : (!torch.int) -> !torch.list<int>
    %true_19 = torch.constant.bool true
    %int6_20 = torch.constant.int 6
    %16 = torch.aten.sum.dim_IntList %14, %15, %true_19, %int6_20 : !torch.vtensor<[5],f32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[1],f32>
    %none_21 = torch.constant.none
    %none_22 = torch.constant.none
    %int4 = torch.constant.int 4
    %cpu_23 = torch.constant.device "cpu"
    %int0_24 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_21, %none_22, %int4, %cpu_23, %int0_24 : !torch.vtensor<[1],si64>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int6_25 = torch.constant.int 6
    %17 = torch.prims.convert_element_type %5, %int6_25 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1],f32>
    %float1.000000e00 = torch.constant.float 1.000000e+00
    %18 = torch.prim.NumToTensor.Scalar %float1.000000e00 : !torch.float -> !torch.vtensor<[],f32>
    %int1_26 = torch.constant.int 1
    %19 = torch.operator "torch.aten.subtract.Tensor"(%17, %18, %int1_26) : (!torch.vtensor<[1],f32>, !torch.vtensor<[],f32>, !torch.int) -> !torch.vtensor<[1],f32> 
    %20 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %none_27 = torch.constant.none
    %none_28 = torch.constant.none
    %none_29 = torch.constant.none
    %false = torch.constant.bool false
    %21 = torch.aten.new_zeros %19, %20, %none_27, %none_28, %none_29, %false : !torch.vtensor<[1],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[],f32>
    %22 = torch.aten.maximum %19, %21 : !torch.vtensor<[1],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[1],f32>
    %23 = torch.operator "torch.aten.divide.Tensor"(%16, %22) : (!torch.vtensor<[1],f32>, !torch.vtensor<[1],f32>) -> !torch.vtensor<[1],f32> 
    %none_30 = torch.constant.none
    %none_31 = torch.constant.none
    %int6_32 = torch.constant.int 6
    %cpu_33 = torch.constant.device "cpu"
    %int0_34 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %23, %none_30, %none_31, %int6_32, %cpu_33, %int0_34 : !torch.vtensor<[1],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %23 : !torch.vtensor<[1],f32>
  }
  func.func @test_reduction_masked_var_cpu_float32_11(%arg0: !torch.vtensor<[5],f32>) -> !torch.vtensor<[],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %int5 = torch.constant.int 5
    %0 = torch.prim.ListConstruct %int5 : (!torch.int) -> !torch.list<int>
    %int4 = torch.constant.int 4
    %none = torch.constant.none
    %cpu = torch.constant.device "cpu"
    %false = torch.constant.bool false
    %1 = torch.aten.ones %0, %int4, %none, %cpu, %false : !torch.list<int>, !torch.int, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[5],si64>
    %int0 = torch.constant.int 0
    %2 = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %int4_0 = torch.constant.int 4
    %3 = torch.aten.sum.dim_IntList %1, %2, %true, %int4_0 : !torch.vtensor<[5],si64>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[1],si64>
    %int0_1 = torch.constant.int 0
    %4 = torch.prim.ListConstruct %int0_1 : (!torch.int) -> !torch.list<int>
    %true_2 = torch.constant.bool true
    %int6 = torch.constant.int 6
    %5 = torch.aten.sum.dim_IntList %arg0, %4, %true_2, %int6 : !torch.vtensor<[5],f32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[1],f32>
    %6 = torch.operator "torch.aten.divide.Tensor"(%5, %3) : (!torch.vtensor<[1],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[1],f32> 
    %int1 = torch.constant.int 1
    %7 = torch.operator "torch.aten.subtract.Tensor"(%arg0, %6, %int1) : (!torch.vtensor<[5],f32>, !torch.vtensor<[1],f32>, !torch.int) -> !torch.vtensor<[5],f32> 
    %8 = torch.aten.mul.Tensor %7, %7 : !torch.vtensor<[5],f32>, !torch.vtensor<[5],f32> -> !torch.vtensor<[5],f32>
    %int0_3 = torch.constant.int 0
    %9 = torch.prim.ListConstruct %int0_3 : (!torch.int) -> !torch.list<int>
    %false_4 = torch.constant.bool false
    %int6_5 = torch.constant.int 6
    %10 = torch.aten.sum.dim_IntList %8, %9, %false_4, %int6_5 : !torch.vtensor<[5],f32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[],f32>
    %11 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %12 = torch.aten.view %3, %11 : !torch.vtensor<[1],si64>, !torch.list<int> -> !torch.vtensor<[],si64>
    %13 = torch.operator "torch.aten.divide.Tensor"(%10, %12) : (!torch.vtensor<[],f32>, !torch.vtensor<[],si64>) -> !torch.vtensor<[],f32> 
    %none_6 = torch.constant.none
    %none_7 = torch.constant.none
    %int6_8 = torch.constant.int 6
    %cpu_9 = torch.constant.device "cpu"
    %int0_10 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %13, %none_6, %none_7, %int6_8, %cpu_9, %int0_10 : !torch.vtensor<[],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %13 : !torch.vtensor<[],f32>
  }
  func.func @test_reduction_masked_var_cpu_float32_12(%arg0: !torch.vtensor<[5],f32>) -> !torch.vtensor<[],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_5_torch.uint8_2> : tensor<5xui8>) : !torch.vtensor<[5],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[5],ui8>, !torch.int -> !torch.vtensor<[5],i1>
    %2 = torch.vtensor.literal(dense<0.000000e+00> : tensor<f32>) : !torch.vtensor<[],f32>
    %3 = torch.vtensor.literal(dense<0.000000e+00> : tensor<f32>) : !torch.vtensor<[],f32>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int0_2 = torch.constant.int 0
    %4 = torch.prim.ListConstruct %int0_2 : (!torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %none_3 = torch.constant.none
    %5 = torch.aten.sum.dim_IntList %1, %4, %true, %none_3 : !torch.vtensor<[5],i1>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1],si64>
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int11_6 = torch.constant.int 11
    %cpu_7 = torch.constant.device "cpu"
    %int0_8 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none_4, %none_5, %int11_6, %cpu_7, %int0_8 : !torch.vtensor<[5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_9 = torch.constant.none
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %7 = torch.aten.where.self %1, %arg0, %6 : !torch.vtensor<[5],i1>, !torch.vtensor<[5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[5],f32>
    %int0_10 = torch.constant.int 0
    %8 = torch.prim.ListConstruct %int0_10 : (!torch.int) -> !torch.list<int>
    %true_11 = torch.constant.bool true
    %int6 = torch.constant.int 6
    %9 = torch.aten.sum.dim_IntList %7, %8, %true_11, %int6 : !torch.vtensor<[5],f32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[1],f32>
    %10 = torch.operator "torch.aten.divide.Tensor"(%9, %5) : (!torch.vtensor<[1],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[1],f32> 
    %int1 = torch.constant.int 1
    %11 = torch.operator "torch.aten.subtract.Tensor"(%arg0, %10, %int1) : (!torch.vtensor<[5],f32>, !torch.vtensor<[1],f32>, !torch.int) -> !torch.vtensor<[5],f32> 
    %12 = torch.aten.mul.Tensor %11, %11 : !torch.vtensor<[5],f32>, !torch.vtensor<[5],f32> -> !torch.vtensor<[5],f32>
    %none_12 = torch.constant.none
    %none_13 = torch.constant.none
    %int11_14 = torch.constant.int 11
    %cpu_15 = torch.constant.device "cpu"
    %int0_16 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none_12, %none_13, %int11_14, %cpu_15, %int0_16 : !torch.vtensor<[5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_17 = torch.constant.none
    %13 = torch.aten.clone %3, %none_17 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %14 = torch.aten.where.self %1, %12, %13 : !torch.vtensor<[5],i1>, !torch.vtensor<[5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[5],f32>
    %int0_18 = torch.constant.int 0
    %15 = torch.prim.ListConstruct %int0_18 : (!torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %int6_19 = torch.constant.int 6
    %16 = torch.aten.sum.dim_IntList %14, %15, %false, %int6_19 : !torch.vtensor<[5],f32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[],f32>
    %17 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %18 = torch.aten.view %5, %17 : !torch.vtensor<[1],si64>, !torch.list<int> -> !torch.vtensor<[],si64>
    %19 = torch.operator "torch.aten.divide.Tensor"(%16, %18) : (!torch.vtensor<[],f32>, !torch.vtensor<[],si64>) -> !torch.vtensor<[],f32> 
    %none_20 = torch.constant.none
    %none_21 = torch.constant.none
    %int6_22 = torch.constant.int 6
    %cpu_23 = torch.constant.device "cpu"
    %int0_24 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %19, %none_20, %none_21, %int6_22, %cpu_23, %int0_24 : !torch.vtensor<[],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %19 : !torch.vtensor<[],f32>
  }
  func.func @test_reduction_masked_var_cpu_float32_13(%arg0: !torch.vtensor<[5,5,5],f32>) -> !torch.vtensor<[5,5],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %int5 = torch.constant.int 5
    %int5_0 = torch.constant.int 5
    %int5_1 = torch.constant.int 5
    %0 = torch.prim.ListConstruct %int5, %int5_0, %int5_1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %int4 = torch.constant.int 4
    %none = torch.constant.none
    %cpu = torch.constant.device "cpu"
    %false = torch.constant.bool false
    %1 = torch.aten.ones %0, %int4, %none, %cpu, %false : !torch.list<int>, !torch.int, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[5,5,5],si64>
    %int1 = torch.constant.int 1
    %2 = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %int4_2 = torch.constant.int 4
    %3 = torch.aten.sum.dim_IntList %1, %2, %true, %int4_2 : !torch.vtensor<[5,5,5],si64>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[5,1,5],si64>
    %int1_3 = torch.constant.int 1
    %4 = torch.prim.ListConstruct %int1_3 : (!torch.int) -> !torch.list<int>
    %true_4 = torch.constant.bool true
    %int6 = torch.constant.int 6
    %5 = torch.aten.sum.dim_IntList %arg0, %4, %true_4, %int6 : !torch.vtensor<[5,5,5],f32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[5,1,5],f32>
    %6 = torch.operator "torch.aten.divide.Tensor"(%5, %3) : (!torch.vtensor<[5,1,5],f32>, !torch.vtensor<[5,1,5],si64>) -> !torch.vtensor<[5,1,5],f32> 
    %int1_5 = torch.constant.int 1
    %7 = torch.operator "torch.aten.subtract.Tensor"(%arg0, %6, %int1_5) : (!torch.vtensor<[5,5,5],f32>, !torch.vtensor<[5,1,5],f32>, !torch.int) -> !torch.vtensor<[5,5,5],f32> 
    %8 = torch.aten.mul.Tensor %7, %7 : !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[5,5,5],f32> -> !torch.vtensor<[5,5,5],f32>
    %int1_6 = torch.constant.int 1
    %9 = torch.prim.ListConstruct %int1_6 : (!torch.int) -> !torch.list<int>
    %false_7 = torch.constant.bool false
    %int6_8 = torch.constant.int 6
    %10 = torch.aten.sum.dim_IntList %8, %9, %false_7, %int6_8 : !torch.vtensor<[5,5,5],f32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[5,5],f32>
    %int5_9 = torch.constant.int 5
    %int5_10 = torch.constant.int 5
    %11 = torch.prim.ListConstruct %int5_9, %int5_10 : (!torch.int, !torch.int) -> !torch.list<int>
    %12 = torch.aten.view %3, %11 : !torch.vtensor<[5,1,5],si64>, !torch.list<int> -> !torch.vtensor<[5,5],si64>
    %none_11 = torch.constant.none
    %none_12 = torch.constant.none
    %int4_13 = torch.constant.int 4
    %cpu_14 = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %12, %none_11, %none_12, %int4_13, %cpu_14, %int0 : !torch.vtensor<[5,5],si64>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int6_15 = torch.constant.int 6
    %13 = torch.prims.convert_element_type %12, %int6_15 : !torch.vtensor<[5,5],si64>, !torch.int -> !torch.vtensor<[5,5],f32>
    %float1.300000e00 = torch.constant.float 1.300000e+00
    %14 = torch.prim.NumToTensor.Scalar %float1.300000e00 : !torch.float -> !torch.vtensor<[],f32>
    %int1_16 = torch.constant.int 1
    %15 = torch.operator "torch.aten.subtract.Tensor"(%13, %14, %int1_16) : (!torch.vtensor<[5,5],f32>, !torch.vtensor<[],f32>, !torch.int) -> !torch.vtensor<[5,5],f32> 
    %16 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %none_17 = torch.constant.none
    %none_18 = torch.constant.none
    %none_19 = torch.constant.none
    %false_20 = torch.constant.bool false
    %17 = torch.aten.new_zeros %15, %16, %none_17, %none_18, %none_19, %false_20 : !torch.vtensor<[5,5],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[],f32>
    %18 = torch.aten.maximum %15, %17 : !torch.vtensor<[5,5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[5,5],f32>
    %19 = torch.operator "torch.aten.divide.Tensor"(%10, %18) : (!torch.vtensor<[5,5],f32>, !torch.vtensor<[5,5],f32>) -> !torch.vtensor<[5,5],f32> 
    %none_21 = torch.constant.none
    %none_22 = torch.constant.none
    %int6_23 = torch.constant.int 6
    %cpu_24 = torch.constant.device "cpu"
    %int0_25 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %19, %none_21, %none_22, %int6_23, %cpu_24, %int0_25 : !torch.vtensor<[5,5],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %19 : !torch.vtensor<[5,5],f32>
  }
  func.func @test_reduction_masked_var_cpu_float32_14(%arg0: !torch.vtensor<[5,5,5],f32>) -> !torch.vtensor<[5,5],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %int5 = torch.constant.int 5
    %int5_0 = torch.constant.int 5
    %int5_1 = torch.constant.int 5
    %0 = torch.prim.ListConstruct %int5, %int5_0, %int5_1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %int4 = torch.constant.int 4
    %none = torch.constant.none
    %cpu = torch.constant.device "cpu"
    %false = torch.constant.bool false
    %1 = torch.aten.ones %0, %int4, %none, %cpu, %false : !torch.list<int>, !torch.int, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[5,5,5],si64>
    %int1 = torch.constant.int 1
    %2 = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %int4_2 = torch.constant.int 4
    %3 = torch.aten.sum.dim_IntList %1, %2, %true, %int4_2 : !torch.vtensor<[5,5,5],si64>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[5,1,5],si64>
    %int1_3 = torch.constant.int 1
    %4 = torch.prim.ListConstruct %int1_3 : (!torch.int) -> !torch.list<int>
    %true_4 = torch.constant.bool true
    %int6 = torch.constant.int 6
    %5 = torch.aten.sum.dim_IntList %arg0, %4, %true_4, %int6 : !torch.vtensor<[5,5,5],f32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[5,1,5],f32>
    %6 = torch.operator "torch.aten.divide.Tensor"(%5, %3) : (!torch.vtensor<[5,1,5],f32>, !torch.vtensor<[5,1,5],si64>) -> !torch.vtensor<[5,1,5],f32> 
    %int1_5 = torch.constant.int 1
    %7 = torch.operator "torch.aten.subtract.Tensor"(%arg0, %6, %int1_5) : (!torch.vtensor<[5,5,5],f32>, !torch.vtensor<[5,1,5],f32>, !torch.int) -> !torch.vtensor<[5,5,5],f32> 
    %8 = torch.aten.mul.Tensor %7, %7 : !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[5,5,5],f32> -> !torch.vtensor<[5,5,5],f32>
    %int1_6 = torch.constant.int 1
    %9 = torch.prim.ListConstruct %int1_6 : (!torch.int) -> !torch.list<int>
    %false_7 = torch.constant.bool false
    %int6_8 = torch.constant.int 6
    %10 = torch.aten.sum.dim_IntList %8, %9, %false_7, %int6_8 : !torch.vtensor<[5,5,5],f32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[5,5],f32>
    %int5_9 = torch.constant.int 5
    %int5_10 = torch.constant.int 5
    %11 = torch.prim.ListConstruct %int5_9, %int5_10 : (!torch.int, !torch.int) -> !torch.list<int>
    %12 = torch.aten.view %3, %11 : !torch.vtensor<[5,1,5],si64>, !torch.list<int> -> !torch.vtensor<[5,5],si64>
    %none_11 = torch.constant.none
    %none_12 = torch.constant.none
    %int4_13 = torch.constant.int 4
    %cpu_14 = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %12, %none_11, %none_12, %int4_13, %cpu_14, %int0 : !torch.vtensor<[5,5],si64>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int6_15 = torch.constant.int 6
    %13 = torch.prims.convert_element_type %12, %int6_15 : !torch.vtensor<[5,5],si64>, !torch.int -> !torch.vtensor<[5,5],f32>
    %float2.000000e00 = torch.constant.float 2.000000e+00
    %14 = torch.prim.NumToTensor.Scalar %float2.000000e00 : !torch.float -> !torch.vtensor<[],f32>
    %int1_16 = torch.constant.int 1
    %15 = torch.operator "torch.aten.subtract.Tensor"(%13, %14, %int1_16) : (!torch.vtensor<[5,5],f32>, !torch.vtensor<[],f32>, !torch.int) -> !torch.vtensor<[5,5],f32> 
    %16 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %none_17 = torch.constant.none
    %none_18 = torch.constant.none
    %none_19 = torch.constant.none
    %false_20 = torch.constant.bool false
    %17 = torch.aten.new_zeros %15, %16, %none_17, %none_18, %none_19, %false_20 : !torch.vtensor<[5,5],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[],f32>
    %18 = torch.aten.maximum %15, %17 : !torch.vtensor<[5,5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[5,5],f32>
    %19 = torch.operator "torch.aten.divide.Tensor"(%10, %18) : (!torch.vtensor<[5,5],f32>, !torch.vtensor<[5,5],f32>) -> !torch.vtensor<[5,5],f32> 
    %none_21 = torch.constant.none
    %none_22 = torch.constant.none
    %int6_23 = torch.constant.int 6
    %cpu_24 = torch.constant.device "cpu"
    %int0_25 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %19, %none_21, %none_22, %int6_23, %cpu_24, %int0_25 : !torch.vtensor<[5,5],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %19 : !torch.vtensor<[5,5],f32>
  }
  func.func @test_reduction_masked_var_cpu_float32_15(%arg0: !torch.vtensor<[5,5,5],f32>) -> !torch.vtensor<[1,1,1],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %int5 = torch.constant.int 5
    %int5_0 = torch.constant.int 5
    %int5_1 = torch.constant.int 5
    %0 = torch.prim.ListConstruct %int5, %int5_0, %int5_1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %int4 = torch.constant.int 4
    %none = torch.constant.none
    %cpu = torch.constant.device "cpu"
    %false = torch.constant.bool false
    %1 = torch.aten.ones %0, %int4, %none, %cpu, %false : !torch.list<int>, !torch.int, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[5,5,5],si64>
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %int2 = torch.constant.int 2
    %2 = torch.prim.ListConstruct %int0, %int1, %int2 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %int4_2 = torch.constant.int 4
    %3 = torch.aten.sum.dim_IntList %1, %2, %true, %int4_2 : !torch.vtensor<[5,5,5],si64>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[1,1,1],si64>
    %int0_3 = torch.constant.int 0
    %int1_4 = torch.constant.int 1
    %int2_5 = torch.constant.int 2
    %4 = torch.prim.ListConstruct %int0_3, %int1_4, %int2_5 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %true_6 = torch.constant.bool true
    %int6 = torch.constant.int 6
    %5 = torch.aten.sum.dim_IntList %arg0, %4, %true_6, %int6 : !torch.vtensor<[5,5,5],f32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[1,1,1],f32>
    %6 = torch.operator "torch.aten.divide.Tensor"(%5, %3) : (!torch.vtensor<[1,1,1],f32>, !torch.vtensor<[1,1,1],si64>) -> !torch.vtensor<[1,1,1],f32> 
    %int1_7 = torch.constant.int 1
    %7 = torch.operator "torch.aten.subtract.Tensor"(%arg0, %6, %int1_7) : (!torch.vtensor<[5,5,5],f32>, !torch.vtensor<[1,1,1],f32>, !torch.int) -> !torch.vtensor<[5,5,5],f32> 
    %8 = torch.aten.mul.Tensor %7, %7 : !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[5,5,5],f32> -> !torch.vtensor<[5,5,5],f32>
    %int0_8 = torch.constant.int 0
    %int1_9 = torch.constant.int 1
    %int2_10 = torch.constant.int 2
    %9 = torch.prim.ListConstruct %int0_8, %int1_9, %int2_10 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %true_11 = torch.constant.bool true
    %int6_12 = torch.constant.int 6
    %10 = torch.aten.sum.dim_IntList %8, %9, %true_11, %int6_12 : !torch.vtensor<[5,5,5],f32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[1,1,1],f32>
    %11 = torch.operator "torch.aten.divide.Tensor"(%10, %3) : (!torch.vtensor<[1,1,1],f32>, !torch.vtensor<[1,1,1],si64>) -> !torch.vtensor<[1,1,1],f32> 
    %none_13 = torch.constant.none
    %none_14 = torch.constant.none
    %int6_15 = torch.constant.int 6
    %cpu_16 = torch.constant.device "cpu"
    %int0_17 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %11, %none_13, %none_14, %int6_15, %cpu_16, %int0_17 : !torch.vtensor<[1,1,1],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %11 : !torch.vtensor<[1,1,1],f32>
  }
  func.func @test_reduction_masked_var_cpu_float32_16(%arg0: !torch.vtensor<[5,5,5],f32>) -> !torch.vtensor<[1,1,1],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_5_5_5_torch.uint8_1> : tensor<5x5x5xui8>) : !torch.vtensor<[5,5,5],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[5,5,5],ui8>, !torch.int -> !torch.vtensor<[5,5,5],i1>
    %2 = torch.vtensor.literal(dense<0.000000e+00> : tensor<f32>) : !torch.vtensor<[],f32>
    %3 = torch.vtensor.literal(dense<0.000000e+00> : tensor<f32>) : !torch.vtensor<[],f32>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %true = torch.constant.bool true
    %none_3 = torch.constant.none
    %4 = torch.aten.sum.dim_IntList %1, %none_2, %true, %none_3 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1],si64>
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int11_6 = torch.constant.int 11
    %cpu_7 = torch.constant.device "cpu"
    %int0_8 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none_4, %none_5, %int11_6, %cpu_7, %int0_8 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_9 = torch.constant.none
    %5 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %6 = torch.aten.where.self %1, %arg0, %5 : !torch.vtensor<[5,5,5],i1>, !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[5,5,5],f32>
    %int0_10 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %int2 = torch.constant.int 2
    %7 = torch.prim.ListConstruct %int0_10, %int1, %int2 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %true_11 = torch.constant.bool true
    %int6 = torch.constant.int 6
    %8 = torch.aten.sum.dim_IntList %6, %7, %true_11, %int6 : !torch.vtensor<[5,5,5],f32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[1,1,1],f32>
    %9 = torch.operator "torch.aten.divide.Tensor"(%8, %4) : (!torch.vtensor<[1,1,1],f32>, !torch.vtensor<[1,1,1],si64>) -> !torch.vtensor<[1,1,1],f32> 
    %int1_12 = torch.constant.int 1
    %10 = torch.operator "torch.aten.subtract.Tensor"(%arg0, %9, %int1_12) : (!torch.vtensor<[5,5,5],f32>, !torch.vtensor<[1,1,1],f32>, !torch.int) -> !torch.vtensor<[5,5,5],f32> 
    %11 = torch.aten.mul.Tensor %10, %10 : !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[5,5,5],f32> -> !torch.vtensor<[5,5,5],f32>
    %none_13 = torch.constant.none
    %none_14 = torch.constant.none
    %int11_15 = torch.constant.int 11
    %cpu_16 = torch.constant.device "cpu"
    %int0_17 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none_13, %none_14, %int11_15, %cpu_16, %int0_17 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_18 = torch.constant.none
    %12 = torch.aten.clone %3, %none_18 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %13 = torch.aten.where.self %1, %11, %12 : !torch.vtensor<[5,5,5],i1>, !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[5,5,5],f32>
    %int0_19 = torch.constant.int 0
    %int1_20 = torch.constant.int 1
    %int2_21 = torch.constant.int 2
    %14 = torch.prim.ListConstruct %int0_19, %int1_20, %int2_21 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %true_22 = torch.constant.bool true
    %int6_23 = torch.constant.int 6
    %15 = torch.aten.sum.dim_IntList %13, %14, %true_22, %int6_23 : !torch.vtensor<[5,5,5],f32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[1,1,1],f32>
    %16 = torch.operator "torch.aten.divide.Tensor"(%15, %4) : (!torch.vtensor<[1,1,1],f32>, !torch.vtensor<[1,1,1],si64>) -> !torch.vtensor<[1,1,1],f32> 
    %none_24 = torch.constant.none
    %none_25 = torch.constant.none
    %int6_26 = torch.constant.int 6
    %cpu_27 = torch.constant.device "cpu"
    %int0_28 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %16, %none_24, %none_25, %int6_26, %cpu_27, %int0_28 : !torch.vtensor<[1,1,1],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %16 : !torch.vtensor<[1,1,1],f32>
  }
  func.func @test_reduction_masked_var_cpu_float32_17(%arg0: !torch.vtensor<[5,5,5],f32>) -> !torch.vtensor<[1,1,1],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_5_5_1_torch.uint8_1> : tensor<5x5x1xui8>) : !torch.vtensor<[5,5,1],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[5,5,1],ui8>, !torch.int -> !torch.vtensor<[5,5,1],i1>
    %2 = torch.vtensor.literal(dense<0.000000e+00> : tensor<f32>) : !torch.vtensor<[],f32>
    %3 = torch.vtensor.literal(dense<0.000000e+00> : tensor<f32>) : !torch.vtensor<[],f32>
    %none = torch.constant.none
    %4 = torch.aten.clone %1, %none : !torch.vtensor<[5,5,1],i1>, !torch.none -> !torch.vtensor<[5,5,1],i1>
    %int5 = torch.constant.int 5
    %int5_0 = torch.constant.int 5
    %int5_1 = torch.constant.int 5
    %5 = torch.prim.ListConstruct %int5, %int5_0, %int5_1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %6 = torch.aten.expand %4, %5, %false : !torch.vtensor<[5,5,1],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[5,5,5],i1>
    %none_2 = torch.constant.none
    %none_3 = torch.constant.none
    %int11_4 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_2, %none_3, %int11_4, %cpu, %int0 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_5 = torch.constant.none
    %none_6 = torch.constant.none
    %int11_7 = torch.constant.int 11
    %cpu_8 = torch.constant.device "cpu"
    %int0_9 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_5, %none_6, %int11_7, %cpu_8, %int0_9 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_10 = torch.constant.none
    %true = torch.constant.bool true
    %none_11 = torch.constant.none
    %7 = torch.aten.sum.dim_IntList %6, %none_10, %true, %none_11 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1],si64>
    %none_12 = torch.constant.none
    %none_13 = torch.constant.none
    %int11_14 = torch.constant.int 11
    %cpu_15 = torch.constant.device "cpu"
    %int0_16 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_12, %none_13, %int11_14, %cpu_15, %int0_16 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_17 = torch.constant.none
    %8 = torch.aten.clone %2, %none_17 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %9 = torch.aten.where.self %6, %arg0, %8 : !torch.vtensor<[5,5,5],i1>, !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[5,5,5],f32>
    %int0_18 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %int2 = torch.constant.int 2
    %10 = torch.prim.ListConstruct %int0_18, %int1, %int2 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %true_19 = torch.constant.bool true
    %int6 = torch.constant.int 6
    %11 = torch.aten.sum.dim_IntList %9, %10, %true_19, %int6 : !torch.vtensor<[5,5,5],f32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[1,1,1],f32>
    %12 = torch.operator "torch.aten.divide.Tensor"(%11, %7) : (!torch.vtensor<[1,1,1],f32>, !torch.vtensor<[1,1,1],si64>) -> !torch.vtensor<[1,1,1],f32> 
    %int1_20 = torch.constant.int 1
    %13 = torch.operator "torch.aten.subtract.Tensor"(%arg0, %12, %int1_20) : (!torch.vtensor<[5,5,5],f32>, !torch.vtensor<[1,1,1],f32>, !torch.int) -> !torch.vtensor<[5,5,5],f32> 
    %14 = torch.aten.mul.Tensor %13, %13 : !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[5,5,5],f32> -> !torch.vtensor<[5,5,5],f32>
    %none_21 = torch.constant.none
    %none_22 = torch.constant.none
    %int11_23 = torch.constant.int 11
    %cpu_24 = torch.constant.device "cpu"
    %int0_25 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_21, %none_22, %int11_23, %cpu_24, %int0_25 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_26 = torch.constant.none
    %15 = torch.aten.clone %3, %none_26 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %16 = torch.aten.where.self %6, %14, %15 : !torch.vtensor<[5,5,5],i1>, !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[5,5,5],f32>
    %int0_27 = torch.constant.int 0
    %int1_28 = torch.constant.int 1
    %int2_29 = torch.constant.int 2
    %17 = torch.prim.ListConstruct %int0_27, %int1_28, %int2_29 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %true_30 = torch.constant.bool true
    %int6_31 = torch.constant.int 6
    %18 = torch.aten.sum.dim_IntList %16, %17, %true_30, %int6_31 : !torch.vtensor<[5,5,5],f32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[1,1,1],f32>
    %19 = torch.operator "torch.aten.divide.Tensor"(%18, %7) : (!torch.vtensor<[1,1,1],f32>, !torch.vtensor<[1,1,1],si64>) -> !torch.vtensor<[1,1,1],f32> 
    %none_32 = torch.constant.none
    %none_33 = torch.constant.none
    %int6_34 = torch.constant.int 6
    %cpu_35 = torch.constant.device "cpu"
    %int0_36 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %19, %none_32, %none_33, %int6_34, %cpu_35, %int0_36 : !torch.vtensor<[1,1,1],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %19 : !torch.vtensor<[1,1,1],f32>
  }
  func.func @test_reduction_masked_var_cpu_float32_18(%arg0: !torch.vtensor<[5,5,5],f32>) -> !torch.vtensor<[1,1,1],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_5_1_5_torch.uint8_1> : tensor<5x1x5xui8>) : !torch.vtensor<[5,1,5],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[5,1,5],ui8>, !torch.int -> !torch.vtensor<[5,1,5],i1>
    %2 = torch.vtensor.literal(dense<0.000000e+00> : tensor<f32>) : !torch.vtensor<[],f32>
    %3 = torch.vtensor.literal(dense<0.000000e+00> : tensor<f32>) : !torch.vtensor<[],f32>
    %none = torch.constant.none
    %4 = torch.aten.clone %1, %none : !torch.vtensor<[5,1,5],i1>, !torch.none -> !torch.vtensor<[5,1,5],i1>
    %int5 = torch.constant.int 5
    %int5_0 = torch.constant.int 5
    %int5_1 = torch.constant.int 5
    %5 = torch.prim.ListConstruct %int5, %int5_0, %int5_1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %6 = torch.aten.expand %4, %5, %false : !torch.vtensor<[5,1,5],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[5,5,5],i1>
    %none_2 = torch.constant.none
    %none_3 = torch.constant.none
    %int11_4 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_2, %none_3, %int11_4, %cpu, %int0 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_5 = torch.constant.none
    %none_6 = torch.constant.none
    %int11_7 = torch.constant.int 11
    %cpu_8 = torch.constant.device "cpu"
    %int0_9 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_5, %none_6, %int11_7, %cpu_8, %int0_9 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_10 = torch.constant.none
    %true = torch.constant.bool true
    %none_11 = torch.constant.none
    %7 = torch.aten.sum.dim_IntList %6, %none_10, %true, %none_11 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1],si64>
    %none_12 = torch.constant.none
    %none_13 = torch.constant.none
    %int11_14 = torch.constant.int 11
    %cpu_15 = torch.constant.device "cpu"
    %int0_16 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_12, %none_13, %int11_14, %cpu_15, %int0_16 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_17 = torch.constant.none
    %8 = torch.aten.clone %2, %none_17 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %9 = torch.aten.where.self %6, %arg0, %8 : !torch.vtensor<[5,5,5],i1>, !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[5,5,5],f32>
    %int0_18 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %int2 = torch.constant.int 2
    %10 = torch.prim.ListConstruct %int0_18, %int1, %int2 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %true_19 = torch.constant.bool true
    %int6 = torch.constant.int 6
    %11 = torch.aten.sum.dim_IntList %9, %10, %true_19, %int6 : !torch.vtensor<[5,5,5],f32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[1,1,1],f32>
    %12 = torch.operator "torch.aten.divide.Tensor"(%11, %7) : (!torch.vtensor<[1,1,1],f32>, !torch.vtensor<[1,1,1],si64>) -> !torch.vtensor<[1,1,1],f32> 
    %int1_20 = torch.constant.int 1
    %13 = torch.operator "torch.aten.subtract.Tensor"(%arg0, %12, %int1_20) : (!torch.vtensor<[5,5,5],f32>, !torch.vtensor<[1,1,1],f32>, !torch.int) -> !torch.vtensor<[5,5,5],f32> 
    %14 = torch.aten.mul.Tensor %13, %13 : !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[5,5,5],f32> -> !torch.vtensor<[5,5,5],f32>
    %none_21 = torch.constant.none
    %none_22 = torch.constant.none
    %int11_23 = torch.constant.int 11
    %cpu_24 = torch.constant.device "cpu"
    %int0_25 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_21, %none_22, %int11_23, %cpu_24, %int0_25 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_26 = torch.constant.none
    %15 = torch.aten.clone %3, %none_26 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %16 = torch.aten.where.self %6, %14, %15 : !torch.vtensor<[5,5,5],i1>, !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[5,5,5],f32>
    %int0_27 = torch.constant.int 0
    %int1_28 = torch.constant.int 1
    %int2_29 = torch.constant.int 2
    %17 = torch.prim.ListConstruct %int0_27, %int1_28, %int2_29 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %true_30 = torch.constant.bool true
    %int6_31 = torch.constant.int 6
    %18 = torch.aten.sum.dim_IntList %16, %17, %true_30, %int6_31 : !torch.vtensor<[5,5,5],f32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[1,1,1],f32>
    %19 = torch.operator "torch.aten.divide.Tensor"(%18, %7) : (!torch.vtensor<[1,1,1],f32>, !torch.vtensor<[1,1,1],si64>) -> !torch.vtensor<[1,1,1],f32> 
    %none_32 = torch.constant.none
    %none_33 = torch.constant.none
    %int6_34 = torch.constant.int 6
    %cpu_35 = torch.constant.device "cpu"
    %int0_36 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %19, %none_32, %none_33, %int6_34, %cpu_35, %int0_36 : !torch.vtensor<[1,1,1],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %19 : !torch.vtensor<[1,1,1],f32>
  }
  func.func @test_reduction_masked_var_cpu_float32_19(%arg0: !torch.vtensor<[5,5,5],f32>) -> !torch.vtensor<[1,1,1],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_1_5_5_torch.uint8_1> : tensor<1x5x5xui8>) : !torch.vtensor<[1,5,5],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[1,5,5],ui8>, !torch.int -> !torch.vtensor<[1,5,5],i1>
    %2 = torch.vtensor.literal(dense<0.000000e+00> : tensor<f32>) : !torch.vtensor<[],f32>
    %3 = torch.vtensor.literal(dense<0.000000e+00> : tensor<f32>) : !torch.vtensor<[],f32>
    %none = torch.constant.none
    %4 = torch.aten.clone %1, %none : !torch.vtensor<[1,5,5],i1>, !torch.none -> !torch.vtensor<[1,5,5],i1>
    %int5 = torch.constant.int 5
    %int5_0 = torch.constant.int 5
    %int5_1 = torch.constant.int 5
    %5 = torch.prim.ListConstruct %int5, %int5_0, %int5_1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %6 = torch.aten.expand %4, %5, %false : !torch.vtensor<[1,5,5],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[5,5,5],i1>
    %none_2 = torch.constant.none
    %none_3 = torch.constant.none
    %int11_4 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_2, %none_3, %int11_4, %cpu, %int0 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_5 = torch.constant.none
    %none_6 = torch.constant.none
    %int11_7 = torch.constant.int 11
    %cpu_8 = torch.constant.device "cpu"
    %int0_9 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_5, %none_6, %int11_7, %cpu_8, %int0_9 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_10 = torch.constant.none
    %true = torch.constant.bool true
    %none_11 = torch.constant.none
    %7 = torch.aten.sum.dim_IntList %6, %none_10, %true, %none_11 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1],si64>
    %none_12 = torch.constant.none
    %none_13 = torch.constant.none
    %int11_14 = torch.constant.int 11
    %cpu_15 = torch.constant.device "cpu"
    %int0_16 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_12, %none_13, %int11_14, %cpu_15, %int0_16 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_17 = torch.constant.none
    %8 = torch.aten.clone %2, %none_17 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %9 = torch.aten.where.self %6, %arg0, %8 : !torch.vtensor<[5,5,5],i1>, !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[5,5,5],f32>
    %int0_18 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %int2 = torch.constant.int 2
    %10 = torch.prim.ListConstruct %int0_18, %int1, %int2 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %true_19 = torch.constant.bool true
    %int6 = torch.constant.int 6
    %11 = torch.aten.sum.dim_IntList %9, %10, %true_19, %int6 : !torch.vtensor<[5,5,5],f32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[1,1,1],f32>
    %12 = torch.operator "torch.aten.divide.Tensor"(%11, %7) : (!torch.vtensor<[1,1,1],f32>, !torch.vtensor<[1,1,1],si64>) -> !torch.vtensor<[1,1,1],f32> 
    %int1_20 = torch.constant.int 1
    %13 = torch.operator "torch.aten.subtract.Tensor"(%arg0, %12, %int1_20) : (!torch.vtensor<[5,5,5],f32>, !torch.vtensor<[1,1,1],f32>, !torch.int) -> !torch.vtensor<[5,5,5],f32> 
    %14 = torch.aten.mul.Tensor %13, %13 : !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[5,5,5],f32> -> !torch.vtensor<[5,5,5],f32>
    %none_21 = torch.constant.none
    %none_22 = torch.constant.none
    %int11_23 = torch.constant.int 11
    %cpu_24 = torch.constant.device "cpu"
    %int0_25 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_21, %none_22, %int11_23, %cpu_24, %int0_25 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_26 = torch.constant.none
    %15 = torch.aten.clone %3, %none_26 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %16 = torch.aten.where.self %6, %14, %15 : !torch.vtensor<[5,5,5],i1>, !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[5,5,5],f32>
    %int0_27 = torch.constant.int 0
    %int1_28 = torch.constant.int 1
    %int2_29 = torch.constant.int 2
    %17 = torch.prim.ListConstruct %int0_27, %int1_28, %int2_29 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %true_30 = torch.constant.bool true
    %int6_31 = torch.constant.int 6
    %18 = torch.aten.sum.dim_IntList %16, %17, %true_30, %int6_31 : !torch.vtensor<[5,5,5],f32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[1,1,1],f32>
    %19 = torch.operator "torch.aten.divide.Tensor"(%18, %7) : (!torch.vtensor<[1,1,1],f32>, !torch.vtensor<[1,1,1],si64>) -> !torch.vtensor<[1,1,1],f32> 
    %none_32 = torch.constant.none
    %none_33 = torch.constant.none
    %int6_34 = torch.constant.int 6
    %cpu_35 = torch.constant.device "cpu"
    %int0_36 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %19, %none_32, %none_33, %int6_34, %cpu_35, %int0_36 : !torch.vtensor<[1,1,1],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %19 : !torch.vtensor<[1,1,1],f32>
  }
  func.func @test_reduction_masked_var_cpu_float32_20(%arg0: !torch.vtensor<[5,5,5],f32>) -> !torch.vtensor<[1,1,1],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_5_5_torch.uint8_1> : tensor<5x5xui8>) : !torch.vtensor<[5,5],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[5,5],ui8>, !torch.int -> !torch.vtensor<[5,5],i1>
    %2 = torch.vtensor.literal(dense<0.000000e+00> : tensor<f32>) : !torch.vtensor<[],f32>
    %3 = torch.vtensor.literal(dense<0.000000e+00> : tensor<f32>) : !torch.vtensor<[],f32>
    %none = torch.constant.none
    %4 = torch.aten.clone %1, %none : !torch.vtensor<[5,5],i1>, !torch.none -> !torch.vtensor<[5,5],i1>
    %int5 = torch.constant.int 5
    %int5_0 = torch.constant.int 5
    %int5_1 = torch.constant.int 5
    %5 = torch.prim.ListConstruct %int5, %int5_0, %int5_1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %6 = torch.aten.expand %4, %5, %false : !torch.vtensor<[5,5],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[5,5,5],i1>
    %none_2 = torch.constant.none
    %none_3 = torch.constant.none
    %int11_4 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_2, %none_3, %int11_4, %cpu, %int0 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_5 = torch.constant.none
    %none_6 = torch.constant.none
    %int11_7 = torch.constant.int 11
    %cpu_8 = torch.constant.device "cpu"
    %int0_9 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_5, %none_6, %int11_7, %cpu_8, %int0_9 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_10 = torch.constant.none
    %true = torch.constant.bool true
    %none_11 = torch.constant.none
    %7 = torch.aten.sum.dim_IntList %6, %none_10, %true, %none_11 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1],si64>
    %none_12 = torch.constant.none
    %none_13 = torch.constant.none
    %int11_14 = torch.constant.int 11
    %cpu_15 = torch.constant.device "cpu"
    %int0_16 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_12, %none_13, %int11_14, %cpu_15, %int0_16 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_17 = torch.constant.none
    %8 = torch.aten.clone %2, %none_17 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %9 = torch.aten.where.self %6, %arg0, %8 : !torch.vtensor<[5,5,5],i1>, !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[5,5,5],f32>
    %int0_18 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %int2 = torch.constant.int 2
    %10 = torch.prim.ListConstruct %int0_18, %int1, %int2 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %true_19 = torch.constant.bool true
    %int6 = torch.constant.int 6
    %11 = torch.aten.sum.dim_IntList %9, %10, %true_19, %int6 : !torch.vtensor<[5,5,5],f32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[1,1,1],f32>
    %12 = torch.operator "torch.aten.divide.Tensor"(%11, %7) : (!torch.vtensor<[1,1,1],f32>, !torch.vtensor<[1,1,1],si64>) -> !torch.vtensor<[1,1,1],f32> 
    %int1_20 = torch.constant.int 1
    %13 = torch.operator "torch.aten.subtract.Tensor"(%arg0, %12, %int1_20) : (!torch.vtensor<[5,5,5],f32>, !torch.vtensor<[1,1,1],f32>, !torch.int) -> !torch.vtensor<[5,5,5],f32> 
    %14 = torch.aten.mul.Tensor %13, %13 : !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[5,5,5],f32> -> !torch.vtensor<[5,5,5],f32>
    %none_21 = torch.constant.none
    %none_22 = torch.constant.none
    %int11_23 = torch.constant.int 11
    %cpu_24 = torch.constant.device "cpu"
    %int0_25 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_21, %none_22, %int11_23, %cpu_24, %int0_25 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_26 = torch.constant.none
    %15 = torch.aten.clone %3, %none_26 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %16 = torch.aten.where.self %6, %14, %15 : !torch.vtensor<[5,5,5],i1>, !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[5,5,5],f32>
    %int0_27 = torch.constant.int 0
    %int1_28 = torch.constant.int 1
    %int2_29 = torch.constant.int 2
    %17 = torch.prim.ListConstruct %int0_27, %int1_28, %int2_29 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %true_30 = torch.constant.bool true
    %int6_31 = torch.constant.int 6
    %18 = torch.aten.sum.dim_IntList %16, %17, %true_30, %int6_31 : !torch.vtensor<[5,5,5],f32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[1,1,1],f32>
    %19 = torch.operator "torch.aten.divide.Tensor"(%18, %7) : (!torch.vtensor<[1,1,1],f32>, !torch.vtensor<[1,1,1],si64>) -> !torch.vtensor<[1,1,1],f32> 
    %none_32 = torch.constant.none
    %none_33 = torch.constant.none
    %int6_34 = torch.constant.int 6
    %cpu_35 = torch.constant.device "cpu"
    %int0_36 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %19, %none_32, %none_33, %int6_34, %cpu_35, %int0_36 : !torch.vtensor<[1,1,1],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %19 : !torch.vtensor<[1,1,1],f32>
  }
  func.func @test_reduction_masked_var_cpu_float32_21(%arg0: !torch.vtensor<[5,5,5],f32>) -> !torch.vtensor<[1,1,1],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_5_torch.uint8_3> : tensor<5xui8>) : !torch.vtensor<[5],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[5],ui8>, !torch.int -> !torch.vtensor<[5],i1>
    %2 = torch.vtensor.literal(dense<0.000000e+00> : tensor<f32>) : !torch.vtensor<[],f32>
    %3 = torch.vtensor.literal(dense<0.000000e+00> : tensor<f32>) : !torch.vtensor<[],f32>
    %none = torch.constant.none
    %4 = torch.aten.clone %1, %none : !torch.vtensor<[5],i1>, !torch.none -> !torch.vtensor<[5],i1>
    %int5 = torch.constant.int 5
    %int5_0 = torch.constant.int 5
    %int5_1 = torch.constant.int 5
    %5 = torch.prim.ListConstruct %int5, %int5_0, %int5_1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %6 = torch.aten.expand %4, %5, %false : !torch.vtensor<[5],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[5,5,5],i1>
    %none_2 = torch.constant.none
    %none_3 = torch.constant.none
    %int11_4 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_2, %none_3, %int11_4, %cpu, %int0 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_5 = torch.constant.none
    %none_6 = torch.constant.none
    %int11_7 = torch.constant.int 11
    %cpu_8 = torch.constant.device "cpu"
    %int0_9 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_5, %none_6, %int11_7, %cpu_8, %int0_9 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_10 = torch.constant.none
    %true = torch.constant.bool true
    %none_11 = torch.constant.none
    %7 = torch.aten.sum.dim_IntList %6, %none_10, %true, %none_11 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1],si64>
    %none_12 = torch.constant.none
    %none_13 = torch.constant.none
    %int11_14 = torch.constant.int 11
    %cpu_15 = torch.constant.device "cpu"
    %int0_16 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_12, %none_13, %int11_14, %cpu_15, %int0_16 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_17 = torch.constant.none
    %8 = torch.aten.clone %2, %none_17 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %9 = torch.aten.where.self %6, %arg0, %8 : !torch.vtensor<[5,5,5],i1>, !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[5,5,5],f32>
    %int0_18 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %int2 = torch.constant.int 2
    %10 = torch.prim.ListConstruct %int0_18, %int1, %int2 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %true_19 = torch.constant.bool true
    %int6 = torch.constant.int 6
    %11 = torch.aten.sum.dim_IntList %9, %10, %true_19, %int6 : !torch.vtensor<[5,5,5],f32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[1,1,1],f32>
    %12 = torch.operator "torch.aten.divide.Tensor"(%11, %7) : (!torch.vtensor<[1,1,1],f32>, !torch.vtensor<[1,1,1],si64>) -> !torch.vtensor<[1,1,1],f32> 
    %int1_20 = torch.constant.int 1
    %13 = torch.operator "torch.aten.subtract.Tensor"(%arg0, %12, %int1_20) : (!torch.vtensor<[5,5,5],f32>, !torch.vtensor<[1,1,1],f32>, !torch.int) -> !torch.vtensor<[5,5,5],f32> 
    %14 = torch.aten.mul.Tensor %13, %13 : !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[5,5,5],f32> -> !torch.vtensor<[5,5,5],f32>
    %none_21 = torch.constant.none
    %none_22 = torch.constant.none
    %int11_23 = torch.constant.int 11
    %cpu_24 = torch.constant.device "cpu"
    %int0_25 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_21, %none_22, %int11_23, %cpu_24, %int0_25 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_26 = torch.constant.none
    %15 = torch.aten.clone %3, %none_26 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %16 = torch.aten.where.self %6, %14, %15 : !torch.vtensor<[5,5,5],i1>, !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[5,5,5],f32>
    %int0_27 = torch.constant.int 0
    %int1_28 = torch.constant.int 1
    %int2_29 = torch.constant.int 2
    %17 = torch.prim.ListConstruct %int0_27, %int1_28, %int2_29 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %true_30 = torch.constant.bool true
    %int6_31 = torch.constant.int 6
    %18 = torch.aten.sum.dim_IntList %16, %17, %true_30, %int6_31 : !torch.vtensor<[5,5,5],f32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[1,1,1],f32>
    %19 = torch.operator "torch.aten.divide.Tensor"(%18, %7) : (!torch.vtensor<[1,1,1],f32>, !torch.vtensor<[1,1,1],si64>) -> !torch.vtensor<[1,1,1],f32> 
    %none_32 = torch.constant.none
    %none_33 = torch.constant.none
    %int6_34 = torch.constant.int 6
    %cpu_35 = torch.constant.device "cpu"
    %int0_36 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %19, %none_32, %none_33, %int6_34, %cpu_35, %int0_36 : !torch.vtensor<[1,1,1],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %19 : !torch.vtensor<[1,1,1],f32>
  }
  func.func @test_reduction_masked_var_cpu_float32_22(%arg0: !torch.vtensor<[5,5,5],f32>) -> !torch.vtensor<[],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %int5 = torch.constant.int 5
    %int5_0 = torch.constant.int 5
    %int5_1 = torch.constant.int 5
    %0 = torch.prim.ListConstruct %int5, %int5_0, %int5_1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %int4 = torch.constant.int 4
    %none = torch.constant.none
    %cpu = torch.constant.device "cpu"
    %false = torch.constant.bool false
    %1 = torch.aten.ones %0, %int4, %none, %cpu, %false : !torch.list<int>, !torch.int, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[5,5,5],si64>
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %int2 = torch.constant.int 2
    %2 = torch.prim.ListConstruct %int0, %int1, %int2 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %int4_2 = torch.constant.int 4
    %3 = torch.aten.sum.dim_IntList %1, %2, %true, %int4_2 : !torch.vtensor<[5,5,5],si64>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[1,1,1],si64>
    %int0_3 = torch.constant.int 0
    %int1_4 = torch.constant.int 1
    %int2_5 = torch.constant.int 2
    %4 = torch.prim.ListConstruct %int0_3, %int1_4, %int2_5 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %true_6 = torch.constant.bool true
    %int6 = torch.constant.int 6
    %5 = torch.aten.sum.dim_IntList %arg0, %4, %true_6, %int6 : !torch.vtensor<[5,5,5],f32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[1,1,1],f32>
    %6 = torch.operator "torch.aten.divide.Tensor"(%5, %3) : (!torch.vtensor<[1,1,1],f32>, !torch.vtensor<[1,1,1],si64>) -> !torch.vtensor<[1,1,1],f32> 
    %int1_7 = torch.constant.int 1
    %7 = torch.operator "torch.aten.subtract.Tensor"(%arg0, %6, %int1_7) : (!torch.vtensor<[5,5,5],f32>, !torch.vtensor<[1,1,1],f32>, !torch.int) -> !torch.vtensor<[5,5,5],f32> 
    %8 = torch.aten.mul.Tensor %7, %7 : !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[5,5,5],f32> -> !torch.vtensor<[5,5,5],f32>
    %int0_8 = torch.constant.int 0
    %int1_9 = torch.constant.int 1
    %int2_10 = torch.constant.int 2
    %9 = torch.prim.ListConstruct %int0_8, %int1_9, %int2_10 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_11 = torch.constant.bool false
    %int6_12 = torch.constant.int 6
    %10 = torch.aten.sum.dim_IntList %8, %9, %false_11, %int6_12 : !torch.vtensor<[5,5,5],f32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[],f32>
    %11 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %12 = torch.aten.view %3, %11 : !torch.vtensor<[1,1,1],si64>, !torch.list<int> -> !torch.vtensor<[],si64>
    %none_13 = torch.constant.none
    %none_14 = torch.constant.none
    %int4_15 = torch.constant.int 4
    %cpu_16 = torch.constant.device "cpu"
    %int0_17 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %12, %none_13, %none_14, %int4_15, %cpu_16, %int0_17 : !torch.vtensor<[],si64>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int6_18 = torch.constant.int 6
    %13 = torch.prims.convert_element_type %12, %int6_18 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[],f32>
    %float1.000000e00 = torch.constant.float 1.000000e+00
    %14 = torch.prim.NumToTensor.Scalar %float1.000000e00 : !torch.float -> !torch.vtensor<[],f32>
    %int1_19 = torch.constant.int 1
    %15 = torch.operator "torch.aten.subtract.Tensor"(%13, %14, %int1_19) : (!torch.vtensor<[],f32>, !torch.vtensor<[],f32>, !torch.int) -> !torch.vtensor<[],f32> 
    %16 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %none_20 = torch.constant.none
    %none_21 = torch.constant.none
    %none_22 = torch.constant.none
    %false_23 = torch.constant.bool false
    %17 = torch.aten.new_zeros %15, %16, %none_20, %none_21, %none_22, %false_23 : !torch.vtensor<[],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[],f32>
    %18 = torch.aten.maximum %15, %17 : !torch.vtensor<[],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[],f32>
    %19 = torch.operator "torch.aten.divide.Tensor"(%10, %18) : (!torch.vtensor<[],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> 
    %none_24 = torch.constant.none
    %none_25 = torch.constant.none
    %int6_26 = torch.constant.int 6
    %cpu_27 = torch.constant.device "cpu"
    %int0_28 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %19, %none_24, %none_25, %int6_26, %cpu_27, %int0_28 : !torch.vtensor<[],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %19 : !torch.vtensor<[],f32>
  }
  func.func @test_reduction_masked_var_cpu_float32_23(%arg0: !torch.vtensor<[5,5,5],f32>) -> !torch.vtensor<[],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_5_5_5_torch.uint8_2> : tensor<5x5x5xui8>) : !torch.vtensor<[5,5,5],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[5,5,5],ui8>, !torch.int -> !torch.vtensor<[5,5,5],i1>
    %2 = torch.vtensor.literal(dense<0.000000e+00> : tensor<f32>) : !torch.vtensor<[],f32>
    %3 = torch.vtensor.literal(dense<0.000000e+00> : tensor<f32>) : !torch.vtensor<[],f32>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %true = torch.constant.bool true
    %none_3 = torch.constant.none
    %4 = torch.aten.sum.dim_IntList %1, %none_2, %true, %none_3 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1],si64>
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int11_6 = torch.constant.int 11
    %cpu_7 = torch.constant.device "cpu"
    %int0_8 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none_4, %none_5, %int11_6, %cpu_7, %int0_8 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_9 = torch.constant.none
    %5 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %6 = torch.aten.where.self %1, %arg0, %5 : !torch.vtensor<[5,5,5],i1>, !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[5,5,5],f32>
    %int0_10 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %int2 = torch.constant.int 2
    %7 = torch.prim.ListConstruct %int0_10, %int1, %int2 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %true_11 = torch.constant.bool true
    %int6 = torch.constant.int 6
    %8 = torch.aten.sum.dim_IntList %6, %7, %true_11, %int6 : !torch.vtensor<[5,5,5],f32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[1,1,1],f32>
    %9 = torch.operator "torch.aten.divide.Tensor"(%8, %4) : (!torch.vtensor<[1,1,1],f32>, !torch.vtensor<[1,1,1],si64>) -> !torch.vtensor<[1,1,1],f32> 
    %int1_12 = torch.constant.int 1
    %10 = torch.operator "torch.aten.subtract.Tensor"(%arg0, %9, %int1_12) : (!torch.vtensor<[5,5,5],f32>, !torch.vtensor<[1,1,1],f32>, !torch.int) -> !torch.vtensor<[5,5,5],f32> 
    %11 = torch.aten.mul.Tensor %10, %10 : !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[5,5,5],f32> -> !torch.vtensor<[5,5,5],f32>
    %none_13 = torch.constant.none
    %none_14 = torch.constant.none
    %int11_15 = torch.constant.int 11
    %cpu_16 = torch.constant.device "cpu"
    %int0_17 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none_13, %none_14, %int11_15, %cpu_16, %int0_17 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_18 = torch.constant.none
    %12 = torch.aten.clone %3, %none_18 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %13 = torch.aten.where.self %1, %11, %12 : !torch.vtensor<[5,5,5],i1>, !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[5,5,5],f32>
    %int0_19 = torch.constant.int 0
    %int1_20 = torch.constant.int 1
    %int2_21 = torch.constant.int 2
    %14 = torch.prim.ListConstruct %int0_19, %int1_20, %int2_21 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %int6_22 = torch.constant.int 6
    %15 = torch.aten.sum.dim_IntList %13, %14, %false, %int6_22 : !torch.vtensor<[5,5,5],f32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[],f32>
    %16 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %17 = torch.aten.view %4, %16 : !torch.vtensor<[1,1,1],si64>, !torch.list<int> -> !torch.vtensor<[],si64>
    %none_23 = torch.constant.none
    %none_24 = torch.constant.none
    %int4 = torch.constant.int 4
    %cpu_25 = torch.constant.device "cpu"
    %int0_26 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %17, %none_23, %none_24, %int4, %cpu_25, %int0_26 : !torch.vtensor<[],si64>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int6_27 = torch.constant.int 6
    %18 = torch.prims.convert_element_type %17, %int6_27 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[],f32>
    %float1.000000e00 = torch.constant.float 1.000000e+00
    %19 = torch.prim.NumToTensor.Scalar %float1.000000e00 : !torch.float -> !torch.vtensor<[],f32>
    %int1_28 = torch.constant.int 1
    %20 = torch.operator "torch.aten.subtract.Tensor"(%18, %19, %int1_28) : (!torch.vtensor<[],f32>, !torch.vtensor<[],f32>, !torch.int) -> !torch.vtensor<[],f32> 
    %21 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %none_29 = torch.constant.none
    %none_30 = torch.constant.none
    %none_31 = torch.constant.none
    %false_32 = torch.constant.bool false
    %22 = torch.aten.new_zeros %20, %21, %none_29, %none_30, %none_31, %false_32 : !torch.vtensor<[],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[],f32>
    %23 = torch.aten.maximum %20, %22 : !torch.vtensor<[],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[],f32>
    %24 = torch.operator "torch.aten.divide.Tensor"(%15, %23) : (!torch.vtensor<[],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> 
    %none_33 = torch.constant.none
    %none_34 = torch.constant.none
    %int6_35 = torch.constant.int 6
    %cpu_36 = torch.constant.device "cpu"
    %int0_37 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %24, %none_33, %none_34, %int6_35, %cpu_36, %int0_37 : !torch.vtensor<[],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %24 : !torch.vtensor<[],f32>
  }
  func.func @test_reduction_masked_var_cpu_float32_24(%arg0: !torch.vtensor<[5,5,5],f32>) -> !torch.vtensor<[],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_5_5_1_torch.uint8_2> : tensor<5x5x1xui8>) : !torch.vtensor<[5,5,1],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[5,5,1],ui8>, !torch.int -> !torch.vtensor<[5,5,1],i1>
    %2 = torch.vtensor.literal(dense<0.000000e+00> : tensor<f32>) : !torch.vtensor<[],f32>
    %3 = torch.vtensor.literal(dense<0.000000e+00> : tensor<f32>) : !torch.vtensor<[],f32>
    %none = torch.constant.none
    %4 = torch.aten.clone %1, %none : !torch.vtensor<[5,5,1],i1>, !torch.none -> !torch.vtensor<[5,5,1],i1>
    %int5 = torch.constant.int 5
    %int5_0 = torch.constant.int 5
    %int5_1 = torch.constant.int 5
    %5 = torch.prim.ListConstruct %int5, %int5_0, %int5_1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %6 = torch.aten.expand %4, %5, %false : !torch.vtensor<[5,5,1],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[5,5,5],i1>
    %none_2 = torch.constant.none
    %none_3 = torch.constant.none
    %int11_4 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_2, %none_3, %int11_4, %cpu, %int0 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_5 = torch.constant.none
    %none_6 = torch.constant.none
    %int11_7 = torch.constant.int 11
    %cpu_8 = torch.constant.device "cpu"
    %int0_9 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_5, %none_6, %int11_7, %cpu_8, %int0_9 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_10 = torch.constant.none
    %true = torch.constant.bool true
    %none_11 = torch.constant.none
    %7 = torch.aten.sum.dim_IntList %6, %none_10, %true, %none_11 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1],si64>
    %none_12 = torch.constant.none
    %none_13 = torch.constant.none
    %int11_14 = torch.constant.int 11
    %cpu_15 = torch.constant.device "cpu"
    %int0_16 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_12, %none_13, %int11_14, %cpu_15, %int0_16 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_17 = torch.constant.none
    %8 = torch.aten.clone %2, %none_17 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %9 = torch.aten.where.self %6, %arg0, %8 : !torch.vtensor<[5,5,5],i1>, !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[5,5,5],f32>
    %int0_18 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %int2 = torch.constant.int 2
    %10 = torch.prim.ListConstruct %int0_18, %int1, %int2 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %true_19 = torch.constant.bool true
    %int6 = torch.constant.int 6
    %11 = torch.aten.sum.dim_IntList %9, %10, %true_19, %int6 : !torch.vtensor<[5,5,5],f32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[1,1,1],f32>
    %12 = torch.operator "torch.aten.divide.Tensor"(%11, %7) : (!torch.vtensor<[1,1,1],f32>, !torch.vtensor<[1,1,1],si64>) -> !torch.vtensor<[1,1,1],f32> 
    %int1_20 = torch.constant.int 1
    %13 = torch.operator "torch.aten.subtract.Tensor"(%arg0, %12, %int1_20) : (!torch.vtensor<[5,5,5],f32>, !torch.vtensor<[1,1,1],f32>, !torch.int) -> !torch.vtensor<[5,5,5],f32> 
    %14 = torch.aten.mul.Tensor %13, %13 : !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[5,5,5],f32> -> !torch.vtensor<[5,5,5],f32>
    %none_21 = torch.constant.none
    %none_22 = torch.constant.none
    %int11_23 = torch.constant.int 11
    %cpu_24 = torch.constant.device "cpu"
    %int0_25 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_21, %none_22, %int11_23, %cpu_24, %int0_25 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_26 = torch.constant.none
    %15 = torch.aten.clone %3, %none_26 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %16 = torch.aten.where.self %6, %14, %15 : !torch.vtensor<[5,5,5],i1>, !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[5,5,5],f32>
    %int0_27 = torch.constant.int 0
    %int1_28 = torch.constant.int 1
    %int2_29 = torch.constant.int 2
    %17 = torch.prim.ListConstruct %int0_27, %int1_28, %int2_29 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_30 = torch.constant.bool false
    %int6_31 = torch.constant.int 6
    %18 = torch.aten.sum.dim_IntList %16, %17, %false_30, %int6_31 : !torch.vtensor<[5,5,5],f32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[],f32>
    %19 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %20 = torch.aten.view %7, %19 : !torch.vtensor<[1,1,1],si64>, !torch.list<int> -> !torch.vtensor<[],si64>
    %none_32 = torch.constant.none
    %none_33 = torch.constant.none
    %int4 = torch.constant.int 4
    %cpu_34 = torch.constant.device "cpu"
    %int0_35 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %20, %none_32, %none_33, %int4, %cpu_34, %int0_35 : !torch.vtensor<[],si64>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int6_36 = torch.constant.int 6
    %21 = torch.prims.convert_element_type %20, %int6_36 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[],f32>
    %float1.000000e00 = torch.constant.float 1.000000e+00
    %22 = torch.prim.NumToTensor.Scalar %float1.000000e00 : !torch.float -> !torch.vtensor<[],f32>
    %int1_37 = torch.constant.int 1
    %23 = torch.operator "torch.aten.subtract.Tensor"(%21, %22, %int1_37) : (!torch.vtensor<[],f32>, !torch.vtensor<[],f32>, !torch.int) -> !torch.vtensor<[],f32> 
    %24 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %none_38 = torch.constant.none
    %none_39 = torch.constant.none
    %none_40 = torch.constant.none
    %false_41 = torch.constant.bool false
    %25 = torch.aten.new_zeros %23, %24, %none_38, %none_39, %none_40, %false_41 : !torch.vtensor<[],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[],f32>
    %26 = torch.aten.maximum %23, %25 : !torch.vtensor<[],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[],f32>
    %27 = torch.operator "torch.aten.divide.Tensor"(%18, %26) : (!torch.vtensor<[],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> 
    %none_42 = torch.constant.none
    %none_43 = torch.constant.none
    %int6_44 = torch.constant.int 6
    %cpu_45 = torch.constant.device "cpu"
    %int0_46 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %27, %none_42, %none_43, %int6_44, %cpu_45, %int0_46 : !torch.vtensor<[],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %27 : !torch.vtensor<[],f32>
  }
  func.func @test_reduction_masked_var_cpu_float32_25(%arg0: !torch.vtensor<[5,5,5],f32>) -> !torch.vtensor<[],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_5_1_5_torch.uint8_2> : tensor<5x1x5xui8>) : !torch.vtensor<[5,1,5],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[5,1,5],ui8>, !torch.int -> !torch.vtensor<[5,1,5],i1>
    %2 = torch.vtensor.literal(dense<0.000000e+00> : tensor<f32>) : !torch.vtensor<[],f32>
    %3 = torch.vtensor.literal(dense<0.000000e+00> : tensor<f32>) : !torch.vtensor<[],f32>
    %none = torch.constant.none
    %4 = torch.aten.clone %1, %none : !torch.vtensor<[5,1,5],i1>, !torch.none -> !torch.vtensor<[5,1,5],i1>
    %int5 = torch.constant.int 5
    %int5_0 = torch.constant.int 5
    %int5_1 = torch.constant.int 5
    %5 = torch.prim.ListConstruct %int5, %int5_0, %int5_1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %6 = torch.aten.expand %4, %5, %false : !torch.vtensor<[5,1,5],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[5,5,5],i1>
    %none_2 = torch.constant.none
    %none_3 = torch.constant.none
    %int11_4 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_2, %none_3, %int11_4, %cpu, %int0 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_5 = torch.constant.none
    %none_6 = torch.constant.none
    %int11_7 = torch.constant.int 11
    %cpu_8 = torch.constant.device "cpu"
    %int0_9 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_5, %none_6, %int11_7, %cpu_8, %int0_9 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_10 = torch.constant.none
    %true = torch.constant.bool true
    %none_11 = torch.constant.none
    %7 = torch.aten.sum.dim_IntList %6, %none_10, %true, %none_11 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1],si64>
    %none_12 = torch.constant.none
    %none_13 = torch.constant.none
    %int11_14 = torch.constant.int 11
    %cpu_15 = torch.constant.device "cpu"
    %int0_16 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_12, %none_13, %int11_14, %cpu_15, %int0_16 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_17 = torch.constant.none
    %8 = torch.aten.clone %2, %none_17 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %9 = torch.aten.where.self %6, %arg0, %8 : !torch.vtensor<[5,5,5],i1>, !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[5,5,5],f32>
    %int0_18 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %int2 = torch.constant.int 2
    %10 = torch.prim.ListConstruct %int0_18, %int1, %int2 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %true_19 = torch.constant.bool true
    %int6 = torch.constant.int 6
    %11 = torch.aten.sum.dim_IntList %9, %10, %true_19, %int6 : !torch.vtensor<[5,5,5],f32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[1,1,1],f32>
    %12 = torch.operator "torch.aten.divide.Tensor"(%11, %7) : (!torch.vtensor<[1,1,1],f32>, !torch.vtensor<[1,1,1],si64>) -> !torch.vtensor<[1,1,1],f32> 
    %int1_20 = torch.constant.int 1
    %13 = torch.operator "torch.aten.subtract.Tensor"(%arg0, %12, %int1_20) : (!torch.vtensor<[5,5,5],f32>, !torch.vtensor<[1,1,1],f32>, !torch.int) -> !torch.vtensor<[5,5,5],f32> 
    %14 = torch.aten.mul.Tensor %13, %13 : !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[5,5,5],f32> -> !torch.vtensor<[5,5,5],f32>
    %none_21 = torch.constant.none
    %none_22 = torch.constant.none
    %int11_23 = torch.constant.int 11
    %cpu_24 = torch.constant.device "cpu"
    %int0_25 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_21, %none_22, %int11_23, %cpu_24, %int0_25 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_26 = torch.constant.none
    %15 = torch.aten.clone %3, %none_26 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %16 = torch.aten.where.self %6, %14, %15 : !torch.vtensor<[5,5,5],i1>, !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[5,5,5],f32>
    %int0_27 = torch.constant.int 0
    %int1_28 = torch.constant.int 1
    %int2_29 = torch.constant.int 2
    %17 = torch.prim.ListConstruct %int0_27, %int1_28, %int2_29 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_30 = torch.constant.bool false
    %int6_31 = torch.constant.int 6
    %18 = torch.aten.sum.dim_IntList %16, %17, %false_30, %int6_31 : !torch.vtensor<[5,5,5],f32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[],f32>
    %19 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %20 = torch.aten.view %7, %19 : !torch.vtensor<[1,1,1],si64>, !torch.list<int> -> !torch.vtensor<[],si64>
    %none_32 = torch.constant.none
    %none_33 = torch.constant.none
    %int4 = torch.constant.int 4
    %cpu_34 = torch.constant.device "cpu"
    %int0_35 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %20, %none_32, %none_33, %int4, %cpu_34, %int0_35 : !torch.vtensor<[],si64>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int6_36 = torch.constant.int 6
    %21 = torch.prims.convert_element_type %20, %int6_36 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[],f32>
    %float1.000000e00 = torch.constant.float 1.000000e+00
    %22 = torch.prim.NumToTensor.Scalar %float1.000000e00 : !torch.float -> !torch.vtensor<[],f32>
    %int1_37 = torch.constant.int 1
    %23 = torch.operator "torch.aten.subtract.Tensor"(%21, %22, %int1_37) : (!torch.vtensor<[],f32>, !torch.vtensor<[],f32>, !torch.int) -> !torch.vtensor<[],f32> 
    %24 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %none_38 = torch.constant.none
    %none_39 = torch.constant.none
    %none_40 = torch.constant.none
    %false_41 = torch.constant.bool false
    %25 = torch.aten.new_zeros %23, %24, %none_38, %none_39, %none_40, %false_41 : !torch.vtensor<[],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[],f32>
    %26 = torch.aten.maximum %23, %25 : !torch.vtensor<[],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[],f32>
    %27 = torch.operator "torch.aten.divide.Tensor"(%18, %26) : (!torch.vtensor<[],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> 
    %none_42 = torch.constant.none
    %none_43 = torch.constant.none
    %int6_44 = torch.constant.int 6
    %cpu_45 = torch.constant.device "cpu"
    %int0_46 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %27, %none_42, %none_43, %int6_44, %cpu_45, %int0_46 : !torch.vtensor<[],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %27 : !torch.vtensor<[],f32>
  }
  func.func @test_reduction_masked_var_cpu_float32_26(%arg0: !torch.vtensor<[5,5,5],f32>) -> !torch.vtensor<[],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_1_5_5_torch.uint8_2> : tensor<1x5x5xui8>) : !torch.vtensor<[1,5,5],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[1,5,5],ui8>, !torch.int -> !torch.vtensor<[1,5,5],i1>
    %2 = torch.vtensor.literal(dense<0.000000e+00> : tensor<f32>) : !torch.vtensor<[],f32>
    %3 = torch.vtensor.literal(dense<0.000000e+00> : tensor<f32>) : !torch.vtensor<[],f32>
    %none = torch.constant.none
    %4 = torch.aten.clone %1, %none : !torch.vtensor<[1,5,5],i1>, !torch.none -> !torch.vtensor<[1,5,5],i1>
    %int5 = torch.constant.int 5
    %int5_0 = torch.constant.int 5
    %int5_1 = torch.constant.int 5
    %5 = torch.prim.ListConstruct %int5, %int5_0, %int5_1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %6 = torch.aten.expand %4, %5, %false : !torch.vtensor<[1,5,5],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[5,5,5],i1>
    %none_2 = torch.constant.none
    %none_3 = torch.constant.none
    %int11_4 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_2, %none_3, %int11_4, %cpu, %int0 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_5 = torch.constant.none
    %none_6 = torch.constant.none
    %int11_7 = torch.constant.int 11
    %cpu_8 = torch.constant.device "cpu"
    %int0_9 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_5, %none_6, %int11_7, %cpu_8, %int0_9 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_10 = torch.constant.none
    %true = torch.constant.bool true
    %none_11 = torch.constant.none
    %7 = torch.aten.sum.dim_IntList %6, %none_10, %true, %none_11 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1],si64>
    %none_12 = torch.constant.none
    %none_13 = torch.constant.none
    %int11_14 = torch.constant.int 11
    %cpu_15 = torch.constant.device "cpu"
    %int0_16 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_12, %none_13, %int11_14, %cpu_15, %int0_16 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_17 = torch.constant.none
    %8 = torch.aten.clone %2, %none_17 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %9 = torch.aten.where.self %6, %arg0, %8 : !torch.vtensor<[5,5,5],i1>, !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[5,5,5],f32>
    %int0_18 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %int2 = torch.constant.int 2
    %10 = torch.prim.ListConstruct %int0_18, %int1, %int2 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %true_19 = torch.constant.bool true
    %int6 = torch.constant.int 6
    %11 = torch.aten.sum.dim_IntList %9, %10, %true_19, %int6 : !torch.vtensor<[5,5,5],f32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[1,1,1],f32>
    %12 = torch.operator "torch.aten.divide.Tensor"(%11, %7) : (!torch.vtensor<[1,1,1],f32>, !torch.vtensor<[1,1,1],si64>) -> !torch.vtensor<[1,1,1],f32> 
    %int1_20 = torch.constant.int 1
    %13 = torch.operator "torch.aten.subtract.Tensor"(%arg0, %12, %int1_20) : (!torch.vtensor<[5,5,5],f32>, !torch.vtensor<[1,1,1],f32>, !torch.int) -> !torch.vtensor<[5,5,5],f32> 
    %14 = torch.aten.mul.Tensor %13, %13 : !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[5,5,5],f32> -> !torch.vtensor<[5,5,5],f32>
    %none_21 = torch.constant.none
    %none_22 = torch.constant.none
    %int11_23 = torch.constant.int 11
    %cpu_24 = torch.constant.device "cpu"
    %int0_25 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_21, %none_22, %int11_23, %cpu_24, %int0_25 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_26 = torch.constant.none
    %15 = torch.aten.clone %3, %none_26 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %16 = torch.aten.where.self %6, %14, %15 : !torch.vtensor<[5,5,5],i1>, !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[5,5,5],f32>
    %int0_27 = torch.constant.int 0
    %int1_28 = torch.constant.int 1
    %int2_29 = torch.constant.int 2
    %17 = torch.prim.ListConstruct %int0_27, %int1_28, %int2_29 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_30 = torch.constant.bool false
    %int6_31 = torch.constant.int 6
    %18 = torch.aten.sum.dim_IntList %16, %17, %false_30, %int6_31 : !torch.vtensor<[5,5,5],f32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[],f32>
    %19 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %20 = torch.aten.view %7, %19 : !torch.vtensor<[1,1,1],si64>, !torch.list<int> -> !torch.vtensor<[],si64>
    %none_32 = torch.constant.none
    %none_33 = torch.constant.none
    %int4 = torch.constant.int 4
    %cpu_34 = torch.constant.device "cpu"
    %int0_35 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %20, %none_32, %none_33, %int4, %cpu_34, %int0_35 : !torch.vtensor<[],si64>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int6_36 = torch.constant.int 6
    %21 = torch.prims.convert_element_type %20, %int6_36 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[],f32>
    %float1.000000e00 = torch.constant.float 1.000000e+00
    %22 = torch.prim.NumToTensor.Scalar %float1.000000e00 : !torch.float -> !torch.vtensor<[],f32>
    %int1_37 = torch.constant.int 1
    %23 = torch.operator "torch.aten.subtract.Tensor"(%21, %22, %int1_37) : (!torch.vtensor<[],f32>, !torch.vtensor<[],f32>, !torch.int) -> !torch.vtensor<[],f32> 
    %24 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %none_38 = torch.constant.none
    %none_39 = torch.constant.none
    %none_40 = torch.constant.none
    %false_41 = torch.constant.bool false
    %25 = torch.aten.new_zeros %23, %24, %none_38, %none_39, %none_40, %false_41 : !torch.vtensor<[],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[],f32>
    %26 = torch.aten.maximum %23, %25 : !torch.vtensor<[],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[],f32>
    %27 = torch.operator "torch.aten.divide.Tensor"(%18, %26) : (!torch.vtensor<[],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> 
    %none_42 = torch.constant.none
    %none_43 = torch.constant.none
    %int6_44 = torch.constant.int 6
    %cpu_45 = torch.constant.device "cpu"
    %int0_46 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %27, %none_42, %none_43, %int6_44, %cpu_45, %int0_46 : !torch.vtensor<[],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %27 : !torch.vtensor<[],f32>
  }
  func.func @test_reduction_masked_var_cpu_float32_27(%arg0: !torch.vtensor<[5,5,5],f32>) -> !torch.vtensor<[],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_5_5_torch.uint8_2> : tensor<5x5xui8>) : !torch.vtensor<[5,5],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[5,5],ui8>, !torch.int -> !torch.vtensor<[5,5],i1>
    %2 = torch.vtensor.literal(dense<0.000000e+00> : tensor<f32>) : !torch.vtensor<[],f32>
    %3 = torch.vtensor.literal(dense<0.000000e+00> : tensor<f32>) : !torch.vtensor<[],f32>
    %none = torch.constant.none
    %4 = torch.aten.clone %1, %none : !torch.vtensor<[5,5],i1>, !torch.none -> !torch.vtensor<[5,5],i1>
    %int5 = torch.constant.int 5
    %int5_0 = torch.constant.int 5
    %int5_1 = torch.constant.int 5
    %5 = torch.prim.ListConstruct %int5, %int5_0, %int5_1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %6 = torch.aten.expand %4, %5, %false : !torch.vtensor<[5,5],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[5,5,5],i1>
    %none_2 = torch.constant.none
    %none_3 = torch.constant.none
    %int11_4 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_2, %none_3, %int11_4, %cpu, %int0 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_5 = torch.constant.none
    %none_6 = torch.constant.none
    %int11_7 = torch.constant.int 11
    %cpu_8 = torch.constant.device "cpu"
    %int0_9 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_5, %none_6, %int11_7, %cpu_8, %int0_9 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_10 = torch.constant.none
    %true = torch.constant.bool true
    %none_11 = torch.constant.none
    %7 = torch.aten.sum.dim_IntList %6, %none_10, %true, %none_11 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1],si64>
    %none_12 = torch.constant.none
    %none_13 = torch.constant.none
    %int11_14 = torch.constant.int 11
    %cpu_15 = torch.constant.device "cpu"
    %int0_16 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_12, %none_13, %int11_14, %cpu_15, %int0_16 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_17 = torch.constant.none
    %8 = torch.aten.clone %2, %none_17 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %9 = torch.aten.where.self %6, %arg0, %8 : !torch.vtensor<[5,5,5],i1>, !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[5,5,5],f32>
    %int0_18 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %int2 = torch.constant.int 2
    %10 = torch.prim.ListConstruct %int0_18, %int1, %int2 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %true_19 = torch.constant.bool true
    %int6 = torch.constant.int 6
    %11 = torch.aten.sum.dim_IntList %9, %10, %true_19, %int6 : !torch.vtensor<[5,5,5],f32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[1,1,1],f32>
    %12 = torch.operator "torch.aten.divide.Tensor"(%11, %7) : (!torch.vtensor<[1,1,1],f32>, !torch.vtensor<[1,1,1],si64>) -> !torch.vtensor<[1,1,1],f32> 
    %int1_20 = torch.constant.int 1
    %13 = torch.operator "torch.aten.subtract.Tensor"(%arg0, %12, %int1_20) : (!torch.vtensor<[5,5,5],f32>, !torch.vtensor<[1,1,1],f32>, !torch.int) -> !torch.vtensor<[5,5,5],f32> 
    %14 = torch.aten.mul.Tensor %13, %13 : !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[5,5,5],f32> -> !torch.vtensor<[5,5,5],f32>
    %none_21 = torch.constant.none
    %none_22 = torch.constant.none
    %int11_23 = torch.constant.int 11
    %cpu_24 = torch.constant.device "cpu"
    %int0_25 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_21, %none_22, %int11_23, %cpu_24, %int0_25 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_26 = torch.constant.none
    %15 = torch.aten.clone %3, %none_26 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %16 = torch.aten.where.self %6, %14, %15 : !torch.vtensor<[5,5,5],i1>, !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[5,5,5],f32>
    %int0_27 = torch.constant.int 0
    %int1_28 = torch.constant.int 1
    %int2_29 = torch.constant.int 2
    %17 = torch.prim.ListConstruct %int0_27, %int1_28, %int2_29 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_30 = torch.constant.bool false
    %int6_31 = torch.constant.int 6
    %18 = torch.aten.sum.dim_IntList %16, %17, %false_30, %int6_31 : !torch.vtensor<[5,5,5],f32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[],f32>
    %19 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %20 = torch.aten.view %7, %19 : !torch.vtensor<[1,1,1],si64>, !torch.list<int> -> !torch.vtensor<[],si64>
    %none_32 = torch.constant.none
    %none_33 = torch.constant.none
    %int4 = torch.constant.int 4
    %cpu_34 = torch.constant.device "cpu"
    %int0_35 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %20, %none_32, %none_33, %int4, %cpu_34, %int0_35 : !torch.vtensor<[],si64>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int6_36 = torch.constant.int 6
    %21 = torch.prims.convert_element_type %20, %int6_36 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[],f32>
    %float1.000000e00 = torch.constant.float 1.000000e+00
    %22 = torch.prim.NumToTensor.Scalar %float1.000000e00 : !torch.float -> !torch.vtensor<[],f32>
    %int1_37 = torch.constant.int 1
    %23 = torch.operator "torch.aten.subtract.Tensor"(%21, %22, %int1_37) : (!torch.vtensor<[],f32>, !torch.vtensor<[],f32>, !torch.int) -> !torch.vtensor<[],f32> 
    %24 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %none_38 = torch.constant.none
    %none_39 = torch.constant.none
    %none_40 = torch.constant.none
    %false_41 = torch.constant.bool false
    %25 = torch.aten.new_zeros %23, %24, %none_38, %none_39, %none_40, %false_41 : !torch.vtensor<[],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[],f32>
    %26 = torch.aten.maximum %23, %25 : !torch.vtensor<[],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[],f32>
    %27 = torch.operator "torch.aten.divide.Tensor"(%18, %26) : (!torch.vtensor<[],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> 
    %none_42 = torch.constant.none
    %none_43 = torch.constant.none
    %int6_44 = torch.constant.int 6
    %cpu_45 = torch.constant.device "cpu"
    %int0_46 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %27, %none_42, %none_43, %int6_44, %cpu_45, %int0_46 : !torch.vtensor<[],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %27 : !torch.vtensor<[],f32>
  }
  func.func @test_reduction_masked_var_cpu_float32_28(%arg0: !torch.vtensor<[5,5,5],f32>) -> !torch.vtensor<[],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_5_torch.uint8_4> : tensor<5xui8>) : !torch.vtensor<[5],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[5],ui8>, !torch.int -> !torch.vtensor<[5],i1>
    %2 = torch.vtensor.literal(dense<0.000000e+00> : tensor<f32>) : !torch.vtensor<[],f32>
    %3 = torch.vtensor.literal(dense<0.000000e+00> : tensor<f32>) : !torch.vtensor<[],f32>
    %none = torch.constant.none
    %4 = torch.aten.clone %1, %none : !torch.vtensor<[5],i1>, !torch.none -> !torch.vtensor<[5],i1>
    %int5 = torch.constant.int 5
    %int5_0 = torch.constant.int 5
    %int5_1 = torch.constant.int 5
    %5 = torch.prim.ListConstruct %int5, %int5_0, %int5_1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %6 = torch.aten.expand %4, %5, %false : !torch.vtensor<[5],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[5,5,5],i1>
    %none_2 = torch.constant.none
    %none_3 = torch.constant.none
    %int11_4 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_2, %none_3, %int11_4, %cpu, %int0 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_5 = torch.constant.none
    %none_6 = torch.constant.none
    %int11_7 = torch.constant.int 11
    %cpu_8 = torch.constant.device "cpu"
    %int0_9 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_5, %none_6, %int11_7, %cpu_8, %int0_9 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_10 = torch.constant.none
    %true = torch.constant.bool true
    %none_11 = torch.constant.none
    %7 = torch.aten.sum.dim_IntList %6, %none_10, %true, %none_11 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1],si64>
    %none_12 = torch.constant.none
    %none_13 = torch.constant.none
    %int11_14 = torch.constant.int 11
    %cpu_15 = torch.constant.device "cpu"
    %int0_16 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_12, %none_13, %int11_14, %cpu_15, %int0_16 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_17 = torch.constant.none
    %8 = torch.aten.clone %2, %none_17 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %9 = torch.aten.where.self %6, %arg0, %8 : !torch.vtensor<[5,5,5],i1>, !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[5,5,5],f32>
    %int0_18 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %int2 = torch.constant.int 2
    %10 = torch.prim.ListConstruct %int0_18, %int1, %int2 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %true_19 = torch.constant.bool true
    %int6 = torch.constant.int 6
    %11 = torch.aten.sum.dim_IntList %9, %10, %true_19, %int6 : !torch.vtensor<[5,5,5],f32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[1,1,1],f32>
    %12 = torch.operator "torch.aten.divide.Tensor"(%11, %7) : (!torch.vtensor<[1,1,1],f32>, !torch.vtensor<[1,1,1],si64>) -> !torch.vtensor<[1,1,1],f32> 
    %int1_20 = torch.constant.int 1
    %13 = torch.operator "torch.aten.subtract.Tensor"(%arg0, %12, %int1_20) : (!torch.vtensor<[5,5,5],f32>, !torch.vtensor<[1,1,1],f32>, !torch.int) -> !torch.vtensor<[5,5,5],f32> 
    %14 = torch.aten.mul.Tensor %13, %13 : !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[5,5,5],f32> -> !torch.vtensor<[5,5,5],f32>
    %none_21 = torch.constant.none
    %none_22 = torch.constant.none
    %int11_23 = torch.constant.int 11
    %cpu_24 = torch.constant.device "cpu"
    %int0_25 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_21, %none_22, %int11_23, %cpu_24, %int0_25 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_26 = torch.constant.none
    %15 = torch.aten.clone %3, %none_26 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %16 = torch.aten.where.self %6, %14, %15 : !torch.vtensor<[5,5,5],i1>, !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[5,5,5],f32>
    %int0_27 = torch.constant.int 0
    %int1_28 = torch.constant.int 1
    %int2_29 = torch.constant.int 2
    %17 = torch.prim.ListConstruct %int0_27, %int1_28, %int2_29 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_30 = torch.constant.bool false
    %int6_31 = torch.constant.int 6
    %18 = torch.aten.sum.dim_IntList %16, %17, %false_30, %int6_31 : !torch.vtensor<[5,5,5],f32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[],f32>
    %19 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %20 = torch.aten.view %7, %19 : !torch.vtensor<[1,1,1],si64>, !torch.list<int> -> !torch.vtensor<[],si64>
    %none_32 = torch.constant.none
    %none_33 = torch.constant.none
    %int4 = torch.constant.int 4
    %cpu_34 = torch.constant.device "cpu"
    %int0_35 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %20, %none_32, %none_33, %int4, %cpu_34, %int0_35 : !torch.vtensor<[],si64>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int6_36 = torch.constant.int 6
    %21 = torch.prims.convert_element_type %20, %int6_36 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[],f32>
    %float1.000000e00 = torch.constant.float 1.000000e+00
    %22 = torch.prim.NumToTensor.Scalar %float1.000000e00 : !torch.float -> !torch.vtensor<[],f32>
    %int1_37 = torch.constant.int 1
    %23 = torch.operator "torch.aten.subtract.Tensor"(%21, %22, %int1_37) : (!torch.vtensor<[],f32>, !torch.vtensor<[],f32>, !torch.int) -> !torch.vtensor<[],f32> 
    %24 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %none_38 = torch.constant.none
    %none_39 = torch.constant.none
    %none_40 = torch.constant.none
    %false_41 = torch.constant.bool false
    %25 = torch.aten.new_zeros %23, %24, %none_38, %none_39, %none_40, %false_41 : !torch.vtensor<[],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[],f32>
    %26 = torch.aten.maximum %23, %25 : !torch.vtensor<[],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[],f32>
    %27 = torch.operator "torch.aten.divide.Tensor"(%18, %26) : (!torch.vtensor<[],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> 
    %none_42 = torch.constant.none
    %none_43 = torch.constant.none
    %int6_44 = torch.constant.int 6
    %cpu_45 = torch.constant.device "cpu"
    %int0_46 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %27, %none_42, %none_43, %int6_44, %cpu_45, %int0_46 : !torch.vtensor<[],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %27 : !torch.vtensor<[],f32>
  }
  func.func @test_reduction_masked_var_cpu_float32_29(%arg0: !torch.vtensor<[5,5,5],f32>) -> !torch.vtensor<[1,1,1],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %int5 = torch.constant.int 5
    %int5_0 = torch.constant.int 5
    %int5_1 = torch.constant.int 5
    %0 = torch.prim.ListConstruct %int5, %int5_0, %int5_1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %int4 = torch.constant.int 4
    %none = torch.constant.none
    %cpu = torch.constant.device "cpu"
    %false = torch.constant.bool false
    %1 = torch.aten.ones %0, %int4, %none, %cpu, %false : !torch.list<int>, !torch.int, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[5,5,5],si64>
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %int2 = torch.constant.int 2
    %2 = torch.prim.ListConstruct %int0, %int1, %int2 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %int4_2 = torch.constant.int 4
    %3 = torch.aten.sum.dim_IntList %1, %2, %true, %int4_2 : !torch.vtensor<[5,5,5],si64>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[1,1,1],si64>
    %int0_3 = torch.constant.int 0
    %int1_4 = torch.constant.int 1
    %int2_5 = torch.constant.int 2
    %4 = torch.prim.ListConstruct %int0_3, %int1_4, %int2_5 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %true_6 = torch.constant.bool true
    %int6 = torch.constant.int 6
    %5 = torch.aten.sum.dim_IntList %arg0, %4, %true_6, %int6 : !torch.vtensor<[5,5,5],f32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[1,1,1],f32>
    %6 = torch.operator "torch.aten.divide.Tensor"(%5, %3) : (!torch.vtensor<[1,1,1],f32>, !torch.vtensor<[1,1,1],si64>) -> !torch.vtensor<[1,1,1],f32> 
    %int1_7 = torch.constant.int 1
    %7 = torch.operator "torch.aten.subtract.Tensor"(%arg0, %6, %int1_7) : (!torch.vtensor<[5,5,5],f32>, !torch.vtensor<[1,1,1],f32>, !torch.int) -> !torch.vtensor<[5,5,5],f32> 
    %8 = torch.aten.mul.Tensor %7, %7 : !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[5,5,5],f32> -> !torch.vtensor<[5,5,5],f32>
    %int0_8 = torch.constant.int 0
    %int1_9 = torch.constant.int 1
    %int2_10 = torch.constant.int 2
    %9 = torch.prim.ListConstruct %int0_8, %int1_9, %int2_10 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %true_11 = torch.constant.bool true
    %int6_12 = torch.constant.int 6
    %10 = torch.aten.sum.dim_IntList %8, %9, %true_11, %int6_12 : !torch.vtensor<[5,5,5],f32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[1,1,1],f32>
    %11 = torch.operator "torch.aten.divide.Tensor"(%10, %3) : (!torch.vtensor<[1,1,1],f32>, !torch.vtensor<[1,1,1],si64>) -> !torch.vtensor<[1,1,1],f32> 
    %none_13 = torch.constant.none
    %none_14 = torch.constant.none
    %int6_15 = torch.constant.int 6
    %cpu_16 = torch.constant.device "cpu"
    %int0_17 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %11, %none_13, %none_14, %int6_15, %cpu_16, %int0_17 : !torch.vtensor<[1,1,1],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %11 : !torch.vtensor<[1,1,1],f32>
  }
  func.func @test_reduction_masked_var_cpu_float32_30(%arg0: !torch.vtensor<[5,5,5],f32>) -> !torch.vtensor<[1,1,1],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_5_5_5_torch.uint8_3> : tensor<5x5x5xui8>) : !torch.vtensor<[5,5,5],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[5,5,5],ui8>, !torch.int -> !torch.vtensor<[5,5,5],i1>
    %2 = torch.vtensor.literal(dense<0.000000e+00> : tensor<f32>) : !torch.vtensor<[],f32>
    %3 = torch.vtensor.literal(dense<0.000000e+00> : tensor<f32>) : !torch.vtensor<[],f32>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %true = torch.constant.bool true
    %none_3 = torch.constant.none
    %4 = torch.aten.sum.dim_IntList %1, %none_2, %true, %none_3 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1],si64>
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int11_6 = torch.constant.int 11
    %cpu_7 = torch.constant.device "cpu"
    %int0_8 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none_4, %none_5, %int11_6, %cpu_7, %int0_8 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_9 = torch.constant.none
    %5 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %6 = torch.aten.where.self %1, %arg0, %5 : !torch.vtensor<[5,5,5],i1>, !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[5,5,5],f32>
    %int0_10 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %int2 = torch.constant.int 2
    %7 = torch.prim.ListConstruct %int0_10, %int1, %int2 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %true_11 = torch.constant.bool true
    %int6 = torch.constant.int 6
    %8 = torch.aten.sum.dim_IntList %6, %7, %true_11, %int6 : !torch.vtensor<[5,5,5],f32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[1,1,1],f32>
    %9 = torch.operator "torch.aten.divide.Tensor"(%8, %4) : (!torch.vtensor<[1,1,1],f32>, !torch.vtensor<[1,1,1],si64>) -> !torch.vtensor<[1,1,1],f32> 
    %int1_12 = torch.constant.int 1
    %10 = torch.operator "torch.aten.subtract.Tensor"(%arg0, %9, %int1_12) : (!torch.vtensor<[5,5,5],f32>, !torch.vtensor<[1,1,1],f32>, !torch.int) -> !torch.vtensor<[5,5,5],f32> 
    %11 = torch.aten.mul.Tensor %10, %10 : !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[5,5,5],f32> -> !torch.vtensor<[5,5,5],f32>
    %none_13 = torch.constant.none
    %none_14 = torch.constant.none
    %int11_15 = torch.constant.int 11
    %cpu_16 = torch.constant.device "cpu"
    %int0_17 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none_13, %none_14, %int11_15, %cpu_16, %int0_17 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_18 = torch.constant.none
    %12 = torch.aten.clone %3, %none_18 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %13 = torch.aten.where.self %1, %11, %12 : !torch.vtensor<[5,5,5],i1>, !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[5,5,5],f32>
    %int0_19 = torch.constant.int 0
    %int1_20 = torch.constant.int 1
    %int2_21 = torch.constant.int 2
    %14 = torch.prim.ListConstruct %int0_19, %int1_20, %int2_21 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %true_22 = torch.constant.bool true
    %int6_23 = torch.constant.int 6
    %15 = torch.aten.sum.dim_IntList %13, %14, %true_22, %int6_23 : !torch.vtensor<[5,5,5],f32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[1,1,1],f32>
    %16 = torch.operator "torch.aten.divide.Tensor"(%15, %4) : (!torch.vtensor<[1,1,1],f32>, !torch.vtensor<[1,1,1],si64>) -> !torch.vtensor<[1,1,1],f32> 
    %none_24 = torch.constant.none
    %none_25 = torch.constant.none
    %int6_26 = torch.constant.int 6
    %cpu_27 = torch.constant.device "cpu"
    %int0_28 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %16, %none_24, %none_25, %int6_26, %cpu_27, %int0_28 : !torch.vtensor<[1,1,1],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %16 : !torch.vtensor<[1,1,1],f32>
  }
  func.func @test_reduction_masked_var_cpu_float32_31(%arg0: !torch.vtensor<[5,5,5],f32>) -> !torch.vtensor<[1,1,1],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_5_5_1_torch.uint8_3> : tensor<5x5x1xui8>) : !torch.vtensor<[5,5,1],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[5,5,1],ui8>, !torch.int -> !torch.vtensor<[5,5,1],i1>
    %2 = torch.vtensor.literal(dense<0.000000e+00> : tensor<f32>) : !torch.vtensor<[],f32>
    %3 = torch.vtensor.literal(dense<0.000000e+00> : tensor<f32>) : !torch.vtensor<[],f32>
    %none = torch.constant.none
    %4 = torch.aten.clone %1, %none : !torch.vtensor<[5,5,1],i1>, !torch.none -> !torch.vtensor<[5,5,1],i1>
    %int5 = torch.constant.int 5
    %int5_0 = torch.constant.int 5
    %int5_1 = torch.constant.int 5
    %5 = torch.prim.ListConstruct %int5, %int5_0, %int5_1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %6 = torch.aten.expand %4, %5, %false : !torch.vtensor<[5,5,1],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[5,5,5],i1>
    %none_2 = torch.constant.none
    %none_3 = torch.constant.none
    %int11_4 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_2, %none_3, %int11_4, %cpu, %int0 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_5 = torch.constant.none
    %none_6 = torch.constant.none
    %int11_7 = torch.constant.int 11
    %cpu_8 = torch.constant.device "cpu"
    %int0_9 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_5, %none_6, %int11_7, %cpu_8, %int0_9 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_10 = torch.constant.none
    %true = torch.constant.bool true
    %none_11 = torch.constant.none
    %7 = torch.aten.sum.dim_IntList %6, %none_10, %true, %none_11 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1],si64>
    %none_12 = torch.constant.none
    %none_13 = torch.constant.none
    %int11_14 = torch.constant.int 11
    %cpu_15 = torch.constant.device "cpu"
    %int0_16 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_12, %none_13, %int11_14, %cpu_15, %int0_16 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_17 = torch.constant.none
    %8 = torch.aten.clone %2, %none_17 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %9 = torch.aten.where.self %6, %arg0, %8 : !torch.vtensor<[5,5,5],i1>, !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[5,5,5],f32>
    %int0_18 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %int2 = torch.constant.int 2
    %10 = torch.prim.ListConstruct %int0_18, %int1, %int2 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %true_19 = torch.constant.bool true
    %int6 = torch.constant.int 6
    %11 = torch.aten.sum.dim_IntList %9, %10, %true_19, %int6 : !torch.vtensor<[5,5,5],f32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[1,1,1],f32>
    %12 = torch.operator "torch.aten.divide.Tensor"(%11, %7) : (!torch.vtensor<[1,1,1],f32>, !torch.vtensor<[1,1,1],si64>) -> !torch.vtensor<[1,1,1],f32> 
    %int1_20 = torch.constant.int 1
    %13 = torch.operator "torch.aten.subtract.Tensor"(%arg0, %12, %int1_20) : (!torch.vtensor<[5,5,5],f32>, !torch.vtensor<[1,1,1],f32>, !torch.int) -> !torch.vtensor<[5,5,5],f32> 
    %14 = torch.aten.mul.Tensor %13, %13 : !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[5,5,5],f32> -> !torch.vtensor<[5,5,5],f32>
    %none_21 = torch.constant.none
    %none_22 = torch.constant.none
    %int11_23 = torch.constant.int 11
    %cpu_24 = torch.constant.device "cpu"
    %int0_25 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_21, %none_22, %int11_23, %cpu_24, %int0_25 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_26 = torch.constant.none
    %15 = torch.aten.clone %3, %none_26 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %16 = torch.aten.where.self %6, %14, %15 : !torch.vtensor<[5,5,5],i1>, !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[5,5,5],f32>
    %int0_27 = torch.constant.int 0
    %int1_28 = torch.constant.int 1
    %int2_29 = torch.constant.int 2
    %17 = torch.prim.ListConstruct %int0_27, %int1_28, %int2_29 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %true_30 = torch.constant.bool true
    %int6_31 = torch.constant.int 6
    %18 = torch.aten.sum.dim_IntList %16, %17, %true_30, %int6_31 : !torch.vtensor<[5,5,5],f32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[1,1,1],f32>
    %19 = torch.operator "torch.aten.divide.Tensor"(%18, %7) : (!torch.vtensor<[1,1,1],f32>, !torch.vtensor<[1,1,1],si64>) -> !torch.vtensor<[1,1,1],f32> 
    %none_32 = torch.constant.none
    %none_33 = torch.constant.none
    %int6_34 = torch.constant.int 6
    %cpu_35 = torch.constant.device "cpu"
    %int0_36 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %19, %none_32, %none_33, %int6_34, %cpu_35, %int0_36 : !torch.vtensor<[1,1,1],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %19 : !torch.vtensor<[1,1,1],f32>
  }
  func.func @test_reduction_masked_var_cpu_float32_32(%arg0: !torch.vtensor<[5,5,5],f32>) -> !torch.vtensor<[1,1,1],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_5_1_5_torch.uint8_3> : tensor<5x1x5xui8>) : !torch.vtensor<[5,1,5],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[5,1,5],ui8>, !torch.int -> !torch.vtensor<[5,1,5],i1>
    %2 = torch.vtensor.literal(dense<0.000000e+00> : tensor<f32>) : !torch.vtensor<[],f32>
    %3 = torch.vtensor.literal(dense<0.000000e+00> : tensor<f32>) : !torch.vtensor<[],f32>
    %none = torch.constant.none
    %4 = torch.aten.clone %1, %none : !torch.vtensor<[5,1,5],i1>, !torch.none -> !torch.vtensor<[5,1,5],i1>
    %int5 = torch.constant.int 5
    %int5_0 = torch.constant.int 5
    %int5_1 = torch.constant.int 5
    %5 = torch.prim.ListConstruct %int5, %int5_0, %int5_1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %6 = torch.aten.expand %4, %5, %false : !torch.vtensor<[5,1,5],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[5,5,5],i1>
    %none_2 = torch.constant.none
    %none_3 = torch.constant.none
    %int11_4 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_2, %none_3, %int11_4, %cpu, %int0 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_5 = torch.constant.none
    %none_6 = torch.constant.none
    %int11_7 = torch.constant.int 11
    %cpu_8 = torch.constant.device "cpu"
    %int0_9 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_5, %none_6, %int11_7, %cpu_8, %int0_9 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_10 = torch.constant.none
    %true = torch.constant.bool true
    %none_11 = torch.constant.none
    %7 = torch.aten.sum.dim_IntList %6, %none_10, %true, %none_11 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1],si64>
    %none_12 = torch.constant.none
    %none_13 = torch.constant.none
    %int11_14 = torch.constant.int 11
    %cpu_15 = torch.constant.device "cpu"
    %int0_16 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_12, %none_13, %int11_14, %cpu_15, %int0_16 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_17 = torch.constant.none
    %8 = torch.aten.clone %2, %none_17 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %9 = torch.aten.where.self %6, %arg0, %8 : !torch.vtensor<[5,5,5],i1>, !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[5,5,5],f32>
    %int0_18 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %int2 = torch.constant.int 2
    %10 = torch.prim.ListConstruct %int0_18, %int1, %int2 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %true_19 = torch.constant.bool true
    %int6 = torch.constant.int 6
    %11 = torch.aten.sum.dim_IntList %9, %10, %true_19, %int6 : !torch.vtensor<[5,5,5],f32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[1,1,1],f32>
    %12 = torch.operator "torch.aten.divide.Tensor"(%11, %7) : (!torch.vtensor<[1,1,1],f32>, !torch.vtensor<[1,1,1],si64>) -> !torch.vtensor<[1,1,1],f32> 
    %int1_20 = torch.constant.int 1
    %13 = torch.operator "torch.aten.subtract.Tensor"(%arg0, %12, %int1_20) : (!torch.vtensor<[5,5,5],f32>, !torch.vtensor<[1,1,1],f32>, !torch.int) -> !torch.vtensor<[5,5,5],f32> 
    %14 = torch.aten.mul.Tensor %13, %13 : !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[5,5,5],f32> -> !torch.vtensor<[5,5,5],f32>
    %none_21 = torch.constant.none
    %none_22 = torch.constant.none
    %int11_23 = torch.constant.int 11
    %cpu_24 = torch.constant.device "cpu"
    %int0_25 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_21, %none_22, %int11_23, %cpu_24, %int0_25 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_26 = torch.constant.none
    %15 = torch.aten.clone %3, %none_26 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %16 = torch.aten.where.self %6, %14, %15 : !torch.vtensor<[5,5,5],i1>, !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[5,5,5],f32>
    %int0_27 = torch.constant.int 0
    %int1_28 = torch.constant.int 1
    %int2_29 = torch.constant.int 2
    %17 = torch.prim.ListConstruct %int0_27, %int1_28, %int2_29 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %true_30 = torch.constant.bool true
    %int6_31 = torch.constant.int 6
    %18 = torch.aten.sum.dim_IntList %16, %17, %true_30, %int6_31 : !torch.vtensor<[5,5,5],f32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[1,1,1],f32>
    %19 = torch.operator "torch.aten.divide.Tensor"(%18, %7) : (!torch.vtensor<[1,1,1],f32>, !torch.vtensor<[1,1,1],si64>) -> !torch.vtensor<[1,1,1],f32> 
    %none_32 = torch.constant.none
    %none_33 = torch.constant.none
    %int6_34 = torch.constant.int 6
    %cpu_35 = torch.constant.device "cpu"
    %int0_36 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %19, %none_32, %none_33, %int6_34, %cpu_35, %int0_36 : !torch.vtensor<[1,1,1],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %19 : !torch.vtensor<[1,1,1],f32>
  }
  func.func @test_reduction_masked_var_cpu_float32_33(%arg0: !torch.vtensor<[5,5,5],f32>) -> !torch.vtensor<[1,1,1],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_1_5_5_torch.uint8_3> : tensor<1x5x5xui8>) : !torch.vtensor<[1,5,5],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[1,5,5],ui8>, !torch.int -> !torch.vtensor<[1,5,5],i1>
    %2 = torch.vtensor.literal(dense<0.000000e+00> : tensor<f32>) : !torch.vtensor<[],f32>
    %3 = torch.vtensor.literal(dense<0.000000e+00> : tensor<f32>) : !torch.vtensor<[],f32>
    %none = torch.constant.none
    %4 = torch.aten.clone %1, %none : !torch.vtensor<[1,5,5],i1>, !torch.none -> !torch.vtensor<[1,5,5],i1>
    %int5 = torch.constant.int 5
    %int5_0 = torch.constant.int 5
    %int5_1 = torch.constant.int 5
    %5 = torch.prim.ListConstruct %int5, %int5_0, %int5_1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %6 = torch.aten.expand %4, %5, %false : !torch.vtensor<[1,5,5],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[5,5,5],i1>
    %none_2 = torch.constant.none
    %none_3 = torch.constant.none
    %int11_4 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_2, %none_3, %int11_4, %cpu, %int0 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_5 = torch.constant.none
    %none_6 = torch.constant.none
    %int11_7 = torch.constant.int 11
    %cpu_8 = torch.constant.device "cpu"
    %int0_9 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_5, %none_6, %int11_7, %cpu_8, %int0_9 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_10 = torch.constant.none
    %true = torch.constant.bool true
    %none_11 = torch.constant.none
    %7 = torch.aten.sum.dim_IntList %6, %none_10, %true, %none_11 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1],si64>
    %none_12 = torch.constant.none
    %none_13 = torch.constant.none
    %int11_14 = torch.constant.int 11
    %cpu_15 = torch.constant.device "cpu"
    %int0_16 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_12, %none_13, %int11_14, %cpu_15, %int0_16 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_17 = torch.constant.none
    %8 = torch.aten.clone %2, %none_17 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %9 = torch.aten.where.self %6, %arg0, %8 : !torch.vtensor<[5,5,5],i1>, !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[5,5,5],f32>
    %int0_18 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %int2 = torch.constant.int 2
    %10 = torch.prim.ListConstruct %int0_18, %int1, %int2 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %true_19 = torch.constant.bool true
    %int6 = torch.constant.int 6
    %11 = torch.aten.sum.dim_IntList %9, %10, %true_19, %int6 : !torch.vtensor<[5,5,5],f32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[1,1,1],f32>
    %12 = torch.operator "torch.aten.divide.Tensor"(%11, %7) : (!torch.vtensor<[1,1,1],f32>, !torch.vtensor<[1,1,1],si64>) -> !torch.vtensor<[1,1,1],f32> 
    %int1_20 = torch.constant.int 1
    %13 = torch.operator "torch.aten.subtract.Tensor"(%arg0, %12, %int1_20) : (!torch.vtensor<[5,5,5],f32>, !torch.vtensor<[1,1,1],f32>, !torch.int) -> !torch.vtensor<[5,5,5],f32> 
    %14 = torch.aten.mul.Tensor %13, %13 : !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[5,5,5],f32> -> !torch.vtensor<[5,5,5],f32>
    %none_21 = torch.constant.none
    %none_22 = torch.constant.none
    %int11_23 = torch.constant.int 11
    %cpu_24 = torch.constant.device "cpu"
    %int0_25 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_21, %none_22, %int11_23, %cpu_24, %int0_25 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_26 = torch.constant.none
    %15 = torch.aten.clone %3, %none_26 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %16 = torch.aten.where.self %6, %14, %15 : !torch.vtensor<[5,5,5],i1>, !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[5,5,5],f32>
    %int0_27 = torch.constant.int 0
    %int1_28 = torch.constant.int 1
    %int2_29 = torch.constant.int 2
    %17 = torch.prim.ListConstruct %int0_27, %int1_28, %int2_29 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %true_30 = torch.constant.bool true
    %int6_31 = torch.constant.int 6
    %18 = torch.aten.sum.dim_IntList %16, %17, %true_30, %int6_31 : !torch.vtensor<[5,5,5],f32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[1,1,1],f32>
    %19 = torch.operator "torch.aten.divide.Tensor"(%18, %7) : (!torch.vtensor<[1,1,1],f32>, !torch.vtensor<[1,1,1],si64>) -> !torch.vtensor<[1,1,1],f32> 
    %none_32 = torch.constant.none
    %none_33 = torch.constant.none
    %int6_34 = torch.constant.int 6
    %cpu_35 = torch.constant.device "cpu"
    %int0_36 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %19, %none_32, %none_33, %int6_34, %cpu_35, %int0_36 : !torch.vtensor<[1,1,1],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %19 : !torch.vtensor<[1,1,1],f32>
  }
  func.func @test_reduction_masked_var_cpu_float32_34(%arg0: !torch.vtensor<[5,5,5],f32>) -> !torch.vtensor<[1,1,1],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_5_5_torch.uint8_3> : tensor<5x5xui8>) : !torch.vtensor<[5,5],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[5,5],ui8>, !torch.int -> !torch.vtensor<[5,5],i1>
    %2 = torch.vtensor.literal(dense<0.000000e+00> : tensor<f32>) : !torch.vtensor<[],f32>
    %3 = torch.vtensor.literal(dense<0.000000e+00> : tensor<f32>) : !torch.vtensor<[],f32>
    %none = torch.constant.none
    %4 = torch.aten.clone %1, %none : !torch.vtensor<[5,5],i1>, !torch.none -> !torch.vtensor<[5,5],i1>
    %int5 = torch.constant.int 5
    %int5_0 = torch.constant.int 5
    %int5_1 = torch.constant.int 5
    %5 = torch.prim.ListConstruct %int5, %int5_0, %int5_1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %6 = torch.aten.expand %4, %5, %false : !torch.vtensor<[5,5],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[5,5,5],i1>
    %none_2 = torch.constant.none
    %none_3 = torch.constant.none
    %int11_4 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_2, %none_3, %int11_4, %cpu, %int0 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_5 = torch.constant.none
    %none_6 = torch.constant.none
    %int11_7 = torch.constant.int 11
    %cpu_8 = torch.constant.device "cpu"
    %int0_9 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_5, %none_6, %int11_7, %cpu_8, %int0_9 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_10 = torch.constant.none
    %true = torch.constant.bool true
    %none_11 = torch.constant.none
    %7 = torch.aten.sum.dim_IntList %6, %none_10, %true, %none_11 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1],si64>
    %none_12 = torch.constant.none
    %none_13 = torch.constant.none
    %int11_14 = torch.constant.int 11
    %cpu_15 = torch.constant.device "cpu"
    %int0_16 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_12, %none_13, %int11_14, %cpu_15, %int0_16 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_17 = torch.constant.none
    %8 = torch.aten.clone %2, %none_17 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %9 = torch.aten.where.self %6, %arg0, %8 : !torch.vtensor<[5,5,5],i1>, !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[5,5,5],f32>
    %int0_18 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %int2 = torch.constant.int 2
    %10 = torch.prim.ListConstruct %int0_18, %int1, %int2 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %true_19 = torch.constant.bool true
    %int6 = torch.constant.int 6
    %11 = torch.aten.sum.dim_IntList %9, %10, %true_19, %int6 : !torch.vtensor<[5,5,5],f32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[1,1,1],f32>
    %12 = torch.operator "torch.aten.divide.Tensor"(%11, %7) : (!torch.vtensor<[1,1,1],f32>, !torch.vtensor<[1,1,1],si64>) -> !torch.vtensor<[1,1,1],f32> 
    %int1_20 = torch.constant.int 1
    %13 = torch.operator "torch.aten.subtract.Tensor"(%arg0, %12, %int1_20) : (!torch.vtensor<[5,5,5],f32>, !torch.vtensor<[1,1,1],f32>, !torch.int) -> !torch.vtensor<[5,5,5],f32> 
    %14 = torch.aten.mul.Tensor %13, %13 : !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[5,5,5],f32> -> !torch.vtensor<[5,5,5],f32>
    %none_21 = torch.constant.none
    %none_22 = torch.constant.none
    %int11_23 = torch.constant.int 11
    %cpu_24 = torch.constant.device "cpu"
    %int0_25 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_21, %none_22, %int11_23, %cpu_24, %int0_25 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_26 = torch.constant.none
    %15 = torch.aten.clone %3, %none_26 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %16 = torch.aten.where.self %6, %14, %15 : !torch.vtensor<[5,5,5],i1>, !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[5,5,5],f32>
    %int0_27 = torch.constant.int 0
    %int1_28 = torch.constant.int 1
    %int2_29 = torch.constant.int 2
    %17 = torch.prim.ListConstruct %int0_27, %int1_28, %int2_29 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %true_30 = torch.constant.bool true
    %int6_31 = torch.constant.int 6
    %18 = torch.aten.sum.dim_IntList %16, %17, %true_30, %int6_31 : !torch.vtensor<[5,5,5],f32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[1,1,1],f32>
    %19 = torch.operator "torch.aten.divide.Tensor"(%18, %7) : (!torch.vtensor<[1,1,1],f32>, !torch.vtensor<[1,1,1],si64>) -> !torch.vtensor<[1,1,1],f32> 
    %none_32 = torch.constant.none
    %none_33 = torch.constant.none
    %int6_34 = torch.constant.int 6
    %cpu_35 = torch.constant.device "cpu"
    %int0_36 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %19, %none_32, %none_33, %int6_34, %cpu_35, %int0_36 : !torch.vtensor<[1,1,1],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %19 : !torch.vtensor<[1,1,1],f32>
  }
  func.func @test_reduction_masked_var_cpu_float32_35(%arg0: !torch.vtensor<[5,5,5],f32>) -> !torch.vtensor<[1,1,1],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_5_torch.uint8_5> : tensor<5xui8>) : !torch.vtensor<[5],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[5],ui8>, !torch.int -> !torch.vtensor<[5],i1>
    %2 = torch.vtensor.literal(dense<0.000000e+00> : tensor<f32>) : !torch.vtensor<[],f32>
    %3 = torch.vtensor.literal(dense<0.000000e+00> : tensor<f32>) : !torch.vtensor<[],f32>
    %none = torch.constant.none
    %4 = torch.aten.clone %1, %none : !torch.vtensor<[5],i1>, !torch.none -> !torch.vtensor<[5],i1>
    %int5 = torch.constant.int 5
    %int5_0 = torch.constant.int 5
    %int5_1 = torch.constant.int 5
    %5 = torch.prim.ListConstruct %int5, %int5_0, %int5_1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %6 = torch.aten.expand %4, %5, %false : !torch.vtensor<[5],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[5,5,5],i1>
    %none_2 = torch.constant.none
    %none_3 = torch.constant.none
    %int11_4 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_2, %none_3, %int11_4, %cpu, %int0 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_5 = torch.constant.none
    %none_6 = torch.constant.none
    %int11_7 = torch.constant.int 11
    %cpu_8 = torch.constant.device "cpu"
    %int0_9 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_5, %none_6, %int11_7, %cpu_8, %int0_9 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_10 = torch.constant.none
    %true = torch.constant.bool true
    %none_11 = torch.constant.none
    %7 = torch.aten.sum.dim_IntList %6, %none_10, %true, %none_11 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.bool, !torch.none -> !torch.vtensor<[1,1,1],si64>
    %none_12 = torch.constant.none
    %none_13 = torch.constant.none
    %int11_14 = torch.constant.int 11
    %cpu_15 = torch.constant.device "cpu"
    %int0_16 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_12, %none_13, %int11_14, %cpu_15, %int0_16 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_17 = torch.constant.none
    %8 = torch.aten.clone %2, %none_17 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %9 = torch.aten.where.self %6, %arg0, %8 : !torch.vtensor<[5,5,5],i1>, !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[5,5,5],f32>
    %int0_18 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %int2 = torch.constant.int 2
    %10 = torch.prim.ListConstruct %int0_18, %int1, %int2 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %true_19 = torch.constant.bool true
    %int6 = torch.constant.int 6
    %11 = torch.aten.sum.dim_IntList %9, %10, %true_19, %int6 : !torch.vtensor<[5,5,5],f32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[1,1,1],f32>
    %12 = torch.operator "torch.aten.divide.Tensor"(%11, %7) : (!torch.vtensor<[1,1,1],f32>, !torch.vtensor<[1,1,1],si64>) -> !torch.vtensor<[1,1,1],f32> 
    %int1_20 = torch.constant.int 1
    %13 = torch.operator "torch.aten.subtract.Tensor"(%arg0, %12, %int1_20) : (!torch.vtensor<[5,5,5],f32>, !torch.vtensor<[1,1,1],f32>, !torch.int) -> !torch.vtensor<[5,5,5],f32> 
    %14 = torch.aten.mul.Tensor %13, %13 : !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[5,5,5],f32> -> !torch.vtensor<[5,5,5],f32>
    %none_21 = torch.constant.none
    %none_22 = torch.constant.none
    %int11_23 = torch.constant.int 11
    %cpu_24 = torch.constant.device "cpu"
    %int0_25 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_21, %none_22, %int11_23, %cpu_24, %int0_25 : !torch.vtensor<[5,5,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_26 = torch.constant.none
    %15 = torch.aten.clone %3, %none_26 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %16 = torch.aten.where.self %6, %14, %15 : !torch.vtensor<[5,5,5],i1>, !torch.vtensor<[5,5,5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[5,5,5],f32>
    %int0_27 = torch.constant.int 0
    %int1_28 = torch.constant.int 1
    %int2_29 = torch.constant.int 2
    %17 = torch.prim.ListConstruct %int0_27, %int1_28, %int2_29 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %true_30 = torch.constant.bool true
    %int6_31 = torch.constant.int 6
    %18 = torch.aten.sum.dim_IntList %16, %17, %true_30, %int6_31 : !torch.vtensor<[5,5,5],f32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[1,1,1],f32>
    %19 = torch.operator "torch.aten.divide.Tensor"(%18, %7) : (!torch.vtensor<[1,1,1],f32>, !torch.vtensor<[1,1,1],si64>) -> !torch.vtensor<[1,1,1],f32> 
    %none_32 = torch.constant.none
    %none_33 = torch.constant.none
    %int6_34 = torch.constant.int 6
    %cpu_35 = torch.constant.device "cpu"
    %int0_36 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %19, %none_32, %none_33, %int6_34, %cpu_35, %int0_36 : !torch.vtensor<[1,1,1],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %19 : !torch.vtensor<[1,1,1],f32>
  }
  func.func @test_reduction_masked_var_cpu_float32_36(%arg0: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[4,5],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %int3 = torch.constant.int 3
    %int4 = torch.constant.int 4
    %int5 = torch.constant.int 5
    %0 = torch.prim.ListConstruct %int3, %int4, %int5 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %int4_0 = torch.constant.int 4
    %none = torch.constant.none
    %cpu = torch.constant.device "cpu"
    %false = torch.constant.bool false
    %1 = torch.aten.ones %0, %int4_0, %none, %cpu, %false : !torch.list<int>, !torch.int, !torch.none, !torch.Device, !torch.bool -> !torch.vtensor<[3,4,5],si64>
    %int0 = torch.constant.int 0
    %2 = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %int4_1 = torch.constant.int 4
    %3 = torch.aten.sum.dim_IntList %1, %2, %true, %int4_1 : !torch.vtensor<[3,4,5],si64>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[1,4,5],si64>
    %int0_2 = torch.constant.int 0
    %4 = torch.prim.ListConstruct %int0_2 : (!torch.int) -> !torch.list<int>
    %true_3 = torch.constant.bool true
    %int6 = torch.constant.int 6
    %5 = torch.aten.sum.dim_IntList %arg0, %4, %true_3, %int6 : !torch.vtensor<[3,4,5],f32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[1,4,5],f32>
    %6 = torch.operator "torch.aten.divide.Tensor"(%5, %3) : (!torch.vtensor<[1,4,5],f32>, !torch.vtensor<[1,4,5],si64>) -> !torch.vtensor<[1,4,5],f32> 
    %int1 = torch.constant.int 1
    %7 = torch.operator "torch.aten.subtract.Tensor"(%arg0, %6, %int1) : (!torch.vtensor<[3,4,5],f32>, !torch.vtensor<[1,4,5],f32>, !torch.int) -> !torch.vtensor<[3,4,5],f32> 
    %8 = torch.aten.mul.Tensor %7, %7 : !torch.vtensor<[3,4,5],f32>, !torch.vtensor<[3,4,5],f32> -> !torch.vtensor<[3,4,5],f32>
    %int0_4 = torch.constant.int 0
    %9 = torch.prim.ListConstruct %int0_4 : (!torch.int) -> !torch.list<int>
    %false_5 = torch.constant.bool false
    %int6_6 = torch.constant.int 6
    %10 = torch.aten.sum.dim_IntList %8, %9, %false_5, %int6_6 : !torch.vtensor<[3,4,5],f32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[4,5],f32>
    %int4_7 = torch.constant.int 4
    %int5_8 = torch.constant.int 5
    %11 = torch.prim.ListConstruct %int4_7, %int5_8 : (!torch.int, !torch.int) -> !torch.list<int>
    %12 = torch.aten.view %3, %11 : !torch.vtensor<[1,4,5],si64>, !torch.list<int> -> !torch.vtensor<[4,5],si64>
    %none_9 = torch.constant.none
    %none_10 = torch.constant.none
    %int4_11 = torch.constant.int 4
    %cpu_12 = torch.constant.device "cpu"
    %int0_13 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %12, %none_9, %none_10, %int4_11, %cpu_12, %int0_13 : !torch.vtensor<[4,5],si64>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int6_14 = torch.constant.int 6
    %13 = torch.prims.convert_element_type %12, %int6_14 : !torch.vtensor<[4,5],si64>, !torch.int -> !torch.vtensor<[4,5],f32>
    %float1.000000e00 = torch.constant.float 1.000000e+00
    %14 = torch.prim.NumToTensor.Scalar %float1.000000e00 : !torch.float -> !torch.vtensor<[],f32>
    %int1_15 = torch.constant.int 1
    %15 = torch.operator "torch.aten.subtract.Tensor"(%13, %14, %int1_15) : (!torch.vtensor<[4,5],f32>, !torch.vtensor<[],f32>, !torch.int) -> !torch.vtensor<[4,5],f32> 
    %16 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %none_16 = torch.constant.none
    %none_17 = torch.constant.none
    %none_18 = torch.constant.none
    %false_19 = torch.constant.bool false
    %17 = torch.aten.new_zeros %15, %16, %none_16, %none_17, %none_18, %false_19 : !torch.vtensor<[4,5],f32>, !torch.list<int>, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[],f32>
    %18 = torch.aten.maximum %15, %17 : !torch.vtensor<[4,5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[4,5],f32>
    %19 = torch.operator "torch.aten.divide.Tensor"(%10, %18) : (!torch.vtensor<[4,5],f32>, !torch.vtensor<[4,5],f32>) -> !torch.vtensor<[4,5],f32> 
    %none_20 = torch.constant.none
    %none_21 = torch.constant.none
    %int6_22 = torch.constant.int 6
    %cpu_23 = torch.constant.device "cpu"
    %int0_24 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %19, %none_20, %none_21, %int6_22, %cpu_23, %int0_24 : !torch.vtensor<[4,5],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %19 : !torch.vtensor<[4,5],f32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_5_5_5_torch.uint8: "0x010000000101000100000001010101010000010000000000000000000100010100000100000100010000000101010001010001000100010101010001000101000001000001010100000001010101010101000100010001000000000000000001010100010000010100010001010001000000010100000000000101010100010001",
      torch_tensor_5_5_1_torch.uint8: "0x0100000001010001000000010101010100000100000000000000000001",
      torch_tensor_5_1_5_torch.uint8: "0x0100000001010001000000010101010100000100000000000000000001",
      torch_tensor_1_5_5_torch.uint8: "0x0100000001010001000000010101010100000100000000000000000001",
      torch_tensor_5_5_torch.uint8: "0x0100000001010001000000010101010100000100000000000000000001",
      torch_tensor_5_torch.uint8: "0x010000000101000100",
      torch_tensor_5_torch.uint8_1: "0x010000000101000100",
      torch_tensor_5_torch.uint8_2: "0x010000000101000100",
      torch_tensor_5_5_5_torch.uint8_1: "0x010000000101000100000001010101010000010000000000000000000100010100000100000100010000000101010001010001000100010101010001000101000001000001010100000001010101010101000100010001000000000000000001010100010000010100010001010001000000010100000000000101010100010001",
      torch_tensor_5_5_1_torch.uint8_1: "0x0100000001010001000000010101010100000100000000000000000001",
      torch_tensor_5_1_5_torch.uint8_1: "0x0100000001010001000000010101010100000100000000000000000001",
      torch_tensor_1_5_5_torch.uint8_1: "0x0100000001010001000000010101010100000100000000000000000001",
      torch_tensor_5_5_torch.uint8_1: "0x0100000001010001000000010101010100000100000000000000000001",
      torch_tensor_5_torch.uint8_3: "0x010000000101000100",
      torch_tensor_5_5_5_torch.uint8_2: "0x010000000101000100000001010101010000010000000000000000000100010100000100000100010000000101010001010001000100010101010001000101000001000001010100000001010101010101000100010001000000000000000001010100010000010100010001010001000000010100000000000101010100010001",
      torch_tensor_5_5_1_torch.uint8_2: "0x0100000001010001000000010101010100000100000000000000000001",
      torch_tensor_5_1_5_torch.uint8_2: "0x0100000001010001000000010101010100000100000000000000000001",
      torch_tensor_1_5_5_torch.uint8_2: "0x0100000001010001000000010101010100000100000000000000000001",
      torch_tensor_5_5_torch.uint8_2: "0x0100000001010001000000010101010100000100000000000000000001",
      torch_tensor_5_torch.uint8_4: "0x010000000101000100",
      torch_tensor_5_5_5_torch.uint8_3: "0x010000000101000100000001010101010000010000000000000000000100010100000100000100010000000101010001010001000100010101010001000101000001000001010100000001010101010101000100010001000000000000000001010100010000010100010001010001000000010100000000000101010100010001",
      torch_tensor_5_5_1_torch.uint8_3: "0x0100000001010001000000010101010100000100000000000000000001",
      torch_tensor_5_1_5_torch.uint8_3: "0x0100000001010001000000010101010100000100000000000000000001",
      torch_tensor_1_5_5_torch.uint8_3: "0x0100000001010001000000010101010100000100000000000000000001",
      torch_tensor_5_5_torch.uint8_3: "0x0100000001010001000000010101010100000100000000000000000001",
      torch_tensor_5_torch.uint8_5: "0x010000000101000100"
    }
  }
#-}
