module @module {
  func.func @test_reduction_masked_prod_cpu_bfloat16_0(%arg0: !torch.vtensor<[],bf16>) -> !torch.vtensor<[],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %arg0, %none, %none_0, %int15, %cpu, %int0 : !torch.vtensor<[],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %arg0 : !torch.vtensor<[],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_1(%arg0: !torch.vtensor<[],bf16>) -> !torch.vtensor<[],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense<1> : tensor<ui8>) : !torch.vtensor<[],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[],ui8>, !torch.int -> !torch.vtensor<[],i1>
    %2 = torch.vtensor.literal(dense<1.000000e+00> : tensor<bf16>) : !torch.vtensor<[],bf16>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],bf16>, !torch.none -> !torch.vtensor<[],bf16>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[],i1>, !torch.vtensor<[],bf16>, !torch.vtensor<[],bf16> -> !torch.vtensor<[],bf16>
    %none_3 = torch.constant.none
    %none_4 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu_5 = torch.constant.device "cpu"
    %int0_6 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %4, %none_3, %none_4, %int15, %cpu_5, %int0_6 : !torch.vtensor<[],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %4 : !torch.vtensor<[],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_2(%arg0: !torch.vtensor<[],bf16>) -> !torch.vtensor<[],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %arg0, %none, %none_0, %int15, %cpu, %int0 : !torch.vtensor<[],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int0_1 = torch.constant.int 0
    %true = torch.constant.bool true
    %none_2 = torch.constant.none
    %0 = torch.aten.prod.dim_int %arg0, %int0_1, %true, %none_2 : !torch.vtensor<[],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[],bf16>
    return %0 : !torch.vtensor<[],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_3(%arg0: !torch.vtensor<[],bf16>) -> !torch.vtensor<[],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense<1> : tensor<ui8>) : !torch.vtensor<[],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[],ui8>, !torch.int -> !torch.vtensor<[],i1>
    %2 = torch.vtensor.literal(dense<1.000000e+00> : tensor<bf16>) : !torch.vtensor<[],bf16>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],bf16>, !torch.none -> !torch.vtensor<[],bf16>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[],i1>, !torch.vtensor<[],bf16>, !torch.vtensor<[],bf16> -> !torch.vtensor<[],bf16>
    %none_3 = torch.constant.none
    %none_4 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu_5 = torch.constant.device "cpu"
    %int0_6 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %4, %none_3, %none_4, %int15, %cpu_5, %int0_6 : !torch.vtensor<[],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int0_7 = torch.constant.int 0
    %true = torch.constant.bool true
    %none_8 = torch.constant.none
    %5 = torch.aten.prod.dim_int %4, %int0_7, %true, %none_8 : !torch.vtensor<[],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[],bf16>
    return %5 : !torch.vtensor<[],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_4(%arg0: !torch.vtensor<[],bf16>) -> !torch.vtensor<[],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %arg0, %none, %none_0, %int15, %cpu, %int0 : !torch.vtensor<[],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int0_1 = torch.constant.int 0
    %false = torch.constant.bool false
    %none_2 = torch.constant.none
    %0 = torch.aten.prod.dim_int %arg0, %int0_1, %false, %none_2 : !torch.vtensor<[],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[],bf16>
    return %0 : !torch.vtensor<[],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_5(%arg0: !torch.vtensor<[],bf16>) -> !torch.vtensor<[],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense<1> : tensor<ui8>) : !torch.vtensor<[],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[],ui8>, !torch.int -> !torch.vtensor<[],i1>
    %2 = torch.vtensor.literal(dense<1.000000e+00> : tensor<bf16>) : !torch.vtensor<[],bf16>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],bf16>, !torch.none -> !torch.vtensor<[],bf16>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[],i1>, !torch.vtensor<[],bf16>, !torch.vtensor<[],bf16> -> !torch.vtensor<[],bf16>
    %none_3 = torch.constant.none
    %none_4 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu_5 = torch.constant.device "cpu"
    %int0_6 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %4, %none_3, %none_4, %int15, %cpu_5, %int0_6 : !torch.vtensor<[],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int0_7 = torch.constant.int 0
    %false = torch.constant.bool false
    %none_8 = torch.constant.none
    %5 = torch.aten.prod.dim_int %4, %int0_7, %false, %none_8 : !torch.vtensor<[],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[],bf16>
    return %5 : !torch.vtensor<[],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_6(%arg0: !torch.vtensor<[],bf16>) -> !torch.vtensor<[],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %arg0, %none, %none_0, %int15, %cpu, %int0 : !torch.vtensor<[],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %arg0 : !torch.vtensor<[],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_7(%arg0: !torch.vtensor<[],bf16>) -> !torch.vtensor<[],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense<1> : tensor<ui8>) : !torch.vtensor<[],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[],ui8>, !torch.int -> !torch.vtensor<[],i1>
    %2 = torch.vtensor.literal(dense<1.000000e+00> : tensor<bf16>) : !torch.vtensor<[],bf16>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],bf16>, !torch.none -> !torch.vtensor<[],bf16>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[],i1>, !torch.vtensor<[],bf16>, !torch.vtensor<[],bf16> -> !torch.vtensor<[],bf16>
    %none_3 = torch.constant.none
    %none_4 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu_5 = torch.constant.device "cpu"
    %int0_6 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %4, %none_3, %none_4, %int15, %cpu_5, %int0_6 : !torch.vtensor<[],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %4 : !torch.vtensor<[],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_8(%arg0: !torch.vtensor<[2],bf16>) -> !torch.vtensor<[],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %arg0, %none, %none_0, %int15, %cpu, %int0 : !torch.vtensor<[2],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int0_1 = torch.constant.int 0
    %false = torch.constant.bool false
    %none_2 = torch.constant.none
    %0 = torch.aten.prod.dim_int %arg0, %int0_1, %false, %none_2 : !torch.vtensor<[2],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[],bf16>
    return %0 : !torch.vtensor<[],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_9(%arg0: !torch.vtensor<[2],bf16>) -> !torch.vtensor<[],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_2_torch.uint8> : tensor<2xui8>) : !torch.vtensor<[2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[2],ui8>, !torch.int -> !torch.vtensor<[2],i1>
    %2 = torch.vtensor.literal(dense<1.000000e+00> : tensor<bf16>) : !torch.vtensor<[],bf16>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],bf16>, !torch.none -> !torch.vtensor<[],bf16>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[2],i1>, !torch.vtensor<[2],bf16>, !torch.vtensor<[],bf16> -> !torch.vtensor<[2],bf16>
    %none_3 = torch.constant.none
    %none_4 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu_5 = torch.constant.device "cpu"
    %int0_6 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %4, %none_3, %none_4, %int15, %cpu_5, %int0_6 : !torch.vtensor<[2],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int0_7 = torch.constant.int 0
    %false = torch.constant.bool false
    %none_8 = torch.constant.none
    %5 = torch.aten.prod.dim_int %4, %int0_7, %false, %none_8 : !torch.vtensor<[2],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[],bf16>
    return %5 : !torch.vtensor<[],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_10(%arg0: !torch.vtensor<[2],bf16>) -> !torch.vtensor<[1],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %arg0, %none, %none_0, %int15, %cpu, %int0 : !torch.vtensor<[2],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int0_1 = torch.constant.int 0
    %true = torch.constant.bool true
    %none_2 = torch.constant.none
    %0 = torch.aten.prod.dim_int %arg0, %int0_1, %true, %none_2 : !torch.vtensor<[2],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[1],bf16>
    return %0 : !torch.vtensor<[1],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_11(%arg0: !torch.vtensor<[2],bf16>) -> !torch.vtensor<[1],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_2_torch.uint8_1> : tensor<2xui8>) : !torch.vtensor<[2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[2],ui8>, !torch.int -> !torch.vtensor<[2],i1>
    %2 = torch.vtensor.literal(dense<1.000000e+00> : tensor<bf16>) : !torch.vtensor<[],bf16>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],bf16>, !torch.none -> !torch.vtensor<[],bf16>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[2],i1>, !torch.vtensor<[2],bf16>, !torch.vtensor<[],bf16> -> !torch.vtensor<[2],bf16>
    %none_3 = torch.constant.none
    %none_4 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu_5 = torch.constant.device "cpu"
    %int0_6 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %4, %none_3, %none_4, %int15, %cpu_5, %int0_6 : !torch.vtensor<[2],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int0_7 = torch.constant.int 0
    %true = torch.constant.bool true
    %none_8 = torch.constant.none
    %5 = torch.aten.prod.dim_int %4, %int0_7, %true, %none_8 : !torch.vtensor<[2],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[1],bf16>
    return %5 : !torch.vtensor<[1],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_12(%arg0: !torch.vtensor<[2],bf16>) -> !torch.vtensor<[],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %arg0, %none, %none_0, %int15, %cpu, %int0 : !torch.vtensor<[2],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int0_1 = torch.constant.int 0
    %false = torch.constant.bool false
    %none_2 = torch.constant.none
    %0 = torch.aten.prod.dim_int %arg0, %int0_1, %false, %none_2 : !torch.vtensor<[2],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[],bf16>
    return %0 : !torch.vtensor<[],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_13(%arg0: !torch.vtensor<[2],bf16>) -> !torch.vtensor<[],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_2_torch.uint8_2> : tensor<2xui8>) : !torch.vtensor<[2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[2],ui8>, !torch.int -> !torch.vtensor<[2],i1>
    %2 = torch.vtensor.literal(dense<1.000000e+00> : tensor<bf16>) : !torch.vtensor<[],bf16>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],bf16>, !torch.none -> !torch.vtensor<[],bf16>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[2],i1>, !torch.vtensor<[2],bf16>, !torch.vtensor<[],bf16> -> !torch.vtensor<[2],bf16>
    %none_3 = torch.constant.none
    %none_4 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu_5 = torch.constant.device "cpu"
    %int0_6 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %4, %none_3, %none_4, %int15, %cpu_5, %int0_6 : !torch.vtensor<[2],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int0_7 = torch.constant.int 0
    %false = torch.constant.bool false
    %none_8 = torch.constant.none
    %5 = torch.aten.prod.dim_int %4, %int0_7, %false, %none_8 : !torch.vtensor<[2],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[],bf16>
    return %5 : !torch.vtensor<[],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_14(%arg0: !torch.vtensor<[2],bf16>) -> !torch.vtensor<[],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %arg0, %none, %none_0, %int15, %cpu, %int0 : !torch.vtensor<[2],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int0_1 = torch.constant.int 0
    %false = torch.constant.bool false
    %none_2 = torch.constant.none
    %0 = torch.aten.prod.dim_int %arg0, %int0_1, %false, %none_2 : !torch.vtensor<[2],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[],bf16>
    return %0 : !torch.vtensor<[],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_15(%arg0: !torch.vtensor<[2],bf16>) -> !torch.vtensor<[],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_2_torch.uint8_3> : tensor<2xui8>) : !torch.vtensor<[2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[2],ui8>, !torch.int -> !torch.vtensor<[2],i1>
    %2 = torch.vtensor.literal(dense<1.000000e+00> : tensor<bf16>) : !torch.vtensor<[],bf16>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],bf16>, !torch.none -> !torch.vtensor<[],bf16>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[2],i1>, !torch.vtensor<[2],bf16>, !torch.vtensor<[],bf16> -> !torch.vtensor<[2],bf16>
    %none_3 = torch.constant.none
    %none_4 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu_5 = torch.constant.device "cpu"
    %int0_6 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %4, %none_3, %none_4, %int15, %cpu_5, %int0_6 : !torch.vtensor<[2],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int0_7 = torch.constant.int 0
    %false = torch.constant.bool false
    %none_8 = torch.constant.none
    %5 = torch.aten.prod.dim_int %4, %int0_7, %false, %none_8 : !torch.vtensor<[2],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[],bf16>
    return %5 : !torch.vtensor<[],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_16(%arg0: !torch.vtensor<[3,5],bf16>) -> !torch.vtensor<[],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %arg0, %none, %none_0, %int15, %cpu, %int0 : !torch.vtensor<[3,5],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int1 = torch.constant.int 1
    %false = torch.constant.bool false
    %none_1 = torch.constant.none
    %0 = torch.aten.prod.dim_int %arg0, %int1, %false, %none_1 : !torch.vtensor<[3,5],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3],bf16>
    %int0_2 = torch.constant.int 0
    %false_3 = torch.constant.bool false
    %none_4 = torch.constant.none
    %1 = torch.aten.prod.dim_int %0, %int0_2, %false_3, %none_4 : !torch.vtensor<[3],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[],bf16>
    return %1 : !torch.vtensor<[],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_17(%arg0: !torch.vtensor<[3,5],bf16>) -> !torch.vtensor<[],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_5_torch.uint8> : tensor<3x5xui8>) : !torch.vtensor<[3,5],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,5],ui8>, !torch.int -> !torch.vtensor<[3,5],i1>
    %2 = torch.vtensor.literal(dense<1.000000e+00> : tensor<bf16>) : !torch.vtensor<[],bf16>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[3,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],bf16>, !torch.none -> !torch.vtensor<[],bf16>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[3,5],i1>, !torch.vtensor<[3,5],bf16>, !torch.vtensor<[],bf16> -> !torch.vtensor<[3,5],bf16>
    %none_3 = torch.constant.none
    %none_4 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu_5 = torch.constant.device "cpu"
    %int0_6 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %4, %none_3, %none_4, %int15, %cpu_5, %int0_6 : !torch.vtensor<[3,5],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int1 = torch.constant.int 1
    %false = torch.constant.bool false
    %none_7 = torch.constant.none
    %5 = torch.aten.prod.dim_int %4, %int1, %false, %none_7 : !torch.vtensor<[3,5],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3],bf16>
    %int0_8 = torch.constant.int 0
    %false_9 = torch.constant.bool false
    %none_10 = torch.constant.none
    %6 = torch.aten.prod.dim_int %5, %int0_8, %false_9, %none_10 : !torch.vtensor<[3],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[],bf16>
    return %6 : !torch.vtensor<[],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_18(%arg0: !torch.vtensor<[3,5],bf16>) -> !torch.vtensor<[],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_5_torch.uint8_1> : tensor<3x5xui8>) : !torch.vtensor<[3,5],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,5],ui8>, !torch.int -> !torch.vtensor<[3,5],i1>
    %2 = torch.vtensor.literal(dense<1.000000e+00> : tensor<bf16>) : !torch.vtensor<[],bf16>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[3,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],bf16>, !torch.none -> !torch.vtensor<[],bf16>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[3,5],i1>, !torch.vtensor<[3,5],bf16>, !torch.vtensor<[],bf16> -> !torch.vtensor<[3,5],bf16>
    %none_3 = torch.constant.none
    %none_4 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu_5 = torch.constant.device "cpu"
    %int0_6 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %4, %none_3, %none_4, %int15, %cpu_5, %int0_6 : !torch.vtensor<[3,5],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int1 = torch.constant.int 1
    %false = torch.constant.bool false
    %none_7 = torch.constant.none
    %5 = torch.aten.prod.dim_int %4, %int1, %false, %none_7 : !torch.vtensor<[3,5],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3],bf16>
    %int0_8 = torch.constant.int 0
    %false_9 = torch.constant.bool false
    %none_10 = torch.constant.none
    %6 = torch.aten.prod.dim_int %5, %int0_8, %false_9, %none_10 : !torch.vtensor<[3],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[],bf16>
    return %6 : !torch.vtensor<[],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_19(%arg0: !torch.vtensor<[3,5],bf16>) -> !torch.vtensor<[],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_5_torch.uint8_2> : tensor<3x5xui8>) : !torch.vtensor<[3,5],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,5],ui8>, !torch.int -> !torch.vtensor<[3,5],i1>
    %2 = torch.vtensor.literal(dense<1.000000e+00> : tensor<bf16>) : !torch.vtensor<[],bf16>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[3,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],bf16>, !torch.none -> !torch.vtensor<[],bf16>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[3,5],i1>, !torch.vtensor<[3,5],bf16>, !torch.vtensor<[],bf16> -> !torch.vtensor<[3,5],bf16>
    %none_3 = torch.constant.none
    %none_4 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu_5 = torch.constant.device "cpu"
    %int0_6 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %4, %none_3, %none_4, %int15, %cpu_5, %int0_6 : !torch.vtensor<[3,5],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int1 = torch.constant.int 1
    %false = torch.constant.bool false
    %none_7 = torch.constant.none
    %5 = torch.aten.prod.dim_int %4, %int1, %false, %none_7 : !torch.vtensor<[3,5],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3],bf16>
    %int0_8 = torch.constant.int 0
    %false_9 = torch.constant.bool false
    %none_10 = torch.constant.none
    %6 = torch.aten.prod.dim_int %5, %int0_8, %false_9, %none_10 : !torch.vtensor<[3],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[],bf16>
    return %6 : !torch.vtensor<[],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_20(%arg0: !torch.vtensor<[3,5],bf16>) -> !torch.vtensor<[],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_5_torch.uint8_3> : tensor<3x5xui8>) : !torch.vtensor<[3,5],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,5],ui8>, !torch.int -> !torch.vtensor<[3,5],i1>
    %2 = torch.vtensor.literal(dense<1.000000e+00> : tensor<bf16>) : !torch.vtensor<[],bf16>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[3,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],bf16>, !torch.none -> !torch.vtensor<[],bf16>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[3,5],i1>, !torch.vtensor<[3,5],bf16>, !torch.vtensor<[],bf16> -> !torch.vtensor<[3,5],bf16>
    %none_3 = torch.constant.none
    %none_4 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu_5 = torch.constant.device "cpu"
    %int0_6 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %4, %none_3, %none_4, %int15, %cpu_5, %int0_6 : !torch.vtensor<[3,5],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int1 = torch.constant.int 1
    %false = torch.constant.bool false
    %none_7 = torch.constant.none
    %5 = torch.aten.prod.dim_int %4, %int1, %false, %none_7 : !torch.vtensor<[3,5],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3],bf16>
    %int0_8 = torch.constant.int 0
    %false_9 = torch.constant.bool false
    %none_10 = torch.constant.none
    %6 = torch.aten.prod.dim_int %5, %int0_8, %false_9, %none_10 : !torch.vtensor<[3],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[],bf16>
    return %6 : !torch.vtensor<[],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_21(%arg0: !torch.vtensor<[3,5],bf16>) -> !torch.vtensor<[1,5],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %arg0, %none, %none_0, %int15, %cpu, %int0 : !torch.vtensor<[3,5],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int0_1 = torch.constant.int 0
    %true = torch.constant.bool true
    %none_2 = torch.constant.none
    %0 = torch.aten.prod.dim_int %arg0, %int0_1, %true, %none_2 : !torch.vtensor<[3,5],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[1,5],bf16>
    return %0 : !torch.vtensor<[1,5],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_22(%arg0: !torch.vtensor<[3,5],bf16>) -> !torch.vtensor<[1,5],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_5_torch.uint8_4> : tensor<3x5xui8>) : !torch.vtensor<[3,5],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,5],ui8>, !torch.int -> !torch.vtensor<[3,5],i1>
    %2 = torch.vtensor.literal(dense<1.000000e+00> : tensor<bf16>) : !torch.vtensor<[],bf16>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[3,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],bf16>, !torch.none -> !torch.vtensor<[],bf16>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[3,5],i1>, !torch.vtensor<[3,5],bf16>, !torch.vtensor<[],bf16> -> !torch.vtensor<[3,5],bf16>
    %none_3 = torch.constant.none
    %none_4 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu_5 = torch.constant.device "cpu"
    %int0_6 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %4, %none_3, %none_4, %int15, %cpu_5, %int0_6 : !torch.vtensor<[3,5],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int0_7 = torch.constant.int 0
    %true = torch.constant.bool true
    %none_8 = torch.constant.none
    %5 = torch.aten.prod.dim_int %4, %int0_7, %true, %none_8 : !torch.vtensor<[3,5],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[1,5],bf16>
    return %5 : !torch.vtensor<[1,5],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_23(%arg0: !torch.vtensor<[3,5],bf16>) -> !torch.vtensor<[1,5],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_5_torch.uint8_5> : tensor<3x5xui8>) : !torch.vtensor<[3,5],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,5],ui8>, !torch.int -> !torch.vtensor<[3,5],i1>
    %2 = torch.vtensor.literal(dense<1.000000e+00> : tensor<bf16>) : !torch.vtensor<[],bf16>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[3,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],bf16>, !torch.none -> !torch.vtensor<[],bf16>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[3,5],i1>, !torch.vtensor<[3,5],bf16>, !torch.vtensor<[],bf16> -> !torch.vtensor<[3,5],bf16>
    %none_3 = torch.constant.none
    %none_4 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu_5 = torch.constant.device "cpu"
    %int0_6 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %4, %none_3, %none_4, %int15, %cpu_5, %int0_6 : !torch.vtensor<[3,5],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int0_7 = torch.constant.int 0
    %true = torch.constant.bool true
    %none_8 = torch.constant.none
    %5 = torch.aten.prod.dim_int %4, %int0_7, %true, %none_8 : !torch.vtensor<[3,5],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[1,5],bf16>
    return %5 : !torch.vtensor<[1,5],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_24(%arg0: !torch.vtensor<[3,5],bf16>) -> !torch.vtensor<[1,5],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_5_torch.uint8_6> : tensor<3x5xui8>) : !torch.vtensor<[3,5],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,5],ui8>, !torch.int -> !torch.vtensor<[3,5],i1>
    %2 = torch.vtensor.literal(dense<1.000000e+00> : tensor<bf16>) : !torch.vtensor<[],bf16>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[3,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],bf16>, !torch.none -> !torch.vtensor<[],bf16>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[3,5],i1>, !torch.vtensor<[3,5],bf16>, !torch.vtensor<[],bf16> -> !torch.vtensor<[3,5],bf16>
    %none_3 = torch.constant.none
    %none_4 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu_5 = torch.constant.device "cpu"
    %int0_6 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %4, %none_3, %none_4, %int15, %cpu_5, %int0_6 : !torch.vtensor<[3,5],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int0_7 = torch.constant.int 0
    %true = torch.constant.bool true
    %none_8 = torch.constant.none
    %5 = torch.aten.prod.dim_int %4, %int0_7, %true, %none_8 : !torch.vtensor<[3,5],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[1,5],bf16>
    return %5 : !torch.vtensor<[1,5],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_25(%arg0: !torch.vtensor<[3,5],bf16>) -> !torch.vtensor<[1,5],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_5_torch.uint8_7> : tensor<3x5xui8>) : !torch.vtensor<[3,5],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,5],ui8>, !torch.int -> !torch.vtensor<[3,5],i1>
    %2 = torch.vtensor.literal(dense<1.000000e+00> : tensor<bf16>) : !torch.vtensor<[],bf16>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[3,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],bf16>, !torch.none -> !torch.vtensor<[],bf16>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[3,5],i1>, !torch.vtensor<[3,5],bf16>, !torch.vtensor<[],bf16> -> !torch.vtensor<[3,5],bf16>
    %none_3 = torch.constant.none
    %none_4 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu_5 = torch.constant.device "cpu"
    %int0_6 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %4, %none_3, %none_4, %int15, %cpu_5, %int0_6 : !torch.vtensor<[3,5],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int0_7 = torch.constant.int 0
    %true = torch.constant.bool true
    %none_8 = torch.constant.none
    %5 = torch.aten.prod.dim_int %4, %int0_7, %true, %none_8 : !torch.vtensor<[3,5],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[1,5],bf16>
    return %5 : !torch.vtensor<[1,5],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_26(%arg0: !torch.vtensor<[3,5],bf16>) -> !torch.vtensor<[3],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %arg0, %none, %none_0, %int15, %cpu, %int0 : !torch.vtensor<[3,5],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int1 = torch.constant.int 1
    %false = torch.constant.bool false
    %none_1 = torch.constant.none
    %0 = torch.aten.prod.dim_int %arg0, %int1, %false, %none_1 : !torch.vtensor<[3,5],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3],bf16>
    return %0 : !torch.vtensor<[3],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_27(%arg0: !torch.vtensor<[3,5],bf16>) -> !torch.vtensor<[3],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_5_torch.uint8_8> : tensor<3x5xui8>) : !torch.vtensor<[3,5],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,5],ui8>, !torch.int -> !torch.vtensor<[3,5],i1>
    %2 = torch.vtensor.literal(dense<1.000000e+00> : tensor<bf16>) : !torch.vtensor<[],bf16>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[3,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],bf16>, !torch.none -> !torch.vtensor<[],bf16>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[3,5],i1>, !torch.vtensor<[3,5],bf16>, !torch.vtensor<[],bf16> -> !torch.vtensor<[3,5],bf16>
    %none_3 = torch.constant.none
    %none_4 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu_5 = torch.constant.device "cpu"
    %int0_6 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %4, %none_3, %none_4, %int15, %cpu_5, %int0_6 : !torch.vtensor<[3,5],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int1 = torch.constant.int 1
    %false = torch.constant.bool false
    %none_7 = torch.constant.none
    %5 = torch.aten.prod.dim_int %4, %int1, %false, %none_7 : !torch.vtensor<[3,5],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3],bf16>
    return %5 : !torch.vtensor<[3],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_28(%arg0: !torch.vtensor<[3,5],bf16>) -> !torch.vtensor<[3],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_5_torch.uint8_9> : tensor<3x5xui8>) : !torch.vtensor<[3,5],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,5],ui8>, !torch.int -> !torch.vtensor<[3,5],i1>
    %2 = torch.vtensor.literal(dense<1.000000e+00> : tensor<bf16>) : !torch.vtensor<[],bf16>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[3,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],bf16>, !torch.none -> !torch.vtensor<[],bf16>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[3,5],i1>, !torch.vtensor<[3,5],bf16>, !torch.vtensor<[],bf16> -> !torch.vtensor<[3,5],bf16>
    %none_3 = torch.constant.none
    %none_4 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu_5 = torch.constant.device "cpu"
    %int0_6 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %4, %none_3, %none_4, %int15, %cpu_5, %int0_6 : !torch.vtensor<[3,5],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int1 = torch.constant.int 1
    %false = torch.constant.bool false
    %none_7 = torch.constant.none
    %5 = torch.aten.prod.dim_int %4, %int1, %false, %none_7 : !torch.vtensor<[3,5],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3],bf16>
    return %5 : !torch.vtensor<[3],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_29(%arg0: !torch.vtensor<[3,5],bf16>) -> !torch.vtensor<[3],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_5_torch.uint8_10> : tensor<3x5xui8>) : !torch.vtensor<[3,5],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,5],ui8>, !torch.int -> !torch.vtensor<[3,5],i1>
    %2 = torch.vtensor.literal(dense<1.000000e+00> : tensor<bf16>) : !torch.vtensor<[],bf16>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[3,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],bf16>, !torch.none -> !torch.vtensor<[],bf16>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[3,5],i1>, !torch.vtensor<[3,5],bf16>, !torch.vtensor<[],bf16> -> !torch.vtensor<[3,5],bf16>
    %none_3 = torch.constant.none
    %none_4 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu_5 = torch.constant.device "cpu"
    %int0_6 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %4, %none_3, %none_4, %int15, %cpu_5, %int0_6 : !torch.vtensor<[3,5],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int1 = torch.constant.int 1
    %false = torch.constant.bool false
    %none_7 = torch.constant.none
    %5 = torch.aten.prod.dim_int %4, %int1, %false, %none_7 : !torch.vtensor<[3,5],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3],bf16>
    return %5 : !torch.vtensor<[3],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_30(%arg0: !torch.vtensor<[3,5],bf16>) -> !torch.vtensor<[3],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_5_torch.uint8_11> : tensor<3x5xui8>) : !torch.vtensor<[3,5],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,5],ui8>, !torch.int -> !torch.vtensor<[3,5],i1>
    %2 = torch.vtensor.literal(dense<1.000000e+00> : tensor<bf16>) : !torch.vtensor<[],bf16>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[3,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],bf16>, !torch.none -> !torch.vtensor<[],bf16>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[3,5],i1>, !torch.vtensor<[3,5],bf16>, !torch.vtensor<[],bf16> -> !torch.vtensor<[3,5],bf16>
    %none_3 = torch.constant.none
    %none_4 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu_5 = torch.constant.device "cpu"
    %int0_6 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %4, %none_3, %none_4, %int15, %cpu_5, %int0_6 : !torch.vtensor<[3,5],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int1 = torch.constant.int 1
    %false = torch.constant.bool false
    %none_7 = torch.constant.none
    %5 = torch.aten.prod.dim_int %4, %int1, %false, %none_7 : !torch.vtensor<[3,5],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3],bf16>
    return %5 : !torch.vtensor<[3],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_31(%arg0: !torch.vtensor<[3,5],bf16>) -> !torch.vtensor<[],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %arg0, %none, %none_0, %int15, %cpu, %int0 : !torch.vtensor<[3,5],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int1 = torch.constant.int 1
    %false = torch.constant.bool false
    %none_1 = torch.constant.none
    %0 = torch.aten.prod.dim_int %arg0, %int1, %false, %none_1 : !torch.vtensor<[3,5],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3],bf16>
    %int0_2 = torch.constant.int 0
    %false_3 = torch.constant.bool false
    %none_4 = torch.constant.none
    %1 = torch.aten.prod.dim_int %0, %int0_2, %false_3, %none_4 : !torch.vtensor<[3],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[],bf16>
    return %1 : !torch.vtensor<[],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_32(%arg0: !torch.vtensor<[3,5],bf16>) -> !torch.vtensor<[],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_5_torch.uint8_12> : tensor<3x5xui8>) : !torch.vtensor<[3,5],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,5],ui8>, !torch.int -> !torch.vtensor<[3,5],i1>
    %2 = torch.vtensor.literal(dense<1.000000e+00> : tensor<bf16>) : !torch.vtensor<[],bf16>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[3,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],bf16>, !torch.none -> !torch.vtensor<[],bf16>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[3,5],i1>, !torch.vtensor<[3,5],bf16>, !torch.vtensor<[],bf16> -> !torch.vtensor<[3,5],bf16>
    %none_3 = torch.constant.none
    %none_4 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu_5 = torch.constant.device "cpu"
    %int0_6 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %4, %none_3, %none_4, %int15, %cpu_5, %int0_6 : !torch.vtensor<[3,5],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int1 = torch.constant.int 1
    %false = torch.constant.bool false
    %none_7 = torch.constant.none
    %5 = torch.aten.prod.dim_int %4, %int1, %false, %none_7 : !torch.vtensor<[3,5],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3],bf16>
    %int0_8 = torch.constant.int 0
    %false_9 = torch.constant.bool false
    %none_10 = torch.constant.none
    %6 = torch.aten.prod.dim_int %5, %int0_8, %false_9, %none_10 : !torch.vtensor<[3],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[],bf16>
    return %6 : !torch.vtensor<[],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_33(%arg0: !torch.vtensor<[3,5],bf16>) -> !torch.vtensor<[],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_5_torch.uint8_13> : tensor<3x5xui8>) : !torch.vtensor<[3,5],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,5],ui8>, !torch.int -> !torch.vtensor<[3,5],i1>
    %2 = torch.vtensor.literal(dense<1.000000e+00> : tensor<bf16>) : !torch.vtensor<[],bf16>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[3,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],bf16>, !torch.none -> !torch.vtensor<[],bf16>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[3,5],i1>, !torch.vtensor<[3,5],bf16>, !torch.vtensor<[],bf16> -> !torch.vtensor<[3,5],bf16>
    %none_3 = torch.constant.none
    %none_4 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu_5 = torch.constant.device "cpu"
    %int0_6 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %4, %none_3, %none_4, %int15, %cpu_5, %int0_6 : !torch.vtensor<[3,5],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int1 = torch.constant.int 1
    %false = torch.constant.bool false
    %none_7 = torch.constant.none
    %5 = torch.aten.prod.dim_int %4, %int1, %false, %none_7 : !torch.vtensor<[3,5],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3],bf16>
    %int0_8 = torch.constant.int 0
    %false_9 = torch.constant.bool false
    %none_10 = torch.constant.none
    %6 = torch.aten.prod.dim_int %5, %int0_8, %false_9, %none_10 : !torch.vtensor<[3],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[],bf16>
    return %6 : !torch.vtensor<[],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_34(%arg0: !torch.vtensor<[3,5],bf16>) -> !torch.vtensor<[],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_5_torch.uint8_14> : tensor<3x5xui8>) : !torch.vtensor<[3,5],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,5],ui8>, !torch.int -> !torch.vtensor<[3,5],i1>
    %2 = torch.vtensor.literal(dense<1.000000e+00> : tensor<bf16>) : !torch.vtensor<[],bf16>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[3,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],bf16>, !torch.none -> !torch.vtensor<[],bf16>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[3,5],i1>, !torch.vtensor<[3,5],bf16>, !torch.vtensor<[],bf16> -> !torch.vtensor<[3,5],bf16>
    %none_3 = torch.constant.none
    %none_4 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu_5 = torch.constant.device "cpu"
    %int0_6 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %4, %none_3, %none_4, %int15, %cpu_5, %int0_6 : !torch.vtensor<[3,5],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int1 = torch.constant.int 1
    %false = torch.constant.bool false
    %none_7 = torch.constant.none
    %5 = torch.aten.prod.dim_int %4, %int1, %false, %none_7 : !torch.vtensor<[3,5],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3],bf16>
    %int0_8 = torch.constant.int 0
    %false_9 = torch.constant.bool false
    %none_10 = torch.constant.none
    %6 = torch.aten.prod.dim_int %5, %int0_8, %false_9, %none_10 : !torch.vtensor<[3],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[],bf16>
    return %6 : !torch.vtensor<[],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_35(%arg0: !torch.vtensor<[3,5],bf16>) -> !torch.vtensor<[],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_5_torch.uint8_15> : tensor<3x5xui8>) : !torch.vtensor<[3,5],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,5],ui8>, !torch.int -> !torch.vtensor<[3,5],i1>
    %2 = torch.vtensor.literal(dense<1.000000e+00> : tensor<bf16>) : !torch.vtensor<[],bf16>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[3,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],bf16>, !torch.none -> !torch.vtensor<[],bf16>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[3,5],i1>, !torch.vtensor<[3,5],bf16>, !torch.vtensor<[],bf16> -> !torch.vtensor<[3,5],bf16>
    %none_3 = torch.constant.none
    %none_4 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu_5 = torch.constant.device "cpu"
    %int0_6 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %4, %none_3, %none_4, %int15, %cpu_5, %int0_6 : !torch.vtensor<[3,5],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int1 = torch.constant.int 1
    %false = torch.constant.bool false
    %none_7 = torch.constant.none
    %5 = torch.aten.prod.dim_int %4, %int1, %false, %none_7 : !torch.vtensor<[3,5],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3],bf16>
    %int0_8 = torch.constant.int 0
    %false_9 = torch.constant.bool false
    %none_10 = torch.constant.none
    %6 = torch.aten.prod.dim_int %5, %int0_8, %false_9, %none_10 : !torch.vtensor<[3],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[],bf16>
    return %6 : !torch.vtensor<[],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_36(%arg0: !torch.vtensor<[3,5],bf16>) -> !torch.vtensor<[1,1],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %arg0, %none, %none_0, %int15, %cpu, %int0 : !torch.vtensor<[3,5],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int1 = torch.constant.int 1
    %true = torch.constant.bool true
    %none_1 = torch.constant.none
    %0 = torch.aten.prod.dim_int %arg0, %int1, %true, %none_1 : !torch.vtensor<[3,5],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3,1],bf16>
    %int0_2 = torch.constant.int 0
    %true_3 = torch.constant.bool true
    %none_4 = torch.constant.none
    %1 = torch.aten.prod.dim_int %0, %int0_2, %true_3, %none_4 : !torch.vtensor<[3,1],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[1,1],bf16>
    return %1 : !torch.vtensor<[1,1],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_37(%arg0: !torch.vtensor<[3,5],bf16>) -> !torch.vtensor<[1,1],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_5_torch.uint8_16> : tensor<3x5xui8>) : !torch.vtensor<[3,5],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,5],ui8>, !torch.int -> !torch.vtensor<[3,5],i1>
    %2 = torch.vtensor.literal(dense<1.000000e+00> : tensor<bf16>) : !torch.vtensor<[],bf16>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[3,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],bf16>, !torch.none -> !torch.vtensor<[],bf16>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[3,5],i1>, !torch.vtensor<[3,5],bf16>, !torch.vtensor<[],bf16> -> !torch.vtensor<[3,5],bf16>
    %none_3 = torch.constant.none
    %none_4 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu_5 = torch.constant.device "cpu"
    %int0_6 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %4, %none_3, %none_4, %int15, %cpu_5, %int0_6 : !torch.vtensor<[3,5],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int1 = torch.constant.int 1
    %true = torch.constant.bool true
    %none_7 = torch.constant.none
    %5 = torch.aten.prod.dim_int %4, %int1, %true, %none_7 : !torch.vtensor<[3,5],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3,1],bf16>
    %int0_8 = torch.constant.int 0
    %true_9 = torch.constant.bool true
    %none_10 = torch.constant.none
    %6 = torch.aten.prod.dim_int %5, %int0_8, %true_9, %none_10 : !torch.vtensor<[3,1],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[1,1],bf16>
    return %6 : !torch.vtensor<[1,1],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_38(%arg0: !torch.vtensor<[3,5],bf16>) -> !torch.vtensor<[1,1],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_5_torch.uint8_17> : tensor<3x5xui8>) : !torch.vtensor<[3,5],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,5],ui8>, !torch.int -> !torch.vtensor<[3,5],i1>
    %2 = torch.vtensor.literal(dense<1.000000e+00> : tensor<bf16>) : !torch.vtensor<[],bf16>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[3,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],bf16>, !torch.none -> !torch.vtensor<[],bf16>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[3,5],i1>, !torch.vtensor<[3,5],bf16>, !torch.vtensor<[],bf16> -> !torch.vtensor<[3,5],bf16>
    %none_3 = torch.constant.none
    %none_4 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu_5 = torch.constant.device "cpu"
    %int0_6 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %4, %none_3, %none_4, %int15, %cpu_5, %int0_6 : !torch.vtensor<[3,5],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int1 = torch.constant.int 1
    %true = torch.constant.bool true
    %none_7 = torch.constant.none
    %5 = torch.aten.prod.dim_int %4, %int1, %true, %none_7 : !torch.vtensor<[3,5],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3,1],bf16>
    %int0_8 = torch.constant.int 0
    %true_9 = torch.constant.bool true
    %none_10 = torch.constant.none
    %6 = torch.aten.prod.dim_int %5, %int0_8, %true_9, %none_10 : !torch.vtensor<[3,1],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[1,1],bf16>
    return %6 : !torch.vtensor<[1,1],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_39(%arg0: !torch.vtensor<[3,5],bf16>) -> !torch.vtensor<[1,1],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_5_torch.uint8_18> : tensor<3x5xui8>) : !torch.vtensor<[3,5],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,5],ui8>, !torch.int -> !torch.vtensor<[3,5],i1>
    %2 = torch.vtensor.literal(dense<1.000000e+00> : tensor<bf16>) : !torch.vtensor<[],bf16>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[3,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],bf16>, !torch.none -> !torch.vtensor<[],bf16>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[3,5],i1>, !torch.vtensor<[3,5],bf16>, !torch.vtensor<[],bf16> -> !torch.vtensor<[3,5],bf16>
    %none_3 = torch.constant.none
    %none_4 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu_5 = torch.constant.device "cpu"
    %int0_6 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %4, %none_3, %none_4, %int15, %cpu_5, %int0_6 : !torch.vtensor<[3,5],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int1 = torch.constant.int 1
    %true = torch.constant.bool true
    %none_7 = torch.constant.none
    %5 = torch.aten.prod.dim_int %4, %int1, %true, %none_7 : !torch.vtensor<[3,5],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3,1],bf16>
    %int0_8 = torch.constant.int 0
    %true_9 = torch.constant.bool true
    %none_10 = torch.constant.none
    %6 = torch.aten.prod.dim_int %5, %int0_8, %true_9, %none_10 : !torch.vtensor<[3,1],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[1,1],bf16>
    return %6 : !torch.vtensor<[1,1],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_40(%arg0: !torch.vtensor<[3,5],bf16>) -> !torch.vtensor<[1,1],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_5_torch.uint8_19> : tensor<3x5xui8>) : !torch.vtensor<[3,5],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,5],ui8>, !torch.int -> !torch.vtensor<[3,5],i1>
    %2 = torch.vtensor.literal(dense<1.000000e+00> : tensor<bf16>) : !torch.vtensor<[],bf16>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[3,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],bf16>, !torch.none -> !torch.vtensor<[],bf16>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[3,5],i1>, !torch.vtensor<[3,5],bf16>, !torch.vtensor<[],bf16> -> !torch.vtensor<[3,5],bf16>
    %none_3 = torch.constant.none
    %none_4 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu_5 = torch.constant.device "cpu"
    %int0_6 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %4, %none_3, %none_4, %int15, %cpu_5, %int0_6 : !torch.vtensor<[3,5],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int1 = torch.constant.int 1
    %true = torch.constant.bool true
    %none_7 = torch.constant.none
    %5 = torch.aten.prod.dim_int %4, %int1, %true, %none_7 : !torch.vtensor<[3,5],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3,1],bf16>
    %int0_8 = torch.constant.int 0
    %true_9 = torch.constant.bool true
    %none_10 = torch.constant.none
    %6 = torch.aten.prod.dim_int %5, %int0_8, %true_9, %none_10 : !torch.vtensor<[3,1],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[1,1],bf16>
    return %6 : !torch.vtensor<[1,1],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_41(%arg0: !torch.vtensor<[3,2,1,2],bf16>) -> !torch.vtensor<[],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %arg0, %none, %none_0, %int15, %cpu, %int0 : !torch.vtensor<[3,2,1,2],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int3 = torch.constant.int 3
    %false = torch.constant.bool false
    %none_1 = torch.constant.none
    %0 = torch.aten.prod.dim_int %arg0, %int3, %false, %none_1 : !torch.vtensor<[3,2,1,2],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3,2,1],bf16>
    %int2 = torch.constant.int 2
    %false_2 = torch.constant.bool false
    %none_3 = torch.constant.none
    %1 = torch.aten.prod.dim_int %0, %int2, %false_2, %none_3 : !torch.vtensor<[3,2,1],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3,2],bf16>
    %int1 = torch.constant.int 1
    %false_4 = torch.constant.bool false
    %none_5 = torch.constant.none
    %2 = torch.aten.prod.dim_int %1, %int1, %false_4, %none_5 : !torch.vtensor<[3,2],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3],bf16>
    %int0_6 = torch.constant.int 0
    %false_7 = torch.constant.bool false
    %none_8 = torch.constant.none
    %3 = torch.aten.prod.dim_int %2, %int0_6, %false_7, %none_8 : !torch.vtensor<[3],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[],bf16>
    return %3 : !torch.vtensor<[],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_42(%arg0: !torch.vtensor<[3,2,1,2],bf16>) -> !torch.vtensor<[],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_2_1_2_torch.uint8> : tensor<3x2x1x2xui8>) : !torch.vtensor<[3,2,1,2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,2,1,2],ui8>, !torch.int -> !torch.vtensor<[3,2,1,2],i1>
    %2 = torch.vtensor.literal(dense<1.000000e+00> : tensor<bf16>) : !torch.vtensor<[],bf16>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],bf16>, !torch.none -> !torch.vtensor<[],bf16>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],bf16>, !torch.vtensor<[],bf16> -> !torch.vtensor<[3,2,1,2],bf16>
    %none_3 = torch.constant.none
    %none_4 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu_5 = torch.constant.device "cpu"
    %int0_6 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %4, %none_3, %none_4, %int15, %cpu_5, %int0_6 : !torch.vtensor<[3,2,1,2],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int3 = torch.constant.int 3
    %false = torch.constant.bool false
    %none_7 = torch.constant.none
    %5 = torch.aten.prod.dim_int %4, %int3, %false, %none_7 : !torch.vtensor<[3,2,1,2],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3,2,1],bf16>
    %int2 = torch.constant.int 2
    %false_8 = torch.constant.bool false
    %none_9 = torch.constant.none
    %6 = torch.aten.prod.dim_int %5, %int2, %false_8, %none_9 : !torch.vtensor<[3,2,1],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3,2],bf16>
    %int1 = torch.constant.int 1
    %false_10 = torch.constant.bool false
    %none_11 = torch.constant.none
    %7 = torch.aten.prod.dim_int %6, %int1, %false_10, %none_11 : !torch.vtensor<[3,2],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3],bf16>
    %int0_12 = torch.constant.int 0
    %false_13 = torch.constant.bool false
    %none_14 = torch.constant.none
    %8 = torch.aten.prod.dim_int %7, %int0_12, %false_13, %none_14 : !torch.vtensor<[3],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[],bf16>
    return %8 : !torch.vtensor<[],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_43(%arg0: !torch.vtensor<[3,2,1,2],bf16>) -> !torch.vtensor<[],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_2_1_1_torch.uint8> : tensor<3x2x1x1xui8>) : !torch.vtensor<[3,2,1,1],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,2,1,1],ui8>, !torch.int -> !torch.vtensor<[3,2,1,1],i1>
    %2 = torch.vtensor.literal(dense<1.000000e+00> : tensor<bf16>) : !torch.vtensor<[],bf16>
    %none = torch.constant.none
    %3 = torch.aten.clone %1, %none : !torch.vtensor<[3,2,1,1],i1>, !torch.none -> !torch.vtensor<[3,2,1,1],i1>
    %int3 = torch.constant.int 3
    %int2 = torch.constant.int 2
    %int1 = torch.constant.int 1
    %int2_0 = torch.constant.int 2
    %4 = torch.prim.ListConstruct %int3, %int2, %int1, %int2_0 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %5 = torch.aten.expand %3, %4, %false : !torch.vtensor<[3,2,1,1],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3,2,1,2],i1>
    %none_1 = torch.constant.none
    %none_2 = torch.constant.none
    %int11_3 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_1, %none_2, %int11_3, %cpu, %int0 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int11_6 = torch.constant.int 11
    %cpu_7 = torch.constant.device "cpu"
    %int0_8 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_4, %none_5, %int11_6, %cpu_7, %int0_8 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_9 = torch.constant.none
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],bf16>, !torch.none -> !torch.vtensor<[],bf16>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],bf16>, !torch.vtensor<[],bf16> -> !torch.vtensor<[3,2,1,2],bf16>
    %none_10 = torch.constant.none
    %none_11 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu_12 = torch.constant.device "cpu"
    %int0_13 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %7, %none_10, %none_11, %int15, %cpu_12, %int0_13 : !torch.vtensor<[3,2,1,2],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int3_14 = torch.constant.int 3
    %false_15 = torch.constant.bool false
    %none_16 = torch.constant.none
    %8 = torch.aten.prod.dim_int %7, %int3_14, %false_15, %none_16 : !torch.vtensor<[3,2,1,2],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3,2,1],bf16>
    %int2_17 = torch.constant.int 2
    %false_18 = torch.constant.bool false
    %none_19 = torch.constant.none
    %9 = torch.aten.prod.dim_int %8, %int2_17, %false_18, %none_19 : !torch.vtensor<[3,2,1],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3,2],bf16>
    %int1_20 = torch.constant.int 1
    %false_21 = torch.constant.bool false
    %none_22 = torch.constant.none
    %10 = torch.aten.prod.dim_int %9, %int1_20, %false_21, %none_22 : !torch.vtensor<[3,2],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3],bf16>
    %int0_23 = torch.constant.int 0
    %false_24 = torch.constant.bool false
    %none_25 = torch.constant.none
    %11 = torch.aten.prod.dim_int %10, %int0_23, %false_24, %none_25 : !torch.vtensor<[3],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[],bf16>
    return %11 : !torch.vtensor<[],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_44(%arg0: !torch.vtensor<[3,2,1,2],bf16>) -> !torch.vtensor<[],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_1_1_2_torch.uint8> : tensor<3x1x1x2xui8>) : !torch.vtensor<[3,1,1,2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,1,1,2],ui8>, !torch.int -> !torch.vtensor<[3,1,1,2],i1>
    %2 = torch.vtensor.literal(dense<1.000000e+00> : tensor<bf16>) : !torch.vtensor<[],bf16>
    %none = torch.constant.none
    %3 = torch.aten.clone %1, %none : !torch.vtensor<[3,1,1,2],i1>, !torch.none -> !torch.vtensor<[3,1,1,2],i1>
    %int3 = torch.constant.int 3
    %int2 = torch.constant.int 2
    %int1 = torch.constant.int 1
    %int2_0 = torch.constant.int 2
    %4 = torch.prim.ListConstruct %int3, %int2, %int1, %int2_0 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %5 = torch.aten.expand %3, %4, %false : !torch.vtensor<[3,1,1,2],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3,2,1,2],i1>
    %none_1 = torch.constant.none
    %none_2 = torch.constant.none
    %int11_3 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_1, %none_2, %int11_3, %cpu, %int0 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int11_6 = torch.constant.int 11
    %cpu_7 = torch.constant.device "cpu"
    %int0_8 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_4, %none_5, %int11_6, %cpu_7, %int0_8 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_9 = torch.constant.none
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],bf16>, !torch.none -> !torch.vtensor<[],bf16>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],bf16>, !torch.vtensor<[],bf16> -> !torch.vtensor<[3,2,1,2],bf16>
    %none_10 = torch.constant.none
    %none_11 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu_12 = torch.constant.device "cpu"
    %int0_13 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %7, %none_10, %none_11, %int15, %cpu_12, %int0_13 : !torch.vtensor<[3,2,1,2],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int3_14 = torch.constant.int 3
    %false_15 = torch.constant.bool false
    %none_16 = torch.constant.none
    %8 = torch.aten.prod.dim_int %7, %int3_14, %false_15, %none_16 : !torch.vtensor<[3,2,1,2],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3,2,1],bf16>
    %int2_17 = torch.constant.int 2
    %false_18 = torch.constant.bool false
    %none_19 = torch.constant.none
    %9 = torch.aten.prod.dim_int %8, %int2_17, %false_18, %none_19 : !torch.vtensor<[3,2,1],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3,2],bf16>
    %int1_20 = torch.constant.int 1
    %false_21 = torch.constant.bool false
    %none_22 = torch.constant.none
    %10 = torch.aten.prod.dim_int %9, %int1_20, %false_21, %none_22 : !torch.vtensor<[3,2],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3],bf16>
    %int0_23 = torch.constant.int 0
    %false_24 = torch.constant.bool false
    %none_25 = torch.constant.none
    %11 = torch.aten.prod.dim_int %10, %int0_23, %false_24, %none_25 : !torch.vtensor<[3],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[],bf16>
    return %11 : !torch.vtensor<[],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_45(%arg0: !torch.vtensor<[3,2,1,2],bf16>) -> !torch.vtensor<[],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_1_2_1_2_torch.uint8> : tensor<1x2x1x2xui8>) : !torch.vtensor<[1,2,1,2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[1,2,1,2],ui8>, !torch.int -> !torch.vtensor<[1,2,1,2],i1>
    %2 = torch.vtensor.literal(dense<1.000000e+00> : tensor<bf16>) : !torch.vtensor<[],bf16>
    %none = torch.constant.none
    %3 = torch.aten.clone %1, %none : !torch.vtensor<[1,2,1,2],i1>, !torch.none -> !torch.vtensor<[1,2,1,2],i1>
    %int3 = torch.constant.int 3
    %int2 = torch.constant.int 2
    %int1 = torch.constant.int 1
    %int2_0 = torch.constant.int 2
    %4 = torch.prim.ListConstruct %int3, %int2, %int1, %int2_0 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %5 = torch.aten.expand %3, %4, %false : !torch.vtensor<[1,2,1,2],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3,2,1,2],i1>
    %none_1 = torch.constant.none
    %none_2 = torch.constant.none
    %int11_3 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_1, %none_2, %int11_3, %cpu, %int0 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int11_6 = torch.constant.int 11
    %cpu_7 = torch.constant.device "cpu"
    %int0_8 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_4, %none_5, %int11_6, %cpu_7, %int0_8 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_9 = torch.constant.none
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],bf16>, !torch.none -> !torch.vtensor<[],bf16>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],bf16>, !torch.vtensor<[],bf16> -> !torch.vtensor<[3,2,1,2],bf16>
    %none_10 = torch.constant.none
    %none_11 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu_12 = torch.constant.device "cpu"
    %int0_13 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %7, %none_10, %none_11, %int15, %cpu_12, %int0_13 : !torch.vtensor<[3,2,1,2],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int3_14 = torch.constant.int 3
    %false_15 = torch.constant.bool false
    %none_16 = torch.constant.none
    %8 = torch.aten.prod.dim_int %7, %int3_14, %false_15, %none_16 : !torch.vtensor<[3,2,1,2],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3,2,1],bf16>
    %int2_17 = torch.constant.int 2
    %false_18 = torch.constant.bool false
    %none_19 = torch.constant.none
    %9 = torch.aten.prod.dim_int %8, %int2_17, %false_18, %none_19 : !torch.vtensor<[3,2,1],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3,2],bf16>
    %int1_20 = torch.constant.int 1
    %false_21 = torch.constant.bool false
    %none_22 = torch.constant.none
    %10 = torch.aten.prod.dim_int %9, %int1_20, %false_21, %none_22 : !torch.vtensor<[3,2],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3],bf16>
    %int0_23 = torch.constant.int 0
    %false_24 = torch.constant.bool false
    %none_25 = torch.constant.none
    %11 = torch.aten.prod.dim_int %10, %int0_23, %false_24, %none_25 : !torch.vtensor<[3],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[],bf16>
    return %11 : !torch.vtensor<[],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_46(%arg0: !torch.vtensor<[3,2,1,2],bf16>) -> !torch.vtensor<[],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_2_1_2_torch.uint8> : tensor<2x1x2xui8>) : !torch.vtensor<[2,1,2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[2,1,2],ui8>, !torch.int -> !torch.vtensor<[2,1,2],i1>
    %2 = torch.vtensor.literal(dense<1.000000e+00> : tensor<bf16>) : !torch.vtensor<[],bf16>
    %none = torch.constant.none
    %3 = torch.aten.clone %1, %none : !torch.vtensor<[2,1,2],i1>, !torch.none -> !torch.vtensor<[2,1,2],i1>
    %int3 = torch.constant.int 3
    %int2 = torch.constant.int 2
    %int1 = torch.constant.int 1
    %int2_0 = torch.constant.int 2
    %4 = torch.prim.ListConstruct %int3, %int2, %int1, %int2_0 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %5 = torch.aten.expand %3, %4, %false : !torch.vtensor<[2,1,2],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3,2,1,2],i1>
    %none_1 = torch.constant.none
    %none_2 = torch.constant.none
    %int11_3 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_1, %none_2, %int11_3, %cpu, %int0 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int11_6 = torch.constant.int 11
    %cpu_7 = torch.constant.device "cpu"
    %int0_8 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_4, %none_5, %int11_6, %cpu_7, %int0_8 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_9 = torch.constant.none
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],bf16>, !torch.none -> !torch.vtensor<[],bf16>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],bf16>, !torch.vtensor<[],bf16> -> !torch.vtensor<[3,2,1,2],bf16>
    %none_10 = torch.constant.none
    %none_11 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu_12 = torch.constant.device "cpu"
    %int0_13 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %7, %none_10, %none_11, %int15, %cpu_12, %int0_13 : !torch.vtensor<[3,2,1,2],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int3_14 = torch.constant.int 3
    %false_15 = torch.constant.bool false
    %none_16 = torch.constant.none
    %8 = torch.aten.prod.dim_int %7, %int3_14, %false_15, %none_16 : !torch.vtensor<[3,2,1,2],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3,2,1],bf16>
    %int2_17 = torch.constant.int 2
    %false_18 = torch.constant.bool false
    %none_19 = torch.constant.none
    %9 = torch.aten.prod.dim_int %8, %int2_17, %false_18, %none_19 : !torch.vtensor<[3,2,1],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3,2],bf16>
    %int1_20 = torch.constant.int 1
    %false_21 = torch.constant.bool false
    %none_22 = torch.constant.none
    %10 = torch.aten.prod.dim_int %9, %int1_20, %false_21, %none_22 : !torch.vtensor<[3,2],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3],bf16>
    %int0_23 = torch.constant.int 0
    %false_24 = torch.constant.bool false
    %none_25 = torch.constant.none
    %11 = torch.aten.prod.dim_int %10, %int0_23, %false_24, %none_25 : !torch.vtensor<[3],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[],bf16>
    return %11 : !torch.vtensor<[],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_47(%arg0: !torch.vtensor<[3,2,1,2],bf16>) -> !torch.vtensor<[],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_2_torch.uint8_4> : tensor<2xui8>) : !torch.vtensor<[2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[2],ui8>, !torch.int -> !torch.vtensor<[2],i1>
    %2 = torch.vtensor.literal(dense<1.000000e+00> : tensor<bf16>) : !torch.vtensor<[],bf16>
    %none = torch.constant.none
    %3 = torch.aten.clone %1, %none : !torch.vtensor<[2],i1>, !torch.none -> !torch.vtensor<[2],i1>
    %int3 = torch.constant.int 3
    %int2 = torch.constant.int 2
    %int1 = torch.constant.int 1
    %int2_0 = torch.constant.int 2
    %4 = torch.prim.ListConstruct %int3, %int2, %int1, %int2_0 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %5 = torch.aten.expand %3, %4, %false : !torch.vtensor<[2],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3,2,1,2],i1>
    %none_1 = torch.constant.none
    %none_2 = torch.constant.none
    %int11_3 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_1, %none_2, %int11_3, %cpu, %int0 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int11_6 = torch.constant.int 11
    %cpu_7 = torch.constant.device "cpu"
    %int0_8 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_4, %none_5, %int11_6, %cpu_7, %int0_8 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_9 = torch.constant.none
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],bf16>, !torch.none -> !torch.vtensor<[],bf16>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],bf16>, !torch.vtensor<[],bf16> -> !torch.vtensor<[3,2,1,2],bf16>
    %none_10 = torch.constant.none
    %none_11 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu_12 = torch.constant.device "cpu"
    %int0_13 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %7, %none_10, %none_11, %int15, %cpu_12, %int0_13 : !torch.vtensor<[3,2,1,2],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int3_14 = torch.constant.int 3
    %false_15 = torch.constant.bool false
    %none_16 = torch.constant.none
    %8 = torch.aten.prod.dim_int %7, %int3_14, %false_15, %none_16 : !torch.vtensor<[3,2,1,2],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3,2,1],bf16>
    %int2_17 = torch.constant.int 2
    %false_18 = torch.constant.bool false
    %none_19 = torch.constant.none
    %9 = torch.aten.prod.dim_int %8, %int2_17, %false_18, %none_19 : !torch.vtensor<[3,2,1],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3,2],bf16>
    %int1_20 = torch.constant.int 1
    %false_21 = torch.constant.bool false
    %none_22 = torch.constant.none
    %10 = torch.aten.prod.dim_int %9, %int1_20, %false_21, %none_22 : !torch.vtensor<[3,2],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3],bf16>
    %int0_23 = torch.constant.int 0
    %false_24 = torch.constant.bool false
    %none_25 = torch.constant.none
    %11 = torch.aten.prod.dim_int %10, %int0_23, %false_24, %none_25 : !torch.vtensor<[3],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[],bf16>
    return %11 : !torch.vtensor<[],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_48(%arg0: !torch.vtensor<[3,2,1,2],bf16>) -> !torch.vtensor<[1,2,1,2],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %arg0, %none, %none_0, %int15, %cpu, %int0 : !torch.vtensor<[3,2,1,2],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int0_1 = torch.constant.int 0
    %true = torch.constant.bool true
    %none_2 = torch.constant.none
    %0 = torch.aten.prod.dim_int %arg0, %int0_1, %true, %none_2 : !torch.vtensor<[3,2,1,2],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[1,2,1,2],bf16>
    return %0 : !torch.vtensor<[1,2,1,2],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_49(%arg0: !torch.vtensor<[3,2,1,2],bf16>) -> !torch.vtensor<[1,2,1,2],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_2_1_2_torch.uint8_1> : tensor<3x2x1x2xui8>) : !torch.vtensor<[3,2,1,2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,2,1,2],ui8>, !torch.int -> !torch.vtensor<[3,2,1,2],i1>
    %2 = torch.vtensor.literal(dense<1.000000e+00> : tensor<bf16>) : !torch.vtensor<[],bf16>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],bf16>, !torch.none -> !torch.vtensor<[],bf16>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],bf16>, !torch.vtensor<[],bf16> -> !torch.vtensor<[3,2,1,2],bf16>
    %none_3 = torch.constant.none
    %none_4 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu_5 = torch.constant.device "cpu"
    %int0_6 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %4, %none_3, %none_4, %int15, %cpu_5, %int0_6 : !torch.vtensor<[3,2,1,2],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int0_7 = torch.constant.int 0
    %true = torch.constant.bool true
    %none_8 = torch.constant.none
    %5 = torch.aten.prod.dim_int %4, %int0_7, %true, %none_8 : !torch.vtensor<[3,2,1,2],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[1,2,1,2],bf16>
    return %5 : !torch.vtensor<[1,2,1,2],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_50(%arg0: !torch.vtensor<[3,2,1,2],bf16>) -> !torch.vtensor<[1,2,1,2],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_2_1_1_torch.uint8_1> : tensor<3x2x1x1xui8>) : !torch.vtensor<[3,2,1,1],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,2,1,1],ui8>, !torch.int -> !torch.vtensor<[3,2,1,1],i1>
    %2 = torch.vtensor.literal(dense<1.000000e+00> : tensor<bf16>) : !torch.vtensor<[],bf16>
    %none = torch.constant.none
    %3 = torch.aten.clone %1, %none : !torch.vtensor<[3,2,1,1],i1>, !torch.none -> !torch.vtensor<[3,2,1,1],i1>
    %int3 = torch.constant.int 3
    %int2 = torch.constant.int 2
    %int1 = torch.constant.int 1
    %int2_0 = torch.constant.int 2
    %4 = torch.prim.ListConstruct %int3, %int2, %int1, %int2_0 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %5 = torch.aten.expand %3, %4, %false : !torch.vtensor<[3,2,1,1],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3,2,1,2],i1>
    %none_1 = torch.constant.none
    %none_2 = torch.constant.none
    %int11_3 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_1, %none_2, %int11_3, %cpu, %int0 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int11_6 = torch.constant.int 11
    %cpu_7 = torch.constant.device "cpu"
    %int0_8 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_4, %none_5, %int11_6, %cpu_7, %int0_8 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_9 = torch.constant.none
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],bf16>, !torch.none -> !torch.vtensor<[],bf16>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],bf16>, !torch.vtensor<[],bf16> -> !torch.vtensor<[3,2,1,2],bf16>
    %none_10 = torch.constant.none
    %none_11 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu_12 = torch.constant.device "cpu"
    %int0_13 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %7, %none_10, %none_11, %int15, %cpu_12, %int0_13 : !torch.vtensor<[3,2,1,2],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int0_14 = torch.constant.int 0
    %true = torch.constant.bool true
    %none_15 = torch.constant.none
    %8 = torch.aten.prod.dim_int %7, %int0_14, %true, %none_15 : !torch.vtensor<[3,2,1,2],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[1,2,1,2],bf16>
    return %8 : !torch.vtensor<[1,2,1,2],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_51(%arg0: !torch.vtensor<[3,2,1,2],bf16>) -> !torch.vtensor<[1,2,1,2],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_1_1_2_torch.uint8_1> : tensor<3x1x1x2xui8>) : !torch.vtensor<[3,1,1,2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,1,1,2],ui8>, !torch.int -> !torch.vtensor<[3,1,1,2],i1>
    %2 = torch.vtensor.literal(dense<1.000000e+00> : tensor<bf16>) : !torch.vtensor<[],bf16>
    %none = torch.constant.none
    %3 = torch.aten.clone %1, %none : !torch.vtensor<[3,1,1,2],i1>, !torch.none -> !torch.vtensor<[3,1,1,2],i1>
    %int3 = torch.constant.int 3
    %int2 = torch.constant.int 2
    %int1 = torch.constant.int 1
    %int2_0 = torch.constant.int 2
    %4 = torch.prim.ListConstruct %int3, %int2, %int1, %int2_0 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %5 = torch.aten.expand %3, %4, %false : !torch.vtensor<[3,1,1,2],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3,2,1,2],i1>
    %none_1 = torch.constant.none
    %none_2 = torch.constant.none
    %int11_3 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_1, %none_2, %int11_3, %cpu, %int0 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int11_6 = torch.constant.int 11
    %cpu_7 = torch.constant.device "cpu"
    %int0_8 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_4, %none_5, %int11_6, %cpu_7, %int0_8 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_9 = torch.constant.none
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],bf16>, !torch.none -> !torch.vtensor<[],bf16>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],bf16>, !torch.vtensor<[],bf16> -> !torch.vtensor<[3,2,1,2],bf16>
    %none_10 = torch.constant.none
    %none_11 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu_12 = torch.constant.device "cpu"
    %int0_13 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %7, %none_10, %none_11, %int15, %cpu_12, %int0_13 : !torch.vtensor<[3,2,1,2],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int0_14 = torch.constant.int 0
    %true = torch.constant.bool true
    %none_15 = torch.constant.none
    %8 = torch.aten.prod.dim_int %7, %int0_14, %true, %none_15 : !torch.vtensor<[3,2,1,2],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[1,2,1,2],bf16>
    return %8 : !torch.vtensor<[1,2,1,2],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_52(%arg0: !torch.vtensor<[3,2,1,2],bf16>) -> !torch.vtensor<[1,2,1,2],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_1_2_1_2_torch.uint8_1> : tensor<1x2x1x2xui8>) : !torch.vtensor<[1,2,1,2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[1,2,1,2],ui8>, !torch.int -> !torch.vtensor<[1,2,1,2],i1>
    %2 = torch.vtensor.literal(dense<1.000000e+00> : tensor<bf16>) : !torch.vtensor<[],bf16>
    %none = torch.constant.none
    %3 = torch.aten.clone %1, %none : !torch.vtensor<[1,2,1,2],i1>, !torch.none -> !torch.vtensor<[1,2,1,2],i1>
    %int3 = torch.constant.int 3
    %int2 = torch.constant.int 2
    %int1 = torch.constant.int 1
    %int2_0 = torch.constant.int 2
    %4 = torch.prim.ListConstruct %int3, %int2, %int1, %int2_0 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %5 = torch.aten.expand %3, %4, %false : !torch.vtensor<[1,2,1,2],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3,2,1,2],i1>
    %none_1 = torch.constant.none
    %none_2 = torch.constant.none
    %int11_3 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_1, %none_2, %int11_3, %cpu, %int0 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int11_6 = torch.constant.int 11
    %cpu_7 = torch.constant.device "cpu"
    %int0_8 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_4, %none_5, %int11_6, %cpu_7, %int0_8 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_9 = torch.constant.none
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],bf16>, !torch.none -> !torch.vtensor<[],bf16>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],bf16>, !torch.vtensor<[],bf16> -> !torch.vtensor<[3,2,1,2],bf16>
    %none_10 = torch.constant.none
    %none_11 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu_12 = torch.constant.device "cpu"
    %int0_13 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %7, %none_10, %none_11, %int15, %cpu_12, %int0_13 : !torch.vtensor<[3,2,1,2],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int0_14 = torch.constant.int 0
    %true = torch.constant.bool true
    %none_15 = torch.constant.none
    %8 = torch.aten.prod.dim_int %7, %int0_14, %true, %none_15 : !torch.vtensor<[3,2,1,2],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[1,2,1,2],bf16>
    return %8 : !torch.vtensor<[1,2,1,2],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_53(%arg0: !torch.vtensor<[3,2,1,2],bf16>) -> !torch.vtensor<[1,2,1,2],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_2_1_2_torch.uint8_1> : tensor<2x1x2xui8>) : !torch.vtensor<[2,1,2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[2,1,2],ui8>, !torch.int -> !torch.vtensor<[2,1,2],i1>
    %2 = torch.vtensor.literal(dense<1.000000e+00> : tensor<bf16>) : !torch.vtensor<[],bf16>
    %none = torch.constant.none
    %3 = torch.aten.clone %1, %none : !torch.vtensor<[2,1,2],i1>, !torch.none -> !torch.vtensor<[2,1,2],i1>
    %int3 = torch.constant.int 3
    %int2 = torch.constant.int 2
    %int1 = torch.constant.int 1
    %int2_0 = torch.constant.int 2
    %4 = torch.prim.ListConstruct %int3, %int2, %int1, %int2_0 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %5 = torch.aten.expand %3, %4, %false : !torch.vtensor<[2,1,2],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3,2,1,2],i1>
    %none_1 = torch.constant.none
    %none_2 = torch.constant.none
    %int11_3 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_1, %none_2, %int11_3, %cpu, %int0 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int11_6 = torch.constant.int 11
    %cpu_7 = torch.constant.device "cpu"
    %int0_8 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_4, %none_5, %int11_6, %cpu_7, %int0_8 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_9 = torch.constant.none
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],bf16>, !torch.none -> !torch.vtensor<[],bf16>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],bf16>, !torch.vtensor<[],bf16> -> !torch.vtensor<[3,2,1,2],bf16>
    %none_10 = torch.constant.none
    %none_11 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu_12 = torch.constant.device "cpu"
    %int0_13 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %7, %none_10, %none_11, %int15, %cpu_12, %int0_13 : !torch.vtensor<[3,2,1,2],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int0_14 = torch.constant.int 0
    %true = torch.constant.bool true
    %none_15 = torch.constant.none
    %8 = torch.aten.prod.dim_int %7, %int0_14, %true, %none_15 : !torch.vtensor<[3,2,1,2],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[1,2,1,2],bf16>
    return %8 : !torch.vtensor<[1,2,1,2],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_54(%arg0: !torch.vtensor<[3,2,1,2],bf16>) -> !torch.vtensor<[1,2,1,2],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_2_torch.uint8_5> : tensor<2xui8>) : !torch.vtensor<[2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[2],ui8>, !torch.int -> !torch.vtensor<[2],i1>
    %2 = torch.vtensor.literal(dense<1.000000e+00> : tensor<bf16>) : !torch.vtensor<[],bf16>
    %none = torch.constant.none
    %3 = torch.aten.clone %1, %none : !torch.vtensor<[2],i1>, !torch.none -> !torch.vtensor<[2],i1>
    %int3 = torch.constant.int 3
    %int2 = torch.constant.int 2
    %int1 = torch.constant.int 1
    %int2_0 = torch.constant.int 2
    %4 = torch.prim.ListConstruct %int3, %int2, %int1, %int2_0 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %5 = torch.aten.expand %3, %4, %false : !torch.vtensor<[2],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3,2,1,2],i1>
    %none_1 = torch.constant.none
    %none_2 = torch.constant.none
    %int11_3 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_1, %none_2, %int11_3, %cpu, %int0 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int11_6 = torch.constant.int 11
    %cpu_7 = torch.constant.device "cpu"
    %int0_8 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_4, %none_5, %int11_6, %cpu_7, %int0_8 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_9 = torch.constant.none
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],bf16>, !torch.none -> !torch.vtensor<[],bf16>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],bf16>, !torch.vtensor<[],bf16> -> !torch.vtensor<[3,2,1,2],bf16>
    %none_10 = torch.constant.none
    %none_11 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu_12 = torch.constant.device "cpu"
    %int0_13 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %7, %none_10, %none_11, %int15, %cpu_12, %int0_13 : !torch.vtensor<[3,2,1,2],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int0_14 = torch.constant.int 0
    %true = torch.constant.bool true
    %none_15 = torch.constant.none
    %8 = torch.aten.prod.dim_int %7, %int0_14, %true, %none_15 : !torch.vtensor<[3,2,1,2],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[1,2,1,2],bf16>
    return %8 : !torch.vtensor<[1,2,1,2],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_55(%arg0: !torch.vtensor<[3,2,1,2],bf16>) -> !torch.vtensor<[3,2,1],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %arg0, %none, %none_0, %int15, %cpu, %int0 : !torch.vtensor<[3,2,1,2],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int3 = torch.constant.int 3
    %false = torch.constant.bool false
    %none_1 = torch.constant.none
    %0 = torch.aten.prod.dim_int %arg0, %int3, %false, %none_1 : !torch.vtensor<[3,2,1,2],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3,2,1],bf16>
    return %0 : !torch.vtensor<[3,2,1],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_56(%arg0: !torch.vtensor<[3,2,1,2],bf16>) -> !torch.vtensor<[3,2,1],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_2_1_2_torch.uint8_2> : tensor<3x2x1x2xui8>) : !torch.vtensor<[3,2,1,2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,2,1,2],ui8>, !torch.int -> !torch.vtensor<[3,2,1,2],i1>
    %2 = torch.vtensor.literal(dense<1.000000e+00> : tensor<bf16>) : !torch.vtensor<[],bf16>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],bf16>, !torch.none -> !torch.vtensor<[],bf16>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],bf16>, !torch.vtensor<[],bf16> -> !torch.vtensor<[3,2,1,2],bf16>
    %none_3 = torch.constant.none
    %none_4 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu_5 = torch.constant.device "cpu"
    %int0_6 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %4, %none_3, %none_4, %int15, %cpu_5, %int0_6 : !torch.vtensor<[3,2,1,2],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int3 = torch.constant.int 3
    %false = torch.constant.bool false
    %none_7 = torch.constant.none
    %5 = torch.aten.prod.dim_int %4, %int3, %false, %none_7 : !torch.vtensor<[3,2,1,2],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3,2,1],bf16>
    return %5 : !torch.vtensor<[3,2,1],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_57(%arg0: !torch.vtensor<[3,2,1,2],bf16>) -> !torch.vtensor<[3,2,1],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_2_1_1_torch.uint8_2> : tensor<3x2x1x1xui8>) : !torch.vtensor<[3,2,1,1],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,2,1,1],ui8>, !torch.int -> !torch.vtensor<[3,2,1,1],i1>
    %2 = torch.vtensor.literal(dense<1.000000e+00> : tensor<bf16>) : !torch.vtensor<[],bf16>
    %none = torch.constant.none
    %3 = torch.aten.clone %1, %none : !torch.vtensor<[3,2,1,1],i1>, !torch.none -> !torch.vtensor<[3,2,1,1],i1>
    %int3 = torch.constant.int 3
    %int2 = torch.constant.int 2
    %int1 = torch.constant.int 1
    %int2_0 = torch.constant.int 2
    %4 = torch.prim.ListConstruct %int3, %int2, %int1, %int2_0 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %5 = torch.aten.expand %3, %4, %false : !torch.vtensor<[3,2,1,1],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3,2,1,2],i1>
    %none_1 = torch.constant.none
    %none_2 = torch.constant.none
    %int11_3 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_1, %none_2, %int11_3, %cpu, %int0 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int11_6 = torch.constant.int 11
    %cpu_7 = torch.constant.device "cpu"
    %int0_8 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_4, %none_5, %int11_6, %cpu_7, %int0_8 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_9 = torch.constant.none
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],bf16>, !torch.none -> !torch.vtensor<[],bf16>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],bf16>, !torch.vtensor<[],bf16> -> !torch.vtensor<[3,2,1,2],bf16>
    %none_10 = torch.constant.none
    %none_11 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu_12 = torch.constant.device "cpu"
    %int0_13 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %7, %none_10, %none_11, %int15, %cpu_12, %int0_13 : !torch.vtensor<[3,2,1,2],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int3_14 = torch.constant.int 3
    %false_15 = torch.constant.bool false
    %none_16 = torch.constant.none
    %8 = torch.aten.prod.dim_int %7, %int3_14, %false_15, %none_16 : !torch.vtensor<[3,2,1,2],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3,2,1],bf16>
    return %8 : !torch.vtensor<[3,2,1],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_58(%arg0: !torch.vtensor<[3,2,1,2],bf16>) -> !torch.vtensor<[3,2,1],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_1_1_2_torch.uint8_2> : tensor<3x1x1x2xui8>) : !torch.vtensor<[3,1,1,2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,1,1,2],ui8>, !torch.int -> !torch.vtensor<[3,1,1,2],i1>
    %2 = torch.vtensor.literal(dense<1.000000e+00> : tensor<bf16>) : !torch.vtensor<[],bf16>
    %none = torch.constant.none
    %3 = torch.aten.clone %1, %none : !torch.vtensor<[3,1,1,2],i1>, !torch.none -> !torch.vtensor<[3,1,1,2],i1>
    %int3 = torch.constant.int 3
    %int2 = torch.constant.int 2
    %int1 = torch.constant.int 1
    %int2_0 = torch.constant.int 2
    %4 = torch.prim.ListConstruct %int3, %int2, %int1, %int2_0 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %5 = torch.aten.expand %3, %4, %false : !torch.vtensor<[3,1,1,2],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3,2,1,2],i1>
    %none_1 = torch.constant.none
    %none_2 = torch.constant.none
    %int11_3 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_1, %none_2, %int11_3, %cpu, %int0 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int11_6 = torch.constant.int 11
    %cpu_7 = torch.constant.device "cpu"
    %int0_8 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_4, %none_5, %int11_6, %cpu_7, %int0_8 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_9 = torch.constant.none
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],bf16>, !torch.none -> !torch.vtensor<[],bf16>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],bf16>, !torch.vtensor<[],bf16> -> !torch.vtensor<[3,2,1,2],bf16>
    %none_10 = torch.constant.none
    %none_11 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu_12 = torch.constant.device "cpu"
    %int0_13 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %7, %none_10, %none_11, %int15, %cpu_12, %int0_13 : !torch.vtensor<[3,2,1,2],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int3_14 = torch.constant.int 3
    %false_15 = torch.constant.bool false
    %none_16 = torch.constant.none
    %8 = torch.aten.prod.dim_int %7, %int3_14, %false_15, %none_16 : !torch.vtensor<[3,2,1,2],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3,2,1],bf16>
    return %8 : !torch.vtensor<[3,2,1],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_59(%arg0: !torch.vtensor<[3,2,1,2],bf16>) -> !torch.vtensor<[3,2,1],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_1_2_1_2_torch.uint8_2> : tensor<1x2x1x2xui8>) : !torch.vtensor<[1,2,1,2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[1,2,1,2],ui8>, !torch.int -> !torch.vtensor<[1,2,1,2],i1>
    %2 = torch.vtensor.literal(dense<1.000000e+00> : tensor<bf16>) : !torch.vtensor<[],bf16>
    %none = torch.constant.none
    %3 = torch.aten.clone %1, %none : !torch.vtensor<[1,2,1,2],i1>, !torch.none -> !torch.vtensor<[1,2,1,2],i1>
    %int3 = torch.constant.int 3
    %int2 = torch.constant.int 2
    %int1 = torch.constant.int 1
    %int2_0 = torch.constant.int 2
    %4 = torch.prim.ListConstruct %int3, %int2, %int1, %int2_0 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %5 = torch.aten.expand %3, %4, %false : !torch.vtensor<[1,2,1,2],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3,2,1,2],i1>
    %none_1 = torch.constant.none
    %none_2 = torch.constant.none
    %int11_3 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_1, %none_2, %int11_3, %cpu, %int0 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int11_6 = torch.constant.int 11
    %cpu_7 = torch.constant.device "cpu"
    %int0_8 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_4, %none_5, %int11_6, %cpu_7, %int0_8 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_9 = torch.constant.none
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],bf16>, !torch.none -> !torch.vtensor<[],bf16>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],bf16>, !torch.vtensor<[],bf16> -> !torch.vtensor<[3,2,1,2],bf16>
    %none_10 = torch.constant.none
    %none_11 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu_12 = torch.constant.device "cpu"
    %int0_13 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %7, %none_10, %none_11, %int15, %cpu_12, %int0_13 : !torch.vtensor<[3,2,1,2],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int3_14 = torch.constant.int 3
    %false_15 = torch.constant.bool false
    %none_16 = torch.constant.none
    %8 = torch.aten.prod.dim_int %7, %int3_14, %false_15, %none_16 : !torch.vtensor<[3,2,1,2],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3,2,1],bf16>
    return %8 : !torch.vtensor<[3,2,1],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_60(%arg0: !torch.vtensor<[3,2,1,2],bf16>) -> !torch.vtensor<[3,2,1],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_2_1_2_torch.uint8_2> : tensor<2x1x2xui8>) : !torch.vtensor<[2,1,2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[2,1,2],ui8>, !torch.int -> !torch.vtensor<[2,1,2],i1>
    %2 = torch.vtensor.literal(dense<1.000000e+00> : tensor<bf16>) : !torch.vtensor<[],bf16>
    %none = torch.constant.none
    %3 = torch.aten.clone %1, %none : !torch.vtensor<[2,1,2],i1>, !torch.none -> !torch.vtensor<[2,1,2],i1>
    %int3 = torch.constant.int 3
    %int2 = torch.constant.int 2
    %int1 = torch.constant.int 1
    %int2_0 = torch.constant.int 2
    %4 = torch.prim.ListConstruct %int3, %int2, %int1, %int2_0 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %5 = torch.aten.expand %3, %4, %false : !torch.vtensor<[2,1,2],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3,2,1,2],i1>
    %none_1 = torch.constant.none
    %none_2 = torch.constant.none
    %int11_3 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_1, %none_2, %int11_3, %cpu, %int0 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int11_6 = torch.constant.int 11
    %cpu_7 = torch.constant.device "cpu"
    %int0_8 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_4, %none_5, %int11_6, %cpu_7, %int0_8 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_9 = torch.constant.none
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],bf16>, !torch.none -> !torch.vtensor<[],bf16>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],bf16>, !torch.vtensor<[],bf16> -> !torch.vtensor<[3,2,1,2],bf16>
    %none_10 = torch.constant.none
    %none_11 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu_12 = torch.constant.device "cpu"
    %int0_13 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %7, %none_10, %none_11, %int15, %cpu_12, %int0_13 : !torch.vtensor<[3,2,1,2],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int3_14 = torch.constant.int 3
    %false_15 = torch.constant.bool false
    %none_16 = torch.constant.none
    %8 = torch.aten.prod.dim_int %7, %int3_14, %false_15, %none_16 : !torch.vtensor<[3,2,1,2],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3,2,1],bf16>
    return %8 : !torch.vtensor<[3,2,1],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_61(%arg0: !torch.vtensor<[3,2,1,2],bf16>) -> !torch.vtensor<[3,2,1],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_2_torch.uint8_6> : tensor<2xui8>) : !torch.vtensor<[2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[2],ui8>, !torch.int -> !torch.vtensor<[2],i1>
    %2 = torch.vtensor.literal(dense<1.000000e+00> : tensor<bf16>) : !torch.vtensor<[],bf16>
    %none = torch.constant.none
    %3 = torch.aten.clone %1, %none : !torch.vtensor<[2],i1>, !torch.none -> !torch.vtensor<[2],i1>
    %int3 = torch.constant.int 3
    %int2 = torch.constant.int 2
    %int1 = torch.constant.int 1
    %int2_0 = torch.constant.int 2
    %4 = torch.prim.ListConstruct %int3, %int2, %int1, %int2_0 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %5 = torch.aten.expand %3, %4, %false : !torch.vtensor<[2],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3,2,1,2],i1>
    %none_1 = torch.constant.none
    %none_2 = torch.constant.none
    %int11_3 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_1, %none_2, %int11_3, %cpu, %int0 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int11_6 = torch.constant.int 11
    %cpu_7 = torch.constant.device "cpu"
    %int0_8 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_4, %none_5, %int11_6, %cpu_7, %int0_8 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_9 = torch.constant.none
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],bf16>, !torch.none -> !torch.vtensor<[],bf16>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],bf16>, !torch.vtensor<[],bf16> -> !torch.vtensor<[3,2,1,2],bf16>
    %none_10 = torch.constant.none
    %none_11 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu_12 = torch.constant.device "cpu"
    %int0_13 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %7, %none_10, %none_11, %int15, %cpu_12, %int0_13 : !torch.vtensor<[3,2,1,2],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int3_14 = torch.constant.int 3
    %false_15 = torch.constant.bool false
    %none_16 = torch.constant.none
    %8 = torch.aten.prod.dim_int %7, %int3_14, %false_15, %none_16 : !torch.vtensor<[3,2,1,2],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3,2,1],bf16>
    return %8 : !torch.vtensor<[3,2,1],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_62(%arg0: !torch.vtensor<[3,2,1,2],bf16>) -> !torch.vtensor<[3,2,1,2],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %arg0, %none, %none_0, %int15, %cpu, %int0 : !torch.vtensor<[3,2,1,2],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int2 = torch.constant.int 2
    %true = torch.constant.bool true
    %none_1 = torch.constant.none
    %0 = torch.aten.prod.dim_int %arg0, %int2, %true, %none_1 : !torch.vtensor<[3,2,1,2],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3,2,1,2],bf16>
    return %0 : !torch.vtensor<[3,2,1,2],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_63(%arg0: !torch.vtensor<[3,2,1,2],bf16>) -> !torch.vtensor<[3,2,1,2],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_2_1_2_torch.uint8_3> : tensor<3x2x1x2xui8>) : !torch.vtensor<[3,2,1,2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,2,1,2],ui8>, !torch.int -> !torch.vtensor<[3,2,1,2],i1>
    %2 = torch.vtensor.literal(dense<1.000000e+00> : tensor<bf16>) : !torch.vtensor<[],bf16>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],bf16>, !torch.none -> !torch.vtensor<[],bf16>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],bf16>, !torch.vtensor<[],bf16> -> !torch.vtensor<[3,2,1,2],bf16>
    %none_3 = torch.constant.none
    %none_4 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu_5 = torch.constant.device "cpu"
    %int0_6 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %4, %none_3, %none_4, %int15, %cpu_5, %int0_6 : !torch.vtensor<[3,2,1,2],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int2 = torch.constant.int 2
    %true = torch.constant.bool true
    %none_7 = torch.constant.none
    %5 = torch.aten.prod.dim_int %4, %int2, %true, %none_7 : !torch.vtensor<[3,2,1,2],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3,2,1,2],bf16>
    return %5 : !torch.vtensor<[3,2,1,2],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_64(%arg0: !torch.vtensor<[3,2,1,2],bf16>) -> !torch.vtensor<[3,2,1,2],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_2_1_1_torch.uint8_3> : tensor<3x2x1x1xui8>) : !torch.vtensor<[3,2,1,1],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,2,1,1],ui8>, !torch.int -> !torch.vtensor<[3,2,1,1],i1>
    %2 = torch.vtensor.literal(dense<1.000000e+00> : tensor<bf16>) : !torch.vtensor<[],bf16>
    %none = torch.constant.none
    %3 = torch.aten.clone %1, %none : !torch.vtensor<[3,2,1,1],i1>, !torch.none -> !torch.vtensor<[3,2,1,1],i1>
    %int3 = torch.constant.int 3
    %int2 = torch.constant.int 2
    %int1 = torch.constant.int 1
    %int2_0 = torch.constant.int 2
    %4 = torch.prim.ListConstruct %int3, %int2, %int1, %int2_0 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %5 = torch.aten.expand %3, %4, %false : !torch.vtensor<[3,2,1,1],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3,2,1,2],i1>
    %none_1 = torch.constant.none
    %none_2 = torch.constant.none
    %int11_3 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_1, %none_2, %int11_3, %cpu, %int0 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int11_6 = torch.constant.int 11
    %cpu_7 = torch.constant.device "cpu"
    %int0_8 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_4, %none_5, %int11_6, %cpu_7, %int0_8 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_9 = torch.constant.none
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],bf16>, !torch.none -> !torch.vtensor<[],bf16>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],bf16>, !torch.vtensor<[],bf16> -> !torch.vtensor<[3,2,1,2],bf16>
    %none_10 = torch.constant.none
    %none_11 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu_12 = torch.constant.device "cpu"
    %int0_13 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %7, %none_10, %none_11, %int15, %cpu_12, %int0_13 : !torch.vtensor<[3,2,1,2],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int2_14 = torch.constant.int 2
    %true = torch.constant.bool true
    %none_15 = torch.constant.none
    %8 = torch.aten.prod.dim_int %7, %int2_14, %true, %none_15 : !torch.vtensor<[3,2,1,2],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3,2,1,2],bf16>
    return %8 : !torch.vtensor<[3,2,1,2],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_65(%arg0: !torch.vtensor<[3,2,1,2],bf16>) -> !torch.vtensor<[3,2,1,2],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_1_1_2_torch.uint8_3> : tensor<3x1x1x2xui8>) : !torch.vtensor<[3,1,1,2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,1,1,2],ui8>, !torch.int -> !torch.vtensor<[3,1,1,2],i1>
    %2 = torch.vtensor.literal(dense<1.000000e+00> : tensor<bf16>) : !torch.vtensor<[],bf16>
    %none = torch.constant.none
    %3 = torch.aten.clone %1, %none : !torch.vtensor<[3,1,1,2],i1>, !torch.none -> !torch.vtensor<[3,1,1,2],i1>
    %int3 = torch.constant.int 3
    %int2 = torch.constant.int 2
    %int1 = torch.constant.int 1
    %int2_0 = torch.constant.int 2
    %4 = torch.prim.ListConstruct %int3, %int2, %int1, %int2_0 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %5 = torch.aten.expand %3, %4, %false : !torch.vtensor<[3,1,1,2],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3,2,1,2],i1>
    %none_1 = torch.constant.none
    %none_2 = torch.constant.none
    %int11_3 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_1, %none_2, %int11_3, %cpu, %int0 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int11_6 = torch.constant.int 11
    %cpu_7 = torch.constant.device "cpu"
    %int0_8 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_4, %none_5, %int11_6, %cpu_7, %int0_8 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_9 = torch.constant.none
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],bf16>, !torch.none -> !torch.vtensor<[],bf16>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],bf16>, !torch.vtensor<[],bf16> -> !torch.vtensor<[3,2,1,2],bf16>
    %none_10 = torch.constant.none
    %none_11 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu_12 = torch.constant.device "cpu"
    %int0_13 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %7, %none_10, %none_11, %int15, %cpu_12, %int0_13 : !torch.vtensor<[3,2,1,2],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int2_14 = torch.constant.int 2
    %true = torch.constant.bool true
    %none_15 = torch.constant.none
    %8 = torch.aten.prod.dim_int %7, %int2_14, %true, %none_15 : !torch.vtensor<[3,2,1,2],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3,2,1,2],bf16>
    return %8 : !torch.vtensor<[3,2,1,2],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_66(%arg0: !torch.vtensor<[3,2,1,2],bf16>) -> !torch.vtensor<[3,2,1,2],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_1_2_1_2_torch.uint8_3> : tensor<1x2x1x2xui8>) : !torch.vtensor<[1,2,1,2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[1,2,1,2],ui8>, !torch.int -> !torch.vtensor<[1,2,1,2],i1>
    %2 = torch.vtensor.literal(dense<1.000000e+00> : tensor<bf16>) : !torch.vtensor<[],bf16>
    %none = torch.constant.none
    %3 = torch.aten.clone %1, %none : !torch.vtensor<[1,2,1,2],i1>, !torch.none -> !torch.vtensor<[1,2,1,2],i1>
    %int3 = torch.constant.int 3
    %int2 = torch.constant.int 2
    %int1 = torch.constant.int 1
    %int2_0 = torch.constant.int 2
    %4 = torch.prim.ListConstruct %int3, %int2, %int1, %int2_0 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %5 = torch.aten.expand %3, %4, %false : !torch.vtensor<[1,2,1,2],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3,2,1,2],i1>
    %none_1 = torch.constant.none
    %none_2 = torch.constant.none
    %int11_3 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_1, %none_2, %int11_3, %cpu, %int0 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int11_6 = torch.constant.int 11
    %cpu_7 = torch.constant.device "cpu"
    %int0_8 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_4, %none_5, %int11_6, %cpu_7, %int0_8 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_9 = torch.constant.none
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],bf16>, !torch.none -> !torch.vtensor<[],bf16>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],bf16>, !torch.vtensor<[],bf16> -> !torch.vtensor<[3,2,1,2],bf16>
    %none_10 = torch.constant.none
    %none_11 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu_12 = torch.constant.device "cpu"
    %int0_13 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %7, %none_10, %none_11, %int15, %cpu_12, %int0_13 : !torch.vtensor<[3,2,1,2],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int2_14 = torch.constant.int 2
    %true = torch.constant.bool true
    %none_15 = torch.constant.none
    %8 = torch.aten.prod.dim_int %7, %int2_14, %true, %none_15 : !torch.vtensor<[3,2,1,2],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3,2,1,2],bf16>
    return %8 : !torch.vtensor<[3,2,1,2],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_67(%arg0: !torch.vtensor<[3,2,1,2],bf16>) -> !torch.vtensor<[3,2,1,2],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_2_1_2_torch.uint8_3> : tensor<2x1x2xui8>) : !torch.vtensor<[2,1,2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[2,1,2],ui8>, !torch.int -> !torch.vtensor<[2,1,2],i1>
    %2 = torch.vtensor.literal(dense<1.000000e+00> : tensor<bf16>) : !torch.vtensor<[],bf16>
    %none = torch.constant.none
    %3 = torch.aten.clone %1, %none : !torch.vtensor<[2,1,2],i1>, !torch.none -> !torch.vtensor<[2,1,2],i1>
    %int3 = torch.constant.int 3
    %int2 = torch.constant.int 2
    %int1 = torch.constant.int 1
    %int2_0 = torch.constant.int 2
    %4 = torch.prim.ListConstruct %int3, %int2, %int1, %int2_0 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %5 = torch.aten.expand %3, %4, %false : !torch.vtensor<[2,1,2],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3,2,1,2],i1>
    %none_1 = torch.constant.none
    %none_2 = torch.constant.none
    %int11_3 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_1, %none_2, %int11_3, %cpu, %int0 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int11_6 = torch.constant.int 11
    %cpu_7 = torch.constant.device "cpu"
    %int0_8 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_4, %none_5, %int11_6, %cpu_7, %int0_8 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_9 = torch.constant.none
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],bf16>, !torch.none -> !torch.vtensor<[],bf16>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],bf16>, !torch.vtensor<[],bf16> -> !torch.vtensor<[3,2,1,2],bf16>
    %none_10 = torch.constant.none
    %none_11 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu_12 = torch.constant.device "cpu"
    %int0_13 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %7, %none_10, %none_11, %int15, %cpu_12, %int0_13 : !torch.vtensor<[3,2,1,2],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int2_14 = torch.constant.int 2
    %true = torch.constant.bool true
    %none_15 = torch.constant.none
    %8 = torch.aten.prod.dim_int %7, %int2_14, %true, %none_15 : !torch.vtensor<[3,2,1,2],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3,2,1,2],bf16>
    return %8 : !torch.vtensor<[3,2,1,2],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_68(%arg0: !torch.vtensor<[3,2,1,2],bf16>) -> !torch.vtensor<[3,2,1,2],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_2_torch.uint8_7> : tensor<2xui8>) : !torch.vtensor<[2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[2],ui8>, !torch.int -> !torch.vtensor<[2],i1>
    %2 = torch.vtensor.literal(dense<1.000000e+00> : tensor<bf16>) : !torch.vtensor<[],bf16>
    %none = torch.constant.none
    %3 = torch.aten.clone %1, %none : !torch.vtensor<[2],i1>, !torch.none -> !torch.vtensor<[2],i1>
    %int3 = torch.constant.int 3
    %int2 = torch.constant.int 2
    %int1 = torch.constant.int 1
    %int2_0 = torch.constant.int 2
    %4 = torch.prim.ListConstruct %int3, %int2, %int1, %int2_0 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %5 = torch.aten.expand %3, %4, %false : !torch.vtensor<[2],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3,2,1,2],i1>
    %none_1 = torch.constant.none
    %none_2 = torch.constant.none
    %int11_3 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_1, %none_2, %int11_3, %cpu, %int0 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int11_6 = torch.constant.int 11
    %cpu_7 = torch.constant.device "cpu"
    %int0_8 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_4, %none_5, %int11_6, %cpu_7, %int0_8 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_9 = torch.constant.none
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],bf16>, !torch.none -> !torch.vtensor<[],bf16>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],bf16>, !torch.vtensor<[],bf16> -> !torch.vtensor<[3,2,1,2],bf16>
    %none_10 = torch.constant.none
    %none_11 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu_12 = torch.constant.device "cpu"
    %int0_13 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %7, %none_10, %none_11, %int15, %cpu_12, %int0_13 : !torch.vtensor<[3,2,1,2],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int2_14 = torch.constant.int 2
    %true = torch.constant.bool true
    %none_15 = torch.constant.none
    %8 = torch.aten.prod.dim_int %7, %int2_14, %true, %none_15 : !torch.vtensor<[3,2,1,2],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3,2,1,2],bf16>
    return %8 : !torch.vtensor<[3,2,1,2],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_69(%arg0: !torch.vtensor<[3,2,1,2],bf16>) -> !torch.vtensor<[],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %arg0, %none, %none_0, %int15, %cpu, %int0 : !torch.vtensor<[3,2,1,2],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int3 = torch.constant.int 3
    %false = torch.constant.bool false
    %none_1 = torch.constant.none
    %0 = torch.aten.prod.dim_int %arg0, %int3, %false, %none_1 : !torch.vtensor<[3,2,1,2],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3,2,1],bf16>
    %int2 = torch.constant.int 2
    %false_2 = torch.constant.bool false
    %none_3 = torch.constant.none
    %1 = torch.aten.prod.dim_int %0, %int2, %false_2, %none_3 : !torch.vtensor<[3,2,1],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3,2],bf16>
    %int1 = torch.constant.int 1
    %false_4 = torch.constant.bool false
    %none_5 = torch.constant.none
    %2 = torch.aten.prod.dim_int %1, %int1, %false_4, %none_5 : !torch.vtensor<[3,2],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3],bf16>
    %int0_6 = torch.constant.int 0
    %false_7 = torch.constant.bool false
    %none_8 = torch.constant.none
    %3 = torch.aten.prod.dim_int %2, %int0_6, %false_7, %none_8 : !torch.vtensor<[3],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[],bf16>
    return %3 : !torch.vtensor<[],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_70(%arg0: !torch.vtensor<[3,2,1,2],bf16>) -> !torch.vtensor<[],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_2_1_2_torch.uint8_4> : tensor<3x2x1x2xui8>) : !torch.vtensor<[3,2,1,2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,2,1,2],ui8>, !torch.int -> !torch.vtensor<[3,2,1,2],i1>
    %2 = torch.vtensor.literal(dense<1.000000e+00> : tensor<bf16>) : !torch.vtensor<[],bf16>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],bf16>, !torch.none -> !torch.vtensor<[],bf16>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],bf16>, !torch.vtensor<[],bf16> -> !torch.vtensor<[3,2,1,2],bf16>
    %none_3 = torch.constant.none
    %none_4 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu_5 = torch.constant.device "cpu"
    %int0_6 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %4, %none_3, %none_4, %int15, %cpu_5, %int0_6 : !torch.vtensor<[3,2,1,2],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int3 = torch.constant.int 3
    %false = torch.constant.bool false
    %none_7 = torch.constant.none
    %5 = torch.aten.prod.dim_int %4, %int3, %false, %none_7 : !torch.vtensor<[3,2,1,2],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3,2,1],bf16>
    %int2 = torch.constant.int 2
    %false_8 = torch.constant.bool false
    %none_9 = torch.constant.none
    %6 = torch.aten.prod.dim_int %5, %int2, %false_8, %none_9 : !torch.vtensor<[3,2,1],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3,2],bf16>
    %int1 = torch.constant.int 1
    %false_10 = torch.constant.bool false
    %none_11 = torch.constant.none
    %7 = torch.aten.prod.dim_int %6, %int1, %false_10, %none_11 : !torch.vtensor<[3,2],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3],bf16>
    %int0_12 = torch.constant.int 0
    %false_13 = torch.constant.bool false
    %none_14 = torch.constant.none
    %8 = torch.aten.prod.dim_int %7, %int0_12, %false_13, %none_14 : !torch.vtensor<[3],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[],bf16>
    return %8 : !torch.vtensor<[],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_71(%arg0: !torch.vtensor<[3,2,1,2],bf16>) -> !torch.vtensor<[],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_2_1_1_torch.uint8_4> : tensor<3x2x1x1xui8>) : !torch.vtensor<[3,2,1,1],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,2,1,1],ui8>, !torch.int -> !torch.vtensor<[3,2,1,1],i1>
    %2 = torch.vtensor.literal(dense<1.000000e+00> : tensor<bf16>) : !torch.vtensor<[],bf16>
    %none = torch.constant.none
    %3 = torch.aten.clone %1, %none : !torch.vtensor<[3,2,1,1],i1>, !torch.none -> !torch.vtensor<[3,2,1,1],i1>
    %int3 = torch.constant.int 3
    %int2 = torch.constant.int 2
    %int1 = torch.constant.int 1
    %int2_0 = torch.constant.int 2
    %4 = torch.prim.ListConstruct %int3, %int2, %int1, %int2_0 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %5 = torch.aten.expand %3, %4, %false : !torch.vtensor<[3,2,1,1],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3,2,1,2],i1>
    %none_1 = torch.constant.none
    %none_2 = torch.constant.none
    %int11_3 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_1, %none_2, %int11_3, %cpu, %int0 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int11_6 = torch.constant.int 11
    %cpu_7 = torch.constant.device "cpu"
    %int0_8 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_4, %none_5, %int11_6, %cpu_7, %int0_8 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_9 = torch.constant.none
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],bf16>, !torch.none -> !torch.vtensor<[],bf16>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],bf16>, !torch.vtensor<[],bf16> -> !torch.vtensor<[3,2,1,2],bf16>
    %none_10 = torch.constant.none
    %none_11 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu_12 = torch.constant.device "cpu"
    %int0_13 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %7, %none_10, %none_11, %int15, %cpu_12, %int0_13 : !torch.vtensor<[3,2,1,2],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int3_14 = torch.constant.int 3
    %false_15 = torch.constant.bool false
    %none_16 = torch.constant.none
    %8 = torch.aten.prod.dim_int %7, %int3_14, %false_15, %none_16 : !torch.vtensor<[3,2,1,2],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3,2,1],bf16>
    %int2_17 = torch.constant.int 2
    %false_18 = torch.constant.bool false
    %none_19 = torch.constant.none
    %9 = torch.aten.prod.dim_int %8, %int2_17, %false_18, %none_19 : !torch.vtensor<[3,2,1],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3,2],bf16>
    %int1_20 = torch.constant.int 1
    %false_21 = torch.constant.bool false
    %none_22 = torch.constant.none
    %10 = torch.aten.prod.dim_int %9, %int1_20, %false_21, %none_22 : !torch.vtensor<[3,2],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3],bf16>
    %int0_23 = torch.constant.int 0
    %false_24 = torch.constant.bool false
    %none_25 = torch.constant.none
    %11 = torch.aten.prod.dim_int %10, %int0_23, %false_24, %none_25 : !torch.vtensor<[3],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[],bf16>
    return %11 : !torch.vtensor<[],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_72(%arg0: !torch.vtensor<[3,2,1,2],bf16>) -> !torch.vtensor<[],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_1_1_2_torch.uint8_4> : tensor<3x1x1x2xui8>) : !torch.vtensor<[3,1,1,2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,1,1,2],ui8>, !torch.int -> !torch.vtensor<[3,1,1,2],i1>
    %2 = torch.vtensor.literal(dense<1.000000e+00> : tensor<bf16>) : !torch.vtensor<[],bf16>
    %none = torch.constant.none
    %3 = torch.aten.clone %1, %none : !torch.vtensor<[3,1,1,2],i1>, !torch.none -> !torch.vtensor<[3,1,1,2],i1>
    %int3 = torch.constant.int 3
    %int2 = torch.constant.int 2
    %int1 = torch.constant.int 1
    %int2_0 = torch.constant.int 2
    %4 = torch.prim.ListConstruct %int3, %int2, %int1, %int2_0 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %5 = torch.aten.expand %3, %4, %false : !torch.vtensor<[3,1,1,2],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3,2,1,2],i1>
    %none_1 = torch.constant.none
    %none_2 = torch.constant.none
    %int11_3 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_1, %none_2, %int11_3, %cpu, %int0 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int11_6 = torch.constant.int 11
    %cpu_7 = torch.constant.device "cpu"
    %int0_8 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_4, %none_5, %int11_6, %cpu_7, %int0_8 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_9 = torch.constant.none
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],bf16>, !torch.none -> !torch.vtensor<[],bf16>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],bf16>, !torch.vtensor<[],bf16> -> !torch.vtensor<[3,2,1,2],bf16>
    %none_10 = torch.constant.none
    %none_11 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu_12 = torch.constant.device "cpu"
    %int0_13 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %7, %none_10, %none_11, %int15, %cpu_12, %int0_13 : !torch.vtensor<[3,2,1,2],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int3_14 = torch.constant.int 3
    %false_15 = torch.constant.bool false
    %none_16 = torch.constant.none
    %8 = torch.aten.prod.dim_int %7, %int3_14, %false_15, %none_16 : !torch.vtensor<[3,2,1,2],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3,2,1],bf16>
    %int2_17 = torch.constant.int 2
    %false_18 = torch.constant.bool false
    %none_19 = torch.constant.none
    %9 = torch.aten.prod.dim_int %8, %int2_17, %false_18, %none_19 : !torch.vtensor<[3,2,1],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3,2],bf16>
    %int1_20 = torch.constant.int 1
    %false_21 = torch.constant.bool false
    %none_22 = torch.constant.none
    %10 = torch.aten.prod.dim_int %9, %int1_20, %false_21, %none_22 : !torch.vtensor<[3,2],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3],bf16>
    %int0_23 = torch.constant.int 0
    %false_24 = torch.constant.bool false
    %none_25 = torch.constant.none
    %11 = torch.aten.prod.dim_int %10, %int0_23, %false_24, %none_25 : !torch.vtensor<[3],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[],bf16>
    return %11 : !torch.vtensor<[],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_73(%arg0: !torch.vtensor<[3,2,1,2],bf16>) -> !torch.vtensor<[],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_1_2_1_2_torch.uint8_4> : tensor<1x2x1x2xui8>) : !torch.vtensor<[1,2,1,2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[1,2,1,2],ui8>, !torch.int -> !torch.vtensor<[1,2,1,2],i1>
    %2 = torch.vtensor.literal(dense<1.000000e+00> : tensor<bf16>) : !torch.vtensor<[],bf16>
    %none = torch.constant.none
    %3 = torch.aten.clone %1, %none : !torch.vtensor<[1,2,1,2],i1>, !torch.none -> !torch.vtensor<[1,2,1,2],i1>
    %int3 = torch.constant.int 3
    %int2 = torch.constant.int 2
    %int1 = torch.constant.int 1
    %int2_0 = torch.constant.int 2
    %4 = torch.prim.ListConstruct %int3, %int2, %int1, %int2_0 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %5 = torch.aten.expand %3, %4, %false : !torch.vtensor<[1,2,1,2],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3,2,1,2],i1>
    %none_1 = torch.constant.none
    %none_2 = torch.constant.none
    %int11_3 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_1, %none_2, %int11_3, %cpu, %int0 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int11_6 = torch.constant.int 11
    %cpu_7 = torch.constant.device "cpu"
    %int0_8 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_4, %none_5, %int11_6, %cpu_7, %int0_8 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_9 = torch.constant.none
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],bf16>, !torch.none -> !torch.vtensor<[],bf16>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],bf16>, !torch.vtensor<[],bf16> -> !torch.vtensor<[3,2,1,2],bf16>
    %none_10 = torch.constant.none
    %none_11 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu_12 = torch.constant.device "cpu"
    %int0_13 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %7, %none_10, %none_11, %int15, %cpu_12, %int0_13 : !torch.vtensor<[3,2,1,2],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int3_14 = torch.constant.int 3
    %false_15 = torch.constant.bool false
    %none_16 = torch.constant.none
    %8 = torch.aten.prod.dim_int %7, %int3_14, %false_15, %none_16 : !torch.vtensor<[3,2,1,2],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3,2,1],bf16>
    %int2_17 = torch.constant.int 2
    %false_18 = torch.constant.bool false
    %none_19 = torch.constant.none
    %9 = torch.aten.prod.dim_int %8, %int2_17, %false_18, %none_19 : !torch.vtensor<[3,2,1],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3,2],bf16>
    %int1_20 = torch.constant.int 1
    %false_21 = torch.constant.bool false
    %none_22 = torch.constant.none
    %10 = torch.aten.prod.dim_int %9, %int1_20, %false_21, %none_22 : !torch.vtensor<[3,2],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3],bf16>
    %int0_23 = torch.constant.int 0
    %false_24 = torch.constant.bool false
    %none_25 = torch.constant.none
    %11 = torch.aten.prod.dim_int %10, %int0_23, %false_24, %none_25 : !torch.vtensor<[3],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[],bf16>
    return %11 : !torch.vtensor<[],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_74(%arg0: !torch.vtensor<[3,2,1,2],bf16>) -> !torch.vtensor<[],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_2_1_2_torch.uint8_4> : tensor<2x1x2xui8>) : !torch.vtensor<[2,1,2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[2,1,2],ui8>, !torch.int -> !torch.vtensor<[2,1,2],i1>
    %2 = torch.vtensor.literal(dense<1.000000e+00> : tensor<bf16>) : !torch.vtensor<[],bf16>
    %none = torch.constant.none
    %3 = torch.aten.clone %1, %none : !torch.vtensor<[2,1,2],i1>, !torch.none -> !torch.vtensor<[2,1,2],i1>
    %int3 = torch.constant.int 3
    %int2 = torch.constant.int 2
    %int1 = torch.constant.int 1
    %int2_0 = torch.constant.int 2
    %4 = torch.prim.ListConstruct %int3, %int2, %int1, %int2_0 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %5 = torch.aten.expand %3, %4, %false : !torch.vtensor<[2,1,2],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3,2,1,2],i1>
    %none_1 = torch.constant.none
    %none_2 = torch.constant.none
    %int11_3 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_1, %none_2, %int11_3, %cpu, %int0 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int11_6 = torch.constant.int 11
    %cpu_7 = torch.constant.device "cpu"
    %int0_8 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_4, %none_5, %int11_6, %cpu_7, %int0_8 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_9 = torch.constant.none
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],bf16>, !torch.none -> !torch.vtensor<[],bf16>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],bf16>, !torch.vtensor<[],bf16> -> !torch.vtensor<[3,2,1,2],bf16>
    %none_10 = torch.constant.none
    %none_11 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu_12 = torch.constant.device "cpu"
    %int0_13 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %7, %none_10, %none_11, %int15, %cpu_12, %int0_13 : !torch.vtensor<[3,2,1,2],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int3_14 = torch.constant.int 3
    %false_15 = torch.constant.bool false
    %none_16 = torch.constant.none
    %8 = torch.aten.prod.dim_int %7, %int3_14, %false_15, %none_16 : !torch.vtensor<[3,2,1,2],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3,2,1],bf16>
    %int2_17 = torch.constant.int 2
    %false_18 = torch.constant.bool false
    %none_19 = torch.constant.none
    %9 = torch.aten.prod.dim_int %8, %int2_17, %false_18, %none_19 : !torch.vtensor<[3,2,1],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3,2],bf16>
    %int1_20 = torch.constant.int 1
    %false_21 = torch.constant.bool false
    %none_22 = torch.constant.none
    %10 = torch.aten.prod.dim_int %9, %int1_20, %false_21, %none_22 : !torch.vtensor<[3,2],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3],bf16>
    %int0_23 = torch.constant.int 0
    %false_24 = torch.constant.bool false
    %none_25 = torch.constant.none
    %11 = torch.aten.prod.dim_int %10, %int0_23, %false_24, %none_25 : !torch.vtensor<[3],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[],bf16>
    return %11 : !torch.vtensor<[],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_75(%arg0: !torch.vtensor<[3,2,1,2],bf16>) -> !torch.vtensor<[],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_2_torch.uint8_8> : tensor<2xui8>) : !torch.vtensor<[2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[2],ui8>, !torch.int -> !torch.vtensor<[2],i1>
    %2 = torch.vtensor.literal(dense<1.000000e+00> : tensor<bf16>) : !torch.vtensor<[],bf16>
    %none = torch.constant.none
    %3 = torch.aten.clone %1, %none : !torch.vtensor<[2],i1>, !torch.none -> !torch.vtensor<[2],i1>
    %int3 = torch.constant.int 3
    %int2 = torch.constant.int 2
    %int1 = torch.constant.int 1
    %int2_0 = torch.constant.int 2
    %4 = torch.prim.ListConstruct %int3, %int2, %int1, %int2_0 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %5 = torch.aten.expand %3, %4, %false : !torch.vtensor<[2],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3,2,1,2],i1>
    %none_1 = torch.constant.none
    %none_2 = torch.constant.none
    %int11_3 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_1, %none_2, %int11_3, %cpu, %int0 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int11_6 = torch.constant.int 11
    %cpu_7 = torch.constant.device "cpu"
    %int0_8 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_4, %none_5, %int11_6, %cpu_7, %int0_8 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_9 = torch.constant.none
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],bf16>, !torch.none -> !torch.vtensor<[],bf16>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],bf16>, !torch.vtensor<[],bf16> -> !torch.vtensor<[3,2,1,2],bf16>
    %none_10 = torch.constant.none
    %none_11 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu_12 = torch.constant.device "cpu"
    %int0_13 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %7, %none_10, %none_11, %int15, %cpu_12, %int0_13 : !torch.vtensor<[3,2,1,2],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int3_14 = torch.constant.int 3
    %false_15 = torch.constant.bool false
    %none_16 = torch.constant.none
    %8 = torch.aten.prod.dim_int %7, %int3_14, %false_15, %none_16 : !torch.vtensor<[3,2,1,2],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3,2,1],bf16>
    %int2_17 = torch.constant.int 2
    %false_18 = torch.constant.bool false
    %none_19 = torch.constant.none
    %9 = torch.aten.prod.dim_int %8, %int2_17, %false_18, %none_19 : !torch.vtensor<[3,2,1],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3,2],bf16>
    %int1_20 = torch.constant.int 1
    %false_21 = torch.constant.bool false
    %none_22 = torch.constant.none
    %10 = torch.aten.prod.dim_int %9, %int1_20, %false_21, %none_22 : !torch.vtensor<[3,2],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3],bf16>
    %int0_23 = torch.constant.int 0
    %false_24 = torch.constant.bool false
    %none_25 = torch.constant.none
    %11 = torch.aten.prod.dim_int %10, %int0_23, %false_24, %none_25 : !torch.vtensor<[3],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[],bf16>
    return %11 : !torch.vtensor<[],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_76(%arg0: !torch.vtensor<[3,2,1,2],bf16>) -> !torch.vtensor<[1,2,1,1],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %arg0, %none, %none_0, %int15, %cpu, %int0 : !torch.vtensor<[3,2,1,2],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int3 = torch.constant.int 3
    %true = torch.constant.bool true
    %none_1 = torch.constant.none
    %0 = torch.aten.prod.dim_int %arg0, %int3, %true, %none_1 : !torch.vtensor<[3,2,1,2],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3,2,1,1],bf16>
    %int0_2 = torch.constant.int 0
    %true_3 = torch.constant.bool true
    %none_4 = torch.constant.none
    %1 = torch.aten.prod.dim_int %0, %int0_2, %true_3, %none_4 : !torch.vtensor<[3,2,1,1],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[1,2,1,1],bf16>
    return %1 : !torch.vtensor<[1,2,1,1],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_77(%arg0: !torch.vtensor<[3,2,1,2],bf16>) -> !torch.vtensor<[1,2,1,1],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_2_1_2_torch.uint8_5> : tensor<3x2x1x2xui8>) : !torch.vtensor<[3,2,1,2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,2,1,2],ui8>, !torch.int -> !torch.vtensor<[3,2,1,2],i1>
    %2 = torch.vtensor.literal(dense<1.000000e+00> : tensor<bf16>) : !torch.vtensor<[],bf16>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],bf16>, !torch.none -> !torch.vtensor<[],bf16>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],bf16>, !torch.vtensor<[],bf16> -> !torch.vtensor<[3,2,1,2],bf16>
    %none_3 = torch.constant.none
    %none_4 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu_5 = torch.constant.device "cpu"
    %int0_6 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %4, %none_3, %none_4, %int15, %cpu_5, %int0_6 : !torch.vtensor<[3,2,1,2],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int3 = torch.constant.int 3
    %true = torch.constant.bool true
    %none_7 = torch.constant.none
    %5 = torch.aten.prod.dim_int %4, %int3, %true, %none_7 : !torch.vtensor<[3,2,1,2],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3,2,1,1],bf16>
    %int0_8 = torch.constant.int 0
    %true_9 = torch.constant.bool true
    %none_10 = torch.constant.none
    %6 = torch.aten.prod.dim_int %5, %int0_8, %true_9, %none_10 : !torch.vtensor<[3,2,1,1],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[1,2,1,1],bf16>
    return %6 : !torch.vtensor<[1,2,1,1],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_78(%arg0: !torch.vtensor<[3,2,1,2],bf16>) -> !torch.vtensor<[1,2,1,1],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_2_1_1_torch.uint8_5> : tensor<3x2x1x1xui8>) : !torch.vtensor<[3,2,1,1],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,2,1,1],ui8>, !torch.int -> !torch.vtensor<[3,2,1,1],i1>
    %2 = torch.vtensor.literal(dense<1.000000e+00> : tensor<bf16>) : !torch.vtensor<[],bf16>
    %none = torch.constant.none
    %3 = torch.aten.clone %1, %none : !torch.vtensor<[3,2,1,1],i1>, !torch.none -> !torch.vtensor<[3,2,1,1],i1>
    %int3 = torch.constant.int 3
    %int2 = torch.constant.int 2
    %int1 = torch.constant.int 1
    %int2_0 = torch.constant.int 2
    %4 = torch.prim.ListConstruct %int3, %int2, %int1, %int2_0 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %5 = torch.aten.expand %3, %4, %false : !torch.vtensor<[3,2,1,1],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3,2,1,2],i1>
    %none_1 = torch.constant.none
    %none_2 = torch.constant.none
    %int11_3 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_1, %none_2, %int11_3, %cpu, %int0 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int11_6 = torch.constant.int 11
    %cpu_7 = torch.constant.device "cpu"
    %int0_8 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_4, %none_5, %int11_6, %cpu_7, %int0_8 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_9 = torch.constant.none
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],bf16>, !torch.none -> !torch.vtensor<[],bf16>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],bf16>, !torch.vtensor<[],bf16> -> !torch.vtensor<[3,2,1,2],bf16>
    %none_10 = torch.constant.none
    %none_11 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu_12 = torch.constant.device "cpu"
    %int0_13 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %7, %none_10, %none_11, %int15, %cpu_12, %int0_13 : !torch.vtensor<[3,2,1,2],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int3_14 = torch.constant.int 3
    %true = torch.constant.bool true
    %none_15 = torch.constant.none
    %8 = torch.aten.prod.dim_int %7, %int3_14, %true, %none_15 : !torch.vtensor<[3,2,1,2],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3,2,1,1],bf16>
    %int0_16 = torch.constant.int 0
    %true_17 = torch.constant.bool true
    %none_18 = torch.constant.none
    %9 = torch.aten.prod.dim_int %8, %int0_16, %true_17, %none_18 : !torch.vtensor<[3,2,1,1],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[1,2,1,1],bf16>
    return %9 : !torch.vtensor<[1,2,1,1],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_79(%arg0: !torch.vtensor<[3,2,1,2],bf16>) -> !torch.vtensor<[1,2,1,1],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_1_1_2_torch.uint8_5> : tensor<3x1x1x2xui8>) : !torch.vtensor<[3,1,1,2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,1,1,2],ui8>, !torch.int -> !torch.vtensor<[3,1,1,2],i1>
    %2 = torch.vtensor.literal(dense<1.000000e+00> : tensor<bf16>) : !torch.vtensor<[],bf16>
    %none = torch.constant.none
    %3 = torch.aten.clone %1, %none : !torch.vtensor<[3,1,1,2],i1>, !torch.none -> !torch.vtensor<[3,1,1,2],i1>
    %int3 = torch.constant.int 3
    %int2 = torch.constant.int 2
    %int1 = torch.constant.int 1
    %int2_0 = torch.constant.int 2
    %4 = torch.prim.ListConstruct %int3, %int2, %int1, %int2_0 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %5 = torch.aten.expand %3, %4, %false : !torch.vtensor<[3,1,1,2],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3,2,1,2],i1>
    %none_1 = torch.constant.none
    %none_2 = torch.constant.none
    %int11_3 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_1, %none_2, %int11_3, %cpu, %int0 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int11_6 = torch.constant.int 11
    %cpu_7 = torch.constant.device "cpu"
    %int0_8 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_4, %none_5, %int11_6, %cpu_7, %int0_8 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_9 = torch.constant.none
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],bf16>, !torch.none -> !torch.vtensor<[],bf16>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],bf16>, !torch.vtensor<[],bf16> -> !torch.vtensor<[3,2,1,2],bf16>
    %none_10 = torch.constant.none
    %none_11 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu_12 = torch.constant.device "cpu"
    %int0_13 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %7, %none_10, %none_11, %int15, %cpu_12, %int0_13 : !torch.vtensor<[3,2,1,2],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int3_14 = torch.constant.int 3
    %true = torch.constant.bool true
    %none_15 = torch.constant.none
    %8 = torch.aten.prod.dim_int %7, %int3_14, %true, %none_15 : !torch.vtensor<[3,2,1,2],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3,2,1,1],bf16>
    %int0_16 = torch.constant.int 0
    %true_17 = torch.constant.bool true
    %none_18 = torch.constant.none
    %9 = torch.aten.prod.dim_int %8, %int0_16, %true_17, %none_18 : !torch.vtensor<[3,2,1,1],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[1,2,1,1],bf16>
    return %9 : !torch.vtensor<[1,2,1,1],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_80(%arg0: !torch.vtensor<[3,2,1,2],bf16>) -> !torch.vtensor<[1,2,1,1],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_1_2_1_2_torch.uint8_5> : tensor<1x2x1x2xui8>) : !torch.vtensor<[1,2,1,2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[1,2,1,2],ui8>, !torch.int -> !torch.vtensor<[1,2,1,2],i1>
    %2 = torch.vtensor.literal(dense<1.000000e+00> : tensor<bf16>) : !torch.vtensor<[],bf16>
    %none = torch.constant.none
    %3 = torch.aten.clone %1, %none : !torch.vtensor<[1,2,1,2],i1>, !torch.none -> !torch.vtensor<[1,2,1,2],i1>
    %int3 = torch.constant.int 3
    %int2 = torch.constant.int 2
    %int1 = torch.constant.int 1
    %int2_0 = torch.constant.int 2
    %4 = torch.prim.ListConstruct %int3, %int2, %int1, %int2_0 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %5 = torch.aten.expand %3, %4, %false : !torch.vtensor<[1,2,1,2],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3,2,1,2],i1>
    %none_1 = torch.constant.none
    %none_2 = torch.constant.none
    %int11_3 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_1, %none_2, %int11_3, %cpu, %int0 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int11_6 = torch.constant.int 11
    %cpu_7 = torch.constant.device "cpu"
    %int0_8 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_4, %none_5, %int11_6, %cpu_7, %int0_8 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_9 = torch.constant.none
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],bf16>, !torch.none -> !torch.vtensor<[],bf16>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],bf16>, !torch.vtensor<[],bf16> -> !torch.vtensor<[3,2,1,2],bf16>
    %none_10 = torch.constant.none
    %none_11 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu_12 = torch.constant.device "cpu"
    %int0_13 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %7, %none_10, %none_11, %int15, %cpu_12, %int0_13 : !torch.vtensor<[3,2,1,2],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int3_14 = torch.constant.int 3
    %true = torch.constant.bool true
    %none_15 = torch.constant.none
    %8 = torch.aten.prod.dim_int %7, %int3_14, %true, %none_15 : !torch.vtensor<[3,2,1,2],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3,2,1,1],bf16>
    %int0_16 = torch.constant.int 0
    %true_17 = torch.constant.bool true
    %none_18 = torch.constant.none
    %9 = torch.aten.prod.dim_int %8, %int0_16, %true_17, %none_18 : !torch.vtensor<[3,2,1,1],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[1,2,1,1],bf16>
    return %9 : !torch.vtensor<[1,2,1,1],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_81(%arg0: !torch.vtensor<[3,2,1,2],bf16>) -> !torch.vtensor<[1,2,1,1],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_2_1_2_torch.uint8_5> : tensor<2x1x2xui8>) : !torch.vtensor<[2,1,2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[2,1,2],ui8>, !torch.int -> !torch.vtensor<[2,1,2],i1>
    %2 = torch.vtensor.literal(dense<1.000000e+00> : tensor<bf16>) : !torch.vtensor<[],bf16>
    %none = torch.constant.none
    %3 = torch.aten.clone %1, %none : !torch.vtensor<[2,1,2],i1>, !torch.none -> !torch.vtensor<[2,1,2],i1>
    %int3 = torch.constant.int 3
    %int2 = torch.constant.int 2
    %int1 = torch.constant.int 1
    %int2_0 = torch.constant.int 2
    %4 = torch.prim.ListConstruct %int3, %int2, %int1, %int2_0 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %5 = torch.aten.expand %3, %4, %false : !torch.vtensor<[2,1,2],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3,2,1,2],i1>
    %none_1 = torch.constant.none
    %none_2 = torch.constant.none
    %int11_3 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_1, %none_2, %int11_3, %cpu, %int0 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int11_6 = torch.constant.int 11
    %cpu_7 = torch.constant.device "cpu"
    %int0_8 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_4, %none_5, %int11_6, %cpu_7, %int0_8 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_9 = torch.constant.none
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],bf16>, !torch.none -> !torch.vtensor<[],bf16>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],bf16>, !torch.vtensor<[],bf16> -> !torch.vtensor<[3,2,1,2],bf16>
    %none_10 = torch.constant.none
    %none_11 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu_12 = torch.constant.device "cpu"
    %int0_13 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %7, %none_10, %none_11, %int15, %cpu_12, %int0_13 : !torch.vtensor<[3,2,1,2],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int3_14 = torch.constant.int 3
    %true = torch.constant.bool true
    %none_15 = torch.constant.none
    %8 = torch.aten.prod.dim_int %7, %int3_14, %true, %none_15 : !torch.vtensor<[3,2,1,2],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3,2,1,1],bf16>
    %int0_16 = torch.constant.int 0
    %true_17 = torch.constant.bool true
    %none_18 = torch.constant.none
    %9 = torch.aten.prod.dim_int %8, %int0_16, %true_17, %none_18 : !torch.vtensor<[3,2,1,1],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[1,2,1,1],bf16>
    return %9 : !torch.vtensor<[1,2,1,1],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_82(%arg0: !torch.vtensor<[3,2,1,2],bf16>) -> !torch.vtensor<[1,2,1,1],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_2_torch.uint8_9> : tensor<2xui8>) : !torch.vtensor<[2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[2],ui8>, !torch.int -> !torch.vtensor<[2],i1>
    %2 = torch.vtensor.literal(dense<1.000000e+00> : tensor<bf16>) : !torch.vtensor<[],bf16>
    %none = torch.constant.none
    %3 = torch.aten.clone %1, %none : !torch.vtensor<[2],i1>, !torch.none -> !torch.vtensor<[2],i1>
    %int3 = torch.constant.int 3
    %int2 = torch.constant.int 2
    %int1 = torch.constant.int 1
    %int2_0 = torch.constant.int 2
    %4 = torch.prim.ListConstruct %int3, %int2, %int1, %int2_0 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %5 = torch.aten.expand %3, %4, %false : !torch.vtensor<[2],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3,2,1,2],i1>
    %none_1 = torch.constant.none
    %none_2 = torch.constant.none
    %int11_3 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_1, %none_2, %int11_3, %cpu, %int0 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int11_6 = torch.constant.int 11
    %cpu_7 = torch.constant.device "cpu"
    %int0_8 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_4, %none_5, %int11_6, %cpu_7, %int0_8 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_9 = torch.constant.none
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],bf16>, !torch.none -> !torch.vtensor<[],bf16>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],bf16>, !torch.vtensor<[],bf16> -> !torch.vtensor<[3,2,1,2],bf16>
    %none_10 = torch.constant.none
    %none_11 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu_12 = torch.constant.device "cpu"
    %int0_13 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %7, %none_10, %none_11, %int15, %cpu_12, %int0_13 : !torch.vtensor<[3,2,1,2],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int3_14 = torch.constant.int 3
    %true = torch.constant.bool true
    %none_15 = torch.constant.none
    %8 = torch.aten.prod.dim_int %7, %int3_14, %true, %none_15 : !torch.vtensor<[3,2,1,2],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3,2,1,1],bf16>
    %int0_16 = torch.constant.int 0
    %true_17 = torch.constant.bool true
    %none_18 = torch.constant.none
    %9 = torch.aten.prod.dim_int %8, %int0_16, %true_17, %none_18 : !torch.vtensor<[3,2,1,1],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[1,2,1,1],bf16>
    return %9 : !torch.vtensor<[1,2,1,1],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_83(%arg0: !torch.vtensor<[3,2,1,2],bf16>) -> !torch.vtensor<[3,1],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %arg0, %none, %none_0, %int15, %cpu, %int0 : !torch.vtensor<[3,2,1,2],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int3 = torch.constant.int 3
    %false = torch.constant.bool false
    %none_1 = torch.constant.none
    %0 = torch.aten.prod.dim_int %arg0, %int3, %false, %none_1 : !torch.vtensor<[3,2,1,2],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3,2,1],bf16>
    %int1 = torch.constant.int 1
    %false_2 = torch.constant.bool false
    %none_3 = torch.constant.none
    %1 = torch.aten.prod.dim_int %0, %int1, %false_2, %none_3 : !torch.vtensor<[3,2,1],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3,1],bf16>
    return %1 : !torch.vtensor<[3,1],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_84(%arg0: !torch.vtensor<[3,2,1,2],bf16>) -> !torch.vtensor<[3,1],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_2_1_2_torch.uint8_6> : tensor<3x2x1x2xui8>) : !torch.vtensor<[3,2,1,2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,2,1,2],ui8>, !torch.int -> !torch.vtensor<[3,2,1,2],i1>
    %2 = torch.vtensor.literal(dense<1.000000e+00> : tensor<bf16>) : !torch.vtensor<[],bf16>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],bf16>, !torch.none -> !torch.vtensor<[],bf16>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],bf16>, !torch.vtensor<[],bf16> -> !torch.vtensor<[3,2,1,2],bf16>
    %none_3 = torch.constant.none
    %none_4 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu_5 = torch.constant.device "cpu"
    %int0_6 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %4, %none_3, %none_4, %int15, %cpu_5, %int0_6 : !torch.vtensor<[3,2,1,2],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int3 = torch.constant.int 3
    %false = torch.constant.bool false
    %none_7 = torch.constant.none
    %5 = torch.aten.prod.dim_int %4, %int3, %false, %none_7 : !torch.vtensor<[3,2,1,2],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3,2,1],bf16>
    %int1 = torch.constant.int 1
    %false_8 = torch.constant.bool false
    %none_9 = torch.constant.none
    %6 = torch.aten.prod.dim_int %5, %int1, %false_8, %none_9 : !torch.vtensor<[3,2,1],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3,1],bf16>
    return %6 : !torch.vtensor<[3,1],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_85(%arg0: !torch.vtensor<[3,2,1,2],bf16>) -> !torch.vtensor<[3,1],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_2_1_1_torch.uint8_6> : tensor<3x2x1x1xui8>) : !torch.vtensor<[3,2,1,1],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,2,1,1],ui8>, !torch.int -> !torch.vtensor<[3,2,1,1],i1>
    %2 = torch.vtensor.literal(dense<1.000000e+00> : tensor<bf16>) : !torch.vtensor<[],bf16>
    %none = torch.constant.none
    %3 = torch.aten.clone %1, %none : !torch.vtensor<[3,2,1,1],i1>, !torch.none -> !torch.vtensor<[3,2,1,1],i1>
    %int3 = torch.constant.int 3
    %int2 = torch.constant.int 2
    %int1 = torch.constant.int 1
    %int2_0 = torch.constant.int 2
    %4 = torch.prim.ListConstruct %int3, %int2, %int1, %int2_0 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %5 = torch.aten.expand %3, %4, %false : !torch.vtensor<[3,2,1,1],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3,2,1,2],i1>
    %none_1 = torch.constant.none
    %none_2 = torch.constant.none
    %int11_3 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_1, %none_2, %int11_3, %cpu, %int0 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int11_6 = torch.constant.int 11
    %cpu_7 = torch.constant.device "cpu"
    %int0_8 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_4, %none_5, %int11_6, %cpu_7, %int0_8 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_9 = torch.constant.none
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],bf16>, !torch.none -> !torch.vtensor<[],bf16>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],bf16>, !torch.vtensor<[],bf16> -> !torch.vtensor<[3,2,1,2],bf16>
    %none_10 = torch.constant.none
    %none_11 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu_12 = torch.constant.device "cpu"
    %int0_13 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %7, %none_10, %none_11, %int15, %cpu_12, %int0_13 : !torch.vtensor<[3,2,1,2],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int3_14 = torch.constant.int 3
    %false_15 = torch.constant.bool false
    %none_16 = torch.constant.none
    %8 = torch.aten.prod.dim_int %7, %int3_14, %false_15, %none_16 : !torch.vtensor<[3,2,1,2],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3,2,1],bf16>
    %int1_17 = torch.constant.int 1
    %false_18 = torch.constant.bool false
    %none_19 = torch.constant.none
    %9 = torch.aten.prod.dim_int %8, %int1_17, %false_18, %none_19 : !torch.vtensor<[3,2,1],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3,1],bf16>
    return %9 : !torch.vtensor<[3,1],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_86(%arg0: !torch.vtensor<[3,2,1,2],bf16>) -> !torch.vtensor<[3,1],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_1_1_2_torch.uint8_6> : tensor<3x1x1x2xui8>) : !torch.vtensor<[3,1,1,2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,1,1,2],ui8>, !torch.int -> !torch.vtensor<[3,1,1,2],i1>
    %2 = torch.vtensor.literal(dense<1.000000e+00> : tensor<bf16>) : !torch.vtensor<[],bf16>
    %none = torch.constant.none
    %3 = torch.aten.clone %1, %none : !torch.vtensor<[3,1,1,2],i1>, !torch.none -> !torch.vtensor<[3,1,1,2],i1>
    %int3 = torch.constant.int 3
    %int2 = torch.constant.int 2
    %int1 = torch.constant.int 1
    %int2_0 = torch.constant.int 2
    %4 = torch.prim.ListConstruct %int3, %int2, %int1, %int2_0 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %5 = torch.aten.expand %3, %4, %false : !torch.vtensor<[3,1,1,2],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3,2,1,2],i1>
    %none_1 = torch.constant.none
    %none_2 = torch.constant.none
    %int11_3 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_1, %none_2, %int11_3, %cpu, %int0 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int11_6 = torch.constant.int 11
    %cpu_7 = torch.constant.device "cpu"
    %int0_8 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_4, %none_5, %int11_6, %cpu_7, %int0_8 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_9 = torch.constant.none
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],bf16>, !torch.none -> !torch.vtensor<[],bf16>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],bf16>, !torch.vtensor<[],bf16> -> !torch.vtensor<[3,2,1,2],bf16>
    %none_10 = torch.constant.none
    %none_11 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu_12 = torch.constant.device "cpu"
    %int0_13 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %7, %none_10, %none_11, %int15, %cpu_12, %int0_13 : !torch.vtensor<[3,2,1,2],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int3_14 = torch.constant.int 3
    %false_15 = torch.constant.bool false
    %none_16 = torch.constant.none
    %8 = torch.aten.prod.dim_int %7, %int3_14, %false_15, %none_16 : !torch.vtensor<[3,2,1,2],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3,2,1],bf16>
    %int1_17 = torch.constant.int 1
    %false_18 = torch.constant.bool false
    %none_19 = torch.constant.none
    %9 = torch.aten.prod.dim_int %8, %int1_17, %false_18, %none_19 : !torch.vtensor<[3,2,1],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3,1],bf16>
    return %9 : !torch.vtensor<[3,1],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_87(%arg0: !torch.vtensor<[3,2,1,2],bf16>) -> !torch.vtensor<[3,1],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_1_2_1_2_torch.uint8_6> : tensor<1x2x1x2xui8>) : !torch.vtensor<[1,2,1,2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[1,2,1,2],ui8>, !torch.int -> !torch.vtensor<[1,2,1,2],i1>
    %2 = torch.vtensor.literal(dense<1.000000e+00> : tensor<bf16>) : !torch.vtensor<[],bf16>
    %none = torch.constant.none
    %3 = torch.aten.clone %1, %none : !torch.vtensor<[1,2,1,2],i1>, !torch.none -> !torch.vtensor<[1,2,1,2],i1>
    %int3 = torch.constant.int 3
    %int2 = torch.constant.int 2
    %int1 = torch.constant.int 1
    %int2_0 = torch.constant.int 2
    %4 = torch.prim.ListConstruct %int3, %int2, %int1, %int2_0 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %5 = torch.aten.expand %3, %4, %false : !torch.vtensor<[1,2,1,2],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3,2,1,2],i1>
    %none_1 = torch.constant.none
    %none_2 = torch.constant.none
    %int11_3 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_1, %none_2, %int11_3, %cpu, %int0 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int11_6 = torch.constant.int 11
    %cpu_7 = torch.constant.device "cpu"
    %int0_8 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_4, %none_5, %int11_6, %cpu_7, %int0_8 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_9 = torch.constant.none
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],bf16>, !torch.none -> !torch.vtensor<[],bf16>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],bf16>, !torch.vtensor<[],bf16> -> !torch.vtensor<[3,2,1,2],bf16>
    %none_10 = torch.constant.none
    %none_11 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu_12 = torch.constant.device "cpu"
    %int0_13 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %7, %none_10, %none_11, %int15, %cpu_12, %int0_13 : !torch.vtensor<[3,2,1,2],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int3_14 = torch.constant.int 3
    %false_15 = torch.constant.bool false
    %none_16 = torch.constant.none
    %8 = torch.aten.prod.dim_int %7, %int3_14, %false_15, %none_16 : !torch.vtensor<[3,2,1,2],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3,2,1],bf16>
    %int1_17 = torch.constant.int 1
    %false_18 = torch.constant.bool false
    %none_19 = torch.constant.none
    %9 = torch.aten.prod.dim_int %8, %int1_17, %false_18, %none_19 : !torch.vtensor<[3,2,1],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3,1],bf16>
    return %9 : !torch.vtensor<[3,1],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_88(%arg0: !torch.vtensor<[3,2,1,2],bf16>) -> !torch.vtensor<[3,1],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_2_1_2_torch.uint8_6> : tensor<2x1x2xui8>) : !torch.vtensor<[2,1,2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[2,1,2],ui8>, !torch.int -> !torch.vtensor<[2,1,2],i1>
    %2 = torch.vtensor.literal(dense<1.000000e+00> : tensor<bf16>) : !torch.vtensor<[],bf16>
    %none = torch.constant.none
    %3 = torch.aten.clone %1, %none : !torch.vtensor<[2,1,2],i1>, !torch.none -> !torch.vtensor<[2,1,2],i1>
    %int3 = torch.constant.int 3
    %int2 = torch.constant.int 2
    %int1 = torch.constant.int 1
    %int2_0 = torch.constant.int 2
    %4 = torch.prim.ListConstruct %int3, %int2, %int1, %int2_0 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %5 = torch.aten.expand %3, %4, %false : !torch.vtensor<[2,1,2],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3,2,1,2],i1>
    %none_1 = torch.constant.none
    %none_2 = torch.constant.none
    %int11_3 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_1, %none_2, %int11_3, %cpu, %int0 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int11_6 = torch.constant.int 11
    %cpu_7 = torch.constant.device "cpu"
    %int0_8 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_4, %none_5, %int11_6, %cpu_7, %int0_8 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_9 = torch.constant.none
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],bf16>, !torch.none -> !torch.vtensor<[],bf16>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],bf16>, !torch.vtensor<[],bf16> -> !torch.vtensor<[3,2,1,2],bf16>
    %none_10 = torch.constant.none
    %none_11 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu_12 = torch.constant.device "cpu"
    %int0_13 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %7, %none_10, %none_11, %int15, %cpu_12, %int0_13 : !torch.vtensor<[3,2,1,2],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int3_14 = torch.constant.int 3
    %false_15 = torch.constant.bool false
    %none_16 = torch.constant.none
    %8 = torch.aten.prod.dim_int %7, %int3_14, %false_15, %none_16 : !torch.vtensor<[3,2,1,2],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3,2,1],bf16>
    %int1_17 = torch.constant.int 1
    %false_18 = torch.constant.bool false
    %none_19 = torch.constant.none
    %9 = torch.aten.prod.dim_int %8, %int1_17, %false_18, %none_19 : !torch.vtensor<[3,2,1],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3,1],bf16>
    return %9 : !torch.vtensor<[3,1],bf16>
  }
  func.func @test_reduction_masked_prod_cpu_bfloat16_89(%arg0: !torch.vtensor<[3,2,1,2],bf16>) -> !torch.vtensor<[3,1],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_2_torch.uint8_10> : tensor<2xui8>) : !torch.vtensor<[2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[2],ui8>, !torch.int -> !torch.vtensor<[2],i1>
    %2 = torch.vtensor.literal(dense<1.000000e+00> : tensor<bf16>) : !torch.vtensor<[],bf16>
    %none = torch.constant.none
    %3 = torch.aten.clone %1, %none : !torch.vtensor<[2],i1>, !torch.none -> !torch.vtensor<[2],i1>
    %int3 = torch.constant.int 3
    %int2 = torch.constant.int 2
    %int1 = torch.constant.int 1
    %int2_0 = torch.constant.int 2
    %4 = torch.prim.ListConstruct %int3, %int2, %int1, %int2_0 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %5 = torch.aten.expand %3, %4, %false : !torch.vtensor<[2],i1>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3,2,1,2],i1>
    %none_1 = torch.constant.none
    %none_2 = torch.constant.none
    %int11_3 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_1, %none_2, %int11_3, %cpu, %int0 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int11_6 = torch.constant.int 11
    %cpu_7 = torch.constant.device "cpu"
    %int0_8 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_4, %none_5, %int11_6, %cpu_7, %int0_8 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_9 = torch.constant.none
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],bf16>, !torch.none -> !torch.vtensor<[],bf16>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],bf16>, !torch.vtensor<[],bf16> -> !torch.vtensor<[3,2,1,2],bf16>
    %none_10 = torch.constant.none
    %none_11 = torch.constant.none
    %int15 = torch.constant.int 15
    %cpu_12 = torch.constant.device "cpu"
    %int0_13 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %7, %none_10, %none_11, %int15, %cpu_12, %int0_13 : !torch.vtensor<[3,2,1,2],bf16>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int3_14 = torch.constant.int 3
    %false_15 = torch.constant.bool false
    %none_16 = torch.constant.none
    %8 = torch.aten.prod.dim_int %7, %int3_14, %false_15, %none_16 : !torch.vtensor<[3,2,1,2],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3,2,1],bf16>
    %int1_17 = torch.constant.int 1
    %false_18 = torch.constant.bool false
    %none_19 = torch.constant.none
    %9 = torch.aten.prod.dim_int %8, %int1_17, %false_18, %none_19 : !torch.vtensor<[3,2,1],bf16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3,1],bf16>
    return %9 : !torch.vtensor<[3,1],bf16>
  }
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_2_torch.uint8: "0x010000000101",
      torch_tensor_2_torch.uint8_1: "0x010000000101",
      torch_tensor_2_torch.uint8_2: "0x010000000101",
      torch_tensor_2_torch.uint8_3: "0x010000000101",
      torch_tensor_3_5_torch.uint8: "0x01000000010100010000000101010101000001",
      torch_tensor_3_5_torch.uint8_1: "0x01000000010100010000000101010101000001",
      torch_tensor_3_5_torch.uint8_2: "0x01000000010100010000000101010101000001",
      torch_tensor_3_5_torch.uint8_3: "0x01000000010100010000000101010101000001",
      torch_tensor_3_5_torch.uint8_4: "0x01000000010100010000000101010101000001",
      torch_tensor_3_5_torch.uint8_5: "0x01000000010100010000000101010101000001",
      torch_tensor_3_5_torch.uint8_6: "0x01000000010100010000000101010101000001",
      torch_tensor_3_5_torch.uint8_7: "0x01000000010100010000000101010101000001",
      torch_tensor_3_5_torch.uint8_8: "0x01000000010100010000000101010101000001",
      torch_tensor_3_5_torch.uint8_9: "0x01000000010100010000000101010101000001",
      torch_tensor_3_5_torch.uint8_10: "0x01000000010100010000000101010101000001",
      torch_tensor_3_5_torch.uint8_11: "0x01000000010100010000000101010101000001",
      torch_tensor_3_5_torch.uint8_12: "0x01000000010100010000000101010101000001",
      torch_tensor_3_5_torch.uint8_13: "0x01000000010100010000000101010101000001",
      torch_tensor_3_5_torch.uint8_14: "0x01000000010100010000000101010101000001",
      torch_tensor_3_5_torch.uint8_15: "0x01000000010100010000000101010101000001",
      torch_tensor_3_5_torch.uint8_16: "0x01000000010100010000000101010101000001",
      torch_tensor_3_5_torch.uint8_17: "0x01000000010100010000000101010101000001",
      torch_tensor_3_5_torch.uint8_18: "0x01000000010100010000000101010101000001",
      torch_tensor_3_5_torch.uint8_19: "0x01000000010100010000000101010101000001",
      torch_tensor_3_2_1_2_torch.uint8: "0x01000000010100010000000101010101",
      torch_tensor_3_2_1_1_torch.uint8: "0x01000000010100010000",
      torch_tensor_3_1_1_2_torch.uint8: "0x01000000010100010000",
      torch_tensor_1_2_1_2_torch.uint8: "0x0100000001010001",
      torch_tensor_2_1_2_torch.uint8: "0x0100000001010001",
      torch_tensor_2_torch.uint8_4: "0x010000000101",
      torch_tensor_3_2_1_2_torch.uint8_1: "0x01000000010100010000000101010101",
      torch_tensor_3_2_1_1_torch.uint8_1: "0x01000000010100010000",
      torch_tensor_3_1_1_2_torch.uint8_1: "0x01000000010100010000",
      torch_tensor_1_2_1_2_torch.uint8_1: "0x0100000001010001",
      torch_tensor_2_1_2_torch.uint8_1: "0x0100000001010001",
      torch_tensor_2_torch.uint8_5: "0x010000000101",
      torch_tensor_3_2_1_2_torch.uint8_2: "0x01000000010100010000000101010101",
      torch_tensor_3_2_1_1_torch.uint8_2: "0x01000000010100010000",
      torch_tensor_3_1_1_2_torch.uint8_2: "0x01000000010100010000",
      torch_tensor_1_2_1_2_torch.uint8_2: "0x0100000001010001",
      torch_tensor_2_1_2_torch.uint8_2: "0x0100000001010001",
      torch_tensor_2_torch.uint8_6: "0x010000000101",
      torch_tensor_3_2_1_2_torch.uint8_3: "0x01000000010100010000000101010101",
      torch_tensor_3_2_1_1_torch.uint8_3: "0x01000000010100010000",
      torch_tensor_3_1_1_2_torch.uint8_3: "0x01000000010100010000",
      torch_tensor_1_2_1_2_torch.uint8_3: "0x0100000001010001",
      torch_tensor_2_1_2_torch.uint8_3: "0x0100000001010001",
      torch_tensor_2_torch.uint8_7: "0x010000000101",
      torch_tensor_3_2_1_2_torch.uint8_4: "0x01000000010100010000000101010101",
      torch_tensor_3_2_1_1_torch.uint8_4: "0x01000000010100010000",
      torch_tensor_3_1_1_2_torch.uint8_4: "0x01000000010100010000",
      torch_tensor_1_2_1_2_torch.uint8_4: "0x0100000001010001",
      torch_tensor_2_1_2_torch.uint8_4: "0x0100000001010001",
      torch_tensor_2_torch.uint8_8: "0x010000000101",
      torch_tensor_3_2_1_2_torch.uint8_5: "0x01000000010100010000000101010101",
      torch_tensor_3_2_1_1_torch.uint8_5: "0x01000000010100010000",
      torch_tensor_3_1_1_2_torch.uint8_5: "0x01000000010100010000",
      torch_tensor_1_2_1_2_torch.uint8_5: "0x0100000001010001",
      torch_tensor_2_1_2_torch.uint8_5: "0x0100000001010001",
      torch_tensor_2_torch.uint8_9: "0x010000000101",
      torch_tensor_3_2_1_2_torch.uint8_6: "0x01000000010100010000000101010101",
      torch_tensor_3_2_1_1_torch.uint8_6: "0x01000000010100010000",
      torch_tensor_3_1_1_2_torch.uint8_6: "0x01000000010100010000",
      torch_tensor_1_2_1_2_torch.uint8_6: "0x0100000001010001",
      torch_tensor_2_1_2_torch.uint8_6: "0x0100000001010001",
      torch_tensor_2_torch.uint8_10: "0x010000000101"
    }
  }
#-}
