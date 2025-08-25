module @module {
  func.func @test_reduction_masked_amin_cpu_float32_0(%arg0: !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %false = torch.constant.bool false
    %1 = torch.aten.amin %arg0, %0, %false : !torch.vtensor<[],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[],f32>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int6, %cpu, %int0 : !torch.vtensor<[],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %1 : !torch.vtensor<[],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_1(%arg0: !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense<1> : tensor<ui8>) : !torch.vtensor<[],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[],ui8>, !torch.int -> !torch.vtensor<[],i1>
    %2 = torch.vtensor.literal(dense<0x7F800000> : tensor<f32>) : !torch.vtensor<[],f32>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[],i1>, !torch.vtensor<[],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[],f32>
    %5 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %false = torch.constant.bool false
    %6 = torch.aten.amin %4, %5, %false : !torch.vtensor<[],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[],f32>
    %none_3 = torch.constant.none
    %none_4 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu_5 = torch.constant.device "cpu"
    %int0_6 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_3, %none_4, %int6, %cpu_5, %int0_6 : !torch.vtensor<[],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %6 : !torch.vtensor<[],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_2(%arg0: !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %1 = torch.aten.amin %arg0, %0, %true : !torch.vtensor<[],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[],f32>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu = torch.constant.device "cpu"
    %int0_1 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int6, %cpu, %int0_1 : !torch.vtensor<[],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %1 : !torch.vtensor<[],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_3(%arg0: !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense<1> : tensor<ui8>) : !torch.vtensor<[],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[],ui8>, !torch.int -> !torch.vtensor<[],i1>
    %2 = torch.vtensor.literal(dense<0x7F800000> : tensor<f32>) : !torch.vtensor<[],f32>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[],i1>, !torch.vtensor<[],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[],f32>
    %int0_3 = torch.constant.int 0
    %5 = torch.prim.ListConstruct %int0_3 : (!torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %6 = torch.aten.amin %4, %5, %true : !torch.vtensor<[],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[],f32>
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu_6 = torch.constant.device "cpu"
    %int0_7 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_4, %none_5, %int6, %cpu_6, %int0_7 : !torch.vtensor<[],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %6 : !torch.vtensor<[],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_4(%arg0: !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %1 = torch.aten.amin %arg0, %0, %false : !torch.vtensor<[],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[],f32>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu = torch.constant.device "cpu"
    %int0_1 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int6, %cpu, %int0_1 : !torch.vtensor<[],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %1 : !torch.vtensor<[],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_5(%arg0: !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense<1> : tensor<ui8>) : !torch.vtensor<[],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[],ui8>, !torch.int -> !torch.vtensor<[],i1>
    %2 = torch.vtensor.literal(dense<0x7F800000> : tensor<f32>) : !torch.vtensor<[],f32>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[],i1>, !torch.vtensor<[],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[],f32>
    %int0_3 = torch.constant.int 0
    %5 = torch.prim.ListConstruct %int0_3 : (!torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %6 = torch.aten.amin %4, %5, %false : !torch.vtensor<[],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[],f32>
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu_6 = torch.constant.device "cpu"
    %int0_7 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_4, %none_5, %int6, %cpu_6, %int0_7 : !torch.vtensor<[],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %6 : !torch.vtensor<[],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_6(%arg0: !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %false = torch.constant.bool false
    %1 = torch.aten.amin %arg0, %0, %false : !torch.vtensor<[],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[],f32>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int6, %cpu, %int0 : !torch.vtensor<[],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %1 : !torch.vtensor<[],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_7(%arg0: !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense<1> : tensor<ui8>) : !torch.vtensor<[],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[],ui8>, !torch.int -> !torch.vtensor<[],i1>
    %2 = torch.vtensor.literal(dense<0x7F800000> : tensor<f32>) : !torch.vtensor<[],f32>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[],i1>, !torch.vtensor<[],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[],f32>
    %5 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %false = torch.constant.bool false
    %6 = torch.aten.amin %4, %5, %false : !torch.vtensor<[],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[],f32>
    %none_3 = torch.constant.none
    %none_4 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu_5 = torch.constant.device "cpu"
    %int0_6 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_3, %none_4, %int6, %cpu_5, %int0_6 : !torch.vtensor<[],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %6 : !torch.vtensor<[],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_8(%arg0: !torch.vtensor<[2],f32>) -> !torch.vtensor<[],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %1 = torch.aten.amin %arg0, %0, %false : !torch.vtensor<[2],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[],f32>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu = torch.constant.device "cpu"
    %int0_1 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int6, %cpu, %int0_1 : !torch.vtensor<[],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %1 : !torch.vtensor<[],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_9(%arg0: !torch.vtensor<[2],f32>) -> !torch.vtensor<[],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_2_torch.uint8> : tensor<2xui8>) : !torch.vtensor<[2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[2],ui8>, !torch.int -> !torch.vtensor<[2],i1>
    %2 = torch.vtensor.literal(dense<0x7F800000> : tensor<f32>) : !torch.vtensor<[],f32>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[2],i1>, !torch.vtensor<[2],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[2],f32>
    %int0_3 = torch.constant.int 0
    %5 = torch.prim.ListConstruct %int0_3 : (!torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %6 = torch.aten.amin %4, %5, %false : !torch.vtensor<[2],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[],f32>
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu_6 = torch.constant.device "cpu"
    %int0_7 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_4, %none_5, %int6, %cpu_6, %int0_7 : !torch.vtensor<[],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %6 : !torch.vtensor<[],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_10(%arg0: !torch.vtensor<[2],f32>) -> !torch.vtensor<[1],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %1 = torch.aten.amin %arg0, %0, %true : !torch.vtensor<[2],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1],f32>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu = torch.constant.device "cpu"
    %int0_1 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int6, %cpu, %int0_1 : !torch.vtensor<[1],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %1 : !torch.vtensor<[1],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_11(%arg0: !torch.vtensor<[2],f32>) -> !torch.vtensor<[1],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_2_torch.uint8_1> : tensor<2xui8>) : !torch.vtensor<[2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[2],ui8>, !torch.int -> !torch.vtensor<[2],i1>
    %2 = torch.vtensor.literal(dense<0x7F800000> : tensor<f32>) : !torch.vtensor<[],f32>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[2],i1>, !torch.vtensor<[2],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[2],f32>
    %int0_3 = torch.constant.int 0
    %5 = torch.prim.ListConstruct %int0_3 : (!torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %6 = torch.aten.amin %4, %5, %true : !torch.vtensor<[2],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1],f32>
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu_6 = torch.constant.device "cpu"
    %int0_7 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_4, %none_5, %int6, %cpu_6, %int0_7 : !torch.vtensor<[1],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %6 : !torch.vtensor<[1],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_12(%arg0: !torch.vtensor<[2],f32>) -> !torch.vtensor<[],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %1 = torch.aten.amin %arg0, %0, %false : !torch.vtensor<[2],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[],f32>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu = torch.constant.device "cpu"
    %int0_1 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int6, %cpu, %int0_1 : !torch.vtensor<[],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %1 : !torch.vtensor<[],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_13(%arg0: !torch.vtensor<[2],f32>) -> !torch.vtensor<[],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_2_torch.uint8_2> : tensor<2xui8>) : !torch.vtensor<[2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[2],ui8>, !torch.int -> !torch.vtensor<[2],i1>
    %2 = torch.vtensor.literal(dense<0x7F800000> : tensor<f32>) : !torch.vtensor<[],f32>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[2],i1>, !torch.vtensor<[2],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[2],f32>
    %int0_3 = torch.constant.int 0
    %5 = torch.prim.ListConstruct %int0_3 : (!torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %6 = torch.aten.amin %4, %5, %false : !torch.vtensor<[2],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[],f32>
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu_6 = torch.constant.device "cpu"
    %int0_7 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_4, %none_5, %int6, %cpu_6, %int0_7 : !torch.vtensor<[],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %6 : !torch.vtensor<[],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_14(%arg0: !torch.vtensor<[2],f32>) -> !torch.vtensor<[],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %1 = torch.aten.amin %arg0, %0, %false : !torch.vtensor<[2],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[],f32>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu = torch.constant.device "cpu"
    %int0_1 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int6, %cpu, %int0_1 : !torch.vtensor<[],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %1 : !torch.vtensor<[],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_15(%arg0: !torch.vtensor<[2],f32>) -> !torch.vtensor<[],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_2_torch.uint8_3> : tensor<2xui8>) : !torch.vtensor<[2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[2],ui8>, !torch.int -> !torch.vtensor<[2],i1>
    %2 = torch.vtensor.literal(dense<0x7F800000> : tensor<f32>) : !torch.vtensor<[],f32>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[2],i1>, !torch.vtensor<[2],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[2],f32>
    %int0_3 = torch.constant.int 0
    %5 = torch.prim.ListConstruct %int0_3 : (!torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %6 = torch.aten.amin %4, %5, %false : !torch.vtensor<[2],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[],f32>
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu_6 = torch.constant.device "cpu"
    %int0_7 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_4, %none_5, %int6, %cpu_6, %int0_7 : !torch.vtensor<[],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %6 : !torch.vtensor<[],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_16(%arg0: !torch.vtensor<[3,5],f32>) -> !torch.vtensor<[],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %0 = torch.prim.ListConstruct %int0, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %1 = torch.aten.amin %arg0, %0, %false : !torch.vtensor<[3,5],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[],f32>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu = torch.constant.device "cpu"
    %int0_1 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int6, %cpu, %int0_1 : !torch.vtensor<[],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %1 : !torch.vtensor<[],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_17(%arg0: !torch.vtensor<[3,5],f32>) -> !torch.vtensor<[],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_5_torch.uint8> : tensor<3x5xui8>) : !torch.vtensor<[3,5],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,5],ui8>, !torch.int -> !torch.vtensor<[3,5],i1>
    %2 = torch.vtensor.literal(dense<0x7F800000> : tensor<f32>) : !torch.vtensor<[],f32>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[3,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[3,5],i1>, !torch.vtensor<[3,5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[3,5],f32>
    %int0_3 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %5 = torch.prim.ListConstruct %int0_3, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %6 = torch.aten.amin %4, %5, %false : !torch.vtensor<[3,5],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[],f32>
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu_6 = torch.constant.device "cpu"
    %int0_7 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_4, %none_5, %int6, %cpu_6, %int0_7 : !torch.vtensor<[],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %6 : !torch.vtensor<[],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_18(%arg0: !torch.vtensor<[3,5],f32>) -> !torch.vtensor<[],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_5_torch.uint8_1> : tensor<3x5xui8>) : !torch.vtensor<[3,5],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,5],ui8>, !torch.int -> !torch.vtensor<[3,5],i1>
    %2 = torch.vtensor.literal(dense<0x7F800000> : tensor<f32>) : !torch.vtensor<[],f32>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[3,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[3,5],i1>, !torch.vtensor<[3,5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[3,5],f32>
    %int0_3 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %5 = torch.prim.ListConstruct %int0_3, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %6 = torch.aten.amin %4, %5, %false : !torch.vtensor<[3,5],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[],f32>
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu_6 = torch.constant.device "cpu"
    %int0_7 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_4, %none_5, %int6, %cpu_6, %int0_7 : !torch.vtensor<[],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %6 : !torch.vtensor<[],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_19(%arg0: !torch.vtensor<[3,5],f32>) -> !torch.vtensor<[],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_5_torch.uint8_2> : tensor<3x5xui8>) : !torch.vtensor<[3,5],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,5],ui8>, !torch.int -> !torch.vtensor<[3,5],i1>
    %2 = torch.vtensor.literal(dense<0x7F800000> : tensor<f32>) : !torch.vtensor<[],f32>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[3,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[3,5],i1>, !torch.vtensor<[3,5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[3,5],f32>
    %int0_3 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %5 = torch.prim.ListConstruct %int0_3, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %6 = torch.aten.amin %4, %5, %false : !torch.vtensor<[3,5],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[],f32>
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu_6 = torch.constant.device "cpu"
    %int0_7 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_4, %none_5, %int6, %cpu_6, %int0_7 : !torch.vtensor<[],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %6 : !torch.vtensor<[],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_20(%arg0: !torch.vtensor<[3,5],f32>) -> !torch.vtensor<[],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_5_torch.uint8_3> : tensor<3x5xui8>) : !torch.vtensor<[3,5],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,5],ui8>, !torch.int -> !torch.vtensor<[3,5],i1>
    %2 = torch.vtensor.literal(dense<0x7F800000> : tensor<f32>) : !torch.vtensor<[],f32>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[3,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[3,5],i1>, !torch.vtensor<[3,5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[3,5],f32>
    %int0_3 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %5 = torch.prim.ListConstruct %int0_3, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %6 = torch.aten.amin %4, %5, %false : !torch.vtensor<[3,5],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[],f32>
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu_6 = torch.constant.device "cpu"
    %int0_7 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_4, %none_5, %int6, %cpu_6, %int0_7 : !torch.vtensor<[],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %6 : !torch.vtensor<[],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_21(%arg0: !torch.vtensor<[3,5],f32>) -> !torch.vtensor<[1,5],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %1 = torch.aten.amin %arg0, %0, %true : !torch.vtensor<[3,5],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,5],f32>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu = torch.constant.device "cpu"
    %int0_1 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int6, %cpu, %int0_1 : !torch.vtensor<[1,5],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %1 : !torch.vtensor<[1,5],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_22(%arg0: !torch.vtensor<[3,5],f32>) -> !torch.vtensor<[1,5],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_5_torch.uint8_4> : tensor<3x5xui8>) : !torch.vtensor<[3,5],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,5],ui8>, !torch.int -> !torch.vtensor<[3,5],i1>
    %2 = torch.vtensor.literal(dense<0x7F800000> : tensor<f32>) : !torch.vtensor<[],f32>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[3,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[3,5],i1>, !torch.vtensor<[3,5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[3,5],f32>
    %int0_3 = torch.constant.int 0
    %5 = torch.prim.ListConstruct %int0_3 : (!torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %6 = torch.aten.amin %4, %5, %true : !torch.vtensor<[3,5],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,5],f32>
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu_6 = torch.constant.device "cpu"
    %int0_7 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_4, %none_5, %int6, %cpu_6, %int0_7 : !torch.vtensor<[1,5],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %6 : !torch.vtensor<[1,5],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_23(%arg0: !torch.vtensor<[3,5],f32>) -> !torch.vtensor<[1,5],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_5_torch.uint8_5> : tensor<3x5xui8>) : !torch.vtensor<[3,5],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,5],ui8>, !torch.int -> !torch.vtensor<[3,5],i1>
    %2 = torch.vtensor.literal(dense<0x7F800000> : tensor<f32>) : !torch.vtensor<[],f32>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[3,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[3,5],i1>, !torch.vtensor<[3,5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[3,5],f32>
    %int0_3 = torch.constant.int 0
    %5 = torch.prim.ListConstruct %int0_3 : (!torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %6 = torch.aten.amin %4, %5, %true : !torch.vtensor<[3,5],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,5],f32>
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu_6 = torch.constant.device "cpu"
    %int0_7 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_4, %none_5, %int6, %cpu_6, %int0_7 : !torch.vtensor<[1,5],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %6 : !torch.vtensor<[1,5],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_24(%arg0: !torch.vtensor<[3,5],f32>) -> !torch.vtensor<[1,5],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_5_torch.uint8_6> : tensor<3x5xui8>) : !torch.vtensor<[3,5],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,5],ui8>, !torch.int -> !torch.vtensor<[3,5],i1>
    %2 = torch.vtensor.literal(dense<0x7F800000> : tensor<f32>) : !torch.vtensor<[],f32>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[3,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[3,5],i1>, !torch.vtensor<[3,5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[3,5],f32>
    %int0_3 = torch.constant.int 0
    %5 = torch.prim.ListConstruct %int0_3 : (!torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %6 = torch.aten.amin %4, %5, %true : !torch.vtensor<[3,5],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,5],f32>
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu_6 = torch.constant.device "cpu"
    %int0_7 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_4, %none_5, %int6, %cpu_6, %int0_7 : !torch.vtensor<[1,5],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %6 : !torch.vtensor<[1,5],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_25(%arg0: !torch.vtensor<[3,5],f32>) -> !torch.vtensor<[1,5],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_5_torch.uint8_7> : tensor<3x5xui8>) : !torch.vtensor<[3,5],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,5],ui8>, !torch.int -> !torch.vtensor<[3,5],i1>
    %2 = torch.vtensor.literal(dense<0x7F800000> : tensor<f32>) : !torch.vtensor<[],f32>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[3,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[3,5],i1>, !torch.vtensor<[3,5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[3,5],f32>
    %int0_3 = torch.constant.int 0
    %5 = torch.prim.ListConstruct %int0_3 : (!torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %6 = torch.aten.amin %4, %5, %true : !torch.vtensor<[3,5],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,5],f32>
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu_6 = torch.constant.device "cpu"
    %int0_7 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_4, %none_5, %int6, %cpu_6, %int0_7 : !torch.vtensor<[1,5],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %6 : !torch.vtensor<[1,5],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_26(%arg0: !torch.vtensor<[3,5],f32>) -> !torch.vtensor<[3],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %int1 = torch.constant.int 1
    %0 = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %1 = torch.aten.amin %arg0, %0, %false : !torch.vtensor<[3,5],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3],f32>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int6, %cpu, %int0 : !torch.vtensor<[3],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %1 : !torch.vtensor<[3],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_27(%arg0: !torch.vtensor<[3,5],f32>) -> !torch.vtensor<[3],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_5_torch.uint8_8> : tensor<3x5xui8>) : !torch.vtensor<[3,5],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,5],ui8>, !torch.int -> !torch.vtensor<[3,5],i1>
    %2 = torch.vtensor.literal(dense<0x7F800000> : tensor<f32>) : !torch.vtensor<[],f32>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[3,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[3,5],i1>, !torch.vtensor<[3,5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[3,5],f32>
    %int1 = torch.constant.int 1
    %5 = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %6 = torch.aten.amin %4, %5, %false : !torch.vtensor<[3,5],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3],f32>
    %none_3 = torch.constant.none
    %none_4 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu_5 = torch.constant.device "cpu"
    %int0_6 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_3, %none_4, %int6, %cpu_5, %int0_6 : !torch.vtensor<[3],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %6 : !torch.vtensor<[3],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_28(%arg0: !torch.vtensor<[3,5],f32>) -> !torch.vtensor<[3],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_5_torch.uint8_9> : tensor<3x5xui8>) : !torch.vtensor<[3,5],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,5],ui8>, !torch.int -> !torch.vtensor<[3,5],i1>
    %2 = torch.vtensor.literal(dense<0x7F800000> : tensor<f32>) : !torch.vtensor<[],f32>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[3,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[3,5],i1>, !torch.vtensor<[3,5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[3,5],f32>
    %int1 = torch.constant.int 1
    %5 = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %6 = torch.aten.amin %4, %5, %false : !torch.vtensor<[3,5],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3],f32>
    %none_3 = torch.constant.none
    %none_4 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu_5 = torch.constant.device "cpu"
    %int0_6 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_3, %none_4, %int6, %cpu_5, %int0_6 : !torch.vtensor<[3],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %6 : !torch.vtensor<[3],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_29(%arg0: !torch.vtensor<[3,5],f32>) -> !torch.vtensor<[3],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_5_torch.uint8_10> : tensor<3x5xui8>) : !torch.vtensor<[3,5],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,5],ui8>, !torch.int -> !torch.vtensor<[3,5],i1>
    %2 = torch.vtensor.literal(dense<0x7F800000> : tensor<f32>) : !torch.vtensor<[],f32>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[3,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[3,5],i1>, !torch.vtensor<[3,5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[3,5],f32>
    %int1 = torch.constant.int 1
    %5 = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %6 = torch.aten.amin %4, %5, %false : !torch.vtensor<[3,5],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3],f32>
    %none_3 = torch.constant.none
    %none_4 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu_5 = torch.constant.device "cpu"
    %int0_6 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_3, %none_4, %int6, %cpu_5, %int0_6 : !torch.vtensor<[3],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %6 : !torch.vtensor<[3],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_30(%arg0: !torch.vtensor<[3,5],f32>) -> !torch.vtensor<[3],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_5_torch.uint8_11> : tensor<3x5xui8>) : !torch.vtensor<[3,5],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,5],ui8>, !torch.int -> !torch.vtensor<[3,5],i1>
    %2 = torch.vtensor.literal(dense<0x7F800000> : tensor<f32>) : !torch.vtensor<[],f32>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[3,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[3,5],i1>, !torch.vtensor<[3,5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[3,5],f32>
    %int1 = torch.constant.int 1
    %5 = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %6 = torch.aten.amin %4, %5, %false : !torch.vtensor<[3,5],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3],f32>
    %none_3 = torch.constant.none
    %none_4 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu_5 = torch.constant.device "cpu"
    %int0_6 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_3, %none_4, %int6, %cpu_5, %int0_6 : !torch.vtensor<[3],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %6 : !torch.vtensor<[3],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_31(%arg0: !torch.vtensor<[3,5],f32>) -> !torch.vtensor<[],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %0 = torch.prim.ListConstruct %int0, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %1 = torch.aten.amin %arg0, %0, %false : !torch.vtensor<[3,5],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[],f32>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu = torch.constant.device "cpu"
    %int0_1 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int6, %cpu, %int0_1 : !torch.vtensor<[],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %1 : !torch.vtensor<[],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_32(%arg0: !torch.vtensor<[3,5],f32>) -> !torch.vtensor<[],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_5_torch.uint8_12> : tensor<3x5xui8>) : !torch.vtensor<[3,5],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,5],ui8>, !torch.int -> !torch.vtensor<[3,5],i1>
    %2 = torch.vtensor.literal(dense<0x7F800000> : tensor<f32>) : !torch.vtensor<[],f32>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[3,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[3,5],i1>, !torch.vtensor<[3,5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[3,5],f32>
    %int0_3 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %5 = torch.prim.ListConstruct %int0_3, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %6 = torch.aten.amin %4, %5, %false : !torch.vtensor<[3,5],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[],f32>
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu_6 = torch.constant.device "cpu"
    %int0_7 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_4, %none_5, %int6, %cpu_6, %int0_7 : !torch.vtensor<[],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %6 : !torch.vtensor<[],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_33(%arg0: !torch.vtensor<[3,5],f32>) -> !torch.vtensor<[],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_5_torch.uint8_13> : tensor<3x5xui8>) : !torch.vtensor<[3,5],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,5],ui8>, !torch.int -> !torch.vtensor<[3,5],i1>
    %2 = torch.vtensor.literal(dense<0x7F800000> : tensor<f32>) : !torch.vtensor<[],f32>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[3,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[3,5],i1>, !torch.vtensor<[3,5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[3,5],f32>
    %int0_3 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %5 = torch.prim.ListConstruct %int0_3, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %6 = torch.aten.amin %4, %5, %false : !torch.vtensor<[3,5],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[],f32>
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu_6 = torch.constant.device "cpu"
    %int0_7 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_4, %none_5, %int6, %cpu_6, %int0_7 : !torch.vtensor<[],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %6 : !torch.vtensor<[],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_34(%arg0: !torch.vtensor<[3,5],f32>) -> !torch.vtensor<[],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_5_torch.uint8_14> : tensor<3x5xui8>) : !torch.vtensor<[3,5],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,5],ui8>, !torch.int -> !torch.vtensor<[3,5],i1>
    %2 = torch.vtensor.literal(dense<0x7F800000> : tensor<f32>) : !torch.vtensor<[],f32>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[3,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[3,5],i1>, !torch.vtensor<[3,5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[3,5],f32>
    %int0_3 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %5 = torch.prim.ListConstruct %int0_3, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %6 = torch.aten.amin %4, %5, %false : !torch.vtensor<[3,5],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[],f32>
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu_6 = torch.constant.device "cpu"
    %int0_7 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_4, %none_5, %int6, %cpu_6, %int0_7 : !torch.vtensor<[],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %6 : !torch.vtensor<[],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_35(%arg0: !torch.vtensor<[3,5],f32>) -> !torch.vtensor<[],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_5_torch.uint8_15> : tensor<3x5xui8>) : !torch.vtensor<[3,5],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,5],ui8>, !torch.int -> !torch.vtensor<[3,5],i1>
    %2 = torch.vtensor.literal(dense<0x7F800000> : tensor<f32>) : !torch.vtensor<[],f32>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[3,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[3,5],i1>, !torch.vtensor<[3,5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[3,5],f32>
    %int0_3 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %5 = torch.prim.ListConstruct %int0_3, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %6 = torch.aten.amin %4, %5, %false : !torch.vtensor<[3,5],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[],f32>
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu_6 = torch.constant.device "cpu"
    %int0_7 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_4, %none_5, %int6, %cpu_6, %int0_7 : !torch.vtensor<[],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %6 : !torch.vtensor<[],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_36(%arg0: !torch.vtensor<[3,5],f32>) -> !torch.vtensor<[1,1],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %0 = torch.prim.ListConstruct %int0, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %1 = torch.aten.amin %arg0, %0, %true : !torch.vtensor<[3,5],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,1],f32>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu = torch.constant.device "cpu"
    %int0_1 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int6, %cpu, %int0_1 : !torch.vtensor<[1,1],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %1 : !torch.vtensor<[1,1],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_37(%arg0: !torch.vtensor<[3,5],f32>) -> !torch.vtensor<[1,1],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_5_torch.uint8_16> : tensor<3x5xui8>) : !torch.vtensor<[3,5],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,5],ui8>, !torch.int -> !torch.vtensor<[3,5],i1>
    %2 = torch.vtensor.literal(dense<0x7F800000> : tensor<f32>) : !torch.vtensor<[],f32>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[3,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[3,5],i1>, !torch.vtensor<[3,5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[3,5],f32>
    %int0_3 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %5 = torch.prim.ListConstruct %int0_3, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %6 = torch.aten.amin %4, %5, %true : !torch.vtensor<[3,5],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,1],f32>
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu_6 = torch.constant.device "cpu"
    %int0_7 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_4, %none_5, %int6, %cpu_6, %int0_7 : !torch.vtensor<[1,1],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %6 : !torch.vtensor<[1,1],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_38(%arg0: !torch.vtensor<[3,5],f32>) -> !torch.vtensor<[1,1],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_5_torch.uint8_17> : tensor<3x5xui8>) : !torch.vtensor<[3,5],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,5],ui8>, !torch.int -> !torch.vtensor<[3,5],i1>
    %2 = torch.vtensor.literal(dense<0x7F800000> : tensor<f32>) : !torch.vtensor<[],f32>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[3,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[3,5],i1>, !torch.vtensor<[3,5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[3,5],f32>
    %int0_3 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %5 = torch.prim.ListConstruct %int0_3, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %6 = torch.aten.amin %4, %5, %true : !torch.vtensor<[3,5],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,1],f32>
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu_6 = torch.constant.device "cpu"
    %int0_7 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_4, %none_5, %int6, %cpu_6, %int0_7 : !torch.vtensor<[1,1],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %6 : !torch.vtensor<[1,1],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_39(%arg0: !torch.vtensor<[3,5],f32>) -> !torch.vtensor<[1,1],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_5_torch.uint8_18> : tensor<3x5xui8>) : !torch.vtensor<[3,5],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,5],ui8>, !torch.int -> !torch.vtensor<[3,5],i1>
    %2 = torch.vtensor.literal(dense<0x7F800000> : tensor<f32>) : !torch.vtensor<[],f32>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[3,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[3,5],i1>, !torch.vtensor<[3,5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[3,5],f32>
    %int0_3 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %5 = torch.prim.ListConstruct %int0_3, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %6 = torch.aten.amin %4, %5, %true : !torch.vtensor<[3,5],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,1],f32>
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu_6 = torch.constant.device "cpu"
    %int0_7 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_4, %none_5, %int6, %cpu_6, %int0_7 : !torch.vtensor<[1,1],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %6 : !torch.vtensor<[1,1],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_40(%arg0: !torch.vtensor<[3,5],f32>) -> !torch.vtensor<[1,1],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_5_torch.uint8_19> : tensor<3x5xui8>) : !torch.vtensor<[3,5],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,5],ui8>, !torch.int -> !torch.vtensor<[3,5],i1>
    %2 = torch.vtensor.literal(dense<0x7F800000> : tensor<f32>) : !torch.vtensor<[],f32>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[3,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[3,5],i1>, !torch.vtensor<[3,5],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[3,5],f32>
    %int0_3 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %5 = torch.prim.ListConstruct %int0_3, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %6 = torch.aten.amin %4, %5, %true : !torch.vtensor<[3,5],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,1],f32>
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu_6 = torch.constant.device "cpu"
    %int0_7 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_4, %none_5, %int6, %cpu_6, %int0_7 : !torch.vtensor<[1,1],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %6 : !torch.vtensor<[1,1],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_41(%arg0: !torch.vtensor<[3,2,1,2],f32>) -> !torch.vtensor<[],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %int2 = torch.constant.int 2
    %int3 = torch.constant.int 3
    %0 = torch.prim.ListConstruct %int0, %int1, %int2, %int3 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %1 = torch.aten.amin %arg0, %0, %false : !torch.vtensor<[3,2,1,2],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[],f32>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu = torch.constant.device "cpu"
    %int0_1 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int6, %cpu, %int0_1 : !torch.vtensor<[],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %1 : !torch.vtensor<[],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_42(%arg0: !torch.vtensor<[3,2,1,2],f32>) -> !torch.vtensor<[],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_2_1_2_torch.uint8> : tensor<3x2x1x2xui8>) : !torch.vtensor<[3,2,1,2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,2,1,2],ui8>, !torch.int -> !torch.vtensor<[3,2,1,2],i1>
    %2 = torch.vtensor.literal(dense<0x7F800000> : tensor<f32>) : !torch.vtensor<[],f32>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[3,2,1,2],f32>
    %int0_3 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %int2 = torch.constant.int 2
    %int3 = torch.constant.int 3
    %5 = torch.prim.ListConstruct %int0_3, %int1, %int2, %int3 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %6 = torch.aten.amin %4, %5, %false : !torch.vtensor<[3,2,1,2],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[],f32>
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu_6 = torch.constant.device "cpu"
    %int0_7 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_4, %none_5, %int6, %cpu_6, %int0_7 : !torch.vtensor<[],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %6 : !torch.vtensor<[],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_43(%arg0: !torch.vtensor<[3,2,1,2],f32>) -> !torch.vtensor<[],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_2_1_1_torch.uint8> : tensor<3x2x1x1xui8>) : !torch.vtensor<[3,2,1,1],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,2,1,1],ui8>, !torch.int -> !torch.vtensor<[3,2,1,1],i1>
    %2 = torch.vtensor.literal(dense<0x7F800000> : tensor<f32>) : !torch.vtensor<[],f32>
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
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[3,2,1,2],f32>
    %int0_10 = torch.constant.int 0
    %int1_11 = torch.constant.int 1
    %int2_12 = torch.constant.int 2
    %int3_13 = torch.constant.int 3
    %8 = torch.prim.ListConstruct %int0_10, %int1_11, %int2_12, %int3_13 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_14 = torch.constant.bool false
    %9 = torch.aten.amin %7, %8, %false_14 : !torch.vtensor<[3,2,1,2],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[],f32>
    %none_15 = torch.constant.none
    %none_16 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu_17 = torch.constant.device "cpu"
    %int0_18 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %9, %none_15, %none_16, %int6, %cpu_17, %int0_18 : !torch.vtensor<[],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %9 : !torch.vtensor<[],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_44(%arg0: !torch.vtensor<[3,2,1,2],f32>) -> !torch.vtensor<[],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_1_1_2_torch.uint8> : tensor<3x1x1x2xui8>) : !torch.vtensor<[3,1,1,2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,1,1,2],ui8>, !torch.int -> !torch.vtensor<[3,1,1,2],i1>
    %2 = torch.vtensor.literal(dense<0x7F800000> : tensor<f32>) : !torch.vtensor<[],f32>
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
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[3,2,1,2],f32>
    %int0_10 = torch.constant.int 0
    %int1_11 = torch.constant.int 1
    %int2_12 = torch.constant.int 2
    %int3_13 = torch.constant.int 3
    %8 = torch.prim.ListConstruct %int0_10, %int1_11, %int2_12, %int3_13 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_14 = torch.constant.bool false
    %9 = torch.aten.amin %7, %8, %false_14 : !torch.vtensor<[3,2,1,2],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[],f32>
    %none_15 = torch.constant.none
    %none_16 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu_17 = torch.constant.device "cpu"
    %int0_18 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %9, %none_15, %none_16, %int6, %cpu_17, %int0_18 : !torch.vtensor<[],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %9 : !torch.vtensor<[],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_45(%arg0: !torch.vtensor<[3,2,1,2],f32>) -> !torch.vtensor<[],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_1_2_1_2_torch.uint8> : tensor<1x2x1x2xui8>) : !torch.vtensor<[1,2,1,2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[1,2,1,2],ui8>, !torch.int -> !torch.vtensor<[1,2,1,2],i1>
    %2 = torch.vtensor.literal(dense<0x7F800000> : tensor<f32>) : !torch.vtensor<[],f32>
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
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[3,2,1,2],f32>
    %int0_10 = torch.constant.int 0
    %int1_11 = torch.constant.int 1
    %int2_12 = torch.constant.int 2
    %int3_13 = torch.constant.int 3
    %8 = torch.prim.ListConstruct %int0_10, %int1_11, %int2_12, %int3_13 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_14 = torch.constant.bool false
    %9 = torch.aten.amin %7, %8, %false_14 : !torch.vtensor<[3,2,1,2],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[],f32>
    %none_15 = torch.constant.none
    %none_16 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu_17 = torch.constant.device "cpu"
    %int0_18 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %9, %none_15, %none_16, %int6, %cpu_17, %int0_18 : !torch.vtensor<[],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %9 : !torch.vtensor<[],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_46(%arg0: !torch.vtensor<[3,2,1,2],f32>) -> !torch.vtensor<[],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_2_1_2_torch.uint8> : tensor<2x1x2xui8>) : !torch.vtensor<[2,1,2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[2,1,2],ui8>, !torch.int -> !torch.vtensor<[2,1,2],i1>
    %2 = torch.vtensor.literal(dense<0x7F800000> : tensor<f32>) : !torch.vtensor<[],f32>
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
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[3,2,1,2],f32>
    %int0_10 = torch.constant.int 0
    %int1_11 = torch.constant.int 1
    %int2_12 = torch.constant.int 2
    %int3_13 = torch.constant.int 3
    %8 = torch.prim.ListConstruct %int0_10, %int1_11, %int2_12, %int3_13 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_14 = torch.constant.bool false
    %9 = torch.aten.amin %7, %8, %false_14 : !torch.vtensor<[3,2,1,2],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[],f32>
    %none_15 = torch.constant.none
    %none_16 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu_17 = torch.constant.device "cpu"
    %int0_18 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %9, %none_15, %none_16, %int6, %cpu_17, %int0_18 : !torch.vtensor<[],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %9 : !torch.vtensor<[],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_47(%arg0: !torch.vtensor<[3,2,1,2],f32>) -> !torch.vtensor<[],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_2_torch.uint8_4> : tensor<2xui8>) : !torch.vtensor<[2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[2],ui8>, !torch.int -> !torch.vtensor<[2],i1>
    %2 = torch.vtensor.literal(dense<0x7F800000> : tensor<f32>) : !torch.vtensor<[],f32>
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
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[3,2,1,2],f32>
    %int0_10 = torch.constant.int 0
    %int1_11 = torch.constant.int 1
    %int2_12 = torch.constant.int 2
    %int3_13 = torch.constant.int 3
    %8 = torch.prim.ListConstruct %int0_10, %int1_11, %int2_12, %int3_13 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_14 = torch.constant.bool false
    %9 = torch.aten.amin %7, %8, %false_14 : !torch.vtensor<[3,2,1,2],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[],f32>
    %none_15 = torch.constant.none
    %none_16 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu_17 = torch.constant.device "cpu"
    %int0_18 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %9, %none_15, %none_16, %int6, %cpu_17, %int0_18 : !torch.vtensor<[],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %9 : !torch.vtensor<[],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_48(%arg0: !torch.vtensor<[3,2,1,2],f32>) -> !torch.vtensor<[1,2,1,2],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %1 = torch.aten.amin %arg0, %0, %true : !torch.vtensor<[3,2,1,2],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,2,1,2],f32>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu = torch.constant.device "cpu"
    %int0_1 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int6, %cpu, %int0_1 : !torch.vtensor<[1,2,1,2],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %1 : !torch.vtensor<[1,2,1,2],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_49(%arg0: !torch.vtensor<[3,2,1,2],f32>) -> !torch.vtensor<[1,2,1,2],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_2_1_2_torch.uint8_1> : tensor<3x2x1x2xui8>) : !torch.vtensor<[3,2,1,2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,2,1,2],ui8>, !torch.int -> !torch.vtensor<[3,2,1,2],i1>
    %2 = torch.vtensor.literal(dense<0x7F800000> : tensor<f32>) : !torch.vtensor<[],f32>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[3,2,1,2],f32>
    %int0_3 = torch.constant.int 0
    %5 = torch.prim.ListConstruct %int0_3 : (!torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %6 = torch.aten.amin %4, %5, %true : !torch.vtensor<[3,2,1,2],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,2,1,2],f32>
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu_6 = torch.constant.device "cpu"
    %int0_7 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_4, %none_5, %int6, %cpu_6, %int0_7 : !torch.vtensor<[1,2,1,2],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %6 : !torch.vtensor<[1,2,1,2],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_50(%arg0: !torch.vtensor<[3,2,1,2],f32>) -> !torch.vtensor<[1,2,1,2],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_2_1_1_torch.uint8_1> : tensor<3x2x1x1xui8>) : !torch.vtensor<[3,2,1,1],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,2,1,1],ui8>, !torch.int -> !torch.vtensor<[3,2,1,1],i1>
    %2 = torch.vtensor.literal(dense<0x7F800000> : tensor<f32>) : !torch.vtensor<[],f32>
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
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[3,2,1,2],f32>
    %int0_10 = torch.constant.int 0
    %8 = torch.prim.ListConstruct %int0_10 : (!torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %9 = torch.aten.amin %7, %8, %true : !torch.vtensor<[3,2,1,2],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,2,1,2],f32>
    %none_11 = torch.constant.none
    %none_12 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu_13 = torch.constant.device "cpu"
    %int0_14 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %9, %none_11, %none_12, %int6, %cpu_13, %int0_14 : !torch.vtensor<[1,2,1,2],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %9 : !torch.vtensor<[1,2,1,2],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_51(%arg0: !torch.vtensor<[3,2,1,2],f32>) -> !torch.vtensor<[1,2,1,2],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_1_1_2_torch.uint8_1> : tensor<3x1x1x2xui8>) : !torch.vtensor<[3,1,1,2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,1,1,2],ui8>, !torch.int -> !torch.vtensor<[3,1,1,2],i1>
    %2 = torch.vtensor.literal(dense<0x7F800000> : tensor<f32>) : !torch.vtensor<[],f32>
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
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[3,2,1,2],f32>
    %int0_10 = torch.constant.int 0
    %8 = torch.prim.ListConstruct %int0_10 : (!torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %9 = torch.aten.amin %7, %8, %true : !torch.vtensor<[3,2,1,2],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,2,1,2],f32>
    %none_11 = torch.constant.none
    %none_12 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu_13 = torch.constant.device "cpu"
    %int0_14 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %9, %none_11, %none_12, %int6, %cpu_13, %int0_14 : !torch.vtensor<[1,2,1,2],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %9 : !torch.vtensor<[1,2,1,2],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_52(%arg0: !torch.vtensor<[3,2,1,2],f32>) -> !torch.vtensor<[1,2,1,2],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_1_2_1_2_torch.uint8_1> : tensor<1x2x1x2xui8>) : !torch.vtensor<[1,2,1,2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[1,2,1,2],ui8>, !torch.int -> !torch.vtensor<[1,2,1,2],i1>
    %2 = torch.vtensor.literal(dense<0x7F800000> : tensor<f32>) : !torch.vtensor<[],f32>
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
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[3,2,1,2],f32>
    %int0_10 = torch.constant.int 0
    %8 = torch.prim.ListConstruct %int0_10 : (!torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %9 = torch.aten.amin %7, %8, %true : !torch.vtensor<[3,2,1,2],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,2,1,2],f32>
    %none_11 = torch.constant.none
    %none_12 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu_13 = torch.constant.device "cpu"
    %int0_14 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %9, %none_11, %none_12, %int6, %cpu_13, %int0_14 : !torch.vtensor<[1,2,1,2],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %9 : !torch.vtensor<[1,2,1,2],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_53(%arg0: !torch.vtensor<[3,2,1,2],f32>) -> !torch.vtensor<[1,2,1,2],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_2_1_2_torch.uint8_1> : tensor<2x1x2xui8>) : !torch.vtensor<[2,1,2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[2,1,2],ui8>, !torch.int -> !torch.vtensor<[2,1,2],i1>
    %2 = torch.vtensor.literal(dense<0x7F800000> : tensor<f32>) : !torch.vtensor<[],f32>
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
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[3,2,1,2],f32>
    %int0_10 = torch.constant.int 0
    %8 = torch.prim.ListConstruct %int0_10 : (!torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %9 = torch.aten.amin %7, %8, %true : !torch.vtensor<[3,2,1,2],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,2,1,2],f32>
    %none_11 = torch.constant.none
    %none_12 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu_13 = torch.constant.device "cpu"
    %int0_14 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %9, %none_11, %none_12, %int6, %cpu_13, %int0_14 : !torch.vtensor<[1,2,1,2],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %9 : !torch.vtensor<[1,2,1,2],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_54(%arg0: !torch.vtensor<[3,2,1,2],f32>) -> !torch.vtensor<[1,2,1,2],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_2_torch.uint8_5> : tensor<2xui8>) : !torch.vtensor<[2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[2],ui8>, !torch.int -> !torch.vtensor<[2],i1>
    %2 = torch.vtensor.literal(dense<0x7F800000> : tensor<f32>) : !torch.vtensor<[],f32>
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
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[3,2,1,2],f32>
    %int0_10 = torch.constant.int 0
    %8 = torch.prim.ListConstruct %int0_10 : (!torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %9 = torch.aten.amin %7, %8, %true : !torch.vtensor<[3,2,1,2],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,2,1,2],f32>
    %none_11 = torch.constant.none
    %none_12 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu_13 = torch.constant.device "cpu"
    %int0_14 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %9, %none_11, %none_12, %int6, %cpu_13, %int0_14 : !torch.vtensor<[1,2,1,2],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %9 : !torch.vtensor<[1,2,1,2],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_55(%arg0: !torch.vtensor<[3,2,1,2],f32>) -> !torch.vtensor<[3,2,1],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %int3 = torch.constant.int 3
    %0 = torch.prim.ListConstruct %int3 : (!torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %1 = torch.aten.amin %arg0, %0, %false : !torch.vtensor<[3,2,1,2],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3,2,1],f32>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int6, %cpu, %int0 : !torch.vtensor<[3,2,1],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %1 : !torch.vtensor<[3,2,1],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_56(%arg0: !torch.vtensor<[3,2,1,2],f32>) -> !torch.vtensor<[3,2,1],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_2_1_2_torch.uint8_2> : tensor<3x2x1x2xui8>) : !torch.vtensor<[3,2,1,2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,2,1,2],ui8>, !torch.int -> !torch.vtensor<[3,2,1,2],i1>
    %2 = torch.vtensor.literal(dense<0x7F800000> : tensor<f32>) : !torch.vtensor<[],f32>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[3,2,1,2],f32>
    %int3 = torch.constant.int 3
    %5 = torch.prim.ListConstruct %int3 : (!torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %6 = torch.aten.amin %4, %5, %false : !torch.vtensor<[3,2,1,2],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3,2,1],f32>
    %none_3 = torch.constant.none
    %none_4 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu_5 = torch.constant.device "cpu"
    %int0_6 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_3, %none_4, %int6, %cpu_5, %int0_6 : !torch.vtensor<[3,2,1],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %6 : !torch.vtensor<[3,2,1],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_57(%arg0: !torch.vtensor<[3,2,1,2],f32>) -> !torch.vtensor<[3,2,1],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_2_1_1_torch.uint8_2> : tensor<3x2x1x1xui8>) : !torch.vtensor<[3,2,1,1],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,2,1,1],ui8>, !torch.int -> !torch.vtensor<[3,2,1,1],i1>
    %2 = torch.vtensor.literal(dense<0x7F800000> : tensor<f32>) : !torch.vtensor<[],f32>
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
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[3,2,1,2],f32>
    %int3_10 = torch.constant.int 3
    %8 = torch.prim.ListConstruct %int3_10 : (!torch.int) -> !torch.list<int>
    %false_11 = torch.constant.bool false
    %9 = torch.aten.amin %7, %8, %false_11 : !torch.vtensor<[3,2,1,2],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3,2,1],f32>
    %none_12 = torch.constant.none
    %none_13 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu_14 = torch.constant.device "cpu"
    %int0_15 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %9, %none_12, %none_13, %int6, %cpu_14, %int0_15 : !torch.vtensor<[3,2,1],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %9 : !torch.vtensor<[3,2,1],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_58(%arg0: !torch.vtensor<[3,2,1,2],f32>) -> !torch.vtensor<[3,2,1],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_1_1_2_torch.uint8_2> : tensor<3x1x1x2xui8>) : !torch.vtensor<[3,1,1,2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,1,1,2],ui8>, !torch.int -> !torch.vtensor<[3,1,1,2],i1>
    %2 = torch.vtensor.literal(dense<0x7F800000> : tensor<f32>) : !torch.vtensor<[],f32>
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
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[3,2,1,2],f32>
    %int3_10 = torch.constant.int 3
    %8 = torch.prim.ListConstruct %int3_10 : (!torch.int) -> !torch.list<int>
    %false_11 = torch.constant.bool false
    %9 = torch.aten.amin %7, %8, %false_11 : !torch.vtensor<[3,2,1,2],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3,2,1],f32>
    %none_12 = torch.constant.none
    %none_13 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu_14 = torch.constant.device "cpu"
    %int0_15 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %9, %none_12, %none_13, %int6, %cpu_14, %int0_15 : !torch.vtensor<[3,2,1],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %9 : !torch.vtensor<[3,2,1],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_59(%arg0: !torch.vtensor<[3,2,1,2],f32>) -> !torch.vtensor<[3,2,1],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_1_2_1_2_torch.uint8_2> : tensor<1x2x1x2xui8>) : !torch.vtensor<[1,2,1,2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[1,2,1,2],ui8>, !torch.int -> !torch.vtensor<[1,2,1,2],i1>
    %2 = torch.vtensor.literal(dense<0x7F800000> : tensor<f32>) : !torch.vtensor<[],f32>
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
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[3,2,1,2],f32>
    %int3_10 = torch.constant.int 3
    %8 = torch.prim.ListConstruct %int3_10 : (!torch.int) -> !torch.list<int>
    %false_11 = torch.constant.bool false
    %9 = torch.aten.amin %7, %8, %false_11 : !torch.vtensor<[3,2,1,2],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3,2,1],f32>
    %none_12 = torch.constant.none
    %none_13 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu_14 = torch.constant.device "cpu"
    %int0_15 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %9, %none_12, %none_13, %int6, %cpu_14, %int0_15 : !torch.vtensor<[3,2,1],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %9 : !torch.vtensor<[3,2,1],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_60(%arg0: !torch.vtensor<[3,2,1,2],f32>) -> !torch.vtensor<[3,2,1],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_2_1_2_torch.uint8_2> : tensor<2x1x2xui8>) : !torch.vtensor<[2,1,2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[2,1,2],ui8>, !torch.int -> !torch.vtensor<[2,1,2],i1>
    %2 = torch.vtensor.literal(dense<0x7F800000> : tensor<f32>) : !torch.vtensor<[],f32>
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
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[3,2,1,2],f32>
    %int3_10 = torch.constant.int 3
    %8 = torch.prim.ListConstruct %int3_10 : (!torch.int) -> !torch.list<int>
    %false_11 = torch.constant.bool false
    %9 = torch.aten.amin %7, %8, %false_11 : !torch.vtensor<[3,2,1,2],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3,2,1],f32>
    %none_12 = torch.constant.none
    %none_13 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu_14 = torch.constant.device "cpu"
    %int0_15 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %9, %none_12, %none_13, %int6, %cpu_14, %int0_15 : !torch.vtensor<[3,2,1],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %9 : !torch.vtensor<[3,2,1],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_61(%arg0: !torch.vtensor<[3,2,1,2],f32>) -> !torch.vtensor<[3,2,1],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_2_torch.uint8_6> : tensor<2xui8>) : !torch.vtensor<[2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[2],ui8>, !torch.int -> !torch.vtensor<[2],i1>
    %2 = torch.vtensor.literal(dense<0x7F800000> : tensor<f32>) : !torch.vtensor<[],f32>
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
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[3,2,1,2],f32>
    %int3_10 = torch.constant.int 3
    %8 = torch.prim.ListConstruct %int3_10 : (!torch.int) -> !torch.list<int>
    %false_11 = torch.constant.bool false
    %9 = torch.aten.amin %7, %8, %false_11 : !torch.vtensor<[3,2,1,2],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3,2,1],f32>
    %none_12 = torch.constant.none
    %none_13 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu_14 = torch.constant.device "cpu"
    %int0_15 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %9, %none_12, %none_13, %int6, %cpu_14, %int0_15 : !torch.vtensor<[3,2,1],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %9 : !torch.vtensor<[3,2,1],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_62(%arg0: !torch.vtensor<[3,2,1,2],f32>) -> !torch.vtensor<[3,2,1,2],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %int2 = torch.constant.int 2
    %0 = torch.prim.ListConstruct %int2 : (!torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %1 = torch.aten.amin %arg0, %0, %true : !torch.vtensor<[3,2,1,2],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3,2,1,2],f32>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int6, %cpu, %int0 : !torch.vtensor<[3,2,1,2],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %1 : !torch.vtensor<[3,2,1,2],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_63(%arg0: !torch.vtensor<[3,2,1,2],f32>) -> !torch.vtensor<[3,2,1,2],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_2_1_2_torch.uint8_3> : tensor<3x2x1x2xui8>) : !torch.vtensor<[3,2,1,2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,2,1,2],ui8>, !torch.int -> !torch.vtensor<[3,2,1,2],i1>
    %2 = torch.vtensor.literal(dense<0x7F800000> : tensor<f32>) : !torch.vtensor<[],f32>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[3,2,1,2],f32>
    %int2 = torch.constant.int 2
    %5 = torch.prim.ListConstruct %int2 : (!torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %6 = torch.aten.amin %4, %5, %true : !torch.vtensor<[3,2,1,2],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3,2,1,2],f32>
    %none_3 = torch.constant.none
    %none_4 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu_5 = torch.constant.device "cpu"
    %int0_6 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_3, %none_4, %int6, %cpu_5, %int0_6 : !torch.vtensor<[3,2,1,2],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %6 : !torch.vtensor<[3,2,1,2],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_64(%arg0: !torch.vtensor<[3,2,1,2],f32>) -> !torch.vtensor<[3,2,1,2],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_2_1_1_torch.uint8_3> : tensor<3x2x1x1xui8>) : !torch.vtensor<[3,2,1,1],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,2,1,1],ui8>, !torch.int -> !torch.vtensor<[3,2,1,1],i1>
    %2 = torch.vtensor.literal(dense<0x7F800000> : tensor<f32>) : !torch.vtensor<[],f32>
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
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[3,2,1,2],f32>
    %int2_10 = torch.constant.int 2
    %8 = torch.prim.ListConstruct %int2_10 : (!torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %9 = torch.aten.amin %7, %8, %true : !torch.vtensor<[3,2,1,2],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3,2,1,2],f32>
    %none_11 = torch.constant.none
    %none_12 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu_13 = torch.constant.device "cpu"
    %int0_14 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %9, %none_11, %none_12, %int6, %cpu_13, %int0_14 : !torch.vtensor<[3,2,1,2],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %9 : !torch.vtensor<[3,2,1,2],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_65(%arg0: !torch.vtensor<[3,2,1,2],f32>) -> !torch.vtensor<[3,2,1,2],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_1_1_2_torch.uint8_3> : tensor<3x1x1x2xui8>) : !torch.vtensor<[3,1,1,2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,1,1,2],ui8>, !torch.int -> !torch.vtensor<[3,1,1,2],i1>
    %2 = torch.vtensor.literal(dense<0x7F800000> : tensor<f32>) : !torch.vtensor<[],f32>
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
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[3,2,1,2],f32>
    %int2_10 = torch.constant.int 2
    %8 = torch.prim.ListConstruct %int2_10 : (!torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %9 = torch.aten.amin %7, %8, %true : !torch.vtensor<[3,2,1,2],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3,2,1,2],f32>
    %none_11 = torch.constant.none
    %none_12 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu_13 = torch.constant.device "cpu"
    %int0_14 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %9, %none_11, %none_12, %int6, %cpu_13, %int0_14 : !torch.vtensor<[3,2,1,2],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %9 : !torch.vtensor<[3,2,1,2],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_66(%arg0: !torch.vtensor<[3,2,1,2],f32>) -> !torch.vtensor<[3,2,1,2],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_1_2_1_2_torch.uint8_3> : tensor<1x2x1x2xui8>) : !torch.vtensor<[1,2,1,2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[1,2,1,2],ui8>, !torch.int -> !torch.vtensor<[1,2,1,2],i1>
    %2 = torch.vtensor.literal(dense<0x7F800000> : tensor<f32>) : !torch.vtensor<[],f32>
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
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[3,2,1,2],f32>
    %int2_10 = torch.constant.int 2
    %8 = torch.prim.ListConstruct %int2_10 : (!torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %9 = torch.aten.amin %7, %8, %true : !torch.vtensor<[3,2,1,2],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3,2,1,2],f32>
    %none_11 = torch.constant.none
    %none_12 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu_13 = torch.constant.device "cpu"
    %int0_14 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %9, %none_11, %none_12, %int6, %cpu_13, %int0_14 : !torch.vtensor<[3,2,1,2],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %9 : !torch.vtensor<[3,2,1,2],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_67(%arg0: !torch.vtensor<[3,2,1,2],f32>) -> !torch.vtensor<[3,2,1,2],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_2_1_2_torch.uint8_3> : tensor<2x1x2xui8>) : !torch.vtensor<[2,1,2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[2,1,2],ui8>, !torch.int -> !torch.vtensor<[2,1,2],i1>
    %2 = torch.vtensor.literal(dense<0x7F800000> : tensor<f32>) : !torch.vtensor<[],f32>
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
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[3,2,1,2],f32>
    %int2_10 = torch.constant.int 2
    %8 = torch.prim.ListConstruct %int2_10 : (!torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %9 = torch.aten.amin %7, %8, %true : !torch.vtensor<[3,2,1,2],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3,2,1,2],f32>
    %none_11 = torch.constant.none
    %none_12 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu_13 = torch.constant.device "cpu"
    %int0_14 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %9, %none_11, %none_12, %int6, %cpu_13, %int0_14 : !torch.vtensor<[3,2,1,2],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %9 : !torch.vtensor<[3,2,1,2],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_68(%arg0: !torch.vtensor<[3,2,1,2],f32>) -> !torch.vtensor<[3,2,1,2],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_2_torch.uint8_7> : tensor<2xui8>) : !torch.vtensor<[2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[2],ui8>, !torch.int -> !torch.vtensor<[2],i1>
    %2 = torch.vtensor.literal(dense<0x7F800000> : tensor<f32>) : !torch.vtensor<[],f32>
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
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[3,2,1,2],f32>
    %int2_10 = torch.constant.int 2
    %8 = torch.prim.ListConstruct %int2_10 : (!torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %9 = torch.aten.amin %7, %8, %true : !torch.vtensor<[3,2,1,2],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3,2,1,2],f32>
    %none_11 = torch.constant.none
    %none_12 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu_13 = torch.constant.device "cpu"
    %int0_14 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %9, %none_11, %none_12, %int6, %cpu_13, %int0_14 : !torch.vtensor<[3,2,1,2],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %9 : !torch.vtensor<[3,2,1,2],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_69(%arg0: !torch.vtensor<[3,2,1,2],f32>) -> !torch.vtensor<[],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %int2 = torch.constant.int 2
    %int3 = torch.constant.int 3
    %0 = torch.prim.ListConstruct %int0, %int1, %int2, %int3 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %1 = torch.aten.amin %arg0, %0, %false : !torch.vtensor<[3,2,1,2],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[],f32>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu = torch.constant.device "cpu"
    %int0_1 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int6, %cpu, %int0_1 : !torch.vtensor<[],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %1 : !torch.vtensor<[],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_70(%arg0: !torch.vtensor<[3,2,1,2],f32>) -> !torch.vtensor<[],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_2_1_2_torch.uint8_4> : tensor<3x2x1x2xui8>) : !torch.vtensor<[3,2,1,2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,2,1,2],ui8>, !torch.int -> !torch.vtensor<[3,2,1,2],i1>
    %2 = torch.vtensor.literal(dense<0x7F800000> : tensor<f32>) : !torch.vtensor<[],f32>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[3,2,1,2],f32>
    %int0_3 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %int2 = torch.constant.int 2
    %int3 = torch.constant.int 3
    %5 = torch.prim.ListConstruct %int0_3, %int1, %int2, %int3 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %6 = torch.aten.amin %4, %5, %false : !torch.vtensor<[3,2,1,2],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[],f32>
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu_6 = torch.constant.device "cpu"
    %int0_7 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_4, %none_5, %int6, %cpu_6, %int0_7 : !torch.vtensor<[],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %6 : !torch.vtensor<[],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_71(%arg0: !torch.vtensor<[3,2,1,2],f32>) -> !torch.vtensor<[],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_2_1_1_torch.uint8_4> : tensor<3x2x1x1xui8>) : !torch.vtensor<[3,2,1,1],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,2,1,1],ui8>, !torch.int -> !torch.vtensor<[3,2,1,1],i1>
    %2 = torch.vtensor.literal(dense<0x7F800000> : tensor<f32>) : !torch.vtensor<[],f32>
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
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[3,2,1,2],f32>
    %int0_10 = torch.constant.int 0
    %int1_11 = torch.constant.int 1
    %int2_12 = torch.constant.int 2
    %int3_13 = torch.constant.int 3
    %8 = torch.prim.ListConstruct %int0_10, %int1_11, %int2_12, %int3_13 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_14 = torch.constant.bool false
    %9 = torch.aten.amin %7, %8, %false_14 : !torch.vtensor<[3,2,1,2],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[],f32>
    %none_15 = torch.constant.none
    %none_16 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu_17 = torch.constant.device "cpu"
    %int0_18 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %9, %none_15, %none_16, %int6, %cpu_17, %int0_18 : !torch.vtensor<[],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %9 : !torch.vtensor<[],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_72(%arg0: !torch.vtensor<[3,2,1,2],f32>) -> !torch.vtensor<[],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_1_1_2_torch.uint8_4> : tensor<3x1x1x2xui8>) : !torch.vtensor<[3,1,1,2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,1,1,2],ui8>, !torch.int -> !torch.vtensor<[3,1,1,2],i1>
    %2 = torch.vtensor.literal(dense<0x7F800000> : tensor<f32>) : !torch.vtensor<[],f32>
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
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[3,2,1,2],f32>
    %int0_10 = torch.constant.int 0
    %int1_11 = torch.constant.int 1
    %int2_12 = torch.constant.int 2
    %int3_13 = torch.constant.int 3
    %8 = torch.prim.ListConstruct %int0_10, %int1_11, %int2_12, %int3_13 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_14 = torch.constant.bool false
    %9 = torch.aten.amin %7, %8, %false_14 : !torch.vtensor<[3,2,1,2],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[],f32>
    %none_15 = torch.constant.none
    %none_16 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu_17 = torch.constant.device "cpu"
    %int0_18 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %9, %none_15, %none_16, %int6, %cpu_17, %int0_18 : !torch.vtensor<[],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %9 : !torch.vtensor<[],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_73(%arg0: !torch.vtensor<[3,2,1,2],f32>) -> !torch.vtensor<[],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_1_2_1_2_torch.uint8_4> : tensor<1x2x1x2xui8>) : !torch.vtensor<[1,2,1,2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[1,2,1,2],ui8>, !torch.int -> !torch.vtensor<[1,2,1,2],i1>
    %2 = torch.vtensor.literal(dense<0x7F800000> : tensor<f32>) : !torch.vtensor<[],f32>
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
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[3,2,1,2],f32>
    %int0_10 = torch.constant.int 0
    %int1_11 = torch.constant.int 1
    %int2_12 = torch.constant.int 2
    %int3_13 = torch.constant.int 3
    %8 = torch.prim.ListConstruct %int0_10, %int1_11, %int2_12, %int3_13 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_14 = torch.constant.bool false
    %9 = torch.aten.amin %7, %8, %false_14 : !torch.vtensor<[3,2,1,2],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[],f32>
    %none_15 = torch.constant.none
    %none_16 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu_17 = torch.constant.device "cpu"
    %int0_18 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %9, %none_15, %none_16, %int6, %cpu_17, %int0_18 : !torch.vtensor<[],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %9 : !torch.vtensor<[],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_74(%arg0: !torch.vtensor<[3,2,1,2],f32>) -> !torch.vtensor<[],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_2_1_2_torch.uint8_4> : tensor<2x1x2xui8>) : !torch.vtensor<[2,1,2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[2,1,2],ui8>, !torch.int -> !torch.vtensor<[2,1,2],i1>
    %2 = torch.vtensor.literal(dense<0x7F800000> : tensor<f32>) : !torch.vtensor<[],f32>
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
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[3,2,1,2],f32>
    %int0_10 = torch.constant.int 0
    %int1_11 = torch.constant.int 1
    %int2_12 = torch.constant.int 2
    %int3_13 = torch.constant.int 3
    %8 = torch.prim.ListConstruct %int0_10, %int1_11, %int2_12, %int3_13 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_14 = torch.constant.bool false
    %9 = torch.aten.amin %7, %8, %false_14 : !torch.vtensor<[3,2,1,2],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[],f32>
    %none_15 = torch.constant.none
    %none_16 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu_17 = torch.constant.device "cpu"
    %int0_18 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %9, %none_15, %none_16, %int6, %cpu_17, %int0_18 : !torch.vtensor<[],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %9 : !torch.vtensor<[],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_75(%arg0: !torch.vtensor<[3,2,1,2],f32>) -> !torch.vtensor<[],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_2_torch.uint8_8> : tensor<2xui8>) : !torch.vtensor<[2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[2],ui8>, !torch.int -> !torch.vtensor<[2],i1>
    %2 = torch.vtensor.literal(dense<0x7F800000> : tensor<f32>) : !torch.vtensor<[],f32>
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
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[3,2,1,2],f32>
    %int0_10 = torch.constant.int 0
    %int1_11 = torch.constant.int 1
    %int2_12 = torch.constant.int 2
    %int3_13 = torch.constant.int 3
    %8 = torch.prim.ListConstruct %int0_10, %int1_11, %int2_12, %int3_13 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false_14 = torch.constant.bool false
    %9 = torch.aten.amin %7, %8, %false_14 : !torch.vtensor<[3,2,1,2],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[],f32>
    %none_15 = torch.constant.none
    %none_16 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu_17 = torch.constant.device "cpu"
    %int0_18 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %9, %none_15, %none_16, %int6, %cpu_17, %int0_18 : !torch.vtensor<[],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %9 : !torch.vtensor<[],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_76(%arg0: !torch.vtensor<[3,2,1,2],f32>) -> !torch.vtensor<[1,2,1,1],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %int0 = torch.constant.int 0
    %int3 = torch.constant.int 3
    %0 = torch.prim.ListConstruct %int0, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %1 = torch.aten.amin %arg0, %0, %true : !torch.vtensor<[3,2,1,2],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,2,1,1],f32>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu = torch.constant.device "cpu"
    %int0_1 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int6, %cpu, %int0_1 : !torch.vtensor<[1,2,1,1],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %1 : !torch.vtensor<[1,2,1,1],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_77(%arg0: !torch.vtensor<[3,2,1,2],f32>) -> !torch.vtensor<[1,2,1,1],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_2_1_2_torch.uint8_5> : tensor<3x2x1x2xui8>) : !torch.vtensor<[3,2,1,2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,2,1,2],ui8>, !torch.int -> !torch.vtensor<[3,2,1,2],i1>
    %2 = torch.vtensor.literal(dense<0x7F800000> : tensor<f32>) : !torch.vtensor<[],f32>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[3,2,1,2],f32>
    %int0_3 = torch.constant.int 0
    %int3 = torch.constant.int 3
    %5 = torch.prim.ListConstruct %int0_3, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %6 = torch.aten.amin %4, %5, %true : !torch.vtensor<[3,2,1,2],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,2,1,1],f32>
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu_6 = torch.constant.device "cpu"
    %int0_7 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_4, %none_5, %int6, %cpu_6, %int0_7 : !torch.vtensor<[1,2,1,1],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %6 : !torch.vtensor<[1,2,1,1],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_78(%arg0: !torch.vtensor<[3,2,1,2],f32>) -> !torch.vtensor<[1,2,1,1],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_2_1_1_torch.uint8_5> : tensor<3x2x1x1xui8>) : !torch.vtensor<[3,2,1,1],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,2,1,1],ui8>, !torch.int -> !torch.vtensor<[3,2,1,1],i1>
    %2 = torch.vtensor.literal(dense<0x7F800000> : tensor<f32>) : !torch.vtensor<[],f32>
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
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[3,2,1,2],f32>
    %int0_10 = torch.constant.int 0
    %int3_11 = torch.constant.int 3
    %8 = torch.prim.ListConstruct %int0_10, %int3_11 : (!torch.int, !torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %9 = torch.aten.amin %7, %8, %true : !torch.vtensor<[3,2,1,2],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,2,1,1],f32>
    %none_12 = torch.constant.none
    %none_13 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu_14 = torch.constant.device "cpu"
    %int0_15 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %9, %none_12, %none_13, %int6, %cpu_14, %int0_15 : !torch.vtensor<[1,2,1,1],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %9 : !torch.vtensor<[1,2,1,1],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_79(%arg0: !torch.vtensor<[3,2,1,2],f32>) -> !torch.vtensor<[1,2,1,1],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_1_1_2_torch.uint8_5> : tensor<3x1x1x2xui8>) : !torch.vtensor<[3,1,1,2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,1,1,2],ui8>, !torch.int -> !torch.vtensor<[3,1,1,2],i1>
    %2 = torch.vtensor.literal(dense<0x7F800000> : tensor<f32>) : !torch.vtensor<[],f32>
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
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[3,2,1,2],f32>
    %int0_10 = torch.constant.int 0
    %int3_11 = torch.constant.int 3
    %8 = torch.prim.ListConstruct %int0_10, %int3_11 : (!torch.int, !torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %9 = torch.aten.amin %7, %8, %true : !torch.vtensor<[3,2,1,2],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,2,1,1],f32>
    %none_12 = torch.constant.none
    %none_13 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu_14 = torch.constant.device "cpu"
    %int0_15 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %9, %none_12, %none_13, %int6, %cpu_14, %int0_15 : !torch.vtensor<[1,2,1,1],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %9 : !torch.vtensor<[1,2,1,1],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_80(%arg0: !torch.vtensor<[3,2,1,2],f32>) -> !torch.vtensor<[1,2,1,1],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_1_2_1_2_torch.uint8_5> : tensor<1x2x1x2xui8>) : !torch.vtensor<[1,2,1,2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[1,2,1,2],ui8>, !torch.int -> !torch.vtensor<[1,2,1,2],i1>
    %2 = torch.vtensor.literal(dense<0x7F800000> : tensor<f32>) : !torch.vtensor<[],f32>
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
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[3,2,1,2],f32>
    %int0_10 = torch.constant.int 0
    %int3_11 = torch.constant.int 3
    %8 = torch.prim.ListConstruct %int0_10, %int3_11 : (!torch.int, !torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %9 = torch.aten.amin %7, %8, %true : !torch.vtensor<[3,2,1,2],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,2,1,1],f32>
    %none_12 = torch.constant.none
    %none_13 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu_14 = torch.constant.device "cpu"
    %int0_15 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %9, %none_12, %none_13, %int6, %cpu_14, %int0_15 : !torch.vtensor<[1,2,1,1],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %9 : !torch.vtensor<[1,2,1,1],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_81(%arg0: !torch.vtensor<[3,2,1,2],f32>) -> !torch.vtensor<[1,2,1,1],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_2_1_2_torch.uint8_5> : tensor<2x1x2xui8>) : !torch.vtensor<[2,1,2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[2,1,2],ui8>, !torch.int -> !torch.vtensor<[2,1,2],i1>
    %2 = torch.vtensor.literal(dense<0x7F800000> : tensor<f32>) : !torch.vtensor<[],f32>
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
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[3,2,1,2],f32>
    %int0_10 = torch.constant.int 0
    %int3_11 = torch.constant.int 3
    %8 = torch.prim.ListConstruct %int0_10, %int3_11 : (!torch.int, !torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %9 = torch.aten.amin %7, %8, %true : !torch.vtensor<[3,2,1,2],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,2,1,1],f32>
    %none_12 = torch.constant.none
    %none_13 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu_14 = torch.constant.device "cpu"
    %int0_15 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %9, %none_12, %none_13, %int6, %cpu_14, %int0_15 : !torch.vtensor<[1,2,1,1],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %9 : !torch.vtensor<[1,2,1,1],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_82(%arg0: !torch.vtensor<[3,2,1,2],f32>) -> !torch.vtensor<[1,2,1,1],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_2_torch.uint8_9> : tensor<2xui8>) : !torch.vtensor<[2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[2],ui8>, !torch.int -> !torch.vtensor<[2],i1>
    %2 = torch.vtensor.literal(dense<0x7F800000> : tensor<f32>) : !torch.vtensor<[],f32>
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
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[3,2,1,2],f32>
    %int0_10 = torch.constant.int 0
    %int3_11 = torch.constant.int 3
    %8 = torch.prim.ListConstruct %int0_10, %int3_11 : (!torch.int, !torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %9 = torch.aten.amin %7, %8, %true : !torch.vtensor<[3,2,1,2],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,2,1,1],f32>
    %none_12 = torch.constant.none
    %none_13 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu_14 = torch.constant.device "cpu"
    %int0_15 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %9, %none_12, %none_13, %int6, %cpu_14, %int0_15 : !torch.vtensor<[1,2,1,1],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %9 : !torch.vtensor<[1,2,1,1],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_83(%arg0: !torch.vtensor<[3,2,1,2],f32>) -> !torch.vtensor<[3,1],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %int1 = torch.constant.int 1
    %int3 = torch.constant.int 3
    %0 = torch.prim.ListConstruct %int1, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %1 = torch.aten.amin %arg0, %0, %false : !torch.vtensor<[3,2,1,2],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3,1],f32>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int6, %cpu, %int0 : !torch.vtensor<[3,1],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %1 : !torch.vtensor<[3,1],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_84(%arg0: !torch.vtensor<[3,2,1,2],f32>) -> !torch.vtensor<[3,1],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_2_1_2_torch.uint8_6> : tensor<3x2x1x2xui8>) : !torch.vtensor<[3,2,1,2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,2,1,2],ui8>, !torch.int -> !torch.vtensor<[3,2,1,2],i1>
    %2 = torch.vtensor.literal(dense<0x7F800000> : tensor<f32>) : !torch.vtensor<[],f32>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[3,2,1,2],f32>
    %int1 = torch.constant.int 1
    %int3 = torch.constant.int 3
    %5 = torch.prim.ListConstruct %int1, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %6 = torch.aten.amin %4, %5, %false : !torch.vtensor<[3,2,1,2],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3,1],f32>
    %none_3 = torch.constant.none
    %none_4 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu_5 = torch.constant.device "cpu"
    %int0_6 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %6, %none_3, %none_4, %int6, %cpu_5, %int0_6 : !torch.vtensor<[3,1],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %6 : !torch.vtensor<[3,1],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_85(%arg0: !torch.vtensor<[3,2,1,2],f32>) -> !torch.vtensor<[3,1],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_2_1_1_torch.uint8_6> : tensor<3x2x1x1xui8>) : !torch.vtensor<[3,2,1,1],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,2,1,1],ui8>, !torch.int -> !torch.vtensor<[3,2,1,1],i1>
    %2 = torch.vtensor.literal(dense<0x7F800000> : tensor<f32>) : !torch.vtensor<[],f32>
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
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[3,2,1,2],f32>
    %int1_10 = torch.constant.int 1
    %int3_11 = torch.constant.int 3
    %8 = torch.prim.ListConstruct %int1_10, %int3_11 : (!torch.int, !torch.int) -> !torch.list<int>
    %false_12 = torch.constant.bool false
    %9 = torch.aten.amin %7, %8, %false_12 : !torch.vtensor<[3,2,1,2],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3,1],f32>
    %none_13 = torch.constant.none
    %none_14 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu_15 = torch.constant.device "cpu"
    %int0_16 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %9, %none_13, %none_14, %int6, %cpu_15, %int0_16 : !torch.vtensor<[3,1],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %9 : !torch.vtensor<[3,1],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_86(%arg0: !torch.vtensor<[3,2,1,2],f32>) -> !torch.vtensor<[3,1],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_1_1_2_torch.uint8_6> : tensor<3x1x1x2xui8>) : !torch.vtensor<[3,1,1,2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,1,1,2],ui8>, !torch.int -> !torch.vtensor<[3,1,1,2],i1>
    %2 = torch.vtensor.literal(dense<0x7F800000> : tensor<f32>) : !torch.vtensor<[],f32>
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
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[3,2,1,2],f32>
    %int1_10 = torch.constant.int 1
    %int3_11 = torch.constant.int 3
    %8 = torch.prim.ListConstruct %int1_10, %int3_11 : (!torch.int, !torch.int) -> !torch.list<int>
    %false_12 = torch.constant.bool false
    %9 = torch.aten.amin %7, %8, %false_12 : !torch.vtensor<[3,2,1,2],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3,1],f32>
    %none_13 = torch.constant.none
    %none_14 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu_15 = torch.constant.device "cpu"
    %int0_16 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %9, %none_13, %none_14, %int6, %cpu_15, %int0_16 : !torch.vtensor<[3,1],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %9 : !torch.vtensor<[3,1],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_87(%arg0: !torch.vtensor<[3,2,1,2],f32>) -> !torch.vtensor<[3,1],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_1_2_1_2_torch.uint8_6> : tensor<1x2x1x2xui8>) : !torch.vtensor<[1,2,1,2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[1,2,1,2],ui8>, !torch.int -> !torch.vtensor<[1,2,1,2],i1>
    %2 = torch.vtensor.literal(dense<0x7F800000> : tensor<f32>) : !torch.vtensor<[],f32>
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
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[3,2,1,2],f32>
    %int1_10 = torch.constant.int 1
    %int3_11 = torch.constant.int 3
    %8 = torch.prim.ListConstruct %int1_10, %int3_11 : (!torch.int, !torch.int) -> !torch.list<int>
    %false_12 = torch.constant.bool false
    %9 = torch.aten.amin %7, %8, %false_12 : !torch.vtensor<[3,2,1,2],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3,1],f32>
    %none_13 = torch.constant.none
    %none_14 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu_15 = torch.constant.device "cpu"
    %int0_16 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %9, %none_13, %none_14, %int6, %cpu_15, %int0_16 : !torch.vtensor<[3,1],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %9 : !torch.vtensor<[3,1],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_88(%arg0: !torch.vtensor<[3,2,1,2],f32>) -> !torch.vtensor<[3,1],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_2_1_2_torch.uint8_6> : tensor<2x1x2xui8>) : !torch.vtensor<[2,1,2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[2,1,2],ui8>, !torch.int -> !torch.vtensor<[2,1,2],i1>
    %2 = torch.vtensor.literal(dense<0x7F800000> : tensor<f32>) : !torch.vtensor<[],f32>
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
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[3,2,1,2],f32>
    %int1_10 = torch.constant.int 1
    %int3_11 = torch.constant.int 3
    %8 = torch.prim.ListConstruct %int1_10, %int3_11 : (!torch.int, !torch.int) -> !torch.list<int>
    %false_12 = torch.constant.bool false
    %9 = torch.aten.amin %7, %8, %false_12 : !torch.vtensor<[3,2,1,2],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3,1],f32>
    %none_13 = torch.constant.none
    %none_14 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu_15 = torch.constant.device "cpu"
    %int0_16 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %9, %none_13, %none_14, %int6, %cpu_15, %int0_16 : !torch.vtensor<[3,1],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %9 : !torch.vtensor<[3,1],f32>
  }
  func.func @test_reduction_masked_amin_cpu_float32_89(%arg0: !torch.vtensor<[3,2,1,2],f32>) -> !torch.vtensor<[3,1],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_2_torch.uint8_10> : tensor<2xui8>) : !torch.vtensor<[2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[2],ui8>, !torch.int -> !torch.vtensor<[2],i1>
    %2 = torch.vtensor.literal(dense<0x7F800000> : tensor<f32>) : !torch.vtensor<[],f32>
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
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],f32>, !torch.none -> !torch.vtensor<[],f32>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],f32>, !torch.vtensor<[],f32> -> !torch.vtensor<[3,2,1,2],f32>
    %int1_10 = torch.constant.int 1
    %int3_11 = torch.constant.int 3
    %8 = torch.prim.ListConstruct %int1_10, %int3_11 : (!torch.int, !torch.int) -> !torch.list<int>
    %false_12 = torch.constant.bool false
    %9 = torch.aten.amin %7, %8, %false_12 : !torch.vtensor<[3,2,1,2],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3,1],f32>
    %none_13 = torch.constant.none
    %none_14 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu_15 = torch.constant.device "cpu"
    %int0_16 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %9, %none_13, %none_14, %int6, %cpu_15, %int0_16 : !torch.vtensor<[3,1],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    return %9 : !torch.vtensor<[3,1],f32>
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
