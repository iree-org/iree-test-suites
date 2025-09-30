module @module {
  func.func @test_reduction_masked_argmax_cpu_float16_0(%arg0: !torch.vtensor<[],f16>) -> !torch.vtensor<[],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %none = torch.constant.none
    %false = torch.constant.bool false
    %0 = torch.aten.argmax %arg0, %none, %false : !torch.vtensor<[],f16>, !torch.none, !torch.bool -> !torch.vtensor<[],si64>
    %none_0 = torch.constant.none
    %none_1 = torch.constant.none
    %int4 = torch.constant.int 4
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %0, %none_0, %none_1, %int4, %cpu, %int0 : !torch.vtensor<[],si64>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int5 = torch.constant.int 5
    %1 = torch.prims.convert_element_type %0, %int5 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[],f16>
    return %1 : !torch.vtensor<[],f16>
  }
  func.func @test_reduction_masked_argmax_cpu_float16_1(%arg0: !torch.vtensor<[],f16>) -> !torch.vtensor<[],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense<1> : tensor<ui8>) : !torch.vtensor<[],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[],ui8>, !torch.int -> !torch.vtensor<[],i1>
    %2 = torch.vtensor.literal(dense<0xFC00> : tensor<f16>) : !torch.vtensor<[],f16>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],f16>, !torch.none -> !torch.vtensor<[],f16>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[],i1>, !torch.vtensor<[],f16>, !torch.vtensor<[],f16> -> !torch.vtensor<[],f16>
    %none_3 = torch.constant.none
    %false = torch.constant.bool false
    %5 = torch.aten.argmax %4, %none_3, %false : !torch.vtensor<[],f16>, !torch.none, !torch.bool -> !torch.vtensor<[],si64>
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int4 = torch.constant.int 4
    %cpu_6 = torch.constant.device "cpu"
    %int0_7 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_4, %none_5, %int4, %cpu_6, %int0_7 : !torch.vtensor<[],si64>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int5 = torch.constant.int 5
    %6 = torch.prims.convert_element_type %5, %int5 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[],f16>
    return %6 : !torch.vtensor<[],f16>
  }
  func.func @test_reduction_masked_argmax_cpu_float16_2(%arg0: !torch.vtensor<[],f16>) -> !torch.vtensor<[],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %int0 = torch.constant.int 0
    %true = torch.constant.bool true
    %0 = torch.aten.argmax %arg0, %int0, %true : !torch.vtensor<[],f16>, !torch.int, !torch.bool -> !torch.vtensor<[],si64>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int4 = torch.constant.int 4
    %cpu = torch.constant.device "cpu"
    %int0_1 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %0, %none, %none_0, %int4, %cpu, %int0_1 : !torch.vtensor<[],si64>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int5 = torch.constant.int 5
    %1 = torch.prims.convert_element_type %0, %int5 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[],f16>
    return %1 : !torch.vtensor<[],f16>
  }
  func.func @test_reduction_masked_argmax_cpu_float16_3(%arg0: !torch.vtensor<[],f16>) -> !torch.vtensor<[],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense<1> : tensor<ui8>) : !torch.vtensor<[],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[],ui8>, !torch.int -> !torch.vtensor<[],i1>
    %2 = torch.vtensor.literal(dense<0xFC00> : tensor<f16>) : !torch.vtensor<[],f16>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],f16>, !torch.none -> !torch.vtensor<[],f16>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[],i1>, !torch.vtensor<[],f16>, !torch.vtensor<[],f16> -> !torch.vtensor<[],f16>
    %int0_3 = torch.constant.int 0
    %true = torch.constant.bool true
    %5 = torch.aten.argmax %4, %int0_3, %true : !torch.vtensor<[],f16>, !torch.int, !torch.bool -> !torch.vtensor<[],si64>
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int4 = torch.constant.int 4
    %cpu_6 = torch.constant.device "cpu"
    %int0_7 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_4, %none_5, %int4, %cpu_6, %int0_7 : !torch.vtensor<[],si64>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int5 = torch.constant.int 5
    %6 = torch.prims.convert_element_type %5, %int5 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[],f16>
    return %6 : !torch.vtensor<[],f16>
  }
  func.func @test_reduction_masked_argmax_cpu_float16_4(%arg0: !torch.vtensor<[],f16>) -> !torch.vtensor<[],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %int-1 = torch.constant.int -1
    %false = torch.constant.bool false
    %0 = torch.aten.argmax %arg0, %int-1, %false : !torch.vtensor<[],f16>, !torch.int, !torch.bool -> !torch.vtensor<[],si64>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int4 = torch.constant.int 4
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %0, %none, %none_0, %int4, %cpu, %int0 : !torch.vtensor<[],si64>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int5 = torch.constant.int 5
    %1 = torch.prims.convert_element_type %0, %int5 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[],f16>
    return %1 : !torch.vtensor<[],f16>
  }
  func.func @test_reduction_masked_argmax_cpu_float16_5(%arg0: !torch.vtensor<[],f16>) -> !torch.vtensor<[],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense<1> : tensor<ui8>) : !torch.vtensor<[],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[],ui8>, !torch.int -> !torch.vtensor<[],i1>
    %2 = torch.vtensor.literal(dense<0xFC00> : tensor<f16>) : !torch.vtensor<[],f16>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],f16>, !torch.none -> !torch.vtensor<[],f16>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[],i1>, !torch.vtensor<[],f16>, !torch.vtensor<[],f16> -> !torch.vtensor<[],f16>
    %int-1 = torch.constant.int -1
    %false = torch.constant.bool false
    %5 = torch.aten.argmax %4, %int-1, %false : !torch.vtensor<[],f16>, !torch.int, !torch.bool -> !torch.vtensor<[],si64>
    %none_3 = torch.constant.none
    %none_4 = torch.constant.none
    %int4 = torch.constant.int 4
    %cpu_5 = torch.constant.device "cpu"
    %int0_6 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_3, %none_4, %int4, %cpu_5, %int0_6 : !torch.vtensor<[],si64>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int5 = torch.constant.int 5
    %6 = torch.prims.convert_element_type %5, %int5 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[],f16>
    return %6 : !torch.vtensor<[],f16>
  }
  func.func @test_reduction_masked_argmax_cpu_float16_6(%arg0: !torch.vtensor<[2],f16>) -> !torch.vtensor<[],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %none = torch.constant.none
    %false = torch.constant.bool false
    %0 = torch.aten.argmax %arg0, %none, %false : !torch.vtensor<[2],f16>, !torch.none, !torch.bool -> !torch.vtensor<[],si64>
    %none_0 = torch.constant.none
    %none_1 = torch.constant.none
    %int4 = torch.constant.int 4
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %0, %none_0, %none_1, %int4, %cpu, %int0 : !torch.vtensor<[],si64>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int5 = torch.constant.int 5
    %1 = torch.prims.convert_element_type %0, %int5 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[],f16>
    return %1 : !torch.vtensor<[],f16>
  }
  func.func @test_reduction_masked_argmax_cpu_float16_7(%arg0: !torch.vtensor<[2],f16>) -> !torch.vtensor<[],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_2_torch.uint8> : tensor<2xui8>) : !torch.vtensor<[2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[2],ui8>, !torch.int -> !torch.vtensor<[2],i1>
    %2 = torch.vtensor.literal(dense<0xFC00> : tensor<f16>) : !torch.vtensor<[],f16>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],f16>, !torch.none -> !torch.vtensor<[],f16>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[2],i1>, !torch.vtensor<[2],f16>, !torch.vtensor<[],f16> -> !torch.vtensor<[2],f16>
    %none_3 = torch.constant.none
    %false = torch.constant.bool false
    %5 = torch.aten.argmax %4, %none_3, %false : !torch.vtensor<[2],f16>, !torch.none, !torch.bool -> !torch.vtensor<[],si64>
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int4 = torch.constant.int 4
    %cpu_6 = torch.constant.device "cpu"
    %int0_7 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_4, %none_5, %int4, %cpu_6, %int0_7 : !torch.vtensor<[],si64>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int5 = torch.constant.int 5
    %6 = torch.prims.convert_element_type %5, %int5 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[],f16>
    return %6 : !torch.vtensor<[],f16>
  }
  func.func @test_reduction_masked_argmax_cpu_float16_8(%arg0: !torch.vtensor<[2],f16>) -> !torch.vtensor<[1],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %int0 = torch.constant.int 0
    %true = torch.constant.bool true
    %0 = torch.aten.argmax %arg0, %int0, %true : !torch.vtensor<[2],f16>, !torch.int, !torch.bool -> !torch.vtensor<[1],si64>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int4 = torch.constant.int 4
    %cpu = torch.constant.device "cpu"
    %int0_1 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %0, %none, %none_0, %int4, %cpu, %int0_1 : !torch.vtensor<[1],si64>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int5 = torch.constant.int 5
    %1 = torch.prims.convert_element_type %0, %int5 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1],f16>
    return %1 : !torch.vtensor<[1],f16>
  }
  func.func @test_reduction_masked_argmax_cpu_float16_9(%arg0: !torch.vtensor<[2],f16>) -> !torch.vtensor<[1],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_2_torch.uint8_1> : tensor<2xui8>) : !torch.vtensor<[2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[2],ui8>, !torch.int -> !torch.vtensor<[2],i1>
    %2 = torch.vtensor.literal(dense<0xFC00> : tensor<f16>) : !torch.vtensor<[],f16>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],f16>, !torch.none -> !torch.vtensor<[],f16>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[2],i1>, !torch.vtensor<[2],f16>, !torch.vtensor<[],f16> -> !torch.vtensor<[2],f16>
    %int0_3 = torch.constant.int 0
    %true = torch.constant.bool true
    %5 = torch.aten.argmax %4, %int0_3, %true : !torch.vtensor<[2],f16>, !torch.int, !torch.bool -> !torch.vtensor<[1],si64>
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int4 = torch.constant.int 4
    %cpu_6 = torch.constant.device "cpu"
    %int0_7 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_4, %none_5, %int4, %cpu_6, %int0_7 : !torch.vtensor<[1],si64>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int5 = torch.constant.int 5
    %6 = torch.prims.convert_element_type %5, %int5 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1],f16>
    return %6 : !torch.vtensor<[1],f16>
  }
  func.func @test_reduction_masked_argmax_cpu_float16_10(%arg0: !torch.vtensor<[2],f16>) -> !torch.vtensor<[],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %int-1 = torch.constant.int -1
    %false = torch.constant.bool false
    %0 = torch.aten.argmax %arg0, %int-1, %false : !torch.vtensor<[2],f16>, !torch.int, !torch.bool -> !torch.vtensor<[],si64>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int4 = torch.constant.int 4
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %0, %none, %none_0, %int4, %cpu, %int0 : !torch.vtensor<[],si64>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int5 = torch.constant.int 5
    %1 = torch.prims.convert_element_type %0, %int5 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[],f16>
    return %1 : !torch.vtensor<[],f16>
  }
  func.func @test_reduction_masked_argmax_cpu_float16_11(%arg0: !torch.vtensor<[2],f16>) -> !torch.vtensor<[],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_2_torch.uint8_2> : tensor<2xui8>) : !torch.vtensor<[2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[2],ui8>, !torch.int -> !torch.vtensor<[2],i1>
    %2 = torch.vtensor.literal(dense<0xFC00> : tensor<f16>) : !torch.vtensor<[],f16>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],f16>, !torch.none -> !torch.vtensor<[],f16>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[2],i1>, !torch.vtensor<[2],f16>, !torch.vtensor<[],f16> -> !torch.vtensor<[2],f16>
    %int-1 = torch.constant.int -1
    %false = torch.constant.bool false
    %5 = torch.aten.argmax %4, %int-1, %false : !torch.vtensor<[2],f16>, !torch.int, !torch.bool -> !torch.vtensor<[],si64>
    %none_3 = torch.constant.none
    %none_4 = torch.constant.none
    %int4 = torch.constant.int 4
    %cpu_5 = torch.constant.device "cpu"
    %int0_6 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_3, %none_4, %int4, %cpu_5, %int0_6 : !torch.vtensor<[],si64>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int5 = torch.constant.int 5
    %6 = torch.prims.convert_element_type %5, %int5 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[],f16>
    return %6 : !torch.vtensor<[],f16>
  }
  func.func @test_reduction_masked_argmax_cpu_float16_12(%arg0: !torch.vtensor<[3,5],f16>) -> !torch.vtensor<[],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %none = torch.constant.none
    %false = torch.constant.bool false
    %0 = torch.aten.argmax %arg0, %none, %false : !torch.vtensor<[3,5],f16>, !torch.none, !torch.bool -> !torch.vtensor<[],si64>
    %none_0 = torch.constant.none
    %none_1 = torch.constant.none
    %int4 = torch.constant.int 4
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %0, %none_0, %none_1, %int4, %cpu, %int0 : !torch.vtensor<[],si64>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int5 = torch.constant.int 5
    %1 = torch.prims.convert_element_type %0, %int5 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[],f16>
    return %1 : !torch.vtensor<[],f16>
  }
  func.func @test_reduction_masked_argmax_cpu_float16_13(%arg0: !torch.vtensor<[3,5],f16>) -> !torch.vtensor<[],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_5_torch.uint8> : tensor<3x5xui8>) : !torch.vtensor<[3,5],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,5],ui8>, !torch.int -> !torch.vtensor<[3,5],i1>
    %2 = torch.vtensor.literal(dense<0xFC00> : tensor<f16>) : !torch.vtensor<[],f16>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[3,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],f16>, !torch.none -> !torch.vtensor<[],f16>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[3,5],i1>, !torch.vtensor<[3,5],f16>, !torch.vtensor<[],f16> -> !torch.vtensor<[3,5],f16>
    %none_3 = torch.constant.none
    %false = torch.constant.bool false
    %5 = torch.aten.argmax %4, %none_3, %false : !torch.vtensor<[3,5],f16>, !torch.none, !torch.bool -> !torch.vtensor<[],si64>
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int4 = torch.constant.int 4
    %cpu_6 = torch.constant.device "cpu"
    %int0_7 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_4, %none_5, %int4, %cpu_6, %int0_7 : !torch.vtensor<[],si64>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int5 = torch.constant.int 5
    %6 = torch.prims.convert_element_type %5, %int5 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[],f16>
    return %6 : !torch.vtensor<[],f16>
  }
  func.func @test_reduction_masked_argmax_cpu_float16_14(%arg0: !torch.vtensor<[3,5],f16>) -> !torch.vtensor<[],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_5_torch.uint8_1> : tensor<3x5xui8>) : !torch.vtensor<[3,5],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,5],ui8>, !torch.int -> !torch.vtensor<[3,5],i1>
    %2 = torch.vtensor.literal(dense<0xFC00> : tensor<f16>) : !torch.vtensor<[],f16>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[3,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],f16>, !torch.none -> !torch.vtensor<[],f16>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[3,5],i1>, !torch.vtensor<[3,5],f16>, !torch.vtensor<[],f16> -> !torch.vtensor<[3,5],f16>
    %none_3 = torch.constant.none
    %false = torch.constant.bool false
    %5 = torch.aten.argmax %4, %none_3, %false : !torch.vtensor<[3,5],f16>, !torch.none, !torch.bool -> !torch.vtensor<[],si64>
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int4 = torch.constant.int 4
    %cpu_6 = torch.constant.device "cpu"
    %int0_7 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_4, %none_5, %int4, %cpu_6, %int0_7 : !torch.vtensor<[],si64>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int5 = torch.constant.int 5
    %6 = torch.prims.convert_element_type %5, %int5 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[],f16>
    return %6 : !torch.vtensor<[],f16>
  }
  func.func @test_reduction_masked_argmax_cpu_float16_15(%arg0: !torch.vtensor<[3,5],f16>) -> !torch.vtensor<[],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_5_torch.uint8_2> : tensor<3x5xui8>) : !torch.vtensor<[3,5],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,5],ui8>, !torch.int -> !torch.vtensor<[3,5],i1>
    %2 = torch.vtensor.literal(dense<0xFC00> : tensor<f16>) : !torch.vtensor<[],f16>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[3,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],f16>, !torch.none -> !torch.vtensor<[],f16>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[3,5],i1>, !torch.vtensor<[3,5],f16>, !torch.vtensor<[],f16> -> !torch.vtensor<[3,5],f16>
    %none_3 = torch.constant.none
    %false = torch.constant.bool false
    %5 = torch.aten.argmax %4, %none_3, %false : !torch.vtensor<[3,5],f16>, !torch.none, !torch.bool -> !torch.vtensor<[],si64>
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int4 = torch.constant.int 4
    %cpu_6 = torch.constant.device "cpu"
    %int0_7 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_4, %none_5, %int4, %cpu_6, %int0_7 : !torch.vtensor<[],si64>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int5 = torch.constant.int 5
    %6 = torch.prims.convert_element_type %5, %int5 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[],f16>
    return %6 : !torch.vtensor<[],f16>
  }
  func.func @test_reduction_masked_argmax_cpu_float16_16(%arg0: !torch.vtensor<[3,5],f16>) -> !torch.vtensor<[],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_5_torch.uint8_3> : tensor<3x5xui8>) : !torch.vtensor<[3,5],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,5],ui8>, !torch.int -> !torch.vtensor<[3,5],i1>
    %2 = torch.vtensor.literal(dense<0xFC00> : tensor<f16>) : !torch.vtensor<[],f16>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[3,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],f16>, !torch.none -> !torch.vtensor<[],f16>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[3,5],i1>, !torch.vtensor<[3,5],f16>, !torch.vtensor<[],f16> -> !torch.vtensor<[3,5],f16>
    %none_3 = torch.constant.none
    %false = torch.constant.bool false
    %5 = torch.aten.argmax %4, %none_3, %false : !torch.vtensor<[3,5],f16>, !torch.none, !torch.bool -> !torch.vtensor<[],si64>
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int4 = torch.constant.int 4
    %cpu_6 = torch.constant.device "cpu"
    %int0_7 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_4, %none_5, %int4, %cpu_6, %int0_7 : !torch.vtensor<[],si64>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int5 = torch.constant.int 5
    %6 = torch.prims.convert_element_type %5, %int5 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[],f16>
    return %6 : !torch.vtensor<[],f16>
  }
  func.func @test_reduction_masked_argmax_cpu_float16_17(%arg0: !torch.vtensor<[3,5],f16>) -> !torch.vtensor<[1,5],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %int0 = torch.constant.int 0
    %true = torch.constant.bool true
    %0 = torch.aten.argmax %arg0, %int0, %true : !torch.vtensor<[3,5],f16>, !torch.int, !torch.bool -> !torch.vtensor<[1,5],si64>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int4 = torch.constant.int 4
    %cpu = torch.constant.device "cpu"
    %int0_1 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %0, %none, %none_0, %int4, %cpu, %int0_1 : !torch.vtensor<[1,5],si64>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int5 = torch.constant.int 5
    %1 = torch.prims.convert_element_type %0, %int5 : !torch.vtensor<[1,5],si64>, !torch.int -> !torch.vtensor<[1,5],f16>
    return %1 : !torch.vtensor<[1,5],f16>
  }
  func.func @test_reduction_masked_argmax_cpu_float16_18(%arg0: !torch.vtensor<[3,5],f16>) -> !torch.vtensor<[1,5],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_5_torch.uint8_4> : tensor<3x5xui8>) : !torch.vtensor<[3,5],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,5],ui8>, !torch.int -> !torch.vtensor<[3,5],i1>
    %2 = torch.vtensor.literal(dense<0xFC00> : tensor<f16>) : !torch.vtensor<[],f16>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[3,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],f16>, !torch.none -> !torch.vtensor<[],f16>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[3,5],i1>, !torch.vtensor<[3,5],f16>, !torch.vtensor<[],f16> -> !torch.vtensor<[3,5],f16>
    %int0_3 = torch.constant.int 0
    %true = torch.constant.bool true
    %5 = torch.aten.argmax %4, %int0_3, %true : !torch.vtensor<[3,5],f16>, !torch.int, !torch.bool -> !torch.vtensor<[1,5],si64>
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int4 = torch.constant.int 4
    %cpu_6 = torch.constant.device "cpu"
    %int0_7 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_4, %none_5, %int4, %cpu_6, %int0_7 : !torch.vtensor<[1,5],si64>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int5 = torch.constant.int 5
    %6 = torch.prims.convert_element_type %5, %int5 : !torch.vtensor<[1,5],si64>, !torch.int -> !torch.vtensor<[1,5],f16>
    return %6 : !torch.vtensor<[1,5],f16>
  }
  func.func @test_reduction_masked_argmax_cpu_float16_19(%arg0: !torch.vtensor<[3,5],f16>) -> !torch.vtensor<[1,5],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_5_torch.uint8_5> : tensor<3x5xui8>) : !torch.vtensor<[3,5],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,5],ui8>, !torch.int -> !torch.vtensor<[3,5],i1>
    %2 = torch.vtensor.literal(dense<0xFC00> : tensor<f16>) : !torch.vtensor<[],f16>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[3,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],f16>, !torch.none -> !torch.vtensor<[],f16>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[3,5],i1>, !torch.vtensor<[3,5],f16>, !torch.vtensor<[],f16> -> !torch.vtensor<[3,5],f16>
    %int0_3 = torch.constant.int 0
    %true = torch.constant.bool true
    %5 = torch.aten.argmax %4, %int0_3, %true : !torch.vtensor<[3,5],f16>, !torch.int, !torch.bool -> !torch.vtensor<[1,5],si64>
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int4 = torch.constant.int 4
    %cpu_6 = torch.constant.device "cpu"
    %int0_7 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_4, %none_5, %int4, %cpu_6, %int0_7 : !torch.vtensor<[1,5],si64>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int5 = torch.constant.int 5
    %6 = torch.prims.convert_element_type %5, %int5 : !torch.vtensor<[1,5],si64>, !torch.int -> !torch.vtensor<[1,5],f16>
    return %6 : !torch.vtensor<[1,5],f16>
  }
  func.func @test_reduction_masked_argmax_cpu_float16_20(%arg0: !torch.vtensor<[3,5],f16>) -> !torch.vtensor<[1,5],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_5_torch.uint8_6> : tensor<3x5xui8>) : !torch.vtensor<[3,5],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,5],ui8>, !torch.int -> !torch.vtensor<[3,5],i1>
    %2 = torch.vtensor.literal(dense<0xFC00> : tensor<f16>) : !torch.vtensor<[],f16>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[3,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],f16>, !torch.none -> !torch.vtensor<[],f16>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[3,5],i1>, !torch.vtensor<[3,5],f16>, !torch.vtensor<[],f16> -> !torch.vtensor<[3,5],f16>
    %int0_3 = torch.constant.int 0
    %true = torch.constant.bool true
    %5 = torch.aten.argmax %4, %int0_3, %true : !torch.vtensor<[3,5],f16>, !torch.int, !torch.bool -> !torch.vtensor<[1,5],si64>
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int4 = torch.constant.int 4
    %cpu_6 = torch.constant.device "cpu"
    %int0_7 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_4, %none_5, %int4, %cpu_6, %int0_7 : !torch.vtensor<[1,5],si64>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int5 = torch.constant.int 5
    %6 = torch.prims.convert_element_type %5, %int5 : !torch.vtensor<[1,5],si64>, !torch.int -> !torch.vtensor<[1,5],f16>
    return %6 : !torch.vtensor<[1,5],f16>
  }
  func.func @test_reduction_masked_argmax_cpu_float16_21(%arg0: !torch.vtensor<[3,5],f16>) -> !torch.vtensor<[1,5],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_5_torch.uint8_7> : tensor<3x5xui8>) : !torch.vtensor<[3,5],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,5],ui8>, !torch.int -> !torch.vtensor<[3,5],i1>
    %2 = torch.vtensor.literal(dense<0xFC00> : tensor<f16>) : !torch.vtensor<[],f16>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[3,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],f16>, !torch.none -> !torch.vtensor<[],f16>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[3,5],i1>, !torch.vtensor<[3,5],f16>, !torch.vtensor<[],f16> -> !torch.vtensor<[3,5],f16>
    %int0_3 = torch.constant.int 0
    %true = torch.constant.bool true
    %5 = torch.aten.argmax %4, %int0_3, %true : !torch.vtensor<[3,5],f16>, !torch.int, !torch.bool -> !torch.vtensor<[1,5],si64>
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int4 = torch.constant.int 4
    %cpu_6 = torch.constant.device "cpu"
    %int0_7 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_4, %none_5, %int4, %cpu_6, %int0_7 : !torch.vtensor<[1,5],si64>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int5 = torch.constant.int 5
    %6 = torch.prims.convert_element_type %5, %int5 : !torch.vtensor<[1,5],si64>, !torch.int -> !torch.vtensor<[1,5],f16>
    return %6 : !torch.vtensor<[1,5],f16>
  }
  func.func @test_reduction_masked_argmax_cpu_float16_22(%arg0: !torch.vtensor<[3,5],f16>) -> !torch.vtensor<[3],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %int-1 = torch.constant.int -1
    %false = torch.constant.bool false
    %0 = torch.aten.argmax %arg0, %int-1, %false : !torch.vtensor<[3,5],f16>, !torch.int, !torch.bool -> !torch.vtensor<[3],si64>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int4 = torch.constant.int 4
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %0, %none, %none_0, %int4, %cpu, %int0 : !torch.vtensor<[3],si64>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int5 = torch.constant.int 5
    %1 = torch.prims.convert_element_type %0, %int5 : !torch.vtensor<[3],si64>, !torch.int -> !torch.vtensor<[3],f16>
    return %1 : !torch.vtensor<[3],f16>
  }
  func.func @test_reduction_masked_argmax_cpu_float16_23(%arg0: !torch.vtensor<[3,5],f16>) -> !torch.vtensor<[3],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_5_torch.uint8_8> : tensor<3x5xui8>) : !torch.vtensor<[3,5],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,5],ui8>, !torch.int -> !torch.vtensor<[3,5],i1>
    %2 = torch.vtensor.literal(dense<0xFC00> : tensor<f16>) : !torch.vtensor<[],f16>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[3,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],f16>, !torch.none -> !torch.vtensor<[],f16>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[3,5],i1>, !torch.vtensor<[3,5],f16>, !torch.vtensor<[],f16> -> !torch.vtensor<[3,5],f16>
    %int-1 = torch.constant.int -1
    %false = torch.constant.bool false
    %5 = torch.aten.argmax %4, %int-1, %false : !torch.vtensor<[3,5],f16>, !torch.int, !torch.bool -> !torch.vtensor<[3],si64>
    %none_3 = torch.constant.none
    %none_4 = torch.constant.none
    %int4 = torch.constant.int 4
    %cpu_5 = torch.constant.device "cpu"
    %int0_6 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_3, %none_4, %int4, %cpu_5, %int0_6 : !torch.vtensor<[3],si64>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int5 = torch.constant.int 5
    %6 = torch.prims.convert_element_type %5, %int5 : !torch.vtensor<[3],si64>, !torch.int -> !torch.vtensor<[3],f16>
    return %6 : !torch.vtensor<[3],f16>
  }
  func.func @test_reduction_masked_argmax_cpu_float16_24(%arg0: !torch.vtensor<[3,5],f16>) -> !torch.vtensor<[3],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_5_torch.uint8_9> : tensor<3x5xui8>) : !torch.vtensor<[3,5],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,5],ui8>, !torch.int -> !torch.vtensor<[3,5],i1>
    %2 = torch.vtensor.literal(dense<0xFC00> : tensor<f16>) : !torch.vtensor<[],f16>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[3,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],f16>, !torch.none -> !torch.vtensor<[],f16>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[3,5],i1>, !torch.vtensor<[3,5],f16>, !torch.vtensor<[],f16> -> !torch.vtensor<[3,5],f16>
    %int-1 = torch.constant.int -1
    %false = torch.constant.bool false
    %5 = torch.aten.argmax %4, %int-1, %false : !torch.vtensor<[3,5],f16>, !torch.int, !torch.bool -> !torch.vtensor<[3],si64>
    %none_3 = torch.constant.none
    %none_4 = torch.constant.none
    %int4 = torch.constant.int 4
    %cpu_5 = torch.constant.device "cpu"
    %int0_6 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_3, %none_4, %int4, %cpu_5, %int0_6 : !torch.vtensor<[3],si64>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int5 = torch.constant.int 5
    %6 = torch.prims.convert_element_type %5, %int5 : !torch.vtensor<[3],si64>, !torch.int -> !torch.vtensor<[3],f16>
    return %6 : !torch.vtensor<[3],f16>
  }
  func.func @test_reduction_masked_argmax_cpu_float16_25(%arg0: !torch.vtensor<[3,5],f16>) -> !torch.vtensor<[3],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_5_torch.uint8_10> : tensor<3x5xui8>) : !torch.vtensor<[3,5],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,5],ui8>, !torch.int -> !torch.vtensor<[3,5],i1>
    %2 = torch.vtensor.literal(dense<0xFC00> : tensor<f16>) : !torch.vtensor<[],f16>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[3,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],f16>, !torch.none -> !torch.vtensor<[],f16>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[3,5],i1>, !torch.vtensor<[3,5],f16>, !torch.vtensor<[],f16> -> !torch.vtensor<[3,5],f16>
    %int-1 = torch.constant.int -1
    %false = torch.constant.bool false
    %5 = torch.aten.argmax %4, %int-1, %false : !torch.vtensor<[3,5],f16>, !torch.int, !torch.bool -> !torch.vtensor<[3],si64>
    %none_3 = torch.constant.none
    %none_4 = torch.constant.none
    %int4 = torch.constant.int 4
    %cpu_5 = torch.constant.device "cpu"
    %int0_6 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_3, %none_4, %int4, %cpu_5, %int0_6 : !torch.vtensor<[3],si64>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int5 = torch.constant.int 5
    %6 = torch.prims.convert_element_type %5, %int5 : !torch.vtensor<[3],si64>, !torch.int -> !torch.vtensor<[3],f16>
    return %6 : !torch.vtensor<[3],f16>
  }
  func.func @test_reduction_masked_argmax_cpu_float16_26(%arg0: !torch.vtensor<[3,5],f16>) -> !torch.vtensor<[3],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_5_torch.uint8_11> : tensor<3x5xui8>) : !torch.vtensor<[3,5],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,5],ui8>, !torch.int -> !torch.vtensor<[3,5],i1>
    %2 = torch.vtensor.literal(dense<0xFC00> : tensor<f16>) : !torch.vtensor<[],f16>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[3,5],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],f16>, !torch.none -> !torch.vtensor<[],f16>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[3,5],i1>, !torch.vtensor<[3,5],f16>, !torch.vtensor<[],f16> -> !torch.vtensor<[3,5],f16>
    %int-1 = torch.constant.int -1
    %false = torch.constant.bool false
    %5 = torch.aten.argmax %4, %int-1, %false : !torch.vtensor<[3,5],f16>, !torch.int, !torch.bool -> !torch.vtensor<[3],si64>
    %none_3 = torch.constant.none
    %none_4 = torch.constant.none
    %int4 = torch.constant.int 4
    %cpu_5 = torch.constant.device "cpu"
    %int0_6 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_3, %none_4, %int4, %cpu_5, %int0_6 : !torch.vtensor<[3],si64>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int5 = torch.constant.int 5
    %6 = torch.prims.convert_element_type %5, %int5 : !torch.vtensor<[3],si64>, !torch.int -> !torch.vtensor<[3],f16>
    return %6 : !torch.vtensor<[3],f16>
  }
  func.func @test_reduction_masked_argmax_cpu_float16_27(%arg0: !torch.vtensor<[3,2,1,2],f16>) -> !torch.vtensor<[],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %none = torch.constant.none
    %false = torch.constant.bool false
    %0 = torch.aten.argmax %arg0, %none, %false : !torch.vtensor<[3,2,1,2],f16>, !torch.none, !torch.bool -> !torch.vtensor<[],si64>
    %none_0 = torch.constant.none
    %none_1 = torch.constant.none
    %int4 = torch.constant.int 4
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %0, %none_0, %none_1, %int4, %cpu, %int0 : !torch.vtensor<[],si64>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int5 = torch.constant.int 5
    %1 = torch.prims.convert_element_type %0, %int5 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[],f16>
    return %1 : !torch.vtensor<[],f16>
  }
  func.func @test_reduction_masked_argmax_cpu_float16_28(%arg0: !torch.vtensor<[3,2,1,2],f16>) -> !torch.vtensor<[],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_2_1_2_torch.uint8> : tensor<3x2x1x2xui8>) : !torch.vtensor<[3,2,1,2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,2,1,2],ui8>, !torch.int -> !torch.vtensor<[3,2,1,2],i1>
    %2 = torch.vtensor.literal(dense<0xFC00> : tensor<f16>) : !torch.vtensor<[],f16>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],f16>, !torch.none -> !torch.vtensor<[],f16>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],f16>, !torch.vtensor<[],f16> -> !torch.vtensor<[3,2,1,2],f16>
    %none_3 = torch.constant.none
    %false = torch.constant.bool false
    %5 = torch.aten.argmax %4, %none_3, %false : !torch.vtensor<[3,2,1,2],f16>, !torch.none, !torch.bool -> !torch.vtensor<[],si64>
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int4 = torch.constant.int 4
    %cpu_6 = torch.constant.device "cpu"
    %int0_7 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_4, %none_5, %int4, %cpu_6, %int0_7 : !torch.vtensor<[],si64>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int5 = torch.constant.int 5
    %6 = torch.prims.convert_element_type %5, %int5 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[],f16>
    return %6 : !torch.vtensor<[],f16>
  }
  func.func @test_reduction_masked_argmax_cpu_float16_29(%arg0: !torch.vtensor<[3,2,1,2],f16>) -> !torch.vtensor<[],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_2_1_1_torch.uint8> : tensor<3x2x1x1xui8>) : !torch.vtensor<[3,2,1,1],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,2,1,1],ui8>, !torch.int -> !torch.vtensor<[3,2,1,1],i1>
    %2 = torch.vtensor.literal(dense<0xFC00> : tensor<f16>) : !torch.vtensor<[],f16>
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
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],f16>, !torch.none -> !torch.vtensor<[],f16>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],f16>, !torch.vtensor<[],f16> -> !torch.vtensor<[3,2,1,2],f16>
    %none_10 = torch.constant.none
    %false_11 = torch.constant.bool false
    %8 = torch.aten.argmax %7, %none_10, %false_11 : !torch.vtensor<[3,2,1,2],f16>, !torch.none, !torch.bool -> !torch.vtensor<[],si64>
    %none_12 = torch.constant.none
    %none_13 = torch.constant.none
    %int4 = torch.constant.int 4
    %cpu_14 = torch.constant.device "cpu"
    %int0_15 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %8, %none_12, %none_13, %int4, %cpu_14, %int0_15 : !torch.vtensor<[],si64>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int5 = torch.constant.int 5
    %9 = torch.prims.convert_element_type %8, %int5 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[],f16>
    return %9 : !torch.vtensor<[],f16>
  }
  func.func @test_reduction_masked_argmax_cpu_float16_30(%arg0: !torch.vtensor<[3,2,1,2],f16>) -> !torch.vtensor<[],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_1_1_2_torch.uint8> : tensor<3x1x1x2xui8>) : !torch.vtensor<[3,1,1,2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,1,1,2],ui8>, !torch.int -> !torch.vtensor<[3,1,1,2],i1>
    %2 = torch.vtensor.literal(dense<0xFC00> : tensor<f16>) : !torch.vtensor<[],f16>
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
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],f16>, !torch.none -> !torch.vtensor<[],f16>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],f16>, !torch.vtensor<[],f16> -> !torch.vtensor<[3,2,1,2],f16>
    %none_10 = torch.constant.none
    %false_11 = torch.constant.bool false
    %8 = torch.aten.argmax %7, %none_10, %false_11 : !torch.vtensor<[3,2,1,2],f16>, !torch.none, !torch.bool -> !torch.vtensor<[],si64>
    %none_12 = torch.constant.none
    %none_13 = torch.constant.none
    %int4 = torch.constant.int 4
    %cpu_14 = torch.constant.device "cpu"
    %int0_15 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %8, %none_12, %none_13, %int4, %cpu_14, %int0_15 : !torch.vtensor<[],si64>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int5 = torch.constant.int 5
    %9 = torch.prims.convert_element_type %8, %int5 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[],f16>
    return %9 : !torch.vtensor<[],f16>
  }
  func.func @test_reduction_masked_argmax_cpu_float16_31(%arg0: !torch.vtensor<[3,2,1,2],f16>) -> !torch.vtensor<[],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_1_2_1_2_torch.uint8> : tensor<1x2x1x2xui8>) : !torch.vtensor<[1,2,1,2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[1,2,1,2],ui8>, !torch.int -> !torch.vtensor<[1,2,1,2],i1>
    %2 = torch.vtensor.literal(dense<0xFC00> : tensor<f16>) : !torch.vtensor<[],f16>
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
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],f16>, !torch.none -> !torch.vtensor<[],f16>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],f16>, !torch.vtensor<[],f16> -> !torch.vtensor<[3,2,1,2],f16>
    %none_10 = torch.constant.none
    %false_11 = torch.constant.bool false
    %8 = torch.aten.argmax %7, %none_10, %false_11 : !torch.vtensor<[3,2,1,2],f16>, !torch.none, !torch.bool -> !torch.vtensor<[],si64>
    %none_12 = torch.constant.none
    %none_13 = torch.constant.none
    %int4 = torch.constant.int 4
    %cpu_14 = torch.constant.device "cpu"
    %int0_15 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %8, %none_12, %none_13, %int4, %cpu_14, %int0_15 : !torch.vtensor<[],si64>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int5 = torch.constant.int 5
    %9 = torch.prims.convert_element_type %8, %int5 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[],f16>
    return %9 : !torch.vtensor<[],f16>
  }
  func.func @test_reduction_masked_argmax_cpu_float16_32(%arg0: !torch.vtensor<[3,2,1,2],f16>) -> !torch.vtensor<[],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_2_1_2_torch.uint8> : tensor<2x1x2xui8>) : !torch.vtensor<[2,1,2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[2,1,2],ui8>, !torch.int -> !torch.vtensor<[2,1,2],i1>
    %2 = torch.vtensor.literal(dense<0xFC00> : tensor<f16>) : !torch.vtensor<[],f16>
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
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],f16>, !torch.none -> !torch.vtensor<[],f16>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],f16>, !torch.vtensor<[],f16> -> !torch.vtensor<[3,2,1,2],f16>
    %none_10 = torch.constant.none
    %false_11 = torch.constant.bool false
    %8 = torch.aten.argmax %7, %none_10, %false_11 : !torch.vtensor<[3,2,1,2],f16>, !torch.none, !torch.bool -> !torch.vtensor<[],si64>
    %none_12 = torch.constant.none
    %none_13 = torch.constant.none
    %int4 = torch.constant.int 4
    %cpu_14 = torch.constant.device "cpu"
    %int0_15 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %8, %none_12, %none_13, %int4, %cpu_14, %int0_15 : !torch.vtensor<[],si64>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int5 = torch.constant.int 5
    %9 = torch.prims.convert_element_type %8, %int5 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[],f16>
    return %9 : !torch.vtensor<[],f16>
  }
  func.func @test_reduction_masked_argmax_cpu_float16_33(%arg0: !torch.vtensor<[3,2,1,2],f16>) -> !torch.vtensor<[],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_2_torch.uint8_3> : tensor<2xui8>) : !torch.vtensor<[2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[2],ui8>, !torch.int -> !torch.vtensor<[2],i1>
    %2 = torch.vtensor.literal(dense<0xFC00> : tensor<f16>) : !torch.vtensor<[],f16>
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
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],f16>, !torch.none -> !torch.vtensor<[],f16>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],f16>, !torch.vtensor<[],f16> -> !torch.vtensor<[3,2,1,2],f16>
    %none_10 = torch.constant.none
    %false_11 = torch.constant.bool false
    %8 = torch.aten.argmax %7, %none_10, %false_11 : !torch.vtensor<[3,2,1,2],f16>, !torch.none, !torch.bool -> !torch.vtensor<[],si64>
    %none_12 = torch.constant.none
    %none_13 = torch.constant.none
    %int4 = torch.constant.int 4
    %cpu_14 = torch.constant.device "cpu"
    %int0_15 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %8, %none_12, %none_13, %int4, %cpu_14, %int0_15 : !torch.vtensor<[],si64>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int5 = torch.constant.int 5
    %9 = torch.prims.convert_element_type %8, %int5 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[],f16>
    return %9 : !torch.vtensor<[],f16>
  }
  func.func @test_reduction_masked_argmax_cpu_float16_34(%arg0: !torch.vtensor<[3,2,1,2],f16>) -> !torch.vtensor<[1,2,1,2],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %int0 = torch.constant.int 0
    %true = torch.constant.bool true
    %0 = torch.aten.argmax %arg0, %int0, %true : !torch.vtensor<[3,2,1,2],f16>, !torch.int, !torch.bool -> !torch.vtensor<[1,2,1,2],si64>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int4 = torch.constant.int 4
    %cpu = torch.constant.device "cpu"
    %int0_1 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %0, %none, %none_0, %int4, %cpu, %int0_1 : !torch.vtensor<[1,2,1,2],si64>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int5 = torch.constant.int 5
    %1 = torch.prims.convert_element_type %0, %int5 : !torch.vtensor<[1,2,1,2],si64>, !torch.int -> !torch.vtensor<[1,2,1,2],f16>
    return %1 : !torch.vtensor<[1,2,1,2],f16>
  }
  func.func @test_reduction_masked_argmax_cpu_float16_35(%arg0: !torch.vtensor<[3,2,1,2],f16>) -> !torch.vtensor<[1,2,1,2],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_2_1_2_torch.uint8_1> : tensor<3x2x1x2xui8>) : !torch.vtensor<[3,2,1,2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,2,1,2],ui8>, !torch.int -> !torch.vtensor<[3,2,1,2],i1>
    %2 = torch.vtensor.literal(dense<0xFC00> : tensor<f16>) : !torch.vtensor<[],f16>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],f16>, !torch.none -> !torch.vtensor<[],f16>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],f16>, !torch.vtensor<[],f16> -> !torch.vtensor<[3,2,1,2],f16>
    %int0_3 = torch.constant.int 0
    %true = torch.constant.bool true
    %5 = torch.aten.argmax %4, %int0_3, %true : !torch.vtensor<[3,2,1,2],f16>, !torch.int, !torch.bool -> !torch.vtensor<[1,2,1,2],si64>
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %int4 = torch.constant.int 4
    %cpu_6 = torch.constant.device "cpu"
    %int0_7 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_4, %none_5, %int4, %cpu_6, %int0_7 : !torch.vtensor<[1,2,1,2],si64>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int5 = torch.constant.int 5
    %6 = torch.prims.convert_element_type %5, %int5 : !torch.vtensor<[1,2,1,2],si64>, !torch.int -> !torch.vtensor<[1,2,1,2],f16>
    return %6 : !torch.vtensor<[1,2,1,2],f16>
  }
  func.func @test_reduction_masked_argmax_cpu_float16_36(%arg0: !torch.vtensor<[3,2,1,2],f16>) -> !torch.vtensor<[1,2,1,2],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_2_1_1_torch.uint8_1> : tensor<3x2x1x1xui8>) : !torch.vtensor<[3,2,1,1],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,2,1,1],ui8>, !torch.int -> !torch.vtensor<[3,2,1,1],i1>
    %2 = torch.vtensor.literal(dense<0xFC00> : tensor<f16>) : !torch.vtensor<[],f16>
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
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],f16>, !torch.none -> !torch.vtensor<[],f16>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],f16>, !torch.vtensor<[],f16> -> !torch.vtensor<[3,2,1,2],f16>
    %int0_10 = torch.constant.int 0
    %true = torch.constant.bool true
    %8 = torch.aten.argmax %7, %int0_10, %true : !torch.vtensor<[3,2,1,2],f16>, !torch.int, !torch.bool -> !torch.vtensor<[1,2,1,2],si64>
    %none_11 = torch.constant.none
    %none_12 = torch.constant.none
    %int4 = torch.constant.int 4
    %cpu_13 = torch.constant.device "cpu"
    %int0_14 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %8, %none_11, %none_12, %int4, %cpu_13, %int0_14 : !torch.vtensor<[1,2,1,2],si64>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int5 = torch.constant.int 5
    %9 = torch.prims.convert_element_type %8, %int5 : !torch.vtensor<[1,2,1,2],si64>, !torch.int -> !torch.vtensor<[1,2,1,2],f16>
    return %9 : !torch.vtensor<[1,2,1,2],f16>
  }
  func.func @test_reduction_masked_argmax_cpu_float16_37(%arg0: !torch.vtensor<[3,2,1,2],f16>) -> !torch.vtensor<[1,2,1,2],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_1_1_2_torch.uint8_1> : tensor<3x1x1x2xui8>) : !torch.vtensor<[3,1,1,2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,1,1,2],ui8>, !torch.int -> !torch.vtensor<[3,1,1,2],i1>
    %2 = torch.vtensor.literal(dense<0xFC00> : tensor<f16>) : !torch.vtensor<[],f16>
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
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],f16>, !torch.none -> !torch.vtensor<[],f16>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],f16>, !torch.vtensor<[],f16> -> !torch.vtensor<[3,2,1,2],f16>
    %int0_10 = torch.constant.int 0
    %true = torch.constant.bool true
    %8 = torch.aten.argmax %7, %int0_10, %true : !torch.vtensor<[3,2,1,2],f16>, !torch.int, !torch.bool -> !torch.vtensor<[1,2,1,2],si64>
    %none_11 = torch.constant.none
    %none_12 = torch.constant.none
    %int4 = torch.constant.int 4
    %cpu_13 = torch.constant.device "cpu"
    %int0_14 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %8, %none_11, %none_12, %int4, %cpu_13, %int0_14 : !torch.vtensor<[1,2,1,2],si64>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int5 = torch.constant.int 5
    %9 = torch.prims.convert_element_type %8, %int5 : !torch.vtensor<[1,2,1,2],si64>, !torch.int -> !torch.vtensor<[1,2,1,2],f16>
    return %9 : !torch.vtensor<[1,2,1,2],f16>
  }
  func.func @test_reduction_masked_argmax_cpu_float16_38(%arg0: !torch.vtensor<[3,2,1,2],f16>) -> !torch.vtensor<[1,2,1,2],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_1_2_1_2_torch.uint8_1> : tensor<1x2x1x2xui8>) : !torch.vtensor<[1,2,1,2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[1,2,1,2],ui8>, !torch.int -> !torch.vtensor<[1,2,1,2],i1>
    %2 = torch.vtensor.literal(dense<0xFC00> : tensor<f16>) : !torch.vtensor<[],f16>
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
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],f16>, !torch.none -> !torch.vtensor<[],f16>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],f16>, !torch.vtensor<[],f16> -> !torch.vtensor<[3,2,1,2],f16>
    %int0_10 = torch.constant.int 0
    %true = torch.constant.bool true
    %8 = torch.aten.argmax %7, %int0_10, %true : !torch.vtensor<[3,2,1,2],f16>, !torch.int, !torch.bool -> !torch.vtensor<[1,2,1,2],si64>
    %none_11 = torch.constant.none
    %none_12 = torch.constant.none
    %int4 = torch.constant.int 4
    %cpu_13 = torch.constant.device "cpu"
    %int0_14 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %8, %none_11, %none_12, %int4, %cpu_13, %int0_14 : !torch.vtensor<[1,2,1,2],si64>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int5 = torch.constant.int 5
    %9 = torch.prims.convert_element_type %8, %int5 : !torch.vtensor<[1,2,1,2],si64>, !torch.int -> !torch.vtensor<[1,2,1,2],f16>
    return %9 : !torch.vtensor<[1,2,1,2],f16>
  }
  func.func @test_reduction_masked_argmax_cpu_float16_39(%arg0: !torch.vtensor<[3,2,1,2],f16>) -> !torch.vtensor<[1,2,1,2],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_2_1_2_torch.uint8_1> : tensor<2x1x2xui8>) : !torch.vtensor<[2,1,2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[2,1,2],ui8>, !torch.int -> !torch.vtensor<[2,1,2],i1>
    %2 = torch.vtensor.literal(dense<0xFC00> : tensor<f16>) : !torch.vtensor<[],f16>
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
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],f16>, !torch.none -> !torch.vtensor<[],f16>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],f16>, !torch.vtensor<[],f16> -> !torch.vtensor<[3,2,1,2],f16>
    %int0_10 = torch.constant.int 0
    %true = torch.constant.bool true
    %8 = torch.aten.argmax %7, %int0_10, %true : !torch.vtensor<[3,2,1,2],f16>, !torch.int, !torch.bool -> !torch.vtensor<[1,2,1,2],si64>
    %none_11 = torch.constant.none
    %none_12 = torch.constant.none
    %int4 = torch.constant.int 4
    %cpu_13 = torch.constant.device "cpu"
    %int0_14 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %8, %none_11, %none_12, %int4, %cpu_13, %int0_14 : !torch.vtensor<[1,2,1,2],si64>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int5 = torch.constant.int 5
    %9 = torch.prims.convert_element_type %8, %int5 : !torch.vtensor<[1,2,1,2],si64>, !torch.int -> !torch.vtensor<[1,2,1,2],f16>
    return %9 : !torch.vtensor<[1,2,1,2],f16>
  }
  func.func @test_reduction_masked_argmax_cpu_float16_40(%arg0: !torch.vtensor<[3,2,1,2],f16>) -> !torch.vtensor<[1,2,1,2],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_2_torch.uint8_4> : tensor<2xui8>) : !torch.vtensor<[2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[2],ui8>, !torch.int -> !torch.vtensor<[2],i1>
    %2 = torch.vtensor.literal(dense<0xFC00> : tensor<f16>) : !torch.vtensor<[],f16>
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
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],f16>, !torch.none -> !torch.vtensor<[],f16>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],f16>, !torch.vtensor<[],f16> -> !torch.vtensor<[3,2,1,2],f16>
    %int0_10 = torch.constant.int 0
    %true = torch.constant.bool true
    %8 = torch.aten.argmax %7, %int0_10, %true : !torch.vtensor<[3,2,1,2],f16>, !torch.int, !torch.bool -> !torch.vtensor<[1,2,1,2],si64>
    %none_11 = torch.constant.none
    %none_12 = torch.constant.none
    %int4 = torch.constant.int 4
    %cpu_13 = torch.constant.device "cpu"
    %int0_14 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %8, %none_11, %none_12, %int4, %cpu_13, %int0_14 : !torch.vtensor<[1,2,1,2],si64>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int5 = torch.constant.int 5
    %9 = torch.prims.convert_element_type %8, %int5 : !torch.vtensor<[1,2,1,2],si64>, !torch.int -> !torch.vtensor<[1,2,1,2],f16>
    return %9 : !torch.vtensor<[1,2,1,2],f16>
  }
  func.func @test_reduction_masked_argmax_cpu_float16_41(%arg0: !torch.vtensor<[3,2,1,2],f16>) -> !torch.vtensor<[3,2,1],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %int-1 = torch.constant.int -1
    %false = torch.constant.bool false
    %0 = torch.aten.argmax %arg0, %int-1, %false : !torch.vtensor<[3,2,1,2],f16>, !torch.int, !torch.bool -> !torch.vtensor<[3,2,1],si64>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int4 = torch.constant.int 4
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %0, %none, %none_0, %int4, %cpu, %int0 : !torch.vtensor<[3,2,1],si64>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int5 = torch.constant.int 5
    %1 = torch.prims.convert_element_type %0, %int5 : !torch.vtensor<[3,2,1],si64>, !torch.int -> !torch.vtensor<[3,2,1],f16>
    return %1 : !torch.vtensor<[3,2,1],f16>
  }
  func.func @test_reduction_masked_argmax_cpu_float16_42(%arg0: !torch.vtensor<[3,2,1,2],f16>) -> !torch.vtensor<[3,2,1],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_2_1_2_torch.uint8_2> : tensor<3x2x1x2xui8>) : !torch.vtensor<[3,2,1,2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,2,1,2],ui8>, !torch.int -> !torch.vtensor<[3,2,1,2],i1>
    %2 = torch.vtensor.literal(dense<0xFC00> : tensor<f16>) : !torch.vtensor<[],f16>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],f16>, !torch.none -> !torch.vtensor<[],f16>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],f16>, !torch.vtensor<[],f16> -> !torch.vtensor<[3,2,1,2],f16>
    %int-1 = torch.constant.int -1
    %false = torch.constant.bool false
    %5 = torch.aten.argmax %4, %int-1, %false : !torch.vtensor<[3,2,1,2],f16>, !torch.int, !torch.bool -> !torch.vtensor<[3,2,1],si64>
    %none_3 = torch.constant.none
    %none_4 = torch.constant.none
    %int4 = torch.constant.int 4
    %cpu_5 = torch.constant.device "cpu"
    %int0_6 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_3, %none_4, %int4, %cpu_5, %int0_6 : !torch.vtensor<[3,2,1],si64>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int5 = torch.constant.int 5
    %6 = torch.prims.convert_element_type %5, %int5 : !torch.vtensor<[3,2,1],si64>, !torch.int -> !torch.vtensor<[3,2,1],f16>
    return %6 : !torch.vtensor<[3,2,1],f16>
  }
  func.func @test_reduction_masked_argmax_cpu_float16_43(%arg0: !torch.vtensor<[3,2,1,2],f16>) -> !torch.vtensor<[3,2,1],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_2_1_1_torch.uint8_2> : tensor<3x2x1x1xui8>) : !torch.vtensor<[3,2,1,1],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,2,1,1],ui8>, !torch.int -> !torch.vtensor<[3,2,1,1],i1>
    %2 = torch.vtensor.literal(dense<0xFC00> : tensor<f16>) : !torch.vtensor<[],f16>
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
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],f16>, !torch.none -> !torch.vtensor<[],f16>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],f16>, !torch.vtensor<[],f16> -> !torch.vtensor<[3,2,1,2],f16>
    %int-1 = torch.constant.int -1
    %false_10 = torch.constant.bool false
    %8 = torch.aten.argmax %7, %int-1, %false_10 : !torch.vtensor<[3,2,1,2],f16>, !torch.int, !torch.bool -> !torch.vtensor<[3,2,1],si64>
    %none_11 = torch.constant.none
    %none_12 = torch.constant.none
    %int4 = torch.constant.int 4
    %cpu_13 = torch.constant.device "cpu"
    %int0_14 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %8, %none_11, %none_12, %int4, %cpu_13, %int0_14 : !torch.vtensor<[3,2,1],si64>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int5 = torch.constant.int 5
    %9 = torch.prims.convert_element_type %8, %int5 : !torch.vtensor<[3,2,1],si64>, !torch.int -> !torch.vtensor<[3,2,1],f16>
    return %9 : !torch.vtensor<[3,2,1],f16>
  }
  func.func @test_reduction_masked_argmax_cpu_float16_44(%arg0: !torch.vtensor<[3,2,1,2],f16>) -> !torch.vtensor<[3,2,1],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_1_1_2_torch.uint8_2> : tensor<3x1x1x2xui8>) : !torch.vtensor<[3,1,1,2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,1,1,2],ui8>, !torch.int -> !torch.vtensor<[3,1,1,2],i1>
    %2 = torch.vtensor.literal(dense<0xFC00> : tensor<f16>) : !torch.vtensor<[],f16>
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
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],f16>, !torch.none -> !torch.vtensor<[],f16>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],f16>, !torch.vtensor<[],f16> -> !torch.vtensor<[3,2,1,2],f16>
    %int-1 = torch.constant.int -1
    %false_10 = torch.constant.bool false
    %8 = torch.aten.argmax %7, %int-1, %false_10 : !torch.vtensor<[3,2,1,2],f16>, !torch.int, !torch.bool -> !torch.vtensor<[3,2,1],si64>
    %none_11 = torch.constant.none
    %none_12 = torch.constant.none
    %int4 = torch.constant.int 4
    %cpu_13 = torch.constant.device "cpu"
    %int0_14 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %8, %none_11, %none_12, %int4, %cpu_13, %int0_14 : !torch.vtensor<[3,2,1],si64>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int5 = torch.constant.int 5
    %9 = torch.prims.convert_element_type %8, %int5 : !torch.vtensor<[3,2,1],si64>, !torch.int -> !torch.vtensor<[3,2,1],f16>
    return %9 : !torch.vtensor<[3,2,1],f16>
  }
  func.func @test_reduction_masked_argmax_cpu_float16_45(%arg0: !torch.vtensor<[3,2,1,2],f16>) -> !torch.vtensor<[3,2,1],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_1_2_1_2_torch.uint8_2> : tensor<1x2x1x2xui8>) : !torch.vtensor<[1,2,1,2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[1,2,1,2],ui8>, !torch.int -> !torch.vtensor<[1,2,1,2],i1>
    %2 = torch.vtensor.literal(dense<0xFC00> : tensor<f16>) : !torch.vtensor<[],f16>
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
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],f16>, !torch.none -> !torch.vtensor<[],f16>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],f16>, !torch.vtensor<[],f16> -> !torch.vtensor<[3,2,1,2],f16>
    %int-1 = torch.constant.int -1
    %false_10 = torch.constant.bool false
    %8 = torch.aten.argmax %7, %int-1, %false_10 : !torch.vtensor<[3,2,1,2],f16>, !torch.int, !torch.bool -> !torch.vtensor<[3,2,1],si64>
    %none_11 = torch.constant.none
    %none_12 = torch.constant.none
    %int4 = torch.constant.int 4
    %cpu_13 = torch.constant.device "cpu"
    %int0_14 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %8, %none_11, %none_12, %int4, %cpu_13, %int0_14 : !torch.vtensor<[3,2,1],si64>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int5 = torch.constant.int 5
    %9 = torch.prims.convert_element_type %8, %int5 : !torch.vtensor<[3,2,1],si64>, !torch.int -> !torch.vtensor<[3,2,1],f16>
    return %9 : !torch.vtensor<[3,2,1],f16>
  }
  func.func @test_reduction_masked_argmax_cpu_float16_46(%arg0: !torch.vtensor<[3,2,1,2],f16>) -> !torch.vtensor<[3,2,1],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_2_1_2_torch.uint8_2> : tensor<2x1x2xui8>) : !torch.vtensor<[2,1,2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[2,1,2],ui8>, !torch.int -> !torch.vtensor<[2,1,2],i1>
    %2 = torch.vtensor.literal(dense<0xFC00> : tensor<f16>) : !torch.vtensor<[],f16>
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
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],f16>, !torch.none -> !torch.vtensor<[],f16>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],f16>, !torch.vtensor<[],f16> -> !torch.vtensor<[3,2,1,2],f16>
    %int-1 = torch.constant.int -1
    %false_10 = torch.constant.bool false
    %8 = torch.aten.argmax %7, %int-1, %false_10 : !torch.vtensor<[3,2,1,2],f16>, !torch.int, !torch.bool -> !torch.vtensor<[3,2,1],si64>
    %none_11 = torch.constant.none
    %none_12 = torch.constant.none
    %int4 = torch.constant.int 4
    %cpu_13 = torch.constant.device "cpu"
    %int0_14 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %8, %none_11, %none_12, %int4, %cpu_13, %int0_14 : !torch.vtensor<[3,2,1],si64>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int5 = torch.constant.int 5
    %9 = torch.prims.convert_element_type %8, %int5 : !torch.vtensor<[3,2,1],si64>, !torch.int -> !torch.vtensor<[3,2,1],f16>
    return %9 : !torch.vtensor<[3,2,1],f16>
  }
  func.func @test_reduction_masked_argmax_cpu_float16_47(%arg0: !torch.vtensor<[3,2,1,2],f16>) -> !torch.vtensor<[3,2,1],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_2_torch.uint8_5> : tensor<2xui8>) : !torch.vtensor<[2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[2],ui8>, !torch.int -> !torch.vtensor<[2],i1>
    %2 = torch.vtensor.literal(dense<0xFC00> : tensor<f16>) : !torch.vtensor<[],f16>
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
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],f16>, !torch.none -> !torch.vtensor<[],f16>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],f16>, !torch.vtensor<[],f16> -> !torch.vtensor<[3,2,1,2],f16>
    %int-1 = torch.constant.int -1
    %false_10 = torch.constant.bool false
    %8 = torch.aten.argmax %7, %int-1, %false_10 : !torch.vtensor<[3,2,1,2],f16>, !torch.int, !torch.bool -> !torch.vtensor<[3,2,1],si64>
    %none_11 = torch.constant.none
    %none_12 = torch.constant.none
    %int4 = torch.constant.int 4
    %cpu_13 = torch.constant.device "cpu"
    %int0_14 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %8, %none_11, %none_12, %int4, %cpu_13, %int0_14 : !torch.vtensor<[3,2,1],si64>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int5 = torch.constant.int 5
    %9 = torch.prims.convert_element_type %8, %int5 : !torch.vtensor<[3,2,1],si64>, !torch.int -> !torch.vtensor<[3,2,1],f16>
    return %9 : !torch.vtensor<[3,2,1],f16>
  }
  func.func @test_reduction_masked_argmax_cpu_float16_48(%arg0: !torch.vtensor<[3,2,1,2],f16>) -> !torch.vtensor<[3,2,1,2],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %int2 = torch.constant.int 2
    %true = torch.constant.bool true
    %0 = torch.aten.argmax %arg0, %int2, %true : !torch.vtensor<[3,2,1,2],f16>, !torch.int, !torch.bool -> !torch.vtensor<[3,2,1,2],si64>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int4 = torch.constant.int 4
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %0, %none, %none_0, %int4, %cpu, %int0 : !torch.vtensor<[3,2,1,2],si64>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int5 = torch.constant.int 5
    %1 = torch.prims.convert_element_type %0, %int5 : !torch.vtensor<[3,2,1,2],si64>, !torch.int -> !torch.vtensor<[3,2,1,2],f16>
    return %1 : !torch.vtensor<[3,2,1,2],f16>
  }
  func.func @test_reduction_masked_argmax_cpu_float16_49(%arg0: !torch.vtensor<[3,2,1,2],f16>) -> !torch.vtensor<[3,2,1,2],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_2_1_2_torch.uint8_3> : tensor<3x2x1x2xui8>) : !torch.vtensor<[3,2,1,2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,2,1,2],ui8>, !torch.int -> !torch.vtensor<[3,2,1,2],i1>
    %2 = torch.vtensor.literal(dense<0xFC00> : tensor<f16>) : !torch.vtensor<[],f16>
    %none = torch.constant.none
    %none_0 = torch.constant.none
    %int11_1 = torch.constant.int 11
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %1, %none, %none_0, %int11_1, %cpu, %int0 : !torch.vtensor<[3,2,1,2],i1>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %none_2 = torch.constant.none
    %3 = torch.aten.clone %2, %none_2 : !torch.vtensor<[],f16>, !torch.none -> !torch.vtensor<[],f16>
    %4 = torch.aten.where.self %1, %arg0, %3 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],f16>, !torch.vtensor<[],f16> -> !torch.vtensor<[3,2,1,2],f16>
    %int2 = torch.constant.int 2
    %true = torch.constant.bool true
    %5 = torch.aten.argmax %4, %int2, %true : !torch.vtensor<[3,2,1,2],f16>, !torch.int, !torch.bool -> !torch.vtensor<[3,2,1,2],si64>
    %none_3 = torch.constant.none
    %none_4 = torch.constant.none
    %int4 = torch.constant.int 4
    %cpu_5 = torch.constant.device "cpu"
    %int0_6 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %5, %none_3, %none_4, %int4, %cpu_5, %int0_6 : !torch.vtensor<[3,2,1,2],si64>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int5 = torch.constant.int 5
    %6 = torch.prims.convert_element_type %5, %int5 : !torch.vtensor<[3,2,1,2],si64>, !torch.int -> !torch.vtensor<[3,2,1,2],f16>
    return %6 : !torch.vtensor<[3,2,1,2],f16>
  }
  func.func @test_reduction_masked_argmax_cpu_float16_50(%arg0: !torch.vtensor<[3,2,1,2],f16>) -> !torch.vtensor<[3,2,1,2],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_2_1_1_torch.uint8_3> : tensor<3x2x1x1xui8>) : !torch.vtensor<[3,2,1,1],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,2,1,1],ui8>, !torch.int -> !torch.vtensor<[3,2,1,1],i1>
    %2 = torch.vtensor.literal(dense<0xFC00> : tensor<f16>) : !torch.vtensor<[],f16>
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
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],f16>, !torch.none -> !torch.vtensor<[],f16>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],f16>, !torch.vtensor<[],f16> -> !torch.vtensor<[3,2,1,2],f16>
    %int2_10 = torch.constant.int 2
    %true = torch.constant.bool true
    %8 = torch.aten.argmax %7, %int2_10, %true : !torch.vtensor<[3,2,1,2],f16>, !torch.int, !torch.bool -> !torch.vtensor<[3,2,1,2],si64>
    %none_11 = torch.constant.none
    %none_12 = torch.constant.none
    %int4 = torch.constant.int 4
    %cpu_13 = torch.constant.device "cpu"
    %int0_14 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %8, %none_11, %none_12, %int4, %cpu_13, %int0_14 : !torch.vtensor<[3,2,1,2],si64>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int5 = torch.constant.int 5
    %9 = torch.prims.convert_element_type %8, %int5 : !torch.vtensor<[3,2,1,2],si64>, !torch.int -> !torch.vtensor<[3,2,1,2],f16>
    return %9 : !torch.vtensor<[3,2,1,2],f16>
  }
  func.func @test_reduction_masked_argmax_cpu_float16_51(%arg0: !torch.vtensor<[3,2,1,2],f16>) -> !torch.vtensor<[3,2,1,2],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_1_1_2_torch.uint8_3> : tensor<3x1x1x2xui8>) : !torch.vtensor<[3,1,1,2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[3,1,1,2],ui8>, !torch.int -> !torch.vtensor<[3,1,1,2],i1>
    %2 = torch.vtensor.literal(dense<0xFC00> : tensor<f16>) : !torch.vtensor<[],f16>
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
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],f16>, !torch.none -> !torch.vtensor<[],f16>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],f16>, !torch.vtensor<[],f16> -> !torch.vtensor<[3,2,1,2],f16>
    %int2_10 = torch.constant.int 2
    %true = torch.constant.bool true
    %8 = torch.aten.argmax %7, %int2_10, %true : !torch.vtensor<[3,2,1,2],f16>, !torch.int, !torch.bool -> !torch.vtensor<[3,2,1,2],si64>
    %none_11 = torch.constant.none
    %none_12 = torch.constant.none
    %int4 = torch.constant.int 4
    %cpu_13 = torch.constant.device "cpu"
    %int0_14 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %8, %none_11, %none_12, %int4, %cpu_13, %int0_14 : !torch.vtensor<[3,2,1,2],si64>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int5 = torch.constant.int 5
    %9 = torch.prims.convert_element_type %8, %int5 : !torch.vtensor<[3,2,1,2],si64>, !torch.int -> !torch.vtensor<[3,2,1,2],f16>
    return %9 : !torch.vtensor<[3,2,1,2],f16>
  }
  func.func @test_reduction_masked_argmax_cpu_float16_52(%arg0: !torch.vtensor<[3,2,1,2],f16>) -> !torch.vtensor<[3,2,1,2],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_1_2_1_2_torch.uint8_3> : tensor<1x2x1x2xui8>) : !torch.vtensor<[1,2,1,2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[1,2,1,2],ui8>, !torch.int -> !torch.vtensor<[1,2,1,2],i1>
    %2 = torch.vtensor.literal(dense<0xFC00> : tensor<f16>) : !torch.vtensor<[],f16>
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
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],f16>, !torch.none -> !torch.vtensor<[],f16>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],f16>, !torch.vtensor<[],f16> -> !torch.vtensor<[3,2,1,2],f16>
    %int2_10 = torch.constant.int 2
    %true = torch.constant.bool true
    %8 = torch.aten.argmax %7, %int2_10, %true : !torch.vtensor<[3,2,1,2],f16>, !torch.int, !torch.bool -> !torch.vtensor<[3,2,1,2],si64>
    %none_11 = torch.constant.none
    %none_12 = torch.constant.none
    %int4 = torch.constant.int 4
    %cpu_13 = torch.constant.device "cpu"
    %int0_14 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %8, %none_11, %none_12, %int4, %cpu_13, %int0_14 : !torch.vtensor<[3,2,1,2],si64>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int5 = torch.constant.int 5
    %9 = torch.prims.convert_element_type %8, %int5 : !torch.vtensor<[3,2,1,2],si64>, !torch.int -> !torch.vtensor<[3,2,1,2],f16>
    return %9 : !torch.vtensor<[3,2,1,2],f16>
  }
  func.func @test_reduction_masked_argmax_cpu_float16_53(%arg0: !torch.vtensor<[3,2,1,2],f16>) -> !torch.vtensor<[3,2,1,2],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_2_1_2_torch.uint8_3> : tensor<2x1x2xui8>) : !torch.vtensor<[2,1,2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[2,1,2],ui8>, !torch.int -> !torch.vtensor<[2,1,2],i1>
    %2 = torch.vtensor.literal(dense<0xFC00> : tensor<f16>) : !torch.vtensor<[],f16>
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
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],f16>, !torch.none -> !torch.vtensor<[],f16>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],f16>, !torch.vtensor<[],f16> -> !torch.vtensor<[3,2,1,2],f16>
    %int2_10 = torch.constant.int 2
    %true = torch.constant.bool true
    %8 = torch.aten.argmax %7, %int2_10, %true : !torch.vtensor<[3,2,1,2],f16>, !torch.int, !torch.bool -> !torch.vtensor<[3,2,1,2],si64>
    %none_11 = torch.constant.none
    %none_12 = torch.constant.none
    %int4 = torch.constant.int 4
    %cpu_13 = torch.constant.device "cpu"
    %int0_14 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %8, %none_11, %none_12, %int4, %cpu_13, %int0_14 : !torch.vtensor<[3,2,1,2],si64>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int5 = torch.constant.int 5
    %9 = torch.prims.convert_element_type %8, %int5 : !torch.vtensor<[3,2,1,2],si64>, !torch.int -> !torch.vtensor<[3,2,1,2],f16>
    return %9 : !torch.vtensor<[3,2,1,2],f16>
  }
  func.func @test_reduction_masked_argmax_cpu_float16_54(%arg0: !torch.vtensor<[3,2,1,2],f16>) -> !torch.vtensor<[3,2,1,2],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_2_torch.uint8_6> : tensor<2xui8>) : !torch.vtensor<[2],ui8>
    %int11 = torch.constant.int 11
    %1 = torch.prims.convert_element_type %0, %int11 : !torch.vtensor<[2],ui8>, !torch.int -> !torch.vtensor<[2],i1>
    %2 = torch.vtensor.literal(dense<0xFC00> : tensor<f16>) : !torch.vtensor<[],f16>
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
    %6 = torch.aten.clone %2, %none_9 : !torch.vtensor<[],f16>, !torch.none -> !torch.vtensor<[],f16>
    %7 = torch.aten.where.self %5, %arg0, %6 : !torch.vtensor<[3,2,1,2],i1>, !torch.vtensor<[3,2,1,2],f16>, !torch.vtensor<[],f16> -> !torch.vtensor<[3,2,1,2],f16>
    %int2_10 = torch.constant.int 2
    %true = torch.constant.bool true
    %8 = torch.aten.argmax %7, %int2_10, %true : !torch.vtensor<[3,2,1,2],f16>, !torch.int, !torch.bool -> !torch.vtensor<[3,2,1,2],si64>
    %none_11 = torch.constant.none
    %none_12 = torch.constant.none
    %int4 = torch.constant.int 4
    %cpu_13 = torch.constant.device "cpu"
    %int0_14 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %8, %none_11, %none_12, %int4, %cpu_13, %int0_14 : !torch.vtensor<[3,2,1,2],si64>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int5 = torch.constant.int 5
    %9 = torch.prims.convert_element_type %8, %int5 : !torch.vtensor<[3,2,1,2],si64>, !torch.int -> !torch.vtensor<[3,2,1,2],f16>
    return %9 : !torch.vtensor<[3,2,1,2],f16>
  }
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_2_torch.uint8: "0x010000000101",
      torch_tensor_2_torch.uint8_1: "0x010000000101",
      torch_tensor_2_torch.uint8_2: "0x010000000101",
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
      torch_tensor_3_2_1_2_torch.uint8: "0x01000000010100010000000101010101",
      torch_tensor_3_2_1_1_torch.uint8: "0x01000000010100010000",
      torch_tensor_3_1_1_2_torch.uint8: "0x01000000010100010000",
      torch_tensor_1_2_1_2_torch.uint8: "0x0100000001010001",
      torch_tensor_2_1_2_torch.uint8: "0x0100000001010001",
      torch_tensor_2_torch.uint8_3: "0x010000000101",
      torch_tensor_3_2_1_2_torch.uint8_1: "0x01000000010100010000000101010101",
      torch_tensor_3_2_1_1_torch.uint8_1: "0x01000000010100010000",
      torch_tensor_3_1_1_2_torch.uint8_1: "0x01000000010100010000",
      torch_tensor_1_2_1_2_torch.uint8_1: "0x0100000001010001",
      torch_tensor_2_1_2_torch.uint8_1: "0x0100000001010001",
      torch_tensor_2_torch.uint8_4: "0x010000000101",
      torch_tensor_3_2_1_2_torch.uint8_2: "0x01000000010100010000000101010101",
      torch_tensor_3_2_1_1_torch.uint8_2: "0x01000000010100010000",
      torch_tensor_3_1_1_2_torch.uint8_2: "0x01000000010100010000",
      torch_tensor_1_2_1_2_torch.uint8_2: "0x0100000001010001",
      torch_tensor_2_1_2_torch.uint8_2: "0x0100000001010001",
      torch_tensor_2_torch.uint8_5: "0x010000000101",
      torch_tensor_3_2_1_2_torch.uint8_3: "0x01000000010100010000000101010101",
      torch_tensor_3_2_1_1_torch.uint8_3: "0x01000000010100010000",
      torch_tensor_3_1_1_2_torch.uint8_3: "0x01000000010100010000",
      torch_tensor_1_2_1_2_torch.uint8_3: "0x0100000001010001",
      torch_tensor_2_1_2_torch.uint8_3: "0x0100000001010001",
      torch_tensor_2_torch.uint8_6: "0x010000000101"
    }
  }
#-}
