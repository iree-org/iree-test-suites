module @module {
  func.func @test_reduction_prod_cpu_float16_0(%arg0: !torch.vtensor<[5,5,5],f16>) -> !torch.vtensor<[],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %none = torch.constant.none
    %0 = torch.aten.prod %arg0, %none : !torch.vtensor<[5,5,5],f16>, !torch.none -> !torch.vtensor<[],f16>
    return %0 : !torch.vtensor<[],f16>
  }
  func.func @test_reduction_prod_cpu_float16_1(%arg0: !torch.vtensor<[5,5,5],f16>) -> !torch.vtensor<[5,5],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %int0 = torch.constant.int 0
    %false = torch.constant.bool false
    %none = torch.constant.none
    %0 = torch.aten.prod.dim_int %arg0, %int0, %false, %none : !torch.vtensor<[5,5,5],f16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[5,5],f16>
    return %0 : !torch.vtensor<[5,5],f16>
  }
  func.func @test_reduction_prod_cpu_float16_2(%arg0: !torch.vtensor<[5,5,5],f16>) -> !torch.vtensor<[],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %none = torch.constant.none
    %0 = torch.aten.prod %arg0, %none : !torch.vtensor<[5,5,5],f16>, !torch.none -> !torch.vtensor<[],f16>
    return %0 : !torch.vtensor<[],f16>
  }
  func.func @test_reduction_prod_cpu_float16_3(%arg0: !torch.vtensor<[5,5,5],f16>) -> !torch.vtensor<[5,5],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %int1 = torch.constant.int 1
    %false = torch.constant.bool false
    %none = torch.constant.none
    %0 = torch.aten.prod.dim_int %arg0, %int1, %false, %none : !torch.vtensor<[5,5,5],f16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[5,5],f16>
    return %0 : !torch.vtensor<[5,5],f16>
  }
  func.func @test_reduction_prod_cpu_float16_4(%arg0: !torch.vtensor<[5,5,5],f16>) -> !torch.vtensor<[],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %none = torch.constant.none
    %0 = torch.aten.prod %arg0, %none : !torch.vtensor<[5,5,5],f16>, !torch.none -> !torch.vtensor<[],f16>
    return %0 : !torch.vtensor<[],f16>
  }
  func.func @test_reduction_prod_cpu_float16_5(%arg0: !torch.vtensor<[5,5,5],f16>) -> !torch.vtensor<[5,5],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %int2 = torch.constant.int 2
    %false = torch.constant.bool false
    %none = torch.constant.none
    %0 = torch.aten.prod.dim_int %arg0, %int2, %false, %none : !torch.vtensor<[5,5,5],f16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[5,5],f16>
    return %0 : !torch.vtensor<[5,5],f16>
  }
  func.func @test_reduction_prod_cpu_float16_6(%arg0: !torch.vtensor<[],f16>) -> !torch.vtensor<[],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %none = torch.constant.none
    %0 = torch.aten.prod %arg0, %none : !torch.vtensor<[],f16>, !torch.none -> !torch.vtensor<[],f16>
    return %0 : !torch.vtensor<[],f16>
  }
  func.func @test_reduction_prod_cpu_float16_7(%arg0: !torch.vtensor<[],f16>) -> !torch.vtensor<[],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %int0 = torch.constant.int 0
    %false = torch.constant.bool false
    %none = torch.constant.none
    %0 = torch.aten.prod.dim_int %arg0, %int0, %false, %none : !torch.vtensor<[],f16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[],f16>
    return %0 : !torch.vtensor<[],f16>
  }
  func.func @test_reduction_prod_cpu_float16_8(%arg0: !torch.vtensor<[1],f16>) -> !torch.vtensor<[],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %none = torch.constant.none
    %0 = torch.aten.prod %arg0, %none : !torch.vtensor<[1],f16>, !torch.none -> !torch.vtensor<[],f16>
    return %0 : !torch.vtensor<[],f16>
  }
  func.func @test_reduction_prod_cpu_float16_9(%arg0: !torch.vtensor<[1],f16>) -> !torch.vtensor<[],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %int0 = torch.constant.int 0
    %false = torch.constant.bool false
    %none = torch.constant.none
    %0 = torch.aten.prod.dim_int %arg0, %int0, %false, %none : !torch.vtensor<[1],f16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[],f16>
    return %0 : !torch.vtensor<[],f16>
  }
  func.func @test_reduction_prod_cpu_float16_10(%arg0: !torch.vtensor<[0],f16>) -> !torch.vtensor<[],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %none = torch.constant.none
    %0 = torch.aten.prod %arg0, %none : !torch.vtensor<[0],f16>, !torch.none -> !torch.vtensor<[],f16>
    return %0 : !torch.vtensor<[],f16>
  }
  func.func @test_reduction_prod_cpu_float16_11(%arg0: !torch.vtensor<[0],f16>) -> !torch.vtensor<[],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %int0 = torch.constant.int 0
    %false = torch.constant.bool false
    %none = torch.constant.none
    %0 = torch.aten.prod.dim_int %arg0, %int0, %false, %none : !torch.vtensor<[0],f16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[],f16>
    return %0 : !torch.vtensor<[],f16>
  }
  func.func @test_reduction_prod_cpu_float16_12(%arg0: !torch.vtensor<[5,5,5],f16>) -> !torch.vtensor<[],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %none = torch.constant.none
    %0 = torch.aten.prod %arg0, %none : !torch.vtensor<[5,5,5],f16>, !torch.none -> !torch.vtensor<[],f16>
    return %0 : !torch.vtensor<[],f16>
  }
  func.func @test_reduction_prod_cpu_float16_13(%arg0: !torch.vtensor<[5,5,5],f16>) -> !torch.vtensor<[5,5],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %int1 = torch.constant.int 1
    %false = torch.constant.bool false
    %none = torch.constant.none
    %0 = torch.aten.prod.dim_int %arg0, %int1, %false, %none : !torch.vtensor<[5,5,5],f16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[5,5],f16>
    return %0 : !torch.vtensor<[5,5],f16>
  }
  func.func @test_reduction_prod_cpu_float16_14(%arg0: !torch.vtensor<[5,5,5],f16>) -> !torch.vtensor<[],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %none = torch.constant.none
    %0 = torch.aten.prod %arg0, %none : !torch.vtensor<[5,5,5],f16>, !torch.none -> !torch.vtensor<[],f16>
    return %0 : !torch.vtensor<[],f16>
  }
  func.func @test_reduction_prod_cpu_float16_15(%arg0: !torch.vtensor<[5,5,5],f16>) -> !torch.vtensor<[5,5],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %int1 = torch.constant.int 1
    %false = torch.constant.bool false
    %none = torch.constant.none
    %0 = torch.aten.prod.dim_int %arg0, %int1, %false, %none : !torch.vtensor<[5,5,5],f16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[5,5],f16>
    return %0 : !torch.vtensor<[5,5],f16>
  }
  func.func @test_reduction_prod_cpu_float16_16(%arg0: !torch.vtensor<[5,5,5],f16>) -> !torch.vtensor<[],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %none = torch.constant.none
    %0 = torch.aten.prod %arg0, %none : !torch.vtensor<[5,5,5],f16>, !torch.none -> !torch.vtensor<[],f16>
    return %0 : !torch.vtensor<[],f16>
  }
  func.func @test_reduction_prod_cpu_float16_17(%arg0: !torch.vtensor<[5,5,5],f16>) -> !torch.vtensor<[5,5],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %int1 = torch.constant.int 1
    %false = torch.constant.bool false
    %none = torch.constant.none
    %0 = torch.aten.prod.dim_int %arg0, %int1, %false, %none : !torch.vtensor<[5,5,5],f16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[5,5],f16>
    return %0 : !torch.vtensor<[5,5],f16>
  }
  func.func @test_reduction_prod_cpu_float16_18(%arg0: !torch.vtensor<[5,5,5],f16>) -> !torch.vtensor<[],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %none = torch.constant.none
    %0 = torch.aten.prod %arg0, %none : !torch.vtensor<[5,5,5],f16>, !torch.none -> !torch.vtensor<[],f16>
    return %0 : !torch.vtensor<[],f16>
  }
  func.func @test_reduction_prod_cpu_float16_19(%arg0: !torch.vtensor<[5,5,5],f16>) -> !torch.vtensor<[5,5],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %int1 = torch.constant.int 1
    %false = torch.constant.bool false
    %int5 = torch.constant.int 5
    %0 = torch.aten.prod.dim_int %arg0, %int1, %false, %int5 : !torch.vtensor<[5,5,5],f16>, !torch.int, !torch.bool, !torch.int -> !torch.vtensor<[5,5],f16>
    return %0 : !torch.vtensor<[5,5],f16>
  }
  func.func @test_reduction_prod_cpu_float16_20(%arg0: !torch.vtensor<[5,5,5],f16>) -> !torch.vtensor<[1,5,5],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %int0 = torch.constant.int 0
    %true = torch.constant.bool true
    %none = torch.constant.none
    %0 = torch.aten.prod.dim_int %arg0, %int0, %true, %none : !torch.vtensor<[5,5,5],f16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[1,5,5],f16>
    return %0 : !torch.vtensor<[1,5,5],f16>
  }
  func.func @test_reduction_prod_cpu_float16_21(%arg0: !torch.vtensor<[5,5,5],f16>) -> !torch.vtensor<[5,1,5],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %int1 = torch.constant.int 1
    %true = torch.constant.bool true
    %none = torch.constant.none
    %0 = torch.aten.prod.dim_int %arg0, %int1, %true, %none : !torch.vtensor<[5,5,5],f16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[5,1,5],f16>
    return %0 : !torch.vtensor<[5,1,5],f16>
  }
  func.func @test_reduction_prod_cpu_float16_22(%arg0: !torch.vtensor<[5,5,5],f16>) -> !torch.vtensor<[5,5,1],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %int2 = torch.constant.int 2
    %true = torch.constant.bool true
    %none = torch.constant.none
    %0 = torch.aten.prod.dim_int %arg0, %int2, %true, %none : !torch.vtensor<[5,5,5],f16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[5,5,1],f16>
    return %0 : !torch.vtensor<[5,5,1],f16>
  }
  func.func @test_reduction_prod_cpu_float16_23(%arg0: !torch.vtensor<[],f16>) -> !torch.vtensor<[],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %int0 = torch.constant.int 0
    %true = torch.constant.bool true
    %none = torch.constant.none
    %0 = torch.aten.prod.dim_int %arg0, %int0, %true, %none : !torch.vtensor<[],f16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[],f16>
    return %0 : !torch.vtensor<[],f16>
  }
  func.func @test_reduction_prod_cpu_float16_24(%arg0: !torch.vtensor<[1],f16>) -> !torch.vtensor<[1],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %int0 = torch.constant.int 0
    %true = torch.constant.bool true
    %none = torch.constant.none
    %0 = torch.aten.prod.dim_int %arg0, %int0, %true, %none : !torch.vtensor<[1],f16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[1],f16>
    return %0 : !torch.vtensor<[1],f16>
  }
  func.func @test_reduction_prod_cpu_float16_25(%arg0: !torch.vtensor<[0],f16>) -> !torch.vtensor<[1],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %int0 = torch.constant.int 0
    %true = torch.constant.bool true
    %none = torch.constant.none
    %0 = torch.aten.prod.dim_int %arg0, %int0, %true, %none : !torch.vtensor<[0],f16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[1],f16>
    return %0 : !torch.vtensor<[1],f16>
  }
  func.func @test_reduction_prod_cpu_float16_26(%arg0: !torch.vtensor<[5,5,5],f16>) -> !torch.vtensor<[5,1,5],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %int1 = torch.constant.int 1
    %true = torch.constant.bool true
    %none = torch.constant.none
    %0 = torch.aten.prod.dim_int %arg0, %int1, %true, %none : !torch.vtensor<[5,5,5],f16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[5,1,5],f16>
    return %0 : !torch.vtensor<[5,1,5],f16>
  }
  func.func @test_reduction_prod_cpu_float16_27(%arg0: !torch.vtensor<[5,5,5],f16>) -> !torch.vtensor<[5,1,5],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %int1 = torch.constant.int 1
    %true = torch.constant.bool true
    %none = torch.constant.none
    %0 = torch.aten.prod.dim_int %arg0, %int1, %true, %none : !torch.vtensor<[5,5,5],f16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[5,1,5],f16>
    return %0 : !torch.vtensor<[5,1,5],f16>
  }
  func.func @test_reduction_prod_cpu_float16_28(%arg0: !torch.vtensor<[5,5,5],f16>) -> !torch.vtensor<[5,1,5],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %int1 = torch.constant.int 1
    %true = torch.constant.bool true
    %none = torch.constant.none
    %0 = torch.aten.prod.dim_int %arg0, %int1, %true, %none : !torch.vtensor<[5,5,5],f16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[5,1,5],f16>
    return %0 : !torch.vtensor<[5,1,5],f16>
  }
  func.func @test_reduction_prod_cpu_float16_29(%arg0: !torch.vtensor<[5,5,5],f16>) -> !torch.vtensor<[5,1,5],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %int1 = torch.constant.int 1
    %true = torch.constant.bool true
    %int5 = torch.constant.int 5
    %0 = torch.aten.prod.dim_int %arg0, %int1, %true, %int5 : !torch.vtensor<[5,5,5],f16>, !torch.int, !torch.bool, !torch.int -> !torch.vtensor<[5,1,5],f16>
    return %0 : !torch.vtensor<[5,1,5],f16>
  }
  func.func @test_reduction_prod_cpu_float16_30(%arg0: !torch.vtensor<[5,5],f16>) -> !torch.vtensor<[],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %none = torch.constant.none
    %0 = torch.aten.prod %arg0, %none : !torch.vtensor<[5,5],f16>, !torch.none -> !torch.vtensor<[],f16>
    return %0 : !torch.vtensor<[],f16>
  }
  func.func @test_reduction_prod_cpu_float16_31(%arg0: !torch.vtensor<[3,3,3],f16>) -> !torch.vtensor<[3,3],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %int1 = torch.constant.int 1
    %false = torch.constant.bool false
    %none = torch.constant.none
    %0 = torch.aten.prod.dim_int %arg0, %int1, %false, %none : !torch.vtensor<[3,3,3],f16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3,3],f16>
    return %0 : !torch.vtensor<[3,3],f16>
  }
  func.func @test_reduction_prod_cpu_float16_32(%arg0: !torch.vtensor<[3,3,3],f16>) -> !torch.vtensor<[3,1,3],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %int1 = torch.constant.int 1
    %true = torch.constant.bool true
    %none = torch.constant.none
    %0 = torch.aten.prod.dim_int %arg0, %int1, %true, %none : !torch.vtensor<[3,3,3],f16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3,1,3],f16>
    return %0 : !torch.vtensor<[3,1,3],f16>
  }
  func.func @test_reduction_prod_cpu_float16_33(%arg0: !torch.vtensor<[3,0],f16>) -> !torch.vtensor<[3],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %int1 = torch.constant.int 1
    %false = torch.constant.bool false
    %none = torch.constant.none
    %0 = torch.aten.prod.dim_int %arg0, %int1, %false, %none : !torch.vtensor<[3,0],f16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3],f16>
    return %0 : !torch.vtensor<[3],f16>
  }
  func.func @test_reduction_prod_cpu_float16_34(%arg0: !torch.vtensor<[3,0],f16>) -> !torch.vtensor<[3,1],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %int1 = torch.constant.int 1
    %true = torch.constant.bool true
    %none = torch.constant.none
    %0 = torch.aten.prod.dim_int %arg0, %int1, %true, %none : !torch.vtensor<[3,0],f16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[3,1],f16>
    return %0 : !torch.vtensor<[3,1],f16>
  }
  func.func @test_reduction_prod_cpu_float16_35(%arg0: !torch.vtensor<[4],f16>) -> !torch.vtensor<[],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %none = torch.constant.none
    %0 = torch.aten.prod %arg0, %none : !torch.vtensor<[4],f16>, !torch.none -> !torch.vtensor<[],f16>
    return %0 : !torch.vtensor<[],f16>
  }
  func.func @test_reduction_prod_cpu_float16_36(%arg0: !torch.vtensor<[],f16>) -> !torch.vtensor<[],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %none = torch.constant.none
    %0 = torch.aten.prod %arg0, %none : !torch.vtensor<[],f16>, !torch.none -> !torch.vtensor<[],f16>
    return %0 : !torch.vtensor<[],f16>
  }
  func.func @test_reduction_prod_cpu_float16_37(%arg0: !torch.vtensor<[],f16>) -> !torch.vtensor<[],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %int0 = torch.constant.int 0
    %false = torch.constant.bool false
    %none = torch.constant.none
    %0 = torch.aten.prod.dim_int %arg0, %int0, %false, %none : !torch.vtensor<[],f16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[],f16>
    return %0 : !torch.vtensor<[],f16>
  }
  func.func @test_reduction_prod_cpu_float16_38(%arg0: !torch.vtensor<[],f16>) -> !torch.vtensor<[],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %int0 = torch.constant.int 0
    %true = torch.constant.bool true
    %none = torch.constant.none
    %0 = torch.aten.prod.dim_int %arg0, %int0, %true, %none : !torch.vtensor<[],f16>, !torch.int, !torch.bool, !torch.none -> !torch.vtensor<[],f16>
    return %0 : !torch.vtensor<[],f16>
  }
}
