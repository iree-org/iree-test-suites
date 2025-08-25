module @module {
  func.func @test_reduction_sum_cpu_int32_0(%arg0: !torch.vtensor<[],si32>) -> !torch.vtensor<[],si64> attributes {torch.assume_strict_symbolic_shapes} {
    %none = torch.constant.none
    %0 = torch.aten.sum %arg0, %none : !torch.vtensor<[],si32>, !torch.none -> !torch.vtensor<[],si64>
    return %0 : !torch.vtensor<[],si64>
  }
  func.func @test_reduction_sum_cpu_int32_1(%arg0: !torch.vtensor<[],si32>) -> !torch.vtensor<[],si64> attributes {torch.assume_strict_symbolic_shapes} {
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %none = torch.constant.none
    %1 = torch.aten.sum.dim_IntList %arg0, %0, %true, %none : !torch.vtensor<[],si32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[],si64>
    return %1 : !torch.vtensor<[],si64>
  }
  func.func @test_reduction_sum_cpu_int32_2(%arg0: !torch.vtensor<[],si32>) -> !torch.vtensor<[],si64> attributes {torch.assume_strict_symbolic_shapes} {
    %int-1 = torch.constant.int -1
    %0 = torch.prim.ListConstruct %int-1 : (!torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %none = torch.constant.none
    %1 = torch.aten.sum.dim_IntList %arg0, %0, %false, %none : !torch.vtensor<[],si32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[],si64>
    return %1 : !torch.vtensor<[],si64>
  }
  func.func @test_reduction_sum_cpu_int32_3(%arg0: !torch.vtensor<[],si32>) -> !torch.vtensor<[],si64> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %false = torch.constant.bool false
    %none = torch.constant.none
    %1 = torch.aten.sum.dim_IntList %arg0, %0, %false, %none : !torch.vtensor<[],si32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[],si64>
    return %1 : !torch.vtensor<[],si64>
  }
  func.func @test_reduction_sum_cpu_int32_4(%arg0: !torch.vtensor<[2],si32>) -> !torch.vtensor<[],si64> attributes {torch.assume_strict_symbolic_shapes} {
    %none = torch.constant.none
    %0 = torch.aten.sum %arg0, %none : !torch.vtensor<[2],si32>, !torch.none -> !torch.vtensor<[],si64>
    return %0 : !torch.vtensor<[],si64>
  }
  func.func @test_reduction_sum_cpu_int32_5(%arg0: !torch.vtensor<[2],si32>) -> !torch.vtensor<[1],si64> attributes {torch.assume_strict_symbolic_shapes} {
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %none = torch.constant.none
    %1 = torch.aten.sum.dim_IntList %arg0, %0, %true, %none : !torch.vtensor<[2],si32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1],si64>
    return %1 : !torch.vtensor<[1],si64>
  }
  func.func @test_reduction_sum_cpu_int32_6(%arg0: !torch.vtensor<[2],si32>) -> !torch.vtensor<[],si64> attributes {torch.assume_strict_symbolic_shapes} {
    %int-1 = torch.constant.int -1
    %0 = torch.prim.ListConstruct %int-1 : (!torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %none = torch.constant.none
    %1 = torch.aten.sum.dim_IntList %arg0, %0, %false, %none : !torch.vtensor<[2],si32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[],si64>
    return %1 : !torch.vtensor<[],si64>
  }
  func.func @test_reduction_sum_cpu_int32_7(%arg0: !torch.vtensor<[2],si32>) -> !torch.vtensor<[],si64> attributes {torch.assume_strict_symbolic_shapes} {
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %none = torch.constant.none
    %1 = torch.aten.sum.dim_IntList %arg0, %0, %false, %none : !torch.vtensor<[2],si32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[],si64>
    return %1 : !torch.vtensor<[],si64>
  }
  func.func @test_reduction_sum_cpu_int32_8(%arg0: !torch.vtensor<[3,5],si32>) -> !torch.vtensor<[],si64> attributes {torch.assume_strict_symbolic_shapes} {
    %none = torch.constant.none
    %0 = torch.aten.sum %arg0, %none : !torch.vtensor<[3,5],si32>, !torch.none -> !torch.vtensor<[],si64>
    return %0 : !torch.vtensor<[],si64>
  }
  func.func @test_reduction_sum_cpu_int32_9(%arg0: !torch.vtensor<[3,5],si32>) -> !torch.vtensor<[1,5],si64> attributes {torch.assume_strict_symbolic_shapes} {
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %none = torch.constant.none
    %1 = torch.aten.sum.dim_IntList %arg0, %0, %true, %none : !torch.vtensor<[3,5],si32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,5],si64>
    return %1 : !torch.vtensor<[1,5],si64>
  }
  func.func @test_reduction_sum_cpu_int32_10(%arg0: !torch.vtensor<[3,5],si32>) -> !torch.vtensor<[3],si64> attributes {torch.assume_strict_symbolic_shapes} {
    %int-1 = torch.constant.int -1
    %0 = torch.prim.ListConstruct %int-1 : (!torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %none = torch.constant.none
    %1 = torch.aten.sum.dim_IntList %arg0, %0, %false, %none : !torch.vtensor<[3,5],si32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[3],si64>
    return %1 : !torch.vtensor<[3],si64>
  }
  func.func @test_reduction_sum_cpu_int32_11(%arg0: !torch.vtensor<[3,5],si32>) -> !torch.vtensor<[],si64> attributes {torch.assume_strict_symbolic_shapes} {
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %0 = torch.prim.ListConstruct %int0, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %none = torch.constant.none
    %1 = torch.aten.sum.dim_IntList %arg0, %0, %false, %none : !torch.vtensor<[3,5],si32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[],si64>
    return %1 : !torch.vtensor<[],si64>
  }
  func.func @test_reduction_sum_cpu_int32_12(%arg0: !torch.vtensor<[3,5],si32>) -> !torch.vtensor<[1,1],si64> attributes {torch.assume_strict_symbolic_shapes} {
    %int0 = torch.constant.int 0
    %int-1 = torch.constant.int -1
    %0 = torch.prim.ListConstruct %int0, %int-1 : (!torch.int, !torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %none = torch.constant.none
    %1 = torch.aten.sum.dim_IntList %arg0, %0, %true, %none : !torch.vtensor<[3,5],si32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,1],si64>
    return %1 : !torch.vtensor<[1,1],si64>
  }
  func.func @test_reduction_sum_cpu_int32_13(%arg0: !torch.vtensor<[3,2,1,2],si32>) -> !torch.vtensor<[],si64> attributes {torch.assume_strict_symbolic_shapes} {
    %none = torch.constant.none
    %0 = torch.aten.sum %arg0, %none : !torch.vtensor<[3,2,1,2],si32>, !torch.none -> !torch.vtensor<[],si64>
    return %0 : !torch.vtensor<[],si64>
  }
  func.func @test_reduction_sum_cpu_int32_14(%arg0: !torch.vtensor<[3,2,1,2],si32>) -> !torch.vtensor<[1,2,1,2],si64> attributes {torch.assume_strict_symbolic_shapes} {
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %none = torch.constant.none
    %1 = torch.aten.sum.dim_IntList %arg0, %0, %true, %none : !torch.vtensor<[3,2,1,2],si32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,2,1,2],si64>
    return %1 : !torch.vtensor<[1,2,1,2],si64>
  }
  func.func @test_reduction_sum_cpu_int32_15(%arg0: !torch.vtensor<[3,2,1,2],si32>) -> !torch.vtensor<[3,2,1],si64> attributes {torch.assume_strict_symbolic_shapes} {
    %int-1 = torch.constant.int -1
    %0 = torch.prim.ListConstruct %int-1 : (!torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %none = torch.constant.none
    %1 = torch.aten.sum.dim_IntList %arg0, %0, %false, %none : !torch.vtensor<[3,2,1,2],si32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[3,2,1],si64>
    return %1 : !torch.vtensor<[3,2,1],si64>
  }
  func.func @test_reduction_sum_cpu_int32_16(%arg0: !torch.vtensor<[3,2,1,2],si32>) -> !torch.vtensor<[3,2,1,2],si64> attributes {torch.assume_strict_symbolic_shapes} {
    %int2 = torch.constant.int 2
    %0 = torch.prim.ListConstruct %int2 : (!torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %none = torch.constant.none
    %1 = torch.aten.sum.dim_IntList %arg0, %0, %true, %none : !torch.vtensor<[3,2,1,2],si32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[3,2,1,2],si64>
    return %1 : !torch.vtensor<[3,2,1,2],si64>
  }
  func.func @test_reduction_sum_cpu_int32_17(%arg0: !torch.vtensor<[3,2,1,2],si32>) -> !torch.vtensor<[],si64> attributes {torch.assume_strict_symbolic_shapes} {
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %int2 = torch.constant.int 2
    %int3 = torch.constant.int 3
    %0 = torch.prim.ListConstruct %int0, %int1, %int2, %int3 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %none = torch.constant.none
    %1 = torch.aten.sum.dim_IntList %arg0, %0, %false, %none : !torch.vtensor<[3,2,1,2],si32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[],si64>
    return %1 : !torch.vtensor<[],si64>
  }
  func.func @test_reduction_sum_cpu_int32_18(%arg0: !torch.vtensor<[3,2,1,2],si32>) -> !torch.vtensor<[1,2,1,1],si64> attributes {torch.assume_strict_symbolic_shapes} {
    %int0 = torch.constant.int 0
    %int-1 = torch.constant.int -1
    %0 = torch.prim.ListConstruct %int0, %int-1 : (!torch.int, !torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %none = torch.constant.none
    %1 = torch.aten.sum.dim_IntList %arg0, %0, %true, %none : !torch.vtensor<[3,2,1,2],si32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,2,1,1],si64>
    return %1 : !torch.vtensor<[1,2,1,1],si64>
  }
  func.func @test_reduction_sum_cpu_int32_19(%arg0: !torch.vtensor<[3,2,1,2],si32>) -> !torch.vtensor<[3,1],si64> attributes {torch.assume_strict_symbolic_shapes} {
    %int1 = torch.constant.int 1
    %int3 = torch.constant.int 3
    %0 = torch.prim.ListConstruct %int1, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %none = torch.constant.none
    %1 = torch.aten.sum.dim_IntList %arg0, %0, %false, %none : !torch.vtensor<[3,2,1,2],si32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[3,1],si64>
    return %1 : !torch.vtensor<[3,1],si64>
  }
}
