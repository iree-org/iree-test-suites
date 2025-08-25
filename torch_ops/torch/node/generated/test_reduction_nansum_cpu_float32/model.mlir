module @module {
  func.func @test_reduction_nansum_cpu_float32_0(%arg0: !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %none = torch.constant.none
    %false = torch.constant.bool false
    %none_0 = torch.constant.none
    %0 = torch.operator "torch.aten.nansum"(%arg0, %none, %false, %none_0) : (!torch.vtensor<[],f32>, !torch.none, !torch.bool, !torch.none) -> !torch.vtensor<[],f32> 
    return %0 : !torch.vtensor<[],f32>
  }
  func.func @test_reduction_nansum_cpu_float32_1(%arg0: !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %none = torch.constant.none
    %false = torch.constant.bool false
    %none_0 = torch.constant.none
    %0 = torch.operator "torch.aten.nansum"(%arg0, %none, %false, %none_0) : (!torch.vtensor<[],f32>, !torch.none, !torch.bool, !torch.none) -> !torch.vtensor<[],f32> 
    return %0 : !torch.vtensor<[],f32>
  }
  func.func @test_reduction_nansum_cpu_float32_2(%arg0: !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %none = torch.constant.none
    %1 = torch.operator "torch.aten.nansum"(%arg0, %0, %true, %none) : (!torch.vtensor<[],f32>, !torch.list<int>, !torch.bool, !torch.none) -> !torch.vtensor<[],f32> 
    return %1 : !torch.vtensor<[],f32>
  }
  func.func @test_reduction_nansum_cpu_float32_3(%arg0: !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %int-1 = torch.constant.int -1
    %0 = torch.prim.ListConstruct %int-1 : (!torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %none = torch.constant.none
    %1 = torch.operator "torch.aten.nansum"(%arg0, %0, %false, %none) : (!torch.vtensor<[],f32>, !torch.list<int>, !torch.bool, !torch.none) -> !torch.vtensor<[],f32> 
    return %1 : !torch.vtensor<[],f32>
  }
  func.func @test_reduction_nansum_cpu_float32_4(%arg0: !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %false = torch.constant.bool false
    %none = torch.constant.none
    %1 = torch.operator "torch.aten.nansum"(%arg0, %0, %false, %none) : (!torch.vtensor<[],f32>, !torch.list<int>, !torch.bool, !torch.none) -> !torch.vtensor<[],f32> 
    return %1 : !torch.vtensor<[],f32>
  }
  func.func @test_reduction_nansum_cpu_float32_5(%arg0: !torch.vtensor<[2],f32>) -> !torch.vtensor<[],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %none = torch.constant.none
    %false = torch.constant.bool false
    %none_0 = torch.constant.none
    %0 = torch.operator "torch.aten.nansum"(%arg0, %none, %false, %none_0) : (!torch.vtensor<[2],f32>, !torch.none, !torch.bool, !torch.none) -> !torch.vtensor<[],f32> 
    return %0 : !torch.vtensor<[],f32>
  }
  func.func @test_reduction_nansum_cpu_float32_6(%arg0: !torch.vtensor<[2],f32>) -> !torch.vtensor<[],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %none = torch.constant.none
    %false = torch.constant.bool false
    %none_0 = torch.constant.none
    %0 = torch.operator "torch.aten.nansum"(%arg0, %none, %false, %none_0) : (!torch.vtensor<[2],f32>, !torch.none, !torch.bool, !torch.none) -> !torch.vtensor<[],f32> 
    return %0 : !torch.vtensor<[],f32>
  }
  func.func @test_reduction_nansum_cpu_float32_7(%arg0: !torch.vtensor<[2],f32>) -> !torch.vtensor<[1],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %none = torch.constant.none
    %1 = torch.operator "torch.aten.nansum"(%arg0, %0, %true, %none) : (!torch.vtensor<[2],f32>, !torch.list<int>, !torch.bool, !torch.none) -> !torch.vtensor<[1],f32> 
    return %1 : !torch.vtensor<[1],f32>
  }
  func.func @test_reduction_nansum_cpu_float32_8(%arg0: !torch.vtensor<[2],f32>) -> !torch.vtensor<[],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %int-1 = torch.constant.int -1
    %0 = torch.prim.ListConstruct %int-1 : (!torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %none = torch.constant.none
    %1 = torch.operator "torch.aten.nansum"(%arg0, %0, %false, %none) : (!torch.vtensor<[2],f32>, !torch.list<int>, !torch.bool, !torch.none) -> !torch.vtensor<[],f32> 
    return %1 : !torch.vtensor<[],f32>
  }
  func.func @test_reduction_nansum_cpu_float32_9(%arg0: !torch.vtensor<[2],f32>) -> !torch.vtensor<[],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %none = torch.constant.none
    %1 = torch.operator "torch.aten.nansum"(%arg0, %0, %false, %none) : (!torch.vtensor<[2],f32>, !torch.list<int>, !torch.bool, !torch.none) -> !torch.vtensor<[],f32> 
    return %1 : !torch.vtensor<[],f32>
  }
  func.func @test_reduction_nansum_cpu_float32_10(%arg0: !torch.vtensor<[3,5],f32>) -> !torch.vtensor<[],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %none = torch.constant.none
    %false = torch.constant.bool false
    %none_0 = torch.constant.none
    %0 = torch.operator "torch.aten.nansum"(%arg0, %none, %false, %none_0) : (!torch.vtensor<[3,5],f32>, !torch.none, !torch.bool, !torch.none) -> !torch.vtensor<[],f32> 
    return %0 : !torch.vtensor<[],f32>
  }
  func.func @test_reduction_nansum_cpu_float32_11(%arg0: !torch.vtensor<[3,5],f32>) -> !torch.vtensor<[],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %none = torch.constant.none
    %false = torch.constant.bool false
    %none_0 = torch.constant.none
    %0 = torch.operator "torch.aten.nansum"(%arg0, %none, %false, %none_0) : (!torch.vtensor<[3,5],f32>, !torch.none, !torch.bool, !torch.none) -> !torch.vtensor<[],f32> 
    return %0 : !torch.vtensor<[],f32>
  }
  func.func @test_reduction_nansum_cpu_float32_12(%arg0: !torch.vtensor<[3,5],f32>) -> !torch.vtensor<[1,5],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %none = torch.constant.none
    %1 = torch.operator "torch.aten.nansum"(%arg0, %0, %true, %none) : (!torch.vtensor<[3,5],f32>, !torch.list<int>, !torch.bool, !torch.none) -> !torch.vtensor<[1,5],f32> 
    return %1 : !torch.vtensor<[1,5],f32>
  }
  func.func @test_reduction_nansum_cpu_float32_13(%arg0: !torch.vtensor<[3,5],f32>) -> !torch.vtensor<[3],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %int-1 = torch.constant.int -1
    %0 = torch.prim.ListConstruct %int-1 : (!torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %none = torch.constant.none
    %1 = torch.operator "torch.aten.nansum"(%arg0, %0, %false, %none) : (!torch.vtensor<[3,5],f32>, !torch.list<int>, !torch.bool, !torch.none) -> !torch.vtensor<[3],f32> 
    return %1 : !torch.vtensor<[3],f32>
  }
  func.func @test_reduction_nansum_cpu_float32_14(%arg0: !torch.vtensor<[3,5],f32>) -> !torch.vtensor<[],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %0 = torch.prim.ListConstruct %int0, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %none = torch.constant.none
    %1 = torch.operator "torch.aten.nansum"(%arg0, %0, %false, %none) : (!torch.vtensor<[3,5],f32>, !torch.list<int>, !torch.bool, !torch.none) -> !torch.vtensor<[],f32> 
    return %1 : !torch.vtensor<[],f32>
  }
  func.func @test_reduction_nansum_cpu_float32_15(%arg0: !torch.vtensor<[3,5],f32>) -> !torch.vtensor<[1,1],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %int0 = torch.constant.int 0
    %int-1 = torch.constant.int -1
    %0 = torch.prim.ListConstruct %int0, %int-1 : (!torch.int, !torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %none = torch.constant.none
    %1 = torch.operator "torch.aten.nansum"(%arg0, %0, %true, %none) : (!torch.vtensor<[3,5],f32>, !torch.list<int>, !torch.bool, !torch.none) -> !torch.vtensor<[1,1],f32> 
    return %1 : !torch.vtensor<[1,1],f32>
  }
  func.func @test_reduction_nansum_cpu_float32_16(%arg0: !torch.vtensor<[3,2,1,2],f32>) -> !torch.vtensor<[],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %none = torch.constant.none
    %false = torch.constant.bool false
    %none_0 = torch.constant.none
    %0 = torch.operator "torch.aten.nansum"(%arg0, %none, %false, %none_0) : (!torch.vtensor<[3,2,1,2],f32>, !torch.none, !torch.bool, !torch.none) -> !torch.vtensor<[],f32> 
    return %0 : !torch.vtensor<[],f32>
  }
  func.func @test_reduction_nansum_cpu_float32_17(%arg0: !torch.vtensor<[3,2,1,2],f32>) -> !torch.vtensor<[],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %none = torch.constant.none
    %false = torch.constant.bool false
    %none_0 = torch.constant.none
    %0 = torch.operator "torch.aten.nansum"(%arg0, %none, %false, %none_0) : (!torch.vtensor<[3,2,1,2],f32>, !torch.none, !torch.bool, !torch.none) -> !torch.vtensor<[],f32> 
    return %0 : !torch.vtensor<[],f32>
  }
  func.func @test_reduction_nansum_cpu_float32_18(%arg0: !torch.vtensor<[3,2,1,2],f32>) -> !torch.vtensor<[1,2,1,2],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %none = torch.constant.none
    %1 = torch.operator "torch.aten.nansum"(%arg0, %0, %true, %none) : (!torch.vtensor<[3,2,1,2],f32>, !torch.list<int>, !torch.bool, !torch.none) -> !torch.vtensor<[1,2,1,2],f32> 
    return %1 : !torch.vtensor<[1,2,1,2],f32>
  }
  func.func @test_reduction_nansum_cpu_float32_19(%arg0: !torch.vtensor<[3,2,1,2],f32>) -> !torch.vtensor<[3,2,1],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %int-1 = torch.constant.int -1
    %0 = torch.prim.ListConstruct %int-1 : (!torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %none = torch.constant.none
    %1 = torch.operator "torch.aten.nansum"(%arg0, %0, %false, %none) : (!torch.vtensor<[3,2,1,2],f32>, !torch.list<int>, !torch.bool, !torch.none) -> !torch.vtensor<[3,2,1],f32> 
    return %1 : !torch.vtensor<[3,2,1],f32>
  }
  func.func @test_reduction_nansum_cpu_float32_20(%arg0: !torch.vtensor<[3,2,1,2],f32>) -> !torch.vtensor<[3,2,1,2],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %int2 = torch.constant.int 2
    %0 = torch.prim.ListConstruct %int2 : (!torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %none = torch.constant.none
    %1 = torch.operator "torch.aten.nansum"(%arg0, %0, %true, %none) : (!torch.vtensor<[3,2,1,2],f32>, !torch.list<int>, !torch.bool, !torch.none) -> !torch.vtensor<[3,2,1,2],f32> 
    return %1 : !torch.vtensor<[3,2,1,2],f32>
  }
  func.func @test_reduction_nansum_cpu_float32_21(%arg0: !torch.vtensor<[3,2,1,2],f32>) -> !torch.vtensor<[],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %int2 = torch.constant.int 2
    %int3 = torch.constant.int 3
    %0 = torch.prim.ListConstruct %int0, %int1, %int2, %int3 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %none = torch.constant.none
    %1 = torch.operator "torch.aten.nansum"(%arg0, %0, %false, %none) : (!torch.vtensor<[3,2,1,2],f32>, !torch.list<int>, !torch.bool, !torch.none) -> !torch.vtensor<[],f32> 
    return %1 : !torch.vtensor<[],f32>
  }
  func.func @test_reduction_nansum_cpu_float32_22(%arg0: !torch.vtensor<[3,2,1,2],f32>) -> !torch.vtensor<[1,2,1,1],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %int0 = torch.constant.int 0
    %int-1 = torch.constant.int -1
    %0 = torch.prim.ListConstruct %int0, %int-1 : (!torch.int, !torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %none = torch.constant.none
    %1 = torch.operator "torch.aten.nansum"(%arg0, %0, %true, %none) : (!torch.vtensor<[3,2,1,2],f32>, !torch.list<int>, !torch.bool, !torch.none) -> !torch.vtensor<[1,2,1,1],f32> 
    return %1 : !torch.vtensor<[1,2,1,1],f32>
  }
  func.func @test_reduction_nansum_cpu_float32_23(%arg0: !torch.vtensor<[3,2,1,2],f32>) -> !torch.vtensor<[3,1],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %int1 = torch.constant.int 1
    %int3 = torch.constant.int 3
    %0 = torch.prim.ListConstruct %int1, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %none = torch.constant.none
    %1 = torch.operator "torch.aten.nansum"(%arg0, %0, %false, %none) : (!torch.vtensor<[3,2,1,2],f32>, !torch.list<int>, !torch.bool, !torch.none) -> !torch.vtensor<[3,1],f32> 
    return %1 : !torch.vtensor<[3,1],f32>
  }
  func.func @test_reduction_nansum_cpu_float32_24(%arg0: !torch.vtensor<[3],f32>) -> !torch.vtensor<[],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %none = torch.constant.none
    %false = torch.constant.bool false
    %none_0 = torch.constant.none
    %0 = torch.operator "torch.aten.nansum"(%arg0, %none, %false, %none_0) : (!torch.vtensor<[3],f32>, !torch.none, !torch.bool, !torch.none) -> !torch.vtensor<[],f32> 
    return %0 : !torch.vtensor<[],f32>
  }
  func.func @test_reduction_nansum_cpu_float32_25(%arg0: !torch.vtensor<[3],f32>) -> !torch.vtensor<[],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %none = torch.constant.none
    %false = torch.constant.bool false
    %none_0 = torch.constant.none
    %0 = torch.operator "torch.aten.nansum"(%arg0, %none, %false, %none_0) : (!torch.vtensor<[3],f32>, !torch.none, !torch.bool, !torch.none) -> !torch.vtensor<[],f32> 
    return %0 : !torch.vtensor<[],f32>
  }
  func.func @test_reduction_nansum_cpu_float32_26(%arg0: !torch.vtensor<[3],f32>) -> !torch.vtensor<[1],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %none = torch.constant.none
    %1 = torch.operator "torch.aten.nansum"(%arg0, %0, %true, %none) : (!torch.vtensor<[3],f32>, !torch.list<int>, !torch.bool, !torch.none) -> !torch.vtensor<[1],f32> 
    return %1 : !torch.vtensor<[1],f32>
  }
  func.func @test_reduction_nansum_cpu_float32_27(%arg0: !torch.vtensor<[3],f32>) -> !torch.vtensor<[],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %int-1 = torch.constant.int -1
    %0 = torch.prim.ListConstruct %int-1 : (!torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %none = torch.constant.none
    %1 = torch.operator "torch.aten.nansum"(%arg0, %0, %false, %none) : (!torch.vtensor<[3],f32>, !torch.list<int>, !torch.bool, !torch.none) -> !torch.vtensor<[],f32> 
    return %1 : !torch.vtensor<[],f32>
  }
  func.func @test_reduction_nansum_cpu_float32_28(%arg0: !torch.vtensor<[3],f32>) -> !torch.vtensor<[],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %none = torch.constant.none
    %1 = torch.operator "torch.aten.nansum"(%arg0, %0, %false, %none) : (!torch.vtensor<[3],f32>, !torch.list<int>, !torch.bool, !torch.none) -> !torch.vtensor<[],f32> 
    return %1 : !torch.vtensor<[],f32>
  }
  func.func @test_reduction_nansum_cpu_float32_29(%arg0: !torch.vtensor<[2,2],f32>) -> !torch.vtensor<[],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %none = torch.constant.none
    %false = torch.constant.bool false
    %none_0 = torch.constant.none
    %0 = torch.operator "torch.aten.nansum"(%arg0, %none, %false, %none_0) : (!torch.vtensor<[2,2],f32>, !torch.none, !torch.bool, !torch.none) -> !torch.vtensor<[],f32> 
    return %0 : !torch.vtensor<[],f32>
  }
  func.func @test_reduction_nansum_cpu_float32_30(%arg0: !torch.vtensor<[2,2],f32>) -> !torch.vtensor<[],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %none = torch.constant.none
    %false = torch.constant.bool false
    %none_0 = torch.constant.none
    %0 = torch.operator "torch.aten.nansum"(%arg0, %none, %false, %none_0) : (!torch.vtensor<[2,2],f32>, !torch.none, !torch.bool, !torch.none) -> !torch.vtensor<[],f32> 
    return %0 : !torch.vtensor<[],f32>
  }
  func.func @test_reduction_nansum_cpu_float32_31(%arg0: !torch.vtensor<[2,2],f32>) -> !torch.vtensor<[1,2],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %none = torch.constant.none
    %1 = torch.operator "torch.aten.nansum"(%arg0, %0, %true, %none) : (!torch.vtensor<[2,2],f32>, !torch.list<int>, !torch.bool, !torch.none) -> !torch.vtensor<[1,2],f32> 
    return %1 : !torch.vtensor<[1,2],f32>
  }
  func.func @test_reduction_nansum_cpu_float32_32(%arg0: !torch.vtensor<[2,2],f32>) -> !torch.vtensor<[2],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %int-1 = torch.constant.int -1
    %0 = torch.prim.ListConstruct %int-1 : (!torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %none = torch.constant.none
    %1 = torch.operator "torch.aten.nansum"(%arg0, %0, %false, %none) : (!torch.vtensor<[2,2],f32>, !torch.list<int>, !torch.bool, !torch.none) -> !torch.vtensor<[2],f32> 
    return %1 : !torch.vtensor<[2],f32>
  }
  func.func @test_reduction_nansum_cpu_float32_33(%arg0: !torch.vtensor<[2,2],f32>) -> !torch.vtensor<[],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %0 = torch.prim.ListConstruct %int0, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %none = torch.constant.none
    %1 = torch.operator "torch.aten.nansum"(%arg0, %0, %false, %none) : (!torch.vtensor<[2,2],f32>, !torch.list<int>, !torch.bool, !torch.none) -> !torch.vtensor<[],f32> 
    return %1 : !torch.vtensor<[],f32>
  }
  func.func @test_reduction_nansum_cpu_float32_34(%arg0: !torch.vtensor<[2,2],f32>) -> !torch.vtensor<[1,1],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %int0 = torch.constant.int 0
    %int-1 = torch.constant.int -1
    %0 = torch.prim.ListConstruct %int0, %int-1 : (!torch.int, !torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %none = torch.constant.none
    %1 = torch.operator "torch.aten.nansum"(%arg0, %0, %true, %none) : (!torch.vtensor<[2,2],f32>, !torch.list<int>, !torch.bool, !torch.none) -> !torch.vtensor<[1,1],f32> 
    return %1 : !torch.vtensor<[1,1],f32>
  }
}
