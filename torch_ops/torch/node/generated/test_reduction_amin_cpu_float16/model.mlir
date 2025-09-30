module @module {
  func.func @test_reduction_amin_cpu_float16_0(%arg0: !torch.vtensor<[],f16>) -> !torch.vtensor<[],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %false = torch.constant.bool false
    %1 = torch.aten.amin %arg0, %0, %false : !torch.vtensor<[],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[],f16>
    return %1 : !torch.vtensor<[],f16>
  }
  func.func @test_reduction_amin_cpu_float16_1(%arg0: !torch.vtensor<[],f16>) -> !torch.vtensor<[],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %1 = torch.aten.amin %arg0, %0, %true : !torch.vtensor<[],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[],f16>
    return %1 : !torch.vtensor<[],f16>
  }
  func.func @test_reduction_amin_cpu_float16_2(%arg0: !torch.vtensor<[],f16>) -> !torch.vtensor<[],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %int-1 = torch.constant.int -1
    %0 = torch.prim.ListConstruct %int-1 : (!torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %1 = torch.aten.amin %arg0, %0, %false : !torch.vtensor<[],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[],f16>
    return %1 : !torch.vtensor<[],f16>
  }
  func.func @test_reduction_amin_cpu_float16_3(%arg0: !torch.vtensor<[],f16>) -> !torch.vtensor<[],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %false = torch.constant.bool false
    %1 = torch.aten.amin %arg0, %0, %false : !torch.vtensor<[],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[],f16>
    return %1 : !torch.vtensor<[],f16>
  }
  func.func @test_reduction_amin_cpu_float16_4(%arg0: !torch.vtensor<[2],f16>) -> !torch.vtensor<[],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %false = torch.constant.bool false
    %1 = torch.aten.amin %arg0, %0, %false : !torch.vtensor<[2],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[],f16>
    return %1 : !torch.vtensor<[],f16>
  }
  func.func @test_reduction_amin_cpu_float16_5(%arg0: !torch.vtensor<[2],f16>) -> !torch.vtensor<[1],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %1 = torch.aten.amin %arg0, %0, %true : !torch.vtensor<[2],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1],f16>
    return %1 : !torch.vtensor<[1],f16>
  }
  func.func @test_reduction_amin_cpu_float16_6(%arg0: !torch.vtensor<[2],f16>) -> !torch.vtensor<[],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %int-1 = torch.constant.int -1
    %0 = torch.prim.ListConstruct %int-1 : (!torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %1 = torch.aten.amin %arg0, %0, %false : !torch.vtensor<[2],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[],f16>
    return %1 : !torch.vtensor<[],f16>
  }
  func.func @test_reduction_amin_cpu_float16_7(%arg0: !torch.vtensor<[2],f16>) -> !torch.vtensor<[],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %1 = torch.aten.amin %arg0, %0, %false : !torch.vtensor<[2],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[],f16>
    return %1 : !torch.vtensor<[],f16>
  }
  func.func @test_reduction_amin_cpu_float16_8(%arg0: !torch.vtensor<[3,5],f16>) -> !torch.vtensor<[],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %false = torch.constant.bool false
    %1 = torch.aten.amin %arg0, %0, %false : !torch.vtensor<[3,5],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[],f16>
    return %1 : !torch.vtensor<[],f16>
  }
  func.func @test_reduction_amin_cpu_float16_9(%arg0: !torch.vtensor<[3,5],f16>) -> !torch.vtensor<[1,5],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %1 = torch.aten.amin %arg0, %0, %true : !torch.vtensor<[3,5],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,5],f16>
    return %1 : !torch.vtensor<[1,5],f16>
  }
  func.func @test_reduction_amin_cpu_float16_10(%arg0: !torch.vtensor<[3,5],f16>) -> !torch.vtensor<[3],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %int-1 = torch.constant.int -1
    %0 = torch.prim.ListConstruct %int-1 : (!torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %1 = torch.aten.amin %arg0, %0, %false : !torch.vtensor<[3,5],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3],f16>
    return %1 : !torch.vtensor<[3],f16>
  }
  func.func @test_reduction_amin_cpu_float16_11(%arg0: !torch.vtensor<[3,5],f16>) -> !torch.vtensor<[],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %0 = torch.prim.ListConstruct %int0, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %1 = torch.aten.amin %arg0, %0, %false : !torch.vtensor<[3,5],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[],f16>
    return %1 : !torch.vtensor<[],f16>
  }
  func.func @test_reduction_amin_cpu_float16_12(%arg0: !torch.vtensor<[3,5],f16>) -> !torch.vtensor<[1,1],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %int0 = torch.constant.int 0
    %int-1 = torch.constant.int -1
    %0 = torch.prim.ListConstruct %int0, %int-1 : (!torch.int, !torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %1 = torch.aten.amin %arg0, %0, %true : !torch.vtensor<[3,5],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,1],f16>
    return %1 : !torch.vtensor<[1,1],f16>
  }
  func.func @test_reduction_amin_cpu_float16_13(%arg0: !torch.vtensor<[3,2,1,2],f16>) -> !torch.vtensor<[],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %false = torch.constant.bool false
    %1 = torch.aten.amin %arg0, %0, %false : !torch.vtensor<[3,2,1,2],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[],f16>
    return %1 : !torch.vtensor<[],f16>
  }
  func.func @test_reduction_amin_cpu_float16_14(%arg0: !torch.vtensor<[3,2,1,2],f16>) -> !torch.vtensor<[1,2,1,2],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %int0 = torch.constant.int 0
    %0 = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %1 = torch.aten.amin %arg0, %0, %true : !torch.vtensor<[3,2,1,2],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,2,1,2],f16>
    return %1 : !torch.vtensor<[1,2,1,2],f16>
  }
  func.func @test_reduction_amin_cpu_float16_15(%arg0: !torch.vtensor<[3,2,1,2],f16>) -> !torch.vtensor<[3,2,1],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %int-1 = torch.constant.int -1
    %0 = torch.prim.ListConstruct %int-1 : (!torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %1 = torch.aten.amin %arg0, %0, %false : !torch.vtensor<[3,2,1,2],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3,2,1],f16>
    return %1 : !torch.vtensor<[3,2,1],f16>
  }
  func.func @test_reduction_amin_cpu_float16_16(%arg0: !torch.vtensor<[3,2,1,2],f16>) -> !torch.vtensor<[3,2,1,2],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %int2 = torch.constant.int 2
    %0 = torch.prim.ListConstruct %int2 : (!torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %1 = torch.aten.amin %arg0, %0, %true : !torch.vtensor<[3,2,1,2],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3,2,1,2],f16>
    return %1 : !torch.vtensor<[3,2,1,2],f16>
  }
  func.func @test_reduction_amin_cpu_float16_17(%arg0: !torch.vtensor<[3,2,1,2],f16>) -> !torch.vtensor<[],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %int2 = torch.constant.int 2
    %int3 = torch.constant.int 3
    %0 = torch.prim.ListConstruct %int0, %int1, %int2, %int3 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %1 = torch.aten.amin %arg0, %0, %false : !torch.vtensor<[3,2,1,2],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[],f16>
    return %1 : !torch.vtensor<[],f16>
  }
  func.func @test_reduction_amin_cpu_float16_18(%arg0: !torch.vtensor<[3,2,1,2],f16>) -> !torch.vtensor<[1,2,1,1],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %int0 = torch.constant.int 0
    %int-1 = torch.constant.int -1
    %0 = torch.prim.ListConstruct %int0, %int-1 : (!torch.int, !torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %1 = torch.aten.amin %arg0, %0, %true : !torch.vtensor<[3,2,1,2],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,2,1,1],f16>
    return %1 : !torch.vtensor<[1,2,1,1],f16>
  }
  func.func @test_reduction_amin_cpu_float16_19(%arg0: !torch.vtensor<[3,2,1,2],f16>) -> !torch.vtensor<[3,1],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %int1 = torch.constant.int 1
    %int3 = torch.constant.int 3
    %0 = torch.prim.ListConstruct %int1, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %1 = torch.aten.amin %arg0, %0, %false : !torch.vtensor<[3,2,1,2],f16>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3,1],f16>
    return %1 : !torch.vtensor<[3,1],f16>
  }
}
