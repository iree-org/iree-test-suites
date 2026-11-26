module {
  func.func @main(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> (tensor<4xi32>, tensor<4xi32>) {
    %0:2 = "stablehlo.sort"(%arg0, %arg1) <{dimension = 0 : i64, is_stable = true}> ({
    ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>, %arg4: tensor<i32>, %arg5: tensor<i32>):
      %1 = stablehlo.compare  LT, %arg2, %arg3 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    }) : (tensor<4xi32>, tensor<4xi32>) -> (tensor<4xi32>, tensor<4xi32>)
    return %0#0, %0#1 : tensor<4xi32>, tensor<4xi32>
  }
}

