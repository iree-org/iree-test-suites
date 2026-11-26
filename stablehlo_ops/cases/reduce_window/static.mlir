module {
  func.func @main(%arg0: tensor<4xi32>, %arg1: tensor<i32>) -> tensor<3xi32> {
    %0 = "stablehlo.reduce_window"(%arg0, %arg1) <{window_dimensions = array<i64: 2>, window_strides = array<i64: 1>}> ({
    ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>):
      %1 = stablehlo.add %arg2, %arg3 : tensor<i32>
      stablehlo.return %1 : tensor<i32>
    }) : (tensor<4xi32>, tensor<i32>) -> tensor<3xi32>
    return %0 : tensor<3xi32>
  }
}

