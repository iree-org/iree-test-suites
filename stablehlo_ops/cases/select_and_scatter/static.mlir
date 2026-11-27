module {
  func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<2xf32>, %arg2: tensor<f32>) -> tensor<4xf32> {
    %0 = "stablehlo.select_and_scatter"(%arg0, %arg1, %arg2) <{window_dimensions = array<i64: 2>, window_strides = array<i64: 2>}> ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %1 = stablehlo.compare  GE, %arg3, %arg4 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    }, {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %1 = stablehlo.add %arg3, %arg4 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<4xf32>, tensor<2xf32>, tensor<f32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}

