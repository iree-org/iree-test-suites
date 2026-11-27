module {
  func.func @main(%arg0: tensor<?xi32>, %arg1: tensor<f32>) -> (tensor<?xi32>, tensor<f32>) {
    %0:2 = stablehlo.optimization_barrier %arg0, %arg1 : tensor<?xi32>, tensor<f32>
    return %0#0, %0#1 : tensor<?xi32>, tensor<f32>
  }
}

