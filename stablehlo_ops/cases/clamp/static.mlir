module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<3xf32>, %arg2: tensor<f32>) -> tensor<3xf32> {
    %0 = stablehlo.clamp %arg0, %arg1, %arg2 : (tensor<f32>, tensor<3xf32>, tensor<f32>) -> tensor<3xf32>
    return %0 : tensor<3xf32>
  }
}

