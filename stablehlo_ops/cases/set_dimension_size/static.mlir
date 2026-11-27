module {
  func.func @main(%input: tensor<3xi32>, %size: tensor<i32>) -> tensor<?xi32, #stablehlo.bounds<3>> {
    %result = stablehlo.set_dimension_size %input, %size, dim = 0 : (tensor<3xi32>, tensor<i32>) -> tensor<?xi32, #stablehlo.bounds<3>>
    return %result : tensor<?xi32, #stablehlo.bounds<3>>
  }
}
