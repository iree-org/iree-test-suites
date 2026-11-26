module {
  func.func @main(%arg0: tensor<2x3xi32>) -> tensor<i32> {
    %0 = stablehlo.get_dimension_size %arg0, dim = 1 : (tensor<2x3xi32>) -> tensor<i32>
    return %0 : tensor<i32>
  }
}

