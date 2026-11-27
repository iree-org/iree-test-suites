module {
  func.func @main(%input: tensor<?xi32>, %size: tensor<i32>) -> tensor<?xi32> {
    %result = stablehlo.set_dimension_size %input, %size, dim = 0 : (tensor<?xi32>, tensor<i32>) -> tensor<?xi32>
    return %result : tensor<?xi32>
  }
}
