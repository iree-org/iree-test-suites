module {
  func.func @main(%arg0: tensor<?xi32>, %arg1: tensor<2x1xi32>, %arg2: tensor<1xi64>) -> tensor<2xi32> {
    %0 = "stablehlo.dynamic_gather"(%arg0, %arg1, %arg2) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>}> : (tensor<?xi32>, tensor<2x1xi32>, tensor<1xi64>) -> tensor<2xi32>
    return %0 : tensor<2xi32>
  }
}

