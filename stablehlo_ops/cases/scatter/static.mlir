module {
  func.func @main(%arg0: tensor<3xi32>, %arg1: tensor<2x1xi32>, %arg2: tensor<2xi32>) -> tensor<3xi32> {
    %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
      %1 = stablehlo.add %arg3, %arg4 : tensor<i32>
      stablehlo.return %1 : tensor<i32>
    }) : (tensor<3xi32>, tensor<2x1xi32>, tensor<2xi32>) -> tensor<3xi32>
    return %0 : tensor<3xi32>
  }
}

