module {
  func.func @main(%arg0: tensor<2xi1>, %arg1: tensor<2xi32>, %arg2: tensor<2xi32>) -> tensor<2xi32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<2xi1>, tensor<2xi32>
    return %0 : tensor<2xi32>
  }
}

