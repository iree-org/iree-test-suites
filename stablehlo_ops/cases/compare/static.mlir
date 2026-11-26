module {
  func.func @main(%arg0: tensor<3xi32>, %arg1: tensor<3xi32>) -> tensor<3xi1> {
    %0 = stablehlo.compare  LT, %arg0, %arg1 : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi1>
    return %0 : tensor<3xi1>
  }
}

