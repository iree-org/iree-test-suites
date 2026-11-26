module {
  func.func @main(%arg0: tensor<3xi32>, %arg1: tensor<2xi32>) -> tensor<5xi32> {
    %0 = stablehlo.concatenate %arg0, %arg1, dim = 0 : (tensor<3xi32>, tensor<2xi32>) -> tensor<5xi32>
    return %0 : tensor<5xi32>
  }
}

