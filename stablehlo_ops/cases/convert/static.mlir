module {
  func.func @main(%arg0: tensor<3xf32>) -> tensor<3xi32> {
    %0 = stablehlo.convert %arg0 : (tensor<3xf32>) -> tensor<3xi32>
    return %0 : tensor<3xi32>
  }
}

