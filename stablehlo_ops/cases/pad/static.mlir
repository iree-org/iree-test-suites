module {
  func.func @main(%arg0: tensor<3xi32>, %arg1: tensor<i32>) -> tensor<8xi32> {
    %0 = stablehlo.pad %arg0, %arg1, low = [1], high = [2], interior = [1] : (tensor<3xi32>, tensor<i32>) -> tensor<8xi32>
    return %0 : tensor<8xi32>
  }
}

