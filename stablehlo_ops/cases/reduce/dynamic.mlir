module {
  func.func @main(%arg0: tensor<?x?xi32>, %arg1: tensor<i32>) -> tensor<?xi32> {
    %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [1] : (tensor<?x?xi32>, tensor<i32>) -> tensor<?xi32>
    return %0 : tensor<?xi32>
  }
}

