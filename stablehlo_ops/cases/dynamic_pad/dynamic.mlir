module {
  func.func @main(%arg0: tensor<?xi32>, %arg1: tensor<i32>, %arg2: tensor<1xi64>, %arg3: tensor<1xi64>, %arg4: tensor<1xi64>) -> tensor<?xi32> {
    %0 = stablehlo.dynamic_pad %arg0, %arg1, %arg2, %arg3, %arg4 : (tensor<?xi32>, tensor<i32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xi32>
    return %0 : tensor<?xi32>
  }
}

