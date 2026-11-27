module {
  func.func @main(%a: tensor<10xf32>, %b: tensor<10xf32>) -> tensor<10xf32> {
    %result = stablehlo.subtract %a, %b : tensor<10xf32>
    return %result : tensor<10xf32>
  }
}
