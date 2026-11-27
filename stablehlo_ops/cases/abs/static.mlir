module {
  func.func @main(%a: tensor<10xf32>) -> tensor<10xf32> {
    %result = stablehlo.abs %a : tensor<10xf32>
    return %result : tensor<10xf32>
  }
}
