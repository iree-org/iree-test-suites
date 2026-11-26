module {
  func.func @main(%a: tensor<?xf32>) -> tensor<?xf32> {
    %result = stablehlo.abs %a : tensor<?xf32>
    return %result : tensor<?xf32>
  }
}
