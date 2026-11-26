module {
  func.func @main(%a: tensor<?xf32>, %b: tensor<?xf32>) -> tensor<?xf32> {
    %result = stablehlo.add %a, %b : tensor<?xf32>
    return %result : tensor<?xf32>
  }
}
