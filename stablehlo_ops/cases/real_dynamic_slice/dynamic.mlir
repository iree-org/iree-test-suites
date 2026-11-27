module {
  func.func @main(%input: tensor<?xf32>, %start: tensor<1xi64>, %limit: tensor<1xi64>, %stride: tensor<1xi64>) -> tensor<?xf32> {
    %slice = stablehlo.real_dynamic_slice %input, %start, %limit, %stride : (tensor<?xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xf32>
    return %slice : tensor<?xf32>
  }
}
