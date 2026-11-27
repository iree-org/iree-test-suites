module {
  func.func @main(%input: tensor<10xf32>, %start: tensor<1xi64>, %limit: tensor<1xi64>, %stride: tensor<1xi64>) -> tensor<3xf32> {
    %slice = stablehlo.real_dynamic_slice %input, %start, %limit, %stride : (tensor<10xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xf32>
    return %slice : tensor<3xf32>
  }
}
