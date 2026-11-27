module {
  func.func @main(%input: tensor<10xf32>, %start: tensor<1xi64>, %limit: tensor<1xi64>, %stride: tensor<1xi64>) -> tensor<3xf32> {
    // The manifest supplies start [1], limit [6], and stride [2].
    %slice = stablehlo.slice %input [1:6:2] : (tensor<10xf32>) -> tensor<3xf32>
    return %slice : tensor<3xf32>
  }
}
