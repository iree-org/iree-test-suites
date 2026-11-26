// Static reference for feature_index = 1: (x - mean) / sqrt(variance + epsilon) * scale + offset.
module {
  func.func @main(%x: tensor<2x2xf32>, %scale: tensor<2xf32>, %offset: tensor<2xf32>, %mean: tensor<2xf32>, %variance: tensor<2xf32>) -> tensor<2x2xf32> {
    %epsilon = stablehlo.constant dense<1.000000e-03> : tensor<2xf32>
    %adjusted = stablehlo.add %variance, %epsilon : tensor<2xf32>
    %stddev = stablehlo.sqrt %adjusted : tensor<2xf32>
    %means = stablehlo.broadcast_in_dim %mean, dims = [1] : (tensor<2xf32>) -> tensor<2x2xf32>
    %stddevs = stablehlo.broadcast_in_dim %stddev, dims = [1] : (tensor<2xf32>) -> tensor<2x2xf32>
    %scales = stablehlo.broadcast_in_dim %scale, dims = [1] : (tensor<2xf32>) -> tensor<2x2xf32>
    %offsets = stablehlo.broadcast_in_dim %offset, dims = [1] : (tensor<2xf32>) -> tensor<2x2xf32>
    %centered = stablehlo.subtract %x, %means : tensor<2x2xf32>
    %normalized = stablehlo.divide %centered, %stddevs : tensor<2x2xf32>
    %scaled = stablehlo.multiply %normalized, %scales : tensor<2x2xf32>
    %result = stablehlo.add %scaled, %offsets : tensor<2x2xf32>
    return %result : tensor<2x2xf32>
  }
}
