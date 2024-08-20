func.func @matmul_accumulate_512x128xbf16_times_128x512xbf16_into_512x512xbf16(%lhs: tensor<512x128xbf16>, %rhs: tensor<128x512xbf16>, %acc: tensor<512x512xbf16>) -> tensor<512x512xbf16> {
  %result = linalg.matmul ins(%lhs, %rhs: tensor<512x128xbf16>, tensor<128x512xbf16>) outs(%acc: tensor<512x512xbf16>) -> tensor<512x512xbf16>
  return %result: tensor<512x512xbf16>
}

func.func @matmul_512x128xbf16_times_128x512xbf16_into_512x512xbf16(%lhs: tensor<512x128xbf16>, %rhs: tensor<128x512xbf16>) -> tensor<512x512xbf16> {
  %init_acc = tensor.empty() : tensor<512x512xbf16>
  %c0_acc_type = arith.constant 0.0: bf16
  %acc = linalg.fill ins(%c0_acc_type : bf16) outs(%init_acc : tensor<512x512xbf16>) -> tensor<512x512xbf16>
  %result = linalg.matmul ins(%lhs, %rhs: tensor<512x128xbf16>, tensor<128x512xbf16>) outs(%acc: tensor<512x512xbf16>) -> tensor<512x512xbf16>
  return %result: tensor<512x512xbf16>
}

