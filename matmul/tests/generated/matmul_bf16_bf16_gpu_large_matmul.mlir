func.func @matmul_457x330xbf16_times_330x512xbf16_into_457x512xbf16(%lhs: tensor<457x330xbf16>, %rhs: tensor<330x512xbf16>) -> tensor<457x512xbf16> {
  %init_acc = tensor.empty() : tensor<457x512xbf16>
  %c0_acc_type = arith.constant 0.0: bf16
  %acc = linalg.fill ins(%c0_acc_type : bf16) outs(%init_acc : tensor<457x512xbf16>) -> tensor<457x512xbf16>
  %result = linalg.matmul ins(%lhs, %rhs: tensor<457x330xbf16>, tensor<330x512xbf16>) outs(%acc: tensor<457x512xbf16>) -> tensor<457x512xbf16>
  return %result: tensor<457x512xbf16>
}

func.func @matmul_457x330xbf16_times_330x514xbf16_into_457x514xbf16(%lhs: tensor<457x330xbf16>, %rhs: tensor<330x514xbf16>) -> tensor<457x514xbf16> {
  %init_acc = tensor.empty() : tensor<457x514xbf16>
  %c0_acc_type = arith.constant 0.0: bf16
  %acc = linalg.fill ins(%c0_acc_type : bf16) outs(%init_acc : tensor<457x514xbf16>) -> tensor<457x514xbf16>
  %result = linalg.matmul ins(%lhs, %rhs: tensor<457x330xbf16>, tensor<330x514xbf16>) outs(%acc: tensor<457x514xbf16>) -> tensor<457x514xbf16>
  return %result: tensor<457x514xbf16>
}

func.func @matmul_438x330xbf16_times_330x514xbf16_into_438x514xbf16(%lhs: tensor<438x330xbf16>, %rhs: tensor<330x514xbf16>) -> tensor<438x514xbf16> {
  %init_acc = tensor.empty() : tensor<438x514xbf16>
  %c0_acc_type = arith.constant 0.0: bf16
  %acc = linalg.fill ins(%c0_acc_type : bf16) outs(%init_acc : tensor<438x514xbf16>) -> tensor<438x514xbf16>
  %result = linalg.matmul ins(%lhs, %rhs: tensor<438x330xbf16>, tensor<330x514xbf16>) outs(%acc: tensor<438x514xbf16>) -> tensor<438x514xbf16>
  return %result: tensor<438x514xbf16>
}

func.func @matmul_540x332xbf16_times_332x516xbf16_into_540x516xbf16(%lhs: tensor<540x332xbf16>, %rhs: tensor<332x516xbf16>) -> tensor<540x516xbf16> {
  %init_acc = tensor.empty() : tensor<540x516xbf16>
  %c0_acc_type = arith.constant 0.0: bf16
  %acc = linalg.fill ins(%c0_acc_type : bf16) outs(%init_acc : tensor<540x516xbf16>) -> tensor<540x516xbf16>
  %result = linalg.matmul ins(%lhs, %rhs: tensor<540x332xbf16>, tensor<332x516xbf16>) outs(%acc: tensor<540x516xbf16>) -> tensor<540x516xbf16>
  return %result: tensor<540x516xbf16>
}

func.func @matmul_1000x4xbf16_times_4x512xbf16_into_1000x512xbf16(%lhs: tensor<1000x4xbf16>, %rhs: tensor<4x512xbf16>) -> tensor<1000x512xbf16> {
  %init_acc = tensor.empty() : tensor<1000x512xbf16>
  %c0_acc_type = arith.constant 0.0: bf16
  %acc = linalg.fill ins(%c0_acc_type : bf16) outs(%init_acc : tensor<1000x512xbf16>) -> tensor<1000x512xbf16>
  %result = linalg.matmul ins(%lhs, %rhs: tensor<1000x4xbf16>, tensor<4x512xbf16>) outs(%acc: tensor<1000x512xbf16>) -> tensor<1000x512xbf16>
  return %result: tensor<1000x512xbf16>
}

func.func @matmul_4x1000xbf16_times_1000x512xbf16_into_4x512xbf16(%lhs: tensor<4x1000xbf16>, %rhs: tensor<1000x512xbf16>) -> tensor<4x512xbf16> {
  %init_acc = tensor.empty() : tensor<4x512xbf16>
  %c0_acc_type = arith.constant 0.0: bf16
  %acc = linalg.fill ins(%c0_acc_type : bf16) outs(%init_acc : tensor<4x512xbf16>) -> tensor<4x512xbf16>
  %result = linalg.matmul ins(%lhs, %rhs: tensor<4x1000xbf16>, tensor<1000x512xbf16>) outs(%acc: tensor<4x512xbf16>) -> tensor<4x512xbf16>
  return %result: tensor<4x512xbf16>
}

func.func @matmul_512x1000xbf16_times_1000x4xbf16_into_512x4xbf16(%lhs: tensor<512x1000xbf16>, %rhs: tensor<1000x4xbf16>) -> tensor<512x4xbf16> {
  %init_acc = tensor.empty() : tensor<512x4xbf16>
  %c0_acc_type = arith.constant 0.0: bf16
  %acc = linalg.fill ins(%c0_acc_type : bf16) outs(%init_acc : tensor<512x4xbf16>) -> tensor<512x4xbf16>
  %result = linalg.matmul ins(%lhs, %rhs: tensor<512x1000xbf16>, tensor<1000x4xbf16>) outs(%acc: tensor<512x4xbf16>) -> tensor<512x4xbf16>
  return %result: tensor<512x4xbf16>
}

func.func @matmul_512x128xbf16_times_128x500xbf16_into_512x500xbf16(%lhs: tensor<512x128xbf16>, %rhs: tensor<128x500xbf16>) -> tensor<512x500xbf16> {
  %init_acc = tensor.empty() : tensor<512x500xbf16>
  %c0_acc_type = arith.constant 0.0: bf16
  %acc = linalg.fill ins(%c0_acc_type : bf16) outs(%init_acc : tensor<512x500xbf16>) -> tensor<512x500xbf16>
  %result = linalg.matmul ins(%lhs, %rhs: tensor<512x128xbf16>, tensor<128x500xbf16>) outs(%acc: tensor<512x500xbf16>) -> tensor<512x500xbf16>
  return %result: tensor<512x500xbf16>
}

func.func @matmul_457x160xbf16_times_160x512xbf16_into_457x512xbf16(%lhs: tensor<457x160xbf16>, %rhs: tensor<160x512xbf16>) -> tensor<457x512xbf16> {
  %init_acc = tensor.empty() : tensor<457x512xbf16>
  %c0_acc_type = arith.constant 0.0: bf16
  %acc = linalg.fill ins(%c0_acc_type : bf16) outs(%init_acc : tensor<457x512xbf16>) -> tensor<457x512xbf16>
  %result = linalg.matmul ins(%lhs, %rhs: tensor<457x160xbf16>, tensor<160x512xbf16>) outs(%acc: tensor<457x512xbf16>) -> tensor<457x512xbf16>
  return %result: tensor<457x512xbf16>
}

func.func @matmul_512x330xbf16_times_330x512xbf16_into_512x512xbf16(%lhs: tensor<512x330xbf16>, %rhs: tensor<330x512xbf16>) -> tensor<512x512xbf16> {
  %init_acc = tensor.empty() : tensor<512x512xbf16>
  %c0_acc_type = arith.constant 0.0: bf16
  %acc = linalg.fill ins(%c0_acc_type : bf16) outs(%init_acc : tensor<512x512xbf16>) -> tensor<512x512xbf16>
  %result = linalg.matmul ins(%lhs, %rhs: tensor<512x330xbf16>, tensor<330x512xbf16>) outs(%acc: tensor<512x512xbf16>) -> tensor<512x512xbf16>
  return %result: tensor<512x512xbf16>
}

