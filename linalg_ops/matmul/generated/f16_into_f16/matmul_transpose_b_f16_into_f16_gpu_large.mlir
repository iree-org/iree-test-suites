func.func @matmul_457x330xf16_times_512x330xf16_into_457x512xf16(%lhs: tensor<457x330xf16>, %rhs: tensor<512x330xf16>) -> tensor<457x512xf16> {
  %init_acc = tensor.empty() : tensor<457x512xf16>
  %c0_acc_type = arith.constant 0.0: f16
  %acc = linalg.fill ins(%c0_acc_type : f16) outs(%init_acc : tensor<457x512xf16>) -> tensor<457x512xf16>
  %result = linalg.matmul_transpose_b ins(%lhs, %rhs: tensor<457x330xf16>, tensor<512x330xf16>) outs(%acc: tensor<457x512xf16>) -> tensor<457x512xf16>
  return %result: tensor<457x512xf16>
}

func.func @matmul_457x330xf16_times_514x330xf16_into_457x514xf16(%lhs: tensor<457x330xf16>, %rhs: tensor<514x330xf16>) -> tensor<457x514xf16> {
  %init_acc = tensor.empty() : tensor<457x514xf16>
  %c0_acc_type = arith.constant 0.0: f16
  %acc = linalg.fill ins(%c0_acc_type : f16) outs(%init_acc : tensor<457x514xf16>) -> tensor<457x514xf16>
  %result = linalg.matmul_transpose_b ins(%lhs, %rhs: tensor<457x330xf16>, tensor<514x330xf16>) outs(%acc: tensor<457x514xf16>) -> tensor<457x514xf16>
  return %result: tensor<457x514xf16>
}

func.func @matmul_438x330xf16_times_514x330xf16_into_438x514xf16(%lhs: tensor<438x330xf16>, %rhs: tensor<514x330xf16>) -> tensor<438x514xf16> {
  %init_acc = tensor.empty() : tensor<438x514xf16>
  %c0_acc_type = arith.constant 0.0: f16
  %acc = linalg.fill ins(%c0_acc_type : f16) outs(%init_acc : tensor<438x514xf16>) -> tensor<438x514xf16>
  %result = linalg.matmul_transpose_b ins(%lhs, %rhs: tensor<438x330xf16>, tensor<514x330xf16>) outs(%acc: tensor<438x514xf16>) -> tensor<438x514xf16>
  return %result: tensor<438x514xf16>
}

func.func @matmul_540x332xf16_times_516x332xf16_into_540x516xf16(%lhs: tensor<540x332xf16>, %rhs: tensor<516x332xf16>) -> tensor<540x516xf16> {
  %init_acc = tensor.empty() : tensor<540x516xf16>
  %c0_acc_type = arith.constant 0.0: f16
  %acc = linalg.fill ins(%c0_acc_type : f16) outs(%init_acc : tensor<540x516xf16>) -> tensor<540x516xf16>
  %result = linalg.matmul_transpose_b ins(%lhs, %rhs: tensor<540x332xf16>, tensor<516x332xf16>) outs(%acc: tensor<540x516xf16>) -> tensor<540x516xf16>
  return %result: tensor<540x516xf16>
}

func.func @matmul_1000x4xf16_times_512x4xf16_into_1000x512xf16(%lhs: tensor<1000x4xf16>, %rhs: tensor<512x4xf16>) -> tensor<1000x512xf16> {
  %init_acc = tensor.empty() : tensor<1000x512xf16>
  %c0_acc_type = arith.constant 0.0: f16
  %acc = linalg.fill ins(%c0_acc_type : f16) outs(%init_acc : tensor<1000x512xf16>) -> tensor<1000x512xf16>
  %result = linalg.matmul_transpose_b ins(%lhs, %rhs: tensor<1000x4xf16>, tensor<512x4xf16>) outs(%acc: tensor<1000x512xf16>) -> tensor<1000x512xf16>
  return %result: tensor<1000x512xf16>
}

func.func @matmul_4x1000xf16_times_512x1000xf16_into_4x512xf16(%lhs: tensor<4x1000xf16>, %rhs: tensor<512x1000xf16>) -> tensor<4x512xf16> {
  %init_acc = tensor.empty() : tensor<4x512xf16>
  %c0_acc_type = arith.constant 0.0: f16
  %acc = linalg.fill ins(%c0_acc_type : f16) outs(%init_acc : tensor<4x512xf16>) -> tensor<4x512xf16>
  %result = linalg.matmul_transpose_b ins(%lhs, %rhs: tensor<4x1000xf16>, tensor<512x1000xf16>) outs(%acc: tensor<4x512xf16>) -> tensor<4x512xf16>
  return %result: tensor<4x512xf16>
}

func.func @matmul_512x1000xf16_times_4x1000xf16_into_512x4xf16(%lhs: tensor<512x1000xf16>, %rhs: tensor<4x1000xf16>) -> tensor<512x4xf16> {
  %init_acc = tensor.empty() : tensor<512x4xf16>
  %c0_acc_type = arith.constant 0.0: f16
  %acc = linalg.fill ins(%c0_acc_type : f16) outs(%init_acc : tensor<512x4xf16>) -> tensor<512x4xf16>
  %result = linalg.matmul_transpose_b ins(%lhs, %rhs: tensor<512x1000xf16>, tensor<4x1000xf16>) outs(%acc: tensor<512x4xf16>) -> tensor<512x4xf16>
  return %result: tensor<512x4xf16>
}

func.func @matmul_512x128xf16_times_500x128xf16_into_512x500xf16(%lhs: tensor<512x128xf16>, %rhs: tensor<500x128xf16>) -> tensor<512x500xf16> {
  %init_acc = tensor.empty() : tensor<512x500xf16>
  %c0_acc_type = arith.constant 0.0: f16
  %acc = linalg.fill ins(%c0_acc_type : f16) outs(%init_acc : tensor<512x500xf16>) -> tensor<512x500xf16>
  %result = linalg.matmul_transpose_b ins(%lhs, %rhs: tensor<512x128xf16>, tensor<500x128xf16>) outs(%acc: tensor<512x500xf16>) -> tensor<512x500xf16>
  return %result: tensor<512x500xf16>
}

func.func @matmul_457x160xf16_times_512x160xf16_into_457x512xf16(%lhs: tensor<457x160xf16>, %rhs: tensor<512x160xf16>) -> tensor<457x512xf16> {
  %init_acc = tensor.empty() : tensor<457x512xf16>
  %c0_acc_type = arith.constant 0.0: f16
  %acc = linalg.fill ins(%c0_acc_type : f16) outs(%init_acc : tensor<457x512xf16>) -> tensor<457x512xf16>
  %result = linalg.matmul_transpose_b ins(%lhs, %rhs: tensor<457x160xf16>, tensor<512x160xf16>) outs(%acc: tensor<457x512xf16>) -> tensor<457x512xf16>
  return %result: tensor<457x512xf16>
}

func.func @matmul_512x330xf16_times_512x330xf16_into_512x512xf16(%lhs: tensor<512x330xf16>, %rhs: tensor<512x330xf16>) -> tensor<512x512xf16> {
  %init_acc = tensor.empty() : tensor<512x512xf16>
  %c0_acc_type = arith.constant 0.0: f16
  %acc = linalg.fill ins(%c0_acc_type : f16) outs(%init_acc : tensor<512x512xf16>) -> tensor<512x512xf16>
  %result = linalg.matmul_transpose_b ins(%lhs, %rhs: tensor<512x330xf16>, tensor<512x330xf16>) outs(%acc: tensor<512x512xf16>) -> tensor<512x512xf16>
  return %result: tensor<512x512xf16>
}

