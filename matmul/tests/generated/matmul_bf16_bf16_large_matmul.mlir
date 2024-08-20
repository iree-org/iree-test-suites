func.func @matmul_accumulate_DYNxDYNxbf16_times_DYNxDYNxbf16_into_DYNxDYNxbf16(%lhs: tensor<?x?xbf16>, %rhs: tensor<?x?xbf16>, %acc: tensor<?x?xbf16>) -> tensor<?x?xbf16> {
  %result = linalg.matmul ins(%lhs, %rhs: tensor<?x?xbf16>, tensor<?x?xbf16>) outs(%acc: tensor<?x?xbf16>) -> tensor<?x?xbf16>
  return %result: tensor<?x?xbf16>
}

func.func @matmul_accumulate_123x456xbf16_times_456x789xbf16_into_123x789xbf16(%lhs: tensor<123x456xbf16>, %rhs: tensor<456x789xbf16>, %acc: tensor<123x789xbf16>) -> tensor<123x789xbf16> {
  %result = linalg.matmul ins(%lhs, %rhs: tensor<123x456xbf16>, tensor<456x789xbf16>) outs(%acc: tensor<123x789xbf16>) -> tensor<123x789xbf16>
  return %result: tensor<123x789xbf16>
}

func.func @matmul_DYNxDYNxbf16_times_DYNxDYNxbf16_into_DYNxDYNxbf16(%lhs: tensor<?x?xbf16>, %rhs: tensor<?x?xbf16>) -> tensor<?x?xbf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %acc_dim0 = tensor.dim %lhs, %c0 : tensor<?x?xbf16>
  %acc_dim1 = tensor.dim %rhs, %c1 : tensor<?x?xbf16>
  %init_acc = tensor.empty(%acc_dim0, %acc_dim1) : tensor<?x?xbf16>
  %c0_acc_type = arith.constant 0.0: bf16
  %acc = linalg.fill ins(%c0_acc_type : bf16) outs(%init_acc : tensor<?x?xbf16>) -> tensor<?x?xbf16>
  %result = linalg.matmul ins(%lhs, %rhs: tensor<?x?xbf16>, tensor<?x?xbf16>) outs(%acc: tensor<?x?xbf16>) -> tensor<?x?xbf16>
  return %result: tensor<?x?xbf16>
}

func.func @matmul_654x321xbf16_times_321x234xbf16_into_654x234xbf16(%lhs: tensor<654x321xbf16>, %rhs: tensor<321x234xbf16>) -> tensor<654x234xbf16> {
  %init_acc = tensor.empty() : tensor<654x234xbf16>
  %c0_acc_type = arith.constant 0.0: bf16
  %acc = linalg.fill ins(%c0_acc_type : bf16) outs(%init_acc : tensor<654x234xbf16>) -> tensor<654x234xbf16>
  %result = linalg.matmul ins(%lhs, %rhs: tensor<654x321xbf16>, tensor<321x234xbf16>) outs(%acc: tensor<654x234xbf16>) -> tensor<654x234xbf16>
  return %result: tensor<654x234xbf16>
}

func.func @matmul_accumulate_1x1000xbf16_times_1000x1000xbf16_into_1x1000xbf16(%lhs: tensor<1x1000xbf16>, %rhs: tensor<1000x1000xbf16>, %acc: tensor<1x1000xbf16>) -> tensor<1x1000xbf16> {
  %result = linalg.matmul ins(%lhs, %rhs: tensor<1x1000xbf16>, tensor<1000x1000xbf16>) outs(%acc: tensor<1x1000xbf16>) -> tensor<1x1000xbf16>
  return %result: tensor<1x1000xbf16>
}

func.func @matmul_accumulate_1000x1000xbf16_times_1000x1xbf16_into_1000x1xbf16(%lhs: tensor<1000x1000xbf16>, %rhs: tensor<1000x1xbf16>, %acc: tensor<1000x1xbf16>) -> tensor<1000x1xbf16> {
  %result = linalg.matmul ins(%lhs, %rhs: tensor<1000x1000xbf16>, tensor<1000x1xbf16>) outs(%acc: tensor<1000x1xbf16>) -> tensor<1000x1xbf16>
  return %result: tensor<1000x1xbf16>
}

func.func @matmul_1000x1000xbf16_times_1000x1xbf16_into_1000x1xbf16(%lhs: tensor<1000x1000xbf16>, %rhs: tensor<1000x1xbf16>) -> tensor<1000x1xbf16> {
  %init_acc = tensor.empty() : tensor<1000x1xbf16>
  %c0_acc_type = arith.constant 0.0: bf16
  %acc = linalg.fill ins(%c0_acc_type : bf16) outs(%init_acc : tensor<1000x1xbf16>) -> tensor<1000x1xbf16>
  %result = linalg.matmul ins(%lhs, %rhs: tensor<1000x1000xbf16>, tensor<1000x1xbf16>) outs(%acc: tensor<1000x1xbf16>) -> tensor<1000x1xbf16>
  return %result: tensor<1000x1xbf16>
}

