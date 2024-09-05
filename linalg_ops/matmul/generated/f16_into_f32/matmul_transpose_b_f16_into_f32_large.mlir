func.func @matmul_accumulate_DYNxDYNxf16_times_DYNxDYNxf16_into_DYNxDYNxf32(%lhs: tensor<?x?xf16>, %rhs: tensor<?x?xf16>, %acc: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %result = linalg.matmul_transpose_b ins(%lhs, %rhs: tensor<?x?xf16>, tensor<?x?xf16>) outs(%acc: tensor<?x?xf32>) -> tensor<?x?xf32>

  return %result: tensor<?x?xf32>
}

func.func @matmul_accumulate_123x456xf16_times_789x456xf16_into_123x789xf32(%lhs: tensor<123x456xf16>, %rhs: tensor<789x456xf16>, %acc: tensor<123x789xf32>) -> tensor<123x789xf32> {
  %result = linalg.matmul_transpose_b ins(%lhs, %rhs: tensor<123x456xf16>, tensor<789x456xf16>) outs(%acc: tensor<123x789xf32>) -> tensor<123x789xf32>

  return %result: tensor<123x789xf32>
}

func.func @matmul_DYNxDYNxf16_times_DYNxDYNxf16_into_DYNxDYNxf32(%lhs: tensor<?x?xf16>, %rhs: tensor<?x?xf16>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %acc_dim0 = tensor.dim %lhs, %c0 : tensor<?x?xf16>
  %acc_dim1 = tensor.dim %rhs, %c1 : tensor<?x?xf16>
  %init_acc = tensor.empty(%acc_dim0, %acc_dim1) : tensor<?x?xf32>
  %c0_acc_type = arith.constant 0.0: f32
  %acc = linalg.fill ins(%c0_acc_type : f32) outs(%init_acc : tensor<?x?xf32>) -> tensor<?x?xf32>
  %result = linalg.matmul_transpose_b ins(%lhs, %rhs: tensor<?x?xf16>, tensor<?x?xf16>) outs(%acc: tensor<?x?xf32>) -> tensor<?x?xf32>
  return %result: tensor<?x?xf32>
}

func.func @matmul_654x321xf16_times_234x321xf16_into_654x234xf32(%lhs: tensor<654x321xf16>, %rhs: tensor<234x321xf16>) -> tensor<654x234xf32> {
  %init_acc = tensor.empty() : tensor<654x234xf32>
  %c0_acc_type = arith.constant 0.0: f32
  %acc = linalg.fill ins(%c0_acc_type : f32) outs(%init_acc : tensor<654x234xf32>) -> tensor<654x234xf32>
  %result = linalg.matmul_transpose_b ins(%lhs, %rhs: tensor<654x321xf16>, tensor<234x321xf16>) outs(%acc: tensor<654x234xf32>) -> tensor<654x234xf32>
  return %result: tensor<654x234xf32>
}

func.func @matmul_accumulate_1x1000xf16_times_1000x1000xf16_into_1x1000xf32(%lhs: tensor<1x1000xf16>, %rhs: tensor<1000x1000xf16>, %acc: tensor<1x1000xf32>) -> tensor<1x1000xf32> {
  %result = linalg.matmul_transpose_b ins(%lhs, %rhs: tensor<1x1000xf16>, tensor<1000x1000xf16>) outs(%acc: tensor<1x1000xf32>) -> tensor<1x1000xf32>

  return %result: tensor<1x1000xf32>
}

func.func @matmul_accumulate_1000x1000xf16_times_1x1000xf16_into_1000x1xf32(%lhs: tensor<1000x1000xf16>, %rhs: tensor<1x1000xf16>, %acc: tensor<1000x1xf32>) -> tensor<1000x1xf32> {
  %result = linalg.matmul_transpose_b ins(%lhs, %rhs: tensor<1000x1000xf16>, tensor<1x1000xf16>) outs(%acc: tensor<1000x1xf32>) -> tensor<1000x1xf32>

  return %result: tensor<1000x1xf32>
}

func.func @matmul_1000x1000xf16_times_1x1000xf16_into_1000x1xf32(%lhs: tensor<1000x1000xf16>, %rhs: tensor<1x1000xf16>) -> tensor<1000x1xf32> {
  %init_acc = tensor.empty() : tensor<1000x1xf32>
  %c0_acc_type = arith.constant 0.0: f32
  %acc = linalg.fill ins(%c0_acc_type : f32) outs(%init_acc : tensor<1000x1xf32>) -> tensor<1000x1xf32>
  %result = linalg.matmul_transpose_b ins(%lhs, %rhs: tensor<1000x1000xf16>, tensor<1x1000xf16>) outs(%acc: tensor<1000x1xf32>) -> tensor<1000x1xf32>
  return %result: tensor<1000x1xf32>
}

