func.func @matmul_accumulate_DYNxDYNxi8_times_DYNxDYNxi8_into_DYNxDYNxi32(%lhs: tensor<?x?xi8>, %rhs: tensor<?x?xi8>, %acc: tensor<?x?xi32>) -> tensor<?x?xi32> {
  %result = linalg.matmul_transpose_b ins(%lhs, %rhs: tensor<?x?xi8>, tensor<?x?xi8>) outs(%acc: tensor<?x?xi32>) -> tensor<?x?xi32>

  return %result: tensor<?x?xi32>
}

func.func @matmul_accumulate_123x456xi8_times_789x456xi8_into_123x789xi32(%lhs: tensor<123x456xi8>, %rhs: tensor<789x456xi8>, %acc: tensor<123x789xi32>) -> tensor<123x789xi32> {
  %result = linalg.matmul_transpose_b ins(%lhs, %rhs: tensor<123x456xi8>, tensor<789x456xi8>) outs(%acc: tensor<123x789xi32>) -> tensor<123x789xi32>

  return %result: tensor<123x789xi32>
}

func.func @matmul_DYNxDYNxi8_times_DYNxDYNxi8_into_DYNxDYNxi32(%lhs: tensor<?x?xi8>, %rhs: tensor<?x?xi8>) -> tensor<?x?xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %acc_dim0 = tensor.dim %lhs, %c0 : tensor<?x?xi8>
  %acc_dim1 = tensor.dim %rhs, %c1 : tensor<?x?xi8>
  %init_acc = tensor.empty(%acc_dim0, %acc_dim1) : tensor<?x?xi32>
  %c0_acc_type = arith.constant 0: i32
  %acc = linalg.fill ins(%c0_acc_type : i32) outs(%init_acc : tensor<?x?xi32>) -> tensor<?x?xi32>
  %result = linalg.matmul_transpose_b ins(%lhs, %rhs: tensor<?x?xi8>, tensor<?x?xi8>) outs(%acc: tensor<?x?xi32>) -> tensor<?x?xi32>
  return %result: tensor<?x?xi32>
}

func.func @matmul_654x321xi8_times_234x321xi8_into_654x234xi32(%lhs: tensor<654x321xi8>, %rhs: tensor<234x321xi8>) -> tensor<654x234xi32> {
  %init_acc = tensor.empty() : tensor<654x234xi32>
  %c0_acc_type = arith.constant 0: i32
  %acc = linalg.fill ins(%c0_acc_type : i32) outs(%init_acc : tensor<654x234xi32>) -> tensor<654x234xi32>
  %result = linalg.matmul_transpose_b ins(%lhs, %rhs: tensor<654x321xi8>, tensor<234x321xi8>) outs(%acc: tensor<654x234xi32>) -> tensor<654x234xi32>
  return %result: tensor<654x234xi32>
}

func.func @matmul_accumulate_1x1000xi8_times_1000x1000xi8_into_1x1000xi32(%lhs: tensor<1x1000xi8>, %rhs: tensor<1000x1000xi8>, %acc: tensor<1x1000xi32>) -> tensor<1x1000xi32> {
  %result = linalg.matmul_transpose_b ins(%lhs, %rhs: tensor<1x1000xi8>, tensor<1000x1000xi8>) outs(%acc: tensor<1x1000xi32>) -> tensor<1x1000xi32>

  return %result: tensor<1x1000xi32>
}

func.func @matmul_accumulate_1000x1000xi8_times_1x1000xi8_into_1000x1xi32(%lhs: tensor<1000x1000xi8>, %rhs: tensor<1x1000xi8>, %acc: tensor<1000x1xi32>) -> tensor<1000x1xi32> {
  %result = linalg.matmul_transpose_b ins(%lhs, %rhs: tensor<1000x1000xi8>, tensor<1x1000xi8>) outs(%acc: tensor<1000x1xi32>) -> tensor<1000x1xi32>

  return %result: tensor<1000x1xi32>
}

func.func @matmul_1000x1000xi8_times_1x1000xi8_into_1000x1xi32(%lhs: tensor<1000x1000xi8>, %rhs: tensor<1x1000xi8>) -> tensor<1000x1xi32> {
  %init_acc = tensor.empty() : tensor<1000x1xi32>
  %c0_acc_type = arith.constant 0: i32
  %acc = linalg.fill ins(%c0_acc_type : i32) outs(%init_acc : tensor<1000x1xi32>) -> tensor<1000x1xi32>
  %result = linalg.matmul_transpose_b ins(%lhs, %rhs: tensor<1000x1000xi8>, tensor<1x1000xi8>) outs(%acc: tensor<1000x1xi32>) -> tensor<1000x1xi32>
  return %result: tensor<1000x1xi32>
}

