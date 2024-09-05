func.func @matmul_accumulate_DYNxDYNxbf16_times_DYNxDYNxbf16_into_DYNxDYNxbf16(%lhs: tensor<?x?xbf16>, %rhs: tensor<?x?xbf16>, %acc: tensor<?x?xbf16>) -> tensor<?x?xbf16> {
  %result = linalg.matmul_transpose_b ins(%lhs, %rhs: tensor<?x?xbf16>, tensor<?x?xbf16>) outs(%acc: tensor<?x?xbf16>) -> tensor<?x?xbf16>
  return %result: tensor<?x?xbf16>
}

func.func @matmul_accumulate_1x1xbf16_times_1x1xbf16_into_1x1xbf16(%lhs: tensor<1x1xbf16>, %rhs: tensor<1x1xbf16>, %acc: tensor<1x1xbf16>) -> tensor<1x1xbf16> {
  %result = linalg.matmul_transpose_b ins(%lhs, %rhs: tensor<1x1xbf16>, tensor<1x1xbf16>) outs(%acc: tensor<1x1xbf16>) -> tensor<1x1xbf16>
  return %result: tensor<1x1xbf16>
}

func.func @matmul_DYNxDYNxbf16_times_DYNxDYNxbf16_into_DYNxDYNxbf16(%lhs: tensor<?x?xbf16>, %rhs: tensor<?x?xbf16>) -> tensor<?x?xbf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %acc_dim0 = tensor.dim %lhs, %c0 : tensor<?x?xbf16>
  %acc_dim1 = tensor.dim %rhs, %c1 : tensor<?x?xbf16>
  %init_acc = tensor.empty(%acc_dim0, %acc_dim1) : tensor<?x?xbf16>
  %c0_acc_type = arith.constant 0.0: bf16
  %acc = linalg.fill ins(%c0_acc_type : bf16) outs(%init_acc : tensor<?x?xbf16>) -> tensor<?x?xbf16>
  %result = linalg.matmul_transpose_b ins(%lhs, %rhs: tensor<?x?xbf16>, tensor<?x?xbf16>) outs(%acc: tensor<?x?xbf16>) -> tensor<?x?xbf16>
  return %result: tensor<?x?xbf16>
}

func.func @matmul_1x1xbf16_times_1x1xbf16_into_1x1xbf16(%lhs: tensor<1x1xbf16>, %rhs: tensor<1x1xbf16>) -> tensor<1x1xbf16> {
  %init_acc = tensor.empty() : tensor<1x1xbf16>
  %c0_acc_type = arith.constant 0.0: bf16
  %acc = linalg.fill ins(%c0_acc_type : bf16) outs(%init_acc : tensor<1x1xbf16>) -> tensor<1x1xbf16>
  %result = linalg.matmul_transpose_b ins(%lhs, %rhs: tensor<1x1xbf16>, tensor<1x1xbf16>) outs(%acc: tensor<1x1xbf16>) -> tensor<1x1xbf16>
  return %result: tensor<1x1xbf16>
}

func.func @matmul_accumulate_2x2xbf16_times_2x2xbf16_into_2x2xbf16(%lhs: tensor<2x2xbf16>, %rhs: tensor<2x2xbf16>, %acc: tensor<2x2xbf16>) -> tensor<2x2xbf16> {
  %result = linalg.matmul_transpose_b ins(%lhs, %rhs: tensor<2x2xbf16>, tensor<2x2xbf16>) outs(%acc: tensor<2x2xbf16>) -> tensor<2x2xbf16>
  return %result: tensor<2x2xbf16>
}

func.func @matmul_accumulate_4x4xbf16_times_4x4xbf16_into_4x4xbf16(%lhs: tensor<4x4xbf16>, %rhs: tensor<4x4xbf16>, %acc: tensor<4x4xbf16>) -> tensor<4x4xbf16> {
  %result = linalg.matmul_transpose_b ins(%lhs, %rhs: tensor<4x4xbf16>, tensor<4x4xbf16>) outs(%acc: tensor<4x4xbf16>) -> tensor<4x4xbf16>
  return %result: tensor<4x4xbf16>
}

func.func @matmul_accumulate_8x8xbf16_times_8x8xbf16_into_8x8xbf16(%lhs: tensor<8x8xbf16>, %rhs: tensor<8x8xbf16>, %acc: tensor<8x8xbf16>) -> tensor<8x8xbf16> {
  %result = linalg.matmul_transpose_b ins(%lhs, %rhs: tensor<8x8xbf16>, tensor<8x8xbf16>) outs(%acc: tensor<8x8xbf16>) -> tensor<8x8xbf16>
  return %result: tensor<8x8xbf16>
}

func.func @matmul_accumulate_9x9xbf16_times_9x9xbf16_into_9x9xbf16(%lhs: tensor<9x9xbf16>, %rhs: tensor<9x9xbf16>, %acc: tensor<9x9xbf16>) -> tensor<9x9xbf16> {
  %result = linalg.matmul_transpose_b ins(%lhs, %rhs: tensor<9x9xbf16>, tensor<9x9xbf16>) outs(%acc: tensor<9x9xbf16>) -> tensor<9x9xbf16>
  return %result: tensor<9x9xbf16>
}

func.func @matmul_accumulate_6x13xbf16_times_3x13xbf16_into_6x3xbf16(%lhs: tensor<6x13xbf16>, %rhs: tensor<3x13xbf16>, %acc: tensor<6x3xbf16>) -> tensor<6x3xbf16> {
  %result = linalg.matmul_transpose_b ins(%lhs, %rhs: tensor<6x13xbf16>, tensor<3x13xbf16>) outs(%acc: tensor<6x3xbf16>) -> tensor<6x3xbf16>
  return %result: tensor<6x3xbf16>
}

func.func @matmul_15x37xbf16_times_7x37xbf16_into_15x7xbf16(%lhs: tensor<15x37xbf16>, %rhs: tensor<7x37xbf16>) -> tensor<15x7xbf16> {
  %init_acc = tensor.empty() : tensor<15x7xbf16>
  %c0_acc_type = arith.constant 0.0: bf16
  %acc = linalg.fill ins(%c0_acc_type : bf16) outs(%init_acc : tensor<15x7xbf16>) -> tensor<15x7xbf16>
  %result = linalg.matmul_transpose_b ins(%lhs, %rhs: tensor<15x37xbf16>, tensor<7x37xbf16>) outs(%acc: tensor<15x7xbf16>) -> tensor<15x7xbf16>
  return %result: tensor<15x7xbf16>
}

func.func @matmul_accumulate_81x19xbf16_times_41x19xbf16_into_81x41xbf16(%lhs: tensor<81x19xbf16>, %rhs: tensor<41x19xbf16>, %acc: tensor<81x41xbf16>) -> tensor<81x41xbf16> {
  %result = linalg.matmul_transpose_b ins(%lhs, %rhs: tensor<81x19xbf16>, tensor<41x19xbf16>) outs(%acc: tensor<81x41xbf16>) -> tensor<81x41xbf16>
  return %result: tensor<81x41xbf16>
}

func.func @matmul_accumulate_1x10xbf16_times_10x10xbf16_into_1x10xbf16(%lhs: tensor<1x10xbf16>, %rhs: tensor<10x10xbf16>, %acc: tensor<1x10xbf16>) -> tensor<1x10xbf16> {
  %result = linalg.matmul_transpose_b ins(%lhs, %rhs: tensor<1x10xbf16>, tensor<10x10xbf16>) outs(%acc: tensor<1x10xbf16>) -> tensor<1x10xbf16>
  return %result: tensor<1x10xbf16>
}

func.func @matmul_1x10xbf16_times_10x10xbf16_into_1x10xbf16(%lhs: tensor<1x10xbf16>, %rhs: tensor<10x10xbf16>) -> tensor<1x10xbf16> {
  %init_acc = tensor.empty() : tensor<1x10xbf16>
  %c0_acc_type = arith.constant 0.0: bf16
  %acc = linalg.fill ins(%c0_acc_type : bf16) outs(%init_acc : tensor<1x10xbf16>) -> tensor<1x10xbf16>
  %result = linalg.matmul_transpose_b ins(%lhs, %rhs: tensor<1x10xbf16>, tensor<10x10xbf16>) outs(%acc: tensor<1x10xbf16>) -> tensor<1x10xbf16>
  return %result: tensor<1x10xbf16>
}

func.func @matmul_accumulate_10x1xbf16_times_10x1xbf16_into_10x10xbf16(%lhs: tensor<10x1xbf16>, %rhs: tensor<10x1xbf16>, %acc: tensor<10x10xbf16>) -> tensor<10x10xbf16> {
  %result = linalg.matmul_transpose_b ins(%lhs, %rhs: tensor<10x1xbf16>, tensor<10x1xbf16>) outs(%acc: tensor<10x10xbf16>) -> tensor<10x10xbf16>
  return %result: tensor<10x10xbf16>
}

func.func @matmul_accumulate_10x10xbf16_times_1x10xbf16_into_10x1xbf16(%lhs: tensor<10x10xbf16>, %rhs: tensor<1x10xbf16>, %acc: tensor<10x1xbf16>) -> tensor<10x1xbf16> {
  %result = linalg.matmul_transpose_b ins(%lhs, %rhs: tensor<10x10xbf16>, tensor<1x10xbf16>) outs(%acc: tensor<10x1xbf16>) -> tensor<10x1xbf16>
  return %result: tensor<10x1xbf16>
}

func.func @matmul_10x10xbf16_times_1x10xbf16_into_10x1xbf16(%lhs: tensor<10x10xbf16>, %rhs: tensor<1x10xbf16>) -> tensor<10x1xbf16> {
  %init_acc = tensor.empty() : tensor<10x1xbf16>
  %c0_acc_type = arith.constant 0.0: bf16
  %acc = linalg.fill ins(%c0_acc_type : bf16) outs(%init_acc : tensor<10x1xbf16>) -> tensor<10x1xbf16>
  %result = linalg.matmul_transpose_b ins(%lhs, %rhs: tensor<10x10xbf16>, tensor<1x10xbf16>) outs(%acc: tensor<10x1xbf16>) -> tensor<10x1xbf16>
  return %result: tensor<10x1xbf16>
}

