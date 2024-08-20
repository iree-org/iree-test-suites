func.func @matmul_accumulate_DYNxDYNxf8E4M3FNUZ_times_DYNxDYNxf8E4M3FNUZ_into_DYNxDYNxf32(%lhs: tensor<?x?xf32>, %rhs: tensor<?x?xf32>, %acc: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %lhs_casted = arith.truncf %lhs: tensor<?x?xf32> to tensor<?x?xf8E4M3FNUZ>
  %rhs_casted = arith.truncf %rhs: tensor<?x?xf32> to tensor<?x?xf8E4M3FNUZ>
  %result = linalg.matmul ins(%lhs_casted, %rhs_casted: tensor<?x?xf8E4M3FNUZ>, tensor<?x?xf8E4M3FNUZ>) outs(%acc: tensor<?x?xf32>) -> tensor<?x?xf32>  return %result: tensor<?x?xf32>
}

func.func @matmul_accumulate_123x456xf8E4M3FNUZ_times_456x789xf8E4M3FNUZ_into_123x789xf32(%lhs: tensor<123x456xf32>, %rhs: tensor<456x789xf32>, %acc: tensor<123x789xf32>) -> tensor<123x789xf32> {
  %lhs_casted = arith.truncf %lhs: tensor<123x456xf32> to tensor<123x456xf8E4M3FNUZ>
  %rhs_casted = arith.truncf %rhs: tensor<456x789xf32> to tensor<456x789xf8E4M3FNUZ>
  %result = linalg.matmul ins(%lhs_casted, %rhs_casted: tensor<123x456xf8E4M3FNUZ>, tensor<456x789xf8E4M3FNUZ>) outs(%acc: tensor<123x789xf32>) -> tensor<123x789xf32>  return %result: tensor<123x789xf32>
}

func.func @matmul_DYNxDYNxf8E4M3FNUZ_times_DYNxDYNxf8E4M3FNUZ_into_DYNxDYNxf32(%lhs: tensor<?x?xf32>, %rhs: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %acc_dim0 = tensor.dim %lhs, %c0 : tensor<?x?xf32>
  %acc_dim1 = tensor.dim %rhs, %c1 : tensor<?x?xf32>
  %init_acc = tensor.empty(%acc_dim0, %acc_dim1) : tensor<?x?xf32>
  %c0_acc_type = arith.constant 0.0: f32
  %acc = linalg.fill ins(%c0_acc_type : f32) outs(%init_acc : tensor<?x?xf32>) -> tensor<?x?xf32>
  %lhs_casted = arith.truncf %lhs: tensor<?x?xf32> to tensor<?x?xf8E4M3FNUZ>
  %rhs_casted = arith.truncf %rhs: tensor<?x?xf32> to tensor<?x?xf8E4M3FNUZ>
  %result = linalg.matmul ins(%lhs_casted, %rhs_casted: tensor<?x?xf8E4M3FNUZ>, tensor<?x?xf8E4M3FNUZ>) outs(%acc: tensor<?x?xf32>) -> tensor<?x?xf32>  return %result: tensor<?x?xf32>
}

func.func @matmul_654x321xf8E4M3FNUZ_times_321x234xf8E4M3FNUZ_into_654x234xf32(%lhs: tensor<654x321xf32>, %rhs: tensor<321x234xf32>) -> tensor<654x234xf32> {
  %init_acc = tensor.empty() : tensor<654x234xf32>
  %c0_acc_type = arith.constant 0.0: f32
  %acc = linalg.fill ins(%c0_acc_type : f32) outs(%init_acc : tensor<654x234xf32>) -> tensor<654x234xf32>
  %lhs_casted = arith.truncf %lhs: tensor<654x321xf32> to tensor<654x321xf8E4M3FNUZ>
  %rhs_casted = arith.truncf %rhs: tensor<321x234xf32> to tensor<321x234xf8E4M3FNUZ>
  %result = linalg.matmul ins(%lhs_casted, %rhs_casted: tensor<654x321xf8E4M3FNUZ>, tensor<321x234xf8E4M3FNUZ>) outs(%acc: tensor<654x234xf32>) -> tensor<654x234xf32>  return %result: tensor<654x234xf32>
}

func.func @matmul_accumulate_1x1000xf8E4M3FNUZ_times_1000x1000xf8E4M3FNUZ_into_1x1000xf32(%lhs: tensor<1x1000xf32>, %rhs: tensor<1000x1000xf32>, %acc: tensor<1x1000xf32>) -> tensor<1x1000xf32> {
  %lhs_casted = arith.truncf %lhs: tensor<1x1000xf32> to tensor<1x1000xf8E4M3FNUZ>
  %rhs_casted = arith.truncf %rhs: tensor<1000x1000xf32> to tensor<1000x1000xf8E4M3FNUZ>
  %result = linalg.matmul ins(%lhs_casted, %rhs_casted: tensor<1x1000xf8E4M3FNUZ>, tensor<1000x1000xf8E4M3FNUZ>) outs(%acc: tensor<1x1000xf32>) -> tensor<1x1000xf32>  return %result: tensor<1x1000xf32>
}

func.func @matmul_accumulate_1000x1000xf8E4M3FNUZ_times_1000x1xf8E4M3FNUZ_into_1000x1xf32(%lhs: tensor<1000x1000xf32>, %rhs: tensor<1000x1xf32>, %acc: tensor<1000x1xf32>) -> tensor<1000x1xf32> {
  %lhs_casted = arith.truncf %lhs: tensor<1000x1000xf32> to tensor<1000x1000xf8E4M3FNUZ>
  %rhs_casted = arith.truncf %rhs: tensor<1000x1xf32> to tensor<1000x1xf8E4M3FNUZ>
  %result = linalg.matmul ins(%lhs_casted, %rhs_casted: tensor<1000x1000xf8E4M3FNUZ>, tensor<1000x1xf8E4M3FNUZ>) outs(%acc: tensor<1000x1xf32>) -> tensor<1000x1xf32>  return %result: tensor<1000x1xf32>
}

func.func @matmul_1000x1000xf8E4M3FNUZ_times_1000x1xf8E4M3FNUZ_into_1000x1xf32(%lhs: tensor<1000x1000xf32>, %rhs: tensor<1000x1xf32>) -> tensor<1000x1xf32> {
  %init_acc = tensor.empty() : tensor<1000x1xf32>
  %c0_acc_type = arith.constant 0.0: f32
  %acc = linalg.fill ins(%c0_acc_type : f32) outs(%init_acc : tensor<1000x1xf32>) -> tensor<1000x1xf32>
  %lhs_casted = arith.truncf %lhs: tensor<1000x1000xf32> to tensor<1000x1000xf8E4M3FNUZ>
  %rhs_casted = arith.truncf %rhs: tensor<1000x1xf32> to tensor<1000x1xf8E4M3FNUZ>
  %result = linalg.matmul ins(%lhs_casted, %rhs_casted: tensor<1000x1000xf8E4M3FNUZ>, tensor<1000x1xf8E4M3FNUZ>) outs(%acc: tensor<1000x1xf32>) -> tensor<1000x1xf32>  return %result: tensor<1000x1xf32>
}

