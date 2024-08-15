module {
  func.func @test_reduce_sum_square_negative_axes_keepdims_random_expanded(%arg0: !torch.vtensor<[3,2,2],f32>, %arg1: !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,1,2],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Mul"(%arg0, %arg0) : (!torch.vtensor<[3,2,2],f32>, !torch.vtensor<[3,2,2],f32>) -> !torch.vtensor<[3,2,2],f32> 
    %1 = torch.operator "onnx.ReduceSum"(%0, %arg1) {torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[3,2,2],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,1,2],f32> 
    return %1 : !torch.vtensor<[3,1,2],f32>
  }
}

