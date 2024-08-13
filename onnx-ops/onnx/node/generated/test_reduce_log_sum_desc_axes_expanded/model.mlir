module {
  func.func @test_reduce_log_sum_desc_axes_expanded(%arg0: !torch.vtensor<[3,4,5],f32>, %arg1: !torch.vtensor<[2],si64>) -> !torch.vtensor<[3],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.ReduceSum"(%arg0, %arg1) {torch.onnx.keepdims = 0 : si64} : (!torch.vtensor<[3,4,5],f32>, !torch.vtensor<[2],si64>) -> !torch.vtensor<[],f32> 
    %1 = torch.operator "onnx.Log"(%0) : (!torch.vtensor<[],f32>) -> !torch.vtensor<[3],f32> 
    return %1 : !torch.vtensor<[3],f32>
  }
}

