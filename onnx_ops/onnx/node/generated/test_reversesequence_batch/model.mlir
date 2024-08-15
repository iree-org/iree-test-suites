module {
  func.func @test_reversesequence_batch(%arg0: !torch.vtensor<[4,4],f32>, %arg1: !torch.vtensor<[4],si64>) -> !torch.vtensor<[4,4],f32> attributes {torch.onnx_meta.ir_version = 5 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.ReverseSequence"(%arg0, %arg1) {torch.onnx.batch_axis = 0 : si64, torch.onnx.time_axis = 1 : si64} : (!torch.vtensor<[4,4],f32>, !torch.vtensor<[4],si64>) -> !torch.vtensor<[4,4],f32> 
    return %0 : !torch.vtensor<[4,4],f32>
  }
}

