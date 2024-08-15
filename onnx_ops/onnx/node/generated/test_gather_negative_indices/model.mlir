module {
  func.func @test_gather_negative_indices(%arg0: !torch.vtensor<[10],f32>, %arg1: !torch.vtensor<[3],si64>) -> !torch.vtensor<[3],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Gather"(%arg0, %arg1) {torch.onnx.axis = 0 : si64} : (!torch.vtensor<[10],f32>, !torch.vtensor<[3],si64>) -> !torch.vtensor<[3],f32> 
    return %0 : !torch.vtensor<[3],f32>
  }
}

