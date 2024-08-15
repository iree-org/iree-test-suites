module {
  func.func @test_leakyrelu(%arg0: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.LeakyRelu"(%arg0) {torch.onnx.alpha = 1.000000e-01 : f32} : (!torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> 
    return %0 : !torch.vtensor<[3,4,5],f32>
  }
}

