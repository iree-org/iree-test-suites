module {
  func.func @test_bernoulli_seed(%arg0: !torch.vtensor<[10],f32>) -> !torch.vtensor<[10],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Bernoulli"(%arg0) {torch.onnx.seed = 0.000000e+00 : f32} : (!torch.vtensor<[10],f32>) -> !torch.vtensor<[10],f32> 
    return %0 : !torch.vtensor<[10],f32>
  }
}

