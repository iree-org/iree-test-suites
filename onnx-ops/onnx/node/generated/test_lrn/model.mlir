module {
  func.func @test_lrn(%arg0: !torch.vtensor<[5,5,5,5],f32>) -> !torch.vtensor<[5,5,5,5],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.LRN"(%arg0) {torch.onnx.alpha = 2.000000e-04 : f32, torch.onnx.beta = 5.000000e-01 : f32, torch.onnx.bias = 2.000000e+00 : f32, torch.onnx.size = 3 : si64} : (!torch.vtensor<[5,5,5,5],f32>) -> !torch.vtensor<[5,5,5,5],f32> 
    return %0 : !torch.vtensor<[5,5,5,5],f32>
  }
}

