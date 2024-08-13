module {
  func.func @test_quantizelinear_int16(%arg0: !torch.vtensor<[16],f32>, %arg1: !torch.vtensor<[],f32>, %arg2: !torch.vtensor<[],si16>) -> !torch.vtensor<[16],si16> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 21 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.QuantizeLinear"(%arg0, %arg1, %arg2) : (!torch.vtensor<[16],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[],si16>) -> !torch.vtensor<[16],si16> 
    return %0 : !torch.vtensor<[16],si16>
  }
}

