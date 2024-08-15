module {
  func.func @test_quantizelinear_uint16(%arg0: !torch.vtensor<[12],f32>, %arg1: !torch.vtensor<[],f32>, %arg2: !torch.vtensor<[],ui16>) -> !torch.vtensor<[12],ui16> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 21 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.QuantizeLinear"(%arg0, %arg1, %arg2) : (!torch.vtensor<[12],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[],ui16>) -> !torch.vtensor<[12],ui16> 
    return %0 : !torch.vtensor<[12],ui16>
  }
}

