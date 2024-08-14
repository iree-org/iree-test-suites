module {
  func.func @test_dequantizelinear_e5m2(%arg0: !torch.vtensor<[5],f8E5M2>, %arg1: !torch.vtensor<[],f32>) -> !torch.vtensor<[5],f32> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 21 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.DequantizeLinear"(%arg0, %arg1) {torch.onnx.axis = 0 : si64} : (!torch.vtensor<[5],f8E5M2>, !torch.vtensor<[],f32>) -> !torch.vtensor<[5],f32> 
    return %0 : !torch.vtensor<[5],f32>
  }
}

