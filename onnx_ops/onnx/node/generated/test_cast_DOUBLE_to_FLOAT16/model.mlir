module {
  func.func @test_cast_DOUBLE_to_FLOAT16(%arg0: !torch.vtensor<[3,4],f64>) -> !torch.vtensor<[3,4],f16> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 21 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Cast"(%arg0) {torch.onnx.to = 10 : si64} : (!torch.vtensor<[3,4],f64>) -> !torch.vtensor<[3,4],f16> 
    return %0 : !torch.vtensor<[3,4],f16>
  }
}

