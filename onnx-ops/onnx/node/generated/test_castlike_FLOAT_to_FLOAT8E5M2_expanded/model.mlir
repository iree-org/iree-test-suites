module {
  func.func @test_castlike_FLOAT_to_FLOAT8E5M2_expanded(%arg0: !torch.vtensor<[3,4],f32>, %arg1: !torch.vtensor<[1],f8E5M2>) -> !torch.vtensor<[3,4],f8E5M2> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 21 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Cast"(%arg0) {torch.onnx.saturate = 1 : si64, torch.onnx.to = 19 : si64} : (!torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f8E5M2> 
    return %0 : !torch.vtensor<[3,4],f8E5M2>
  }
}

