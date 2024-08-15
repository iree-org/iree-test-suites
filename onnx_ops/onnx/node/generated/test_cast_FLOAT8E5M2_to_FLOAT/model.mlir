module {
  func.func @test_cast_FLOAT8E5M2_to_FLOAT(%arg0: !torch.vtensor<[3,5],f8E5M2>) -> !torch.vtensor<[3,5],f32> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 21 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Cast"(%arg0) {torch.onnx.to = 1 : si64} : (!torch.vtensor<[3,5],f8E5M2>) -> !torch.vtensor<[3,5],f32> 
    return %0 : !torch.vtensor<[3,5],f32>
  }
}

