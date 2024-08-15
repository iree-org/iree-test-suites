module {
  func.func @test_round(%arg0: !torch.vtensor<[15],f32>) -> !torch.vtensor<[15],f32> attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Round"(%arg0) : (!torch.vtensor<[15],f32>) -> !torch.vtensor<[15],f32> 
    return %0 : !torch.vtensor<[15],f32>
  }
}

