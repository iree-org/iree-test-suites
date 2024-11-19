module {
  func.func @test_mish(%arg0: !torch.vtensor<[10000],f32>) -> !torch.vtensor<[10000],f32> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 22 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Mish"(%arg0) : (!torch.vtensor<[10000],f32>) -> !torch.vtensor<[10000],f32> 
    return %0 : !torch.vtensor<[10000],f32>
  }
}

