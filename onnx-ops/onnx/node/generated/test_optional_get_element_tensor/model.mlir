module {
  func.func @test_optional_get_element_tensor(%arg0: !torch.vtensor<[4],f32>) -> !torch.vtensor<[4],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.OptionalGetElement"(%arg0) : (!torch.vtensor<[4],f32>) -> !torch.vtensor<[4],f32> 
    return %0 : !torch.vtensor<[4],f32>
  }
}

