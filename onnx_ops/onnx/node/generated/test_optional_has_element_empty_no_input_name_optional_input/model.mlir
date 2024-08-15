module {
  func.func @test_optional_has_element_empty_no_input_name_optional_input() -> !torch.vtensor<[],i1> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.OptionalHasElement"(%none) : (!torch.none) -> !torch.vtensor<[],i1> 
    return %0 : !torch.vtensor<[],i1>
  }
}

