module {
  func.func @test_string_concat_zero_dimensional(%arg0: !torch.vtensor<[],!torch.str>, %arg1: !torch.vtensor<[],!torch.str>) -> !torch.vtensor<[],!torch.str> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.StringConcat"(%arg0, %arg1) : (!torch.vtensor<[],!torch.str>, !torch.vtensor<[],!torch.str>) -> !torch.vtensor<[],!torch.str> 
    return %0 : !torch.vtensor<[],!torch.str>
  }
}

