module {
  func.func @test_equal_string(%arg0: !torch.vtensor<[2],!torch.str>, %arg1: !torch.vtensor<[2],!torch.str>) -> !torch.vtensor<[2],i1> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 19 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Equal"(%arg0, %arg1) : (!torch.vtensor<[2],!torch.str>, !torch.vtensor<[2],!torch.str>) -> !torch.vtensor<[2],i1> 
    return %0 : !torch.vtensor<[2],i1>
  }
}

