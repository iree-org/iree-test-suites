module {
  func.func @test_strnormalizer_export_monday_insensintive_upper_twodim(%arg0: !torch.vtensor<[1,6],!torch.str>) -> !torch.vtensor<[1,4],!torch.str> attributes {torch.onnx_meta.ir_version = 5 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.StringNormalizer"(%arg0) {torch.onnx.case_change_action = "UPPER", torch.onnx.stopwords = ["monday"]} : (!torch.vtensor<[1,6],!torch.str>) -> !torch.vtensor<[1,4],!torch.str> 
    return %0 : !torch.vtensor<[1,4],!torch.str>
  }
}

