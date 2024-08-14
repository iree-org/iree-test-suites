module {
  func.func @test_strnormalizer_export_monday_casesensintive_lower(%arg0: !torch.vtensor<[4],!torch.str>) -> !torch.vtensor<[3],!torch.str> attributes {torch.onnx_meta.ir_version = 5 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.StringNormalizer"(%arg0) {torch.onnx.case_change_action = "LOWER", torch.onnx.is_case_sensitive = 1 : si64, torch.onnx.stopwords = ["monday"]} : (!torch.vtensor<[4],!torch.str>) -> !torch.vtensor<[3],!torch.str> 
    return %0 : !torch.vtensor<[3],!torch.str>
  }
}

