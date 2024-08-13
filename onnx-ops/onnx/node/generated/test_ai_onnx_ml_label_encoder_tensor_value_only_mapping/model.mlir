module {
  func.func @test_ai_onnx_ml_label_encoder_tensor_value_only_mapping(%arg0: !torch.vtensor<[5],!torch.str>) -> !torch.vtensor<[5],si16> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_versions = {ai.onnx.ml = 4 : si64}, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.LabelEncoder"(%arg0) {torch.onnx.default_tensor = dense<42> : tensor<1xsi16>, torch.onnx.keys_strings = ["a", "b", "c"], torch.onnx.values_tensor = dense<[0, 1, 2]> : tensor<3xsi16>} : (!torch.vtensor<[5],!torch.str>) -> !torch.vtensor<[5],si16> 
    return %0 : !torch.vtensor<[5],si16>
  }
}

