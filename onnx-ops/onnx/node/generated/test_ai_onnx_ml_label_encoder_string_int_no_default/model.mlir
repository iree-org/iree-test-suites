module {
  func.func @test_ai_onnx_ml_label_encoder_string_int_no_default(%arg0: !torch.vtensor<[5],!torch.str>) -> !torch.vtensor<[5],si64> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_versions = {ai.onnx.ml = 4 : si64}, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.LabelEncoder"(%arg0) {torch.onnx.keys_strings = ["a", "b", "c"], torch.onnx.values_int64s = [0 : si64, 1 : si64, 2 : si64]} : (!torch.vtensor<[5],!torch.str>) -> !torch.vtensor<[5],si64> 
    return %0 : !torch.vtensor<[5],si64>
  }
}

