module {
  func.func @test_ai_onnx_ml_array_feature_extractor(%arg0: !torch.vtensor<[3,4],f32>, %arg1: !torch.vtensor<[2],si64>) -> !torch.vtensor<[3,2],f32> attributes {torch.onnx_meta.ir_version = 3 : si64, torch.onnx_meta.opset_versions = {ai.onnx.ml = 1 : si64}, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.ArrayFeatureExtractor"(%arg0, %arg1) : (!torch.vtensor<[3,4],f32>, !torch.vtensor<[2],si64>) -> !torch.vtensor<[3,2],f32> 
    return %0 : !torch.vtensor<[3,2],f32>
  }
}

