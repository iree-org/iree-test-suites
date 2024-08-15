module {
  func.func @test_thresholdedrelu_expanded_ver18(%arg0: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Constant"() {torch.onnx.value_float = 2.000000e+00 : f32} : () -> !torch.vtensor<[],f32> 
    %1 = torch.operator "onnx.CastLike"(%0, %arg0) : (!torch.vtensor<[],f32>, !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[],f32> 
    %2 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0.000000e+00> : tensor<f32>} : () -> !torch.vtensor<[],f32> 
    %3 = torch.operator "onnx.CastLike"(%2, %arg0) : (!torch.vtensor<[],f32>, !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[],f32> 
    %4 = torch.operator "onnx.Less"(%1, %arg0) : (!torch.vtensor<[],f32>, !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],i1> 
    %5 = torch.operator "onnx.Where"(%4, %arg0, %3) : (!torch.vtensor<[3,4,5],i1>, !torch.vtensor<[3,4,5],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[3,4,5],f32> 
    return %5 : !torch.vtensor<[3,4,5],f32>
  }
}

