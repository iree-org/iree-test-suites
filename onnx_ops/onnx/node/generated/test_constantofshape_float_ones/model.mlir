module {
  func.func @test_constantofshape_float_ones(%arg0: !torch.vtensor<[3],si64>) -> !torch.vtensor<[4,3,2],f32> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 21 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.ConstantOfShape"(%arg0) {torch.onnx.value = dense<1.000000e+00> : tensor<1xf32>} : (!torch.vtensor<[3],si64>) -> !torch.vtensor<[4,3,2],f32> 
    return %0 : !torch.vtensor<[4,3,2],f32>
  }
}

