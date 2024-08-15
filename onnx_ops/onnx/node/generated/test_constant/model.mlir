module {
  func.func @test_constant() -> !torch.vtensor<[5,5],f32> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 21 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<[[1.76405239, 0.400157213, 9.787380e-01, 2.24089313, 1.867558], [-0.977277874, 0.950088441, -0.151357204, -0.103218853, 0.410598516], [0.144043565, 1.45427346, 0.761037707, 0.121675014, 0.443863243], [0.333674341, 1.49407911, -0.205158263, 0.313067704, -0.854095757], [-2.55298972, 0.653618574, 0.864436209, -7.421650e-01, 2.26975465]]> : tensor<5x5xf32>} : () -> !torch.vtensor<[5,5],f32> 
    return %0 : !torch.vtensor<[5,5],f32>
  }
}

