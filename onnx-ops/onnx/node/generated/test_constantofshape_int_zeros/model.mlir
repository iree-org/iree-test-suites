module {
  func.func @test_constantofshape_int_zeros(%arg0: !torch.vtensor<[2],si64>) -> !torch.vtensor<[10,6],si32> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 21 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.ConstantOfShape"(%arg0) {torch.onnx.value = dense<0> : tensor<1xsi32>} : (!torch.vtensor<[2],si64>) -> !torch.vtensor<[10,6],si32> 
    return %0 : !torch.vtensor<[10,6],si32>
  }
}

