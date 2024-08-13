module {
  func.func @test_celu_expanded(%arg0: !torch.vtensor<[3,3,3,1],f32>) -> !torch.vtensor<[3,3,3,1],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<2.000000e+00> : tensor<1xf32>} : () -> !torch.vtensor<[1],f32> 
    %1 = torch.operator "onnx.Div"(%arg0, %0) : (!torch.vtensor<[3,3,3,1],f32>, !torch.vtensor<[1],f32>) -> !torch.vtensor<[3,3,3,1],f32> 
    %2 = torch.operator "onnx.Elu"(%1) {torch.onnx.alpha = 1.000000e+00 : f32} : (!torch.vtensor<[3,3,3,1],f32>) -> !torch.vtensor<[3,3,3,1],f32> 
    %3 = torch.operator "onnx.Mul"(%0, %2) : (!torch.vtensor<[1],f32>, !torch.vtensor<[3,3,3,1],f32>) -> !torch.vtensor<[3,3,3,1],f32> 
    return %3 : !torch.vtensor<[3,3,3,1],f32>
  }
}

