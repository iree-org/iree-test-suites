module {
  func.func @test_softsign_example_expanded_ver18(%arg0: !torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<1.000000e+00> : tensor<f32>} : () -> !torch.vtensor<[],f32> 
    %1 = torch.operator "onnx.CastLike"(%0, %arg0) : (!torch.vtensor<[],f32>, !torch.vtensor<[3],f32>) -> !torch.vtensor<[],f32> 
    %2 = torch.operator "onnx.Abs"(%arg0) : (!torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32> 
    %3 = torch.operator "onnx.Add"(%1, %2) : (!torch.vtensor<[],f32>, !torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32> 
    %4 = torch.operator "onnx.Div"(%arg0, %3) : (!torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32> 
    return %4 : !torch.vtensor<[3],f32>
  }
}

