module {
  func.func @test_prelu_broadcast_expanded(%arg0: !torch.vtensor<[3,4,5],f32>, %arg1: !torch.vtensor<[5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0.000000e+00> : tensor<f32>} : () -> !torch.vtensor<[],f32> 
    %1 = torch.operator "onnx.CastLike"(%0, %arg0) : (!torch.vtensor<[],f32>, !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[],f32> 
    %2 = torch.operator "onnx.Less"(%arg0, %1) : (!torch.vtensor<[3,4,5],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[3,4,5],i1> 
    %3 = torch.operator "onnx.Mul"(%arg1, %arg0) : (!torch.vtensor<[5],f32>, !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> 
    %4 = torch.operator "onnx.Where"(%2, %3, %arg0) : (!torch.vtensor<[3,4,5],i1>, !torch.vtensor<[3,4,5],f32>, !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> 
    return %4 : !torch.vtensor<[3,4,5],f32>
  }
}

