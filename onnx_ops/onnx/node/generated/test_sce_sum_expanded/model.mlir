module {
  func.func @test_sce_sum_expanded(%arg0: !torch.vtensor<[3,5],f32>, %arg1: !torch.vtensor<[3],si64>) -> !torch.vtensor<[],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<[0, 0, -1]> : tensor<3xsi64>} : () -> !torch.vtensor<[3],si64> 
    %1 = torch.operator "onnx.Reshape"(%arg0, %0) : (!torch.vtensor<[3,5],f32>, !torch.vtensor<[3],si64>) -> !torch.vtensor<[3,5,1],f32> 
    %2 = torch.operator "onnx.Transpose"(%1) {torch.onnx.perm = [0 : si64, 2 : si64, 1 : si64]} : (!torch.vtensor<[3,5,1],f32>) -> !torch.vtensor<[3,1,5],f32> 
    %3 = torch.operator "onnx.LogSoftmax"(%2) {torch.onnx.axis = 2 : si64} : (!torch.vtensor<[3,1,5],f32>) -> !torch.vtensor<[3,1,5],f32> 
    %4 = torch.operator "onnx.Transpose"(%3) {torch.onnx.perm = [0 : si64, 2 : si64, 1 : si64]} : (!torch.vtensor<[3,1,5],f32>) -> !torch.vtensor<[3,5,1],f32> 
    %5 = torch.operator "onnx.Shape"(%arg0) : (!torch.vtensor<[3,5],f32>) -> !torch.vtensor<[2],si64> 
    %6 = torch.operator "onnx.Reshape"(%4, %5) : (!torch.vtensor<[3,5,1],f32>, !torch.vtensor<[2],si64>) -> !torch.vtensor<[3,5],f32> 
    %7 = torch.operator "onnx.NegativeLogLikelihoodLoss"(%6, %arg1) {torch.onnx.reduction = "sum"} : (!torch.vtensor<[3,5],f32>, !torch.vtensor<[3],si64>) -> !torch.vtensor<[],f32> 
    return %7 : !torch.vtensor<[],f32>
  }
}

