module {
  func.func @test_nllloss_NCd1d2d3d4d5_none_no_weight_expanded(%arg0: !torch.vtensor<[3,5,6,6,5,3,4],f32>, %arg1: !torch.vtensor<[3,6,6,5,3,4],si64>) -> !torch.vtensor<[3,6,6,5,3,4],f32> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 22 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0> : tensor<1xsi64>} : () -> !torch.vtensor<[1],si64> 
    %1 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<1> : tensor<1xsi64>} : () -> !torch.vtensor<[1],si64> 
    %2 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<1> : tensor<1xsi64>} : () -> !torch.vtensor<[1],si64> 
    %3 = torch.operator "onnx.Unsqueeze"(%arg1, %2) : (!torch.vtensor<[3,6,6,5,3,4],si64>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,1,6,6,5,3,4],si64> 
    %4 = torch.operator "onnx.GatherElements"(%arg0, %3) {torch.onnx.axis = 1 : si64} : (!torch.vtensor<[3,5,6,6,5,3,4],f32>, !torch.vtensor<[3,1,6,6,5,3,4],si64>) -> !torch.vtensor<[3,1,6,6,5,3,4],f32> 
    %5 = torch.operator "onnx.Neg"(%4) : (!torch.vtensor<[3,1,6,6,5,3,4],f32>) -> !torch.vtensor<[3,1,6,6,5,3,4],f32> 
    %6 = torch.operator "onnx.Slice"(%5, %0, %1, %1) : (!torch.vtensor<[3,1,6,6,5,3,4],f32>, !torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,1,6,6,5,3,4],f32> 
    %7 = torch.operator "onnx.Squeeze"(%6, %2) : (!torch.vtensor<[3,1,6,6,5,3,4],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,6,6,5,3,4],f32> 
    return %7 : !torch.vtensor<[3,6,6,5,3,4],f32>
  }
}

