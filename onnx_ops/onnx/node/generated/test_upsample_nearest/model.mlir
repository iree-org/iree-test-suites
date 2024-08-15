module {
  func.func @test_upsample_nearest(%arg0: !torch.vtensor<[1,1,2,2],f32>, %arg1: !torch.vtensor<[4],f32>) -> !torch.vtensor<[1,1,4,6],f32> attributes {torch.onnx_meta.ir_version = 4 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<[0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00]> : tensor<8xf32>} : () -> !torch.vtensor<[8],f32> 
    %1 = torch.operator "onnx.Resize"(%arg0, %0, %arg1) {torch.onnx.mode = "nearest"} : (!torch.vtensor<[1,1,2,2],f32>, !torch.vtensor<[8],f32>, !torch.vtensor<[4],f32>) -> !torch.vtensor<[1,1,4,6],f32> 
    return %1 : !torch.vtensor<[1,1,4,6],f32>
  }
}

