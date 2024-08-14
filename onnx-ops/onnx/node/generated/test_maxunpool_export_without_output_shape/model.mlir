module {
  func.func @test_maxunpool_export_without_output_shape(%arg0: !torch.vtensor<[1,1,2,2],f32>, %arg1: !torch.vtensor<[1,1,2,2],si64>) -> !torch.vtensor<[1,1,4,4],f32> attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.MaxUnpool"(%arg0, %arg1) {torch.onnx.kernel_shape = [2 : si64, 2 : si64], torch.onnx.strides = [2 : si64, 2 : si64]} : (!torch.vtensor<[1,1,2,2],f32>, !torch.vtensor<[1,1,2,2],si64>) -> !torch.vtensor<[1,1,4,4],f32> 
    return %0 : !torch.vtensor<[1,1,4,4],f32>
  }
}

