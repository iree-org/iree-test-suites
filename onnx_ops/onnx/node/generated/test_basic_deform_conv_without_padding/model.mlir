module {
  func.func @test_basic_deform_conv_without_padding(%arg0: !torch.vtensor<[1,1,3,3],f32>, %arg1: !torch.vtensor<[1,1,2,2],f32>, %arg2: !torch.vtensor<[1,8,2,2],f32>) -> !torch.vtensor<[1,1,2,2],f32> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 22 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.DeformConv"(%arg0, %arg1, %arg2) {torch.onnx.kernel_shape = [2 : si64, 2 : si64], torch.onnx.pads = [0 : si64, 0 : si64, 0 : si64, 0 : si64]} : (!torch.vtensor<[1,1,3,3],f32>, !torch.vtensor<[1,1,2,2],f32>, !torch.vtensor<[1,8,2,2],f32>) -> !torch.vtensor<[1,1,2,2],f32> 
    return %0 : !torch.vtensor<[1,1,2,2],f32>
  }
}

