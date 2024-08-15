module {
  func.func @test_deform_conv_with_multiple_offset_groups(%arg0: !torch.vtensor<[1,2,3,3],f32>, %arg1: !torch.vtensor<[1,2,2,2],f32>, %arg2: !torch.vtensor<[1,16,2,2],f32>) -> !torch.vtensor<[1,1,2,2],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 19 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.DeformConv"(%arg0, %arg1, %arg2) {torch.onnx.kernel_shape = [2 : si64, 2 : si64], torch.onnx.offset_group = 2 : si64, torch.onnx.pads = [0 : si64, 0 : si64, 0 : si64, 0 : si64]} : (!torch.vtensor<[1,2,3,3],f32>, !torch.vtensor<[1,2,2,2],f32>, !torch.vtensor<[1,16,2,2],f32>) -> !torch.vtensor<[1,1,2,2],f32> 
    return %0 : !torch.vtensor<[1,1,2,2],f32>
  }
}

