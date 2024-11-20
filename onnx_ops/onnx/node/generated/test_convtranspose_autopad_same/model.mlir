module {
  func.func @test_convtranspose_autopad_same(%arg0: !torch.vtensor<[1,1,3,3],f32>, %arg1: !torch.vtensor<[1,2,3,3],f32>) -> !torch.vtensor<[1,2,6,6],f32> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 22 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.ConvTranspose"(%arg0, %arg1) {torch.onnx.auto_pad = "SAME_UPPER", torch.onnx.strides = [2 : si64, 2 : si64]} : (!torch.vtensor<[1,1,3,3],f32>, !torch.vtensor<[1,2,3,3],f32>) -> !torch.vtensor<[1,2,6,6],f32> 
    return %0 : !torch.vtensor<[1,2,6,6],f32>
  }
}

