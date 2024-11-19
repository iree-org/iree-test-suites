module {
  func.func @test_convtranspose_group_2_image_3(%arg0: !torch.vtensor<[3,2,3,3],f32>, %arg1: !torch.vtensor<[2,1,3,3],f32>) -> !torch.vtensor<[3,2,5,5],f32> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 22 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.ConvTranspose"(%arg0, %arg1) {torch.onnx.group = 2 : si64} : (!torch.vtensor<[3,2,3,3],f32>, !torch.vtensor<[2,1,3,3],f32>) -> !torch.vtensor<[3,2,5,5],f32> 
    return %0 : !torch.vtensor<[3,2,5,5],f32>
  }
}

