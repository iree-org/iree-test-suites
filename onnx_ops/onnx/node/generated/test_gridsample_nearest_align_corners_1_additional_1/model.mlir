module {
  func.func @test_gridsample_nearest_align_corners_1_additional_1(%arg0: !torch.vtensor<[1,1,3,2],f32>, %arg1: !torch.vtensor<[1,2,4,2],f32>) -> !torch.vtensor<[1,1,2,4],f32> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 22 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.GridSample"(%arg0, %arg1) {torch.onnx.align_corners = 1 : si64, torch.onnx.mode = "nearest"} : (!torch.vtensor<[1,1,3,2],f32>, !torch.vtensor<[1,2,4,2],f32>) -> !torch.vtensor<[1,1,2,4],f32> 
    return %0 : !torch.vtensor<[1,1,2,4],f32>
  }
}

