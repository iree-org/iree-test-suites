module {
  func.func @test_resize_tf_crop_and_resize(%arg0: !torch.vtensor<[1,1,4,4],f32>, %arg1: !torch.vtensor<[8],f32>, %arg2: !torch.vtensor<[4],si64>) -> !torch.vtensor<[1,1,3,3],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 19 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Resize"(%arg0, %arg1, %none, %arg2) {torch.onnx.coordinate_transformation_mode = "tf_crop_and_resize", torch.onnx.mode = "linear"} : (!torch.vtensor<[1,1,4,4],f32>, !torch.vtensor<[8],f32>, !torch.none, !torch.vtensor<[4],si64>) -> !torch.vtensor<[1,1,3,3],f32> 
    return %0 : !torch.vtensor<[1,1,3,3],f32>
  }
}

