module {
  func.func @test_roialign_aligned_true(%arg0: !torch.vtensor<[1,1,10,10],f32>, %arg1: !torch.vtensor<[3,4],f32>, %arg2: !torch.vtensor<[3],si64>) -> !torch.vtensor<[3,1,5,5],f32> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 22 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.RoiAlign"(%arg0, %arg1, %arg2) {torch.onnx.coordinate_transformation_mode = "half_pixel", torch.onnx.output_height = 5 : si64, torch.onnx.output_width = 5 : si64, torch.onnx.sampling_ratio = 2 : si64, torch.onnx.spatial_scale = 1.000000e+00 : f32} : (!torch.vtensor<[1,1,10,10],f32>, !torch.vtensor<[3,4],f32>, !torch.vtensor<[3],si64>) -> !torch.vtensor<[3,1,5,5],f32> 
    return %0 : !torch.vtensor<[3,1,5,5],f32>
  }
}

