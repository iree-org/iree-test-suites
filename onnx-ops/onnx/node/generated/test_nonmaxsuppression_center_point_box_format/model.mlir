module {
  func.func @test_nonmaxsuppression_center_point_box_format(%arg0: !torch.vtensor<[1,6,4],f32>, %arg1: !torch.vtensor<[1,1,6],f32>, %arg2: !torch.vtensor<[1],si64>, %arg3: !torch.vtensor<[1],f32>, %arg4: !torch.vtensor<[1],f32>) -> !torch.vtensor<[3,3],si64> attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.NonMaxSuppression"(%arg0, %arg1, %arg2, %arg3, %arg4) {torch.onnx.center_point_box = 1 : si64} : (!torch.vtensor<[1,6,4],f32>, !torch.vtensor<[1,1,6],f32>, !torch.vtensor<[1],si64>, !torch.vtensor<[1],f32>, !torch.vtensor<[1],f32>) -> !torch.vtensor<[3,3],si64> 
    return %0 : !torch.vtensor<[3,3],si64>
  }
}

