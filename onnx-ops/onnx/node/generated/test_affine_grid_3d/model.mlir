module {
  func.func @test_affine_grid_3d(%arg0: !torch.vtensor<[2,3,4],f32>, %arg1: !torch.vtensor<[5],si64>) -> !torch.vtensor<[2,4,5,6,3],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.AffineGrid"(%arg0, %arg1) {torch.onnx.align_corners = 0 : si64} : (!torch.vtensor<[2,3,4],f32>, !torch.vtensor<[5],si64>) -> !torch.vtensor<[2,4,5,6,3],f32> 
    return %0 : !torch.vtensor<[2,4,5,6,3],f32>
  }
}

