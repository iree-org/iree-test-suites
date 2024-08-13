module {
  func.func @test_reshape_allowzero_reordered(%arg0: !torch.vtensor<[0,3,4],f32>, %arg1: !torch.vtensor<[3],si64>) -> !torch.vtensor<[3,4,0],f32> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 21 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Reshape"(%arg0, %arg1) {torch.onnx.allowzero = 1 : si64} : (!torch.vtensor<[0,3,4],f32>, !torch.vtensor<[3],si64>) -> !torch.vtensor<[3,4,0],f32> 
    return %0 : !torch.vtensor<[3,4,0],f32>
  }
}

