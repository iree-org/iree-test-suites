module {
  func.func @test_reshape_negative_extended_dims(%arg0: !torch.vtensor<[2,3,4],f32>, %arg1: !torch.vtensor<[4],si64>) -> !torch.vtensor<[1,2,3,4],f32> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 21 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Reshape"(%arg0, %arg1) : (!torch.vtensor<[2,3,4],f32>, !torch.vtensor<[4],si64>) -> !torch.vtensor<[1,2,3,4],f32> 
    return %0 : !torch.vtensor<[1,2,3,4],f32>
  }
}

