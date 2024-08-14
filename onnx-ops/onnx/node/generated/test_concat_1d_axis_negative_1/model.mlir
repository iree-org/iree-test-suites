module {
  func.func @test_concat_1d_axis_negative_1(%arg0: !torch.vtensor<[2],f32>, %arg1: !torch.vtensor<[2],f32>) -> !torch.vtensor<[4],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Concat"(%arg0, %arg1) {torch.onnx.axis = -1 : si64} : (!torch.vtensor<[2],f32>, !torch.vtensor<[2],f32>) -> !torch.vtensor<[4],f32> 
    return %0 : !torch.vtensor<[4],f32>
  }
}

