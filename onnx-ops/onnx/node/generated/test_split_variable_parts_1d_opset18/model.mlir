module {
  func.func @test_split_variable_parts_1d_opset18(%arg0: !torch.vtensor<[6],f32>, %arg1: !torch.vtensor<[2],si64>) -> (!torch.vtensor<[2],f32>, !torch.vtensor<[4],f32>) attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0:2 = torch.operator "onnx.Split"(%arg0, %arg1) {torch.onnx.axis = 0 : si64} : (!torch.vtensor<[6],f32>, !torch.vtensor<[2],si64>) -> (!torch.vtensor<[2],f32>, !torch.vtensor<[4],f32>) 
    return %0#0, %0#1 : !torch.vtensor<[2],f32>, !torch.vtensor<[4],f32>
  }
}

