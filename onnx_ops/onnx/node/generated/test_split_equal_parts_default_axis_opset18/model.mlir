module {
  func.func @test_split_equal_parts_default_axis_opset18(%arg0: !torch.vtensor<[6],f32>) -> (!torch.vtensor<[2],f32>, !torch.vtensor<[2],f32>, !torch.vtensor<[2],f32>) attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0:3 = torch.operator "onnx.Split"(%arg0) {torch.onnx.num_outputs = 3 : si64} : (!torch.vtensor<[6],f32>) -> (!torch.vtensor<[2],f32>, !torch.vtensor<[2],f32>, !torch.vtensor<[2],f32>) 
    return %0#0, %0#1, %0#2 : !torch.vtensor<[2],f32>, !torch.vtensor<[2],f32>, !torch.vtensor<[2],f32>
  }
}

