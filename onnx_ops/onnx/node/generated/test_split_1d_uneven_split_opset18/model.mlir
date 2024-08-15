module {
  func.func @test_split_1d_uneven_split_opset18(%arg0: !torch.vtensor<[7],f32>) -> (!torch.vtensor<[2],f32>, !torch.vtensor<[2],f32>, !torch.vtensor<[2],f32>, !torch.vtensor<[1],f32>) attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0:4 = torch.operator "onnx.Split"(%arg0) {torch.onnx.num_outputs = 4 : si64} : (!torch.vtensor<[7],f32>) -> (!torch.vtensor<[2],f32>, !torch.vtensor<[2],f32>, !torch.vtensor<[2],f32>, !torch.vtensor<[1],f32>) 
    return %0#0, %0#1, %0#2, %0#3 : !torch.vtensor<[2],f32>, !torch.vtensor<[2],f32>, !torch.vtensor<[2],f32>, !torch.vtensor<[1],f32>
  }
}

