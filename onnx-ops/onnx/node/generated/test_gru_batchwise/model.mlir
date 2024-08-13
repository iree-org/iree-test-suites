module {
  func.func @test_gru_batchwise(%arg0: !torch.vtensor<[3,1,2],f32>, %arg1: !torch.vtensor<[1,18,2],f32>, %arg2: !torch.vtensor<[1,18,6],f32>) -> (!torch.vtensor<[3,1,1,6],f32>, !torch.vtensor<[3,1,6],f32>) attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0:2 = torch.operator "onnx.GRU"(%arg0, %arg1, %arg2) {torch.onnx.hidden_size = 6 : si64, torch.onnx.layout = 1 : si64} : (!torch.vtensor<[3,1,2],f32>, !torch.vtensor<[1,18,2],f32>, !torch.vtensor<[1,18,6],f32>) -> (!torch.vtensor<[3,1,1,6],f32>, !torch.vtensor<[3,1,6],f32>) 
    return %0#0, %0#1 : !torch.vtensor<[3,1,1,6],f32>, !torch.vtensor<[3,1,6],f32>
  }
}

