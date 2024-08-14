module {
  func.func @test_top_k_smallest(%arg0: !torch.vtensor<[3,4],f32>, %arg1: !torch.vtensor<[1],si64>) -> (!torch.vtensor<[3,3],f32>, !torch.vtensor<[3,3],si64>) attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0:2 = torch.operator "onnx.TopK"(%arg0, %arg1) {torch.onnx.axis = 1 : si64, torch.onnx.largest = 0 : si64, torch.onnx.sorted = 1 : si64} : (!torch.vtensor<[3,4],f32>, !torch.vtensor<[1],si64>) -> (!torch.vtensor<[3,3],f32>, !torch.vtensor<[3,3],si64>) 
    return %0#0, %0#1 : !torch.vtensor<[3,3],f32>, !torch.vtensor<[3,3],si64>
  }
}

