module {
  func.func @test_unique_sorted_with_axis(%arg0: !torch.vtensor<[3,3],f32>) -> (!torch.vtensor<[2,3],f32>, !torch.vtensor<[2],si64>, !torch.vtensor<[3],si64>, !torch.vtensor<[2],si64>) attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0:4 = torch.operator "onnx.Unique"(%arg0) {torch.onnx.axis = 0 : si64, torch.onnx.sorted = 1 : si64} : (!torch.vtensor<[3,3],f32>) -> (!torch.vtensor<[2,3],f32>, !torch.vtensor<[2],si64>, !torch.vtensor<[3],si64>, !torch.vtensor<[2],si64>) 
    return %0#0, %0#1, %0#2, %0#3 : !torch.vtensor<[2,3],f32>, !torch.vtensor<[2],si64>, !torch.vtensor<[3],si64>, !torch.vtensor<[2],si64>
  }
}

