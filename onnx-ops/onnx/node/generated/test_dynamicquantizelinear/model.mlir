module {
  func.func @test_dynamicquantizelinear(%arg0: !torch.vtensor<[6],f32>) -> (!torch.vtensor<[6],ui8>, !torch.vtensor<[],f32>, !torch.vtensor<[],ui8>) attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0:3 = torch.operator "onnx.DynamicQuantizeLinear"(%arg0) : (!torch.vtensor<[6],f32>) -> (!torch.vtensor<[6],ui8>, !torch.vtensor<[],f32>, !torch.vtensor<[],ui8>) 
    return %0#0, %0#1, %0#2 : !torch.vtensor<[6],ui8>, !torch.vtensor<[],f32>, !torch.vtensor<[],ui8>
  }
}

