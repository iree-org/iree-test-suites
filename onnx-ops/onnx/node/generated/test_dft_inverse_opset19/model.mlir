module {
  func.func @test_dft_inverse_opset19(%arg0: !torch.vtensor<[1,10,10,2],f32>) -> !torch.vtensor<[1,10,10,2],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 19 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.DFT"(%arg0) {torch.onnx.axis = 1 : si64, torch.onnx.inverse = 1 : si64} : (!torch.vtensor<[1,10,10,2],f32>) -> !torch.vtensor<[1,10,10,2],f32> 
    return %0 : !torch.vtensor<[1,10,10,2],f32>
  }
}

