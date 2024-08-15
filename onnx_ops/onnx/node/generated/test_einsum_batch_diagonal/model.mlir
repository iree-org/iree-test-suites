module {
  func.func @test_einsum_batch_diagonal(%arg0: !torch.vtensor<[3,5,5],f64>) -> !torch.vtensor<[3,5],f64> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Einsum"(%arg0) {torch.onnx.equation = "...ii ->...i"} : (!torch.vtensor<[3,5,5],f64>) -> !torch.vtensor<[3,5],f64> 
    return %0 : !torch.vtensor<[3,5],f64>
  }
}

