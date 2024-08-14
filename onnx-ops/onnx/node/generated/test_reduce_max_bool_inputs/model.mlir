module {
  func.func @test_reduce_max_bool_inputs(%arg0: !torch.vtensor<[4,2],i1>, %arg1: !torch.vtensor<[1],si64>) -> !torch.vtensor<[4,1],i1> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.ReduceMax"(%arg0, %arg1) {torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[4,2],i1>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[4,1],i1> 
    return %0 : !torch.vtensor<[4,1],i1>
  }
}

