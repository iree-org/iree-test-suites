module {
  func.func @test_gathernd_example_int32_batch_dim1(%arg0: !torch.vtensor<[2,2,2],si32>, %arg1: !torch.vtensor<[2,1],si64>) -> !torch.vtensor<[2,2],si32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.GatherND"(%arg0, %arg1) {torch.onnx.batch_dims = 1 : si64} : (!torch.vtensor<[2,2,2],si32>, !torch.vtensor<[2,1],si64>) -> !torch.vtensor<[2,2],si32> 
    return %0 : !torch.vtensor<[2,2],si32>
  }
}

