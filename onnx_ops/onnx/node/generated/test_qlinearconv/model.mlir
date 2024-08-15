module {
  func.func @test_qlinearconv(%arg0: !torch.vtensor<[1,1,7,7],ui8>, %arg1: !torch.vtensor<[],f32>, %arg2: !torch.vtensor<[],ui8>, %arg3: !torch.vtensor<[1,1,1,1],ui8>, %arg4: !torch.vtensor<[1],f32>, %arg5: !torch.vtensor<[1],ui8>, %arg6: !torch.vtensor<[],f32>, %arg7: !torch.vtensor<[],ui8>) -> !torch.vtensor<[1,1,7,7],ui8> attributes {torch.onnx_meta.ir_version = 5 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.QLinearConv"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7) : (!torch.vtensor<[1,1,7,7],ui8>, !torch.vtensor<[],f32>, !torch.vtensor<[],ui8>, !torch.vtensor<[1,1,1,1],ui8>, !torch.vtensor<[1],f32>, !torch.vtensor<[1],ui8>, !torch.vtensor<[],f32>, !torch.vtensor<[],ui8>) -> !torch.vtensor<[1,1,7,7],ui8> 
    return %0 : !torch.vtensor<[1,1,7,7],ui8>
  }
}

