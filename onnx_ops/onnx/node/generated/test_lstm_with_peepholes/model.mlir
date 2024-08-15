module {
  func.func @test_lstm_with_peepholes(%arg0: !torch.vtensor<[1,2,4],f32>, %arg1: !torch.vtensor<[1,12,4],f32>, %arg2: !torch.vtensor<[1,12,3],f32>, %arg3: !torch.vtensor<[1,24],f32>, %arg4: !torch.vtensor<[2],si32>, %arg5: !torch.vtensor<[1,2,3],f32>, %arg6: !torch.vtensor<[1,2,3],f32>, %arg7: !torch.vtensor<[1,9],f32>) -> !torch.vtensor<[1,2,3],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0:2 = torch.operator "onnx.LSTM"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7) {torch.onnx.hidden_size = 3 : si64} : (!torch.vtensor<[1,2,4],f32>, !torch.vtensor<[1,12,4],f32>, !torch.vtensor<[1,12,3],f32>, !torch.vtensor<[1,24],f32>, !torch.vtensor<[2],si32>, !torch.vtensor<[1,2,3],f32>, !torch.vtensor<[1,2,3],f32>, !torch.vtensor<[1,9],f32>) -> (!torch.none, !torch.vtensor<[1,2,3],f32>) 
    return %0#1 : !torch.vtensor<[1,2,3],f32>
  }
}

