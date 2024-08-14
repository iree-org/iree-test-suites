module {
  func.func @test_qlinearmatmul_2D_uint8_float16(%arg0: !torch.vtensor<[2,4],ui8>, %arg1: !torch.vtensor<[1],f16>, %arg2: !torch.vtensor<[1],ui8>, %arg3: !torch.vtensor<[4,3],ui8>, %arg4: !torch.vtensor<[1],f16>, %arg5: !torch.vtensor<[1],ui8>, %arg6: !torch.vtensor<[1],f16>, %arg7: !torch.vtensor<[1],ui8>) -> !torch.vtensor<[2,3],ui8> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 21 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.QLinearMatMul"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7) : (!torch.vtensor<[2,4],ui8>, !torch.vtensor<[1],f16>, !torch.vtensor<[1],ui8>, !torch.vtensor<[4,3],ui8>, !torch.vtensor<[1],f16>, !torch.vtensor<[1],ui8>, !torch.vtensor<[1],f16>, !torch.vtensor<[1],ui8>) -> !torch.vtensor<[2,3],ui8> 
    return %0 : !torch.vtensor<[2,3],ui8>
  }
}

