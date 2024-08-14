module {
  func.func @test_image_decoder_decode_jpeg_bgr(%arg0: !torch.vtensor<[1058],ui8>) -> !torch.vtensor<[32,32,3],ui8> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.ImageDecoder"(%arg0) {torch.onnx.pixel_format = "BGR"} : (!torch.vtensor<[1058],ui8>) -> !torch.vtensor<[32,32,3],ui8> 
    return %0 : !torch.vtensor<[32,32,3],ui8>
  }
}

