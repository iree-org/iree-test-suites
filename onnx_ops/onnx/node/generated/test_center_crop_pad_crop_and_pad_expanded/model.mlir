module {
  func.func @test_center_crop_pad_crop_and_pad_expanded(%arg0: !torch.vtensor<[20,8,3],f32>, %arg1: !torch.vtensor<[3],si64>) -> !torch.vtensor<[10,10,3],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<2> : tensor<1xsi64>} : () -> !torch.vtensor<[1],si64> 
    %1 = torch.operator "onnx.Shape"(%arg0) : (!torch.vtensor<[20,8,3],f32>) -> !torch.vtensor<[3],si64> 
    %2 = torch.operator "onnx.Max"(%1, %arg1) : (!torch.vtensor<[3],si64>, !torch.vtensor<[3],si64>) -> !torch.vtensor<[3],si64> 
    %3 = torch.operator "onnx.Sub"(%2, %1) : (!torch.vtensor<[3],si64>, !torch.vtensor<[3],si64>) -> !torch.vtensor<[3],si64> 
    %4 = torch.operator "onnx.Div"(%3, %0) : (!torch.vtensor<[3],si64>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3],si64> 
    %5 = torch.operator "onnx.Sub"(%3, %4) : (!torch.vtensor<[3],si64>, !torch.vtensor<[3],si64>) -> !torch.vtensor<[3],si64> 
    %6 = torch.operator "onnx.Concat"(%4, %5) {torch.onnx.axis = 0 : si64} : (!torch.vtensor<[3],si64>, !torch.vtensor<[3],si64>) -> !torch.vtensor<[6],si64> 
    %7 = torch.operator "onnx.Pad"(%arg0, %6) : (!torch.vtensor<[20,8,3],f32>, !torch.vtensor<[6],si64>) -> !torch.vtensor<[?,?,?],f32> 
    %8 = torch.operator "onnx.Shape"(%7) : (!torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[3],si64> 
    %9 = torch.operator "onnx.Sub"(%8, %arg1) : (!torch.vtensor<[3],si64>, !torch.vtensor<[3],si64>) -> !torch.vtensor<[3],si64> 
    %10 = torch.operator "onnx.Div"(%9, %0) : (!torch.vtensor<[3],si64>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3],si64> 
    %11 = torch.operator "onnx.Add"(%10, %arg1) : (!torch.vtensor<[3],si64>, !torch.vtensor<[3],si64>) -> !torch.vtensor<[3],si64> 
    %12 = torch.operator "onnx.Slice"(%7, %10, %11) : (!torch.vtensor<[?,?,?],f32>, !torch.vtensor<[3],si64>, !torch.vtensor<[3],si64>) -> !torch.vtensor<[10,10,3],f32> 
    return %12 : !torch.vtensor<[10,10,3],f32>
  }
}

