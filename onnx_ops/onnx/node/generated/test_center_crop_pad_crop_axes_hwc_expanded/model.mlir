module {
  func.func @test_center_crop_pad_crop_axes_hwc_expanded(%arg0: !torch.vtensor<[20,8,3],f32>, %arg1: !torch.vtensor<[2],si64>) -> !torch.vtensor<[10,9,3],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<2> : tensor<1xsi64>} : () -> !torch.vtensor<[1],si64> 
    %1 = torch.operator "onnx.Constant"() {torch.onnx.value_ints = [0 : si64, 1 : si64]} : () -> !torch.vtensor<[2],si64> 
    %2 = torch.operator "onnx.Shape"(%arg0) : (!torch.vtensor<[20,8,3],f32>) -> !torch.vtensor<[3],si64> 
    %3 = torch.operator "onnx.Gather"(%2, %1) : (!torch.vtensor<[3],si64>, !torch.vtensor<[2],si64>) -> !torch.vtensor<[2],si64> 
    %4 = torch.operator "onnx.Max"(%3, %arg1) : (!torch.vtensor<[2],si64>, !torch.vtensor<[2],si64>) -> !torch.vtensor<[2],si64> 
    %5 = torch.operator "onnx.Sub"(%4, %3) : (!torch.vtensor<[2],si64>, !torch.vtensor<[2],si64>) -> !torch.vtensor<[2],si64> 
    %6 = torch.operator "onnx.Div"(%5, %0) : (!torch.vtensor<[2],si64>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[2],si64> 
    %7 = torch.operator "onnx.Sub"(%5, %6) : (!torch.vtensor<[2],si64>, !torch.vtensor<[2],si64>) -> !torch.vtensor<[2],si64> 
    %8 = torch.operator "onnx.Concat"(%6, %7) {torch.onnx.axis = 0 : si64} : (!torch.vtensor<[2],si64>, !torch.vtensor<[2],si64>) -> !torch.vtensor<[4],si64> 
    %9 = torch.operator "onnx.Pad"(%arg0, %8, %none, %1) : (!torch.vtensor<[20,8,3],f32>, !torch.vtensor<[4],si64>, !torch.none, !torch.vtensor<[2],si64>) -> !torch.vtensor<[?,?,?],f32> 
    %10 = torch.operator "onnx.Shape"(%9) : (!torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[3],si64> 
    %11 = torch.operator "onnx.Gather"(%10, %1) : (!torch.vtensor<[3],si64>, !torch.vtensor<[2],si64>) -> !torch.vtensor<[2],si64> 
    %12 = torch.operator "onnx.Sub"(%11, %arg1) : (!torch.vtensor<[2],si64>, !torch.vtensor<[2],si64>) -> !torch.vtensor<[2],si64> 
    %13 = torch.operator "onnx.Div"(%12, %0) : (!torch.vtensor<[2],si64>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[2],si64> 
    %14 = torch.operator "onnx.Add"(%13, %arg1) : (!torch.vtensor<[2],si64>, !torch.vtensor<[2],si64>) -> !torch.vtensor<[2],si64> 
    %15 = torch.operator "onnx.Slice"(%9, %13, %14, %1) : (!torch.vtensor<[?,?,?],f32>, !torch.vtensor<[2],si64>, !torch.vtensor<[2],si64>, !torch.vtensor<[2],si64>) -> !torch.vtensor<[10,9,3],f32> 
    return %15 : !torch.vtensor<[10,9,3],f32>
  }
}

