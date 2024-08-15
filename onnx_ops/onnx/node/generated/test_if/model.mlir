module {
  func.func @test_if(%arg0: !torch.vtensor<[],i1>) -> !torch.vtensor<[5],f32> attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.If"(%arg0) : (!torch.vtensor<[],i1>) -> !torch.vtensor<[5],f32> {
      %1 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<_> : tensor<5xf32>} : () -> !torch.vtensor<[5],f32> 
      torch.operator_terminator %1 : !torch.vtensor<[5],f32>
    }, {
      %1 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<__1> : tensor<5xf32>} : () -> !torch.vtensor<[5],f32> 
      torch.operator_terminator %1 : !torch.vtensor<[5],f32>
    }
    return %0 : !torch.vtensor<[5],f32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      _: "0x080000000000A0400000804000004040000000400000803F",
      __1: "0x080000000000803F0000004000004040000080400000A040"
    }
  }
#-}

