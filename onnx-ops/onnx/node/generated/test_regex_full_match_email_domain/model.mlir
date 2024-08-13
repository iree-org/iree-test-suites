module {
  func.func @test_regex_full_match_email_domain(%arg0: !torch.vtensor<[2,2],!torch.str>) -> !torch.vtensor<[2,2],i1> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.RegexFullMatch"(%arg0) {torch.onnx.pattern = "(\\W|^)[\\w.\\-]{0,25}@(yahoo|gmail)\\.com(\\W|$)"} : (!torch.vtensor<[2,2],!torch.str>) -> !torch.vtensor<[2,2],i1> 
    return %0 : !torch.vtensor<[2,2],i1>
  }
}

