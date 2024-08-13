module {
  func.func @test_tfidfvectorizer_tf_uniandbigrams_skip5(%arg0: !torch.vtensor<[12],si32>) -> !torch.vtensor<[7],f32> attributes {torch.onnx_meta.ir_version = 4 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.TfIdfVectorizer"(%arg0) {torch.onnx.max_gram_length = 2 : si64, torch.onnx.max_skip_count = 5 : si64, torch.onnx.min_gram_length = 1 : si64, torch.onnx.mode = "TF", torch.onnx.ngram_counts = [0 : si64, 4 : si64], torch.onnx.ngram_indexes = [0 : si64, 1 : si64, 2 : si64, 3 : si64, 4 : si64, 5 : si64, 6 : si64], torch.onnx.pool_int64s = [2 : si64, 3 : si64, 5 : si64, 4 : si64, 5 : si64, 6 : si64, 7 : si64, 8 : si64, 6 : si64, 7 : si64]} : (!torch.vtensor<[12],si32>) -> !torch.vtensor<[7],f32> 
    return %0 : !torch.vtensor<[7],f32>
  }
}

