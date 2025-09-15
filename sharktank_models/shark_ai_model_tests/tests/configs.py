##################### File For Model Configurations #########################

MODELS = {

    ########################################################################
    # Llama 70b Fp16
    ########################################################################

    "llama-70b-fp16": {
        "irpa": "/shark-dev/70b/instruct/weights/llama3.1_70b_instruct_fp16.irpa",
        "dtype": "fp16",
        "tokenizer": "/shark-dev/70b/instruct/tokenizer.json",
        "tokenizer_config": "/shark-dev/70b/instruct/tokenizer_config.json",
        "kv_dtype": "float16",
        "benchmark_model": "llama-70B-FP16",
        "benchmarks" = [
            ("prefill_bs4", [
                "4x2048xsi64", "4xsi64", "4x64xsi64", "513x5242880xf16"
            ], 2048),

            ("decode_bs4", [
                "4x1xsi64", "4xsi64", "4xsi64", "4x65xsi64", "513x5242880xf16"
            ],2048),
        ],
        "benchmark_repetitions": 3,
        "attention_kernel": "sharktank",
        "device_block_count": "4096",
        "gold_number": "x",
        "bs_prefil": 4,
        "bs_decode": 4,
        "extra_export_flags_list": [],
    },





    ########################################################################
    # Llama 70b Fp8
    ########################################################################

    "llama-70b-fp8": {
        "irpa": "/shark-dev/70b/fp8/attnf8/native_fp8_e4m3fnuz_llama3_70b.irpa",
        "dtype": "fp8",
        "tokenizer": "/shark-dev/70b/instruct/tokenizer.json",
        "tokenizer_config": "/shark-dev/70b/instruct/tokenizer_config.json",
        "kv_dtype": "float8_e4m3fnuz",
        "benchmark_model": "llama-70B-FP8",
        "device_block_count": "4096",
        "gold_number": "x",
        "bs_prefil": 4,
        "bs_decode": 4,
        "extra_export_flags_list": [],






    ########################################################################
    # Llama 8b Fp16
    ########################################################################

    },
    "llama-8b-fp16": {
        "irpa": "/shark-dev/8b/instruct/weights/llama3.1_8b_instruct_fp16.irpa",
        "dtype": "fp16",
        "tokenizer": "/shark-dev/8b/instruct/tokenizer.json",
        "tokenizer_config": "/shark-dev/8b/instruct/tokenizer_config.json",
        "kv_dtype": "float16",
        "benchmark_model": "llama-8B-FP16",
        "benchmarks" = [
            ("prefill_bs4", [
                "@/shark-dev/8b/prefill_args_bs4_2048_stride_32/tokens.npy",
                "@/shark-dev/8b/prefill_args_bs4_2048_stride_32/seq_lens.npy",
                "@/shark-dev/8b/prefill_args_bs4_2048_stride_32/seq_block_ids.npy",
                "@/shark-dev/8b/prefill_args_bs4_2048_stride_32/cs_f16.npy"
            ], 2048),
            ("decode_bs4", [
                "@/shark-dev/8b/decode_args_bs4_2048_stride_32/next_tokens.npy",
                "@/shark-dev/8b/decode_args_bs4_2048_stride_32/seq_lens.npy",
                "@/shark-dev/8b/decode_args_bs4_2048_stride_32/start_positions.npy",
                "@/shark-dev/8b/decode_args_bs4_2048_stride_32/seq_block_ids.npy",
                "@/shark-dev/8b/decode_args_bs4_2048_stride_32/cs_f16.npy"
            ], 2048),
        ],
        "benchmark_repetitions"=3,
        "attention_kernel": "sharktank",
        "device_block_count": "4096",
        "gold_number": "x",
        "bs_prefil": 4,
        "bs_decode": 4,
        "extra_export_flags_list": [],
    },





    ########################################################################
    # Llama 8b Fp8
    ########################################################################

    "llama-8b-fp8": {
        "irpa": "/shark-dev/8b/fp8/attnf8/native_fp8_e4m3fnuz_llama3_8b.irpa",
        "dtype": "fp8",
        "tokenizer": "/shark-dev/8b/instruct/tokenizer.json",
        "tokenizer_config": "/shark-dev/8b/instruct/tokenizer_config.json",
        "kv_dtype": "float8_e4m3fnuz",
        "benchmark_model": "llama-8B-FP8",
        "benchmarks" = [
            ("prefill_bs4", [
                "4x2048xi64 4xi64 4x64xi64 261x2097152xf8E4M3FNUZ"
            ], 2048),
            ("decode_bs4", [
                "4x1xi64 4xi64 4xi64 4x65xi64 261x2097152xf8E4M3FNUZ"
            ], 2048),
        ],
        "benchmark_repetitions": 3,
        "attention_kernel": "sharktank",
        "device_block_count": "4096",
        "gold_number": "x",
        "bs_prefil": 4,
        "bs_decode": 4,
        "extra_export_flags_list": [],
    },





    ########################################################################
    # Mistral Nemo Instruct 2407 FP8
    ########################################################################

    "mistral": {
        "irpa": "/shark-dev/mistral_instruct/instruct.irpa",
        "dtype": "mistral_fp8",
        "tokenizer": "/shark-dev/mistral_instruct/tokenizer.json",
        "tokenizer_config": "/shark-dev/mistral_instruct/tokenizer_config.json",
        "kv_dtype": "float8_e4m3fnuz",
        "benchmark_model": "mistral-nemo-instruct-fp8",
        "benchmarks" = [
            ("prefill_bs4", [
                "4x2048xsi64 4xsi64 4x32xsi64 2048x2621440xf8E4M3FNUZ"
            ], 2048),
            ("decode_bs32", [
                "32x1xsi64 32xsi64 32xsi64 32x32xsi64 2048x2621440xf8E4M3FNUZ"
            ], 2048),
        ],
        "benchmark_repetitions": 5,
        "attention_kernel" "torch",
        "device_block_count": "4096",
        "gold_number": "x",
        "bs_prefil": 4,
        "bs_decode": 32,
        "extra_export_flags_list": [],
    },
}