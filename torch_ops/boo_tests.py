"""
# Summary

Convolution tests generated from MIOpen driver commands.

These tests are generated with boo instead of torch
because boo can better control which layouts are selected.
"""

import torch
from iree.turbine.kernel.boo.op_exports.registry import BooOpRegistry

from generate import signature_to_formulas, GenConfig, gen_config

# fmt: off
CONFIGS = [
    ("nhwc_hwcf",                                      "convfp16 -n 2 -c 3 -H 32 -W 32 -k 8 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 --in_layout NHWC --fil_layout HWCN --out_layout NHWC"),
    ("nchw_fchw",                                      "convfp16 -n 2 -c 3 -H 32 -W 32 -k 8 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1"),
    ("nhwc_fhwc",                                      "convfp16 -n 2 -c 3 -H 32 -W 32 -k 8 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC"),
    ("nhwc_bf16_3x3_strided_unaligned_mn_fwd",         "convbfp16 -n 16 -c 64 -H 225 -W 225 -k 64 -y 3 -x 3 -p 1 -q 1 -u 3 -v 3 -l 1 -j 1 -g 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC -F 1"),
    ("nhwc_bf16_3x3_strided_unaligned_mn_bwd_input",    "convbfp16 -n 16 -c 64 -H 225 -W 225 -k 64 -y 3 -x 3 -p 1 -q 1 -u 3 -v 3 -l 1 -j 1 -g 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC -F 2"),
    ("nhwc_bf16_3x3_strided_unaligned_mn_bwd_weight",   "convbfp16 -n 16 -c 64 -H 225 -W 225 -k 64 -y 3 -x 3 -p 1 -q 1 -u 3 -v 3 -l 1 -j 1 -g 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC -F 4"),
    ("nchw_bf16_3x3_strided_unaligned_mn_fwd",          "convbfp16 -n 16 -c 64 -H 225 -W 225 -k 64 -y 3 -x 3 -p 1 -q 1 -u 3 -v 3 -l 1 -j 1 -g 1 --in_layout NCHW --fil_layout NCHW --out_layout NCHW -F 1"),
    ("nchw_bf16_3x3_strided_unaligned_mn_bwd_input",    "convbfp16 -n 16 -c 64 -H 225 -W 225 -k 64 -y 3 -x 3 -p 1 -q 1 -u 3 -v 3 -l 1 -j 1 -g 1 --in_layout NCHW --fil_layout NCHW --out_layout NCHW -F 2"),
    ("nchw_bf16_3x3_strided_unaligned_mn_bwd_weight",   "convbfp16 -n 16 -c 64 -H 225 -W 225 -k 64 -y 3 -x 3 -p 1 -q 1 -u 3 -v 3 -l 1 -j 1 -g 1 --in_layout NCHW --fil_layout NCHW --out_layout NCHW -F 4"),
    ("nhwc_bf16_3x3_unaligned_k_fwd",                   "convbfp16 -n 16 -c 49 -H 192 -W 128 -k 32 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC -F 1"),
    ("nhwc_bf16_3x3_unaligned_k_bwd_input",             "convbfp16 -n 16 -c 49 -H 192 -W 128 -k 32 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC -F 2"),
    ("nhwc_bf16_3x3_unaligned_k_bwd_weight",            "convbfp16 -n 16 -c 49 -H 192 -W 128 -k 32 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC -F 4"),
    ("nchw_bf16_3x3_unaligned_k_fwd",                   "convbfp16 -n 16 -c 49 -H 192 -W 128 -k 32 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1 --in_layout NCHW --fil_layout NCHW --out_layout NCHW -F 1"),
    ("nchw_bf16_3x3_unaligned_k_bwd_input",             "convbfp16 -n 16 -c 49 -H 192 -W 128 -k 32 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1 --in_layout NCHW --fil_layout NCHW --out_layout NCHW -F 2"),
    ("nchw_bf16_3x3_unaligned_k_bwd_weight",            "convbfp16 -n 16 -c 49 -H 192 -W 128 -k 32 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1 --in_layout NCHW --fil_layout NCHW --out_layout NCHW -F 4"),
    ("nhwc_bf16_3x1_dilated_d2_fwd",                    "convbfp16 -n 16 -c 96 -H 48 -W 32 -k 96 -y 3 -x 1 -p 2 -q 0 -u 1 -v 1 -l 2 -j 2 -g 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC -F 1"),
    ("nhwc_bf16_3x1_dilated_d2_bwd_input",              "convbfp16 -n 16 -c 96 -H 48 -W 32 -k 96 -y 3 -x 1 -p 2 -q 0 -u 1 -v 1 -l 2 -j 2 -g 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC -F 2"),
    ("nhwc_bf16_3x1_dilated_d2_bwd_weight",             "convbfp16 -n 16 -c 96 -H 48 -W 32 -k 96 -y 3 -x 1 -p 2 -q 0 -u 1 -v 1 -l 2 -j 2 -g 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC -F 4"),
    ("nchw_bf16_3x1_dilated_d2_fwd",                    "convbfp16 -n 16 -c 96 -H 48 -W 32 -k 96 -y 3 -x 1 -p 2 -q 0 -u 1 -v 1 -l 2 -j 2 -g 1 --in_layout NCHW --fil_layout NCHW --out_layout NCHW -F 1"),
    ("nchw_bf16_3x1_dilated_d2_bwd_input",              "convbfp16 -n 16 -c 96 -H 48 -W 32 -k 96 -y 3 -x 1 -p 2 -q 0 -u 1 -v 1 -l 2 -j 2 -g 1 --in_layout NCHW --fil_layout NCHW --out_layout NCHW -F 2"),
    ("nchw_bf16_3x1_dilated_d2_bwd_weight",             "convbfp16 -n 16 -c 96 -H 48 -W 32 -k 96 -y 3 -x 1 -p 2 -q 0 -u 1 -v 1 -l 2 -j 2 -g 1 --in_layout NCHW --fil_layout NCHW --out_layout NCHW -F 4"),
    ("nhwc_bf16_3x3_grouped_fwd",                       "convbfp16 -n 128 -c 384 -H 48 -W 32 -k 384 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 6 --in_layout NHWC --fil_layout NHWC --out_layout NHWC -F 1"),
    ("nhwc_bf16_3x3_grouped_bwd_input",                 "convbfp16 -n 128 -c 384 -H 48 -W 32 -k 384 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 6 --in_layout NHWC --fil_layout NHWC --out_layout NHWC -F 2"),
    ("nhwc_bf16_3x3_grouped_bwd_weight",                "convbfp16 -n 128 -c 384 -H 48 -W 32 -k 384 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 6 --in_layout NHWC --fil_layout NHWC --out_layout NHWC -F 4"),
    ("nchw_bf16_3x3_grouped_fwd",                       "convbfp16 -n 128 -c 384 -H 48 -W 32 -k 384 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 6 --in_layout NCHW --fil_layout NCHW --out_layout NCHW -F 1"),
    ("nchw_bf16_3x3_grouped_bwd_input",                 "convbfp16 -n 128 -c 384 -H 48 -W 32 -k 384 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 6 --in_layout NCHW --fil_layout NCHW --out_layout NCHW -F 2"),
    ("nchw_bf16_3x3_grouped_bwd_weight",                "convbfp16 -n 128 -c 384 -H 48 -W 32 -k 384 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 6 --in_layout NCHW --fil_layout NCHW --out_layout NCHW -F 4"),
    ("ndhwc_bf16_1x3x3_3d_grouped_fwd",                 "convbfp16 -n 16 -c 288 --in_d 8 -H 48 -W 32 -k 288 --fil_d 1 -y 3 -x 3 --pad_d 0 -p 1 -q 1 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -g 3 --in_layout NDHWC --fil_layout NDHWC --out_layout NDHWC -F 1"),
    ("ndhwc_bf16_1x3x3_3d_grouped_fwd_bias",            "convbfp16 -n 16 -c 288 --in_d 8 -H 48 -W 32 -k 288 --fil_d 1 -y 3 -x 3 --pad_d 0 -p 1 -q 1 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -g 3 -b 1 --in_layout NDHWC --fil_layout NDHWC --out_layout NDHWC -F 1"),
    ("ndhwc_bf16_1x3x3_3d_grouped_bwd_input",           "convbfp16 -n 16 -c 288 --in_d 8 -H 48 -W 32 -k 288 --fil_d 1 -y 3 -x 3 --pad_d 0 -p 1 -q 1 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -g 3 --in_layout NDHWC --fil_layout NDHWC --out_layout NDHWC -F 2"),
    ("ndhwc_bf16_1x3x3_3d_grouped_bwd_weight",          "convbfp16 -n 16 -c 288 --in_d 8 -H 48 -W 32 -k 288 --fil_d 1 -y 3 -x 3 --pad_d 0 -p 1 -q 1 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -g 3 --in_layout NDHWC --fil_layout NDHWC --out_layout NDHWC -F 4"),
    ("ncdhw_bf16_1x3x3_3d_grouped_fwd",                 "convbfp16 -n 16 -c 288 --in_d 8 -H 48 -W 32 -k 288 --fil_d 1 -y 3 -x 3 --pad_d 0 -p 1 -q 1 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -g 3 --in_layout NCDHW --fil_layout NCDHW --out_layout NCDHW -F 1"),
    ("ncdhw_bf16_1x3x3_3d_grouped_fwd_bias",            "convbfp16 -n 16 -c 288 --in_d 8 -H 48 -W 32 -k 288 --fil_d 1 -y 3 -x 3 --pad_d 0 -p 1 -q 1 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -g 3 -b 1 --in_layout NCDHW --fil_layout NCDHW --out_layout NCDHW -F 1"),
    ("ncdhw_bf16_1x3x3_3d_grouped_bwd_input",           "convbfp16 -n 16 -c 288 --in_d 8 -H 48 -W 32 -k 288 --fil_d 1 -y 3 -x 3 --pad_d 0 -p 1 -q 1 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -g 3 --in_layout NCDHW --fil_layout NCDHW --out_layout NCDHW -F 2"),
    ("ncdhw_bf16_1x3x3_3d_grouped_bwd_weight",          "convbfp16 -n 16 -c 288 --in_d 8 -H 48 -W 32 -k 288 --fil_d 1 -y 3 -x 3 --pad_d 0 -p 1 -q 1 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 -g 3 --in_layout NCDHW --fil_layout NCDHW --out_layout NCDHW -F 4"),
    ("nhwc_bf16_3x3_dilated_d4_fwd",                    "convbfp16 -n 16 -c 48 -H 48 -W 32 -k 48 -y 3 -x 3 -p 4 -q 4 -u 1 -v 1 -l 4 -j 4 -g 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC -F 1"),
    ("nhwc_bf16_3x3_dilated_d4_bwd_input",              "convbfp16 -n 16 -c 48 -H 48 -W 32 -k 48 -y 3 -x 3 -p 4 -q 4 -u 1 -v 1 -l 4 -j 4 -g 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC -F 2"),
    ("nhwc_bf16_3x3_dilated_d4_bwd_weight",             "convbfp16 -n 16 -c 48 -H 48 -W 32 -k 48 -y 3 -x 3 -p 4 -q 4 -u 1 -v 1 -l 4 -j 4 -g 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC -F 4"),
    ("nchw_bf16_3x3_dilated_d4_fwd",                    "convbfp16 -n 16 -c 48 -H 48 -W 32 -k 48 -y 3 -x 3 -p 4 -q 4 -u 1 -v 1 -l 4 -j 4 -g 1 --in_layout NCHW --fil_layout NCHW --out_layout NCHW -F 1"),
    ("nchw_bf16_3x3_dilated_d4_bwd_input",              "convbfp16 -n 16 -c 48 -H 48 -W 32 -k 48 -y 3 -x 3 -p 4 -q 4 -u 1 -v 1 -l 4 -j 4 -g 1 --in_layout NCHW --fil_layout NCHW --out_layout NCHW -F 2"),
    ("nchw_bf16_3x3_dilated_d4_bwd_weight",             "convbfp16 -n 16 -c 48 -H 48 -W 32 -k 48 -y 3 -x 3 -p 4 -q 4 -u 1 -v 1 -l 4 -j 4 -g 1 --in_layout NCHW --fil_layout NCHW --out_layout NCHW -F 4"),
    ("nhwc_fp16_3x3_unaligned_mn_fwd",                  "convfp16 -n 32 -c 256 -H 25 -W 25 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC -F 1"),
    ("nhwc_fp16_3x3_unaligned_mn_bwd_input",            "convfp16 -n 32 -c 256 -H 25 -W 25 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC -F 2"),
    ("nhwc_fp16_3x3_unaligned_mn_bwd_weight",           "convfp16 -n 32 -c 256 -H 25 -W 25 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC -F 4"),
    ("nchw_fp16_3x3_unaligned_mn_fwd",                  "convfp16 -n 32 -c 256 -H 25 -W 25 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1 --in_layout NCHW --fil_layout NCHW --out_layout NCHW -F 1"),
    ("nchw_fp16_3x3_unaligned_mn_bwd_input",            "convfp16 -n 32 -c 256 -H 25 -W 25 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1 --in_layout NCHW --fil_layout NCHW --out_layout NCHW -F 2"),
    ("nchw_fp16_3x3_unaligned_mn_bwd_weight",           "convfp16 -n 32 -c 256 -H 25 -W 25 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1 --in_layout NCHW --fil_layout NCHW --out_layout NCHW -F 4"),
]
# fmt: on

for name, cmd_str in CONFIGS:
    signature = BooOpRegistry.parse_command(cmd_str.split())
    args = signature_to_formulas(signature)
    # Reset dynamo state so each config is traced fresh (avoids recompile limit).
    torch.compiler.reset()
    gen_config(signature, GenConfig(name, args=args, seed=1, rtol=5e-3, atol=1e-3))
