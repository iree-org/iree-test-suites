"""
# Summary

Convolution tests generated from MIOpen driver commands.

These tests are generated with boo instead of torch
because boo can better control which layouts are selected.
"""

import torch
from iree.turbine.kernel.boo.op_exports.registry import BooOpRegistry

from generate import signature_to_formulas, GenConfig, gen_config

# For bf16 we use rtol=2**-7 (i.e. 1 ULP due to 7 mantissa bits).
# (name, rtol, atol, driver_command)
# fmt: off
CONFIGS = [
    ("nhwc_hwcf",                                      5e-3,  1e-3, "convfp16 -n 2 -c 3 -H 32 -W 32 -k 8 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 --in_layout NHWC --fil_layout HWCN --out_layout NHWC"),
    ("nchw_fchw",                                      5e-3,  1e-3, "convfp16 -n 2 -c 3 -H 32 -W 32 -k 8 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1"),
    ("nhwc_fhwc",                                      5e-3,  1e-3, "convfp16 -n 2 -c 3 -H 32 -W 32 -k 8 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC"),
    ("nhwc_bf16_3x3_strided_unaligned_mn_fwd",         2**-7, 1e-3, "convbfp16 -n 16 -c 64 -H 225 -W 225 -k 64 -y 3 -x 3 -p 1 -q 1 -u 3 -v 3 -l 1 -j 1 -g 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC -F 1"),
]
# fmt: on

for name, rtol, atol, cmd_str in CONFIGS:
    signature = BooOpRegistry.parse_command(cmd_str.split())
    args = signature_to_formulas(signature)
    torch.compiler.reset()
    gen_config(signature, GenConfig(name, args=args, seed=1, rtol=rtol, atol=atol))
