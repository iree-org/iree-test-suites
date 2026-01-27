"""
# Summary

File containing convolution tests.
Generates tests for the following layouts.

* NCHW_FCHW
* NHWC_FHWC
* NHWC_HWCF

These tests are generated with boo instead of torch
because boo can better control which layouts are selected.
"""

import os

import torch
from iree.turbine.kernel.boo.op_exports.registry import BooOpRegistry
from iree.turbine.kernel.boo.op_exports.conv import ConvSignature

from generate import signature_to_formulas, GenConfig, gen_config

miopen_driver_command = "convfp16 -n 32 -c 3 -H 224 -W 224 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 --in_layout NHWC --fil_layout HWCN --out_layout NHWC".split()
signature = BooOpRegistry.parse_command(miopen_driver_command)
args = signature_to_formulas(signature)
name = "nhwc_hwcf"
gen_config(signature, GenConfig(name, args=args, seed=1, rtol=1e-3, atol=1e-3))

miopen_driver_command = "convfp16 -n 32 -c 3 -H 224 -W 224 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1".split()
signature = BooOpRegistry.parse_command(miopen_driver_command)
args = signature_to_formulas(signature)
name = "nchw_fchw"
gen_config(signature, GenConfig(name, args=args, seed=1, rtol=1e-3, atol=1e-3))

miopen_driver_command = "convfp16 -n 32 -c 3 -H 224 -W 224 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 --in_layout NHWC --fil_layout NHWC --out_layout NHWC".split()
signature = BooOpRegistry.parse_command(miopen_driver_command)
args = signature_to_formulas(signature)
name = "nhwc_fhwc"
gen_config(signature, GenConfig(name, args=args, seed=1, rtol=1e-3, atol=1e-3))
