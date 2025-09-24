from typing import Callable
import unittest
import inspect
from pathlib import Path
import shutil

import numpy as np
import ml_dtypes

import torch

from torch.testing._internal.common_device_type import (
    ops,
    onlyOn,
    instantiate_device_type_tests,
    OpDTypes,
)
from torch.testing._internal.common_methods_invocations import (
    unary_ufuncs,
    binary_ufuncs,
    reduction_ops,
    shape_funcs,
)

from iree.turbine import aot

# Write generated files to a subfolder.
THIS_DIR = Path(__file__).parent
GENERATED_FILES_OUTPUT_ROOT = THIS_DIR / "torch/node/generated"
IMPORT_SUCCESSES_FILE_PATH = GENERATED_FILES_OUTPUT_ROOT / "import_successes.txt"
IMPORT_FAILURES_FILE_PATH = GENERATED_FILES_OUTPUT_ROOT / "import_failures.txt"


def filter_funcs(ufuncs):
    return [
        x
        for x in ufuncs
        if not x.name.startswith("_refs") and not x.name.startswith("special")
    ]


class Test(unittest.TestCase):
    current_test_name: str
    successes: list[str] = []
    failures: list[str] = []

    def set_test_name(self, name: str):
        self.current_test_name = name

    def _test(self, device: torch.device, dtype: torch.dtype, op, name: str):
        test_dir = GENERATED_FILES_OUTPUT_ROOT / name
        test_dir.mkdir(parents=True, exist_ok=True)
        model_path = test_dir / "model.mlir"

        samples = list(op.sample_inputs(device, dtype))

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.constant_args = [sample.args for sample in samples]
                self.constant_kwargs = [sample.kwargs for sample in samples]

        model = Model()
        fxb = aot.FxProgramsBuilder(model)

        try:
            for i, sample in enumerate(samples):
                test_name = f"{name}_{i}"

                @fxb.export_program(args=(sample.input,), name=test_name)
                def _(root_module, arg):
                    return op(
                        arg,
                        *root_module.constant_args[i],
                        **root_module.constant_kwargs[i],
                    )

            exported = aot.export(fxb)
            exported.save_mlir(model_path)
            for i, sample in enumerate(samples):
                input_path = test_dir / f"input_{i}.npy"
                output_path = test_dir / f"output_{i}.npy"
                expected_output = op(sample.input, *sample.args, **sample.kwargs)
                np.save(input_path, sample.input.cpu().detach().numpy())
                np.save(output_path, expected_output.cpu().detach().numpy())
            print(f"Successfully exported {name}")
            self.successes += [name]
        except Exception as e:
            print(f"Failed to export {name}")
            print(e)
            shutil.rmtree(path)
            self.failures += [name]

    @ops(
        filter_funcs(reduction_ops),
        allowed_dtypes=[
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.int32,
            torch.int64,
        ],
    )
    def test_reduction(self, device: torch.device, dtype: torch.dtype, op):
        self._test(device, dtype, op, self.current_test_name)


instantiate_device_type_tests(Test, globals(), only_for=["cpu"])

cpu_backend = TestCPU
generic_members = set(cpu_backend.__dict__.keys())
generic_tests = [x for x in generic_members if x.startswith("test")]

instance = cpu_backend()
for test in generic_tests:
    # TODO: This kills parallelism, but I really don't know how else to pass this.
    instance.set_test_name(test)
    fun = getattr(cpu_backend, test)
    fun(instance)

with open(IMPORT_SUCCESSES_FILE_PATH, "wt") as f:
    f.write("\n".join(instance.successes))
with open(IMPORT_FAILURES_FILE_PATH, "wt") as f:
    f.write("\n".join(instance.failures))
