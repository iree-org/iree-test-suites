from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any
import itertools

import torch
import iree.turbine.aot as aot
import json


def camel_to_snake(name):
    result = ""
    for i, char in enumerate(name):
        if char.isupper() and i > 0:
            result += "_"
        result += char.lower()
    return result


class TestGenerator(ABC, torch.nn.Module):

    def __init__(self, *args, **kwargs):
        self.path = Path(camel_to_snake("Test" + type(self).__name__))
        self.path.mkdir(exist_ok=True)
        self.test_config = {}
        super().__init__()

    def save_mlir(self, *args):
        exported_module = aot.export(self, *args)
        exported_module.save_mlir(self.path / "test.mlir")

    def save_inputs(self, *args):
        inputs = []
        for idx, input in enumerate(args):
            fname = f"input{idx}.npy"
            path = self.path / fname
            np.save(path, input)
            inputs.append(fname)
        self.test_config["inputs"] = inputs

    def save_results(self, *args):
        expected_outputs = []
        for idx, result in enumerate(args):
            fname = f"result{idx}.npy"
            path = self.path / fname
            np.save(path, result)
            expected_outputs.append(fname)
        self.test_config["expected_outputs"] = expected_outputs

    def save_config(self):
        with open(self.path / "run_module_io_flags.txt", "w") as config:
            for file in self.test_config["inputs"]:
                print("--input=@" + str(file), file=config)
            for file in self.test_config["expected_outputs"]:
                print("--expected_output=@" + str(file), file=config)

    def generate_test(self):
        inputs = self.generate_inputs()
        self.save_mlir(*inputs)
        results = self.generate_expected_value(*inputs)

        self.save_inputs(*inputs)
        self.save_results(*results)
        self.save_config()

    def generate_expected_value(self, *args):
        results = self.forward(*args)
        return [results]

    @abstractmethod
    def generate_inputs(self): ...

    @abstractmethod
    def forward(self, *args): ...


class MatMulOp(TestGenerator):

    def forward(self, left, right):
        return left @ right

    def generate_inputs(self):
        return torch.rand(64, 64), torch.rand(64, 64)


@dataclass
class ModuleRunner:
    module: torch.nn.Module
    """
    Module to be exported.
    The total number of times this module will be run is the cross product
    of argss and kwargss.
    """

    argss: tuple[tuple[torch.Tensor]] | None = field(default=((),))
    """
    Sequence of arguments to `forward`.
    """

    kwargss: tuple[dict[Any]] | None = field(default=({},))
    """
    Sequence of kwargs to `forward`.
    """

    def run(self) -> tuple[Any]:
        return (
            self.module.forward(*args, **kwargs)
            for args, kwargs in itertools.product(self.argss, self.kwargss)
        )


@dataclass
class TestModuleExporter(ModuleRunner):
    path: str | Path = field(default=None)
    """
    Path of exported MLIR module.

    By default it will use the path will be the name of the module.
    """

    export_program_kwargss: tuple[dict[Any]] | None = field(default=({},))
    """
    Sequence kwargs to export_program.

    The total number of times this module will be exported is the
    cross product of argss, kwargss and exported_program_kwargss.

    The name argument in self.export_programs_kwargs is ignored.
    """

    @staticmethod
    def save_function_args(path, export_program_name, args: list[torch.Tensor]):
        inputs = []
        for idx, arg in enumerate(args):
            fname = export_program_name + f"_input{idx}.npy"
            file = path / fname
            np.save(file, arg)
            inputs.append(fname)
        return inputs

    @staticmethod
    def save_function_expected_results(
        path, export_program_name, results: list[torch.Tensor]
    ):
        outputs = []
        for idx, result in enumerate(results):
            fname = export_program_name + f"_result{idx}.npy"
            file = path / fname
            np.save(file, result)
            outputs.append(fname)
        return outputs

    def export(self):
        path = Path(camel_to_snake("Test" + type(self.module).__name__))
        path.mkdir(exist_ok=True)

        fxb = aot.FxProgramsBuilder(self.module)
        config = []

        for idx, (args, kwargs, export_program_kwargs) in enumerate(
            itertools.product(self.argss, self.kwargss, self.export_program_kwargss)
        ):
            export_program_kwargs["name"] = self.module._get_name() + "_" + str(idx)
            entry_point = export_program_kwargs["name"]
            input_files = TestModuleExporter.save_function_args(path, entry_point, args)

            @fxb.export_program(args=args, kwargs=kwargs, **export_program_kwargs)
            def _(module, *args, **kwargs):
                return module.forward(*args, **kwargs)

            expected_output = self.module.forward(*args, **kwargs)
            output_files = TestModuleExporter.save_function_expected_results(
                path, entry_point, (expected_output,)
            )

            config.append(
                {
                    "function": entry_point,
                    "inputs": input_files,
                    "expected_outputs": output_files,
                }
            )

        exported = aot.export(fxb)

        exported.save_mlir(path / "test.mlir")
        with open(path / "run_module_io_flags.json", "w") as config_file:
            json.dump(config, config_file, indent=4)


class MatMulOp2(torch.nn.Module):
    def forward(self, left, right):
        return left + right


if __name__ == "__main__":
    matmulop = MatMulOp()
    matmulop.generate_test()
    TestModuleExporter(
        MatMulOp2(), ((torch.rand(64, 64), torch.rand(64, 64)), (torch.rand(32, 32, dtype=torch.float16), torch.rand(32, 32, dtype=torch.float16)))
    ).export()
