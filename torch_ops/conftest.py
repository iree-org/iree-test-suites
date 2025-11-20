# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass
from pathlib import Path
import typing
import numbers
import csv
import logging
import json
import os
import pytest
import subprocess
import glob
import numpy as np
import os

from iree import compiler as ireec
from utils import customJSONDecoder, Formula

THIS_DIR = Path(__file__).parent
TEST_DATA_FLAGFILE_NAME = "run_module_io_flags.json"


def pytest_addoption(parser):
    # List of configuration files following this schema:
    #   {
    #     "config_name": str,
    #     "iree_compile_flags": list of str,
    #     "iree_run_module_flags": list of str,
    #     "skip_compile_tests": list of str,
    #     "skip_run_tests": list of str,
    #     "expected_compile_failures": list of str,
    #     "expected_run_failures": list of str
    #   }
    #
    # For example, to test on CPU with the `llvm-cpu` backend and `local-task` device:
    #   {
    #     "config_name": "cpu_llvm_task",
    #     "iree_compile_flags": ["--iree-hal-target-backends=llvm-cpu"],
    #     "iree_run_module_flags": ["--device=local-task"],
    #     "skip_compile_tests": [],
    #     "skip_run_tests": [],
    #     "expected_compile_failures": ["test_abs"],
    #     "expected_run_failures": ["test_add"],
    #   }
    #
    # The list of files can be specified in (by order of preference):
    #   1. The `--config-files` argument
    #       e.g. `pytest ... --config-files foo.json bar.json`
    #   2. The `IREE_TEST_CONFIG_FILES` environment variable
    #       e.g. `set IREE_TEST_CONFIG_FILES=foo.json;bar.json`
    #   3. A default config file used for testing the test suite itself
    default_config_files = [
        f for f in os.getenv("IREE_TEST_CONFIG_FILES", "").split(";") if f
    ]
    if not default_config_files:
        default_config_files = [
            THIS_DIR / "configs" / "torch_ops_cpu_llvm_sync.json",
        ]
    parser.addoption(
        "--config-files",
        action="store",
        nargs="*",
        default=default_config_files,
        help="List of config JSON files used to build test cases",
    )

    parser.addoption(
        "--ignore-xfails",
        action="store_true",
        default=False,
        help="Ignores expected compile/run failures from configs, to print all error output",
    )

    parser.addoption(
        "--skip-all-runs",
        action="store_true",
        default=False,
        help="Skips all 'run' tests, overriding 'skip_run_tests' in configs",
    )


def pytest_sessionstart(session):
    session.config.iree_test_configs = []
    for config_file in session.config.getoption("config_files"):
        with open(config_file) as f:
            test_config = json.load(f)

            # Sanity check the config file structure before going any further.
            def check_field(field_name):
                if field_name not in test_config:
                    raise ValueError(
                        f"config file '{config_file}' is missing a '{field_name}' field"
                    )

            check_field("config_name")
            check_field("iree_compile_flags")
            check_field("iree_run_module_flags")

            session.config.iree_test_configs.append(test_config)


def pytest_collect_file(parent, file_path):
    """A test corresponds to a file named ${TEST_DATA_FLAGFILE_NAME}.

    In order for the test to succeed though, the test should have the
    folder structure expected. Which is:

    ../${MODULE_NAME}/${TEST_NAME}/{correcntess,benchmark}/${TEST_DATA_FLAGFILE_NAME}

    The ${MODULE_NAME} corresponds to the MLIR module and this is where the MLIR module
    should be stored.
    """
    if file_path.name == TEST_DATA_FLAGFILE_NAME:
        return MlirCompileRunTest.from_parent(parent, path=file_path)


@dataclass(frozen=True)
class IreeCompileAndRunTestSpec:
    """Specification for an IREE "compile and run" test."""

    # full path containing the test (i.e., run_io_module.json file).
    test_directory: Path

    # full path to file
    input_mlir_file: Path

    # Name of input MLIR file in a format accepted by IREE (e.g. torch, tosa, or linalg dialect).
    # Including file suffix, e.g. 'model.mlir' or 'model.mlirbc'.
    input_mlir_name: str

    # Stem of input MLIR file, excluding file suffix, e.g. 'model'.
    input_mlir_stem: str

    # Name of flagfile in the same directory as the input MLIR, containing flags like:
    #   --input=i64=@input_0.bin
    #   --expected_output=i64=@output_0.bin
    data_flagfile_name: str

    # Name of the test configuration, e.g. "cpu_llvm_sync".
    # This will be used in generated files and test case names.
    test_name: str

    # Flags to pass to `iree-compile`, e.g. ["--iree-hal-target-backends=llvm-cpu"].
    iree_compile_flags: typing.List[str]

    # Flags to pass to `iree-run-module`, e.g. ["--device=local-task"].
    # These will be passed in addition to `--flagfile={data_flagfile_name}`.
    iree_run_module_flags: typing.List[str]

    # True if compilation is expected to succeed. If false, the test will be marked XFAIL.
    expect_compile_success: bool

    # True if running is expected to succeed. If false, the test will be marked XFAIL.
    expect_run_success: bool

    # Golden time for benchmark runs
    golden_time_ms: typing.Dict[str, float]

    # True to only compile the test and skip running.
    skip_run: bool

    # Json decoded dictionary
    json_obj: typing.Dict

    # Arguments to be passed to the function being tested.
    args: None | typing.List[typing.Any]

    # Key word arguments to be passed to the function being tested.
    kwargs: None | typing.Dict


@dataclass(frozen=True)
class IreeCorrectnessTestSpec(IreeCompileAndRunTestSpec):
    # List of expected output.
    # We expect a list of npy files.
    expected_output: typing.List[str]


class MlirCompileRunTest(pytest.File):
    """Collector for MLIR -> compile -> run tests anchored on a file."""

    @dataclass(frozen=True)
    class TestCase:
        mlir_file: Path
        runtime_flagfile: str

    def discover_test_cases(self):
        """Discovers test cases with run_module_io_flags.txt files.

        We expect the following structure

        generated/${CLASSNAME}/${FILE}.mlir
        generated/${CLASSNAME}/{correctness,benchmark}/${TESTNAME}/${TEST_DATA_FLAGFILE_NAME}

        Multiple correctness and benchmark tests may share an MLIR module.
        """
        test_cases = []

        mlir_files = sorted(self.path.parent.parent.parent.glob("*.mlir*"))
        assert len(mlir_files) <= 1, "Test directories may only contain one .mlir file"
        mlir_file = mlir_files[0].resolve()

        if self.path.name == TEST_DATA_FLAGFILE_NAME:
            test_cases.append(
                MlirCompileRunTest.TestCase(
                    mlir_file=mlir_file,
                    runtime_flagfile=TEST_DATA_FLAGFILE_NAME,
                )
            )

        return test_cases

    def collect(self):
        # This test directory corresponds to the path
        # generated/${CLASSNAME}/{correctness,benchmark}/${TESTNAME}/
        test_directory = self.path.parent.resolve()

        relative_test_directory = test_directory.relative_to(THIS_DIR).as_posix()
        test_directory_name = test_directory.name

        test_cases = self.discover_test_cases()
        if len(test_cases) == 0:
            logging.getLogger().debug(f"No test cases for '{test_directory_name}'")
            return []

        for config in self.config.iree_test_configs:
            if relative_test_directory in config.get("skip_compile_tests", []):
                continue

            expect_compile_success = self.config.getoption(
                "ignore_xfails"
            ) or relative_test_directory not in config.get(
                "expected_compile_failures", []
            )
            expect_run_success = self.config.getoption(
                "ignore_xfails"
            ) or relative_test_directory not in config.get("expected_run_failures", [])
            skip_run = self.config.getoption(
                "skip_all_runs"
            ) or relative_test_directory in config.get("skip_run_tests", [])
            config_name = config["config_name"]

            for test_case in test_cases:
                # Generate test item names like 'model.mlir::cpu_llvm_sync'.
                # These show up in pytest output.
                mlir_file = test_case.mlir_file
                name_parts = [e for e in [mlir_file.name, config_name] if e]
                item_name = "::".join(name_parts)

                with open(test_directory / self.path, "r") as json_file:
                    json_obj = json.load(json_file, object_hook=customJSONDecoder)

                spec = IreeCompileAndRunTestSpec(
                    test_directory=test_directory,
                    input_mlir_file=mlir_file,
                    input_mlir_name=mlir_file.name,
                    input_mlir_stem=mlir_file.stem,
                    data_flagfile_name=test_case.runtime_flagfile,
                    test_name=config_name,
                    iree_compile_flags=config["iree_compile_flags"],
                    iree_run_module_flags=config["iree_run_module_flags"],
                    expect_compile_success=expect_compile_success,
                    expect_run_success=expect_run_success,
                    golden_time_ms=config["golden_times_ms"],
                    skip_run=skip_run,
                    json_obj=json_obj,
                    args=json_obj["args"],
                    kwargs=json_obj["kwargs"],
                )

                marker = json_obj["marker"]

                marker_name, marker_args, marker_kwargs = marker.values()
                marker_callable = getattr(pytest.mark, marker_name)
                if marker_name == "correctness_test":
                    test = IreeCompileRunItem.from_parent(
                        self, name=item_name, spec=spec
                    )
                elif marker_name == "benchmark_test":
                    test = IreeBenchmarkItem.from_parent(
                        self, name=item_name, spec=spec
                    )
                test.add_marker(marker_callable(*marker_args, **marker_kwargs))
                yield test


class IreeBaseTest(pytest.Item):
    """Test invocation item for an IREE test case."""

    spec: IreeCompileAndRunTestSpec

    def __init__(self, spec, **kwargs):
        super().__init__(**kwargs)
        self.spec = spec

        relative_test_directory = self.spec.test_directory.relative_to(
            THIS_DIR
        ).as_posix()
        self.user_properties.append(
            ("relative_test_directory_name", relative_test_directory)
        )
        self.user_properties.append(("input_mlir_name", self.spec.input_mlir_file))
        self.user_properties.append(("test_name", self.spec.test_name))
        self.vmfb_name = f"{self.spec.input_mlir_stem}_{self.spec.test_name}.vmfb"
        self.test_cwd = self.spec.test_directory

        self.pregen_args = self.spec.args
        self.pregen_kwargs = self.spec.kwargs
        self.json_obj = self.spec.json_obj

        compile_args = ["iree-compile", self.spec.input_mlir_file]
        compile_args.extend(self.spec.iree_compile_flags)
        compile_args.extend(["-o", self.vmfb_name])
        self.compile_cmd = subprocess.list2cmdline(compile_args)

    def test_compile(self):
        cwd = self.test_cwd
        logging.getLogger().info(
            f"Launching compile command:\n" f"cd {cwd} && {self.compile_cmd}"  #
        )
        proc = subprocess.run(
            self.compile_cmd, shell=True, capture_output=True, cwd=cwd
        )
        if proc.returncode != 0:
            raise IreeCompileException(
                process=proc,
                cwd=cwd,
                input_mlir_file=self.spec.input_mlir_file,
                compile_cmd=self.compile_cmd,
            )
        return self.vmfb_name

    def repr_failure(self, excinfo):
        """Called when self.runtest() raises an exception."""
        return super().repr_failure(excinfo)

    def reportinfo(self):
        display_name = (
            f"{self.path.parent.name}::{self.spec.input_mlir_name}::{self.name}"
        )
        return self.path, 0, f"IREE compile and run: {display_name}"

    # Defining this for pytest-retry to avoid an AttributeError.
    def _initrequest(self):
        pass

    def generate_values(self, *pregen_args, **pregen_kwargs):
        args = []
        for pregen_arg in pregen_args:
            if isinstance(pregen_arg, Formula):
                args.append(pregen_arg.numpy())
            else:
                args.append(pregen_arg)

        kwargs = {}
        for pregen_key, pregen_val in pregen_kwargs.items():
            if isinstance(pregen_val, Formula):
                kwargs[pregen_key] = pregen_val.numpy()
            else:
                kwargs[pregen_key] = pregen_val

        return args, kwargs

    def generate_input_npy_files(self, *args, **kwargs):
        if kwargs:
            raise ValueError("Cannot handle kwargs yet")

        new_args = []
        for idx, arg in enumerate(args):
            path = self.test_cwd / f"input_{idx}.npy"
            np.save(path, arg)
            new_args.append(path)

        return new_args, {}

    def runtest(self):
        # We want to test two phases: 'compile', and 'run'.
        # A test can be marked as expected to fail at either stage, with these
        # possible outcomes:

        # Expect 'compile' | Expect 'run' | Actual 'compile' | Actual 'run' | Result
        # ---------------- | ------------ | ---------------- | ------------ | ------
        #
        # PASS             | PASS         | PASS             | PASS         | PASS
        # PASS             | PASS         | FAIL             | N/A          | FAIL
        # PASS             | PASS         | PASS             | FAIL         | FAIL
        #
        # PASS             | FAIL         | PASS             | PASS         | XPASS
        # PASS             | FAIL         | FAIL             | N/A          | FAIL
        # PASS             | FAIL         | PASS             | FAIL         | XFAIL
        #
        # FAIL             | N/A          | PASS             | PASS         | XPASS
        # FAIL             | N/A          | FAIL             | N/A          | XFAIL
        # FAIL             | N/A          | PASS             | FAIL         | XPASS

        # * XFAIL and PASS are acceptable outcomes - they mean that the list of
        #   expected failures in the config file matched the test run.
        # * FAIL means that something expected to work did not. That's an error.
        # * XPASS means that a test is newly passing and can be removed from the
        #   expected failures list.

        if not self.spec.expect_compile_success:
            self.add_marker(
                pytest.mark.xfail(
                    raises=IreeCompileException,
                    strict=True,
                    reason="Expected compilation to fail (included in 'expected_compile_failures')",
                )
            )
        if not self.spec.expect_run_success:
            self.add_marker(
                pytest.mark.xfail(
                    raises=IreeRunException,
                    strict=True,
                    reason="Expected run to fail (included in 'expected_run_failures')",
                )
            )

        vmfb = self.test_compile()

        if self.spec.skip_run:
            return

        self.test_run(vmfb)


class IreeCompileRunItem(IreeBaseTest):
    """Test invocation item for an IREE compile + run test case."""

    def initialize_correctness_test(self):
        marker = self.get_closest_marker("correctness_test")

        np.random.seed(marker.kwargs["seed"])
        self.entry_point = marker.kwargs["entry_point"]
        self.rtol = marker.kwargs.get("rtol", 1e-05)
        self.atol = marker.kwargs.get("atol", 1e-08)
        self.equal_nan = marker.kwargs.get("equal_nan", False)
        self.expected_results = self.json_obj["expected_results"]

    def test_run(self, vmfb):
        self.initialize_correctness_test()
        run_args = ["iree-run-module", f"--module={str(vmfb)}"]
        run_args.extend(self.spec.iree_run_module_flags)
        run_args.extend([f"--function={self.entry_point}"])

        args, kwargs = self.generate_values(*self.pregen_args, **self.pregen_kwargs)
        input_npy_files, input_kwargs = self.generate_input_npy_files(*args, **kwargs)

        for input in input_npy_files:
            run_args.append(f"--input=@{input}")

        cwd = self.test_cwd

        observed = []
        for idx, expected_result in enumerate(self.expected_results):
            observed_results_file = f"observed_outputs{idx}.npy"
            observed.append(f"{observed_results_file}")

        for obs in observed:
            run_args.append(f"--output=@{obs}")

        self.run_cmd = subprocess.list2cmdline(run_args)

        logging.getLogger().info(
            f"Launching run command:\n" f"cd {cwd} && {self.run_cmd}"  #
        )

        proc = subprocess.run(self.run_cmd, shell=True, capture_output=True, cwd=cwd)
        if proc.returncode != 0:
            raise IreeRunException(
                cwd=cwd,
                process=proc,
                input_mlir_file=self.spec.input_mlir_file,
                compile_cmd=self.compile_cmd,
                run_cmd=self.run_cmd,
            )

        expected = self.expected_results
        for exp, obs in zip(expected, observed, strict=True):
            exp_arr = np.load(cwd / exp)
            obs_arr = np.load(cwd / obs)
            assert np.allclose(
                exp_arr,
                obs_arr,
                rtol=self.rtol,
                atol=self.atol,
                equal_nan=self.equal_nan,
            )


# TODO(@amd-eochoalo): Have a version of this that does not depend
# on rocprofv3 for different targets. Once gpu timestamp is available
# Remove rocprofv3 dependency and use iree-benchmark-{module,executable}
class IreeBenchmarkItem(IreeBaseTest):
    """Test invocation item for an IREE compile + run test case."""

    def initialize_benchmark_test(self):
        marker = self.get_closest_marker("benchmark_test")
        np.random.seed(marker.kwargs["seed"])
        self.entry_point = marker.kwargs["entry_point"]

    def test_run(self, vmfb):
        self.initialize_benchmark_test()
        parent = self.test_cwd.parent
        grand_parent = parent.parent
        golden_time_key = f"{grand_parent.name}/{parent.name}"
        run_args = [
            "iree-benchmark-module",
            "--benchmark_format=json",
            f"--module={str(vmfb)}",
        ]
        run_args.extend(self.spec.iree_run_module_flags)
        run_args.extend([f"--function={self.entry_point}"])

        args, kwargs = self.generate_values(*self.pregen_args, **self.pregen_kwargs)
        input_npy_files, input_kwargs = self.generate_input_npy_files(*args, **kwargs)

        for input in input_npy_files:
            run_args.append(f"--input=@{input}")

        cwd = self.test_cwd

        self.run_cmd = subprocess.list2cmdline(run_args)

        logging.getLogger().info(
            f"Launching run command:\n" f"cd {cwd} && {self.run_cmd}"  #
        )

        proc = subprocess.run(self.run_cmd, shell=True, capture_output=True, cwd=cwd)
        if proc.returncode != 0:
            raise IreeRunException(
                cwd=cwd,
                process=proc,
                input_mlir_file=self.spec.input_mlir_file,
                compile_cmd=self.compile_cmd,
                run_cmd=self.run_cmd,
            )

        json_obj = json.loads(proc.stdout.decode("utf-8"))
        observed_time = json_obj["benchmarks"][0]["real_time"]
        expected_golden_time = self.spec.golden_time_ms[golden_time_key]
        assert observed_time <= (expected_golden_time * 1.1)


class IreeCompileException(Exception):
    """Compiler exception that preserves the command line and output."""

    def __init__(
        self,
        process: subprocess.CompletedProcess,
        cwd: Path,
        input_mlir_file: Path,
        compile_cmd: str,
    ):
        try:
            errs = process.stderr.decode("utf-8")
        except:
            errs = str(process.stderr)
        try:
            outs = process.stdout.decode("utf-8")
        except:
            outs = str(process.stdout)

        test_github_url = (
            "https://github.com/iree-org/iree-test-suites/blob/main/torch_ops/"
            + cwd.relative_to(THIS_DIR).as_posix()
        )

        with open(input_mlir_file) as f:
            input_mlir = f.read()

            super().__init__(
                f"Error invoking iree-compile\n"
                f"Error code: {process.returncode}\n"
                f"Stderr diagnostics:\n{errs}\n\n"
                f"Stdout diagnostics:\n{outs}\n\n"
                f"Test case source:\n"
                f"  {test_github_url}\n\n"
                f"Input program:\n"
                f"```\n{input_mlir}```\n\n"
                f"Compiled with:\n"
                f"  cd {cwd} && {compile_cmd}\n\n"
            )


class IreeRunException(Exception):
    """Runtime exception that preserves the command line and output."""

    def __init__(
        self,
        cwd: Path,
        process: subprocess.CompletedProcess,
        input_mlir_file: Path,
        compile_cmd: str,
        run_cmd: str,
    ):
        try:
            errs = process.stderr.decode("utf-8")
        except:
            errs = str(process.stderr)
        try:
            outs = process.stdout.decode("utf-8")
        except:
            outs = str(process.stdout)

        test_github_url = (
            "https://github.com/iree-org/iree-test-suites/blob/main/torch_ops/"
            + cwd.relative_to(THIS_DIR).as_posix()
        )

        with open(input_mlir_file) as f:
            input_mlir = f.read()

            super().__init__(
                f"Error invoking iree-run-module\n"
                f"Error code: {process.returncode}\n"
                f"Stderr diagnostics:\n{errs}\n"
                f"Stdout diagnostics:\n{outs}\n"
                f"Test case source:\n"
                f"  {test_github_url}\n\n"
                f"Input program:\n"
                f"```\n{input_mlir}```\n\n"
                f"Compiled with:\n"
                f"  cd {cwd} && {compile_cmd}\n\n"
                f"Run with:\n"
                f"  cd {cwd} && {run_cmd}\n\n"
            )


class IreeXFailCompileRunException(Exception):
    pass
