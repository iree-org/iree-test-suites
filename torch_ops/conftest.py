# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass
from pathlib import Path
from typing import List
import csv
import logging
import pyjson5
import os
import pytest
import subprocess
import glob
import numpy as np

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
            test_config = pyjson5.load(f)

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
    if file_path.name == TEST_DATA_FLAGFILE_NAME:
        return MlirCompileRunTest.from_parent(parent, path=file_path)

    if file_path.suffix == ".json":
        with open(file_path) as f:
            test_cases_json = pyjson5.load(f)
            if test_cases_json.get("file_format", "") == "test_cases_v0":
                return MlirCompileRunTest.from_parent(parent, path=file_path)


@dataclass(frozen=True)
class IreeCompileAndRunTestSpec:
    """Specification for an IREE "compile and run" test."""

    # Directory where test input files are located.
    test_directory: Path

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
    iree_compile_flags: List[str]

    # Flags to pass to `iree-run-module`, e.g. ["--device=local-task"].
    # These will be passed in addition to `--flagfile={data_flagfile_name}`.
    iree_run_module_flags: List[str]

    # True if compilation is expected to succeed. If false, the test will be marked XFAIL.
    expect_compile_success: bool

    # True if running is expected to succeed. If false, the test will be marked XFAIL.
    expect_run_success: bool

    # True to only compile the test and skip running.
    skip_run: bool


class MlirCompileRunTest(pytest.File):
    """Collector for MLIR -> compile -> run tests anchored on a file."""

    @dataclass(frozen=True)
    class TestCase:
        mlir_file: str
        runtime_flagfile: str

    def discover_test_cases(self):
        """Discovers test cases with run_module_io_flags.txt files."""
        test_cases = []

        mlir_files = sorted(self.path.parent.glob("*.mlir*"))
        assert len(mlir_files) <= 1, "Test directories may only contain one .mlir file"
        mlir_file = mlir_files[0]

        if self.path.name == TEST_DATA_FLAGFILE_NAME:
            test_cases.append(
                MlirCompileRunTest.TestCase(
                    mlir_file=mlir_file,
                    runtime_flagfile=TEST_DATA_FLAGFILE_NAME,
                )
            )

        return test_cases

    def collect(self):
        # Expected directory structure:
        #   path/to/test_abs/
        #     - *.mlir[bc]
        #     - run_module_io_flags.txt
        #   path/to/test_add/
        #     - *.mlir[bc]
        #     - run_module_io_flags.txt

        test_directory = self.path.parent
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

                spec = IreeCompileAndRunTestSpec(
                    test_directory=test_directory,
                    input_mlir_name=mlir_file.name,
                    input_mlir_stem=mlir_file.stem,
                    data_flagfile_name=test_case.runtime_flagfile,
                    test_name=config_name,
                    iree_compile_flags=config["iree_compile_flags"],
                    iree_run_module_flags=config["iree_run_module_flags"],
                    expect_compile_success=expect_compile_success,
                    expect_run_success=expect_run_success,
                    skip_run=skip_run,
                )
                with open(test_directory / spec.data_flagfile_name, "r") as config:
                    json = pyjson5.load(config)

                markers = json["markers"]

                for marker_name, args_kwargs in markers.items():
                    marker_callable = getattr(pytest.mark, marker_name)
                    marker_args = args_kwargs["args"]
                    marker_kwargs = args_kwargs["kwargs"]
                    if marker_name == "correctness":
                        test = IreeCompileRunItem.from_parent(
                            self, name=item_name, spec=spec
                        )
                    elif marker_name == "benchmark":
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
        self.user_properties.append(("input_mlir_name", self.spec.input_mlir_name))
        self.user_properties.append(("test_name", self.spec.test_name))

        self.test_cwd = self.spec.test_directory

        vmfb_name = f"{self.spec.input_mlir_stem}_{self.spec.test_name}.vmfb"

        compile_args = ["iree-compile", self.spec.input_mlir_name]
        compile_args.extend(self.spec.iree_compile_flags)
        compile_args.extend(["-o", vmfb_name])
        self.compile_cmd = subprocess.list2cmdline(compile_args)

        run_args = ["iree-run-module", f"--module={vmfb_name}"]
        run_args.extend(self.spec.iree_run_module_flags)
        with open(self.test_cwd / self.spec.data_flagfile_name, "r") as config:
            json = pyjson5.load(config)
        self.markers = json["markers"]
        for input in json["inputs"]:
            run_args.append(f"--input=@{input}")
        for output in json["observed_outputs"]:
            run_args.append(f"--output=@{output}")
        self.run_cmd = run_args
        self.run_options = json

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
                input_mlir_name=self.spec.input_mlir_name,
                compile_cmd=self.compile_cmd,
            )

    def repr_failure(self, excinfo):
        """Called when self.runtest() raises an exception."""
        if isinstance(excinfo.value, (IreeCompileException, IreeRunException)):
            return "\n".join(excinfo.value.args)
        if isinstance(excinfo.value, IreeXFailCompileRunException):
            return (
                "Expected compile failure but run failed (move to 'expected_run_failures'):\n"
                + "\n".join(excinfo.value.__cause__.args)
            )
        return super().repr_failure(excinfo)

    def reportinfo(self):
        display_name = (
            f"{self.path.parent.name}::{self.spec.input_mlir_name}::{self.name}"
        )
        return self.path, 0, f"IREE compile and run: {display_name}"

    # Defining this for pytest-retry to avoid an AttributeError.
    def _initrequest(self):
        pass

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

        self.test_compile()

        if self.spec.skip_run:
            return

        try:
            self.test_run()
        except IreeRunException as e:
            if not self.spec.expect_compile_success:
                raise IreeXFailCompileRunException from e
            raise e


class IreeCompileRunItem(IreeBaseTest):
    """Test invocation item for an IREE compile + run test case."""

    def test_run(self):
        cwd = self.test_cwd
        logging.getLogger().info(
            f"Launching run command:\n" f"cd {cwd} && {self.run_cmd}"  #
        )

        self.run_cmd = subprocess.list2cmdline(self.run_cmd)
        proc = subprocess.run(self.run_cmd, shell=True, capture_output=True, cwd=cwd)
        if proc.returncode != 0:
            raise IreeRunException(
                process=proc,
                cwd=cwd,
                input_mlir_name=self.spec.input_mlir_name,
                compile_cmd=self.compile_cmd,
                run_cmd=self.run_cmd,
            )

        observed = self.run_options["observed_outputs"]
        expected = self.run_options["expected_outputs"]
        expected.sort()
        observed.sort()
        rtol = self.run_options["rtol"]
        atol = self.run_options["atol"]
        equal_nan = self.run_options["equal_nan"]
        for exp, obs in zip(expected, observed, strict=True):
            exp_arr = np.load(cwd / exp)
            obs_arr = np.load(cwd / obs)
            assert np.allclose(
                exp_arr, obs_arr, rtol=rtol, atol=atol, equal_nan=equal_nan
            )


class IreeBenchmarkItem(IreeBaseTest):
    """Test invocation item for an IREE compile + run test case."""

    def test_run(self):
        cwd = self.test_cwd
        logging.getLogger().info(
            f"Launching run command:\n" f"cd {cwd} && {self.run_cmd}"  #
        )

        # TODO(@amd-eochoalo): investigate how to use the json
        # format.
        outfile = "test"
        self.run_cmd = [
            "rocprofv3",
            "--kernel-trace",
            "--output-file",
            outfile,
            "--output-format",
            "csv",
            "--",
        ] + self.run_cmd
        self.run_cmd = subprocess.list2cmdline(self.run_cmd)

        # appended automatically by rocprofv3
        outfile = outfile + "_kernel_trace.csv"
        outfile = cwd / outfile
        proc = subprocess.run(self.run_cmd, shell=True, capture_output=True, cwd=cwd)

        # Unit is ns according to
        # https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/docs-6.2.1/how-to/using-rocprofv3.html#output-file-fields
        agg_gpu_timestamp_ns = 0

        with open(outfile, "r") as f:
            perfdata = csv.DictReader(f)
            for row in perfdata:
                agg_gpu_timestamp_ns += int(row["End_Timestamp"]) - int(
                    row["Start_Timestamp"]
                )

        if proc.returncode != 0:
            raise IreeRunException(
                process=proc,
                cwd=cwd,
                input_mlir_name=self.spec.input_mlir_name,
                compile_cmd=self.compile_cmd,
                run_cmd=self.run_cmd,
            )


class IreeCompileException(Exception):
    """Compiler exception that preserves the command line and output."""

    def __init__(
        self,
        process: subprocess.CompletedProcess,
        cwd: Path,
        input_mlir_name: Path,
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

        with open(cwd / input_mlir_name) as f:
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
        process: subprocess.CompletedProcess,
        cwd: Path,
        input_mlir_name: Path,
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

        with open(cwd / input_mlir_name) as f:
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
