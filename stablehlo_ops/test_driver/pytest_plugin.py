"""Collect run_module_io.json files with the ONNX/Torch backend conventions."""

from collections.abc import Callable, Iterator
from functools import partial
import os
import tempfile
from pathlib import Path
from typing import Any

import pytest

from .artifacts import read_json, read_tensor
from .runner import (
    RunnerError,
    compile_module,
    run_module,
    save_iree_inputs,
)
from .reference import evaluate

ROOT = Path(__file__).resolve().parents[1]


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("stablehlo")
    group.addoption(
        "--config-files",
        nargs="+",
        default=[p for p in os.getenv("IREE_TEST_CONFIG_FILES", "").split(";") if p]
        or [str(ROOT / "configs/stablehlo_ops_cpu_llvm_sync.json")],
        help="Backend configuration JSON files",
    )
    group.addoption(
        "--ignore-xfails",
        action="store_true",
        help="Ignore configured expected failures",
    )
    group.addoption(
        "--skip-all-runs",
        action="store_true",
        help="Compile cases without running them",
    )


def pytest_configure(config: pytest.Config) -> None:
    configurations = []
    names = set()
    for path in config.getoption("config_files"):
        backend = read_json(Path(path))
        for key in ("config_name", "iree_compile_flags", "iree_run_module_flags"):
            if key not in backend:
                raise pytest.UsageError(f"Missing {key} in {path}")
        if backend["config_name"] in names:
            raise pytest.UsageError(f"Duplicate backend name: {backend['config_name']}")
        names.add(backend["config_name"])
        configurations.append(backend)
    config.stablehlo_backends = configurations


def pytest_collect_file(
    parent: pytest.Collector, file_path: Path
) -> "StablehloFile | None":
    if file_path.name == "run_module_io.json":
        return StablehloFile.from_parent(parent, path=file_path)


class StablehloFile(pytest.File):
    def collect(self) -> Iterator[pytest.Function]:
        case = read_json(self.path)
        sources = {"static": case["file_name"], "dynamic": case["dynamic_file_name"]}
        for variant, name in sources.items():
            if not (self.path.parent / name).is_file():
                raise self.CollectError(f"Missing {variant} MLIR: {name}")
        for backend in self.config.stablehlo_backends:
            yield pytest.Function.from_parent(
                self,
                name=backend["config_name"],
                callobj=partial(
                    run_pair, self.path, case, sources, backend, self.config
                ),
            )


def run_pair(
    path: Path,
    case: dict[str, Any],
    sources: dict[str, str],
    backend: dict[str, Any],
    config: pytest.Config,
) -> None:
    failures = []
    known_failures = []

    def check(stage: str, identifier: str, action: Callable[[], None]) -> bool:
        expected = not config.getoption("ignore_xfails") and identifier in backend.get(
            f"expected_{stage}_failures", []
        )
        try:
            action()
        except RunnerError as error:
            issues = known_failures if expected else failures
            issues.append(f"{identifier} {stage}: {error}")
            return False
        if expected:
            failures.append(f"Unexpected {stage} success: {identifier}")
            return False
        return True

    with tempfile.TemporaryDirectory(prefix="stablehlo-pair-") as temporary:
        tmp_path = Path(temporary)
        runnable = []
        for variant, name in sources.items():
            identifier = f"{path.parent.name}/{variant}"
            if identifier in backend.get("skip_compile_tests", []):
                continue
            work = tmp_path / variant
            work.mkdir()
            vmfb = work / "module.vmfb"
            compile_action = partial(compile_module, path.parent / name, backend, vmfb)
            if check("compile", identifier, compile_action) and not (
                case.get("compile_only_reason")
                or config.getoption("skip_all_runs")
                or identifier in backend.get("skip_run_tests", [])
            ):
                runnable.append((identifier, vmfb))

        if runnable:
            inputs = [read_tensor(path.parent / name) for name in case["inputs"]]
            expected = evaluate(path.parent / case["file_name"], inputs)
            input_files = save_iree_inputs(tmp_path, inputs)
            for identifier, vmfb in runnable:
                run_action = partial(run_module, input_files, backend, vmfb, expected)
                check("run", identifier, run_action)

    if failures:
        pytest.fail("\n".join(failures), pytrace=False)
    if known_failures:
        pytest.xfail("\n".join(known_failures))
    if all(
        f"{path.parent.name}/{variant}" in backend.get("skip_compile_tests", [])
        for variant in sources
    ):
        pytest.skip("Both variants included in skip_compile_tests")
