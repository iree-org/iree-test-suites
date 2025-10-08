import pytest
import logging
from pathlib import Path
import logging
from pytest_iree.module import ModuleArtifact
from pytest_iree.azure import AzureArtifact
from pytest_iree.artifact import Artifact
from pytest_iree.irpa_gen import RandomIRPAArtifact

logger = logging.getLogger(__name__)


class TestBase(pytest.Item):
    def __init__(self, *, test_data: dict, **kwargs):
        super().__init__(**kwargs)
        self.test_data = test_data

        # Verify test.
        # TODO: Verify spec as described in README.md.
        assert "markers" in test_data, "Test data must contain 'markers' field"

        self.external_file_directory = Path(
            self.config.getoption("external_file_directory")
        ).resolve()
        self.artifact_dir = Path(self.config.getoption("artifact_directory")).resolve()
        self.force_recompile = self.config.getoption("force_recompile")
        self.module_directory = Path(
            self.config.getoption("module_directory")
        ).resolve()
        self.status = "N/A"

        # Add markers.
        for marker in test_data["markers"]:
            self.add_marker(marker)

    def _get_module(self, module: str) -> ModuleArtifact:
        return ModuleArtifact(
            artifact_base_dir=self.artifact_dir,
            module_base_dir=self.module_directory,
            module=module,
            external_file_dir=self.external_file_directory,
            force_recompile=self.force_recompile,
        )

    def _get_modules(self) -> list[ModuleArtifact]:
        # Compile all required modules.
        modules = self.test_data.get("modules")
        assert modules is not None, "Test data must contain 'modules' field"
        module_artifacts = []
        for module in modules:
            module_artifact = self._get_module(module)
            module_artifacts.append(module_artifact)
        return module_artifacts

    def _get_weights(self) -> list[tuple[str, Artifact]]:
        """
        Returns a list of (scope, Artifact) tuples for the model weights.
        """
        # Get the model weights.
        weights = self.test_data.get("weights", [])
        weight_artifacts: list[tuple[str, Artifact]] = []
        for weight in weights:
            if weight["type"] == "url":
                url = weight["url"]
                artifact = AzureArtifact(
                    artifact_base_dir=self.artifact_dir,
                    url=url,
                )
            elif weight["type"] == "random":
                module = weight["module"]
                seed = weight["seed"]
                module = self._get_module(module)
                artifact = RandomIRPAArtifact(
                    artifact_base_dir=self.artifact_dir,
                    module=module,
                    seed=seed,
                )
            else:
                raise ValueError(f"Unknown weight type: {weight['type']}")
            scope = weight["scope"]
            weight_artifacts.append((scope, artifact))
        return weight_artifacts

    def _get_arg_strings(self, json_field: str) -> list[str]:
        """
        Get argument strings based on the argument spec:
        {
            "url": "<url to file>",
            "value": "<literal value> | <file path> | <byte string>"
        }

        All fields are optional, but at least one must be present.
        The resulting argument string is a concatenation of the fields:
          <value>=@<url_file_path>
        """
        args = self.test_data.get(json_field, [])
        arg_strings: list[str] = []
        for arg in args:
            url = arg.get("url", None)
            value = arg.get("value", None)
            strings = []
            if value is not None:
                strings.append(value)
            if url is not None:
                artifact = AzureArtifact(artifact_base_dir=self.artifact_dir, url=url)
                artifact.join()
                strings.append(f"@{str(artifact.path.absolute())}")
            assert (
                len(strings) > 0
            ), f"{json_field} entry must have either 'url' or 'value'"
            arg_strings.append("=".join(strings))
        return arg_strings

    def _get_common_run_args(
        self, weight_artifacts: list[tuple[str, Artifact]]
    ) -> list[str]:
        # Get the inputs / outputs / expected outputs.
        inputs = self._get_arg_strings("inputs")
        outputs = self._get_arg_strings("outputs")
        expected_outputs = self._get_arg_strings("expected_outputs")

        # Create args for iree-run-module.
        args = []
        for scope, weight_artifact in weight_artifacts:
            weight_artifact.join()
            args.append(f"--parameters={scope}={str(weight_artifact.path.absolute())}")
        for input in inputs:
            args.append(f"--input={input}")
        for output in outputs:
            args.append(f"--output={output}")
        for expected_output in expected_outputs:
            args.append(f"--expected_output={expected_output}")
        # Add additional args.
        run_args = self.test_data.get("run_args", [])
        args.extend(run_args)
        return args

    def reportinfo(self):
        return self.path, 0, f"usecase: {self.name}"

    @classmethod
    def get_test_type(cls) -> str:
        ...

    @classmethod
    def get_test_headers(cls) -> list[str]:
        ...

    def get_test_summary(self) -> list:
        ...
