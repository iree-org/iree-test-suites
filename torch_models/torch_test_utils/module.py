from pathlib import Path
import logging
import logging
import json
from torch_test_utils.artifact import Artifact
from torch_test_utils.azure import AzureArtifact
from torch_test_utils.utils import iree_compile

logger = logging.getLogger(__name__)


class ModuleArtifact(Artifact):
    """Represents a vmfb module artifact that can be loaded by IREE. Each
    artifact's uniqueness is defined by it's module name. The module definition
    can be found at `<module_base_dir>/Path(<module>).json`.

    The artifact creates a directory structure under the artifact_base_dir as:
    artifact_base_dir/
        modules/
            <module1>/
                <module1>.vmfb
            <module2>
                <module2>.vmfb
                ...
    """

    def _clean_module_name(self, module: str) -> str:
        return module.replace("/", "_").replace("\\", "_")

    def __init__(
        self,
        artifact_base_dir: Path,
        module_base_dir: Path,
        module: str,
        external_file_dir: Path,
        force_recompile: bool = False,
    ):
        artifact_dir = artifact_base_dir / "modules"
        module_artifact_dir = artifact_dir / Path(module)
        name = "module.vmfb"
        module_json = (module_base_dir / Path(module)).with_suffix(".json")
        assert (
            module_json.exists()
        ), f"Module definition '{module_json}' does not exist."

        super().__init__(module_artifact_dir, name)

        self.artifact_base_dir = artifact_base_dir
        self.module_json = module_json
        self.module_artifact_dir = module_artifact_dir
        self.external_file_dir = external_file_dir
        self.force_recompile = force_recompile

    def join(self):
        # Check if the module file already exists.
        if (
            not self.force_recompile
            and self.path.exists()
            and self.path.stat().st_size > 0
        ):
            logger.info(f"  Skipping '{self.path}' download - file exists")
            return
        module_data = json.loads(self.module_json.read_text())
        mlir_url = module_data["mlir"]
        assert (
            "blob.core.windows.net" in mlir_url
        ), "Only Azure Blob Storage is supported currently."
        mlir_artifact = AzureArtifact(self.artifact_base_dir, mlir_url)
        mlir_artifact.join()
        # Compile the MLIR file to a VMFB file.
        compiler_flags = module_data.get("compiler_flags", [])
        iree_compile(
            source=mlir_artifact.path,
            output=self.path,
            cwd=self.external_file_dir,
            args=compiler_flags,
        )
