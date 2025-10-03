from pathlib import Path
import logging
import logging
import json
from pytest_iree.artifact import Artifact
from pytest_iree.azure import AzureArtifact
from pytest_iree.utils import iree_compile
import shutil

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

        # Delete the directory to force recompilation of module if requested.
        if force_recompile and module_artifact_dir.exists():
            logger.info(
                f"  Force recompilation - removing existing module artifact directory '{module_artifact_dir}'"
            )
            shutil.rmtree(module_artifact_dir)

        super().__init__(module_artifact_dir, name)

        self.artifact_base_dir = artifact_base_dir
        self.module_json = module_json
        self.module_artifact_dir = module_artifact_dir
        self.external_file_dir = external_file_dir
        self.compstat_path = module_artifact_dir / "compilation_info.json"

    def _needs_recompile(self) -> bool:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return True
        if not self.compstat_path.exists() or self.compstat_path.stat().st_size == 0:
            return True
        return False

    def join(self):
        super().join()
        # Check if the module file already exists.
        if not self._needs_recompile():
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
        # Add compilation stats path and format.
        compiler_flags += [
            f"--iree-scheduling-dump-statistics-file={self.compstat_path.resolve()}",
            "--iree-scheduling-dump-statistics-format=json",
        ]
        iree_compile(
            source=mlir_artifact.path,
            output=self.path,
            cwd=self.external_file_dir,
            args=compiler_flags,
        )

    def get_compstats(self) -> dict:
        """Get the compilation statistics from the compilation_info.json file."""
        self.join()
        assert (
            self.compstat_path.exists()
        ), f"Compilation stats file '{self.compstat_path}' does not exist."
        compstats = json.loads(self.compstat_path.read_text())
        return compstats
