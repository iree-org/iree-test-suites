from pathlib import Path
import logging
import subprocess
from pytest_iree.artifact import Artifact

logger = logging.getLogger(__name__)

class GitLfsArtifact(Artifact):
    """
    Represents an artifact that is stored in the repo using git lfs.
    """

    def __init__(
            self,
            module_base_dir: Path,
            filename: str,
    ):
        super().__init__(module_base_dir, filename)

    def pull_lfs_file(self):
        if (self.path.exists()):
            return
        
        logger.info(f"  Pulling git LFS file '{self.path}'")
        command = [
            "git", "lfs", "pull", f"--include={self.name}", '--exclude""']
        logger.debug(f"  Running command: \n  cd {self.module_path}\n  {subprocess.list2cmdline(command)}")
        subprocess.run(command, check=True, cwd=self.artifact_dir)

    def join(self):
        super().join()
        self.pull_lfs_file()
    

