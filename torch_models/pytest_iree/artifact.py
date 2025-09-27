from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class Artifact:
    """Some form of artifact materialized to disk. The name for each artifact must be unique."""

    def __init__(
        self,
        artifact_dir: Path,
        name: str,
    ):
        self.artifact_dir = artifact_dir
        self.name = name

    @property
    def path(self) -> Path:
        return self.artifact_dir / self.name

    def join(self):
        """Waits for the artifact to become available."""
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __str__(self):
        return str(self.path)
