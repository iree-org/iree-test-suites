from pathlib import Path
import logging
import hashlib
import mmap
import uuid
import json
import requests
from pytest_iree.artifact import Artifact

logger = logging.getLogger(__name__)


class AzureArtifact(Artifact):
    """Represents an azure artifact that can be downloaded from a URL. Each
    artifact's uniqueness is determined by it's URL. Two different artifacts
    with the same URL will map to the same local file and will be cached.

    The artifact creates a directory structure under the artifact_base_dir as:
    artifact_base_dir/
        azure/
           manifest.json
           <artifact1>
           <artifact2>
           ...

    The manifest.json file maps the URL to the local filename.
    """

    def __init__(
        self,
        artifact_base_dir: Path,
        url: str,
    ):
        # Check if this file already exists in the artifact directory.
        artifact_dir = artifact_base_dir / "azure"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        # Create a manifest.json file to hold the URL -> local filename
        # mapping, if it doesn't already exist.
        manifest_path = artifact_dir / "manifest.json"
        if not manifest_path.exists():
            manifest_path.write_text("{}")
        # Read the manifest.json file to check if a mapping already exists.
        manifest = json.loads(manifest_path.read_text())
        if url in manifest:
            name = manifest[url]
        else:
            # Create a unique name for the artifact using uuid, with the same file extension.
            file_extension = url.rsplit(".", 1)[-1]
            name = f"{uuid.uuid4()}.{file_extension}"
            manifest[url] = name
            manifest_path.write_text(json.dumps(manifest, indent=2))

        super().__init__(artifact_dir, name)
        self.url = url

    def human_readable_size(self, size, decimal_places=2):
        unit = "PiB"
        for unit in ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]:
            if size < 1024.0:
                break
            size /= 1024.0
        return f"{size:.{decimal_places}f} {unit}"

    def get_azure_md5(self, remote_file: str, content_md5: str | None):
        """Gets the content_md5 hash for a blob on Azure, if available."""
        if not content_md5:
            logger.warning(
                f"  Remote file '{remote_file}' on Azure is missing the "
                "'content_md5' property, can't check if local matches remote"
            )
        return content_md5

    def get_local_md5(self, local_file_path: Path):
        """Gets the content_md5 hash for a lolca file, if it exists."""
        if not local_file_path.exists() or local_file_path.stat().st_size == 0:
            return None

        with open(local_file_path) as file, mmap.mmap(
            file.fileno(), 0, access=mmap.ACCESS_READ
        ) as file:
            return hashlib.md5(file).digest()

    def download_azure_artifact(self):
        """
        Download artifact from Azure Blob Storage using direct HTTP requests.
        This avoids Azure SDK authentication issues on VMs with managed identity.
        """
        remote_file_name = self.url.rsplit("/", 1)[-1]

        # Use requests library for direct HTTP access to avoid Azure SDK
        # authentication complexity. Public blobs should be accessible via HTTP.
        try:
            # First, do a HEAD request to get blob properties
            logger.info(f"  Checking blob properties for '{remote_file_name}'")
            head_response = requests.head(self.url, timeout=30)
            head_response.raise_for_status()
            
            blob_size = int(head_response.headers.get('Content-Length', 0))
            content_md5_b64 = head_response.headers.get('Content-MD5')
            
            # Convert base64 MD5 to bytes if present
            azure_md5 = None
            if content_md5_b64:
                import base64
                azure_md5 = base64.b64decode(content_md5_b64)
            
            blob_size_str = self.human_readable_size(blob_size)
            local_md5 = self.get_local_md5(self.path)

            if azure_md5 and azure_md5 == local_md5:
                logger.info(
                    f"  Skipping '{remote_file_name}' download ({blob_size_str}) "
                    "- local MD5 hash matches"
                )
                return

            # Download the blob
            if not local_md5:
                logger.info(
                    f"  Downloading '{remote_file_name}' ({blob_size_str}) "
                    f"to '{self.path}'"
                )
            else:
                logger.info(
                    f"  Downloading '{remote_file_name}' ({blob_size_str}) "
                    f"to '{self.path}' (local MD5 does not match)"
                )

            # Stream download with progress
            response = requests.get(self.url, stream=True, timeout=300)
            response.raise_for_status()
            
            with open(self.path, mode="wb") as local_blob:
                for chunk in response.iter_content(chunk_size=1024 * 1024 * 32):  # 32 MiB chunks
                    if chunk:
                        local_blob.write(chunk)
            
            logger.info(f"  Downloaded '{remote_file_name}' successfully")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"  Failed to download '{remote_file_name}': {e}")
            raise

    def join(self):
        super().join()
        self.download_azure_artifact()
