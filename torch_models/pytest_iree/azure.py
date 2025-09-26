from pathlib import Path
from azure.storage.blob import BlobClient, BlobProperties
import logging
import hashlib
import mmap
import uuid
import re
import tqdm
import logging
import json
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

    def get_azure_md5(self, remote_file: str, azure_blob_properties: BlobProperties):
        """Gets the content_md5 hash for a blob on Azure, if available."""
        content_settings = azure_blob_properties.get("content_settings")
        if not content_settings:
            return None
        azure_md5 = content_settings.get("content_md5")
        if not azure_md5:
            logger.warning(
                f"  Remote file '{remote_file}' on Azure is missing the "
                "'content_md5' property, can't check if local matches remote"
            )
        return azure_md5

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
        Checks the hashes between the local file and azure file.
        """
        remote_file_name = self.url.rsplit("/", 1)[-1]

        # Extract path components from Azure URL to use with the Azure Storage Blobs
        # client library for Python (https://pypi.org/project/azure-storage-blob/).
        #
        # For example:
        #   https://sharkpublic.blob.core.windows.net/sharkpublic/path/to/blob.txt
        #                                            ^           ^
        #   account_url:    https://sharkpublic.blob.core.windows.net
        #   container_name: sharkpublic
        #   blob_name:      path/to/blob.txt
        result = re.search(r"(https.+\.net)/([^/]+)/(.+)", self.url)
        assert result, f"Failed to parse Azure URL '{self.url}'"
        account_url = result.groups()[0]
        container_name = result.groups()[1]
        blob_name = result.groups()[2]

        # Move azure logging to DEBUG because it is too verbose.
        azure_logger = logging.getLogger("azure").setLevel(logging.ERROR)

        with BlobClient(
            account_url,
            container_name,
            blob_name,
            max_chunk_get_size=1024 * 1024 * 32,  # 32 MiB
            max_single_get_size=1024 * 1024 * 32,  # 32 MiB
            logger=azure_logger,
        ) as blob_client:
            blob_properties = blob_client.get_blob_properties()
            blob_size_str = self.human_readable_size(blob_properties.size)
            azure_md5 = self.get_azure_md5(self.url, blob_properties)
            local_md5 = self.get_local_md5(self.path)

            if azure_md5 and azure_md5 == local_md5:
                logger.info(
                    f"  Skipping '{remote_file_name}' download ({blob_size_str}) "
                    "- local MD5 hash matches"
                )
                return

            with tqdm.tqdm(
                total=blob_properties.size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=f"    {remote_file_name}",
                file=None,
            ) as pbar:

                def progress_hook(current, total):
                    pbar.update(current - pbar.n)
                    # TODO: This can probably be done at a lesser frequenct
                    # interval.
                    logger.info(str(pbar))

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
                with open(self.path, mode="wb") as local_blob:
                    download_stream = blob_client.download_blob(
                        max_concurrency=4, progress_hook=progress_hook
                    )
                    local_blob.write(download_stream.readall())

    def join(self):
        super().join()
        self.download_azure_artifact()
