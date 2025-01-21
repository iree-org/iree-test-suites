# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import abc
import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class CacheManager:
    """Manager class for multiple CacheScope instances.

    Each registered scope can independently manage a set of cached files.
    Symlinks from the cache are created into a common working directory. For example:

    cache_dir_1/
      onnx_models/    <--- GitHubLFSRepositoryCacheScope for https://github.com/onnx/models
        validated/vision/classification/
          mnist/
            model/
              mnist-12.onnx
          mobilenet/
            model/
              mobilenetv2-12.onnx
      some_other_repository/
        ...

    cache_dir_2/
      ...

    working_directory/
      model_zoo/                          <--- scope name
        validated/vision/classification/  <--- subdirectory
          mnist-12.mlir                   <--- generated file during a test run
          mnist-12.onnx                   <--- symlink to cached file
          mobilenetv2-12.mlir             <--- generated file during a test run
          mobilenetv2-12.onnx             <--- symlink to cached file
    """

    def __init__(self, working_directory: Path):
        self.cache_scopes = []
        self.working_directory = working_directory

    def get_file_in_cache(self, scope_name: str, relative_path: str) -> Path:
        """Gets the path to a file in the cache, if it exists"""
        logger.info(f"Getting file from {scope_name} cache: {relative_path}")
        for cache_scope in self.cache_scopes:
            if cache_scope.scope_name == scope_name:
                return cache_scope.get_file(relative_path)
        raise ValueError(f"Unknown cache scope: '{cache_scope}'")

    def get_file_in_working_directory(
        self, scope_name: str, relative_path: str, subdirectory: str
    ) -> Path:
        """Gets the path to a file from the cache, symlinked into the working directory"""
        file_in_cache = self.get_file_in_cache(scope_name, relative_path)

        # Create symlink from cache dir to working directory.
        working_subdirectory = self.working_directory / scope_name / subdirectory
        logger.debug(f"Working subdirectory: {working_subdirectory}")
        working_subdirectory.mkdir(parents=True, exist_ok=True)

        file_name = relative_path.rsplit("/", 1)[-1]
        working_subdirectory_file = working_subdirectory / file_name
        logger.debug(f"Symlinking '{working_subdirectory_file}' to '{file_in_cache}'")
        if working_subdirectory_file.is_symlink():
            if os.path.samefile(str(working_subdirectory_file), str(file_in_cache)):
                logger.debug("  Expected symlink already exists")
                return working_subdirectory_file
            os.remove(working_subdirectory_file)
        elif working_subdirectory_file.exists():
            logger.warning("  Non-symlink file exists. Replacing with a symlink")
            os.remove(working_subdirectory_file)
        os.symlink(src=file_in_cache, dst=working_subdirectory_file)
        return working_subdirectory_file


class CacheScope(abc.ABC):
    """Abstract base class for a cache scope."""

    def __init__(self, scope_name: str):
        self.scope_name = scope_name

    @abc.abstractmethod
    def get_file(self, relative_path: str) -> Path:
        """Get the path to a file loaded from the cache"""


class GitHubLFSRepositoryCacheScope(CacheScope):
    """Cache scope backed by a GitHub repository using Git LFS for files of interest."""

    def __init__(
        self,
        scope_name: str,
        cache_dir: Path,
        repository_name: str,
        clone_method: str = "https",
    ):
        super().__init__(scope_name)

        self.repository_name = repository_name
        self.local_repository_dir = cache_dir / repository_name.replace("/", "_")

        self.setup_github_repository(
            repository_name=repository_name, clone_method=clone_method
        )

    def setup_github_repository(self, repository_name: str, clone_method: str):
        logger.info(f"Setting up GitHub repository '{repository_name}'")

        logger.info("Checking for working 'git lfs' (https://git-lfs.com/)")
        subprocess.run(["git", "lfs", "env"], capture_output=True, check=True)

        # Skip if the directory already exists (and is a git directory).
        if self.local_repository_dir.is_dir():
            logger.info(f"Directory '{self.local_repository_dir}' already exists")
            subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                cwd=self.local_repository_dir,
                capture_output=True,
                check=True,
            )
            return

        # Directory does not exist yet, clone.
        if clone_method == "https":
            remote_url = f"https://github.com/{repository_name}.git"
        else:
            remote_url = f"git@github.com:{repository_name}.git"
        logger.info(f"Cloning {remote_url} into '{self.local_repository_dir}'")
        subprocess.run(
            ["git", "clone", remote_url, self.local_repository_dir], check=True
        )

    def pull_lfs_file(self, file_relative_path: str):
        logger.debug(
            f"Pulling git LFS file '{self.local_repository_dir / file_relative_path}'"
        )
        command = [
            "git",
            "lfs",
            "pull",
            f"--include={file_relative_path}",
            '--exclude=""',
        ]
        logger.debug(
            f"Running command:\n  cd {self.local_repository_dir}\n  {subprocess.list2cmdline(command)}"
        )
        subprocess.run(command, check=True, cwd=self.local_repository_dir)

    def get_file(self, relative_path: str) -> Path:
        # Log file information for easier reproduction outside of the test suite.
        direct_download_url = (
            f"https://github.com/{self.repository_name}/raw/main/{relative_path}"
        )
        logger.info(
            f"Getting file '{relative_path}' from cache. Direct download URL:\n  {direct_download_url}"
        )

        self.pull_lfs_file(relative_path)
        return self.local_repository_dir / relative_path
