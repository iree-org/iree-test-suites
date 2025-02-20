# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from setuptools import find_namespace_packages, setup

setup(
    name=f"iree-regression-suite",
    version=f"0.1dev1",
    packages=find_namespace_packages(
        include=[
            "ireers_tools",
        ],
    ),
    install_requires=[
        "numpy",
        "pytest",
        "pytest-xdist",
        "pytest-depends",
        "pytest-retry",
        "pytest-timeout",
        "pytest-xdist",
        "pytest-check",
        "tabulate",
        "tqdm",
        "azure-storage-blob",
    ],
    extras_require={},
)
