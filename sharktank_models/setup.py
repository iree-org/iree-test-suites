# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from setuptools import find_namespace_packages, setup

setup(
    name="sharktank_model_tests",
    version=f"0.1dev1",
    packages=find_namespace_packages(
        include=[
            "ireers_tools",
        ],
    ),
    install_requires=[
        "azure-storage-blob",
        "iree-base-compiler",
        "iree-base-runtime",
        "ml_dtypes",
        "numpy",
        "pytest",
        "pytest-check",
        "pytest-html",
        "pytest-reportlog",
        "pytest-retry",
        "pytest-timeout",
        "pytest-xdist",
        "tabulate",
    ],
    extras_require={},
)
