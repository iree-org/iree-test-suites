# Matrix Multiplication (Matmul) Tests

These are being migrated from
https://github.com/iree-org/iree/tree/main/tests/e2e/matmul.

## Prerequisites

## Quickstart

First ensure you have the prerequisites from
https://iree.dev/building-from-source/getting-started/, including CMake, a
compiler like clang, and Python.

1. Get the IREE compiler tools, either from release packages or a source build:

    * To use Python packages:

        ```bash
        python -m venv .venv
        source .venv/bin/activate
        python -m pip install -r requirements-iree.txt
        export IREE_HOST_BIN_DIR=.venv/bin
        ```

    * To use a source build:

        ```bash
        export IREE_HOST_BIN_DIR=path/to/iree-build
        ```

2. Configure:

    * To let CMake `FetchContent` download its own copy of the IREE repository:

        ```bash
        # For now this is where you would set other options like
        #   -DIREE_HIP_TEST_TARGET_CHIP=gfx90a
        # Tests should be decoupled from the core CMake project as much as possible.

        cmake -G Ninja -B build/ . \
          -DCMAKE_BUILD_TYPE=RelWithDebInfo \
          -DIREE_HOST_BIN_DIR=${IREE_HOST_BIN_DIR}
        ```

    * To use a local IREE repository:

        ```bash
        # For now this is where you would set other options like
        #   -DIREE_HIP_TEST_TARGET_CHIP=gfx90a
        # Tests should be decoupled from the core CMake project as much as possible.

        cmake -G Ninja -B build/ . \
          -DCMAKE_BUILD_TYPE=RelWithDebInfo \
          -DIREE_USE_LOCAL_REPO=ON \
          -DIREE_LOCAL_REPO_PATH=/path/to/iree \
          -DIREE_HOST_BIN_DIR=${IREE_HOST_BIN_DIR}
        ```

3. Build:

    ```bash
    cmake --build build/
    cmake --build build/ --target iree-matmul-test-suite-deps
    ```

4. Run tests:

    ```bash
    ctest --test-dir build/ -R iree-test-suites
    ```
