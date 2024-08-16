# Matrix Multiplication (Matmul) Tests

These are being migrated from
https://github.com/iree-org/iree/tree/main/tests/e2e/matmul.

## Prerequisites

## Quickstart

First ensure you have the prerequisites from
https://iree.dev/building-from-source/getting-started/.

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

    ```bash
    cmake -G Ninja -B build/ . \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DIREE_HOST_BIN_DIR=${IREE_HOST_BIN_DIR}
    ```

3. Build:

    ```bash
    cmake --build build/
    cmake --build build/ --target iree-test-deps
    ```
