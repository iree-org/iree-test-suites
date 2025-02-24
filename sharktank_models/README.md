# Sharktank Model Tests

This test suite includes small scale versions of Large Language Models (LLMs)
and other Generative AI (GenAI) programs exported using the
[sharktank package](https://github.com/nod-ai/shark-ai/tree/main/sharktank)
built as part of the [shark-ai project](https://github.com/nod-ai/shark-ai).

## Quickstart

1. Download files through [git lfs](https://git-lfs.com/) as needed:

    ```bash
    git lfs install
    git lfs pull --include="*"

     git lfs ls-files
     # 37f90b4754 * sharktank_models/llama3.1/assets/toy_llama.irpa
     # 7172acdf43 * sharktank_models/llama3.1/assets/toy_llama.mlir
     # e997647ecc * sharktank_models/llama3.1/assets/toy_llama_tp2.irpa
     # b7b2f5a206 * sharktank_models/llama3.1/assets/toy_llama_tp2.mlir
     # 917845c887 * sharktank_models/llama3.1/assets/toy_llama_tp2.rank0.irpa
     # 9ab51093c4 * sharktank_models/llama3.1/assets/toy_llama_tp2.rank1.irpa
     ```

2. Set up your virtual environment and install requirements:

    ```bash
    cd sharktank_models

    python -m venv .venv
    source .venv/bin/activate
    python -m pip install -e sharktank_models/
    ```

    * To use IREE from nightly pre-release Python packages:

        ```bash
        python -m pip install -r sharktank_models/requirements-iree.txt
        ```

    * To use a custom version of IREE follow the instructions for
      [building the IREE Python packages from source](https://iree.dev/building-from-source/getting-started/#python-bindings).

3. Run pytest using typical flags:

    ```bash
    pytest \
      -rA \
      -m "target_cpu" \
      --timeout=300 \
      --durations=0 \
      --log-cli-level=info
    ```

    See https://docs.pytest.org/en/stable/how-to/usage.html for other options.

## Advanced pytest usage

* The `log-cli-level` level can also be set to `debug`, `warning`, or `error`.
  See https://docs.pytest.org/en/stable/how-to/logging.html.
* Run only tests matching a name pattern:

    ```bash
    pytest -k llama
    ```

* Run tests that require an AMD GPU
  (https://docs.pytest.org/en/stable/example/markers.html):

    ```bash
    pytest -m "target_hip"
    ```

* Ignore xfail marks
  (https://docs.pytest.org/en/stable/how-to/skipping.html#ignoring-xfail):

    ```bash
    pytest --runxfail
    ```

* Run tests in parallel using https://pytest-xdist.readthedocs.io/
  (note that this swallows some logging):

    ```bash
    # Run with an automatic number of threads (usually one per CPU core).
    pytest -n auto

    # Run on an explicit number of threads.
    pytest -n 4
    ```

* Create an HTML report using https://pytest-html.readthedocs.io/en/latest/index.html

    ```bash
    pytest --html=report.html --self-contained-html --log-cli-level=info
    ```

    See also
    https://docs.pytest.org/en/latest/how-to/output.html#creating-junitxml-format-files

## Running quality tests

Please refer to [Quality tests README](regression_tests/README.md) to run tests

## Running benchmark tests

Please refer to [Benchmark tests README](benchmarks/README.md) to run tests

Note: for benchmark tests to run, you will need `vmfbs` files available
