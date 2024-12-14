# LiteRT (Formely TFLite) Model Tests

This test suite exercises
[LiteRT, formely known as TensorFlow Lite](https://ai.google.dev/edge/litert)
models. Most pretrained models are sourced from https://www.kaggle.com/models.

Testing *currently* follows several stages:

```mermaid
graph LR
  Model["Download model"]
  Model --> ImportMLIR["Import into MLIR"]
  ImportMLIR --> CompileIREE["Compile with IREE"]
```

Testing *could* also test inference and compare with LiteRT:

```mermaid
graph LR
  Model --> ImportMLIR["Import into MLIR"]
  ImportMLIR --> CompileIREE["Compile with IREE"]
  CompileIREE --> RunIREE["Run with IREE"]
  RunIREE --> Check

  Model --> LoadLiteRT["Load into LiteRT"]
  LoadLiteRT --> RunLiteRT["Run with LiteRT"]
  RunLiteRT --> Check

  Check["Compare results"]
```

## Quickstart

1. Set up your virtual environment and install requirements:

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt
    ```

    * To use `iree-compile` and `iree-run-module` from Python packages:

        ```bash
        python -m pip install -r requirements-iree.txt
        ```

    * To use a custom version of IREE follow the instructions for
      [building the IREE Python packages from source](https://iree.dev/building-from-source/getting-started/#python-bindings),
      including the extra steps for the TFLite importer.

2. Run pytest using typical flags:

    ```bash
    pytest \
      -rA \
      --log-cli-level=info \
      --durations=0
    ```

    See https://docs.pytest.org/en/stable/how-to/usage.html for other options.

## Advanced pytest usage

* The `log-cli-level` level can also be set to `debug`, `warning`, or `error`.
  See https://docs.pytest.org/en/stable/how-to/logging.html.
* Run only tests matching a name pattern:

    ```bash
    pytest -k resnet
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

* Create an HTMl report using https://pytest-html.readthedocs.io/en/latest/index.html

    ```bash
    pytest --html=report.html --self-contained-html --log-cli-level=info
    ```

    See also
    https://docs.pytest.org/en/latest/how-to/output.html#creating-junitxml-format-files

## Test suite implementation details

### Kaggle

Models are downloaded using https://github.com/Kaggle/kagglehub.

By default, kagglehub caches downloads at `~/.cache/kagglehub/models/`. This
can be overriden by setting the `KAGGLEHUB_CACHE` environment variable. See the
[`kagglehub/config.py` source](https://github.com/Kaggle/kagglehub/blob/main/src/kagglehub/config.py)
for other configuration options.

### Working with `.mlirbc` files

The `iree-import-tflite` tool outputs MLIR bytecode (`.mlirbc`) by default. To
convert to MLIR text (`.mlir`):

```bash
iree-ir-tool cp input.mlirbc -o output.mlir
```
