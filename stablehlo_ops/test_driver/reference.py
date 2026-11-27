"""Run eval_module on the mlir functions to invoke the stablehlo interpreter
and get reference outputs
"""

from pathlib import Path
import numpy as np

from mlir import ir
from mlir.dialects import stablehlo


def evaluate(source: Path, inputs: list[np.ndarray]) -> list[np.ndarray]:
    with ir.Context() as context:
        stablehlo.register_dialect(context)
        # Fallbacks might be used in case the interpreter is not able to interpret the usual test but
        # there exists equivalent code that can be used to compute the reference. e.g a decomposed version.
        fallback = source.with_name("stablehlo_interpreter_fallback.mlir")
        if fallback.is_file():
            source = fallback
        module = ir.Module.parse(source.read_text())
        module.operation.verify()
        results = stablehlo.eval_module(
            module, [ir.DenseElementsAttr.get(value) for value in inputs]
        )
        return [np.array(result) for result in results]
