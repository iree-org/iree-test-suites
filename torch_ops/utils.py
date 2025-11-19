import dataclasses
import json
import numbers
import numpy as np
import pytest
import typing

try:
    # Optionally import torch.
    # Should only be needed when generating the tests,
    # not when running the tests.
    import torch
except:
    ...

@dataclasses.dataclass(frozen=True, kw_only=True)
class Formula:
    """
    Represents the formula:

    (coeff * numpy.random.rand(*shape) + offset).astype(dtype)
    """
    shape: typing.Tuple[int, ...]
    dtype: type = np.dtype("float32")
    coeff: numbers.Number = 1
    offset: numbers.Number = 0

    def numpy(self):
        return (self.coeff * np.random.rand(*self.shape) + self.offset).astype(self.dtype)

    def torch(self):
        return torch.from_numpy(self.numpy())

    def toJSONEncoder(self):
        """
        Ensure all fields in dataclass are able to be encoded into JSON.

        self.dtype:type cannot be encoded into JSON
        """
        return {"Formula": {"shape": self.shape, "dtype": self.dtype.name, "coeff": self.coeff, "offset": self.offset}}

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Formula):
            return obj.toJSONEncoder()
        if isinstance(obj, pytest.Mark):
            mark = obj
            return {"name": mark.name, "args": mark.args, "kwargs": mark.kwargs}
        return super().default(obj)

def customJSONDecoder(d):
    if kwargs := d.get("Formula"):
        return Formula(**kwargs)
    return d
