import logging
from pathlib import Path
from pytest_iree.artifact import Artifact
from pytest_iree.module import ModuleArtifact
from enum import Enum
import mmap
import hashlib
import numpy as np
import ml_dtypes
import tqdm

logger = logging.getLogger(__name__)


class RandomIRPAArtifact(Artifact):
    """Represents an IRPA file which is generated for a module with random
    values.

    The artifact creates a directory structure under the artifact_base_dir as:
    artifact_base_dir/
        irpa_gen/
            <module1>/
                param_<seed>.irpa
                ...
            <module2>/
                param_<seed>.irpa
                ...

    TODO: Do proper scoping of parameters. Currently, to handle a file with
    multiple scopes, the user can pass the same randomly generated irpa file
    with another scope, and when passed to iree runtime, it will end up mapping
    to the same file, with different scopes.
    """

    def __init__(
        self,
        artifact_base_dir: Path,
        module: ModuleArtifact,
        seed: int,
    ):
        artifact_dir = artifact_base_dir / "random_irpa"
        param_artifact_dir = artifact_dir / Path(module.module_name)
        name = "param_" + str(seed) + ".irpa"

        super().__init__(param_artifact_dir, name)

        self.module = module
        self.seed = seed
        self.md5_hash_path = param_artifact_dir / (name + ".md5")

    def get_local_md5(self, local_file_path: Path):
        """Gets the content_md5 hash for a lolca file, if it exists."""
        if not local_file_path.exists() or local_file_path.stat().st_size == 0:
            return None

        with open(local_file_path) as file, mmap.mmap(
            file.fileno(), 0, access=mmap.ACCESS_READ
        ) as file:
            return hashlib.md5(file).digest()

    def _needs_regenerate(self, mlir_path) -> bool:
        if not self.path.exists():
            return True
        # Check if the mlir file's md5 hash matches the one from which this
        # param was generated from.
        if not self.md5_hash_path.exists():
            return True
        md5_hash = self.get_local_md5(mlir_path)
        assert md5_hash is not None
        existing_md5_hash = self.md5_hash_path.read_bytes()
        if md5_hash != existing_md5_hash:
            return True
        return False

    def _check_imports(self):
        try:
            import iree.runtime  # noqa: F401
            import iree.compiler  # noqa: F401
        except Exception as e:
            logger.error(
                "Could not import iree python bindings. Generating random irpa requires iree python bindings"
            )
            raise e

    def _match_named_parameter_regex(self, param: str):
        # general form:
        # (stream|flow).parameter.named<("scope"::)?<name>> : tensor<shapextype>
        #
        # examples:
        # stream scoped: stream.parameter.named<"model"::"unet.mid_block.resnets.1.norm1.weight"> : tensor<1280xf16>
        # flow scoped: flow.parameter.named<"model"::"unet.mid_block.resnets.1.norm1.weight"> : tensor<1280xf16>
        # flow unscoped: flow.parameter.named<"unet.mid_block.resnets.1.norm1.weight"> : tensor<1280xf16>
        #
        # TODO: Someone should really add attribute python bindings for these and
        # use them instead of this regex. This is prone to breaking if we change
        # the assembly syntax / printer. For now, this is a workaround.
        import re

        regex = (
            r'#(stream|flow)\.parameter\.named<(?:(?:"([^"]*)")::)?(?:"([^"]+)")> : .*'
        )
        match = re.match(regex, param)
        if match:
            scope = match.group(2) if match.group(2) else ""
            name = match.group(3)
            return scope, name
        return None

    def _collect_params(self, mlir_path):
        from iree.compiler.dialects import util as util_d
        from iree.compiler.ir import Module, Context, Location, ShapedType

        params: list[tuple[str, ShapedType]] = []
        with Context(), Location.unknown():
            with open(mlir_path) as f:
                mlir_mod = Module.parse(f.read())
                for op in mlir_mod.body.operations:
                    if isinstance(op, util_d.GlobalOp):
                        if op.initial_value is None:
                            continue
                        init_value = str(op.initial_value)
                        match = self._match_named_parameter_regex(init_value)
                        if match:
                            _, name = match
                            shaped_type = ShapedType(op.type_.value)
                            params.append((name, shaped_type))
        return params

    def _mlir_dtype_to_numpy_dtype(self, el_ty):
        import iree.compiler.ir as iree_ir

        if isinstance(el_ty, iree_ir.IntegerType):
            if el_ty.width == 8:
                return np.uint8 if el_ty.is_unsigned else np.int8
            elif el_ty.width == 16:
                return np.uint16 if el_ty.is_unsigned else np.int16
            elif el_ty.width == 32:
                return np.uint32 if el_ty.is_unsigned else np.int32
            elif el_ty.width == 64:
                return np.uint64 if el_ty.is_unsigned else np.int64
            else:
                raise ValueError(f"NYI integer width: {el_ty.width}")
        elif isinstance(el_ty, iree_ir.F64Type):
            return np.float64
        elif isinstance(el_ty, iree_ir.F32Type):
            return np.float32
        elif isinstance(el_ty, iree_ir.F16Type):
            return np.float16
        elif isinstance(el_ty, iree_ir.BF16Type):
            return ml_dtypes.bfloat16
        else:
            raise ValueError(f"NYI floating point type: {el_ty}")

    def join(self):
        self._check_imports()
        super().join()
        self.module.join()
        mlir_path = self.module.get_mlir_path()
        if not self._needs_regenerate(mlir_path):
            logger.info(f"  Skipping '{self.path}' generation - file exists")
            return
        params = self._collect_params(mlir_path)

        self._check_imports()
        import iree.runtime as rt

        rng = np.random.default_rng(self.seed)
        random_param = rt.ParameterIndex()
        np_dtype = None
        with tqdm.tqdm(
            total=len(params), desc=f"Generating {self.path.name}", unit="param"
        ) as pbar:
            for key, shape in params:
                el_ty = shape.element_type
                np_dtype = self._mlir_dtype_to_numpy_dtype(el_ty)

                if np.issubdtype(np_dtype, np.floating):
                    # For floats, sample from a normal distribution with mean
                    # 0.0 and stddev 0.01.
                    array = rng.normal(loc=0.0, scale=0.01, size=shape.shape).astype(
                        np_dtype
                    )
                elif np.issubdtype(np_dtype, np.integer):
                    # For integers, create a random number between the min and max of the dtype
                    info = np.iinfo(np_dtype)
                    array = rng.integers(
                        info.min, info.max, size=shape.shape, dtype=np_dtype
                    )
                else:
                    raise ValueError(
                        f"Unsupported numpy dtype {np_dtype} for parameter {key}"
                    )
                random_param.add_buffer(key, array)
                pbar.update(1)
                if pbar.n % 100 == 0:
                    logger.info(str(pbar))

        random_param.create_archive_file(str(self.path))
        # Save the md5 hash of the mlir file from which this param was generated
        md5_hash = self.get_local_md5(mlir_path)
        assert md5_hash is not None
        self.md5_hash_path.write_bytes(md5_hash)
