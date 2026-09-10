"""Shared GeoBench metadata deserialization helpers."""

import ast
import io
import pickle


class _StubUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> type:
        if module == "geobench.dataset":
            return type(name, (), {})
        return super().find_class(module, name)


def unpickle_metadata(value: object) -> dict:
    """Deserialize trusted GeoBench metadata without evaluating a string expression."""
    if isinstance(value, str):
        value = ast.literal_eval(value)
    elif not isinstance(value, bytes) and hasattr(value, "tobytes"):
        value = value.tobytes()
    if not isinstance(value, bytes):
        raise TypeError(f"Expected pickled metadata bytes, got {type(value).__name__}.")

    try:
        return pickle.loads(value)
    except (ModuleNotFoundError, AttributeError):  # allow-except: legacy GeoBench pickle classes.
        return _StubUnpickler(io.BytesIO(value)).load()
