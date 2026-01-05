# src/isotope/_optional.py
"""Helpers for optional dependency handling."""

from typing import Any


def _create_missing_dependency_class(class_name: str, package: str) -> type:
    """Create a placeholder class that raises ImportError on instantiation.

    This allows the class to be imported and used in type hints, but raises
    a helpful error when someone tries to instantiate it without the required
    optional dependency installed.

    Args:
        class_name: Name of the class being created
        package: Name of the optional package (used in error message)

    Returns:
        A class that raises ImportError on __init__
    """

    class MissingDependencyClass:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                f"{class_name} requires the '{package}' package. "
                f"Install it with: pip install isotope-rag[{package}]"
            )

        def __class_getitem__(cls, item: Any) -> type:
            # Support generic type hints like MissingClass[T]
            return cls

    MissingDependencyClass.__name__ = class_name
    MissingDependencyClass.__qualname__ = class_name
    MissingDependencyClass.__module__ = "isotope"

    return MissingDependencyClass
