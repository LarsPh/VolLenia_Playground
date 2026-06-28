"""Search framework for Plan 07 sensorimotor VolLenia experiments."""

from importlib import import_module

from .config import parse_args


def main() -> None:
    import_module("vollenia_search.runner").main()

__all__ = ["main", "parse_args"]
