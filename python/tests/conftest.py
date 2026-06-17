from __future__ import annotations

import pytest
import torch


def pytest_sessionstart(session: pytest.Session) -> None:
    if not torch.cuda.is_available():
        raise pytest.UsageError(
            "VolLenia PyTorch tests require CUDA PyTorch. "
            "Run `uv sync --refresh`, then verify torch.cuda.is_available() is true."
        )
