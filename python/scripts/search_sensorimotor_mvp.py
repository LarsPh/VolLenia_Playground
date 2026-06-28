from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vollenia_search import runner as _runner


if __name__ == "__main__":
    _runner.main()
else:
    sys.modules[__name__] = _runner
