from __future__ import annotations

"""Shared markdown explanations for metrics.

This module collects the ``explanation_md`` strings from all criterion
modules so that pages can reuse them without duplicating text.
"""

from typing import Dict

from metrics.criterion1 import C1_DEFS
from metrics.criterion2 import C2_DEFS
from metrics.criterion3 import C3_DEFS
from metrics.criterion4 import C4_DEFS
from metrics.criterion5 import C5_DEFS
from metrics.criterion6 import C6_DEFS
from metrics.criterion7 import C7_DEFS
from metrics.criterion8 import C8_DEFS
from metrics.criterion9 import C9_DEFS

MD_EXPLANATIONS: Dict[str, str] = {}
for defs in [
    C1_DEFS,
    C2_DEFS,
    C3_DEFS,
    C4_DEFS,
    C5_DEFS,
    C6_DEFS,
    C7_DEFS,
    C8_DEFS,
    C9_DEFS,
]:
    for d in defs:
        if getattr(d, "explanation_md", None):
            MD_EXPLANATIONS[d.dist_col] = d.explanation_md  # type: ignore[attr-defined]

__all__ = ["MD_EXPLANATIONS"]
