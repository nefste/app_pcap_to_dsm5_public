# metrics/base.py
import numpy as np

STATUS_ORDER = {"OK": 0, "Caution": 1, "N/A": 2}

def status_from_thresholds(value, ok, warn=None, higher_is_worse=True):
    """
    Three Conditions: OK, Caution, N/A.
    - higher_is_worse=True: value <= ok -> OK, else Caution
    - higher_is_worse=False: value >= ok -> OK, else Caution
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ("N/A", "blue")
    if higher_is_worse:
        return ("OK", "green") if value <= ok else ("Caution", "orange")
    else:
        return ("OK", "green") if value >= ok else ("Caution", "orange")

def thresholds_text(ok, warn=None, higher_is_worse=True):
    if higher_is_worse:
        return f"Status ranges: **OK ≤ {ok}**; **Caution > {ok}** *(higher is worse)*"
    else:
        return f"Status ranges: **OK ≥ {ok}**; **Caution < {ok}** *(lower is worse)*"

def make_metric(label, value, fmt, status_tuple, ranges_str,
                latex_formula=None, latex_numbers=None, heuristic_md=None, missing_md=None,
                dist_col: str | None = None, range_cfg: dict | None = None):
    return {
        "label": label,
        "value": value,
        "fmt": fmt,
        "status_tuple": status_tuple,
        "ranges_str": ranges_str,
        "latex_formula": latex_formula,
        "latex_numbers": latex_numbers,
        "heuristic_md": heuristic_md,
        "missing_md": missing_md,
        "dist_col": dist_col,        # column in ALL_DAILY
        "range_cfg": range_cfg,      # {"ok": <float>, "higher_is_worse": <bool>}
    }
