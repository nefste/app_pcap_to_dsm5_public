from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd


@dataclass
class MetricAutoTune:
    metric: str
    lo: float
    mid: float
    hi: float
    invert: bool
    weight: float
    centers: Tuple[float, float, float]
    boundaries: Tuple[float, float]
    counts: Tuple[int, int, int]
    p10: float
    p50: float
    p90: float


@dataclass
class AutoTuneResult:
    criterion: str
    metrics: List[MetricAutoTune]

    def to_spec(self) -> Dict[str, Any]:
        """Return cfg_state-compatible spec mapping metric -> { w, mf:{...} }"""
        out: Dict[str, Any] = {}
        for m in self.metrics:
            out[m.metric] = {
                "w": float(m.weight),
                "mf": {
                    "type": "tri",
                    "lo": float(m.lo),
                    "mid": float(m.mid),
                    "hi": float(m.hi),
                    "invert": bool(m.invert),
                },
            }
        return out


def _kmeans_1d(x: np.ndarray, k: int = 3, max_iter: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Simple 1D k-means returning (centers_sorted, labels).
    
    - Initializes centers from quantiles to be robust.
    - If any cluster ends empty, keeps its previous center.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.array([0.0, 0.0, 0.0], dtype=float), np.empty(0, dtype=int)

    # Initialize centers by quantiles inside (0,1) to avoid extremes
    qs = np.linspace(0.15, 0.85, k)
    centers = np.quantile(x, qs)
    for _ in range(max_iter):
        # Assign
        d2 = (x[:, None] - centers[None, :]) ** 2
        labels = np.argmin(d2, axis=1)
        new_centers = centers.copy()
        for j in range(k):
            mask = labels == j
            if np.any(mask):
                new_centers[j] = float(np.mean(x[mask]))
        if np.allclose(new_centers, centers, rtol=1e-5, atol=1e-8):
            centers = new_centers
            break
        centers = new_centers
    # Ensure monotonic order left->right and remap labels
    order = np.argsort(centers)
    centers_sorted = centers[order]
    inv_map = {int(order[i]): i for i in range(k)}
    labels_sorted = np.array([inv_map[int(l)] for l in labels], dtype=int)
    return centers_sorted, labels_sorted


def _cluster_boundaries_from_centers(centers: np.ndarray) -> Tuple[float, float]:
    """Return two boundaries (t12, t23) as midpoints between sorted centers.
    
    Fallback to slightly separated medians if degenerate.
    """
    c1, c2, c3 = [float(c) for c in centers]
    t12 = 0.5 * (c1 + c2)
    t23 = 0.5 * (c2 + c3)
    if not np.isfinite(t12) or not np.isfinite(t23) or not (t12 < t23):
        # Degenerate; make a minimal separation
        med = float(np.median(centers))
        eps = float(max(1e-9, 0.05 * (np.max(centers) - np.min(centers) + 1e-9)))
        t12, t23 = med - eps, med + eps
    return float(t12), float(t23)


def _robust_spread(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    q10, q90 = np.quantile(x, [0.10, 0.90])
    spread = float(q90 - q10)
    return max(0.0, spread)


def auto_tune_for_criterion(
    criterion: str,
    selected_metrics: List[str],
    daily_df: pd.DataFrame,
    hiw_map: Dict[str, bool],
    *,
    weight_strategy: str = "spread",
) -> AutoTuneResult:
    """Infer weights and membership parameters per metric using 1D k-means.
    
    Strategy per metric m:
    - Clean daily series (drop NaN/inf).
    - Fit 1D KMeans (k=3), get centers c1<c2<c3 and boundaries t12,t23.
    - Use an open triangular membership aligned to direction:
        * higher_is_worse=True  -> invert=False,  lo=t12, mid=t23, hi=+inf (right-open)
        * higher_is_worse=False -> invert=True,   lo=t12, mid=t23, hi=+inf (right-open, then inverted)
    - Weight ∝ robust spread (P90-P10), normalized to sum 1 across selected metrics.
    """
    metrics: List[MetricAutoTune] = []

    spreads: Dict[str, float] = {}
    per_metric_tmp: Dict[str, Dict[str, Any]] = {}

    for m in selected_metrics:
        if m not in daily_df.columns:
            continue
        s = pd.to_numeric(daily_df[m], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        x = s.values.astype(float)
        if x.size < 5:
            # Fallback to quantiles with minimal data
            if x.size == 0:
                centers = np.array([0.0, 0.0, 0.0])
                labels = np.empty(0, dtype=int)
            else:
                centers = np.quantile(x, [0.2, 0.5, 0.8])
                # pseudo-labels via nearest center
                labels = np.argmin((x[:, None] - centers[None, :]) ** 2, axis=1)
        else:
            centers, labels = _kmeans_1d(x, k=3)

        # Safety: ensure strictly increasing centers
        centers = np.sort(centers)
        t12, t23 = _cluster_boundaries_from_centers(centers)

        hiw = bool(hiw_map.get(m, True))
        invert = (not hiw)
        lo = float(t12)
        mid = float(t23)
        hi = float(np.inf)

        # Robust stats
        p10, p50, p90 = [float(q) for q in np.quantile(x, [0.10, 0.50, 0.90])] if x.size else (0.0, 0.0, 0.0)
        counts = (
            int(np.sum(labels == 0)) if labels.size else 0,
            int(np.sum(labels == 1)) if labels.size else 0,
            int(np.sum(labels == 2)) if labels.size else 0,
        )

        # Weighting measure per metric according to strategy
        ws = (weight_strategy or "spread").strip().lower()
        if ws in {"var", "variance"}:
            measure = float(np.var(x)) if x.size else 0.0
        elif ws in {"equal", "uniform"}:
            measure = 1.0
        else:
            measure = _robust_spread(x)

        per_metric_tmp[m] = {
            "centers": centers,
            "boundaries": (t12, t23),
            "counts": counts,
            "invert": invert,
            "lo": lo,
            "mid": mid,
            "hi": hi,
            "p10": p10,
            "p50": p50,
            "p90": p90,
        }
        spreads[m] = max(0.0, float(measure))

    # Normalize weights by spread (fallback to equal if all zero)
    total_spread = float(sum(spreads.values()))
    n = max(1, len(per_metric_tmp))
    for m, tmp in per_metric_tmp.items():
        if total_spread > 0:
            w = float(spreads[m] / total_spread)
        else:
            w = 1.0 / n
        metrics.append(
            MetricAutoTune(
                metric=m,
                lo=float(tmp["lo"]),
                mid=float(tmp["mid"]),
                hi=float(tmp["hi"]),
                invert=bool(tmp["invert"]),
                weight=float(w),
                centers=(float(tmp["centers"][0]), float(tmp["centers"][1]), float(tmp["centers"][2])),
                boundaries=(float(tmp["boundaries"][0]), float(tmp["boundaries"][1])),
                counts=(int(tmp["counts"][0]), int(tmp["counts"][1]), int(tmp["counts"][2])),
                p10=float(tmp["p10"]),
                p50=float(tmp["p50"]),
                p90=float(tmp["p90"]),
            )
        )

    # Ensure stable order
    metrics.sort(key=lambda z: z.metric)
    return AutoTuneResult(criterion=criterion, metrics=metrics)
