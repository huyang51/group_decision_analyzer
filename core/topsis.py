"""TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class TOPSISResult:
    """Result of a TOPSIS analysis."""
    scores: Dict[str, float]          # 各方案的相对接近度
    ranking: List[str]                # 排名 (从优到劣)
    ideal_solution: List[float]       # 正理想解
    nadir_solution: List[float]       # 负理想解
    distances_to_ideal: Dict[str, float]   # 到正理想解的距离
    distances_to_nadir: Dict[str, float]   # 到负理想解的距离


def run_topsis(
    criteria: List[str],
    score_matrix: List[List[float]],
    alternative_names: List[str],
    criteria_weights: Dict[str, float],
    criteria_types: Optional[List[str]] = None,
) -> TOPSISResult:
    """Run a complete TOPSIS analysis.

    Args:
        criteria: list of criterion names.
        score_matrix: m alternatives x n criteria score matrix (list of lists).
        alternative_names: list of alternative names (length m).
        criteria_weights: dict mapping criterion name -> weight (should sum to 1).
        criteria_types: list of "benefit" or "cost" for each criterion.
                        Default: all "benefit" (higher is better).

    Returns:
        TOPSISResult with scores, ranking, ideal/nadir solutions, distances.
    """
    n_criteria = len(criteria)
    n_alts = len(alternative_names)

    if criteria_types is None:
        criteria_types = ["benefit"] * n_criteria

    # Step 1: Vector normalization
    matrix = np.array(score_matrix, dtype=float)  # shape (n_alts, n_criteria)
    col_norms = np.sqrt((matrix ** 2).sum(axis=0))
    col_norms[col_norms == 0] = 1.0  # avoid division by zero
    normalized = matrix / col_norms

    # Step 2: Weighted normalized matrix
    weights_arr = np.array([criteria_weights.get(c, 1.0 / n_criteria) for c in criteria], dtype=float)
    weighted = normalized * weights_arr

    # Step 3: Ideal and nadir solutions
    ideal = np.zeros(n_criteria)
    nadir = np.zeros(n_criteria)
    for j in range(n_criteria):
        if criteria_types[j] == "benefit":
            ideal[j] = weighted[:, j].max()
            nadir[j] = weighted[:, j].min()
        else:  # cost
            ideal[j] = weighted[:, j].min()
            nadir[j] = weighted[:, j].max()

    # Step 4: Distances to ideal and nadir
    distances_ideal = np.sqrt(((weighted - ideal) ** 2).sum(axis=1))
    distances_nadir = np.sqrt(((weighted - nadir) ** 2).sum(axis=1))

    # Step 5: Relative closeness
    denom = distances_ideal + distances_nadir
    denom[denom == 0] = 1.0  # avoid division by zero
    closeness = distances_nadir / denom

    # Step 6: Build result
    scores = {name: float(closeness[i]) for i, name in enumerate(alternative_names)}
    sorted_alts = sorted(alternative_names, key=lambda x: -scores[x])

    return TOPSISResult(
        scores={name: round(scores[name], 4) for name in alternative_names},
        ranking=sorted_alts,
        ideal_solution=[round(float(v), 4) for v in ideal],
        nadir_solution=[round(float(v), 4) for v in nadir],
        distances_to_ideal={name: round(float(distances_ideal[i]), 4) for i, name in enumerate(alternative_names)},
        distances_to_nadir={name: round(float(distances_nadir[i]), 4) for i, name in enumerate(alternative_names)},
    )
