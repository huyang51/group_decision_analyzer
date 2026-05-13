"""Analytic Hierarchy Process (AHP) for multi-criteria decision analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class AHPResult:
    """Result of an AHP analysis."""
    weights: Dict[str, float]
    CR: float
    ranking: List[str]
    is_consistent: bool
    lambda_max: float


def compute_ahp_weights(criteria_matrix: List[List[float]]) -> Tuple[np.ndarray, float]:
    """Compute AHP weight vector from a pairwise comparison matrix using the eigenvalue method.

    Args:
        criteria_matrix: n x n matrix where [i][j] = importance of i relative to j.
                         Scale: 1 = equal, 3 = moderate, 5 = strong, 7 = very strong, 9 = extreme.

    Returns:
        (weights, lambda_max) — normalized weight vector and principal eigenvalue.
    """
    matrix = np.array(criteria_matrix, dtype=float)
    n = matrix.shape[0]

    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    # Find the eigenvalue with the largest real part
    idx = np.argmax(eigenvalues.real)
    lambda_max = eigenvalues[idx].real

    # Weight vector from the corresponding eigenvector (take real part, normalize)
    weights_raw = eigenvectors[:, idx].real
    weights = weights_raw / weights_raw.sum()

    return weights, lambda_max


def compute_consistency_ratio(
    matrix: List[List[float]],
    weights: np.ndarray,
) -> float:
    """Compute the Consistency Ratio (CR) for the AHP judgment matrix.

    CR = CI / RI, where CI = (lambda_max - n) / (n - 1).
    RI is the Random Index for the given matrix size.

    CR <= 0.10 is considered acceptable.
    """
    n = len(matrix)
    if n <= 2:
        return 0.0  # 1x1 and 2x2 matrices are always consistent

    arr = np.array(matrix, dtype=float)
    lambda_max = np.real(np.linalg.eigvals(arr).max())
    ci = (lambda_max - n) / (n - 1)

    # Random Index (Saaty's table)
    ri_table = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
    ri = ri_table.get(n, 1.49)

    if ri == 0:
        return 0.0
    return ci / ri


def rank_alternatives(
    criteria_weights: Dict[str, float],
    alternatives_matrix: List[List[float]],
    alternative_names: List[str],
) -> List[Tuple[str, float]]:
    """Rank alternatives by weighted sum of criteria scores.

    Args:
        criteria_weights: dict mapping criterion name -> weight.
        alternatives_matrix: m alternatives x n criteria score matrix.
        alternative_names: list of alternative names.

    Returns:
        List of (alternative_name, score) sorted by score descending.
    """
    scores_arr = np.array(alternatives_matrix, dtype=float)
    weights_arr = np.array(list(criteria_weights.values()), dtype=float)

    # Normalize each criterion column to sum to 1 (benefit normalization)
    col_sums = scores_arr.sum(axis=0)
    col_sums[col_sums == 0] = 1  # avoid division by zero
    normalized = scores_arr / col_sums

    # Weighted sum
    final_scores = normalized @ weights_arr

    result = [(name, float(score)) for name, score in zip(alternative_names, final_scores)]
    result.sort(key=lambda x: -x[1])
    return result


def run_ahp(
    criteria_names: List[str],
    criteria_matrix: List[List[float]],
    alternative_names: List[str],
    alternatives_matrix: List[List[float]],
) -> AHPResult:
    """Run a complete AHP analysis.

    Args:
        criteria_names: list of criterion names.
        criteria_matrix: pairwise comparison matrix for criteria.
        alternative_names: list of alternative names.
        alternatives_matrix: score matrix (alternatives x criteria).

    Returns:
        AHPResult with weights, CR, ranking, consistency flag.
    """
    weights_arr, lambda_max = compute_ahp_weights(criteria_matrix)
    cr = compute_consistency_ratio(criteria_matrix, weights_arr)

    criteria_weights = {name: float(w) for name, w in zip(criteria_names, weights_arr)}

    ranking_pairs = rank_alternatives(criteria_weights, alternatives_matrix, alternative_names)
    ranking = [name for name, _ in ranking_pairs]

    return AHPResult(
        weights={name: round(w, 4) for name, w in criteria_weights.items()},
        CR=round(cr, 4),
        ranking=ranking,
        is_consistent=cr <= 0.10,
        lambda_max=round(lambda_max, 4),
    )
