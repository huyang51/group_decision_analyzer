"""MCDM Sensitivity Analysis — vary criterion weights and track rank changes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from .models import DecisionMatrix
from .topsis import run_topsis
from .ahp import rank_alternatives


@dataclass
class SensitivityResult:
    """Result of MCDM sensitivity analysis."""
    base_ranking: List[str]
    weight_variations: List[Dict[str, float]]   # 测试的权重组合
    rankings_per_variation: List[List[str]]      # 每种权重下的排名
    rank_changes: Dict[str, List[int]]           # 每个方案在不同权重下的排名变化
    critical_criterion: str                      # 对排名影响最大的准则
    stability_score: float                       # 排名稳定性 (0-1, 1=完全稳定)


def run_sensitivity_analysis(
    decision_matrix: DecisionMatrix,
    base_weights: Dict[str, float],
    method: str = "topsis",
    variation_range: Tuple[float, float] = (0.1, 0.9),
    n_steps: int = 10,
) -> SensitivityResult:
    """Run sensitivity analysis by varying each criterion's weight.

    For each criterion, its weight is varied from variation_range[0] to
    variation_range[1] in n_steps. Other criteria weights are adjusted
    proportionally so the total sums to 1.

    Args:
        decision_matrix: the DecisionMatrix object.
        base_weights: baseline criteria weights (dict mapping criterion name -> weight).
        method: "topsis" or "ahp".
        variation_range: (min_weight, max_weight) for the varied criterion.
        n_steps: number of variation steps.

    Returns:
        SensitivityResult with rankings, rank changes, critical criterion, stability.
    """
    criteria = decision_matrix.criteria
    alts = decision_matrix.alternatives
    score_matrix_list = decision_matrix.get_average_matrix_as_list()

    # Compute base ranking
    if method == "topsis":
        base_result = run_topsis(criteria, score_matrix_list, alts, base_weights)
        base_ranking = base_result.ranking
    else:
        base_ranked = rank_alternatives(base_weights, score_matrix_list, alts)
        base_ranking = [name for name, _ in base_ranked]

    weight_variations = []
    rankings_per_variation = []
    rank_changes: Dict[str, List[int]] = {alt: [] for alt in alts}

    test_weights = np.linspace(variation_range[0], variation_range[1], n_steps)

    for crit in criteria:
        for w in test_weights:
            # Build new weight vector
            new_weights = dict(base_weights)
            new_weights[crit] = w

            # Adjust other weights proportionally
            other_crits = [c for c in criteria if c != crit]
            other_sum = sum(base_weights[c] for c in other_crits)
            remaining = 1.0 - w

            if other_sum > 0:
                for c in other_crits:
                    new_weights[c] = (base_weights[c] / other_sum) * remaining
            else:
                # Edge case: all other weights were 0
                share = remaining / len(other_crits) if other_crits else 0
                for c in other_crits:
                    new_weights[c] = share

            # Run MCDM with new weights
            if method == "topsis":
                result = run_topsis(criteria, score_matrix_list, alts, new_weights)
                ranking = result.ranking
            else:
                ranked = rank_alternatives(new_weights, score_matrix_list, alts)
                ranking = [name for name, _ in ranked]

            weight_variations.append({c: round(new_weights[c], 4) for c in criteria})
            rankings_per_variation.append(ranking)

            # Record rank for each alternative
            for alt in alts:
                rank = ranking.index(alt) + 1
                rank_changes[alt].append(rank)

    # Find critical criterion: the one whose variation causes the most rank volatility
    crit_volatility = {}
    steps_per_crit = n_steps
    for i, crit in enumerate(criteria):
        volatility = 0.0
        start_idx = i * steps_per_crit
        end_idx = start_idx + steps_per_crit
        for alt in alts:
            ranks_slice = rank_changes[alt][start_idx:end_idx]
            if len(ranks_slice) > 1:
                volatility += np.std(ranks_slice)
        crit_volatility[crit] = volatility

    critical_criterion = max(crit_volatility, key=crit_volatility.get)

    # Stability score: proportion of variations where the top-ranked alternative stays the same
    top_in_base = base_ranking[0]
    stable_count = sum(1 for r in rankings_per_variation if r[0] == top_in_base)
    stability_score = stable_count / len(rankings_per_variation) if rankings_per_variation else 1.0

    return SensitivityResult(
        base_ranking=base_ranking,
        weight_variations=weight_variations,
        rankings_per_variation=rankings_per_variation,
        rank_changes=rank_changes,
        critical_criterion=critical_criterion,
        stability_score=round(stability_score, 4),
    )
