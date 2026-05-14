"""Voting power indices for group decision analysis.

Implements:
- Shapley-Shubik power index (Shapley & Shubik, 1954)
- Banzhaf power index (Banzhaf, 1965)

These indices quantify each voter's pivotal influence in a weighted
voting system.
"""

from __future__ import annotations

from itertools import permutations
from typing import Dict, List, Optional

from .models import Ballot, PowerIndexResult


def compute_shapley_shubik(
    ballot: Ballot,
    weights: Optional[List[float]] = None,
) -> Dict[str, float]:
    """Compute Shapley-Shubik power index for each voter.

    For each permutation of voters, find the "pivotal" voter — the one
    whose addition makes the coalition go from losing to winning.
    SS index = proportion of permutations where the voter is pivotal.

    When weights are not given, all voters are treated as equal (weight=1).
    A simple majority threshold is used: floor(n/2) + 1.

    For n ≤ 10, exhaustive enumeration of all n! permutations is feasible.
    """
    n = ballot.num_voters
    names = [v.name for v in ballot.voters]

    if weights is None:
        weights = [1.0] * n

    threshold = sum(weights) / 2.0  # simple majority

    pivotal_counts: Dict[str, int] = {name: 0 for name in names}
    total_perms = 0

    # Enumerate all permutations (feasible for n ≤ 10)
    for perm in permutations(range(n)):
        total_perms += 1
        cumulative = 0.0
        for idx in perm:
            cumulative += weights[idx]
            if cumulative > threshold:
                pivotal_counts[names[idx]] += 1
                break

    # Normalize
    ss_index: Dict[str, float] = {}
    for name in names:
        ss_index[name] = pivotal_counts[name] / total_perms if total_perms > 0 else 0.0

    return ss_index


def compute_banzhaf(
    ballot: Ballot,
    weights: Optional[List[float]] = None,
) -> Dict[str, float]:
    """Compute Banzhaf power index for each voter.

    For each coalition, check if adding a voter changes the outcome
    from losing to winning (that voter is "critical").
    Banzhaf index = (critical count for voter) / (total critical counts).

    When weights are not given, all voters are treated as equal (weight=1).
    """
    n = ballot.num_voters
    names = [v.name for v in ballot.voters]

    if weights is None:
        weights = [1.0] * n

    threshold = sum(weights) / 2.0

    # Count critical occurrences for each voter
    critical_counts: Dict[str, int] = {name: 0 for name in names}

    # Enumerate all 2^n coalitions
    total_coalitions = 1 << n
    for mask in range(total_coalitions):
        # Compute weight of this coalition
        coalition_weight = sum(weights[i] for i in range(n) if mask & (1 << i))

        # Check each voter not in the coalition
        for i in range(n):
            if mask & (1 << i):
                continue
            # Check if adding voter i makes the coalition winning
            if coalition_weight <= threshold and coalition_weight + weights[i] > threshold:
                critical_counts[names[i]] += 1

    # Normalize
    total_critical = sum(critical_counts.values())
    banzhaf_index: Dict[str, float] = {}
    for name in names:
        banzhaf_index[name] = critical_counts[name] / total_critical if total_critical > 0 else 0.0

    return banzhaf_index


def compute_power_indices(
    ballot: Ballot,
    weights: Optional[List[float]] = None,
) -> PowerIndexResult:
    """Compute both Shapley-Shubik and Banzhaf power indices."""
    n = ballot.num_voters
    names = [v.name for v in ballot.voters]

    ss = compute_shapley_shubik(ballot, weights)
    banzhaf = compute_banzhaf(ballot, weights)

    # Compute pivotal counts for display
    if weights is None:
        weights = [1.0] * n
    threshold = sum(weights) / 2.0

    pivotal_counts: Dict[str, int] = {name: 0 for name in names}
    total_coalitions = 1 << n
    for mask in range(total_coalitions):
        coalition_weight = sum(weights[i] for i in range(n) if mask & (1 << i))
        for i in range(n):
            if mask & (1 << i):
                continue
            if coalition_weight <= threshold and coalition_weight + weights[i] > threshold:
                pivotal_counts[names[i]] += 1

    return PowerIndexResult(
        shapley_shubik=ss,
        banzhaf=banzhaf,
        pivotal_count=pivotal_counts,
        total_coalitions=total_coalitions,
    )
