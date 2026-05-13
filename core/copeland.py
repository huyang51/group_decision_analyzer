"""Copeland scoring method: wins minus losses in pairwise comparisons."""

from __future__ import annotations

from typing import Dict, List, Tuple

from .models import Ballot, CopelandScores, PairwiseResult
from .condorcet import compute_pairwise_matrix


def compute_copeland_scores(ballot: Ballot) -> CopelandScores:
    """Compute Copeland scores for each alternative.

    Copeland score = number of pairwise wins - number of pairwise losses.
    Ties count as 0.5 win and 0.5 loss.
    """
    matrix = compute_pairwise_matrix(ballot)
    alts = ballot.alternatives

    wins: Dict[str, float] = {a: 0 for a in alts}
    losses: Dict[str, float] = {a: 0 for a in alts}

    for (a, b), result in matrix.items():
        if result.winner_votes > result.loser_votes:
            wins[result.winner] += 1
            losses[result.loser] += 1
        else:
            # Tie
            wins[a] += 0.5
            wins[b] += 0.5
            losses[a] += 0.5
            losses[b] += 0.5

    scores: Dict[str, int] = {}
    for alt in alts:
        scores[alt] = wins[alt] - losses[alt]

    ranking = sorted(alts, key=lambda a: (-scores[a], a))

    return CopelandScores(
        scores=scores,
        wins={a: int(wins[a]) if wins[a] == int(wins[a]) else wins[a] for a in alts},
        losses={a: int(losses[a]) if losses[a] == int(losses[a]) else losses[a] for a in alts},
        ranking=ranking,
    )
