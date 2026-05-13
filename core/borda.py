"""Borda count voting method."""

from __future__ import annotations

from typing import Dict, List

from .models import Ballot, BordaScores


def compute_borda_scores(ballot: Ballot) -> BordaScores:
    """Compute Borda count scores.

    For m alternatives, the voter's 1st choice gets (m-1) points,
    2nd gets (m-2), ..., last gets 0.
    """
    m = ballot.num_alternatives
    scores: Dict[str, float] = {alt: 0.0 for alt in ballot.alternatives}

    for voter in ballot.voters:
        for rank, alt in enumerate(voter.preference):
            scores[alt] += m - 1 - rank

    # Sort by score descending, then alphabetically for tie-breaking
    ranking = sorted(scores.keys(), key=lambda a: (-scores[a], a))

    return BordaScores(scores=scores, ranking=ranking)


def rank_by_borda(scores: BordaScores) -> List[str]:
    """Return the Borda ranking as a list."""
    return scores.ranking
