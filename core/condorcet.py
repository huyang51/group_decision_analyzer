"""Condorcet pairwise comparison, cycle detection, and dominance graph."""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from .models import Ballot, PairwiseResult


def compute_pairwise_matrix(
    ballot: Ballot,
) -> Dict[Tuple[str, str], PairwiseResult]:
    """Compute pairwise comparison results for all alternative pairs.

    Returns a dict keyed by (alt_a, alt_b) where a < b alphabetically,
    with PairwiseResult recording how many voters prefer each side.
    """
    alts = ballot.alternatives
    matrix: Dict[Tuple[str, str], PairwiseResult] = {}

    for i in range(len(alts)):
        for j in range(i + 1, len(alts)):
            a, b = alts[i], alts[j]
            votes_a = sum(1 for v in ballot.voters if v.prefers(a, b))
            votes_b = ballot.num_voters - votes_a
            key = (a, b) if a < b else (b, a)
            if votes_a >= votes_b:
                matrix[key] = PairwiseResult(
                    winner=a, loser=b,
                    winner_votes=votes_a, loser_votes=votes_b,
                )
            else:
                matrix[key] = PairwiseResult(
                    winner=b, loser=a,
                    winner_votes=votes_b, loser_votes=votes_a,
                )
    return matrix


def _get_winner(matrix: Dict[Tuple[str, str], PairwiseResult], a: str, b: str) -> str:
    """Get the winner between a and b from the pairwise matrix."""
    key = (a, b) if a < b else (b, a)
    result = matrix[key]
    return result.winner


def find_condorcet_winner(
    matrix: Dict[Tuple[str, str], PairwiseResult],
    alternatives: List[str],
) -> Optional[str]:
    """Find the Condorcet winner — the alternative that beats all others pairwise.

    Returns None if no Condorcet winner exists.
    """
    for candidate in alternatives:
        is_winner = True
        for other in alternatives:
            if candidate == other:
                continue
            if _get_winner(matrix, candidate, other) != candidate:
                is_winner = False
                break
        if is_winner:
            return candidate
    return None


def detect_cycle(
    matrix: Dict[Tuple[str, str], PairwiseResult],
    alternatives: List[str],
) -> Optional[List[str]]:
    """Detect if there is a Condorcet cycle (rock-paper-scissors pattern).

    Uses DFS to find a cycle in the dominance directed graph.
    Returns the cycle path (e.g., ['A', 'B', 'C', 'A']) or None.
    """
    # Build adjacency list: edge from winner to loser
    adj: Dict[str, List[str]] = {a: [] for a in alternatives}
    for (_, _), result in matrix.items():
        adj[result.winner].append(result.loser)

    # DFS-based cycle detection
    WHITE, GRAY, BLACK = 0, 1, 2
    color: Dict[str, int] = {a: WHITE for a in alternatives}
    parent: Dict[str, Optional[str]] = {a: None for a in alternatives}

    def dfs(u: str) -> Optional[List[str]]:
        color[u] = GRAY
        for v in adj[u]:
            if color[v] == GRAY:
                # Found cycle — reconstruct path
                cycle = [v, u]
                node = u
                while parent[node] is not None and parent[node] != v:
                    node = parent[node]
                    cycle.append(node)
                cycle.reverse()
                return cycle
            if color[v] == WHITE:
                parent[v] = u
                result = dfs(v)
                if result is not None:
                    return result
        color[u] = BLACK
        return None

    for alt in alternatives:
        if color[alt] == WHITE:
            result = dfs(alt)
            if result is not None:
                return result
    return None


def build_dominance_graph(
    matrix: Dict[Tuple[str, str], PairwiseResult],
) -> List[Tuple[str, str]]:
    """Build directed edges for the dominance graph.

    Returns list of (winner, loser) tuples for visualization.
    """
    edges: List[Tuple[str, str]] = []
    for (_, _), result in matrix.items():
        edges.append((result.winner, result.loser))
    return edges


def compute_full_condorcet(ballot: Ballot) -> dict:
    """Run all Condorcet analyses on a ballot and return a summary dict."""
    matrix = compute_pairwise_matrix(ballot)
    winner = find_condorcet_winner(matrix, ballot.alternatives)
    cycle = detect_cycle(matrix, ballot.alternatives)
    edges = build_dominance_graph(matrix)

    return {
        "pairwise_matrix": matrix,
        "condorcet_winner": winner,
        "cycle_detected": cycle is not None,
        "cycle_path": cycle,
        "dominance_edges": edges,
    }
