"""Game theory utilities: Nash equilibrium detection and best response computation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class NashResult:
    """Result of Nash equilibrium analysis."""
    equilibrium_strategies: List[Tuple[str, str]]
    is_locked: bool
    explanation: str
    payoff_matrix_desc: str
    best_responses: Dict[str, List[str]]


def compute_best_response(
    payoff_matrix: List[List[Tuple[float, float]]],
    player_idx: int,
    opponent_strategy_idx: int,
    strategies: List[str],
) -> List[str]:
    """Find the best response(s) for a player given the opponent's strategy.

    Args:
        payoff_matrix: n x m matrix of (player1_payoff, player2_payoff) tuples.
        player_idx: 0 for player 1, 1 for player 2.
        opponent_strategy_idx: index of the opponent's chosen strategy.
        strategies: list of strategy names for the player.

    Returns:
        List of best response strategy names (may be multiple if tied).
    """
    if player_idx == 0:
        # Player 1 chooses row; opponent (player 2) has fixed column
        payoffs = [payoff_matrix[i][opponent_strategy_idx][0] for i in range(len(strategies))]
    else:
        # Player 2 chooses column; opponent (player 1) has fixed row
        payoffs = [payoff_matrix[opponent_strategy_idx][j][1] for j in range(len(strategies))]

    max_payoff = max(payoffs)
    return [strategies[i] for i, p in enumerate(payoffs) if p == max_payoff]


def detect_nash_equilibrium(
    payoff_matrix: List[List[Tuple[float, float]]],
    strategies_p1: List[str],
    strategies_p2: List[str],
) -> NashResult:
    """Detect all pure-strategy Nash equilibria in a 2-player game.

    A Nash equilibrium is a strategy profile where neither player can
    improve their payoff by unilaterally changing their strategy.

    Args:
        payoff_matrix: n x m matrix of (p1_payoff, p2_payoff) tuples.
        strategies_p1: Player 1's strategy names (rows).
        strategies_p2: Player 2's strategy names (columns).

    Returns:
        NashResult with equilibria, explanation, and best response mapping.
    """
    n = len(strategies_p1)
    m = len(strategies_p2)

    equilibria: List[Tuple[str, str]] = []
    best_responses: Dict[str, List[str]] = {}

    # Compute best responses for each player
    # Player 1's best response to each of Player 2's strategies
    for j, s2 in enumerate(strategies_p2):
        br = compute_best_response(payoff_matrix, 0, j, strategies_p1)
        for b in br:
            key = f"P1→{s2}"
            best_responses.setdefault(key, []).append(b)

    # Player 2's best response to each of Player 1's strategies
    for i, s1 in enumerate(strategies_p1):
        br = compute_best_response(payoff_matrix, 1, i, strategies_p2)
        for b in br:
            key = f"P2→{s1}"
            best_responses.setdefault(key, []).append(b)

    # Check each cell for Nash equilibrium
    for i in range(n):
        for j in range(m):
            p1_payoff, p2_payoff = payoff_matrix[i][j]

            # Is row i a best response for P1 to column j?
            p1_best = compute_best_response(payoff_matrix, 0, j, strategies_p1)
            # Is column j a best response for P2 to row i?
            p2_best = compute_best_response(payoff_matrix, 1, i, strategies_p2)

            if strategies_p1[i] in p1_best and strategies_p2[j] in p2_best:
                equilibria.append((strategies_p1[i], strategies_p2[j]))

    is_locked = len(equilibria) == 1

    # Build payoff matrix description
    lines = ["| | " + " | ".join(strategies_p2) + " |"]
    lines.append("|---" * (m + 1) + "|")
    for i, s1 in enumerate(strategies_p1):
        cells = [f"({payoff_matrix[i][j][0]},{payoff_matrix[i][j][1]})" for j in range(m)]
        lines.append(f"| {s1} | " + " | ".join(cells) + " |")
    matrix_desc = "\n".join(lines)

    if equilibria:
        eq_str = ", ".join(f"({s1}, {s2})" for s1, s2 in equilibria)
        explanation = f"找到 {len(equilibria)} 个纳什均衡: {eq_str}。"
        if is_locked:
            explanation += " 均衡唯一，博弈被锁定。"
        else:
            explanation += " 存在多个均衡，结果取决于协调机制。"
    else:
        explanation = "未找到纯策略纳什均衡。博弈可能存在混合策略均衡。"

    return NashResult(
        equilibrium_strategies=equilibria,
        is_locked=is_locked,
        explanation=explanation,
        payoff_matrix_desc=matrix_desc,
        best_responses=best_responses,
    )
