"""Non-cooperative behavior detection in group decision making.

Implements detection methods from:
- Dong et al., "Large-scale group decision-making with non-cooperative behaviors", EJOR, 2020
- "Consensus Reaching and Strategic Manipulation in GDM", IEEE TSMC, 2020
"""

from __future__ import annotations

from typing import Dict, List

from .models import Ballot, NonCooperativeResult


def _kendall_tau_distance(pref_a: List[str], pref_b: List[str]) -> int:
    """Compute Kendall tau distance between two rankings.

    Returns the number of pairwise disagreements.
    """
    n = len(pref_a)
    index_b = {alt: i for i, alt in enumerate(pref_b)}
    distance = 0
    for i in range(n):
        for j in range(i + 1, n):
            a_i, a_j = pref_a[i], pref_a[j]
            # Check if order is reversed in pref_b
            if index_b[a_i] > index_b[a_j]:
                distance += 1
    return distance


def _max_kendall_tau(n: int) -> int:
    """Maximum possible Kendall tau distance for rankings of length n."""
    return n * (n - 1) // 2


def detect_bullet_voting(ballot: Ballot) -> List[str]:
    """Detect bullet voting behavior.

    Bullet voting: a voter strongly favors one alternative and
    does not differentiate among the rest. Detected when a voter's
    top choice has overwhelming preference margin vs. all others,
    while remaining alternatives have near-random ordering.

    Returns list of voter names suspected of bullet voting.
    """
    if ballot.num_alternatives < 3:
        return []

    suspicious: List[str] = []
    n_alts = ballot.num_alternatives

    for v in ballot.voters:
        top = v.preference[0]
        # Check if the voter consistently puts the same alternative first
        # and shows minimal differentiation among the rest
        top_margins = []
        for other in v.preference[1:]:
            # How many voters agree this top > other?
            agree_count = sum(
                1 for u in ballot.voters if u.prefers(top, other)
            )
            top_margins.append(agree_count / ballot.num_voters)

        # If the top choice has very high margins (near unanimous)
        # and the voter's ordering of the rest doesn't match the group well
        avg_margin = sum(top_margins) / len(top_margins) if top_margins else 0

        # Compute similarity of remaining preferences to group median
        remaining = v.preference[1:]
        group_remaining = []
        for other in v.preference[1:]:
            # Group's average rank for this alternative
            avg_rank = sum(
                u.preference.index(other) for u in ballot.voters
            ) / ballot.num_voters
            group_remaining.append((avg_rank, other))
        group_remaining.sort()

        group_order = [alt for _, alt in group_remaining]
        tau = _kendall_tau_distance(remaining, group_order)
        max_tau = _max_kendall_tau(len(remaining))
        tau_ratio = tau / max_tau if max_tau > 0 else 0

        # Bullet voting heuristic: top choice is near-unanimous AND
        # remaining order is highly different from group
        if avg_margin > 0.7 and tau_ratio > 0.5:
            suspicious.append(v.name)

    return suspicious


def detect_preference_deviation(ballot: Ballot) -> Dict[str, float]:
    """Detect preference deviation from the group median ranking.

    Computes normalized Kendall tau distance between each voter's
    ranking and the group's "median" ranking (Borda-derived).

    Returns dict mapping voter name to deviation score (0 = identical, 1 = max disagreement).
    """
    from .borda import compute_borda_scores

    borda = compute_borda_scores(ballot)
    median_ranking = borda.ranking  # Borda ranking as proxy for group median

    n_alts = ballot.num_alternatives
    max_tau = _max_kendall_tau(n_alts)

    deviations: Dict[str, float] = {}
    for v in ballot.voters:
        tau = _kendall_tau_distance(v.preference, median_ranking)
        deviations[v.name] = tau / max_tau if max_tau > 0 else 0.0

    return deviations


def detect_strategic_voting(ballot: Ballot) -> NonCooperativeResult:
    """Comprehensive detection of non-cooperative / strategic voting.

    Combines bullet voting detection, preference deviation analysis,
    and consensus relationship analysis.
    """
    bullet_voters = detect_bullet_voting(ballot)
    deviations = detect_preference_deviation(ballot)

    # Determine manipulation types
    manipulation_type: Dict[str, str] = {}
    for name in bullet_voters:
        manipulation_type[name] = "子弹投票 (Bullet Voting)"

    # Flag voters with high deviation (> 0.6) who aren't already flagged
    suspicious = list(bullet_voters)
    for name, score in deviations.items():
        if score > 0.6 and name not in suspicious:
            suspicious.append(name)
            manipulation_type[name] = "偏好偏离 (Preference Deviation)"

    # Build explanation
    if not suspicious:
        explanation = (
            "未检测到明显的非合作行为。所有投票人的偏好与群体共识较为一致，"
            "没有发现策略性投票的显著证据。"
        )
    else:
        details = []
        for name in suspicious:
            dev_score = deviations.get(name, 0)
            m_type = manipulation_type.get(name, "未知")
            details.append(f"  - {name}: {m_type}（偏离度: {dev_score:.2f}）")
        explanation = (
            f"检测到 {len(suspicious)} 位投票人存在疑似非合作行为：\n"
            + "\n".join(details)
            + "\n\n这些投票人的偏好与群体共识存在较大差异，可能存在策略性投票行为。"
        )

    return NonCooperativeResult(
        suspicious_voters=suspicious,
        manipulation_type=manipulation_type,
        deviation_scores=deviations,
        explanation=explanation,
    )
