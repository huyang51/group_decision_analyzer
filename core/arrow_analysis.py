"""Arrow's Impossibility Theorem analysis.

Checks four conditions:
1. Unrestricted Domain — voters can have any preference ordering
2. Pareto Efficiency — if everyone prefers A to B, society prefers A to B
3. Independence of Irrelevant Alternatives (IIA) — simplified check
4. Non-Dictatorship — no single voter determines the outcome
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .models import ArrowCheckResult, Ballot, PairwiseResult
from .condorcet import compute_pairwise_matrix, find_condorcet_winner


def check_unrestricted_domain(ballot: Ballot) -> Tuple[bool, str]:
    """Check if the domain allows all possible preference orderings.

    In our system, voters can express any complete ranking, so this is satisfied.
    """
    return True, "所有投票者可以自由表达任意偏好排序，无域限制。"


def check_pareto_efficiency(
    ballot: Ballot,
    matrix: Dict[Tuple[str, str], PairwiseResult],
) -> Tuple[bool, str]:
    """Check Pareto efficiency: if ALL voters prefer A over B,
    then A must be socially preferred to B.

    This is trivially satisfied in pairwise majority rule when unanimity exists.
    """
    violations = []
    for (a, b), result in matrix.items():
        # Check if all voters prefer a over b
        all_prefer_a = all(v.prefers(a, b) for v in ballot.voters)
        all_prefer_b = all(v.prefers(b, a) for v in ballot.voters)

        if all_prefer_a and result.winner != a:
            violations.append(f"全体偏好 {a}>{b}，但社会选择 {b}>{a}")
        if all_prefer_b and result.winner != b:
            violations.append(f"全体偏好 {b}>{a}，但社会选择 {a}>{b}")

    if violations:
        return False, "帕累托效率违反: " + "; ".join(violations)
    return True, "帕累托效率满足：当所有投票者一致偏好 A>B 时，社会排序也为 A>B。"


def check_iia(
    ballot: Ballot,
    matrix: Dict[Tuple[str, str], PairwiseResult],
) -> Tuple[bool, str]:
    """Simplified IIA check: the social preference between A and B
    should depend only on individual preferences between A and B,
    not on other alternatives.

    We approximate this by checking if removing a third alternative
    changes the pairwise outcome between any two alternatives.
    """
    alts = ballot.alternatives
    if len(alts) < 3:
        return True, "少于3个方案，IIA 条件平凡满足。"

    violations = []
    for removed_alt in alts:
        remaining = [a for a in alts if a != removed_alt]
        # Build sub-ballot with only remaining alternatives
        from .models import Voter, Ballot as BallotModel
        sub_voters = []
        for v in ballot.voters:
            sub_pref = [a for a in v.preference if a in remaining]
            sub_voters.append(Voter(name=v.name, preference=sub_pref))
        sub_ballot = BallotModel(voters=sub_voters, alternatives=remaining)
        sub_matrix = compute_pairwise_matrix(sub_ballot)

        # Compare pairwise outcomes
        for i in range(len(remaining)):
            for j in range(i + 1, len(remaining)):
                a, b = remaining[i], remaining[j]
                key = (a, b) if a < b else (b, a)
                orig_winner = matrix[key].winner
                sub_winner = sub_matrix[key].winner
                if orig_winner != sub_winner:
                    violations.append(
                        f"移除 {removed_alt} 后，{a} vs {b} 的赢家从 "
                        f"{orig_winner} 变为 {sub_winner}"
                    )

    if violations:
        return False, "IIA 违反: " + "; ".join(violations)
    return True, "IIA 满足：移除任何第三方方案不改变剩余方案的两两比较结果。"


def check_non_dictatorship(ballot: Ballot) -> Tuple[bool, str]:
    """Check that no single voter is a dictator (their preference always
    becomes the social preference regardless of others).
    """
    if ballot.num_voters <= 1:
        return False, "仅有一位投票者，该投票者是天然的独裁者。"

    alts = ballot.alternatives
    from .models import Voter, Ballot as BallotModel

    for potential_dictator in ballot.voters:
        is_dictator = True
        # For every pair, check if social outcome matches dictator's preference
        for i in range(len(alts)):
            for j in range(i + 1, len(alts)):
                a, b = alts[i], alts[j]
                # Create a ballot where dictator has a>b, others have b>a
                test_voters = []
                for v in ballot.voters:
                    if v.name == potential_dictator.name:
                        test_voters.append(Voter(name=v.name, preference=potential_dictator.preference))
                    else:
                        # Reverse preference for non-dictators
                        rev = list(reversed(v.preference))
                        test_voters.append(Voter(name=v.name, preference=rev))
                test_ballot = BallotModel(voters=test_voters, alternatives=alts)
                test_matrix = compute_pairwise_matrix(test_ballot)
                key = (a, b) if a < b else (b, a)
                social_winner = test_matrix[key].winner
                dict_prefers_a = potential_dictator.prefers(a, b)
                expected = a if dict_prefers_a else b
                if social_winner != expected:
                    is_dictator = False
                    break
            if not is_dictator:
                break

        if is_dictator:
            return False, f"检测到独裁者: {potential_dictator.name} 的偏好总是决定社会选择。"

    return True, "非独裁性满足：没有任何单个投票者的偏好能始终决定社会选择。"


def run_full_arrow_check(ballot: Ballot) -> ArrowCheckResult:
    """Run all four Arrow condition checks."""
    matrix = compute_pairwise_matrix(ballot)

    ud_ok, ud_detail = check_unrestricted_domain(ballot)
    pe_ok, pe_detail = check_pareto_efficiency(ballot, matrix)
    iia_ok, iia_detail = check_iia(ballot, matrix)
    nd_ok, nd_detail = check_non_dictatorship(ballot)

    all_pass = all([ud_ok, pe_ok, iia_ok, nd_ok])
    if all_pass:
        summary = "所有四个条件均满足，社会选择函数在此案例中符合理性要求。"
    else:
        failed = []
        if not pe_ok:
            failed.append("帕累托效率")
        if not iia_ok:
            failed.append("IIA")
        if not nd_ok:
            failed.append("非独裁性")
        summary = f"以下条件未满足: {', '.join(failed)}。这体现了 Arrow 不可能定理的核心困境。"

    return ArrowCheckResult(
        pareto_efficient=pe_ok,
        pareto_detail=pe_detail,
        iia_satisfied=iia_ok,
        iia_detail=iia_detail,
        non_dictator=nd_ok,
        non_dictator_detail=nd_detail,
        unrestricted_domain=ud_ok,
        unrestricted_domain_detail=ud_detail,
        summary=summary,
    )
