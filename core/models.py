"""Data models for group decision analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any


@dataclass
class Voter:
    """A voter with a ranked preference over alternatives."""
    name: str
    preference: List[str]  # index 0 = most preferred

    def prefers(self, a: str, b: str) -> bool:
        """Return True if this voter prefers a over b."""
        return self.preference.index(a) < self.preference.index(b)


@dataclass
class Ballot:
    """A collection of voters and the set of alternatives."""
    voters: List[Voter]
    alternatives: List[str]

    @property
    def num_voters(self) -> int:
        return len(self.voters)

    @property
    def num_alternatives(self) -> int:
        return len(self.alternatives)

    def validate(self) -> None:
        """Validate that all voters rank all alternatives consistently."""
        alt_set = set(self.alternatives)
        for v in self.voters:
            pref_set = set(v.preference)
            if pref_set != alt_set:
                missing = alt_set - pref_set
                extra = pref_set - alt_set
                detail = ""
                if missing:
                    detail += f" 缺少: {missing}"
                if extra:
                    detail += f" 多余: {extra}"
                raise ValueError(
                    f"投票人 '{v.name}' 的偏好与方案不一致。{detail}"
                )


@dataclass
class PairwiseResult:
    """Result of a pairwise comparison between two alternatives."""
    winner: str
    loser: str
    winner_votes: int
    loser_votes: int

    @property
    def margin(self) -> int:
        return self.winner_votes - self.loser_votes


@dataclass
class BordaScores:
    """Borda count scores for each alternative."""
    scores: Dict[str, float]
    ranking: List[str]  # sorted by score descending

    def get_score(self, alt: str) -> float:
        return self.scores.get(alt, 0.0)


@dataclass
class ArrowCheckResult:
    """Result of Arrow's impossibility theorem checks."""
    pareto_efficient: bool
    pareto_detail: str
    iia_satisfied: bool
    iia_detail: str
    non_dictator: bool
    non_dictator_detail: str
    unrestricted_domain: bool
    unrestricted_domain_detail: str
    summary: str


@dataclass
class CopelandScores:
    """Copeland method scores (wins minus losses)."""
    scores: Dict[str, int]
    wins: Dict[str, int]
    losses: Dict[str, int]
    ranking: List[str]


@dataclass
class SimulationStep:
    """A single step in a dynamic simulation."""
    step_num: int
    description: str
    ballot: Ballot
    condorcet_winner: Optional[str]
    borda_ranking: List[str]
    cycle_detected: bool


@dataclass
class DebateRecord:
    """Record of a multi-agent debate."""
    speeches: List[Dict[str, str]]  # list of {"agent": name, "content": text, "round": int}
    summary: str
    total_rounds: int


@dataclass
class AnalysisResult:
    """Aggregated result of all analyses on a ballot."""
    ballot: Ballot
    # Pairwise
    pairwise_matrix: Dict[Tuple[str, str], PairwiseResult]
    condorcet_winner: Optional[str]
    cycle_detected: bool
    cycle_path: Optional[List[str]]
    dominance_edges: List[Tuple[str, str]]
    # Borda
    borda: BordaScores
    # Copeland
    copeland: CopelandScores
    # Arrow
    arrow: ArrowCheckResult
    # Simulation (optional)
    simulation_steps: Optional[List[SimulationStep]] = None
