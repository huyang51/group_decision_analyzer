"""Data models for group decision analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

import numpy as np
from numpy import ndarray


@dataclass
class DecisionMatrix:
    """Multi-criteria decision matrix (per-voter score matrices)."""
    criteria: List[str]                                        # 准则名称列表
    voter_score_matrices: Dict[str, Dict[str, List[float]]]   # voter_name → {alt_name → scores}
    alternatives: List[str]                                    # 方案名称列表

    # Backward-compatible property: average matrix as if shared
    @property
    def score_matrix(self) -> Dict[str, List[float]]:
        """Return average score matrix across all voters (backward compat)."""
        return self.get_average_matrix()

    def get_voter_matrix(self, voter_name: str) -> Dict[str, List[float]]:
        """Get score matrix for a specific voter."""
        return self.voter_score_matrices[voter_name]

    def get_voter_matrix_as_list(self, voter_name: str) -> List[List[float]]:
        """Get score matrix as list-of-lists for TOPSIS/AHP consumption."""
        mat = self.voter_score_matrices[voter_name]
        return [mat[a] for a in self.alternatives]

    def get_average_matrix(self) -> Dict[str, List[float]]:
        """Get average score matrix across all voters (for sensitivity analysis)."""
        result = {}
        for alt in self.alternatives:
            all_scores = [self.voter_score_matrices[vn][alt] for vn in self.voter_score_matrices]
            result[alt] = np.mean(all_scores, axis=0).tolist()
        return result

    def get_average_matrix_as_list(self) -> List[List[float]]:
        """Average matrix as list-of-lists."""
        avg = self.get_average_matrix()
        return [avg[a] for a in self.alternatives]

    def get_scores(self, alt: str) -> List[float]:
        return self.score_matrix[alt]

    def get_criterion_scores(self, criterion_idx: int) -> Dict[str, float]:
        return {alt: scores[criterion_idx] for alt, scores in self.score_matrix.items()}


@dataclass
class Voter:
    """A voter with a ranked preference over alternatives."""
    name: str
    preference: List[str]  # index 0 = most preferred
    criteria_weights: Optional[Dict[str, float]] = None  # 该投票人的准则权重
    voting_weight: float = 1.0  # 投票权重，用于权力指数计算（SS/Banzhaf）

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
    # MCDM (optional)
    decision_matrix: Optional[DecisionMatrix] = None
    ahp_results: Optional[Dict[str, Any]] = None       # voter_name → AHPResult
    topsis_results: Optional[Dict[str, Any]] = None     # voter_name → TOPSISResult
    sensitivity_results: Optional[Any] = None            # SensitivityResult


@dataclass
class ConsensusResult:
    """Result of consensus analysis (GDM methods)."""
    consensus_degree: float                          # 0-1, 1=完全共识
    preference_matrix: ndarray                       # 偏好相似度矩阵
    voter_names: List[str]                           # 投票人名称列表
    voter_distances: Dict[str, float]                # 每个投票人与群体平均的距离
    clusters: List[List[str]]                        # 意见相近的投票人分组
    min_cost_adjustments: Dict[str, Dict[str, float]] = field(default_factory=dict)  # 达成共识的最小调整
    dynamics_history: Optional[List[Dict[str, float]]] = None  # 意见动力学历史


@dataclass
class PowerIndexResult:
    """Result of voting power index analysis."""
    shapley_shubik: Dict[str, float]   # 每个投票人的SS指数
    banzhaf: Dict[str, float]          # 每个投票人的Banzhaf指数
    pivotal_count: Dict[str, int]      # 每个投票人成为关键票的次数
    total_coalitions: int              # 总联盟数


@dataclass
class NonCooperativeResult:
    """Result of non-cooperative behavior detection."""
    suspicious_voters: List[str]       # 疑似策略性投票的投票人
    manipulation_type: Dict[str, str]  # 操纵类型
    deviation_scores: Dict[str, float] # 偏离"诚实投票"的程度
    explanation: str                   # 综合解释
