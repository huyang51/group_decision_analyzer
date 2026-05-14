"""Consensus analysis for group decision making.

Implements:
- Consensus degree measurement (Herrera-Viedma et al., 2020)
- Minimum cost consensus model (Ben-Arieh et al., 2020)
- Bounded confidence opinion dynamics (Hegselmann-Krause model, EJOR 2022)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .models import Ballot, ConsensusResult


def compute_preference_matrix(ballot: Ballot) -> np.ndarray:
    """Convert ranked preferences to a preference matrix.

    Returns an (n_voters x n_voters) similarity matrix where
    entry [i][j] = proportion of alternative-pairs on which
    voter i and voter j agree.
    """
    n = ballot.num_voters
    alts = ballot.alternatives
    num_pairs = len(alts) * (len(alts) - 1) // 2
    if num_pairs == 0:
        return np.ones((n, n))

    # Build pairwise preference vectors for each voter
    voter_vectors = []
    for v in ballot.voters:
        vec = []
        for i in range(len(alts)):
            for j in range(i + 1, len(alts)):
                vec.append(1 if v.prefers(alts[i], alts[j]) else 0)
        voter_vectors.append(vec)

    # Compute similarity = 1 - normalized Hamming distance
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                sim_matrix[i][j] = 1.0
            else:
                agreements = sum(
                    1 for k in range(num_pairs) if voter_vectors[i][k] == voter_vectors[j][k]
                )
                sim_matrix[i][j] = agreements / num_pairs

    return sim_matrix


def compute_consensus_degree(ballot: Ballot) -> float:
    """Compute the consensus degree (CD) of a ballot.

    CD = average pairwise agreement across all voter pairs and alternative-pairs.
    CD = 1.0 means full consensus; lower values indicate disagreement.

    Reference: Herrera-Viedma et al., Information Fusion, 2020.
    """
    n = ballot.num_voters
    if n < 2:
        return 1.0

    sim_matrix = compute_preference_matrix(ballot)
    # Average of all off-diagonal entries
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += sim_matrix[i][j]
            count += 1

    return total / count if count > 0 else 1.0


def compute_voter_distances(ballot: Ballot) -> Dict[str, float]:
    """Compute each voter's distance from the group average preference profile.

    Distance = 1 - average similarity to all other voters.
    """
    sim_matrix = compute_preference_matrix(ballot)
    n = ballot.num_voters
    distances = {}
    for i, v in enumerate(ballot.voters):
        if n <= 1:
            distances[v.name] = 0.0
        else:
            avg_sim = (sim_matrix[i].sum() - 1.0) / (n - 1)  # exclude self
            distances[v.name] = 1.0 - avg_sim
    return distances


def cluster_voters(ballot: Ballot, threshold: float = 0.5) -> List[List[str]]:
    """Cluster voters based on preference similarity.

    Voters with similarity > threshold are grouped together using
    a simple connected-components approach on the similarity graph.
    """
    sim_matrix = compute_preference_matrix(ballot)
    n = ballot.num_voters
    names = [v.name for v in ballot.voters]

    # Build adjacency: connected if similarity > threshold
    visited = [False] * n
    clusters = []

    for i in range(n):
        if visited[i]:
            continue
        # BFS from i
        cluster = []
        queue = [i]
        visited[i] = True
        while queue:
            node = queue.pop(0)
            cluster.append(names[node])
            for j in range(n):
                if not visited[j] and sim_matrix[node][j] > threshold:
                    visited[j] = True
                    queue.append(j)
        clusters.append(cluster)

    return clusters


def compute_min_cost_consensus(
    ballot: Ballot,
    target_consensus: float = 0.8,
) -> ConsensusResult:
    """Minimum cost consensus model.

    Computes the minimum preference adjustments needed to reach
    the target consensus degree. Uses a greedy approach:
    identify the voter farthest from the group and suggest
    the smallest adjustments to bring them closer.

    Reference: Ben-Arieh et al., Information Sciences, 2020.
    """
    current_cd = compute_consensus_degree(ballot)
    voter_distances = compute_voter_distances(ballot)
    sim_matrix = compute_preference_matrix(ballot)
    clusters = cluster_voters(ballot)

    adjustments: Dict[str, Dict[str, float]] = {}

    if current_cd < target_consensus:
        # Find the voter(s) farthest from the group
        sorted_voters = sorted(voter_distances.items(), key=lambda x: -x[1])
        n = ballot.num_voters
        alts = ballot.alternatives

        # Suggest adjustments for the most distant voters
        for v_name, dist in sorted_voters:
            if dist <= 0:
                break
            # Find which alternative-pairs this voter disagrees on most
            voter_idx = next(
                i for i, v in enumerate(ballot.voters) if v.name == v_name
            )
            voter = ballot.voters[voter_idx]

            # Compute group majority preference for each pair
            voter_adj: Dict[str, float] = {}
            for i in range(len(alts)):
                for j in range(i + 1, len(alts)):
                    a, b = alts[i], alts[j]
                    majority_prefers_a = sum(
                        1 for v in ballot.voters if v.prefers(a, b)
                    ) > n / 2
                    voter_prefers_a = voter.prefers(a, b)
                    if majority_prefers_a != voter_prefers_a:
                        # This pair contributes to disagreement
                        pair_key = f"{a} vs {b}"
                        voter_adj[pair_key] = 1.0 / (len(alts) * (len(alts) - 1) / 2)

            if voter_adj:
                adjustments[v_name] = voter_adj

    return ConsensusResult(
        consensus_degree=current_cd,
        preference_matrix=sim_matrix,
        voter_names=[v.name for v in ballot.voters],
        voter_distances=voter_distances,
        clusters=clusters,
        min_cost_adjustments=adjustments,
    )


def run_opinion_dynamics(
    ballot: Ballot,
    epsilon: float = 0.3,
    max_rounds: int = 20,
) -> List[Dict[str, float]]:
    """Bounded confidence opinion dynamics (Hegselmann-Krause model).

    Each voter's preference ranking is encoded as a pairwise comparison
    vector. PCA is used to project these vectors onto the first principal
    component, which captures the main axis of disagreement among voters.
    This 1D opinion value is then used in the HK bounded-confidence model.

    In each round, each voter's opinion moves toward the average of all
    voters within epsilon distance (the "confidence bound").

    Returns a list of rounds, each mapping voter name to their
    normalized opinion value (0-1 scale).

    Reference: Hegselmann & Krause, JASSS 2002; EJOR 2022.
    """
    n = ballot.num_voters
    alts = ballot.alternatives
    m = len(alts)

    if n < 2 or m < 2:
        return []

    # Build pairwise preference matrix: n voters × C(m,2) pairs
    num_pairs = m * (m - 1) // 2
    pref_matrix = np.zeros((n, num_pairs))
    for i, v in enumerate(ballot.voters):
        pair_idx = 0
        for a in range(m):
            for b in range(a + 1, m):
                pref_matrix[i, pair_idx] = 1.0 if v.prefers(alts[a], alts[b]) else 0.0
                pair_idx += 1

    # Project to 1D using PCA (first principal component)
    # This captures the main axis of disagreement among voters
    pref_centered = pref_matrix - pref_matrix.mean(axis=0)
    U, S, Vt = np.linalg.svd(pref_centered, full_matrices=False)
    pc1 = U[:, 0] * S[0]  # first PC scores

    # Normalize to [0, 1]
    pc_min, pc_max = pc1.min(), pc1.max()
    if pc_max - pc_min > 1e-10:
        pc1 = (pc1 - pc_min) / (pc_max - pc_min)
    else:
        pc1 = np.full(n, 0.5)

    # Initialize opinions from PCA projection
    opinions: Dict[str, List[float]] = {}
    for i, v in enumerate(ballot.voters):
        opinions[v.name] = [float(pc1[i])]

    # Run dynamics
    history: List[Dict[str, float]] = [{v.name: opinions[v.name][0] for v in ballot.voters}]

    for _round in range(max_rounds):
        current = {v.name: opinions[v.name][-1] for v in ballot.voters}
        new_opinions: Dict[str, float] = {}

        for v in ballot.voters:
            # Find neighbors within epsilon
            neighbors = [
                u for u in ballot.voters
                if abs(current[u.name] - current[v.name]) <= epsilon
            ]
            if neighbors:
                new_opinions[v.name] = sum(current[u.name] for u in neighbors) / len(neighbors)
            else:
                new_opinions[v.name] = current[v.name]

        # Check convergence
        converged = all(
            abs(new_opinions[v.name] - current[v.name]) < 1e-6
            for v in ballot.voters
        )

        for v in ballot.voters:
            opinions[v.name].append(new_opinions[v.name])

        history.append(new_opinions)

        if converged:
            break

    return history
