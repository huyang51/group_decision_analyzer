"""Dynamic simulation: add/remove voters and track how results change."""

from __future__ import annotations

from typing import List, Optional

from .models import Ballot, SimulationStep, Voter
from .condorcet import compute_full_condorcet
from .borda import compute_borda_scores
from .copeland import compute_copeland_scores


def add_voter(ballot: Ballot, voter: Voter) -> Ballot:
    """Create a new ballot with an additional voter."""
    if set(voter.preference) != set(ballot.alternatives):
        raise ValueError(
            f"Voter preference {voter.preference} "
            f"does not match alternatives {ballot.alternatives}"
        )
    new_voters = list(ballot.voters) + [voter]
    return Ballot(voters=new_voters, alternatives=list(ballot.alternatives))


def remove_voter(ballot: Ballot, voter_name: str) -> Ballot:
    """Create a new ballot with the named voter removed."""
    new_voters = [v for v in ballot.voters if v.name != voter_name]
    if len(new_voters) == len(ballot.voters):
        raise ValueError(f"Voter '{voter_name}' not found in ballot")
    return Ballot(voters=new_voters, alternatives=list(ballot.alternatives))


def run_simulation_sequence(
    initial_ballot: Ballot,
    steps: List[dict],
) -> List[SimulationStep]:
    """Run a sequence of simulation steps.

    Each step is a dict with:
        - 'action': 'add' or 'remove'
        - 'voter': Voter object (for add) or voter name (for remove)
        - 'description': text description of the step

    Returns a list of SimulationStep objects, one per step.
    """
    results: List[SimulationStep] = []
    current_ballot = initial_ballot

    # Step 0: initial state
    condorcet = compute_full_condorcet(current_ballot)
    borda = compute_borda_scores(current_ballot)
    copeland = compute_copeland_scores(current_ballot)
    results.append(SimulationStep(
        step_num=0,
        description="初始状态",
        ballot=current_ballot,
        condorcet_winner=condorcet["condorcet_winner"],
        borda_ranking=borda.ranking,
        cycle_detected=condorcet["cycle_detected"],
    ))

    for i, step in enumerate(steps, 1):
        action = step["action"]
        desc = step.get("description", f"步骤 {i}")

        if action == "add":
            current_ballot = add_voter(current_ballot, step["voter"])
        elif action == "remove":
            current_ballot = remove_voter(current_ballot, step["voter"])
        else:
            raise ValueError(f"Unknown action: {action}")

        condorcet = compute_full_condorcet(current_ballot)
        borda = compute_borda_scores(current_ballot)
        copeland = compute_copeland_scores(current_ballot)
        results.append(SimulationStep(
            step_num=i,
            description=desc,
            ballot=current_ballot,
            condorcet_winner=condorcet["condorcet_winner"],
            borda_ranking=borda.ranking,
            cycle_detected=condorcet["cycle_detected"],
        ))

    return results
