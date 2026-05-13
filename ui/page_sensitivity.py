"""Page: Sensitivity Analysis — change one voter's preference and compare results."""

from __future__ import annotations

import streamlit as st

from ..core.models import Voter, Ballot
from ..core.condorcet import compute_full_condorcet
from ..core.borda import compute_borda_scores
from ..core.copeland import compute_copeland_scores


def render_page():
    """Render the sensitivity analysis page."""
    st.header("敏感性分析 — 测试偏好变化的影响")

    ballot = st.session_state.get("ballot")
    if not ballot:
        st.warning("请先在「建模」页面输入或选择一个决策场景。")
        return

    # Select voter to modify
    voter_names = [v.name for v in ballot.voters]
    selected_voter = st.selectbox("选择要修改偏好的投票人", voter_names, key="sens_voter")

    original_voter = next(v for v in ballot.voters if v.name == selected_voter)

    st.subheader(f"当前偏好: {selected_voter}")
    st.write(" > ".join(original_voter.preference))

    # New preference input
    st.subheader("修改偏好")
    new_prefs = []
    cols = st.columns(len(ballot.alternatives))
    for i, alt in enumerate(ballot.alternatives):
        with cols[i]:
            rank = st.number_input(
                f"{alt} 排名",
                min_value=1,
                max_value=len(ballot.alternatives),
                value=original_voter.preference.index(alt) + 1,
                key=f"sens_rank_{alt}",
            )
            new_prefs.append((rank, alt))

    new_prefs.sort()
    new_pref_list = [alt for _, alt in new_prefs]

    # Run comparison
    if st.button("运行敏感性分析", type="primary", key="run_sensitivity"):
        # Original results
        orig_condorcet = compute_full_condorcet(ballot)
        orig_borda = compute_borda_scores(ballot)
        orig_copeland = compute_copeland_scores(ballot)

        # Modified ballot
        new_voter = Voter(name=selected_voter, preference=new_pref_list)
        modified_voters = [new_voter if v.name == selected_voter else v for v in ballot.voters]
        modified_ballot = Ballot(voters=modified_voters, alternatives=ballot.alternatives)

        mod_condorcet = compute_full_condorcet(modified_ballot)
        mod_borda = compute_borda_scores(modified_ballot)
        mod_copeland = compute_copeland_scores(modified_ballot)

        # Display comparison
        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("修改前")
            if orig_condorcet["condorcet_winner"]:
                st.success(f"孔多塞赢家: {orig_condorcet['condorcet_winner']}")
            else:
                st.warning("无孔多塞赢家")
            if orig_condorcet["cycle_detected"]:
                st.error(f"循环: {' → '.join(orig_condorcet['cycle_path'])}")
            st.write("**Borda 排名:**")
            for i, alt in enumerate(orig_borda.ranking, 1):
                st.write(f"  {i}. {alt} ({orig_borda.scores[alt]})")

        with col2:
            st.subheader("修改后")
            if mod_condorcet["condorcet_winner"]:
                st.success(f"孔多塞赢家: {mod_condorcet['condorcet_winner']}")
            else:
                st.warning("无孔多塞赢家")
            if mod_condorcet["cycle_detected"]:
                st.error(f"循环: {' → '.join(mod_condorcet['cycle_path'])}")
            st.write("**Borda 排名:**")
            for i, alt in enumerate(mod_borda.ranking, 1):
                st.write(f"  {i}. {alt} ({mod_borda.scores[alt]})")

        # Summary of changes
        st.divider()
        st.subheader("变化分析")

        cycle_broken = orig_condorcet["cycle_detected"] and not mod_condorcet["cycle_detected"]
        cycle_created = not orig_condorcet["cycle_detected"] and mod_condorcet["cycle_detected"]
        winner_changed = orig_condorcet["condorcet_winner"] != mod_condorcet["condorcet_winner"]
        borda_changed = orig_borda.ranking != mod_borda.ranking

        if cycle_broken:
            st.success("循环被打破！偏好变化消除了循环。")
        if cycle_created:
            st.error("循环产生！偏好变化引入了新的循环。")
        if winner_changed:
            st.warning(f"孔多塞赢家变化: {orig_condorcet['condorcet_winner']} → {mod_condorcet['condorcet_winner']}")
        if borda_changed:
            st.info("Borda 排名发生变化。")
        if not any([cycle_broken, cycle_created, winner_changed, borda_changed]):
            st.success("偏好变化未改变最终结果，系统具有一定稳健性。")
