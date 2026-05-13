"""Page: Dynamic Simulation — add/remove voters and track result changes."""

from __future__ import annotations

import streamlit as st

from ..core.models import Voter, Ballot
from ..core.simulation import add_voter, remove_voter, run_simulation_sequence
from ..core.condorcet import compute_full_condorcet
from ..core.borda import compute_borda_scores
from ..core.visualization import plot_simulation_results


def render_page():
    """Render the simulation page."""
    st.header("步骤3：动态推演 — 模拟利益相关者变化")

    ballot = st.session_state.get("ballot")
    if not ballot:
        st.warning("请先在「建模」页面输入或选择一个决策场景。")
        return

    st.subheader("当前利益相关者")
    for v in ballot.voters:
        col1, col2, col3 = st.columns([3, 5, 2])
        with col1:
            st.write(f"**{v.name}**")
        with col2:
            st.write(" > ".join(v.preference))
        with col3:
            if st.button("删除", key=f"del_{v.name}"):
                try:
                    new_ballot = remove_voter(ballot, v.name)
                    st.session_state["ballot"] = new_ballot
                    st.rerun()
                except ValueError:
                    st.error("不能删除：至少需要2位投票人。")

    st.divider()

    # Add voter form
    st.subheader("添加利益相关者")
    with st.form("add_voter_form"):
        new_name = st.text_input("名称", value=f"投票人{len(ballot.voters)+1}")
        new_prefs = []
        cols = st.columns(len(ballot.alternatives))
        for i, alt in enumerate(ballot.alternatives):
            with cols[i]:
                rank = st.number_input(
                    f"{alt} 排名",
                    min_value=1,
                    max_value=len(ballot.alternatives),
                    value=i + 1,
                    key=f"new_rank_{alt}",
                )
                new_prefs.append((rank, alt))

        submitted = st.form_submit_button("添加")
        if submitted:
            new_prefs.sort()
            pref_list = [alt for _, alt in new_prefs]
            if len(set(pref_list)) != len(ballot.alternatives):
                st.error("偏好排序包含重复方案，请检查。")
            else:
                new_voter = Voter(name=new_name, preference=pref_list)
                new_ballot = add_voter(ballot, new_voter)
                st.session_state["ballot"] = new_ballot
                st.success(f"已添加 {new_name}")
                st.rerun()

    st.divider()

    # Simulation results
    st.subheader("推演结果")
    current_ballot = st.session_state.get("ballot")
    if current_ballot:
        condorcet = compute_full_condorcet(current_ballot)
        borda = compute_borda_scores(current_ballot)

        col1, col2 = st.columns(2)
        with col1:
            if condorcet["condorcet_winner"]:
                st.success(f"孔多塞赢家: **{condorcet['condorcet_winner']}**")
            else:
                st.warning("无孔多塞赢家")
            if condorcet["cycle_detected"]:
                st.error(f"循环: {' → '.join(condorcet['cycle_path'])}")

        with col2:
            st.write("**Borda 排名:**")
            for i, alt in enumerate(borda.ranking, 1):
                st.write(f"  {i}. {alt} ({borda.scores[alt]}分)")

    # Timeline chart (if we have simulation history)
    if "simulation_steps" in st.session_state and st.session_state["simulation_steps"]:
        st.subheader("推演时间线")
        fig = plot_simulation_results(st.session_state["simulation_steps"])
        st.plotly_chart(fig, use_container_width=True)
