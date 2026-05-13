"""Page: Case Library — browse and load predefined cases."""

from __future__ import annotations

import streamlit as st

from ..cases.case_library import CaseLibrary
from ..core.models import Voter, Ballot
from ..core.condorcet import compute_full_condorcet
from ..core.borda import compute_borda_scores


def render_page():
    """Render the case library page."""
    st.header("案例库 — 经典决策案例分析")

    case_lib = CaseLibrary()
    case_names = case_lib.get_case_names()

    if not case_names:
        st.warning("案例库为空。请确保 cases/data/ 目录下有JSON文件。")
        return

    selected_name = st.selectbox("选择案例", case_names, key="case_select")
    case = case_lib.load_case(selected_name)

    if not case:
        st.error(f"加载案例失败: {selected_name}")
        return

    # Case info
    st.subheader(case.get("name", ""))
    st.markdown(case.get("description", ""))
    st.info(f"教学点：{case.get('teaching_point', '')}")

    # Scenarios
    scenarios = case.get("scenarios", [])
    for i, scenario in enumerate(scenarios):
        with st.expander(f"场景 {i+1}: {scenario.get('name', '')}", expanded=(i == 0)):
            alternatives = scenario.get("alternatives", [])
            voter_data = scenario.get("voters", [])

            # Show preferences
            st.write("**偏好矩阵:**")
            for v in voter_data:
                st.write(f"  {v['name']}: {' > '.join(v['preference'])}")

            # Pre-compute results
            voters = [Voter(name=v["name"], preference=v["preference"]) for v in voter_data]
            ballot = Ballot(voters=voters, alternatives=alternatives)

            try:
                ballot.validate()
            except ValueError as e:
                st.error(f"数据错误: {e}")
                continue

            condorcet = compute_full_condorcet(ballot)
            borda = compute_borda_scores(ballot)

            col1, col2 = st.columns(2)
            with col1:
                if condorcet["condorcet_winner"]:
                    st.success(f"孔多塞赢家: {condorcet['condorcet_winner']}")
                else:
                    st.warning("无孔多塞赢家")
                if condorcet["cycle_detected"]:
                    st.error(f"循环: {' → '.join(condorcet['cycle_path'])}")

            with col2:
                st.write("**Borda 排名:**")
                for rank, alt in enumerate(borda.ranking, 1):
                    st.write(f"  {rank}. {alt} ({borda.scores[alt]}分)")

            # Load button
            if st.button(f"加载场景 {i+1} 到分析", key=f"load_case_{i}"):
                st.session_state["ballot"] = ballot
                st.session_state["scenario_desc"] = case.get("description", "")
                st.session_state["case_name"] = selected_name
                st.success(f"已加载: {selected_name} — 场景 {i+1}")
