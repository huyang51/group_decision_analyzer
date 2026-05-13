"""Page: Modeling — scenario input and preference configuration."""

from __future__ import annotations

import json
import streamlit as st

from ..cases.case_library import CaseLibrary
from ..core.models import Voter, Ballot


def _build_custom_ballot() -> Ballot:
    """Build a ballot from custom user input."""
    num_voters = st.number_input("投票人数", min_value=2, max_value=10, value=3, key="num_voters")
    num_alts = st.number_input("方案数", min_value=2, max_value=8, value=3, key="num_alts")

    st.subheader("方案名称")
    alternatives = []
    for i in range(num_alts):
        alt = st.text_input(f"方案 {i+1}", value=chr(65 + i), key=f"alt_{i}")
        alternatives.append(alt)

    st.subheader("投票者偏好（从最偏好到最不偏好）")
    voters = []
    for i in range(num_voters):
        cols = st.columns([1, num_alts])
        with cols[0]:
            name = st.text_input("名称", value=f"投票人{i+1}", key=f"voter_name_{i}", label_visibility="collapsed")
        prefs = []
        for j in range(num_alts):
            with cols[j + 1] if j + 1 < len(cols) else cols[-1]:
                pref = st.selectbox(
                    f"第{j+1}偏好",
                    alternatives,
                    index=j,
                    key=f"pref_{i}_{j}",
                    label_visibility="collapsed",
                )
                prefs.append(pref)
        # Validate uniqueness
        if len(set(prefs)) == len(alts):
            voters.append(Voter(name=name, preference=prefs))
        else:
            st.warning(f"{name} 的偏好包含重复方案，请检查。")

    return Ballot(voters=voters, alternatives=alternatives)


def _load_preset_case(case_lib: CaseLibrary) -> tuple:
    """Load a preset case from the case library."""
    case_names = case_lib.get_case_names()
    if not case_names:
        st.warning("案例库为空，请确保 cases/data/ 目录下有JSON文件。")
        return None, None, None

    selected_case = st.selectbox("选择案例", case_names, key="preset_case")
    case = case_lib.load_case(selected_case)

    if not case:
        return None, None, None

    st.markdown(f"**{case.get('name', '')}**")
    st.info(case.get("description", ""))
    st.caption(f"教学点：{case.get('teaching_point', '')}")

    # Select scenario
    scenarios = case.get("scenarios", [])
    if not scenarios:
        return None, None, None

    scenario_names = [s.get("name", f"场景{i+1}") for i, s in enumerate(scenarios)]
    selected_scenario_name = st.selectbox("选择场景", scenario_names, key="preset_scenario")
    scenario_idx = scenario_names.index(selected_scenario_name)
    scenario = scenarios[scenario_idx]

    # Build ballot
    alternatives = scenario.get("alternatives", [])
    voters = []
    for v in scenario.get("voters", []):
        voters.append(Voter(name=v["name"], preference=v["preference"]))
    ballot = Ballot(voters=voters, alternatives=alternatives)

    return ballot, case.get("description", ""), selected_case


def render_page():
    """Render the modeling page."""
    st.header("步骤1：建模 — 输入决策场景")

    tab_preset, tab_custom = st.tabs(["预设案例", "自定义输入"])

    with tab_preset:
        case_lib = CaseLibrary()
        ballot, description, case_name = _load_preset_case(case_lib)
        if ballot:
            st.session_state["ballot"] = ballot
            st.session_state["scenario_desc"] = description
            st.session_state["case_name"] = case_name
            st.success(f"已加载：{case_name} — {len(ballot.voters)}位投票人, {len(ballot.alternatives)}个方案")

            # Show preference table
            st.subheader("偏好矩阵")
            import pandas as pd
            data = []
            for v in ballot.voters:
                row = {"投票人": v.name}
                for rank, alt in enumerate(v.preference):
                    row[f"第{rank+1}偏好"] = alt
                data.append(row)
            st.dataframe(pd.DataFrame(data), use_container_width=True)

    with tab_custom:
        ballot = _build_custom_ballot()
        if ballot and len(ballot.voters) >= 2 and len(ballot.alternatives) >= 2:
            st.session_state["ballot"] = ballot
            st.session_state["scenario_desc"] = "自定义决策场景"
            st.session_state["case_name"] = "custom"

    # Analysis button
    st.divider()
    if st.button("运行分析", type="primary", use_container_width=True):
        if "ballot" in st.session_state:
            st.session_state["run_analysis"] = True
            st.success("分析参数已就绪，请前往「核心分析」页面查看结果。")
        else:
            st.error("请先输入或选择一个决策场景。")
