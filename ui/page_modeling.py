"""Page: Modeling — scenario input and preference configuration."""

from __future__ import annotations

import json
import streamlit as st

from ..cases.case_library import CaseLibrary
from ..core.models import Voter, Ballot, DecisionMatrix


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
        if len(set(prefs)) == num_alts:
            voters.append(Voter(name=name, preference=prefs))
        else:
            st.warning(f"{name} 的偏好包含重复方案，请检查。")

    return Ballot(voters=voters, alternatives=alternatives)


def _on_preset_change():
    st.session_state["modeling_source"] = "preset"


def _load_preset_case(case_lib: CaseLibrary) -> tuple:
    """Load a preset case from the case library."""
    case_names = case_lib.get_case_names()
    if not case_names:
        st.warning("案例库为空，请确保 cases/data/ 目录下有JSON文件。")
        return None, None, None

    selected_case = st.selectbox("选择案例", case_names, key="preset_case", on_change=_on_preset_change)
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
    selected_scenario_name = st.selectbox("选择场景", scenario_names, key="preset_scenario", on_change=_on_preset_change)
    scenario_idx = scenario_names.index(selected_scenario_name)
    scenario = scenarios[scenario_idx]

    # Build ballot
    alternatives = scenario.get("alternatives", [])
    voters = []
    for v in scenario.get("voters", []):
        cw = v.get("criteria_weights")
        vw = v.get("voting_weight", 1.0)
        voters.append(Voter(name=v["name"], preference=v["preference"],
                            criteria_weights=cw, voting_weight=vw))
    ballot = Ballot(voters=voters, alternatives=alternatives)

    # Build decision matrix if criteria and per-voter score_matrix are present
    criteria = scenario.get("criteria", [])
    voter_score_matrices = {}
    for v in scenario.get("voters", []):
        sm = v.get("score_matrix")
        if sm:
            voter_score_matrices[v["name"]] = sm
    if criteria and voter_score_matrices:
        dm = DecisionMatrix(criteria=criteria, voter_score_matrices=voter_score_matrices, alternatives=alternatives)
        st.session_state["decision_matrix"] = dm
    else:
        st.session_state.pop("decision_matrix", None)

    return ballot, case.get("description", ""), selected_case


def render_page():
    """Render the modeling page."""
    st.header("步骤1：建模 — 输入决策场景")

    tab_preset, tab_custom = st.tabs(["预设案例", "自定义输入"])

    with tab_preset:
        case_lib = CaseLibrary()
        ballot, description, case_name = _load_preset_case(case_lib)
        if ballot:
            if st.session_state.get("modeling_source", "preset") == "preset":
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
            st.dataframe(pd.DataFrame(data), width="stretch")

            # Show decision matrix if available (per-voter)
            dm = st.session_state.get("decision_matrix")
            if dm:
                st.subheader("决策矩阵（方案×准则评分）")
                voter_names = list(dm.voter_score_matrices.keys())
                selected_voter = st.selectbox("查看投票人评分矩阵", voter_names + ["平均矩阵"], key="modeling_voter_matrix")
                if selected_voter == "平均矩阵":
                    display_matrix = dm.get_average_matrix()
                else:
                    display_matrix = dm.get_voter_matrix(selected_voter)
                dm_data = []
                for alt in dm.alternatives:
                    row = {"方案": alt}
                    for i, crit in enumerate(dm.criteria):
                        row[crit] = display_matrix[alt][i]
                    dm_data.append(row)
                st.dataframe(pd.DataFrame(dm_data), width="stretch")

                # Show voter criteria weights
                voters_with_weights = [v for v in ballot.voters if v.criteria_weights]
                if voters_with_weights:
                    st.subheader("投票人准则权重")
                    import plotly.graph_objects as go
                    fig = go.Figure()
                    for v in voters_with_weights:
                        fig.add_trace(go.Scatterpolar(
                            r=[v.criteria_weights.get(c, 0) for c in dm.criteria] + [v.criteria_weights.get(dm.criteria[0], 0)],
                            theta=dm.criteria + [dm.criteria[0]],
                            fill="toself",
                            name=v.name,
                            opacity=0.6,
                        ))
                    fig.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                        height=500,
                        template="plotly_white",
                        title="各投票人准则权重雷达图",
                    )
                    st.plotly_chart(fig, width="stretch")

    with tab_custom:
        with st.form("custom_form"):
            ballot = _build_custom_ballot()
            submitted = st.form_submit_button("应用自定义场景")
            if submitted and ballot and len(ballot.voters) >= 2 and len(ballot.alternatives) >= 2:
                st.session_state["ballot"] = ballot
                st.session_state["scenario_desc"] = "自定义决策场景"
                st.session_state["case_name"] = "custom"
                st.session_state["modeling_source"] = "custom"

    # Analysis button
    st.divider()
    if st.button("运行分析", type="primary", width="stretch"):
        if "ballot" in st.session_state:
            st.session_state["run_analysis"] = True
            st.success("分析参数已就绪，请前往「核心分析」页面查看结果。")
        else:
            st.error("请先输入或选择一个决策场景。")
