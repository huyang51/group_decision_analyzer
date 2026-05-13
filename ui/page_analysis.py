"""Page: Core Analysis — Condorcet, Borda/Copeland, Theory checks, Debate."""

from __future__ import annotations

import streamlit as st

from ..core.models import Ballot, AnalysisResult
from ..core.condorcet import compute_full_condorcet, compute_pairwise_matrix
from ..core.borda import compute_borda_scores
from ..core.copeland import compute_copeland_scores
from ..core.arrow_analysis import run_full_arrow_check
from ..core.ahp import run_ahp
from ..core.game_theory import detect_nash_equilibrium
from ..core.behavioral import diagnose_groupthink, classify_risk_domain
from ..core.visualization import (
    plot_pairwise_matrix,
    plot_borda_scores,
    plot_copeland_scores,
    plot_cycle_diagram,
    plot_ahp_hierarchy,
    plot_groupthink_radar,
    plot_risk_profiles,
)


def _run_analysis(ballot: Ballot) -> AnalysisResult:
    """Run all core analyses on a ballot."""
    condorcet = compute_full_condorcet(ballot)
    borda = compute_borda_scores(ballot)
    copeland = compute_copeland_scores(ballot)
    arrow = run_full_arrow_check(ballot)

    return AnalysisResult(
        ballot=ballot,
        pairwise_matrix=condorcet["pairwise_matrix"],
        condorcet_winner=condorcet["condorcet_winner"],
        cycle_detected=condorcet["cycle_detected"],
        cycle_path=condorcet["cycle_path"],
        dominance_edges=condorcet["dominance_edges"],
        borda=borda,
        copeland=copeland,
        arrow=arrow,
    )


def _format_analysis_summary(result: AnalysisResult) -> str:
    """Format analysis result as text summary for agents."""
    lines = []
    lines.append("## 分析结果摘要\n")

    # Condorcet
    lines.append("### 孔多塞分析")
    if result.condorcet_winner:
        lines.append(f"- 孔多塞赢家: **{result.condorcet_winner}**")
    else:
        lines.append("- 无孔多塞赢家")
    if result.cycle_detected:
        lines.append(f"- 循环路径: {' → '.join(result.cycle_path)}")

    # Pairwise matrix
    lines.append("\n### 两两配对比较")
    for (a, b), pr in result.pairwise_matrix.items():
        lines.append(f"- {pr.winner} 击败 {pr.loser}: {pr.winner_votes} vs {pr.loser_votes}")

    # Borda
    lines.append("\n### Borda 排名")
    for i, alt in enumerate(result.borda.ranking, 1):
        lines.append(f"  {i}. {alt}: {result.borda.scores[alt]}分")

    # Copeland
    lines.append("\n### Copeland 排名")
    for i, alt in enumerate(result.copeland.ranking, 1):
        lines.append(f"  {i}. {alt}: 得分{result.copeland.scores[alt]}")

    # Arrow
    lines.append("\n### Arrow 定理检验")
    lines.append(f"- {result.arrow.summary}")

    return "\n".join(lines)


def render_page():
    """Render the analysis page."""
    st.header("步骤2：核心分析")

    ballot = st.session_state.get("ballot")
    if not ballot:
        st.warning("请先在「建模」页面输入或选择一个决策场景。")
        return

    # Run analysis
    result = _run_analysis(ballot)
    st.session_state["analysis_result"] = result

    # Store formatted summary for agents
    summary_text = _format_analysis_summary(result)
    st.session_state["analysis_summary"] = summary_text

    tab1, tab2, tab3, tab4 = st.tabs(["孔多塞分析", "Borda/Copeland", "理论检验", "辩论记录"])

    # Tab 1: Condorcet
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("配对比较矩阵")
            fig_matrix = plot_pairwise_matrix(result.pairwise_matrix, ballot.alternatives)
            st.plotly_chart(fig_matrix, width="stretch")

        with col2:
            st.subheader("结果")
            if result.condorcet_winner:
                st.success(f"孔多塞赢家: **{result.condorcet_winner}**")
            else:
                st.error("无孔多塞赢家")
            if result.cycle_detected:
                st.warning(f"检测到循环: {' → '.join(result.cycle_path)}")

        st.subheader("优势关系图")
        fig_cycle = plot_cycle_diagram(result.dominance_edges, ballot.alternatives, result.cycle_path)
        st.plotly_chart(fig_cycle, width="stretch")

    # Tab 2: Borda/Copeland
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Borda 计票")
            fig_borda = plot_borda_scores(result.borda)
            st.plotly_chart(fig_borda, width="stretch")

        with col2:
            st.subheader("Copeland 计分")
            fig_copeland = plot_copeland_scores(result.copeland)
            st.plotly_chart(fig_copeland, width="stretch")

        # Ranking table
        st.subheader("综合排名")
        import pandas as pd
        ranking_data = []
        for i, alt in enumerate(ballot.alternatives):
            ranking_data.append({
                "方案": alt,
                "Borda分数": result.borda.scores[alt],
                "Borda排名": result.borda.ranking.index(alt) + 1,
                "Copeland得分": result.copeland.scores[alt],
                "Copeland排名": result.copeland.ranking.index(alt) + 1,
            })
        st.dataframe(pd.DataFrame(ranking_data), width="stretch")

    # Tab 3: Theory checks
    with tab3:
        st.subheader("Arrow 不可能定理检验")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**无限制域**: {'满足' if result.arrow.unrestricted_domain else '违反'}")
            st.caption(result.arrow.unrestricted_domain_detail)
            st.markdown(f"**帕累托效率**: {'满足' if result.arrow.pareto_efficient else '违反'}")
            st.caption(result.arrow.pareto_detail)
        with col2:
            st.markdown(f"**IIA (无关方案独立性)**: {'满足' if result.arrow.iia_satisfied else '违反'}")
            st.caption(result.arrow.iia_detail)
            st.markdown(f"**非独裁性**: {'满足' if result.arrow.non_dictator else '违反'}")
            st.caption(result.arrow.non_dictator_detail)
        st.info(result.arrow.summary)

        st.divider()

        # Groupthink diagnosis (template mode)
        st.subheader("群体思维诊断")
        diagnosis = diagnose_groupthink(st.session_state.get("scenario_desc", ""))
        fig_radar = plot_groupthink_radar(diagnosis)
        st.plotly_chart(fig_radar, width="stretch")
        st.caption("注：此为模板诊断。结合场景描述和辩论结果可获得更精确的评估。")

        # Risk profiles
        st.subheader("前景理论风险域分析")
        profiles = []
        for v in ballot.voters:
            # Simple heuristic: first preference = gain, last preference = loss
            gains = [v.preference[0]]
            losses = [v.preference[-1]]
            profile = classify_risk_domain(v.name, gains, losses)
            profiles.append(profile)

        for p in profiles:
            with st.expander(f"{p.stakeholder} — {p.domain} ({p.risk_preference})"):
                st.write(p.description)

        fig_risk = plot_risk_profiles(profiles)
        st.plotly_chart(fig_risk, width="stretch")

    # Tab 4: Debate
    with tab4:
        st.subheader("多智能体辩论")

        import os
        if not os.getenv("DEEPSEEK_API_KEY", ""):
            st.warning("辩论功能需要配置 DEEPSEEK_API_KEY。请在 .env 文件中设置。")
            st.code("DEEPSEEK_API_KEY=your_key_here\nDEEPSEEK_BASE_URL=https://api.deepseek.com\nDEEPSEEK_MODEL=deepseek-chat")
        else:
            if st.button("启动辩论", key="start_debate", type="primary"):
                from ..agents.debate import DebateManager
                debate_mgr = DebateManager()

                with st.spinner("辩论进行中..."):
                    case_context = st.session_state.get("scenario_desc", "")
                    debate_record = debate_mgr.run_debate(
                        analysis_result=summary_text,
                        case_context=case_context,
                    )

                st.session_state["debate_record"] = debate_record
                st.success(f"辩论完成，共 {debate_record.total_rounds} 轮发言。")

            # Display debate record
            debate_record = st.session_state.get("debate_record")
            if debate_record:
                for speech in debate_record.speeches:
                    agent_name = speech["agent"]
                    content = speech["content"]
                    round_num = speech["round"]

                    icon_map = {
                        "分析师": "📊",
                        "批判者": "🔍",
                        "叙事者": "📖",
                        "建模者": "🏗️",
                        "协商者": "🤝",
                        "协调者": "🎯",
                    }
                    icon = icon_map.get(agent_name, "💬")

                    with st.expander(f"{icon} 第{round_num}轮 — {agent_name}", expanded=(round_num == 1)):
                        st.markdown(content)

                st.divider()
                st.subheader("协调者最终总结")
                st.markdown(debate_record.summary)
