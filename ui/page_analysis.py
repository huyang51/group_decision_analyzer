"""Page: Core Analysis — Condorcet, Borda/Copeland, GDM methods, Debate."""

from __future__ import annotations

import os

import streamlit as st
import pandas as pd

from ..core.models import Ballot, AnalysisResult, DecisionMatrix
from ..core.condorcet import compute_full_condorcet, compute_pairwise_matrix
from ..core.borda import compute_borda_scores
from ..core.copeland import compute_copeland_scores
from ..core.arrow_analysis import run_full_arrow_check
from ..core.ahp import run_ahp
from ..core.topsis import run_topsis
from ..core.mcdm_sensitivity import run_sensitivity_analysis
from ..core.game_theory import detect_nash_equilibrium
from ..core.behavioral import diagnose_groupthink, classify_risk_domain
from ..core.consensus import (
    compute_consensus_degree, compute_voter_distances, cluster_voters,
    compute_min_cost_consensus, run_opinion_dynamics,
)
from ..core.power_index import compute_power_indices
from ..core.non_cooperative import detect_strategic_voting
from ..core.visualization import (
    plot_pairwise_matrix,
    plot_borda_scores,
    plot_copeland_scores,
    plot_cycle_diagram,
    plot_ahp_hierarchy,
    plot_groupthink_radar,
    plot_risk_profiles,
    plot_consensus_heatmap,
    plot_power_index_bar,
    plot_opinion_dynamics,
    plot_non_cooperative_radar,
    plot_decision_matrix_heatmap,
    plot_criteria_weights_radar,
    plot_mcdm_ranking_comparison,
    plot_sensitivity_tornado,
)


def _run_analysis(ballot: Ballot) -> AnalysisResult:
    """Run all core analyses on a ballot."""
    condorcet = compute_full_condorcet(ballot)
    borda = compute_borda_scores(ballot)
    copeland = compute_copeland_scores(ballot)
    arrow = run_full_arrow_check(ballot)

    result = AnalysisResult(
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

    # Run MCDM if decision matrix is available
    dm = st.session_state.get("decision_matrix")
    if dm:
        result.decision_matrix = dm

        # AHP per voter (using each voter's own score matrix)
        ahp_results = {}
        for v in ballot.voters:
            if v.criteria_weights:
                voter_score_list = dm.get_voter_matrix_as_list(v.name)
                ranked = run_ahp(
                    dm.criteria,
                    [[1.0] * len(dm.criteria)] * len(dm.criteria),  # placeholder
                    dm.alternatives,
                    voter_score_list,
                )
                # Override weights with voter's criteria weights
                from ..core.ahp import rank_alternatives
                ranking_pairs = rank_alternatives(v.criteria_weights, voter_score_list, dm.alternatives)
                ranked.ranking = [name for name, _ in ranking_pairs]
                ranked.weights = v.criteria_weights
                ahp_results[v.name] = ranked
        result.ahp_results = ahp_results

        # TOPSIS per voter (using each voter's own score matrix)
        topsis_results = {}
        for v in ballot.voters:
            if v.criteria_weights:
                voter_score_list = dm.get_voter_matrix_as_list(v.name)
                topsis_res = run_topsis(dm.criteria, voter_score_list, dm.alternatives, v.criteria_weights)
                topsis_results[v.name] = topsis_res
        result.topsis_results = topsis_results

        # Sensitivity analysis (using average score matrix and average weights)
        avg_weights = {}
        voters_with_weights = [v for v in ballot.voters if v.criteria_weights]
        if voters_with_weights:
            for crit in dm.criteria:
                avg_weights[crit] = sum(v.criteria_weights.get(crit, 0) for v in voters_with_weights) / len(voters_with_weights)
            result.sensitivity_results = run_sensitivity_analysis(dm, avg_weights)

    return result


def _format_analysis_summary(result: AnalysisResult, ballot: Ballot) -> str:
    """Format analysis result as text summary for agents (includes GDM metrics)."""
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

    # ─── GDM Metrics ───

    # Consensus
    lines.append("\n### 共识度分析 (GDM)")
    consensus_result = compute_min_cost_consensus(ballot)
    lines.append(f"- 共识度: {consensus_result.consensus_degree:.2%}")
    lines.append(f"- 投票人分组数: {len(consensus_result.clusters)}")
    for i, cluster in enumerate(consensus_result.clusters):
        lines.append(f"  - 组{i+1}: {', '.join(cluster)}")
    dist_items = sorted(consensus_result.voter_distances.items(), key=lambda x: -x[1])
    lines.append("- 偏离度排名: " + ", ".join(f"{n}({d:.3f})" for n, d in dist_items))

    # Power index
    if ballot.num_voters <= 10:
        lines.append("\n### 投票权力指数 (GDM)")
        voter_weights = [v.voting_weight for v in ballot.voters]
        power_result = compute_power_indices(ballot, weights=voter_weights)
        for name in power_result.shapley_shubik:
            lines.append(
                f"- {name}: SS={power_result.shapley_shubik[name]:.4f}, "
                f"Banzhaf={power_result.banzhaf[name]:.4f}, "
                f"关键票={power_result.pivotal_count[name]}次"
            )

    # Non-cooperative behavior
    if ballot.num_voters >= 3:
        lines.append("\n### 非合作行为检测 (GDM)")
        nc_result = detect_strategic_voting(ballot)
        if nc_result.suspicious_voters:
            lines.append(f"- 疑似非合作投票人: {', '.join(nc_result.suspicious_voters)}")
            for name in nc_result.suspicious_voters:
                m_type = nc_result.manipulation_type.get(name, "未知")
                score = nc_result.deviation_scores.get(name, 0)
                lines.append(f"  - {name}: {m_type} (偏离度: {score:.3f})")
        else:
            lines.append("- 未检测到明显的非合作行为")
        for name, score in nc_result.deviation_scores.items():
            lines.append(f"- {name} 偏离度: {score:.3f}")

    # MCDM results
    if result.topsis_results:
        lines.append("\n### 多属性决策分析 (MCDM)")
        lines.append("#### TOPSIS各视角排名")
        for voter_name, topsis_res in result.topsis_results.items():
            ranking_str = " > ".join(topsis_res.ranking)
            lines.append(f"- {voter_name}: {ranking_str}")

    if result.sensitivity_results:
        sens = result.sensitivity_results
        lines.append(f"\n#### 灵敏度分析")
        lines.append(f"- 基准排名: {' > '.join(sens.base_ranking)}")
        lines.append(f"- 关键准则: {sens.critical_criterion}")
        lines.append(f"- 排名稳定性: {sens.stability_score:.2%}")

    return "\n".join(lines)


def _opinion_variance(opinions: dict) -> float:
    """Compute variance of opinion values."""
    values = list(opinions.values())
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


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

    # Store formatted summary for agents (includes GDM metrics)
    summary_text = _format_analysis_summary(result, ballot)
    st.session_state["analysis_summary"] = summary_text

    tab1, tab2, tab3, tab4 = st.tabs(["孔多塞分析", "Borda/Copeland", "GDM分析与理论检验", "辩论记录"])

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

    # Tab 3: Theory checks + GDM methods
    with tab3:
        # Sub-tabs for theory checks and GDM methods
        gdm_tab1, gdm_tab2, gdm_tab3, gdm_tab4, gdm_tab5, gdm_tab6 = st.tabs([
            "Arrow定理", "共识分析", "权力分析", "非合作行为检测", "意见动力学", "多属性决策(MCDM)",
        ])

        # Sub-tab 1: Arrow theorem
        with gdm_tab1:
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
                gains = [v.preference[0]]
                losses = [v.preference[-1]]
                profile = classify_risk_domain(v.name, gains, losses)
                profiles.append(profile)

            for p in profiles:
                with st.expander(f"{p.stakeholder} — {p.domain} ({p.risk_preference})"):
                    st.write(p.description)

            fig_risk = plot_risk_profiles(profiles)
            st.plotly_chart(fig_risk, width="stretch")

        # Sub-tab 2: Consensus analysis
        with gdm_tab2:
            st.subheader("共识度度量 (Herrera-Viedma et al., 2020)")
            consensus_result = compute_min_cost_consensus(ballot)

            # Consensus degree with progress bar and interpretation
            cd = consensus_result.consensus_degree
            st.progress(cd)
            if cd >= 0.9:
                st.success(f"共识度: **{cd:.2%}** — 高度共识，投票人偏好高度一致。")
            elif cd >= 0.7:
                st.info(f"共识度: **{cd:.2%}** — 中等共识，存在一定分歧但整体可接受。")
            elif cd >= 0.5:
                st.warning(f"共识度: **{cd:.2%}** — 低共识，投票人间存在明显分歧。")
            else:
                st.error(f"共识度: **{cd:.2%}** — 极低共识，投票人偏好严重对立。")

            col1, col2, col3 = st.columns(3)
            with col1:
                avg_dist = sum(consensus_result.voter_distances.values()) / len(consensus_result.voter_distances) if consensus_result.voter_distances else 0
                st.metric("平均偏离度", f"{avg_dist:.3f}")
            with col2:
                st.metric("投票人分组数", str(len(consensus_result.clusters)))
            with col3:
                most_distant = max(consensus_result.voter_distances.items(), key=lambda x: x[1]) if consensus_result.voter_distances else ("—", 0)
                st.metric("最偏离投票人", most_distant[0])

            # Similarity heatmap
            fig_heatmap = plot_consensus_heatmap(consensus_result)
            st.plotly_chart(fig_heatmap, width="stretch")

            # Voter distances
            st.subheader("投票人偏离度")
            for name, dist in sorted(consensus_result.voter_distances.items(), key=lambda x: -x[1]):
                bar_color = "red" if dist > 0.5 else "orange" if dist > 0.3 else "green"
                st.markdown(f"**{name}**: {dist:.3f} " + f":{bar_color}[{'█' * int(dist * 20)}{'░' * (20 - int(dist * 20))}]")

            # Clusters
            st.subheader("投票人聚类")
            for i, cluster in enumerate(consensus_result.clusters):
                st.write(f"**组 {i+1}**: {', '.join(cluster)}")

            # Min-cost consensus adjustments
            if consensus_result.min_cost_adjustments:
                st.subheader("最小成本共识建议")
                st.write(f"当前共识度 {consensus_result.consensus_degree:.2%} 未达理想水平，以下投票人需调整偏好：")
                for voter_name, adj in consensus_result.min_cost_adjustments.items():
                    with st.expander(f"{voter_name} 的调整建议"):
                        for pair, cost in adj.items():
                            st.write(f"- {pair}: 调整成本 {cost:.3f}")

        # Sub-tab 3: Power analysis
        with gdm_tab3:
            st.subheader("投票权力指数 (Shapley-Shubik & Banzhaf)")

            if ballot.num_voters > 10:
                st.warning("投票人超过10人时，Shapley-Shubik指数计算量过大（n!），已跳过。")
                st.info("Banzhaf指数仍可计算，但建议使用加权投票模型。")
            else:
                with st.spinner("计算投票权力指数中（Shapley-Shubik排列枚举）..."):
                    voter_weights = [v.voting_weight for v in ballot.voters]
                    power_result = compute_power_indices(ballot, weights=voter_weights)

                # Display indices table
                power_data = []
                for name in power_result.shapley_shubik:
                    power_data.append({
                        "投票人": name,
                        "Shapley-Shubik": f"{power_result.shapley_shubik[name]:.4f}",
                        "Banzhaf": f"{power_result.banzhaf[name]:.4f}",
                        "关键票次数": power_result.pivotal_count[name],
                    })
                st.dataframe(pd.DataFrame(power_data), width="stretch")

                # Bar chart
                fig_power = plot_power_index_bar(power_result)
                st.plotly_chart(fig_power, width="stretch")

                # Interpretation
                ss_vals = list(power_result.shapley_shubik.values())
                ss_max_name = max(power_result.shapley_shubik, key=power_result.shapley_shubik.get)
                ss_min_name = min(power_result.shapley_shubik, key=power_result.shapley_shubik.get)
                ss_max = power_result.shapley_shubik[ss_max_name]
                ss_min = power_result.shapley_shubik[ss_min_name]

                n = ballot.num_voters
                equal_share = 1.0 / n

                if abs(ss_max - ss_min) < 0.01:
                    st.success(f"权力分布均衡：所有投票人权力指数接近 {equal_share:.3f}（1/{n}）。")
                elif ss_max > equal_share * 1.5:
                    st.warning(
                        f"权力集中：**{ss_max_name}** 的SS指数为 {ss_max:.4f}，"
                        f"是 **{ss_min_name}**（{ss_min:.4f}）的 {ss_max/ss_min:.1f} 倍。"
                    )
                else:
                    st.info(f"权力分布较均匀，最强投票人 **{ss_max_name}**（{ss_max:.4f}）与最弱 **{ss_min_name}**（{ss_min:.4f}）差距不大。")

                st.caption(
                    f"总联盟数: {power_result.total_coalitions} | "
                    f"Shapley-Shubik指数衡量投票人在所有排列中成为关键票的概率；"
                    f"Banzhaf指数衡量投票人在所有联盟中改变结果的能力。"
                )

        # Sub-tab 4: Non-cooperative behavior detection
        with gdm_tab4:
            st.subheader("非合作行为检测 (Dong et al., EJOR 2020)")

            st.caption(
                "检测策略性投票行为，包括子弹投票（只投第一偏好）、"
                "偏好偏离（与群体共识严重不一致）等非合作行为。"
            )

            if ballot.num_voters < 3:
                st.info("投票人少于3人时，非合作行为检测意义有限。")
            else:
                nc_result = detect_strategic_voting(ballot)

                # Summary banner
                if nc_result.suspicious_voters:
                    st.warning(f"检测到 **{len(nc_result.suspicious_voters)}** 位投票人存在疑似非合作行为")
                    for name in nc_result.suspicious_voters:
                        m_type = nc_result.manipulation_type.get(name, "未知")
                        score = nc_result.deviation_scores.get(name, 0)
                        st.markdown(f"- **{name}**: {m_type}（偏离度: {score:.3f}）")
                else:
                    st.success("未检测到明显的非合作行为。所有投票人偏好与群体共识较为一致。")

                # Radar chart
                fig_nc = plot_non_cooperative_radar(nc_result)
                st.plotly_chart(fig_nc, width="stretch")

                # Detailed explanation
                with st.expander("检测方法说明", expanded=False):
                    st.markdown(nc_result.explanation)

                # Deviation scores table
                st.subheader("投票人偏离度排名")
                dev_data = []
                for name, score in sorted(nc_result.deviation_scores.items(), key=lambda x: -x[1]):
                    suspicious = "⚠️ 疑似" if name in nc_result.suspicious_voters else "✓ 正常"
                    m_type = nc_result.manipulation_type.get(name, "—")
                    dev_data.append({
                        "状态": suspicious,
                        "投票人": name,
                        "偏离度": f"{score:.3f}",
                        "操纵类型": m_type,
                    })
                st.dataframe(pd.DataFrame(dev_data), width="stretch")

                # Interpretation
                avg_dev = sum(nc_result.deviation_scores.values()) / len(nc_result.deviation_scores) if nc_result.deviation_scores else 0
                if avg_dev < 0.2:
                    st.info(f"群体平均偏离度为 {avg_dev:.3f}，整体投票行为诚实。")
                elif avg_dev < 0.4:
                    st.warning(f"群体平均偏离度为 {avg_dev:.3f}，存在一定程度的偏好差异。")
                else:
                    st.error(f"群体平均偏离度为 {avg_dev:.3f}，群体内部分歧较大。")

        # Sub-tab 5: Opinion dynamics
        with gdm_tab5:
            st.subheader("有界信任意见动力学 (Hegselmann-Krause模型)")

            st.caption(
                "模拟投票人之间的意见交互过程。每轮中，每个投票人的意见向"
                "信任阈值ε范围内的其他投票人的平均意见靠拢。"
                "ε越大表示越开放，越容易达成共识；ε越小表示越固执，越可能形成多个意见簇。"
            )

            col1, col2 = st.columns(2)
            with col1:
                epsilon = st.slider("信任阈值 (ε)", 0.0, 1.0, 0.2, 0.05,
                                    help="投票人只听取意见距离在ε以内的其他人的意见")
            with col2:
                max_rounds = st.number_input("最大轮次", 5, 50, 20)

            # Auto-run with default parameters, and re-run on parameter change
            history = run_opinion_dynamics(ballot, epsilon=epsilon, max_rounds=max_rounds)

            if history:
                # Convergence info
                final = history[-1]
                n_rounds = len(history) - 1
                if n_rounds > 0:
                    prev = history[-2]
                    max_change = max(abs(final[n] - prev[n]) for n in final)
                    converged = max_change < 1e-5
                else:
                    converged = True
                    max_change = 0

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("收敛轮次", str(n_rounds) if converged else f"{n_rounds}+")
                with col2:
                    st.metric("最终意见方差", f"{_opinion_variance(final):.4f}")
                with col3:
                    st.metric("状态", "已收敛" if converged else "未收敛")

                # Dynamics chart
                fig_dynamics = plot_opinion_dynamics(history)
                st.plotly_chart(fig_dynamics, width="stretch")

                # Final state
                st.subheader("最终意见分布")
                for name, val in sorted(final.items(), key=lambda x: -x[1]):
                    bar_width = int(val * 20)
                    st.markdown(f"**{name}**: {val:.4f} " + f"{'█' * bar_width}{'░' * (20 - bar_width)}")

                # Interpretation
                if converged:
                    final_vals = list(final.values())
                    if max(final_vals) - min(final_vals) < 1e-5:
                        st.success(f"意见已在第 {n_rounds} 轮达成完全共识（值: {final_vals[0]:.4f}）。")
                    else:
                        st.info(f"意见动力学已在第 {n_rounds} 轮稳定，但未达成完全共识。最终意见分布在 [{min(final_vals):.4f}, {max(final_vals):.4f}]。")
                else:
                    # Check for clusters
                    sorted_opinions = sorted(final.values())
                    clusters_found = 1
                    for i in range(1, len(sorted_opinions)):
                        if sorted_opinions[i] - sorted_opinions[i-1] > epsilon * 0.5:
                            clusters_found += 1
                    if clusters_found > 1:
                        st.warning(f"意见未收敛，形成 **{clusters_found}** 个意见簇。ε={epsilon} 时投票人无法跨越意见鸿沟。尝试增大ε值。")
                    else:
                        st.info(f"意见尚未完全收敛（最大变化: {max_change:.6f}），继续运行可能收敛。")
            else:
                st.info("投票人或方案数量不足，无法运行动力学模拟。")

        # Sub-tab 6: MCDM analysis
        with gdm_tab6:
            st.subheader("多属性决策分析 (MCDM)")

            dm = st.session_state.get("decision_matrix")
            if not dm:
                st.info("当前案例未包含多属性决策数据（准则、评分矩阵）。请加载包含 criteria 和 score_matrix 的案例（如OpenAI政变事件扩展场景）。")
            else:
                # Decision matrix heatmap (per-voter selector)
                st.subheader("决策矩阵热力图")
                voter_names = list(dm.voter_score_matrices.keys())
                heatmap_mode = st.radio(
                    "热力图模式", ["平均矩阵", "按投票人"],
                    horizontal=True, key="heatmap_mode",
                )
                if heatmap_mode == "按投票人":
                    selected_voter = st.selectbox("选择投票人", voter_names, key="heatmap_voter_select")
                    fig_dm = plot_decision_matrix_heatmap(dm, voter_name=selected_voter)
                else:
                    fig_dm = plot_decision_matrix_heatmap(dm)
                st.plotly_chart(fig_dm, width="stretch")

                # Criteria weights radar
                voters_with_weights = [v for v in ballot.voters if v.criteria_weights]
                if voters_with_weights:
                    st.subheader("投票人准则权重对比")
                    voter_weights = {v.name: v.criteria_weights for v in voters_with_weights}
                    fig_radar = plot_criteria_weights_radar(dm.criteria, voter_weights)
                    st.plotly_chart(fig_radar, width="stretch")

                # AHP results per voter
                if result.ahp_results:
                    st.subheader("AHP 各视角排名")
                    ahp_ranking_data = []
                    for voter_name, ahp_res in result.ahp_results.items():
                        row = {"投票人": voter_name}
                        for rank, alt in enumerate(ahp_res.ranking, 1):
                            row[f"第{rank}名"] = alt
                        row["CR"] = f"{ahp_res.CR:.4f}"
                        row["一致性"] = "✓" if ahp_res.is_consistent else "✗"
                        ahp_ranking_data.append(row)
                    st.dataframe(pd.DataFrame(ahp_ranking_data), width="stretch")

                    # Show weights used
                    with st.expander("各投票人准则权重详情"):
                        for voter_name, ahp_res in result.ahp_results.items():
                            st.write(f"**{voter_name}**: {ahp_res.weights}")

                # TOPSIS results per voter
                if result.topsis_results:
                    st.subheader("TOPSIS 各视角排名")
                    topsis_ranking_data = []
                    for voter_name, topsis_res in result.topsis_results.items():
                        row = {"投票人": voter_name}
                        for rank, alt in enumerate(topsis_res.ranking, 1):
                            row[f"第{rank}名"] = alt
                        # Show closeness scores
                        for alt in dm.alternatives:
                            row[f"{alt}(接近度)"] = f"{topsis_res.scores[alt]:.4f}"
                        topsis_ranking_data.append(row)
                    st.dataframe(pd.DataFrame(topsis_ranking_data), width="stretch")

                # MCDM vs Social Choice comparison
                if result.topsis_results:
                    st.subheader("MCDM排名 vs 社会选择排名")
                    social_rankings = {
                        "Borda": result.borda.ranking,
                        "Copeland": result.copeland.ranking,
                    }
                    mcdm_rankings = {}
                    for voter_name, topsis_res in result.topsis_results.items():
                        mcdm_rankings[f"TOPSIS({voter_name})"] = topsis_res.ranking
                    if result.ahp_results:
                        for voter_name, ahp_res in result.ahp_results.items():
                            mcdm_rankings[f"AHP({voter_name})"] = ahp_res.ranking

                    fig_compare = plot_mcdm_ranking_comparison(mcdm_rankings, social_rankings)
                    st.plotly_chart(fig_compare, width="stretch")

                # Sensitivity analysis
                if result.sensitivity_results:
                    st.subheader("灵敏度分析")
                    sens = result.sensitivity_results

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("基准排名", " > ".join(sens.base_ranking))
                    with col2:
                        st.metric("关键准则", sens.critical_criterion)
                    with col3:
                        st.metric("排名稳定性", f"{sens.stability_score:.2%}")

                    # Tornado diagram for selected alternative
                    selected_alt = st.selectbox(
                        "选择方案查看灵敏度龙卷风图",
                        dm.alternatives,
                        key="sensitivity_alt_select",
                    )
                    fig_tornado = plot_sensitivity_tornado(sens, selected_alt)
                    st.plotly_chart(fig_tornado, width="stretch")

                    # Rank changes table
                    st.subheader("各方案排名变化明细")
                    rank_data = []
                    for alt in dm.alternatives:
                        ranks = sens.rank_changes[alt]
                        rank_data.append({
                            "方案": alt,
                            "最高排名": min(ranks),
                            "最低排名": max(ranks),
                            "平均排名": f"{sum(ranks)/len(ranks):.1f}",
                            "排名波动": f"{max(ranks) - min(ranks)}",
                        })
                    st.dataframe(pd.DataFrame(rank_data), width="stretch")

    # Tab 4: Debate
    with tab4:
        st.subheader("多智能体辩论")

        if not os.getenv("DEEPSEEK_API_KEY", ""):
            st.warning("辩论功能需要配置 DEEPSEEK_API_KEY。请在 .env 文件中设置。")
            st.code("DEEPSEEK_API_KEY=your_key_here\nDEEPSEEK_BASE_URL=https://api.deepseek.com\nDEEPSEEK_MODEL=deepseek-chat")
        else:
            if st.button("启动辩论", key="start_debate", type="primary"):
                from ..agents.debate import DebateManager
                debate_mgr = DebateManager()

                try:
                    with st.spinner("辩论进行中..."):
                        case_context = st.session_state.get("scenario_desc", "")
                        debate_record = debate_mgr.run_debate(
                            analysis_result=summary_text,
                            case_context=case_context,
                        )

                    st.session_state["debate_record"] = debate_record
                    st.success(f"辩论完成，共 {debate_record.total_rounds} 轮发言。")
                except Exception as e:
                    st.error(f"辩论过程中发生错误: {e}")
                    st.info("请检查网络连接和API配置后重试。")

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
