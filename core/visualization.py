"""Plotly chart generation for group decision analysis."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import math

import plotly.graph_objects as go
import networkx as nx

# Built-in color palette (replaces plotly.express dependency)
_COLORS = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
]

from .models import (
    PairwiseResult, BordaScores, SimulationStep, CopelandScores,
    ConsensusResult, PowerIndexResult, NonCooperativeResult,
    DecisionMatrix,
)


def plot_pairwise_matrix(
    matrix: Dict[Tuple[str, str], PairwiseResult],
    alternatives: List[str],
) -> go.Figure:
    """Create a heatmap of pairwise comparison margins."""
    n = len(alternatives)
    # Build a matrix: cell[i][j] = margin of i over j (positive = i wins)
    z = [[0.0] * n for _ in range(n)]
    text = [[""] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i == j:
                z[i][j] = 0
                text[i][j] = "-"
            else:
                a, b = alternatives[i], alternatives[j]
                key = (a, b) if a < b else (b, a)
                result = matrix[key]
                if result.winner == a:
                    z[i][j] = result.margin
                    text[i][j] = f"{result.winner_votes}:{result.loser_votes}"
                else:
                    z[i][j] = -result.margin
                    text[i][j] = f"{result.loser_votes}:{result.winner_votes}"

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=alternatives,
        y=alternatives,
        text=text,
        texttemplate="%{text}",
        colorscale="RdYlGn",
        zmid=0,
        colorbar=dict(title="胜出票数差"),
    ))
    fig.update_layout(
        title="两两配对比较矩阵",
        xaxis_title="被比较方案",
        yaxis_title="比较方案",
        height=400,
        template="plotly_white",
    )
    return fig


def plot_borda_scores(borda: BordaScores) -> go.Figure:
    """Create a bar chart of Borda scores."""
    alts = borda.ranking
    scores = [borda.scores[a] for a in alts]

    colors = _COLORS[:len(alts)]

    fig = go.Figure(data=go.Bar(
        x=alts,
        y=scores,
        marker_color=colors,
        text=scores,
        textposition="outside",
    ))
    fig.update_layout(
        title="Borda 计票分数",
        xaxis_title="方案",
        yaxis_title="Borda 分数",
        height=400,
        template="plotly_white",
    )
    return fig


def plot_copeland_scores(copeland: CopelandScores) -> go.Figure:
    """Create a bar chart of Copeland scores."""
    alts = copeland.ranking
    scores = [copeland.scores[a] for a in alts]
    wins = [copeland.wins[a] for a in alts]
    losses = [copeland.losses[a] for a in alts]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="胜场", x=alts, y=wins, marker_color="#2ecc71"))
    fig.add_trace(go.Bar(name="负场", x=alts, y=[-l for l in losses], marker_color="#e74c3c"))
    fig.add_trace(go.Bar(
        name="净得分", x=alts, y=scores,
        marker_color="#3498db", text=scores, textposition="outside",
    ))
    fig.update_layout(
        title="Copeland 计分",
        xaxis_title="方案",
        yaxis_title="得分",
        barmode="group",
        height=400,
        template="plotly_white",
    )
    return fig


def plot_cycle_diagram(
    dominance_edges: List[Tuple[str, str]],
    alternatives: List[str],
    cycle_path: Optional[List[str]] = None,
) -> go.Figure:
    """Create a directed graph showing dominance relationships."""
    G = nx.DiGraph()
    G.add_nodes_from(alternatives)
    G.add_edges_from(dominance_edges)

    pos = nx.spring_layout(G, seed=42, k=2)

    # Edge traces
    edge_x, edge_y = [], []
    cycle_edges = set()
    if cycle_path and len(cycle_path) > 1:
        for k in range(len(cycle_path) - 1):
            cycle_edges.add((cycle_path[k], cycle_path[k + 1]))

    normal_x, normal_y = [], []
    cycle_x, cycle_y = [], []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        if edge in cycle_edges:
            cycle_x.extend([x0, x1, None])
            cycle_y.extend([y0, y1, None])
        else:
            normal_x.extend([x0, x1, None])
            normal_y.extend([y0, y1, None])

    fig = go.Figure()

    # Normal edges
    fig.add_trace(go.Scatter(
        x=normal_x, y=normal_y, mode="lines",
        line=dict(width=1, color="#95a5a6"),
        hoverinfo="none", showlegend=False,
    ))

    # Cycle edges
    if cycle_x:
        fig.add_trace(go.Scatter(
            x=cycle_x, y=cycle_y, mode="lines",
            line=dict(width=3, color="#e74c3c"),
            hoverinfo="none", name="循环路径",
        ))

    # Add arrows
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        # Shorten line to not overlap with node
        dx, dy = x1 - x0, y1 - y0
        length = math.sqrt(dx**2 + dy**2) or 1
        shrink = 0.08
        ax = x1 - dx / length * shrink
        ay = y1 - dy / length * shrink
        fig.add_annotation(
            x=ax, y=ay, ax=x0, ay=y0,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1.5,
            arrowwidth=2,
            arrowcolor="#e74c3c" if edge in cycle_edges else "#95a5a6",
        )

    # Node traces
    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    node_text = list(G.nodes())

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        marker=dict(size=30, color="#3498db", line=dict(width=2, color="white")),
        text=node_text, textposition="middle center",
        textfont=dict(color="white", size=14),
        hoverinfo="text", showlegend=False,
    ))

    fig.update_layout(
        title="方案优势关系图（箭头指向被击败方案）",
        showlegend=bool(cycle_path),
        height=500,
        template="plotly_white",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    return fig


def plot_simulation_results(steps: List[SimulationStep]) -> go.Figure:
    """Create a timeline showing how rankings change across simulation steps."""
    from .borda import compute_borda_scores

    if not steps:
        return go.Figure()

    all_alts = set()
    for s in steps:
        all_alts.update(s.borda_ranking)
    alts = sorted(all_alts)

    fig = go.Figure()

    for alt in alts:
        y_values = []
        x_labels = []
        for s in steps:
            # Use Borda score as the metric
            borda = compute_borda_scores(s.ballot)
            y_values.append(borda.scores.get(alt, 0))
            x_labels.append(f"步骤{s.step_num}: {s.description}")

        fig.add_trace(go.Scatter(
            x=list(range(len(steps))),
            y=y_values,
            mode="lines+markers",
            name=alt,
            text=x_labels,
            hovertemplate="%{text}<br>Borda分数: %{y}",
        ))

    # Mark cycle steps
    for s in steps:
        if s.cycle_detected:
            fig.add_vrect(
                x0=s.step_num - 0.3, x1=s.step_num + 0.3,
                fillcolor="red", opacity=0.1,
                layer="below", line_width=0,
            )

    fig.update_layout(
        title="动态推演：Borda 分数变化",
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(len(steps))),
            ticktext=[f"步骤{s.step_num}" for s in steps],
        ),
        xaxis_title="推演步骤",
        yaxis_title="Borda 分数",
        height=450,
        template="plotly_white",
    )
    return fig


def plot_ahp_hierarchy(
    criteria: List[str],
    alternatives: List[str],
    scores: Dict[str, float],
) -> go.Figure:
    """Create a horizontal bar chart showing AHP final scores for alternatives."""
    names = list(scores.keys())
    values = list(scores.values())

    # Sort by score descending
    sorted_pairs = sorted(zip(names, values), key=lambda x: -x[1])
    names, values = zip(*sorted_pairs) if sorted_pairs else ([], [])

    fig = go.Figure(data=go.Bar(
        x=list(values),
        y=list(names),
        orientation="h",
        marker_color=_COLORS[:len(names)],
        text=[f"{v:.3f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title="AHP 方案综合得分",
        xaxis_title="综合得分",
        yaxis_title="方案",
        height=max(300, len(names) * 60),
        template="plotly_white",
    )
    return fig


def plot_groupthink_radar(diagnosis) -> go.Figure:
    """Create a radar chart of the 8 groupthink symptoms."""
    from .behavioral import GROUPTHINK_SYMPTOMS

    labels = [label for _, label, _ in GROUPTHINK_SYMPTOMS]
    values = [1 if diagnosis.symptoms.get(key, False) else 0 for key, _, _ in GROUPTHINK_SYMPTOMS]
    # Close the polygon
    labels_plot = labels + [labels[0]]
    values_plot = values + [values[0]]

    fig = go.Figure(data=go.Scatterpolar(
        r=values_plot,
        theta=labels_plot,
        fill="toself",
        fillcolor="rgba(231, 76, 60, 0.3)",
        line=dict(color="#e74c3c", width=2),
        name="群体思维症状",
    ))
    fig.update_layout(
        title=f"群体思维诊断 — 风险等级: {diagnosis.risk_level} ({diagnosis.score}/{diagnosis.total})",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickvals=[0, 1], ticktext=["否", "是"])),
        height=500,
        template="plotly_white",
    )
    return fig


def plot_risk_profiles(profiles: List) -> go.Figure:
    """Create a scatter plot of risk profiles (losses vs gains) with quadrant labels."""
    fig = go.Figure()

    for p in profiles:
        # Count gains and losses from the profile
        domain_color = "#e74c3c" if "损失" in p.domain else "#2ecc71"
        risk_marker = "triangle-up" if "寻求" in p.risk_preference else "circle"

        fig.add_trace(go.Scatter(
            x=[0], y=[0],  # placeholder; use description-based positioning
            mode="markers+text",
            marker=dict(size=16, color=domain_color, symbol=risk_marker, line=dict(width=1, color="white")),
            text=[p.stakeholder],
            textposition="top center",
            name=p.stakeholder,
            hovertext=p.description,
            hoverinfo="text",
        ))

    # Add quadrant annotations
    fig.add_annotation(x=0.5, y=0.5, text="收益域·风险规避", showarrow=False, font=dict(size=12, color="#27ae60"))
    fig.add_annotation(x=-0.5, y=0.5, text="损失域·风险寻求", showarrow=False, font=dict(size=12, color="#c0392b"))
    fig.add_annotation(x=0.5, y=-0.5, text="收益域·风险寻求", showarrow=False, font=dict(size=12, color="#f39c12"))
    fig.add_annotation(x=-0.5, y=-0.5, text="损失域·风险规避", showarrow=False, font=dict(size=12, color="#8e44ad"))

    fig.update_layout(
        title="前景理论：利益相关者风险域分布",
        xaxis=dict(title="← 损失域    |    收益域 →", range=[-1, 1], zeroline=True, showticklabels=False),
        yaxis=dict(title="← 风险规避    |    风险寻求 →", range=[-1, 1], zeroline=True, showticklabels=False),
        height=450,
        template="plotly_white",
        showlegend=True,
    )
    return fig


# ─── GDM/LSGDM Visualization Functions ───────────────────────────────


def plot_consensus_heatmap(
    consensus_result: ConsensusResult,
) -> go.Figure:
    """Create a heatmap of voter preference similarity matrix."""
    import numpy as np

    voter_names = consensus_result.voter_names
    sim_matrix = consensus_result.preference_matrix

    # Format text with 2 decimal places
    n = len(voter_names)
    text = [[f"{sim_matrix[i][j]:.2f}" for j in range(n)] for i in range(n)]

    fig = go.Figure(data=go.Heatmap(
        z=sim_matrix.tolist(),
        x=voter_names,
        y=voter_names,
        text=text,
        texttemplate="%{text}",
        colorscale="RdYlGn",
        zmin=0,
        zmax=1,
        colorbar=dict(title="相似度"),
    ))
    fig.update_layout(
        title=f"投票人偏好相似度矩阵 (共识度: {consensus_result.consensus_degree:.2%})",
        height=max(450, len(voter_names) * 55),
        xaxis=dict(tickangle=-45, tickfont=dict(size=10)),
        yaxis=dict(tickfont=dict(size=10)),
        template="plotly_white",
    )
    return fig


def plot_power_index_bar(power_result: PowerIndexResult) -> go.Figure:
    """Create a grouped bar chart comparing Shapley-Shubik and Banzhaf indices."""
    names = list(power_result.shapley_shubik.keys())
    ss_values = [power_result.shapley_shubik[n] for n in names]
    banzhaf_values = [power_result.banzhaf[n] for n in names]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Shapley-Shubik",
        x=names,
        y=ss_values,
        marker_color="#3498db",
        text=[f"{v:.3f}" for v in ss_values],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="Banzhaf",
        x=names,
        y=banzhaf_values,
        marker_color="#e67e22",
        text=[f"{v:.3f}" for v in banzhaf_values],
        textposition="outside",
    ))
    fig.update_layout(
        title="投票权力指数对比",
        xaxis_title="投票人",
        yaxis_title="权力指数",
        barmode="group",
        height=max(400, len(names) * 55),
        xaxis=dict(tickangle=-45, tickfont=dict(size=10)),
        template="plotly_white",
    )
    return fig


def plot_opinion_dynamics(dynamics_history: list) -> go.Figure:
    """Create a line chart showing opinion dynamics convergence.

    dynamics_history: list of dicts, each mapping voter name to opinion value.
    """
    if not dynamics_history:
        return go.Figure()

    voter_names = list(dynamics_history[0].keys())
    rounds = list(range(len(dynamics_history)))

    fig = go.Figure()
    for i, name in enumerate(voter_names):
        values = [d[name] for d in dynamics_history]
        fig.add_trace(go.Scatter(
            x=rounds,
            y=values,
            mode="lines+markers",
            name=name,
            line=dict(color=_COLORS[i % len(_COLORS)], width=2),
        ))

    fig.update_layout(
        title="有界信任意见动力学 (Hegselmann-Krause模型)",
        xaxis_title="轮次",
        yaxis_title="意见值 (0-1)",
        height=max(450, len(voter_names) * 50 + 100),
        legend=dict(orientation="v", x=1.05, y=1, font=dict(size=9)),
        template="plotly_white",
    )
    return fig


def plot_non_cooperative_radar(nc_result: NonCooperativeResult) -> go.Figure:
    """Create a radar chart showing deviation scores for each voter."""
    names = list(nc_result.deviation_scores.keys())
    values = [nc_result.deviation_scores[n] for n in names]

    # Close the polygon
    names_plot = names + [names[0]]
    values_plot = values + [values[0]]

    # Color suspicious voters differently
    colors = []
    for name in names:
        if name in nc_result.suspicious_voters:
            colors.append("#e74c3c")
        else:
            colors.append("#2ecc71")

    fig = go.Figure(data=go.Scatterpolar(
        r=values_plot,
        theta=names_plot,
        fill="toself",
        fillcolor="rgba(231, 76, 60, 0.2)",
        line=dict(color="#e74c3c", width=2),
        name="偏离度",
    ))

    suspicious_count = len(nc_result.suspicious_voters)
    title_suffix = f" (⚠ {suspicious_count}人疑似)" if suspicious_count > 0 else " (正常)"

    fig.update_layout(
        title=f"非合作行为检测 — 偏离度雷达图{title_suffix}",
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1]),
            angularaxis=dict(tickfont=dict(size=9)),
        ),
        height=550,
        template="plotly_white",
    )
    return fig


# ─── MCDM Visualization Functions ────────────────────────────────────


def plot_decision_matrix_heatmap(
    decision_matrix: DecisionMatrix,
    voter_name: Optional[str] = None,
) -> go.Figure:
    """Create a heatmap of the decision matrix (alternatives x criteria scores).

    Args:
        decision_matrix: the DecisionMatrix object.
        voter_name: if specified, show that voter's score matrix;
                    if None, show the average matrix across all voters.
    """
    alts = decision_matrix.alternatives
    criteria = decision_matrix.criteria
    n_alts = len(alts)

    if voter_name:
        score_dict = decision_matrix.get_voter_matrix(voter_name)
        title = f"决策矩阵 — {voter_name}（方案×准则评分）"
    else:
        score_dict = decision_matrix.get_average_matrix()
        title = "决策矩阵 — 平均（方案×准则评分）"

    z = []
    text = []
    for alt in alts:
        scores = score_dict[alt]
        z.append(scores)
        text.append([str(s) for s in scores])

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=criteria,
        y=alts,
        text=text,
        texttemplate="%{text}",
        colorscale="RdYlGn",
        colorbar=dict(title="评分"),
    ))
    fig.update_layout(
        title=title,
        xaxis_title="准则",
        yaxis_title="方案",
        height=max(300, n_alts * 60),
        template="plotly_white",
    )
    return fig


def plot_criteria_weights_radar(
    criteria: List[str],
    voter_weights: Dict[str, Dict[str, float]],
) -> go.Figure:
    """Create a radar chart comparing criteria weights across voters."""
    fig = go.Figure()
    for i, (voter_name, weights) in enumerate(voter_weights.items()):
        r_values = [weights.get(c, 0) for c in criteria]
        # Close the polygon
        r_values_closed = r_values + [r_values[0]]
        theta_closed = criteria + [criteria[0]]

        fig.add_trace(go.Scatterpolar(
            r=r_values_closed,
            theta=theta_closed,
            fill="toself",
            name=voter_name,
            opacity=0.5,
            line=dict(color=_COLORS[i % len(_COLORS)]),
        ))

    fig.update_layout(
        title="各投票人准则权重对比",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        height=500,
        template="plotly_white",
        showlegend=True,
    )
    return fig


def plot_mcdm_ranking_comparison(
    mcdm_rankings: Dict[str, List[str]],
    social_rankings: Dict[str, List[str]],
) -> go.Figure:
    """Create a heatmap comparing MCDM rankings vs social choice rankings."""
    all_rankings = {**social_rankings, **mcdm_rankings}
    methods = list(all_rankings.keys())

    # Collect all alternatives
    all_alts = []
    for ranking in all_rankings.values():
        for alt in ranking:
            if alt not in all_alts:
                all_alts.append(alt)

    # Build rank matrix: rows=methods, cols=alternatives
    z = []
    text = []
    for method in methods:
        ranking = all_rankings[method]
        row = []
        row_text = []
        for alt in all_alts:
            if alt in ranking:
                rank = ranking.index(alt) + 1
                row.append(rank)
                row_text.append(f"#{rank}")
            else:
                row.append(0)
                row_text.append("—")
        z.append(row)
        text.append(row_text)

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=all_alts,
        y=methods,
        text=text,
        texttemplate="%{text}",
        colorscale="RdYlGn_r",
        colorbar=dict(title="排名"),
    ))
    fig.update_layout(
        title="MCDM排名 vs 社会选择排名 (数字=排名, 越小越好)",
        xaxis_title="方案",
        yaxis_title="方法",
        height=max(300, len(methods) * 45 + 100),
        margin=dict(l=200),
        yaxis=dict(tickfont=dict(size=9)),
        template="plotly_white",
    )
    return fig


def plot_sensitivity_tornado(sensitivity, alternative: str) -> go.Figure:
    """Create a tornado diagram showing how criterion weight changes affect an alternative's rank."""
    from .mcdm_sensitivity import SensitivityResult

    criteria_set = set()
    for w in sensitivity.weight_variations:
        criteria_set.update(w.keys())
    criteria = sorted(criteria_set)

    n_steps = len(sensitivity.rank_changes.get(alternative, []))
    if n_steps == 0:
        return go.Figure()

    # Group rank changes by criterion
    ranks = sensitivity.rank_changes[alternative]
    n_crit = len(criteria)
    steps_per_crit = math.ceil(n_steps / n_crit) if n_crit > 0 else n_steps

    fig = go.Figure()

    crit_labels = []
    min_ranks = []
    max_ranks = []
    base_rank = sensitivity.base_ranking.index(alternative) + 1 if alternative in sensitivity.base_ranking else 0

    for i, crit in enumerate(criteria):
        start_idx = i * steps_per_crit
        end_idx = start_idx + steps_per_crit
        ranks_slice = ranks[start_idx:end_idx]
        if ranks_slice:
            min_r = min(ranks_slice)
            max_r = max(ranks_slice)
            crit_labels.append(crit)
            min_ranks.append(min_r)
            max_ranks.append(max_r)

    # Tornado bars: horizontal bars from min_rank to max_rank
    for i, crit in enumerate(crit_labels):
        fig.add_trace(go.Bar(
            y=[crit],
            x=[max_ranks[i] - min_ranks[i]],
            base=[min_ranks[i]],
            orientation="h",
            marker_color=_COLORS[i % len(_COLORS)],
            text=[f"#{min_ranks[i]} ~ #{max_ranks[i]}"],
            textposition="outside",
            showlegend=False,
        ))

    # Add base rank line
    fig.add_vline(x=base_rank, line_dash="dash", line_color="red",
                  annotation_text=f"基准排名 #{base_rank}")

    fig.update_layout(
        title=f"灵敏度分析 — {alternative} 的排名变化（龙卷风图）",
        xaxis_title="排名（数字越小越好）",
        yaxis_title="准则",
        height=max(300, len(crit_labels) * 60 + 100),
        template="plotly_white",
        barmode="overlay",
    )
    return fig
