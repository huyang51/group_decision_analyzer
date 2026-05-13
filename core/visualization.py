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

from .models import PairwiseResult, BordaScores, SimulationStep, CopelandScores


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
