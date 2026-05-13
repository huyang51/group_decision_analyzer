"""Behavioral decision theory: Groupthink diagnosis and Prospect Theory risk domain classification."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


GROUPTHINK_SYMPTOMS = [
    ("illusion_of_invulnerability", "无敌幻觉", "群体过度自信，忽视风险"),
    ("collective_rationalization", "集体合理化", "群体为决策找借口，忽视警告信号"),
    ("belief_in_inherent_morality", "道德信念", "群体相信自身目标天然正确"),
    ("stereotyping_outgroups", "外部刻板印象", "将反对者视为愚蠢、邪恶或软弱"),
    ("pressure_on_dissenters", "压制异议", "对持不同意见者施加压力"),
    ("self_censorship", "自我审查", "成员主动隐藏疑虑和分歧"),
    ("illusion_of_unanimity", "一致幻觉", "沉默被误读为同意"),
    ("mindguards", "思想卫士", "某些成员主动屏蔽不利信息"),
]


@dataclass
class GroupthinkDiagnosis:
    """Result of a Groupthink diagnosis based on 8 symptoms."""
    symptoms: Dict[str, bool]
    symptom_details: Dict[str, str]
    score: int
    total: int
    conclusion: str
    risk_level: str  # "低", "中", "高"


@dataclass
class RiskProfile:
    """A stakeholder's risk profile under Prospect Theory."""
    stakeholder: str
    domain: str  # "损失域" or "收益域"
    risk_preference: str  # "风险寻求" or "风险规避"
    reference_point: str
    description: str


def diagnose_groupthink(events_description: str, symptom_checks: Optional[Dict[str, bool]] = None) -> GroupthinkDiagnosis:
    """Diagnose groupthink risk based on the 8 Janis symptoms.

    If symptom_checks is provided (from user input or LLM), use those values.
    Otherwise, return a template for manual assessment.

    Args:
        events_description: text description of the group decision context.
        symptom_checks: optional dict mapping symptom key -> True/False.

    Returns:
        GroupthinkDiagnosis with symptom analysis and risk level.
    """
    symptoms: Dict[str, bool] = {}
    details: Dict[str, str] = {}

    if symptom_checks:
        for key, label, desc in GROUPTHINK_SYMPTOMS:
            symptoms[key] = symptom_checks.get(key, False)
            details[key] = f"{label}: {'存在' if symptoms[key] else '未发现'} — {desc}"
    else:
        # Return template for manual assessment
        for key, label, desc in GROUPTHINK_SYMPTOMS:
            symptoms[key] = False
            details[key] = f"{label}: 待评估 — {desc}"

    score = sum(1 for v in symptoms.values() if v)
    total = len(GROUPTHINK_SYMPTOMS)

    if score <= 2:
        risk_level = "低"
        conclusion = f"群体思维风险较低（{score}/{total} 症状匹配）。群体决策过程相对健康。"
    elif score <= 5:
        risk_level = "中"
        conclusion = f"群体思维风险中等（{score}/{total} 症状匹配）。建议引入外部视角和结构化辩论。"
    else:
        risk_level = "高"
        conclusion = f"群体思维风险高（{score}/{total} 症状匹配）。强烈建议：指定魔鬼代言人、匿名投票、分组独立讨论。"

    return GroupthinkDiagnosis(
        symptoms=symptoms,
        symptom_details=details,
        score=score,
        total=total,
        conclusion=conclusion,
        risk_level=risk_level,
    )


def classify_risk_domain(
    stakeholder: str,
    gains: List[str],
    losses: List[str],
    reference_point: str = "现状",
) -> RiskProfile:
    """Classify a stakeholder's risk domain using Prospect Theory.

    Prospect Theory: people are risk-averse in gains, risk-seeking in losses.

    Args:
        stakeholder: name of the stakeholder.
        gains: list of potential gains.
        losses: list of potential losses.
        reference_point: what the stakeholder considers the status quo.

    Returns:
        RiskProfile with domain classification and risk preference.
    """
    gain_count = len(gains)
    loss_count = len(losses)

    if loss_count > gain_count:
        domain = "损失域"
        risk_preference = "风险寻求"
        description = (
            f"{stakeholder} 面临更多损失（{loss_count}项损失 vs {gain_count}项收益），"
            f"根据前景理论，处于损失域时倾向于风险寻求——更可能押注高风险选项以避免确定的损失。"
        )
    elif gain_count > loss_count:
        domain = "收益域"
        risk_preference = "风险规避"
        description = (
            f"{stakeholder} 拥有更多收益（{gain_count}项收益 vs {loss_count}项损失），"
            f"根据前景理论，处于收益域时倾向于风险规避——更可能锁定确定收益而非冒险。"
        )
    else:
        # Equal gains and losses — use loss aversion (losses loom larger)
        domain = "损失域（损失厌恶）"
        risk_preference = "风险寻求"
        description = (
            f"{stakeholder} 收益与损失相当，但损失厌恶效应使损失的心理权重约为收益的2倍，"
            f"实际处于损失域，倾向于风险寻求。"
        )

    return RiskProfile(
        stakeholder=stakeholder,
        domain=domain,
        risk_preference=risk_preference,
        reference_point=reference_point,
        description=description,
    )
