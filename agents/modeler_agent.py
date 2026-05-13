"""Modeler agent: structures raw scenario descriptions into decision models."""

from __future__ import annotations

from typing import Optional

from .base_agent import BaseAgent

SYSTEM_PROMPT = """你是一个群决策分析系统的建模智能体。你的职责是：
1. 从场景描述中识别所有利益相关者（stakeholders）
2. 根据场景建议可行的备选方案（alternatives）
3. 构建决策准则（criteria）和评估维度

请用中文回复，输出结构化的JSON格式。"""


class ModelerAgent(BaseAgent):
    """Structures raw scenario descriptions into formal decision models."""

    def identify_stakeholders(self, scenario: str) -> str:
        """Identify stakeholders from a scenario description."""
        user_msg = f"""## 场景描述
{scenario}

请识别这个决策场景中的所有利益相关者，列出：
1. 每个利益相关者的名称
2. 他们的核心利益和诉求
3. 他们的影响力程度（高/中/低）
4. 他们之间的关系（联盟/对立/中立）

请用结构化格式输出。"""

        return self.chat(SYSTEM_PROMPT, user_msg, temperature=0.3, max_tokens=1500)

    def suggest_alternatives(self, scenario: str, stakeholders: str) -> str:
        """Suggest feasible alternatives based on scenario and stakeholders."""
        user_msg = f"""## 场景描述
{scenario}

## 已识别的利益相关者
{stakeholders}

请根据场景和利益相关者的诉求，建议3-5个可行的备选方案。每个方案需要：
1. 方案名称（简短）
2. 方案描述
3. 主要支持者和反对者
4. 关键优势和风险"""

        return self.chat(SYSTEM_PROMPT, user_msg, temperature=0.5, max_tokens=1500)

    def build_decision_criteria(self, scenario: str) -> str:
        """Build decision criteria for evaluating alternatives."""
        user_msg = f"""## 场景描述
{scenario}

请为这个群决策场景构建评估准则，包括：
1. 3-5个关键评估维度
2. 每个维度的权重建议（总和为1）
3. 每个维度的评估标准说明

请考虑效率、公平性、可行性、风险等通用维度，以及场景特有的维度。"""

        return self.chat(SYSTEM_PROMPT, user_msg, temperature=0.3, max_tokens=1200)
