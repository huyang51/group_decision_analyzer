"""Coordinator agent: orchestrates the analysis pipeline."""

from __future__ import annotations

from typing import Optional

from .base_agent import BaseAgent

SYSTEM_PROMPT = """你是一个群决策分析系统的协调智能体。你的职责是：
1. 理解用户输入的投票偏好数据和场景描述
2. 决定需要运行哪些分析方法（孔多塞分析、Borda计票、Copeland方法、Arrow定理检验）
3. 将分析结果整理成结构化摘要

请用中文回复。输出格式要求：
- 先简述场景和投票结构
- 列出将要执行的分析步骤
- 总结关键发现"""


RECOMMENDATION_PROMPT = """你是一个群决策分析系统的协调智能体，现在需要基于完整的辩论记录给出最终决策建议。
请综合技术分析结果和各方观点，给出平衡、务实的建议。"""


class CoordinatorAgent(BaseAgent):
    """Orchestrates the analysis pipeline and produces a summary."""

    def plan_analysis(self, scenario: str, voter_data: str) -> str:
        """Given a scenario description and voter data, produce an analysis plan."""
        user_msg = f"""## 场景描述
{scenario}

## 投票者偏好数据
{voter_data}

请分析这个群决策场景，说明将执行哪些分析方法，并预判可能的结果。"""

        return self.chat(SYSTEM_PROMPT, user_msg, temperature=0.5)

    def summarize_results(self, scenario: str, analysis_summary: str) -> str:
        """Summarize the complete analysis results."""
        user_msg = f"""## 场景
{scenario}

## 分析结果摘要
{analysis_summary}

请将以上分析结果整合成一份结构化摘要，突出关键发现和决策建议。"""

        return self.chat(SYSTEM_PROMPT, user_msg, temperature=0.6)

    def generate_final_recommendation(self, debate_summary: str, analysis_result: str) -> str:
        """Generate a final recommendation by synthesizing debate and analysis."""
        user_msg = f"""## 技术分析结果
{analysis_result}

## 辩论摘要
{debate_summary}

请基于以上技术分析和辩论讨论，生成最终决策建议。要求：
1. 明确推荐的方案（或方案组合）
2. 推荐理由（综合技术分析和各方观点）
3. 实施建议和风险提示
4. 备选方案（如果推荐方案不可行）"""

        return self.chat(RECOMMENDATION_PROMPT, user_msg, temperature=0.4, max_tokens=1500)
