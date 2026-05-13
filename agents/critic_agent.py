"""Critic agent: provides critical analysis of decision results."""

from __future__ import annotations

from typing import Optional

from .base_agent import BaseAgent

SYSTEM_PROMPT = """你是一个群决策分析系统的批判智能体。你的职责是：
1. 对分析结果提出质疑和反面论证
2. 检验方法论的合理性
3. 识别潜在的策略性投票和操纵风险
4. 从Arrow不可能定理角度审视结果

请用中文回复，保持建设性批判的态度。"""


class CriticAgent(BaseAgent):
    """Provides critical analysis and devil's advocate perspectives."""

    def critique_analysis(self, analysis_result: str) -> str:
        """Provide a comprehensive critique of the analysis results."""
        user_msg = f"""## 分析结果
{analysis_result}

请从以下角度对分析结果进行批判性审视：

1. **Arrow定理视角**：该结果是否违反了Arrow不可能定理的某个条件？如果投票规则满足了所有条件，那一定存在独裁者吗？

2. **策略性投票风险**：是否存在投票人可以通过谎报偏好获益的可能？投票结果是否容易被操纵？

3. **群体思维检验**：如果这是真实决策场景，是否存在群体思维的迹象？异议是否被充分表达？

4. **方法局限性**：孔多塞、Borda、Copeland三种方法给出的结论是否一致？如果不一致，哪种方法更可信？

5. **反事实推理**：如果某个投票人改变偏好，结果会如何变化？结果的稳健性如何？"""

        return self.chat(SYSTEM_PROMPT, user_msg, temperature=0.6, max_tokens=2000)

    def check_methodology(self, methods_used: str) -> str:
        """Question the appropriateness of the chosen methods."""
        user_msg = f"""## 使用的分析方法
{methods_used}

请质疑这些方法选择的合理性：
1. 是否遗漏了重要的分析维度？
2. 方法之间是否存在理论冲突？
3. 对于当前场景，是否有更合适的方法？
4. 这些方法各自的假设是否在当前场景中成立？"""

        return self.chat(SYSTEM_PROMPT, user_msg, temperature=0.5, max_tokens=1500)
