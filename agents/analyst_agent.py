"""Analyst agent: interprets numerical results into natural language."""

from __future__ import annotations

from typing import Optional

from .base_agent import BaseAgent

SYSTEM_PROMPT = """你是一个群决策分析系统的分析智能体。你的职责是：
1. 解读孔多塞配对比较矩阵的数字含义
2. 解释Borda分数和排名的意义
3. 解释Copeland得分的含义
4. 判断是否存在循环偏好（孔多塞悖论）
5. 将技术性分析结果翻译成通俗易懂的自然语言

请用中文回复，语言要清晰、专业但不晦涩。"""


class AnalystAgent(BaseAgent):
    """Interprets numerical analysis results into natural language."""

    def interpret_condorcet(
        self,
        pairwise_matrix_desc: str,
        condorcet_winner: Optional[str],
        cycle_detected: bool,
        cycle_path: Optional[str] = None,
    ) -> str:
        """Interpret Condorcet analysis results."""
        user_msg = f"""## 孔多塞分析结果

配对比较矩阵：
{pairwise_matrix_desc}

孔多塞赢家：{condorcet_winner if condorcet_winner else "无"}
循环检测：{"检测到循环" if cycle_detected else "无循环"}
{f"循环路径：{cycle_path}" if cycle_path else ""}

请解读以上结果的含义，说明：
1. 谁在两两对决中获胜最多？
2. 是否存在孔多塞赢家？如果不存在，原因是什么？
3. 如果存在循环，这对决策意味着什么？"""

        return self.chat(SYSTEM_PROMPT, user_msg, temperature=0.5)

    def interpret_borda(self, borda_desc: str) -> str:
        """Interpret Borda count results."""
        user_msg = f"""## Borda 计票结果

{borda_desc}

请解读Borda分数的含义，说明：
1. 排名是否与直觉一致？
2. 分数差距说明了什么？
3. 与孔多塞分析相比，结论是否一致？"""

        return self.chat(SYSTEM_PROMPT, user_msg, temperature=0.5)

    def interpret_full_analysis(self, full_data: str) -> str:
        """Interpret a complete analysis result."""
        user_msg = f"""## 完整分析数据

{full_data}

请提供全面的分析解读，综合孔多塞、Borda、Copeland三种方法的结果，指出一致性和差异。"""

        return self.chat(SYSTEM_PROMPT, user_msg, temperature=0.5)
