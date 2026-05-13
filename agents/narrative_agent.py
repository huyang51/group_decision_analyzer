"""Narrative agent: generates compelling narratives and historical analogies."""

from __future__ import annotations

from typing import Optional

from .base_agent import BaseAgent

SYSTEM_PROMPT = """你是一个群决策分析系统的叙事智能体。你的职责是：
1. 将技术性分析结果转化为引人入胜的案例叙述
2. 引入历史类比来说明决策模式的普遍性
3. 用故事化的语言让抽象概念更易理解

请用中文回复，语言生动但保持学术严谨。"""


class NarrativeAgent(BaseAgent):
    """Generates narratives and historical analogies for decision analysis."""

    def generate_narrative(self, analysis_data: str, case_context: str) -> str:
        """Generate a 300-500 word narrative case study."""
        user_msg = f"""## 分析数据
{analysis_data}

## 案例背景
{case_context}

请将以上分析结果写成一个引人入胜的案例叙述（300-500字），包括：
1. 背景铺垫：谁在做决定？为什么这个决定重要？
2. 冲突核心：分歧在哪里？各方立场是什么？
3. 分析洞察：数据揭示了什么模式？
4. 启示总结：这个案例能教会我们什么？

语言风格：学术叙事，既专业又可读。"""

        return self.chat(SYSTEM_PROMPT, user_msg, temperature=0.7, max_tokens=2000)

    def draw_historical_analogy(self, analysis_data: str, case_library: str = "") -> str:
        """Draw historical analogies to illustrate the decision pattern."""
        user_msg = f"""## 当前分析结果
{analysis_data}

{f"## 已知案例库{chr(10)}{case_library}" if case_library else ""}

请引入1-2个历史类比来说明当前决策模式的普遍性。可以参考：
1. 历史上的著名投票悖论（如18世纪孔多塞发现的循环偏好）
2. 政治决策中的类似案例（如选举制度改革争议）
3. 商业决策中的类比（如企业并购中的多方博弈）

说明类比的相似之处和差异。"""

        return self.chat(SYSTEM_PROMPT, user_msg, temperature=0.7, max_tokens=1500)
