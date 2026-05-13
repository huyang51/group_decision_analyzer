"""Deliberation agent: simulates how stakeholders respond to analysis results."""

from __future__ import annotations

from typing import Optional

from .base_agent import BaseAgent

SYSTEM_PROMPT = """你是一个群决策分析系统的协商智能体。你的职责是：
1. 模拟各方在看到分析结果后的立场变化
2. 预测偏好可能如何转移
3. 识别可能的妥协空间和共识区域
4. 提出促进共识的协商策略

请用中文回复，保持客观中立的视角。"""


class DeliberationAgent(BaseAgent):
    """Simulates post-analysis deliberation and preference shifts."""

    def simulate_deliberation(self, analysis_result: str, stakeholders: str) -> str:
        """Simulate how stakeholders would respond to seeing analysis results."""
        user_msg = f"""## 分析结果
{analysis_result}

## 利益相关者
{stakeholders}

假设各方都能看到以上分析结果，请模拟他们的反应：

1. **各方立场变化**：每个利益相关者看到结果后，立场会如何变化？
2. **妥协空间**：哪些方案可能成为各方都能接受的折中选项？
3. **谈判筹码**：各方的谈判筹码是什么？谁最可能让步？
4. **共识路径**：从当前位置到共识，最可能的路径是什么？

请以"协商模拟"的形式，预测各方的发言和立场演变。"""

        return self.chat(SYSTEM_PROMPT, user_msg, temperature=0.7, max_tokens=2000)

    def predict_preference_shift(
        self, current_prefs: str, analysis_insight: str
    ) -> str:
        """Predict how preferences might shift given analytical insights."""
        user_msg = f"""## 当前偏好结构
{current_prefs}

## 分析洞察
{analysis_insight}

基于分析揭示的模式（如循环、均衡、风险域），预测：
1. 哪些投票人最可能改变偏好？
2. 偏好变化的方向是什么？
3. 变化后的结果会如何改变？
4. 是否存在"关键投票人"——其偏好变化能打破僵局？"""

        return self.chat(SYSTEM_PROMPT, user_msg, temperature=0.5, max_tokens=1500)
