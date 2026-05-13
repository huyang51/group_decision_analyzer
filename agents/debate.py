"""Debate manager: orchestrates multi-agent debate on decision analysis results."""

from __future__ import annotations

from typing import Dict, Generator, List, Optional

from ..core.models import DebateRecord
from .analyst_agent import AnalystAgent
from .critic_agent import CriticAgent
from .narrative_agent import NarrativeAgent
from .modeler_agent import ModelerAgent
from .deliberation_agent import DeliberationAgent
from .coordinator_agent import CoordinatorAgent


DEBATE_SYSTEM_PROMPT = """你正在参与一场关于群决策分析的学术辩论。请根据你的角色定位发言。
你可以引用和反驳其他参与者的观点，但要保持学术礼貌。
请用中文回复，发言长度控制在200-400字。"""


class DebateManager:
    """Manages the multi-agent debate process.

    Debate flow:
    1. Analyst presents technical findings
    2. Critic challenges the analysis
    3. Narrative agent provides context and analogies
    4. Modeler agent suggests structural improvements
    5. Deliberation agent simulates stakeholder responses
    6. Coordinator summarizes and synthesizes
    """

    def __init__(
        self,
        analyst: Optional[AnalystAgent] = None,
        critic: Optional[CriticAgent] = None,
        narrative: Optional[NarrativeAgent] = None,
        modeler: Optional[ModelerAgent] = None,
        deliberation: Optional[DeliberationAgent] = None,
        coordinator: Optional[CoordinatorAgent] = None,
    ):
        self.analyst = analyst or AnalystAgent()
        self.critic = critic or CriticAgent()
        self.narrative = narrative or NarrativeAgent()
        self.modeler = modeler or ModelerAgent()
        self.deliberation = deliberation or DeliberationAgent()
        self.coordinator = coordinator or CoordinatorAgent()

    def is_available(self) -> bool:
        """Check if all agents have API access."""
        return self.analyst.is_available()

    def run_debate(
        self,
        analysis_result: str,
        case_context: str,
        stakeholders: str = "",
    ) -> DebateRecord:
        """Run the full debate sequence.

        Each agent sees all previous speeches as context.

        Returns:
            DebateRecord with all speeches and final summary.
        """
        speeches: List[Dict[str, str]] = []
        context_so_far = ""

        # Round 1: Analyst presents
        analyst_prompt = f"""## 分析结果
{analysis_result}

## 案例背景
{case_context}

请作为分析者，总结技术分析的关键发现，为辩论开场。重点说明：
- 孔多塞赢家/循环状态
- Borda和Copeland排名
- Arrow定理检验结果
- 最重要的发现是什么"""

        analyst_speech = self.analyst.chat(DEBATE_SYSTEM_PROMPT, analyst_prompt, temperature=0.4, max_tokens=1200)
        speeches.append({"agent": "分析师", "content": analyst_speech, "round": 1})
        context_so_far += f"\n\n### 分析师发言\n{analyst_speech}"

        # Round 2: Critic challenges
        critic_prompt = f"""## 分析结果
{analysis_result}
{context_so_far}

作为批判者，请对分析师的发现提出质疑。你可以说：
- 方法论是否恰当？
- 是否存在策略性投票的可能？
- Arrow定理告诉我们什么？
- 分析是否遗漏了什么？"""

        critic_speech = self.critic.chat(DEBATE_SYSTEM_PROMPT, critic_prompt, temperature=0.5, max_tokens=1200)
        speeches.append({"agent": "批判者", "content": critic_speech, "round": 2})
        context_so_far += f"\n\n### 批判者发言\n{critic_speech}"

        # Round 3: Narrative provides context
        narrative_prompt = f"""## 分析结果
{analysis_result}
{context_so_far}

作为叙事者，请为辩论提供历史和理论背景：
- 引入一个相关的决策案例类比
- 从行为决策理论角度解读结果
- 这个案例在更广阔的知识图谱中处于什么位置？"""

        narrative_speech = self.narrative.chat(DEBATE_SYSTEM_PROMPT, narrative_prompt, temperature=0.7, max_tokens=1200)
        speeches.append({"agent": "叙事者", "content": narrative_speech, "round": 3})
        context_so_far += f"\n\n### 叙事者发言\n{narrative_speech}"

        # Round 4: Modeler (if available) suggests structure
        if self.modeler and self.modeler.is_available():
            modeler_prompt = f"""## 分析结果
{analysis_result}
{context_so_far}

作为建模者，请审视当前的决策模型结构：
- 利益相关者识别是否完整？
- 备选方案是否穷尽？
- 是否有遗漏的关键维度？
- 建议如何改进决策框架？"""

            modeler_speech = self.modeler.chat(DEBATE_SYSTEM_PROMPT, modeler_prompt, temperature=0.5, max_tokens=1200)
            speeches.append({"agent": "建模者", "content": modeler_speech, "round": 4})
            context_so_far += f"\n\n### 建模者发言\n{modeler_speech}"

        # Round 5: Deliberation simulates response
        deliberation_prompt = f"""## 分析结果
{analysis_result}
{context_so_far}

## 利益相关者
{stakeholders if stakeholders else "（请从分析结果中推断利益相关者）"}

作为协商模拟者，请：
- 预测各方看到以上辩论后的反应
- 指出可能的妥协空间
- 识别共识的障碍
- 提出促进共识的建议"""

        deliberation_speech = self.deliberation.chat(DEBATE_SYSTEM_PROMPT, deliberation_prompt, temperature=0.6, max_tokens=1200)
        speeches.append({"agent": "协商者", "content": deliberation_speech, "round": len(speeches) + 1})
        context_so_far += f"\n\n### 协商者发言\n{deliberation_speech}"

        # Final round: Coordinator summarizes
        coordinator_prompt = f"""## 分析结果
{analysis_result}

## 完整辩论记录
{context_so_far}

作为协调者，请综合以上所有发言，给出最终结论：
1. 辩论中最重要的共识点
2. 仍然存在的分歧
3. 最终的决策建议
4. 需要进一步分析的问题"""

        coordinator_speech = self.coordinator.chat(DEBATE_SYSTEM_PROMPT, coordinator_prompt, temperature=0.4, max_tokens=1500)
        speeches.append({"agent": "协调者", "content": coordinator_speech, "round": len(speeches) + 1})

        return DebateRecord(
            speeches=speeches,
            summary=coordinator_speech,
            total_rounds=len(speeches),
        )

    def run_debate_stream(
        self,
        analysis_result: str,
        case_context: str,
        stakeholders: str = "",
    ) -> Generator[Dict[str, str], None, None]:
        """Run debate with streaming, yielding each speech as it's generated.

        Yields dicts with keys: agent, content, round, is_final.
        """
        speeches: List[Dict[str, str]] = []
        context_so_far = ""

        agents_sequence = [
            ("分析师", self.analyst, f"""## 分析结果
{analysis_result}

## 案例背景
{case_context}

请作为分析者，总结技术分析的关键发现，为辩论开场。重点说明孔多塞赢家/循环状态、Borda和Copeland排名、Arrow定理检验结果。"""),
            ("批判者", self.critic, f"""## 分析结果
{analysis_result}

{{context}}

作为批判者，请对分析师的发现提出质疑：方法论是否恰当？是否存在策略性投票可能？Arrow定理告诉我们什么？"""),
            ("叙事者", self.narrative, f"""## 分析结果
{analysis_result}

{{context}}

作为叙事者，请为辩论提供历史和理论背景，引入一个相关的决策案例类比。"""),
            ("协商者", self.deliberation, f"""## 分析结果
{analysis_result}

{{context}}

## 利益相关者
{stakeholders if stakeholders else "（请从分析结果中推断）"}

作为协商模拟者，预测各方反应和妥协空间。"""),
            ("协调者", self.coordinator, f"""## 分析结果
{analysis_result}

{{context}}

作为协调者，请综合所有发言给出最终结论和决策建议。"""),
        ]

        for i, (agent_name, agent, prompt_template) in enumerate(agents_sequence, 1):
            prompt = prompt_template.replace("{context}", context_so_far) if "{context}" in prompt_template else prompt_template
            speech_text = agent.chat(DEBATE_SYSTEM_PROMPT, prompt, temperature=0.5, max_tokens=1200)
            speech = {"agent": agent_name, "content": speech_text, "round": i, "is_final": i == len(agents_sequence)}
            speeches.append({"agent": agent_name, "content": speech_text, "round": i})
            context_so_far += f"\n\n### {agent_name}发言\n{speech_text}"
            yield speech
