"""Sidebar navigation for the Streamlit app."""

from __future__ import annotations

import streamlit as st


PAGES = {
    "建模": "modeling",
    "核心分析": "analysis",
    "动态推演": "simulation",
    "敏感性分析": "sensitivity",
    "案例库": "cases",
    "理论参考": "theory",
}


def render_sidebar() -> str:
    """Render the sidebar navigation and return the selected page key."""
    with st.sidebar:
        st.title("群决策分析系统")
        st.caption("Multi-Agent Group Decision Analyzer")
        st.divider()

        selected = st.radio(
            "导航",
            list(PAGES.keys()),
            index=0,
            key="nav_radio",
        )

        st.divider()

        # API status
        import os
        api_key = os.getenv("DEEPSEEK_API_KEY", "")
        if api_key:
            st.success("DeepSeek API: 已配置")
        else:
            st.warning("DeepSeek API: 未配置\n\n辩论功能需要API Key。计算和图表功能无需API。")

        st.divider()
        st.caption("基于Arrow不可能定理、孔多塞方法、\nBorda计票、Copeland方法、\nAHP层次分析、纳什均衡、\n前景理论的多智能体分析系统。")

    return PAGES[selected]
