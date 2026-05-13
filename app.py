"""Streamlit entry point for the Group Decision Analyzer."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure both the package root and its parent are on sys.path.
# This allows running from either:
#   cd class_project && streamlit run group_decision_analyzer/app.py
#   cd group_decision_analyzer && streamlit run app.py
_here = Path(__file__).resolve().parent
_parent = _here.parent
for p in (str(_here), str(_parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

import streamlit as st

# Load .env if available
try:
    from dotenv import load_dotenv
    load_dotenv(_here / ".env")
except ImportError:
    pass


def main():
    st.set_page_config(
        page_title="群决策分析系统",
        page_icon="🗳️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Import UI modules — try absolute (from parent dir) then direct (from inside)
    try:
        from group_decision_analyzer.ui.sidebar import render_sidebar
        from group_decision_analyzer.ui import (
            page_modeling,
            page_analysis,
            page_simulation,
            page_sensitivity,
            page_cases,
            page_theory,
        )
    except ModuleNotFoundError:
        from ui.sidebar import render_sidebar
        from ui import (
            page_modeling,
            page_analysis,
            page_simulation,
            page_sensitivity,
            page_cases,
            page_theory,
        )

    # Render sidebar and get selected page
    selected_page = render_sidebar()

    # Route to pages
    page_map = {
        "modeling": page_modeling.render_page,
        "analysis": page_analysis.render_page,
        "simulation": page_simulation.render_page,
        "sensitivity": page_sensitivity.render_page,
        "cases": page_cases.render_page,
        "theory": page_theory.render_page,
    }

    render_fn = page_map.get(selected_page)
    if render_fn:
        render_fn()
    else:
        st.error(f"未知页面: {selected_page}")


if __name__ == "__main__":
    main()
