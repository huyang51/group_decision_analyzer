"""Page: Theory Reference — display group decision theory documentation."""

from __future__ import annotations

import streamlit as st
from pathlib import Path


THEORY_FILE = Path(__file__).parent.parent.parent / "group_decision_theory.md"


def render_page():
    """Render the theory reference page."""
    st.header("理论参考 — 群决策理论基础")

    if not THEORY_FILE.exists():
        st.warning("理论文档文件未找到。请确保 group_decision_theory.md 在项目根目录。")
        return

    content = THEORY_FILE.read_text(encoding="utf-8")

    # Split by top-level headings for collapsible sections
    sections = []
    current_title = ""
    current_lines = []

    for line in content.split("\n"):
        if line.startswith("# ") and not line.startswith("## "):
            if current_title:
                sections.append((current_title, "\n".join(current_lines)))
            current_title = line.lstrip("# ").strip()
            current_lines = []
        elif line.startswith("## "):
            if current_title:
                sections.append((current_title, "\n".join(current_lines)))
            current_title = line.lstrip("## ").strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_title:
        sections.append((current_title, "\n".join(current_lines)))

    # Render sections
    for title, body in sections:
        with st.expander(title, expanded=False):
            st.markdown(body)
