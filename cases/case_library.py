"""Case library: loads and manages predefined decision analysis cases."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional


DATA_DIR = Path(__file__).parent / "data"


class CaseLibrary:
    """Manages a collection of predefined decision analysis cases."""

    def __init__(self):
        self._cases: Dict[str, dict] = {}
        self._load_all_cases()

    def _load_all_cases(self) -> None:
        """Load all JSON case files from the data directory."""
        if not DATA_DIR.exists():
            return
        for json_file in DATA_DIR.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    case_data = json.load(f)
                name = case_data.get("name", json_file.stem)
                self._cases[name] = case_data
            except (json.JSONDecodeError, KeyError):
                continue

    def get_case_names(self) -> List[str]:
        """Return a list of all available case names."""
        return sorted(self._cases.keys())

    def load_case(self, name: str) -> Optional[dict]:
        """Load a case by name. Returns None if not found."""
        return self._cases.get(name)

    def get_case_summary(self, name: str) -> str:
        """Return a brief summary of a case."""
        case = self._cases.get(name)
        if not case:
            return f"案例 '{name}' 未找到。"
        desc = case.get("description", "")
        point = case.get("teaching_point", "")
        return f"{desc}\n\n教学点：{point}"

    def get_scenario(self, case_name: str, scenario_idx: int = 0) -> Optional[dict]:
        """Get a specific scenario from a case."""
        case = self._cases.get(case_name)
        if not case:
            return None
        scenarios = case.get("scenarios", [])
        if 0 <= scenario_idx < len(scenarios):
            return scenarios[scenario_idx]
        return None
