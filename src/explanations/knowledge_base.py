"""
Cybersecurity Knowledge Base and Ontological Mappings.
Provides MITRE ATT&CK, CVE, and semantic prototype metadata for network threats.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional


class ThreatKnowledgeBase:
    """Interface to structured cybersecurity knowledge base."""

    def __init__(self, kb_path: Optional[str] = None):
        if kb_path is None:
            kb_path = Path(__file__).resolve().parent.parent.parent / "data" / "mitre_cve_kb.json"

        with open(kb_path, "r", encoding="utf-8") as f:
            self._raw = json.load(f)

        self.classes: Dict[str, Dict[str, Any]] = self._raw["classes"]
        self.held_out: Dict[str, List[str]] = self._raw.get("held_out_zero_shot", {})

        self.class_by_id = {v["class_id"]: k for k, v in self.classes.items()}

    def get_class_info(self, class_name_or_id: Any) -> Dict[str, Any]:
        """Returns structured metadata for a threat class."""
        if isinstance(class_name_or_id, int):
            class_name = self.class_by_id.get(class_name_or_id, "Benign")
        else:
            class_name = str(class_name_or_id)

        return self.classes.get(class_name, self.classes["Benign"])

    def get_mitre_info(self, class_name_or_id: Any) -> Dict[str, str]:
        """Returns MITRE ATT&CK tactic and technique."""
        info = self.get_class_info(class_name_or_id)
        return info.get("mitre", {"tactic": "None", "technique_id": "None", "technique_name": "Unknown"})

    def get_cve_list(self, class_name_or_id: Any) -> List[str]:
        """Returns associated CVE IDs."""
        info = self.get_class_info(class_name_or_id)
        return info.get("cve", [])

    def get_severity(self, class_name_or_id: Any) -> float:
        """Returns normalized severity s_t in [0, 1]."""
        info = self.get_class_info(class_name_or_id)
        return float(info.get("severity", 0.5))

    def get_cvss(self, class_name_or_id: Any) -> float:
        """Returns CVSS score in [0.0, 10.0]."""
        info = self.get_class_info(class_name_or_id)
        return float(info.get("cvss", 5.0))
