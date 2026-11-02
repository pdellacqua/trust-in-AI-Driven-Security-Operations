"""Tests for explanation generator and knowledge base."""

import unittest
from src.explanations.knowledge_base import ThreatKnowledgeBase
from src.explanations.generator import AdaptiveExplanationGenerator


class TestExplanations(unittest.TestCase):

    def test_knowledge_base_retrieval(self):
        kb = ThreatKnowledgeBase()
        self.assertIn("DDoS", kb.classes)
        self.assertIn("Infiltration", kb.classes)

        ddos_info = kb.get_class_info("DDoS")
        self.assertEqual(ddos_info["mitre"]["technique_id"], "T1498")
        self.assertGreater(len(ddos_info["cve"]), 0)

    def test_explanation_levels_rendering(self):
        gen = AdaptiveExplanationGenerator()

        e0 = gen.generate("E0", "Infiltration", reliability=0.88)
        self.assertEqual(e0["level"], "E0")
        self.assertNotIn("mitre_attack", e0)

        e1 = gen.generate("E1", "Infiltration", reliability=0.88)
        self.assertEqual(e1["level"], "E1")
        self.assertIn("calibrated_reliability", e1)

        e2 = gen.generate("E2", "Infiltration", reliability=0.88)
        self.assertEqual(e2["level"], "E2")
        self.assertIn("mitre_attack", e2)
        self.assertIn("cve_references", e2)

        e3 = gen.generate("E3", "Infiltration", reliability=0.88)
        self.assertEqual(e3["level"], "E3")
        self.assertIn("alternative_hypotheses", e3)
        self.assertIn("verification_checklist", e3)


if __name__ == "__main__":
    unittest.main()
