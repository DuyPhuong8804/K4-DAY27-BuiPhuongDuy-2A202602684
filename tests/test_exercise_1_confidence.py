import unittest
from unittest.mock import patch

from common.github import PullRequest
from common.schemas import PRAnalysis
from exercises import exercise_1_confidence as exercise


class ConfidenceGraphTests(unittest.TestCase):
    def test_confidence_boundaries_reach_expected_terminal(self):
        pr = PullRequest(
            url="https://github.com/example/repo/pull/1",
            owner="example", repo="repo", number=1, title="Example PR",
            author="reviewer", base_ref="main", head_ref="feature",
            head_sha="abcdef1234", diff="+ example", files_changed=["example.py"],
        )
        cases = [
            (0.0, "escalate", "pending_escalation"),
            (0.5799, "escalate", "pending_escalation"),
            (0.58, "human_approval", "pending_human_approval"),
            (0.7299, "human_approval", "pending_human_approval"),
            (0.73, "auto_approve", "auto_approved"),
            (1.0, "auto_approve", "auto_approved"),
        ]
        for confidence, decision, action in cases:
            with self.subTest(confidence=confidence):
                analysis = PRAnalysis(
                    summary="Example review", confidence=confidence,
                    confidence_reasoning="Boundary test",
                )
                with patch.object(exercise, "fetch_pr", return_value=pr), patch.object(
                    exercise, "get_llm"
                ) as get_llm:
                    get_llm.return_value.with_structured_output.return_value.invoke.return_value = analysis
                    final = exercise.build_graph().invoke({"pr_url": pr.url})
                self.assertEqual(final["decision"], decision)
                self.assertEqual(final["final_action"], action)
                self.assertEqual(final["analysis"], analysis)


if __name__ == "__main__":
    unittest.main()
