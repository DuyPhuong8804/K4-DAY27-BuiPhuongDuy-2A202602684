import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from streamlit.testing.v1 import AppTest

from common.github import PullRequest
from common.schemas import PRAnalysis
from exercises import exercise_4_audit as exercise

APP_PATH = str(Path(__file__).resolve().parents[1] / "app.py")


class StreamlitReviewTests(unittest.TestCase):
    def setUp(self):
        Path("scratch").mkdir(exist_ok=True)
        temp = tempfile.TemporaryDirectory(dir="scratch")
        self.addCleanup(temp.cleanup)
        env = patch.dict(os.environ, {
            "HITL_DB_PATH": str(Path(temp.name) / "app.db"), "GITHUB_USER": "test-reviewer",
        })
        env.start()
        self.addCleanup(env.stop)
        pr = PullRequest(
            url="https://github.com/example/repo/pull/1", owner="example", repo="repo",
            number=1, title="Example PR", author="reviewer", base_ref="main",
            head_ref="feature", head_sha="abcdef1234", diff="+ example", files_changed=["example.py"],
        )
        for name, kwargs in (
            ("fetch_pr", {"return_value": pr}), ("get_llm", {}), ("post_review_comment", {}),
        ):
            mock = patch.object(exercise, name, **kwargs)
            setattr(self, name, mock.start())
            self.addCleanup(mock.stop)

    def start_review(self, confidence=0.65):
        self.get_llm.return_value.with_structured_output.return_value.ainvoke = AsyncMock(return_value=PRAnalysis(
            summary="Review summary", confidence=confidence, confidence_reasoning="Needs context",
            risk_factors=["Weak password hashing"], escalation_questions=["Why MD5 in auth.py?", "Is sync.py HTTPS?"],
        ))
        app = AppTest.from_file(APP_PATH).run(timeout=20)
        self.assertFalse(app.exception)
        app.text_input(key="pr_url_input").set_value("https://github.com/example/repo/pull/1")
        self.click(app, "Run review")
        return app

    def click(self, app, label):
        next(button for button in app.button if button.label == label).click().run(timeout=20)
        self.assertFalse(app.exception)

    def test_approval_buttons_and_refresh_restore_checkpoint(self):
        app = self.start_review()
        self.assertIn("Approval requested", app.subheader[0].value)
        self.post_review_comment.assert_not_called()
        thread = app.session_state["thread_id"]
        restored = AppTest.from_file(APP_PATH)
        restored.query_params["thread"] = thread
        restored.run(timeout=20)
        self.assertFalse(restored.exception)
        self.assertEqual(restored.session_state["thread_id"], thread)
        self.assertIn("Approval requested", restored.subheader[0].value)
        self.assertEqual(self.fetch_pr.call_count, 1)
        self.click(restored, "Approve")
        self.assertEqual(restored.session_state["final"]["final_action"], "committed")
        self.assertTrue(restored.success)
        self.post_review_comment.assert_called_once()
        restored.run(timeout=20)
        self.post_review_comment.assert_called_once()

    def test_reject_and_edit_do_not_post(self):
        for choice in ("Reject", "Edit"):
            with self.subTest(choice=choice):
                app = self.start_review()
                self.click(app, choice)
                self.assertEqual(app.session_state["final"]["final_action"], "rejected")
                self.assertTrue(app.warning)
                self.post_review_comment.assert_not_called()

    def test_escalation_requires_answers_then_posts_refined_review(self):
        app = self.start_review(0.4)
        self.assertIn("Strong escalation", app.subheader[0].value)
        self.assertTrue(app.error)
        self.click(app, "Submit answers")
        self.post_review_comment.assert_not_called()
        for field in app.text_input:
            if field.label in {"Why MD5 in auth.py?", "Is sync.py HTTPS?"}:
                field.set_value("Test context")
        refined = PRAnalysis(summary="Refined review", confidence=0.7, confidence_reasoning="Clarified")
        self.get_llm.return_value.with_structured_output.return_value.ainvoke.return_value = refined
        self.click(app, "Submit answers")
        self.assertEqual(app.session_state["final"]["analysis"], refined)
        self.assertTrue(app.success)
        self.post_review_comment.assert_called_once()
        self.assertIn("Reviewer answered escalation questions", self.post_review_comment.call_args.args[1])

    def test_auto_success_and_post_failure_are_distinct(self):
        app = self.start_review(0.8)
        self.assertTrue(app.success)
        self.assertTrue(any("80%" in metric.value for metric in app.metric))
        self.assertTrue(any("Review summary" in item.value for item in app.markdown))
        self.post_review_comment.side_effect = RuntimeError("Test failure")
        failed = self.start_review(0.8)
        self.assertFalse(failed.success)
        self.assertTrue(failed.error)


if __name__ == "__main__":
    unittest.main()
