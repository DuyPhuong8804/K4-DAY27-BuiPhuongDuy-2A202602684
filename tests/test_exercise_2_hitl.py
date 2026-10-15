import unittest
from unittest.mock import patch

from langgraph.types import Command

from common.github import PullRequest
from common.schemas import PRAnalysis, ReviewComment
from exercises import exercise_2_hitl as exercise


class HumanApprovalTests(unittest.TestCase):
    def test_pause_then_resume_posts_only_for_approve(self):
        pr = PullRequest(
            url="https://github.com/example/repo/pull/1",
            owner="example", repo="repo", number=1, title="Example PR",
            author="reviewer", base_ref="main", head_ref="feature",
            head_sha="abcdef1234", diff="+ example\n" * 300,
            files_changed=["example.py"],
        )
        analysis = PRAnalysis(
            summary="Review summary", confidence=0.65,
            confidence_reasoning="Migration needs confirmation",
            comments=[ReviewComment(
                file="example.py", line=1, severity="issue", body="Check migration",
            )],
        )
        for choice in ("approve", "reject", "edit"):
            with self.subTest(choice=choice), patch.object(
                exercise, "fetch_pr", return_value=pr
            ), patch.object(exercise, "get_llm") as get_llm, patch.object(
                exercise, "post_review_comment"
            ) as post:
                get_llm.return_value.with_structured_output.return_value.invoke.return_value = analysis
                app = exercise.build_graph()
                cfg = {"configurable": {"thread_id": choice}}
                paused = app.invoke({"pr_url": pr.url, "thread_id": choice}, cfg)
                post.assert_not_called()
                self.assertNotIn("final_action", paused)
                self.assertEqual(paused["__interrupt__"][0].value, {
                    "kind": "approval_request", "confidence": 0.65,
                    "confidence_reasoning": analysis.confidence_reasoning,
                    "summary": analysis.summary,
                    "comments": [c.model_dump() for c in analysis.comments],
                    "diff_preview": pr.diff[:2000],
                })
                self.assertEqual(app.get_state(cfg).next, ("human_approval",))
                final = app.invoke(Command(resume={
                    "choice": choice, "feedback": "Reviewer feedback",
                }), cfg)
                self.assertEqual(final["human_choice"], choice)
                self.assertEqual(final["human_feedback"], "Reviewer feedback")
                self.assertEqual(app.get_state(cfg).next, ())
                if choice == "approve":
                    self.assertEqual(final["final_action"], "committed")
                    post.assert_called_once_with(pr.url, exercise._render_comment_body(final))
                else:
                    self.assertEqual(final["final_action"], "rejected")
                    post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
