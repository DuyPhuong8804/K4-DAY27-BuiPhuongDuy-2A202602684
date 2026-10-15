import unittest
from unittest.mock import patch

from langgraph.types import Command

from common.github import PullRequest
from common.schemas import PRAnalysis
from exercises import exercise_3_escalation as exercise


class EscalationTests(unittest.TestCase):
    def test_questions_resume_refine_and_post_once(self):
        pr = PullRequest(
            url="https://github.com/example/repo/pull/2",
            owner="example", repo="repo", number=2, title="Add login",
            author="reviewer", base_ref="main", head_ref="feature",
            head_sha="abcdef1234", diff="+ password_hash = md5(password)",
            files_changed=["auth.py"],
        )
        for questions in (["Why MD5 in auth.py?", "Is this production login?"], []):
            with self.subTest(questions=questions), patch.object(
                exercise, "fetch_pr", return_value=pr
            ), patch.object(exercise, "get_llm") as get_llm, patch.object(
                exercise, "post_review_comment"
            ) as post:
                initial = PRAnalysis(
                    summary="Login needs context", confidence=0.4,
                    confidence_reasoning="Password hashing requirements unknown",
                    risk_factors=["Weak password hashing"],
                    escalation_questions=questions,
                )
                refined = PRAnalysis(
                    summary="Replace MD5 before production", confidence=0.7,
                    confidence_reasoning="Reviewer clarified deployment intent",
                )
                llm = get_llm.return_value.with_structured_output.return_value
                llm.invoke.side_effect = [initial, refined]
                app = exercise.build_graph()
                cfg = {"configurable": {"thread_id": "escalation-test"}}
                paused = app.invoke({"pr_url": pr.url}, cfg)
                payload = paused["__interrupt__"][0].value
                expected_questions = questions or [
                    "What is the intent of this PR?", "Any migration concerns?",
                ]
                self.assertEqual(payload, {
                    "kind": "escalation", "pr_url": pr.url,
                    "confidence": initial.confidence,
                    "confidence_reasoning": initial.confidence_reasoning,
                    "summary": initial.summary, "risk_factors": initial.risk_factors,
                    "questions": expected_questions,
                })
                post.assert_not_called()
                self.assertEqual(llm.invoke.call_count, 1)
                answers = {q: "Context for: " + q for q in expected_questions}
                final = app.invoke(Command(resume=answers), cfg)
                self.assertEqual(final["escalation_answers"], answers)
                self.assertEqual(final["analysis"], refined)
                self.assertEqual(final["final_action"], "committed_after_escalation")
                self.assertEqual(app.get_state(cfg).next, ())
                synthesis_context = llm.invoke.call_args.args[0][1]["content"]
                for value in (pr.diff, initial.model_dump_json(), *answers, *answers.values()):
                    self.assertIn(value, synthesis_context)
                post.assert_called_once_with(pr.url, exercise._render_comment_body(final))
                body = post.call_args.args[1]
                self.assertIn(refined.summary, body)
                self.assertIn("Reviewer answered escalation questions", body)
                for question, answer in answers.items():
                    self.assertIn(question, body)
                    self.assertIn(answer, body)


if __name__ == "__main__":
    unittest.main()
