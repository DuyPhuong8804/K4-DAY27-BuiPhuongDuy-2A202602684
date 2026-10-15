import os
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import AsyncMock, patch

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from rich.console import Console

from common.db import db_conn, replay_events
from common.github import PullRequest
from common.schemas import PRAnalysis, risk_level_for
from audit import replay
from exercises import exercise_4_audit as exercise


class AuditTests(unittest.IsolatedAsyncioTestCase):
    async def test_all_routes_persist_events_and_resume_after_reopening_database(self):
        pr = PullRequest(
            url="https://github.com/example/repo/pull/1",
            owner="example", repo="repo", number=1, title="Example PR",
            author="reviewer", base_ref="main", head_ref="feature",
            head_sha="abcdef1234", diff="+ example", files_changed=["example.py"],
        )
        cases = [
            (0.8, None, False, "auto_committed"),
            (0.8, None, True, "auto_commit_failed"),
            (0.65, "approve", False, "committed"),
            (0.65, "approve", True, "commit_failed"),
            (0.65, "reject", False, "rejected"),
            (0.65, "edit", False, "rejected"),
            (0.4, "escalate", False, "committed"),
        ]
        Path("scratch").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir="scratch") as temp:
            path = str(Path(temp) / "audit.db")
            with patch.dict(os.environ, {"HITL_DB_PATH": path, "GITHUB_USER": "test-reviewer"}):
                for index, (confidence, choice, fail, expected_action) in enumerate(cases):
                    with self.subTest(choice=choice, fail=fail), patch.object(
                        exercise, "fetch_pr", return_value=pr
                    ), patch.object(exercise, "get_llm") as get_llm, patch.object(
                        exercise, "post_review_comment", side_effect=RuntimeError("Test post failure") if fail else None
                    ) as post:
                        initial = PRAnalysis(
                            summary="Initial review", confidence=confidence,
                            confidence_reasoning="Needs context",
                            escalation_questions=["Why this change in example.py?"],
                        )
                        refined = PRAnalysis(
                            summary="Refined review", confidence=0.7,
                            confidence_reasoning="Context clarified",
                        )
                        llm = get_llm.return_value.with_structured_output.return_value
                        llm.ainvoke = AsyncMock(side_effect=[initial, refined])
                        thread = f"test-{index}"
                        cfg = {"configurable": {"thread_id": thread}}
                        async with AsyncSqliteSaver.from_conn_string(path) as cp:
                            await cp.setup()
                            result = await exercise.build_graph(cp).ainvoke(
                                {"pr_url": pr.url, "thread_id": thread}, cfg,
                            )
                        if choice:
                            post.assert_not_called()
                            payload = result["__interrupt__"][0].value
                            answer = (
                                {q: "Test context" for q in payload["questions"]}
                                if choice == "escalate" else {"choice": choice, "feedback": "Test feedback"}
                            )
                            if choice == "reject":
                                with patch.object(exercise, "handle_interrupt", return_value=answer):
                                    await exercise.run(pr.url, thread)
                                async with AsyncSqliteSaver.from_conn_string(path) as cp:
                                    result = (await exercise.build_graph(cp).aget_state(cfg)).values
                                self.assertEqual(llm.ainvoke.call_count, 1)
                            else:
                                async with AsyncSqliteSaver.from_conn_string(path) as cp:
                                    app = exercise.build_graph(cp)
                                    self.assertTrue((await app.aget_state(cfg)).next)
                                    result = await app.ainvoke(Command(resume=answer), cfg)
                        self.assertEqual(result["final_action"], expected_action)
                        self.assertEqual(post.call_count, int(choice not in {"reject", "edit"}))
                        events = await replay_events(thread)
                        actions = [e["action"] for e in events]
                        expected = ["fetch_pr", "analyze", "route"]
                        if choice:
                            human_node = "escalate" if choice == "escalate" else "human_approval"
                            expected += [human_node] * 3
                            expected += ["synthesize", "commit"] if choice == "escalate" else ["commit"]
                            self.assertEqual(events[5]["reviewer_id"], "test-reviewer")
                            self.assertEqual(events[5]["decision"], choice)
                            self.assertIn("Test context" if choice == "escalate" else "Test feedback", events[5]["reason"])
                            self.assertTrue(all(e["reviewer_id"] is None for e in events[:5]))
                        else:
                            expected += ["auto_approve"]
                        self.assertEqual(actions, expected)
                        self.assertEqual(events[0]["confidence"], 0)
                        self.assertEqual(events[0]["risk_level"], "med")
                        self.assertEqual(events[1]["decision"], "pending")
                        self.assertEqual(events[2]["decision"], "auto" if choice is None else "escalate" if choice == "escalate" else "pending")
                        for event in events:
                            self.assertIn(event["decision"], {"auto", "approve", "reject", "edit", "escalate", "pending"})
                            self.assertEqual(event["agent_id"], exercise.AGENT_ID)
                            self.assertGreaterEqual(event["execution_time_ms"], 0)
                            self.assertTrue(event["timestamp"])
                            if event["action"] != "fetch_pr":
                                self.assertEqual(event["risk_level"], risk_level_for(event["confidence"]))
                        if choice == "escalate":
                            self.assertEqual(events[-2]["confidence"], refined.confidence)
                            self.assertEqual(events[-1]["confidence"], refined.confidence)
                        self.assertIn("commit_failed" if fail else expected_action, events[-1]["reason"])
                async with db_conn() as conn:
                    async with conn.execute("SELECT COUNT(*) FROM audit_events WHERE risk_level='high' AND decision='auto'") as cursor:
                        self.assertEqual((await cursor.fetchone())[0], 0)
                output = StringIO()
                with patch.object(replay, "Console", return_value=Console(file=output, width=300)):
                    await replay.list_threads()
                escalation_row = next(line for line in output.getvalue().splitlines() if "test-6" in line)
                self.assertIn("high", escalation_row)


if __name__ == "__main__":
    unittest.main()
