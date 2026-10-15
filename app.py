"""Streamlit approval UI. Run: uv run streamlit run app.py"""

from __future__ import annotations

import asyncio
import uuid

import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from common.db import db_conn, db_path
from common.github import parse_pr_url
from exercises.exercise_4_audit import build_graph


def checkpoint_result(snapshot) -> dict:
    result = dict(snapshot.values)
    pending = [item for task in snapshot.tasks for item in task.interrupts]
    if pending:
        result["__interrupt__"] = pending
    return result


async def load_session(thread_id: str) -> dict:
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        graph = build_graph(cp)
        return checkpoint_result(await graph.aget_state({"configurable": {"thread_id": thread_id}}))


async def recent_sessions() -> list[dict]:
    async with db_conn() as conn:
        async with conn.execute("""
            SELECT thread_id, pr_url,
                   CASE MAX(CASE risk_level WHEN 'high' THEN 3 WHEN 'med' THEN 2 ELSE 1 END)
                       WHEN 3 THEN 'high' WHEN 2 THEN 'med' ELSE 'low' END AS worst_risk,
                   MAX(timestamp) AS last_event
            FROM audit_events
            GROUP BY thread_id, pr_url
            ORDER BY MAX(timestamp) DESC
            LIMIT 25
        """) as cursor:
            return [dict(row) for row in await cursor.fetchall()]


async def run_graph(pr_url: str, thread_id: str, resume_value=None):
    """Start or resume one graph turn using the same durable checkpoint."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        graph = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}
        snapshot = await graph.aget_state(cfg)
        if snapshot.values:
            if snapshot.values["pr_url"] != pr_url:
                raise ValueError("This thread belongs to a different PR URL")
            if not snapshot.next:
                return checkpoint_result(snapshot)
            if resume_value is None and any(task.interrupts for task in snapshot.tasks):
                return checkpoint_result(snapshot)
        if resume_value is not None:
            if not snapshot.values:
                raise ValueError("No saved session to resume")
            return await graph.ainvoke(Command(resume=resume_value), cfg)
        if snapshot.values:
            return await graph.ainvoke(None, cfg)
        return await graph.ainvoke({"pr_url": pr_url, "thread_id": thread_id}, cfg)


def save_result(result: dict) -> None:
    pending = result.get("__interrupt__")
    st.session_state.interrupt_payload = pending[0].value if pending else None
    st.session_state.final = None if pending else result


def open_session(thread_id: str) -> bool:
    result = asyncio.run(load_session(thread_id))
    if not result:
        st.warning("This session has no saved checkpoint.")
        return False
    st.session_state.thread_id = thread_id
    st.session_state.pr_url = result["pr_url"]
    st.session_state.pr_url_input = result["pr_url"]
    st.query_params["thread"] = thread_id
    save_result(result)
    return True


def render_approval_card(payload: dict) -> dict | None:
    st.subheader(f"Approval requested — confidence {payload['confidence']:.0%}")
    st.caption(payload["confidence_reasoning"])
    st.markdown(payload["summary"])
    for comment in payload.get("comments", []):
        st.markdown(
            f"- **[{comment['severity']}]** `{comment['file']}:{comment.get('line') or '?'}` — {comment['body']}"
        )
    with st.expander("Diff"):
        st.code(payload.get("diff_preview", ""), language="diff")
    feedback = st.text_input("Feedback (optional)", key=f"feedback_{st.session_state.thread_id}")
    cols = st.columns(3)
    for col, choice in zip(cols, ("approve", "reject", "edit")):
        if col.button(choice.title(), type="primary" if choice == "approve" else "secondary"):
            return {"choice": choice, "feedback": feedback}
    st.caption("Edit saves your feedback and ends this review without posting.")
    return None


def render_escalation_card(payload: dict) -> dict | None:
    st.subheader(f"Strong escalation — confidence {payload['confidence']:.0%}")
    st.caption(payload["confidence_reasoning"])
    if payload.get("risk_factors"):
        st.error("Risks:\n\n" + "\n".join(f"- {risk}" for risk in payload["risk_factors"]))
    st.markdown(payload["summary"])
    st.caption("Submitting answers refines the review and posts it to GitHub.")
    with st.form(f"escalation_{st.session_state.thread_id}"):
        answers = {
            question: st.text_input(question, key=f"answer_{st.session_state.thread_id}_{index}")
            for index, question in enumerate(payload["questions"])
        }
        submitted = st.form_submit_button("Submit answers", type="primary")
    if submitted:
        if not all(answer.strip() for answer in answers.values()):
            st.warning("Please answer every question before submitting.")
        else:
            return {question: answer.strip() for question, answer in answers.items()}
    return None


def render_final(final: dict) -> None:
    action = final.get("final_action")
    if action in {"committed", "auto_committed", "committed_after_escalation", "auto_approved"}:
        st.success("Review posted to GitHub.")
        st.link_button("View comments on GitHub", st.session_state.pr_url + "#discussion_bucket")
    elif action == "rejected":
        st.warning("Review ended — no comment posted.")
    elif action in {"commit_failed", "auto_commit_failed"}:
        st.error("Could not post the review. Check your GitHub token and repository access.")
    else:
        st.info("This session has not reached a final review outcome.")
    analysis = final.get("analysis")
    if analysis is not None:
        st.metric("Confidence", f"{analysis.confidence:.0%}")
        st.markdown(analysis.summary)
        with st.expander("Review details"):
            st.markdown(analysis.confidence_reasoning)
            for comment in analysis.comments:
                st.markdown(f"- **[{comment.severity}]** `{comment.file}:{comment.line or '?'}` — {comment.body}")


def main():
    load_dotenv()
    st.set_page_config(page_title="HITL PR Review", layout="wide")
    for key, default in {"thread_id": None, "pr_url": "", "pr_url_input": "", "interrupt_payload": None, "final": None}.items():
        if key not in st.session_state:
            st.session_state[key] = default
    st.title("HITL PR Review Agent")
    st.caption("Review a pull request, resolve uncertainty, and keep a trace of each decision.")

    requested_thread = st.query_params.get("thread")
    if requested_thread and requested_thread != st.session_state.thread_id:
        try:
            open_session(requested_thread)
        except Exception as error:
            st.error(f"Could not restore the session ({type(error).__name__}). Check the audit database.")
            st.stop()

    with st.sidebar:
        st.header("Recent sessions")
        try:
            rows = asyncio.run(recent_sessions())
            if rows:
                selected = st.dataframe(
                    rows, hide_index=True, on_select="rerun", selection_mode="single-row",
                    key=f"recent_{st.session_state.thread_id}",
                    column_config={"pr_url": st.column_config.LinkColumn("PR"), "thread_id": "Session"},
                )
                if selected.selection.rows:
                    row = rows[selected.selection.rows[0]]
                    if row["thread_id"] != st.session_state.thread_id and open_session(row["thread_id"]):
                        st.rerun()
            else:
                st.caption("No review sessions yet.")
        except Exception as error:
            st.error(f"Could not load sessions ({type(error).__name__}).")

    with st.form("start"):
        pr_url = st.text_input(
            "PR URL", key="pr_url_input",
            placeholder="https://github.com/VinUni-AI20k/PR-Demo/pull/1",
        )
        submitted = st.form_submit_button("Run review", type="primary")
    if submitted:
        pr_url = pr_url.strip()
        try:
            if not pr_url.startswith("https://github.com/"):
                raise ValueError("Use a GitHub HTTPS pull request URL")
            parse_pr_url(pr_url)
        except ValueError:
            st.error("Enter a PR URL such as https://github.com/owner/repo/pull/1.")
        else:
            st.session_state.pr_url = pr_url
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.interrupt_payload = None
            st.session_state.final = None
            st.query_params["thread"] = st.session_state.thread_id
            try:
                with st.spinner("Fetching PR + asking the LLM..."):
                    save_result(asyncio.run(run_graph(pr_url, st.session_state.thread_id)))
                st.rerun()
            except Exception as error:
                st.error(f"Could not run the review ({type(error).__name__}). Check your API credentials and connection.")

    payload = st.session_state.interrupt_payload
    if payload is not None:
        if payload["kind"] == "approval_request":
            answer = render_approval_card(payload)
        elif payload["kind"] == "escalation":
            answer = render_escalation_card(payload)
        else:
            st.error("This session contains an unsupported review request.")
            answer = None
        if answer is not None:
            try:
                with st.spinner("Resuming review..."):
                    save_result(asyncio.run(run_graph(
                        st.session_state.pr_url, st.session_state.thread_id, resume_value=answer,
                    )))
                st.rerun()
            except Exception as error:
                st.error(f"Could not continue the review ({type(error).__name__}). Your checkpoint is saved.")
    if st.session_state.final is not None:
        render_final(st.session_state.final)
    if st.session_state.thread_id:
        st.caption(f"thread_id = {st.session_state.thread_id}")
        st.code(f"uv run python -m audit.replay --thread {st.session_state.thread_id}", language="bash")


if __name__ == "__main__":
    main()
