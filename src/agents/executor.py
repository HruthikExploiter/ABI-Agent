"""
src/agents/executor.py
The Executor Nodes — each one calls a specific tool from src/tools.py
and writes the result back to AgentState.
Three executors: analysis, sql, visualization.
Plus the Responder that writes the final answer.
"""

from __future__ import annotations

import textwrap
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from loguru import logger

from src.agents.state import AgentState
from src.utils.llm_factory import build_llm


# ── Analysis Executor ────────────────────────────────────────────────────────

def analysis_executor_node(state: AgentState) -> dict[str, Any]:
    """
    Calls PolarsAnalyst to answer the question using Polars code.

    Reads:  state["user_question"], state["raw_dataframe"], state["plan"]
    Writes: state["analysis_result"], state["generated_code"],
            state["next_node"], state["error"]
    """
    from src.tools import PolarsAnalyst

    question = state["user_question"]
    lf = state.get("raw_dataframe")

    if lf is None:
        return {
            "error": "No dataset loaded. Please upload a CSV file first.",
            "next_node": "responder",
        }

    logger.info(f"[AnalysisExecutor] Running for: '{question}'")

    analyst = PolarsAnalyst()
    result = analyst.analyze(question, lf)

    if result["success"]:
        plan = state.get("plan", {})
        # If plan requires visualization, go to viz_node next
        next_node = "viz_node" if plan.get("requires_viz") else "responder"

        return {
            "analysis_result": result["result_df"],
            "generated_code": result["generated_code"],
            "error": None,
            "next_node": next_node,
            "messages": [
                AIMessage(
                    content=(
                        f"Analysis complete after "
                        f"{result['attempts']} attempt(s).\n"
                        f"Code used:\n{result['generated_code']}"
                    )
                )
            ],
        }
    else:
        logger.error(f"[AnalysisExecutor] Failed: {result['error']}")
        return {
            "error": result["error"],
            "generated_code": result["generated_code"],
            "next_node": "responder",
        }


# ── SQL Executor ─────────────────────────────────────────────────────────────

def sql_executor_node(state: AgentState) -> dict[str, Any]:
    """
    Calls SQLGenerator to produce and validate a SQL query.

    Reads:  state["user_question"], state["raw_dataframe"]
    Writes: state["sql_query"], state["sql_result"], state["error"]
    """
    from src.tools import SQLGenerator

    question = state["user_question"]
    lf = state.get("raw_dataframe")

    if lf is None:
        return {
            "error": "No dataset loaded.",
            "next_node": "responder",
        }

    logger.info(f"[SQLExecutor] Generating SQL for: '{question}'")

    generator = SQLGenerator()
    result = generator.generate(question, lf)

    if result["success"]:
        return {
            "sql_query": result["sql_query"],
            "sql_result": result["result_df"],
            "error": None,
            "next_node": "responder",
            "messages": [
                AIMessage(
                    content=f"SQL generated:\n```sql\n{result['sql_query']}\n```"
                )
            ],
        }
    else:
        return {
            "sql_query": result.get("sql_query", ""),
            "error": result["error"],
            "next_node": "responder",
        }


# ── Visualization Executor ────────────────────────────────────────────────────

def viz_executor_node(state: AgentState) -> dict[str, Any]:
    """
    Calls VizGenerator to create a Plotly chart.

    Reads:  state["analysis_result"], state["user_question"], state["plan"]
    Writes: state["figure"], state["error"]
    """
    from src.tools import VizGenerator

    question = state["user_question"]
    df = state.get("analysis_result")

    if df is None:
        lf = state.get("raw_dataframe")
        if lf is None:
            return {
                "error": "No data available for visualization.",
                "next_node": "responder",
            }
        df = lf.collect()

    plan = state.get("plan", {})
    chart_type = plan.get("chart_type", "")
    viz_question = (
        f"{chart_type} chart: {question}" if chart_type else question
    )

    logger.info(f"[VizExecutor] Generating chart for: '{viz_question}'")

    generator = VizGenerator()
    result = generator.generate(viz_question, df)

    if result["success"]:
        return {
            "figure": result["figure"],
            "error": None,
            "next_node": "responder",
            "messages": [
                AIMessage(
                    content=f"Chart created after {result['attempts']} attempt(s)."
                )
            ],
        }
    else:
        return {
            "error": result["error"],
            "next_node": "responder",
        }


# ── Responder Node ────────────────────────────────────────────────────────────

_RESPONDER_PROMPT = textwrap.dedent("""
    You are the final layer of a Supply Chain Business Intelligence Agent.
    Your job is to write a clear, friendly, plain English answer
    based on the computed results given to you.

    Rules:
    - Start with the direct answer to the question.
    - Highlight key numbers or trends.
    - If there was an error, explain it simply and suggest a fix.
    - Keep it under 150 words.
    - Never mention code, LazyFrames, or technical terms.
""").strip()


def responder_node(state: AgentState) -> dict[str, Any]:
    """
    Reads all results from state and writes a plain English final answer.

    Reads:  state["user_question"], state["analysis_result"],
            state["sql_result"], state["error"]
    Writes: state["final_answer"]
    """
    llm = build_llm()
    question = state["user_question"]
    context_parts = [f"User Question: {question}"]

    if error := state.get("error"):
        context_parts.append(f"Error: {error}")

    if (df := state.get("analysis_result")) is not None:
        preview = df.head(10).to_pandas().to_string(index=False)
        context_parts.append(f"Analysis Result:\n{preview}")

    if sql := state.get("sql_query"):
        context_parts.append(f"SQL Query:\n{sql}")

    if (sql_df := state.get("sql_result")) is not None:
        preview = sql_df.head(10).to_pandas().to_string(index=False)
        context_parts.append(f"SQL Result:\n{preview}")

    if state.get("figure") is not None:
        context_parts.append("A chart has been generated and displayed.")

    context = "\n\n".join(context_parts)

    logger.info("[Responder] Writing final answer.")

    response = llm.invoke([
        SystemMessage(content=_RESPONDER_PROMPT),
        HumanMessage(content=context),
    ])

    final_answer = str(response.content)

    return {
        "final_answer": final_answer,
        "next_node": "end",
        "messages": [AIMessage(content=final_answer)],
    }