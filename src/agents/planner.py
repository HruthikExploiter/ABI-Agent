"""
src/agents/planner.py
The Planner Node — reads the user question and decides
what needs to be done and which executor should handle it.
Runs FIRST before any data analysis happens.
"""

from __future__ import annotations

import json
import re
import textwrap
from typing import Any

import polars as pl
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from loguru import logger

from src.agents.state import AgentState
from src.utils.llm_factory import build_llm

_PLANNER_PROMPT = textwrap.dedent("""
    You are the Planner for a Supply Chain Business Intelligence Agent.

    Your only job is to read the user's question and return a JSON plan.
    Output ONLY valid JSON inside <plan> tags. Nothing else outside the tags.

    JSON format:
    {
        "intent": "analysis or sql or visualization or multi",
        "primary_task": "short description of main task",
        "requires_sql": true or false,
        "requires_viz": true or false,
        "chart_type": "bar or line or pie or scatter or null",
        "complexity": "low or medium or high",
        "routing": "executor or sql_node or viz_node or multi"
    }

    Rules for routing:
    - "executor"  → user wants data analysis only
    - "sql_node"  → user explicitly wants SQL query
    - "viz_node"  → user wants a chart only
    - "multi"     → user wants analysis AND a chart (most common)
""").strip()


def planner_node(state: AgentState) -> dict[str, Any]:
    """
    Planner Node for LangGraph.

    Reads:  state["user_question"], state["raw_dataframe"]
    Writes: state["plan"], state["next_node"], state["messages"]
    """
    llm = build_llm()
    question = state["user_question"]
    lf = state.get("raw_dataframe")

    # Get column names only — no data is loaded here
    schema_info = ""
    if lf is not None:
        cols = "\n".join(
            f"  - {col}: {dtype}"
            for col, dtype in lf.collect_schema().items()
        )
        schema_info = f"\nDataset columns:\n{cols}"

    user_prompt = f"User Question: {question}{schema_info}"

    logger.info(f"[Planner] Planning for: '{question}'")

    response = llm.invoke([
        SystemMessage(content=_PLANNER_PROMPT),
        HumanMessage(content=user_prompt),
    ])

    response_text = str(response.content)

    # Extract JSON from <plan> tags
    match = re.search(r"<plan>(.*?)</plan>", response_text, re.DOTALL)
    raw_json = match.group(1).strip() if match else response_text.strip()

    try:
        plan = json.loads(raw_json)
    except json.JSONDecodeError:
        logger.warning("[Planner] Could not parse JSON — using safe default.")
        plan = {
            "intent": "analysis",
            "primary_task": question,
            "requires_sql": False,
            "requires_viz": False,
            "chart_type": None,
            "complexity": "low",
            "routing": "executor",
        }

    next_node = plan.get("routing", "executor")
    # "multi" means analysis first, then viz
    if next_node == "multi":
        next_node = "executor"

    logger.info(f"[Planner] Intent={plan.get('intent')} | Route={next_node}")

    return {
        "plan": plan,
        "next_node": next_node,
        "messages": [
            AIMessage(content=f"Plan created: {json.dumps(plan, indent=2)}")
        ],
    }