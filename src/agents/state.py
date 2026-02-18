"""
src/agents/state.py
Defines the shared state object that all agents read from and write to.
Think of it as the shared whiteboard for the entire agent system.
"""

from __future__ import annotations

from typing import Annotated, Any
import polars as pl
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """
    Central state shared across all LangGraph nodes.
    Every agent reads from this and writes back to this.
    Nothing is passed directly between agents.
    """

    # The full conversation message history
    # add_messages means new messages are APPENDED not overwritten
    messages: Annotated[list[BaseMessage], add_messages]

    # What the user asked
    user_question: str

    # The uploaded CSV as a Polars LazyFrame (no data loaded yet)
    raw_dataframe: Any  # pl.LazyFrame — using Any to avoid TypedDict issues

    # Written by Planner, read by Executor
    plan: dict[str, Any]

    # Routing signal — tells LangGraph which node to go to next
    next_node: str

    # Written by Analysis Executor
    analysis_result: Any  # pl.DataFrame or None

    # The Polars code that was generated and ran
    generated_code: str

    # Written by SQL Executor
    sql_query: str
    sql_result: Any  # pl.DataFrame or None

    # Written by Viz Executor
    figure: Any  # plotly Figure or None

    # Error message if something went wrong
    error: Any  # str or None

    # How many self-healing retries have happened
    retry_count: int

    # The final plain English answer shown to the user
    final_answer: str