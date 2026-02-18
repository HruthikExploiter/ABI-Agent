"""
src/agents/graph.py
Assembles all agent nodes into a LangGraph StateGraph.
This is the flowchart of the entire system — who runs when.

Flow:
  START → planner → executor or sql_node or viz_node → responder → END
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from src.agents.executor import (
    analysis_executor_node,
    responder_node,
    sql_executor_node,
    viz_executor_node,
)
from src.agents.planner import planner_node
from src.agents.state import AgentState


def _route_after_planner(state: AgentState) -> str:
    """
    Reads state["next_node"] after planner runs.
    Returns the name of the next node to execute.
    """
    route = state.get("next_node", "executor")
    valid = {"executor", "sql_node", "viz_node", "responder"}
    if route not in valid:
        return "executor"
    return route


def _route_after_executor(state: AgentState) -> str:
    """
    After analysis executor runs, go to viz_node or responder.
    Depends on whether the plan required a visualization.
    """
    route = state.get("next_node", "responder")
    if route == "viz_node":
        return "viz_node"
    return "responder"


def build_graph():
    """
    Builds and compiles the full ABI Agent LangGraph.

    Returns:
        A compiled LangGraph ready for .invoke() calls.

    Usage:
        graph = build_graph()
        result = graph.invoke({
            "user_question": "Top 5 suppliers by revenue?",
            "raw_dataframe": lf,
            "messages": [],
            "retry_count": 0,
            "plan": {},
            "next_node": "",
            "analysis_result": None,
            "generated_code": "",
            "sql_query": "",
            "sql_result": None,
            "figure": None,
            "error": None,
            "final_answer": "",
        })
    """
    builder = StateGraph(AgentState)

    # Register all nodes
    builder.add_node("planner", planner_node)
    builder.add_node("executor", analysis_executor_node)
    builder.add_node("sql_node", sql_executor_node)
    builder.add_node("viz_node", viz_executor_node)
    builder.add_node("responder", responder_node)

    # Entry point — always start at planner
    builder.add_edge(START, "planner")

    # After planner — route based on intent
    builder.add_conditional_edges(
        "planner",
        _route_after_planner,
        {
            "executor": "executor",
            "sql_node": "sql_node",
            "viz_node": "viz_node",
            "responder": "responder",
        },
    )

    # After analysis executor — go to viz or straight to responder
    builder.add_conditional_edges(
        "executor",
        _route_after_executor,
        {
            "viz_node": "viz_node",
            "responder": "responder",
        },
    )

    # SQL and Viz always go straight to responder
    builder.add_edge("sql_node", "responder")
    builder.add_edge("viz_node", "responder")

    # Responder is always the last stop
    builder.add_edge("responder", END)

    return builder.compile()


# Compiled graph — import this directly in app.py and tools.py
abi_graph = build_graph()