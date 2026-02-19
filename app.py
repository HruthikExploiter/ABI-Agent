"""
app.py
The Streamlit web interface for the ABI Agent.
Run with: streamlit run app.py
"""

from __future__ import annotations

import tempfile

from dotenv import load_dotenv
load_dotenv()

import polars as pl
import streamlit as st

from src.config import get_settings

# ── Page Setup ────────────────────────────────────────────────────────────────

cfg = get_settings()

st.set_page_config(
    page_title=cfg.streamlit.page_title,
    page_icon=cfg.streamlit.page_icon,
    layout=cfg.streamlit.layout,
)

# ── Session State ─────────────────────────────────────────────────────────────

if "lf" not in st.session_state:
    st.session_state.lf = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📦 ABI Agent")
    st.divider()

    # Step 1 — Upload first (most important)
    st.markdown("### 📁 Upload Dataset")
    uploaded = st.file_uploader(
        "CSV file",
        type=["csv"],
        help="Upload your supply chain CSV file.",
    )

    if uploaded:
        with st.spinner("Loading dataset..."):
            try:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".csv"
                ) as tmp:
                    tmp.write(uploaded.getvalue())
                    tmp_path = tmp.name

                st.session_state.lf = pl.scan_csv(tmp_path)
                schema = st.session_state.lf.collect_schema()

                st.success(f"✓ Loaded — {len(schema)} columns")

                with st.expander("View Columns"):
                    for col, dtype in schema.items():
                        st.text(f"  {col}: {dtype}")

            except Exception as e:
                st.error(f"Error loading file: {e}")

    st.divider()

    # Step 2 — Example questions right below upload
    st.markdown("### 💡 Try asking:")
    examples = [
        "Top 5 suppliers by total revenue?",
        "Bar chart of revenue by category",
        "Products with highest lead time?",
        "SQL for all delayed orders",
        "Total inventory value by warehouse",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["pending"] = ex

    st.divider()

    # LLM info at bottom
    st.markdown("### ⚙️ LLM Provider")
    st.info(
        f"**{cfg.llm.provider.title()}**\n\n"
        f"`{cfg.llm.active_primary_model}`"
    )

    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# ── Main Area ─────────────────────────────────────────────────────────────────

st.markdown("# 📦 Autonomous business intelligence agent")
st.markdown("#### Supply Chain Intelligence Platform")

# Show chat history
for i, msg in enumerate(st.session_state.chat_history):
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("figure"):
            st.plotly_chart(
                msg["figure"],
                use_container_width=True,
                key=f"fig_{i}",
            )
        if msg.get("df") is not None:
            with st.expander("View Data", expanded=False):
                st.dataframe(
                    msg["df"].to_pandas(),
                    use_container_width=True,
                    key=f"df_{i}",
                )
        if msg.get("code"):
            with st.expander("Generated Polars Code", expanded=False):
                st.code(msg["code"], language="python")
        if msg.get("sql"):
            with st.expander("Generated SQL", expanded=False):
                st.code(msg["sql"], language="sql")

# ── Chat Input ────────────────────────────────────────────────────────────────

question = st.chat_input(
    "Ask a supply chain question...",
    disabled=st.session_state.lf is None,
)

# Handle sidebar button clicks — only if no typed question
if not question and "pending" in st.session_state:
    question = st.session_state.pop("pending")

if not question and st.session_state.lf is None:
    st.info("⬅️ Upload a CSV file from the sidebar to get started.")

if question:
    if st.session_state.lf is None:
        st.warning("Please upload a CSV file first.")
        st.stop()

    # Show user message immediately
    with st.chat_message("user"):
        st.write(question)
    st.session_state.chat_history.append({
        "role": "user",
        "content": question,
    })

    # Run the agent and show response
    with st.chat_message("assistant"):
        with st.spinner("Agent thinking..."):
            try:
                from src.agents.graph import abi_graph

                initial_state = {
                    "messages": [],
                    "user_question": question,
                    "raw_dataframe": st.session_state.lf,
                    "plan": {},
                    "next_node": "",
                    "analysis_result": None,
                    "generated_code": "",
                    "sql_query": "",
                    "sql_result": None,
                    "figure": None,
                    "error": None,
                    "retry_count": 0,
                    "final_answer": "",
                }

                result = abi_graph.invoke(initial_state)

                # Display final answer
                answer = result.get("final_answer", "No answer generated.")
                st.write(answer)

                # Display chart if generated
                if fig := result.get("figure"):
                    st.plotly_chart(fig, use_container_width=True)

                # Display data tables
                analysis_df = result.get("analysis_result")
                if analysis_df is not None:
                    with st.expander("View Data Table"):
                        st.dataframe(
                            analysis_df.to_pandas(),
                            use_container_width=True,
                        )

                sql_df = result.get("sql_result")
                if sql_df is not None:
                    with st.expander("View SQL Result"):
                        st.dataframe(
                            sql_df.to_pandas(),
                            use_container_width=True,
                        )

                # Display generated code
                if code := result.get("generated_code"):
                    with st.expander("Generated Polars Code"):
                        st.code(code, language="python")

                if sql := result.get("sql_query"):
                    with st.expander("Generated SQL"):
                        st.code(sql, language="sql")

                # Save assistant response to history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "figure": result.get("figure"),
                    "df": (
                        analysis_df
                        if analysis_df is not None
                        else sql_df
                    ),
                    "code": result.get("generated_code"),
                    "sql": result.get("sql_query"),
                })

            except Exception as e:
                st.error(f"Agent error: {e}")
                st.exception(e)
