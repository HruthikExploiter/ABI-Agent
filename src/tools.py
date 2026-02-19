"""
src/tools.py
The core intelligence of the ABI Agent.
Three tools: PolarsAnalyst, SQLGenerator, VizGenerator.
Each tool generates code using an LLM and executes it.
If the code fails, it sends the error back to the LLM to fix — self-healing.
"""

from __future__ import annotations

import re
import textwrap
import traceback
from typing import Any

import duckdb
import polars as pl
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from src.utils.llm_factory import build_llm


# ── Helper Functions ─────────────────────────────────────────────────────────

def _extract_tag(text: str, tag: str) -> str:
    """Pull content from between XML tags in LLM response."""
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return ""
    return match.group(1).strip()


def _get_schema_info(lf: pl.LazyFrame) -> str:
    """Get column names and types without loading any data."""
    schema = lf.collect_schema()
    lines = [f"  - {col}: {dtype}" for col, dtype in schema.items()]
    return "Dataset columns:\n" + "\n".join(lines)


# ── PolarsAnalyst ─────────────────────────────────────────────────────────────

_POLARS_SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert Polars (Python) data analyst.

    STRICT RULES — follow every single one:
    1. The input LazyFrame is named `lf`. Never redefine it.
    2. Always use LazyFrame operations and call .collect() at the very end.
    3. Assign your final result to a variable named `result`.
    4. `result` must be a polars.DataFrame (not a LazyFrame).
    5. Do NOT import polars — it is already available as `pl`.
    6. Do NOT print anything.

    CRITICAL POLARS API RULES:
    - Use group_by() NOT groupby()
    - Use pl.col("column_name") to reference columns
    - Use .alias("new_name") to rename columns
    - Use .sort("column", descending=True) to sort
    - Chain everything: lf.filter(...).group_by(...).agg(...).sort(...).collect()

    CORRECT EXAMPLE:
    result = (
        lf
        .group_by(pl.col("category"))
        .agg(pl.col("total_revenue").sum().alias("revenue"))
        .sort("revenue", descending=True)
        .collect()
    )

    Output format — use these exact tags:
    <thinking>your reasoning here</thinking>
    <plan>your step by step plan</plan>
    <code>your python code here</code>
""").strip()


class PolarsAnalyst:
    """
    Takes a natural language question + LazyFrame.
    Asks LLM to write Polars code.
    Runs the code.
    If it fails, sends the error back to LLM to fix and retries.
    """

    def __init__(self) -> None:
        self._settings = None
        self._max_retries = 3

    def analyze(self, question: str, lf: pl.LazyFrame) -> dict[str, Any]:
        """
        Main method. Takes a question and LazyFrame, returns a result dict.

        Args:
            question: Natural language business question.
            lf:       Polars LazyFrame of the uploaded data.

        Returns:
            dict with keys: success, result_df, generated_code,
                            error, attempts
        """
        schema_info = _get_schema_info(lf)
        error_history: list[str] = []
        last_code = ""

        for attempt in range(1, self._max_retries + 2):
            # Switch to fallback model on last attempt
            use_fallback = attempt > self._max_retries
            llm = build_llm(use_fallback=use_fallback)

            logger.info(f"[PolarsAnalyst] Attempt {attempt}")

            user_message = self._build_prompt(
                question, schema_info, error_history
            )

            try:
                response = llm.invoke([
                    SystemMessage(content=_POLARS_SYSTEM_PROMPT),
                    HumanMessage(content=user_message),
                ])

                response_text = str(response.content)
                code = _extract_tag(response_text, "code")
                last_code = code

                if not code:
                    raise ValueError(
                        "LLM did not return a <code> block. "
                        "Raw response: " + response_text[:300]
                    )

                # Run the generated code
                result_df = self._run_code(code, lf)

                logger.success(
                    f"[PolarsAnalyst] Success on attempt {attempt}"
                )
                return {
                    "success": True,
                    "result_df": result_df,
                    "generated_code": code,
                    "error": None,
                    "attempts": attempt,
                }

            except Exception as exc:
                error_msg = (
                    f"Attempt {attempt} failed.\n"
                    f"Error type: {type(exc).__name__}\n"
                    f"Error message: {exc}\n"
                    f"Traceback:\n{traceback.format_exc()}"
                )
                logger.warning(f"[PolarsAnalyst] {error_msg}")
                error_history.append(error_msg)

                if attempt > self._max_retries:
                    logger.error("[PolarsAnalyst] All retries exhausted.")
                    return {
                        "success": False,
                        "result_df": None,
                        "generated_code": last_code,
                        "error": str(exc),
                        "attempts": attempt,
                    }

        return {
            "success": False,
            "result_df": None,
            "generated_code": last_code,
            "error": "Unexpected exit from retry loop.",
            "attempts": self._max_retries + 1,
        }

    @staticmethod
    def _build_prompt(
            question: str,
            schema_info: str,
            error_history: list[str],
    ) -> str:
        """Build the user prompt, injecting errors for self-healing."""
        base = textwrap.dedent(f"""
            {schema_info}

            Question: {question}

            Write Polars code to answer this question.
            Remember:
            - Input is `lf` (LazyFrame). Do not redefine it.
            - End with: result = lf.[...].collect()
            - result must be a polars.DataFrame
        """).strip()

        if not error_history:
            return base

        errors = "\n\n---\n".join(error_history)
        return textwrap.dedent(f"""
            {base}

            YOUR PREVIOUS CODE FAILED. Study these errors and fix them:

            {errors}

            Write corrected code now. Pay attention to:
            - group_by() not groupby()
            - .alias() for renaming columns
            - .collect() at the end
            - Correct column names from the schema above
        """).strip()

    @staticmethod
    def _run_code(code: str, lf: pl.LazyFrame) -> pl.DataFrame:
        """
        Execute LLM-generated code in a controlled namespace.
        Only pl and lf are available — nothing else.
        """
        namespace: dict[str, Any] = {
            "pl": pl,
            "lf": lf,
        }

        exec(code, namespace)  # noqa: S102

        result = namespace.get("result")

        if result is None:
            raise ValueError(
                "Code did not define a `result` variable. "
                "Make sure your code ends with: result = lf.[...].collect()"
            )

        if not isinstance(result, pl.DataFrame):
            raise ValueError(
                f"`result` must be a polars.DataFrame, "
                f"got {type(result).__name__}. "
                f"Did you forget .collect() at the end?"
            )

        return result


# ── SQLGenerator ──────────────────────────────────────────────────────────────

_SQL_SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert SQL analyst for supply chain data.
    Write DuckDB-compatible SQL against a table named `sc_data`.

    Rules:
    1. Output only valid DuckDB SQL.
    2. Use standard SQL: GROUP BY, CTEs, window functions are fine.
    3. Wrap your SQL in <sql> tags.
    4. Nothing outside the tags except <thinking> if needed.
""").strip()


class SQLGenerator:
    """
    Takes a natural language question and generates a validated SQL query.
    Runs the SQL against the actual data using DuckDB.
    """

    def generate(self, question: str, lf: pl.LazyFrame) -> dict[str, Any]:
        """
        Generate and run a SQL query for the given question.

        Args:
            question: Natural language business question.
            lf:       Polars LazyFrame of the data.

        Returns:
            dict with keys: success, sql_query, result_df, error
        """
        llm = build_llm()
        schema_info = _get_schema_info(lf)

        user_message = textwrap.dedent(f"""
            {schema_info}

            Table name in DuckDB: sc_data

            Question: {question}

            Write a DuckDB SQL query to answer this.
        """).strip()

        sql_query = ""

        try:
            response = llm.invoke([
                SystemMessage(content=_SQL_SYSTEM_PROMPT),
                HumanMessage(content=user_message),
            ])

            response_text = str(response.content)
            sql_query = _extract_tag(response_text, "sql")

            if not sql_query:
                raise ValueError(
                    "LLM did not return a <sql> block."
                )

            # Collect the LazyFrame so DuckDB can query it
            sc_data = lf.collect()  # noqa: F841 — used by DuckDB

            conn = duckdb.connect()
            result_df = conn.execute(sql_query).pl()
            conn.close()

            logger.success("[SQLGenerator] SQL executed successfully.")
            return {
                "success": True,
                "sql_query": sql_query,
                "result_df": result_df,
                "error": None,
            }

        except Exception as exc:
            logger.error(f"[SQLGenerator] Failed: {exc}")
            return {
                "success": False,
                "sql_query": sql_query,
                "result_df": None,
                "error": str(exc),
            }


# ── VizGenerator ──────────────────────────────────────────────────────────────

_VIZ_SYSTEM_PROMPT = textwrap.dedent("""
    You are a data visualization expert using Plotly in Python.

    Rules:
    1. The DataFrame is named `df` and is already in scope.
    2. `go` (plotly.graph_objects) is already imported.
    3. `px` (plotly.express) is already imported.
    4. Assign the final chart to a variable named `fig`.
    5. Use template="plotly_dark" for all charts.
    6. Always add a clear title and axis labels.
    7. Do NOT import anything.

    CORRECT EXAMPLES:

    Bar chart:
    fig = px.bar(
        df.to_pandas(),
        x="category",
        y="revenue",
        title="Revenue by Category",
        template="plotly_dark",
    )

    Line chart:
    fig = px.line(
        df.to_pandas(),
        x="order_date",
        y="total_revenue",
        title="Revenue Over Time",
        template="plotly_dark",
    )

    IMPORTANT:
    - Always call df.to_pandas() before passing to plotly
    - Use px (plotly.express) not go for simple charts
    - Column names must exactly match the DataFrame columns shown

    Output format:
    <thinking>reasoning</thinking>
    <plan>chart plan</plan>
    <code>python code here</code>
""").strip()


class VizGenerator:
    """
    Takes a natural language chart request and a DataFrame.
    Generates and executes Plotly code.
    Self-heals on errors just like PolarsAnalyst.
    """

    def __init__(self) -> None:
        self._max_retries = 3

    def generate(
            self, question: str, df: pl.DataFrame
    ) -> dict[str, Any]:
        """
        Generate a Plotly figure from a chart description.

        Args:
            question: Natural language chart request.
            df:       Polars DataFrame with the data to visualize.

        Returns:
            dict with keys: success, figure, generated_code,
                            error, attempts
        """
        import plotly.express as px
        import plotly.graph_objects as go

        col_info = ", ".join(
            f"{c}({t})" for c, t in df.collect_schema().items()
        )
        sample = df.head(3).to_dicts()

        base_prompt = textwrap.dedent(f"""
            Columns: {col_info}
            Sample rows: {sample}
            Chart request: {question}
            Write Plotly code. DataFrame is `df`. 
            Assign result to `fig`.
        """).strip()

        error_history: list[str] = []
        last_code = ""

        for attempt in range(1, self._max_retries + 2):
            use_fallback = attempt > self._max_retries
            llm = build_llm(use_fallback=use_fallback)

            if error_history:
                errors = "\n\n---\n".join(error_history)
                prompt = base_prompt + f"\n\nFix these errors:\n{errors}"
            else:
                prompt = base_prompt

            try:
                response = llm.invoke([
                    SystemMessage(content=_VIZ_SYSTEM_PROMPT),
                    HumanMessage(content=prompt),
                ])

                response_text = str(response.content)
                code = _extract_tag(response_text, "code")
                last_code = code

                if not code:
                    raise ValueError("LLM returned no <code> block.")

                namespace: dict[str, Any] = {
                    "df": df,
                    "go": go,
                    "px": px,
                    "pl": pl,
                }
                exec(code, namespace)  # noqa: S102

                fig = namespace.get("fig")
                if fig is None:
                    raise ValueError(
                        "Code did not define a `fig` variable."
                    )

                logger.success(
                    f"[VizGenerator] Chart created on attempt {attempt}"
                )
                return {
                    "success": True,
                    "figure": fig,
                    "generated_code": code,
                    "error": None,
                    "attempts": attempt,
                }

            except Exception as exc:
                error_msg = (
                    f"Attempt {attempt}: {type(exc).__name__}: {exc}\n"
                    f"{traceback.format_exc()}"
                )
                logger.warning(f"[VizGenerator] {error_msg}")
                error_history.append(error_msg)

                if attempt > self._max_retries:
                    return {
                        "success": False,
                        "figure": None,
                        "generated_code": last_code,
                        "error": str(exc),
                        "attempts": attempt,
                    }

        return {
            "success": False,
            "figure": None,
            "generated_code": last_code,
            "error": "Unexpected exit.",
            "attempts": self._max_retries + 1,
        }
