# 📦 ABI Agent — Autonomous Business Intelligence for Supply Chain

An AI-powered agent that takes a messy supply chain dataset, cleans it using Polars, and uses an Agentic Loop (ReAct) to answer complex business questions, generate visualizations, and self-correct its own code if it hits an error.

## App link
https://abi-agent.streamlit.app/

---

## 🎯 What It Does

Upload any supply chain CSV and ask questions in plain English:

- **"What are the top 5 suppliers by total revenue?"** → Polars analysis + data table
- **"Show a bar chart of revenue by product category"** → Auto-generated Plotly chart
- **"Generate SQL to find all delayed orders"** → Validated DuckDB SQL query
- **"Monthly revenue trend as a line chart"** → Time series visualization

If the generated code fails, the agent **automatically sends the error back to the LLM and fixes it** — no manual intervention needed.

---

## 🏗️ Architecture

```
User Question
     │
     ▼
┌─────────────┐
│   Planner   │  Classifies intent → analysis / sql / visualization
└──────┬──────┘
       │ (conditional routing via LangGraph)
       ▼
┌─────────────────────────────────────┐
│  Executor (tools.py)                │
│                                     │
│  PolarsAnalyst  → Polars code       │
│  SQLGenerator   → DuckDB SQL        │
│  VizGenerator   → Plotly chart      │
│                                     │
│  Each tool has a Self-Healing Loop: │
│  fail → send error to LLM → retry  │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────┐
│  Responder  │  Synthesizes plain English answer
└──────┬──────┘
       │
       ▼
  Streamlit UI
```

---

## 🛠️ Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Orchestration | LangGraph | Stateful multi-agent with deterministic routing |
| Data Engineering | Polars | 10-50x faster than Pandas, lazy evaluation |
| LLM | Google Gemini / Groq Llama | Model-agnostic via config |
| Visualization | Plotly | Interactive charts |
| Frontend | Streamlit | Rapid UI with chat interface |
| SQL Engine | DuckDB | In-process SQL on Polars DataFrames |
| Config | Pydantic Settings | Type-safe settings with env override |

---

## 📁 Project Structure

```
abi-agent/
│
├── app.py                      # Streamlit entry point
├── requirements.txt            # All dependencies
├── .env.example                # API key template
│
├── config/
│   └── config.yaml             # LLM provider, agent settings
│
├── src/
│   ├── config.py               # Pydantic settings loader
│   ├── tools.py                # PolarsAnalyst, SQLGenerator, VizGenerator
│   │
│   ├── agents/
│   │   ├── state.py            # Shared AgentState (TypedDict)
│   │   ├── planner.py          # Intent classification node
│   │   ├── executor.py         # Tool invocation nodes
│   │   └── graph.py            # LangGraph StateGraph assembly
│   │
│   └── utils/
│       └── llm_factory.py      # Model-agnostic LLM builder
│
├── data/
│   ├── raw/                    # Uploaded datasets
│   └── processed/              # Cleaned outputs
│
└── scripts/
    └── generate_demo_data.py   # Generates 5000-row demo CSV
```

---

## ⚡ Quickstart

### 1. Clone the repository
```bash
git clone https://github.com/HruthikExploiter/abi-agent.git
cd abi-agent
```

### 2. Create virtual environment
```bash
py -3.13.12 -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt --only-binary=:all:
```

### 4. Configure API keys
```bash
cp .env.example .env
```
Edit `.env` and add your keys:
```
GOOGLE_API_KEY=your_google_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Choose your LLM provider
Edit `config/config.yaml`:
```yaml
llm:
  provider: "groq"   # or "google"
```

### 6. Generate demo data
```bash
python scripts/generate_demo_data.py
```

### 7. Run the app
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 🔑 Getting API Keys

| Provider | Free Tier | Link |
|---|---|---|
| Google Gemini | 15 requests/min | [aistudio.google.com](https://aistudio.google.com) |
| Groq | 14,400 requests/day | [console.groq.com](https://console.groq.com) |
| LangSmith (optional) | Free tracing | [smith.langchain.com](https://smith.langchain.com) |

---

## 🔄 Switching LLM Provider

No code changes needed. Edit one line in `config/config.yaml`:

```yaml
# Use Groq (Llama 3.3)
provider: "groq"

# Use Google (Gemini 2.0)
provider: "google"
```

---

## 🧠 Self-Healing Code Execution

The most powerful feature. When the LLM generates broken Polars code:

```
Attempt 1: LLM writes code → exec() fails
           ↓ error + traceback sent back to LLM
Attempt 2: LLM fixes the bug → exec() succeeds ✓

If all retries fail → switches to fallback LLM model
```

This means the agent recovers from API changes, wrong column names,
and incorrect Polars syntax without any manual intervention.

---

## 📊 Demo Dataset

The demo CSV contains 5,000 rows of realistic supply chain data:

| Column | Type | Description |
|---|---|---|
| order_id | String | Unique order identifier |
| product_name | String | Product name |
| category | String | Product category |
| quantity | Integer | Units ordered |
| unit_price | Float | Price per unit |
| total_revenue | Float | quantity × unit_price |
| order_date | Date | When order was placed |
| lead_time_days | Integer | Days from order to delivery |
| supplier_name | String | Supplier company name |
| warehouse_id | String | Fulfillment warehouse |
| inventory_level | Integer | Current stock level |
| status | String | delivered/shipped/pending/returned/delayed |

---

## 🚀 Key Design Decisions

**Why Polars over Pandas?**
Polars uses lazy evaluation — for a 10GB CSV, only the rows/columns needed for the query are read from disk. Pandas would load the entire file into memory.

**Why LangGraph over a simple LLM call?**
LangGraph gives deterministic, stateful routing. The Planner decides once — then the right executor runs. No random tool loops or unpredictable agent behavior.

**Why model-agnostic design?**
Real companies switch LLM providers constantly based on cost and performance. A factory pattern with config-driven provider selection is the production pattern used in industry.

---

## 👨‍💻 Author

Hruthik Gajjala

**Skills demonstrated:**
- Agentic AI systems (LangGraph, ReAct pattern)
- High-performance data engineering (Polars, DuckDB)
- Production software patterns (factory pattern, pydantic settings, lazy evaluation)
- Full-stack AI application (Streamlit frontend + Python backend)
