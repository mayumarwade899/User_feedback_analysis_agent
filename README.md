# 🧠 User Feedback Analysis Agent

> An intelligent multi-agent pipeline that automatically classifies, analyzes, and converts raw user feedback into structured, actionable tickets — powered by **LangGraph**, **Google Gemini**, **FastAPI**, and **Streamlit**.

## 📖 Overview

Processing thousands of user reviews and support emails manually is slow, inconsistent, and expensive. This project automates that entire workflow using a **LangGraph multi-agent system** that:

1. **Ingests** raw feedback from CSV sources (reviews, emails)
2. **Classifies** each item by type — Bug Report, Feature Request, Complaint, Praise, or General
3. **Extracts deep insights** using specialized AI agents (bug reproduction steps, feature impact, etc.)
4. **Builds structured tickets** with title, description, category, and priority
5. **Validates ticket quality** via a dedicated QA agent
6. **Outputs** clean CSVs and a live Streamlit dashboard with metrics

---

## 🏗️ Architecture

```
Input CSVs (reviews, emails)
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│              LangGraph Feedback Pipeline                 │
│                                                         │
│  ┌──────────────────────┐                               │
│  │ FeedbackClassifier   │ ── classifies category        │
│  │       Agent          │    & priority                 │
│  └──────────┬───────────┘                               │
│             │  conditional routing                      │
│      ┌──────┴──────┐                                    │
│      ▼             ▼                                    │
│  ┌──────────┐  ┌──────────────┐                         │
│  │   Bug    │  │   Feature    │  (routed by category)   │
│  │ Insights │  │   Insights   │                         │
│  │  Agent   │  │    Agent     │                         │
│  └────┬─────┘  └──────┬───────┘                         │
│       └────────┬───────┘                                │
│                ▼                                        │
│       ┌─────────────────┐                               │
│       │  TicketBuilder  │ ── builds structured ticket   │
│       │     Agent       │                               │
│       └────────┬────────┘                               │
│                ▼                                        │
│       ┌─────────────────┐                               │
│       │  TicketQuality  │ ── validates ticket quality   │
│       │     Agent       │                               │
│       └────────┬────────┘                               │
└────────────────┼────────────────────────────────────────┘
                 │
        ┌────────▼────────┐
        │  Output CSVs    │    ┌─────────────────────┐
        │ + Metrics + Log │───▶│ Streamlit Dashboard  │
        └─────────────────┘    └─────────────────────┘
                               ┌─────────────────────┐
                          ───▶ │  FastAPI REST API    │
                               └─────────────────────┘
```

---

## 🤖 Agent Breakdown

| Agent | Role | LLM Used |
|---|---|---|
| `FeedbackClassifierAgent` | Classifies feedback into category + priority | Gemini |
| `BugInsightsAgent` | Extracts bug severity, reproduction steps, affected components | Gemini |
| `FeatureInsightsAgent` | Extracts business value, user impact, effort estimate | Gemini |
| `TicketBuilderAgent` | Constructs a structured ticket from all upstream data | Rule-based |
| `TicketQualityAgent` | Validates completeness, clarity, and quality of each ticket | Gemini |

---

## 🧩 Tech Stack

| Layer | Technology |
|---|---|
| **Agent Orchestration** | LangGraph 1.0 (StateGraph with conditional routing) |
| **LLM** | Google Gemini via `langchain-google-genai` |
| **Backend API** | FastAPI + Uvicorn |
| **Dashboard** | Streamlit 1.52 |
| **Data Processing** | Pandas, NumPy, scikit-learn |
| **Configuration** | YAML + Pydantic dataclasses |
| **Environment** | python-dotenv |
| **Testing** | pytest |

---

## 📂 Project Structure

```
User_feedback_analysis_agent/
│
├── data/                                   # Input data sources
│   ├── emails.csv                          # Raw support email feedback
│   ├── reviews.csv                         # Raw product reviews
│   └── expected_classification.csv         # Ground truth for evaluation
│
├── outputs/                                # Auto-generated pipeline outputs
│   ├── generated_tickets.csv               # Structured tickets
│   ├── metrics.csv                         # Pipeline performance metrics
│   └── processing_log.csv                  # Per-record processing log
│
├── services/                               # REST API layer
│   ├── __init__.py
│   └── api.py                              # FastAPI endpoints
│
├── src/feedback_automation/                # Core package
│   ├── agents/                             # Individual agent implementations
│   │   ├── __init__.py
│   │   ├── base.py                         # Base agent class
│   │   ├── feedback_classifier.py          # Classification agent
│   │   ├── bug_insights_agent.py           # Bug analysis agent
│   │   ├── feature_insights_agent.py       # Feature analysis agent
│   │   ├── ticket_builder_agent.py         # Ticket construction agent
│   │   └── ticket_quality_agent.py         # Quality validation agent
│   ├── __init__.py
│   ├── config.py                           # ApplicationConfig (YAML-backed)
│   ├── graph.py                            # LangGraph StateGraph definition
│   ├── llm.py                              # LLM initialization & JSON parsing
│   ├── schemas.py                          # Pydantic models & TypedDict state
│   └── utils.py                            # Shared utilities
│
├── app.py                                  # Streamlit dashboard entry point
├── requirements.txt                        # Pinned dependencies
├── .env.example                            # Environment variable template
└── README.md
```

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.11+
- A [Google Gemini API key](https://ai.google.dev/) (free tier available)

### 1. Clone the repository

```bash
git clone https://github.com/mayumarwade899/User_feedback_analysis_agent.git
cd User_feedback_analysis_agent
```

### 2. Create a virtual environment

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in your values:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
MODEL_NAME=gemini-1.5-flash
```

---

## 🚀 Running the Project

### Option A — Streamlit Dashboard (Recommended)

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

- Click **"Execute Pipeline"** to process all feedback from `data/`
- View generated tickets, metrics, and processing logs in real time

### Option B — FastAPI REST API

```bash
uvicorn services.api:app --reload
```

API will be available at [http://localhost:8000](http://localhost:8000)

Interactive docs at [http://localhost:8000/docs](http://localhost:8000/docs)

**Example request:**

```bash
curl -X POST http://localhost:8000/process \
  -H "Content-Type: application/json" \
  -d '{
    "source_id": "review_001",
    "source_type": "review",
    "content": "The app crashes every time I try to upload a file larger than 10MB."
  }'
```

---

## 📊 Data Formats

### Input (`data/reviews.csv`, `data/emails.csv`)

| Column | Type | Description |
|---|---|---|
| `source_id` | string | Unique identifier for the feedback item |
| `source_type` | string | `review` or `email` |
| `content` | string | Raw feedback text |

### Output (`outputs/generated_tickets.csv`)

| Column | Description |
|---|---|
| `ticket_id` | Auto-generated unique ticket ID |
| `title` | AI-generated concise ticket title |
| `description` | Full ticket description with context |
| `category` | `BUG`, `FEATURE`, `COMPLAINT`, `PRAISE`, or `GENERAL` |
| `priority` | `HIGH`, `MEDIUM`, or `LOW` |
| `source_id` | Reference back to original feedback |
| `metadata` | JSON blob with agent-extracted insights |

---

## 🧠 How the LangGraph Pipeline Works

The pipeline uses a **conditional StateGraph** to route each feedback record through the right agents:

```python
graph.set_entry_point("classify")
graph.add_conditional_edges(
    "classify",
    route_after_classification,
    {
        "bug":     "bug_insights",
        "feature": "feature_insights",
        "other":   "ticket_builder",   # skips insight agents
    }
)
graph.add_edge("bug_insights",     "ticket_builder")
graph.add_edge("feature_insights", "ticket_builder")
graph.add_edge("ticket_builder",   "ticket_quality")
graph.add_edge("ticket_quality",   END)
```

**State** (`GraphState` TypedDict) flows through every node, accumulating:
- `record` — the original `FeedbackRecord`
- `classification` — output of `FeedbackClassifierAgent`
- `bug_insights` / `feature_insights` — output of specialist agents
- `ticket` — built by `TicketBuilderAgent`
- `quality_assessment` — final validation from `TicketQualityAgent`

---

## 🧪 Running Tests

```bash
pytest
```

---

## 📈 Example Output

**Input:**
```
"The search feature is completely broken on mobile. It returns no results 
for any query. This started after the v2.1.0 update."
```

**Generated Ticket:**
```json
{
  "ticket_id": "BUG-review_042-1a2b3c",
  "title": "Search returns no results on mobile after v2.1.0 update",
  "category": "BUG",
  "priority": "HIGH",
  "description": "Users report search functionality returning zero results on mobile devices following the v2.1.0 release. Affects all search queries.",
  "metadata": {
    "affected_version": "v2.1.0",
    "platform": "mobile",
    "severity": "high",
    "reproduction_steps": ["Open app on mobile", "Navigate to search", "Enter any query", "Observe empty results"]
  }
}
```

## 🗺️ Roadmap / Future Improvements

- [ ] Add LangSmith tracing for agent observability
- [ ] Integrate RAGAs evaluation for classification quality scoring
- [ ] Support additional input sources (Slack, Zendesk, Jira webhooks)
- [ ] Add conversation memory for multi-turn feedback clarification
- [ ] Dockerize the full stack (`docker-compose` for API + dashboard)
- [ ] Deploy to cloud (Railway / Render for backend, Streamlit Cloud for UI)
- [ ] Add a duplicate detection agent to merge similar tickets


## 👤 Author

**Mayur Marwade**
- GitHub: [@mayumarwade899](https://github.com/mayumarwade899)
