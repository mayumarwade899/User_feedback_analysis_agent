from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict

BASE_DIR = Path(__file__).resolve().parent
SOURCE_DIR = BASE_DIR / "src"

if SOURCE_DIR.exists() and SOURCE_DIR.as_posix() not in sys.path:
    sys.path.insert(0, SOURCE_DIR.as_posix())

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from dotenv import load_dotenv
from feedback_automation.config import ApplicationConfig
from feedback_automation.graph import FeedbackPipeline

load_dotenv()

st.set_page_config(
    page_title="Feedback Intelligence Dashboard",
    layout="wide",
)

@st.cache_resource
def init_pipeline(config_file: Path | None = None) -> FeedbackPipeline:
    """
    Initialize and cache the feedback pipeline.
    """
    config = ApplicationConfig.load(config_file)
    return FeedbackPipeline(config)


@st.cache_data
def read_outputs(config: ApplicationConfig) -> Dict[str, pd.DataFrame]:
    """
    Load generated CSV outputs into DataFrames.
    """
    dataframes: Dict[str, pd.DataFrame] = {}

    output_map = {
        "tickets": config.paths.tickets_output_file,
        "logs": config.paths.pipeline_log_file,
        "metrics": config.paths.metrics_file,
    }

    for key, path in output_map.items():
        if path.exists():
            df = pd.read_csv(path)
            if key == "tickets" and "metadata" in df.columns:
                df["metadata"] = df["metadata"].apply(_safe_json)
            dataframes[key] = df

    return dataframes


def _safe_json(value: str):
    try:
        return json.loads(value)
    except Exception:
        return value


def pipeline_runner(pipeline: FeedbackPipeline):
    st.subheader("Run Processing Pipeline")

    if st.button("Execute Pipeline", type="primary"):
        with st.spinner("Analyzing feedback..."):
            results = pipeline.process()
            pipeline.write_outputs(results)

        st.session_state.pipeline_executed = True 
        st.success(f"{len(results)} feedback records processed successfully.")
        st.cache_data.clear()


def metrics_panel(outputs: Dict[str, pd.DataFrame]):
    st.subheader("Key Metrics")

    metrics_df = outputs.get("metrics")
    if metrics_df is None or metrics_df.empty:
        st.info("No metrics available yet.")
        return

    row = metrics_df.iloc[0]
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Processed", int(row.get("processed", 0)))
    col2.metric("By Category", str(row.get("by_category", {})))
    col3.metric("By Priority", str(row.get("by_priority", {})))


def tickets_panel(outputs: Dict[str, pd.DataFrame], config: ApplicationConfig):
    st.subheader("Generated Tickets")

    tickets_df = outputs.get("tickets")
    if tickets_df is None or tickets_df.empty:
        st.info("No tickets found. Run the pipeline first.")
        return

    st.dataframe(tickets_df, use_container_width=True)

    st.markdown("### Edit Ticket")

    ticket_ids = tickets_df["ticket_id"].tolist()
    selected_ticket = st.selectbox("Ticket ID", ticket_ids)

    record = tickets_df[tickets_df["ticket_id"] == selected_ticket].iloc[0]

    title = st.text_input("Title", record["title"])
    priority_levels = ["Critical", "High", "Medium", "Low"]
    priority = st.selectbox(
        "Priority",
        priority_levels,
        index=priority_levels.index(record["priority"]),
    )
    description = st.text_area(
        "Description",
        record["description"],
        height=200,
    )

    if st.button("Save Changes"):
        tickets_df.loc[
            tickets_df["ticket_id"] == selected_ticket,
            ["title", "priority", "description"],
        ] = [title, priority, description]

        tickets_df.to_csv(config.paths.tickets_output_file, index=False)
        st.success(f"Ticket {selected_ticket} updated.")
        st.cache_data.clear()


def logs_panel(outputs: Dict[str, pd.DataFrame]):
    st.subheader("Processing Logs")

    log_df = outputs.get("logs")
    if log_df is None or log_df.empty:
        st.info("No logs available.")
        return

    st.dataframe(
        log_df.sort_values("timestamp", ascending=False),
        use_container_width=True,
    )


def graph_visual_section(pipeline: FeedbackPipeline):
    st.subheader("Pipeline Graph")
    mermaid = pipeline.mermaid_diagram()
    _render_mermaid(mermaid)
    with st.expander("Mermaid source"):
        st.code(mermaid, language="mermaid")
    with st.expander("ASCII view"):
        st.code(pipeline.ascii_diagram())

def _render_mermaid(mermaid_diagram: str, height: int = 400) -> None:
    components.html(
        f"""
        <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
        <div class="mermaid">
        {mermaid_diagram}
        </div>
        <script>mermaid.initialize({{ startOnLoad: true }});</script>
        """,
        height=height,
    )


def expected_vs_actual_section(pipeline: FeedbackPipeline, outputs: Dict[str, pd.DataFrame]):
    st.subheader("Classification Accuracy")
    config = pipeline.config
    if not config.paths.expected_labels_file.exists():
        st.info("No expected classifications file found.")
        return
    if st.button("Compare to expected"):
        with st.spinner("Running comparison..."):
            results = pipeline.process()
            df = pipeline.compare_with_expected(results)
        st.dataframe(df, use_container_width=True)


def main() -> None:
    if "pipeline_executed" not in st.session_state:
        st.session_state.pipeline_executed = False

    st.title("Intelligent Feedback Processing System")
    st.caption("Multi-agent workflow powered by LangGraph")

    config_input = st.sidebar.text_input(
        "Configuration file path",
        value="config/config.yaml",
    )
    config_file = Path(config_input)

    pipeline = init_pipeline(config_file if config_file.exists() else None)
    st.sidebar.json(pipeline.config.to_dict())

    pipeline_runner(pipeline)

    if not st.session_state.pipeline_executed:
        st.info("Click **Execute Pipeline** to view results.")
        return
    
    outputs = read_outputs(pipeline.config)

    metrics_panel(outputs)
    tickets_panel(outputs, pipeline.config)
    logs_panel(outputs)
    graph_visual_section(pipeline)
    expected_vs_actual_section(pipeline, outputs)


if __name__ == "__main__":
    main()
