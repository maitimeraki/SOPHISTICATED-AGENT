# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 🚀 Architecture Overview

The core of the application is the **Sophisticated Agent**, implemented in [sophiscated_agent.py](sophiscated_agent.py). This agent orchestrates a complex, multi-step workflow using a state graph (`StateGraph`) to process user queries.

**Key Architectural Components:**

1.  **Agent Workflow:** The agent follows a defined sequence of steps:
    *   `anonymize_question` $\rightarrow$ `planner` $\rightarrow$ `de_anonymize_plan` $\rightarrow$ `break_down_plan` $\rightarrow$ `task_handler` $\rightarrow$ (Retrieval Steps) $\rightarrow$ `answer`.
    *   The workflow is visualized using a network graph, which is logged via MLflow.
2.  **Data Retrieval (RAG):** The system implements a Retrieval-Augmented Generation (RAG) pattern, sourcing context from three distinct vector stores:
    *   `chunks_vector_store`: General chunked content.
    *   `chapter_summaries_vector_store`: Summarized chapter content.
    *   `book_quotes_vectorstore`: Specific book quotes.
3.  **Technology Stack:**
    *   **Frontend/UI:** Streamlit is used to host the interactive web application.
    *   **LLM/Orchestration:** LangChain and LangGraph manage the complex agent state machine.
    *   **ML Tracking:** MLflow is used to log the execution state, metrics, and network graphs for reproducibility.
    *   **Data Processing:** The system relies on `PyPDF2` and `pdfplumber` for PDF content extraction.

## 🛠️ Development Commands

### 1. Setup and Dependencies
To install all required dependencies, use the following command:
```bash
pip install -r requirements.txt
```

### 2. Running the Application
The application is a Streamlit web app. To run the main agent interface:
```bash
streamlit run sophisticated_agent.py
```

### 3. Testing and Development
*   **Unit Testing:** Unit tests should focus on the utility functions in [functions_for_pipeline.py](functions_for_pipeline.py) and the graph logic in [sophiscated_agent.py](sophiscated_agent.py).
*   **MLflow Tracking:** Remember that the agent execution automatically logs metrics and artifacts to the MLflow server (default URI: `http://localhost:5000`).
*   **Single Test:** For testing a specific function, import it directly into a temporary script and run it, or use the `pytest` framework if configured.

## 💡 Development Notes

*   **State Management:** The agent's state is critical. Changes to the node IDs or the sequence of edges in `create_network_graph` will directly impact the workflow.
*   **Data Sources:** When modifying retrieval logic, ensure the correct vector store (`chunks_vector_store`, `chapter_summaries_vector_store`, or `book_quotes_vectorstore`) is targeted, as they serve different levels of context granularity.
*   **MLflow Context:** Always ensure the MLflow tracking server is running before executing the agent to guarantee proper logging of experiments.
