import sys
import os
import streamlit as st
from functions_for_pipeline import *
import streamlit.components.v1 as components
import mlflow
from mlflow import langchain
import tempfile
from pyvis.network import Network
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler('./logs/sophiscated_agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


langchain.autolog()
logger.info("LangChain autolog enabled")  
# Use local file store — skips the URL print entirely
mlflow.set_tracking_uri(f"file:///{os.path.abspath('./mlruns').replace('\\', '/')}")
# Specify the experiment you just created for your GenAI application.
mlflow.set_experiment("Sophisticated Agent Experiment")


def create_network_graph(current_state):
    """
    Create a network graph visualization of the agent's current state.

    Args:
        current_state (str): The current state of the agent.

    Returns:
        Network: The network graph visualization.
    """
    try:
        logger.debug(f"create_network_graph called with current_state: {current_state}")
        net = Network(directed=True, notebook=True, height="300px", width="100%")
        logger.debug("Created Network object")
        net.toggle_physics(False) # Disable physics for better layout
        nodes = [
        {"id": "anonymize_question", "label": "anonymize_question", "x": 0, "y": 0},
        {"id": "planner", "label": "planner", "x": 175*1.75, "y": -100},
        {"id": "de_anonymize_plan", "label": "de_anonymize_plan", "x": 350*1.75, "y": -100},
        {"id": "break_down_plan", "label": "break_down_plan", "x": 525*1.75, "y": -100},
        {"id": "task_handler", "label": "task_handler", "x": 700*1.75, "y": 0},
        {"id": "retrieve_chunks", "label": "retrieve_chunks", "x": 875*1.75, "y": +200},
        {"id": "retrieve_summaries", "label": "retrieve_summaries", "x": 875*1.75, "y": +100},
        {"id": "retrieve_book_quotes", "label": "retrieve_book_quotes", "x": 875*1.75, "y": 0},
        {"id": "answer", "label": "answer", "x": 875*1.75, "y": -100},
        {"id": "replan", "label": "replan", "x": 1050*1.75, "y": 0},
        {"id": "can_be_answered_already", "label": "can_be_answered_already", "x": 1225*1.75, "y": 0},
        {"id": "get_final_answer", "label": "get_final_answer", "x": 1400*1.75, "y": 0}
    ]
        edges = [
            ("anonymize_question", "planner"),
            ("planner", "de_anonymize_plan"),
            ("de_anonymize_plan", "break_down_plan"),
            ("break_down_plan", "task_handler"),
            ("task_handler", "retrieve_chunks"),
            ("task_handler", "retrieve_summaries"),
            ("task_handler", "retrieve_book_quotes"),
            ("task_handler", "answer"),
            ("retrieve_chunks", "replan"),
            ("retrieve_summaries", "replan"),
            ("retrieve_book_quotes", "replan"),
            ("answer", "replan"),
            ("replan", "can_be_answered_already"),
            ("replan", "break_down_plan"),
            ("can_be_answered_already", "get_final_answer")
        ]
        for node in nodes:
            color= "#00FF00" if node["id"] == current_state else "#FF69B4"  # Green if current, else pink
            net.add_node(node["id"], label=node["label"], color=color, x=node["x"], y=node["y"], physics=False)
            logger.debug(f"Added node: {node['id']}")

        logger.debug(f"Added {len(nodes)} nodes to network")

        # Add edges with a default color
        for edge in edges:
            net.add_edge(edge[0], edge[1], color="#808080")  # Set edge color to gray
            logger.debug(f"Added edge: {edge[0]} -> {edge[1]}")

        logger.debug(f"Added {len(edges)} edges to network")

        net.options.edges.smooth.type = "straight" # Set edges to straight lines
        logger.info(f"Network graph created successfully for state: {current_state}")
        return net
    except Exception as e:
        logger.error(f"Error in create_network_graph: {str(e)}", exc_info=True)
        raise

def compute_initial_position(net):
    """
    Compute the initial position of the nodes in  network graph.

    Args:
        net (Network): The network graph visualization.

    Returns:
        dict: The initial position of the graph.
    """
    net.barnes_hut()
    return {node["id"]: {"x": node["x"], "y": node["y"]}     for node in net.nodes}

def save_and_display_graph(net):
    """
    Save the network graph visualization to an HTML file and display it in Streamlit.

    Args:
        net (Network): The network graph visualization.
    """
    try:
        logger.debug("save_and_display_graph called")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
            logger.debug(f"Created temporary file: {tmp_file.name}")
            net.write_html(tmp_file.name, notebook=True)
            logger.debug("Wrote HTML to file")
            tmp_file.flush()

            logger.info(f"Logging artifact to MLflow: {tmp_file.name}")
            mlflow.log_artifact(tmp_file.name, artifact_path="network_graphs")

            with open(tmp_file.name, "r", encoding="utf-8") as f:
                html_content = f.read()
                logger.info(f"Read HTML content: {len(html_content)} characters")
                return html_content
    except IOError as e:
        logger.error(f"File I/O error in save_and_display_graph: {str(e)}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Error in save_and_display_graph: {str(e)}", exc_info=True)
        raise
        
        
def updates_placeholders_and_graph(agent_state_value, placeholders, graph_placeholder, previous_values, previous_state):
    """
    Update the placeholders and graph in the Streamlit app based on the current state.

    Args:
        agent_state_value (dict): The current state value of the agent.
        placeholders (dict): The placeholders to display the steps.
        graph_placeholder (Streamlit.placeholder): The placeholder to display the network graph.
        previous_values (dict): The previous values of the placeholders.
        previous_state: The previous state of the agent.

    Returns:
        tuple: Updated previous_values and previous_state.
    """
    current_state = agent_state_value.get("curr_state")
    # Update the graph visualization
    if current_state: 
        net = create_network_graph(current_state)
        graph_html = save_and_display_graph(net)
        graph_placeholder.empty()
        with graph_placeholder.container():
            components.html(graph_html, height=350)
    
    # Update placeholders only if the state has changed (i.e., we've finished visiting the previous node)
    if current_state != previous_state and previous_state is not None:
        for key, placeholder in placeholders.items():
            if key in previous_values and previous_values[key] is not None:
                if isinstance(previous_values[key],list):
                    formatted_value = "\n".join([f"{i+1}. {item}" for i, item in enumerate(previous_values[key])])
                else:
                    formatted_value = previous_values[key]
                placeholder.markdown(f"{formatted_value}")
    # Update previous values and state for the next iteration
    for key in placeholders:
        if key in agent_state_value:
            previous_values[key] = agent_state_value[key]
            
    return previous_values, current_state
            
        
        
# ============================================================================
# PHASE 3: Streaming response support
# ============================================================================

def stream_response(response_placeholder, initial_response: str):
    """Phase 3: Show response progressively. Returns user-perceivable answer early."""
    if initial_response:
        response_placeholder.markdown(initial_response)
        logger.info("Phase 3: Streamed initial response to user at 5-10s mark")


# ============================================================================
# END PHASE 3 STREAMING
# ============================================================================

def execute_plan_and_print_steps(inputs, plan_and_execute_app, placeholders, graph_placeholder, recurtion_limit=45):
    """
    Execute the plan and print the steps in the Streamlit app.

    Args:
        inputs (dict): The inputs to the plan.
        plan_and_execute_app (StateGraph): The compiled plan and execute app.
        placeholders (dict): The placeholders to display the steps.
        graph_placeholder (Streamlit.placeholder): The placeholder to display the network graph.
        recursion_limit (int): The recursion limit for the plan execution.

    Returns:
        str: The final response from the agent.
    """
    config = {"recursion_limit": recurtion_limit}
    agent_state_value = None
    progress_bar = st.progress(0)
    count_step = 0
    previous_state = None
    previous_values = {key: None for key in placeholders}

    try:
        logger.info(f"Starting plan execution with recursion_limit: {recurtion_limit}")
        logger.debug(f"Inputs: {inputs}")

        for plan_output in plan_and_execute_app.stream(inputs, config=config):
            count_step += 1
            logger.debug(f"Processing step {count_step}")

            for step, agent_state_value in plan_output.items():
                logger.info(f"Step {count_step}: {step}")
                print(f"Step: {step}, Agent State Value: {agent_state_value}")

                try:
                    mlflow.log_metric("steps_executed", count_step)
                    mlflow.log_text(str(agent_state_value), artifact_file=f"step_{count_step}_state.txt")
                    logger.debug(f"Logged step {count_step} to MLflow")
                except Exception as mlflow_error:
                    logger.warning(f"Failed to log to MLflow: {str(mlflow_error)}")

                previous_values, previous_state = updates_placeholders_and_graph(agent_state_value, placeholders, graph_placeholder, previous_values, previous_state)
                progress_bar.progress(count_step / recurtion_limit)

                if count_step >= recurtion_limit:
                    logger.warning(f"Reached recursion limit: {recurtion_limit}")
                    break

        logger.info("Plan execution completed, displaying final results")

        for key, placeholder in placeholders.items():
            if previous_values is not None:
                if key in previous_values and previous_values[key] is not None:
                    if isinstance(previous_values[key], list):
                        formatted_value = "\n".join([f"{i+1}. {item}" for i, item in enumerate(previous_values[key])])
                    else:
                        formatted_value = previous_values[key]
                    placeholder.markdown(f"{formatted_value}")
                    logger.debug(f"Updated placeholder: {key}")

        response = agent_state_value.get('response', "No response generated.") if agent_state_value else "No response generated."
        logger.info(f"Final response: {response[:100]}")
        return response

    except Exception as e:
        logger.error(f"Error during execution: {str(e)}", exc_info=True)
        st.error(f"Error during execution: {e}")
        return "An error occurred during execution."
def main():
    """Main function to run the Streamlit app
    """
    try:
        logger.info("Starting main Streamlit app")
        st.set_page_config(page_title="Agent Network Graph", page_icon="🌐", layout="wide")
        st.title("Agent Network Graph Visualization")
        logger.debug("Streamlit page config set")

        # Phase 1 + Phase 3: Initialize caches in Streamlit session state
        if 'embedding_cache' not in st.session_state:
            st.session_state.embedding_cache = EmbeddingCache()
            logger.info("Phase 1: Initialized embedding cache in session state")

        if 'request_cache' not in st.session_state:
            from functions_for_pipeline import _REQUEST_CACHE
            st.session_state.request_cache = _REQUEST_CACHE
            logger.info("Phase 3: Initialized request cache in session state")

        # Load the current state of the agent
        logger.info("Creating agent...")
        plan_and_execute_app = create_agent()
        logger.info("Agent created successfully")

        question = st.text_input("Enter your question:")
        logger.debug(f"Question input: {question[:50] if question else 'No input'}")

        if st.button("Submit"):
            logger.info(f"Processing question: {question[:50]}")
            inputs = {"question": question}

            st.markdown("**Graph**")
            graph_placeholder = st.empty()  # Placeholder for the graph

            col1, col2, col3 = st.columns([1, 1, 4])
            with col1:
                st.markdown("**Plan**")
            with col2:
                st.markdown("**Past Steps**")
            with col3:
                st.markdown("**Aggregated Context**")

            placeholders = {
                "plan": col1.empty(),
                "past_steps": col2.empty(),
                "aggregated_context": col3.empty()
            }
            logger.debug("Created placeholders")

            try:
                with mlflow.start_run(run_name="Sophisticated Agent Execution"):
                    logger.info("MLflow run started")
                    mlflow.log_param("question", question)
                    logger.debug("Logged question to MLflow")

                    response = execute_plan_and_print_steps(inputs, plan_and_execute_app, placeholders, graph_placeholder, recurtion_limit=45)
                    logger.info(f"Plan execution completed, response length: {len(response)}")

                    mlflow.log_text(response, artifact_file="final_response.txt")
                    logger.info("Logged final response to MLflow")

            except Exception as mlflow_error:
                logger.error(f"MLflow error: {str(mlflow_error)}", exc_info=True)
                raise

            st.write("**Final Answer:**")
            st.write(response)
            logger.info("Displayed final answer in UI")

    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}", exc_info=True)
        st.error(f"Fatal error: {str(e)}")
        raise


if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("Sophisticated Agent Application Started")
    logger.info("=" * 80)
    try:
        main()
    except Exception as e:
        logger.critical(f"Application crashed: {str(e)}", exc_info=True)
        raise