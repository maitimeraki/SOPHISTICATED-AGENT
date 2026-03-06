import streamlit as st
from functions_for_pipeline import *
import streamlit.components.v1 as components
import mlflow
from mlflow import langchain
import tempfile
from pyvis.network import Network


langchain.autolog()  
# Specify the tracking URI for the MLflow server.
mlflow.set_tracking_uri("http://localhost:5000")
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
    net = Network(directed=True, notebook=True, height="300px", width="100%")
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
    # Add edges with a default color
    for edge in edges:
        net.add_edge(edge[0], edge[1], color="#808080")  # Set edge color to gray
        
    net.options.edges.smooth.type = "straight" # Set edges to straight lines
    return net

def compute_initial_position(net):
    """
    Compute the initial position of the nodes in  network graph.

    Args:
        net (Network): The network graph visualization.

    Returns:
        dict: The initial position of the graph.
    """
    net.barnes_hut()
    return {node["id"]: {"x": node["x"], "y": node["y"]} for node in net.nodes}

def save_and_display_graph(net):
    """
    Save the network graph visualization to an HTML file and display it in Streamlit.

    Args:
        net (Network): The network graph visualization.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        net.write_html(tmp_file.name,notebook=True)
        tmp_file.flush()
        mlflow.log_artifact(tmp_file.name, artifact_path="network_graphs")
        with open(tmp_file.name, "r", encoding="utf-8") as f:
            html_content = f.read()
            return html_content
        
        
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
    agent_state_value =None
    progress_bar = st.progress(0)
    count_step =0
    previous_state = None
    previous_values = {key:None for key in placeholders}
    
    try:
        for plan_output in plan_and_execute_app.stream(inputs, config=config):
            count_step += 1
            for step, agent_state_value in plan_output.items():
                print(f"Step: {step}, Agent State Value: {agent_state_value}")
                mlflow.log_metric("steps_executed", count_step)
                mlflow.log_text(str(agent_state_value), artifact_file=f"step_{count_step}_state.txt")
                previous_values, previous_state = updates_placeholders_and_graph(agent_state_value, placeholders, graph_placeholder, previous_values, previous_state)
                progress_bar.progress(count_step/recurtion_limit)
                if count_step >= recurtion_limit:
                    break
                
        for key, placeholder in placeholders.items():
            if key in previous_values and previous_values[key] is not None:
                if isinstance(previous_values[key],list):
                    formatted_value = "\n".join([f"{i+1}. {item}" for i, item in enumerate(previous_values[key])])
                else:
                    formatted_value = previous_values[key]
                placeholder.markdown(f"{formatted_value}")
                
        response = agent_state_value.get('response', "No response generated.") if agent_state_value else "No response generated."
        return response
                
    except Exception as e:
        st.error(f"Error during execution: {e}")
        return "An error occurred during execution."
def main():
    """Main function to run the Streamlit app
    """
    st.set_page_config(page_title="Agent Network Graph", page_icon="🌐", layout="wide")
    st.title("Agent Network Graph Visualization")
    
    # Load the current state of the agent
    plan_and_execute_app = create_agent()
    
    question = st.text_input("Enter your question:")
    if st.button("Submit"):
        inputs = {"question": question}
        
        st.markdown("**Graph**")
        graph_placeholder = st.empty()  # Placeholder for the graph
        
        col1, col2, col3 = st.columns([1,1,4])
        with col1:
            st.markdown("**Plan**")
        with col2:
            st.markdown("**Past Steps**")
        with col3:
            st.markdown("**Aggregated Context**")
            
        placeholders ={
            "plan": col1.empty(),
            "past_steps": col2.empty(),
            "aggregated_context": col3.empty()
        }
        with mlflow.start_run(run_name="Sophisticated Agent Execution"):
            mlflow.log_param("question", question)
            response = execute_plan_and_print_steps(inputs, plan_and_execute_app, placeholders, graph_placeholder, recurtion_limit=45)
            mlflow.log_text(response, artifact_file="final_response.txt")
            
            
        st.write("**Final Answer:**")
        st.write(response)
        
        
        
if __name__ == "__main__":
    main()