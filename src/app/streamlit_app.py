"""
Streamlit app for running and visualizing leadership emergence simulations.
"""

import streamlit as st
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.base_model import LeadershipEmergenceModel
from src.visualization.plot_outcomes import (
    plot_identity_evolution,
    plot_network_metrics,
    plot_leadership_network,
    plot_interaction_heatmap
)
import matplotlib.pyplot as plt

def main():
    st.title("Leadership Emergence Simulation")
    st.write("""
    This app demonstrates the emergence of leadership through agent-based modeling.
    Adjust the parameters and run simulations to see how different conditions affect leadership emergence.
    """)
    
    # Sidebar for parameters
    st.sidebar.header("Simulation Parameters")
    
    n_agents = st.sidebar.slider(
        "Number of Agents",
        min_value=3,
        max_value=10,
        value=4,
        help="Number of agents in the simulation"
    )
    
    initial_li_equal = st.sidebar.checkbox(
        "Equal Initial Leader Identity",
        value=True,
        help="If checked, all agents start with equal leader identity"
    )
    
    li_change_rate = st.sidebar.slider(
        "Identity Change Rate",
        min_value=0.5,
        max_value=5.0,
        value=2.0,
        step=0.5,
        help="Rate at which identities change after interactions"
    )
    
    n_steps = st.sidebar.slider(
        "Number of Steps",
        min_value=10,
        max_value=200,
        value=100,
        step=10,
        help="Number of simulation steps to run"
    )
    
    # Run simulation button
    if st.sidebar.button("Run Simulation"):
        # Initialize model
        model = LeadershipEmergenceModel(
            n_agents=n_agents,
            initial_li_equal=initial_li_equal,
            li_change_rate=li_change_rate
        )
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Run simulation
        for step in range(n_steps):
            model.step()
            progress = (step + 1) / n_steps
            progress_bar.progress(progress)
            status_text.text(f"Running step {step + 1}/{n_steps}")
        
        status_text.text("Simulation complete! Generating visualizations...")
        
        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs([
            "Identity Evolution",
            "Network Metrics",
            "Leadership Network",
            "Interaction Patterns"
        ])
        
        with tab1:
            st.header("Identity Evolution")
            fig, ax = plt.subplots(figsize=(10, 6))
            plot_identity_evolution(model.history)
            st.pyplot(fig)
            plt.close()
            
            # Add statistics
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Mean Leader Identity",
                    f"{sum(agent.leader_identity for agent in model.agents) / model.n_agents:.2f}"
                )
            with col2:
                st.metric(
                    "Mean Follower Identity",
                    f"{sum(agent.follower_identity for agent in model.agents) / model.n_agents:.2f}"
                )
        
        with tab2:
            st.header("Network Metrics")
            fig, ax = plt.subplots(figsize=(12, 5))
            plot_network_metrics(model.history)
            st.pyplot(fig)
            plt.close()
            
            # Add metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Network Density", f"{model.history['density'][-1]:.2f}")
            with col2:
                st.metric("Leadership Centralization", f"{model.history['centralization'][-1]:.2f}")
        
        with tab3:
            st.header("Leadership Network")
            fig, ax = plt.subplots(figsize=(8, 8))
            plot_leadership_network(model.interaction_network)
            st.pyplot(fig)
            plt.close()
            
            st.write("""
            Node size represents leadership perception (larger = stronger leader).
            Edge thickness represents interaction strength.
            """)
        
        with tab4:
            st.header("Interaction Patterns")
            fig, ax = plt.subplots(figsize=(8, 6))
            plot_interaction_heatmap(model)
            st.pyplot(fig)
            plt.close()
            
            st.write("""
            Heatmap shows the strength of leadership perceptions between agents.
            Darker colors indicate stronger leadership recognition.
            """)
        
        # Final status
        status_text.text("Analysis complete! Adjust parameters and run again to compare.")

if __name__ == "__main__":
    main() 