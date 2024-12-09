"""
Shiny application for Leadership Emergence Simulation.
"""

import os
import sys
from pathlib import Path
import json
import pandas as pd
import io

# Ensure proper path handling
root_dir = Path(__file__).parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from shiny import App, render, ui, reactive
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import networkx as nx
from matplotlib.patches import Circle

from src.models.base_model import (
    BaseLeadershipModel,
    SchemaModel,
    NetworkModel,
    SchemaNetworkModel
)

from src.models.metrics import (
    calculate_identity_variance,
    calculate_perception_agreement,
    calculate_claiming_granting_correlation,
    calculate_network_metrics,
    calculate_emergence_lag,
    calculate_identity_behavior_consistency
)

app_ui = ui.page_fluid(
    ui.panel_title("Leadership Emergence Simulation"),
    
    ui.tags.head(
        ui.tags.script("""
        window.Shiny.addCustomMessageHandler('toggleVisibility', function(message) {
            document.getElementById(message.id).style.display = message.display;
        });
        """)
    ),
    
    # Model Selection Landing Page
    ui.div(
        {"id": "model_selection_page"},
        ui.h2("Select Model Type"),
        ui.row(
            ui.column(
                3,
                ui.card(
                    ui.card_header("Base Model"),
                    ui.card_body(
                        "Simple leadership emergence through dyadic interactions",
                        ui.br(),
                        ui.br(),
                        ui.input_action_button("select_base", "Select Base Model", class_="btn-primary btn-block")
                    )
                )
            ),
            ui.column(
                3,
                ui.card(
                    ui.card_header("Schema Model"),
                    ui.card_body(
                        "Leadership emergence with cognitive schemas",
                        ui.br(),
                        ui.br(),
                        ui.input_action_button("select_schema", "Select Schema Model", class_="btn-primary btn-block")
                    )
                )
            ),
            ui.column(
                3,
                ui.card(
                    ui.card_header("Network Model"),
                    ui.card_body(
                        "Leadership emergence in social networks",
                        ui.br(),
                        ui.br(),
                        ui.input_action_button("select_network", "Select Network Model", class_="btn-primary btn-block")
                    )
                )
            ),
            ui.column(
                3,
                ui.card(
                    ui.card_header("Combined Model"),
                    ui.card_body(
                        "Leadership emergence with schemas and networks",
                        ui.br(),
                        ui.br(),
                        ui.input_action_button("select_combined", "Select Combined Model", class_="btn-primary btn-block")
                    )
                )
            )
        )
    ),
    
    # Simulation Interface (initially hidden)
    ui.div(
        {"id": "simulation_interface", "style": "display: none"},
        ui.row(
            ui.column(
                3,
                ui.card(
                    ui.card_header("Simulation Parameters"),
                    ui.card_body(
                        ui.h4("Simulation Mode"),
                        ui.input_radio_buttons(
                            "sim_mode",
                            None,
                            {
                                "analysis": "Analysis Mode",
                                "step": "Step-by-Step Mode"
                            },
                            selected="analysis"
                        ),
                        ui.input_slider("n_agents", "Number of Agents", 2, 10, 4),
                        ui.input_checkbox("initial_li_equal", "Equal Initial Leader Identity", True),
                        ui.input_slider("li_change_rate", "Identity Change Rate", 0.5, 5, 2),
                        
                        # Analysis Mode Controls
                        ui.panel_conditional(
                            "input.sim_mode === 'analysis'",
                            ui.div(
                                ui.input_slider("n_steps", "Number of Steps", 10, 200, 100),
                                ui.input_action_button(
                                    "run_sim", 
                                    "Run Full Simulation", 
                                    class_="btn-success btn-block"
                                ),
                                ui.br(),
                                ui.br()
                            )
                        ),
                        
                        # Step Mode Controls
                        ui.panel_conditional(
                            "input.sim_mode === 'step'",
                            ui.div(
                                ui.input_action_button(
                                    "init_sim", 
                                    "Initialize Simulation", 
                                    class_="btn-primary btn-block"
                                ),
                                ui.br(),
                                ui.br(),
                                ui.input_action_button(
                                    "step_sim", 
                                    "Step Forward", 
                                    class_="btn-info btn-block"
                                ),
                                ui.br(),
                                ui.br(),
                                ui.card(
                                    ui.card_header("Current Interaction"),
                                    ui.card_body(
                                        ui.output_text("interaction_details")
                                    )
                                )
                            )
                        ),
                        
                        ui.input_action_button(
                            "reset_sim", 
                            "Reset", 
                            class_="btn-warning btn-block"
                        ),
                        ui.br(),
                        ui.br(),
                        ui.output_text("sim_status")
                    )
                )
            ),
            ui.column(
                9,
                # Analysis Mode View
                ui.panel_conditional(
                    "input.sim_mode === 'analysis'",
                    ui.navset_tab(
                        ui.nav_panel(
                            "Identity Evolution",
                            ui.row(
                                ui.column(12, ui.output_plot("identity_plot"))
                            ),
                            ui.row(
                                ui.column(6, ui.value_box(
                                    "Mean Leader Identity",
                                    ui.output_text("mean_li"),
                                    theme="primary"
                                )),
                                ui.column(6, ui.value_box(
                                    "Mean Follower Identity",
                                    ui.output_text("mean_fi"),
                                    theme="info"
                                ))
                            )
                        ),
                        ui.nav_panel(
                            "Role Stabilization",
                            ui.row(
                                ui.column(12, ui.output_plot("variance_plot"))
                            ),
                            ui.row(
                                ui.column(6, ui.value_box(
                                    "Emergence Lag",
                                    ui.output_text("emergence_lag"),
                                    theme="warning"
                                )),
                                ui.column(6, ui.value_box(
                                    "Within-Group Agreement",
                                    ui.output_text("perception_agreement"),
                                    theme="success"
                                ))
                            )
                        ),
                        ui.nav_panel(
                            "Network Metrics",
                            ui.row(
                                ui.column(12, ui.output_plot("network_plot"))
                            ),
                            ui.row(
                                ui.column(4, ui.value_box(
                                    "Network Density",
                                    ui.output_text("density"),
                                    theme="primary"
                                )),
                                ui.column(4, ui.value_box(
                                    "Centralization",
                                    ui.output_text("centralization"),
                                    theme="info"
                                )),
                                ui.column(4, ui.value_box(
                                    "Modularity",
                                    ui.output_text("modularity"),
                                    theme="success"
                                ))
                            )
                        ),
                        ui.nav_panel(
                            "Claiming-Granting",
                            ui.row(
                                ui.column(12, ui.output_plot("claim_grant_plot"))
                            ),
                            ui.row(
                                ui.column(6, ui.value_box(
                                    "Claim-Grant Correlation",
                                    ui.output_text("claim_grant_corr"),
                                    theme="primary"
                                )),
                                ui.column(6, ui.value_box(
                                    "Identity-Behavior Consistency",
                                    ui.output_text("identity_behavior_consistency"),
                                    theme="info"
                                ))
                            )
                        ),
                        id="viz_tabs"
                    )
                ),
                
                # Step Mode View
                ui.panel_conditional(
                    "input.sim_mode === 'step'",
                    ui.div(
                        ui.row(
                            ui.column(
                                12,
                                ui.card(
                                    ui.card_header("Network State"),
                                    ui.card_body(
                                        ui.output_plot("step_network_plot", height="500px")
                                    )
                                )
                            )
                        ),
                        ui.row(
                            ui.column(
                                6,
                                ui.card(
                                    ui.card_header("Agent Details"),
                                    ui.card_body(
                                        ui.output_ui("agent_details")
                                    )
                                )
                            ),
                            ui.column(
                                6,
                                ui.card(
                                    ui.card_header("Recent History"),
                                    ui.card_body(
                                        ui.output_plot("step_history_plot", height="300px")
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
    )
)

def server(input, output, session):
    rv = reactive.Value({
        'model': None,
        'current_step': 0,
        'network_pos': None,
        'agents': None,
        'selected_model': None
    })
    
    async def toggle_visibility(element_id, show=True):
        """Helper function to show/hide elements"""
        await session.send_custom_message(
            'toggleVisibility',
            {'id': element_id, 'display': 'block' if show else 'none'}
        )
    
    # Model Selection Handlers
    @reactive.Effect
    @reactive.event(input.select_base)
    async def select_base_model():
        rv.set({
            'selected_model': 'Base Model',
            'model': None,
            'current_step': 0,
            'network_pos': None,
            'agents': None
        })
        ui.update_radio_buttons("sim_mode", selected="analysis")
        await toggle_visibility("simulation_interface", True)
        await toggle_visibility("model_selection_page", False)
    
    @reactive.Effect
    @reactive.event(input.select_schema)
    async def select_schema_model():
        rv.set({
            'selected_model': 'Schema Model',
            'model': None,
            'current_step': 0,
            'network_pos': None,
            'agents': None
        })
        ui.update_radio_buttons("sim_mode", selected="analysis")
        await toggle_visibility("simulation_interface", True)
        await toggle_visibility("model_selection_page", False)
    
    @reactive.Effect
    @reactive.event(input.select_network)
    async def select_network_model():
        rv.set({
            'selected_model': 'Network Model',
            'model': None,
            'current_step': 0,
            'network_pos': None,
            'agents': None
        })
        ui.update_radio_buttons("sim_mode", selected="analysis")
        await toggle_visibility("simulation_interface", True)
        await toggle_visibility("model_selection_page", False)
    
    @reactive.Effect
    @reactive.event(input.select_combined)
    async def select_combined_model():
        rv.set({
            'selected_model': 'Combined Model',
            'model': None,
            'current_step': 0,
            'network_pos': None,
            'agents': None
        })
        ui.update_radio_buttons("sim_mode", selected="analysis")
        await toggle_visibility("simulation_interface", True)
        await toggle_visibility("model_selection_page", False)
    
    @reactive.Effect
    @reactive.event(input.reset_sim)
    async def reset_simulation():
        # Return to model selection page
        await toggle_visibility("simulation_interface", False)
        await toggle_visibility("model_selection_page", True)
        current = rv.get()
        rv.set({
            'model': None,
            'current_step': 0,
            'network_pos': None,
            'agents': None,
            'selected_model': None
        })
    
    # Rest of your existing server functions...
    # (Keep all the existing simulation functions unchanged)
    
    @reactive.Effect
    @reactive.event(input.init_sim)
    def initialize_simulation():
        model_type = rv.get().get('selected_model', 'Base Model')
        model_class = {
            'Base Model': BaseLeadershipModel,
            'Schema Model': SchemaModel,
            'Network Model': NetworkModel,
            'Combined Model': SchemaNetworkModel
        }.get(model_type, BaseLeadershipModel)
        
        m = model_class(
            n_agents=input.n_agents(),
            initial_li_equal=input.initial_li_equal(),
            li_change_rate=input.li_change_rate()
        )
        rv.set({
            'model': m,
            'current_step': 0,
            'network_pos': None,
            'agents': m.agents,
            'selected_model': model_type
        })
    
    @reactive.Effect
    @reactive.event(input.step_sim)
    def step_simulation():
        current = rv.get()
        if current['model'] is not None:
            m = current['model']
            agent1, agent2 = m._select_interaction_pair()
            m.last_interaction = (agent1.id, agent2.id)
            m.step()
            rv.set({
                'model': m,
                'current_step': current['current_step'] + 1,
                'network_pos': current['network_pos'],
                'agents': m.agents,
                'selected_model': current.get('selected_model')
            })
    
    @reactive.Effect
    @reactive.event(input.run_sim)
    async def run_simulation():
        current = rv.get()
        model_type = current.get('selected_model', 'Base Model')
        
        model_class = {
            'Base Model': BaseLeadershipModel,
            'Schema Model': SchemaModel,
            'Network Model': NetworkModel,
            'Combined Model': SchemaNetworkModel
        }.get(model_type, BaseLeadershipModel)
        
        # Initialize model
        m = model_class(
            n_agents=input.n_agents(),
            initial_li_equal=input.initial_li_equal(),
            li_change_rate=input.li_change_rate()
        )
        
        # Run simulation with progress updates
        n_steps = input.n_steps()
        with ui.Progress(min=0, max=n_steps) as p:
            for step in range(n_steps):
                p.set(value=step, message=f"Running step {step + 1}")
                m.step()
                # Update state periodically
                if step % 10 == 0:  # Update every 10 steps
                    rv.set({
                        'model': m,
                        'current_step': step,
                        'network_pos': None,
                        'agents': m.agents,
                        'selected_model': model_type
                    })
        
        # Final update
        rv.set({
            'model': m,
            'current_step': n_steps,
            'network_pos': None,
            'agents': m.agents,
            'selected_model': model_type
        })
    
    @output
    @render.text
    def sim_status():
        current = rv.get()
        if current['model'] is None:
            return "Ready to initialize simulation."
        return f"Step {current['current_step']}"
    
    @output
    @render.text
    def mean_li():
        current = rv.get()
        if current['agents'] is None:
            return "0.00"
        return f"{np.mean([agent.leader_identity for agent in current['agents']]):.2f}"
    
    @output
    @render.text
    def mean_fi():
        current = rv.get()
        if current['agents'] is None:
            return "0.00"
        return f"{np.mean([agent.follower_identity for agent in current['agents']]):.2f}"
    
    @output
    @render.plot
    def identity_plot():
        current = rv.get()
        if current['model'] is None:
            return plt.figure()
        
        m = current['model']
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for agent in m.agents:
            ax.plot(agent.leader_identity_history, label=f"Agent {agent.id} LI")
            ax.plot(agent.follower_identity_history, '--', label=f"Agent {agent.id} FI")
        
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Identity Strength")
        ax.set_title("Identity Evolution Over Time")
        ax.legend()
        return fig
    
    @output
    @render.plot
    def variance_plot():
        current = rv.get()
        if current['model'] is None:
            return plt.figure()
        
        m = current['model']
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate variances over time
        li_var = []
        fi_var = []
        for t in range(len(m.agents[0].leader_identity_history)):
            li_values = [agent.leader_identity_history[t] for agent in m.agents]
            fi_values = [agent.follower_identity_history[t] for agent in m.agents]
            li_var.append(np.var(li_values))
            fi_var.append(np.var(fi_values))
        
        ax.plot(li_var, label='Leader Identity Variance')
        ax.plot(fi_var, label='Follower Identity Variance')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Variance')
        ax.set_title('Role Stabilization Over Time')
        ax.legend()
        return fig
    
    @output
    @render.plot
    def network_plot():
        current = rv.get()
        if current['model'] is None or not hasattr(current['model'], 'interaction_network'):
            return plt.figure()
        
        m = current['model']
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Use stored positions or create new ones
        if current['network_pos'] is None:
            pos = nx.spring_layout(m.interaction_network)
            current['network_pos'] = pos
            rv.set(current)
        else:
            pos = current['network_pos']
        
        # Draw the network
        nx.draw_networkx_edges(m.interaction_network, pos, alpha=0.2)
        
        # Draw nodes with size based on leader identity
        node_colors = [agent.leader_identity for agent in m.agents]
        nx.draw_networkx_nodes(m.interaction_network, pos, 
                             node_color=node_colors, 
                             node_size=1000,
                             cmap=plt.cm.viridis)
        
        # Add labels
        labels = {i: f"Agent {i}\nLI: {agent.leader_identity:.2f}" 
                 for i, agent in enumerate(m.agents)}
        nx.draw_networkx_labels(m.interaction_network, pos, labels)
        
        # Highlight last interaction if available
        if hasattr(m, 'last_interaction'):
            ax.add_patch(Circle(pos[m.last_interaction[0]], 0.15, 
                              fill=False, color='red'))
            ax.add_patch(Circle(pos[m.last_interaction[1]], 0.15, 
                              fill=False, color='blue'))
        
        ax.set_title("Interaction Network")
        return fig
    
    @output
    @render.plot
    def claim_grant_plot():
        current = rv.get()
        if current['model'] is None:
            return plt.figure()
        
        m = current['model']
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate claiming and granting for each agent
        claims = []
        grants = []
        for agent in m.agents:
            claims.append(agent.leader_identity)
            grants.append(1 - agent.follower_identity)  # Convert to granting
        
        ax.scatter(claims, grants)
        for i, (claim, grant) in enumerate(zip(claims, grants)):
            ax.annotate(f"Agent {i}", (claim, grant))
        
        ax.plot([0, 1], [0, 1], '--', color='gray')  # Identity line
        ax.set_xlabel("Leadership Claiming (Leader Identity)")
        ax.set_ylabel("Leadership Granting (1 - Follower Identity)")
        ax.set_title("Leadership Claiming vs Granting")
        return fig
    
    @output
    @render.text
    def emergence_lag():
        current = rv.get()
        if current['model'] is None:
            return "N/A"
        
        m = current['model']
        # Find when role differentiation stabilizes
        role_diffs = []
        for t in range(len(m.agents[0].leader_identity_history)):
            diffs = []
            for agent in m.agents:
                diff = agent.leader_identity_history[t] - agent.follower_identity_history[t]
                diffs.append(abs(diff))
            role_diffs.append(np.mean(diffs))
        
        # Find first point where role diff stays above threshold
        threshold = 0.2
        window = 10
        for i in range(len(role_diffs) - window):
            if all(d > threshold for d in role_diffs[i:i+window]):
                return str(i)
        return "N/A"
    
    @output
    @render.text
    def perception_agreement():
        current = rv.get()
        if current['model'] is None:
            return "0.00"
        
        m = current['model']
        # Calculate agreement as correlation between agents' perceptions
        perceptions = []
        for agent in m.agents:
            row = []
            for other in m.agents:
                if agent != other:
                    edge_data = m.interaction_network.get_edge_data(agent.id, other.id)
                    weight = edge_data['weight'] if edge_data else 0
                    row.append(weight)
            perceptions.append(row)
        
        # Calculate average correlation between perception vectors
        correlations = []
        for i in range(len(perceptions)):
            for j in range(i+1, len(perceptions)):
                if len(perceptions[i]) > 0 and len(perceptions[j]) > 0:
                    corr = np.corrcoef(perceptions[i], perceptions[j])[0,1]
                    if not np.isnan(corr):
                        correlations.append(corr)
        
        return f"{np.mean(correlations) if correlations else 0:.2f}"
    
    @output
    @render.text
    def density():
        current = rv.get()
        if current['model'] is None:
            return "0.00"
        
        m = current['model']
        density = nx.density(m.interaction_network)
        return f"{density:.2f}"
    
    @output
    @render.text
    def centralization():
        current = rv.get()
        if current['model'] is None:
            return "0.00"
        
        m = current['model']
        # Calculate degree centralization
        n = len(m.agents)
        if n < 2:
            return "0.00"
        
        degrees = [d for n, d in m.interaction_network.degree()]
        max_degree = max(degrees)
        sum_of_differences = sum(max_degree - d for d in degrees)
        max_possible = (n-1) * (n-2)
        centralization = sum_of_differences / max_possible if max_possible > 0 else 0
        return f"{centralization:.2f}"
    
    @output
    @render.text
    def modularity():
        current = rv.get()
        if current['model'] is None:
            return "0.00"
        
        m = current['model']
        # Use community detection to calculate modularity
        communities = list(nx.community.greedy_modularity_communities(m.interaction_network.to_undirected()))
        modularity = nx.community.modularity(m.interaction_network.to_undirected(), communities)
        return f"{modularity:.2f}"
    
    @output
    @render.text
    def claim_grant_corr():
        current = rv.get()
        if current['model'] is None:
            return "0.00"
        
        m = current['model']
        # Calculate correlation between claiming and granting behaviors
        claims = []
        grants = []
        for agent in m.agents:
            claims.append(agent.leader_identity)
            grants.append(1 - agent.follower_identity)  # Convert to granting
        
        if len(claims) > 1:
            corr = np.corrcoef(claims, grants)[0,1]
            return f"{corr:.2f}"
        return "0.00"
    
    @output
    @render.text
    def identity_behavior_consistency():
        current = rv.get()
        if current['model'] is None:
            return "0.00"
        
        m = current['model']
        # Calculate correlation between identities and network positions
        li_values = [agent.leader_identity for agent in m.agents]
        fi_values = [agent.follower_identity for agent in m.agents]
        
        in_centrality = list(nx.in_degree_centrality(m.interaction_network).values())
        out_centrality = list(nx.out_degree_centrality(m.interaction_network).values())
        
        if len(li_values) > 1:
            li_corr = np.corrcoef(li_values, in_centrality)[0,1]
            fi_corr = np.corrcoef(fi_values, out_centrality)[0,1]
            return f"LI: {li_corr:.2f}, FI: {fi_corr:.2f}"
        return "0.00"
    
    @output
    @render.text
    def interaction_details():
        """Display details about the current/last interaction."""
        current = rv.get()
        if current['model'] is None:
            return "Initialize simulation to see interactions"
        
        m = current['model']
        if not hasattr(m, 'last_interaction'):
            return "No interactions yet"
        
        agent1_id, agent2_id = m.last_interaction
        agent1 = m.agents[agent1_id]
        agent2 = m.agents[agent2_id]
        
        return (
            f"Step {current['current_step']}\n"
            f"Agent {agent1_id} (LI: {agent1.leader_identity:.2f}, FI: {agent1.follower_identity:.2f})\n"
            f"interacted with\n"
            f"Agent {agent2_id} (LI: {agent2.leader_identity:.2f}, FI: {agent2.follower_identity:.2f})"
        )
    
    @output
    @render.ui
    def agent_details():
        """Display detailed information about all agents."""
        current = rv.get()
        if current['model'] is None:
            return ui.p("Initialize simulation to see agent details")
        
        m = current['model']
        cards = []
        
        for agent in m.agents:
            # Determine if this agent was involved in the last interaction
            is_active = False
            if hasattr(m, 'last_interaction'):
                is_active = agent.id in m.last_interaction
            
            style = {
                "margin": "5px",
                "padding": "10px",
                "border": "2px solid #4CAF50" if is_active else "1px solid #dee2e6",
                "background-color": "#f0fff0" if is_active else "#ffffff"
            }
            
            cards.append(
                ui.div(
                    ui.h4(f"Agent {agent.id}"),
                    ui.p(f"Leader Identity: {agent.leader_identity:.2f}"),
                    ui.p(f"Follower Identity: {agent.follower_identity:.2f}"),
                    ui.p(f"Role Differentiation: {agent.leader_identity - agent.follower_identity:.2f}"),
                    style="; ".join(f"{k}: {v}" for k, v in style.items())
                )
            )
        
        return ui.div(cards)
    
    @output
    @render.plot
    def step_network_plot():
        """Display the current state of the interaction network."""
        current = rv.get()
        if current['model'] is None or not hasattr(current['model'], 'interaction_network'):
            return plt.figure()
        
        m = current['model']
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Use stored positions or create new ones
        if current['network_pos'] is None:
            pos = nx.spring_layout(m.interaction_network)
            current['network_pos'] = pos
            rv.set(current)
        else:
            pos = current['network_pos']
        
        # Draw the network with larger elements for better visibility
        nx.draw_networkx_edges(m.interaction_network, pos, alpha=0.2, width=2)
        
        # Draw nodes with size based on leader identity
        node_colors = [agent.leader_identity for agent in m.agents]
        nodes = nx.draw_networkx_nodes(m.interaction_network, pos, 
                                     node_color=node_colors, 
                                     node_size=2000,
                                     cmap=plt.cm.viridis)
        
        # Add labels with more information
        labels = {i: f"Agent {i}\nLI: {agent.leader_identity:.2f}\nFI: {agent.follower_identity:.2f}" 
                 for i, agent in enumerate(m.agents)}
        nx.draw_networkx_labels(m.interaction_network, pos, labels, font_size=10)
        
        # Highlight last interaction if available
        if hasattr(m, 'last_interaction'):
            ax.add_patch(Circle(pos[m.last_interaction[0]], 0.2, 
                              fill=False, color='red', linewidth=3))
            ax.add_patch(Circle(pos[m.last_interaction[1]], 0.2, 
                              fill=False, color='blue', linewidth=3))
            
            # Add interaction labels
            ax.text(pos[m.last_interaction[0]][0], pos[m.last_interaction[0]][1] + 0.25, 
                   "Leader", color='red', ha='center')
            ax.text(pos[m.last_interaction[1]][0], pos[m.last_interaction[1]][1] + 0.25, 
                   "Follower", color='blue', ha='center')
        
        ax.set_title(f"Interaction Network (Step {current['current_step']})")
        plt.colorbar(nodes, label="Leader Identity")
        return fig
    
    @output
    @render.plot
    def step_history_plot():
        """Display recent history of identity changes."""
        current = rv.get()
        if current['model'] is None:
            return plt.figure()
        
        m = current['model']
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot only the most recent steps for clarity
        window = min(20, len(m.agents[0].leader_identity_history))
        for agent in m.agents:
            li_history = agent.leader_identity_history[-window:]
            fi_history = agent.follower_identity_history[-window:]
            steps = range(max(0, current['current_step'] - window + 1), current['current_step'] + 1)
            
            # Highlight active agents
            alpha = 1.0
            if hasattr(m, 'last_interaction'):
                alpha = 1.0 if agent.id in m.last_interaction else 0.3
            
            ax.plot(steps, li_history, '-', label=f"Agent {agent.id} LI", alpha=alpha)
            ax.plot(steps, fi_history, '--', label=f"Agent {agent.id} FI", alpha=alpha)
        
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Identity Strength")
        ax.set_title("Recent Identity Changes")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        return fig

app = App(app_ui, server)
