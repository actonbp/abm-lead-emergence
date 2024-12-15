"""
Shiny application for Leadership Emergence Simulation.
"""

import os
import sys
from pathlib import Path
import json
import yaml
import pandas as pd
import io
import numpy as np
import networkx as nx
from scipy import stats

# Ensure proper path handling
root_dir = Path(__file__).parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from shiny import App, render, ui, reactive
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import seaborn as sns

from src.models import (
    BaseLeadershipModel,
    SchemaModel,
    NetworkModel,
    SchemaNetworkModel
)
from src.simulation.runner import SimulationConfig, BatchRunner
from src.models.metrics import (
    calculate_identity_variance,
    calculate_perception_agreement,
    calculate_claiming_granting_correlation,
    calculate_network_metrics,
    calculate_emergence_lag,
    calculate_identity_behavior_consistency
)

# Create output directory if it doesn't exist
output_dir = root_dir / "outputs" / "validation_runs"
output_dir.mkdir(parents=True, exist_ok=True)

# Load configurations
CONFIG_DIR = root_dir / "config"
with open(CONFIG_DIR / "base.yaml", "r") as f:
    base_config = yaml.safe_load(f)

variant_configs = {}
variants_dir = CONFIG_DIR / "variants"
if variants_dir.exists():
    for variant_file in variants_dir.glob("*.yaml"):
        with open(variant_file, "r") as f:
            variant_configs[variant_file.stem] = yaml.safe_load(f)

app_ui = ui.page_fluid(
    ui.panel_title("Leadership Emergence Simulation"),
    
    # Add JavaScript handler for visibility toggling
    ui.tags.head(
        ui.tags.script("""
            $(document).ready(function() {
                Shiny.addCustomMessageHandler('toggleVisibility', function(message) {
                    $('#' + message.id).css('display', message.display);
                });
            });
        """)
    ),
    
    # Model Selection Landing Page
    ui.div(
        {"id": "model_selection_page"},
        ui.h2("Select Model Variant"),
        ui.row(
            ui.column(
                3,
                ui.card(
                    ui.card_header("SIP Hierarchical"),
                    ui.card_body(
                        variant_configs["sip_hierarchical"]["description"],
                        ui.br(),
                        ui.br(),
                        ui.input_action_button("select_sip", "Select SIP Model", class_="btn-primary btn-block")
                    )
                )
            ),
            ui.column(
                3,
                ui.card(
                    ui.card_header("SCP Dynamic"),
                    ui.card_body(
                        variant_configs["scp_dynamic"]["description"],
                        ui.br(),
                        ui.br(),
                        ui.input_action_button("select_scp", "Select SCP Model", class_="btn-primary btn-block")
                    )
                )
            ),
            ui.column(
                3,
                ui.card(
                    ui.card_header("SI Prototype"),
                    ui.card_body(
                        variant_configs["si_prototype"]["description"],
                        ui.br(),
                        ui.br(),
                        ui.input_action_button("select_si", "Select SI Model", class_="btn-primary btn-block")
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
                        ui.h4("Model Configuration"),
                        ui.output_text("selected_config_info"),
                        ui.br(),
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
                        
                        # Core Parameters
                        ui.input_slider(
                            "n_agents", 
                            "Number of Agents", 
                            min=2, 
                            max=10, 
                            value=6
                        ),
                        ui.input_slider(
                            "n_steps", 
                            "Number of Steps", 
                            min=10, 
                            max=200, 
                            value=100
                        ),
                        
                        # Validation Thresholds
                        ui.h4("Validation Thresholds"),
                        ui.output_ui("validation_thresholds"),
                        
                        # Controls - Analysis Mode
                        ui.panel_conditional(
                            "input.sim_mode === 'analysis'",
                            ui.div(
                                ui.card(
                                    ui.card_header("Analysis Controls"),
                                    ui.card_body(
                                        ui.p("Run a complete simulation and analyze results."),
                                        ui.input_action_button(
                                            "run_sim", 
                                            "Run Full Simulation", 
                                            class_="btn-success btn-lg btn-block"
                                        ),
                                        ui.br(),
                                        ui.br(),
                                        ui.download_button(
                                            "download_results",
                                            "Download Results",
                                            class_="btn-info btn-block"
                                        )
                                    )
                                )
                            )
                        ),
                        
                        # Controls - Step Mode
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
                ui.panel_conditional(
                    "input.sim_mode === 'step'",
                    ui.navset_tab(
                        ui.nav_panel(
                            "ABM Visualization",
                            ui.row(
                                ui.column(
                                    12,
                                    ui.card(
                                        ui.card_header("Agent Interaction Network"),
                                        ui.card_body(ui.output_plot("step_network_plot"))
                                    )
                                )
                            ),
                            ui.row(
                                ui.column(
                                    6,
                                    ui.card(
                                        ui.card_header("Recent Identity Changes"),
                                        ui.card_body(ui.output_plot("step_history_plot"))
                                    )
                                ),
                                ui.column(
                                    6,
                                    ui.card(
                                        ui.card_header("Agent Details"),
                                        ui.card_body(ui.output_ui("agent_details"))
                                    )
                                )
                            ),
                            ui.row(
                                ui.column(
                                    12,
                                    ui.card(
                                        ui.card_header("Interaction Details"),
                                        ui.card_body(ui.output_text("interaction_details"))
                                    )
                                )
                            )
                        )
                    )
                ),
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
                            "Network Analysis",
                            ui.row(
                                ui.column(12, ui.output_plot("network_plot"))
                            ),
                            ui.row(
                                ui.column(6, ui.value_box(
                                    "Network Density",
                                    ui.output_text("density"),
                                    theme="primary"
                                )),
                                ui.column(6, ui.value_box(
                                    "Centralization",
                                    ui.output_text("centralization"),
                                    theme="info"
                                ))
                            )
                        ),
                        ui.nav_panel(
                            "Validation Results",
                            ui.row(
                                ui.column(12, ui.output_ui("validation_results"))
                            ),
                            ui.row(
                                ui.column(12, ui.output_plot("variance_plot"))
                            )
                        ),
                        id="analysis_tabs"
                    )
                )
            )
        )
    )
)

def _map_config_to_model_params(config):
    """Map YAML configuration to model parameters."""
    sim_props = config.get('parameters', {}).get('simulation_properties', {})
    agent_props = config.get('parameters', {}).get('agent_properties', {})
    identity_props = agent_props.get('identity_representation', {})
    
    return {
        'n_agents': sim_props.get('group_size', 6),
        'initial_li_equal': identity_props.get('leader_identity', {}).get('initial_value', 50) == 50,
        'li_change_rate': 2.0,  # Default value
        'interaction_radius': 1.0,
        'memory_length': config.get('parameters', {}).get('interaction_rules', {}).get('memory_length', 0)
    }

def server(input, output, session):
    MODEL_CLASSES = {
        'sip_hierarchical': SchemaModel,
        'scp_dynamic': NetworkModel,
        'si_prototype': SchemaNetworkModel
    }
    
    rv = reactive.Value({
        'model': None,
        'current_step': 0,
        'network_pos': None,
        'agents': None,
        'selected_variant': None,
        'config': None,
        'validation_results': None
    })
    
    def get_current_state():
        """Safely get current state with defaults."""
        try:
            return rv.get()
        except:
            return {
                'model': None,
                'current_step': 0,
                'network_pos': None,
                'agents': None,
                'selected_variant': None,
                'config': None,
                'validation_results': None
            }
    
    async def toggle_visibility(element_id, show=True):
        """Helper function to show/hide elements"""
        await session.send_custom_message(
            'toggleVisibility',
            {'id': element_id, 'display': 'block' if show else 'none'}
        )
    
    # Model Selection Handlers
    @reactive.Effect
    @reactive.event(input.select_sip)
    async def select_sip_model():
        rv.set({
            'selected_variant': 'sip_hierarchical',
            'config': variant_configs['sip_hierarchical'],
            'model': None,
            'current_step': 0,
            'network_pos': None,
            'agents': None,
            'validation_results': None
        })
        await toggle_visibility("simulation_interface", True)
        await toggle_visibility("model_selection_page", False)

    @reactive.Effect
    @reactive.event(input.select_scp)
    async def select_scp_model():
        rv.set({
            'selected_variant': 'scp_dynamic',
            'config': variant_configs['scp_dynamic'],
            'model': None,
            'current_step': 0,
            'network_pos': None,
            'agents': None,
            'validation_results': None
        })
        await toggle_visibility("simulation_interface", True)
        await toggle_visibility("model_selection_page", False)

    @reactive.Effect
    @reactive.event(input.select_si)
    async def select_si_model():
        rv.set({
            'selected_variant': 'si_prototype',
            'config': variant_configs['si_prototype'],
            'model': None,
            'current_step': 0,
            'network_pos': None,
            'agents': None,
            'validation_results': None
        })
        await toggle_visibility("simulation_interface", True)
        await toggle_visibility("model_selection_page", False)

    @reactive.Effect
    @reactive.event(input.reset_sim)
    async def reset_simulation():
        # Return to model selection page
        await toggle_visibility("simulation_interface", False)
        await toggle_visibility("model_selection_page", True)
        rv.set({
            'model': None,
            'current_step': 0,
            'network_pos': None,
            'agents': None,
            'selected_variant': None,
            'config': None,
            'validation_results': None
        })
    
    @reactive.Effect
    @reactive.event(input.init_sim)
    def initialize_simulation():
        current = get_current_state()
        variant = current.get('selected_variant')
        if variant is None:
            return
            
        model_class = MODEL_CLASSES.get(variant, BaseLeadershipModel)
        config = current.get('config')
        if config is None:
            return
            
        model_params = _map_config_to_model_params(config)
        
        m = model_class(config=model_params)
        new_state = current.copy()
        new_state.update({
            'model': m,
            'current_step': 0,
            'network_pos': None,
            'agents': m.agents
        })
        rv.set(new_state)
    
    @reactive.Effect
    @reactive.event(input.step_sim)
    def step_simulation():
        current = get_current_state()
        model = current.get('model')
        if model is None:
            return
            
        agent1, agent2 = model._select_interaction_pair()
        model.last_interaction = (agent1.id, agent2.id)
        model.step()
        
        new_state = current.copy()
        new_state.update({
            'model': model,
            'current_step': current.get('current_step', 0) + 1,
            'agents': model.agents
        })
        rv.set(new_state)
    
    @reactive.Effect
    @reactive.event(input.run_sim)
    async def run_simulation():
        current = get_current_state()
        config = current.get('config')
        if config is None:
            return
        
        variant = current.get('selected_variant')
        if variant is None:
            return
            
        model_class = MODEL_CLASSES.get(variant, BaseLeadershipModel)
        
        # Map configuration to model parameters
        model_params = _map_config_to_model_params(config)
        model_params['n_agents'] = input.n_agents()  # Use slider value
        
        # Create simulation configuration
        sim_config = SimulationConfig(
            model_params=model_params,
            n_steps=input.n_steps(),
            random_seed=42
        )
        
        # Run simulation
        runner = BatchRunner(model_class, Path("outputs/validation_runs"))
        result_file = runner.run_single_simulation(
            sim_config,
            f"validation_{variant}",
            Path("outputs/validation_runs")
        )
        
        # Load results and calculate validation metrics
        with open(result_file) as f:
            results = json.load(f)
        
        # Create and store model for visualization
        model = model_class(config=model_params)
        
        # Extract history arrays
        history = results['history']
        li_history = np.array([state['leader_identities'] for state in history])
        fi_history = np.array([state['follower_identities'] for state in history])
        
        # Update model with history
        for t, (li_state, fi_state) in enumerate(zip(li_history, fi_history)):
            for agent, li, fi in zip(model.agents, li_state, fi_state):
                agent.leader_identity = li
                agent.follower_identity = fi
                agent.leader_identity_history.append(li)
                agent.follower_identity_history.append(fi)
        
        # Calculate metrics
        li_var, fi_var = calculate_identity_variance(li_history, fi_history)
        emergence_lag = calculate_emergence_lag(li_history, fi_history)
        
        # Create a temporary network from the final state
        final_network = nx.DiGraph()
        model_state = results['model_state']
        if 'interaction_network' in model_state:
            network_data = model_state['interaction_network']
            final_network.add_nodes_from(network_data['nodes'])
            for u, v, d in network_data['edges']:
                final_network.add_edge(u, v, **d)
        
        # Update model's network
        model.interaction_network = final_network
        
        # Calculate network metrics
        network_metrics = calculate_network_metrics(final_network)
        
        # Calculate identity-behavior consistency using final state
        claiming = np.sum(nx.to_numpy_array(final_network), axis=1)
        granting = np.sum(nx.to_numpy_array(final_network), axis=0)
        
        li_final = li_history[-1]
        fi_final = fi_history[-1]
        
        li_corr = stats.pearsonr(li_final, claiming)[0] if len(claiming) > 1 else 0
        fi_corr = stats.pearsonr(fi_final, granting)[0] if len(granting) > 1 else 0
        
        validation_results = {
            'identity_stabilization': float(np.mean(li_var)),
            'variance_reduction': float(np.mean(fi_var)),
            'emergence_lag': float(emergence_lag),
            'network_density': float(network_metrics['density']),
            'centralization': float(network_metrics['centralization']),
            'li_behavior_consistency': float(li_corr),
            'fi_behavior_consistency': float(fi_corr)
        }
        
        # Update state with results and model
        new_state = current.copy()
        new_state.update({
            'model': model,
            'validation_results': validation_results,
            'current_step': input.n_steps(),
            'network_pos': nx.spring_layout(final_network),
            'agents': model.agents
        })
        rv.set(new_state)
        
        # Force plot updates
        await session.send_custom_message(
            'toggleVisibility',
            {'id': 'analysis_tabs', 'display': 'block'}
        )

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
    @reactive.event(input.run_sim)
    def identity_plot():
        current = get_current_state()
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
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        return fig
    
    @output
    @render.plot
    @reactive.event(input.run_sim)
    def variance_plot():
        current = get_current_state()
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
        plt.tight_layout()
        return fig
    
    @output
    @render.plot
    @reactive.event(input.run_sim)
    def network_plot():
        current = get_current_state()
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
        
        ax.set_title("Final Interaction Network")
        plt.colorbar(nodes, label="Leader Identity")
        plt.tight_layout()
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

    @output
    @render.text
    def selected_config_info():
        """Display information about selected configuration."""
        current = get_current_state()
        config = current.get('config')
        if config is None:
            return "No configuration selected"
        
        return (
            f"Model: {config.get('model_name', 'Unknown')}\n"
            f"Theory: {config.get('theoretical_basis', 'Unknown')}\n"
        )

    @output
    @render.ui
    def validation_thresholds():
        """Display validation thresholds from config."""
        current = get_current_state()
        config = current.get('config')
        if config is None:
            return ui.div()
        
        thresholds = config.get('validation', {}).get('metrics_thresholds', {})
        
        return ui.div([
            ui.p(f"{metric}: {threshold}")
            for metric, threshold in thresholds.items()
        ])

    @output
    @render.ui
    def validation_results():
        """Display validation results compared to thresholds."""
        current = get_current_state()
        results = current.get('validation_results')
        if results is None:
            return ui.p("Run simulation to see validation results")
        
        config = current.get('config')
        if config is None:
            return ui.p("No configuration selected")
            
        thresholds = config.get('validation', {}).get('metrics_thresholds', {})
        
        cards = []
        for metric, value in results.items():
            threshold = thresholds.get(metric, "N/A")
            passed = "✓" if _check_threshold(value, threshold) else "✗"
            
            cards.append(
                ui.card(
                    ui.card_header(metric),
                    ui.card_body(
                        ui.p(f"Value: {value:.2f}"),
                        ui.p(f"Threshold: {threshold}"),
                        ui.p(f"Status: {passed}")
                    )
                )
            )
        
        return ui.div(
            ui.row([ui.column(4, card) for card in cards])
        )

    def _check_threshold(value, threshold):
        """Check if value meets threshold condition."""
        if threshold == "N/A":
            return True
        
        if ">=" in threshold:
            return value >= float(threshold.replace(">=", "").strip())
        elif "<=" in threshold:
            return value <= float(threshold.replace("<=", "").strip())
        elif "between" in threshold:
            low, high = map(float, threshold.replace("between", "").strip().split("and"))
            return low <= value <= high
        
        return True

    @output
    @render.download
    def download_results():
        """Create downloadable results file."""
        def download():
            current = rv.get()
            if current['model'] is None:
                return None
            
            # Prepare results dictionary
            results = {
                'model_variant': current['selected_variant'],
                'parameters': current['config']['parameters'],
                'metrics': current['validation_results'],
                'history': {
                    'leader_identities': [
                        agent.leader_identity_history 
                        for agent in current['model'].agents
                    ],
                    'follower_identities': [
                        agent.follower_identity_history
                        for agent in current['model'].agents
                    ]
                }
            }
            
            # Convert to JSON
            return json.dumps(results, indent=2)
        
        return download

    # Add mode selection handler
    @reactive.Effect
    @reactive.event(input.sim_mode)
    def handle_mode_change():
        current = get_current_state()
        new_state = current.copy()
        
        # Reset simulation state when changing modes
        new_state.update({
            'model': None,
            'current_step': 0,
            'network_pos': None,
            'agents': None,
            'validation_results': None
        })
        rv.set(new_state)

app = App(app_ui, server)
