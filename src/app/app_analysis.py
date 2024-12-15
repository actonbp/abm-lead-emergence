"""
Analysis app for Leadership Emergence Simulation.
Focused on running full simulations and analyzing results.
"""

import os
import sys
from pathlib import Path
import json
import yaml
import numpy as np
import networkx as nx
from scipy import stats

# Ensure proper path handling
root_dir = Path(__file__).parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from shiny import App, render, ui, reactive
import matplotlib.pyplot as plt
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
    ui.panel_title("Leadership Emergence Analysis"),
    
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
    
    # Analysis Interface
    ui.div(
        {"id": "analysis_interface", "style": "display: none"},
        ui.row(
            ui.column(
                3,
                ui.card(
                    ui.card_header("Analysis Parameters"),
                    ui.card_body(
                        ui.h4("Model Configuration"),
                        ui.output_text("selected_config_info"),
                        ui.br(),
                        
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
                        
                        # Controls
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
                        ),
                        ui.br(),
                        ui.br(),
                        ui.input_action_button(
                            "reset_sim", 
                            "Reset", 
                            class_="btn-warning btn-block"
                        )
                    )
                )
            ),
            ui.column(
                9,
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
        'li_change_rate': 2.0,
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
        'validation_results': None,
        'history': None
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
                'validation_results': None,
                'history': None
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
            'validation_results': None,
            'history': None
        })
        await toggle_visibility("analysis_interface", True)
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
            'validation_results': None,
            'history': None
        })
        await toggle_visibility("analysis_interface", True)
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
            'validation_results': None,
            'history': None
        })
        await toggle_visibility("analysis_interface", True)
        await toggle_visibility("model_selection_page", False)

    @reactive.Effect
    @reactive.event(input.reset_sim)
    async def reset_simulation():
        await toggle_visibility("analysis_interface", False)
        await toggle_visibility("model_selection_page", True)
        rv.set({
            'model': None,
            'current_step': 0,
            'network_pos': None,
            'agents': None,
            'selected_variant': None,
            'config': None,
            'validation_results': None,
            'history': None
        })
    
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
        model_params['n_agents'] = input.n_agents()
        
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
            'agents': model.agents,
            'history': history
        })
        rv.set(new_state)
    
    @output
    @render.text
    def mean_li():
        current = get_current_state()
        if current['agents'] is None:
            return "0.00"
        return f"{np.mean([agent.leader_identity for agent in current['agents']]):.2f}"
    
    @output
    @render.text
    def mean_fi():
        current = get_current_state()
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
        
        if current['network_pos'] is None:
            pos = nx.spring_layout(m.interaction_network)
            current['network_pos'] = pos
            rv.set(current)
        else:
            pos = current['network_pos']
        
        nx.draw_networkx_edges(m.interaction_network, pos, alpha=0.2, width=2)
        
        node_colors = [agent.leader_identity for agent in m.agents]
        nodes = nx.draw_networkx_nodes(m.interaction_network, pos, 
                                     node_color=node_colors, 
                                     node_size=2000,
                                     cmap=plt.cm.viridis)
        
        labels = {i: f"Agent {i}\nLI: {agent.leader_identity:.2f}\nFI: {agent.follower_identity:.2f}" 
                 for i, agent in enumerate(m.agents)}
        nx.draw_networkx_labels(m.interaction_network, pos, labels, font_size=10)
        
        ax.set_title("Final Interaction Network")
        plt.colorbar(nodes, label="Leader Identity")
        plt.tight_layout()
        return fig
    
    @output
    @render.text
    def density():
        current = get_current_state()
        if current['model'] is None:
            return "0.00"
        
        m = current['model']
        density = nx.density(m.interaction_network)
        return f"{density:.2f}"
    
    @output
    @render.text
    def centralization():
        current = get_current_state()
        if current['model'] is None:
            return "0.00"
        
        m = current['model']
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
    def selected_config_info():
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
        def download():
            current = get_current_state()
            if current['model'] is None:
                return None
            
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
            
            return json.dumps(results, indent=2)
        
        return download

app = App(app_ui, server) 