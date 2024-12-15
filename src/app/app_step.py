"""
Step-by-step visualization app for Leadership Emergence Simulation.
"""

import os
import sys
from pathlib import Path
import json
import yaml
import numpy as np
import networkx as nx
import scipy.stats as stats

# Ensure proper path handling
root_dir = Path(__file__).parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from shiny import App, render, ui, reactive
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from src.models import (
    BaseLeadershipModel,
    SchemaModel,
    NetworkModel,
    SchemaNetworkModel
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
    ui.tags.head(
        ui.tags.style("""
            .app-container { padding: 20px; }
            .model-card {
                height: 100%;
                transition: transform 0.2s;
                border: 1px solid #e0e0e0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .model-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            }
            .model-card .card-header {
                background-color: #f8f9fa;
                border-bottom: 2px solid #007bff;
                font-weight: bold;
                padding: 15px;
            }
            .model-card .card-body {
                padding: 20px;
            }
            .btn-block {
                margin-top: 15px;
                padding: 10px;
            }
            .simulation-card {
                margin-bottom: 20px;
                border: 1px solid #e0e0e0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .simulation-card .card-header {
                background-color: #f8f9fa;
                border-bottom: 2px solid #28a745;
                font-weight: bold;
                padding: 15px;
            }
            .nav-tabs {
                border-bottom: 2px solid #dee2e6;
            }
            .nav-tabs .nav-link.active {
                border-bottom: 2px solid #007bff;
                font-weight: bold;
            }
            .control-button {
                width: 100%;
                margin: 5px 0;
                padding: 10px;
            }
            .model-info {
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 15px;
            }
            .step-count {
                font-size: 1.2em;
                font-weight: bold;
                color: #007bff;
                text-align: center;
                margin: 10px 0;
            }
            .plot-container {
                background-color: white;
                padding: 15px;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .metric-card {
                background-color: white;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 15px;
                border-left: 4px solid #007bff;
            }
            .metric-value {
                font-size: 1.5em;
                font-weight: bold;
                color: #007bff;
            }
            .metric-label {
                color: #666;
                font-size: 0.9em;
            }
            .validation-status {
                padding: 5px 10px;
                border-radius: 3px;
                font-weight: bold;
            }
            .validation-ok {
                background-color: #d4edda;
                color: #155724;
            }
            .validation-warning {
                background-color: #fff3cd;
                color: #856404;
            }
            .validation-error {
                background-color: #f8d7da;
                color: #721c24;
            }
            .agent-card {
                background-color: #f8f9fa;
                padding: 10px;
                margin-bottom: 10px;
                border-radius: 5px;
                border-left: 4px solid #28a745;
            }
            .agent-card.leader {
                border-left-color: #dc3545;
            }
            .agent-card.follower {
                border-left-color: #007bff;
            }
            .identity-value {
                font-weight: bold;
                color: #495057;
            }
            .trend-positive {
                color: #155724;
            }
            .trend-neutral {
                color: #856404;
            }
            .trend-negative {
                color: #721c24;
            }
            .parameter-table {
                margin: 15px 0;
            }
            .parameter-table th {
                background-color: #f8f9fa;
                font-weight: bold;
            }
            .parameter-table td {
                vertical-align: middle;
            }
            .validation-metrics th {
                background-color: #f8f9fa;
                font-weight: bold;
            }
            .validation-status {
                display: inline-block;
                width: 24px;
                height: 24px;
                line-height: 24px;
                text-align: center;
                border-radius: 50%;
                font-weight: bold;
            }
            .validation-ok {
                background-color: #d4edda;
                color: #155724;
            }
            .validation-error {
                background-color: #f8d7da;
                color: #721c24;
            }
            .validation-warning {
                background-color: #fff3cd;
                color: #856404;
            }
            .metric-value {
                font-size: 1.2em;
                font-weight: bold;
            }
            .metric-threshold {
                color: #6c757d;
                font-size: 0.9em;
            }
            .model-parameter {
                font-weight: bold;
                color: #495057;
            }
            .parameter-description {
                color: #6c757d;
                font-style: italic;
            }
            .metric-box {
                background-color: #f8f9fa;
                border-radius: 5px;
                padding: 10px;
                text-align: center;
                margin-bottom: 10px;
                border-left: 4px solid #007bff;
            }
            .metric-label {
                font-size: 0.9em;
                color: #666;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            .metric-value {
                font-size: 1.5em;
                font-weight: bold;
                display: inline-block;
                margin: 5px 0;
            }
            .validation-icon {
                font-size: 1.2em;
                margin-left: 5px;
            }
            .validation-ok .metric-value,
            .validation-ok .validation-icon {
                color: #28a745;
            }
            .validation-warning .metric-value,
            .validation-warning .validation-icon {
                color: #ffc107;
            }
            .validation-error .metric-value,
            .validation-error .validation-icon {
                color: #dc3545;
            }
            .validation-card {
                background-color: #fff;
                border-radius: 5px;
                padding: 15px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .validation-messages {
                margin-top: 10px;
                padding: 10px;
                border-radius: 5px;
            }
            .validation-message {
                margin: 5px 0;
                padding: 5px 10px;
                border-radius: 3px;
            }
            .validation-message:before {
                margin-right: 8px;
                font-weight: bold;
            }
            .validation-ok .validation-message {
                background-color: #d4edda;
                color: #155724;
            }
            .validation-warning .validation-message {
                background-color: #fff3cd;
                color: #856404;
            }
            .validation-error .validation-message {
                background-color: #f8d7da;
                color: #721c24;
            }
            .metric-info {
                position: relative;
                display: inline-block;
                cursor: help;
            }
            
            .metric-info .tooltip-text {
                visibility: hidden;
                min-width: 300px;
                max-width: 400px;
                width: auto;
                background-color: #f8f9fa;
                color: #212529;
                text-align: left;
                border-radius: 6px;
                padding: 15px;
                border: 1px solid #dee2e6;
                box-shadow: 0 4px 8px rgba(0,0,0,0.15);
                
                /* Position the tooltip */
                position: absolute;
                z-index: 1000;
                bottom: 130%;
                left: 0;
                transform: translateX(-25%);
                
                /* Fade in */
                opacity: 0;
                transition: opacity 0.2s;
                
                /* Ensure text wrapping */
                white-space: normal;
                word-wrap: break-word;
            }
            
            /* Add arrow */
            .metric-info .tooltip-text::after {
                content: "";
                position: absolute;
                top: 100%;
                left: 30%;
                margin-left: -10px;
                border-width: 10px;
                border-style: solid;
                border-color: #f8f9fa transparent transparent transparent;
            }
            
            .metric-info:hover .tooltip-text {
                visibility: visible;
                opacity: 1;
            }
            
            .metric-definition {
                margin-bottom: 8px;
                font-size: 0.95em;
                line-height: 1.4;
            }
            
            .metric-example {
                font-style: italic;
                color: #6c757d;
                font-size: 0.9em;
                line-height: 1.4;
                margin-bottom: 8px;
                padding-left: 8px;
                border-left: 3px solid #e9ecef;
            }
            
            .metric-interpretation {
                font-weight: 500;
                color: #495057;
                font-size: 0.9em;
                margin-top: 8px;
                padding-top: 8px;
                border-top: 1px solid #e9ecef;
            }
            
            .info-icon {
                color: #007bff;
                margin-left: 5px;
                font-size: 0.9em;
            }
            
            /* Ensure tooltip stays visible when near table edges */
            .table td:first-child .tooltip-text {
                left: 0;
                transform: translateX(0);
            }
            
            .table td:last-child .tooltip-text {
                left: auto;
                right: 0;
                transform: translateX(0);
            }
        """)
    ),
    ui.panel_title(
        ui.h1(
            "Leadership Emergence Step-by-Step Simulation",
            class_="text-center mb-4"
        )
    ),
    
    ui.div(
        {"class": "app-container"},
        ui.output_ui("model_selection_page"),
        ui.output_ui("simulation_interface")
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

def calculate_entropy_metrics(model):
    """Calculate entropy-based metrics for hierarchy emergence."""
    n_agents = len(model.agents)
    
    # Calculate entropy based on leader identity distribution
    li_values = np.array([agent.leader_identity for agent in model.agents])
    
    # Normalize leader identities to probabilities for entropy calculation
    if np.sum(li_values) > 0:  # Only normalize if there are non-zero values
        li_probs = li_values / np.sum(li_values)
        system_entropy = -np.sum(li_probs * np.log2(li_probs + 1e-10))
    else:
        system_entropy = 0
    
    # Calculate leadership perception matrix based on actual model mechanics
    if hasattr(model, 'interaction_network'):
        # Initialize perceptions matrix
        perceptions = np.zeros((n_agents, n_agents))
        
        # Only consider direct grant interactions
        for i, j in model.interaction_network.edges():
            # Only count successful grants (weight indicates successful claims)
            edge_data = model.interaction_network[i][j]
            if edge_data.get('weight', 0) > 0:
                # j granted i's claim, so j perceives i as leader
                perceptions[j, i] = edge_data['weight']
    else:
        perceptions = np.zeros((n_agents, n_agents))
    
    # Calculate hierarchy metrics
    max_entropy = -np.log2(1/n_agents)
    hierarchy_clarity = 1 - (system_entropy / max_entropy) if max_entropy > 0 else 0
    
    # Calculate rank consensus based on leader identity rankings
    li_rankings = np.argsort(-li_values)  # Higher LI = higher rank
    rank_correlations = []
    
    # Only calculate correlations if there's enough variation in the values
    if len(set(li_values)) > 1:  # Check if there are at least 2 different values
        for i in range(n_agents):
            for j in range(i+1, n_agents):
                # Only calculate if both agents have different values
                if li_values[i] != li_values[j]:
                    try:
                        # Calculate correlation between their rankings of all agents
                        agent_i_ranks = np.argsort([-li_values[i], -li_values[j]])
                        agent_j_ranks = np.argsort([-li_values[j], -li_values[i]])
                        if not np.array_equal(agent_i_ranks, agent_j_ranks):  # Only if rankings differ
                            corr = stats.spearmanr(agent_i_ranks, agent_j_ranks)[0]
                            if not np.isnan(corr):
                                rank_correlations.append(corr)
                    except:
                        continue
    
    rank_consensus = np.mean(rank_correlations) if rank_correlations else 0
    
    # Calculate structural stability from interaction network
    if hasattr(model, 'interaction_network'):
        degrees = [d for n, d in model.interaction_network.degree()]
        structural_stability = 1 - (np.std(degrees) / np.mean(degrees) if np.mean(degrees) > 0 else 0)
    else:
        structural_stability = 0
    
    return {
        'system_entropy': system_entropy,
        'hierarchy_clarity': hierarchy_clarity,
        'rank_consensus': rank_consensus,
        'individual_entropies': [0] * n_agents,  # No longer using individual entropies
        'structural_stability': structural_stability,
        'perception_matrix': perceptions
    }

def check_simulation_validity(model):
    """Enhanced validation checks with sophisticated criteria."""
    metrics = calculate_entropy_metrics(model)
    li_values = [agent.leader_identity for agent in model.agents]
    fi_values = [agent.follower_identity for agent in model.agents]
    
    messages = []
    status_class = 'validation-ok'
    
    # Value range checks
    if any(li > 100 or li < 0 for li in li_values) or any(fi > 100 or fi < 0 for fi in fi_values):
        messages.append('Error: Identity values out of valid range (0-100)')
        status_class = 'validation-error'
    
    # Variance checks
    li_std = np.std(li_values)
    fi_std = np.std(fi_values)
    if li_std < 0.01 or fi_std < 0.01:
        messages.append('Warning: Very low variance in identities - possible stagnation')
        status_class = 'validation-warning'
    elif li_std > 30 or fi_std > 30:
        if metrics['structural_stability'] < 0.3:
            messages.append('Warning: High variance with unstable structure')
            status_class = 'validation-warning'
    
    # Hierarchy emergence checks
    if metrics['hierarchy_clarity'] < 0.2:
        messages.append('Warning: No clear hierarchy emerging')
        status_class = 'validation-warning'
    
    # Role differentiation checks
    role_diffs = [li - fi for li, fi in zip(li_values, fi_values)]
    max_diff = max(abs(d) for d in role_diffs)
    if max_diff < 2:
        if metrics['rank_consensus'] < 0.3:
            messages.append('Warning: Low role differentiation with poor consensus')
            status_class = 'validation-warning'
    
    # Structural checks
    if metrics['structural_stability'] < 0.2:
        messages.append('Warning: Unstable interaction patterns')
        status_class = 'validation-warning'
    
    # Add positive validations
    if metrics['hierarchy_clarity'] > 0.6:
        messages.append('✓ Clear hierarchy has emerged')
    if metrics['rank_consensus'] > 0.7:
        messages.append('✓ Strong agreement on leadership structure')
    if metrics['structural_stability'] > 0.7:
        messages.append('✓ Stable interaction patterns established')
    
    if not messages:
        messages.append('All validation checks passed')
    
    return {
        'class': status_class,
        'message': '\n'.join(messages),
        'metrics': {
            'hierarchy_clarity': metrics['hierarchy_clarity'],
            'rank_consensus': metrics['rank_consensus'],
            'system_entropy': metrics['system_entropy'],
            'structural_stability': metrics['structural_stability']
        }
    }

def server(input, output, session):
    MODEL_CLASSES = {
        'base_derue': BaseLeadershipModel,
        'sip_hierarchical': SchemaModel,
        'scp_dynamic': NetworkModel,
        'si_prototype': SchemaNetworkModel
    }
    
    # Initialize reactive values
    page_state = reactive.Value('selection')
    model_state = reactive.Value({
        'model': None,
        'current_step': 0,
        'network_pos': None,
        'agents': None,
        'selected_variant': None,
        'config': None,
    })
    
    def get_current_state():
        """Safely get current state with defaults."""
        try:
            return model_state.get()
        except:
            return {
                'model': None,
                'current_step': 0,
                'network_pos': None,
                'agents': None,
                'selected_variant': None,
                'config': None,
            }
    
    @output
    @render.ui
    @reactive.event(page_state)
    def model_selection_page():
        if page_state.get() != 'selection':
            return ui.div()
        
        return ui.div(
            ui.h2("Select Model Variant", class_="text-center mb-4"),
            ui.row(
                ui.column(
                    3,
                    ui.div(
                        {"class": "model-card"},
                        ui.card(
                            ui.card_header("Base DeRue Model"),
                            ui.card_body(
                                ui.h4("Core Leadership Emergence"),
                                ui.p(
                                    "Original model with basic mechanisms: "
                                    "leadership characteristics, ILTs, claims/grants, "
                                    "and identity development through interactions."
                                ),
                                ui.div(
                                    ui.input_action_button(
                                        "select_base",
                                        "Select Base Model",
                                        class_="btn-primary btn-block"
                                    )
                                )
                            )
                        )
                    )
                ),
                ui.column(
                    3,
                    ui.div(
                        {"class": "model-card"},
                        ui.card(
                            ui.card_header("SIP Hierarchical"),
                            ui.card_body(
                                ui.h4("Social-Interactionist Perspective"),
                                ui.p(variant_configs["sip_hierarchical"]["description"]),
                                ui.div(
                                    ui.input_action_button(
                                        "select_sip",
                                        "Select SIP Model",
                                        class_="btn-primary btn-block"
                                    )
                                )
                            )
                        )
                    )
                ),
                ui.column(
                    3,
                    ui.div(
                        {"class": "model-card"},
                        ui.card(
                            ui.card_header("SCP Dynamic"),
                            ui.card_body(
                                ui.h4("Social-Cognitive Perspective"),
                                ui.p(variant_configs["scp_dynamic"]["description"]),
                                ui.div(
                                    ui.input_action_button(
                                        "select_scp",
                                        "Select SCP Model",
                                        class_="btn-primary btn-block"
                                    )
                                )
                            )
                        )
                    )
                ),
                ui.column(
                    3,
                    ui.div(
                        {"class": "model-card"},
                        ui.card(
                            ui.card_header("SI Prototype"),
                            ui.card_body(
                                ui.h4("Social Identity Theory"),
                                ui.p(variant_configs["si_prototype"]["description"]),
                                ui.div(
                                    ui.input_action_button(
                                        "select_si",
                                        "Select SI Model",
                                        class_="btn-primary btn-block"
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
    
    @output
    @render.ui
    @reactive.event(page_state)
    def simulation_interface():
        if page_state.get() != 'simulation':
            return ui.div()
        
        return ui.div(
            ui.row(
                ui.column(
                    3,
                    ui.div(
                        {"class": "simulation-card"},
                        ui.card(
                            ui.card_header("Simulation Parameters"),
                            ui.card_body(
                                ui.div(
                                    {"class": "model-info"},
                                    ui.h4("Model Configuration"),
                                    ui.output_text("selected_config_info")
                                ),
                                
                                ui.div(
                                    {"class": "simulation-controls mt-4"},
                                    ui.input_action_button(
                                        "init_sim", 
                                        "Initialize Simulation", 
                                        class_="btn-primary control-button"
                                    ),
                                    ui.input_action_button(
                                        "step_sim", 
                                        "Step Forward", 
                                        class_="btn-info control-button mt-2"
                                    ),
                                    ui.input_action_button(
                                        "reset_sim", 
                                        "Reset", 
                                        class_="btn-warning control-button mt-2"
                                    )
                                ),
                                
                                ui.div(
                                    {"class": "step-count mt-3"},
                                    ui.output_text("sim_status")
                                )
                            )
                        )
                    )
                ),
                ui.column(
                    9,
                    ui.navset_tab(
                        ui.nav_panel(
                            "ABM Visualization",
                            ui.div(
                                {"class": "plot-container"},
                                ui.row(
                                    ui.column(
                                        6,
                                        ui.card(
                                            ui.card_header("Agent Interaction Network"),
                                            ui.card_body(ui.output_plot("step_network_plot"))
                                        )
                                    ),
                                    ui.column(
                                        6,
                                        ui.card(
                                            ui.card_header("Leadership Perceptions"),
                                            ui.card_body(ui.output_plot("perception_network_plot"))
                                        )
                                    )
                                ),
                                ui.row(
                                    ui.column(
                                        6,
                                        ui.card(
                                            ui.card_header("Agent Details"),
                                            ui.card_body(ui.output_ui("agent_details"))
                                        )
                                    ),
                                    ui.column(
                                        6,
                                        ui.card(
                                            ui.card_header("Interaction Details"),
                                            ui.card_body(ui.output_ui("interaction_details"))
                                        )
                                    )
                                )
                            )
                        ),
                        ui.nav_panel(
                            "Model Parameters",
                            ui.div(
                                {"class": "plot-container"},
                                ui.row(
                                    ui.column(
                                        12,
                                        ui.card(
                                            ui.card_header("Core Model Parameters"),
                                            ui.card_body(
                                                ui.div(
                                                    {"class": "parameter-table"},
                                                    ui.tags.table(
                                                        {"class": "table table-striped"},
                                                        ui.tags.thead(
                                                            ui.tags.tr(
                                                                ui.tags.th("Parameter"),
                                                                ui.tags.th("Value"),
                                                                ui.tags.th("Description")
                                                            )
                                                        ),
                                                        ui.tags.tbody(
                                                            ui.output_ui("parameter_rows")
                                                        )
                                                    )
                                                )
                                            )
                                        )
                                    )
                                ),
                                ui.row(
                                    ui.column(
                                        12,
                                        ui.card(
                                            ui.card_header("Validation Metrics"),
                                            ui.card_body(
                                                ui.output_ui("validation_metrics_table")
                                            )
                                        )
                                    )
                                )
                            )
                        ),
                        ui.nav_panel(
                            "Step Logic",
                            ui.div(
                                {"class": "plot-container"},
                                ui.output_ui("step_logic_content")
                            )
                        )
                    )
                )
            )
        )
    
    # Model Selection Handlers
    @reactive.Effect
    @reactive.event(input.select_sip)
    def select_sip_model():
        print("SIP model selected")
        current = get_current_state()
        current.update({
            'selected_variant': 'sip_hierarchical',
            'config': variant_configs['sip_hierarchical'],
            'model': None,
            'current_step': 0,
            'network_pos': None,
            'agents': None,
        })
        model_state.set(current)
        page_state.set('simulation')
        print(f"Updated state - page: {page_state.get()}, model: {model_state.get()['selected_variant']}")

    @reactive.Effect
    @reactive.event(input.select_scp)
    def select_scp_model():
        current = get_current_state()
        current.update({
            'selected_variant': 'scp_dynamic',
            'config': variant_configs['scp_dynamic'],
            'model': None,
            'current_step': 0,
            'network_pos': None,
            'agents': None,
        })
        model_state.set(current)
        page_state.set('simulation')

    @reactive.Effect
    @reactive.event(input.select_si)
    def select_si_model():
        current = get_current_state()
        current.update({
            'selected_variant': 'si_prototype',
            'config': variant_configs['si_prototype'],
            'model': None,
            'current_step': 0,
            'network_pos': None,
            'agents': None,
        })
        model_state.set(current)
        page_state.set('simulation')

    @reactive.Effect
    @reactive.event(input.reset_sim)
    def reset_simulation():
        current = get_current_state()
        current.update({
            'model': None,
            'current_step': 0,
            'network_pos': None,
            'agents': None,
            'selected_variant': None,
            'config': None,
        })
        model_state.set(current)
        page_state.set('selection')
    
    @reactive.Effect
    @reactive.event(input.init_sim)
    def initialize_model():
        # First reset the current state
        current = {
            'model': None,
            'current_step': 0,
            'network_pos': None,
            'agents': None,
            'selected_variant': None,
            'config': None,
        }
        
        # Get current parameter values
        config = {
            'n_agents': input.n_agents(),
            'initial_li_equal': input.initial_li_equal(),
            'li_change_rate': input.identity_change_rate(),
            'identity_change_rate': input.identity_change_rate(),
            'ilt_match_algorithm': input.ilt_match_algorithm(),
            'ilt_match_params': {
                'sigma': input.gaussian_sigma(),
                'k': input.sigmoid_k(),
                'threshold': input.threshold_value()
            },
            'characteristics_range': input.characteristics_range(),
            'ilt_range': input.ilt_range(),
            'initial_identity': input.initial_identity(),
            'claim_multiplier': input.claim_multiplier(),
            'grant_multiplier': input.grant_multiplier()
        }
        
        # Initialize model with config
        model = BaseLeadershipModel(config)
        
        # Update state with new model and config
        current.update({
            'config': config,
            'model': model,
            'current_step': 0,
            'network_pos': None,
            'agents': model.agents  # Initialize agents from the new model
        })
        
        model_state.set(current)
    
    @reactive.Effect
    @reactive.event(input.step_sim)
    def step_simulation():
        current = get_current_state()
        model = current.get('model')
        if model is None:
            return
            
        # Select interaction pair
        agent1, agent2 = model._select_interaction_pair()
        model.last_interaction = (agent1.id, agent2.id)
        
        # Calculate match between characteristics and ILTs
        # For claims: match against own ILT
        agent1_self_match = 1 - abs(agent1.characteristics - agent1.ilt) / 100  # Match against own ILT
        agent2_self_match = 1 - abs(agent2.characteristics - agent2.ilt) / 100  # Match against own ILT
        
        # For grants: match against other's ILT
        agent1_other_match = 1 - abs(agent1.characteristics - agent2.ilt) / 100  # Match against other's ILT
        agent2_other_match = 1 - abs(agent2.characteristics - agent1.ilt) / 100  # Match against other's ILT
        
        # Store match values for display
        model.last_agent1_match = agent1_other_match  # Show match that matters for granting
        model.last_agent2_match = agent2_other_match  # Show match that matters for granting
        
        # Probabilistic claiming based on self-match
        # Higher self-match = higher chance to claim
        agent1_claim_prob = agent1_self_match
        agent2_claim_prob = agent2_self_match
        
        # Make claim decisions probabilistically
        agent1_claims = model.rng.random() < agent1_claim_prob
        agent2_claims = model.rng.random() < agent2_claim_prob
        
        # Store claim decisions and probabilities for display
        model.last_agent1_claimed = agent1_claims
        model.last_agent2_claimed = agent2_claims
        model.last_agent1_claim_prob = agent1_claim_prob
        model.last_agent2_claim_prob = agent2_claim_prob
        
        # Calculate grant probabilities based on other-match
        agent1_grant_prob = 0
        agent2_grant_prob = 0
        if agent1_claims:
            agent2_grant_prob = agent1_other_match  # How well agent1 matches agent2's ILT
        if agent2_claims:
            agent1_grant_prob = agent2_other_match  # How well agent2 matches agent1's ILT
            
        # Make grant decisions probabilistically
        agent1_grants = False
        agent2_grants = False
        if agent2_claims:
            agent1_grants = model.rng.random() < agent1_grant_prob
        if agent1_claims:
            agent2_grants = model.rng.random() < agent2_grant_prob
            
        # Store grant decisions and probabilities
        model.last_agent1_granted = agent1_grants
        model.last_agent2_granted = agent2_grants
        model.last_agent1_grant_prob = agent1_grant_prob
        model.last_agent2_grant_prob = agent2_grant_prob
        
        # Update model state based on claims and grants
        grant_given = (agent1_claims and agent2_grants) or (agent2_claims and agent1_grants)
        model.last_grant_given = grant_given
        
        # Update identities and perceptions based on interaction outcome
        if grant_given:
            # Initialize interaction network if it doesn't exist
            if not hasattr(model, 'interaction_network'):
                model.interaction_network = nx.DiGraph()
                for i in range(len(model.agents)):
                    model.interaction_network.add_node(i)
            
            if agent1_claims and agent2_grants:
                # Update identities
                agent1.leader_identity = min(100, agent1.leader_identity + model.li_change_rate)
                agent2.follower_identity = min(100, agent2.follower_identity + model.li_change_rate)
                # Update leadership perceptions - agent2 perceives agent1 as leader
                if not model.interaction_network.has_edge(agent2.id, agent1.id):
                    model.interaction_network.add_edge(agent2.id, agent1.id, weight=1)
                else:
                    model.interaction_network[agent2.id][agent1.id]['weight'] += 1
                    
            if agent2_claims and agent1_grants:
                # Update identities
                agent2.leader_identity = min(100, agent2.leader_identity + model.li_change_rate)
                agent1.follower_identity = min(100, agent1.follower_identity + model.li_change_rate)
                # Update leadership perceptions - agent1 perceives agent2 as leader
                if not model.interaction_network.has_edge(agent1.id, agent2.id):
                    model.interaction_network.add_edge(agent1.id, agent2.id, weight=1)
                else:
                    model.interaction_network[agent1.id][agent2.id]['weight'] += 1
        else:
            if agent1_claims:
                agent1.leader_identity = max(0, agent1.leader_identity - model.li_change_rate)
            if agent2_claims:
                agent2.leader_identity = max(0, agent2.leader_identity - model.li_change_rate)
        
        new_state = current.copy()
        new_state.update({
            'model': model,
            'current_step': current.get('current_step', 0) + 1,
            'agents': model.agents
        })
        model_state.set(new_state)
    
    @output
    @render.text
    def sim_status():
        current = get_current_state()
        if current['model'] is None:
            return "Ready to initialize simulation."
        return f"Step {current['current_step']}"
    
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
    def agent_details():
        current = get_current_state()
        if current['model'] is None:
            return ui.p("Initialize simulation to see agent details")
        
        m = current['model']
        metrics = calculate_entropy_metrics(m)
        validation = check_simulation_validity(m)
        
        # Define thresholds for metrics
        thresholds = {
            'hierarchy_clarity': 0.5,
            'rank_consensus': 0.6,
            'structural_stability': 0.7,
            'system_entropy': 2.0  # Lower is better
        }
        
        return ui.div(
            ui.div(
                {"class": "metric-card"},
                ui.h4("Hierarchy Structure"),
                ui.div(
                    ui.div(
                        {"class": "row"},
                        ui.div(
                            {"class": "col-4"},
                            ui.div(
                                {"class": "metric-box"},
                                ui.p(
                                    ui.span("Clarity", class_="metric-label"),
                                    ui.br(),
                                    ui.span(
                                        f"{metrics['hierarchy_clarity']:.2f}",
                                        class_=f"metric-value {_get_validation_class(metrics['hierarchy_clarity'], thresholds['hierarchy_clarity'])}"
                                    ),
                                    ui.span(
                                        " ✓" if metrics['hierarchy_clarity'] >= thresholds['hierarchy_clarity'] else " ✗",
                                        class_=f"validation-icon {_get_validation_class(metrics['hierarchy_clarity'], thresholds['hierarchy_clarity'])}"
                                    )
                                )
                            )
                        ),
                        ui.div(
                            {"class": "col-4"},
                            ui.div(
                                {"class": "metric-box"},
                                ui.p(
                                    ui.span("Consensus", class_="metric-label"),
                                    ui.br(),
                                    ui.span(
                                        f"{metrics['rank_consensus']:.2f}",
                                        class_=f"metric-value {_get_validation_class(metrics['rank_consensus'], thresholds['rank_consensus'])}"
                                    ),
                                    ui.span(
                                        " ✓" if metrics['rank_consensus'] >= thresholds['rank_consensus'] else " ✗",
                                        class_=f"validation-icon {_get_validation_class(metrics['rank_consensus'], thresholds['rank_consensus'])}"
                                    )
                                )
                            )
                        ),
                        ui.div(
                            {"class": "col-4"},
                            ui.div(
                                {"class": "metric-box"},
                                ui.p(
                                    ui.span("Stability", class_="metric-label"),
                                    ui.br(),
                                    ui.span(
                                        f"{metrics['structural_stability']:.2f}",
                                        class_=f"metric-value {_get_validation_class(metrics['structural_stability'], thresholds['structural_stability'])}"
                                    ),
                                    ui.span(
                                        " ✓" if metrics['structural_stability'] >= thresholds['structural_stability'] else " ✗",
                                        class_=f"validation-icon {_get_validation_class(metrics['structural_stability'], thresholds['structural_stability'])}"
                                    )
                                )
                            )
                        )
                    ),
                    ui.div(
                        {"class": "row mt-2"},
                        ui.div(
                            {"class": "col-12"},
                            ui.div(
                                {"class": "metric-box"},
                                ui.p(
                                    ui.span("System Entropy", class_="metric-label"),
                                    ui.br(),
                                    ui.span(
                                        f"{metrics['system_entropy']:.2f}",
                                        class_=f"metric-value {_get_validation_class(metrics['system_entropy'], thresholds['system_entropy'], lower_is_better=True)}"
                                    ),
                                    ui.span(
                                        " ✓" if metrics['system_entropy'] <= thresholds['system_entropy'] else " ✗",
                                        class_=f"validation-icon {_get_validation_class(metrics['system_entropy'], thresholds['system_entropy'], lower_is_better=True)}"
                                    )
                                )
                            )
                        )
                    )
                )
            ),
            ui.div(
                {"class": "validation-card mt-3"},
                ui.h4("Validation Messages"),
                ui.div(
                    {"class": f"validation-messages {validation['class']}"},
                    *[ui.p(msg, class_="validation-message") for msg in validation['message'].split('\n')]
                )
            )
        )
    
    @output
    @render.ui
    def interaction_details():
        current = get_current_state()
        if current['model'] is None:
            return ui.p("Initialize simulation to see interactions")
        
        m = current['model']
        if not hasattr(m, 'last_interaction'):
            return ui.p("No interactions yet")
        
        agent1_id, agent2_id = m.last_interaction
        agent1 = m.agents[agent1_id]
        agent2 = m.agents[agent2_id]
        
        # Get claim/grant information from the model if available
        agent1_claimed = getattr(m, 'last_agent1_claimed', False)
        agent2_claimed = getattr(m, 'last_agent2_claimed', False)
        grant_given = getattr(m, 'last_grant_given', False)
        
        # Create narrative sections for each step
        sections = []
        
        # Selection narrative
        selection_div = ui.div(
            {"class": "interaction-step"},
            ui.h4({"class": "step-header"}, "👥 Agent Selection"),
            ui.p(
                f"Agent {agent1_id} and Agent {agent2_id} were selected for interaction.",
                ui.br(),
                ui.span(
                    f"Initial states: ",
                    ui.br(),
                    f"Agent {agent1_id}: LI={agent1.leader_identity:.1f}, FI={agent1.follower_identity:.1f}",
                    ui.br(),
                    f"Agent {agent2_id}: LI={agent2.leader_identity:.1f}, FI={agent2.follower_identity:.1f}",
                    class_="detail-text"
                )
            )
        )
        sections.append(selection_div)
        
        # Claims and Grants narrative
        claims_text = []
        if agent1_claimed and agent2_claimed:
            claims_text.append(
                f"Both agents attempted to claim leadership. "
                f"(Agent {agent1_id}: {getattr(m, 'last_agent1_claim_prob', 0):.1%} claim chance, "
                f"Agent {agent2_id}: {getattr(m, 'last_agent2_claim_prob', 0):.1%} claim chance)"
            )
        elif agent1_claimed:
            claims_text.append(
                f"Agent {agent1_id} attempted to claim leadership ({getattr(m, 'last_agent1_claim_prob', 0):.1%} chance), "
                f"while Agent {agent2_id} did not ({getattr(m, 'last_agent2_claim_prob', 0):.1%} chance)."
            )
        elif agent2_claimed:
            claims_text.append(
                f"Agent {agent2_id} attempted to claim leadership ({getattr(m, 'last_agent2_claim_prob', 0):.1%} chance), "
                f"while Agent {agent1_id} did not ({getattr(m, 'last_agent1_claim_prob', 0):.1%} chance)."
            )
        else:
            claims_text.append(
                f"Neither agent attempted to claim leadership. "
                f"(Agent {agent1_id}: {getattr(m, 'last_agent1_claim_prob', 0):.1%} chance, "
                f"Agent {agent2_id}: {getattr(m, 'last_agent2_claim_prob', 0):.1%} chance)"
            )
        
        claims_div = ui.div(
            {"class": "interaction-step"},
            ui.h4({"class": "step-header"}, "👑 Leadership Claims & Grants"),
            ui.p(
                claims_text[0],
                ui.br(),
                ui.span(
                    f"Agent {agent1_id} ILT match: {getattr(m, 'last_agent1_match', 0):.1%}, "
                    f"Agent {agent2_id} ILT match: {getattr(m, 'last_agent2_match', 0):.1%}",
                    class_="detail-text"
                ),
                ui.br(),
                ui.span(
                    f"Grant probabilities - Agent {agent1_id}: {getattr(m, 'last_agent1_grant_prob', 0):.1%}, "
                    f"Agent {agent2_id}: {getattr(m, 'last_agent2_grant_prob', 0):.1%}",
                    class_="detail-text"
                ),
                ui.br(),
                ui.span(
                    "Each agent can both claim (based on own characteristics vs own ILT) "
                    "and grant (based on other's characteristics vs own ILT)",
                    class_="detail-text"
                )
            )
        )
        sections.append(claims_div)
        
        # Outcome narrative
        outcome_text = []
        agent1_granted = getattr(m, 'last_agent1_granted', False)
        agent2_granted = getattr(m, 'last_agent2_granted', False)
        
        if agent1_claimed and agent2_claimed:
            if agent1_granted and agent2_granted:
                outcome_text = [
                    "Both agents successfully established mutual leadership recognition.",
                    "Each agent's characteristics matched the other's leadership expectations."
                ]
            elif agent1_granted:
                outcome_text = [
                    f"Only Agent {agent2_id}'s leadership claim was granted.",
                    f"Agent {agent1_id} granted leadership based on ILT match ({getattr(m, 'last_agent1_grant_prob', 0):.1%} chance)."
                ]
            elif agent2_granted:
                outcome_text = [
                    f"Only Agent {agent1_id}'s leadership claim was granted.",
                    f"Agent {agent2_id} granted leadership based on ILT match ({getattr(m, 'last_agent2_grant_prob', 0):.1%} chance)."
                ]
            else:
                outcome_text = [
                    "Neither agent's leadership claim was granted.",
                    "Their characteristics did not sufficiently match each other's ILTs."
                ]
        elif agent1_claimed:
            if agent2_granted:
                outcome_text = [
                    f"Agent {agent1_id} successfully emerged as a leader.",
                    f"Agent {agent2_id} granted leadership based on ILT match ({getattr(m, 'last_agent2_grant_prob', 0):.1%} chance)."
                ]
            else:
                outcome_text = [
                    f"Agent {agent1_id}'s leadership claim was unsuccessful.",
                    f"Agent {agent2_id} did not grant leadership (grant chance: {getattr(m, 'last_agent2_grant_prob', 0):.1%})."
                ]
        elif agent2_claimed:
            if agent1_granted:
                outcome_text = [
                    f"Agent {agent2_id} successfully emerged as a leader.",
                    f"Agent {agent1_id} granted leadership based on ILT match ({getattr(m, 'last_agent1_grant_prob', 0):.1%} chance)."
                ]
            else:
                outcome_text = [
                    f"Agent {agent2_id}'s leadership claim was unsuccessful.",
                    f"Agent {agent1_id} did not grant leadership (grant chance: {getattr(m, 'last_agent1_grant_prob', 0):.1%})."
                ]
        else:
            outcome_text = [
                "No leadership emergence occurred in this interaction.",
                "Neither agent's characteristics matched their own ILT enough to claim."
            ]
        
        outcome_div = ui.div(
            {"class": "interaction-step"},
            ui.h4({"class": "step-header"}, "🎯 Interaction Outcome"),
            ui.p(
                outcome_text[0],
                ui.br(),
                ui.span(outcome_text[1] if len(outcome_text) > 1 else "", class_="detail-text")
            )
        )
        sections.append(outcome_div)
        
        # Add CSS for interaction details
        style = ui.tags.style("""
            .interaction-step {
                background: white;
                padding: 15px;
                margin-bottom: 15px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                border-left: 4px solid #007bff;
            }
            
            .step-header {
                color: #007bff;
                font-size: 1.1em;
                margin-bottom: 10px;
                display: flex;
                align-items: center;
                gap: 8px;
            }
            
            .detail-text {
                color: #666;
                font-size: 0.9em;
                font-style: italic;
                display: block;
                margin-top: 5px;
            }
            
            .success-text {
                color: #28a745;
                font-weight: bold;
            }
            
            .failure-text {
                color: #dc3545;
                font-weight: bold;
            }
            
            .interaction-step:hover {
                border-left-color: #0056b3;
                background: #f8f9fa;
            }
        """)
        
        return ui.div(
            style,
            ui.h3(f"Step {current['current_step']} Interaction Details"),
            *sections
        )
    
    @output
    @render.plot
    def step_network_plot():
        current = get_current_state()
        if current['model'] is None or not hasattr(current['model'], 'interaction_network'):
            return plt.figure()
        
        m = current['model']
        fig, ax = plt.subplots(figsize=(10, 10))
        
        if current['network_pos'] is None:
            pos = nx.spring_layout(m.interaction_network)
            current['network_pos'] = pos
            model_state.set(current)
        else:
            pos = current['network_pos']
        
        # Draw edges with varying width based on interaction frequency
        edge_weights = nx.get_edge_attributes(m.interaction_network, 'weight')
        if edge_weights:
            max_weight = max(edge_weights.values())
            widths = [2 * w/max_weight for w in edge_weights.values()]
            nx.draw_networkx_edges(m.interaction_network, pos, 
                                 width=widths,
                                 alpha=0.3,
                                 edge_color='gray',
                                 arrowsize=20)  # Make arrows more visible
        
        # Draw nodes with fixed size and fixed color scale (0-100)
        node_colors = [agent.leader_identity for agent in m.agents]
        nodes = nx.draw_networkx_nodes(m.interaction_network, pos, 
                                     node_color=node_colors,
                                     node_size=2000,  # Fixed size
                                     cmap=plt.cm.viridis,
                                     vmin=0, vmax=100)  # Fixed color scale
        
        # Enhanced labels showing agent state
        labels = {}
        for i, agent in enumerate(m.agents):
            labels[i] = f"A{i}\n{agent.leader_identity:.0f}"  # Shorter labels with just LI
        
        nx.draw_networkx_labels(m.interaction_network, pos, labels, font_size=8)
        
        # Highlight current interaction pair
        if hasattr(m, 'last_interaction'):
            ax.add_patch(Circle(pos[m.last_interaction[0]], 0.2, 
                              fill=False, color='red', linewidth=3))
            ax.add_patch(Circle(pos[m.last_interaction[1]], 0.2, 
                              fill=False, color='blue', linewidth=3))
            
            # Add interaction labels
            ax.text(pos[m.last_interaction[0]][0], pos[m.last_interaction[0]][1] + 0.25, 
                   "Current", color='red', ha='center')
            ax.text(pos[m.last_interaction[1]][0], pos[m.last_interaction[1]][1] + 0.25, 
                   "Interaction", color='blue', ha='center')
        
        ax.set_title(f"Interaction Network (Step {current['current_step']})")
        plt.colorbar(nodes, label="Leadership Identity (0-100)")
        plt.tight_layout()
        return fig
    
    @output
    @render.plot
    def perception_network_plot():
        current = get_current_state()
        if current['model'] is None:
            return plt.figure()
        
        m = current['model']
        entropy_metrics = calculate_entropy_metrics(m)
        perceptions = entropy_metrics['perception_matrix'].copy()  # Make a copy to modify
        
        # Update diagonal with current leadership identities
        for i, agent in enumerate(m.agents):
            perceptions[i, i] = agent.leader_identity
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create mask for diagonal elements
        mask = np.zeros_like(perceptions, dtype=bool)
        np.fill_diagonal(mask, True)
        
        # Plot off-diagonal elements with new colormap
        im = ax.imshow(np.ma.array(perceptions, mask=mask), 
                      cmap='YlOrRd', vmin=0, vmax=20)  # Yellow-Orange-Red colormap
        
        # Plot diagonal elements in light gray
        diagonal = np.ma.array(perceptions, mask=~mask)
        ax.imshow(diagonal, cmap='Greys', alpha=0.2, vmin=0, vmax=100)
        
        # Add labels with smaller font
        ax.set_xticks(np.arange(len(m.agents)))
        ax.set_yticks(np.arange(len(m.agents)))
        
        # Create more concise labels
        row_labels = [f'A{i}' for i in range(len(m.agents))]  # Shorter labels
        col_labels = [f'A{i}' for i in range(len(m.agents))]
        
        ax.set_xticklabels(col_labels, fontsize=6)
        ax.set_yticklabels(row_labels, fontsize=6)
        
        # Rotate x labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
        
        # Add colorbar with better label
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("# Successful Claims", rotation=-90, va="bottom", fontsize=8)
        
        # Add title with smaller font
        ax.set_title("Leadership Perception Network", fontsize=10, pad=10)
        
        # Add text annotations with smaller font
        for i in range(len(m.agents)):
            for j in range(len(m.agents)):
                value = perceptions[i, j]
                if i == j:  # Diagonal elements (self-identity)
                    text = f"{value:.0f}"  # Just show the value
                    color = 'gray'
                else:  # Off-diagonal elements (granted leadership)
                    if value > 0:
                        text = f"{value:.0f}"
                        color = 'black' if value < 10 else 'white'
                    else:
                        text = ""
                        color = 'black'
                
                ax.text(j, i, text, ha="center", va="center", color=color,
                       fontsize=6, fontweight='normal')
        
        plt.tight_layout()
        return fig

    @output
    @render.ui
    def parameter_rows():
        current = get_current_state()
        
        return ui.div(
            {"class": "parameter-controls"},
            ui.h4("Model Configuration"),
            
            # Core Parameters
            ui.div(
                {"class": "parameter-section"},
                ui.h5("Core Parameters"),
                ui.input_slider(
                    "n_agents", 
                    "Number of Agents",
                    min=2, max=10, value=2
                ),
                ui.input_slider(
                    "identity_change_rate",
                    "Identity Change Rate",
                    min=0.1, max=5.0, value=2.0, step=0.1
                ),
                ui.input_checkbox(
                    "initial_li_equal",
                    "Initial Leader Identity Equal",
                    value=True
                ),
            ),
            
            # ILT Matching Parameters
            ui.div(
                {"class": "parameter-section"},
                ui.h5("ILT Matching Parameters"),
                ui.input_select(
                    "ilt_match_algorithm",
                    "Matching Algorithm",
                    {
                        "euclidean": "Euclidean Distance",
                        "gaussian": "Gaussian Similarity",
                        "sigmoid": "Sigmoid Function",
                        "threshold": "Threshold-based"
                    },
                    selected="euclidean"
                ),
                ui.div(
                    {"id": "gaussian-params", "class": "algorithm-params"},
                    ui.input_slider(
                        "gaussian_sigma",
                        "Gaussian Sigma",
                        min=5.0, max=50.0, value=20.0, step=1.0
                    )
                ),
                ui.div(
                    {"id": "sigmoid-params", "class": "algorithm-params"},
                    ui.input_slider(
                        "sigmoid_k",
                        "Sigmoid Steepness",
                        min=1.0, max=20.0, value=10.0, step=0.5
                    )
                ),
                ui.div(
                    {"id": "threshold-params", "class": "algorithm-params"},
                    ui.input_slider(
                        "threshold_value",
                        "Threshold Value",
                        min=5.0, max=30.0, value=15.0, step=1.0
                    )
                )
            ),
            
            # Initial Value Ranges
            ui.div(
                {"class": "parameter-section"},
                ui.h5("Initial Value Ranges"),
                ui.input_slider(
                    "characteristics_range",
                    "Initial Characteristics Range",
                    min=0, max=100, value=[40, 60]
                ),
                ui.input_slider(
                    "ilt_range",
                    "Initial ILT Range",
                    min=0, max=100, value=[40, 60]
                ),
                ui.input_numeric(
                    "initial_identity",
                    "Initial Identity Value",
                    value=50, min=0, max=100
                )
            ),
            
            # Probability Multipliers
            ui.div(
                {"class": "parameter-section"},
                ui.h5("Probability Adjustments"),
                ui.input_slider(
                    "claim_multiplier",
                    "Claim Probability Multiplier",
                    min=0.1, max=1.0, value=0.7, step=0.05
                ),
                ui.input_slider(
                    "grant_multiplier",
                    "Grant Probability Multiplier",
                    min=0.1, max=1.0, value=0.6, step=0.05
                )
            ),
            
            # Save & Reset Button
            ui.div(
                {"class": "parameter-section mt-4"},
                ui.input_action_button(
                    "save_and_reset",
                    "Save Parameters & Reset Simulation",
                    class_="btn-primary btn-block"
                )
            )
        )
    
    @reactive.Effect
    @reactive.event(input.save_and_reset)
    def handle_save_and_reset():
        current = get_current_state()
        
        # Update configuration with new parameter values
        new_config = {
            'n_agents': input.n_agents(),
            'initial_li_equal': input.initial_li_equal(),
            'li_change_rate': input.identity_change_rate(),
            'identity_change_rate': input.identity_change_rate(),
            'ilt_match_algorithm': input.ilt_match_algorithm(),
            'ilt_match_params': {
                'sigma': input.gaussian_sigma(),
                'k': input.sigmoid_k(),
                'threshold': input.threshold_value()
            },
            'characteristics_range': input.characteristics_range(),
            'ilt_range': input.ilt_range(),
            'initial_identity': input.initial_identity(),
            'claim_multiplier': input.claim_multiplier(),
            'grant_multiplier': input.grant_multiplier()
        }
        
        # Update current state
        current.update({
            'config': new_config,
            'model': None,
            'current_step': 0,
            'network_pos': None,
            'agents': None
        })
        
        model_state.set(current)
    
    # Add CSS for parameter controls
    ui.tags.style("""
        .parameter-section {
            background: white;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .parameter-section h5 {
            color: #007bff;
            margin-bottom: 15px;
            padding-bottom: 5px;
            border-bottom: 2px solid #e9ecef;
        }
        
        .algorithm-params {
            padding-left: 15px;
            margin-top: 10px;
            border-left: 3px solid #e9ecef;
        }
        
        .parameter-controls .form-group {
            margin-bottom: 15px;
        }
        
        .parameter-controls label {
            font-weight: 500;
            color: #495057;
        }
        
        .parameter-controls .form-text {
            color: #6c757d;
            font-size: 0.875rem;
        }
        
        .btn-block {
            width: 100%;
            padding: 10px;
            font-weight: 500;
        }
    """)

    # Define metric explanations
    METRIC_EXPLANATIONS = {
        'hierarchy_clarity': {
            'definition': 'Measures how clear and well-defined the leadership hierarchy is within the group.',
            'example': 'High clarity (>0.6) means clear leaders and followers are emerging. Low clarity suggests confusion about roles.',
            'interpretation': 'Range 0-1: Higher is better'
        },
        'rank_consensus': {
            'definition': 'Shows how much agents agree on who the leaders are in the group.',
            'example': 'High consensus (>0.7) means agents agree on leadership rankings. Low consensus indicates disagreement.',
            'interpretation': 'Range 0-1: Higher is better'
        },
        'structural_stability': {
            'definition': 'Indicates how stable the interaction patterns and relationships are over time.',
            'example': 'High stability (>0.7) shows consistent leadership patterns. Low stability suggests chaotic interactions.',
            'interpretation': 'Range 0-1: Higher is better'
        },
        'system_entropy': {
            'definition': 'Measures the overall randomness or disorder in the leadership structure.',
            'example': 'Low entropy (<2.0) indicates organized leadership. High entropy suggests random, unstructured interactions.',
            'interpretation': 'Range 0+: Lower is better'
        }
    }

    @output
    @render.ui
    def validation_metrics_table():
        current = get_current_state()
        if current['model'] is None:
            return ui.p("Initialize simulation to see validation metrics")
        
        metrics = calculate_entropy_metrics(current['model'])
        validation = check_simulation_validity(current['model'])
        
        # Define thresholds
        thresholds = {
            'hierarchy_clarity': 0.5,
            'rank_consensus': 0.6,
            'structural_stability': 0.7,
            'system_entropy': 2.0  # Lower is better
        }
        
        rows = []
        for metric, value in metrics.items():
            if metric in ['perception_matrix', 'individual_entropies']:
                continue
                
            threshold = thresholds.get(metric)
            if threshold is None:
                continue
                
            # Determine if metric meets threshold
            meets_threshold = False
            if metric == 'system_entropy':
                meets_threshold = value <= threshold
            else:
                meets_threshold = value >= threshold
            
            # Create status icon
            status_icon = "✓" if meets_threshold else "✗"
            status_class = "validation-ok" if meets_threshold else "validation-error"
            
            # Create metric name with tooltip
            metric_info = METRIC_EXPLANATIONS.get(metric, {})
            metric_name = ui.div(
                {"class": "metric-info"},
                metric.replace('_', ' ').title(),
                ui.span("ⓘ", class_="info-icon"),
                ui.div(
                    {"class": "tooltip-text"},
                    ui.p(metric_info.get('definition', ''), class_="metric-definition"),
                    ui.p(metric_info.get('example', ''), class_="metric-example"),
                    ui.p(metric_info.get('interpretation', ''), class_="metric-interpretation")
                )
            )
            
            # Create and append table row
            row = ui.tags.tr(
                ui.tags.td(metric_name),
                ui.tags.td(f"{value:.2f}"),
                ui.tags.td(f"{threshold:.2f}"),
                ui.tags.td(
                    ui.div(
                        {"class": f"validation-status {status_class}"},
                        status_icon
                    )
                )
            )
            rows.append(row)
        
        # Create and return table
        return ui.tags.table(
            {"class": "table table-striped"},
            ui.tags.thead(
                ui.tags.tr(
                    ui.tags.th("Metric ⓘ"),
                    ui.tags.th("Current Value"),
                    ui.tags.th("Threshold"),
                    ui.tags.th("Status")
                )
            ),
            ui.tags.tbody(rows)
        )

    # Add handler for base model selection
    @reactive.Effect
    @reactive.event(input.select_base)
    def select_base_model():
        current = get_current_state()
        current.update({
            'selected_variant': 'base_derue',
            'config': {
                'model_name': 'Base DeRue Model',
                'theoretical_basis': 'Core Leadership Emergence Mechanisms',
                'description': 'Original model with basic mechanisms: leadership characteristics, ILTs, claims/grants, and identity development.',
                'parameters': {
                    'simulation_properties': {
                        'group_size': 6
                    },
                    'agent_properties': {
                        'leader_characteristics': {
                            'initial_range': [30, 70]
                        },
                        'follower_characteristics': {
                            'initial_range': [30, 70]
                        },
                        'ilt': {
                            'weight_range': [0.5, 1.5],
                            'prototype_range': [40, 60]
                        }
                    },
                    'interaction_rules': {
                        'identity_change_rate': 2.0,
                        'ilt_match_threshold': 0.7
                    }
                }
            },
            'model': None,
            'current_step': 0,
            'network_pos': None,
            'agents': None,
        })
        model_state.set(current)
        page_state.set('simulation')

    # Add CSS for step logic styling
    ui.tags.style("""
        .step-logic {
            font-family: 'Monaco', 'Consolas', monospace;
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin: 10px 0;
        }
        
        .step-panel {
            background: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-left: 4px solid #007bff;
            transition: transform 0.2s;
        }
        
        .step-panel:hover {
            transform: translateX(5px);
        }
        
        .step-panel h3 {
            color: #007bff;
            margin-bottom: 15px;
            font-size: 1.3em;
        }
        
        .step-panel p {
            color: #333;
            line-height: 1.6;
            margin-bottom: 15px;
        }
        
        .step-detail {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            color: #666;
            line-height: 1.8;
        }
        
        .current-step {
            border-left-color: #28a745;
            background: #f8fff8;
        }
        
        .current-step h3 {
            color: #28a745;
        }
        
        .step-note {
            font-style: italic;
            color: #666;
            display: block;
            margin-top: 10px;
        }
    """)

    @output
    @render.ui
    def step_logic_content():
        current = get_current_state()
        model = current.get('model')
        
        if model is None:
            return ui.p("Initialize simulation to see step logic")
            
        last_interaction = getattr(model, 'last_interaction', None)
        current_step = current.get('current_step', 0)
        
        sections = []
        
        # Current Step Status
        if last_interaction is not None:
            current_info = ui.div(
                {"class": "step-panel current-step"},
                ui.h3("Current Interaction"),
                ui.p(
                    f"Step {current_step}: Agent {last_interaction[0]} and Agent {last_interaction[1]} are interacting.",
                    ui.br(),
                    ui.span(
                        "Watch the network visualization to see how this affects the group structure.",
                        class_="step-note"
                    )
                )
            )
            sections.append(current_info)
        
        # Step 1: Selection
        selection = ui.div(
            {"class": "step-panel"},
            ui.h3("Step 1: Agent Selection"),
            ui.p(
                "Two agents are randomly chosen to interact with each other. "
                "This represents a spontaneous interaction opportunity in the group."
            )
        )
        sections.append(selection)
        
        # Step 2: Leadership Claims
        claims = ui.div(
            {"class": "step-panel"},
            ui.h3("Step 2: Leadership Claims"),
            ui.p(
                "Each agent decides if they want to be a leader in this interaction. "
                "They are more likely to claim leadership if their characteristics match "
                "what they think a leader should be like."
            ),
            ui.div(
                {"class": "step-detail"},
                "• Agents compare their characteristics to their ideal leader image",
                ui.br(),
                "• Better matches increase the chance of claiming leadership",
                ui.br(),
                "• Both agents can try to claim leadership at the same time"
            )
        )
        sections.append(claims)
        
        # Step 3: Leadership Recognition
        grants = ui.div(
            {"class": "step-panel"},
            ui.h3("Step 3: Leadership Recognition"),
            ui.p(
                "When an agent claims leadership, the other agent decides whether to accept them as a leader. "
                "This decision is based on how well the claiming agent matches their idea of a good leader."
            ),
            ui.div(
                {"class": "step-detail"},
                "• The potential follower compares the claimant to their ideal leader",
                ui.br(),
                "• Better matches increase the chance of accepting the leadership claim",
                ui.br(),
                "• Both agents can recognize each other as leaders if both claimed"
            )
        )
        sections.append(grants)
        
        # Step 4: Identity Changes
        identity = ui.div(
            {"class": "step-panel"},
            ui.h3("Step 4: Identity Development"),
            ui.p(
                "The interaction outcome affects how the agents see themselves as leaders or followers. "
                "Successful leadership claims strengthen these self-views."
            ),
            ui.div(
                {"class": "step-detail"},
                "• Successful leaders become more confident in leading",
                ui.br(),
                "• Followers become more comfortable in following",
                ui.br(),
                "• Failed leadership claims reduce leadership confidence"
            )
        )
        sections.append(identity)
        
        # Step 5: Group Structure
        network = ui.div(
            {"class": "step-panel"},
            ui.h3("Step 5: Group Structure Update"),
            ui.p(
                "The interaction affects the overall group structure. "
                "Successful leadership emergence creates patterns of influence in the group."
            ),
            ui.div(
                {"class": "step-detail"},
                "• Leadership relationships are recorded",
                ui.br(),
                "• The group's hierarchy becomes clearer over time",
                ui.br(),
                "• Patterns of leadership and following emerge"
            )
        )
        sections.append(network)
        
        return ui.div(
            {"class": "step-logic-container"},
            *sections
        )

def _get_trend_class(value, threshold):
    """Helper to get CSS class based on metric value."""
    if value >= threshold:
        return "trend-positive"
    elif value >= threshold * 0.7:
        return "trend-neutral"
    else:
        return "trend-negative"

def _get_validation_class(value, threshold, lower_is_better=False):
    """Helper to get CSS class based on metric value compared to threshold."""
    if lower_is_better:
        if value <= threshold:
            return "validation-ok"
        elif value <= threshold * 1.5:
            return "validation-warning"
        else:
            return "validation-error"
    else:
        if value >= threshold:
            return "validation-ok"
        elif value >= threshold * 0.7:
            return "validation-warning"
        else:
            return "validation-error"

app = App(app_ui, server) 