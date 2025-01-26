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
from shiny.ui import span  # Add span import
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

try:
    from src.models.base import BaseLeadershipModel
except ImportError:
    from src.models.base_model import BaseLeadershipModel

from src.models import (
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
            .documentation-button {
                position: absolute;
                top: 20px;
                right: 20px;
                padding: 10px 20px;
                background: #f8f9fa;
                border: 2px solid #007bff;
                border-radius: 8px;
                color: #007bff;
                text-decoration: none;
                transition: all 0.2s ease;
            }
            .documentation-button:hover {
                background: #007bff;
                color: white;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .documentation-icon {
                margin-right: 8px;
            }
        """)
    ),
    ui.panel_title(
        ui.h1(
            "Leadership Emergence Step-by-Step Simulation",
            class_="text-center mb-4 main-title"
        ),
        ui.download_button(
            "download_docs",
            "📚 Download Documentation",
            class_="documentation-button"
        )
    ),
    
    ui.div(
        {"class": "app-container"},
        ui.output_ui("model_selection_page"),
        ui.output_ui("simulation_interface")
    ),
    
    ui.tags.style("""
        body {
            background-color: #f5f7fa;
        }
        
        .main-title {
            color: #2c3e50;
            font-weight: 600;
            margin: 30px 0;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
        }
        
        .app-container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .simulation-card {
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 30px;
            border: none;
        }
        
        .simulation-card .card-header {
            background: linear-gradient(135deg, #3498db, #2980b9);
            color: white;
            border-radius: 12px 12px 0 0;
            padding: 20px;
            font-size: 1.2em;
            border: none;
        }
        
        .simulation-card .card-body {
            padding: 25px;
        }
        
        .model-info {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            border-left: 4px solid #3498db;
        }
        
        .simulation-controls {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        
        .control-button {
            border-radius: 8px;
            padding: 12px;
            font-weight: 500;
            transition: all 0.3s ease;
            border: none;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .control-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        
        .btn-primary.control-button {
            background: linear-gradient(135deg, #3498db, #2980b9);
        }
        
        .btn-info.control-button {
            background: linear-gradient(135deg, #2ecc71, #27ae60);
        }
        
        .btn-warning.control-button {
            background: linear-gradient(135deg, #e74c3c, #c0392b);
        }
        
        .plot-container {
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            padding: 25px;
            margin-bottom: 30px;
        }
        
        .nav-tabs {
            border: none;
            margin-bottom: 20px;
        }
        
        .nav-tabs .nav-link {
            border: none;
            color: #7f8c8d;
            padding: 12px 20px;
            border-radius: 8px;
            margin-right: 10px;
            transition: all 0.3s ease;
        }
        
        .nav-tabs .nav-link:hover {
            background: #f8f9fa;
            color: #2c3e50;
        }
        
        .nav-tabs .nav-link.active {
            background: linear-gradient(135deg, #3498db, #2980b9);
            color: white;
            font-weight: 500;
        }
        
        .step-count {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            font-size: 1.3em;
            color: #2c3e50;
            margin-top: 20px;
            border-left: 4px solid #3498db;
        }
        
        /* Card layouts */
        .card {
            border: none;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        
        .card-header {
            background: linear-gradient(135deg, #3498db, #2980b9);
            color: white;
            border-radius: 12px 12px 0 0 !important;
            padding: 15px 20px;
            font-weight: 500;
        }
        
        .card-body {
            padding: 20px;
        }
        
        /* Responsive adjustments */
        @media (max-width: 768px) {
            .app-container {
                padding: 10px;
            }
            
            .simulation-card .card-header {
                padding: 15px;
            }
            
            .control-button {
                padding: 10px;
            }
        }
    """)
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
    """Calculate entropy-based metrics for hierarchy emergence with improved robustness."""
    n_agents = len(model.agents)
    
    # Get leader identity values
    li_values = np.array([agent.leader_identity for agent in model.agents])
    
    # 1. IMPROVED HIERARCHY CLARITY
    # Consider both the spread of values and their distribution
    li_range = np.max(li_values) - np.min(li_values)
    li_std = np.std(li_values)
    
    # Calculate normalized metrics
    range_component = li_range / 100.0  # Normalize by max possible range
    std_component = li_std / (100.0 / np.sqrt(12))  # Normalize by max possible std dev
    
    # Calculate entropy component (weighted less now)
    if np.sum(li_values) > 0:
        li_probs = li_values / np.sum(li_values)
        system_entropy = -np.sum(li_probs * np.log2(li_probs + 1e-10))
        max_entropy = -np.log2(1/n_agents)
        entropy_component = 1 - (system_entropy / max_entropy) if max_entropy > 0 else 0
    else:
        entropy_component = 0
    
    # Combine components with weights
    hierarchy_clarity = (0.4 * range_component + 
                        0.4 * std_component + 
                        0.2 * entropy_component)
    
    # 2. IMPROVED RANK CONSENSUS
    # Use Kendall's W for multiple rankers
    rankings = []
    for i in range(n_agents):
        # Each agent ranks all agents (including themselves) based on leader identity
        agent_ranking = np.argsort(-li_values)  # Higher LI = higher rank
        rankings.append(agent_ranking)
    
    rankings = np.array(rankings)
    
    # Calculate mean ranking for each agent
    mean_ranks = np.mean(rankings, axis=0)
    
    # Calculate sum of squared deviations
    S = np.sum((rankings - mean_ranks) ** 2)
    
    # Calculate maximum possible S
    max_S = (n_agents ** 2 * (n_agents ** 2 - 1)) / 12
    
    # Calculate Kendall's W
    if max_S > 0:
        rank_consensus = S / max_S
    else:
        rank_consensus = 0
    
    # 3. Calculate structural stability from interaction network
    if hasattr(model, 'interaction_network'):
        degrees = [d for n, d in model.interaction_network.degree()]
        mean_degree = np.mean(degrees) if degrees else 0
        structural_stability = 1.0 - (np.std(degrees) / mean_degree if mean_degree > 0 else 0.0)
    else:
        structural_stability = 0.0
    
    # 4. System Entropy (kept for backwards compatibility but improved)
    # Now considers both the distribution and the magnitude of differences
    if np.sum(li_values) > 0:
        # Calculate normalized differences between consecutive ranked agents
        sorted_li = np.sort(li_values)
        diffs = np.diff(sorted_li)
        normalized_diffs = diffs / np.max(diffs) if np.max(diffs) > 0 else np.zeros_like(diffs)
        
        # Calculate entropy considering the gaps between ranks
        gap_entropy = -np.sum(normalized_diffs * np.log2(normalized_diffs + 1e-10)) / len(normalized_diffs)
        
        # Normalize to 0-5 range (lower is better)
        system_entropy = gap_entropy * (5.0 / np.log2(n_agents))
    else:
        system_entropy = 5.0  # Maximum entropy when all values are 0
    
    return {
        'system_entropy': system_entropy,
        'hierarchy_clarity': hierarchy_clarity,
        'rank_consensus': rank_consensus,
        'structural_stability': structural_stability,
        'perception_matrix': np.zeros((n_agents, n_agents))  # Kept for compatibility
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
        'scp_dynamic': NetworkModel
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
                            ui.card_header("Social Interactionist/Base Model"),
                            ui.card_body(
                                ui.h4("Core Leadership Emergence"),
                                ui.p(
                                    "Foundational social interactionist model with key mechanisms: "
                                    "leadership characteristics, ILTs, claims/grants, "
                                    "and identity development through social interactions."
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
                                    ui.output_ui("control_buttons")
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
                                                ),
                                                ui.div(
                                                    {"class": "mt-4"},
                                                    ui.input_action_button(
                                                        "save_model_params",
                                                        "Save Model Parameters",
                                                        class_="btn-primary"
                                                    ),
                                                    ui.input_action_button(
                                                        "save_model_and_reset",
                                                        "Save & Reset Simulation",
                                                        class_="btn-warning ml-2"
                                                    )
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        ),
                        ui.nav_panel(
                            "Identity Change Rules",
                            ui.div(
                                {"class": "plot-container"},
                                ui.div(
                                    {"class": "identity-rules-container"},
                                    ui.h4("Identity Change Rules", class_="section-title"),
                                    ui.p("Configure how different interaction paths affect agent identities:", class_="section-description"),
                                    
                                    # Flow diagram container
                                    ui.div(
                                        {"class": "flow-diagram"},
                                        
                                        # Agent Making Decision Section
                                        ui.div(
                                            {"class": "flow-section"},
                                            ui.h5("Agent Making Decision", class_="flow-title"),
                                            
                                            # Start node
                                            ui.div(
                                                {"class": "flow-node start-node"},
                                                ui.div({"class": "node-content"},
                                                    ui.h6("Agent"),
                                                    ui.p("Initial State", class_="node-description")
                                                )
                                            ),
                                            
                                            # Decision paths
                                            ui.div(
                                                {"class": "decision-paths"},
                                                # Claim path
                                                ui.div(
                                                    {"class": "path-branch"},
                                                    ui.div({"class": "path-arrow"}),
                                                    ui.div(
                                                        {"class": "path-label"},
                                                        "Claims Leadership"
                                                    ),
                                                    ui.div(
                                                        {"class": "outcomes-container"},
                                                        # Granted outcome
                                                        ui.div(
                                                            {"class": "outcome-node success"},
                                                            ui.h6("Claim Granted"),
                                                            ui.div(
                                                                {"class": "identity-changes"},
                                                                ui.div(
                                                                    {"class": "change-input"},
                                                                    ui.tags.label(
                                                                        ui.span("Leader Identity ", class_="label-text"),
                                                                        ui.span("+", class_="change-direction positive")
                                                                    ),
                                                                    ui.input_numeric(
                                                                        "claim_granted_li",
                                                                        None,
                                                                        value=2.0,
                                                                        min=-5.0,
                                                                        max=5.0,
                                                                        step=0.1
                                                                    )
                                                                ),
                                                                ui.div(
                                                                    {"class": "change-input"},
                                                                    ui.tags.label(
                                                                        ui.span("Follower Identity ", class_="label-text"),
                                                                        ui.span("-", class_="change-direction negative")
                                                                    ),
                                                                    ui.input_numeric(
                                                                        "claim_granted_fi",
                                                                        None,
                                                                        value=-1.0,
                                                                        min=-5.0,
                                                                        max=5.0,
                                                                        step=0.1
                                                                    )
                                                                )
                                                            )
                                                        ),
                                                        # Rejected outcome
                                                        ui.div(
                                                            {"class": "outcome-node failure"},
                                                            ui.h6("Claim Rejected"),
                                                            ui.div(
                                                                {"class": "identity-changes"},
                                                                ui.div(
                                                                    {"class": "change-input"},
                                                                    ui.tags.label(
                                                                        ui.span("Leader Identity ", class_="label-text"),
                                                                        ui.span("-", class_="change-direction negative")
                                                                    ),
                                                                    ui.input_numeric(
                                                                        "claim_rejected_li",
                                                                        None,
                                                                        value=-1.0,
                                                                        min=-5.0,
                                                                        max=5.0,
                                                                        step=0.1
                                                                    )
                                                                ),
                                                                ui.div(
                                                                    {"class": "change-input"},
                                                                    ui.tags.label(
                                                                        ui.span("Follower Identity ", class_="label-text"),
                                                                        ui.span("-", class_="change-direction negative")
                                                                    ),
                                                                    ui.input_numeric(
                                                                        "claim_rejected_fi",
                                                                        None,
                                                                        value=-0.5,
                                                                        min=-5.0,
                                                                        max=5.0,
                                                                        step=0.1
                                                                    )
                                                                )
                                                            )
                                                        )
                                                    )
                                                ),
                                                # No claim path
                                                ui.div(
                                                    {"class": "path-branch"},
                                                    ui.div({"class": "path-arrow"}),
                                                    ui.div(
                                                        {"class": "path-label"},
                                                        "No Claim"
                                                    ),
                                                    ui.div(
                                                        {"class": "outcome-node neutral"},
                                                        ui.h6("Passive Behavior"),
                                                        ui.div(
                                                            {"class": "identity-changes"},
                                                            ui.div(
                                                                {"class": "change-input"},
                                                                ui.tags.label(
                                                                    ui.span("Leader Identity ", class_="label-text"),
                                                                    ui.span("-", class_="change-direction negative")
                                                                ),
                                                                ui.input_numeric(
                                                                    "no_claim_li",
                                                                    None,
                                                                    value=-0.5,
                                                                    min=-5.0,
                                                                    max=5.0,
                                                                    step=0.1
                                                                )
                                                            ),
                                                            ui.div(
                                                                {"class": "change-input"},
                                                                ui.tags.label(
                                                                    ui.span("Follower Identity ", class_="label-text"),
                                                                    ui.span("+", class_="change-direction positive")
                                                                ),
                                                                ui.input_numeric(
                                                                    "no_claim_fi",
                                                                    None,
                                                                    value=0.5,
                                                                    min=-5.0,
                                                                    max=5.0,
                                                                    step=0.1
                                                                )
                                                            )
                                                        )
                                                    )
                                                )
                                            )
                                        ),
                                        
                                        # Agent Receiving Claim Section
                                        ui.div(
                                            {"class": "flow-section"},
                                            ui.h5("Agent Receiving Claim", class_="flow-title"),
                                            
                                            # Start node
                                            ui.div(
                                                {"class": "flow-node start-node"},
                                                ui.div({"class": "node-content"},
                                                    ui.h6("Other Agent"),
                                                    ui.p("Receives Claim", class_="node-description")
                                                )
                                            ),
                                            
                                            # Decision paths
                                            ui.div(
                                                {"class": "decision-paths"},
                                                # Grant path
                                                ui.div(
                                                    {"class": "path-branch"},
                                                    ui.div({"class": "path-arrow"}),
                                                    ui.div(
                                                        {"class": "path-label"},
                                                        "Evaluates Claim"
                                                    ),
                                                    ui.div(
                                                        {"class": "outcomes-container"},
                                                        # Grants leadership
                                                        ui.div(
                                                            {"class": "outcome-node success"},
                                                            ui.h6("Grants Leadership"),
                                                            ui.div(
                                                                {"class": "identity-changes"},
                                                                ui.div(
                                                                    {"class": "change-input"},
                                                                    ui.tags.label(
                                                                        ui.span("Leader Identity ", class_="label-text"),
                                                                        ui.span("-", class_="change-direction negative")
                                                                    ),
                                                                    ui.input_numeric(
                                                                        "grant_given_li",
                                                                        None,
                                                                        value=-1.0,
                                                                        min=-5.0,
                                                                        max=5.0,
                                                                        step=0.1
                                                                    )
                                                                ),
                                                                ui.div(
                                                                    {"class": "change-input"},
                                                                    ui.tags.label(
                                                                        ui.span("Follower Identity ", class_="label-text"),
                                                                        ui.span("+", class_="change-direction positive")
                                                                    ),
                                                                    ui.input_numeric(
                                                                        "grant_given_fi",
                                                                        None,
                                                                        value=2.0,
                                                                        min=-5.0,
                                                                        max=5.0,
                                                                        step=0.1
                                                                    )
                                                                )
                                                            )
                                                        ),
                                                        # Withholds grant
                                                        ui.div(
                                                            {"class": "outcome-node failure"},
                                                            ui.h6("Withholds Grant"),
                                                            ui.div(
                                                                {"class": "identity-changes"},
                                                                ui.div(
                                                                    {"class": "change-input"},
                                                                    ui.tags.label(
                                                                        ui.span("Leader Identity ", class_="label-text"),
                                                                        ui.span("±", class_="change-direction neutral")
                                                                    ),
                                                                    ui.input_numeric(
                                                                        "grant_withheld_li",
                                                                        None,
                                                                        value=0.0,
                                                                        min=-5.0,
                                                                        max=5.0,
                                                                        step=0.1
                                                                    )
                                                                ),
                                                                ui.div(
                                                                    {"class": "change-input"},
                                                                    ui.tags.label(
                                                                        ui.span("Follower Identity ", class_="label-text"),
                                                                        ui.span("-", class_="change-direction negative")
                                                                    ),
                                                                    ui.input_numeric(
                                                                        "grant_withheld_fi",
                                                                        None,
                                                                        value=-0.5,
                                                                        min=-5.0,
                                                                        max=5.0,
                                                                        step=0.1
                                                                    )
                                                                )
                                                            )
                                                        )
                                                    )
                                                )
                                            )
                                        )
                                    )
                                ),
                                
                                ui.tags.style("""
                                    .identity-rules-container {
                                        background: white;
                                        padding: 30px;
                                        border-radius: 12px;
                                        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                                        max-width: 1200px;
                                        margin: 0 auto;
                                    }
                                    
                                    .section-title {
                                        color: #2c3e50;
                                        font-size: 1.5em;
                                        margin-bottom: 10px;
                                        position: relative;
                                        display: inline-block;
                                    }
                                    
                                    .section-title::after {
                                        content: '';
                                        position: absolute;
                                        bottom: -5px;
                                        left: 0;
                                        width: 100%;
                                        height: 2px;
                                        background: linear-gradient(to right, #3498db, transparent);
                                    }
                                    
                                    .flow-diagram {
                                        display: flex;
                                        flex-direction: column;
                                        gap: 40px;
                                        transition: all 0.3s ease;
                                    }
                                    
                                    @media (max-width: 768px) {
                                        .flow-diagram {
                                            gap: 20px;
                                        }
                                    }
                                    
                                    .flow-section {
                                        background: #f8f9fa;
                                        padding: 25px;
                                        border-radius: 8px;
                                        position: relative;
                                        transition: transform 0.2s ease, box-shadow 0.2s ease;
                                    }
                                    
                                    .flow-section:hover {
                                        transform: translateY(-2px);
                                        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                                    }
                                    
                                    .flow-title {
                                        color: #34495e;
                                        font-size: 1.2em;
                                        margin-bottom: 20px;
                                        display: flex;
                                        align-items: center;
                                        gap: 8px;
                                    }
                                    
                                    .flow-title::before {
                                        content: '👤';
                                        font-size: 1.1em;
                                    }
                                    
                                    .flow-node {
                                        background: white;
                                        border-radius: 8px;
                                        padding: 15px;
                                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                                        margin-bottom: 20px;
                                        border-left: 4px solid #3498db;
                                        transition: all 0.2s ease;
                                    }
                                    
                                    .flow-node:hover {
                                        border-left-width: 6px;
                                        transform: translateX(2px);
                                    }
                                    
                                    .path-branch {
                                        flex: 1;
                                        display: flex;
                                        flex-direction: column;
                                        align-items: center;
                                        position: relative;
                                        transition: transform 0.2s ease;
                                    }
                                    
                                    .path-branch:hover {
                                        transform: translateY(-2px);
                                    }
                                    
                                    .path-arrow {
                                        width: 2px;
                                        height: 30px;
                                        background: #bdc3c7;
                                        margin-bottom: 10px;
                                        position: relative;
                                        transition: height 0.2s ease;
                                    }
                                    
                                    .path-branch:hover .path-arrow {
                                        height: 35px;
                                    }
                                    
                                    .path-label {
                                        background: #e9ecef;
                                        padding: 8px 15px;
                                        border-radius: 20px;
                                        font-weight: 500;
                                        color: #2c3e50;
                                        margin-bottom: 15px;
                                        transition: all 0.2s ease;
                                        cursor: help;
                                    }
                                    
                                    .path-label:hover {
                                        background: #dee2e6;
                                        transform: scale(1.05);
                                    }
                                    
                                    .outcome-node {
                                        padding: 15px;
                                        border-radius: 8px;
                                        background: white;
                                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                                        transition: all 0.2s ease;
                                        position: relative;
                                    }
                                    
                                    .outcome-node:hover {
                                        transform: translateY(-2px);
                                        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
                                    }
                                    
                                    .outcome-node.success {
                                        border-left: 4px solid #2ecc71;
                                    }
                                    
                                    .outcome-node.success:hover {
                                        border-left-width: 6px;
                                    }
                                    
                                    .outcome-node.failure {
                                        border-left: 4px solid #e74c3c;
                                    }
                                    
                                    .outcome-node.failure:hover {
                                        border-left-width: 6px;
                                    }
                                    
                                    .outcome-node.neutral {
                                        border-left: 4px solid #95a5a6;
                                    }
                                    
                                    .outcome-node.neutral:hover {
                                        border-left-width: 6px;
                                    }
                                    
                                    .change-input {
                                        display: flex;
                                        flex-direction: column;
                                        gap: 5px;
                                        position: relative;
                                    }
                                    
                                    .change-input label {
                                        display: flex;
                                        align-items: center;
                                        gap: 5px;
                                        font-size: 0.9em;
                                        color: #495057;
                                        cursor: help;
                                    }
                                    
                                    .change-direction {
                                        font-weight: bold;
                                        width: 24px;
                                        height: 24px;
                                        display: inline-flex;
                                        align-items: center;
                                        justify-content: center;
                                        border-radius: 50%;
                                        transition: all 0.2s ease;
                                    }
                                    
                                    .change-direction:hover {
                                        transform: scale(1.1);
                                    }
                                    
                                    .change-input input {
                                        width: 100%;
                                        padding: 8px;
                                        border: 1px solid #ced4da;
                                        border-radius: 4px;
                                        transition: all 0.2s ease;
                                        font-size: 0.95em;
                                    }
                                    
                                    .change-input input:hover {
                                        border-color: #adb5bd;
                                    }
                                    
                                    .change-input input:focus {
                                        border-color: #80bdff;
                                        outline: none;
                                        box-shadow: 0 0 0 0.2rem rgba(0,123,255,0.25);
                                    }
                                    
                                    /* Tooltip styles */
                                    .tooltip-wrapper {
                                        position: relative;
                                        display: inline-block;
                                    }
                                    
                                    .tooltip-content {
                                        visibility: hidden;
                                        position: absolute;
                                        bottom: 125%;
                                        left: 50%;
                                        transform: translateX(-50%);
                                        background: #2c3e50;
                                        color: white;
                                        padding: 8px 12px;
                                        border-radius: 4px;
                                        font-size: 0.85em;
                                        white-space: nowrap;
                                        z-index: 1000;
                                        opacity: 0;
                                        transition: opacity 0.2s ease;
                                    }
                                    
                                    .tooltip-wrapper:hover .tooltip-content {
                                        visibility: visible;
                                        opacity: 1;
                                    }
                                    
                                    /* Reset button styles */
                                    .reset-button {
                                        position: absolute;
                                        top: 10px;
                                        right: 10px;
                                        background: none;
                                        border: none;
                                        color: #6c757d;
                                        cursor: pointer;
                                        padding: 5px;
                                        transition: all 0.2s ease;
                                    }
                                    
                                    .reset-button:hover {
                                        color: #343a40;
                                        transform: rotate(90deg);
                                    }
                                    
                                    /* Preview styles */
                                    .preview-container {
                                        background: #f8f9fa;
                                        padding: 10px;
                                        border-radius: 4px;
                                        margin-top: 10px;
                                        font-size: 0.9em;
                                        color: #666;
                                        transition: all 0.2s ease;
                                    }
                                    
                                    .preview-container:hover {
                                        background: #e9ecef;
                                    }
                                    
                                    .preview-value {
                                        font-weight: bold;
                                        color: #2c3e50;
                                    }
                                    
                                    /* Responsive adjustments */
                                    @media (max-width: 768px) {
                                        .identity-rules-container {
                                            padding: 15px;
                                        }
                                        
                                        .flow-section {
                                            padding: 15px;
                                        }
                                        
                                        .decision-paths {
                                            flex-direction: column;
                                        }
                                        
                                        .path-branch {
                                            width: 100%;
                                        }
                                    }
                                """)
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

    @output
    @render.ui
    def control_buttons():
        current = get_current_state()
        model = current.get('model')
        
        if model is None:
            # Show Initialize button when no simulation is running
            return ui.div(
                ui.input_action_button(
                    "init_sim", 
                    "Start Simulation", 
                    class_="btn-primary control-button"
                ),
                ui.input_action_button(
                    "back_to_models", 
                    "← Back to Model Selection", 
                    class_="btn-secondary control-button mt-2"
                )
            )
        else:
            # Show simulation control buttons when simulation is running
            return ui.div(
                ui.input_action_button(
                    "step_sim", 
                    span("Step Forward ", ui.tags.i({"class": "fas fa-step-forward"})), 
                    class_="btn-success control-button"
                ),
                ui.input_action_button(
                    "reset_sim", 
                    span("Reset Simulation ", ui.tags.i({"class": "fas fa-redo"})), 
                    class_="btn-warning control-button mt-2"
                ),
                ui.input_action_button(
                    "back_to_models", 
                    span("← Change Model", ui.tags.i({"class": "fas fa-exchange-alt"})), 
                    class_="btn-secondary control-button mt-2"
                )
            )

    @reactive.Effect
    @reactive.event(input.back_to_models)
    def handle_back_to_models():
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
    @reactive.event(input.reset_sim)
    def handle_reset_simulation():
        """Reset the simulation using current parameters."""
        try:
            current = get_current_state()
            config = current.get('config')
            
            if config is None:
                ui.notification_show("No configuration available. Please save parameters first.", type="error")
                return
            
            # Create new model instance with current config
            model = BaseLeadershipModel(config)
            
            # Initialize interaction network
            model.interaction_network = nx.DiGraph()
            for i in range(config['n_agents']):
                model.interaction_network.add_node(i)
            
            # Clear all visualizations and reset state
            current.update({
                'model': model,
                'current_step': 0,
                'network_pos': None,
                'agents': model.agents,
                'last_interaction': None  # Clear last interaction
            })
            
            # Clear any stored interaction data
            if hasattr(model, 'last_interaction'):
                del model.last_interaction
            if hasattr(model, 'last_agent1_claimed'):
                del model.last_agent1_claimed
            if hasattr(model, 'last_agent2_claimed'):
                del model.last_agent2_claimed
            
            model_state.set(current)
            ui.notification_show("Simulation reset successfully!", type="message")
            
        except Exception as e:
            ui.notification_show(f"Error resetting simulation: {str(e)}", type="error")

    @reactive.Effect
    @reactive.event(input.init_sim)
    def initialize_model():
        print("Starting model initialization...")  # Debug print
        
        # Use default configuration
        config = {
            'n_agents': 4,
            'initial_li_equal': True,
            'li_change_rate': 2.0,
            'identity_change_rate': 2.0,
            'ilt_match_algorithm': 'euclidean',
            'ilt_match_params': {
                'sigma': 20.0,
                'k': 10.0,
                'threshold': 15.0
            },
            'characteristics_range': [40, 60],
            'ilt_range': [40, 60],
            'initial_identity': 50,
            'claim_multiplier': 0.7,
            'grant_multiplier': 0.6
        }
        
        try:
            # Initialize model with config
            print("Creating BaseLeadershipModel...")  # Debug print
            model = BaseLeadershipModel(config)
            print("Model created successfully")  # Debug print
            
            # Initialize interaction network
            print("Initializing interaction network...")  # Debug print
            model.interaction_network = nx.DiGraph()
            for i in range(config['n_agents']):
                model.interaction_network.add_node(i)
            print("Network initialized successfully")  # Debug print
            
            # Update state with new model and config
            current = {
                'config': config,
                'model': model,
                'current_step': 0,
                'network_pos': None,
                'agents': model.agents,
                'selected_variant': 'base_derue'
            }
            
            print("Updating model state...")  # Debug print
            model_state.set(current)
            print("Setting page state to simulation...")  # Debug print
            page_state.set('simulation')
            print("Initialization complete")  # Debug print
            
        except Exception as e:
            import traceback
            print(f"Error initializing model: {str(e)}")  # Debug print
            print("Full traceback:")  # Debug print
            print(traceback.format_exc())  # Print full traceback
            raise
    
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
        agent1_self_match = 1 - abs(agent1.characteristics - agent1.ilt) / 100
        agent2_self_match = 1 - abs(agent2.characteristics - agent2.ilt) / 100
        
        agent1_other_match = 1 - abs(agent1.characteristics - agent2.ilt) / 100
        agent2_other_match = 1 - abs(agent2.characteristics - agent1.ilt) / 100
        
        model.last_agent1_match = agent1_other_match
        model.last_agent2_match = agent2_other_match
        
        # Probabilistic claiming based on self-match
        agent1_claim_prob = agent1_self_match
        agent2_claim_prob = agent2_self_match
        
        agent1_claims = model.rng.random() < agent1_claim_prob
        agent2_claims = model.rng.random() < agent2_claim_prob
        
        model.last_agent1_claimed = agent1_claims
        model.last_agent2_claimed = agent2_claims
        model.last_agent1_claim_prob = agent1_claim_prob
        model.last_agent2_claim_prob = agent2_claim_prob
        
        # Calculate grant probabilities based on other-match
        agent1_grant_prob = 0
        agent2_grant_prob = 0
        if agent1_claims:
            agent2_grant_prob = agent1_other_match
        if agent2_claims:
            agent1_grant_prob = agent2_other_match
            
        # Make grant decisions probabilistically
        agent1_grants = False
        agent2_grants = False
        if agent2_claims:
            agent1_grants = model.rng.random() < agent1_grant_prob
        if agent1_claims:
            agent2_grants = model.rng.random() < agent2_grant_prob
            
        model.last_agent1_granted = agent1_grants
        model.last_agent2_granted = agent2_grants
        model.last_agent1_grant_prob = agent1_grant_prob
        model.last_agent2_grant_prob = agent2_grant_prob
        
        # Update model state based on claims and grants
        grant_given = (agent1_claims and agent2_grants) or (agent2_claims and agent1_grants)
        model.last_grant_given = grant_given
        
        # Update identities based on interaction outcomes using UI values
        if agent1_claims:
            if agent2_grants:
                # Claim was granted - use claim_granted values
                agent1.leader_identity = min(100, agent1.leader_identity + input.claim_granted_li())
                agent1.follower_identity = max(0, agent1.follower_identity + input.claim_granted_fi())
                # Update network
                if not model.interaction_network.has_edge(agent2.id, agent1.id):
                    model.interaction_network.add_edge(agent2.id, agent1.id, weight=1)
                else:
                    model.interaction_network[agent2.id][agent1.id]['weight'] += 1
            else:
                # Claim was rejected - use claim_rejected values
                agent1.leader_identity = max(0, agent1.leader_identity + input.claim_rejected_li())
                agent1.follower_identity = max(0, agent1.follower_identity + input.claim_rejected_fi())
        
        if agent2_claims:
            if agent1_grants:
                # Claim was granted - use claim_granted values
                agent2.leader_identity = min(100, agent2.leader_identity + input.claim_granted_li())
                agent2.follower_identity = max(0, agent2.follower_identity + input.claim_granted_fi())
                # Update network
                if not model.interaction_network.has_edge(agent1.id, agent2.id):
                    model.interaction_network.add_edge(agent1.id, agent2.id, weight=1)
                else:
                    model.interaction_network[agent1.id][agent2.id]['weight'] += 1
            else:
                # Claim was rejected - use claim_rejected values
                agent2.leader_identity = max(0, agent2.leader_identity + input.claim_rejected_li())
                agent2.follower_identity = max(0, agent2.follower_identity + input.claim_rejected_fi())
        
        # Update granting agent identities
        if agent1_grants:
            # Agent 1 gave grant - use grant_given values
            agent1.leader_identity = max(0, agent1.leader_identity + input.grant_given_li())
            agent1.follower_identity = min(100, agent1.follower_identity + input.grant_given_fi())
        elif agent2_claims:  # Agent 1 withheld grant
            agent1.leader_identity = max(0, agent1.leader_identity + input.grant_withheld_li())
            agent1.follower_identity = max(0, agent1.follower_identity + input.grant_withheld_fi())
            
        if agent2_grants:
            # Agent 2 gave grant - use grant_given values
            agent2.leader_identity = max(0, agent2.leader_identity + input.grant_given_li())
            agent2.follower_identity = min(100, agent2.follower_identity + input.grant_given_fi())
        elif agent1_claims:  # Agent 2 withheld grant
            agent2.leader_identity = max(0, agent2.leader_identity + input.grant_withheld_li())
            agent2.follower_identity = max(0, agent2.follower_identity + input.grant_withheld_fi())
        
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
            return ui.div(
                {"class": "interaction-step"},
                ui.h4({"class": "step-header"}, "👥 Initial State"),
                ui.p(
                    "The simulation is ready to begin. Click 'Start Simulation' to initialize the agents.",
                    ui.br(),
                    ui.span(
                        "Agents will be created with random initial characteristics and identities.",
                        class_="detail-text"
                    )
                )
            )
        
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
        if current['model'] is None:
            # Create initial state visualization
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.text(0.5, 0.5, "Click 'Start Simulation' to begin",
                   ha='center', va='center', fontsize=12)
            ax.set_title("Agent Interaction Network")
            ax.set_xticks([])
            ax.set_yticks([])
            return fig
        
        m = current['model']
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Initialize positions if not already set
        if current['network_pos'] is None:
            current['network_pos'] = nx.spring_layout(m.interaction_network)
            model_state.set(current)
        
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
                                 arrowsize=20)
        
        # Draw nodes with fixed size and color scale
        node_colors = [agent.leader_identity for agent in m.agents]
        nodes = nx.draw_networkx_nodes(m.interaction_network, pos, 
                                     node_color=node_colors,
                                     node_size=2000,
                                     cmap=plt.cm.viridis,
                                     vmin=0, vmax=100)
        
        # Add labels
        labels = {i: f"A{i}\n{agent.leader_identity:.0f}/{agent.follower_identity:.0f}"
                 for i, agent in enumerate(m.agents)}
        nx.draw_networkx_labels(m.interaction_network, pos, labels, font_size=8)
        
        # Highlight current interaction pair if exists
        if hasattr(m, 'last_interaction'):
            ax.add_patch(Circle(pos[m.last_interaction[0]], 0.2, 
                              fill=False, color='red', linewidth=3))
            ax.add_patch(Circle(pos[m.last_interaction[1]], 0.2, 
                              fill=False, color='blue', linewidth=3))
            
            ax.text(pos[m.last_interaction[0]][0], pos[m.last_interaction[0]][1] + 0.25, 
                   "Current", color='red', ha='center')
            ax.text(pos[m.last_interaction[1]][0], pos[m.last_interaction[1]][1] + 0.25, 
                   "Interaction", color='blue', ha='center')
        
        ax.set_title(f"Interaction Network (Step {current['current_step']})")
        plt.colorbar(nodes, label="Leadership Identity (0-100)")
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        return fig

    @output
    @render.plot
    def perception_network_plot():
        current = get_current_state()
        if current['model'] is None:
            # Create empty figure with message
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, "Initialize simulation to see perceptions",
                   ha='center', va='center', fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            return fig
        
        m = current['model']
        entropy_metrics = calculate_entropy_metrics(m)
        perceptions = entropy_metrics['perception_matrix'].copy()
        
        # Update diagonal with current leadership identities
        for i, agent in enumerate(m.agents):
            perceptions[i, i] = agent.leader_identity
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create mask for diagonal elements
        mask = np.zeros_like(perceptions, dtype=bool)
        np.fill_diagonal(mask, True)
        
        # Plot off-diagonal elements
        im = ax.imshow(np.ma.array(perceptions, mask=mask), 
                      cmap='YlOrRd', vmin=0, vmax=20)
        
        # Plot diagonal elements
        diagonal = np.ma.array(perceptions, mask=~mask)
        ax.imshow(diagonal, cmap='Greys', alpha=0.2, vmin=0, vmax=100)
        
        # Add labels
        ax.set_xticks(np.arange(len(m.agents)))
        ax.set_yticks(np.arange(len(m.agents)))
        
        row_labels = [f'A{i}' for i in range(len(m.agents))]
        col_labels = [f'A{i}' for i in range(len(m.agents))]
        
        ax.set_xticklabels(col_labels, fontsize=8)
        ax.set_yticklabels(row_labels, fontsize=8)
        
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
        
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("# Successful Claims", rotation=-90, va="bottom", fontsize=8)
        
        ax.set_title("Leadership Perception Network", fontsize=10, pad=10)
        
        # Add text annotations
        for i in range(len(m.agents)):
            for j in range(len(m.agents)):
                value = perceptions[i, j]
                if i == j:
                    text = f"{value:.0f}"
                    color = 'gray'
                else:
                    if value > 0:
                        text = f"{value:.0f}"
                        color = 'black' if value < 10 else 'white'
                    else:
                        text = ""
                        color = 'black'
                
                ax.text(j, i, text, ha="center", va="center", color=color,
                       fontsize=8, fontweight='normal')
        
        plt.tight_layout()
        return fig
    
    @output
    @render.ui
    def parameter_rows():
        current = get_current_state()
        
        return ui.tags.div(
            {"class": "parameter-container"},
            
            # Basic Setup Section
            ui.tags.div(
                {"class": "parameter-card"},
                ui.tags.h4("Basic Setup", {"class": "card-title"}),
                ui.tags.p("Essential simulation parameters", {"class": "card-description"}),
                
                ui.tags.div(
                    {"class": "parameter-group"},
                    # Number of Agents
                    ui.tags.div(
                        {"class": "parameter-item"},
                        ui.tags.label(
                            "Number of Agents",
                            ui.tags.span("ⓘ", {"class": "info-icon"}),
                            ui.tags.div(
                                "The number of agents that will participate in leadership interactions.",
                                {"class": "tooltip-text"}
                            )
                        ),
                        ui.input_numeric(
                            "n_agents",
                            None,
                            value=4,
                            min=2,
                            max=10
                        )
                    ),
                    
                    # Identity Change Rate
                    ui.tags.div(
                        {"class": "parameter-item"},
                        ui.tags.label(
                            "Identity Change Rate",
                            ui.tags.span("ⓘ", {"class": "info-icon"}),
                            ui.tags.div(
                                "How quickly agents' leader and follower identities change after interactions.",
                                {"class": "tooltip-text"}
                            )
                        ),
                        ui.input_slider(
                            "identity_change_rate",
                            None,
                            min=0.1,
                            max=5.0,
                            value=2.0,
                            step=0.1
                        )
                    )
                )
            ),
            
            # Initial Conditions Section
            ui.tags.div(
                {"class": "parameter-card"},
                ui.tags.div(
                    {"class": "card-header-with-toggle"},
                    ui.tags.h4("Initial Conditions", {"class": "card-title"}),
                    ui.input_checkbox("show_initial_conditions", "Show Advanced", False)
                ),
                ui.tags.div(
                    {"class": "parameter-group"},
                    # Basic Initial Settings
                    ui.tags.div(
                        {"class": "parameter-item"},
                        ui.tags.label("Initial Leader Identity"),
                        ui.input_numeric(
                            "initial_identity",
                            None,
                            value=50,
                            min=0,
                            max=100
                        )
                    ),
                    ui.tags.div(
                        {"class": "parameter-item"},
                        ui.input_checkbox(
                            "initial_li_equal",
                            "Start with Equal Identities",
                            value=True
                        )
                    ),
                    # Advanced Initial Settings (conditionally shown)
                    ui.output_ui("advanced_initial_conditions")
                )
            ),
            
            # Interaction Rules Section
            ui.tags.div(
                {"class": "parameter-card"},
                ui.tags.div(
                    {"class": "card-header-with-toggle"},
                    ui.tags.h4("Interaction Rules", {"class": "card-title"}),
                    ui.input_checkbox("show_interaction_rules", "Show Advanced", False)
                ),
                ui.tags.div(
                    {"class": "parameter-group"},
                    # Basic Interaction Settings
                    ui.tags.div(
                        {"class": "parameter-item"},
                        ui.tags.label(
                            "Claim Threshold",
                            ui.tags.span("ⓘ", {"class": "info-icon"}),
                            ui.tags.div(
                                "Minimum self-confidence needed to claim leadership.",
                                {"class": "tooltip-text"}
                            )
                        ),
                        ui.input_slider(
                            "claim_threshold",
                            None,
                            min=0.0,
                            max=1.0,
                            value=0.6,
                            step=0.1
                        )
                    ),
                    ui.tags.div(
                        {"class": "parameter-item"},
                        ui.tags.label(
                            "Grant Threshold",
                            ui.tags.span("ⓘ", {"class": "info-icon"}),
                            ui.tags.div(
                                "Minimum leadership match needed to grant leadership.",
                                {"class": "tooltip-text"}
                            )
                        ),
                        ui.input_slider(
                            "grant_threshold",
                            None,
                            min=0.0,
                            max=1.0,
                            value=0.7,
                            step=0.1
                        )
                    ),
                    # Advanced Interaction Settings (conditionally shown)
                    ui.output_ui("advanced_interaction_rules")
                )
            ),
            
            # Add CSS for parameter organization
            ui.tags.style("""
                .parameter-card {
                    background: white;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    margin-bottom: 20px;
                    padding: 20px;
                }
                
                .card-header-with-toggle {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 15px;
                }
                
                .parameter-group {
                    display: grid;
                    gap: 15px;
                }
                
                .parameter-item {
                    background: #f8f9fa;
                    padding: 15px;
                    border-radius: 6px;
                    transition: all 0.2s ease;
                }
                
                .parameter-item:hover {
                    background: #e9ecef;
                }
                
                .advanced-section {
                    margin-top: 15px;
                    padding-top: 15px;
                    border-top: 1px dashed #dee2e6;
                }
                
                .info-icon {
                    color: #007bff;
                    cursor: help;
                    margin-left: 5px;
                }
                
                .tooltip-text {
                    display: none;
                    position: absolute;
                    background: #2c3e50;
                    color: white;
                    padding: 8px 12px;
                    border-radius: 4px;
                    font-size: 0.9em;
                    z-index: 1000;
                    max-width: 250px;
                }
                
                .info-icon:hover + .tooltip-text {
                    display: block;
                }
            """)
        )
    
    @reactive.Effect
    @reactive.event(input.save_model_and_reset)
    def handle_save_model_and_reset():
        """Save model parameters and reset simulation."""
        try:
            handle_save_model_params()
            handle_reset_simulation()
        except Exception as e:
            ui.notification_show(f"Error saving and resetting: {str(e)}", type="error")

    @reactive.Effect
    @reactive.event(input.save_model_params)
    def handle_save_model_params():
        """Handle saving model parameters without resetting."""
        try:
            current = get_current_state()
            if current.get('config') is None:
                current['config'] = {}
            
            # Update model parameters in config
            current['config'].update({
                'n_agents': input.n_agents(),
                'initial_li_equal': input.initial_li_equal(),
                'identity_change_rate': input.identity_change_rate(),
                'ilt_match_algorithm': input.ilt_match_algorithm(),
                'ilt_match_params': {
                    'sigma': input.gaussian_sigma(),
                    'k': input.sigmoid_k(),
                    'threshold': input.threshold_value()
                }
            })
            
            model_state.set(current)
            ui.notification_show("Model parameters saved successfully!", type="message")
            
        except Exception as e:
            ui.notification_show(f"Error saving model parameters: {str(e)}", type="error")

    @reactive.Effect
    @reactive.event(input.save_identity_params)
    def handle_save_identity_params():
        """Handle saving identity parameters without resetting."""
        try:
            current = get_current_state()
            if current.get('config') is None:
                current['config'] = {}
            
            # Update identity parameters in config
            current['config'].update({
                'identity_rules': {
                    'claim_granted': {
                        'leader_identity': input.claim_granted_li(),
                        'follower_identity': input.claim_granted_fi()
                    },
                    'claim_rejected': {
                        'leader_identity': input.claim_rejected_li(),
                        'follower_identity': input.claim_rejected_fi()
                    },
                    'grant_given': {
                        'leader_identity': input.grant_given_li(),
                        'follower_identity': input.grant_given_fi()
                    },
                    'grant_withheld': {
                        'leader_identity': input.grant_withheld_li(),
                        'follower_identity': input.grant_withheld_fi()
                    }
                }
            })
            
            model_state.set(current)
            ui.notification_show("Identity parameters saved successfully!", type="message")
            
        except Exception as e:
            ui.notification_show(f"Error saving identity parameters: {str(e)}", type="error")

    @reactive.Effect
    @reactive.event(input.save_identity_and_reset)
    def handle_save_identity_and_reset():
        """Save identity parameters and reset simulation."""
        try:
            handle_save_identity_params()
            handle_reset_simulation()
        except Exception as e:
            ui.notification_show(f"Error saving and resetting: {str(e)}", type="error")

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
            return ui.tags.p("Initialize simulation to see validation metrics")
        
        metrics = calculate_entropy_metrics(current['model'])
        validation = check_simulation_validity(current['model'])
        
        # Define thresholds
        thresholds = {
            'hierarchy_clarity': 0.5,
            'rank_consensus': 0.6,
            'structural_stability': 0.7,
            'system_entropy': 2.0  # Lower is better
        }
        
        # Create table rows
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
            metric_name = ui.tags.div(
                {"class": "metric-info"},
                metric.replace('_', ' ').title(),
                ui.tags.span("ⓘ", {"class": "info-icon"}),
                ui.tags.div(
                    {"class": "tooltip-text"},
                    ui.tags.p(metric_info.get('definition', ''), {"class": "metric-definition"}),
                    ui.tags.p(metric_info.get('example', ''), {"class": "metric-example"}),
                    ui.tags.p(metric_info.get('interpretation', ''), {"class": "metric-interpretation"})
                )
            )
            
            # Create table row
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
        
        # Create and return complete table
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
            ui.tags.tbody(*rows)  # Use unpacking operator to add all rows
        )
    
    # Add handler for base model selection
    @reactive.Effect
    @reactive.event(input.select_base)
    def select_base_model():
        current = get_current_state()
        current.update({
            'selected_variant': 'base_derue',
            'config': {
                'model_name': 'Base Social Interactionist Model',
                'theoretical_basis': 'Social Interactionist Leadership Emergence',
                'description': 'Social interactionist model with core mechanisms: leadership characteristics, ILTs, claims/grants, and identity development through social interactions.',
                'parameters': {
                    'simulation_properties': {
                        'group_size': 4  # Changed default to 4 agents
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

    @output
    @render.ui
    def advanced_initial_conditions():
        """Render advanced initial condition parameters when checkbox is checked."""
        if not input.show_initial_conditions():
            return None
            
        return ui.tags.div(
            {"class": "advanced-section"},
            ui.tags.h5("Advanced Initial Settings", {"class": "section-subtitle"}),
            
            # Characteristics Range
            ui.tags.div(
                {"class": "parameter-item"},
                ui.tags.label(
                    "Initial Characteristics Range",
                    ui.tags.span("ⓘ", {"class": "info-icon"}),
                    ui.tags.div(
                        "Range for initial leadership characteristics.",
                        {"class": "tooltip-text"}
                    )
                ),
                ui.input_slider(
                    "characteristics_range",
                    None,
                    min=0,
                    max=100,
                    value=[40, 60]
                )
            ),
            
            # ILT Range
            ui.tags.div(
                {"class": "parameter-item"},
                ui.tags.label(
                    "ILT Prototype Range",
                    ui.tags.span("ⓘ", {"class": "info-icon"}),
                    ui.tags.div(
                        "Range for initial Implicit Leadership Theory values.",
                        {"class": "tooltip-text"}
                    )
                ),
                ui.input_slider(
                    "ilt_range",
                    None,
                    min=0,
                    max=100,
                    value=[40, 60]
                )
            )
        )
    
    @output
    @render.ui
    def advanced_interaction_rules():
        """Render advanced interaction rule parameters when checkbox is checked."""
        if not input.show_interaction_rules():
            return None
            
        return ui.tags.div(
            {"class": "advanced-section"},
            ui.tags.h5("Advanced Interaction Settings", {"class": "section-subtitle"}),
            
            # ILT Matching Method
            ui.tags.div(
                {"class": "parameter-item"},
                ui.tags.label(
                    "ILT Matching Method",
                    ui.tags.span("ⓘ", {"class": "info-icon"}),
                    ui.tags.div(
                        "How agents compare others' characteristics to their ideal leader prototype.",
                        {"class": "tooltip-text"}
                    )
                ),
                ui.input_select(
                    "ilt_match_algorithm",
                    None,
                    {
                        "euclidean": "Simple Direct Comparison",
                        "gaussian": "Flexible Matching",
                        "sigmoid": "Clear Cutoff",
                        "threshold": "Yes/No Decision"
                    }
                )
            ),
            
            # Method-specific parameters
            ui.tags.div(
                {"class": "parameter-item"},
                ui.tags.label(
                    "Method Parameters",
                    ui.tags.span("ⓘ", {"class": "info-icon"}),
                    ui.tags.div(
                        "Specific parameters for the selected matching method.",
                        {"class": "tooltip-text"}
                    )
                ),
                ui.input_slider(
                    "gaussian_sigma",
                    "Flexibility (Gaussian)",
                    min=5.0,
                    max=50.0,
                    value=20.0,
                    step=1.0
                ),
                ui.input_slider(
                    "sigmoid_k",
                    "Decision Sharpness (Sigmoid)",
                    min=1.0,
                    max=20.0,
                    value=10.0,
                    step=0.5
                ),
                ui.input_slider(
                    "threshold_value",
                    "Required Similarity (Threshold)",
                    min=5.0,
                    max=30.0,
                    value=15.0,
                    step=1.0
                )
            )
        )

    @output
    @render.download(filename="leadership_emergence_documentation")
    def download_docs():
        """Create and serve the documentation using Quarto."""
        def create_documentation():
            import tempfile
            import os
            from pathlib import Path
            import subprocess
            import shutil

            # Create a temporary directory for Quarto files
            with tempfile.TemporaryDirectory() as temp_dir:
                # Create Quarto document
                qmd_path = Path(temp_dir) / "documentation.qmd"
                with open(qmd_path, "w") as f:
                    f.write("""---
title: "Leadership Emergence Simulation Documentation"
format:
  pdf:
    toc: true
    number-sections: true
    colorlinks: true
  html:
    toc: true
    code-fold: true
    theme: cosmo
---

## Introduction

This document provides detailed explanations of the metrics, parameters, and theoretical foundations 
of the Leadership Emergence Simulation. Understanding these elements will help you make the most of 
the simulation and interpret its results effectively.

## Simulation Metrics

```{python}
#| echo: false
#| warning: false
import pandas as pd

metrics = {
    'Hierarchy Clarity': {
        'Definition': 'Measures how clear and well-defined the leadership hierarchy is',
        'Example': 'High clarity (>0.6) means clear leaders and followers are emerging',
        'Interpretation': 'Range 0-1: Higher is better'
    },
    'Rank Consensus': {
        'Definition': 'Agreement among agents about leadership rankings',
        'Example': 'High consensus (>0.7) means agents agree on who leads',
        'Interpretation': 'Range 0-1: Higher is better'
    }
}

df = pd.DataFrame.from_dict(metrics, orient='index')
df.style.set_properties(**{'text-align': 'left'})
```

## Parameters

### Basic Parameters

#### Number of Agents
The number of agents participating in the simulation. More agents create more complex interaction patterns.

::: {.callout-tip}
## Recommended Range
2-10 agents is optimal for observing emergence patterns while maintaining interpretability
:::

#### Identity Change Rate
Controls how quickly agents' leader and follower identities change based on interactions.

```{python}
#| echo: false
#| label: fig-identity-change
#| fig-cap: "Effect of Identity Change Rate on Emergence Speed"

import matplotlib.pyplot as plt
import numpy as np

rates = np.linspace(0.1, 5.0, 100)
emergence_speed = 1 - np.exp(-rates)

plt.figure(figsize=(8, 4))
plt.plot(rates, emergence_speed)
plt.xlabel('Identity Change Rate')
plt.ylabel('Relative Emergence Speed')
plt.grid(True, alpha=0.3)
```

### Advanced Parameters

#### ILT Matching Methods

Different methods for comparing agent characteristics to leadership prototypes:

1. **Euclidean Distance**
   - Simple direct comparison
   - $d = \sqrt{\sum(x_i - y_i)^2}$

2. **Gaussian Similarity**
   - More forgiving of small differences
   - $s = e^{-\frac{d^2}{2\sigma^2}}$

3. **Sigmoid Function**
   - Sharp transition between acceptance/rejection
   - $s = \frac{1}{1 + e^{-k(x-x_0)}}$

4. **Threshold-based**
   - Binary decision
   - $s = \begin{cases} 1 & \text{if } d < threshold \\ 0 & \text{otherwise} \end{cases}$

## Theoretical Background

### Social Identity Theory
How group memberships and collective identity influence leadership emergence.

### Implicit Leadership Theory
How people's mental models of leadership affect who they accept as leaders.

::: {.callout-note}
## Key Insight
ILTs act as cognitive filters that shape both leadership claims and grants.
:::

### Identity Development
How leadership and follower identities evolve through social interaction.

### Group Dynamics
How patterns of influence and hierarchy emerge from individual interactions.

## References

::: {#refs}
:::
""")

                # Run Quarto to generate PDF
                subprocess.run(["quarto", "render", str(qmd_path), "--to", "pdf"], check=True)
                
                # Read the generated PDF
                pdf_path = Path(temp_dir) / "documentation.pdf"
                with open(pdf_path, "rb") as f:
                    return f.read()

        return create_documentation()

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