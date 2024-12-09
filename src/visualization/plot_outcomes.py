import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from typing import Dict, List

def plot_identity_evolution(history: Dict[str, List], title: str = "Identity Evolution"):
    """Plot the evolution of leader and follower identities over time."""
    plt.figure(figsize=(12, 6))
    
    # Convert to numpy arrays for easier manipulation
    li_data = np.array(history['leader_identities'])
    fi_data = np.array(history['follower_identities'])
    
    # Plot individual trajectories
    time = range(len(li_data))
    for i in range(li_data.shape[1]):
        plt.plot(time, li_data[:, i], 'b-', alpha=0.3, label='Leader Identity' if i == 0 else "")
        plt.plot(time, fi_data[:, i], 'r-', alpha=0.3, label='Follower Identity' if i == 0 else "")
    
    # Plot means
    plt.plot(time, li_data.mean(axis=1), 'b-', linewidth=2, label='Mean Leader Identity')
    plt.plot(time, fi_data.mean(axis=1), 'r-', linewidth=2, label='Mean Follower Identity')
    
    plt.xlabel('Time Step')
    plt.ylabel('Identity Strength')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
def plot_network_metrics(history: Dict[str, List], title: str = "Network Evolution"):
    """Plot the evolution of network metrics over time."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    time = range(len(history['centralization']))
    
    # Plot centralization
    ax1.plot(time, history['centralization'], 'g-', linewidth=2)
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Centralization')
    ax1.set_title('Leadership Centralization')
    ax1.grid(True, alpha=0.3)
    
    # Plot density
    ax2.plot(time, history['density'], 'b-', linewidth=2)
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Density')
    ax2.set_title('Network Density')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()

def plot_leadership_network(network: nx.DiGraph, title: str = "Leadership Network"):
    """Plot the current state of the leadership network."""
    plt.figure(figsize=(10, 10))
    
    # Get edge weights for width
    edges = network.edges(data=True)
    weights = [d['weight'] * 2 for (u, v, d) in edges]
    
    # Calculate node sizes based on in-degree centrality
    centrality = nx.in_degree_centrality(network)
    node_sizes = [v * 3000 for v in centrality.values()]
    
    # Create layout
    pos = nx.spring_layout(network)
    
    # Draw network
    nx.draw_networkx_nodes(network, pos, node_size=node_sizes, 
                          node_color=list(centrality.values()),
                          cmap=plt.cm.YlOrRd)
    nx.draw_networkx_edges(network, pos, width=weights, 
                          edge_color='gray', alpha=0.5,
                          arrowsize=20)
    nx.draw_networkx_labels(network, pos)
    
    plt.title(title)
    plt.axis('off')

def create_summary_plots(model, title: str = "Leadership Emergence Simulation"):
    """Create a comprehensive set of plots for model outcomes."""
    plt.figure(figsize=(15, 10))
    
    # Identity evolution
    plt.subplot(2, 2, 1)
    plot_identity_evolution(model.history, "Identity Evolution")
    
    # Network metrics
    plt.subplot(2, 2, 2)
    plot_network_metrics(model.history, "Network Evolution")
    
    # Current network state
    plt.subplot(2, 2, (3, 4))
    plot_leadership_network(model.interaction_network, "Current Leadership Network")
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
def plot_interaction_heatmap(model, title: str = "Interaction Patterns"):
    """Plot a heatmap of interaction patterns between agents."""
    plt.figure(figsize=(8, 6))
    
    # Create adjacency matrix
    adj_matrix = nx.to_numpy_array(model.interaction_network)
    
    # Plot heatmap
    sns.heatmap(adj_matrix, annot=True, cmap='YlOrRd', 
                xticklabels=range(model.n_agents),
                yticklabels=range(model.n_agents))
    
    plt.title(title)
    plt.xlabel('Target Agent')
    plt.ylabel('Source Agent') 