"""
Metrics for validating leadership emergence patterns.
"""

import numpy as np
from scipy import stats
import networkx as nx

def calculate_identity_variance(li_history, fi_history):
    """Calculate variance in leader and follower identities over time."""
    li_variance = np.var(li_history, axis=1)
    fi_variance = np.var(fi_history, axis=1)
    return li_variance, fi_variance

def calculate_perception_agreement(interaction_network):
    """Calculate within-group agreement on leadership perceptions."""
    adj_matrix = nx.to_numpy_array(interaction_network)
    # Calculate rwg (within-group agreement) using variance
    variance = np.var(adj_matrix)
    expected_variance = 1.0  # Assuming ratings on 0-1 scale
    rwg = 1 - (variance / expected_variance)
    return rwg

def calculate_claiming_granting_correlation(interaction_network):
    """Calculate correlation between claiming and granting behaviors."""
    adj_matrix = nx.to_numpy_array(interaction_network)
    claiming = np.sum(adj_matrix, axis=1)  # Row sums (outgoing ties)
    granting = np.sum(adj_matrix, axis=0)  # Column sums (incoming ties)
    correlation, p_value = stats.pearsonr(claiming, granting)
    return correlation, p_value

def calculate_network_metrics(interaction_network):
    """Calculate various network-based leadership metrics."""
    try:
        # Convert weights to binary edges using a threshold
        binary_network = nx.DiGraph()
        binary_network.add_nodes_from(interaction_network.nodes())
        
        # Add edges where weight exceeds mean weight
        weights = [d['weight'] for (u, v, d) in interaction_network.edges(data=True)]
        if weights:
            threshold = np.mean(weights)
            for u, v, d in interaction_network.edges(data=True):
                if d['weight'] > threshold:
                    binary_network.add_edge(u, v)
        
        metrics = {
            'density': nx.density(binary_network),
            'centralization': calculate_centralization(binary_network),
            'clustering': calculate_clustering(binary_network),
            'modularity': calculate_modularity(binary_network)
        }
        return metrics
    except Exception as e:
        print(f"Error calculating network metrics: {e}")
        return {
            'density': 0.0,
            'centralization': 0.0,
            'clustering': 0.0,
            'modularity': 0.0
        }

def calculate_centralization(network):
    """Calculate degree centralization of the network."""
    n = network.number_of_nodes()
    if n < 2:
        return 0
    
    # Calculate degree centrality for all nodes
    centrality = nx.in_degree_centrality(network)
    centrality_values = list(centrality.values())
    
    if not centrality_values:
        return 0
    
    # Find maximum centrality
    max_centrality = max(centrality_values)
    
    # Calculate sum of differences from maximum
    sum_diff = sum(max_centrality - c for c in centrality_values)
    
    # Calculate maximum possible centralization for a directed network
    max_possible = (n - 1)
    
    # Return centralization index
    return sum_diff / max_possible if max_possible > 0 else 0

def calculate_clustering(network):
    """Calculate average clustering coefficient with error handling."""
    try:
        if network.number_of_edges() == 0:
            return 0.0
        
        # Convert to undirected for clustering calculation
        undirected = network.to_undirected()
        return nx.average_clustering(undirected)
    except:
        return 0.0

def calculate_modularity(network):
    """Calculate modularity of the network using community detection."""
    try:
        if network.number_of_edges() == 0:
            return 0.0
            
        # Convert to undirected and remove self-loops for community detection
        undirected = network.to_undirected()
        undirected.remove_edges_from(nx.selfloop_edges(undirected))
        
        if undirected.number_of_edges() == 0:
            return 0.0
        
        # Use fast greedy community detection
        communities = nx.community.louvain_communities(undirected)
        if not communities:
            return 0.0
            
        return nx.community.modularity(undirected, communities)
    except Exception as e:
        print(f"Error calculating modularity: {e}")
        return 0.0

def calculate_emergence_lag(li_history, fi_history, threshold=0.1):
    """Calculate the time steps needed for roles to stabilize."""
    li_variance = np.var(li_history, axis=1)
    fi_variance = np.var(fi_history, axis=1)
    
    # Find point where variance stays below threshold
    li_stable = np.where(li_variance < threshold)[0]
    fi_stable = np.where(fi_variance < threshold)[0]
    
    if len(li_stable) > 0 and len(fi_stable) > 0:
        return max(li_stable[0], fi_stable[0])
    return len(li_variance)  # Return max time if no stabilization

def calculate_identity_behavior_consistency(model):
    """Calculate correlation between identity scores and behavior."""
    adj_matrix = nx.to_numpy_array(model.interaction_network)
    claiming = np.sum(adj_matrix, axis=1)
    granting = np.sum(adj_matrix, axis=0)
    
    li_scores = np.array([agent.leader_identity for agent in model.agents])
    fi_scores = np.array([agent.follower_identity for agent in model.agents])
    
    # Check for constant arrays
    li_claim_corr = 0.0
    fi_grant_corr = 0.0
    
    if np.std(li_scores) > 0 and np.std(claiming) > 0:
        li_claim_corr, _ = stats.pearsonr(li_scores, claiming)
    
    if np.std(fi_scores) > 0 and np.std(granting) > 0:
        fi_grant_corr, _ = stats.pearsonr(fi_scores, granting)
    
    return li_claim_corr, fi_grant_corr 