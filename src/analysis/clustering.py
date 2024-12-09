"""
Clustering analysis for leadership emergence patterns.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import hdbscan
from umap import UMAP

class PatternAnalyzer:
    """Analyzes patterns in leadership emergence data using clustering."""
    
    def __init__(
        self,
        features: pd.DataFrame,
        n_clusters: int = None,
        random_state: int = None
    ):
        self.features = features
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.scaled_features = None
        self.cluster_labels = None
        self.embedding = None
    
    def preprocess(self, features_to_use: List[str] = None) -> np.ndarray:
        """Preprocess features for clustering."""
        if features_to_use is None:
            features_to_use = self.features.columns
        
        # Scale features
        self.scaled_features = self.scaler.fit_transform(
            self.features[features_to_use]
        )
        return self.scaled_features
    
    def find_clusters_kmeans(self) -> Tuple[np.ndarray, float]:
        """Find clusters using k-means."""
        if self.scaled_features is None:
            self.preprocess()
        
        # Run k-means
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state
        )
        self.cluster_labels = kmeans.fit_predict(self.scaled_features)
        
        # Calculate silhouette score
        silhouette = silhouette_score(
            self.scaled_features,
            self.cluster_labels
        )
        
        return self.cluster_labels, silhouette
    
    def find_clusters_hdbscan(
        self,
        min_cluster_size: int = 5
    ) -> Tuple[np.ndarray, float]:
        """Find clusters using HDBSCAN."""
        if self.scaled_features is None:
            self.preprocess()
        
        # Run HDBSCAN
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            gen_min_span_tree=True
        )
        self.cluster_labels = clusterer.fit_predict(self.scaled_features)
        
        # Calculate validity metrics
        validity = np.mean(clusterer.probabilities_)
        
        return self.cluster_labels, validity
    
    def create_embedding(
        self,
        n_components: int = 2,
        n_neighbors: int = 15
    ) -> np.ndarray:
        """Create low-dimensional embedding using UMAP."""
        if self.scaled_features is None:
            self.preprocess()
        
        # Run UMAP
        reducer = UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            random_state=self.random_state
        )
        self.embedding = reducer.fit_transform(self.scaled_features)
        
        return self.embedding
    
    def analyze_clusters(self) -> pd.DataFrame:
        """Analyze characteristics of each cluster."""
        if self.cluster_labels is None:
            raise ValueError("Must run clustering before analysis")
        
        # Create DataFrame with cluster assignments
        results = pd.DataFrame({
            'cluster': self.cluster_labels
        })
        
        # Calculate cluster statistics
        cluster_stats = []
        for cluster in np.unique(self.cluster_labels):
            if cluster == -1:  # Noise points in HDBSCAN
                continue
                
            mask = self.cluster_labels == cluster
            cluster_features = self.features[mask]
            
            stats = {
                'cluster': cluster,
                'size': np.sum(mask),
                'mean_li': cluster_features['mean_final_li'].mean(),
                'mean_fi': cluster_features['mean_final_fi'].mean(),
                'mean_role_diff': cluster_features['mean_role_diff'].mean(),
                'mean_stability': cluster_features['time_to_li_stability'].mean()
            }
            cluster_stats.append(stats)
        
        return pd.DataFrame(cluster_stats)
    
    def get_exemplars(self, n_exemplars: int = 3) -> Dict[int, List[int]]:
        """Find exemplar simulations for each cluster."""
        if self.cluster_labels is None or self.scaled_features is None:
            raise ValueError("Must run clustering before finding exemplars")
        
        exemplars = {}
        cluster_centers = {}
        
        # Calculate cluster centers
        for cluster in np.unique(self.cluster_labels):
            if cluster == -1:  # Noise points in HDBSCAN
                continue
                
            mask = self.cluster_labels == cluster
            cluster_features = self.scaled_features[mask]
            cluster_centers[cluster] = np.mean(cluster_features, axis=0)
        
        # Find closest points to centers
        for cluster, center in cluster_centers.items():
            mask = self.cluster_labels == cluster
            cluster_features = self.scaled_features[mask]
            
            # Calculate distances to center
            distances = np.linalg.norm(
                cluster_features - center,
                axis=1
            )
            
            # Get indices of closest points
            closest_indices = np.argsort(distances)[:n_exemplars]
            exemplars[cluster] = np.where(mask)[0][closest_indices]
        
        return exemplars 