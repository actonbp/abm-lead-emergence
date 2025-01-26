"""
ML pipeline for analyzing leadership emergence patterns and optimizing parameters.
"""

from typing import Dict, List, Any, Optional
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.stats import linregress
import matplotlib.pyplot as plt
from pathlib import Path

class MLPipeline:
    """Class for ML-driven analysis of leadership emergence simulations."""
    
    def __init__(
        self,
        n_iterations: int = 30,
        batch_size: int = 5,
        n_clusters: int = 3
    ):
        """Initialize ML pipeline.
        
        Args:
            n_iterations: Number of optimization iterations
            batch_size: Number of parallel evaluations per iteration
            n_clusters: Number of clusters for pattern analysis
        """
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
    
    def run_optimization(
        self,
        initial_configs: List[Dict],
        initial_results: List[Dict],
        parameter_space: Dict,
        n_iterations: Optional[int] = None
    ) -> Dict:
        """Run Bayesian optimization to find optimal parameters."""
        if n_iterations is None:
            n_iterations = self.n_iterations
            
        # Extract features and targets from initial results
        X = self._extract_features(initial_configs)
        y = self._calculate_objectives(initial_results)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize best results
        best_idx = np.argmax(y)
        best_configs = [initial_configs[best_idx]]
        best_scores = [y[best_idx]]
        best_results = [initial_results[best_idx]]
        
        # Run optimization iterations
        for _ in range(n_iterations):
            # Generate new candidates
            candidates = self._generate_candidates(
                parameter_space,
                X_scaled,
                y,
                self.batch_size
            )
            
            # For now, just store the candidates with placeholder results
            best_configs.extend(candidates)
            best_scores.extend([0.0] * len(candidates))
            best_results.extend([{'metrics': {}, 'histories': []} for _ in range(len(candidates))])
        
        return {
            'configs': best_configs,
            'scores': best_scores,
            'results': best_results
        }
    
    def analyze_results(
        self,
        configs: List[Dict],
        results: List[Dict]
    ) -> Dict:
        """Analyze parameter sweep results."""
        # Extract features and calculate objectives
        X = self._extract_features(configs)
        y = self._calculate_objectives(results)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Cluster configurations
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Find best configurations
        best_idx = np.argsort(y)[-3:][::-1]  # Top 3
        best_configs = [configs[i] for i in best_idx]
        best_scores = [y[i] for i in best_idx]
        
        # Calculate parameter importance
        importance = self._calculate_parameter_importance(X, y)
        
        return {
            'best_configs': best_configs,
            'best_scores': best_scores,
            'clusters': clusters.tolist(),
            'parameter_importance': importance
        }
    
    def _extract_features(self, configs: List[Dict]) -> np.ndarray:
        """Extract numerical features from configurations."""
        # Get sorted parameter names from first config
        self.param_names = sorted(configs[0].keys())
        
        # Get unique values for categorical parameters
        self.categorical_values = {}
        for param_name in self.param_names:
            values = set()
            for config in configs:
                if isinstance(config[param_name], str):
                    values.add(config[param_name])
            if values:
                self.categorical_values[param_name] = sorted(values)
        
        # Track feature names and indices
        self.feature_names = []
        self.param_to_features = {param: [] for param in self.param_names}
        feature_idx = 0
        
        # First, determine feature structure
        for param_name in self.param_names:
            value = configs[0][param_name]  # Use first config as reference
            if isinstance(value, (int, float, bool)):
                self.feature_names.append(param_name)
                self.param_to_features[param_name].append(feature_idx)
                feature_idx += 1
            elif isinstance(value, str):
                values = self.categorical_values[param_name]
                for v in values:
                    self.feature_names.append(f"{param_name}_{v}")
                    self.param_to_features[param_name].append(feature_idx)
                    feature_idx += 1
        
        # Convert configs to feature matrix
        features = []
        for config in configs:
            feature_vector = []
            for param_name in self.param_names:
                value = config[param_name]
                if isinstance(value, (int, float)):
                    feature_vector.append(float(value))
                elif isinstance(value, bool):
                    feature_vector.append(1.0 if value else 0.0)
                elif isinstance(value, str):
                    values = self.categorical_values[param_name]
                    one_hot = [1.0 if value == v else 0.0 for v in values]
                    feature_vector.extend(one_hot)
            features.append(feature_vector)
        
        return np.array(features)
    
    def _calculate_objectives(self, results: List[Dict]) -> np.ndarray:
        """Calculate objective values from results."""
        objectives = []
        for result in results:
            # Extract relevant metrics
            metrics = result.get('metrics', {})
            
            # Calculate weighted sum of metrics
            objective = (
                0.3 * metrics.get('kendall_w', 0.0) +
                0.3 * metrics.get('krippendorff_alpha', 0.0) +
                -0.2 * metrics.get('normalized_entropy', 0.0) +
                0.2 * metrics.get('top_leader_agreement', 0.0)
            )
            objectives.append(objective)
        
        return np.array(objectives)
    
    def _generate_candidates(
        self,
        parameter_space: Dict,
        X: np.ndarray,
        y: np.ndarray,
        n_candidates: int
    ) -> List[Dict]:
        """Generate new candidate configurations."""
        # For now, just return random configurations
        candidates = []
        for _ in range(n_candidates):
            config = {}
            for param_name, param_info in parameter_space.items():
                if param_info['type'] == 'continuous':
                    low, high = param_info['range']
                    config[param_name] = np.random.uniform(low, high)
                elif param_info['type'] == 'discrete':
                    config[param_name] = np.random.choice(param_info['values'])
            candidates.append(config)
        return candidates
    
    def _calculate_parameter_importance(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, float]:
        """Calculate parameter importance using correlation analysis."""
        importance = {}
        
        for param_name in self.param_names:
            feature_indices = self.param_to_features[param_name]
            if not feature_indices:
                print(f"Warning: No features found for parameter {param_name}")
                importance[param_name] = 0.0
                continue
                
            correlations = []
            for idx in feature_indices:
                if idx >= X.shape[1]:
                    print(f"Warning: Feature index {idx} out of bounds for parameter {param_name}")
                    continue
                correlation = np.corrcoef(X[:, idx], y)[0, 1]
                if not np.isnan(correlation):
                    correlations.append(abs(correlation))
            
            # Use max correlation for categorical parameters
            importance[param_name] = max(correlations) if correlations else 0.0
        
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)) 