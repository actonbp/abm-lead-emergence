"""
Analysis pipeline for leadership emergence simulations.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import matplotlib.pyplot as plt
from datetime import datetime

from ..features.time_series import extract_time_series_features
from ..simulation.runner import SimulationRunner
from .clustering import PatternAnalyzer

class AnalysisPipeline:
    """Runs complete analysis workflow on simulation data."""
    
    def __init__(
        self,
        output_dir: str = "data/results",
        random_seed: int = None
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.random_seed = random_seed
        
        # Store results
        self.features_df = None
        self.cluster_labels = None
        self.pattern_stats = None
        
    def run_analysis(
        self,
        simulation_results: List[Dict],
        analysis_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run complete analysis pipeline."""
        # Extract features
        print("Extracting features...")
        features = []
        for result in simulation_results:
            sim_features = extract_time_series_features(result["history"])
            sim_features.update({
                "replication": result["replication"],
                **{f"param_{k}": v for k, v in result["parameters"].items()}
            })
            features.append(sim_features)
        
        self.features_df = pd.DataFrame(features)
        
        # Run clustering
        print("Finding patterns...")
        analyzer = PatternAnalyzer(
            self.features_df,
            n_clusters=analysis_params.get("n_clusters", 3),
            random_state=self.random_seed
        )
        
        # Try both clustering methods
        kmeans_labels, kmeans_score = analyzer.find_clusters_kmeans()
        hdbscan_labels, hdbscan_score = analyzer.find_clusters_hdbscan(
            min_cluster_size=analysis_params.get("min_cluster_size", 5)
        )
        
        # Use better clustering (based on scores)
        if kmeans_score > hdbscan_score:
            self.cluster_labels = kmeans_labels
            clustering_method = "kmeans"
            clustering_score = kmeans_score
        else:
            self.cluster_labels = hdbscan_labels
            clustering_method = "hdbscan"
            clustering_score = hdbscan_score
        
        # Analyze patterns
        print("Analyzing patterns...")
        self.pattern_stats = analyzer.analyze_clusters()
        exemplars = analyzer.get_exemplars(
            n_exemplars=analysis_params.get("n_exemplars", 3)
        )
        
        # Create visualizations
        print("Creating visualizations...")
        self._create_visualizations()
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"analysis_results_{timestamp}.json"
        
        results = {
            "metadata": {
                "n_simulations": len(simulation_results),
                "clustering_method": clustering_method,
                "clustering_score": float(clustering_score),
                "analysis_params": analysis_params,
                "timestamp": timestamp
            },
            "pattern_stats": self.pattern_stats.to_dict(orient="records"),
            "exemplars": {str(k): v.tolist() for k, v in exemplars.items()}
        }
        
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def _create_visualizations(self):
        """Create standard set of visualizations."""
        # Pattern summary plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        
        # Plot 1: Mean LI/FI by cluster
        self.pattern_stats.plot(
            kind='bar',
            x='cluster',
            y=['mean_li', 'mean_fi'],
            ax=axes[0,0],
            title='Identity Levels by Pattern'
        )
        
        # Plot 2: Role differentiation by cluster
        self.pattern_stats.plot(
            kind='bar',
            x='cluster',
            y='mean_role_diff',
            ax=axes[0,1],
            title='Role Differentiation by Pattern'
        )
        
        # Plot 3: Stability by cluster
        self.pattern_stats.plot(
            kind='bar',
            x='cluster',
            y='mean_stability',
            ax=axes[1,0],
            title='Time to Stability by Pattern'
        )
        
        # Plot 4: Cluster sizes
        self.pattern_stats.plot(
            kind='pie',
            y='size',
            ax=axes[1,1],
            title='Pattern Distribution'
        )
        
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(self.output_dir / f"pattern_summary_{timestamp}.png")
        plt.close() 