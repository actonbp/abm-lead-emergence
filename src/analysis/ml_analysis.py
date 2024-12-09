"""
Machine learning analysis of leadership emergence simulation results.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple

class LeadershipMLAnalysis:
    """Analyzes leadership emergence patterns using machine learning."""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.features_df = None
        self.params_df = None
        
    def load_data(self) -> None:
        """Load and combine all analysis results."""
        features_path = self.results_dir / "analysis/features"
        
        # Load all feature files
        dfs = []
        for file in features_path.glob("*.csv"):
            df = pd.read_csv(file)
            dfs.append(df)
        
        self.features_df = pd.concat(dfs, ignore_index=True)
        
        # Extract parameters from index
        param_cols = [col for col in self.features_df.columns 
                     if col.startswith("param_")]
        self.params_df = self.features_df[param_cols]
        
        # Keep only feature columns
        self.features_df = self.features_df.drop(columns=param_cols)
    
    def run_pca(self, n_components: int = 2) -> pd.DataFrame:
        """Perform PCA on features."""
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(self.features_df)
        
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(scaled_features)
        
        # Create DataFrame with PCA results
        pca_df = pd.DataFrame(
            pca_result,
            columns=[f"PC{i+1}" for i in range(n_components)]
        )
        
        # Add parameter columns
        for col in self.params_df.columns:
            pca_df[col] = self.params_df[col]
            
        # Calculate explained variance
        explained_var = pca.explained_variance_ratio_
        print("\nExplained variance ratio:")
        for i, var in enumerate(explained_var):
            print(f"PC{i+1}: {var:.3f}")
            
        return pca_df
    
    def cluster_analysis(
        self,
        n_clusters: int = 3,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Perform cluster analysis on features."""
        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(self.features_df)
        
        # Fit KMeans
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state
        )
        clusters = kmeans.fit_predict(scaled_features)
        
        # Create results DataFrame
        cluster_df = pd.DataFrame({
            "Cluster": clusters
        })
        
        # Add parameter columns
        for col in self.params_df.columns:
            cluster_df[col] = self.params_df[col]
        
        # Calculate cluster statistics
        cluster_stats = {}
        for i in range(n_clusters):
            mask = clusters == i
            cluster_stats[f"cluster_{i}"] = {
                "size": np.sum(mask),
                "centroid": kmeans.cluster_centers_[i],
                "inertia": np.sum(
                    np.linalg.norm(
                        scaled_features[mask] - kmeans.cluster_centers_[i],
                        axis=1
                    )**2
                )
            }
            
        return cluster_df, cluster_stats
    
    def visualize_clusters(
        self,
        cluster_df: pd.DataFrame,
        output_dir: str = None
    ) -> None:
        """Create visualization of clusters with parameter distributions."""
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
        # Create TSNE embedding for visualization
        tsne = TSNE(n_components=2, random_state=42)
        tsne_result = tsne.fit_transform(self.features_df)
        
        # Plot TSNE clusters
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            tsne_result[:, 0],
            tsne_result[:, 1],
            c=cluster_df["Cluster"],
            cmap="viridis"
        )
        plt.colorbar(scatter)
        plt.title("t-SNE visualization of clusters")
        
        if output_dir:
            plt.savefig(output_dir / "tsne_clusters.png")
        plt.close()
        
        # Plot parameter distributions per cluster
        param_cols = [col for col in cluster_df.columns 
                     if col.startswith("param_")]
        
        for param in param_cols:
            plt.figure(figsize=(10, 6))
            sns.boxplot(
                data=cluster_df,
                x="Cluster",
                y=param
            )
            plt.title(f"{param} distribution by cluster")
            
            if output_dir:
                plt.savefig(output_dir / f"{param}_by_cluster.png")
            plt.close()

def main():
    """Run ML analysis on simulation results."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run ML analysis on leadership emergence results"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="data/results",
        help="Directory containing simulation results"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/ml_analysis",
        help="Directory for ML analysis output"
    )
    args = parser.parse_args()
    
    # Initialize analysis
    analysis = LeadershipMLAnalysis(args.results_dir)
    
    # Load data
    print("Loading data...")
    analysis.load_data()
    
    # Run PCA
    print("\nRunning PCA...")
    pca_df = analysis.run_pca()
    
    # Run clustering
    print("\nPerforming cluster analysis...")
    cluster_df, cluster_stats = analysis.cluster_analysis()
    
    # Create visualizations
    print("\nCreating visualizations...")
    analysis.visualize_clusters(
        cluster_df,
        output_dir=args.output_dir
    )
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pca_df.to_csv(output_dir / "pca_results.csv", index=False)
    cluster_df.to_csv(output_dir / "cluster_results.csv", index=False)
    
    print(f"\nAnalysis complete! Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main() 