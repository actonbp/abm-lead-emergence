"""
Batch feature extraction from simulation results.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

from .time_series import TimeSeriesFeatureExtractor

logger = logging.getLogger(__name__)


class BatchFeatureExtractor:
    """Extracts features from batches of simulation results."""
    
    def __init__(
        self,
        feature_extractor: TimeSeriesFeatureExtractor,
        n_jobs: int = -1
    ):
        """Initialize batch feature extractor.
        
        Args:
            feature_extractor: Feature extractor for individual simulations
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        self.feature_extractor = feature_extractor
        
        # Set number of jobs
        if n_jobs < 0:
            self.n_jobs = multiprocessing.cpu_count()
        else:
            self.n_jobs = min(n_jobs, multiprocessing.cpu_count())
    
    def extract_batch_features(
        self,
        batch_dir: str,
        output_file: Optional[str] = None
    ) -> pd.DataFrame:
        """Extract features from all simulations in a batch.
        
        Args:
            batch_dir: Directory containing simulation results
            output_file: Optional path to save features CSV
            
        Returns:
            DataFrame containing extracted features
        """
        batch_dir = Path(batch_dir)
        
        # Load metadata
        metadata_file = batch_dir / "metadata.json"
        if not metadata_file.exists():
            logger.error(f"No metadata file found in {batch_dir}")
            return pd.DataFrame()
            
        with open(metadata_file) as f:
            metadata = json.load(f)
        
        # Get all result files
        result_files = list(batch_dir.glob("*_run_*.json"))
        if not result_files:
            logger.error(f"No result files found in {batch_dir}")
            return pd.DataFrame()
        
        # Extract features in parallel
        logger.info(f"Extracting features from {len(result_files)} simulations...")
        
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = []
            for result_file in result_files:
                future = executor.submit(
                    self._extract_single_features,
                    result_file
                )
                futures.append((result_file, future))
            
            # Process results as they complete
            features_list = []
            for result_file, future in futures:
                try:
                    features = future.result()
                    if features is not None:
                        features_list.append(features)
                except Exception as e:
                    logger.error(f"Failed to extract features from {result_file}: {str(e)}")
        
        if not features_list:
            logger.error("No features could be extracted from any simulation")
            return pd.DataFrame()
        
        # Combine all features
        features_df = pd.DataFrame(features_list)
        
        # Save to file if requested
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            features_df.to_csv(output_file, index=False)
            logger.info(f"Features saved to: {output_file}")
        
        return features_df
    
    def _extract_single_features(self, result_file: Path) -> Optional[Dict[str, Any]]:
        """Extract features from a single simulation result.
        
        Args:
            result_file: Path to simulation result file
            
        Returns:
            Dict containing extracted features and parameters
        """
        try:
            # Load simulation results
            with open(result_file) as f:
                results = json.load(f)
            
            # Extract features
            features = self.feature_extractor.extract_features(results)
            
            # Add parameters
            feature_dict = {
                'run_id': result_file.stem,
                **results['parameters'],
                **features.__dict__
            }
            
            return feature_dict
            
        except Exception as e:
            logger.error(f"Error extracting features from {result_file}: {str(e)}")
            return None 