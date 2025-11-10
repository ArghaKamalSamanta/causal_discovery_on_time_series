"""
Data Generation Module
Generates Syn-6 synthetic data and loads fMRI data
"""

import numpy as np
import scipy.io as sio
from pathlib import Path
import json

class DataGenerator:
    """Generate synthetic time series data and load fMRI data"""
    
    def __init__(self, random_seed=42):
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def generate_syn6_data(self, n_samples=1000, lag=2):
        """
        Generate Syn-6 synthetic data with 6 variables and specified lag
        Based on CDANs paper methodology
        
        Args:
            n_samples: Number of time points
            lag: Maximum lag period
            
        Returns:
            data: Generated time series data (n_samples x 6)
            true_graph: Ground truth causal graph
        """
        # Initialize variables
        X = np.zeros((n_samples + lag, 6))
        
        # Generate noise terms (non-Gaussian)
        eps = np.random.standard_t(df=5, size=(n_samples + lag, 6)) * 0.5
        
        # Generate data according to structural equations
        for t in range(lag, n_samples + lag):
            # X1: Autocorrelated variable
            X[t, 0] = 0.6 * X[t-1, 0] + eps[t, 0]
            
            # X2: Lagged dependency on X1 with time dependency (changing module)
            X[t, 1] = 0.8 * X[t-1, 0] + 1.5 * np.sin(t / 50) + eps[t, 1]
            
            # X3: Lagged dependency on X2 with autocorrelation
            X[t, 2] = 0.7 * X[t-lag, 1] + 0.5 * X[t-lag, 2] + eps[t, 2]
            
            # X4: Contemporaneous dependency on X3
            X[t, 3] = 0.6 * X[t, 2] + eps[t, 3]
            
            # X5: Lagged dependency on X4 with time dependency (changing module)
            X[t, 4] = 0.8 * X[t-lag, 3] + 0.8 * np.sin(t / 20) + eps[t, 4]
            
            # X6: Contemporaneous dependency on X5
            X[t, 5] = 0.7 * X[t, 4] + eps[t, 5]
        
        # Remove initial lag period
        data = X[lag:, :]
        
        # Create ground truth graph structure
        # Format: (source, target, lag)
        true_edges = [
            (0, 0, 1),  # X1(t-1) -> X1(t)
            (0, 1, 1),  # X1(t-1) -> X2(t)
            (1, 2, lag),  # X2(t-lag) -> X3(t)
            (2, 2, lag),  # X3(t-lag) -> X3(t)
            (2, 3, 0),  # X3(t) -> X4(t)
            (3, 4, lag),  # X4(t-lag) -> X5(t)
            (4, 5, 0),  # X5(t) -> X6(t)
        ]
        
        # Changing modules: X2 and X5 depend on time
        changing_modules = [1, 4]
        
        true_graph = {
            'edges': true_edges,
            'changing_modules': changing_modules,
            'n_vars': 6,
            'max_lag': lag
        }
        
        return data, true_graph
    
    def load_fmri_data(self, data_path, sim_index=1):
        """
        Load fMRI simulation data
        
        Args:
            data_path: Path to fMRI data directory
            sim_index: Simulation index (1-28)
            
        Returns:
            data: Time series data for first subject
            true_graph: Ground truth network
            metadata: Additional information
        """
        file_path = Path(data_path) / f'sim{sim_index}.mat'
        
        if not file_path.exists():
            raise FileNotFoundError(f"fMRI data file not found: {file_path}")
        
        # Load MATLAB file
        mat_data = sio.loadmat(str(file_path))
        
        # Extract data
        ts = mat_data['ts']
        net = mat_data['net']
        Nsubjects = int(mat_data['Nsubjects'][0, 0])
        Ntimepoints = int(mat_data['Ntimepoints'][0, 0])
        Nnodes = int(mat_data['Nnodes'][0, 0])
        
        # Get first subject's data
        data = ts[:Ntimepoints, :]
        
        # Get ground truth network (average across subjects)
        true_network = np.squeeze(np.mean(net, axis=0))
        
        metadata = {
            'n_subjects': Nsubjects,
            'n_timepoints': Ntimepoints,
            'n_nodes': Nnodes,
            'sim_index': sim_index
        }
        
        return data, true_network, metadata
    
    def save_data(self, data, filename, metadata=None):
        """Save generated data to file"""
        np.save(filename, data)
        if metadata:
            with open(filename.replace('.npy', '_metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)


def main():
    """Generate and save datasets"""
    generator = DataGenerator()
    
    # Generate Syn-6 data
    print("Generating Syn-6 data...")
    syn6_data, syn6_graph = generator.generate_syn6_data(n_samples=1000, lag=2)
    generator.save_data(syn6_data, 'syn6_data.npy', syn6_graph)
    print(f"Syn-6 data generated: shape {syn6_data.shape}")
    
    # Load fMRI data (example for sim1)
    print("\nLoading fMRI data...")
    try:
        fmri_data, fmri_net, fmri_meta = generator.load_fmri_data('./fmri_data', sim_index=1)
        generator.save_data(fmri_data, 'fmri_sim1_data.npy', fmri_meta)
        print(f"fMRI data loaded: shape {fmri_data.shape}")
    except FileNotFoundError as e:
        print(f"Warning: {e}")
    
    print("\nData generation complete!")


if __name__ == "__main__":
    main()