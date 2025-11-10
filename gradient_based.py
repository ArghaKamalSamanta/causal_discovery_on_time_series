"""
Gradient-Based Causal Discovery Algorithms for Time Series
Implements: DYNOTEARS, NTS-NOTEARS
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.special import expit as sigmoid
import warnings
warnings.filterwarnings('ignore')


class DYNOTEARSMethod:
    """
    DYNOTEARS: Structure learning from time-series data
    Based on: https://github.com/quantumblacklabs/causalnex
    """
    
    def __init__(self, data, max_lag=2):
        """
        Initialize DYNOTEARS
        
        Args:
            data: Time series data (n_samples x n_vars)
            max_lag: Maximum lag to consider
        """
        self.data = data
        self.n_samples, self.n_vars = data.shape
        self.max_lag = max_lag
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def run(self, lambda1=0.01, lambda2=0.01, max_iter=100, h_tol=1e-8):
        """
        Run DYNOTEARS algorithm
        
        Args:
            lambda1: L1 regularization parameter
            lambda2: DAG regularization parameter
            max_iter: Maximum iterations
            h_tol: Tolerance for acyclicity constraint
            
        Returns:
            results: Dictionary with intra-slice and inter-slice graphs
        """
        # Prepare lagged data
        X, X_lag = self._prepare_data()
        
        # Initialize weight matrices
        W_intra = torch.zeros(self.n_vars, self.n_vars, requires_grad=True, device=self.device)
        W_inter = torch.zeros(self.n_vars * self.max_lag, self.n_vars, requires_grad=True, device=self.device)
        
        optimizer = torch.optim.Adam([W_intra, W_inter], lr=0.001)
        
        # Convert data to torch tensors
        X_torch = torch.FloatTensor(X).to(self.device)
        X_lag_torch = torch.FloatTensor(X_lag).to(self.device)
        
        # Optimization loop
        for iteration in range(max_iter):
            optimizer.zero_grad()
            
            # Compute predictions
            # pred = X_torch @ W_intra.T + X_lag_torch @ W_inter.T
            pred = X_torch @ W_intra.T + X_lag_torch @ W_inter
            
            # Compute loss
            loss_fit = 0.5 / X.shape[0] * torch.sum((X_torch - pred) ** 2)
            loss_l1 = lambda1 * (torch.sum(torch.abs(W_intra)) + torch.sum(torch.abs(W_inter)))
            h = self._compute_acyclicity(W_intra)
            loss = loss_fit + loss_l1 + lambda2 * h
            
            # Backpropagate
            loss.backward()
            optimizer.step()
            
            # Check convergence
            if h.item() < h_tol:
                break
        
        # Extract results
        W_intra_np = W_intra.detach().cpu().numpy()
        W_inter_np = W_inter.detach().cpu().numpy()
        
        # Threshold small values
        threshold = 0.3
        # W_intra_np[np.abs(W_intra_np) < threshold] = 0
        # W_inter_np[np.abs(W_inter_np) < threshold] = 0

        W_intra_np = W_intra_np * (np.abs(W_intra_np) > threshold)
        W_inter_np = W_inter_np * (np.abs(W_inter_np) > threshold)
        
        # Convert to standard format
        graph = self._convert_to_graph(W_intra_np, W_inter_np)
        
        return {
            'graph': graph,
            'W_intra': W_intra_np,
            'W_inter': W_inter_np,
            'algorithm': 'DYNOTEARS'
        }
    
    def _prepare_data(self):
        """Prepare lagged data matrices"""
        n = self.n_samples - self.max_lag
        X = self.data[self.max_lag:, :]
        
        X_lag = []
        for lag in range(1, self.max_lag + 1):
            X_lag.append(self.data[self.max_lag - lag:-lag, :])
        
        X_lag = np.hstack(X_lag)
        return X, X_lag
    
    def _compute_acyclicity(self, W):
        """Compute acyclicity constraint h(W) = tr(e^(W◦W)) - d"""
        d = W.shape[0]
        M = W * W
        E = torch.matrix_exp(M)
        h = torch.trace(E) - d
        return h
    
    def _convert_to_graph(self, W_intra, W_inter):
        """Convert weight matrices to graph format"""
        graph = np.zeros((self.n_vars, self.n_vars, self.max_lag + 1))
        
        # Contemporaneous effects
        graph[:, :, 0] = W_intra
        
        # Lagged effects
        for lag in range(1, self.max_lag + 1):
            start_idx = (lag - 1) * self.n_vars
            end_idx = lag * self.n_vars
            graph[:, :, lag] = W_inter[start_idx:end_idx, :].T
        
        return graph


class NTSNOTEARSMethod:
    """
    NTS-NOTEARS: Learning Nonparametric Temporal DAGs with Time-Series Data
    Based on: https://github.com/xiangyu-sun-789/NTS-NOTEARS
    """
    
    def __init__(self, data, max_lag=2):
        """
        Initialize NTS-NOTEARS
        
        Args:
            data: Time series data (n_samples x n_vars)
            max_lag: Maximum lag to consider
        """
        self.data = data
        self.n_samples, self.n_vars = data.shape
        self.max_lag = max_lag
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def run(self, lambda1=0.01, lambda2=0.01, max_iter=100, hidden_layers=[10]):
        """
        Run NTS-NOTEARS algorithm using 1D CNNs
        
        Args:
            lambda1: L1 regularization
            lambda2: DAG regularization
            max_iter: Maximum iterations
            hidden_layers: Hidden layer sizes for CNN
            
        Returns:
            results: Dictionary with causal graph
        """
        # Build model
        model = TemporalCNN(self.n_vars, self.max_lag, hidden_layers).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Prepare data
        X_tensor = torch.FloatTensor(self.data).to(self.device)
        
        # Training loop
        for iteration in range(max_iter):
            optimizer.zero_grad()
            
            # Forward pass
            pred, W = model(X_tensor)
            
            # Compute loss
            loss_fit = torch.mean((X_tensor[self.max_lag:] - pred) ** 2)
            loss_l1 = lambda1 * torch.sum(torch.abs(W))
            h = self._compute_acyclicity(W)
            loss = loss_fit + loss_l1 + lambda2 * h
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if iteration % 20 == 0 and h.item() < 1e-8:
                break
        
        # Extract causal graph
        with torch.no_grad():
            _, W_final = model(X_tensor)
            W_np = W_final.cpu().numpy()
        
        # Threshold and convert
        threshold = 0.3
        W_np[np.abs(W_np) < threshold] = 0
        
        graph = self._extract_graph(W_np, model)
        
        return {
            'graph': graph,
            'weights': W_np,
            'algorithm': 'NTS-NOTEARS'
        }
    
    def _compute_acyclicity(self, W):
        # """Compute acyclicity constraint"""
        # # Simplified version for demonstration
        # d = W.shape[0]
        # M = W * W
        # # Use approximation to avoid expensive matrix exponential
        # h = torch.trace(torch.matrix_power(torch.eye(d, device=W.device) + M / d, d)) - d
        # return h

        """
        Compute acyclicity constraint using matrix exponential as in NOTEARS.

        h(W) = tr(exp(W ⊙ W)) - d = 0  if and only if W is acyclic.
        (⊙ is element-wise multiplication)

        This is the *actual* formulation from NOTEARS (Zheng et al., 2018)
        and retained in NTS-NOTEARS for instantaneous (non-lagged) edges.
        """
        d = W.shape[0]
        # Compute matrix exponential of elementwise square
        expm = torch.matrix_exp(W * W)
        h = torch.trace(expm) - d
        return h
    
    # def _extract_graph(self, W, model):
    #     # """Extract temporal graph from CNN model"""
    #     # graph = np.zeros((self.n_vars, self.n_vars, self.max_lag + 1))
        
    #     # # Extract from convolutional layers
    #     # # This is a simplified extraction
    #     # for lag in range(self.max_lag + 1):
    #     #     graph[:, :, lag] = W
        
    #     # return graph

    #     """
    #     Extract temporal causal graph from convolutional weights.
    #     This is the *actual* temporal edge extraction process
    #     following the NTS-NOTEARS paper.

    #     The convolutional kernels encode lagged dependencies,
    #     while the learned W matrix encodes instantaneous edges.
    #     """
    #     # Get instantaneous dependencies
    #     W_inst = model.W.detach().cpu().numpy()

    #     # Extract lagged dependencies from CNN kernels
    #     lagged_graph = np.zeros((self.n_vars, self.n_vars, self.max_lag))
    #     for conv in model.conv_layers:
    #         # Each kernel shape: (out_channels, in_channels, kernel_size)
    #         weights = conv.weight.detach().cpu().numpy()
    #         kernel_size = weights.shape[2]

    #         # Aggregate effect per lag
    #         for lag in range(min(self.max_lag, kernel_size)):
    #             # Average over hidden channels
    #             lagged_graph[:, :, lag] += np.mean(weights[:, :, lag], axis=0)

    #     # Combine instantaneous + lagged
    #     full_graph = np.concatenate(
    #         [W_inst[:, :, np.newaxis], lagged_graph], axis=2
    #     )

    #     # Thresholding small weights
    #     full_graph[np.abs(full_graph) < 0.1] = 0

    #     return full_graph

    def _extract_graph(self, W, model):
        """
        Extract temporal causal graph - paper uses lagged dependencies only
        """
        # Get instantaneous dependencies from W matrix
        W_inst = W.copy()
        W_inst = W_inst * (np.abs(W_inst) > 0.1)  # Threshold
        
        # Extract lagged dependencies from CNN - paper methodology
        n_vars = self.n_vars
        max_lag = self.max_lag
        
        # Initialize: shape (n_vars, n_vars, max_lag+1)
        graph = np.zeros((n_vars, n_vars, max_lag + 1))
        
        # Instantaneous effects (lag 0)
        graph[:, :, 0] = W_inst
        
        # Lagged effects from convolutional weights
        # The paper extracts these from kernel importance
        for conv_layer in model.conv_layers:
            kernel_weights = conv_layer.weight.detach().cpu().numpy()
            # kernel shape: (out_channels, in_channels, kernel_size)
            
            for lag_idx in range(min(max_lag, kernel_weights.shape[2] - 1)):
                # Average over channels to get variable-to-variable strength
                lag_weights = np.mean(np.abs(kernel_weights[:, :, lag_idx]), axis=0)

                if lag_weights.ndim == 1:
                    lag_weights = np.diag(lag_weights)
                
                # Reshape to (n_vars, n_vars) if needed
                if lag_weights.shape[0] == n_vars:
                    graph[:, :, lag_idx + 1] += lag_weights[:n_vars, :n_vars]
        
        # Final threshold
        graph = graph * (np.abs(graph) > 0.1)
        
        return graph


class TemporalCNN(nn.Module):
    """1D CNN for temporal causal discovery"""
    
    def __init__(self, n_vars, max_lag, hidden_layers=[10]):
        super().__init__()
        self.n_vars = n_vars
        self.max_lag = max_lag
        
        # Learnable weight matrix for DAG structure
        self.W = nn.Parameter(torch.zeros(n_vars, n_vars))
        
        # 1D Convolutional layers for temporal patterns
        self.conv_layers = nn.ModuleList()
        in_channels = n_vars
        
        for hidden_size in hidden_layers:
            self.conv_layers.append(
                nn.Conv1d(in_channels, hidden_size, kernel_size=max_lag + 1, padding=max_lag)
            )
            in_channels = hidden_size
        
        # Output layer
        self.output_layer = nn.Conv1d(in_channels, n_vars, kernel_size=1)
    
    def forward(self, X):
        """
        Forward pass
        
        Args:
            X: Input tensor (n_samples, n_vars)
            
        Returns:
            pred: Predictions (n_samples - max_lag, n_vars)
            W: Weight matrix
        """
        n_samples = X.shape[0]
        
        # Reshape for 1D conv: (batch, channels, length)
        x = X.T.unsqueeze(0)  # (1, n_vars, n_samples)
        
        # Apply conv layers
        for conv in self.conv_layers:
            x = torch.relu(conv(x))
        
        # Output layer
        x = self.output_layer(x)

        # x shape: (1, n_vars, L_out), due to padding L_out = n_samples + max_lag
        # Trim extra padding at the start to match original length
        x = x[:, :, self.max_lag:n_samples + self.max_lag]
        
        # Reshape back
        pred = x.squeeze(0).T[self.max_lag:]
        
        # Apply structural constraint
        W_masked = self.W * (torch.abs(self.W) > 0.1).float()
        
        return pred, W_masked


def main():
    """Test gradient-based methods"""
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Generate test data
    n_samples = 500
    n_vars = 3
    data = np.random.randn(n_samples, n_vars) * 0.5
    
    for t in range(1, n_samples):
        data[t, 1] += 0.6 * data[t-1, 0]
        data[t, 2] += 0.4 * data[t-1, 1]
    
    print("Testing DYNOTEARS...")
    dynotears = DYNOTEARSMethod(data, max_lag=2)
    results_dyno = dynotears.run(max_iter=50)
    print(f"DYNOTEARS completed: {results_dyno['algorithm']}")
    print(f"Graph shape: {results_dyno['graph'].shape}")
    
    print("\nTesting NTS-NOTEARS...")
    nts = NTSNOTEARSMethod(data, max_lag=2)
    results_nts = nts.run(max_iter=50)
    print(f"NTS-NOTEARS completed: {results_nts['algorithm']}")
    print(f"Graph shape: {results_nts['graph'].shape}")


if __name__ == "__main__":
    main()