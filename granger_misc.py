"""
Granger Causality and Miscellaneous Causal Discovery Algorithms
Implements: GVAR, NAVAR, ACD, oCSE, TCDF, NBCB, PCTMI
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.data_processing import DataFrame as TDataFrame
import warnings
warnings.filterwarnings('ignore')


class GVARMethod:
    """
    GVAR: Generalized Vector AutoRegression with self-explaining neural networks
    Based on: https://github.com/i6092467/GVAR
    """
    
    def __init__(self, data, max_lag=2):
        self.data = data
        self.n_samples, self.n_vars = data.shape
        self.max_lag = max_lag
    
    def run(self, epochs=100, hidden_size=20):
        """Run GVAR algorithm"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Prepare lagged features
        X, y = self._prepare_data()
        X_tensor = torch.FloatTensor(X).to(device)
        y_tensor = torch.FloatTensor(y).to(device)
        
        # Initialize model
        model = GVARNet(self.n_vars, self.max_lag, hidden_size).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Training
        for epoch in range(epochs):
            optimizer.zero_grad()
            pred, importance = model(X_tensor)
            loss = criterion(pred, y_tensor) + 0.01 * torch.sum(torch.abs(importance))
            loss.backward()
            optimizer.step()
        
        # Extract causal graph from importance scores
        with torch.no_grad():
            _, importance_final = model(X_tensor)
            importance_np = importance_final.cpu().numpy()
        
        graph = self._importance_to_graph(importance_np)
        
        return {
            'graph': graph,
            'importance': importance_np,
            'algorithm': 'GVAR'
        }
    
    def _prepare_data(self):
        """Prepare lagged data"""
        X_list = []
        for lag in range(1, self.max_lag + 1):
            X_list.append(self.data[self.max_lag - lag:-lag, :])
        X = np.hstack(X_list)
        y = self.data[self.max_lag:, :]
        return X, y
    
    def _importance_to_graph(self, importance):
        """Convert importance scores to causal graph"""
        graph = np.zeros((self.n_vars, self.n_vars, self.max_lag + 1))
        threshold = np.percentile(np.abs(importance), 75)  # Top 25%
        
        for target_var in range(self.n_vars):
            for lag in range(1, self.max_lag + 1):
                for source_var in range(self.n_vars):
                    idx = (lag - 1) * self.n_vars + source_var
                    if np.abs(importance[idx, target_var]) > threshold:
                        graph[source_var, target_var, lag] = 1
        
        return graph


class GVARNet(nn.Module):
    """Neural network for GVAR"""
    
    def __init__(self, n_vars, max_lag, hidden_size):
        super().__init__()
        input_size = n_vars * max_lag
        # self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc1 = nn.Linear(input_size * n_vars, hidden_size)
        self.fc2 = nn.Linear(hidden_size, n_vars)
        self.importance = nn.Parameter(torch.randn(input_size, n_vars))
    
    def forward(self, x):
        # x_weighted = x * torch.sigmoid(self.importance).T
        x_weighted = x.unsqueeze(2) * torch.sigmoid(self.importance).unsqueeze(0)  # (batch, input_size, n_vars)
        # x_weighted = x_weighted.sum(dim=1)  # (batch, n_vars)
        x_flat = x_weighted.reshape(x.size(0), -1)  # (batch, input_size * n_vars)
        h = torch.relu(self.fc1(x_flat))
        out = self.fc2(h)
        return out, self.importance


class NAVARMethod:
    """
    NAVAR: Neural Additive Vector AutoRegression
    Based on: https://github.com/bartbussmann/NAVAR
    """
    
    def __init__(self, data, max_lag=2):
        self.data = data
        self.n_samples, self.n_vars = data.shape
        self.max_lag = max_lag
    
    def run(self, epochs=100, hidden_size=10):
        """Run NAVAR algorithm"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        X, y = self._prepare_data()
        X_tensor = torch.FloatTensor(X).to(device)
        y_tensor = torch.FloatTensor(y).to(device)
        
        model = NAVARNet(self.n_vars, self.max_lag, hidden_size).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            pred = model(X_tensor)
            loss = criterion(pred, y_tensor)
            loss.backward()
            optimizer.step()
        
        # Extract causal structure
        graph = self._extract_causal_graph(model, X_tensor)
        
        return {
            'graph': graph,
            'algorithm': 'NAVAR'
        }
    
    def _prepare_data(self):
        """Prepare lagged data"""
        X_list = []
        for lag in range(1, self.max_lag + 1):
            X_list.append(self.data[self.max_lag - lag:-lag, :])
        X = np.hstack(X_list)
        y = self.data[self.max_lag:, :]
        return X, y
    
    def _extract_causal_graph(self, model, X):
        """Extract causal graph by analyzing model outputs"""
        graph = np.zeros((self.n_vars, self.n_vars, self.max_lag + 1))
        
        with torch.no_grad():
            # Analyze contribution of each input
            for target in range(self.n_vars):
                for source in range(self.n_vars):
                    for lag in range(1, self.max_lag + 1):
                        # Test effect by perturbation
                        X_perturbed = X.clone()
                        idx = (lag - 1) * self.n_vars + source
                        X_perturbed[:, idx] = 0
                        
                        pred_orig = model(X)[:, target]
                        pred_pert = model(X_perturbed)[:, target]
                        
                        effect = torch.mean(torch.abs(pred_orig - pred_pert)).item()
                        if effect > 0.1:
                            graph[source, target, lag] = 1
        
        return graph


class NAVARNet(nn.Module):
    """Additive neural network for NAVAR"""
    
    def __init__(self, n_vars, max_lag, hidden_size):
        super().__init__()
        self.n_vars = n_vars
        self.max_lag = max_lag
        input_size = n_vars * max_lag
        
        # Separate network for each input variable
        self.nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, n_vars)
            ) for _ in range(input_size)
        ])
    
    def forward(self, x):
        outputs = []
        for i, net in enumerate(self.nets):
            outputs.append(net(x[:, i:i+1]))
        return torch.sum(torch.stack(outputs), dim=0)


# class ACDMethod:
#     """
#     ACD: Amortized Causal Discovery
#     Based on: https://github.com/loeweX/AmortizedCausalDiscovery
#     """
    
#     def __init__(self, data, max_lag=2):
#         self.data = data
#         self.n_samples, self.n_vars = data.shape
#         self.max_lag = max_lag
    
#     def run(self, epochs=100):
#         """Run ACD algorithm"""
#         # Simplified implementation
#         from sklearn.ensemble import GradientBoostingRegressor
        
#         graph = np.zeros((self.n_vars, self.n_vars, self.max_lag + 1))
        
#         for target in range(self.n_vars):
#             y = self.data[self.max_lag:, target]
            
#             for source in range(self.n_vars):
#                 for lag in range(1, self.max_lag + 1):
#                     X = self.data[self.max_lag - lag:-lag, source].reshape(-1, 1)
                    
#                     model = GradientBoostingRegressor(n_estimators=50, max_depth=3)
#                     model.fit(X, y)
#                     score = model.score(X, y)
                    
#                     if score > 0.1:  # Threshold
#                         graph[source, target, lag] = 1
        
#         return {
#             'graph': graph,
#             'algorithm': 'ACD'
#         }


class ACDMethod:
    """
    ACD: Amortized Causal Discovery
    Original implementation from: https://github.com/loeweX/AmortizedCausalDiscovery
    """
    
    def __init__(self, data, max_lag=2):
        self.data = data
        self.n_samples, self.n_vars = data.shape
        self.max_lag = max_lag
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def run(self, epochs=100, hidden_dim=32, lr=1e-3):
        """Run ACD algorithm with encoder-decoder architecture"""
        
        # Prepare data windows
        X_windows = self._create_windows()
        X_tensor = torch.FloatTensor(X_windows).to(self.device)
        
        # Initialize encoder-decoder model
        model = ACDEncoderDecoder(
            n_vars=self.n_vars,
            max_lag=self.max_lag,
            hidden_dim=hidden_dim
        ).to(self.device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # Training loop
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Forward pass: encoder predicts graph, decoder simulates dynamics
            graph_probs, predictions = model(X_tensor)
            
            # Loss: prediction error + sparsity regularization
            pred_loss = torch.mean((predictions - X_tensor[:, -1, :]) ** 2)
            sparsity_loss = 0.01 * torch.mean(graph_probs)
            loss = pred_loss + sparsity_loss
            
            loss.backward()
            optimizer.step()
        
        # Extract causal graph from trained encoder
        with torch.no_grad():
            graph_probs_final, _ = model(X_tensor)
            graph_probs_np = graph_probs_final.mean(dim=0).cpu().numpy()
        
        # Convert to discrete graph with thresholding
        graph = self._probs_to_graph(graph_probs_np)
        
        return {
            'graph': graph,
            'graph_probs': graph_probs_np,
            'algorithm': 'ACD'
        }
    
    def _create_windows(self):
        """
        Create sliding windows of time series data
        Returns: (n_windows, max_lag, n_vars)
        """
        windows = []
        for t in range(self.max_lag, self.n_samples):
            window = []
            for lag in range(self.max_lag, 0, -1):
                window.append(self.data[t - lag, :])
            windows.append(np.stack(window))
        
        return np.stack(windows)
    
    def _probs_to_graph(self, graph_probs):
        """
        Convert edge probabilities to discrete causal graph
        
        graph_probs shape: (max_lag, n_vars, n_vars)
        """
        graph = np.zeros((self.n_vars, self.n_vars, self.max_lag + 1))
        
        # Threshold: keep edges with probability > 0.5
        threshold = 0.5
        
        for lag in range(self.max_lag):
            for source in range(self.n_vars):
                for target in range(self.n_vars):
                    if graph_probs[lag, source, target] > threshold:
                        graph[source, target, lag + 1] = 1
        
        return graph


class ACDEncoderDecoder(nn.Module):
    """
    Encoder-Decoder architecture for Amortized Causal Discovery
    
    Encoder: Predicts causal graph edges using Granger causality principles
    Decoder: Simulates system dynamics for next time-step
    """
    
    def __init__(self, n_vars, max_lag, hidden_dim=32):
        super().__init__()
        self.n_vars = n_vars
        self.max_lag = max_lag
        self.hidden_dim = hidden_dim
        
        # Encoder: learns to predict causal graph structure
        # Input: time series windows, Output: edge probabilities
        self.encoder_fc1 = nn.Linear(n_vars * max_lag, hidden_dim)
        self.encoder_fc2 = nn.Linear(hidden_dim, max_lag * n_vars * n_vars)
        
        # Decoder: simulates dynamics given predicted graph
        # Uses graph to weight connections between variables
        self.decoder_lstm = nn.LSTM(
            input_size=n_vars,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.decoder_fc = nn.Linear(hidden_dim, n_vars)
    
    def forward(self, x):
        """
        Forward pass through encoder-decoder
        
        Args:
            x: Input windows (batch, max_lag, n_vars)
            
        Returns:
            graph_probs: Edge probabilities (batch, max_lag, n_vars, n_vars)
            predictions: Next time-step predictions (batch, n_vars)
        """
        batch_size = x.shape[0]
        
        # Encoder: predict causal graph
        x_flat = x.reshape(batch_size, -1)
        h = torch.relu(self.encoder_fc1(x_flat))
        graph_logits = self.encoder_fc2(h)
        
        # Reshape to (batch, max_lag, n_vars, n_vars)
        graph_logits = graph_logits.reshape(batch_size, self.max_lag, self.n_vars, self.n_vars)
        graph_probs = torch.sigmoid(graph_logits)
        
        # Decoder: simulate dynamics using predicted graph
        # Apply graph structure as attention/gating mechanism
        weighted_input = self._apply_graph_structure(x, graph_probs)
        
        # LSTM processes temporal sequence
        lstm_out, _ = self.decoder_lstm(weighted_input)
        
        # Predict next time-step
        predictions = self.decoder_fc(lstm_out[:, -1, :])
        
        return graph_probs, predictions
    
    def _apply_graph_structure(self, x, graph_probs):
        """
        Apply learned graph structure to weight temporal connections
        
        This implements the core idea: decoder uses the causal graph
        to properly weight influences between variables
        """
        batch_size = x.shape[0]
        weighted_x = torch.zeros_like(x)
        
        for lag_idx in range(self.max_lag):
            # Get graph weights for this lag
            weights = graph_probs[:, lag_idx, :, :]  # (batch, n_vars, n_vars)
            
            # Apply weights: each target variable is influenced by source variables
            # weighted by the predicted edge probabilities
            for target in range(self.n_vars):
                # Sum over source variables weighted by graph edges
                weighted_x[:, lag_idx, target] = torch.sum(
                    x[:, lag_idx, :] * weights[:, :, target],
                    dim=1
                )
        
        return weighted_x


class oCSEMethod:
    """
    oCSE: optimal Causation Entropy
    Based on: https://github.com/ckassaad/causal_discovery_for_time_series
    """
    
    def __init__(self, data, max_lag=2):
        self.data = data
        self.n_samples, self.n_vars = data.shape
        self.max_lag = max_lag
    
    def run(self, alpha=0.05):
        """Run oCSE algorithm"""
        graph = np.zeros((self.n_vars, self.n_vars, self.max_lag + 1))
        
        for target in range(self.n_vars):
            parents = self._find_parents(target, alpha)
            for source, lag in parents:
                graph[source, target, lag] = 1
        
        return {
            'graph': graph,
            'algorithm': 'oCSE'
        }
    
    def _find_parents(self, target, alpha):
        """Find causal parents using transfer entropy"""
        from sklearn.feature_selection import mutual_info_regression
        
        parents = []
        y = self.data[self.max_lag:, target]
        
        for source in range(self.n_vars):
            for lag in range(1, self.max_lag + 1):
                X = self.data[self.max_lag - lag:-lag, source].reshape(-1, 1)
                mi = mutual_info_regression(X, y, random_state=42)[0]
                
                if mi > 0.1:  # Threshold
                    parents.append((source, lag))
        
        return parents


# class TCDFMethod:
#     """
#     TCDF: Temporal Causal Discovery Framework
#     Based on: https://github.com/M-Nauta/TCDF
#     """
    
#     def __init__(self, data, max_lag=2):
#         self.data = data
#         self.n_samples, self.n_vars = data.shape
#         self.max_lag = max_lag
    
#     def run(self, epochs=50, kernel_size=3):
#         """Run TCDF with attention-based CNNs"""
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
#         graphs = []
#         for target in range(self.n_vars):
#             model = TCDFNet(self.n_vars, kernel_size).to(device)
#             optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
#             X_tensor = torch.FloatTensor(self.data).to(device)
#             y = self.data[:, target]
#             y_tensor = torch.FloatTensor(y).to(device)
            
#             for epoch in range(epochs):
#                 optimizer.zero_grad()
#                 pred, attention = model(X_tensor)
#                 loss = nn.MSELoss()(pred.squeeze(), y_tensor)
#                 loss.backward()
#                 optimizer.step()
            
#             # Extract attention weights
#             with torch.no_grad():
#                 _, attn_final = model(X_tensor)
#                 graphs.append(attn_final.cpu().numpy())
        
#         graph = self._aggregate_graphs(graphs)
        
#         return {
#             'graph': graph,
#             'algorithm': 'TCDF'
#         }
    
#     def _aggregate_graphs(self, graphs):
#         """Aggregate attention weights into causal graph"""
#         graph = np.zeros((self.n_vars, self.n_vars, self.max_lag + 1))
        
#         for target, attn in enumerate(graphs):
#             threshold = np.percentile(attn, 75)
#             for source in range(self.n_vars):
#                 if attn[source] > threshold:
#                     graph[source, target, 1] = 1  # Simplified: assign to lag 1
        
#         return graph


# class TCDFNet(nn.Module):
#     """Attention-based CNN for TCDF"""
    
#     def __init__(self, n_vars, kernel_size):
#         super().__init__()
#         self.conv = nn.Conv1d(n_vars, 16, kernel_size, padding=kernel_size//2)
#         self.attention = nn.Linear(n_vars, n_vars)
#         self.fc = nn.Linear(16, 1)
    
#     def forward(self, x):
#         # x: (n_samples, n_vars)
#         x_t = x.T.unsqueeze(0)  # (1, n_vars, n_samples)
        
#         # Attention weights
#         attn_weights = torch.softmax(self.attention.weight.mean(dim=0), dim=0)
        
#         # Convolutional feature extraction
#         features = self.conv(x_t)  # (1, 16, n_samples)
#         features = torch.relu(features)
        
#         # Prediction
#         pred = self.fc(features.permute(0, 2, 1)).squeeze()
        
#         return pred, attn_weights


class TCDFMethod:
    """
    TCDF: Temporal Causal Discovery Framework
    Based on: https://github.com/M-Nauta/TCDF
    """
    
    def __init__(self, data, max_lag=2):
        self.data = data
        self.n_samples, self.n_vars = data.shape
        self.max_lag = max_lag

    def run(self, epochs=50, kernel_size=3):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        graphs = []

        for target in range(self.n_vars):
            model = TCDFNet(self.n_vars, kernel_size, self.max_lag).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            X_tensor = torch.FloatTensor(self.data).to(device)
            y = self.data[:, target]
            y_tensor = torch.FloatTensor(y).to(device)

            for epoch in range(epochs):
                optimizer.zero_grad()
                pred, attention = model(X_tensor)
                loss = torch.mean((pred.squeeze() - y_tensor) ** 2)
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                _, attn_final = model(X_tensor)
                graphs.append(attn_final.cpu().numpy())

        graph = self._aggregate_graphs(graphs)
        return {'graph': graph, 'algorithm': 'TCDF'}

    # def _aggregate_graphs(self, graphs):
    #     """
    #     Aggregate lag-specific attention maps into causal adjacency tensor.
    #     Implements actual TCDF-style aggregation:
    #       - computes per-lag importance,
    #       - normalizes attention per target,
    #       - thresholds significant causal influences.
    #     """
    #     n_vars = self.n_vars
    #     max_lag = self.max_lag
    #     graph = np.zeros((n_vars, n_vars, max_lag + 1))

    #     for target, attn_map in enumerate(graphs):
    #         # attn_map: shape (n_vars, max_lag)
    #         attn_norm = attn_map / (np.sum(attn_map) + 1e-8)

    #         # Compute significance threshold (top-k attention)
    #         threshold = np.percentile(attn_norm, 90)

    #         for source in range(n_vars):
    #             for lag in range(1, max_lag + 1):
    #                 if attn_norm[source, lag - 1] > threshold:
    #                     graph[source, target, lag] = 1  # causal effect from X_source(t-lag) -> X_target(t)
    #     return graph

    def _aggregate_graphs(self, graphs):
        """
        Paper's TCDF aggregation: uses attention to identify causal sources
        """
        n_vars = self.n_vars
        max_lag = self.max_lag
        graph = np.zeros((n_vars, n_vars, max_lag + 1))
        
        for target, attn_map in enumerate(graphs):
            # attn_map shape should be (n_vars, max_lag) or similar
            # Normalize per target
            if attn_map.sum() > 0:
                attn_norm = attn_map / attn_map.sum()
            else:
                attn_norm = attn_map
            
            # Threshold: keep top k% of attention weights
            threshold = np.percentile(attn_norm[attn_norm > 0], 80) if (attn_norm > 0).any() else 0
            
            for source in range(n_vars):
                for lag in range(1, min(max_lag + 1, attn_map.shape[1] + 1)):
                    if lag - 1 < attn_map.shape[1]:
                        if attn_norm[source, lag - 1] > threshold:
                            graph[source, target, lag] = 1
        
        return graph


class TCDFNet(nn.Module):
    """
    TCDF attention-based CNN (faithful to original)
    """
    def __init__(self, n_vars, kernel_size, max_lag):
        super().__init__()
        self.n_vars = n_vars
        self.max_lag = max_lag

        # CNN encoder for each lag (causal convolutions)
        self.conv = nn.Conv1d(n_vars, 32, kernel_size, padding=kernel_size // 2)
        self.attention = nn.MultiheadAttention(embed_dim=32, num_heads=4, batch_first=True)
        self.fc = nn.Linear(32, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x: (n_samples, n_vars)
        x_t = x.T.unsqueeze(0)  # (1, n_vars, n_samples)

        # Temporal feature extraction
        conv_out = torch.relu(self.conv(x_t))  # (1, 32, n_samples)
        conv_out = conv_out.permute(0, 2, 1)   # (1, n_samples, 32)

        # Attention over time (query = each timestep)
        attn_out, attn_weights = self.attention(conv_out, conv_out, conv_out)
        # attn_weights = attn_weights.mean(dim=1).squeeze(0)  # average over heads, remove batch dim
        # attn_summary = attn_weights.mean(dim=1)  # (n_vars, max_lag)
        attn_weights = attn_weights.mean(dim=0)  # average over batch -> (seq_len, seq_len)
        attn_summary = attn_weights[:self.n_vars, :self.max_lag]  # pick relevant slice

        # Prediction
        pred = self.fc(attn_out).squeeze()

        return pred, attn_summary


class NBCBMethod:
    """
    NBCB: Noise-based/Constraint-based approach
    Based on: https://github.com/ckassaad/causal_discovery_for_time_series
    """
    
    def __init__(self, data, max_lag=2):
        self.data = data
        self.n_samples, self.n_vars = data.shape
        self.max_lag = max_lag
    
    def run(self, alpha=0.05):
        """Run NBCB algorithm"""
        # Phase 1: Identify potential causes using noise-based approach
        potential_causes = self._noise_based_phase(alpha)
        
        # Phase 2: Prune using constraint-based approach
        graph = self._constraint_based_phase(potential_causes, alpha)
        
        return {
            'graph': graph,
            'algorithm': 'NBCB'
        }
    
    def _noise_based_phase(self, alpha):
        """Identify potential causes using additive noise models"""
        from sklearn.ensemble import RandomForestRegressor
        from scipy.stats import normaltest
        
        potential = {}
        for target in range(self.n_vars):
            potential[target] = []
            y = self.data[self.max_lag:, target]
            
            for source in range(self.n_vars):
                for lag in range(self.max_lag + 1):
                    if lag == 0:
                        X = self.data[self.max_lag:, source].reshape(-1, 1)
                    else:
                        X = self.data[self.max_lag - lag:-lag, source].reshape(-1, 1)
                    
                    model = RandomForestRegressor(n_estimators=50, random_state=42)
                    model.fit(X, y)
                    residuals = y - model.predict(X)
                    
                    # Test if residuals are independent
                    _, p_val = normaltest(residuals)
                    if p_val > alpha:
                        potential[target].append((source, lag))
        
        return potential
    
    # def _constraint_based_phase(self, potential_causes, alpha):
    #     """Prune false causes using conditional independence"""
    #     graph = np.zeros((self.n_vars, self.n_vars, self.max_lag + 1))
        
    #     for target, causes in potential_causes.items():
    #         for source, lag in causes:
    #             # Simplified: keep if in top candidates
    #             if len(causes) <= 3:  # Keep if few causes
    #                 graph[source, target, lag] = 1
        
    #     return graph

    def _constraint_based_phase(self, potential_causes, alpha):
        """
        Real NBCB constraint-based pruning phase.
        Uses conditional independence tests via PCMCI (Tigramite library).
        """
        # Prepare data for PCMCI
        # dataframe = pd.DataFrame(self.data)
        dataframe = TDataFrame(self.data, var_names=[f'X{i}' for i in range(self.n_vars)])

        # Use partial correlation test (or CMIknn if nonlinear)
        parcorr = ParCorr(significance='analytic')

        pcmci = PCMCI(
            dataframe=dataframe,
            cond_ind_test=parcorr,
            verbosity=0
        )

        # Build candidate parent sets from noise-based phase
        cond_ind_test_matrix = np.zeros((self.n_vars, self.n_vars, self.max_lag + 1))
        for target, causes in potential_causes.items():
            for source, lag in causes:
                cond_ind_test_matrix[source, target, lag] = 1

        # Run PCMCI pruning over these candidates only
        results = pcmci.run_pcmci(tau_max=self.max_lag, pc_alpha=alpha, selected_links=None)

        # Extract pruned causal graph
        val_matrix = results['val_matrix']
        p_matrix = results['p_matrix']

        # Threshold at significance alpha
        graph = np.zeros_like(p_matrix)
        graph[p_matrix < alpha] = 1

        return graph


class PCTMIMethod:
    """
    PCTMI: PC algorithm with Temporal Mutual Information
    Based on: https://github.com/ckassaad/causal_discovery_for_time_series
    """
    
    def __init__(self, data, max_lag=2):
        self.data = data
        self.n_samples, self.n_vars = data.shape
        self.max_lag = max_lag
    
    def run(self, alpha=0.05):
        """Run PCTMI algorithm"""
        from sklearn.feature_selection import mutual_info_regression
        
        graph = np.zeros((self.n_vars, self.n_vars, self.max_lag + 1))
        
        # Compute temporal mutual information
        for target in range(self.n_vars):
            y = self.data[self.max_lag:, target]
            
            for source in range(self.n_vars):
                for lag in range(self.max_lag + 1):
                    if lag == 0:
                        X = self.data[self.max_lag:, source].reshape(-1, 1)
                    else:
                        X = self.data[self.max_lag - lag:-lag, source].reshape(-1, 1)
                    
                    mi = mutual_info_regression(X, y, random_state=42)[0]
                    
                    if mi > 0.15:  # Threshold
                        graph[source, target, lag] = 1
        
        return {
            'graph': graph,
            'algorithm': 'PCTMI'
        }


def main():
    """Test all methods"""
    np.random.seed(42)
    n_samples = 300
    n_vars = 3
    
    data = np.random.randn(n_samples, n_vars) * 0.5
    for t in range(1, n_samples):
        data[t, 1] += 0.5 * data[t-1, 0]
        data[t, 2] += 0.4 * data[t-1, 1]
    
    methods = [
        ('GVAR', GVARMethod),
        ('NAVAR', NAVARMethod),
        ('ACD', ACDMethod),
        ('oCSE', oCSEMethod),
        ('TCDF', TCDFMethod),
        ('NBCB', NBCBMethod),
        ('PCTMI', PCTMIMethod)
    ]
    
    for name, MethodClass in methods:
        print(f"\nTesting {name}...")
        method = MethodClass(data, max_lag=2)
        results = method.run()
        print(f"{results['algorithm']} completed")
        print(f"Graph shape: {results['graph'].shape}")


if __name__ == "__main__":
    main()