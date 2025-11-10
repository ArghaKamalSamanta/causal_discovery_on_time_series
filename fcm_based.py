"""
Functional Causal Model (FCM) Based Algorithms for Time Series
Implements: VarLiNGAM, TiMINo
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')


class VarLiNGAMMethod:
    """
    VarLiNGAM: Vector Autoregressive Linear Non-Gaussian Acyclic Model
    Implementation: https://lingam.readthedocs.io/en/latest/tutorial/var.html
    """
    
    def __init__(self, data, max_lag=2):
        """
        Initialize VarLiNGAM
        
        Args:
            data: Time series data (n_samples x n_vars)
            max_lag: Maximum lag to consider
        """
        self.data = data
        self.n_samples, self.n_vars = data.shape
        self.max_lag = max_lag
    
    def run(self, prune=True, alpha=0.05):
        """
        Run VarLiNGAM algorithm
        
        Args:
            prune: Whether to prune insignificant edges
            alpha: Significance level for pruning
            
        Returns:
            results: Dictionary with causal graph and coefficients
        """
        try:
            from lingam import VARLiNGAM
        except ImportError:
            print("Warning: lingam package not installed. Install with: pip install lingam")
            return self._fallback_method()
        
        # Initialize and fit VARLiNGAM
        model = VARLiNGAM(lags=self.max_lag, prune=prune, criterion='bic')
        model.fit(self.data)
        
        # Extract causal graph
        # adjacency_matrices_ has shape (n_lags, n_vars, n_vars)
        causal_order = model.causal_order_
        adjacency_matrices = model.adjacency_matrices_
        
        # Convert to standard format: (n_vars, n_vars, n_lags)
        # graph = np.zeros((self.n_vars, self.n_vars, self.max_lag + 1))
        graph = np.zeros((self.n_vars, self.n_vars, len(adjacency_matrices) + 1))
        
        # Contemporaneous effects (lag 0)
        if hasattr(model, 'adjacency_matrix_'):
            graph[:, :, 0] = model.adjacency_matrix_
        
        # Lagged effects
        for lag_idx in range(len(adjacency_matrices)):
            graph[:, :, lag_idx + 1] = adjacency_matrices[lag_idx].T
        
        return {
            'graph': graph,
            'causal_order': causal_order,
            'adjacency_matrices': adjacency_matrices,
            'algorithm': 'VarLiNGAM'
        }
    
    def _fallback_method(self):
        """Fallback using basic VAR estimation"""
        from statsmodels.tsa.api import VAR
        
        model = VAR(self.data)
        results = model.fit(maxlags=self.max_lag, ic='bic')
        
        # Extract coefficients
        graph = np.zeros((self.n_vars, self.n_vars, self.max_lag + 1))
        
        for lag in range(1, self.max_lag + 1):
            coef_matrix = results.params[lag * self.n_vars:(lag + 1) * self.n_vars, :]
            graph[:, :, lag] = coef_matrix.T
        
        # Threshold small coefficients
        threshold = 0.1
        graph[np.abs(graph) < threshold] = 0
        
        return {
            'graph': graph,
            'algorithm': 'VarLiNGAM (fallback VAR)'
        }


# class TiMINoMethod:
#     """
#     TiMINo: Time-series Models with Independent Noise
#     Based on: https://github.com/ckassaad/causal_discovery_for_time_series
#     """
    
#     def __init__(self, data, max_lag=2):
#         """
#         Initialize TiMINo
        
#         Args:
#             data: Time series data (n_samples x n_vars)
#             max_lag: Maximum lag to consider
#         """
#         self.data = data
#         self.n_samples, self.n_vars = data.shape
#         self.max_lag = max_lag
    
#     def run(self, alpha=0.05, method='kernel'):
#         """
#         Run TiMINo algorithm
        
#         Args:
#             alpha: Significance level
#             method: Independence test method ('kernel' or 'hsic')
            
#         Returns:
#             results: Dictionary with causal graph
#         """
#         try:
#             from causal_discovery_ts import TiMINO as TiMINO_lib
#         except ImportError:
#             print("Warning: causal_discovery_ts not available. Using simplified implementation.")
#             return self._simplified_timino(alpha)
        
#         # Run TiMINo
#         model = TiMINO_lib(sig_level=alpha, lag=self.max_lag)
#         graph_dict = model.fit(self.data)
        
#         # Convert to standard format
#         graph = self._dict_to_graph(graph_dict)
        
#         return {
#             'graph': graph,
#             'algorithm': 'TiMINo'
#         }
    
#     def _simplified_timino(self, alpha):
#         """
#         Simplified TiMINo implementation
#         Uses independence tests on residuals
#         """
#         from sklearn.gaussian_process import GaussianProcessRegressor
#         from sklearn.gaussian_process.kernels import RBF
#         from scipy.stats import kstest
        
#         graph = np.zeros((self.n_vars, self.n_vars, self.max_lag + 1))
        
#         # For each variable, find its causes
#         for j in range(self.n_vars):
#             target = self.data[self.max_lag:, j]
            
#             # Test each potential cause
#             for i in range(self.n_vars):
#                 # Test contemporaneous effect
#                 X_contemp = self.data[self.max_lag:, i].reshape(-1, 1)
                
#                 if self._test_independence(X_contemp, target, alpha):
#                     graph[i, j, 0] = 1
                
#                 # Test lagged effects
#                 for lag in range(1, self.max_lag + 1):
#                     X_lag = self.data[self.max_lag - lag:-lag, i].reshape(-1, 1)
                    
#                     if self._test_independence(X_lag, target, alpha):
#                         graph[i, j, lag] = 1
        
#         return {
#             'graph': graph,
#             'algorithm': 'TiMINo (simplified)'
#         }
    
#     def _test_independence(self, X, y, alpha):
#         """
#         Test if y is independent of X using nonlinear regression residuals
        
#         Returns True if dependent (p_value < alpha)
#         """
#         from sklearn.ensemble import RandomForestRegressor
#         from scipy.stats import pearsonr
        
#         # Fit nonlinear model
#         model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
#         model.fit(X, y)
        
#         # Get residuals
#         pred = model.predict(X)
#         residuals = y - pred
        
#         # Test if residuals are independent of X
#         corr, p_val = pearsonr(X.flatten(), residuals)
        
#         return p_val < alpha
    
#     def _dict_to_graph(self, graph_dict):
#         """Convert dictionary graph to matrix format"""
#         graph = np.zeros((self.n_vars, self.n_vars, self.max_lag + 1))
        
#         for (i, j, lag), val in graph_dict.items():
#             if lag <= self.max_lag:
#                 graph[i, j, lag] = val
        
#         return graph


class TiMINoMethod:
    """
    TiMINo: Time-series Models with Independent Noise
    Original implementation using R
    Based on: https://github.com/ckassaad/causal_discovery_for_time_series
    """
    
    def __init__(self, data, max_lag=2):
        """
        Initialize TiMINo
        
        Args:
            data: Time series data (n_samples x n_vars)
            max_lag: Maximum lag to consider
        """
        self.data = data
        self.n_samples, self.n_vars = data.shape
        self.max_lag = max_lag
    
    def run(self, alpha=0.05):
        """
        Run TiMINo algorithm
        
        Args:
            alpha: Significance level
            
        Returns:
            results: Dictionary with causal graph
        """
        import pandas as pd
        
        # Convert data to pandas DataFrame
        data_df = pd.DataFrame(
            self.data, 
            columns=[f'X{i}' for i in range(self.n_vars)]
        )
        
        try:
            # Call R implementation
            from scripts_R import run_R
            
            g_df, _ = run_R(
                "timino", 
                [[data_df, "data"], [alpha, "sig_level"], [self.max_lag, "nlags"]]
            )
            
            # Convert dataframe to graph format
            graph = self._dataframe_to_graph(g_df)
            
            return {
                'graph': graph,
                'dataframe': g_df,
                'algorithm': 'TiMINo'
            }
            
        except Exception as e:
            print(f"Warning: R-based TiMINo failed ({str(e)}). Using simplified implementation.")
            return self._simplified_timino(alpha)
    
    def _dataframe_to_graph(self, df):
        """
        Convert R output dataframe to graph format
        
        The dataframe format from R:
        - Row names and column names are variable names
        - Values: 0 = no edge, 1 = self-loop, 2 = directed edge
        """
        # Initialize graph with contemporaneous edges only (lag 0)
        # TiMINo returns summary graph without explicit lags
        graph = np.zeros((self.n_vars, self.n_vars, self.max_lag + 1))
        
        # Get variable names from dataframe
        var_names = df.columns.tolist()
        
        # Parse the dataframe
        for i, name_x in enumerate(var_names):
            for j, name_y in enumerate(var_names):
                value = df[name_y].loc[name_x]
                
                if i == j and value > 0:
                    # Self-loop (autocorrelation) - assign to lag 1
                    graph[i, i, 1] = 1
                elif i != j and value == 2:
                    # Directed edge - assign to lag 0 (contemporaneous)
                    graph[i, j, 0] = 1
        
        return graph
    
    def _simplified_timino(self, alpha):
        """
        Simplified TiMINo implementation
        Uses independence tests on residuals
        """
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF
        from scipy.stats import kstest
        
        graph = np.zeros((self.n_vars, self.n_vars, self.max_lag + 1))
        
        # For each variable, find its causes
        for j in range(self.n_vars):
            target = self.data[self.max_lag:, j]
            
            # Test each potential cause
            for i in range(self.n_vars):
                # Test contemporaneous effect
                X_contemp = self.data[self.max_lag:, i].reshape(-1, 1)
                
                if self._test_independence(X_contemp, target, alpha):
                    graph[i, j, 0] = 1
                
                # Test lagged effects
                for lag in range(1, self.max_lag + 1):
                    X_lag = self.data[self.max_lag - lag:-lag, i].reshape(-1, 1)
                    
                    if self._test_independence(X_lag, target, alpha):
                        graph[i, j, lag] = 1
        
        return {
            'graph': graph,
            'algorithm': 'TiMINo (simplified)'
        }
    
    def _test_independence(self, X, y, alpha):
        """
        Test if y is independent of X using nonlinear regression residuals
        
        Returns True if dependent (p_value < alpha)
        """
        from sklearn.ensemble import RandomForestRegressor
        from scipy.stats import pearsonr
        
        # Fit nonlinear model
        model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        model.fit(X, y)
        
        # Get residuals
        pred = model.predict(X)
        residuals = y - pred
        
        # Test if residuals are independent of X
        corr, p_val = pearsonr(X.flatten(), residuals)
        
        return p_val < alpha

def main():
    """Test FCM-based methods"""
    # Generate simple test data
    np.random.seed(42)
    n_samples = 500
    n_vars = 3
    
    data = np.random.randn(n_samples, n_vars)
    # Add some temporal dependencies
    for t in range(1, n_samples):
        data[t, 1] += 0.5 * data[t-1, 0]
        data[t, 2] += 0.3 * data[t-1, 1]
    
    print("Testing VarLiNGAM...")
    varlingam = VarLiNGAMMethod(data, max_lag=2)
    results_var = varlingam.run()
    print(f"VarLiNGAM completed: {results_var['algorithm']}")
    print(f"Graph shape: {results_var['graph'].shape}")
    
    print("\nTesting TiMINo...")
    timino = TiMINoMethod(data, max_lag=2)
    results_timino = timino.run()
    print(f"TiMINo completed: {results_timino['algorithm']}")
    print(f"Graph shape: {results_timino['graph'].shape}")


if __name__ == "__main__":
    main()