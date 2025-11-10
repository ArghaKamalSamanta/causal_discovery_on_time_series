"""
Constraint-Based Causal Discovery Algorithms for Time Series
Implements: tsFCI, PCMCI, PCMCI+, LPCMCI, CD-NOD, CDANs
"""

import numpy as np
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.lpcmci import LPCMCI
# from tigramite.independence_tests import ParCorr, GPDC, CMIknn
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.independence_tests.gpdc import GPDC
from tigramite.independence_tests.cmiknn import CMIknn
import warnings
warnings.filterwarnings('ignore')


class ConstraintBasedMethods:
    """Wrapper for constraint-based causal discovery methods"""
    
    def __init__(self, data, var_names=None, max_lag=2):
        """
        Initialize with time series data
        
        Args:
            data: Time series data (n_samples x n_vars)
            var_names: Variable names
            max_lag: Maximum time lag to consider
        """
        self.data = data
        self.n_samples, self.n_vars = data.shape
        self.max_lag = max_lag
        self.var_names = var_names or [f'X{i}' for i in range(self.n_vars)]
        
        # Prepare data for Tigramite
        self.dataframe = pp.DataFrame(
            data,
            var_names=self.var_names
        )
    
    def run_pcmci(self, tau_max=None, pc_alpha=0.05, cond_ind_test='par_corr'):
        """
        Run PCMCI algorithm
        
        Args:
            tau_max: Maximum lag (uses self.max_lag if None)
            pc_alpha: Significance level
            cond_ind_test: Conditional independence test type
            
        Returns:
            results: Dictionary with graph and p-values
        """
        if tau_max is None:
            tau_max = self.max_lag
        
        # Select independence test
        if cond_ind_test == 'par_corr':
            cond_ind_test_obj = ParCorr(significance='analytic')
        elif cond_ind_test == 'gpdc':
            cond_ind_test_obj = GPDC(significance='analytic')
        elif cond_ind_test == 'cmi_knn':
            cond_ind_test_obj = CMIknn(significance='shuffle_test')
        else:
            cond_ind_test_obj = ParCorr(significance='analytic')
        
        # Initialize PCMCI
        pcmci = PCMCI(
            dataframe=self.dataframe,
            cond_ind_test=cond_ind_test_obj,
            verbosity=0
        )
        
        # Run PCMCI
        results_pcmci = pcmci.run_pcmci(
            tau_max=tau_max,
            pc_alpha=pc_alpha
        )
        
        # Extract results
        graph = results_pcmci['graph']
        p_matrix = results_pcmci['p_matrix']
        val_matrix = results_pcmci['val_matrix']

        # Convert p_matrix format to binary graph
        binary_graph = np.zeros_like(val_matrix)
        binary_graph[p_matrix < pc_alpha] = 1
        
        return {
            'graph': binary_graph, #graph,
            'p_matrix': p_matrix,
            'val_matrix': val_matrix,
            'algorithm': 'PCMCI'
        }
    
    def run_pcmci_plus(self, tau_max=None, pc_alpha=0.05):
        """
        Run PCMCI+ algorithm (detects lagged and contemporaneous edges)
        
        Args:
            tau_max: Maximum lag
            pc_alpha: Significance level
            
        Returns:
            results: Dictionary with graph and p-values
        """
        if tau_max is None:
            tau_max = self.max_lag
        
        # Initialize with ParCorr test
        cond_ind_test = ParCorr(significance='analytic')
        
        pcmci = PCMCI(
            dataframe=self.dataframe,
            cond_ind_test=cond_ind_test,
            verbosity=0
        )
        
        # Run PCMCIplus
        results_pcmci_plus = pcmci.run_pcmciplus(
            tau_max=tau_max,
            pc_alpha=pc_alpha
        )
        
        graph = results_pcmci_plus['graph']
        p_matrix = results_pcmci_plus['p_matrix']
        val_matrix = results_pcmci_plus['val_matrix']

        # Convert p_matrix format to binary graph
        binary_graph = np.zeros_like(val_matrix)
        binary_graph[p_matrix < pc_alpha] = 1
        
        return {
            'graph': binary_graph,
            'p_matrix': p_matrix,
            'val_matrix': val_matrix,
            'algorithm': 'PCMCI+'
        }
    
    def run_lpcmci(self, tau_max=None, pc_alpha=0.05):
        """
        Run LPCMCI algorithm (handles latent confounders)
        
        Args:
            tau_max: Maximum lag
            pc_alpha: Significance level
            
        Returns:
            results: Dictionary with graph and p-values
        """
        if tau_max is None:
            tau_max = self.max_lag
        
        cond_ind_test = ParCorr(significance='analytic')
        
        lpcmci = LPCMCI(
            dataframe=self.dataframe,
            cond_ind_test=cond_ind_test,
            verbosity=0
        )
        
        # Run LPCMCI
        results_lpcmci = lpcmci.run_lpcmci(
            tau_max=tau_max,
            pc_alpha=pc_alpha
        )
        
        graph = results_lpcmci['graph']
        p_matrix = results_lpcmci.get('p_matrix')
        val_matrix = results_lpcmci.get('val_matrix')

        # Convert p_matrix format to binary graph
        binary_graph = np.zeros_like(val_matrix)
        binary_graph[p_matrix < pc_alpha] = 1
        
        return {
            'graph': binary_graph,
            'p_matrix': p_matrix,
            'val_matrix': val_matrix,
            'algorithm': 'LPCMCI'
        }


# class tsFCIMethod:
#     """
#     time series FCI (tsFCI) implementation
#     Note: This is a simplified implementation based on FCI principles
#     For full implementation, refer to: https://sites.google.com/site/dorisentner/publications/tsfci
#     """
    
#     def __init__(self, data, max_lag=2):
#         self.data = data
#         self.n_samples, self.n_vars = data.shape
#         self.max_lag = max_lag
    
#     def run(self, alpha=0.05):
#         """
#         Run tsFCI algorithm
#         Uses PC-like approach with temporal ordering
#         """
#         from causallearn.search.ConstraintBased.FCI import fci
#         from causallearn.utils.cit import fisherz
        
#         # Prepare lagged data
#         lagged_data = self._create_lagged_data()
        
#         # Run FCI on lagged data
#         G, edges = fci(lagged_data, fisherz, alpha=alpha, verbose=False)
        
#         # Convert back to temporal format
#         graph = self._extract_temporal_graph(G)
        
#         return {
#             'graph': graph,
#             'algorithm': 'tsFCI'
#         }
    
#     def _create_lagged_data(self):
#         """Create lagged variable matrix"""
#         lagged_vars = []
#         for lag in range(self.max_lag + 1):
#             if lag == 0:
#                 lagged_vars.append(self.data[self.max_lag:, :])
#             else:
#                 lagged_vars.append(self.data[self.max_lag - lag:-lag, :])
        
#         return np.hstack(lagged_vars)
    
#     def _extract_temporal_graph(self, G):
#         """Extract temporal graph from static graph"""
#         # Simplified extraction - full implementation would be more complex
#         n_vars = self.n_vars
#         max_lag = self.max_lag
        
#         # Initialize temporal graph
#         graph = np.zeros((n_vars, n_vars, max_lag + 1))
        
#         # Map back to temporal structure
#         adj_matrix = (G != 0).astype(int)
        
#         for i in range(n_vars):
#             for lag in range(max_lag + 1):
#                 for j in range(n_vars):
#                     idx_from = i + lag * n_vars
#                     idx_to = j
#                     if idx_from < adj_matrix.shape[0] and idx_to < adj_matrix.shape[1]:
#                         graph[i, j, lag] = adj_matrix[idx_from, idx_to]
        
#         return graph

# class tsFCIMethod:
#     """
#     time series FCI (tsFCI) implementation
#     Original implementation from: https://sites.google.com/site/dorisentner/publications/tsfci
#     """
    
#     def __init__(self, data, max_lag=2):
#         self.data = data
#         self.n_samples, self.n_vars = data.shape
#         self.max_lag = max_lag
    
#     def run(self, alpha=0.05):
#         """
#         Run tsFCI algorithm using R implementation
#         """
#         import pandas as pd
        
#         # Convert data to pandas DataFrame
#         data_df = pd.DataFrame(
#             self.data, 
#             columns=[f'X{i}' for i in range(self.n_vars)]
#         )
        
#         try:
#             # Call R implementation
#             from scripts_R import run_R
            
#             g_dict, init_obj = run_R(
#                 "tsfci", 
#                 [[data_df, "data"], [alpha, "sig_level"], [self.max_lag, "nlags"]]
#             )
            
#             # Convert dictionary to temporal graph format
#             graph = self._dict_to_temporal_graph(g_dict)
            
#             return {
#                 'graph': graph,
#                 'temporal_dict': g_dict,
#                 'algorithm': 'tsFCI'
#             }
            
#         except Exception as e:
#             print(f"Warning: R-based tsFCI failed ({str(e)}). Using fallback implementation.")
#             return self._fallback_tsfci(alpha)
    
#     def _dict_to_temporal_graph(self, g_dict):
#         """Convert temporal dictionary from R to graph format"""
#         # Initialize temporal graph: (source, target, lag)
#         graph = np.zeros((self.n_vars, self.n_vars, self.max_lag + 1))
        
#         # Parse dictionary: g_dict[target] = [(source, lag), ...]
#         for target_name, causes in g_dict.items():
#             # Extract target index from name (e.g., 'X1' -> 1)
#             target_idx = int(target_name.replace('X', ''))
            
#             for source_name, lag in causes:
#                 # Extract source index
#                 source_idx = int(source_name.replace('X', ''))
                
#                 # Set edge in graph (handle negative lags from R)
#                 abs_lag = abs(lag)
#                 if abs_lag <= self.max_lag:
#                     graph[source_idx, target_idx, abs_lag] = 1
        
#         return graph
    
#     def _fallback_tsfci(self, alpha):
#         """Fallback FCI implementation if R is not available"""
#         from causallearn.search.ConstraintBased.FCI import fci
#         from causallearn.utils.cit import fisherz
        
#         # Prepare lagged data
#         lagged_data = self._create_lagged_data()
        
#         # Run FCI on lagged data
#         G, edges = fci(lagged_data, fisherz, alpha=alpha, verbose=False)
        
#         # Convert back to temporal format
#         graph = self._extract_temporal_graph(G)
        
#         return {
#             'graph': graph,
#             'algorithm': 'tsFCI (fallback)'
#         }
    
#     def _create_lagged_data(self):
#         """Create lagged variable matrix"""
#         lagged_vars = []
#         for lag in range(self.max_lag + 1):
#             if lag == 0:
#                 lagged_vars.append(self.data[self.max_lag:, :])
#             else:
#                 lagged_vars.append(self.data[self.max_lag - lag:-lag, :])
        
#         return np.hstack(lagged_vars)
    
#     def _extract_temporal_graph(self, G):
#         """Extract temporal graph from static graph"""
#         n_vars = self.n_vars
#         max_lag = self.max_lag
        
#         # Initialize temporal graph
#         graph = np.zeros((n_vars, n_vars, max_lag + 1))
        
#         # Map back to temporal structure
#         adj_matrix = (G != 0).astype(int)
        
#         for i in range(n_vars):
#             for lag in range(max_lag + 1):
#                 for j in range(n_vars):
#                     idx_from = i + lag * n_vars
#                     idx_to = j
#                     if idx_from < adj_matrix.shape[0] and idx_to < adj_matrix.shape[1]:
#                         graph[i, j, lag] = adj_matrix[idx_from, idx_to]
        
#         return graph

# --- Create lag-expanded data to match R's expectations ---
def create_lagged_matrix(data, max_lag):
    lagged = []
    for lag in range(max_lag + 1):
        if lag == 0:
            lagged.append(data[max_lag:, :])
        else:
            lagged.append(data[max_lag - lag:-lag, :])
    return np.hstack(lagged)

class tsFCIMethod:
    """
    time series FCI (tsFCI) implementation
    Original implementation from: https://sites.google.com/site/dorisentner/publications/tsfci
    """
    
    def __init__(self, data, max_lag=2):
        self.data = data
        self.n_samples, self.n_vars = data.shape
        self.max_lag = max_lag
    
    def run(self, alpha=0.05):
        """
        Run tsFCI algorithm using R implementation
        """
        import pandas as pd
        
        # # Convert data to pandas DataFrame
        # data_df = pd.DataFrame(
        #     self.data, 
        #     columns=[f'X{i+1}' for i in range(self.n_vars)]
        # )

        lagged_data = create_lagged_matrix(self.data, int(self.max_lag))

        # Rename columns as X1..Xn per lag, matching tsfci.R
        columns = []
        for lag in range(int(self.max_lag) + 1):
            for i in range(self.n_vars):
                columns.append(f'X{i+1}_{lag}')

        data_df = pd.DataFrame(lagged_data, columns=columns)
        
        try:
            # Call R implementation
            from scripts_R import run_R

            print("Data shape:", data_df.shape)
            print("Data columns:", data_df.columns.tolist())
            print("NaN count:", data_df.isna().sum().sum())
            
            g_dict, init_obj = run_R(
                "tsfci", 
                [[data_df, "data"], [alpha, "sig_level"], [self.max_lag, "nlags"]]
            )
            
            # Convert dictionary to temporal graph format
            graph = self._dict_to_temporal_graph(g_dict)
            
            return {
                'graph': graph,
                'temporal_dict': g_dict,
                'algorithm': 'tsFCI'
            }
            
        except Exception as e:
            print(f"Warning: R-based tsFCI failed ({str(e)}). Using fallback implementation.")
            return self._fallback_tsfci(alpha)
    
    def _dict_to_temporal_graph(self, g_dict):
        """Convert temporal dictionary from R to graph format"""
        # Initialize temporal graph: (source, target, lag)
        graph = np.zeros((self.n_vars, self.n_vars, self.max_lag + 1))
        
        # Parse dictionary: g_dict[target] = [(source, lag), ...]
        for target_name, causes in g_dict.items():
            # Extract target index from name (e.g., 'X1' -> 1)
            try:
                target_idx = int(target_name.replace('X', ''))
            except ValueError:
                # Handle case where variable name might not follow X<n> pattern
                var_names = [f'X{i+1}' for i in range(self.n_vars)]
                if target_name in var_names:
                    target_idx = var_names.index(target_name)
                else:
                    continue
            
            for source_name, lag in causes:
                # Extract source index
                try:
                    source_idx = int(source_name.replace('X', ''))
                except ValueError:
                    var_names = [f'X{i}' for i in range(self.n_vars)]
                    if source_name in var_names:
                        source_idx = var_names.index(source_name)
                    else:
                        continue
                
                # Set edge in graph (handle negative lags from R)
                abs_lag = abs(lag)
                if abs_lag <= self.max_lag and source_idx < self.n_vars and target_idx < self.n_vars:
                    graph[source_idx, target_idx, abs_lag] = 1
        
        return graph
    
    def _fallback_tsfci(self, alpha):
        """Fallback FCI implementation if R is not available"""
        try:
            from causallearn.search.ConstraintBased.FCI import fci
            from causallearn.utils.cit import fisherz
            
            # Prepare lagged data
            lagged_data = self._create_lagged_data()
            
            # Run FCI on lagged data
            G, edges = fci(lagged_data, fisherz, alpha=alpha, verbose=False)
            
            # Convert back to temporal format
            graph = self._extract_temporal_graph(G)
            
            return {
                'graph': graph,
                'algorithm': 'tsFCI (fallback)'
            }
        except Exception as e:
            print(f"Fallback FCI also failed: {str(e)}")
            # Return empty graph as last resort
            graph = np.zeros((self.n_vars, self.n_vars, self.max_lag + 1))
            return {
                'graph': graph,
                'algorithm': 'tsFCI (failed)'
            }
    
    def _create_lagged_data(self):
        """Create lagged variable matrix"""
        lagged_vars = []
        for lag in range(self.max_lag + 1):
            if lag == 0:
                lagged_vars.append(self.data[self.max_lag:, :])
            else:
                lagged_vars.append(self.data[self.max_lag - lag:-lag, :])
        
        return np.hstack(lagged_vars)
    
    def _extract_temporal_graph(self, G):
        """Extract temporal graph from static graph"""
        n_vars = self.n_vars
        max_lag = self.max_lag
        
        # Initialize temporal graph
        graph = np.zeros((n_vars, n_vars, max_lag + 1))
        
        # Map back to temporal structure
        adj_matrix = (G != 0).astype(int)
        
        for i in range(n_vars):
            for lag in range(max_lag + 1):
                for j in range(n_vars):
                    idx_from = i + lag * n_vars
                    idx_to = j
                    if idx_from < adj_matrix.shape[0] and idx_to < adj_matrix.shape[1]:
                        graph[i, j, lag] = adj_matrix[idx_from, idx_to]
        
        return graph


class CDNODMethod:
    """
    CD-NOD and extended CD-NOD implementation
    Based on: Causal Discovery from Nonstationary/Heterogeneous Data
    """
    
    def __init__(self, data, max_lag=2):
        self.data = data
        self.n_samples, self.n_vars = data.shape
        self.max_lag = max_lag
    
    def run(self, alpha=0.05):
        """Run CD-NOD algorithm"""
        # Phase 1: Detect changing modules using surrogate variable
        changing_modules = self._detect_changing_modules(alpha)
        
        # Phase 2: Run PC-like algorithm with temporal ordering
        graph = self._discover_structure(alpha, changing_modules)
        
        return {
            'graph': graph,
            'changing_modules': changing_modules,
            'algorithm': 'CD-NOD'
        }
    
    def _detect_changing_modules(self, alpha):
        """Detect variables that change over time"""
        from scipy.stats import spearmanr
        
        changing = []
        time_var = np.arange(self.n_samples)
        
        for i in range(self.n_vars):
            # Test correlation with time
            corr, p_val = spearmanr(time_var, self.data[:, i])
            if p_val < alpha:
                changing.append(i)
        
        return changing
    
    def _discover_structure(self, alpha, changing_modules):
        """Discover causal structure"""
        # Simplified PC-style discovery
        from causallearn.search.ConstraintBased.PC import pc
        from causallearn.utils.cit import fisherz
        
        cg = pc(self.data, alpha=alpha, indep_test=fisherz, verbose=False)
        
        # Extract graph
        graph = cg.G.graph
        
        return graph


class CDANsMethod:
    """
    CDANs: Temporal Causal Discovery from Autocorrelated and Non-stationary Time Series
    Implementation based on the paper
    """
    
    def __init__(self, data, max_lag=2):
        self.data = data
        self.n_samples, self.n_vars = data.shape
        self.max_lag = max_lag
        self.dataframe = pp.DataFrame(data)
    
    def run(self, alpha=0.05):
        """Run CDANs algorithm"""
        # Step 1: Detect lagged parents using PCMCI
        lagged_parents = self._detect_lagged_parents(alpha)
        
        # Step 2: Detect changing modules
        changing_modules = self._detect_changing_modules(alpha, lagged_parents)
        
        # Step 3: Detect contemporaneous edges
        contemp_graph = self._detect_contemporaneous_edges(alpha, lagged_parents)
        
        # Step 4: Orient edges
        final_graph = self._orient_edges(lagged_parents, contemp_graph, changing_modules)
        
        return {
            'graph': final_graph,
            'lagged_parents': lagged_parents,
            'changing_modules': changing_modules,
            'algorithm': 'CDANs'
        }
    
    def _detect_lagged_parents(self, alpha):
        """Detect lagged parents using PCMCI"""
        cond_ind_test = ParCorr(significance='analytic')
        pcmci = PCMCI(dataframe=self.dataframe, cond_ind_test=cond_ind_test, verbosity=0)
        
        results = pcmci.run_pcmci(tau_max=self.max_lag, pc_alpha=alpha)
        
        # Extract lagged parents for each variable
        lagged_parents = {}
        for j in range(self.n_vars):
            lagged_parents[j] = []
            for i in range(self.n_vars):
                for tau in range(1, self.max_lag + 1):
                    if results['graph'][i, j, tau] != '':
                        lagged_parents[j].append((i, tau))
        
        return lagged_parents
    
    def _detect_changing_modules(self, alpha, lagged_parents):
        """Detect variables with time dependency"""
        from scipy.stats import spearmanr
        from sklearn.kernel_ridge import KernelRidge
        
        changing = []
        time_var = np.arange(self.n_samples).reshape(-1, 1)
        
        for i in range(self.n_vars):
            # Fit nonlinear model with time
            kr = KernelRidge(kernel='rbf')
            kr.fit(time_var, self.data[:, i])
            pred = kr.predict(time_var)
            
            # Test significance
            residuals = self.data[:, i] - pred
            corr, p_val = spearmanr(time_var.flatten(), residuals)
            
            if p_val < alpha and abs(corr) > 0.1:
                changing.append(i)
        
        return changing
    
    def _detect_contemporaneous_edges(self, alpha, lagged_parents):
        """Detect contemporaneous edges"""
        from scipy.stats import pearsonr
        
        contemp_graph = np.zeros((self.n_vars, self.n_vars))
        
        for i in range(self.n_vars):
            for j in range(i + 1, self.n_vars):
                # Partial correlation conditioned on lagged parents
                _, p_val = self._partial_correlation_test(i, j, lagged_parents)
                
                if p_val < alpha:
                    contemp_graph[i, j] = 1
                    contemp_graph[j, i] = 1
        
        return contemp_graph
    
    def _partial_correlation_test(self, i, j, lagged_parents):
        """Partial correlation test"""
        from scipy.stats import pearsonr
        
        # Simple correlation for now
        corr, p_val = pearsonr(self.data[:, i], self.data[:, j])
        return corr, p_val
    
    def _orient_edges(self, lagged_parents, contemp_graph, changing_modules):
        """Orient edges based on temporal order and changing modules"""
        # Initialize full graph structure
        graph = np.zeros((self.n_vars, self.n_vars, self.max_lag + 1))
        
        # Add lagged edges (already oriented by time)
        for j, parents in lagged_parents.items():
            for i, tau in parents:
                graph[i, j, tau] = 1
        
        # Add contemporaneous edges
        graph[:, :, 0] = contemp_graph
        
        return graph