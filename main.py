"""
Main Runner for Time Series Causal Discovery Experiments
Runs all algorithms on Syn-6 and fMRI datasets
"""

import numpy as np
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import data generation
from data_generation import DataGenerator

# Import algorithm implementations
from constraint_based import (
    ConstraintBasedMethods, tsFCIMethod, CDNODMethod, CDANsMethod
)
from fcm_based import VarLiNGAMMethod, TiMINoMethod
from gradient_based import DYNOTEARSMethod, NTSNOTEARSMethod
from granger_misc import (
    GVARMethod, NAVARMethod, ACDMethod, oCSEMethod,
    TCDFMethod, NBCBMethod, PCTMIMethod
)

# Import evaluation
from evaluation import Evaluator, ResultLogger


class ExperimentRunner:
    """Run causal discovery experiments"""
    
    def __init__(self, data, true_graph, dataset_name, max_lag=2):
        """
        Initialize experiment runner
        
        Args:
            data: Time series data
            true_graph: Ground truth causal graph
            dataset_name: Name of the dataset
            max_lag: Maximum lag to consider
        """
        self.data = data
        self.true_graph = true_graph
        self.dataset_name = dataset_name
        self.max_lag = max_lag
        
        # Initialize evaluator and logger
        self.evaluator = Evaluator(true_graph)
        self.logger = ResultLogger(
            log_file=f'results_{dataset_name}.log',
            json_file=f'results_{dataset_name}.json'
        )
        
        self.results = {}
    
    def run_all_algorithms(self):
        """Run all implemented algorithms"""
        self.logger.log(f"\n{'='*80}")
        self.logger.log(f"Starting experiments on {self.dataset_name} dataset")
        self.logger.log(f"Data shape: {self.data.shape}, Max lag: {self.max_lag}")
        self.logger.log(f"{'='*80}\n")
        
        # Define all algorithms to run
        algorithms = [
            # Constraint-based
            ('PCMCI', self._run_pcmci),
            ('PCMCI+', self._run_pcmci_plus),
            ('LPCMCI', self._run_lpcmci),
            # ('tsFCI', self._run_tsfci),
            ('CD-NOD', self._run_cdnod),
            ('CDANs', self._run_cdans),
            
            # FCM-based
            ('VarLiNGAM', self._run_varlingam),
            ('TiMINo', self._run_timino),
            
            # Gradient-based
            ('DYNOTEARS', self._run_dynotears),
            ('NTS-NOTEARS', self._run_nts_notears),
            
            # Granger causality-based
            ('GVAR', self._run_gvar),
            ('NAVAR', self._run_navar),
            ('ACD', self._run_acd),
            
            # Miscellaneous
            ('oCSE', self._run_ocse),
            ('TCDF', self._run_tcdf),
            ('NBCB', self._run_nbcb),
            ('PCTMI', self._run_pctmi),
        ]
        
        # Run each algorithm with progress bar
        for alg_name, alg_func in tqdm(algorithms, desc=f"Running algorithms on {self.dataset_name}"):
            # try:
            #     self.logger.log(f"\nRunning {alg_name}...", print_console=True)
            #     start_time = time.time()
                
            #     result = alg_func()
                
            #     elapsed_time = time.time() - start_time
            #     self.logger.log(f"{alg_name} completed in {elapsed_time:.2f} seconds")
                
            #     # Evaluate
            #     metrics = self.evaluator.evaluate(result, alg_name)
            #     metrics['runtime_seconds'] = round(elapsed_time, 2)
                
            #     self.logger.log_metrics(metrics)
            #     self.results[alg_name] = result
                
            # except Exception as e:
            #     self.logger.log(f"Error running {alg_name}: {str(e)}")
            #     print(f"  ⚠ {alg_name} failed: {str(e)}")

            self.logger.log(f"\nRunning {alg_name}...", print_console=True)
            start_time = time.time()
            
            result = alg_func()
            
            elapsed_time = time.time() - start_time
            self.logger.log(f"{alg_name} completed in {elapsed_time:.2f} seconds")
            
            # Evaluate
            metrics = self.evaluator.evaluate(result, alg_name)
            metrics['runtime_seconds'] = round(elapsed_time, 2)
            
            self.logger.log_metrics(metrics)
            self.results[alg_name] = result
        
        # Compare all results
        self._compare_results()
        
        # Save results
        self.logger.save_json()
        self.logger.print_summary_table()
    
    # Constraint-based methods
    def _run_pcmci(self):
        method = ConstraintBasedMethods(self.data, max_lag=self.max_lag)
        return method.run_pcmci()
    
    def _run_pcmci_plus(self):
        method = ConstraintBasedMethods(self.data, max_lag=self.max_lag)
        return method.run_pcmci_plus()
    
    def _run_lpcmci(self):
        method = ConstraintBasedMethods(self.data, max_lag=self.max_lag)
        return method.run_lpcmci()
    
    def _run_tsfci(self):
        method = tsFCIMethod(self.data, max_lag=self.max_lag)
        return method.run()
    
    def _run_cdnod(self):
        method = CDNODMethod(self.data, max_lag=self.max_lag)
        return method.run()
    
    def _run_cdans(self):
        method = CDANsMethod(self.data, max_lag=self.max_lag)
        return method.run()
    
    # FCM-based methods
    def _run_varlingam(self):
        method = VarLiNGAMMethod(self.data, max_lag=self.max_lag)
        return method.run()
    
    def _run_timino(self):
        method = TiMINoMethod(self.data, max_lag=self.max_lag)
        return method.run()
    
    # Gradient-based methods
    def _run_dynotears(self):
        method = DYNOTEARSMethod(self.data, max_lag=self.max_lag)
        return method.run(max_iter=50)
    
    def _run_nts_notears(self):
        method = NTSNOTEARSMethod(self.data, max_lag=self.max_lag)
        return method.run(max_iter=50)
    
    # Granger causality methods
    def _run_gvar(self):
        method = GVARMethod(self.data, max_lag=self.max_lag)
        return method.run(epochs=50)
    
    def _run_navar(self):
        method = NAVARMethod(self.data, max_lag=self.max_lag)
        return method.run(epochs=50)
    
    def _run_acd(self):
        method = ACDMethod(self.data, max_lag=self.max_lag)
        return method.run(epochs=50)
    
    # Miscellaneous methods
    def _run_ocse(self):
        method = oCSEMethod(self.data, max_lag=self.max_lag)
        return method.run()
    
    def _run_tcdf(self):
        method = TCDFMethod(self.data, max_lag=self.max_lag)
        return method.run(epochs=30)
    
    def _run_nbcb(self):
        method = NBCBMethod(self.data, max_lag=self.max_lag)
        return method.run()
    
    def _run_pctmi(self):
        method = PCTMIMethod(self.data, max_lag=self.max_lag)
        return method.run()
    
    def _compare_results(self):
        """Compare all algorithm results"""
        comparison = self.evaluator.compare_multiple_algorithms(self.results)
        self.logger.log_comparison(comparison)


def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("Time Series Causal Discovery - Comprehensive Evaluation")
    print("="*80 + "\n")
    
    # Initialize data generator
    generator = DataGenerator(random_seed=42)
    
    # ========== Experiment 1: Syn-6 Dataset ==========
    print("\n[1/2] Generating Syn-6 dataset...")
    syn6_data, syn6_graph = generator.generate_syn6_data(n_samples=1000, lag=2)
    print(f"Syn-6 data generated: shape {syn6_data.shape}")
    print(f"True edges: {len(syn6_graph['edges'])}")
    print(f"Changing modules: {syn6_graph['changing_modules']}")
    
    print("\n" + "-"*80)
    print("Running experiments on Syn-6 dataset")
    print("-"*80)
    
    syn6_runner = ExperimentRunner(
        data=syn6_data,
        true_graph=syn6_graph,
        dataset_name='syn6',
        max_lag=2
    )
    syn6_runner.run_all_algorithms()
    
    # ========== Experiment 2: fMRI Dataset ==========
    print("\n\n[2/2] Loading fMRI dataset...")
    try:
        fmri_data, fmri_net, fmri_meta = generator.load_fmri_data(
            './fMRI',
            sim_index=1
        )
        print(f"fMRI ground truth type: {type(fmri_net)}")
        print(f"fMRI ground truth shape: {fmri_net.shape if hasattr(fmri_net, 'shape') else 'no shape'}")
        print(f"fMRI data loaded: shape {fmri_data.shape}")
        print(f"Simulation {fmri_meta['sim_index']}: "
              f"{fmri_meta['n_subjects']} subjects, "
              f"{fmri_meta['n_timepoints']} timepoints, "
              f"{fmri_meta['n_nodes']} nodes")
        
        # For fMRI, we use the network as ground truth
        # Convert to temporal format (simplified)
        fmri_graph_dict = {
            'n_vars': fmri_meta['n_nodes'],
            'max_lag': 1  # Use lag 1 for fMRI
        }
        
        print("\n" + "-"*80)
        print("Running experiments on fMRI dataset")
        print("-"*80)
        
        # fmri_runner = ExperimentRunner(
        #     data=fmri_data,
        #     true_graph=fmri_net,  # Use network directly
        #     dataset_name='fmri',
        #     max_lag=1
        # )

        # Convert fMRI network to temporal format:
        if isinstance(fmri_net, np.ndarray) and len(fmri_net.shape) == 2:
            # Convert 2D network to temporal format with lag=1
            fmri_graph = {
                'edges': set((i, j, 1) for i in range(fmri_net.shape[0]) 
                             for j in range(fmri_net.shape[1]) 
                             if abs(fmri_net[i, j]) > 0),
                'n_vars': fmri_net.shape[0],
                'max_lag': 1
            }
            fmri_runner = ExperimentRunner(
                data=fmri_data,
                true_graph=fmri_graph,  # Use converted format
                dataset_name='fmri',
                max_lag=1
            )
            fmri_runner.run_all_algorithms()
        
    except FileNotFoundError as e:
        print(f"\nWarning: fMRI data not found. Skipping fMRI experiments.")
        print(f"Error: {e}")
        print("Please ensure fMRI data is in './fmri_data/' directory")
    
    print("\n" + "="*80)
    print("All experiments completed!")
    print("Results saved to:")
    print("  - results_syn6.log, results_syn6.json")
    print("  - results_fmri.log, results_fmri.json (if fMRI data available)")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()