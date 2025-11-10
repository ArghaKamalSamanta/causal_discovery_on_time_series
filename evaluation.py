"""
Evaluation Utilities for Causal Discovery Algorithms
Computes TPR, FDR, SHD metrics
"""

import numpy as np
import json
from datetime import datetime


class Evaluator:
    """Evaluate causal discovery results"""
    
    def __init__(self, true_graph):
        """
        Initialize evaluator
        
        Args:
            true_graph: Ground truth causal graph
                       Can be adjacency matrix or dict with 'edges' key
        """
        if isinstance(true_graph, dict):
            self.true_edges = set(true_graph.get('edges', []))
            self.n_vars = true_graph.get('n_vars', 0)
            self.max_lag = true_graph.get('max_lag', 0)
            self.changing_modules = true_graph.get('changing_modules', [])
        else:
            self.true_graph_matrix = true_graph
            self.true_edges = self._matrix_to_edges(true_graph)
            self.n_vars = true_graph.shape[0]
            self.max_lag = true_graph.shape[2] - 1 if len(true_graph.shape) == 3 else 0
            self.changing_modules = []
    
    def evaluate(self, predicted_graph, algorithm_name="Unknown"):
        """
        Evaluate predicted graph against ground truth
        
        Args:
            predicted_graph: Predicted causal graph (matrix or dict)
            algorithm_name: Name of the algorithm
            
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        # Convert predicted graph to edge set
        if isinstance(predicted_graph, dict):
            pred_edges = self._extract_edges_from_dict(predicted_graph)
        else:
            pred_edges = self._matrix_to_edges(predicted_graph)
        
        # Compute metrics
        tp, fp, fn, tn = self._compute_confusion_matrix(self.true_edges, pred_edges)
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fdr = fp / (tp + fp) if (tp + fp) > 0 else 0
        shd = fp + fn  # Structural Hamming Distance
        
        # Precision and F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tpr
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics = {
            'algorithm': algorithm_name,
            'TPR': round(tpr, 3),
            'FDR': round(fdr, 3),
            'SHD': int(shd),
            'Precision': round(precision, 3),
            'Recall': round(recall, 3),
            'F1': round(f1, 3),
            'TP': int(tp),
            'FP': int(fp),
            'FN': int(fn),
            'TN': int(tn),
            'num_predicted_edges': len(pred_edges),
            'num_true_edges': len(self.true_edges)
        }
        
        return metrics
    
    # def _matrix_to_edges(self, graph_matrix):
    #     """Convert adjacency matrix to set of edges"""
    #     edges = set()
        
    #     if len(graph_matrix.shape) == 2:
    #         # Static graph
    #         for i in range(graph_matrix.shape[0]):
    #             for j in range(graph_matrix.shape[1]):
    #                 if graph_matrix[i, j] != 0:
    #                     edges.add((i, j, 0))
    #     elif len(graph_matrix.shape) == 3:
    #         # Temporal graph (source, target, lag)
    #         for lag in range(graph_matrix.shape[2]):
    #             for i in range(graph_matrix.shape[0]):
    #                 for j in range(graph_matrix.shape[1]):
    #                     if graph_matrix[i, j, lag] != 0:
    #                         edges.add((i, j, lag))
        
    #     return edges

    def _matrix_to_edges(self, graph_matrix):
        """Convert adjacency matrix to set of edges"""
        edges = set()
        
        if len(graph_matrix.shape) == 2:
            # Static graph
            for i in range(graph_matrix.shape[0]):
                for j in range(graph_matrix.shape[1]):
                    if abs(graph_matrix[i, j]) > 1e-6:  # Add threshold for floating point
                        edges.add((i, j, 0))
        elif len(graph_matrix.shape) == 3:
            # Temporal graph (source, target, lag)
            # CRITICAL: Only count directed edges (source -> target at specific lag)
            for lag in range(graph_matrix.shape[2]):
                for i in range(graph_matrix.shape[0]):
                    for j in range(graph_matrix.shape[1]):
                        # Paper likely excludes self-loops at lag 0
                        if abs(graph_matrix[i, j, lag]) > 1e-6:
                            if not (lag == 0 and i == j):  # Exclude contemporaneous self-loops
                                edges.add((i, j, lag))
        
        return edges
    
    def _extract_edges_from_dict(self, graph_dict):
        """Extract edges from dictionary format"""
        if 'graph' in graph_dict:
            return self._matrix_to_edges(graph_dict['graph'])
        elif 'edges' in graph_dict:
            return set(graph_dict['edges'])
        else:
            return set()
    
    def _compute_confusion_matrix(self, true_edges, pred_edges):
        """Compute TP, FP, FN, TN"""
        tp = len(true_edges & pred_edges)  # Intersection
        fp = len(pred_edges - true_edges)  # In pred but not in true
        fn = len(true_edges - pred_edges)  # In true but not in pred
        
        # Estimate TN (all possible edges minus actual positives)
        if self.n_vars > 0:
            max_possible_edges = self.n_vars * self.n_vars * (self.max_lag + 1)
            tn = max_possible_edges - tp - fp - fn
        else:
            tn = 0
        
        return tp, fp, fn, tn
    
    def compare_multiple_algorithms(self, results_dict):
        """
        Compare results from multiple algorithms
        
        Args:
            results_dict: Dictionary mapping algorithm names to their results
            
        Returns:
            comparison: Dictionary with comparative metrics
        """
        all_metrics = []
        
        for alg_name, result in results_dict.items():
            metrics = self.evaluate(result, alg_name)
            all_metrics.append(metrics)
        
        # Sort by F1 score
        all_metrics.sort(key=lambda x: x['F1'], reverse=True)
        
        comparison = {
            'metrics': all_metrics,
            'best_by_f1': all_metrics[0]['algorithm'] if all_metrics else None,
            'best_by_tpr': max(all_metrics, key=lambda x: x['TPR'])['algorithm'] if all_metrics else None,
            'best_by_fdr': min(all_metrics, key=lambda x: x['FDR'])['algorithm'] if all_metrics else None,
            'best_by_shd': min(all_metrics, key=lambda x: x['SHD'])['algorithm'] if all_metrics else None
        }
        
        return comparison


class ResultLogger:
    """Log results to file"""
    
    def __init__(self, log_file='results.log', json_file='results.json'):
        self.log_file = log_file
        self.json_file = json_file
        self.results = []
    
    def log(self, message, print_console=False):
        """Write message to log file"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        
        with open(self.log_file, 'a') as f:
            f.write(log_message + '\n')
        
        if print_console:
            print(message)
    
    def log_metrics(self, metrics):
        """Log evaluation metrics"""
        self.results.append(metrics)
        
        # Format metrics nicely
        message = f"\n{'='*60}\n"
        message += f"Algorithm: {metrics['algorithm']}\n"
        message += f"TPR: {metrics['TPR']:.3f} | FDR: {metrics['FDR']:.3f} | SHD: {metrics['SHD']}\n"
        message += f"Precision: {metrics['Precision']:.3f} | Recall: {metrics['Recall']:.3f} | F1: {metrics['F1']:.3f}\n"
        message += f"True Positives: {metrics['TP']} | False Positives: {metrics['FP']}\n"
        message += f"False Negatives: {metrics['FN']} | True Negatives: {metrics['TN']}\n"
        message += f"Predicted Edges: {metrics['num_predicted_edges']} | True Edges: {metrics['num_true_edges']}\n"
        message += f"{'='*60}\n"
        
        self.log(message, print_console=False)
    
    def log_comparison(self, comparison):
        """Log comparison results"""
        message = f"\n{'='*60}\n"
        message += "ALGORITHM COMPARISON\n"
        message += f"{'='*60}\n"
        message += f"Best by F1 Score: {comparison['best_by_f1']}\n"
        message += f"Best by TPR: {comparison['best_by_tpr']}\n"
        message += f"Best by FDR (lowest): {comparison['best_by_fdr']}\n"
        message += f"Best by SHD (lowest): {comparison['best_by_shd']}\n"
        message += f"\nRanked by F1 Score:\n"
        for i, metrics in enumerate(comparison['metrics'], 1):
            message += f"{i}. {metrics['algorithm']}: F1={metrics['F1']:.3f}, TPR={metrics['TPR']:.3f}, FDR={metrics['FDR']:.3f}, SHD={metrics['SHD']}\n"
        message += f"{'='*60}\n"
        
        self.log(message, print_console=True)
    
    def save_json(self):
        """Save all results to JSON file"""
        output = {
            'timestamp': datetime.now().isoformat(),
            'results': self.results
        }
        
        with open(self.json_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        self.log(f"Results saved to {self.json_file}")
    
    def print_summary_table(self):
        """Print a summary table of all results"""
        if not self.results:
            print("No results to display")
            return
        
        # Print header
        print(f"\n{'='*100}")
        print(f"{'Algorithm':<25} {'TPR':>8} {'FDR':>8} {'SHD':>8} {'Precision':>10} {'F1':>8} {'Edges':>8}")
        print(f"{'='*100}")
        
        # Sort by F1
        sorted_results = sorted(self.results, key=lambda x: x['F1'], reverse=True)
        
        for result in sorted_results:
            print(f"{result['algorithm']:<25} {result['TPR']:>8.3f} {result['FDR']:>8.3f} "
                  f"{result['SHD']:>8} {result['Precision']:>10.3f} {result['F1']:>8.3f} "
                  f"{result['num_predicted_edges']:>8}")
        
        print(f"{'='*100}\n")


def main():
    """Test evaluation utilities"""
    # Create synthetic true graph
    true_edges = [(0, 0, 1), (0, 1, 1), (1, 2, 2), (2, 2, 2), (2, 3, 0), (3, 4, 2), (4, 5, 0)]
    true_graph = {
        'edges': true_edges,
        'changing_modules': [1, 4],
        'n_vars': 6,
        'max_lag': 2
    }
    
    # Create evaluator
    evaluator = Evaluator(true_graph)
    
    # Simulate predicted edges (with some errors)
    pred_edges = [(0, 0, 1), (0, 1, 1), (1, 2, 2), (2, 3, 0), (4, 5, 0), (1, 3, 1)]  # 5 correct, 1 wrong
    pred_graph = {'edges': pred_edges}
    
    # Evaluate
    metrics = evaluator.evaluate(pred_graph, "Test Algorithm")
    
    # Print results
    print("Evaluation Results:")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    # Test logger
    logger = ResultLogger('test_results.log', 'test_results.json')
    logger.log_metrics(metrics)
    logger.save_json()
    logger.print_summary_table()


if __name__ == "__main__":
    main()