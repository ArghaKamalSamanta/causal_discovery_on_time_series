# Time Series Causal Discovery - Comprehensive Evaluation

This project implements and evaluates 17 time series causal discovery algorithms on synthetic and real-world datasets.

## Implemented Algorithms

### Constraint-Based Methods
- **PCMCI**: PC algorithm with momentary conditional independence
- **PCMCI+**: Extension with contemporaneous edges
- **LPCMCI**: Latent PCMCI for handling confounders
- **tsFCI**: Time series Fast Causal Inference
- **CD-NOD**: Causal Discovery from Nonstationary/Heterogeneous Data
- **CDANs**: Causal Discovery from Autocorrelated and Non-stationary data

### FCM-Based Methods
- **VarLiNGAM**: Vector Autoregressive Linear Non-Gaussian Acyclic Model
- **TiMINo**: Time-series Models with Independent Noise

### Gradient-Based Methods
- **DYNOTEARS**: Dynamic NOTEARS for time series
- **NTS-NOTEARS**: Nonparametric Temporal Structure learning

### Granger Causality Methods
- **GVAR**: Generalized Vector AutoRegression
- **NAVAR**: Neural Additive Vector AutoRegression
- **ACD**: Amortized Causal Discovery

### Miscellaneous Methods
- **oCSE**: optimal Causation Entropy
- **TCDF**: Temporal Causal Discovery Framework
- **NBCB**: Noise-based/Constraint-based approach
- **PCTMI**: PC with Temporal Mutual Information

## Installation

```bash
# Create virtual environment (recommended)
python -m venv causal_env
source causal_env/bin/activate  # On Windows: causal_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Additional Setup

Some packages may require additional installation:

```bash
# For lingam (VarLiNGAM)
pip install lingam

# For causal-learn (tsFCI, CD-NOD)
pip install causal-learn

# For tigramite (PCMCI, PCMCI+, LPCMCI)
pip install tigramite
```

## Project Structure

```
.
├── data_generation.py      # Data generation and loading
├── constraint_based.py     # Constraint-based algorithms
├── fcm_based.py           # FCM-based algorithms
├── gradient_based.py      # Gradient-based algorithms
├── granger_misc.py        # Granger & miscellaneous methods
├── evaluation.py          # Evaluation metrics and logging
├── main.py               # Main experiment runner
├── requirements.txt      # Python dependencies
└── fmri_data/           # fMRI simulation data (download separately)
```

## Usage

### Quick Start

Run all experiments:

```bash
python main.py
```

This will:
1. Generate Syn-6 synthetic data
2. Load fMRI data (if available)
3. Run all 17 algorithms on both datasets
4. Evaluate and compare results
5. Save results to log and JSON files

### Individual Algorithm Testing

Test specific algorithm categories:

```bash
# Test constraint-based methods
python constraint_based.py

# Test FCM-based methods
python fcm_based.py

# Test gradient-based methods
python gradient_based.py

# Test Granger & miscellaneous methods
python granger_misc.py
```

### Data Generation Only

```bash
python data_generation.py
```

## Datasets

### Syn-6 Dataset
- **Variables**: 6
- **Lag period**: 2
- **Features**: Autocorrelation, non-stationarity, changing modules
- **Automatically generated**

### fMRI Dataset
- **Source**: BOLD time series simulations
- **Variables**: 50 network nodes
- **Time points**: Variable by simulation
- **Download**: Place .mat files in `./fmri_data/` directory

## Output

### Log Files
- `results_syn6.log`: Detailed results for Syn-6 dataset
- `results_fmri.log`: Detailed results for fMRI dataset

### JSON Files
- `results_syn6.json`: Structured results for Syn-6
- `results_fmri.json`: Structured results for fMRI

### Metrics Reported
- **TPR** (True Positive Rate / Recall)
- **FDR** (False Discovery Rate)
- **SHD** (Structural Hamming Distance)
- **Precision**
- **F1 Score**
- **Runtime** (seconds)

## Evaluation Metrics

### True Positive Rate (TPR)
```
TPR = TP / (TP + FN)
```
Higher is better. Measures the proportion of true causal edges correctly identified.

### False Discovery Rate (FDR)
```
FDR = FP / (TP + FP)
```
Lower is better. Measures the proportion of false edges among discovered edges.

### Structural Hamming Distance (SHD)
```
SHD = FP + FN
```
Lower is better. Total number of edge additions and deletions needed to match true graph.

## Customization

### Modify Data Generation
Edit `data_generation.py` to change:
- Number of samples
- Lag period
- Noise distribution
- Structural equations

### Adjust Algorithm Parameters
Each algorithm has configurable parameters:

```python
# Example: Modify PCMCI parameters
method = ConstraintBasedMethods(data, max_lag=3)
result = method.run_pcmci(pc_alpha=0.01, cond_ind_test='gpdc')
```

### Add Custom Algorithms
Create a new method class and add to `main.py`:

```python
class CustomMethod:
    def __init__(self, data, max_lag=2):
        self.data = data
        self.max_lag = max_lag
    
    def run(self):
        # Your implementation
        graph = ...
        return {'graph': graph, 'algorithm': 'Custom'}
```

## Troubleshooting

### Missing Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

### GPU Support (Optional)
For faster gradient-based methods:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Memory Issues
Reduce data size or run algorithms individually:
```python
# In main.py, comment out algorithms
algorithms = [
    ('PCMCI', self._run_pcmci),
    # ('PCMCI+', self._run_pcmci_plus),  # Commented out
    # ...
]
```

## References

See the survey papers for detailed algorithm descriptions:
- Hasan et al. (2023): "A Survey on Causal Discovery Methods for I.I.D. and Time Series Data"
- Ferdous et al. (2023): "CDANs: Temporal Causal Discovery from Autocorrelated and Non-Stationary Time Series Data"

## Citation

If you use this code, please cite the original papers for each algorithm you use.

## License

This project is for research and educational purposes.