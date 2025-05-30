# Keystroke Dynamics Authentication Evaluation

## Overview

This repository provides a comprehensive evaluation framework for keystroke dynamics authentication methods. It implements multiple algorithms and statistical analysis tools to enable fair, standardized comparison of different approaches in keystroke-based biometric authentication.

## Purpose

- **Standardized Evaluation**: Consistent evaluation protocol following Killourhy & Maxion (2009)
- **Statistical Rigor**: Comprehensive statistical testing including Wilcoxon signed-rank test
- **Fair Comparison**: Unbiased implementation of multiple authentication methods
- **Research Reproducibility**: Complete transparency in evaluation methodology

## Implemented Methods

### Traditional Distance-Based Approaches
- **Manhattan Distance**: `Σ|xi - μi|`
- **Euclidean Distance**: `√Σ(xi - μi)²`
- **Scaled Manhattan Distance**: `Σ|xi - μi|/σi`

### Robust Statistical Approach
- **Adaptive Statistical Profile (ASP)**: Novel approach using robust statistics
  - Median-based central tendency (outlier-resistant)
  - IQR and MAD for robust scale estimation
  - Adaptive feature weighting based on stability and reliability
  - Multi-metric distance fusion: `0.5×Median + 0.3×MAD + 0.2×Std`

## Dataset

**CMU Keystroke Dynamics Dataset** (Killourhy & Maxion, 2009):
- **Users**: 51 participants
- **Samples**: 20,400 keystroke recordings
- **Features**: 31 timing features (dwell time, flight time)
- **Password**: ".tie5Roanl" (fixed password)
- **Sessions**: Multiple sessions per user

## Evaluation Protocol

### Data Splitting
- **Training**: 50% of each user's samples (temporal order maintained)
- **Testing**: Remaining 50% for genuine user evaluation
- **Impostor Testing**: Samples from all other users

### Performance Metrics
- **Equal Error Rate (EER)**: Primary metric
- **False Rejection Rate (FRR)**: Legitimate user rejection rate
- **False Acceptance Rate (FAR)**: Impostor acceptance rate

### Statistical Analysis
- **Wilcoxon Signed-Rank Test**: Non-parametric significance testing
- **Effect Size**: Cohen's d for practical significance
- **Confidence Intervals**: 95% CI using bootstrap method
- **Win Rate Analysis**: Head-to-head comparison statistics

## Requirements

- Python 3.7+
- pandas
- numpy
- scipy
- CMU Keystroke Dynamics Dataset

## Installation

1. Clone this repository:
```bash
git clone https://github.com/[your-username]/keystroke-dynamics-evaluation.git
cd keystroke-dynamics-evaluation
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download the CMU Keystroke Dynamics Dataset:
   - Visit: https://www.cs.cmu.edu/~keystroke/
   - Download `DSL-StrongPasswordData.csv`
   - Place it in the project directory

## Usage

### Basic Evaluation
```python
from keystroke_dynamics_evaluator import KeystrokeDynamicsEvaluator

# Initialize evaluator
evaluator = KeystrokeDynamicsEvaluator('DSL-StrongPasswordData.csv')

# Run comprehensive evaluation
results = evaluator.run_full_evaluation()

# Display results
print(results['final_report'])
```

### Command Line Usage
```bash
python keystroke_dynamics_evaluator.py
```

## Results Summary

Based on evaluation with 51 users from the CMU dataset:

| Method | Mean EER | Std Dev | Median EER |
|--------|----------|---------|------------|
| **Adaptive Statistical Profile** | **11.99%** | ±7.76% | 11.50% |
| Scaled Manhattan | 12.91% | ±9.34% | 10.50% |
| Manhattan Distance | 19.58% | ±10.14% | 18.25% |
| Euclidean Distance | 21.43% | ±10.01% | 21.00% |

### Statistical Significance
- **ASP vs Manhattan**: p < 0.001, Cohen's d = 1.528 (large effect)
- **ASP vs Euclidean**: p < 0.001, Cohen's d = 1.552 (large effect)
- **ASP vs Scaled Manhattan**: p < 0.05, Cohen's d = 0.344 (small effect)

## File Structure

```
keystroke-dynamics-evaluation/
├── README.md                           # This documentation
├── keystroke_dynamics_evaluator.py     # Main evaluation framework
├── requirements.txt                    # Python dependencies
├── sample_output.txt                   # Detailed results output
└── LICENSE                            # License information
```

## Methodology Transparency

### Data Integrity
- **No Data Leakage**: Strict train-test separation
- **Temporal Consistency**: Chronological data splitting
- **Reproducible Sampling**: Fixed random seeds for consistency

### Statistical Validity
- **Non-parametric Testing**: Appropriate for non-normal distributions
- **Multiple Comparison Awareness**: Effect sizes reported alongside p-values
- **Practical Significance**: Both statistical and practical significance evaluated

### Implementation Correctness
- **Standard Compliance**: Follows Killourhy & Maxion (2009) protocol
- **Code Transparency**: All calculations explicitly implemented
- **Reproducible Results**: Deterministic execution with fixed seeds

## Author

**Yuji Umemoto**  
Lecturer
Faculty of International Cultural Studies  
Kwassui Women's University  
Nagasaki, Japan

## Acknowledgments

- CMU Keystroke Dynamics Dataset: Kevin S. Killourhy and Roy A. Maxion
- Statistical methodology: Killourhy & Maxion (2009) evaluation protocol
- Research support: Kwassui Women's University and Nagasaki Institute of Applied Science

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.