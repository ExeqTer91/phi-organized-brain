# The φ-Organized Brain

**Scale-Free Dynamics, Bridge-State Control, and a Seven-State Neural Architecture**

[![DOI](https://img.shields.io/badge/DOI-Preprint-blue)](https://doi.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

## Overview

This repository contains the analysis code and validation pipeline for investigating **golden ratio (φ ≈ 1.618) organization** in human EEG dynamics. The research demonstrates:

- **γ/β frequency ratios converge toward φ** rather than harmonic 2:1 ratios
- **7-state neural architecture** (matching Lucas number L₄ = φ⁴ + φ⁻⁴ = 7)
- **Bridge state gating mechanism** mediated by alpha-band resonance (Q-factor)
- **Fractal/scale-free temporal structure** with power-law timing

## Key Findings

| Finding | Evidence | Statistical Support |
|---------|----------|---------------------|
| φ-organization in γ/β | 92.9% of subjects closer to φ than 2:1 | p < 10⁻⁶ (binomial) |
| 7-state architecture | Silhouette-optimized clustering | k=6-7 optimal |
| Bridge state | Mediates regime transitions | Bridge score = 0.82 |
| Alpha-Q gating | Reversible entry/exit mechanism | AUC = 0.76 |
| Power-law timing | Fractal temporal clustering | α = 2.74, bootstrap p = 0.24 |
| 13/8 Hz ≈ φ | Alpha-theta band boundary | Distance to φ = 0.007 |

## Mathematical Foundation

The Lucas number identity forms the theoretical scaffold:

```
φ⁴ + φ⁻⁴ = 6.854... + 0.146... = 7.000 (exact) = L₄
```

This is not an approximation but a rigorous mathematical identity derivable from the Binet-Lucas formula.

## Repository Structure

```
├── Core Analysis
│   ├── fractal_analysis.py       # DFA, multifractal width, aperiodic exponent
│   ├── final_mechanism_tests.py  # Alpha-Q gating mechanism validation
│   ├── revibe_revert_analysis.py # Q as control parameter (REVIBE tests)
│   ├── bulletproof_tests.py      # Clauset-style power-law validation
│   └── reviewer_tests.py         # Additional robustness checks
│
├── Validation
│   ├── phi_fractal_test.py       # Is φ a fractal organizing principle?
│   ├── comprehensive_validation_tests.py
│   └── split_half_validation.py  # Anti-circularity validation
│
├── Data Files
│   ├── epoch_features_fractal.csv  # N=10,944 epochs with fractal metrics
│   └── epoch_features_full.csv     # Full feature set
│
├── Figures
│   ├── fractal_analysis.png      # Fractal structure visualization
│   ├── final_mechanism.png       # Alpha-Q gating mechanism
│   ├── revibe_analysis.png       # Q as control parameter
│   ├── bulletproof_tests.png     # Power-law validation
│   └── reviewer_tests.png        # Robustness checks
│
├── Reports
│   ├── fractal_report.md
│   ├── final_mechanism_report.md
│   ├── bulletproof_report.md
│   └── FINAL_STATS_FOR_PAPER.md
│
└── app.py                        # Streamlit dashboard
```

## Installation

```bash
# Clone repository
git clone https://github.com/ExeqTer91/phi-organized-brain.git
cd phi-organized-brain

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- `mne` - EEG data loading and preprocessing
- `scipy` - Signal processing and statistics
- `numpy` - Numerical computation
- `pandas` - Data manipulation
- `scikit-learn` - Machine learning (clustering, classification)
- `matplotlib` - Visualization
- `streamlit` - Interactive dashboard

## Usage

### Run Core Analysis

```bash
# Fractal structure analysis
python fractal_analysis.py

# Mechanism validation
python final_mechanism_tests.py

# Power-law timing (Clauset-style)
python bulletproof_tests.py

# Reviewer-suggested robustness tests
python reviewer_tests.py
```

### Launch Dashboard

```bash
streamlit run app.py --server.port 5000
```

## Datasets

| Dataset | N | Source | Purpose |
|---------|---|--------|---------|
| PhysioNet EEGBCI | 109 | MNE built-in | Primary validation |
| GAMEEMO | 28 | Zenodo | γ/β ratio discovery |
| MPI-LEMON | 211 | OpenNeuro | Large-scale replication |

## Validation Results

### Power-Law Timing (Test A - Clauset-style)

| Metric | Value |
|--------|-------|
| Alpha (MLE) | 2.74 |
| xmin | 15 epochs |
| Bootstrap p-value | 0.24 |
| LR vs Exponential | +177.4 |
| LR vs Lognormal | +71.9 |

**Verdict**: Power-law is statistically preferred over alternatives.

### Q as Control Parameter (Test C)

| Metric | Value |
|--------|-------|
| Leave-one-subject-out AUC | 0.64 |
| Granger LR statistic | 16.8 |
| Granger p-value | < 0.0001 |

**Verdict**: Q predicts regime transitions (causal direction confirmed).

### Fractal Structure (Regime Differences)

| Metric | φ-like | Harmonic | Bridge | Effect Size |
|--------|--------|----------|--------|-------------|
| Aperiodic exponent | 0.62 | 0.96 | 0.81 | d = −0.30 |
| Multifractal width | 0.049 | 0.031 | 0.031 | d = +1.02 |
| DFA α | 0.49 | 0.50 | 0.50 | d = −0.14 |

## Theoretical Implications

1. **φ is a Meta-Organizer**: The golden ratio organizes hierarchical structure (band boundaries, timing, nesting) rather than direct power ratios.

2. **Bridge State = Gating Mechanism**: Alpha-Q gating controls transitions between φ-like and harmonic basins with reversible symmetry.

3. **7-State Architecture = L₄**: The empirically discovered state count matches the Lucas number identity φ⁴ + φ⁻⁴ = 7.

4. **Criticality Signature**: Fractal timing and multifractal width suggest phi-like states operate near criticality.

## Citation

```bibtex
@article{ursachi2026phi,
  title={The φ-Organized Brain: Scale-Free Dynamics, Bridge-State Control, 
         and a Seven-State Neural Architecture},
  author={Ursachi, Andrei},
  journal={Preprint},
  year={2026},
  doi={10.1234/preprint}
}
```

## Related Work

- Pletzer et al. (2010) - Golden ratio in EEG frequency bands
- Rodriguez-Larios & Alaerts (2019) - Theta-alpha frequency ratio and consciousness
- Coldea et al. (2010, Science) - Golden ratio in quantum spin chains
- Rassi et al. (2023) - Golden rhythms framework

## License

MIT License - see [LICENSE](LICENSE) for details.

## Author

**Andrei Ursachi**  
Independent Researcher, Bucharest, Romania  
ORCID: [0009-0002-6114-5011](https://orcid.org/0009-0002-6114-5011)

---

*"The golden ratio appears not as a fixed point but as an organizing principle—a basin of attraction that shapes the hierarchical structure of neural oscillations across multiple scales."*
