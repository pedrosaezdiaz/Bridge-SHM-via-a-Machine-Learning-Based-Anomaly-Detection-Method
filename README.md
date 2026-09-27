# Bridge-SHM-via-a-Machine-Learning-Based-Anomaly-Detection-Method
This repository documents my participation in the structural instrumentation and monitoring project for the Arroyo de las Huertas de Mateo viaduct, located in Minglanilla (Cuenca, Spain), carried out during my internship in the R&D department at CEMOSA. 

This GitHub repository should be reviewed in conjunction with the project report for full context: [📄 View Internship Report](<Internship Report SHM - Pedro Sáez.pdf>).

## File Directory and Description

This project implements a **structural health monitoring (SHM) pipeline** for bridge pot bearings, combining classical state estimation (Kalman filtering), deep learning-based anomaly detection (TimesNet-style blocks), and a two-layer ensemble regression system for sensor signal correction. Files are grouped by module below. 

Note on Data: This repository does not include the sensor datasets used in this project due to confidentiality/privacy restrictions. File paths in the scripts are placeholders and should be updated to point to your own dataset with a matching schema (see column names referenced in each script).

---

### Module: Sensor Signal Correction (Ensemble Learning)

**File Name**: `Two_Layer_Ensemble.py`

**Purpose**: Implements a two-layer stacked ensemble (SVR + XGBoost base models → LASSO meta-model) that corrects temperature-induced drift in inclination and displacement sensor readings across four pot bearings (P1, P5, P9, P15).

**Key Functions/Classes**:
- `create_base_models()` — Grid-searches SVR (linear/RBF/poly) and XGBoost regressors per sensor target.
- `train_ensemble_models()` — Generates out-of-fold and test-set predictions using time-series cross-validation (`TimeSeriesSplit`).
- `calculate_weights_and_aggregate()` — Combines base-model predictions using error-based and time-decay weighting.
- `train_lasso_meta_model()` — Trains per-sensor LASSO meta-models on stacked base-model outputs, with hyperparameter search.
- `plot_predictions_and_residuals()` / `plot_residuals_for_plant()` — Visualizes raw vs. corrected signals and residual behavior against temperature.

**Dependencies**: `scikit-learn` (SVR, Lasso, GridSearchCV, TimeSeriesSplit), `xgboost`, `numpy`, `pandas`, `matplotlib`. No internal file imports — standalone script.

---

### Module: Deep Learning Anomaly Detection

**File Name**: `TimesBlock.py`

**Purpose**: Implements a custom PyTorch **TimesNet-style** architecture that captures periodic patterns in sensor time series via FFT-based period detection and 2D inception convolutions, used as an autoencoder-style reconstruction model for anomaly detection.

**Key Functions/Classes**:
- `avg_amp_fft()` / `arg_topk_A()` — Extracts dominant frequencies/periods from a signal via FFT.
- `reshape_2D()` / `reshape_1D()` — Converts 1D time series to/from 2D representations for convolutional processing.
- `BasicInceptionBlock2D` — Multi-branch (1x1, 3x3, 5x5, pooled) 2D convolutional block.
- `TimesBlock` — Single layer combining period detection, reshaping, and inception convolution.
- `TimesBlockModel` — Stacks multiple `TimesBlock` layers with optional input/output projections.
- `train_timesblock_model()` — Trains the model via MSE reconstruction loss (Adam + LR scheduler).
- `compute_layer_mse()` — Computes per-layer, per-timestep reconstruction error.
- `calculate_gpd_threshold()` — Derives an extreme-value (Generalized Pareto Distribution) anomaly threshold from reconstruction errors.

**Dependencies**: `torch`, `torch.nn`/`torch.optim`, `scipy.fft`, `scipy.stats` (genpareto), `numpy`, `matplotlib`, `scikit-learn` (StandardScaler). No internal file imports — this is the base model consumed by `Model_Validation.py`.

---

**File Name**: `Model_Validation.py`

**Purpose**: Validates the `TimesBlock` anomaly detector by synthetically injecting labeled noise/anomalies into sensor data, then benchmarking detection performance (accuracy, precision, recall, F1, ROC-AUC) across different GPD threshold configurations.

**Key Functions/Classes**:
- `BridgeAnomalyLabeler` — Class that generates synthetic noise ensembles and injects controlled anomalies into clean signals to build ground-truth labeled test sets.
  - `generate_noise_parameters()` / `generate_white_noise()` / `generate_point_noise()` — Builds randomized noise ensembles.
  - `create_labeled_dataset()` — Injects anomalies at a target rate and labels each timestep.
  - `evaluate_detection_performance()` — Computes accuracy/precision/recall/F1 vs. ground truth.
  - `plot_data_comparison()` / `plot_confusion_matrix()` — Diagnostic visualizations.
- Main script sweeps percentile/alpha threshold grids, selects the best F1-scoring configuration, and plots ROC curves and precision/recall/F1 vs. threshold.

**Dependencies**: **Internal** → imports `train_timesblock_model`, `compute_layer_mse`, `TimesBlockModel`, `calculate_gpd_threshold` from `TimesBlock.py`. **External** → `torch`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn` (metrics, StandardScaler), `scipy.stats` (genpareto).

---

### Module: Physics-Based State Estimation

**File Name**: `Filtrado_de_Kalman.py`

**Purpose**: Implements a physics-informed **Kalman filter** for structural health monitoring of bridge pot bearings (based on Erazo et al., 2019), estimating hidden structural states (temperature, inclination, displacement + velocities) from noisy sensor data and deriving a damage index from residual spectral analysis.

**Key Functions/Classes**:
- `StructuralHealthMonitoringKalman` — Core class encapsulating the state-space Kalman filter.
  - `identify_system_matrices()` — Builds state-transition (A), input (B), observation (C), and covariance (Q, R) matrices from modal frequencies, temperature-adjusted stiffness, and measured variance.
  - `_temperature_modulus_relationship()` — Models elastic modulus vs. temperature (steel) per the paper's Eq. 33.
  - `kalman_yo()` — Runs the iterative Kalman filter (predict/update loop) with Cholesky-stabilized gain computation, returning estimated states and residuals.
  - `compute_residual_psd()` — Computes residual power spectral density via Welch's method.
  - `whiteness_test()` — Bayesian whiteness test on residuals (sign-change statistic).
  - `damage_index()` — Computes a normalized structural damage index (Eq. 32) from spectral moments.
- Main script loads/preprocesses multi-sensor data for P1/P5/P9/P15, runs the filter per bearing, and plots measured vs. estimated states, residual PSDs, and damage indices.

**Dependencies**: `numpy`, `pandas`, `scipy` (`signal`, `linalg`, `welch`), `matplotlib`. No internal file imports — standalone script; outputs (`monitoring_data_*.npy`) feed into `Model_Validation.py` / `TimesBlock.py`.

---

> **Usage Notes**: Recommended execution order: `Filtrado_de_Kalman.py` → `TimesBlock.py` / `Model_Validation.py` → `Two_Layer_Ensemble.py`.
