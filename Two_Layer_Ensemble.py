#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 08:34:52 2025

@author: psd6587
"""

#Import packages
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.base import clone
import warnings
from xgboost import XGBRegressor
warnings.filterwarnings('ignore')



# Load and preprocess data (keeping your existing preprocessing)
inc_data = pd.read_csv("/home/CEMOSA/psd6587/Escritorio/Datos/datos_INC (NEW).csv")
LVDT_data = pd.read_csv("/home/CEMOSA/psd6587/Escritorio/Datos/datos_LVDT (NEW).csv")
TEMP_data = pd.read_csv("/home/CEMOSA/psd6587/Escritorio/Datos/datos_TEMP (NEW).csv")

# Date preprocessing from Timestamp to date format
inc_data['Date'] = pd.to_datetime(inc_data['Date'])
LVDT_data['Date'] = pd.to_datetime(LVDT_data['Date'])
TEMP_data['Date'] = pd.to_datetime(TEMP_data['Date'])

# Downsample to 10min intervals (1/600 Hz)
# Computes the mean within each widow
inc_data = inc_data.set_index('Date').resample('10min').mean().ffill()
LVDT_data = LVDT_data.set_index('Date').resample('10min').mean().ffill()
TEMP_data = TEMP_data.set_index('Date').resample('10min').mean().ffill()

# Common time index to ensure full temporal alignment
common_index = inc_data.index.intersection(LVDT_data.index).intersection(TEMP_data.index)
inc_data = inc_data.loc[common_index]
LVDT_data = LVDT_data.loc[common_index]
TEMP_data = TEMP_data.loc[common_index]


# Prepare features (x_i in the literature) for each pot bearing 
# For P1-P5-P9-P15: two features; north and south temperature

#Features for P1 bearing 
feature_temp_P1_north = TEMP_data['P1 TEMP NORTE AI C-1'].to_numpy().reshape(-1,1)
feature_temp_P1_south = TEMP_data['P1 TEMP SUR AI C-2'].to_numpy().reshape(-1,1)

#Features for P5 bearing 
feature_temp_P5_north = TEMP_data['P5 TEMP NORTE AI J-1'].to_numpy().reshape(-1,1)
feature_temp_P5_south = TEMP_data['P5 TEMP SUR AI J-2'].to_numpy().reshape(-1,1)

#Features for P9 bearing 
feature_temp_P9_north = TEMP_data['P9 TEMP NORTE AI H-1'].to_numpy().reshape(-1,1)
feature_temp_P9_south = TEMP_data['P9 TEMP SUR AI H-2'].to_numpy().reshape(-1,1)

#Features for P15 bearing 
feature_temp_P15_north = TEMP_data['P15 TEMP NORTE AI E-1'].to_numpy().reshape(-1,1)
feature_temp_P15_south = TEMP_data['P15 TEMP SUR AI E-2'].to_numpy().reshape(-1,1)





#Target sensors for P1-P5-P9-P15 bearings: 2 inclinations and 3 displacements
#Correspond to entries y_i in the literature
P1_sensor_targets = {
    'inc_X': inc_data['P1 INC AXIS X AI C-3'].to_numpy().reshape(-1,1),
    'inc_Y': inc_data['P1 INC AXIS  Y AI C-4'].to_numpy().reshape(-1,1),
    'LVDT_transv': LVDT_data['P1 LVDT TRANSV AI C-5'].to_numpy().reshape(-1,1),
    'LVDT_fixed': LVDT_data['P1 LVDT FIJO AI D-1'].to_numpy().reshape(-1,1),
    'LVDT_free': LVDT_data['P1 LVDT LIBRE AI C-6'].to_numpy().reshape(-1,1)
}

P5_sensor_targets = {
    'inc_X': inc_data['P5 INC AXIS X AI J-3'].to_numpy().reshape(-1,1),
    'inc_Y': inc_data['P5 INC AXIS Y AI J-4'].to_numpy().reshape(-1,1),
    'LVDT_transv': LVDT_data['P5 LVDT TRANSV AI J-5'].to_numpy().reshape(-1,1),
    'LVDT_fixed': LVDT_data['P5 LVDT FIJO AI I-1'].to_numpy().reshape(-1,1),
    'LVDT_free': LVDT_data['P5 LVDT LIBRE AI J-6'].to_numpy().reshape(-1,1)
}

P9_sensor_targets = {
    'inc_X': inc_data['P9 INC AXIS X AI H-3'].to_numpy().reshape(-1,1),
    'inc_Y': inc_data['P9 INC AXIS Y AI H-4'].to_numpy().reshape(-1,1),
    'LVDT_transv': LVDT_data['P9 LVDT TRANSV AI H-5'].to_numpy().reshape(-1,1),
    'LVDT_fixed': LVDT_data['P9 LVDT FIJO AI G-1'].to_numpy().reshape(-1,1),
    'LVDT_free': LVDT_data['P9 LVDT LIBRE AI H-6'].to_numpy().reshape(-1,1)
}

P15_sensor_targets = {
    'inc_X': inc_data['P15 INC AXIS X AI E-3'].to_numpy().reshape(-1,1),
    'inc_Y': inc_data['P15 INC AXIS Y  AI E-4'].to_numpy().reshape(-1,1),
    'LVDT_transv': LVDT_data['P15 LVDT  TRANSV AI E-5'].to_numpy().reshape(-1,1),
    'LVDT_fixed': LVDT_data['P15 LVDT FIJO  AI F-1'].to_numpy().reshape(-1,1),
    'LVDT_free': LVDT_data['P15  LVDT LIBRE AI E-6'].to_numpy().reshape(-1,1)
}

#Features (temperatures) are stacked into a (T,2) matrix for each pot bearing
X_all_P1 = np.column_stack([feature_temp_P1_north, feature_temp_P1_south])
X_all_P5 = np.column_stack([feature_temp_P5_north, feature_temp_P5_south])
X_all_P9 = np.column_stack([feature_temp_P9_north, feature_temp_P9_south])
X_all_P15 = np.column_stack([feature_temp_P15_north, feature_temp_P15_south])

#Targets (inclination and displacements) are stacked into a (T,5) matrix 
#for each pot bearing
y_all_P1 = np.column_stack([target.flatten() for target in P1_sensor_targets.values()])
y_all_P5 = np.column_stack([target.flatten() for target in P5_sensor_targets.values()])
y_all_P9 = np.column_stack([target.flatten() for target in P9_sensor_targets.values()])
y_all_P15 = np.column_stack([target.flatten() for target in P15_sensor_targets.values()])



#Define the name of the sensors: x/y-axis inclination and transversal, 
#fixed and free LVDT sensor values
sensor_names = list(P1_sensor_targets.keys())


#Since targets and features are chronologically ordered, 
#the train-test split applies no shuffle to avoid data-leckeage on time
#indices. X_train_P corresponds to R^* in the literature and X_test_P to the
#set R.

#Train-test split P1:
X_train_P1, X_test_P1, y_train_P1, y_test_P1 = train_test_split(
    X_all_P1, y_all_P1, test_size=0.2, shuffle=False)

#Train-test split P5:
X_train_P5, X_test_P5, y_train_P5, y_test_P5 = train_test_split(
    X_all_P5, y_all_P5, test_size=0.2, shuffle=False)


#Train-test split P9:
X_train_P9, X_test_P9, y_train_P9, y_test_P9 = train_test_split(
    X_all_P9, y_all_P9, test_size=0.2, shuffle=False)


#Train-test split P1:
X_train_P15, X_test_P15, y_train_P15, y_test_P15 = train_test_split(
    X_all_P15, y_all_P15, test_size=0.2, shuffle=False)

#Scale features: separate for each pot bearing
#The features are scaled using training data only to prevent data leakage.
#Scale P1 features
scaler_x_P1 = StandardScaler()
X_train_P1_scaled = scaler_x_P1.fit_transform(X_train_P1)
X_test_P1_scaled = scaler_x_P1.transform(X_test_P1)
X_all_P1_scaled = scaler_x_P1.transform(X_all_P1)

#Scale P5 features
scaler_x_P5 = StandardScaler()
X_train_P5_scaled = scaler_x_P5.fit_transform(X_train_P5)
X_test_P5_scaled = scaler_x_P5.transform(X_test_P5)
X_all_P5_scaled = scaler_x_P5.transform(X_all_P5)

#Scale P9 features
scaler_x_P9 = StandardScaler()
X_train_P9_scaled = scaler_x_P9.fit_transform(X_train_P9)
X_test_P9_scaled = scaler_x_P9.transform(X_test_P9)
X_all_P9_scaled = scaler_x_P9.transform(X_all_P9)

#Scale P15 features
scaler_x_P15 = StandardScaler()
X_train_P15_scaled = scaler_x_P15.fit_transform(X_train_P15)
X_test_P15_scaled = scaler_x_P15.transform(X_test_P15)
X_all_P15_scaled = scaler_x_P15.transform(X_all_P15)

#Scale targets: separate for each pot bearing 
#The target values are scaled using training data only to prevent data leakage.
#Ensures consistent scaling across train, test, and full datasets (per sensor).

#Scale P1 targets
scaler_y_P1 = StandardScaler()
y_train_P1_scaled = scaler_y_P1.fit_transform(y_train_P1)
y_test_P1_scaled = scaler_y_P1.transform(y_test_P1)
y_all_P1_scaled = scaler_y_P1.transform(y_all_P1)

#Scale P5 target
scaler_y_P5 = StandardScaler()
y_train_P5_scaled = scaler_y_P5.fit_transform(y_train_P5)
y_test_P5_scaled = scaler_y_P5.transform(y_test_P5)
y_all_P5_scaled = scaler_y_P5.transform(y_all_P5)

#Scale P9 target
scaler_y_P9 = StandardScaler()
y_train_P9_scaled = scaler_y_P9.fit_transform(y_train_P9)
y_test_P9_scaled = scaler_y_P9.transform(y_test_P9)
y_all_P9_scaled = scaler_y_P9.transform(y_all_P9)

#Scale P15 target
scaler_y_P15 = StandardScaler()
y_train_P15_scaled = scaler_y_P15.fit_transform(y_train_P15)
y_test_P15_scaled = scaler_y_P15.transform(y_test_P15)
y_all_P15_scaled = scaler_y_P15.transform(y_all_P15)


#Create the optimal SVR and XGBoost base models given the input train data
def create_base_models(X_train, y_train):
    """
    Train and tune base regression models for each sensor target.

    For each sensor in the multi-output target `y_train`, this function performs hyperparameter tuning 
    of (SVR) with different kernels (linear, RBF, polynomial) and XGBoost regressors. It selects the best-tuned model for each sensor
    and each model type based on cross-validation.

    Parameters:
        X_train (np.array): Training feature matrix of shape (n_samples , n_features).
        y_train (np.array): Training target matrix of shape (n_samples, n_sensors), where each 
                            column corresponds to a different sensor's target values.
    The number of samples (n_samples ) corresponds to the number of time instances.

    Returns:
        best_models (dict): Nested dictionary where the first-level keys are sensor identifiers (e.g., 'sensor_0'), 
                            and the second-level keys are model types/kernels (e.g., 'linear', 'rbf', 'poly', 'xgb'). 
                            The values are the best-fit trained model instances for each sensor-model combination.
    """
    # Param grids per model
    param_grid_linear = {
        'C': [0.01, 0.1, 1, 10, 100],
        'epsilon': [0.01, 0.1, 0.5]
    }
    param_grid_rbf = {
        'C': [0.01, 0.1, 1, 10, 100],
        'epsilon': [0.01, 0.1, 0.5],
        'gamma': ['scale', 'auto']
    }
    param_grid_poly = {
        'C': [0.1, 1, 10],
        'epsilon': [0.01, 0.1, 0.5],
        'degree': [2, 3]
    }
    param_grid_xgb = {
        'n_estimators': [50, 100],
        'max_depth': [2, 4],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8]
    }

    best_models = {}

    for i in range(y_train.shape[1]):
        best_models[f"Sensor_{i}"] = {}

        # SVR - Linear
        svr_linear = SVR(kernel='linear')
        gscv_linear = GridSearchCV(svr_linear, param_grid_linear, cv=3)
        gscv_linear.fit(X_train, y_train[:, i])
        best_models[f"Sensor_{i}"]['SVR_linear'] = gscv_linear.best_estimator_
        print(f"Sensor {i} SVR-linear best: {gscv_linear.best_params_}")

        # SVR - RBF
        svr_rbf = SVR(kernel='rbf')
        gscv_rbf = GridSearchCV(svr_rbf, param_grid_rbf, cv=3)
        gscv_rbf.fit(X_train, y_train[:, i])
        best_models[f"Sensor_{i}"]['SVR_rbf'] = gscv_rbf.best_estimator_
        print(f"Sensor {i} SVR-rbf best: {gscv_rbf.best_params_}")

        # SVR - Poly
        svr_poly = SVR(kernel='poly')
        gscv_poly = GridSearchCV(svr_poly, param_grid_poly, cv=3)
        gscv_poly.fit(X_train, y_train[:, i])
        best_models[f"Sensor_{i}"]['SVR_poly'] = gscv_poly.best_estimator_
        print(f"Sensor {i} SVR-poly best: {gscv_poly.best_params_}")

        # XGBoost
        xgb_model = XGBRegressor(objective='reg:squarederror', verbosity=0)
        gscv_xgb = GridSearchCV(xgb_model, param_grid_xgb, cv=3)
        gscv_xgb.fit(X_train, y_train[:, i])
        best_models[f"Sensor_{i}"]['XGBoost'] = gscv_xgb.best_estimator_
        print(f"Sensor {i} XGBoost best: {gscv_xgb.best_params_}")

    return best_models



#TimeSeriesSplit: cross-validation that preserves temporal order.
#`time_scaling_lambda` controls the decay rate of time-based weights 
#during ensemble aggregation as explained in equation () in the literature
k_fold = TimeSeriesSplit(n_splits=5)
time_scaling_lambda = 100



# Ensemble training
def train_ensemble_models(X_train, y_train, X_test, y_all, k_fold, best_models):
    """Train ensemble models with proper out-of-fold predictions and test set evaluation

    Args:
        X_train: Training features (n_samples_train, n_features)
        y_train: Training targets (n_samples_train, n_sensors)
        X_test: Test features (n_samples_test, n_features)
        k_fold: TimeSeriesSplit object for cross-validation
    
    Returns:
        model_oof_predictions: Out-of-fold predictions for training data (B_k)
        model_test_predictions: Test set predictions for each base model
        for each fold F_j(R)
        model_errors: RMSE errors per fold
"""
    
    
    
    
    # Storage for predictions and errors
    model_oof_predictions = {}
    model_test_predictions = {}
    model_errors = {}
    
    # Initialize storage with combined keys: base_model 
    for sensor_name, kernel_models in best_models.items():
        for kernel_name  in kernel_models:
            model_name = f"{sensor_name}_{kernel_name}"
            model_oof_predictions[model_name] = np.zeros(y_train.shape[0])
            model_test_predictions[model_name] = []
            model_errors[model_name] = []
    
    for sensor_name, base_models in best_models.items():
        for kernel_name, base_model in base_models.items():  
            model_name = f"{sensor_name}_{kernel_name}"
            print(f"Training {model_name}...")

            fold_test_predictions = []
            fold_errors = []
        
            for fold, (train_idx, val_idx) in enumerate(k_fold.split(X_train)):
                model = clone(base_model)
            
                X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
                y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
                
                # Note: Each model is for a single sensor -> get sensor index from sensor_name
                sensor_idx = int(sensor_name.split('_')[1])
                
                # Train model on corresponding sensor target only (1D)
                model.fit(X_fold_train, y_fold_train[:, sensor_idx])
                
                # Predict on val fold 
                val_pred = model.predict(X_fold_val)
                
                # Store out-of-fold predictions in full vector at val indices
                model_oof_predictions[model_name][val_idx] = val_pred
                
                # Predict on test set 
                test_pred = model.predict(X_test)
                fold_test_predictions.append(test_pred)
                
                # Calculate fold error (RMSE for this sensor)
                fold_error = np.sqrt(mean_squared_error(y_fold_val[:, sensor_idx], val_pred))
                fold_errors.append(fold_error)
            
            # Save predictions and errors per model
            model_test_predictions[model_name] = fold_test_predictions
            model_errors[model_name] = np.array(fold_errors)
    
    return model_oof_predictions, model_test_predictions, model_errors



#Function that combines outputs from the different base models and computes
#meta model training and test sets 
def calculate_weights_and_aggregate(test_preds, errors, k_fold, time_scaling_lambda=1):
    """Calculate weighted average of predictions using error and time-based weights 

    Args:
        test_preds: Dictionary of test set predictions per model per fold
        errors: Dictionary of RMSE errors per model per fold
        k_fold: TimeSeriesSplit object used for CV
        time_scaling_lambda: Controls time-based weighting strength
    
    Returns:
        aggregated_test_preds: Weighted average predictions for test set
        This corresponds to \mathbf{M} in the literature
"""

    base_models = list(test_preds.keys())
    #Training is done per-sensor, per base model
    
    # Calculate time-based weights (beta)
    sum_folds_time = sum([(i+1)**(1/time_scaling_lambda) for i in range(k_fold.n_splits)])
    beta_weights = np.array([
        ((i+1)**(1/time_scaling_lambda)) / sum_folds_time 
        for i in range(k_fold.n_splits)
    ])
    
    # Aggregate predictions for each model
    aggregated_test_preds = {}
    
    
    for model_name in base_models:
        # Error-based weights (alpha)
        model_errors = errors[model_name]
        total_error = np.sum(model_errors)
        #alpha_weights = model_errors / total_error if total_error > 0 else np.ones_like(model_errors) / len(model_errors)
        #Reemplazar definicion de pesos alpha 
        inv_errors = np.exp(-model_errors)  # Errores más grandes reciben menor peso exponencialmente
        alpha_weights = inv_errors / (np.sum(inv_errors))

    
        # Combined weights
        gamma_weights = 0.5* alpha_weights + 0.5* beta_weights
        
        # Aggregate test predictions
        test_pred_weighted = np.zeros_like(test_preds[model_name][0])
        for i, weight in enumerate(gamma_weights):
            test_pred_weighted += weight * test_preds[model_name][i]
        
        aggregated_test_preds[model_name] = test_pred_weighted
        
        
    return aggregated_test_preds



def train_lasso_meta_model(oof_preds, agg_test_preds, y_train_scaled, sensor_names, param_grid=None):
    """
    Train independent LASSO meta-models for each sensor using only its base model predictions.
    
    Args:
        oof_preds: Out-of-fold predictions dict {model_name: predictions}
        agg_test_preds: Aggregated test predictions dict {model_name: predictions}  
        y_train_scaled: Scaled target values (n_train_samples, n_sensors)
        sensor_names: List of sensor identifiers
        param_grid: Optional custom parameter grid for GridSearchCV
        
    Returns:
        dict: Trained LASSO models keyed by sensor name
        dict: Best alpha values keyed by sensor name
        ndarray: Test set predictions (n_test_samples, n_sensors)
        ndarray: Full dataset predictions (n_train_samples + n_test_samples, n_sensors)
    """
    #Parameter values for alpha^lasso as defined in the literature
    #These grids can be adjusted for better training of the model: 
    #specially avoiding correlations in the residuals
    sensor_param_grids = {
        'inc_X': {'alpha': np.logspace(-7, -2, 10)},
        'inc_Y': {'alpha': np.logspace(-7, -2, 10)},
        'LVDT_transv': {'alpha': np.logspace(-6, -1, 10)},
        'LVDT_fixed': {'alpha': np.logspace(-6, -1, 10)},
        'LVDT_free': {'alpha': np.logspace(-7, -1, 10)},
}

    lasso_models = {}
    best_alphas = {}
    meta_test_pred_scaled = []
    
    for sensor_idx, sensor_name in enumerate(sensor_names):
        print(f"\nTraining LASSO meta-model for {sensor_name}...")
        
        # Only use meta models for this specific sensor
        sensor_model_names = [name for name in oof_preds.keys() 
                             if name.startswith(f"Sensor_{sensor_idx}_")]
        
        print(f"Using base models: {sensor_model_names}")
        
        # Crear meta-features solo con predicciones de este sensor (4 modelos base)
        X_meta_train_sensor = np.column_stack([oof_preds[name] for name in sensor_model_names])
        X_meta_test_sensor = np.column_stack([agg_test_preds[name] for name in sensor_model_names])
        
        # Grid search con TimeSeriesSplit
        lasso = Lasso(max_iter=1000000, random_state=42)
        grid_search = GridSearchCV(
            estimator=lasso,
            param_grid=sensor_param_grids[sensor_name],
            scoring='neg_mean_absolute_error',
            cv=TimeSeriesSplit(n_splits=5),
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_meta_train_sensor, y_train_scaled[:, sensor_idx])
        
        # Store best model
        best_model = grid_search.best_estimator_
        lasso_models[sensor_name] = best_model
        best_alphas[sensor_name] = grid_search.best_params_['alpha']
        
        # Generate predictions on X_meta_test_sensor
        test_pred = best_model.predict(X_meta_test_sensor)
        meta_test_pred_scaled.append(test_pred)
        
        print(f"Best alpha for {sensor_name}: {best_alphas[sensor_name]:.6f}")
    
    # Stack predictions
    meta_test_pred_scaled = np.column_stack(meta_test_pred_scaled)
    
    # Reconstruct full predictions (train + test)
    meta_full_pred_scaled = []
    for sensor_idx, sensor_name in enumerate(sensor_names):
        sensor_model_names = [name for name in oof_preds.keys() 
                             if name.startswith(f"Sensor_{sensor_idx}_")]
        
        # Combine train + test meta-features for this sensor
        X_meta_train_sensor = np.column_stack([oof_preds[name] for name in sensor_model_names])
        X_meta_test_sensor = np.column_stack([agg_test_preds[name] for name in sensor_model_names])
        X_meta_full_sensor = np.vstack([X_meta_train_sensor, X_meta_test_sensor])
        
        # Predict full dataset for this sensor
        full_pred = lasso_models[sensor_name].predict(X_meta_full_sensor)
        meta_full_pred_scaled.append(full_pred)
    
    meta_full_pred_scaled = np.column_stack(meta_full_pred_scaled)
    
    return lasso_models, best_alphas, meta_test_pred_scaled, meta_full_pred_scaled



def plot_predictions_and_residuals(temp_north, temp_south, residuals, 
                                   sensor_names, y_all_unscaled, meta_full_pred, plant_name):
    """
    Plot raw vs. predicted signals and residuals vs. temperature for each sensor.

    Args:
        temp_north (ndarray): North temperature readings for test set.
        temp_south (ndarray): South temperature readings for test set.
        residuals (ndarray): Residuals between predictions and targets (n_samples, n_sensors).
        sensor_names (list): List of sensor name strings.
        y_all_unscaled (ndarray): True unscaled signals (full dataset).
        meta_full_pred (ndarray): Predicted signals from meta-model (unscaled, full dataset).
        plant_name (string): Name of the Pot Bearing (P1-P5-P9-P15)
    """
    fig, axes = plt.subplots(len(sensor_names), 2, figsize=(16, 4 * len(sensor_names)))
    
    for i, sensor_name in enumerate(sensor_names):
        # --- Line Plot: Prediction vs Actual ---
        ax_line = axes[i, 0]
        time_idx = range(len(y_all_unscaled))
        ax_line.plot(time_idx, y_all_unscaled[:, i], label='Raw Signal', alpha=0.7, linewidth=1)
        ax_line.plot(time_idx, meta_full_pred[:, i], label='Meta-Model Prediction', alpha=0.8, linewidth=1.2)
        ax_line.set_title(f'{sensor_name} - Prediction vs Raw Signal for {plant_name}')
        ax_line.set_xlabel('Time Index (minutes)')
        ax_line.set_ylabel('Signal Value')
        ax_line.legend()
        ax_line.grid(True, alpha=0.3)

        # --- Residuals Scatter Plot: Residuals vs. Temperature ---
        ax_resid = axes[i, 1]
        ax_resid.scatter(temp_north, residuals[:, i], label='North Temp', alpha=0.6, s=10, color='blue')
        ax_resid.scatter(temp_south, residuals[:, i], label='South Temp', alpha=0.6, s=10, color='red')
        ax_resid.set_title(f'{sensor_name} - Residuals vs. Temperature for {plant_name}')
        ax_resid.set_xlabel('Temperature')
        ax_resid.set_ylabel('Residual')
        ax_resid.legend()
        ax_resid.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    
    
    
def plot_residuals_for_plant(y_test_scaled, meta_pred_scaled, X_test_scaled, y_all_scaled, meta_full_pred_scaled, sensor_names, plant_name):
    residuals = y_test_scaled - meta_pred_scaled
    print(f"Plotting residuals for {plant_name}")
    plot_predictions_and_residuals(
        temp_north=X_test_scaled[:, 0],  # North temps
        temp_south=X_test_scaled[:, 1],  # South temps
        residuals=residuals,
        sensor_names=sensor_names,
        y_all_unscaled=y_all_scaled,
        meta_full_pred=meta_full_pred_scaled,
        plant_name=plant_name)

    

if __name__ == "__main__":
    
    #Define best base models for the data of each Pot Bearing
    #Note this takes time, once base models are decided, run the code below
    #only
    best_models_P1 = create_base_models(X_train_P1_scaled, y_train_P1_scaled)
    best_models_P5 = create_base_models(X_train_P5_scaled, y_train_P5_scaled)
    best_models_P9 = create_base_models(X_train_P9_scaled, y_train_P9_scaled)
    best_models_P15 = create_base_models(X_train_P15_scaled, y_train_P15_scaled)

    #Train base models for each Pot Bearing 
    #P1 
    oof_preds_P1, test_preds_P1, errors_P1 = train_ensemble_models(
        X_train_P5_scaled, y_train_P5_scaled, X_test_P5_scaled, y_all_P1_scaled, k_fold, best_models_P1)
    #P5 
    oof_preds_P5, test_preds_P5, errors_P5 = train_ensemble_models(
        X_train_P5_scaled, y_train_P5_scaled, X_test_P5_scaled, y_all_P5_scaled, k_fold, best_models_P5)
    #P9
    oof_preds_P9, test_preds_P9, errors_P9 = train_ensemble_models(
        X_train_P5_scaled, y_train_P5_scaled, X_test_P5_scaled, y_all_P5_scaled, k_fold, best_models_P9)
    #P15
    oof_preds_P15, test_preds_P15, errors_P15 = train_ensemble_models(
        X_train_P5_scaled, y_train_P5_scaled, X_test_P5_scaled, y_all_P5_scaled, k_fold, best_models_P15)


    #Aggregate predictions for input of the second layer of the Meta-Model
    #P1
    agg_test_preds_P1 = calculate_weights_and_aggregate(
       test_preds_P1, errors_P1, k_fold, time_scaling_lambda
       )
    #P5
    agg_test_preds_P5 = calculate_weights_and_aggregate(
       test_preds_P5, errors_P5, k_fold, time_scaling_lambda
       )
    #P9
    agg_test_preds_P9 = calculate_weights_and_aggregate(
       test_preds_P9, errors_P9, k_fold, time_scaling_lambda
       )
    #P15
    agg_test_preds_P15 = calculate_weights_and_aggregate(
       test_preds_P15, errors_P15, k_fold, time_scaling_lambda
       )



    # Train meta-models using LASSO 
    # The function lasso_models returns the corrected signal 
    # Predictions are also recovered to the original scale
    #Lasso P1
    lasso_models_P1, best_alphas_P1, meta_test_pred_scaled_P1, meta_full_pred_scaled_P1 = train_lasso_meta_model(
        oof_preds_P1,
        agg_test_preds_P1,
        y_train_P1_scaled,
        sensor_names, 
        )

    #Lasso P5
    lasso_models_P5, best_alphas_P5, meta_test_pred_scaled_P5, meta_full_pred_scaled_P5 = train_lasso_meta_model(
        oof_preds_P5,
        agg_test_preds_P5,
        y_train_P5_scaled,
        sensor_names, 
        )

    #Lasso P9
    lasso_models_P9, best_alphas_P9, meta_test_pred_scaled_P9, meta_full_pred_scaled_P9 = train_lasso_meta_model(
        oof_preds_P9,
        agg_test_preds_P9,
        y_train_P9_scaled,
        sensor_names, 
        )

    #Lasso P15
    lasso_models_P15, best_alphas_P15, meta_test_pred_scaled_P15, meta_full_pred_scaled_P15 = train_lasso_meta_model(
        oof_preds_P15,
        agg_test_preds_P15,
        y_train_P15_scaled,
        sensor_names, 
        )


    plants = {
        "P1": (y_test_P1_scaled, meta_test_pred_scaled_P1, X_test_P1_scaled, y_all_P1_scaled, meta_full_pred_scaled_P1),
        "P5": (y_test_P5_scaled, meta_test_pred_scaled_P5, X_test_P5_scaled, y_all_P5_scaled, meta_full_pred_scaled_P5),
        "P9": (y_test_P9_scaled, meta_test_pred_scaled_P9, X_test_P9_scaled, y_all_P9_scaled, meta_full_pred_scaled_P9),
        "P15": (y_test_P15_scaled, meta_test_pred_scaled_P15, X_test_P15_scaled, y_all_P15_scaled, meta_full_pred_scaled_P15),
    }

    for plant_name, data in plants.items():
        plot_residuals_for_plant(*data, sensor_names=sensor_names, plant_name=plant_name)


    print("Comment: the parameters used in the param-grid of the LASSO influence the final output heavily")
    print("Comment: the parameters where adjusted to fit as much as possible the results from P5")
    print("Clear correlation in the residuals for many plots suggest the need for further tuning parameters")
    
    print("Store fitted results for P5")
    np.save("meta_full_pred_scaled_P5.npy", meta_full_pred_scaled_P5)
    



