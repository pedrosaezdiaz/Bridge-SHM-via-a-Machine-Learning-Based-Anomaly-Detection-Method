import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
from typing import Tuple, List
import warnings
from scipy.stats import scoreatpercentile
from TimesBlock import train_timesblock_model, compute_layer_mse
from TimesBlock import TimesBlockModel, calculate_gpd_threshold
from scipy.stats import genpareto
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


#Visualizacion
sns.set(style="whitegrid")




class BridgeAnomalyLabeler:
    """
    Anomaly labeler; adds noise to the signal at a random number of time 
    instances and creates a binary vector that classifies each point as anomalous 
    (value 1) or normal (value 0).
    """
    
    def __init__(self, std_max: float = 0.2, std_min: float = 0.1, n_ensembles: int = 100, random_seed: int = 42):
        """
        Defines the initial parameters that modify the deviations 
        in the noise added to the signal.
        """
        self.std_max = std_max
        self.std_min = std_min
        self.n_ensembles = n_ensembles
        self.random_seed = random_seed

    def generate_noise_parameters(self, m: int, n: int, iteration_seed: int = 0) -> np.ndarray:
        """Generate noise parameters for each ensemble."""
        #Fix the seed
        np.random.seed(self.random_seed + iteration_seed)
        std_params = np.zeros((self.n_ensembles, n))
        for i in range(self.n_ensembles):
            for j in range(n):
                rand_val = np.random.uniform(0, 1)
                std_params[i, j] = (self.std_max - self.std_min) * rand_val + self.std_min
        return std_params
    
    def generate_white_noise(self, m: int, n: int, std_params: np.ndarray, iteration_seed: int = 0) -> List[np.ndarray]:
        """Generate white noise for each ensemble."""
        np.random.seed(self.random_seed + iteration_seed + 100)
        noise_ensembles = []
        
        for i in range(self.n_ensembles):
            base_noise = np.random.randn(m, n)
            noise_scaled = np.zeros((m, n))
            for j in range(n):
                col_mean = np.mean(base_noise[:, j])
                col_std = np.std(base_noise[:, j])
                if col_std != 0:
                    standardized = (base_noise[:, j] - col_mean) / col_std
                else:
                    standardized = base_noise[:, j]
                noise_scaled[:, j] = standardized * std_params[i, j]
            noise_ensembles.append(noise_scaled)
        
        return noise_ensembles
    
    
    def generate_point_noise(self, n: int, std_params_row: np.ndarray, iteration_seed: int = 0):
        np.random.seed(self.random_seed + iteration_seed + 100)
        point_ensembles = []
        for i in range(self.n_ensembles):
            base_noise = np.random.randn(1, n)
            standardized = (base_noise - base_noise.mean()) / (base_noise.std() + 1e-6)
            scaled = standardized * std_params_row
            point_ensembles.append(scaled)
        return point_ensembles

    def add_noise_to_signal_corrected(self, signal: np.ndarray, noise_ensembles: List[np.ndarray]) -> np.ndarray:
        """
        Adds noise only to a certain number of points in
        specific.
        
        Parameters:
        -----------
        signal: np.ndarray
            Señal original (puede ser un solo punto o múltiples puntos)
        noise_ensembles: List[np.ndarray]
            Lista de matrices de ruido correspondientes a la señal
    
        Returns:
        --------
        np.ndarray
            Señal con ruido agregado
        """
        m, n = signal.shape
        
        # Crear copia de la señal original
        enhanced_signal = signal.copy()
        
        # Promedio del ruido de todos los ensembles
        avg_noise = np.zeros((m, n))
        for noise in noise_ensembles:
            avg_noise += noise
            
        avg_noise /= self.n_ensembles
        
        # Agregar el ruido promediado
        enhanced_signal += avg_noise
        
        return enhanced_signal

    
    def create_labeled_dataset(self, data: np.ndarray, anomaly_rate: float = 0.05, 
                           iteration_seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a labeled dataset with light noise everywhere and strong noise on anomalies.
        """
        np.random.seed(self.random_seed + iteration_seed + 2000)

        m, n = data.shape
        labels = np.zeros(m)
        noisy_data = data.copy()

        # Determine anomaly indices
        n_anomalies = int(m * anomaly_rate)
        anomaly_indices = np.random.choice(m, n_anomalies, replace=False)
        anomaly_set = set(anomaly_indices)

        # Generate noise
        std_params = self.generate_noise_parameters(m, n)
        noise_ensembles = self.generate_white_noise(m, n, std_params)

        # Apply noise to all points
        for idx in range(m):
            signal = data[idx:idx+1, :]
            noise_per_point = [noise[idx:idx+1, :] for noise in noise_ensembles]
            avg_noise = np.mean(noise_per_point, axis=0)
            if idx in anomaly_set:
                noisy_signal = signal + 50*avg_noise
                labels[idx] = 1
                if np.abs(noisy_signal - signal)<0.5:
                    labels[idx] = 0
            else:
                noisy_signal = signal + 10*avg_noise
                labels[idx] = 0
                if np.abs(noisy_signal - signal)>0.5:
                    labels[idx] = 1

            noisy_data[idx, :] = noisy_signal[0, :]

        return noisy_data, labels
    
            
   
             
    
    def calculate_mse(self, original: np.ndarray, reconstructed: np.ndarray) -> np.ndarray:
        """Calculate Mean Squared Error for each sample."""
        mse_values = np.mean((original - reconstructed) ** 2, axis=1)
        return mse_values
    
    def evaluate_detection_performance(self, true_labels: np.ndarray, 
                                     predicted_labels: np.ndarray) -> dict:
        """Evaluate anomaly detection performance."""
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels, zero_division=0)
        recall = recall_score(true_labels, predicted_labels, zero_division=0)
        f1 = f1_score(true_labels, predicted_labels, zero_division=0)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def plot_data_comparison(self, original_data: np.ndarray, enhanced_data: np.ndarray, 
                           labels: np.ndarray, sensor_idx: int = 0, 
                           sample_points: int = 1000):
        """Plot comparison between original and enhanced data."""
        n_points = min(sample_points, len(original_data))
        indices = np.linspace(0, len(original_data)-1, n_points, dtype=int)
        
        plt.figure(figsize=(15, 8))
        
        # Plot original data
        plt.subplot(2, 1, 1)
        plt.plot(indices, original_data[indices, sensor_idx], 'b-', alpha=0.7, label='Original Signal')
        plt.plot(indices, enhanced_data[indices, sensor_idx], 'r-', alpha=0.7, label='Enhanced Signal')
        anomaly_mask = labels[indices] == 1
        if np.any(anomaly_mask):
            plt.scatter(indices[anomaly_mask], enhanced_data[indices[anomaly_mask], sensor_idx], 
                       c='red', s=20, alpha=0.8, label='Synthetic Anomalies')
        plt.xlabel('Time Step')
        plt.ylabel('Sensor Value')
        plt.title(f'Original vs Enhanced Data - Sensor {sensor_idx}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot difference
        plt.subplot(2, 1, 2)
        difference = enhanced_data[indices, sensor_idx] - original_data[indices, sensor_idx]
        plt.plot(indices, difference, 'g-', alpha=0.7, label='Difference (Enhanced - Original)')
        if np.any(anomaly_mask):
            plt.scatter(indices[anomaly_mask], difference[anomaly_mask], 
                       c='red', s=20, alpha=0.8, label='Anomaly Points')
        plt.xlabel('Time Step')
        plt.ylabel('Difference')
        plt.title('Difference Between Enhanced and Original Signals')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_noise_introduction(self, original_data: np.ndarray, noisy_data: np.ndarray, 
                           labels: np.ndarray, sensor_idx: int = 0, 
                           sample_points: int = 1000):
        n_points = min(sample_points, len(original_data))
        indices = np.linspace(0, len(original_data)-1, n_points, dtype=int)
        
        plt.figure(figsize=(15, 8))
        
        # Plot original data
        plt.subplot(2, 1, 1)
        #plt.plot(indices, original_data[indices, sensor_idx], 'b-', alpha=0.7, label='Original Signal')
        plt.plot(indices, noisy_data[indices, sensor_idx], 'r-', alpha=0.7, label='Noisy Signal')
        anomaly_mask = labels[indices] == 1
        if np.any(anomaly_mask):
            plt.scatter(indices[anomaly_mask], noisy_data[indices[anomaly_mask], sensor_idx], 
                       c='red', s=20, alpha=0.8, label='Synthetic Anomalies')
        plt.xlabel('Time Step (5 minutes)')
        plt.ylabel('Normalized Sensor Value')
        plt.title("Original vs Noise-Injected Signal – Fixed Displacement (LVDT Sensor)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot difference
        plt.subplot(2, 1, 2)
        difference = noisy_data[indices, sensor_idx] - original_data[indices, sensor_idx]
        plt.plot(indices, difference, 'g-', alpha=0.7, label='Difference (Noisy - Original)')
        if np.any(anomaly_mask):
            plt.scatter(indices[anomaly_mask], difference[anomaly_mask], 
                       c='red', s=20, alpha=0.8, label='Anomaly Points')
        plt.xlabel('Time Step')
        plt.ylabel('Difference')
        plt.title('Difference Between Noisy and Original Signals')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    
    def plot_confusion_matrix(self, true_labels: np.ndarray, predicted_labels: np.ndarray):
        """Plot confusion matrix for anomaly detection results."""
        cm = confusion_matrix(true_labels, predicted_labels)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Anomaly'], 
                   yticklabels=['Normal', 'Anomaly'])
        plt.title('Confusion Matrix - Anomaly Detection')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()






if __name__ == "__main__":
    # Cargar solo un sensor (por ejemplo, el 3)
    X0_1D = np.load("monitoring_data_P1.npy")[:, 3:4]

    
    num_time_steps, num_channels = X0_1D.shape
    
    #Split data into training and test (training is performed under clean signal)
    #Normalize
    train_ratio = 0.7
    train_size = int(num_time_steps * train_ratio)
    X_train, X_test = X0_1D[:train_size], X0_1D[train_size:]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    

    # Parámetros del modelo
    d_model = min(256, num_channels * 128)
    k = 3
    num_layers = 3
    min_freq = 1e-6
    max_period = min(168, num_time_steps // 4)

    
     # === ENTRENAMIENTO ===
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32, requires_grad=True).unsqueeze(0)
    model = TimesBlockModel(d_model, k, min_freq, max_period, num_layers, input_dim=num_channels)
    model_train, losses = train_timesblock_model(model, X_train_tensor, num_epochs=10, lr=0.01)
    


    #Calcular MSE
    mse_train_per_layer = compute_layer_mse(model_train, X_train_tensor)
    mse_train_final = mse_train_per_layer[-1]

    # Configuración
    np.random.seed(42)
    torch.manual_seed(42)

    percentiles = [70, 80, 90]
    alphas = [0.6, 0.65, 0.7, 0.8, 0.9, 0.95]
    results = []
    iteration_counter = 0

    for alpha_i in alphas:
        for percentile in percentiles:
            #Threshold con EVA
            threshold_train, _ = calculate_gpd_threshold(mse_train_final,
                                                         percentile=percentile, alpha=alpha_i)

            
            # === ANOMALÍAS ENTRENAMIENTO ===
            anomalies_train_indices_per_layer = [
               np.where(mse > threshold_train)[0] for mse in mse_train_per_layer]
            
            # === Set EVA threshold ===
            mse_test_per_layer = compute_layer_mse(model_train, X_train_tensor)
            mse_test_final = mse_test_per_layer[-1]
            anomalies_test = np.where(mse_test_final > threshold_train)[0]
            timesteps_test = np.arange(train_size, num_time_steps)
            
        
            #Introduce noise to the test set
            labeler = BridgeAnomalyLabeler(std_max=0.2, std_min=0.1, n_ensembles=100, random_seed=42)

            noisy_X_test, labels = labeler.create_labeled_dataset(
                X_test, anomaly_rate=0.05, iteration_seed=iteration_counter
            )

            #Calcula MSE entre training (sin noise) con señal con noise
            noisy_X_test_tensor = torch.tensor(noisy_X_test, dtype=torch.float)
            if noisy_X_test_tensor.ndim == 2:
                noisy_X_test_tensor = noisy_X_test_tensor.unsqueeze(0)
           
            model_train.eval()
            with torch.no_grad():
                model_output = model_train(noisy_X_test_tensor).squeeze(0)  # (T, C)

            # MSE por timestep (no promedio global)
            mse_values = torch.mean((model_output - noisy_X_test_tensor.squeeze(0))**2, dim=1).cpu().numpy()  # (T,)




            # EVA threshold
            percentile_value = np.percentile(mse_values, percentile)
            excesses = mse_values[mse_values > percentile_value] - percentile_value
            print(f"length excesses {len(excesses)}")

            np.random.seed(42 + iteration_counter)
            shape, loc, scale = genpareto.fit(excesses)
            threshold_eva = percentile_value + genpareto.ppf(alpha_i, shape, loc=loc, scale=scale)

            predicted_labels = (mse_values > threshold_eva).astype(int)
            metrics = labeler.evaluate_detection_performance(labels, predicted_labels)

            results.append({
                'alpha': alpha_i,
                'percentile': percentile,
                'threshold_value': threshold_eva,
                'detected': np.sum(predicted_labels),
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'mse_values': mse_values,
                'labels': labels,
                'predicted_labels': predicted_labels,
                'noisy_data': noisy_X_test,
                'number excesses': len(excesses)
            })

            iteration_counter += 1

    # Elegir mejor resultado
    best_result = max(results, key=lambda x: x['f1_score'])

    print("\nMejor configuración:")
    print(f"Alpha: {best_result['alpha']}, Percentile %: {best_result['percentile']}, F1 Score: {best_result['f1_score']:.4f}")

    mse_values = best_result['mse_values']
    labels = best_result['labels']
    predicted_labels = best_result['predicted_labels']
    noisy_data = best_result['noisy_data']

    # Visualizaciones
    plt.figure(figsize=(12, 8))

    # ROC Curve
    plt.subplot(2, 2, 1)
    fpr, tpr, _ = roc_curve(labels, predicted_labels)
    auc = roc_auc_score(labels, predicted_labels)
    plt.plot(fpr, tpr, 'b-', linewidth=2, label="GPD-based Classifier")
    plt.plot([0, 1], [0, 1], 'r--', label="Random Classifier")
    plt.plot([0, 0, 1], [0, 1, 1], 'g--', linewidth=2, label='Perfect Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve (AUC={auc:.4f})')
    plt.legend()
    plt.grid(True)

    # Precision / Recall / F1
    plt.subplot(2, 2, 2)
    thresholds_line = np.linspace(np.min(mse_values), 0.95, 100)
    precisions, recalls, f1s = [], [], []

    for thr in thresholds_line:
        pred = (mse_values > thr).astype(int)
        if np.sum(pred) > 0:
            precisions.append(precision_score(labels, pred, zero_division=0))
            recalls.append(recall_score(labels, pred, zero_division=0))
            f1s.append(f1_score(labels, pred, zero_division=0))
        else:
            precisions.append(0)
            recalls.append(0)
            f1s.append(0)

    plt.plot(thresholds_line, precisions, 'r-', label='Precision', linewidth=2)
    plt.plot(thresholds_line, recalls, 'g-', label='Recall', linewidth=2)
    plt.plot(thresholds_line, f1s, 'b-', label='F1-Score', linewidth=2)
    plt.xlabel('MSE Threshold')
    plt.ylabel('Score')
    plt.title('Score vs MSE Threshold')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


    print("\nMatriz de confusión para el mejor set:")
    labeler.plot_confusion_matrix(labels, predicted_labels)

    print("\nNOISY data comparison:")
    labeler.plot_noise_introduction(X_test, noisy_data, labels, sensor_idx=0)
