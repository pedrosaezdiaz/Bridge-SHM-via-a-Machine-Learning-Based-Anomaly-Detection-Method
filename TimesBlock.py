import numpy as np
from scipy.fft import fft, fftfreq
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import genpareto
import matplotlib.pyplot as plt
from scipy.stats import scoreatpercentile
import torch.optim as optim
from sklearn.preprocessing import StandardScaler



def avg_amp_fft(X):
    """
    Computes FFT of the TxC matrix, its amplitude, and the mean as in equation (1).
    
    Args:
        X (torch.Tensor): Input time series of shape (T, C).
    
    Returns:
        torch.Tensor: Mean amplitude across channels, shape (T,).
    """
    X_fft = torch.fft.fft(X, dim=0)
    X_amp = torch.abs(X_fft)
    X_amp_mean = X_amp.mean(dim=1)
    print(f"amp size {X_amp_mean.shape}")
    return X_amp_mean

def arg_topk_A(amplitudes, k, min_freq, max_period, dt):
    """
    Finds the k dominant frequencies (highest amplitude) and returns their periods.
    
    Args:
        amplitudes (torch.Tensor): FFT amplitudes, shape (T,).
        k (int): Number of dominant frequencies to select.
        min_freq (float): Minimum frequency threshold.
        max_period (int): Maximum allowed period.
        dt (float): Time step.
    
    Returns:
        tuple: (top_k_freqs, top_k_amplitudes, periods).
    """
    length_T = amplitudes.shape[0]
    freqs = torch.fft.fftfreq(length_T, d=dt)  # Use the provided dt
    
    # Remove DC component and upper half of spectrum
    freqs = freqs[1:int(np.floor(length_T / 2))]
    amplitudes = amplitudes[1:int(np.floor(length_T / 2))]
    print(f"freqs shape: {freqs.shape}, amplitudes shape: {amplitudes.shape}")
    
    # Filter frequencies below min_freq - use torch operations to maintain gradients
    valid_mask = freqs > min_freq
    if valid_mask.sum() == 0:
        print(f"Warning: No freqs > {min_freq}. Using unfiltered freqs.")
        valid_mask = torch.ones_like(freqs, dtype=torch.bool)
    
    freqs = freqs[valid_mask]
    amplitudes = amplitudes[valid_mask]
    
    # Select top-k frequencies
    top_k_indices = torch.argsort(amplitudes, descending=True)[:k]
    top_k_freqs = freqs[top_k_indices]
    top_k_amplitudes = amplitudes[top_k_indices]
    
    # Convert frequencies to periods, capped at max_period
    # Use torch operations to maintain gradients
    periods = [
        min(int(np.ceil(length_T / float(f))), max_period)
        for f in top_k_freqs
    ]  # Convert to int for reshaping
    
    return top_k_freqs, top_k_amplitudes, periods

def reshape_2D(X_lminus_1D, period):
    """
    Reshapes 1D tensor to 2D for inception (equation 5).
    
    Args:
        X_lminus_1D (torch.Tensor): Input of shape (T, d_model).
        period (int): Period p_i for reshaping.
    
    Returns:
        torch.Tensor: Reshaped tensor of shape (period, num_blocks, d_model).
    """
    print(f"shape {X_lminus_1D.shape}")
    T, d_model = X_lminus_1D.shape
    remainder = T % period
    if remainder != 0:
        padding_length = period - remainder
        # Use torch.zeros with same device and dtype to maintain gradients
        padding = torch.zeros(padding_length, d_model, device=X_lminus_1D.device, dtype=X_lminus_1D.dtype)
        X_lminus_1D_padded = torch.cat([X_lminus_1D, padding], dim=0)
    else:
        X_lminus_1D_padded = X_lminus_1D
    
    T_padded = X_lminus_1D_padded.shape[0]
    f_i = T_padded // period
    X_l_2D_i = X_lminus_1D_padded.view(f_i, period, d_model)
    X_l_2D_i = X_l_2D_i.permute(1, 0, 2).contiguous()
    return X_l_2D_i


def reshape_1D(hat_X1_2D_i, original_length):
    """
    Transforms a 2D tensor (with reshape and possibly padding) 
    back to a 1D tensor (without padding) (i.e. equation 7)

    Args:
        hat_X1_2D_i (torch.Tensor): Tensor of shape (1, d_model, period, num_blocks) 
        direct from inception_2D
        original_length (int): Original length of time series T (without padding)

    Returns:
        X_1D (torch.Tensor): Tensor of shape (original_length, d_model)
    """
    # Delete batch dimension
    X = hat_X1_2D_i.squeeze(0)  # (d_model, period, num_blocks)

    # Reorganize: (d_model, period, num_blocks) → (period, num_blocks, d_model)
    X = X.permute(1, 2, 0).contiguous()

    # Combine period and num_blocks into single time dimension
    T_padded = X.shape[0] * X.shape[1]
    hat_X_1D_i = X.view(T_padded, -1)  # (T_padded, d_model)

    # Truncate to recover original length
    hat_X_1D_i = hat_X_1D_i[:original_length]

    return hat_X_1D_i


class BasicInceptionBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        assert out_channels % 4 == 0, "out_channels must be divisible by 4"
        self.branch1 = nn.Conv2d(in_channels, out_channels//4, kernel_size=1)
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, kernel_size=1),
            nn.Conv2d(out_channels//4, out_channels//4, kernel_size=3, padding=1)
        )
        self.branch5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, kernel_size=1),
            nn.Conv2d(out_channels//4, out_channels//4, kernel_size=5, padding=2)
        )
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_channels//4, kernel_size=1)
        )
        self.relu = nn.ReLU() #aplicar relu a los output de cada branch individualmente

    def forward(self, x):
        return self.relu(
            torch.cat([
                self.branch1(x),
                self.branch3(x),
                self.branch5(x),
                self.branch_pool(x)
            ], dim=1)
        )

class TimesBlock(nn.Module):
    def __init__(self, d_model, k, min_freq, max_period):
        super().__init__()
        self.d_model = d_model
        self.k = k
        self.min_freq = min_freq
        self.max_period = max_period
        self.inception = BasicInceptionBlock2D(d_model, d_model)

    def forward(self, X_lminus_1D):
        X_amp_mean = avg_amp_fft(X_lminus_1D)
        top_k_frequencies, top_k_amplitudes, periods = arg_topk_A(X_amp_mean, self.k, self.min_freq, self.max_period, dt=1e-10)
        print(X_amp_mean.shape)
        
        # Ensure amplitudes tensor requires gradients
        amplitudes = top_k_amplitudes.clone().detach().requires_grad_(True)
        weights = F.softmax(amplitudes, dim=0)
        
        weighted_sum = torch.zeros_like(X_lminus_1D)
        for i in range(len(periods)):
            X_l_2D_i = reshape_2D(X_lminus_1D, periods[i])
            X_l_2D_i = X_l_2D_i.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
            hat_X_l_2D_i = self.inception(X_l_2D_i)
            hat_X_l_1D_i = reshape_1D(hat_X_l_2D_i, X_lminus_1D.shape[0])  # Fix: use shape[0] not shape[-2]
            weighted_sum += weights[i] * hat_X_l_1D_i
        
        X_l_1D = weighted_sum + X_lminus_1D
        return X_l_1D

class TimesBlockModel(nn.Module):
    """Multi-layer TimesBlock model for proper training"""
    def __init__(self, d_model, k, min_freq, max_period, num_layers, input_dim=None):
        super().__init__()
        self.num_layers = num_layers
        self.input_dim = input_dim
        
        # Input projection if needed
        if input_dim and input_dim != d_model:
            self.input_projection = nn.Linear(input_dim, d_model)
        else:
            self.input_projection = None
            
        # TimesBlock layers
        self.layers = nn.ModuleList([
            TimesBlock(d_model, k, min_freq, max_period) 
            for _ in range(num_layers)
        ])
        
        # Output projection if needed
        if input_dim and input_dim != d_model:
            self.output_projection = nn.Linear(d_model, input_dim)
        else:
            self.output_projection = None
    
    def forward(self, x, return_intermediate=False):
        # x shape: (batch, T, C)
        if x.dim() == 3:
            x = x.squeeze(0)  # Remove batch dimension: (T, C)
        
        # Input projection
        if self.input_projection:
            x = self.input_projection(x)
        
        outputs = [x]  # Store intermediate outputs
        
        # Pass through TimesBlock layers
        for layer in self.layers:
            x = layer(x)
            outputs.append(x)
        
        # Output projection
        if self.output_projection:
            x = self.output_projection(x)
            outputs[-1] = x
        
        if return_intermediate:
            return [out.unsqueeze(0) for out in outputs]  # Add batch dimension back
        else:
            return x.unsqueeze(0)  # Add batch dimension back


def train_timesblock_model(model, data, num_epochs=10, lr=0.01):
    """
    Trains the TimesBlock model using backpropagation to minimize MSE.
    
    Args:
        model (nn.Module): TimesBlockModel instance.
        data (torch.Tensor): Input data of shape (batch, T, C).
        num_epochs (int): Number of training epochs.
        lr (float): Learning rate for Adam optimizer.
    
    Returns:
        nn.Module: Trained model.
        list: Training losses.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    losses = []
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        scheduler.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
        
    return model, losses 

def compute_layer_mse(model, data):
    """
    Computes MSE per timestep for each TimesBlock layer's output in the input space (C channels).
    
    Args:
        model (nn.Module): Trained TimesBlockModel.
        data (torch.Tensor): Input data of shape (batch, T, C).
    
    Returns:
        list: MSE per timestep for each layer, shape (num_layers + 1, T).
    """
    model.eval()
    with torch.no_grad():
        layer_outputs = model(data, return_intermediate=True)
        layer_outputs = layer_outputs[1:]  # Skip the raw input
# List of (batch, T, C or d_model)
        C = data.shape[2]  # Number of channels in input data
        mse_per_layer = []
        for out in layer_outputs:
            if out.shape[2] != C:
                if hasattr(model, 'output_projection'):
                    out = model.output_projection(out)
                else:
                    raise ValueError("Model does not have output_projection to project layers")
            mse = torch.mean((out - data)**2, dim=2).squeeze(0).cpu().numpy()  # (T,)
            mse_per_layer.append(mse)
    return mse_per_layer

def calculate_gpd_threshold(errors, percentile=95, alpha=0.99):
    """
    Calculate EVA threshold using GPD fit.
    
    Args:
        errors (np.array): MSE errors per timestep.
        percentile (float): Percentile for preliminary threshold.
        alpha (float): Confidence level for EVA threshold.
    
    Returns:
        float: EVA threshold.
        np.array: Indices of anomalies.
    """
    threshold_prelim = scoreatpercentile(errors, percentile)
    excesses = errors[errors > threshold_prelim] - threshold_prelim
    if len(excesses) > 0:
        shape, loc, scale = genpareto.fit(excesses)
        threshold_eva = threshold_prelim + genpareto.ppf(alpha, shape, loc=loc, scale=scale)
    else:
        print("No excesses for Pareto: using percentile threshold")
        threshold_eva = threshold_prelim
    anomalies = errors > threshold_eva
    anomaly_indices = np.where(anomalies)[0]
    return threshold_eva, anomaly_indices



if __name__ == "__main__":
    # === CARGA Y NORMALIZACIÓN ===
    X_1D_P5 = np.load("monitoring_data_P1.npy")[:, 1:2]  # Solo un sensor
    #mean, std = X_1D_P5.mean(axis=0), X_1D_P5.std(axis=0)
    #X_1D_P5 = (X_1D_P5 - mean) / std

    num_time_steps, num_channels = X_1D_P5.shape

    # === SPLIT: TRAIN / TEST ===
    train_ratio = 0.7
    train_size = int(num_time_steps * train_ratio)
    X_train, X_test = X_1D_P5[:train_size], X_1D_P5[train_size:]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    

    # === PARÁMETROS DEL MODELO ===
    d_model = min(256, num_channels * 256)
    k = 3
    num_layers = 2
    min_freq = 1e-6
    max_period = min(168, train_size // 4)

    # === ENTRENAMIENTO ===
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32, requires_grad=True).unsqueeze(0)
    model = TimesBlockModel(d_model, k, min_freq, max_period, num_layers, input_dim=num_channels)
    model_train, losses = train_timesblock_model(model, X_train_tensor, num_epochs=10, lr=0.001)

    # === THRESHOLD CON EVA (TRAIN) ===
    mse_train_per_layer = compute_layer_mse(model_train, X_train_tensor)
    mse_train_final = mse_train_per_layer[-1]
    threshold_train, _ = calculate_gpd_threshold(mse_train_final)
    print(f"EVA Threshold (from training): {threshold_train:.4f}")

    # === ANOMALÍAS ENTRENAMIENTO ===
    anomalies_train_indices_per_layer = [
        np.where(mse > threshold_train)[0] for mse in mse_train_per_layer
    ]

    # === EVALUACIÓN TEST ===
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32, requires_grad=True).unsqueeze(0)
    mse_test_per_layer = compute_layer_mse(model_train, X_test_tensor)
    mse_test_final = mse_test_per_layer[-1]
    anomalies_test = np.where(mse_test_final > threshold_train)[0]
    timesteps_test = np.arange(train_size, num_time_steps)

    # === PLOT: MSE por capa (ENTRENAMIENTO) ===
    n_rows = int(np.ceil((num_layers + 1) / 3))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes = axes.flatten()
    for i, (mse, anomalies) in enumerate(zip(mse_train_per_layer, anomalies_train_indices_per_layer)):
        axes[i].plot(np.arange(train_size), mse, color='blue', label=f"Layer {i+1}")
        axes[i].set_title(f"TimesBlock Layer {i+1}", fontsize=10)
        axes[i].set_xlabel("Timestep")
        axes[i].set_ylabel("MSE")
        axes[i].legend()
        axes[i].grid(True)
    fig.suptitle("MSE per Timestep in Training Set", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


    # === PLOT FINAL: Subplots separados ===
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True, gridspec_kw={'hspace': 0.3})
    ax1.plot(timesteps_test, X_test[:, 0], label="Original Signal", color="#2ca02c", linewidth=1.5)
    ax1.set_ylabel("Normalized signal")
    ax1.set_title("Original Normalized Signal (Test Set)", fontsize=14, fontweight='bold')
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right")

    ax2.plot(timesteps_test, mse_test_final, label="Final Layer MSE", color="#1f77b4", linewidth=1.5)
    ax2.axhline(threshold_train, color="#d62728", linestyle="--", linewidth=1.5, label="Train EVA Threshold")
    ax2.scatter(timesteps_test[anomalies_test], mse_test_final[anomalies_test], color="black", marker="x", s=50, label="Anomalies")
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("MSE")
    ax2.set_title("Final Layer MSE with EVA Threshold and Detected Anomalies", fontsize=14, fontweight='bold')
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right")


    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    
