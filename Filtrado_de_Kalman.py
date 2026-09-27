import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, linalg
from scipy.signal import welch
import pandas as pd

class StructuralHealthMonitoringKalman:
    """
    Implementación del filtro de Kalman para monitoreo de salud estructural
    basado en el paper de Erazo et al. (2019)
    """
    
    def __init__(self, n_states=11, n_measurements=6, n_inputs=2):
        """
        Inicializa el filtro de Kalman
        
        Parámetros:
        - n_states: número de estados (posición, velocidad para cada DOF)
        - n_measurements: número de mediciones (temp, inclinación, desplazamiento)
        - n_inputs: número de entradas (excitación ambiental)
        """
        self.n = 11       # dimensión del estado
        self.m = 6  # número de mediciones
        self.r = 2        # número de entradas
        
        # Matrices del sistema (se identificarán desde datos)
        self.A = None    # Matriz de transición de estado
        self.B = None    # Matriz entrada-estado
        self.C = None    # Matriz estado-salida
        self.D = None    # Matriz transmisión directa
        
        # Matrices de covarianza
        self.Q = None    # Covarianza del ruido del proceso
        self.R = None    # Covarianza del ruido de medición
        self.N = None    # Covarianza del ruido de medición puro
        
        # Ganancia de Kalman y matriz de covarianza de error
        self.K = None    # Ganancia de Kalman
        self.P = None    # Matriz de covarianza del error de estado
        
        # Estados y residuos
        self.x_hat = None     # Estado estimado
        self.residuals = []   # Residuos (innovaciones)
        
    def identify_system_matrices(self, data, dt, temperature):
        """
        Identifica las matrices del sistema desde datos medidos
        Usa un enfoque simplificado de identificación en subspace
        
        Parámetros:
        - data: array (N, m) con mediciones [temp, inclinación, desplazamiento]
        - dt: paso de tiempo
        - temperature: temperatura de referencia
        """
        num_timesteps = data.shape[0]
        
        # Crear matrices Hankel para identificación en subespacios
        # Simplificación: usar un modelo AR para aproximar la dinámica
        
        # Matrices de estado discretas (ejemplo para un sistema de 4 DOF)
        # A representa la dinámica del sistema discretizado
        omega1, omega2, omega3, omega4 = 2*np.pi*4.5, 2*np.pi*7.1, 2*np.pi*18.1, 2*np.pi*0.229 # frecuencias naturales (Hz del paper)
        
        zeta = 0.05  # amortiguamiento del 5% (como en el paper)
        
        # Ajustar frecuencias por temperatura (ecuación 33 del paper)
        E_T = self._temperature_modulus_relationship(temperature)
        E_ref = self._temperature_modulus_relationship(20.0)  # temperatura de referencia
        freq_factor = np.sqrt(E_T / E_ref)
        
        omega1 *= freq_factor
        omega2 *= freq_factor
        omega3 *= freq_factor
        omega4 *= freq_factor
        
        # Sistema discreto para cada modo
        # Possible extension of the Kalman Filter, not implemented
        def discrete_mode(omega, zeta, dt):
            """Convierte un modo continuo a discreto"""
            wd = omega * np.sqrt(1 - zeta**2)
            exp_term = np.exp(-zeta * omega * dt)
            cos_term = np.cos(wd * dt)
            sin_term = np.sin(wd * dt)
            
            A_mode = exp_term * np.array([
                [cos_term + (zeta * omega / wd) * sin_term, (1/wd) * sin_term],
                [-omega**2/wd * sin_term, cos_term - (zeta * omega / wd) * sin_term]
            ])
            return A_mode
        
        # Matriz A para 4 DOF 
        self.A = np.zeros((11, 11))
        self.A[0, 0] = 1
        self.A[1, 1] = 1
        self.A[2,2] = 1 #constant bias

        #For each position-velocity pair a submatrix of size 2x2 is defined 
        def simple_A(dt):
            A_simple = np.array([[1,dt], [0,1]])
            
            return A_simple
        
        simple_block = simple_A(dt)

        A_theta_x = discrete_mode(omega1, zeta, dt)
        self.A[3:5, 3:5] = A_theta_x  # theta_x y dot_theta_x
        self.A[3:5, 3:5] = simple_block
    
        A_theta_y = discrete_mode(omega2, zeta, dt)
        self.A[5:7, 5:7] = A_theta_y # theta_y y dot_theta_y
        self.A[5:7, 5:7] = simple_block 

        A_d1 = discrete_mode(omega3, zeta, dt)
        self.A[7:9, 7:9] = A_d1  # d_1 y dot_d_1
        self.A[7:9, 7:9] = simple_block  # d_1 y dot_d_1
        

        A_d2 = discrete_mode(omega4, zeta, dt)
        self.A[9:11, 9:11] = A_d2  # d_2 y dot_d_2
        self.A[9:11, 9:11] = simple_block  
        
        
        print(self.A.shape)

 
        # Matriz B (entrada de excitación ambiental)
        self.B = np.zeros((11, 2))

        # Suponemos que T_norte afecta inclinación x y desplazamiento d1
        #Aqui hay varios parametros a ajustar de como B (ruido ambiental)
        #afecta a la velocidad de las variaciones de los sensores
        self.B[4, 0] = 0.1  # efecto de T_norte en velocidad theta_x
        self.B[8, 0] = 0.1  # efecto de T_norte en velocidad d1
    
        # Supongamos que T_sur afecta inclinación y desplazamiento d2
        self.B[6, 1] = 0.1  # efecto de T_sur en velocidad theta_y
        self.B[10,1] = 0.1   # efecto de T_sur en velocidad d2
        
        #Matriz B durante paso de tiempo (no está claro)
        self.B = self.B * dt 
        print(self.B.shape)

        
        # Matriz C (relaciona estados con mediciones de los sensores)
        #Mi matriz C va a ser parecida a la identidad ya que miden directamente esos
        #valores:  “extrae” directamente un estado
        # [temperatura, inclinación, desplazamiento]
        # Estados: [T1, T2, bias, theta_x, theta_y, d1, d2]
        n_states = 11

        # Número de mediciones (asumiendo que mides todas esas variables)
        n_measurements = 6  # 2 temps + 2 inclinaciones + 2 desplazamientos

        # Inicializar matriz C en ceros
        self.C = np.zeros((n_measurements, n_states))

        # Temperaturas T1 y T2
        self.C[0, 0] = 1  # y1 = T1
        self.C[1, 1] = 1  # y2 = T2

        # Inclinaciones theta_x y theta_y
        self.C[2, 3] = 1  # y3 = theta_x
        self.C[3, 5] = 1  # y4 = theta_y

        # Desplazamientos d1 y d2
        self.C[4, 7] = 1  # y5 = d1
        self.C[5, 9] = 1  # y6 = d2
        
        print(self.C.shape)



        # Matriz D (transmisión directa, usualmente cero)
        #La entrada (fuerza, excitación, ruido ambiental) afecta a los estados
        #internos del sistema (como velocidad o desplazamiento),
        #y no directamente a las mediciones de temperatura, inclinación o desplazamiento.
        self.D = np.zeros((self.m, self.r))
        
        # Covarianzas estimadas desde datos
        # CORREGIDO: Matriz Q más realista
        self.Q = np.eye(2)
        # Ruido menor para temperaturas (cambian lentamente)
        self.Q[0, 0] = 0.1  # T1
        self.Q[1, 1] = 0.1  # T2

        print(self.Q.shape) # Ajustar según nivel de excitación
        
        # Estimar R desde varianza de mediciones
        if data.shape[1] != 6:
            print(f"Advertencia: data tiene {data.shape[1]} columnas, esperadas 6")
            # Ajustar si los datos no tienen la estructura esperada
            measurement_vars = np.ones(6) * 0.1  # valores por defecto
        else:
            measurement_vars = np.var(data, axis=0)
            scales = np.array([0.1, 0.1, 0.1, 0.1, 0.01, 0.01])
            scaled_vars = measurement_vars * scales

        
        self.R = np.diag(scaled_vars)
        print(f"Matriz R shape: {self.R.shape}")
        print(self.R.shape)
        self.N = self.R.copy()
        
        print(f"Matrices identificadas para T = {temperature}°C")
        print(f"Frecuencias ajustadas: {omega1/(2*np.pi):.2f} Hz, {omega2/(2*np.pi):.2f} Hz, {omega3/(2*np.pi):.2f} Hz, {omega4/(2*np.pi):.2f} Hz")
        
    def _temperature_modulus_relationship(self, T_celsius):
        """
        Relación temperatura-módulo elástico del paper para el kalman 
        filter (Ecuación 33)
        Para acero estructural (reemplazar por analisis elementos finitos)
        """
        T = T_celsius
        e0 = 206e9  # GPa convertido a Pa
        e1 = -4.326e-2 * 1e9
        e2 = -3.502e-5 * 1e9
        e3 = -6.592e-8 * 1e9
        
        if T >= 0:
            E = e0 + e1*T + e2*T**2 + e3*T**3
        else:
            E = e0 + e1*T  # aproximación lineal para T < 0°C
            
        return E
        
  

    def kalman_yo(self, measurements, dt, max_iters=1000, tol=1e-6):
        """
        Método iterativo para calcular la matriz P y ganancia de Kalman (K).
        Resuelve la ecuación de Riccati de forma discreta.

        Parameters:
            -----------
            measurements : ndarray, shape (num_timesteps, m)
            Matriz de mediciones, donde m es el número de variables medidas.
       dt : float
            Intervalo de tiempo entre mediciones.
            u_k : ndarray, shape (num_timesteps, p), optional
            Entradas de control para cada paso de tiempo. Si es None, se asume cero.
            max_iters : int, optional
            Número máximo de iteraciones para la convergencia de P (default: 1000).
            tol : float, optional
            Tolerancia para la convergencia de P (default: 1e-6).

        Returns:
            --------
            K : ndarray
            Ganancia de Kalman.
            P : ndarray
            Matriz de covarianza a posteriori.
            states : ndarray, shape (num_timesteps, n)
            Estados estimados en cada paso de tiempo.
            residuals : ndarray, shape (num_timesteps, m)
            Residuos (innovaciones) en cada paso de tiempo.

        Notes:
            ------
            Requiere que la clase tenga los atributos:
        - self.A : Matriz de transición de estado (n x n).
        - self.B : Matriz de control (n x p).
        - self.C : Matriz de observación (m x n).
        - self.Q : Covarianza del ruido del proceso (p x p).
        - self.R : Covarianza del ruido de medición (m x m).
        - self.n : Dimensión del vector de estado.
        """
        # Inicialización
        P_priori = np.eye(self.n) * 0.1  # Covarianza inicial
        Q_process = self.B @ self.Q @ self.B.T
        num_timesteps = measurements.shape[0]
        states = np.zeros((num_timesteps, self.n))
        residuals = np.zeros((num_timesteps, measurements.shape[1]))
    
    
        np.random.seed(42)  # Para reproducibilidad
        u_k = np.random.normal(0, 0.1, (num_timesteps, 2))

        # Estado inicial tomando valores en tiempo t=0 (mejora significativa)
        x_hat = np.zeros(self.n)
        if num_timesteps > 0:
            x_hat[0] = measurements[0, 0]  # T1
            x_hat[1] = measurements[0, 1]  # T2
            x_hat[3] = measurements[0, 2]  # theta_x
            x_hat[5] = measurements[0, 3]  # theta_y
            x_hat[7] = measurements[0, 4]  # d1
            x_hat[9] = measurements[0, 5]  # d2
            if num_timesteps > 1:
                x_hat[4] = (measurements[1, 2] - measurements[0, 2]) / dt  # dot_theta_x
                x_hat[6] = (measurements[1, 3] - measurements[0, 3]) / dt  # dot_theta_y
                x_hat[8] = (measurements[1, 4] - measurements[0, 4]) / dt  # dot_d1
                x_hat[10] = (measurements[1, 5] - measurements[0, 5]) / dt  # dot_d2

        states[0] = x_hat  # Guardar estado inicial

        # Bucle principal del filtro de Kalman
        for k in range(num_timesteps):
            if k > 0:
                # Predicción
                x_hat_priori = self.A @ x_hat + self.B @ u_k[k-1]
                P_priori = self.A @ P_priori @ self.A.T + Q_process

                # Medición predicha
                y_hat = self.C @ x_hat_priori

                # Residuo (innovación)
                residual = measurements[k] - y_hat
                residuals[k] = residual

                # Covarianza de la innovación
                S = self.C @ P_priori @ self.C.T + self.R

                # Ganancia de Kalman (usando Cholesky para estabilidad numérica)
                try:
                    L = np.linalg.cholesky(S)
                    K = np.linalg.solve(L.T, np.linalg.solve(L, (P_priori @ self.C.T).T)).T
                except np.linalg.LinAlgError:
                        # Fallback a la inversa si Cholesky falla
                    K = P_priori @ self.C.T @ np.linalg.inv(S)

                # Actualización
                x_hat = x_hat_priori + K @ residual
                P_posteriori = (np.eye(self.n) - K @ self.C) @ P_priori

                # Guardar resultados
                states[k] = x_hat
                P_priori = P_posteriori  # Actualizar P para el siguiente paso


        # Asignar resultados finales
        self.P = P_priori
        self.K = K

        print("Kalman gain y P calculados con método iterativo.")
        return states, residuals
            

   
    def compute_residual_psd(self, residuals, fs=1/600, sensor_idx=0):
        """
        Calcula la densidad espectral de potencia (PSD) de los residuos
        
        Parámetros:
        - residuals: residuos del filtro
        - fs: frecuencia de muestreo
        - sensor_idx: índice del sensor para análisis
        
        Retorna:
        - frequencies: frecuencias
        - psd: densidad espectral de potencia
        """
        if residuals.ndim > 1:
            residual_signal = residuals[:, sensor_idx]
        else:
            residual_signal = residuals
            
        nperseg = min(1024, len(residual_signal) // 8)
        noverlap = int(nperseg * 0.65)


        # Usar método de Welch para estimar PSD (como menciona el paper)
        frequencies, psd = welch(residual_signal, fs=fs, 
                               nperseg=nperseg,
                               noverlap=noverlap)
        
        return frequencies, psd
    
    def whiteness_test(self, residuals, sensor_idx=0):
        """
        Test de blancura Bayesiano (Ecuación 29 del paper)
        W_T(e) = número de cambios de signo en e
        
        Parámetros:
        - residuals: residuos del filtro
        - sensor_idx: índice del sensor
        
        Retorna:
        - WT: estadístico de blancura
        - is_white: True si el residuo es blanco
        """
        if residuals.ndim > 1:
            e = residuals[:, sensor_idx]
        else:
            e = residuals
            
        # Contar cambios de signo (Ecuación 29)
        sign_changes = np.sum(np.diff(np.sign(e)) != 0)
        WT = sign_changes
        
        # Criterio simple: para señal blanca, esperamos muchos cambios de signo
        # Aproximadamente N/2 cambios para N muestras
        expected_changes = len(e) * 0.4
        print(f"expected changes: {expected_changes}") # umbral ajustable
        is_white = WT > expected_changes
        
        return WT, is_white
    
    def damage_index(self, frequencies, psd, freq_range=(0, 50)):
        """
        Calcula el índice de daño normalizado (Ecuación 32 del paper)
        D = 1 - sqrt(3) * (ω2+ω1)/(ω2-ω1) * [λ2/λ0 - (λ1/λ0)²]^(1/2) / (λ1/λ0)
        
        Parámetros:
        - frequencies: vector de frecuencias
        - psd: densidad espectral de potencia
        - freq_range: rango de frecuencias para análisis
        
        Retorna:
        - D: índice de daño [0, 1]
        """
        # Filtrar frecuencias en el rango especificado
        mask = (frequencies >= freq_range[0]) & (frequencies <= freq_range[1])
        freq_filtered = frequencies[mask]
        freq_filtered_omega =  2 * np.pi * freq_filtered
        psd_filtered = psd[mask] *2*np.pi
        
        if len(freq_filtered) == 0:
            return 0.0
        
        # Calcular momentos espectrales (Ecuación 31)
        df = np.mean(np.diff(freq_filtered)) if len(freq_filtered) > 1 else 1.0

        
        # λ0 = ∫ S(ω) dω (momento 0)
        lambda_0 = np.trapz(psd_filtered, dx=df)
        
        # λ1 = ∫ ω * S(ω) dω (momento 1)
        lambda_1 = np.trapz(freq_filtered * psd_filtered, dx=df)
        
        # λ2 = ∫ ω² * S(ω) dω (momento 2)
        lambda_2 = np.trapz(freq_filtered**2 * psd_filtered, dx=df)
        
        if lambda_0 == 0:
            return 0.0
        
        # Implementar ecuación 32
        omega_1, omega_2 = freq_range[0], freq_range[1]
        
        try:
            term1 = lambda_2 / lambda_0 - (lambda_1 / lambda_0)**2
            if term1 <= 0:
                return 0.0
                
            term2 = np.sqrt(term1) / (lambda_1 / lambda_0)
            term3 = np.sqrt(3) * (omega_2 + omega_1) / (omega_2 - omega_1)
            
            D = 1 - term3 * term2
            
            # Asegurar que D esté en [0, 1]
            D = max(0.0, min(1.0, D))
            
        except (ZeroDivisionError, ValueError):
            print("Coeficiente D no está bien calculado")
            D = 0.0
            
        return D

  


# Ejemplo de uso
if __name__ == "__main__":
    
    
    #Coger datos de P5 y crear vector de estados
    
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
    time_vector = common_index.to_numpy()  # Timestamps absolutos
    time_relative_minutes = ((common_index - common_index[0])/60).total_seconds().to_numpy()  # Minutos desde el inicio
    
    
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


    P1_sensor_targets = {
        'inc_X': inc_data['P1 INC AXIS X AI C-3'].to_numpy().reshape(-1,1),
        'inc_Y': inc_data['P1 INC AXIS  Y AI C-4'].to_numpy().reshape(-1,1),
        'LVDT_transv': LVDT_data['P1 LVDT TRANSV AI C-5'].to_numpy().reshape(-1,1),
        'LVDT_fixed': LVDT_data['P1 LVDT FIJO AI D-1'].to_numpy().reshape(-1,1)
    }

    P5_sensor_targets = {
        'inc_X': inc_data['P5 INC AXIS X AI J-3'].to_numpy().reshape(-1,1),
        'inc_Y': inc_data['P5 INC AXIS Y AI J-4'].to_numpy().reshape(-1,1),
        'LVDT_transv': LVDT_data['P5 LVDT TRANSV AI J-5'].to_numpy().reshape(-1,1),
        'LVDT_fixed': LVDT_data['P5 LVDT FIJO AI I-1'].to_numpy().reshape(-1,1)
    }

    P9_sensor_targets = {
        'inc_X': inc_data['P9 INC AXIS X AI H-3'].to_numpy().reshape(-1,1),
        'inc_Y': inc_data['P9 INC AXIS Y AI H-4'].to_numpy().reshape(-1,1),
        'LVDT_transv': LVDT_data['P9 LVDT TRANSV AI H-5'].to_numpy().reshape(-1,1),
        'LVDT_fixed': LVDT_data['P9 LVDT FIJO AI G-1'].to_numpy().reshape(-1,1)
    }

    P15_sensor_targets = {
        'inc_X': inc_data['P15 INC AXIS X AI E-3'].to_numpy().reshape(-1,1),
        'inc_Y': inc_data['P15 INC AXIS Y  AI E-4'].to_numpy().reshape(-1,1),
        'LVDT_transv': LVDT_data['P15 LVDT  TRANSV AI E-5'].to_numpy().reshape(-1,1),
        'LVDT_fixed': LVDT_data['P15 LVDT FIJO  AI F-1'].to_numpy().reshape(-1,1)
    }


    #Stack temperatura [T_north, T_south]
    stack_temp_P1 = np.column_stack([feature_temp_P1_north, feature_temp_P1_south])
    stack_temp_P5 = np.column_stack([feature_temp_P5_north, feature_temp_P5_south])
    stack_temp_P9 = np.column_stack([feature_temp_P9_north, feature_temp_P9_south])
    stack_temp_P15 = np.column_stack([feature_temp_P15_north, feature_temp_P15_south])
    
    #Stack the dos inclinaciones y dos desplazamientos 
    stack_inc_disp_P1 = np.column_stack([target.flatten() for target in P1_sensor_targets.values()])
    stack_inc_disp_P5 = np.column_stack([target.flatten() for target in P5_sensor_targets.values()])
    stack_inc_disp_P9 = np.column_stack([target.flatten() for target in P9_sensor_targets.values()])
    stack_inc_disp_P15 = np.column_stack([target.flatten() for target in P15_sensor_targets.values()])
    
    #Stack de todos los observables del sistema [T_north, T_south, inc_x, inc_y, d_1, d_2]
    monitoring_data_P1 = np.column_stack([stack_temp_P1, stack_inc_disp_P1])
    monitoring_data_P5 = np.column_stack([stack_temp_P5, stack_inc_disp_P5])
    monitoring_data_P9 = np.column_stack([stack_temp_P9, stack_inc_disp_P9])
    monitoring_data_P15 = np.column_stack([stack_temp_P15, stack_inc_disp_P15])

    dt = 1/600


    # Crear filtro de Kalman para P1
    shm_filter = StructuralHealthMonitoringKalman()
    # Identificar matrices del sistema
    shm_filter.identify_system_matrices(monitoring_data_P1, dt, temperature=20)
    # Calcular ganancia de Kalman para P1
    states_P1, residuals_P1 = shm_filter.kalman_yo(monitoring_data_P1, dt)


    # Crear filtro de Kalman para P5
    shm_filter = StructuralHealthMonitoringKalman()
    # Identificar matrices del sistema
    shm_filter.identify_system_matrices(monitoring_data_P5, dt, temperature=20)
    # Calcular ganancia de Kalman para P5
    states_P5, residuals_P5 = shm_filter.kalman_yo(monitoring_data_P5, dt)


    # Crear filtro de Kalman para P9
    shm_filter = StructuralHealthMonitoringKalman()
    # Identificar matrices del sistema
    shm_filter.identify_system_matrices(monitoring_data_P9, dt, temperature=20)
    # Calcular ganancia de Kalman para P5
    states_P9, residuals_P9 = shm_filter.kalman_yo(monitoring_data_P9, dt)


    # Crear filtro de Kalman para P15
    shm_filter = StructuralHealthMonitoringKalman()
    # Identificar matrices del sistema
    shm_filter.identify_system_matrices(monitoring_data_P15, dt, temperature=20)
    # Calcular ganancia de Kalman para P5
    states_P15, residuals_P15 = shm_filter.kalman_yo(monitoring_data_P15, dt)
    
    print(f"Ganancia de Kalman K:\n{shm_filter.K}")
    
    # Aplicar filtro a datos de P5
    print("\nProcesando datos de P5...")
    

    # Análisis de resultados
    fs = 1/dt
    
    # PSD de residuos 
    freq_P1, psd_P1 = shm_filter.compute_residual_psd(residuals_P1, fs, sensor_idx=3)
    freq_P5, psd_P5 = shm_filter.compute_residual_psd(residuals_P5, fs, sensor_idx=3)
    freq_P9, psd_P9 = shm_filter.compute_residual_psd(residuals_P9, fs, sensor_idx=3)
    freq_P15, psd_P15 = shm_filter.compute_residual_psd(residuals_P15, fs, sensor_idx=3)



    
    # Test de blancura
    wt_P1, is_white_h_P1 = shm_filter.whiteness_test(residuals_P1, sensor_idx=3)
    wt_P5, is_white_h_P5 = shm_filter.whiteness_test(residuals_P5, sensor_idx=3)
    wt_P9, is_white_h_P9 = shm_filter.whiteness_test(residuals_P9, sensor_idx=3)
    wt_P15, is_white_h_P15 = shm_filter.whiteness_test(residuals_P15, sensor_idx=3)

    # Índices de daño
    damage_idx_P1 = shm_filter.damage_index(freq_P1, psd_P1, freq_range=(10, 30))
    damage_idx_P5 = shm_filter.damage_index(freq_P5, psd_P5, freq_range=(10, 30))
    damage_idx_P9 = shm_filter.damage_index(freq_P9, psd_P9, freq_range=(10, 30))
    damage_idx_P15 = shm_filter.damage_index(freq_P15, psd_P15, freq_range=(10, 30))

    
    print(f"\n=== RESULTADOS ===")
    print(f"Test de blancura: WT = {wt_P1}, Blanco: {is_white_h_P1}")
    print(f"Test de blancura: WT = {wt_P5}, Blanco: {is_white_h_P5}")
    print(f"Test de blancura: WT = {wt_P9}, Blanco: {is_white_h_P9}")
    print(f"Test de blancura: WT = {wt_P15}, Blanco: {is_white_h_P15}")
    print(f"Índice de daño (saludable): D = {damage_idx_P1:.4f}")
    print(f"Índice de daño (saludable): D = {damage_idx_P5:.4f}")
    print(f"Índice de daño (saludable): D = {damage_idx_P9:.4f}")
    print(f"Índice de daño (saludable): D = {damage_idx_P15:.4f}")

    
    np.save("monitoring_data_P1.npy", monitoring_data_P1)
    np.save("monitoring_data_P5.npy", monitoring_data_P5)
    np.save("states.npy", states_P1)

# Plot: Temperature (North Sensor)
    plt.figure(figsize=(10, 4))
    plt.plot(time_relative_minutes, monitoring_data_P5[:, 0], label='Measured Temp. North')
    plt.plot(time_relative_minutes, states_P5[:, 0], label='Estimated Temp. North')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Temperature (°C)')
    plt.title('North Temperature: Measured vs Estimated')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plot: Temperature (South Sensor)
    plt.figure(figsize=(10, 4))
    plt.plot(time_relative_minutes, monitoring_data_P5[:, 1], label='Measured Temp. South')
    plt.plot(time_relative_minutes, states_P5[:, 1], label='Estimated Temp. South')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Temperature (°C)')
    plt.title('South Temperature: Measured vs Estimated')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot: Inclination X
    plt.figure(figsize=(10, 4))
    plt.plot(time_relative_minutes, monitoring_data_P5[:, 2], label='Measured Inclination X')
    plt.plot(time_relative_minutes, states_P5[:, 3], label='Estimated Inclination X')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Inclination (rad)')
    plt.title('Inclination X: Measured vs Estimated')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot: Inclination Y
    plt.figure(figsize=(10, 4))
    plt.plot(time_relative_minutes, monitoring_data_P5[:, 3], label='Sensor Measurement (Inclination Y)')
    plt.plot(time_relative_minutes, states_P5[:, 5], label="Kalman Estimate (Inclination Y)")
    plt.xlabel('Time (minutes)')
    plt.ylabel('Inclination (rad)')
    plt.title('Sensor vs Kalman Estimate: P5')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot: LVDT Transversal
    plt.figure(figsize=(10, 4))
    plt.plot(time_relative_minutes, monitoring_data_P5[:, 4], label='Measured Displacement 1')
    plt.plot(time_relative_minutes, states_P5[:, 7], label='Estimated Displacement 1')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Displacement (m)')
    plt.title('LVDT Transversal: Measured vs Estimated')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot: LVDT Fixed
    plt.figure(figsize=(10, 4))
    plt.plot(time_relative_minutes, monitoring_data_P5[:, 5], label='Measured Displacement 2')
    plt.plot(time_relative_minutes, states_P5[:, 9], label='Estimated Displacement 2')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Displacement (m)')
    plt.title('LVDT Fixed: Measured vs Estimated')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot: Residuals (example: sensor_idx = 3)
    plt.figure(figsize=(10, 4))
    plt.plot(residuals_P5[:, 3])
    plt.xlabel('Time Step')
    plt.ylabel('Residual')
    plt.title('Residuals (Sensor 3)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot: PSD of Residuals
    plt.figure(figsize=(10, 4))
    plt.semilogy(freq_P1, psd_P1, label='P1')
    plt.semilogy(freq_P5, psd_P5, label='P5')
    plt.semilogy(freq_P9, psd_P9, label='P9')
    plt.semilogy(freq_P15, psd_P15, label='P15')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD')
    plt.title('Power Spectral Density of Residuals')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot: Damage Index
    plt.figure(figsize=(6, 4))
    plt.bar(['P5'], [damage_idx_P5], color='green', alpha=0.7)
    plt.ylabel('Damage Index D')
    plt.ylim(0, 1)
    plt.title('Damage Index')
    plt.tight_layout()
    plt.show()


    plt.figure(figsize=(6, 4))
    error = monitoring_data_P5[:, 3] - states_P5[:, 5]
    plt.plot(error)
    plt.title('Error entre señal observada y estimada')
    plt.show()

    
    print("Matriz R (ruido de medición):\n", shm_filter.R)
    print("Matriz Q (ruido del proceso):\n", shm_filter.Q)
    print("Ganancia de Kalman K:\n", shm_filter.K)
    
    
    # Valores de D (coeficiente daño estructural) que han sido encontrados antes
    D1, D2, D3, D4 = 0.0485, 0.0000, 0.0634, 0.0770
    
    # Crear la figura
    plt.figure(figsize=(10, 4))
    print("Matriz R (ruido de medición):\n", shm_filter.R)
    print("Matriz Q (ruido del proceso):\n", shm_filter.Q)
    print("Ganancia de Kalman K:\n", shm_filter.K)


    # Valores de D
    D1, D2, D3, D4 = 0.0485, 0.0000, 0.0634, 0.0770
    
    # Crear la figura
    plt.figure(figsize=(10, 4))
    
    # Graficar las PSDs con D en la leyenda
    plt.semilogy(freq_P1, psd_P1, label=f'P1: $D={D1:.4f}$')
    plt.semilogy(freq_P5, psd_P5, label=f'P5: $D={D2:.4f}$')
    plt.semilogy(freq_P9, psd_P9, label=f'P9: $D={D3:.4f}$')
    plt.semilogy(freq_P15, psd_P15, label=f'P15: $D={D4:.4f}$')
    
    # Líneas verticales en 10Hz y 30Hz
    plt.axvline(x=10, color='black', linestyle='--', linewidth=1, label='10 Hz')
    plt.axvline(x=30, color='black', linestyle='--', linewidth=1, label='30 Hz')
    
    # Etiquetas y título
    plt.xlabel('Frequency (Hz)')
    plt.ylabel(r'PSD ($\mathcal{S}_{ee}$)')
    plt.title('Power Spectral Density of Residuals')
    
    # Leyenda y formato
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    
    # Mostrar
    plt.show()
        

    

