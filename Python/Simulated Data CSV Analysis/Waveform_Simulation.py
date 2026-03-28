import numpy as np
import matplotlib.pyplot as plt

def simulate_SEDO(t, V0, tau, fc, theta=0, td=0):
    """
    Single Exponentially Damping Oscillation (SEDO) Pulse.
    
    Parameters:
    t   : Time array (seconds)
    V0  : Peak amplitude
    tau : Decay time constant
    fc  : Center frequency of oscillation (Hz)
    theta : Phase shift (radians)
    td : Time delay (seconds)
    """
    signal = V0 * np.exp(-(t-td) / tau) * np.sin(2 * np.pi * fc * (t-td) + theta)
    signal[(t-td)<0]=0
    return signal

def simulate_DED(t, V0, alpha, beta, td=0):
    """
    Double Exponential Damping (DED) Pulse (without oscillation).
    
    Parameters:
    t     : Time array (seconds)
    V0    : Amplitude scaling factor
    alpha : Decay parameter (controls tail)
    beta  : Rise parameter (controls rise time)
    Note: Typically alpha < beta
    """
    signal = V0 * (np.exp(-alpha * (t-td)) - np.exp(-beta * (t-td)))
    signal[(t-td)<0]=0
    return signal

def simulate_DEDO(t, V0, alpha, beta, fc, theta=0, td=0):
    """
    Double Exponentially Damping Oscillation (DEDO) Pulse.
    
    Parameters:
    t     : Time array (seconds)
    V0    : Amplitude scaling factor
    alpha : Decay parameter
    beta  : Rise parameter
    fc    : Center frequency (Hz)
    theta : Phase shift (radians)
    """
    signal = V0 * (np.exp(-alpha * (t-td)) - np.exp(-beta * (t-td))) * np.sin(2 * np.pi * fc * (t-td) + theta)
    signal[(t-td)<0]=0
    return signal

def simulate_SMG(t, A, t0, tau, fc, td=0):
    """
    Sinusoidally Modulated Gaussian (SMG) Pulse.

    Parameters:
    t   : Time array (seconds)
    A   : Amplitude
    t0  : Pulse center time
    tau : Pulse width parameter
    fc  : Carrier frequency (Hz)
    """
    exponent = -((t - t0 - td)**2) / (2 * (tau**2))
    signal = A * np.exp(exponent) * np.sin(2 * np.pi * fc * (t-t0-td))
    signal[(t-td)<0]=0
    return signal

if __name__ == "__main__":
    
    # --- Simulation Parameters ---
    # Time settings: 0 to 2 microseconds, with high resolution (1 ns step)
    fs = 1e9  # 1 GHz sampling rate
    duration = 2e-6 # 2 microseconds
    t = np.linspace(0, duration, int(duration * fs))

    # 1. SEDO Parameters
    V0_sedo = 1.0
    tau_sedo = 0.4e-6
    fc_sedo = 15e6 # 15 MHz

    # 2. DED Parameters
    V0_ded = 2.0 # Adjusted to normalize peak roughly to 1
    alpha_ded = 2e6 
    beta_ded = 10e6 

    # 3. DEDO Parameters
    V0_dedo = 2.0
    alpha_dedo = 1.5e6
    beta_dedo = 15e6
    fc_dedo = 15e6

    # 4. SMG Parameters
    A_smg = 1.0
    t0_smg = 1.0e-6 # Centered at 1 us
    tau_smg = 0.2e-6
    fc_smg = 20e6

    # --- Generate Signals ---
    y_sedo = simulate_SEDO(t, V0_sedo, tau_sedo, fc_sedo, 0, 2e-7)
    y_ded = simulate_DED(t, V0_ded, alpha_ded, beta_ded, 2e-7)
    y_dedo = simulate_DEDO(t, V0_dedo, alpha_dedo, beta_dedo, fc_dedo)
    y_smg = simulate_SMG(t, A_smg, t0_smg, tau_smg, fc_smg)

    # --- Plotting to resemble Fig. 1 [cite: 1711] ---
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    # Plot SEDO
    axes[0].plot(t * 1e6, y_sedo, color='tab:blue')
    axes[0].set_title('a) SEDO Pulse [Eq. 1]')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)

    # Plot DED
    axes[1].plot(t * 1e6, y_ded, color='tab:orange')
    axes[1].set_title('b) DED Pulse [Eq. 2]')
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(True, alpha=0.3)

    # Plot DEDO
    axes[2].plot(t * 1e6, y_dedo, color='tab:green')
    axes[2].set_title('c) DEDO Pulse [Eq. 3]')
    axes[2].set_ylabel('Amplitude')
    axes[2].grid(True, alpha=0.3)

    # Plot SMG
    axes[3].plot(t * 1e6, y_smg, color='tab:red')
    axes[3].set_title('e) SMG Pulse [Eq. 5]') # Labeled 'e' in Fig 1, 'd' is Heidler (skipped)
    axes[3].set_ylabel('Amplitude')
    axes[3].set_xlabel('Time (µs)')
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()