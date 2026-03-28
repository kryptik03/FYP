import numpy as np
import matplotlib.pyplot as plt

# Parameters
A = 1.0  # Amplitude
alpha = 5e8  # Attenuation coefficient in 1/m
f = 800e6  # Frequency in Hz
# typical PD frequency ranges between 200 MHz to 3GHz

tdelay = 20.1e-9  # Time delay in seconds
Fs = 10e9  # Sampling frequency in Hz
T = 100e-9  # Total time duration in seconds

t = np.arange(0, T+1/Fs, 1 / Fs)  # Time vector

s = np.zeros(len(t))
s[t >= tdelay] = A * np.exp(-alpha * (t[t >= tdelay] - tdelay)) * np.sin(2 * np.pi * f * (t[t >= tdelay] - tdelay))
s_fd = np.fft.fft(s) # frequency domain signal
s_fd_freqs = np.fft.fftfreq(len(s), 1/Fs) # frequency vector

s_noisy = s + np.random.normal(0, 0.05, size=len(s))

s_noisy_fd = np.fft.fft(s_noisy) # frequency domain signal

fig, ax = plt.subplots(2,1, figsize=(20,5))
ax[0].plot(t, s_noisy, label='Noisy Signal')
ax[1].plot(np.fft.fftshift(s_fd_freqs), np.fft.fftshift(np.abs(s_noisy_fd)), label='Frequency Domain Signal')
ax[0].plot(t, s, label='Clean Signal')
ax[1].plot(np.fft.fftshift(s_fd_freqs), np.fft.fftshift(np.abs(s_fd)), label='Frequency Domain Signal')
plt.show()

