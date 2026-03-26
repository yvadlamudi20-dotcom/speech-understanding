import numpy as np

signal = np.sin(2*np.pi*50*np.linspace(0,1,500))

windows = {
    "Rect": np.ones(len(signal)),
    "Hamming": np.hamming(len(signal)),
    "Hanning": np.hanning(len(signal))
}

for name, win in windows.items():
    fft = np.abs(np.fft.fft(signal * win))
    leakage = np.sum(fft) - np.max(fft)
    print(name, "Leakage:", leakage)
