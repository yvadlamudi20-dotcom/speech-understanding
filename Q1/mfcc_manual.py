import numpy as np, librosa, matplotlib.pyplot as plt
from scipy.fftpack import dct

signal, sr = librosa.load("audio.wav", sr=16000)

emphasized = np.append(signal[0], signal[1:] - 0.97 * signal[:-1])
frames = librosa.util.frame(emphasized, frame_length=400, hop_length=160).T.copy()
frames *= np.hamming(400)

fft = np.abs(np.fft.rfft(frames, 512))
power = (fft**2)/512

mel = librosa.filters.mel(sr=sr, n_fft=512, n_mels=26)
mel_energy = np.dot(power, mel.T)

log_mel = np.log(mel_energy + 1e-8)
mfcc = dct(log_mel, axis=1, norm='ortho')[:, :13]

plt.imshow(mfcc.T, aspect='auto')
plt.title("MFCC")
plt.savefig("mfcc.png")
plt.show()
