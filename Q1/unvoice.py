import librosa, numpy as np, matplotlib.pyplot as plt

signal, sr = librosa.load("audio.wav", sr=16000)
frames = librosa.util.frame(signal, frame_length=400, hop_length=160).T.copy()

energy = np.sum(frames**2, axis=1)
voiced = energy > np.mean(energy)

plt.plot(voiced)
plt.title("Voiced/Unvoiced")
plt.savefig("voiced.png")
plt.show()
