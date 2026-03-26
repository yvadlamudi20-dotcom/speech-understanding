from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import librosa, torch

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

signal, sr = librosa.load("audio.wav", sr=16000)

input_values = processor(signal, return_tensors="pt", sampling_rate=sr).input_values
logits = model(input_values).logits

pred_ids = torch.argmax(logits, dim=-1)
text = processor.decode(pred_ids[0])

print("Predicted Text:", text)
