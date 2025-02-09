from tensorflow.keras.models import load_model
import numpy as np
import sentencepiece as spm


sp = spm.SentencePieceProcessor()
sp.load("tkn.model")

VOCAB_SIZE = sp.vocab_size()

# Load trained model
model = load_model("cnn_bi_gru.h5")

# Function to preprocess new audio into Mel-spectrogram
def preprocess_audio(audio_path):
    y, sr = librosa.load(audio_path, sr=22050)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec = tf.image.resize(mel_spec, (128, 129)).numpy()
    mel_spec = (mel_spec - np.min(mel_spec)) / (np.max(mel_spec) - np.min(mel_spec))
    mel_spec = np.expand_dims(mel_spec, axis=(0, -1))  # Shape: (1, 128, 129, 1)
    return mel_spec


# Load trained model
model = load_model("cnn_bi_gru_seq2seq.h5")

# Process new audio
audio_path = "new_audio.wav"
mel_spec = preprocess_audio(audio_path)

# Predict token probabilities
predictions = model.predict(mel_spec)

# Convert probabilities to token sequence
predicted_token_ids = np.argmax(predictions, axis=-1)[0]  # Get most likely tokens

# Decode token sequence to text
predicted_text = sp.decode(predicted_token_ids)
print("Predicted Transcript:", predicted_text)

