import os
import tensorflow as tf
from datasets import load_dataset
import librosa
import numpy as np
import sentencepiece as spm

# Load SentencePiece model
sp = spm.SentencePieceProcessor()
sp.load("tkn.model")

VOCAB_SIZE = sp.vocab_size()

def compute_mel_spectrogram(audio_array, sample_rate):
    """
    Compute the Mel-Spectrogram in decibel scale for a given audio array and sample rate.
    """
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio_array,
        sr=sample_rate,
        n_mels=128,
        fmax=8000
    )
    return librosa.power_to_db(mel_spectrogram, ref=np.max)

def _bytes_feature(value):
    """
    Returns a bytes_list from a string/byte.
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    """
    Returns an int64_list from a single int or list of ints.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

# Adjust this path to point to your Samsung 860 EVO drive.
OUTPUT_DIR = "/Volumes/Samsung 860 EVO/spectrograms"  
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Load dataset in streaming mode to avoid full download
ds_stream = load_dataset("tarteel-ai/EA-DI", split="train", streaming=True)

# 2. Shard parameters
SHARD_SIZE = 1000  
shard_index = 52
sample_index = 0
writer = None

try:
    for sample in ds_stream:
        # If this is the first sample of a shard, open a new TFRecordWriter
        if sample_index % SHARD_SIZE == 0:
            # Close the old writer if it exists
            if writer:
                writer.close()
            tfrecord_filename = os.path.join(
                OUTPUT_DIR, 
                f"train_melspec_{shard_index:05d}.tfrecord" #Output filename
            )
            writer = tf.io.TFRecordWriter(tfrecord_filename)
            print(f"Opened new TFRecord file: {tfrecord_filename}")
            shard_index += 1

        audio_array = sample["audio"]["array"]
        sample_rate = sample["audio"]["sampling_rate"]
        transcript = sample["text"]  # Extract transcript from dataset

        # Compute the spectrogram
        mel_spec_db = compute_mel_spectrogram(audio_array, sample_rate)

        # Convert to float32
        mel_spec_db = mel_spec_db.astype(np.float32)

        # Flatten to a 1D byte string
        mel_spec_bytes = mel_spec_db.tobytes()
        height, width = mel_spec_db.shape

        # Encode transcript using SentencePiece
        encoded_transcript = sp.encode(transcript, out_type=int)

        # Build a tf.train.Example
        feature = {
            "spectrogram": _bytes_feature(mel_spec_bytes),
            "height": _int64_feature(height),
            "width": _int64_feature(width),
            "transcript": _int64_feature(encoded_transcript)  # Store tokenized transcript
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize and write to the current shard
        writer.write(example.SerializeToString())

        sample_index += 1
finally:
    # Make sure the last writer is closed
    if writer:
        writer.close()

print(f"Done! Processed {sample_index} samples into {shard_index} shard(s).")
