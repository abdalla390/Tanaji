import tensorflow as tf
import numpy as np
import librosa
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, GRU, Bidirectional, Reshape
from tensorflow.keras.models import Sequential
import sentencepiece as spm

# Load the trained tokenizer model
sp = spm.SentencePieceProcessor()
sp.load("Tokenizer/TnjTknz.model")

VOCAB_SIZE = sp.vocab_size()

# --- 1. Function to Parse TFRecord Files ---
def _parse_tfrecord_fn(example_proto):
    feature_description = {
        'mel_spec': tf.io.FixedLenFeature([128*129], tf.float32),  # 128x129 Mel-spectrogram
        'tokens': tf.io.VarLenFeature(tf.int64)  
    }
    example = tf.io.parse_single_example(example_proto, feature_description)
    mel_spec = tf.reshape(example['mel_spec'], (128, 129, 1))  # Reshape to correct dimensions
    token_id = tf.one_hot(example['tokens'], depth=sp.vocab_size()) 
    return mel_spec, token_id

# --- 2. Load TFRecord Dataset ---
def load_tfrecord_dataset(tfrecord_path, batch_size=32):
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(_parse_tfrecord_fn)
    dataset = dataset.shuffle(1000).padded_batch(batch_size, padded_shapes=([128, 129, 1], [None]))
    return dataset

# --- 3. Build CNN+Bi-GRU Model ---
def build_model():
    model = Sequential([
        # CNN Layers
        Conv2D(128, (5, 5), activation='relu', padding='same', input_shape=(128, 129, 1)),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(128, (5, 5), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Reshape for RNN Input
        Reshape((32, 129 * 2)),  # Adjusting shape for GRU input
        
        # Bi-GRU Layers
        Bidirectional(GRU(128, return_sequences=True)),
        Bidirectional(GRU(64, return_sequences=True)),
        
        # Dense Layers
        Dense(256, activation='relu'),
        Dropout(0.25),

        # Output: Sequence of tokens
        Dense(VOCAB_SIZE, activation='softmax') 
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# --- 4. Training the Model ---

if __name__ == "__main__":
    
    number_of_tfrecord = 10  

    for shard_index in range(number_of_tfrecord):
        train_dataset = load_tfrecord_dataset(
            f"spectrograms/train_melspec_{shard_index:05d}.tfrecord"
        )
        val_dataset = load_tfrecord_dataset(
            f"spectrograms/val_melspec_{shard_index:05d}.tfrecord"
        )

        model = build_model()
        model.summary()

        model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=50
        )
    
    # Save model after training
    model.save("cnn_bi_gru.h5")
