import tensorflow as tf
import matplotlib.pyplot as plt
import librosa.display

def parse_tfrecord(example_proto):
    """
    Parse a single TF Example containing a flattened spectrogram.
    """
    feature_description = {
        "spectrogram": tf.io.FixedLenFeature([], tf.string),
        "height": tf.io.FixedLenFeature([], tf.int64),
        "width": tf.io.FixedLenFeature([], tf.int64),
        # If you stored 'sr' or anything else, add them here.
    }
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)

    # Decode the raw spectrogram bytes
    spectrogram_raw = parsed_example["spectrogram"]
    height = parsed_example["height"]
    width = parsed_example["width"]

    # Convert raw bytes to float32 tensor
    spectrogram = tf.io.decode_raw(spectrogram_raw, tf.float32)
    spectrogram = tf.reshape(spectrogram, (height, width, 1))

    return spectrogram

# 1. Load the dataset from the TFRecord
dataset = tf.data.TFRecordDataset("train_melspec.tfrecord")

# 2. Map to parse function
dataset = dataset.map(parse_tfrecord)

# 3. Iterate through a few spectrograms and display them
for idx, spectrogram in enumerate(dataset.take(2)):  # show first 2 as an example
    spectrogram_np = spectrogram.numpy().squeeze()   # shape: [height, width]

    plt.figure(figsize=(10, 4))
    # Option A: Use librosa.display (requires sr if you want time/mel axes)
    # Here, we don't have 'sr' stored. So we can just do a raw specshow.
    librosa.display.specshow(spectrogram_np, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Spectrogram #{idx}')
    plt.tight_layout()
    plt.show()

    # Option B: Or simply display as an image:
    # plt.imshow(spectrogram_np, aspect='auto', origin='lower')
    # plt.title(f'Spectrogram #{idx} (raw image)')
    # plt.colorbar()
    # plt.show()
