# ===== Install Necessary Libraries =====
# We use 'try-except' to avoid reinstalling if already present
try:
    import mediapipe_model_maker
except ImportError:
    print("Installing MediaPipe Model Maker...")
    !pip install mediapipe-model-maker
    !pip install tf_keras

import os
import sys
import shutil

# ===== Set Environment Variables =====
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# ===== Ensure Compatible Package Exists =====
try:
    import tf_keras
except ImportError:
    print("Re-installing tf_keras just in case...")
    !pip install tf_keras
    import tf_keras

# ===== Fix: Forcefully Replace Keras Core =====
import tensorflow as tf

# Replace 'keras' module in Python memory with legacy 'tf_keras'
sys.modules["keras"] = tf_keras
# Forcefully replace keras attribute inside tensorflow
tf.keras = tf_keras

print(f"System core replacement complete! Current tf.keras version: {tf.keras.__version__}")

# ===== CRITICAL FIX: Clean Hidden '.ipynb_checkpoints' Files =====
dataset_path = "/content/dataSet"

print(f"\nCleaning dataset at: {dataset_path}")
if os.path.exists(dataset_path):
    # Walk through all folders and delete .ipynb_checkpoints
    for root, dirs, files in os.walk(dataset_path):
        for d in dirs:
            if d == ".ipynb_checkpoints":
                full_path = os.path.join(root, d)
                try:
                    shutil.rmtree(full_path)
                    print(f"   - Removed hidden folder: {full_path}")
                except Exception as e:
                    print(f"   - Failed to remove {full_path}: {e}")
    print("Cleanup complete. Ready to load data.")
else:
    print(f"Error: Dataset path '{dataset_path}' not found!")

# ===== Load MediaPipe & Train =====
from mediapipe_model_maker import gesture_recognizer

try:
    print("\nLoading data...")
    data = gesture_recognizer.Dataset.from_folder(
        dirname=dataset_path,
        hparams=gesture_recognizer.HandDataPreprocessingParams()
    )
    print("Data loaded successfully!")

    # Split data
    train_data, rest_data = data.split(0.8)
    validation_data, test_data = rest_data.split(0.5)

    # Start training
    print("Starting model training...")
    hparams = gesture_recognizer.HParams(export_dir="exported_model")
    options = gesture_recognizer.GestureRecognizerOptions(hparams=hparams)

    model = gesture_recognizer.GestureRecognizer.create(
        train_data=train_data,
        validation_data=validation_data,
        options=options
    )

    # Evaluate
    loss, acc = model.evaluate(test_data, batch_size=1)
    print(f"Test set accuracy: {acc * 100:.2f}%")

    # Export
    model.export_model()
    output_file = os.path.join(hparams.export_dir, 'gesture_recognizer.task')
    print(f"\nModel exported! Please refresh the file browser on the left to download: {output_file}")

except Exception as e:
    print(f"\nAn error occurred: {e}")
    import keras
    print(f"Debug Info - Actual Keras loaded: {keras.__version__}")
