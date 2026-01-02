# -*- coding: utf-8 -*-
"""
================================================================================
CNN IMAGE CLASSIFICATION TRAINING PIPELINE
================================================================================

Model: EfficientNetV2S (Transfer Learning)
Dataset: Garbage Classification (Kaggle)
Author: [Dian Brown]
Date: January 2026

This notebook implements a two-stage transfer learning approach:
  - Stage 1: Train classifier head with frozen backbone (fast convergence)
  - Stage 2: Fine-tune entire model with lower learning rate (higher accuracy)

================================================================================
"""

# ==============================================================================
# CELL 1: IMPORT DEPENDENCIES
# ==============================================================================
# Import all required libraries for:
#   - Data manipulation (numpy)
#   - Deep learning framework (tensorflow, keras)
#   - Visualization (matplotlib)
#   - File path handling (pathlib)
# ==============================================================================

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from pathlib import Path

# ==============================================================================
# CELL 2: VERIFY ENVIRONMENT
# ==============================================================================
# Confirm TensorFlow version and GPU availability.
# ==============================================================================

print("TF:", tf.__version__)
print("GPU:", tf.config.list_physical_devices("GPU"))

# ==============================================================================
# CELL 3: ENABLE MIXED PRECISION TRAINING
# ==============================================================================
# Mixed precision (float16) provides:
#   - ~2x faster training on modern GPUs (Volta, Turing, Ampere)
#   - ~50% memory reduction (allows larger batch sizes)
# Note: Final layer uses float32 for numerical stability.
# ==============================================================================

from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy("mixed_float16")
print("Policy:", mixed_precision.global_policy())

# ==============================================================================
# CELL 4: DOWNLOAD DATASET
# ==============================================================================
# Download dataset from Kaggle using kagglehub.
# ==============================================================================

import kagglehub

path = kagglehub.dataset_download("hassnainzaidi/garbage-classification")
print("Dataset path:", path)

# ==============================================================================
# CELL 5: CONFIGURE DATA DIRECTORIES
# ==============================================================================
# Define paths to train/validation/test splits.
# Standard directory structure expected:
#   dataset/
#     ├── train/
#     │     ├── class_1/
#     │     ├── class_2/
#     │     └── ...
#     ├── val/
#     └── test/
# Note No train/test split was needed as the data was already categorized into train, val, and test folders. (like above structure)
# ==============================================================================

ROOT_DIR = Path(path) / "Garbage classification"
TRAIN_DIR = ROOT_DIR / "train"
VAL_DIR   = ROOT_DIR / "val"
TEST_DIR  = ROOT_DIR / "test"

print("ROOT_DIR:", ROOT_DIR)
print("Train exists:", TRAIN_DIR.exists())
print("Val exists:", VAL_DIR.exists())
print("Test exists:", TEST_DIR.exists())

print("Train folders:", [p.name for p in TRAIN_DIR.iterdir() if p.is_dir()])

# ==============================================================================
# CELL 6: DEFINE HYPERPARAMETERS
# ==============================================================================
# Key training configuration:
#   - SEED: Ensures reproducibility across runs
#   - IMG_SIZE: Input resolution (higher = better accuracy, more VRAM)
#   - BATCH_SIZE_STAGE1: Larger batch for frozen backbone (faster)
#   - BATCH_SIZE_STAGE2: Smaller batch for fine-tuning (prevents OOM)
# ==============================================================================

SEED = 42
IMG_SIZE = (320, 320)        # Safer than 384 on T4, still high accuracy
BATCH_SIZE_STAGE1 = 32       # Frozen backbone stage
BATCH_SIZE_STAGE2 = 8        # Fine-tuning stage (prevents OOM)

# ==============================================================================
# CELL 7: LOAD DATASETS (STAGE 1 - LARGER BATCHES)
# ==============================================================================
# Create tf.data.Dataset objects using Keras utility.
# This automatically:
#   - Loads images from subdirectory structure
#   - Infers class labels from folder names
#   - Resizes images to IMG_SIZE
#   - Batches data for training
# ==============================================================================

train_ds = keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE_STAGE1,
    shuffle=True
)

val_ds = keras.utils.image_dataset_from_directory(
    VAL_DIR,
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE_STAGE1,
    shuffle=False
)

test_ds = keras.utils.image_dataset_from_directory(
    TEST_DIR,
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE_STAGE1,
    shuffle=False
)

# Extract class names and count for model output layer
class_names = train_ds.class_names
NUM_CLASSES = len(class_names)

print("Classes:", class_names)
print("Num classes:", NUM_CLASSES)

# ==============================================================================
# CELL 8: OPTIMIZE DATA PIPELINE PERFORMANCE
# ==============================================================================
# Enable prefetching to overlap data loading with training.
# AUTOTUNE automatically determines optimal buffer size.
# This eliminates I/O bottlenecks during training.
# ==============================================================================

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds   = val_ds.prefetch(AUTOTUNE)
test_ds  = test_ds.prefetch(AUTOTUNE)

# ==============================================================================
# CELL 9: LOAD DATASETS (STAGE 2 - SMALLER BATCHES)
# ==============================================================================
# Create separate datasets with smaller batch size for fine-tuning.
# Fine-tuning requires more memory (gradients for all layers),
# so we reduce batch size to prevent out-of-memory errors.
# ==============================================================================

train_ds_ft = keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE_STAGE2,
    shuffle=True
).prefetch(AUTOTUNE)

val_ds_ft = keras.utils.image_dataset_from_directory(
    VAL_DIR,
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE_STAGE2,
    shuffle=False
).prefetch(AUTOTUNE)

# ==============================================================================
# CELL 10: VISUALIZE SAMPLE IMAGES
# ==============================================================================
# Display a 3x3 grid of training images to verify:
#   - Data loaded correctly
#   - Labels match images
#   - Image quality is acceptable
# ==============================================================================

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.tight_layout()
plt.show()

# ==============================================================================
# CELL 11: DEFINE DATA AUGMENTATION PIPELINE
# ==============================================================================
# Data augmentation artificially increases dataset diversity by applying
# random transformations during training. This helps:
#   - Reduce overfitting
#   - Improve generalization
#   - Make model robust to real-world variations
#
# Augmentations applied:
#   - RandomFlip: Horizontal mirror
#   - RandomRotation: ±5% rotation
#   - RandomZoom: ±10% zoom
#   - RandomTranslation: ±5% shift
#   - RandomContrast: ±10% contrast adjustment
# ==============================================================================

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.05, 0.05),
    layers.RandomContrast(0.1),
], name="augmentation")

# ==============================================================================
# CELL 12: LOAD PRETRAINED BACKBONE
# ==============================================================================
# Use EfficientNetV2S pretrained on ImageNet as feature extractor.
# Transfer learning benefits:
#   - Leverages knowledge from 1M+ ImageNet images
#   - Requires less training data
#   - Achieves higher accuracy faster
#
# include_top=False: Remove original classification head
# weights="imagenet": Load pretrained weights
# ==============================================================================

preprocess = tf.keras.applications.efficientnet_v2.preprocess_input

backbone = tf.keras.applications.EfficientNetV2S(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
)

# Stage 1: Freeze backbone weights (only train new classifier head)
backbone.trainable = False

# ==============================================================================
# CELL 13: BUILD COMPLETE MODEL ARCHITECTURE
# ==============================================================================
# Construct the full model by connecting:
#   1. Input layer (accepts raw images)
#   2. Data augmentation (training only)
#   3. Preprocessing (normalize for backbone)
#   4. Backbone (frozen feature extractor)
#   5. Global Average Pooling (flatten features)
#   6. Dropout (regularization, prevents overfitting)
#   7. Dense output layer (softmax classification)
#
# Note: Output layer uses float32 for numerical stability with mixed precision.
# ==============================================================================

inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = data_augmentation(inputs)
x = preprocess(x)
x = backbone(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)

# IMPORTANT: dtype float32 so outputs aren't float16 (numerical stability)
outputs = layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32")(x)

model = keras.Model(inputs, outputs)
model.summary()

# ==============================================================================
# CELL 14: COMPILE MODEL FOR STAGE 1 TRAINING
# ==============================================================================
# Configure training settings:
#   - Optimizer: Adam with learning rate 1e-3 (relatively high for frozen backbone)
#   - Loss: SparseCategoricalCrossentropy (for integer labels)
#   - Metrics: Accuracy
# ==============================================================================

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)

# ==============================================================================
# CELL 15: DEFINE STAGE 1 CALLBACKS
# ==============================================================================
# Callbacks control training behavior:
#   - ModelCheckpoint: Save best model based on validation accuracy
#   - EarlyStopping: Stop training if no improvement for 3 epochs
#                    (restores best weights automatically)
# ==============================================================================

callbacks_stage1 = [
    keras.callbacks.ModelCheckpoint("best_stage1.keras", save_best_only=True, monitor="val_accuracy", mode="max"),
    keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
]

# ==============================================================================
# CELL 16: STAGE 1 TRAINING - FROZEN BACKBONE
# ==============================================================================
# Train only the classifier head while backbone is frozen.
# This is fast (~minutes) and establishes a good starting point.
# Expected: Rapid accuracy improvement in first few epochs.
# ==============================================================================

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=callbacks_stage1
)

# ==============================================================================
# CELL 17: UNFREEZE BACKBONE FOR FINE-TUNING
# ==============================================================================
# Stage 2: Enable training of backbone layers.
# BatchNormalization layers are kept frozen for stability.
# Fine-tuning allows the model to adapt pretrained features to our specific task.
# ==============================================================================

backbone.trainable = True

# Freeze BatchNorm layers (recommended for stable fine-tuning)
for layer in backbone.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

# ==============================================================================
# CELL 18: RECOMPILE MODEL FOR STAGE 2
# ==============================================================================
# Recompile with much lower learning rate (2e-5) for fine-tuning.
# Low learning rate prevents:
#   - Destroying pretrained features
#   - Catastrophic forgetting
#   - Unstable gradients
# ==============================================================================

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=2e-5),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)

# ==============================================================================
# CELL 19: DEFINE STAGE 2 CALLBACKS
# ==============================================================================
# Enhanced callbacks for fine-tuning:
#   - ModelCheckpoint: Save best fine-tuned model
#   - EarlyStopping: Patience=6 (more epochs for subtle improvements)
#   - ReduceLROnPlateau: Automatically reduce LR when stuck
# ==============================================================================

callbacks_stage2 = [
    keras.callbacks.ModelCheckpoint("best_finetuned.keras", save_best_only=True, monitor="val_accuracy", mode="max"),
    keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=2, min_lr=1e-6, mode="max"),
]

# ==============================================================================
# CELL 20: STAGE 2 TRAINING - FINE-TUNING
# ==============================================================================
# Fine-tune entire model with smaller batches and lower learning rate.
# This is slower but typically improves accuracy by 2-5%.
# Uses separate dataset with smaller batch size to prevent OOM.
# ==============================================================================

history2 = model.fit(
    train_ds_ft,
    validation_data=val_ds_ft,
    epochs=30,
    callbacks=callbacks_stage2
)

# ==============================================================================
# CELL 21: EVALUATE ON TEST SET
# ==============================================================================
# Final evaluation on held-out test set.
# This gives unbiased estimate of real-world performance.
# Test accuracy should be close to validation accuracy.
# ==============================================================================

test_loss, test_acc = model.evaluate(test_ds)
print("Test accuracy:", test_acc)

# ==============================================================================
# CELL 22: GENERATE CLASSIFICATION REPORT
# ==============================================================================
# Detailed per-class metrics using sklearn:
#   - Precision: What % of predictions for each class are correct
#   - Recall: What % of actual class samples are correctly identified
#   - F1-Score: Harmonic mean of precision and recall
#   - Support: Number of samples per class
# ==============================================================================

from sklearn.metrics import confusion_matrix, classification_report

y_true = np.concatenate([y.numpy() for _, y in test_ds], axis=0)
y_prob = model.predict(test_ds)
y_pred = np.argmax(y_prob, axis=1)

print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

# ==============================================================================
# CELL 23: VISUALIZE CONFUSION MATRIX
# ==============================================================================
# Confusion matrix shows:
#   - Diagonal: Correct predictions (higher = better)
#   - Off-diagonal: Misclassifications (reveals class confusion patterns)
# Useful for identifying which classes the model struggles with.
# ==============================================================================

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(range(NUM_CLASSES), class_names, rotation=90)
plt.yticks(range(NUM_CLASSES), class_names)
plt.colorbar()
plt.tight_layout()
plt.show()

# ==============================================================================
# CELL 24: SAVE AND EXPORT MODEL
# ==============================================================================
# Save trained model in Keras format (.keras).
# This preserves:
#   - Model architecture
#   - Trained weights
#   - Training configuration (optimizer, loss, metrics)
#
# For deployment, consider also saving as:
#   - TFLite (mobile/edge devices)
#   - SavedModel (TensorFlow Serving)
#   - ONNX (cross-framework compatibility)
# ==============================================================================

model.save("waste_sorting_model.keras")
print("Saved: waste_sorting_model.keras")

# Download model file (Colab-specific)
from google.colab import files
files.download("waste_sorting_model.keras")