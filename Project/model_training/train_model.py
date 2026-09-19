# ============================================================
#  STEP 2: MODEL TRAINING — LSTM Neural Network
#  Sign Recognition App — Final Year Project
#  Trains LSTM model on collected hand landmark sequences
# ============================================================
#
#  HOW TO RUN:
#  pip install tensorflow scikit-learn matplotlib seaborn numpy
#  python train_model.py
#
#  WHAT IT DOES:
#  - Loads collected .npy landmark sequences from dataset/
#  - Builds LSTM model: Input(30,126) → LSTM → Dense → Softmax
#  - Trains with 80/20 split, early stopping
#  - Saves best model as: models/sign_model.h5
#  - Saves label encoder as: models/labels.npy
#  - Plots accuracy/loss curves + confusion matrix

import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, BatchNormalization,
    Bidirectional, TimeDistributed
)
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)
from tensorflow.keras.utils import to_categorical
import warnings
warnings.filterwarnings('ignore')

# ─── CONFIG ──────────────────────────────────────────────────
DATA_PATH     = os.path.join('..', 'dataset')
MODEL_DIR     = os.path.join('..', 'models')
SEQUENCE_LEN  = 30       # frames per sequence
NUM_FEATURES  = 126      # 21 landmarks × 3 coords × 2 hands
EPOCHS        = 100
BATCH_SIZE    = 32
LEARNING_RATE = 0.001
TEST_SIZE     = 0.2
VAL_SIZE      = 0.1

os.makedirs(MODEL_DIR, exist_ok=True)

# ─── SIGN LABELS + TAMIL MAPPING ─────────────────────────────
TAMIL_MAP = {
    'hello':'வணக்கம்',     'thank_you':'நன்றி',
    'yes':'ஆம்',            'no':'இல்லை',
    'sorry':'மன்னிக்கவும்', 'please':'தயவுசெய்து',
    'love':'அன்பு',         'help':'உதவி',
    'water':'தண்ணீர்',      'good':'நல்லது',
    'bad':'கெட்டது',        'eat':'சாப்பிட',
    'sleep':'தூக்கம்',      'come':'வாருங்கள்',
    'go':'போ',              'zero':'பூஜ்யம்',
    'one':'ஒன்று',          'two':'இரண்டு',
    'three':'மூன்று',       'four':'நான்கு',
    'five':'ஐந்து',         'name':'பெயர்',
    'where':'எங்கே',        'how':'எப்படி',
    'what':'என்ன',
}

# ─── LOAD DATASET ────────────────────────────────────────────
def load_dataset():
    print("\n" + "="*60)
    print("  LOADING DATASET")
    print("="*60)

    sequences, labels = [], []

    # Auto-detect signs from dataset folder
    signs = sorted([
        d for d in os.listdir(DATA_PATH)
        if os.path.isdir(os.path.join(DATA_PATH, d))
    ])
    print(f"  Found signs: {signs}")
    print(f"  Total classes: {len(signs)}")

    for sign in signs:
        sign_path = os.path.join(DATA_PATH, sign)
        seq_folders = sorted(os.listdir(sign_path), key=lambda x: int(x))

        for seq_num in seq_folders:
            seq_path = os.path.join(sign_path, seq_num)
            frames = []

            for frame_num in range(SEQUENCE_LEN):
                npy_path = os.path.join(seq_path, f"{frame_num}.npy")
                if os.path.exists(npy_path):
                    frame_data = np.load(npy_path)
                    frames.append(frame_data)
                else:
                    frames.append(np.zeros(NUM_FEATURES))

            if len(frames) == SEQUENCE_LEN:
                sequences.append(frames)
                labels.append(sign)

    X = np.array(sequences)           # Shape: (samples, 30, 126)
    le = LabelEncoder()
    y_encoded = le.fit_transform(labels)
    y = to_categorical(y_encoded)     # One-hot encode

    print(f"  Dataset shape: {X.shape}")
    print(f"  Labels shape:  {y.shape}")
    print(f"  Classes: {le.classes_}")

    # Save label encoder
    np.save(os.path.join(MODEL_DIR, 'labels.npy'), le.classes_)
    np.save(os.path.join(MODEL_DIR, 'tamil_map.npy'), TAMIL_MAP)
    print(f"  ✅ Labels saved to models/labels.npy")

    return X, y, le.classes_

# ─── BUILD LSTM MODEL ────────────────────────────────────────
def build_model(num_classes, sequence_len=30, num_features=126):
    """
    Architecture:
    Input(30, 126)
      → Bidirectional LSTM(128) → BatchNorm → Dropout
      → Bidirectional LSTM(256) → BatchNorm → Dropout
      → LSTM(128) → BatchNorm → Dropout
      → Dense(128, relu) → Dropout
      → Dense(64, relu)
      → Dense(num_classes, softmax)
    """
    model = Sequential([
        # ── Layer 1: Bidirectional LSTM ──
        Bidirectional(
            LSTM(128, return_sequences=True, activation='tanh'),
            input_shape=(sequence_len, num_features)
        ),
        BatchNormalization(),
        Dropout(0.3),

        # ── Layer 2: Bidirectional LSTM ──
        Bidirectional(
            LSTM(256, return_sequences=True, activation='tanh')
        ),
        BatchNormalization(),
        Dropout(0.3),

        # ── Layer 3: LSTM ──
        LSTM(128, return_sequences=False, activation='tanh'),
        BatchNormalization(),
        Dropout(0.3),

        # ── Dense layers ──
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),

        # ── Output ──
        Dense(num_classes, activation='softmax'),
    ], name='ISL_SignRecognition')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy',
                 tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_acc')]
    )

    return model

# ─── PLOT TRAINING CURVES ────────────────────────────────────
def plot_history(history):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('ISL Sign Recognition — Training Results', fontsize=14, fontweight='bold')

    # Accuracy
    axes[0].plot(history.history['accuracy'],    label='Train', color='#2563eb', linewidth=2)
    axes[0].plot(history.history['val_accuracy'],label='Val',   color='#0d9488', linewidth=2, linestyle='--')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1)

    # Loss
    axes[1].plot(history.history['loss'],    label='Train', color='#dc2626', linewidth=2)
    axes[1].plot(history.history['val_loss'],label='Val',   color='#d97706', linewidth=2, linestyle='--')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    # Top-3 Accuracy
    axes[2].plot(history.history['top3_acc'],    label='Train', color='#7c3aed', linewidth=2)
    axes[2].plot(history.history['val_top3_acc'],label='Val',   color='#db2777', linewidth=2, linestyle='--')
    axes[2].set_title('Top-3 Accuracy')
    axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Top-3 Accuracy')
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print("  ✅ Training curves saved to models/training_curves.png")

# ─── PLOT CONFUSION MATRIX ───────────────────────────────────
def plot_confusion(y_true, y_pred, classes):
    cm = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))
    cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm_pct, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                linewidths=0.5, ax=ax)
    ax.set_title('Confusion Matrix — ISL Sign Recognition', fontsize=14, fontweight='bold', pad=16)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print("  ✅ Confusion matrix saved to models/confusion_matrix.png")

# ─── TRAIN ───────────────────────────────────────────────────
def train():
    # 1. Load data
    X, y, classes = load_dataset()
    num_classes = len(classes)

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=VAL_SIZE/(1-TEST_SIZE),
        random_state=42, stratify=y_train
    )
    print(f"\n  Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")

    # 3. Build model
    model = build_model(num_classes)
    model.summary()

    # 4. Callbacks
    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, 'sign_model.h5'),
            monitor='val_accuracy', save_best_only=True,
            verbose=1, mode='max'
        ),
        EarlyStopping(
            monitor='val_accuracy', patience=20,
            restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=8,
            min_lr=1e-6, verbose=1
        ),
        TensorBoard(log_dir=os.path.join(MODEL_DIR, 'logs'), histogram_freq=1),
    ]

    print("\n" + "="*60)
    print("  TRAINING MODEL")
    print(f"  Classes: {num_classes}  |  Epochs: {EPOCHS}  |  Batch: {BATCH_SIZE}")
    print("="*60)

    # 5. Train
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1,
    )

    # 6. Evaluate on test set
    print("\n" + "="*60)
    print("  EVALUATION ON TEST SET")
    print("="*60)
    test_loss, test_acc, test_top3 = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Test Accuracy  : {test_acc*100:.2f}%")
    print(f"  Test Top-3 Acc : {test_top3*100:.2f}%")
    print(f"  Test Loss      : {test_loss:.4f}")

    # 7. Classification report
    y_pred = model.predict(X_test)
    print("\n  Classification Report:")
    print(classification_report(
        y_test.argmax(axis=1), y_pred.argmax(axis=1),
        target_names=classes
    ))

    # 8. Plots
    plot_history(history)
    plot_confusion(y_test, y_pred, classes)

    # 9. Save TFLite (for mobile deployment)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(os.path.join(MODEL_DIR, 'sign_model.tflite'), 'wb') as f:
        f.write(tflite_model)

    print("\n" + "="*60)
    print("  ✅ TRAINING COMPLETE!")
    print(f"  Model saved: models/sign_model.h5")
    print(f"  TFLite:      models/sign_model.tflite")
    print(f"  Labels:      models/labels.npy")
    print(f"  Final Test Accuracy: {test_acc*100:.2f}%")
    print("="*60)

    return model, history

if __name__ == '__main__':
    tf.random.set_seed(42)
    np.random.seed(42)
    train()
