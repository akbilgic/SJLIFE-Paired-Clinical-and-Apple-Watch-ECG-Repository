import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Conv1D, Activation, Add
from keras.models import Model
from keras.optimizers import Adam


# ======================
# Data Loading
# ======================

def load_ecg_data(apple_data_dir, clinical_data_dir):
    """
    Load paired Apple and Clinical ECG data from directories
    Returns:
        apple_data: numpy array of shape (num_samples, 5000, 1)
        clinical_data: numpy array of shape (num_samples, 5000, 1)
    """
    apple_files = sorted([f for f in os.listdir(apple_data_dir) if f.endswith('.npy')])
    clinical_files = sorted([f for f in os.listdir(clinical_data_dir) if f.endswith('.npy')])
    
    assert len(apple_files) == len(clinical_files), "Mismatched number of Apple/Clinical ECGs"
    
    apple_data = []
    clinical_data = []
    
    for af, cf in zip(apple_files, clinical_files):
        apple = np.load(os.path.join(apple_data_dir, af))
        clinical = np.load(os.path.join(clinical_data_dir, cf))
        
        # Squeeze extra dims, ensure shape (5000,1)
        apple = apple.squeeze().reshape(-1, 1)
        clinical = clinical.squeeze().reshape(-1, 1)
        
        apple_data.append(apple)
        clinical_data.append(clinical)
    
    return np.array(apple_data), np.array(clinical_data)


# ======================
# Data Preparation
# ======================

apple_data, clinical_data = load_ecg_data(
    apple_data_dir='./500hz_10sec_apple',
    clinical_data_dir='./first_lead_clinical'
)

apple_train, apple_test, clinical_train, clinical_test = train_test_split(
    apple_data, clinical_data, test_size=0.2, random_state=42
)


# ======================
# Normalization Utility
# ======================

class Normalizer:
    def __init__(self, data):
        self.mean = np.mean(data)
        self.std = np.std(data)
    
    def normalize(self, x):
        return (x - self.mean) / self.std
    
    def denormalize(self, x):
        return x * self.std + self.mean


apple_normalizer = Normalizer(apple_train)
clinical_normalizer = Normalizer(clinical_train)

X_train = apple_normalizer.normalize(apple_train)
y_train = clinical_normalizer.normalize(clinical_train)
X_test  = apple_normalizer.normalize(apple_test)


# ======================
# (Optional) Data Augmentation
# ======================

def augment_ecg(ecg, noise_level=0.05):
    noise = np.random.normal(0, noise_level * np.std(ecg), ecg.shape)
    return ecg + noise


# ======================
# Model Definition
# ======================

def residual_block(x, filters, kernel_size=3):
    shortcut = x
    x = Conv1D(filters, kernel_size, padding='same')(x)
    x = Activation('relu')(x)
    x = Conv1D(filters, kernel_size, padding='same')(x)
    x = Add()([shortcut, x])
    return x

def build_ecg_transformer(input_shape=(5000, 1)):
    inputs = Input(shape=input_shape)
    
    # Initial convolution
    x = Conv1D(64, 15, padding='same')(inputs)
    x = Activation('relu')(x)
    
    # 6 residual blocks
    for _ in range(6):
        x = residual_block(x, 64)
    
    # Bottleneck & output
    x = Conv1D(32, 3, padding='same')(x)
    x = Activation('relu')(x)
    outputs = Conv1D(1, 3, padding='same')(x)
    
    model = Model(inputs, outputs, name='ecg_transformer')
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='mse',
        metrics=['mae']
    )
    return model

model = build_ecg_transformer()
model.summary()


# ======================
# Training
# ======================

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop]
)


# ======================
# Inference / Conversion
# ======================

def convert_apple_to_clinical(apple_ecg):
    """
    apple_ecg: array of shape (5000,1) or (batch, 5000,1)
    returns: denormalized clinical ECG prediction
    """
    x = apple_ecg
    if x.ndim == 2:
        x = x[np.newaxis, ...]   # make batch of 1
    x_norm = apple_normalizer.normalize(x)
    y_norm = model.predict(x_norm)
    return clinical_normalizer.denormalize(y_norm)


# Example plotting
sample_idx = 0
converted = convert_apple_to_clinical(apple_test[sample_idx])

plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(apple_test[sample_idx].squeeze(), label='Apple ECG')
plt.title('Original Apple ECG')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(converted.squeeze(), label='Converted Clinical ECG', alpha=0.8)
plt.plot(clinical_test[sample_idx].squeeze(), label='True Clinical ECG', alpha=0.5)
plt.title('Conversion Result')
plt.legend()

plt.tight_layout()
plt.show()
