import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import glob
import cv2
import numpy as np
import mediapipe as mp
import os


GESTURES = ['pan', 'up', 'love', 'imobile']
DATA_PATH = 'data'
SEQUENCE_LENGTH = 15
NB_SAMPLES = 30  # Par classe

# Chargement des données
X, y = [], []

for file in glob.glob('data/*.npy'):
    gesture_name = file.split('/')[-1].split('_')[0]
    sequence = np.load(file)
    if sequence.shape == (15, 63):
        X.append(sequence)
        y.append(gesture_name)

X = np.array(X)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Construction du modèle LSTM
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(15, 63)),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(len(GESTURES), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y_categorical, epochs=30, validation_split=0.2)
model.save('lstm_gesture_model.h5')

