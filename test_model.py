import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# === Chargement du modèle et des classes
GESTURES = ['pan', 'up', 'love', 'imobile']
model = load_model('lstm_gesture_model.h5')

label_encoder = LabelEncoder()
label_encoder.fit(GESTURES)

# === Initialisation de MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# === Fonction d'extraction
def extract_keypoints(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0]
        keypoints = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark]).flatten()
    else:
        keypoints = np.zeros(63)
    return keypoints

# === Prédiction temps réel
sequence = []
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Extraire les coordonnées de la main
    keypoints = extract_keypoints(frame)
    sequence.append(keypoints)

    # 2. Garder les 30 dernières frames
    if len(sequence) > 15:
        sequence.pop(0)

    # 3. Faire une prédiction si on a 30 frames
    if len(sequence) == 15:
        input_data = np.expand_dims(sequence, axis=0)
        prediction = model.predict(input_data)[0]
        predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])[0]
        confidence = np.max(prediction)

        if confidence > 0.8:  # seuil de confiance
            cv2.putText(frame, f"{predicted_class} ({confidence:.2f})",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            print(predicted_class)

    # 4. Afficher la vidéo avec le texte
    #cv2.imshow('Détection de geste', frame)

    # 5. Quitter avec 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# === Nettoyage
cap.release()
cv2.destroyAllWindows()
