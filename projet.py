import cv2
import numpy as np
import mediapipe as mp
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints(image):
    # Renvoie un vecteur de 21 points * 3 coordonnées = 63 features
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        keypoints = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark]).flatten()
    else:
        keypoints = np.zeros(63)
    
    return keypoints



GESTURES = ['imobile']
DATA_PATH = 'data'
SEQUENCE_LENGTH = 15
NB_SAMPLES = 30  # Par classe

for gesture in GESTURES:
    for sequence in range(NB_SAMPLES):
        window = []
        cap = cv2.VideoCapture(0)
        for frame_num in range(SEQUENCE_LENGTH):
            ret, frame = cap.read()
            keypoints = extract_keypoints(frame)
            window.append(keypoints)
            cv2.imshow('Collecte', frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

        window = np.array(window)
        np.save(os.path.join(DATA_PATH, gesture + '_' + str(sequence)), window)