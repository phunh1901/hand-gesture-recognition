import cv2 as cv
import mediapipe as mp
import numpy as np 
import pickle

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=1, 
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

def get_landmarks(frame):
    img_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    output = hands.process(img_rgb)
    
    if output.multi_hand_landmarks:
        clean_data = []
        landmarks = output.multi_hand_landmarks[0].landmark
        for lm in landmarks:
            clean_data.extend([lm.x, lm.y, lm.z])
        return clean_data
    
    return [0.0] * 63

try:
    with open('./models/hand_gesture_svm.pkl', 'rb') as f:
        svm = pickle.load(f)
except FileNotFoundError:
    print("Không tìm thấy file hand_gesture_svm.pkl!")
    exit()

cap = cv.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    features = get_landmarks(frame)
    features = np.array(features).reshape(1, -1)

    if np.any(features):
        y_pred = svm.predict(features)
        
        label = str(y_pred[0])
        cv.putText(frame, label, (50, 100), cv.FONT_HERSHEY_SIMPLEX, 
                   2, (255, 0, 0), 3, cv.LINE_AA)

    cv.imshow('Hand Gesture Recognition', frame)
    
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
hands.close()