import cv2 as cv
import mediapipe as mp
import numpy as np 
import pickle


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=2, 
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

THRESHOLD = 0.8 

def get_best_hand_landmarks(frame):
    img_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    output = hands.process(img_rgb)
    
    if output.multi_hand_landmarks:
        if len(output.multi_hand_landmarks) == 1:
            landmarks = output.multi_hand_landmarks[0].landmark
        else:
            best_hand_idx = 0
            max_score = 0
            for i, hand_info in enumerate(output.multi_handedness):
                score = hand_info.classification[0].score
                if score > max_score:
                    max_score = score
                    best_hand_idx = i
            
            landmarks = output.multi_hand_landmarks[best_hand_idx].landmark
        
        clean_data = []
        for lm in landmarks:
            clean_data.extend([lm.x, lm.y, lm.z])
        return clean_data
    
    return None

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
    frame = cv.flip(frame, 1)

    features = get_best_hand_landmarks(frame)

    if features is not None:
        features = np.array(features).reshape(1, -1)
        
        probs = svm.predict_proba(features)
        max_prob = np.max(probs)
        
        if max_prob >= THRESHOLD:
            y_pred = svm.predict(features)
            label = f"{str(y_pred[0])} ({max_prob:.2f})"
            color = (0, 255, 0)
        else:
            label = f"Unknown ({max_prob:.2f})"
            color = (0, 0, 255)
            
        cv.putText(frame, label, (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv.LINE_AA)

    cv.imshow('Hand Gesture Recognition', frame)
    if cv.waitKey(1) == ord('q'): break

cap.release()
cv.destroyAllWindows()
hands.close()