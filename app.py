import cv2 as cv
import mediapipe as mp
import numpy as np
import pickle
import joblib

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=1, 
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

THRESHOLD = 0.9

svm = joblib.load('./models/svm_model.pkl')
scaler = joblib.load('./models/scaler.pkl')


def calc_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calc_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc) + 1e-6)
    return np.arccos(np.clip(cos_angle, -1.0, 1.0))


def extract_features(frame):
    img_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if not result.multi_hand_landmarks:
        return None

    landmarks = result.multi_hand_landmarks[0].landmark

    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    points = []
    for lm in landmarks:
        x = (lm.x - min_x) / (max_x - min_x + 1e-6)
        y = (lm.y - min_y) / (max_y - min_y + 1e-6)
        points.append((x, y))

    features = []

    for (x, y) in points:
        features.extend([x, y])

    fingertip_ids = [4, 8, 12, 16, 20]
    for i in range(len(fingertip_ids)):
        for j in range(i+1, len(fingertip_ids)):
            d = calc_distance(points[fingertip_ids[i]], points[fingertip_ids[j]])
            features.append(d)

    wrist = points[0]
    for tip in fingertip_ids:
        angle = calc_angle(points[tip], wrist, points[8])
        features.append(angle)

    return np.array(features).reshape(1, -1)


cap = cv.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv.flip(frame, 1)

    features = extract_features(frame)

    if features is not None:
        features = scaler.transform(features)

        probs = svm.predict_proba(features)
        max_prob = np.max(probs)
        pred = svm.predict(features)[0]

        if max_prob >= THRESHOLD:
            label = f"{pred} ({max_prob:.2f})"
            color = (0, 255, 0)
        else:
            label = f"Unknown ({max_prob:.2f})"
            color = (0, 0, 255)

        cv.putText(frame, label, (50, 100),
                   cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv.imshow("Hand Gesture Recognition", frame)

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
hands.close()