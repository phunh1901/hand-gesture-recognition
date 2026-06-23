import cv2 as cv
import mediapipe as mp
import numpy as np
import joblib
import time
import pyautogui

# ===== INIT =====
pyautogui.FAILSAFE = False  # tránh crash khi di chuột vào góc màn hình

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7  
)

THRESHOLD = 0.8
HOLD_TIME = 2.0       
COOLDOWN = 2.0     # delay sau khi trigger

current_gesture = None
gesture_start_time = None
last_trigger_time = 0
no_hand_counter = 0
MAX_NO_HAND_FRAMES = 5

# ===== LOAD MODEL =====
with open('./models/hand_gesture_svm.pkl', 'rb') as f:
    model_data = joblib.load(f)

svm = model_data['model']
scaler = model_data['scaler']


# ===== UTILS =====
def apply_clahe(img):
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv.merge((cl, a, b))
    return cv.cvtColor(limg, cv.COLOR_LAB2BGR)


# ===== FEATURE EXTRACT =====
def extract_features(frame):
    # Tiền xử lý chống lóa sáng bằng CLAHE
    processed_frame = apply_clahe(frame)
    img_rgb = cv.cvtColor(processed_frame, cv.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if not result.multi_hand_landmarks:
        return None

    # Vẽ landmarks để trực quan hóa bàn tay lên webcam gốc
    for hand_landmarks in result.multi_hand_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2), # Khớp xanh lá
            mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=2) # Xương đỏ
        )

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

    # (x, y) coordinates
    for (x, y) in points:
        features.extend([x, y])

    # distance giữa các fingertip
    fingertip_ids = [4, 8, 12, 16, 20]
    for i in range(len(fingertip_ids)):
        for j in range(i + 1, len(fingertip_ids)):
            p1 = points[fingertip_ids[i]]
            p2 = points[fingertip_ids[j]]
            dist = np.linalg.norm(np.array(p1) - np.array(p2))
            features.append(dist)

    # angle tại cổ tay
    def calc_angle(a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba = a - b
        bc = c - b
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return np.arccos(np.clip(cos_angle, -1.0, 1.0))

    wrist = points[0]
    for tip in fingertip_ids:
        angle = calc_angle(points[tip], wrist, points[8])
        features.append(angle)

    return np.array(features).reshape(1, -1)


# ===== ACTION =====
def trigger_action(gesture):
    print(f"[ACTION TRIGGERED] {gesture}")

    if gesture == "FULL_SCREEN":
        pyautogui.press('f')

    elif gesture == "MUTE":
        pyautogui.press('m')

    elif gesture == "PLAY_PAUSE":
        pyautogui.press('space')

    elif gesture == "SEEK_BW":
        pyautogui.press('left')

    elif gesture == "SEEK_FW":
        pyautogui.press('right')

    elif gesture == "VOL_DOWN":
        pyautogui.press('down')

    elif gesture == "VOL_UP":
        pyautogui.press('up')


# ===== MAIN =====
cap = cv.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv.flip(frame, 1)

    features = extract_features(frame)

    label = "No Hand"
    color = (0, 0, 255)
    now = time.time()

    if features is not None:
        no_hand_counter = 0  # Reset bộ đếm mất dấu
        features_scaled = scaler.transform(features)

        probs = svm.predict_proba(features_scaled)
        max_prob = np.max(probs)

        if max_prob >= THRESHOLD:
            pred = svm.predict(features_scaled)[0]
            label = f"{pred} ({max_prob:.2f})"
            color = (0, 255, 0)

            # ===== HOLD LOGIC =====
            if pred == current_gesture:
                if gesture_start_time is None:
                    gesture_start_time = now

                elapsed = now - gesture_start_time

                # đủ thời gian giữ + hết cooldown
                if elapsed >= HOLD_TIME and (now - last_trigger_time) > COOLDOWN:
                    trigger_action(pred)
                    last_trigger_time = now
                    gesture_start_time = None  # reset để đếm lại
            else:
                # gesture mới → reset
                current_gesture = pred
                gesture_start_time = now

        else:
            # confidence thấp → reset ngay lập tức
            current_gesture = None
            gesture_start_time = None

    else:
        # Không phát hiện thấy tay (có thể do bị lóa hoặc che khuất tạm thời)
        no_hand_counter += 1
        if no_hand_counter >= MAX_NO_HAND_FRAMES:
            # Thực sự mất dấu tay
            current_gesture = None
            gesture_start_time = None
            label = "No Hand"
            color = (0, 0, 255)
        else:
            # Tạm thời giữ lại trạng thái cũ để tránh giật lag khi demo (Persistence)
            if current_gesture is not None:
                label = f"{current_gesture} (Lost...)"
                color = (0, 255, 255)  # Màu vàng cảnh báo đang cố hồi phục dấu tay

    # ===== UI =====
    cv.putText(frame, label, (50, 100),
               cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Thanh tiến trình hold
    if gesture_start_time is not None:
        elapsed = now - gesture_start_time
        hold_progress = min(int(elapsed / HOLD_TIME * 100), 100)  # fix: clamp tối đa 100%

        cv.putText(frame, f"Holding: {hold_progress}%",
                   (50, 150), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Vẽ thanh progress bar
        bar_x, bar_y, bar_w, bar_h = 50, 170, 300, 20
        filled_w = int(bar_w * hold_progress / 100)
        cv.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (100, 100, 100), 2)
        cv.rectangle(frame, (bar_x, bar_y), (bar_x + filled_w, bar_y + bar_h), (0, 255, 255), -1)

    # Hướng dẫn
    cv.putText(frame, "Press Q to quit", (50, frame.shape[0] - 20),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv.imshow("Gesture Control", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
hands.close()