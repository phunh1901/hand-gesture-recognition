import cv2
import mediapipe as mp
import pandas as pd  
import os
import numpy as np 
from pathlib import Path

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

def calc_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calc_angle(a, b, c): # 2 chiều
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return angle


def image_processed(file_path):
    img = cv2.imread(file_path)
    if img is None:
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if not result.multi_hand_landmarks:
        return None

    landmarks = result.multi_hand_landmarks[0].landmark

    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    points = []
    # Chuẩn hóa tọa độ
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
        angle = calc_angle(points[tip], wrist, points[8])  # so với ngón trỏ
        features.append(angle)

    return features


def make_csv():
    BASE_DIR = Path(__file__).resolve().parent.parent
    dataset_path = BASE_DIR / "data" / "DATASET"
    output_file = BASE_DIR / "data" / "dataset.csv"

    all_data = []

    for class_name in os.listdir(dataset_path):
        class_path = dataset_path / class_name
        
        if not class_path.is_dir():
            continue

        print(f"Processing: {class_name}")

        for img_name in os.listdir(class_path):
            img_path = class_path / img_name
            
            data = image_processed(str(img_path))
            
            if data is not None:
                data.append(class_name)
                all_data.append(data)

    if all_data:
        df = pd.DataFrame(all_data)
        df.to_csv(output_file, index=False)
        print("Saved CSV:", output_file)
        print("Total samples:", len(all_data))
    else:
        print("No valid data!")

    hands.close()


if __name__ == "__main__":
    make_csv()