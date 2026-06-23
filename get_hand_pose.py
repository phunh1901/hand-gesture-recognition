import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands

def draw_styled_hand(image_path, output_name):
    image = cv2.imread(image_path)
    h, w, _ = image.shape 
    h = h*4
    w = w*4
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
        results = hands.process(image_rgb)
        canvas = np.zeros((h, w, 3), dtype=np.uint8) # Nền đen

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Chuyển đổi tọa độ landmark sang pixel
                points = []
                for lm in hand_landmarks.landmark:
                    points.append((int(lm.x * w), int(lm.y * h)))

                # Định nghĩa các vùng ngón tay để vẽ (theo ID của MediaPipe)
                # Mỗi danh sách là tập hợp các điểm tạo nên một "vùng thịt"
                finger_polys = [
                    [0, 1, 2, 3, 4],    # Ngón cái
                    [0, 5, 6, 7, 8],    # Ngón trỏ
                    [0, 9, 10, 11, 12], # Ngón giữa
                    [0, 13, 14, 15, 16],# Ngón áp út
                    [0, 17, 18, 19, 20],# Ngón út
                    [5, 9, 13, 17, 0]   # Lòng bàn tay
                ]

                # Vẽ các vùng đầy (Fills) để tạo hình dáng bàn tay
                for poly in finger_polys:
                    pts = np.array([points[i] for i in poly], np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    # Vẽ vùng màu xám nhạt (mang tính học thuật)
                    cv2.fillPoly(canvas, [pts], (80, 80, 80)) 

                # Vẽ khung xương trắng đè lên trên
                mp.solutions.drawing_utils.draw_landmarks(
                    canvas,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1),
                    mp.solutions.drawing_utils.DrawingSpec(color=(200, 200, 200), thickness=2)
                )

            cv2.imwrite(f'styled_{output_name}.png', canvas)
            print("Đã tạo ảnh bàn tay mô phỏng!")

# Gọi hàm
image_path = r'C:\Users\HP\Desktop\learning\project\project2\project2\data\DATASET\VOL_DOWN\14.png'
output_name = 'VOL_DOWN'
draw_styled_hand(image_path, output_name)

