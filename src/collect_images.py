import cv2 as cv
from pathlib import Path
import mediapipe as mp
import os

def get_image():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    
    Class = 'SEEK_BW'
    TARGET_TOTAL = 300
    
    BASE_DIR = Path(__file__).resolve().parent.parent
    save_dir = BASE_DIR / "data" / "DATASET" / Class
    save_dir.mkdir(parents=True, exist_ok=True)

    existing_files = list(save_dir.glob("*.png"))
    current_count = len(existing_files)
    
    if current_count >= TARGET_TOTAL:
        print(f"Đã đủ {TARGET_TOTAL} ảnh cho lớp {Class}. Không cần thu thập thêm.")
        return

    print(f"Lớp: {Class}")
    print(f"Đã có: {current_count} ảnh. Sẽ bắt đầu lưu từ ảnh thứ {current_count + 1}.")
    print(f"Lưu tại: {save_dir.resolve()}")
    
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return
        
    count = current_count
    total_attempts = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv.flip(frame, 1)
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            total_attempts += 1
            
            if total_attempts % 5 == 0:
                count += 1
                img_path = save_dir / f"{count}.png"
                cv.imwrite(str(img_path), frame)
                print(f"Saved: {img_path}")

            cv.putText(frame, f"Count: {count}/{TARGET_TOTAL}", (10, 50),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv.putText(frame, "No Hand Detected", (10, 50),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv.imshow("Collecting", frame)
        
        if cv.waitKey(1) == ord('q') or count >= TARGET_TOTAL:
            break
    
    cap.release()
    cv.destroyAllWindows()
    hands.close()
    print(f"Tiến trình kết thúc. Hiện có tổng cộng {count} ảnh trong thư mục {Class}.")

if __name__ == "__main__":
    get_image()