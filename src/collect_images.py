import numpy as np
import cv2 as cv
from pathlib import Path
import mediapipe as mp

def get_image():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    
    Class = 'Like'
    Path('../data/DATASET/'+Class).mkdir(parents=True, exist_ok=True)
    
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
        
    count = 0  
    total_attempts = 0
    
    print(f"Bắt đầu thu thập dữ liệu cho lớp: {Class}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            total_attempts += 1
            
            if total_attempts % 5 == 0:
                count += 1
                img_path = f'DATASET/{Class}/{count}.png'
                cv.imwrite(img_path, frame)
                
                cv.putText(frame, f"Saved: {count}", (10, 50), 
                           cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv.putText(frame, "No Hand Detected!", (10, 50), 
                       cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
      
        cv.imshow('Collecting Data', frame)
        
        if cv.waitKey(1) == ord('q') or count >= 300:
            break
  
    cap.release()
    cv.destroyAllWindows()
    hands.close()
    print(f"Đã lưu xong {count} ảnh vào thư mục {Class}.")

if __name__ == "__main__":
    get_image()