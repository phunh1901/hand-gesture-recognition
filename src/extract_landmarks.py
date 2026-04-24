import cv2
import mediapipe as mp
import pandas as pd  
import os
import numpy as np 


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

def image_processed(file_path):
    hand_img = cv2.imread(file_path)
    if hand_img is None:
        return None

    img_rgb = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
    
    output = hands.process(img_rgb) 

    if output.multi_hand_landmarks:
        clean_data = []
        landmarks = output.multi_hand_landmarks[0]
        for lm in landmarks.landmark:
            clean_data.extend([lm.x, lm.y, lm.z])
        return clean_data
    
    return None 


def make_csv():
    mypath = '../data/DATASET'
    output_file = '../data/dataset.csv'
    
    all_data = [] 

    for each_folder in os.listdir(mypath):
        if each_folder.startswith('.'): continue 

        print(f"Processing class: {each_folder}...")
        folder_path = os.path.join(mypath, each_folder)
        
        for each_image in os.listdir(folder_path):
            if each_image.startswith('.'): continue
            
            file_loc = os.path.join(folder_path, each_image)
            data = image_processed(file_loc)
            
            if data is not None:
                data.append(each_folder)
                all_data.append(data)
            else:
                print(f"Skipped (No hand): {each_image}")


    if all_data:
        df = pd.DataFrame(all_data)
        df.to_csv(output_file, mode='a', index=False, header=not os.path.exists(output_file))
        print(f'Thành công! Đã tạo dữ liệu với {len(all_data)} mẫu sạch.')
    else:
        print("Không có dữ liệu hợp lệ nào được tìm thấy.")

    hands.close()

if __name__ == "__main__":
    make_csv()