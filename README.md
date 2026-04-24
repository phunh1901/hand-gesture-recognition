# Hand Gesture Recognition Project 🖐️

Dự án sử dụng MediaPipe để trích xuất tọa độ tay và mô hình SVM để nhận diện cử chỉ thực tế từ Webcam.

## 📁 Cấu trúc Pipeline

1. **Thu thập dữ liệu:** Chạy `collect_images.py` để chụp ảnh tay từ Webcam.
2. **Xử lý đặc trưng:** Chạy `extract_landmarks.py` để chuyển ảnh thành tọa độ 63 điểm (x, y, z).
3. **Phân tích & Tăng cường:** Sử dụng `eda_and_augmentation.ipynb` để làm giàu dữ liệu.
4. **Huấn luyện:** Chạy `model_training.ipynb` để tạo ra file `model.pkl`.
5. **Ứng dụng:** Chạy `app.py` để bắt đầu nhận diện cử chỉ.

## 🛠️ Cài đặt

```bash
pip install -r requirements.txt
```
