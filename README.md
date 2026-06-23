# 🖐️ Hand Gesture Recognition – Điều khiển trình phát media bằng cử chỉ tay

Dự án nhận diện cử chỉ tay theo thời gian thực bằng **MediaPipe** và **SVM (Support Vector Machine)**, cho phép điều khiển trình phát video (YouTube, VLC, v.v.) mà không cần chạm vào bàn phím hay chuột.

---

## 🎯 Tính năng

| Cử chỉ | Hành động | Phím tương ứng |
|---|---|---|
| ✋ **FULL\_SCREEN** | Bật / tắt toàn màn hình | `F` |
| 🤫 **MUTE** | Tắt / bật âm thanh | `M` |
| ✌️ **PLAY\_PAUSE** | Phát / dừng | `Space` |
| 👈 **SEEK\_BW** | Tua lùi | `←` |
| 👉 **SEEK\_FW** | Tua tới | `→` |
| 👇 **VOL\_DOWN** | Giảm âm lượng | `↓` |
| 👆 **VOL\_UP** | Tăng âm lượng | `↑` |

**Cơ chế hoạt động:** Giữ nguyên cử chỉ trong **2 giây** để kích hoạt hành động. Có thanh tiến trình hiển thị trực quan trên màn hình webcam.

---

## 🏗️ Kiến trúc hệ thống

```
Webcam → MediaPipe (trích xuất 21 điểm tay) → Feature Engineering → SVM → Hành động
```

1. **MediaPipe Hands**: Phát hiện và theo dõi bàn tay, trích xuất 21 điểm landmark trong không gian (x, y).
2. **Feature Engineering**: Từ 21 điểm landmark, hệ thống tính thêm:
   - Tọa độ chuẩn hóa của 21 điểm (42 features)
   - Khoảng cách giữa 5 đầu ngón tay (10 features)
   - Góc giữa các ngón tay so với cổ tay (5 features)
   - **Tổng: 57 features**
3. **SVM (RBF Kernel)**: Phân loại cử chỉ với xác suất dự đoán, chỉ trigger khi confidence ≥ 80%.
4. **Temporal Persistence**: Giữ trạng thái tối đa 5 frame khi bị mất dấu tay tạm thời (do ánh sáng hoặc che khuất).
5. **CLAHE Preprocessing**: Tăng cường độ tương phản để hoạt động tốt trong môi trường ánh sáng không đồng đều.

---

## 📁 Cấu trúc thư mục

```
project2/
├── app.py                      # 🚀 File chính – chạy ứng dụng demo
├── requirements.txt            # Danh sách thư viện cần thiết
│
├── src/
│   ├── collect_images.py       # Thu thập ảnh từ webcam để tạo dataset
│   └── extract_landmarks.py   # Trích xuất tọa độ landmark từ ảnh → CSV
│
├── data/
│   ├── DATASET/                # Ảnh gốc thu thập từ webcam
│   │   ├── FULL_SCREEN/
│   │   ├── MUTE/
│   │   ├── PLAY_PAUSE/
│   │   ├── SEEK_BW/
│   │   ├── SEEK_FW/
│   │   ├── VOL_DOWN/
│   │   └── VOL_UP/
│   └── dataset.csv             # Dữ liệu features đã trích xuất (3500 mẫu)
│
├── models/
│   ├── hand_gesture_svm.pkl    # Model SVM + Scaler đã huấn luyện (độ chính xác ~99%)
│   └── scaler.pkl              # StandardScaler (dùng riêng)
│
├── notebooks/
│   ├── eda_augmentation.ipynb  # Phân tích & tăng cường dữ liệu
│   └── model_training.ipynb    # Huấn luyện và đánh giá mô hình
│
└── get_hand_pose.py            # Tiện ích vẽ hình xương tay cho demo/báo cáo
```

---

## ⚙️ Cài đặt

### Yêu cầu hệ thống

- Python **3.10** trở lên
- Webcam
- Windows / Linux / macOS

### Bước 1 – Clone repository

```bash
git clone <repository-url>
cd project2
```

### Bước 2 – Tạo môi trường ảo (khuyến nghị)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### Bước 3 – Cài đặt thư viện

> **Lưu ý:** File `requirements.txt` là danh sách đầy đủ môi trường. Để cài nhanh chỉ các thư viện cốt lõi, dùng lệnh dưới đây:

```bash
pip install opencv-python mediapipe scikit-learn joblib pyautogui numpy pandas
```

Hoặc cài toàn bộ:

```bash
pip install -r requirements.txt
```

---

## 🚀 Sử dụng

### Chạy ứng dụng demo (nhanh nhất)

```bash
python app.py
```

Webcam sẽ bật lên. Đưa bàn tay vào khung hình và giữ cử chỉ trong **2 giây** để kích hoạt hành động tương ứng. Nhấn `Q` để thoát.

**Giao diện webcam hiển thị:**
- 🟢 **Tên cử chỉ + độ tự tin** (góc trên trái)
- 📊 **Thanh tiến trình giữ** (bên dưới tên cử chỉ)
- 🦴 **Khung xương tay** (21 điểm landmark của MediaPipe)
- 🟡 **"(Lost...)"** khi tay bị mất dấu tạm thời nhưng hệ thống vẫn cố hồi phục

---

## 🔬 Quy trình huấn luyện lại mô hình (tùy chọn)

Nếu muốn thu thập thêm dữ liệu hoặc huấn luyện lại, thực hiện theo các bước sau:

### Bước 1 – Thu thập ảnh

Mở `src/collect_images.py`, thay đổi tên lớp ở dòng:
```python
Class = 'TEN_CU_CHI'  # Ví dụ: 'VOL_UP'
```
Sau đó chạy:
```bash
python src/collect_images.py
```
Hệ thống tự động lưu **300 ảnh** vào `data/DATASET/<TEN_CU_CHI>/`.

### Bước 2 – Trích xuất features

```bash
python src/extract_landmarks.py
```
Tạo ra file `data/dataset.csv` với 57 features cho mỗi mẫu.

### Bước 3 – Tăng cường dữ liệu & phân tích (tùy chọn)

Mở và chạy notebook:
```
notebooks/eda_augmentation.ipynb
```

### Bước 4 – Huấn luyện mô hình

```bash
# Chạy notebook
jupyter notebook notebooks/model_training.ipynb
```

Hoặc chạy script:
```bash
cd notebooks
python model_training.py
```

Model mới sẽ được lưu vào `models/hand_gesture_svm.pkl`.

---

## 📊 Kết quả mô hình

| Metric | Giá trị |
|---|---|
| **Accuracy** | ~99.1% |
| **Số lớp** | 7 cử chỉ |
| **Tổng mẫu huấn luyện** | 3.500 mẫu (500/lớp) |
| **Thuật toán** | SVM – Kernel RBF |
| **Confidence threshold** | 80% |

---

## 🛠️ Cải tiến kỹ thuật

- **CLAHE (Contrast Limited Adaptive Histogram Equalization):** Tiền xử lý ảnh giúp hệ thống nhận diện tốt hơn trong điều kiện ánh sáng mạnh hoặc lóa sáng.
- **Temporal State Persistence:** Giữ trạng thái tối đa 5 frame liên tiếp khi mất dấu tay, tránh nhấp nháy khi demo.
- **Data Augmentation:** Tăng cường dữ liệu bằng lật ảnh ngang, thêm nhiễu và thay đổi tỷ lệ để cân bằng các lớp.

---

## 📦 Thư viện chính

| Thư viện | Vai trò |
|---|---|
| `mediapipe` | Phát hiện và theo dõi bàn tay |
| `opencv-python` | Xử lý ảnh và webcam |
| `scikit-learn` | Huấn luyện mô hình SVM |
| `joblib` | Lưu/tải mô hình |
| `pyautogui` | Điều khiển bàn phím |
| `numpy` / `pandas` | Xử lý dữ liệu |

---

## 📝 Lưu ý

- Đảm bảo bàn tay nằm gọn trong khung hình webcam để MediaPipe nhận diện chính xác.
- Nền phía sau bàn tay càng đơn giản (ít họa tiết) thì độ chính xác càng cao.
- Ánh sáng đầy đủ và đồng đều sẽ cho kết quả tốt nhất.
- Hệ thống đã được tối ưu để hoạt động với **một bàn tay** tại một thời điểm.
