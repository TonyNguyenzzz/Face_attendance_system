# Face Recognition Attendance System

Hệ thống điểm danh tự động sử dụng nhận diện khuôn mặt với công nghệ Deep Learning và ONNX.

## 📋 Tổng quan

Đây là hệ thống điểm danh dựa trên nhận diện khuôn mặt, cho phép:
- **Đăng ký khuôn mặt**: Lưu trữ thông tin và embedding khuôn mặt của sinh viên/người dùng
- **Điểm danh tự động**: Nhận diện khuôn mặt và ghi nhận thời gian điểm danh
- **Quản lý dữ liệu**: CRUD thông tin người dùng, xem lịch sử điểm danh
- **Báo cáo thống kê**: Trực quan hóa dữ liệu điểm danh

## 🏗 Kiến trúc hệ thống

### Pipeline xử lý

```
┌──────────────────────┐
│   Ảnh đầu vào        │
│ (Webcam/File/Video)  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  FaceDetector (ONNX) │
│  - Detect Face       │
│  - Detect Keypoints  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Crop ROI           │
│ (Cắt vùng khuôn mặt)  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Align Face         │
│ (Căn chỉnh bằng      │
│  5 keypoints)        │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Preprocess Face      │
│ - Resize             │
│ - Chuyển RGB         │
│ - CLAHE              │
│ - Chuẩn hóa [-1,1]   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Face Embedding Model │
│ (ONNX: ArcFace/      │
│  FaceNet/...)        │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ So sánh với Database │
│ (Nhận diện hoặc      │
│  Đăng ký)            │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Giao diện/Điểm danh  │
│ (Hiển thị, lưu kết quả│
│  thao tác DB)        │
└──────────────────────┘
```

### Các bước xử lý chi tiết

1. **Phát hiện khuôn mặt (Face Detection)**
   - Sử dụng model SCRFD (ONNX) để phát hiện tất cả khuôn mặt trong ảnh
   - Đầu ra: bounding boxes, 5 keypoints (2 mắt, mũi, 2 khóe miệng), confidence

2. **Cắt và căn chỉnh (Crop & Alignment)**
   - Cắt vùng khuôn mặt dựa trên bounding box
   - Sử dụng affine transform để căn chỉnh khuôn mặt về vị trí chuẩn

3. **Tiền xử lý (Preprocessing)**
   - Resize về kích thước chuẩn (160x160)
   - Cân bằng sáng (CLAHE)
   - Chuyển đổi RGB và chuẩn hóa giá trị pixel [-1, 1]

4. **Sinh embedding**
   - Sử dụng model FaceNet512 để sinh vector đặc trưng 512 chiều
   - Vector embedding dùng để so sánh và nhận diện

5. **Nhận diện/Đăng ký**
   - So sánh embedding với database (cosine/Euclidean distance)
   - Nếu khoảng cách < ngưỡng → xác định danh tính
   - Nếu đăng ký → lưu embedding mới vào database

## 📁 Cấu trúc dự án

```
workspace/
├── config/
│   └── config.py              # Cấu hình hệ thống
├── src/
│   ├── gui.py                 # Giao diện người dùng chính
│   ├── cam_realtime.py        # Giao diện camera realtime
│   ├── camera_handler.py      # Xử lý camera
│   ├── database_handler.py    # Quản lý database SQLite
│   ├── face_detector.py       # Module phát hiện khuôn mặt
│   ├── face_embedding.py      # Module sinh embedding
│   ├── face_recognition.py    # Module nhận diện
│   ├── train_worker.py        # Worker xử lý batch
│   ├── logging_config.py      # Cấu hình logging
│   └── onnx_config.py         # Cấu hình ONNX runtime
├── models/                    # Thư mục chứa model ONNX
├── attendance.db              # Database SQLite
└── requirements.txt           # Dependencies
```

## 🚀 Cài đặt

### Yêu cầu hệ thống

- Python 3.8+
- RAM: 4GB+ (khuyến nghị 8GB+)
- GPU (tùy chọn): CUDA 11.x để tăng tốc xử lý

### Các bước cài đặt

1. **Clone repository hoặc tải source code**

2. **Cài đặt dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Tải model ONNX**
   - Đặt file model SCRFD vào thư mục `models/`
   - Tên file: `scrfd_2.5g.onnx` (hoặc cấu hình trong `config.py`)

4. **Chạy ứng dụng**
   ```bash
   cd src
   python gui.py
   ```

## ⚙️ Cấu hình

File cấu hình: `config/config.py`

### Các tham số chính

| Tham số | Giá trị mặc định | Mô tả |
|---------|------------------|-------|
| `detector_confidence` | 0.7 | Ngưỡng phát hiện khuôn mặt |
| `recognition_distance` | 0.35 | Ngưỡng khoảng cách nhận diện |
| `face_model` | "Facenet512" | Model embedding sử dụng |
| `face_size` | (160, 160) | Kích thước ảnh khuôn mặt |
| `use_gpu` | True | Sử dụng GPU acceleration |
| `camera_id` | 0 | ID camera đầu vào |
| `attendance_cooldown` | 300 | Thời gian chờ giữa lần điểm danh (giây) |

## 📦 Các module chính

### 1. FaceDetector (`face_detector.py`)
- Phát hiện khuôn mặt sử dụng SCRFD model
- Trích xuất 5 keypoints cho mỗi khuôn mặt

### 2. FaceEmbedding (`face_embedding.py`)
- Căn chỉnh và tiền xử lý khuôn mặt
- Sinh vector embedding từ model FaceNet512

### 3. AttendanceSystem (`face_recognition.py`)
- So sánh embedding với database
- Quản lý logic điểm danh

### 4. DatabaseHandler (`database_handler.py`)
- Lưu trữ thông tin người dùng
- Quản lý embedding và lịch sử điểm danh
- Xuất báo cáo Excel/PDF

### 5. GUI (`gui.py`)
- Giao diện PyQt5 với các tab:
  - **Camera**: Xem camera và điểm danh realtime
  - **Quản lý người dùng**: Thêm/sửa/xóa thông tin
  - **Thống kê**: Biểu đồ và báo cáo điểm danh
  - **Cấu hình**: Điều chỉnh tham số hệ thống

## 🎯 Tính năng

- ✅ Điểm danh tự động qua camera
- ✅ Đăng ký khuôn mặt mới
- ✅ Tìm kiếm và quản lý người dùng
- ✅ Xuất báo cáo điểm danh (Excel, PDF)
- ✅ Thống kê trực quan (matplotlib)
- ✅ Hỗ trợ đa luồng (multi-threading)
- ✅ Logging chi tiết
- ✅ Cache embedding để tối ưu hiệu suất

## 🔧 Công nghệ sử dụng

- **Deep Learning**: TensorFlow, Keras, ONNX Runtime
- **Computer Vision**: OpenCV, MTCNN, RetinaFace
- **GUI**: PyQt5, Matplotlib
- **Database**: SQLite
- **Face Recognition**: DeepFace, FaceNet, ArcFace
- **Export**: FPDF2, OpenPyXL

## 📝 Lưu ý

- Đảm bảo ánh sáng tốt khi quay camera để tăng độ chính xác
- Có thể điều chỉnh `recognition_distance` để cân bằng giữa precision và recall
- Sử dụng GPU để cải thiện đáng kể hiệu suất xử lý

## 📄 License

Dự án phục vụ mục đích học tập và nghiên cứu.

## 👥 Đóng góp

Mọi đóng góp vui lòng tạo pull request hoặc report issue.
