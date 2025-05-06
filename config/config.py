import os

# Tạo thư mục cần thiết nếu chưa tồn tại
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Thư mục chứa mô hình
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
ensure_dir(MODEL_DIR)

# Thư mục chứa kết quả
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ensure_dir(RESULT_DIR)

# Cấu hình mặc định
DEFAULT_CONFIG = {
    # Thư mục
    "model_dir": MODEL_DIR,
    "result_dir": RESULT_DIR,
    # SCRFD model
    "scrfd_model": "scrfd_2.5g.onnx",
    # Tham số nhận diện
    "detector_confidence": 0.7,  # Ngưỡng phát hiện khuôn mặt (SCRFD)
    "recognition_distance": 0.35,      # Ngưỡng khoảng cách nhận diện - Giảm để tăng độ chính xác
    "recognition_confidence": 0.85,      # Ngưỡng tin cậy - Tăng để giảm nhận diện sai
    "min_confidence": 0.6,  # Ngưỡng tin cậy tối thiểu
    "distance_threshold": 0.95,  # Ngưỡng khoảng cách
    "min_confidence_threshold": 0.6,  # Ngưỡng tin cậy tối thiểu để nhận diện
    "absolute_distance_threshold": 0.6,  # Ngưỡng khoảng cách tối đa
    # Kích thước ảnh đầu vào
    "max_dim": 1080,                # Mặc định
    "face_size": (160, 160),       # Mặc định: (160, 160)
    "target_size": (160, 160),     # Cho phép đồng bộ với các module khác
    # Caching & worker
    "cache_size": 1000,            # Mặc định: 1000
    "max_workers": 4,              # Mặc định: 4
    # Camera
    "camera_id": 0,                # Mặc định: 0
    "camera_fps": 25,              # Mặc định: 25
    # GPU
    "use_gpu": True,
    # Điểm danh
    "attendance_cooldown": 300,    # 5 phút
    "late_threshold": 0,           # Mặc định: 0
    # Model embedding
    "face_model": "Facenet512",
}
