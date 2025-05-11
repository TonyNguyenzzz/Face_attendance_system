import os

# Tạo thư mục cần thiết nếu chưa tồn tại
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Thư mục chứa mô hình
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
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
    
    # Tham số nhận diện khuôn mặt
    "detector_confidence": 0.7,  # Ngưỡng phát hiện khuôn mặt (SCRFD) - Tăng để giảm phát hiện sai
    
    # Tham số nhận diện danh tính 
    "recognition_distance": 0.35,      # Ngưỡng khoảng cách nhận diện chính - Giảm để tăng độ chính xác nhận diện
                                      # Được sử dụng trong giao diện người dùng và là tham số chính để điều chỉnh
    "recognition_confidence": 0.85,    # Ngưỡng tin cậy hiển thị - Tăng để giảm hiển thị nhận diện sai
                                      # Chỉ ảnh hưởng đến việc hiển thị kết quả, không ảnh hưởng đến thuật toán
    
    # Các tham số bên dưới được giữ lại để tương thích với mã nguồn hiện tại
    # Trong các phiên bản tương lai, chúng sẽ được hợp nhất hoặc loại bỏ
    "distance_threshold": 0.35,        # Ngưỡng khoảng cách trong thuật toán - Đồng bộ với recognition_distance
                                      # Lưu ý: Giá trị này được đồng bộ tự động với recognition_distance trong GUI
    "min_confidence_threshold": 0.6,   # Ngưỡng tin cậy tối thiểu để chấp nhận kết quả nhận diện
                                      # Được sử dụng trong thuật toán nhận diện để lọc kết quả không đáng tin cậy
    "absolute_distance_threshold": 0.6, # Ngưỡng khoảng cách tối đa cho phép (giới hạn cứng)
                                      # Đây là giới hạn cứng để ngăn nhận diện sai khi khoảng cách quá lớn
    "min_confidence": 0.6,             # Ngưỡng tin cậy tối thiểu (giữ để tương thích ngược)
    # Kích thước ảnh đầu vào
    "max_dim": 1080,                # Kích thước tối đa của ảnh đầu vào (chiều dài hoặc rộng)
                                    # Ảnh sẽ được resize để chiều lớn nhất không vượt quá giá trị này
    "face_size": (160, 160),        # Kích thước ảnh khuôn mặt sau khi cắt để đưa vào mô hình nhận diện
                                    # Được sử dụng trong face_embedding.py
    "target_size": (160, 160),      # Kích thước mục tiêu cho các module khác
                                    # Đảm bảo tính nhất quán giữa các module
    # Caching & worker
    "cache_size": 1000,            # Số lượng embedding tối đa được lưu trong bộ nhớ cache
                                    # Tăng giá trị này sẽ cải thiện hiệu suất nhưng tiêu tốn nhiều bộ nhớ hơn
    "max_workers": 4,              # Số lượng worker tối đa cho xử lý đa luồng
                                    # Tăng giá trị này có thể cải thiện hiệu suất trên CPU đa nhân
    # Camera
    "camera_id": 0,                # ID của camera được sử dụng (0: camera mặc định)
                                    # Thay đổi giá trị này nếu có nhiều camera được kết nối
    "camera_fps": 25,             # Tốc độ khung hình mục tiêu (frames per second)
                                    # Giảm giá trị này nếu hệ thống chạy chậm
    # GPU
    "use_gpu": True,               # Sử dụng GPU để tăng tốc xử lý (nếu có)
                                    # Đặt thành False nếu gặp vấn đề với GPU hoặc không có GPU
    # Điểm danh
    "attendance_cooldown": 300,    # Thời gian chờ giữa các lần điểm danh (tính bằng giây) - 5 phút
                                    # Ngăn điểm danh trùng lặp trong khoảng thời gian này
    "late_threshold": 0,           # Ngưỡng thời gian trễ (tính bằng phút)
                                    # Đánh dấu là đi trễ nếu điểm danh sau thời gian này
    # Model embedding
    "face_model": "Facenet512",     # Mô hình embedding khuôn mặt được sử dụng
                                    # Facenet512 cung cấp vector 512 chiều với độ chính xác cao
    # Cấu hình logging
    "logging_level": "ERROR",      # Mức độ log: DEBUG, INFO, WARNING, ERROR, CRITICAL
                                    # Đặt thành DEBUG để xem thông tin chi tiết khi gỡ lỗi
    "disable_console_logging": False, # Tắt hoàn toàn log ra console
                                    # Đặt thành True để tắt log ra console, chỉ ghi vào file
}
