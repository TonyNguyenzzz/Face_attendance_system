# camera_handler.py
import cv2
import numpy as np
import logging
from PyQt5.QtCore import QThread, pyqtSignal, QMutex, QMutexLocker

# Thiết lập logger với mức độ cao hơn để loại bỏ thông báo không cần thiết
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

class CameraHandler(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    
    def __init__(self, camera_id=0, fps=30):
        super().__init__()
        self.camera_id = camera_id
        self.fps = fps
        self.running = False
        self.frame = None
        self.lock = QMutex()

    def start(self):
        # Kiểm tra camera trước khi bắt đầu thread
        test_cap = cv2.VideoCapture(self.camera_id)
        if not test_cap.isOpened():
            test_cap.release()
            return False
        test_cap.release()
        
        # Bắt đầu thread nếu camera khả dụng
        super().start()
        return True

    def run(self):
        self.running = True
        cap = None
        try:
            cap = cv2.VideoCapture(self.camera_id)
            
            if not cap.isOpened():
                logger.error(f"Không thể mở camera ID {self.camera_id}")
                self.running = False
                return
                
            while self.running:
                try:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        logger.warning(f"Không đọc được frame từ camera ID {self.camera_id}")
                        # Thử kết nối lại camera sau 1 giây
                        self.msleep(1000)
                        continue
                        
                    with QMutexLocker(self.lock):
                        self.frame = frame.copy()
                    self.frame_ready.emit(frame)
                    
                    # Tính toán thời gian chờ dựa trên FPS
                    wait_time = int(1000 / self.fps)  # Chuyển đổi FPS thành milliseconds
                    self.msleep(wait_time)  # Giới hạn FPS theo cấu hình
                except cv2.error as e:
                    logger.error(f"OpenCV error: {str(e)}")
                    self.msleep(1000)  # Chờ 1 giây trước khi thử lại
                except Exception as e:
                    logger.error(f"Lỗi khi xử lý frame: {str(e)}")
                    self.msleep(1000)  # Chờ 1 giây trước khi thử lại
        except Exception as e:
            logger.error(f"Lỗi camera: {str(e)}")
        finally:
            if cap is not None:
                cap.release()
            self.running = False
            logger.info("Camera thread đã dừng")

    def stop(self):
        """Dừng luồng camera."""
        self.running = False
        self.wait()