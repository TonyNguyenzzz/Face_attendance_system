# camera_handler.py
import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, QMutex, QMutexLocker

class CameraHandler(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    
    def __init__(self, camera_id=0):
        super().__init__()
        self.camera_id = camera_id
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
        cap = cv2.VideoCapture(self.camera_id)
        
        if not cap.isOpened():
            self.running = False
            return
            
        while self.running:
            ret, frame = cap.read()
            if ret:
                with QMutexLocker(self.lock):
                    self.frame = frame.copy()
                self.frame_ready.emit(frame)
            else:
                # Xử lý trường hợp không đọc được frame
                break
            self.msleep(30)  # Giới hạn ~30 FPS
            
        cap.release()
        self.running = False

    def stop(self):
        """Dừng luồng camera."""
        self.running = False
        self.wait()