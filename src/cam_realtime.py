import sys, os, cv2
import logging
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QHBoxLayout, QMessageBox, QStatusBar
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from camera_handler import CameraHandler
from face_detector import FaceDetector

# Thiết lập logger với mức độ cao hơn để loại bỏ thông báo không cần thiết
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

class RealtimeWindow(QMainWindow):
    def __init__(self, cam_id=0, model_path=None, fps=30):
        super().__init__()
        self.setWindowTitle("Real-time Face Detection")
        self.setMinimumSize(800, 600)  # Thiết lập kích thước tối thiểu
        self.detector = FaceDetector(model_path or os.path.join(os.path.dirname(__file__), "models", "model_3.onnx"))
        # Sử dụng tham số fps từ config
        self.camera = CameraHandler(camera_id=cam_id, fps=fps)
        self.camera.frame_ready.connect(self.update_frame)
        self.init_ui()
        
    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        # Video display
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        layout.addWidget(self.video_label)
        
        # Button layout
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self.start_camera)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_camera)
        self.stop_btn.setEnabled(False)  # Ban đầu vô hiệu hóa
        
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        btn_layout.addWidget(self.close_btn)
        layout.addLayout(btn_layout)
        
        # Thêm thanh trạng thái
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready")
        
    def start_camera(self):
        if not self.camera.isRunning():
            success = self.camera.start()
            if success:
                self.start_btn.setEnabled(False)
                self.stop_btn.setEnabled(True)
                self.statusBar.showMessage("Camera running")
            else:
                QMessageBox.critical(self, "Error", "Could not open camera. Please check your camera connection.")
                
    def stop_camera(self):
        if self.camera.isRunning():
            self.camera.stop()
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.statusBar.showMessage("Camera stopped")
            
    def update_frame(self, frame):
        # Detect faces if frame is not None
        if frame is not None:
            try:
                # Detect faces
                boxes, _ = self.detector.detect(frame, input_size=(640, 640))
                
                # Draw bounding boxes
                for box in boxes.astype(int):
                    x1, y1, x2, y2, _ = box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                
                # Convert to QPixmap
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                bytes_per_line = ch * w
                qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                
                # Scale while preserving aspect ratio
                pixmap = QPixmap.fromImage(qt_img)
                self.video_label.setPixmap(pixmap.scaled(
                    self.video_label.width(), self.video_label.height(), 
                    Qt.KeepAspectRatio, Qt.SmoothTransformation
                ))
                
                # Update status with detected face count
                face_count = len(boxes)
                self.statusBar.showMessage(f"Detected {face_count} face{'s' if face_count != 1 else ''}")
            except Exception as e:
                self.statusBar.showMessage(f"Error processing frame: {str(e)}")
                
    def closeEvent(self, event):
        # Ensure camera is stopped when window is closed
        if self.camera.isRunning():
            self.camera.stop()
        event.accept()

if __name__ == "__main__":
    # Import config để sử dụng các tham số
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from config.config import DEFAULT_CONFIG
    
    app = QApplication(sys.argv)
    # Sử dụng các tham số từ config
    window = RealtimeWindow(
        cam_id=DEFAULT_CONFIG.get("camera_id", 0),
        fps=DEFAULT_CONFIG.get("camera_fps", 30)
    )
    window.show()
    sys.exit(app.exec_())