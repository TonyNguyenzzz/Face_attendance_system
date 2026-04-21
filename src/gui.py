"""
Face Recognition Attendance System - GUI Module

This module provides the main graphical user interface for the attendance system,
including camera management, user registration, reporting, and settings.

Author: Nguyen Gia Huy
Version: 1.0
"""

import sys
import os
import logging
import time
import copy
from datetime import datetime
from typing import Optional, List, Tuple, Dict, Any

import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QLabel, QLineEdit, QFormLayout,
    QTableWidget, QTableWidgetItem, QComboBox, QDateEdit, QSpinBox, QDoubleSpinBox,
    QFileDialog, QMessageBox, QFrame, QSplitter, QGroupBox, QCheckBox,
    QProgressBar, QProgressDialog, QScrollArea, QGridLayout, QSizePolicy, 
    QStatusBar, QHeaderView, QSlider
)
from PyQt5.QtGui import QBrush, QColor, QPixmap, QImage, QFont, QIcon
from PyQt5.QtCore import Qt, QTimer, QDate, QThread, pyqtSignal, QSize, QMutexLocker

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from camera_handler import CameraHandler
from database_handler import DatabaseHandler
from face_embedding import FaceEmbedding
from face_recognition import AttendanceSystem
from train_worker import TrainWorker
from config.config import DEFAULT_CONFIG

# Configure environment before other imports
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Setup logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("GUI")
logging.getLogger("FaceEmbedding").setLevel(logging.WARNING)
logging.getLogger("AttendanceSystem").setLevel(logging.WARNING)


class AIProcessingThread(QThread):
    """
    Thread for processing AI face recognition tasks asynchronously.
    
    This thread handles face detection and recognition to prevent blocking
    the main GUI thread during intensive AI computations.
    
    Attributes:
        processing_done: Signal emitted when processing completes with frame and faces
        attendance_system: The attendance system instance for processing
        current_frame: Current frame being processed
        running: Flag to control thread execution
        processing: Flag indicating if currently processing a frame
    """
    
    processing_done = pyqtSignal(np.ndarray, list)
    
    def __init__(self, attendance_system: AttendanceSystem) -> None:
        """
        Initialize the AI processing thread.
        
        Args:
            attendance_system: AttendanceSystem instance for face processing
        """
        super().__init__()
        self.attendance_system = attendance_system
        self.current_frame: Optional[np.ndarray] = None
        self.running: bool = True
        self.processing: bool = False
    
    def set_frame(self, frame: np.ndarray) -> None:
        """
        Set a new frame for processing if not currently busy.
        
        Args:
            frame: Image frame to process
        """
        if not self.processing:
            if frame is not None:
                try:
                    self.current_frame = frame.copy()
                    self.processing = True
                except Exception as e:
                    logger.error(f"Error copying frame in AIProcessingThread: {e}", exc_info=True)
                    self.current_frame = None
                    self.processing = False
            else:
                logger.warning("AIProcessingThread received a None frame. Skipping.")
    
    def run(self) -> None:
        """Main thread loop for processing frames."""
        while self.running:
            if self.current_frame is None:
                self.msleep(10)
                continue
            
            try:
                processed_frame, faces = self.attendance_system.process_image(self.current_frame)
                self.processing_done.emit(processed_frame, faces)
            except Exception as e:
                logger.error(f"AI processing error in thread: {e}", exc_info=True)
            finally:
                self.current_frame = None
                self.processing = False
                self.msleep(10)

STYLESHEET = """
QMainWindow {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ecf0f1, stop:1 #bdc3c7);
}
QTabBar::tab {
    background-color: #ecf0f1;
    color: #2c3e50;
    padding: 8px 15px;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
}
QTabBar::tab:selected, QTabBar::tab:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3498db, stop:1 #2980b9);
    color: white;
}
QPushButton {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3498db, stop:1 #2980b9);
    color: white;
    border-radius: 6px;
    padding: 10px;
}
QPushButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2980b9, stop:1 #1f6391);
}
QLineEdit, QComboBox, QDateEdit, QSpinBox {
    border: 1px solid #dcdcdc;
    border-radius: 4px;
    padding: 6px;
    background-color: #ffffff;
}
QLabel {
    color: #2c3e50;
}
QTableWidget {
    gridline-color: #e0e0e0;
    border: 1px solid #e0e0e0;
    border-radius: 5px;
}
QHeaderView::section {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3498db, stop:1 #2980b9);
    color: white;
    padding: 5px;
}
QGroupBox {
    border: 1px solid #bdc3c7;
    border-radius: 8px;
    background-color: #ffffff;
    font-weight: bold;
    padding: 10px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 5px;
    color: #3498db;
}
QProgressBar {
    border: 1px solid #bdc3c7;
    border-radius: 5px;
    text-align: center;
}
QProgressBar::chunk {
    background-color: #3498db;
    border-radius: 3px;
}
"""

class MainWindow(QMainWindow):
    """
    Main application window for the Face Recognition Attendance System.
    
    This class manages all UI components including tabs for home, registration,
    reporting, and settings. It handles camera operations, face recognition,
    database interactions, and user interface updates.
    
    Attributes:
        db: DatabaseHandler instance for data persistence
        config: Configuration dictionary
        attendance_system: AttendanceSystem instance for face processing
        camera_handler: CameraHandler instance for camera management
        ai_thread: AIProcessingThread for async AI processing
        stats_values: Dictionary storing statistics display labels
        trained_embedding: Trained face embedding model
    """
    
    def __init__(self) -> None:
        """Initialize the main window and all its components."""
        super().__init__()
        self.setWindowTitle("Hệ Thống Điểm Danh Bằng Nhận Diện Khuôn Mặt")
        self.setMinimumSize(1200, 800)
        
        # Initialize database handler with explicit connection name
        self.db = DatabaseHandler(connection_name="gui_connection")
        self.db.ensure_created_at_for_existing_users()
        
        # Load configuration
        self.config = copy.deepcopy(DEFAULT_CONFIG)
        
        # Initialize core systems
        self.attendance_system = AttendanceSystem(self.config, db_handler=self.db)
        self.camera_handler = CameraHandler(self.config.get("camera_id", 0))
        self.ai_thread = AIProcessingThread(self.attendance_system)
        
        # State variables
        self.stats_values: Dict[str, QLabel] = {}
        self.trained_embedding: Optional[np.ndarray] = None
        self.current_capture_frame: Optional[np.ndarray] = None
        self.last_recognition: Optional[Any] = None
        self.last_face_pixmap: Optional[QPixmap] = None
        self.is_processing: bool = False
        
        # Setup UI components
        self.setup_ui()
        
        # Setup status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Sẵn sàng", 5000)
        
        # Initialize timers
        self._setup_timers()
        
        # Setup connections and shortcuts
        self.setup_connections()
        self.init_shortcuts()
        
        # Connect tab change signal
        self.tabs.currentChanged.connect(self.on_tab_changed)
    
    def _setup_timers(self) -> None:
        """Initialize and start all application timers."""
        # Timer for updating attendance info (every 5 seconds)
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_attendance_info)
        self.update_timer.start(5000)
        
        # Timer for updating preview pixmap (10 FPS)
        self.preview_timer = QTimer(self)
        self.preview_timer.timeout.connect(self.update_preview_pixmap)
        self.preview_timer.start(100)
        
        # Timer for feeding frames to AI thread (10 FPS)
        self.frame_feed_timer = QTimer(self)
        self.frame_feed_timer.timeout.connect(self.feed_ai_thread_with_camera_frame)
        self.frame_feed_timer.start(100)
    
    def feed_ai_thread_with_camera_frame(self) -> None:
        """Feed current camera frame to AI processing thread if available."""
        if self.camera_handler.isRunning() and not self.ai_thread.processing:
            with QMutexLocker(self.camera_handler.lock):
                frame = self.camera_handler.frame
            if frame is not None:
                self.ai_thread.set_frame(frame)
    
    def _on_camera_frame(self, frame: np.ndarray) -> None:
        """Handle new camera frame event."""
        self.is_processing = True
        self.show_spinner(True)
    
    def init_shortcuts(self) -> None:
        """Initialize keyboard shortcuts for quick actions."""
        self.start_camera_btn.setShortcut("Ctrl+Shift+C")
        self.take_attendance_btn.setShortcut("Ctrl+Shift+A")
        if hasattr(self, 'register_btn'):
            self.register_btn.setShortcut("Ctrl+Shift+R")
        if hasattr(self, 'export_report_excel_btn'):
            self.export_report_excel_btn.setShortcut("Ctrl+Shift+E")
        if hasattr(self, 'export_report_pdf_btn'):
            self.export_report_pdf_btn.setShortcut("Ctrl+Shift+P")
    
    def show_spinner(self, show: bool) -> None:
        """
        Show or hide the loading spinner.
        
        Args:
            show: True to show spinner, False to hide
        """
        if hasattr(self, 'loading_spinner'):
            if show:
                self.loading_spinner.setText(
                    "<img src='https://i.imgur.com/llF5iyg.gif' width='48' height='48'> Đang xử lý AI..."
                )
                self.loading_spinner.show()
            else:
                self.loading_spinner.hide()
    
    def show_popup(self, message: str, status: str = "info") -> None:
        """
        Display a status message in the status bar.
        
        Args:
            message: Message to display
            status: Status type (info, warning, error) - currently unused
        """
        if hasattr(self, 'statusBar'):
            self.statusBar.showMessage(message, 5000)

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        header = QLabel("HỆ THỐNG ĐIỂM DANH BẰNG NHẬN DIỆN KHUÔN MẶT")
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("color: #2c3e50; font-size: 24px; font-weight: bold; margin: 15px;")
        main_layout.addWidget(header)
        self.main_header = header  # Store reference for dynamic dark mode styling
        
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        self.create_home_tab()
        self.create_register_tab()
        self.create_report_tab()
        self.create_settings_tab()
        
        # Loading spinner
        self.loading_spinner = QLabel()
        self.loading_spinner.setAlignment(Qt.AlignCenter)
        self.loading_spinner.setStyleSheet("background: transparent;")
        self.loading_spinner.hide()
        main_layout.addWidget(self.loading_spinner)
        
        footer = QLabel("2025 Face Recognition Attendance System - Phiên bản 1.0 - Nguyen Gia Huy")
        footer.setAlignment(Qt.AlignCenter)
        footer.setStyleSheet("color: #636e72; font-size: 22px; margin: 14px 0 8px 0; padding: 0; min-height: 32px; font-weight: bold;")
        main_layout.addWidget(footer)
        
        self.setStyleSheet(STYLESHEET)

    def create_home_tab(self):
        home_tab = QWidget()
        layout = QVBoxLayout(home_tab)
        
        # --- Thanh trạng thái hiện đại ---
        statusbar_layout = QHBoxLayout()
        # Tổng số SV
        self.stat_total_icon = QLabel()
        self.stat_total_icon.setPixmap(QIcon("icons/group.png").pixmap(38,38))
        self.stat_total_label = QLabel("Tổng: 0")
        self.stat_total_label.setStyleSheet("color: #f3f6fa; font-weight: bold; margin-right: 8px;")
        statusbar_layout.addWidget(self.stat_total_icon)
        statusbar_layout.addWidget(self.stat_total_label)
        # Đã điểm danh
        self.stat_present_icon = QLabel()
        self.stat_present_icon.setPixmap(QIcon("icons/check.png").pixmap(38,38))
        self.stat_present_label = QLabel("Đã điểm danh: 0")
        self.stat_present_label.setStyleSheet("color: #2ecc71; font-weight: bold; margin-right: 8px;")
        statusbar_layout.addWidget(self.stat_present_icon)
        statusbar_layout.addWidget(self.stat_present_label)
        # Vắng mặt
        self.stat_absent_icon = QLabel()
        self.stat_absent_icon.setPixmap(QIcon("icons/close.png").pixmap(38,38))
        self.stat_absent_label = QLabel("Vắng mặt: 0")
        self.stat_absent_label.setStyleSheet("color: #e74c3c; font-weight: bold; margin-right: 8px;")
        statusbar_layout.addWidget(self.stat_absent_icon)
        statusbar_layout.addWidget(self.stat_absent_label)
        statusbar_layout.addStretch()
        # Nút thao tác nhỏ
        self.stat_refresh_btn = QPushButton()
        self.stat_refresh_btn.setIcon(QIcon("icons/refresh.png"))
        self.stat_refresh_btn.setToolTip("Làm mới")
        self.stat_refresh_btn.setIconSize(QSize(38,38))
        self.stat_refresh_btn.setFixedSize(44,44)
        self.stat_export_btn = QPushButton()
        self.stat_export_btn.setIcon(QIcon("icons/bar_chart.png"))
        self.stat_export_btn.setToolTip("Xuất báo cáo")
        self.stat_export_btn.setIconSize(QSize(38,38))
        self.stat_export_btn.setFixedSize(44,44)
        statusbar_layout.addWidget(self.stat_refresh_btn)
        statusbar_layout.addWidget(self.stat_export_btn)
        layout.addLayout(statusbar_layout)
        # --- End thanh trạng thái ---
        
        splitter = QSplitter(Qt.Horizontal)
        
        # Camera group lớn hơn, info nhỏ hơn
        camera_group = QGroupBox("Camera Điểm Danh")
        camera_layout = QVBoxLayout(camera_group)
        self.camera_view = QLabel()
        self.camera_view.setFixedSize(800, 600)  # tăng kích thước camera
        self.camera_view.setAlignment(Qt.AlignCenter)
        camera_layout.addWidget(self.camera_view)
        
        self.start_camera_btn = QPushButton("Bắt đầu nhận diện")
        self.take_attendance_btn = QPushButton("Điểm danh thủ công")
        self.take_attendance_btn.setEnabled(True)  # Đảm bảo nút luôn được kích hoạt
        self.take_attendance_btn.clicked.connect(self.manual_attendance)  # Kết nối sự kiện click
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.start_camera_btn)
        btn_layout.addWidget(self.take_attendance_btn)
        camera_layout.addLayout(btn_layout)
        
        splitter.addWidget(camera_group)
        splitter.setStretchFactor(0, 3)  # Camera chiếm 3 phần
        
        info_group = QGroupBox("Bảng điều khiển")
        info_layout = QVBoxLayout(info_group)
        
        stats_layout = QHBoxLayout()
        self.stats_labels = [
            ("Tổng số sinh viên:", "#3498db"),
            ("Đã điểm danh:", "#2ecc71"),
            ("Vắng mặt:", "#e74c3c"),
            ("Tỷ lệ điểm danh:", "#f39c12")
        ]
        self.stats_values = {}
        for label_text, color in self.stats_labels:
            stat_frame = QFrame()
            stat_frame.setFrameShape(QFrame.StyledPanel)
            stat_frame.setStyleSheet(f"background-color: {color}; border-radius: 5px; padding: 10px;")
            stat_vbox = QVBoxLayout(stat_frame)
            label = QLabel(label_text)
            label.setStyleSheet("color: white; font-weight: bold;")
            label.setAlignment(Qt.AlignCenter)
            value = QLabel("0")
            value.setStyleSheet("color: white; font-size: 24px; font-weight: bold;")
            value.setAlignment(Qt.AlignCenter)
            self.stats_values[label_text] = value
            stat_vbox.addWidget(label)
            stat_vbox.addWidget(value)
            stats_layout.addWidget(stat_frame)
        info_layout.addLayout(stats_layout)
        
        recent_attendance_label = QLabel("Danh sách điểm danh hôm nay:")
        recent_attendance_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        info_layout.addWidget(recent_attendance_label)
        
        # Nút làm mới danh sách điểm danh
        refresh_layout = QHBoxLayout()
        self.refresh_attendance_btn = QPushButton("Làm mới")
        self.refresh_attendance_btn.setIcon(QIcon("icons/refresh.png"))
        self.clear_today_attendance_btn = QPushButton("Xoá DS hôm nay")
        refresh_layout.addStretch()
        refresh_layout.addWidget(self.refresh_attendance_btn)
        refresh_layout.addWidget(self.clear_today_attendance_btn)
        info_layout.addLayout(refresh_layout)

        self.clear_today_attendance_btn.clicked.connect(self.on_clear_today_attendance_clicked)
        
        self.recent_attendance_table = QTableWidget(0, 4)
        self.recent_attendance_table.setHorizontalHeaderLabels(["Tên", "Mã SV", "Thời gian", "Trạng thái"])
        self.recent_attendance_table.horizontalHeader().setStretchLastSection(True)
        self.recent_attendance_table.setEditTriggers(QTableWidget.NoEditTriggers)
        info_layout.addWidget(self.recent_attendance_table)
        
        splitter.addWidget(info_group)
        splitter.setStretchFactor(1, 1)  # Info nhỏ hơn
        
        layout.addWidget(splitter)
        
        self.tabs.addTab(home_tab, "Trang chính")

        # Khởi tạo số liệu ban đầu
        self.update_statusbar_stats(0, 0, 0)
        self.stat_refresh_btn.clicked.connect(self.update_attendance_info)
        self.stat_export_btn.clicked.connect(self.export_report_to_excel)

    def update_statusbar_stats(self, total: int, present: int, absent: int) -> None:
        """
        Update the status bar statistics display.
        
        Args:
            total: Total number of users
            present: Number of users present today
            absent: Number of users absent today
        """
        self.stat_total_label.setText(f"Tổng: {total}")
        self.stat_present_label.setText(f"Đã điểm danh: {present}")
        self.stat_absent_label.setText(f"Vắng mặt: {absent}")


    def create_register_tab(self):
        # ... [existing code above remains unchanged]

        register_tab = QWidget()
        layout = QVBoxLayout(register_tab)
        
        register_header = QLabel("ĐĂNG KÝ NGƯỜI DÙNG MỚI")
        register_header.setAlignment(Qt.AlignCenter)
        register_header.setStyleSheet("font-size: 22px; font-weight: bold; color: #3498db; margin: 15px 0;")
        layout.addWidget(register_header)
        
        splitter = QSplitter(Qt.Horizontal)
        
        form_group = QGroupBox("Thông tin người dùng")
        form_layout = QFormLayout(form_group)
        self.name_input = QLineEdit()
        self.name_input.setToolTip("Nhập họ và tên đầy đủ")
        self.student_id_input = QLineEdit()
        self.student_id_input.setToolTip("Nhập mã sinh viên duy nhất")
        self.class_input = QComboBox()
        self.class_input.setEditable(True)  # Cho phép nhập lớp mới nếu chưa có trong danh sách
        self.email_input = QLineEdit()
        form_layout.addRow("Họ và tên:", self.name_input)
        form_layout.addRow("Mã sinh viên:", self.student_id_input)
        form_layout.addRow("Lớp:", self.class_input)
        form_layout.addRow("Email:", self.email_input)
        splitter.addWidget(form_group)
        
        face_capture_group = QGroupBox("Thu thập dữ liệu khuôn mặt")
        face_capture_layout = QVBoxLayout(face_capture_group)
        
        self.face_preview = QLabel()
        self.face_preview.setFixedSize(320, 240)
        self.face_preview.setAlignment(Qt.AlignCenter)
        self.face_preview.setStyleSheet("border: 2px solid #e0e0e0; background-color: #000000; border-radius: 5px;")
        self.face_preview.setText("Nhấn 'Bắt đầu thu thập' để kích hoạt camera")
        face_capture_layout.addWidget(self.face_preview)
        
        self.captured_images_layout = QHBoxLayout()
        self.captured_images = []
        for i in range(5):
            img_label = QLabel()
            img_label.setFixedSize(80, 60)
            img_label.setStyleSheet("border: 1px solid #e0e0e0; background-color: #ffffff; border-radius: 3px;")
            self.captured_images.append(img_label)
            self.captured_images_layout.addWidget(img_label)
        face_capture_layout.addLayout(self.captured_images_layout)
        
        capture_controls = QHBoxLayout()
        self.start_capture_btn = QPushButton("Bắt đầu thu thập")
        self.start_capture_btn.setIcon(QIcon("icons/camera.png"))
        self.start_capture_btn.setToolTip("Bật camera để bắt đầu thu thập")
        self.capture_btn = QPushButton("Chụp ảnh")
        self.capture_btn.setEnabled(False)
        self.capture_btn.setIcon(QIcon("icons/capture.png"))
        self.retake_btn = QPushButton("Chụp lại")
        self.retake_btn.setEnabled(False)
        self.retake_btn.setIcon(QIcon("icons/refresh.png"))
        self.upload_image_btn = QPushButton("Tải ảnh lên")
        self.upload_image_btn.setIcon(QIcon("icons/upload.png"))
        capture_controls.addWidget(self.start_capture_btn)
        capture_controls.addWidget(self.capture_btn)
        capture_controls.addWidget(self.retake_btn)
        capture_controls.addWidget(self.upload_image_btn)
        face_capture_layout.addLayout(capture_controls)
        # Kết nối nút chụp ảnh, chụp lại và tải ảnh
        self.capture_btn.clicked.connect(self.capture_image)
        self.retake_btn.clicked.connect(self.retake_image)
        self.upload_image_btn.clicked.connect(self.upload_images)
        
        self.capture_progress = QProgressBar()
        self.capture_progress.setRange(0, 30)
        self.capture_progress.setValue(0)
        self.capture_status = QLabel("0/30 ảnh đã thu thập")
        self.capture_status.setAlignment(Qt.AlignCenter)
        face_capture_layout.addWidget(self.capture_progress)
        face_capture_layout.addWidget(self.capture_status)
        
        splitter.addWidget(face_capture_group)
        
        layout.addWidget(splitter)
        
        register_btn_layout = QHBoxLayout()
        register_btn_layout.addStretch()
        self.register_btn = QPushButton("Đăng ký người dùng")
        self.register_btn.setIcon(QIcon("icons/user-plus.png"))
        self.register_btn.setMinimumWidth(200)
        self.register_btn.setStyleSheet("background-color: #27ae60; font-weight: bold; padding: 10px; font-size: 14px;")
        self.register_btn.setEnabled(False)
        register_btn_layout.addWidget(self.register_btn)
        # Kết nối nút đăng ký người dùng
        self.register_btn.clicked.connect(self.register_user)
        
        self.train_btn = QPushButton("Huấn luyện ảnh")
        self.train_btn.setMinimumWidth(200)
        self.train_btn.setEnabled(False)
        register_btn_layout.addWidget(self.train_btn)
        # Kết nối nút huấn luyện ảnh
        self.train_btn.clicked.connect(self.train_images)
        
        register_btn_layout.addStretch()
        layout.addLayout(register_btn_layout)
        
        users_group = QGroupBox("Danh sách người dùng đã đăng ký")
        users_layout = QVBoxLayout(users_group)
        
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Tìm kiếm:"))
        self.user_search = QLineEdit()
        self.user_search.setPlaceholderText("Nhập tên hoặc mã sinh viên...")
        filter_layout.addWidget(self.user_search)
        # Kết nối tìm kiếm realtime
        self.user_search.textChanged.connect(self.filter_users_table)
        # Kết nối nút bắt đầu thu thập với toggle_capture
        self.start_capture_btn.clicked.connect(self.toggle_capture)
        
        self.users_table = QTableWidget(0, 6)
        self.users_table.setHorizontalHeaderLabels(["STT", "Tên", "Mã SV", "Lớp", "Email", "Thời gian đăng ký"])
        self.users_table.horizontalHeader().setStretchLastSection(True)
        self.users_table.setEditTriggers(QTableWidget.NoEditTriggers)
        # Connect table row selection to input population
        self.users_table.itemSelectionChanged.connect(self.populate_user_inputs)

        actions_layout = QHBoxLayout()
        self.edit_user_btn = QPushButton("Sửa")
        self.edit_user_btn.setIcon(QIcon("icons/edit.png"))
        self.delete_user_btn = QPushButton("Xóa")
        self.delete_user_btn.setIcon(QIcon("icons/trash.png"))
        self.delete_user_btn.setStyleSheet("background-color: #e74c3c;")
        actions_layout.addStretch()
        actions_layout.addWidget(self.edit_user_btn)
        actions_layout.addWidget(self.delete_user_btn)
        
        users_layout.addLayout(filter_layout)
        users_layout.addWidget(self.users_table)
        users_layout.addLayout(actions_layout)
        
        layout.addWidget(users_group)
        
        self.tabs.addTab(register_tab, "Đăng ký người dùng")
        # Luôn hiển thị danh sách user khi khởi tạo tab
        self.filter_users_table()

    def upload_images(self):
        """
        Cho phép người dùng chọn và tải lên ảnh tĩnh để huấn luyện khuôn mặt.
        Ảnh sẽ được lưu vào thư mục dữ liệu của người dùng theo mã sinh viên và lớp.
        """
        import cv2
        name = self.name_input.text().strip()
        student_id = self.student_id_input.text().strip()
        class_name = self.class_input.currentText().strip()
        if not name or not student_id or not class_name:
            QMessageBox.warning(self, "Thiếu thông tin", "Vui lòng nhập đầy đủ họ tên, mã sinh viên và lớp trước khi tải ảnh.")
            return
        files, _ = QFileDialog.getOpenFileNames(self, "Chọn ảnh khuôn mặt", "", "Images (*.png *.jpg *.jpeg)")
        if not files:
            return
        # Đọc ảnh, detect và crop vùng mặt vào self.captured_frames
        self.captured_frames = []
        skipped = 0
        for file_path in files:
            img = cv2.imread(file_path)
            if img is not None:
                faces = self.attendance_system.face_embedding.detect_faces(img)
                if faces:
                    # Chỉ lấy khuôn mặt đầu tiên (nếu có nhiều khuôn mặt)
                    x1, y1, x2, y2 = faces[0]["bbox"]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(img.shape[1] - 1, x2), min(img.shape[0] - 1, y2)
                    if x2 > x1 and y2 > y1:
                        face_img = img[y1:y2, x1:x2]
                        self.captured_frames.append(face_img)
                    else:
                        # logger.warning(f"Ảnh {file_path} có bbox không hợp lệ: [{x1},{y1},{x2},{y2}]")
                        skipped += 1
                else:
                    # logger.warning(f"Không phát hiện khuôn mặt trong ảnh: {file_path}")
                    skipped += 1
        if len(self.captured_frames) == 0:
            QMessageBox.warning(self, "Lỗi", "Không phát hiện được khuôn mặt nào trong các file đã chọn.")
        else:
            self.train_btn.setEnabled(True)
            msg = f"Đã chọn {len(self.captured_frames)} ảnh tĩnh. Đang huấn luyện..."
            if skipped > 0:
                msg += f" (Bỏ qua {skipped} ảnh không có khuôn mặt hợp lệ)"
            self.capture_status.setText(msg)
            self.train_images()


    def create_report_tab(self):
        report_tab = QWidget()
        layout = QVBoxLayout(report_tab)
        
        report_header = QLabel("BÁO CÁO VÀ THỐNG KÊ")
        report_header.setAlignment(Qt.AlignCenter)
        report_header.setStyleSheet("font-size: 22px; font-weight: bold; color: #3498db; margin: 15px 0;")
        layout.addWidget(report_header)
        
        chart_group = QGroupBox("Thống kê điểm danh")
        chart_layout = QVBoxLayout(chart_group)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        chart_layout.addWidget(self.canvas)
        self.update_chart()
        
        filter_group = QGroupBox("Bộ lọc báo cáo")
        filter_layout = QGridLayout(filter_group)
        
        filter_layout.addWidget(QLabel("Từ ngày:"), 0, 0)
        self.from_date = QDateEdit()
        self.from_date.setDate(QDate.currentDate().addDays(-30))
        self.from_date.setCalendarPopup(True)
        filter_layout.addWidget(self.from_date, 0, 1)
        
        filter_layout.addWidget(QLabel("Đến ngày:"), 0, 2)
        self.to_date = QDateEdit()
        self.to_date.setDate(QDate.currentDate())
        self.to_date.setCalendarPopup(True)
        filter_layout.addWidget(self.to_date, 0, 3)
        
        filter_layout.addWidget(QLabel("Lớp:"), 1, 0)
        self.class_filter = QComboBox()
        self.class_filter.addItem("Tất cả")
        filter_layout.addWidget(self.class_filter, 1, 1)
        
        filter_layout.addWidget(QLabel("Trạng thái:"), 1, 2)
        self.status_filter = QComboBox()
        self.status_filter.addItems(["Tất cả", "Có mặt", "Vắng mặt", "Đi muộn"])
        filter_layout.addWidget(self.status_filter, 1, 3)
        
        self.apply_filter_btn = QPushButton("Áp dụng bộ lọc")
        self.apply_filter_btn.setIcon(QIcon("icons/filter.png"))
        filter_layout.addWidget(self.apply_filter_btn, 2, 3)
        
        report_table_group = QGroupBox("Dữ liệu báo cáo")
        report_table_layout = QVBoxLayout(report_table_group)
        
        self.report_table = QTableWidget(0, 6)
        self.report_table.setHorizontalHeaderLabels(["Tên", "Mã SV", "Lớp", "Ngày", "Thời gian", "Trạng thái"])
        self.report_table.horizontalHeader().setStretchLastSection(True)
        self.report_table.setEditTriggers(QTableWidget.NoEditTriggers)
        
        export_layout = QHBoxLayout()
        export_layout.addStretch()
        self.export_report_excel_btn = QPushButton("Xuất Excel")
        self.export_report_excel_btn.setIcon(QIcon("icons/excel.png"))
        self.export_report_pdf_btn = QPushButton("Xuất PDF")
        self.export_report_pdf_btn.setIcon(QIcon("icons/pdf.png"))
        export_layout.addWidget(self.export_report_excel_btn)
        export_layout.addWidget(self.export_report_pdf_btn)
        
        report_table_layout.addWidget(self.report_table)
        report_table_layout.addLayout(export_layout)
        
        summary_group = QGroupBox()
        summary_vlayout = QVBoxLayout()
        summary_vlayout.setContentsMargins(10, 10, 10, 10)
        summary_group.setLayout(summary_vlayout)
        summary_group.setMinimumHeight(110)

        summary_title = QLabel("Tổng kết")
        summary_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #222; margin-bottom: 6px;")
        summary_title.setAlignment(Qt.AlignLeft)
        summary_vlayout.addWidget(summary_title)

        summary_layout = QHBoxLayout()
        summary_layout.setSpacing(20)
        summary_layout.setContentsMargins(5, 0, 5, 0)
        
        stats_items = [
            ("Tổng số sinh viên:", "0", "#3498db"),
            ("Tổng số buổi học:", "0", "#9b59b6"),
            ("Tỷ lệ đi học đầy đủ:", "0%", "#2ecc71"),
            ("Tỷ lệ vắng mặt:", "0%", "#e74c3c")
        ]
        
        self.summary_values = {}
        for i, (label_text, value_text, color) in enumerate(stats_items):
            stat_frame = QFrame()
            stat_frame.setFrameShape(QFrame.StyledPanel)
            stat_frame.setStyleSheet(f"background-color: {color}; border-radius: 8px; padding: 12px;")
            stat_layout = QVBoxLayout()
            stat_layout.setSpacing(4)
            stat_layout.setContentsMargins(8, 4, 8, 4)
            stat_frame.setLayout(stat_layout)
            
            label = QLabel(label_text)
            label.setStyleSheet("font-size: 14px; color: white; font-weight: bold;")
            label.setAlignment(Qt.AlignCenter)
            
            value = QLabel(value_text)
            value.setStyleSheet("font-size: 22px; font-weight: bold; color: white;")
            value.setAlignment(Qt.AlignCenter)
            self.summary_values[label_text] = value
            
            stat_layout.addWidget(label)
            stat_layout.addWidget(value)
            stat_frame.setMinimumWidth(180)
            stat_frame.setMinimumHeight(60)
            summary_layout.addWidget(stat_frame)
        summary_vlayout.addLayout(summary_layout)

        layout.addWidget(chart_group)
        layout.addWidget(filter_group)
        layout.addWidget(report_table_group)
        layout.addWidget(summary_group)
        summary_group.setVisible(True)

        
        self.tabs.addTab(report_tab, "Báo cáo")
        # Kết nối nút áp dụng bộ lọc với hàm apply_filter
        self.apply_filter_btn.clicked.connect(self.apply_filter)
        # Gọi apply_filter ngay khi tạo tab báo cáo
        self.apply_filter()
        # Đảm bảo summary_group luôn hiển thị
        summary_group.setVisible(True)

    def create_settings_tab(self):
        settings_tab = QWidget()
        settings_layout = QVBoxLayout(settings_tab)
        
        # Camera settings
        camera_settings = QGroupBox("Cài đặt camera")
        camera_settings_layout = QFormLayout(camera_settings)
        
        self.camera_source = QComboBox()
        self.camera_source.addItems(["Camera 0", "Camera 1", "Camera 2"])
        self.camera_source.setCurrentIndex(self.config.get("camera_id", 0))
        camera_settings_layout.addRow("Nguồn camera:", self.camera_source)
        
        self.camera_fps = QSpinBox()
        self.camera_fps.setRange(1, 60)
        self.camera_fps.setValue(self.config.get("camera_fps", 30))
        camera_settings_layout.addRow("FPS:", self.camera_fps)
        
        self.camera_resolution = QComboBox()
        self.camera_resolution.addItems(["640x480", "1280x720", "1920x1080"])
        current_res = f"{self.config.get('max_dim', 1080)}x{self.config.get('max_dim', 1080)}"
        self.camera_resolution.setCurrentText(current_res)
        camera_settings_layout.addRow("Độ phân giải:", self.camera_resolution)
        
        settings_layout.addWidget(camera_settings)
        
        # Face recognition settings
        face_settings = QGroupBox("Cài đặt nhận diện khuôn mặt")
        face_settings_layout = QFormLayout(face_settings)
        
        # --- Cài đặt ngưỡng khoảng cách ---
        distance_label = QLabel("Ngưỡng nhận diện khuôn mặt:")
        self.distance_spinbox = QDoubleSpinBox()
        self.distance_spinbox.setRange(0.1, 0.9)
        self.distance_spinbox.setSingleStep(0.01)
        self.distance_spinbox.setValue(self.config.get('recognition_distance', 0.35))
        self.distance_spinbox.setDecimals(2)
        face_settings_layout.addRow(distance_label, self.distance_spinbox)
        
        # --- Cài đặt ngưỡng tin cậy tối thiểu ---
        min_confidence_label = QLabel("Ngưỡng tin cậy tối thiểu:")
        self.min_confidence_spinbox = QDoubleSpinBox()
        self.min_confidence_spinbox.setRange(0.4, 0.9)
        self.min_confidence_spinbox.setSingleStep(0.05)
        self.min_confidence_spinbox.setValue(self.config.get('min_confidence_threshold', 0.6))
        self.min_confidence_spinbox.setDecimals(2)
        self.min_confidence_spinbox.setSuffix("")
        face_settings_layout.addRow(min_confidence_label, self.min_confidence_spinbox)
        
        # --- Cài đặt ngưỡng khoảng cách tuyệt đối ---
        abs_distance_label = QLabel("Ngưỡng khoảng cách tuyệt đối:")
        self.abs_distance_spinbox = QDoubleSpinBox()
        self.abs_distance_spinbox.setRange(0.3, 0.9)
        self.abs_distance_spinbox.setSingleStep(0.05)
        self.abs_distance_spinbox.setValue(self.config.get('absolute_distance_threshold', 0.6))
        self.abs_distance_spinbox.setDecimals(2)
        face_settings_layout.addRow(abs_distance_label, self.abs_distance_spinbox)
        
        # --- Ngưỡng tin cậy hiện tại ---
        self.confidence_threshold = QSpinBox()
        self.confidence_threshold.setRange(30, 100)
        conf_val = int(self.config.get('recognition_confidence', 0.85) * 100)
        self.confidence_threshold.setValue(conf_val)
        self.confidence_threshold.setSuffix("%")
        face_settings_layout.addRow("Ngưỡng tin cậy:", self.confidence_threshold)
        
        # Thêm nút kiểm tra ngưỡng
        test_threshold_btn = QPushButton("Kiểm tra ngưỡng nhận diện")
        test_threshold_btn.setIcon(QIcon("icons/test.png"))
        face_settings_layout.addRow("", test_threshold_btn)
        
        settings_layout.addWidget(face_settings)
        
        # Database settings
        db_settings = QGroupBox("Cài đặt cơ sở dữ liệu")
        db_settings_layout = QFormLayout(db_settings)
        
        backup_btn = QPushButton("Sao lưu cơ sở dữ liệu")
        backup_btn.clicked.connect(self.backup_database)
        db_settings_layout.addRow("", backup_btn)
        
        restore_btn = QPushButton("Khôi phục cơ sở dữ liệu")
        restore_btn.clicked.connect(self.restore_database)
        db_settings_layout.addRow("", restore_btn)
        
        clear_btn = QPushButton("Xóa dữ liệu điểm danh")
        clear_btn.clicked.connect(self.clear_attendance_data)
        db_settings_layout.addRow("", clear_btn)
        
        settings_layout.addWidget(db_settings)
        
        # Save button
        save_btn = QPushButton("Lưu cài đặt")
        save_btn.clicked.connect(self.save_settings)
        settings_layout.addWidget(save_btn)
        
        # Kết nối sự kiện cho nút kiểm tra ngưỡng
        test_threshold_btn.clicked.connect(self.test_recognition_thresholds)
        
        self.tabs.addTab(settings_tab, "Cài đặt")

    def save_settings(self):
        try:
            self.config["camera_id"] = self.camera_source.currentIndex()
            self.config["camera_fps"] = self.camera_fps.value()
            resolution = self.camera_resolution.currentText()
            self.config["max_dim"] = int(resolution.split("x")[1])
            
            # Cập nhật các ngưỡng nhận diện mới
            self.config["recognition_distance"] = self.distance_spinbox.value()
            self.config["recognition_confidence"] = self.confidence_threshold.value() / 100.0
            self.config["min_confidence_threshold"] = self.min_confidence_spinbox.value()
            self.config["absolute_distance_threshold"] = self.abs_distance_spinbox.value()
            
            # Cập nhật cấu hình cho attendance_system
            self.attendance_system.config.update(self.config)
            
            # Cập nhật các tham số cho FaceEmbedding
            if hasattr(self.attendance_system, 'face_embedding'):
                self.attendance_system.face_embedding.config.distance_threshold = self.config["recognition_distance"]
                
            QMessageBox.information(self, "Thành công", "Đã lưu cài đặt. Các thay đổi sẽ có hiệu lực ngay lập tức.")
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Không thể lưu cài đặt: {e}")

    def test_recognition_thresholds(self):
        """Kiểm tra các ngưỡng khác nhau với ảnh hiện tại"""
        # Kiểm tra xem có frame hiện tại không
        if not hasattr(self, 'current_capture_frame') or self.current_capture_frame is None:
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng bật camera trước khi kiểm tra ngưỡng.")
            return
            
        frame = self.current_capture_frame.copy()
        
        # Phát hiện khuôn mặt
        faces = self.attendance_system.face_embedding.detect_faces(frame)
        if not faces:
            QMessageBox.warning(self, "Cảnh báo", "Không phát hiện được khuôn mặt trong khung hình.")
            return
            
        # Lấy khuôn mặt đầu tiên
        bbox = faces[0]["bbox"]
        x1, y1, x2, y2 = bbox
        face_img = frame[y1:y2, x1:x2]
        
        # Kiểm tra các ngưỡng khác nhau
        results = self.attendance_system.face_embedding.test_recognition_thresholds(
            face_img, distance_range=(0.3, 0.9, 0.05))
            
        # Hiển thị kết quả
        if not results:
            QMessageBox.warning(self, "Lỗi", "Không thể trích xuất embedding từ khuôn mặt.")
            return
            
        # Tạo báo cáo
        report = "Kết quả kiểm tra ngưỡng khoảng cách:\n\n"
        report += "| Ngưỡng | Nhận diện | Độ tin cậy |\n"
        report += "|--------|-----------|------------|\n"
        
        for r in results:
            threshold = r["threshold"]
            name = r["result"]
            confidence = r["confidence"]
            report += f"| {threshold:.2f} | {name} | {confidence:.2f} |\n"
            
        # Hiển thị báo cáo
        QMessageBox.information(self, "Kết quả kiểm tra ngưỡng", report)

    def update_distance_threshold(self, value):
        self.config['recognition_distance'] = value
        # Cập nhật trực tiếp vào hệ thống nhận diện nếu đã khởi tạo
        if hasattr(self.attendance_system, 'face_embedding'):
            self.attendance_system.face_embedding.config.distance_threshold = value
        # # logger.info(f"Cập nhật ngưỡng nhận diện: {value}")
        # Cập nhật label giá trị động nếu có
        if hasattr(self, 'distance_value_label'):
            self.distance_value_label.setText(f"{float(value):.2f}")

    def update_confidence_threshold(self, value):
        self.config['recognition_confidence'] = value / 100.0
        # # logger.info(f"Cập nhật ngưỡng tin cậy: {self.config['recognition_confidence']}")
        if hasattr(self.attendance_system, 'face_embedding'):
            self.attendance_system.face_embedding.config.recognition_confidence = self.config['recognition_confidence']


    def _on_distance_slider_changed(self, slider_value):
        val = slider_value / 100.0
        # Cập nhật spinbox và label
        if abs(self.distance_spinbox.value() - val) > 1e-6:
            self.distance_spinbox.setValue(val)
        self.distance_value_label.setText(f"{val:.2f}")
        self.update_distance_threshold(val)

    def _on_distance_spinbox_changed(self, spin_value):
        slider_val = int(round(spin_value * 100))
        if self.distance_slider.value() != slider_val:
            self.distance_slider.setValue(slider_val)
        self.distance_value_label.setText(f"{spin_value:.2f}")
        self.update_distance_threshold(spin_value)


    def set_high_distance_threshold(self):
        self.update_distance_threshold(0.8)


    def setup_connections(self):
        self.start_camera_btn.clicked.connect(self.toggle_camera)
        self.take_attendance_btn.clicked.connect(self.manual_attendance)
        if hasattr(self, 'refresh_attendance_btn'):
            self.refresh_attendance_btn.clicked.connect(self.update_attendance_info)
        self.camera_handler.frame_ready.connect(self._on_camera_frame)
        self.ai_thread.processing_done.connect(self.update_frame)
        if hasattr(self, 'export_report_excel_btn'):
            self.export_report_excel_btn.clicked.connect(self.export_report_to_excel)
        if hasattr(self, 'export_report_pdf_btn'):
            self.export_report_pdf_btn.clicked.connect(self.export_report_to_pdf)
        if hasattr(self, 'save_settings_btn'):
            self.save_settings_btn.clicked.connect(self.save_settings)
        if hasattr(self, 'backup_db_btn'):
            self.backup_db_btn.clicked.connect(self.backup_database)
        if hasattr(self, 'restore_db_btn'):
            self.restore_db_btn.clicked.connect(self.restore_database)
        if hasattr(self, 'clear_attendance_btn'):
            self.clear_attendance_btn.clicked.connect(self.clear_attendance_data)
        if hasattr(self, 'edit_user_btn'):
            self.edit_user_btn.clicked.connect(self.edit_user)
        if hasattr(self, 'delete_user_btn'):
            self.delete_user_btn.clicked.connect(self.delete_user)

    def toggle_camera(self):
        try:
            if not self.camera_handler.isRunning():
                self.statusBar.showMessage("Đang bật camera...")
                self.start_camera_btn.setText("Dừng camera")
                self.take_attendance_btn.setEnabled(True)
                self.camera_handler.start()
                self.is_processing = True
                self.show_spinner(True)
                if not self.ai_thread.isRunning():
                    self.ai_thread.running = True
                    self.ai_thread.start()
            else:
                self.statusBar.showMessage("Camera đã dừng.")
                self.start_camera_btn.setText("Bắt đầu nhận diện")
                self.take_attendance_btn.setEnabled(False)
                self.ai_thread.running = False
                self.camera_handler.stop()
                self.camera_view.clear()
                self.show_spinner(False)
        except Exception as e:
            self.show_popup(f"Không thể điều khiển camera: {e}", "error")


    def update_frame(self, frame: np.ndarray, faces: list):
        try:
            self.is_processing = False
            pix = self.convert_cv_qt(frame)
            if pix.isNull():
                return
            self.camera_view.setPixmap(pix)
            # Hiển thị thông tin nhận diện realtime
            if faces:
                face = faces[0]
                sid = face.get("student_id")
                name = face.get("name")
                conf = face.get("confidence", 0)
                bbox = face.get("bbox")
                # Hiển thị ảnh khuôn mặt và viền màu theo trạng thái
                if bbox is not None:
                    x1, y1, x2, y2 = map(int, bbox)
                    face_img = frame[y1:y2, x1:x2].copy() if y2 > y1 and x2 > x1 else frame
                    face_pix = self.convert_cv_qt(face_img)
                    self.last_face_pixmap = face_pix
                else:
                    self.last_face_pixmap = None
                if sid:
                    self.last_recognition = {
                        "student_id": sid,
                        "name": name,
                        "confidence": conf,
                        "class": face.get("class", None),
                        "status": face.get("status", "present")
                    }
                    self.statusBar.showMessage(f"Đã nhận diện: {name} ({sid}) | Độ tin cậy: {conf:.2f}", 5000)
                    self.show_popup(f"Nhận diện thành công: {name} ({sid})\nĐộ tin cậy: {conf:.2f}", "success")
                else:
                    self.last_recognition = None
                    self.statusBar.showMessage("Không nhận diện được khuôn mặt", 3000)
                    self.show_popup("Không nhận diện được khuôn mặt!", "error")
            else:
                self.last_face_pixmap = None
                self.last_recognition = None
        except Exception as e:
            # logger.error(f"Frame update error: {e}")
            self.show_popup(f"Lỗi cập nhật frame: {e}", "error")


    def manual_attendance(self):
        try:
            # Chỉ điểm danh khuôn mặt đang nhận diện gần nhất
            if hasattr(self, 'last_recognition') and self.last_recognition is not None:
                sid = self.last_recognition.get("student_id")
                name = self.last_recognition.get("name")
                conf = self.last_recognition.get("confidence")
                if sid and name and conf is not None:
                    success = self.attendance_system.record_attendance(sid, name, conf, force=True)
                    if success:
                        self.statusBar.showMessage(f"Đã điểm danh cho {name} ({sid})", 3000)
                        self.update_attendance_info()
                    else:
                        self.statusBar.showMessage(f"Không thể điểm danh cho {name} ({sid})", 3000)
                else:
                    QMessageBox.warning(self, "Lỗi", "Không có khuôn mặt hợp lệ để điểm danh.")
            else:
                QMessageBox.warning(self, "Lỗi", "Không có khuôn mặt hợp lệ để điểm danh.")
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Không thể điểm danh thủ công: {e}")

    def convert_cv_qt(self, cv_img):
        try:
            rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            scaled = qt_img.scaled(self.camera_view.width(), self.camera_view.height(), Qt.KeepAspectRatio)
            return QPixmap.fromImage(scaled)
        except Exception as e:
            # logger.error(f"Error in convert_cv_qt: {e}")
            return QPixmap()

    def toggle_capture(self):
        try:
            if not hasattr(self, 'capture_camera') or not self.capture_camera.isRunning():
                self.capture_camera = CameraHandler(self.config.get("camera_id", 0))
                self.capture_camera.frame_ready.connect(self.update_face_preview)
                self.capture_thread = QThread()
                self.capture_camera.moveToThread(self.capture_thread)
                self.capture_thread.started.connect(self.capture_camera.run)
                self.capture_thread.start()
                self.start_capture_btn.setText('Dừng thu thập')
                self.capture_auto = True
                self.capture_target = 30
                self.capture_index = 0
                self.captured_frames = []
                self.capture_progress.setMaximum(30)
                self.capture_progress.setValue(0)
                self.capture_status.setText(f"0/{30} ảnh đã thu thập")
                self.register_btn.setEnabled(False)
                self.train_btn.setEnabled(False)
                self.capture_btn.setEnabled(True)
                self.retake_btn.setEnabled(True)
            else:
                self.capture_camera.stop()
                self.capture_thread.quit()
                self.capture_thread.wait()
                self.start_capture_btn.setText('Bắt đầu thu thập')
                self.capture_btn.setEnabled(False)
                self.retake_btn.setEnabled(False)
                self.face_preview.clear()
                self.capture_auto = False
        except Exception as e:
            # logger.error(f"Error in toggle_capture: {e}")
            QMessageBox.critical(self, "Lỗi", f"Không thể thực hiện thu thập: {e}")

    def update_face_preview(self, frame):
        try:
            # # logger.info(f"[Register] Frame hash: {hash(frame.tobytes())}")
            self.current_capture_frame = frame.copy()
            if self.capture_auto and self.capture_index < 30:
                if not hasattr(self, 'last_capture_time') or (time.time() - self.last_capture_time) > 1.0:
                    # Detect faces in the frame
                    faces = self.attendance_system.face_embedding.detect_faces(frame)
                    if faces:
                        # Chỉ lấy khuôn mặt đầu tiên
                        bbox = faces[0].get("bbox")
                        if bbox is not None and len(bbox) == 4:
                            x1, y1, x2, y2 = bbox
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(frame.shape[1] - 1, x2), min(frame.shape[0] - 1, y2)
                            if x2 > x1 and y2 > y1:
                                face_img = frame[y1:y2, x1:x2].copy()
                                self.captured_frames.append(face_img)
                                self.capture_index += 1
                                self.capture_progress.setValue(self.capture_index)
                                self.capture_status.setText(f"{self.capture_index}/{30} ảnh đã thu thập")
                                # # logger.info(f"[Capture] Saved face region: bbox=({x1},{y1},{x2},{y2}), shape={face_img.shape}")
                                self.last_capture_time = time.time()
                                if self.capture_index <= 5:
                                    pix = self.convert_cv_qt(face_img)
                                    self.captured_images[self.capture_index - 1].setPixmap(pix.scaled(80, 60, Qt.KeepAspectRatio))
                                if self.capture_index >= 30:
                                    self.capture_auto = False
                                    self.capture_camera.stop()
                                    self.capture_thread.quit()
                                    self.start_capture_btn.setText('Bắt đầu thu thập')
                                    self.train_btn.setEnabled(True)
                            else:
                                # logger.warning(f"[Capture] Invalid bbox: ({x1},{y1},{x2},{y2})")
                                pass
                        else:
                            # logger.warning(f"[Capture] No valid bbox in detected face: {faces[0]}")
                            pass
                    else:
                        # # logger.info("[Capture] No face detected in frame, skip.")
                        pass
        except Exception as e:
            # logger.error(f"Error in update_face_preview: {e}")
            pass

    def update_preview_pixmap(self):
        try:
            if hasattr(self, 'current_capture_frame') and self.current_capture_frame is not None:
                frame = self.current_capture_frame
                pix = self.convert_cv_qt(frame)
                self.face_preview.setPixmap(pix)
        except Exception as e:
            # logger.error(f"Error in update_preview_pixmap: {e}")
            pass
            pass

    def capture_image(self):
        try:
            # Kiểm tra trạng thái thu thập/camera
            if not hasattr(self, 'capture_camera') or not hasattr(self, 'capture_thread') or not self.capture_camera.isRunning() or not self.capture_thread.isRunning():
                QMessageBox.warning(self, "Cảnh báo", "Bạn cần nhấn 'Bắt đầu thu thập' trước khi chụp ảnh.")
                return
            if not (hasattr(self, 'current_capture_frame') and self.current_capture_frame is not None):
                QMessageBox.warning(self, "Cảnh báo", "Không có ảnh hợp lệ từ camera. Hãy kiểm tra lại camera hoặc thử lại.")
                return
            if self.capture_index >= 5:
                QMessageBox.information(self, "Thông báo", "Bạn đã chụp đủ số lượng ảnh tối đa cho mỗi lần.")
                return
            frame = self.current_capture_frame
            self.captured_frames.append(frame.copy())
            self.capture_index += 1
            pix = self.convert_cv_qt(frame)
            self.captured_images[self.capture_index - 1].setPixmap(pix.scaled(80, 60, Qt.KeepAspectRatio))
            self.capture_progress.setValue(self.capture_index)
            self.capture_status.setText(f"{self.capture_index}/{30} ảnh đã thu thập")
            self.retake_btn.setEnabled(True)
            if self.capture_index >= 5:
                self.capture_btn.setEnabled(False)
        except Exception as e:
            # logger.error(f"Error in capture_image: {e}")
            QMessageBox.critical(self, "Lỗi", f"Không thể chụp ảnh: {e}")

    def retake_image(self):
        try:
            if self.capture_index == 0 or not self.captured_frames:
                QMessageBox.warning(self, "Cảnh báo", "Không còn ảnh nào để xóa/chụp lại.")
                return
            self.capture_index -= 1
            if self.captured_frames:
                self.captured_frames.pop()
            if self.capture_index < len(self.captured_images):
                self.captured_images[self.capture_index].clear()
            self.capture_progress.setValue(self.capture_index)
            self.capture_status.setText(f"{self.capture_index}/{30} ảnh đã thu thập")
            self.capture_btn.setEnabled(True)
            if self.capture_index == 0:
                self.retake_btn.setEnabled(False)
        except Exception as e:
            # logger.error(f"Error in retake_image: {e}")
            QMessageBox.critical(self, "Lỗi", f"Không thể chụp lại: {e}")

    def train_images(self):
        try:
            frames = self.captured_frames
            if not frames or len(frames) == 0:
                QMessageBox.warning(self, 'Lỗi', f'Không có ảnh nào để huấn luyện.')
                return
            self.progress_dialog = QProgressDialog("Đang trích xuất embedding...", "Hủy", 0, len(frames), self)
            self.progress_dialog.setWindowTitle("Huấn luyện")
            self.progress_dialog.setValue(0)
            self.progress_dialog.show()

            self.train_worker = TrainWorker(self.attendance_system.face_embedding, frames)
            self.train_worker.progress.connect(self.progress_dialog.setValue)

            def done_slot(embs):
                self.progress_dialog.close()
                if len(embs) == 0:
                    QMessageBox.warning(self, 'Lỗi', 'Không trích xuất được embedding nào.')
                    # logger.error('Không trích xuất được embedding nào khi huấn luyện.')
                    return
                # Lấy trung bình embedding của 30 ảnh
                self.trained_embedding = np.mean(embs, axis=0)
                # # logger.info(f'Đã tính trung bình embedding từ {len(embs)} ảnh.')
                self.reload_embeddings()
                QMessageBox.information(self, 'Hoàn thành', f'Đã huấn luyện từ {len(embs)} ảnh. Hệ thống đã cập nhật dữ liệu nhận diện.')
                # # logger.info('Huấn luyện thành công, đã cập nhật embeddings.')
                self.register_btn.setEnabled(True)

            def error_slot(msg):
                self.progress_dialog.close()
                # logger.error(f"TrainWorker error: {msg}")
                QMessageBox.critical(self, "Lỗi", f"Không thể huấn luyện: {msg}")

            self.train_worker.done.connect(done_slot)
            self.train_worker.error.connect(error_slot)
            self.progress_dialog.canceled.connect(self.train_worker.cancel)
            self.train_worker.start()

        except Exception as e:
            # logger.error(f"Error in train_images: {e}")
            QMessageBox.critical(self, "Lỗi", f"Không thể huấn luyện: {e}")

    def reload_embeddings(self):
        try:
            self.attendance_system.update_embeddings()
            # # logger.info('Reloaded embeddings from database.')
        except Exception as e:
            # logger.error(f'Error reloading embeddings: {e}')
            pass
            pass

    def update_class_comboboxes(self):
        """Đồng bộ danh sách lớp cho cả class_input và class_filter từ database."""
        try:
            classes = self.db.get_all_classes()
            # Cập nhật cho class_input (QComboBox)
            current_class = self.class_input.currentText() if hasattr(self, 'class_input') else ""
            self.class_input.setEditable(True)  # Luôn cho phép nhập mới
            self.class_input.blockSignals(True)
            self.class_input.clear()
            self.class_input.addItems(classes)
            # Giữ lại lựa chọn cũ nếu còn tồn tại
            if current_class in classes:
                self.class_input.setCurrentText(current_class)
            self.class_input.blockSignals(False)
            # Cập nhật cho class_filter (QComboBox)
            current_filter = self.class_filter.currentText() if hasattr(self, 'class_filter') else "Tất cả"
            self.class_filter.blockSignals(True)
            self.class_filter.clear()
            self.class_filter.addItem("Tất cả")
            self.class_filter.addItems(classes)
            # Giữ lại lựa chọn cũ nếu còn tồn tại
            if current_filter in (["Tất cả"] + classes):
                self.class_filter.setCurrentText(current_filter)
            self.class_filter.blockSignals(False)
        except Exception as e:
            # logger.error(f"Lỗi khi đồng bộ danh sách lớp: {e}")
            pass
            pass

    def register_user(self):
        try:
            name = self.name_input.text().strip()
            student_id = self.student_id_input.text().strip()
            class_name = self.class_input.currentText().strip()
            email = self.email_input.text().strip()
            # Kiểm tra tên không rỗng
            if not name:
                QMessageBox.warning(self, 'Lỗi', 'Vui lòng nhập họ tên.')
                return
            # Kiểm tra mã sinh viên đúng 8 số
            if not student_id or not student_id.isdigit() or len(student_id) != 8:
                QMessageBox.warning(self, 'Lỗi', 'Mã sinh viên phải gồm đúng 8 chữ số.')
                return
            # Kiểm tra email nếu có nhập
            import re
            if email and not re.fullmatch(r'[\w.+-]+@(gmail\.com|hus\.edu\.vn)', email):
                QMessageBox.warning(self, 'Lỗi', 'Email phải có định dạng @gmail.com hoặc @hus.edu.vn.')
                return
            if not hasattr(self, 'trained_embedding') or self.trained_embedding is None:
                QMessageBox.warning(self, 'Lỗi', 'Chưa huấn luyện embedding.')
                return
            user_id = self.db.add_user(name, student_id, class_name, email)
            if user_id == 'INVALID_STUDENT_ID':
                QMessageBox.warning(self, 'Lỗi', 'Mã sinh viên phải gồm đúng 8 chữ số.')
                # logger.error(f'Mã sinh viên không hợp lệ: {student_id}')
                return
            if user_id == 'INVALID_EMAIL':
                QMessageBox.warning(self, 'Lỗi', 'Email phải có định dạng @gmail.com hoặc @hus.edu.vn.')
                # logger.error(f'Email không hợp lệ: {email}')
                return
            if user_id == 'EXISTS':
                QMessageBox.warning(self, 'Lỗi', 'Mã sinh viên đã tồn tại.')
                # logger.error(f'Mã sinh viên {student_id} đã tồn tại khi đăng ký.')
                return
            if not user_id:
                QMessageBox.warning(self, 'Lỗi', 'Có lỗi không xác định khi thêm người dùng.')
                # logger.error(f'Lỗi không xác định khi thêm user {student_id}')
                return
            embedding_ok = self.db.update_user_embedding(student_id, self.trained_embedding)
            if not embedding_ok:
                QMessageBox.warning(self, 'Lỗi', 'Lưu embedding không thành công.')
                # logger.error(f'Lưu embedding không thành công cho {student_id}')
                return
            # Cập nhật lại embeddings để nhận diện ngay
            self.reload_embeddings()
            # === Lưu các ảnh đã đăng ký ra thư mục riêng ===
            try:
                # # # logger.debug(f'[DEBUG] name={name}, student_id={student_id}')
                # # # logger.debug(f'[DEBUG] Số lượng self.captured_frames: {len(self.captured_frames)}')
                import re
                # Tạo tên thư mục đúng định dạng: 'Họ và tên _ MãSV'
                dir_name = f"{name} _ {student_id}"
                # Loại bỏ ký tự đặc biệt không hợp lệ cho Windows
                safe_dir_name = re.sub(r'[\\/:*?\"<>|]', '_', dir_name)
                save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'Image', safe_dir_name)
                # # # logger.debug(f'[DEBUG] save_dir: {save_dir}, abs: {os.path.abspath(save_dir)}')
                # # # logger.debug(f'[DEBUG] cwd: {os.getcwd()}')
                if not self.captured_frames or len(self.captured_frames) == 0:
                    # logger.error('[ERROR] Không có ảnh nào trong self.captured_frames để lưu.')
                    QMessageBox.critical(self, 'Lỗi', 'Không có ảnh nào được thu thập để lưu. Vui lòng chụp ảnh trước khi đăng ký.')
                    return
                os.makedirs(save_dir, exist_ok=True)
                # Test write permission by trying to write a dummy file
                test_path = os.path.join(save_dir, 'test_write.tmp')
                try:
                    with open(test_path, 'w') as f:
                        f.write('test')
                    test_result = True
                    os.remove(test_path)
                except Exception as e:
                    test_result = False
                    # logger.error(f'[ERROR] Không thể ghi file test vào thư mục {save_dir}: {e}')
                save_success = True
                for idx, frame in enumerate(self.captured_frames):
                    img_path = os.path.join(save_dir, f'{idx+1:02d}.jpg')
                    try:
                        write_ok = cv2.imwrite(img_path, frame)
                        if not write_ok:
                            # logger.error(f'[LỖI] Lưu ảnh thất bại: {img_path}')
                            save_success = False
                        else:
                            # Lưu embedding cho từng ảnh nếu có
                            if hasattr(self.attendance_system, 'face_embedding'):
                                embedding = self.attendance_system.face_embedding.get_face_embedding(frame)
                                if embedding is not None:
                                    norm = float(np.linalg.norm(embedding))
                                    pass # Đã bỏ qua việc lưu file embedding để giảm thiểu file log không cần thiết
                    except Exception as e:
                        # logger.error(f'[LỖI] Exception khi lưu ảnh hoặc embedding {img_path}: {e}')
                        save_success = False
                if save_success:
                    # # logger.info(f'Đã lưu {len(self.captured_frames)} ảnh đăng ký vào {save_dir}')
                    # Mở thư mục chứa ảnh sau khi lưu thành công
                    import subprocess, sys
                    if sys.platform.startswith('win'):
                        subprocess.Popen(f'explorer "{os.path.abspath(save_dir)}"')
                    elif sys.platform.startswith('darwin'):
                        subprocess.Popen(['open', os.path.abspath(save_dir)])
                    else:
                        subprocess.Popen(['xdg-open', os.path.abspath(save_dir)])
                else:
                    QMessageBox.critical(self, 'Lỗi', 'Có lỗi khi lưu một hoặc nhiều ảnh. Vui lòng kiểm tra lại quyền ghi thư mục hoặc dung lượng ổ đĩa.')
            except Exception as e:
                # logger.error(f'Lỗi khi lưu ảnh đăng ký: {e}')
                QMessageBox.critical(self, 'Lỗi', f'Lỗi khi lưu ảnh đăng ký: {e}')

            QMessageBox.information(self, 'Thành công', f'Đã đăng ký {name} với {len(self.captured_frames)} ảnh. Hệ thống đã cập nhật dữ liệu nhận diện.')
            # # logger.info(f'Đăng ký user {name} thành công và đã cập nhật embeddings.')
            self.name_input.clear()
            self.student_id_input.clear()
            self.class_input.clear()
            self.email_input.clear()
            for lbl in self.captured_images:
                lbl.clear()
            self.capture_progress.setValue(0)
            self.capture_status.setText(f"0/{30} ảnh đã thu thập")
            self.captured_frames = []
            self.trained_embedding = None
            self.register_btn.setEnabled(False)
            self.train_btn.setEnabled(False)
            self.update_users_table()
            self.update_attendance_info()
        except Exception as e:
            # logger.error(f"Error in register_user: {e}")
            pass
            QMessageBox.critical(self, "Lỗi", f"Không thể đăng ký: {e}")

    def populate_user_inputs(self):
        try:
            row = self.users_table.currentRow()
            if row < 0:
                return
            name_item = self.users_table.item(row, 1)
            student_id_item = self.users_table.item(row, 2)
            class_item = self.users_table.item(row, 3)
            email_item = self.users_table.item(row, 4)
            if name_item:
                self.name_input.setText(name_item.text())
            if student_id_item:
                self.student_id_input.setText(student_id_item.text())
            if class_item:
                self.class_input.setCurrentText(class_item.text())
            if email_item:
                self.email_input.setText(email_item.text())
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Không thể tải thông tin người dùng: {e}")

    def filter_users_table(self):

        try:
            search_text = self.user_search.text().lower()
            users = self.db.get_all_users()
            self.users_table.setRowCount(0)
            show_all = (search_text == "")
            filtered_users = []
            if show_all:
                filtered_users = users
            else:
                for user in users:
                    # user = (name, student_id, class_name, email, created_at)
                    if search_text in user[0].lower() or search_text in user[1].lower():
                        filtered_users.append(user)
            for idx, user in enumerate(filtered_users, 1):
                row = self.users_table.rowCount()
                self.users_table.insertRow(row)
                self.users_table.setItem(row, 0, QTableWidgetItem(str(idx)))
                for col, value in enumerate(user):
                    self.users_table.setItem(row, col + 1, QTableWidgetItem(str(value)))
        except Exception as e:
            # logger.error(f"Error in filter_users_table: {e}")
            pass

    def on_clear_today_attendance_clicked(self):
        reply = QMessageBox.question(self, "Xác nhận xoá", "Bạn có chắc chắn muốn xoá toàn bộ danh sách điểm danh hôm nay?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            if self.db.clear_today_attendance():
                QMessageBox.information(self, "Thành công", "Đã xoá danh sách điểm danh hôm nay.")
                self.update_attendance_info()
            else:
                QMessageBox.critical(self, "Lỗi", "Không thể xoá danh sách điểm danh hôm nay.")

    def update_attendance_info(self) -> None:
        """
        Update attendance information display including statistics and recent records.
        
        Fetches today's attendance records from database and updates all relevant
        UI components including tables, labels, and status bar.
        """
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            records = self.db.get_attendance_records(date=today)
            
            # Track unique present students (latest record per student)
            unique_present: Dict[str, tuple] = {}
            for rec in records:
                name, student_id, class_name, date, time, status = rec
                if status == "present":
                    unique_present[student_id] = (name, student_id, class_name, date, time, status)
            
            # Update recent attendance table
            self.recent_attendance_table.setRowCount(len(unique_present))
            for i, (student_id, (name, _, class_name, date, time, status)) in enumerate(unique_present.items()):
                self.recent_attendance_table.setItem(i, 0, QTableWidgetItem(name))
                self.recent_attendance_table.setItem(i, 1, QTableWidgetItem(student_id))
                self.recent_attendance_table.setItem(i, 2, QTableWidgetItem(time))
                status_item = QTableWidgetItem(status)
                status_item.setForeground(QBrush(QColor("#2ecc71" if status == "present" else "#e74c3c")))
                self.recent_attendance_table.setItem(i, 3, status_item)
            
            # Calculate statistics
            total_students = self.db.get_total_students()
            present_today = len(unique_present)
            absent_today = max(0, total_students - present_today)
            attendance_rate = (present_today / total_students * 100) if total_students > 0 else 0
            
            # Update dashboard stats
            self.stats_values["Tổng số sinh viên:"].setText(str(total_students))
            self.stats_values["Đã điểm danh:"].setText(str(present_today))
            self.stats_values["Vắng mặt:"].setText(str(absent_today))
            
            # Update status bar labels
            if hasattr(self, 'stat_total_label'):
                self.stat_total_label.setText(f"Tổng: {total_students}")
            if hasattr(self, 'stat_present_label'):
                self.stat_present_label.setText(f"Đã điểm danh: {present_today}")
            if hasattr(self, 'stat_absent_label'):
                self.stat_absent_label.setText(f"Vắng mặt: {absent_today}")
            
            # Update status bar via helper method
            if hasattr(self, 'update_statusbar_stats'):
                self.update_statusbar_stats(total_students, present_today, absent_today)
                
        except Exception as e:
            logger.error(f"Error updating attendance info: {e}", exc_info=True)

    def update_users_table(self):
        try:
            users = self.db.get_all_users()
            self.users_table.setRowCount(len(users))
            for i, (name, student_id, class_name, email, created_at) in enumerate(users):
                self.users_table.setItem(i, 0, QTableWidgetItem(str(i+1)))
                self.users_table.setItem(i, 1, QTableWidgetItem(name))
                self.users_table.setItem(i, 2, QTableWidgetItem(student_id))
                self.users_table.setItem(i, 3, QTableWidgetItem(class_name))
                self.users_table.setItem(i, 4, QTableWidgetItem(email))
                self.users_table.setItem(i, 5, QTableWidgetItem(str(created_at)))
        except Exception as e:
            # logger.error(f"Error in update_users_table: {e}")
            pass

    def apply_filter(self):
        try:
            from_date = self.from_date.date().toString("yyyy-MM-dd")
            to_date = self.to_date.date().toString("yyyy-MM-dd")
            if from_date > to_date:
                from_date, to_date = to_date, from_date
            class_filter = self.class_filter.currentText()
            status_filter = self.status_filter.currentText()
            # Lấy dữ liệu đã lọc
            records = self.db.get_attendance_records(
                from_date=from_date, to_date=to_date,
                class_name=class_filter if class_filter != "Tất cả" else None,
                status=status_filter if status_filter != "Tất cả" else None
            )
            columns = ["Tên", "Mã SV", "Lớp", "Ngày", "Thời gian", "Trạng thái"]
            # Cập nhật bảng báo cáo
            if hasattr(self, 'report_table'):
                self.report_table.setRowCount(len(records))
                for i, row in enumerate(records):
                    for j, value in enumerate(row):
                        self.report_table.setItem(i, j, QTableWidgetItem(str(value)))
            # Cập nhật thống kê
            # Tổng số sinh viên chỉ tính những người đã đăng ký khuôn mặt (có embedding)
            total_students = self.db.get_total_students()
            total_sessions = len(set(r[3] for r in records)) if records else 0  # r[3]: ngày
            present_count = sum(1 for r in records if r[5] == "present")
            absent_count = sum(1 for r in records if r[5] == "absent")
            denominator = present_count + absent_count if (present_count + absent_count) > 0 else 1
            full_attendance_rate = min(100.0, present_count / denominator * 100)
            absent_rate = min(100.0, absent_count / denominator * 100)
            if hasattr(self, 'summary_values'):
                self.summary_values["Tổng số sinh viên:"].setText(str(total_students))
                self.summary_values["Tổng số buổi học:"].setText(str(total_sessions))
                self.summary_values["Tỷ lệ đi học đầy đủ:"].setText(f"{full_attendance_rate:.1f}%")
                self.summary_values["Tỷ lệ vắng mặt:"].setText(f"{absent_rate:.1f}%")
            # Cập nhật biểu đồ
            if hasattr(self, 'update_chart'):
                self.update_chart(present_count, absent_count)
        except Exception as e:
            # logger.error(f"Error in apply_filter: {e}")
            pass
            pass

    def export_report_to_pdf(self):
        try:
            from fpdf import FPDF
            import os
            from_date = self.from_date.date().toString("yyyy-MM-dd")
            to_date = self.to_date.date().toString("yyyy-MM-dd")
            # Kiểm tra logic ngày, nếu from_date > to_date thì swap lại
            if from_date > to_date:
                from_date, to_date = to_date, from_date
            class_filter = self.class_filter.currentText()
            status_filter = self.status_filter.currentText()
            records = self.db.get_attendance_records(
                from_date=from_date, to_date=to_date,
                class_name=class_filter if class_filter != "Tất cả" else None,
                status=status_filter if status_filter != "Tất cả" else None
            )
            columns = ["Tên", "Mã SV", "Lớp", "Ngày", "Thời gian", "Trạng thái"]
            path, _ = QFileDialog.getSaveFileName(self, "Lưu PDF", "", "PDF files (*.pdf)")
            if not path:
                return
            font_path = os.path.join(os.path.dirname(__file__), "DejaVuSans.ttf")
            if not os.path.exists(font_path):
                QMessageBox.critical(self, "Thiếu font", "Vui lòng tải file DejaVuSans.ttf về thư mục src để xuất PDF tiếng Việt.")
                return
            pdf = FPDF()
            pdf.add_page()
            pdf.add_font('DejaVu', '', font_path, uni=True)
            pdf.set_font('DejaVu', '', 12)
            # Header
            for col in columns:
                pdf.cell(32, 10, col, 1, 0, 'C')
            pdf.ln()
            # Data
            for row in records:
                for item in row:
                    pdf.cell(32, 10, str(item), 1, 0, 'C')
                pdf.ln()
            pdf.output(path)
            QMessageBox.information(self, "Thành công", "Đã xuất báo cáo ra PDF thành công.")
        except ImportError:
            QMessageBox.critical(self, "Thiếu thư viện", "Bạn cần cài đặt thư viện fpdf2: pip install fpdf2")

        except Exception as e:
            # logger.error(f"Error in export_report_to_pdf: {e}")
            pass
            QMessageBox.critical(self, "Lỗi", f"Không thể xuất báo cáo PDF: {e}")

    def export_report_to_excel(self):
        try:
            import pandas as pd
            from_date = self.from_date.date().toString("yyyy-MM-dd")
            to_date = self.to_date.date().toString("yyyy-MM-dd")
            # Kiểm tra logic ngày, nếu from_date > to_date thì swap lại
            if from_date > to_date:
                from_date, to_date = to_date, from_date
            class_filter = self.class_filter.currentText()
            status_filter = self.status_filter.currentText()
            records = self.db.get_attendance_records(
                from_date=from_date, to_date=to_date,
                class_name=class_filter if class_filter != "Tất cả" else None,
                status=status_filter if status_filter != "Tất cả" else None
            )
            columns = ["Tên", "Mã SV", "Lớp", "Ngày", "Thời gian", "Trạng thái"]
            import os
            path, _ = QFileDialog.getSaveFileName(self, "Lưu Excel", "", "Excel files (*.xlsx)")
            if not path:
                return
            df = pd.DataFrame(records, columns=columns)
            df.to_excel(path, index=False)
            QMessageBox.information(self, "Thành công", "Đã xuất báo cáo ra Excel thành công.")
        except ImportError:
            QMessageBox.critical(self, "Thiếu thư viện", "Bạn cần cài đặt thư viện pandas: pip install pandas openpyxl")
        except Exception as e:
            # logger.error(f"Error in export_report_to_excel: {e}")
            QMessageBox.critical(self, "Lỗi", f"Không thể xuất báo cáo Excel: {e}")
    def save_settings(self):
        try:
            self.config["camera_id"] = self.camera_source.currentIndex()
            self.config["camera_fps"] = self.camera_fps.value()
            resolution = self.camera_resolution.currentText()
            self.config["max_dim"] = int(resolution.split("x")[1])
            self.config["recognition_distance"] = self.distance_spinbox.value()
            self.config["recognition_confidence"] = self.confidence_threshold.value() / 100.0
            self.config["min_confidence_threshold"] = self.min_confidence_spinbox.value()
            self.config["absolute_distance_threshold"] = self.abs_distance_spinbox.value()
            self.attendance_system.config.update(self.config)
            QMessageBox.information(self, "Thành công", "Đã lưu cài đặt.")
        except Exception as e:
            # logger.error(f"Error in save_settings: {e}")
            QMessageBox.critical(self, "Lỗi", f"Không thể lưu cài đặt: {e}")
            pass

# Dark Mode logic
    def update_chart(self, present_count=0, absent_count=0):
        try:
            ax = self.figure.add_subplot(111)
            ax.clear()
            ax.bar(["Có mặt", "Vắng mặt"], [present_count, absent_count], color=["#2ecc71", "#e74c3c"])
            ax.set_title("Thống kê điểm danh")
            self.canvas.draw()
        except Exception as e:
            # logger.error(f"Error in update_chart: {e}")
            pass

    def toggle_dark_mode(self, state):
        if state == 2:  # Checked
            dark_stylesheet = """
QWidget { background-color: #23272f; color: #f3f6fa; }
QMainWindow { background: #23272f; }
QTabBar::tab {
    background-color: #2d323d; color: #aeb6c8; padding: 8px 15px;
    border-top-left-radius: 8px; border-top-right-radius: 8px;
    border: 1px solid #3e4451;
}
QTabBar::tab:selected, QTabBar::tab:hover {
    background: #2979ff;
    color: #fff;
}
QPushButton {
    background: #2979ff; color: #fff; border-radius: 6px; padding: 10px; border: 1px solid #2979ff;
    font-weight: bold;
}
QPushButton:hover {
    background: #5393ff; color: #fff;
}
QLineEdit, QComboBox, QDateEdit, QSpinBox, QDoubleSpinBox {
    background: #23272f; color: #f3f6fa; border: 1px solid #3e4451; border-radius: 4px; padding: 6px;
}
QLabel { color: #f3f6fa; }
QTableWidget {
    background: #23272f; color: #f3f6fa; gridline-color: #3e4451; border: 1px solid #3e4451; border-radius: 5px;
    selection-background-color: #374151; selection-color: #00bcd4;
}
QHeaderView::section {
    background: #2979ff; color: #fff; padding: 5px;
    border: none;
}
QGroupBox {
    border: 1px solid #3e4451; border-radius: 8px; background-color: #2d323d; font-weight: bold; padding: 10px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 5px;
    color: #00bcd4;
}
QProgressBar {
    border: 1px solid #3e4451; border-radius: 5px; text-align: center;
    background: #2d323d; color: #00bcd4;
}
QProgressBar::chunk {
    background-color: #2979ff;
    border-radius: 3px;
}
QScrollArea { background: #23272f; }
QCheckBox { color: #aeb6c8; }
QStatusBar { background: #23272f; color: #00bcd4; }
            """
            self.setStyleSheet(dark_stylesheet)
            # Highlight main header in dark mode with strong contrast and glow
            if hasattr(self, 'main_header'):
                self.main_header.setStyleSheet("color: #ffe066; font-size: 28px; font-weight: bold; margin: 15px; text-shadow: 0 0 12px #111, 0 0 4px #e2b714, 1px 1px 0 #000;")
        else:
            self.setStyleSheet(STYLESHEET)
            # Restore main header style in light mode
            if hasattr(self, 'main_header'):
                self.main_header.setStyleSheet("color: #2c3e50; font-size: 24px; font-weight: bold; margin: 15px;")

    def restore_database(self):
        try:
            path = QFileDialog.getOpenFileName(self, "Khôi phục cơ sở dữ liệu", "", "SQLite Database (*.db)")[0]
            if path:
                self.db.restore(path)
                self.reload_embeddings()
                self.update_users_table()
                self.update_attendance_info()
                QMessageBox.information(self, "Thành công", "Đã khôi phục cơ sở dữ liệu.")
        except Exception as e:
            # logger.error(f"Error in restore_database: {e}")
            QMessageBox.critical(self, "Lỗi", f"Không thể khôi phục: {e}")

    def backup_database(self):
        try:
            path = QFileDialog.getSaveFileName(self, "Sao lưu cơ sở dữ liệu", "", "SQLite Database (*.db)")[0]
            if path:
                self.db.backup(path)
                QMessageBox.information(self, "Thành công", "Đã sao lưu cơ sở dữ liệu.")
        except Exception as e:
            # logger.error(f"Error in backup_database: {e}")
            QMessageBox.critical(self, "Lỗi", f"Không thể sao lưu: {e}")

    def edit_user(self):
        try:
            selected = self.users_table.selectedItems()
            if not selected or self.users_table.currentRow() < 0:
                QMessageBox.warning(self, "Lỗi", "Vui lòng chọn một người dùng để sửa.")
                return
            row = self.users_table.currentRow()
            user_id_item = self.users_table.item(row, 2)  # Mã SV (student_id) is column 2
            if user_id_item is None:
                QMessageBox.warning(self, "Lỗi", "Không thể xác định mã sinh viên của người dùng.")
                return
            user_id = user_id_item.text()
            name = self.name_input.text().strip()
            student_id = self.student_id_input.text().strip()
            class_name = self.class_input.currentText().strip()
            email = self.email_input.text().strip()
            if not all([name, student_id]):
                QMessageBox.warning(self, "Lỗi", "Vui lòng nhập đầy đủ thông tin.")
                return
            success = self.db.update_user(user_id, name, student_id, class_name, email)
            if not success:
                QMessageBox.critical(self, "Lỗi", "Không thể cập nhật thông tin người dùng trong cơ sở dữ liệu.")
                return
            if hasattr(self, 'trained_embedding') and self.trained_embedding is not None:
                self.db.update_user_embedding(student_id, self.trained_embedding)
            self.reload_embeddings()
            self.update_users_table()
            self.update_attendance_info()
            QMessageBox.information(self, "Thành công", "Đã cập nhật thông tin người dùng.")
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Không thể sửa người dùng: {e}")

    def clear_attendance_data(self):
        try:
            reply = QMessageBox.question(self, "Xác nhận", "Bạn có chắc chắn muốn xóa toàn bộ dữ liệu điểm danh không?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                if self.db.clear_attendance_data():
                    QMessageBox.information(self, "Thành công", "Đã xóa toàn bộ dữ liệu điểm danh.")
                    self.update_attendance_info()
                else:
                    QMessageBox.critical(self, "Lỗi", "Không thể xóa dữ liệu điểm danh.")
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Không thể xóa dữ liệu điểm danh: {e}")

    def delete_user(self):
        try:
            selected = self.users_table.selectedItems()
            if not selected:
                QMessageBox.warning(self, "Lỗi", "Vui lòng chọn một người dùng để xóa.")
                return
            student_id = selected[2].text()
            reply = QMessageBox.question(self, "Xác nhận", f"Bạn có chắc muốn xóa người dùng {student_id}?",
                                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.db.delete_user(student_id)
                self.reload_embeddings()
                self.update_users_table()
                self.update_attendance_info()
                QMessageBox.information(self, "Thành công", "Đã xóa người dùng.")
        except Exception as e:
            # logger.error(f"Error in delete_user: {e}")
            QMessageBox.critical(self, "Lỗi", f"Không thể xóa người dùng: {e}")

    def on_tab_changed(self, index: int) -> None:
        """
        Handle tab change events.
        
        Args:
            index: Index of the newly selected tab
        """
        # Sync class list when changing tabs
        self.update_class_comboboxes()
        
        # Update user table when switching to registration tab (index 2)
        if index == 2:
            self.filter_users_table()
        
        # Auto-update report when switching to report tab
        if self.tabs.tabText(index) == "Báo cáo":
            self.apply_filter()
    
    def closeEvent(self, event) -> None:
        """
        Handle application close event with proper cleanup.
        
        Ensures all resources (camera, threads, database) are properly closed
        before the application exits.
        
        Args:
            event: Close event
        """
        try:
            # Close attendance system
            self.attendance_system.close()
            
            # Close database connection
            self.db.close()
            
            # Stop camera handler
            if hasattr(self, 'camera_handler') and self.camera_handler.isRunning():
                self.camera_handler.stop()
            
            # Stop AI processing thread
            if self.ai_thread.isRunning():
                self.ai_thread.running = False
                self.ai_thread.quit()
                self.ai_thread.wait()
            
            # Stop capture camera if exists
            if hasattr(self, 'capture_camera') and self.capture_camera.isRunning():
                self.capture_camera.stop()
            
            # Stop capture thread if exists
            if hasattr(self, 'capture_thread') and self.capture_thread.isRunning():
                self.capture_thread.quit()
                self.capture_thread.wait()
            
            # Hide spinner
            self.show_spinner(False)
            
            event.accept()
            
        except Exception as e:
            logger.error(f"Error during application close: {e}", exc_info=True)
            event.accept()

def main() -> None:
    """
    Main entry point for the Face Recognition Attendance System GUI.
    
    Initializes the Qt application, creates the main window, and starts
    the event loop.
    """
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()