"""
Camera handler module for Face Recognition Attendance System.
Provides thread-safe camera capture with configurable FPS.
"""
import cv2
import numpy as np
import logging
from PyQt5.QtCore import QThread, pyqtSignal, QMutex, QMutexLocker
from typing import Optional

logger = logging.getLogger(__name__)


class CameraHandler(QThread):
    """Thread-safe camera handler for real-time video capture."""
    
    frame_ready = pyqtSignal(np.ndarray)
    
    def __init__(self, camera_id: int = 0, fps: int = 30):
        super().__init__()
        self.camera_id = camera_id
        self.fps = fps
        self.running = False
        self._frame: Optional[np.ndarray] = None
        self._lock = QMutex()
        self._cap: Optional[cv2.VideoCapture] = None
    
    def isRunning(self) -> bool:
        """Check if camera thread is running."""
        return self.running
    
    def start(self) -> bool:
        """
        Start the camera thread after verifying camera availability.
        
        Returns:
            True if camera is available and thread started, False otherwise
        """
        # Test camera availability first
        test_cap = cv2.VideoCapture(self.camera_id)
        if not test_cap.isOpened():
            logger.error(f"Cannot open camera {self.camera_id}")
            test_cap.release()
            return False
        test_cap.release()
        
        # Start the thread
        super().start()
        return True
    
    def run(self) -> None:
        """Main thread loop for capturing frames."""
        self.running = True
        
        try:
            self._cap = cv2.VideoCapture(self.camera_id)
            
            if not self._cap.isOpened():
                logger.error(f"Failed to open camera {self.camera_id}")
                self.running = False
                return
            
            while self.running:
                try:
                    ret, frame = self._cap.read()
                    
                    if not ret or frame is None:
                        logger.warning(f"Failed to read frame from camera {self.camera_id}")
                        self.msleep(1000)  # Wait before retry
                        continue
                    
                    # Store frame safely
                    with QMutexLocker(self._lock):
                        self._frame = frame.copy()
                    
                    # Emit signal
                    self.frame_ready.emit(frame)
                    
                    # Control FPS
                    wait_time = int(1000 / self.fps)
                    self.msleep(wait_time)
                    
                except cv2.error as e:
                    logger.error(f"OpenCV error: {e}")
                    self.msleep(1000)
                except Exception as e:
                    logger.error(f"Frame processing error: {e}")
                    self.msleep(1000)
                    
        except Exception as e:
            logger.error(f"Camera thread error: {e}")
        finally:
            self._cleanup()
    
    def _cleanup(self) -> None:
        """Release camera resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self.running = False
        logger.info("Camera thread stopped")
    
    def stop(self) -> None:
        """Stop the camera thread gracefully."""
        self.running = False
        self.wait()
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get the latest captured frame (thread-safe)."""
        with QMutexLocker(self._lock):
            return self._frame.copy() if self._frame is not None else None
