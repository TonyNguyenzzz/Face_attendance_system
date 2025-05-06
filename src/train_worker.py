from PyQt5.QtCore import QThread, pyqtSignal
import numpy as np
import logging
import time

class TrainWorker(QThread):
    # Signal cập nhật: (frame hiện tại, tổng số frame, số embedding thành công)
    progress = pyqtSignal(int, int, int)
    # Signal khi hoàn thành: trả về danh sách embedding và số lượng thất bại
    done = pyqtSignal(list, int)
    error = pyqtSignal(str)

    def __init__(self, face_embedding, frames, delay=0.5):
        """
        Khởi tạo worker thread để tạo face embeddings
        
        Args:
            face_embedding: Đối tượng xử lý face embedding
            frames: Danh sách các frame ảnh cần xử lý
            delay: Thời gian chờ giữa các lần xử lý (giây)
        """
        super().__init__()
        self.face_embedding = face_embedding
        self.frames = frames
        self.delay = delay
        self._is_cancelled = False
        
        # Thiết lập logging
        self.logger = logging.getLogger(__name__)

    def cancel(self):
        """Hủy quá trình xử lý hiện tại"""
        self._is_cancelled = True
        self.logger.info("Train worker cancelled")

    def run(self):
        """Thực hiện trích xuất embedding từ danh sách frames"""
        embs = []
        failed_frames = 0
        total_frames = len(self.frames)
        
        self.logger.info(f"Starting embedding extraction for {total_frames} frames")
        
        try:
            for i, frame in enumerate(self.frames):
                # Kiểm tra nếu quá trình bị hủy
                if self._is_cancelled:
                    self.logger.info("Process cancelled, stopping extraction")
                    break
                
                # Trích xuất embedding
                self.logger.debug(f"Processing frame {i+1}/{total_frames}")
                emb = self.face_embedding.get_face_embedding(frame)
                
                # Xử lý kết quả
                if emb is not None:
                    embs.append(emb)
                    self.logger.debug(f"Successfully extracted embedding from frame {i+1}")
                else:
                    failed_frames += 1
                    self.logger.warning(f"Failed to extract embedding from frame {i+1}")
                
                # Phát tín hiệu tiến độ: (frame hiện tại, tổng frames, số embedding thành công)
                self.progress.emit(i + 1, total_frames, len(embs))
                
                # Delay nếu cần
                if self.delay > 0 and i < total_frames - 1:  # Không delay ở frame cuối cùng
                    time.sleep(self.delay)
            
            # Báo kết quả
            self.logger.info(f"Embedding extraction completed. Success: {len(embs)}, Failed: {failed_frames}")
            self.done.emit(embs, failed_frames)
            
        except Exception as e:
            self.logger.error(f"Error during embedding extraction: {str(e)}")
            self.error.emit(str(e))