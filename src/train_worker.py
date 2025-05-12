from PyQt5.QtCore import QThread, pyqtSignal
import logging
import time
from typing import List, Optional, Any, Dict, Tuple

class TrainWorker(QThread):
    progress = pyqtSignal(int, int, int)
    done = pyqtSignal(list, int)
    error = pyqtSignal(str, dict)

    def __init__(self, face_embedding, frames, delay=0.5, batch_size=1, 
                 embedding_threshold=None, config=None):
        super().__init__()
        self.face_embedding = face_embedding
        self.frames = frames
        self.delay = delay
        self.batch_size = max(1, batch_size)
        self.config = config or {}
        # Sử dụng ngưỡng từ config nếu không được chỉ định
        # Đồng bộ với cách sử dụng trong face_recognition.py và face_embedding.py
        # face_recognition.py sử dụng min_confidence_threshold
        # face_embedding.py sử dụng min_confidence_threshold và absolute_distance_threshold
        self.embedding_threshold = embedding_threshold if embedding_threshold is not None else \
                                  self.config.get("min_confidence_threshold", 0.6)
        
        # Khởi tạo logger với mức độ cao hơn để loại bỏ thông báo không cần thiết
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.ERROR)
        
        self._is_cancelled = False
        self.logger = logging.getLogger(__name__)
        self._current_index = 0
        self._total_frames = len(frames) if frames else 0
        self._successful_embs = []
        self._failed_frames = 0

        if not frames:
            self.logger.warning("No frames provided for processing")

    def __del__(self):
        self.clean_up()

    def clean_up(self):
        self._is_cancelled = True

    def cancel(self):
        self._is_cancelled = True
        self.logger.info("Train worker cancelled")

    def validate_frame(self, frame) -> bool:
        if frame is None:
            return False
        try:
            shape = frame.shape
            if len(shape) < 2 or (len(shape) == 3 and shape[2] not in [1, 3, 4]):
                self.logger.warning(f"Invalid frame shape: {shape}")
                return False
        except (AttributeError, TypeError):
            self.logger.warning("Frame is not a valid image array")
            return False
        return True

    def process_batch(self, batch_frames: List[Any]) -> List[Tuple[int, Optional[Any]]]:
        results = []
        for idx, frame in batch_frames:
            if not self.validate_frame(frame):
                self.logger.warning(f"Frame {idx+1} validation failed, skipping")
                results.append((idx, None))
                continue

            try:
                # Sử dụng các tham số từ config một cách nhất quán
                # Lưu ý: phương thức get_face_embedding chỉ nhận tham số debug_label
                # Các tham số ngưỡng được cấu hình trong FaceEmbedding khi khởi tạo
                # từ config, không truyền trực tiếp vào phương thức
                emb = self.face_embedding.get_face_embedding(
                    frame,
                    debug_label=f"train_frame_{idx+1}"
                )
                # Đã loại bỏ log debug không cần thiết về các ngưỡng
                
                results.append((idx, emb))
            except Exception as e:
                self.logger.error(f"Error processing frame {idx+1}: {str(e)}")
                results.append((idx, None))

        return results

    def _process_next_batch(self):
        if self._is_cancelled or self._current_index >= self._total_frames:
            self._finish_processing()
            return

        batch_end = min(self._current_index + self.batch_size, self._total_frames)
        batch_frames = [(i, self.frames[i]) for i in range(self._current_index, batch_end)]
        batch_results = self.process_batch(batch_frames)

        for idx, emb in batch_results:
            if emb is not None:
                self._successful_embs.append(emb)
            else:
                self._failed_frames += 1

        self._current_index = batch_end
        self.progress.emit(
            self._current_index,
            self._total_frames,
            len(self._successful_embs)
        )

        if not self._is_cancelled and self._current_index < self._total_frames:
            # Thay vì sử dụng QTimer, chỉ đơn giản là sleep trong thread này
            time.sleep(self.delay)
            self._process_next_batch()
        else:
            self._finish_processing()

    def _finish_processing(self):
        if not self._is_cancelled:
            self.logger.info(f"Embedding completed. Success: {len(self._successful_embs)}, Failed: {self._failed_frames}")
            self.done.emit(self._successful_embs, self._failed_frames)

    def run(self):
        try:
            if not self.frames:
                error_info = {"reason": "empty_frames", "message": "No frames to process"}
                self.error.emit("No frames provided for processing", error_info)
                return

            self._current_index = 0
            self._successful_embs = []
            self._failed_frames = 0
            self._is_cancelled = False
            self._process_next_batch()

        except Exception as e:
            error_details = {
                "type": type(e).__name__,
                "message": str(e)
            }
            self.logger.error(f"Error during embedding extraction: {str(e)}", exc_info=True)
            self.error.emit(f"Error during embedding extraction: {str(e)}", error_details)
