"""
Training worker module for Face Recognition Attendance System.
Handles batch processing of face embeddings in a background thread.
"""
from PyQt5.QtCore import QThread, pyqtSignal
import logging
import time
from typing import List, Optional, Any, Dict, Tuple

logger = logging.getLogger(__name__)


class TrainWorker(QThread):
    """Background worker for processing face embeddings."""
    
    progress = pyqtSignal(int, int, int)  # current, total, successful
    done = pyqtSignal(list, int)  # embeddings, failed_count
    error = pyqtSignal(str, dict)  # message, details
    
    def __init__(
        self, 
        face_embedding: Any, 
        frames: List[Any], 
        delay: float = 0.5, 
        batch_size: int = 1,
        embedding_threshold: Optional[float] = None,
        config: Optional[Dict] = None
    ):
        super().__init__()
        self.face_embedding = face_embedding
        self.frames = frames or []
        self.delay = delay
        self.batch_size = max(1, batch_size)
        self.config = config or {}
        
        # Use threshold from config if not provided
        self.embedding_threshold = (
            embedding_threshold if embedding_threshold is not None 
            else self.config.get("min_confidence_threshold", 0.6)
        )
        
        self._is_cancelled = False
        self._current_index = 0
        self._successful_embs: List = []
        self._failed_frames = 0
    
    def __del__(self):
        self.clean_up()
    
    def clean_up(self) -> None:
        """Clean up resources and stop processing."""
        self._is_cancelled = True
    
    def cancel(self) -> None:
        """Cancel the ongoing processing."""
        self._is_cancelled = True
        logger.info("Train worker cancelled")
    
    def validate_frame(self, frame: Any) -> bool:
        """
        Validate that a frame is suitable for processing.
        
        Args:
            frame: Image frame to validate
            
        Returns:
            True if valid, False otherwise
        """
        if frame is None:
            return False
        try:
            shape = frame.shape
            if len(shape) < 2 or (len(shape) == 3 and shape[2] not in [1, 3, 4]):
                logger.warning(f"Invalid frame shape: {shape}")
                return False
        except (AttributeError, TypeError):
            logger.warning("Frame is not a valid image array")
            return False
        return True
    
    def process_batch(self, batch_frames: List[Tuple[int, Any]]) -> List[Tuple[int, Optional[Any]]]:
        """
        Process a batch of frames to extract embeddings.
        
        Args:
            batch_frames: List of (index, frame) tuples
            
        Returns:
            List of (index, embedding_or_none) tuples
        """
        results = []
        for idx, frame in batch_frames:
            if not self.validate_frame(frame):
                logger.warning(f"Frame {idx + 1} validation failed, skipping")
                results.append((idx, None))
                continue
            
            try:
                emb = self.face_embedding.get_face_embedding(
                    frame, 
                    debug_label=f"train_frame_{idx + 1}"
                )
                results.append((idx, emb))
            except Exception as e:
                logger.error(f"Error processing frame {idx + 1}: {e}")
                results.append((idx, None))
        
        return results
    
    def _process_next_batch(self) -> None:
        """Process the next batch of frames."""
        if self._is_cancelled or self._current_index >= len(self.frames):
            self._finish_processing()
            return
        
        # Prepare batch
        batch_end = min(self._current_index + self.batch_size, len(self.frames))
        batch_frames = [(i, self.frames[i]) for i in range(self._current_index, batch_end)]
        
        # Process batch
        batch_results = self.process_batch(batch_frames)
        
        # Collect results
        for idx, emb in batch_results:
            if emb is not None:
                self._successful_embs.append(emb)
            else:
                self._failed_frames += 1
        
        self._current_index = batch_end
        
        # Emit progress
        self.progress.emit(
            self._current_index,
            len(self.frames),
            len(self._successful_embs)
        )
        
        # Continue or finish
        if not self._is_cancelled and self._current_index < len(self.frames):
            time.sleep(self.delay)
            self._process_next_batch()
        else:
            self._finish_processing()
    
    def _finish_processing(self) -> None:
        """Finish processing and emit results."""
        if not self._is_cancelled:
            logger.info(
                f"Embedding completed. Success: {len(self._successful_embs)}, "
                f"Failed: {self._failed_frames}"
            )
            self.done.emit(self._successful_embs, self._failed_frames)
    
    def run(self) -> None:
        """Main thread execution."""
        try:
            if not self.frames:
                error_info = {"reason": "empty_frames", "message": "No frames to process"}
                self.error.emit("No frames provided for processing", error_info)
                return
            
            # Reset state
            self._current_index = 0
            self._successful_embs = []
            self._failed_frames = 0
            self._is_cancelled = False
            
            self._process_next_batch()
            
        except Exception as e:
            error_details = {"type": type(e).__name__, "message": str(e)}
            logger.error(f"Error during embedding extraction: {e}", exc_info=True)
            self.error.emit(f"Error during embedding extraction: {e}", error_details)
