import cv2
import logging
import numpy as np
import os
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import OrderedDict
import threading
from deepface import DeepFace
import concurrent.futures
from queue import Queue

logger = logging.getLogger("FaceEmbedding")

class FaceEmbeddingConfig:
    def __init__(
        self,
        model_dir: str,
        # Thống nhất các thông số về ngưỡng nhận diện
        detection_threshold: float = 0.7,     # Ngưỡng phát hiện khuôn mặt
        recognition_threshold: float = 0.35,  # Ngưỡng khoảng cách nhận diện
        minimum_confidence: float = 0.6,      # Độ tin cậy tối thiểu để chấp nhận kết quả
        max_allowed_distance: float = 0.6,    # Ngưỡng khoảng cách tối đa cho phép
        # Các thông số khác
        max_dim: int = 800,
        face_size: Tuple[int, int] = (160, 160),
        cache_size: int = 1000,
        max_workers: int = 4
    ):
        # Thư mục mô hình
        self.model_dir = model_dir
        
        # Tham số phát hiện và nhận diện
        self.detection_threshold = detection_threshold
        self.recognition_threshold = recognition_threshold
        self.minimum_confidence = minimum_confidence
        self.max_allowed_distance = max_allowed_distance
        
        # Các tham số xử lý hình ảnh
        self.max_dim = max_dim
        self.face_size = face_size
        
        # Tham số hệ thống
        self.cache_size = cache_size
        self.max_workers = max_workers
    
    def get(self, key: str, default=None):
        """Get configuration value with default fallback"""
        return getattr(self, key, default)

def dict_to_face_embedding_config(cfg: dict) -> FaceEmbeddingConfig:
    """Convert dictionary to FaceEmbeddingConfig object"""
    # Đảm bảo tính tương thích ngược với các tham số cũ
    detection_threshold = cfg.get("detector_confidence", cfg.get("min_confidence", 0.7))
    recognition_threshold = cfg.get("recognition_distance", cfg.get("distance_threshold", 0.35))
    minimum_confidence = cfg.get("min_confidence_threshold", cfg.get("min_confidence", 0.6))
    max_allowed_distance = cfg.get("absolute_distance_threshold", 0.6)
    
    return FaceEmbeddingConfig(
        model_dir=cfg.get("model_dir", "./models"),
        detection_threshold=detection_threshold, 
        recognition_threshold=recognition_threshold,
        minimum_confidence=minimum_confidence,
        max_allowed_distance=max_allowed_distance,
        max_dim=cfg.get("max_dim", 800),
        face_size=cfg.get("face_size", (160, 160)),
        cache_size=cfg.get("cache_size", 1000),
        max_workers=cfg.get("max_workers", 4)
    )

class LimitedSizeDict(OrderedDict):
    """Cache dictionary with limited size"""
    def __init__(self, *args, **kwargs):
        self.size_limit = kwargs.pop("size_limit", None)
        OrderedDict.__init__(self, *args, **kwargs)
        self._check_size_limit()

    def __setitem__(self, key, value):
        OrderedDict.__setitem__(self, key, value)
        self._check_size_limit()

    def _check_size_limit(self):
        if self.size_limit is not None:
            while len(self) > self.size_limit:
                self.popitem(last=False)

class FaceEmbedding:
    def __init__(self, config: Any):
        self.config = config if isinstance(config, FaceEmbeddingConfig) else dict_to_face_embedding_config(config)
        
        # Khởi tạo cache cho embedding
        self.embedding_cache = LimitedSizeDict(size_limit=self.config.cache_size)
        self.student_embeddings: Dict[str, Dict[str, Any]] = {}
        self.cache_lock = threading.Lock()
        self.emb_matrix = None
        self.emb_ids = []
        
        # Khởi tạo bộ phát hiện khuôn mặt
        from face_detector import FaceDetector
        model_path = os.path.join(self.config.model_dir, "model_3.onnx")
        if not os.path.exists(model_path):
            logger.error(f"Face detection model not found at: {model_path}")
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.face_detector = FaceDetector(model_path)
        
        # Khởi tạo thread pool executor để xử lý đa luồng
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config.max_workers,
            thread_name_prefix="FaceEmbedding"
        )
        
        logger.info(f"Face Embedding module initialized with DeepFace (facenet512) using {self.config.max_workers} workers")

    def l2_normalize(self, x: np.ndarray, axis=-1, epsilon=1e-10) -> np.ndarray:
        """Chuẩn hóa L2 vector"""
        return x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))

    def cosine_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine distance between two vectors."""
        # Kiểm tra đầu vào hợp lệ
        if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
            return 1.0
        if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape:
            return 1.0
            
        # Chuẩn hóa vector nếu cần
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm < 0.99 or a_norm > 1.01:
            a = a / (a_norm + 1e-10)
        if b_norm < 0.99 or b_norm > 1.01:
            b = b / (b_norm + 1e-10)
            
        # Tính khoảng cách cosine: 1 - độ tương đồng cosine
        dot_product = np.dot(a, b)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        return 1.0 - dot_product

    def _preprocess_face(self, face_img: np.ndarray) -> Optional[np.ndarray]:
        """Tiền xử lý ảnh khuôn mặt trước khi lấy embedding"""
        try:
            if face_img is None or face_img.size == 0:
                return None
                
            # Chuyển đổi không gian màu nếu cần
            if face_img.ndim == 2:
                face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)
            elif face_img.shape[2] == 4:
                face_img = cv2.cvtColor(face_img, cv2.COLOR_RGBA2RGB)
            elif face_img.shape[2] == 3:
                if face_img.dtype != np.uint8:
                    face_img = np.clip(face_img, 0, 255).astype(np.uint8)
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                
            # Thay đổi kích thước và chuẩn hóa
            face_img = cv2.resize(face_img, self.config.face_size, interpolation=cv2.INTER_AREA)
            face_img = face_img.astype(np.float32)
            face_img = (face_img / 127.5) - 1.0  # Chuẩn hóa về [-1,1] cho Facenet
            return face_img
        except Exception as e:
            logger.error(f"Error preprocessing face: {e}")
            return None

    def get_face_embedding(self, face_img: np.ndarray, debug_label=None) -> Optional[np.ndarray]:
        """Lấy vector embedding từ ảnh khuôn mặt"""
        # Kiểm tra đầu vào
        if face_img is None or face_img.size == 0:
            logger.warning("Empty face image provided")
            return None
        if face_img.ndim != 3 or face_img.shape[2] != 3:
            logger.warning("Invalid face image shape")
            return None
            
        # Chuyển đổi kiểu dữ liệu nếu cần
        if face_img.dtype != np.float32:
            face_img = face_img.astype(np.float32)
            
        # Kiểm tra cache
        img_hash = hash(face_img.tobytes())
        with self.cache_lock:
            if img_hash in self.embedding_cache:
                return self.embedding_cache[img_hash]
                
        try:
            # Sử dụng DeepFace để lấy embedding
            embedding_objs = DeepFace.represent(face_img, model_name='Facenet512', enforce_detection=False)
            
            if isinstance(embedding_objs, list) and len(embedding_objs) > 0:
                # Lấy vector embedding và chuyển về float32
                vec = np.array(embedding_objs[0]["embedding"], dtype=np.float32)
                
                # Chuẩn hóa L2
                norm = np.linalg.norm(vec)
                if not (0.99 < norm < 1.01):
                    logger.debug(f"[Embedding][{debug_label}] Normalizing vector with norm: {norm}")
                    vec = vec / (norm + 1e-10)
                
                # Kiểm tra vector hợp lệ
                if np.isnan(vec).any() or np.isinf(vec).any():
                    logger.warning(f"[Embedding][{debug_label}] Invalid embedding vector (NaN/Inf detected)")
                    return None
                    
                # Lưu vào cache
                with self.cache_lock:
                    self.embedding_cache[img_hash] = vec
                
                # Kiểm tra shape và dtype
                if vec.shape[-1] != 512:
                    logger.warning(f"[Embedding] Invalid shape: {vec.shape}")
                    return None
                    
                return vec
            else:
                logger.warning(f"[Embedding][{debug_label}] DeepFace did not return valid embedding")
                return None
        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
            return None

    def recognize_face(self, face_img: np.ndarray) -> Tuple[str, float, Optional[str]]:
        """
        Nhận diện khuôn mặt từ hình ảnh, trả về tên, độ tin cậy và ID.
        """
        logger.debug(f"Recognition database size: {len(self.student_embeddings)}")
        
        # Lấy các tham số từ cấu hình
        recognition_threshold = self.config.recognition_threshold
        minimum_confidence = self.config.minimum_confidence
        max_allowed_distance = self.config.max_allowed_distance
        
        # Kiểm tra đầu vào
        if face_img is None or face_img.size == 0:
            return "Unknown", 0.0, None
        
        # Lấy embedding
        current = self.get_face_embedding(face_img, debug_label="recognize_input")
        if current is None:
            return "Unknown", 0.0, None
        
        # Đảm bảo embedding được chuẩn hóa L2
        norm = np.linalg.norm(current)
        if not (0.99 < norm < 1.01):
            current = (current / (norm + 1e-10)).astype(np.float32)
        
        best_name = "Unknown"
        best_id = None
        min_dist = float("inf")
        
        # Phương pháp 1: Nhận diện nhanh sử dụng ma trận embedding
        if self.emb_matrix is not None and len(self.emb_ids) > 0:
            # Tính cosine distance hàng loạt
            sims = np.dot(self.emb_matrix, current)
            dists = 1 - sims  # cosine distance
            min_idx = np.argmin(dists)
            min_dist = float(dists[min_idx])
            
            # Kiểm tra khoảng cách
            if (min_dist < recognition_threshold and 
                min_dist < max_allowed_distance):
                confidence = 1.0 - min_dist
                
                # Chỉ chấp nhận kết quả có độ tin cậy cao
                if confidence >= minimum_confidence:
                    sid = self.emb_ids[min_idx]
                    data = self.student_embeddings.get(sid, {})
                    best_name = data.get("name", "Unknown")
                    best_id = sid
        else:
            # Phương pháp 2: Duyệt từng embedding (phương pháp dự phòng)
            for sid, data in self.student_embeddings.items():
                try:
                    if "embedding" not in data or data["embedding"] is None:
                        continue
                    
                    emb = data["embedding"]
                    # Đảm bảo emb được chuẩn hóa
                    emb_norm = np.linalg.norm(emb)
                    if not (0.99 < emb_norm < 1.01):
                        emb = emb / (emb_norm + 1e-10)
                    
                    # Tính khoảng cách cosine
                    dist = self.cosine_distance(current, emb)
                    
                    # Kiểm tra cả hai ngưỡng và lấy khoảng cách nhỏ nhất
                    if (dist < recognition_threshold and 
                        dist < max_allowed_distance and 
                        dist < min_dist):
                        min_dist = dist
                        confidence = 1.0 - dist
                        
                        # Chỉ chấp nhận kết quả có độ tin cậy cao
                        if confidence >= minimum_confidence:
                            best_name = data.get("name", "Unknown")
                            best_id = sid
                except Exception as e:
                    logger.error(f"Error comparing with embedding {sid}: {e}")
        
        # Tính độ tin cậy cuối cùng
        confidence = 1.0 - min_dist if best_name != "Unknown" else 0.0
        confidence = max(0.0, min(1.0, confidence))
        
        # Log kết quả nhận diện
        logger.info(f"Recognition result: {best_name}, Distance: {min_dist:.4f}, Confidence: {confidence:.2f}")
        
        return best_name, round(confidence, 2), best_id

    def load_embeddings(self, embeddings: Dict[str, Dict[str, Any]]):
        """
        Load embeddings from database, ensure all are numpy arrays (float32) and l2-normalized.
        """
        if not isinstance(embeddings, dict):
            logger.error(f"Invalid embeddings type: {type(embeddings)}")
            return
            
        valid_count = 0
        invalid_count = 0
        
        # Xử lý từng embedding trong database
        for sid, data in embeddings.items():
            if not isinstance(data, dict) or "embedding" not in data:
                logger.warning(f"Invalid embedding data for student {sid}")
                invalid_count += 1
                continue
                
            emb = data["embedding"]
            
            # Chuyển đổi list/tuple thành numpy array
            if isinstance(emb, (list, tuple)):
                try:
                    emb = np.array(emb, dtype=np.float32)
                except Exception as e:
                    logger.warning(f"Cannot convert embedding list to numpy for {sid}: {e}")
                    data["embedding"] = None
                    invalid_count += 1
                    continue
                    
            # Kiểm tra cấu trúc
            if not isinstance(emb, np.ndarray) or emb.shape != (512,):
                logger.warning(f"Invalid embedding shape for {sid}: {getattr(emb, 'shape', None)}")
                data["embedding"] = None
                invalid_count += 1
                continue
                
            # Chuẩn hóa L2
            norm = np.linalg.norm(emb)
            if not (0.99 < norm < 1.01):
                emb = emb / (norm + 1e-10)
                
            # Lưu embedding đã chuẩn hóa
            data["embedding"] = emb.astype(np.float32)
            valid_count += 1
            
        # Lưu embedding database
        self.student_embeddings = embeddings
        
        # Tạo ma trận embedding cho nhận diện nhanh
        emb_list = []
        id_list = []
        for sid, data in embeddings.items():
            emb = data.get("embedding")
            if (isinstance(emb, np.ndarray) and 
                emb.dtype == np.float32 and 
                emb.shape == (512,) and 
                not np.isnan(emb).any() and 
                not np.isinf(emb).any()):
                emb_list.append(emb)
                id_list.append(sid)
                
        # Tạo ma trận nếu có đủ dữ liệu
        if emb_list:
            self.emb_matrix = np.stack(emb_list, axis=0)
            self.emb_ids = id_list
        else:
            self.emb_matrix = None
            self.emb_ids = []
            
        # Xóa cache
        with self.cache_lock:
            self.embedding_cache.clear()
            
        logger.info(f"Loaded {valid_count} valid embeddings (invalid: {invalid_count})")

    def detect_faces(self, img: np.ndarray) -> List[Dict[str, Any]]:
        """Phát hiện khuôn mặt trong ảnh"""
        if img is None or img.size == 0:
            return []
            
        try:
            # Sử dụng detector với ngưỡng từ cấu hình
            bboxes, kpss = self.face_detector.detect(
                img, 
                thresh=self.config.detection_threshold,
                input_size=(640, 640)
            )
            
            # Chuyển đổi kết quả
            faces = []
            for i, bbox in enumerate(bboxes):
                x1, y1, x2, y2, score = bbox
                face = {
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": float(score)
                }
                if kpss is not None and i < len(kpss):
                    face["kps"] = kpss[i]
                faces.append(face)
                
            return faces
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            return []

    def _process_face(self, args: Tuple[np.ndarray, Dict[str, Any], bool]) -> Optional[Dict[str, Any]]:
        """
        Xử lý một khuôn mặt riêng lẻ - dùng cho đa luồng
        
        Args:
            args: (img, face_info, detect_only)
            
        Returns:
            Kết quả xử lý khuôn mặt
        """
        img, face, detect_only = args
        
        try:
            x1, y1, x2, y2 = face["bbox"]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.shape[1] - 1, x2), min(img.shape[0] - 1, y2)
            
            if x2 <= x1 or y2 <= y1:
                logger.warning(f"Invalid face region: [{x1},{y1},{x2},{y2}]")
                return None
                
            roi = img[y1:y2, x1:x2]
            if roi.size == 0:
                return None
                
            if detect_only:
                return {
                    "bbox": [x1, y1, x2, y2],
                    "confidence": face.get("confidence", 0.0),
                    "kps": face.get("kps", None),
                }
            else:
                # Nhận diện khuôn mặt
                name, conf, sid = self.recognize_face(roi)
                
                return {
                    "name": name,
                    "confidence": conf,
                    "student_id": sid,
                    "bbox": [x1, y1, x2, y2],
                    "kps": face.get("kps", None),
                }
        except Exception as e:
            logger.error(f"Error in _process_face: {e}")
            return None

    def process_image(self, image: Union[str, np.ndarray], detect_only: bool = False) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Xử lý ảnh để phát hiện và nhận diện khuôn mặt (đa luồng)
        
        Args:
            image: Đường dẫn ảnh hoặc mảng numpy
            detect_only: Chỉ phát hiện khuôn mặt, không nhận diện
            
        Returns:
            Tuple(ảnh đã xử lý, danh sách kết quả)
        """
        # Đọc ảnh nếu là đường dẫn
        if isinstance(image, str):
            if not os.path.isfile(image):
                logger.error(f"Image file not found: {image}")
                raise FileNotFoundError(f"Image not found: {image}")
            img = cv2.imread(image)
            if img is None:
                logger.error(f"Failed to read image: {image}")
                raise ValueError(f"Cannot read image: {image}")
        else:
            if image is None or image.size == 0:
                logger.error("Empty image array provided")
                raise ValueError("Invalid empty image array")
            img = image.copy()
            
        # Thay đổi kích thước nếu ảnh quá lớn
        h, w = img.shape[:2]
        if max(h, w) > self.config.max_dim:
            scale = self.config.max_dim / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            
        # Phát hiện khuôn mặt
        start_time = time.time()
        faces = self.detect_faces(img)
        detect_time = time.time() - start_time
        logger.debug(f"Face detection took {detect_time:.3f}s, found {len(faces)} faces")
        
        if not faces:
            return img, []
            
        # Tạo tác vụ cho từng khuôn mặt để xử lý đa luồng
        tasks = [(img, face, detect_only) for face in faces]
        results = []
        
        # Sử dụng thread pool để xử lý các khuôn mặt song song
        process_start = time.time()
        future_results = {self.executor.submit(self._process_face, task): i for i, task in enumerate(tasks)}
        
        # Thu thập kết quả theo thứ tự hoàn thành
        for future in concurrent.futures.as_completed(future_results):
            result = future.result()
            if result is not None:
                results.append(result)
                
        process_time = time.time() - process_start
        logger.debug(f"Face processing took {process_time:.3f}s for {len(faces)} faces ({process_time/max(1, len(faces)):.3f}s per face)")
        
        # Trả về ảnh và kết quả
        return img, results

    def process_image_batch(self, images: List[Union[str, np.ndarray]], detect_only: bool = False) -> List[Tuple[np.ndarray, List[Dict[str, Any]]]]:
        """
        Xử lý một loạt ảnh cùng lúc với đa luồng
        
        Args:
            images: Danh sách đường dẫn ảnh hoặc mảng numpy
            detect_only: Chỉ phát hiện khuôn mặt, không nhận diện
            
        Returns:
            Danh sách (ảnh đã xử lý, kết quả) cho mỗi ảnh đầu vào
        """
        if not images:
            return []
            
        # Tạo tác vụ cho từng ảnh
        future_results = {}
        for i, img in enumerate(images):
            future = self.executor.submit(self.process_image, img, detect_only)
            future_results[future] = i
            
        # Thu thập kết quả theo thứ tự của ảnh đầu vào
        results = [None] * len(images)
        for future in concurrent.futures.as_completed(future_results):
            idx = future_results[future]
            try:
                result = future.result()
                results[idx] = result
            except Exception as e:
                logger.error(f"Error processing image {idx}: {e}")
                results[idx] = (None, [])
                
        return results

    def add_face_embedding(self, student_id: str, face_img: np.ndarray, name: str = None) -> bool:
        """
        Thêm khuôn mặt mới vào cơ sở dữ liệu
        
        Args:
            student_id: ID học sinh
            face_img: Ảnh khuôn mặt
            name: Tên học sinh (tùy chọn)
            
        Returns:
            Thành công hay không
        """
        try:
            # Lấy embedding cho khuôn mặt mới
            embedding = self.get_face_embedding(face_img, debug_label=f"new_face_{student_id}")
            if embedding is None:
                logger.error(f"Could not generate embedding for student {student_id}")
                return False
                
            # Kiểm tra và thêm vào cơ sở dữ liệu
            student_data = self.student_embeddings.get(student_id, {})
            student_data["embedding"] = embedding
            
            if name is not None:
                student_data["name"] = name
                
            self.student_embeddings[student_id] = student_data
            
            # Cập nhật ma trận embedding
            self._update_embedding_matrix()
            
            logger.info(f"Added new face embedding for student {student_id}")
            return True
        except Exception as e:
            logger.error(f"Error adding face embedding: {e}")
            return False
            
    def _update_embedding_matrix(self):
        """Cập nhật ma trận embedding cho nhận diện nhanh"""
        emb_list = []
        id_list = []
        for sid, data in self.student_embeddings.items():
            emb = data.get("embedding")
            if (isinstance(emb, np.ndarray) and 
                emb.dtype == np.float32 and 
                emb.shape == (512,) and 
                not np.isnan(emb).any() and 
                not np.isinf(emb).any()):
                emb_list.append(emb)
                id_list.append(sid)
                
        # Tạo ma trận nếu có đủ dữ liệu
        if emb_list:
            self.emb_matrix = np.stack(emb_list, axis=0)
            self.emb_ids = id_list
        else:
            self.emb_matrix = None
            self.emb_ids = []

    def clear_cache(self):
        """Xóa cache embedding"""
        with self.cache_lock:
            self.embedding_cache.clear()
        logger.info("Embedding cache cleared")
        
    def close(self):
        """Giải phóng tài nguyên"""
        try:
            with self.cache_lock:
                self.embedding_cache.clear()
            self.student_embeddings.clear()
            self.emb_matrix = None
            self.emb_ids = []
            logger.info("Face Embedding module closed and resources released")
        except Exception as e:
            logger.error(f"Error closing Face Embedding module: {e}")