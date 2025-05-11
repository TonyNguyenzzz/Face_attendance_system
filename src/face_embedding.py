import cv2
import logging
import numpy as np
import os
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import OrderedDict
import threading
from deepface import DeepFace

logger = logging.getLogger("FaceEmbedding")

class FaceEmbeddingConfig:
    def __init__(
        self,
        model_dir: str,
        min_confidence: float = 0.5,
        distance_threshold: float = 0.95,
        max_dim: int = 800,
        face_size: Tuple[int, int] = (160, 160),
        cache_size: int = 1000,
        max_workers: int = 4,
        min_confidence_threshold: float = 0.6,
        absolute_distance_threshold: float = 0.6
    ):
        self.model_dir = model_dir
        self.min_confidence = min_confidence
        self.distance_threshold = distance_threshold
        self.max_dim = max_dim
        self.face_size = face_size
        self.cache_size = cache_size
        self.max_workers = max_workers
        self.min_confidence_threshold = min_confidence_threshold
        self.absolute_distance_threshold = absolute_distance_threshold

    def get(self, key: str, default=None):
        """Get configuration value with default fallback"""
        return getattr(self, key, default)

def dict_to_face_embedding_config(cfg: dict) -> FaceEmbeddingConfig:
    return FaceEmbeddingConfig(
        model_dir=cfg.get("model_dir", "./models"),
        min_confidence=cfg.get("min_confidence", 0.5),
        distance_threshold=cfg.get("distance_threshold", 0.95),
        max_dim=cfg.get("max_dim", 800),
        face_size=cfg.get("face_size", (160, 160)),
        cache_size=cfg.get("cache_size", 1000),
        max_workers=cfg.get("max_workers", 4),
        min_confidence_threshold=cfg.get("min_confidence_threshold", 0.6),
        absolute_distance_threshold=cfg.get("absolute_distance_threshold", 0.6)
    )

class LimitedSizeDict(OrderedDict):
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
        self._pool = None
        self.config = config
        self.embedding_cache = LimitedSizeDict(size_limit=getattr(config, 'cache_size', 1000))
        self.student_embeddings: Dict[str, Dict[str, Any]] = {}
        self.cache_lock = threading.Lock()
        from face_detector import FaceDetector
        # Giữ nguyên phát hiện khuôn mặt bằng SCRFD
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        model_path = os.path.join(project_root, "models", "model_3.onnx")
        if not os.path.exists(model_path):
            logger.error(f"Face detection model not found at: {model_path}")
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.face_detector = FaceDetector(model_path)
        # logger.info("Face Embedding module initialized with DeepFace (facenet512)")

    def get_face_embedding(self, face_img: np.ndarray, enforce_detection=False) -> Optional[np.ndarray]:
        try:
            # DeepFace sẽ tự resize và chuẩn hóa ảnh đầu vào
            embedding_objs = DeepFace.represent(face_img, model_name='Facenet512', enforce_detection=enforce_detection)
            if isinstance(embedding_objs, list) and len(embedding_objs) > 0:
                vec = embedding_objs[0]["embedding"]
                return np.array(vec, dtype=np.float32)
            return None
        except Exception as e:
            logger.error(f"Error getting face embedding with DeepFace: {e}")
            return None

    def _preprocess_face(self, face_img: np.ndarray) -> Optional[np.ndarray]:
        try:
            if face_img is None or face_img.size == 0:
                return None
            if face_img.ndim == 2:
                face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)
            elif face_img.shape[2] == 4:
                face_img = cv2.cvtColor(face_img, cv2.COLOR_RGBA2RGB)
            elif face_img.shape[2] == 3:
                if face_img.dtype != np.uint8:
                    face_img = np.clip(face_img, 0, 255).astype(np.uint8)
                # Thêm chuyển đổi BGR -> RGB nếu ảnh có 3 kênh
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            # logger.debug(f"[Preprocess][Before] min={face_img.min()}, max={face_img.max()}, mean={face_img.mean()}")
            face_img = cv2.resize(face_img, self.config.face_size, interpolation=cv2.INTER_AREA)
            face_img = face_img.astype(np.float32)
            face_img = (face_img / 127.5) - 1.0  # Chuẩn hóa về [-1,1] cho Facenet
            # logger.debug(f"[Preprocess][After] min={face_img.min()}, max={face_img.max()}, mean={face_img.mean()}")
            return face_img
        except Exception as e:
            return None

    def l2_normalize(self, x, axis=-1, epsilon=1e-10):
        return x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))

    def cosine(self, a, b):
        """Calculate cosine distance between two vectors."""
        if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
            return 1.0  # Maximum distance for invalid inputs
        if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape:
            return 1.0  # Maximum distance for incompatible shapes
        # Ensure both vectors are L2 normalized
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm < 0.99 or a_norm > 1.01:
            a = a / (a_norm + 1e-10)
        if b_norm < 0.99 or b_norm > 1.01:
            b = b / (b_norm + 1e-10)
        # Calculate dot product
        dot_product = np.dot(a, b)
        # Clip to [-1, 1] to handle numerical issues
        dot_product = np.clip(dot_product, -1.0, 1.0)
        # Cosine distance = 1 - cosine similarity
        return 1.0 - dot_product

    def get_face_embedding(self, face_img: np.ndarray, debug_label=None) -> Optional[np.ndarray]:
        # Ghi log embedding vector ra file nếu có label đặc biệt
        log_to_file = False
        log_file = None
        # if debug_label is not None and ("register" in debug_label or "recognize" in debug_label):
            # log_to_file = True
            # log_file = f"embedding_{debug_label}_{int(time.time())}.txt"
        
        if face_img is None or face_img.size == 0:
            logger.warning("Empty face image provided")
            return None
        if face_img.ndim != 3 or face_img.shape[2] != 3:
            logger.warning("Invalid face image shape")
            return None
        if face_img.dtype != np.float32:
            # logger.info("Auto converting face image dtype from %s to float32", face_img.dtype)
            face_img = face_img.astype(np.float32)
        img_hash = hash(face_img.tobytes())
        with self.cache_lock:
            if img_hash in self.embedding_cache:
                return self.embedding_cache[img_hash]
        try:
            # Sử dụng DeepFace để lấy embedding
            embedding_objs = DeepFace.represent(face_img, model_name='Facenet512', enforce_detection=False)
            if isinstance(embedding_objs, list) and len(embedding_objs) > 0:
                vec = np.array(embedding_objs[0]["embedding"], dtype=np.float32)
                # CHUẨN HÓA L2
                norm = np.linalg.norm(vec)
                if not (0.99 < norm < 1.01):
                    # logger.warning(f"[Embedding][{debug_label}] L2 norm is not close to 1: {norm}, will normalize!")
                    vec = vec / (norm + 1e-10)
                    norm = np.linalg.norm(vec)
                # logger.debug(f"[Embedding][{debug_label}] shape={vec.shape}, dtype={vec.dtype}, norm={norm:.4f}, nan={np.isnan(vec).any()}, inf={np.isinf(vec).any()}")
                # logger.debug(f"[Embedding][{debug_label}] first 10 values: {vec[:10]}")
                # if log_to_file and log_file:
                #     with open(log_file, 'w', encoding='utf-8') as f:
                #         f.write(f"Embedding ({debug_label}):\n")
                #         f.write(str(vec.tolist()) + "\n")
                #         f.write(f"Norm: {norm}\n")
                if np.isnan(vec).any() or np.isinf(vec).any():
                    logger.warning(f"[Embedding][{debug_label}] NAN or INF detected in embedding vector!")
                if not (0.99 < norm < 1.01):
                    logger.warning(f"[Embedding][{debug_label}] L2 norm is not close to 1: {norm}")
                if vec is None:
                    logger.warning(f"[Embedding][{debug_label}] Embedding vector is None!")
                with self.cache_lock:
                    self.embedding_cache[img_hash] = vec
                # Kiểm tra shape và dtype
                if vec.shape[-1] != 512 or vec.dtype != np.float32:
                    logger.warning(f"[Embedding][{debug_label}] Invalid embedding shape or dtype: shape={vec.shape}, dtype={vec.dtype}")
                    return None
                return vec
            else:
                logger.warning(f"[Embedding][{debug_label}] DeepFace did not return valid embedding!")
                return None
        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
            return None

    def batch_get_face_embeddings(self, face_imgs: List[np.ndarray]) -> List[Optional[np.ndarray]]:
        if not face_imgs:
            return []
        results: List[Optional[np.ndarray]] = []
        uncached_imgs = []
        uncached_indices = []
        for i, img in enumerate(face_imgs):
            if img is None or img.size == 0:
                results.append(None)
                continue
            img_hash = hash(img.tobytes())
            with self.cache_lock:
                if img_hash in self.embedding_cache:
                    results.append(self.embedding_cache[img_hash])
                else:
                    results.append(None)
                    uncached_imgs.append(img)
                    uncached_indices.append(i)
        if not uncached_imgs:
            return results
        processed_imgs = [self._preprocess_face(img) for img in uncached_imgs]
        valid_imgs = [img for img in processed_imgs if img is not None]
        valid_indices = [i for i, img in enumerate(processed_imgs) if img is not None]
        if not valid_imgs:
            return results
        try:
            batch_inp = np.stack(valid_imgs, axis=0)
            embs_out = self.emb_session.run(None, {self.emb_input_name: batch_inp})[0]
            for i, orig_list_idx in enumerate(valid_indices):
                vec = embs_out[i]
                idx_in_results = uncached_indices[orig_list_idx]
                results[idx_in_results] = vec
                with self.cache_lock:
                    img_hash2 = hash(uncached_imgs[orig_list_idx].tobytes())
                    self.embedding_cache[img_hash2] = vec
            return results
        except Exception as e:
            logger.error(f"Error in batch embedding generation: {e}")
            return results

    def recognize_face(self, face_img: np.ndarray) -> Tuple[str, float, Optional[str]]:
        """
        Nhận diện khuôn mặt từ hình ảnh, trả về tên, độ tin cậy và ID.
        Đã cải thiện để ngăn chặn nhận diện sai với ngưỡng tin cậy thấp.
        """
        logger.info(f"[Nhận diện] Số lượng embedding trong DB: {len(self.student_embeddings)}")
        logger.info(f"[Nhận diện] Ngưỡng sử dụng (distance_threshold): {self.config.distance_threshold}")
        
        if face_img is None or face_img.size == 0:
            return "Unknown", 0.0, None
        
        # Gán nhãn debug cho embedding nhận diện
        current = self.get_face_embedding(face_img, debug_label="recognize_input")
        if current is None:
            return "Unknown", 0.0, None
        
        # Đảm bảo embedding được chuẩn hóa L2
        norm = np.linalg.norm(current)
        if not (current.dtype == np.float32 and abs(norm - 1.0) < 1e-3):
            current = (current / (norm + 1e-10)).astype(np.float32)
        
        # Kiểm tra thông tin embedding
        logger.debug(f"[Recognize Input] shape={current.shape}, dtype={current.dtype}, norm={norm:.4f}")
        
        # Lấy tham số từ config
        distance_threshold = self.config.distance_threshold
        min_confidence_threshold = self.config.get("min_confidence_threshold", 0.6)
        absolute_distance_threshold = self.config.get("absolute_distance_threshold", 0.6)
        
        logger.debug(f"[Threshold] Recognition threshold = {distance_threshold}")
        logger.debug(f"[Threshold] Min confidence threshold = {min_confidence_threshold}")
        logger.debug(f"[Threshold] Absolute distance threshold = {absolute_distance_threshold}")
        
        best_name = "Unknown"
        best_id = None
        min_dist = float("inf")
        
        # Phương pháp 1: Nhận diện nhanh sử dụng ma trận embedding
        if hasattr(self, "emb_matrix") and self.emb_matrix is not None and len(self.emb_ids) > 0:
            # Tính cosine distance hàng loạt
            sims = np.dot(self.emb_matrix, current)
            dists = 1 - sims  # cosine distance
            min_idx = np.argmin(dists)
            min_dist = float(dists[min_idx])
            
            # Kiểm tra và log thông tin khoảng cách nhỏ nhất
            logger.debug(f"[Nearest] Min distance: {min_dist:.4f}, idx={min_idx}")
            
            # Kiểm tra cả ngưỡng khoảng cách và ngưỡng tin cậy tối thiểu
            if min_dist < distance_threshold and min_dist < absolute_distance_threshold:
                confidence = 1.0 - min_dist
                logger.debug(f"[Confidence] Calculated: {confidence:.4f}, Min required: {min_confidence_threshold}")
                
                # Chỉ chấp nhận kết quả có độ tin cậy cao
                if confidence >= min_confidence_threshold:
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
                    
                    # Sử dụng phương thức cosine đã thêm ở trên
                    dist = self.cosine(current, emb)
                    logger.debug(f"[Cosine Distance] To {sid}: {dist:.4f}")
                    
                    # Kiểm tra cả hai ngưỡng và lấy khoảng cách nhỏ nhất
                    if dist < distance_threshold and dist < absolute_distance_threshold and dist < min_dist:
                        min_dist = dist
                        confidence = 1.0 - dist
                        
                        # Chỉ chấp nhận kết quả có độ tin cậy cao
                        if confidence >= min_confidence_threshold:
                            best_name = data.get("name", "Unknown")
                            best_id = sid
                except Exception as e:
                    logger.error(f"Error comparing with embedding {sid}: {e}")
        
        # Tính độ tin cậy cuối cùng
        confidence = 1.0 - min_dist if best_name != "Unknown" else 0.0
        confidence = max(0.0, min(1.0, confidence))
        
        # Log kết quả nhận diện
        logger.info(f"[Nhận diện] Kết quả: {best_name}, Khoảng cách nhỏ nhất: {min_dist:.4f}, Độ tin cậy: {confidence:.2f}")
        
        return best_name, round(confidence, 2), best_id

    def load_embeddings(self, embeddings: Dict[str, Dict[str, Any]]):
        """
        Load embeddings from DB, ensure all are numpy arrays (float32) and l2-normalized.
        Log and skip invalid/corrupt embedding.
        """
        import numpy as np
        valid_count = 0
        invalid_count = 0
        if not isinstance(embeddings, dict):
            logger.error(f"Invalid embeddings type: {type(embeddings)}")
            return
        for sid, data in embeddings.items():
            if not isinstance(data, dict) or "embedding" not in data:
                logger.warning(f"Invalid embedding data for student {sid}")
                invalid_count += 1
                continue
            emb = data["embedding"]
            # Nếu là list hoặc tuple thì ép về numpy array
            if isinstance(emb, (list, tuple)):
                try:
                    emb = np.array(emb, dtype=np.float32)
                except Exception as e:
                    logger.warning(f"Cannot convert embedding list to numpy for {sid}: {e}")
                    data["embedding"] = None
                    invalid_count += 1
                    continue
            # Nếu không phải numpy array hoặc shape không đúng thì bỏ qua
            if not isinstance(emb, np.ndarray) or emb.shape != (512,):
                logger.warning(f"Embedding for {sid} is not a valid numpy array of shape (512,): {type(emb)}, shape={getattr(emb, 'shape', None)}")
                data["embedding"] = None
                invalid_count += 1
                continue
            # CHUẨN HÓA LẠI L2
            norm = np.linalg.norm(emb)
            if not (0.99 < norm < 1.01):
                logger.warning(f"Embedding for {sid} L2 norm is not close to 1: {norm}, will normalize!")
                emb = emb / (norm + 1e-10)
            # Final L2 normalization and ensure float32 dtype
            emb = self.l2_normalize(emb).astype(np.float32)
            data["embedding"] = emb
            norm = np.linalg.norm(emb)
            # logger.debug(f"[DB Embedding][{sid}] shape={emb.shape}, dtype={emb.dtype}, norm={norm:.4f}, nan={np.isnan(emb).any()}, inf={np.isinf(emb).any()}")
            if np.isnan(emb).any() or np.isinf(emb).any():
                logger.warning(f"[DB Embedding][{sid}] NAN or INF detected in embedding vector!")
            if not (0.99 < norm < 1.01):
                logger.warning(f"[DB Embedding][{sid}] L2 norm is not close to 1: {norm}")
            valid_count += 1
        self.student_embeddings = embeddings
        # Tạo emb_matrix và emb_ids để nhận diện nhanh
        emb_list = []
        id_list = []
        for sid, data in embeddings.items():
            emb = data.get("embedding")
            if isinstance(emb, np.ndarray) and emb.dtype == np.float32 and emb.shape[-1] == 512 and not np.isnan(emb).any() and not np.isinf(emb).any():
                emb_list.append(emb)
                id_list.append(sid)
        if emb_list:
            self.emb_matrix = np.stack(emb_list, axis=0)
            self.emb_ids = id_list
        else:
            self.emb_matrix = None
            self.emb_ids = []
        with self.cache_lock:
            self.embedding_cache.clear()
        # logger.info(f"Loaded {valid_count} valid embeddings out of {len(embeddings)} total (invalid: {invalid_count})")

    def detect_faces(self, img: np.ndarray) -> List[Dict[str, Any]]:
        if img is None or img.size == 0:
            return []
        try:
            bboxes, kpss = self.face_detector.detect(
                img, 
                thresh=self.config.min_confidence,
                input_size=(640, 640)
            )
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

    def extract_face_from_image(self, image_path: str) -> Optional[np.ndarray]:
        if not os.path.isfile(image_path):
            logger.error(f"Image file not found: {image_path}")
            return None
        try:
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Failed to read image: {image_path}")
                return None
            faces = self.detect_faces(img)
            if not faces:
                logger.warning(f"No faces detected in {image_path}")
                return None
            best = max(faces, key=lambda x: x.get("confidence", 0))
            x1, y1, x2, y2 = best["bbox"]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.shape[1] - 1, x2), min(img.shape[0] - 1, y2)
            if x2 <= x1 or y2 <= y1:
                logger.error(f"Invalid face region: [{x1},{y1},{x2},{y2}]")
                return None
            roi = img[y1:y2, x1:x2]
            if roi.size == 0:
                logger.error("Empty face region")
                return None
            if "kps" in best:
                roi = self.align_face(roi, best["kps"])
            return roi
        except Exception as e:
            logger.error(f"Error extracting face from {image_path}: {e}")
            return None

    def process_image(self, image: Union[str, np.ndarray], detect_only: bool = False) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
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
        h, w = img.shape[:2]
        if max(h, w) > self.config.max_dim:
            scale = self.config.max_dim / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        faces = self.detect_faces(img)
        results: List[Dict[str, Any]] = []
        if detect_only:
            for f in faces:
                x1, y1, x2, y2 = f["bbox"]
                results.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": f.get("confidence", 0.0),
                    "kps": f.get("kps", None),
                })
        else:
            face_crops = []
            valid_faces = []
            for f in faces:
                x1, y1, x2, y2 = f["bbox"]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img.shape[1] - 1, x2), min(img.shape[0] - 1, y2)
                if x2 <= x1 or y2 <= y1:
                    logger.warning(f"Invalid face region: [{x1},{y1},{x2},{y2}]")
                    continue
                roi = img[y1:y2, x1:x2]
                if roi.size == 0:
                    continue
                if "kps" in f:
                    roi = self.align_face(roi, f["kps"])
                face_crops.append(roi)
                valid_faces.append(f)
            if face_crops:
                names_confs_ids = [self.recognize_face(crop) for crop in face_crops]
                for i, (name, conf, sid) in enumerate(names_confs_ids):
                    f = valid_faces[i]
                    x1, y1, x2, y2 = f["bbox"]
                    results.append({
                        "name": name,
                        "confidence": conf,
                        "student_id": sid,
                        "bbox": [x1, y1, x2, y2],
                        "kps": f.get("kps", None),
                    })
        return img, results

    def _get_pool(self):
        if self._pool is None or self._pool._broken:
            self._pool = ProcessPoolExecutor(max_workers=self.config.max_workers)
        return self._pool

    def batch_process_images(self, image_paths: List[str], timeout: Optional[float] = None) -> List[Dict[str, Any]]:
        if not image_paths:
            return []
        valid_paths = [p for p in image_paths if os.path.isfile(p)]
        if len(valid_paths) < len(image_paths):
            logger.warning(f"Skipping {len(image_paths) - len(valid_paths)} invalid image paths")
        if not valid_paths:
            return []
        results = []
        try:
            pool = self._get_pool()
            futures = [pool.submit(self.process_single_image, path) for path in valid_paths]
            for future in as_completed(futures, timeout=timeout):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing image in batch: {e}")
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
        return results

    def process_single_image(self, image_path: str) -> Dict[str, Any]:
        try:
            start_time = time.time()
            img, faces = self.process_image(image_path)
            processing_time = time.time() - start_time
            return {
                "path": image_path,
                "success": True,
                "faces": faces,
                "processed_image": img,
                "processing_time": processing_time
            }
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return {
                "path": image_path, 
                "success": False, 
                "error": str(e),
                "faces": []
            }

    def clear_cache(self):
        with self.cache_lock:
            self.embedding_cache.clear()
        # logger.info("Embedding cache cleared")

    def close(self):
        try:
            if self._pool:
                self._pool.shutdown(wait=False)
                self._pool = None
            with self.cache_lock:
                self.embedding_cache.clear()
            self.student_embeddings.clear()
            # logger.info("Face Embedding module closed and resources released")
        except Exception as e:
            logger.error(f"Error closing Face Embedding module: {e}")

    def test_recognition_thresholds(self, test_img: np.ndarray, distance_range=(0.3, 0.9, 0.05)):
        """
        Kiểm tra nhiều ngưỡng khoảng cách khác nhau để tìm ngưỡng tối ưu.
        Hữu ích khi tinh chỉnh hệ thống.
        
        Args:
            test_img: Ảnh khuôn mặt cần kiểm tra
            distance_range: Tuple (start, end, step) cho dải ngưỡng cần thử
            
        Returns:
            List[Dict]: Kết quả nhận diện với các ngưỡng khác nhau
        """
        if test_img is None or test_img.size == 0:
            return []
        
        test_embedding = self.get_face_embedding(test_img, debug_label="test_threshold")
        if test_embedding is None:
            return []
        
        results = []
        original_threshold = self.config.distance_threshold
        
        # Lưu trữ cấu hình hiện tại
        try:
            # Thử nhiều ngưỡng khác nhau
            for threshold in np.arange(distance_range[0], distance_range[1], distance_range[2]):
                self.config.distance_threshold = threshold
                name, confidence, sid = self.recognize_face(test_img)
                results.append({
                    "threshold": threshold,
                    "result": name,
                    "confidence": confidence,
                    "student_id": sid,
                    "status": "Match" if name != "Unknown" else "Unknown"
                })
        finally:
            # Khôi phục cấu hình ban đầu
            self.config.distance_threshold = original_threshold
        
        return results