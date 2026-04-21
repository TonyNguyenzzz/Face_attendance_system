"""
Face Detector module using SCRFD ONNX model.
Provides efficient face detection with landmark support.
"""
import os
import cv2
import numpy as np
import logging
from typing import Optional, Tuple, List
from onnxruntime import InferenceSession

from onnx_config import configure_onnx_providers

logger = logging.getLogger(__name__)


def distance2bbox(
    points: np.ndarray, 
    distance: np.ndarray, 
    max_shape: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """Decode distance prediction to bounding box."""
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    
    if max_shape is not None:
        x1 = np.clip(x1, 0, max_shape[1])
        y1 = np.clip(y1, 0, max_shape[0])
        x2 = np.clip(x2, 0, max_shape[1])
        y2 = np.clip(y2, 0, max_shape[0])
    
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(
    points: np.ndarray, 
    distance: np.ndarray, 
    max_shape: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """Decode distance prediction to keypoints."""
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        
        if max_shape is not None:
            px = np.clip(px, 0, max_shape[1])
            py = np.clip(py, 0, max_shape[0])
        
        preds.append(px)
        preds.append(py)
    
    return np.stack(preds, axis=-1)


class FaceDetector:
    """SCRFD-based face detector with ONNX runtime."""
    
    def __init__(
        self,
        onnx_file: Optional[str] = None,
        session: Optional[InferenceSession] = None,
        nms_thresh: float = 0.4,
        mean_vals: Tuple[float, float, float] = (127.5, 127.5, 127.5),
        scale: float = 1.0 / 128,
        max_cache_size: int = 100
    ):
        """
        Initialize FaceDetector with ONNX model.
        
        Args:
            onnx_file: Path to ONNX model file
            session: Pre-initialized ONNX session
            nms_thresh: NMS threshold for filtering boxes
            mean_vals: Mean values for input normalization
            scale: Scale factor for input normalization
            max_cache_size: Maximum size for anchor center cache
        """
        self.session = session
        self.batched = False
        self.nms_thresh = nms_thresh
        self.mean_vals = mean_vals
        self.scale = scale
        self.max_cache_size = max_cache_size
        self.center_cache: dict = {}
        
        if self.session is None:
            if not onnx_file or not os.path.exists(onnx_file):
                raise FileNotFoundError(f"ONNX model file not found: {onnx_file}")
            
            providers = configure_onnx_providers()
            self.session = InferenceSession(onnx_file, providers=providers)
        
        self._init_vars()
    
    def _init_vars(self) -> None:
        """Initialize model configuration variables."""
        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape
        
        # Check if dynamic input size
        if isinstance(input_shape[2], str):
            self.input_size = None
        else:
            self.input_size = tuple(input_shape[2:4][::-1])
        
        self.input_name = input_cfg.name
        outputs = self.session.get_outputs()
        
        if len(outputs[0].shape) == 3:
            self.batched = True
        
        self.output_names = [o.name for o in outputs]
        self.use_kps = False
        self._num_anchors = 1
        
        # Configure based on output count
        output_configs = {
            6: (3, [8, 16, 32], 2, False),
            9: (3, [8, 16, 32], 2, True),
            10: (5, [8, 16, 32, 64, 128], 1, False),
            15: (5, [8, 16, 32, 64, 128], 1, True),
        }
        
        if len(outputs) in output_configs:
            cfg = output_configs[len(outputs)]
            self.fmc, self._feat_stride_fpn, self._num_anchors, self.use_kps = cfg
    
    def _get_anchor_centers(
        self, 
        height: int, 
        width: int, 
        stride: int
    ) -> np.ndarray:
        """Get anchor centers for a feature map."""
        key = (height, width, stride)
        
        if key in self.center_cache:
            return self.center_cache[key]
        
        # Calculate anchor centers
        anchor_centers = np.mgrid[:height, :width][::-1].astype(np.float32)
        anchor_centers = np.stack(anchor_centers, axis=-1)
        anchor_centers = (anchor_centers * stride).reshape((-1, 2))
        
        if self._num_anchors > 1:
            anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1)
            anchor_centers = anchor_centers.reshape((-1, 2))
        
        # Manage cache size
        if len(self.center_cache) >= self.max_cache_size:
            self.center_cache.pop(next(iter(self.center_cache)))
        
        self.center_cache[key] = anchor_centers
        return anchor_centers
    
    def _process_feature_level(
        self,
        scores: np.ndarray,
        boxes: np.ndarray,
        points: Optional[np.ndarray],
        anchor_centers: np.ndarray,
        stride: int,
        thresh: float
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Process detections from a single feature level."""
        pos_index = np.where(scores >= thresh)[0]
        
        if len(pos_index) == 0:
            return None, None, None
        
        bboxes = distance2bbox(anchor_centers, boxes)
        
        pos_scores = scores[pos_index]
        pos_bboxes = bboxes[pos_index]
        
        pos_kpss = None
        if self.use_kps and points is not None:
            kpss = distance2kps(anchor_centers, points)
            kpss = kpss.reshape((kpss.shape[0], -1, 2))
            pos_kpss = kpss[pos_index]
        
        return pos_scores, pos_bboxes, pos_kpss
    
    def forward(
        self, 
        img: np.ndarray, 
        thresh: float
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """Perform forward pass for face detection."""
        scores_list, bboxes_list, kps_list = [], [], []
        
        try:
            input_size = tuple(img.shape[0:2][::-1])
            blob = cv2.dnn.blobFromImage(
                img, self.scale, input_size, self.mean_vals, swapRB=True
            )
            
            net_outs = self.session.run(self.output_names, {self.input_name: blob})
            
            input_height, input_width = blob.shape[2:]
            
            for idx, stride in enumerate(self._feat_stride_fpn):
                if self.batched:
                    scores = net_outs[idx][0]
                    boxes = net_outs[idx + self.fmc][0] * stride
                    points = net_outs[idx + self.fmc * 2][0] * stride if self.use_kps else None
                else:
                    scores = net_outs[idx]
                    boxes = net_outs[idx + self.fmc] * stride
                    points = net_outs[idx + self.fmc * 2] * stride if self.use_kps else None
                
                height = input_height // stride
                width = input_width // stride
                
                anchor_centers = self._get_anchor_centers(height, width, stride)
                
                pos_scores, pos_bboxes, pos_kpss = self._process_feature_level(
                    scores, boxes, points, anchor_centers, stride, thresh
                )
                
                if pos_scores is not None:
                    scores_list.append(pos_scores)
                    bboxes_list.append(pos_bboxes)
                    if pos_kpss is not None:
                        kps_list.append(pos_kpss)
            
            return scores_list, bboxes_list, kps_list
            
        except Exception as e:
            logger.error(f"Forward pass error: {e}")
            return [], [], []
    
    def detect(
        self,
        image: np.ndarray,
        thresh: float = 0.5,
        input_size: Optional[Tuple[int, int]] = None,
        max_num: int = 0,
        metric: str = 'default'
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Detect faces in an image.
        
        Args:
            image: Input image (BGR format)
            thresh: Confidence threshold
            input_size: Input size for model
            max_num: Maximum number of faces to return
            metric: Selection metric ('default' or 'max')
            
        Returns:
            Tuple of (bounding_boxes, keypoints)
        """
        if image is None or image.size == 0:
            return np.array([]), None
        
        effective_input_size = input_size or self.input_size
        if effective_input_size is None:
            raise ValueError("input_size must be provided")
        
        try:
            # Resize image
            im_ratio = float(image.shape[0]) / image.shape[1]
            model_ratio = float(effective_input_size[1]) / effective_input_size[0]
            
            if im_ratio > model_ratio:
                new_height = effective_input_size[1]
                new_width = int(new_height / im_ratio)
            else:
                new_width = effective_input_size[0]
                new_height = int(new_width * im_ratio)
            
            det_scale = float(new_height) / image.shape[0]
            resized_img = cv2.resize(image, (new_width, new_height))
            
            # Prepare padded image
            det_img = np.zeros(
                (effective_input_size[1], effective_input_size[0], 3), 
                dtype=np.uint8
            )
            det_img[:new_height, :new_width, :] = resized_img
            
            # Forward pass
            scores_list, bboxes_list, kps_list = self.forward(det_img, thresh)
            
            if not scores_list:
                return np.array([]), None
            
            # Combine results
            scores = np.vstack(scores_list)
            order = scores.ravel().argsort()[::-1]
            bboxes = np.vstack(bboxes_list) / det_scale
            
            kpss = None
            if self.use_kps and kps_list:
                kpss = np.vstack(kps_list) / det_scale
                kpss = kpss[order, :, :]
            
            # Apply NMS
            pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
            pre_det = pre_det[order, :]
            keep = self.nms(pre_det)
            det = pre_det[keep, :]
            
            # Limit number of detections
            if 0 < max_num < det.shape[0]:
                area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                img_center = image.shape[0] // 2, image.shape[1] // 2
                
                offsets = np.vstack([
                    (det[:, 0] + det[:, 2]) / 2 - img_center[1],
                    (det[:, 1] + det[:, 3]) / 2 - img_center[0]
                ])
                offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                
                values = area if metric == 'max' else area - offset_dist_squared * 2.0
                bindex = np.argsort(values)[::-1][:max_num]
                
                det = det[bindex, :]
                if kpss is not None:
                    kpss = kpss[bindex, :, :]
            
            return det, kpss
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return np.array([]), None
    
    def nms(self, dets: np.ndarray) -> List[int]:
        """Apply Non-Maximum Suppression."""
        if dets.shape[0] == 0:
            return []
        
        x1, y1 = dets[:, 0], dets[:, 1]
        x2, y2 = dets[:, 2], dets[:, 3]
        scores = dets[:, 4]
        
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            index = np.where(ovr <= self.nms_thresh)[0]
            order = order[index + 1]
        
        return keep
