import os
import cv2
import numpy as np
import onnxruntime as ort
import logging
from onnxruntime import InferenceSession

# Thiết lập logger với mức độ cao hơn để loại bỏ thông báo không cần thiết
logger = logging.getLogger("FaceDetector")
logger.setLevel(logging.WARNING)

def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.
    
    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4 boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.
        
    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clip(min=0, max=max_shape[1])
        y1 = y1.clip(min=0, max=max_shape[0])
        x2 = x2.clip(min=0, max=max_shape[1])
        y2 = y2.clip(min=0, max=max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)

def distance2kps(points, distance, max_shape=None):
    """Decode distance prediction to keypoints.
    
    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4 boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.
        
    Returns:
        Tensor: Decoded keypoints.
    """
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        if max_shape is not None:
            px = px.clip(min=0, max=max_shape[1])
            py = py.clip(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)

class FaceDetector:
    def __init__(self, onnx_file=None, session=None, nms_thresh=0.4, 
                 mean_vals=(127.5, 127.5, 127.5), scale=1.0/128, 
                 max_cache_size=100):
        """Initialize FaceDetector with ONNX model.
        
        Args:
            onnx_file (str): Path to ONNX model file.
            session (InferenceSession): Pre-initialized ONNX session.
            nms_thresh (float): NMS threshold for filtering boxes.
            mean_vals (tuple): Mean values for input normalization.
            scale (float): Scale factor for input normalization.
            max_cache_size (int): Maximum size for center cache.
        """
        self.session = session
        self.batched = False
        self.nms_thresh = nms_thresh
        self.mean_vals = mean_vals
        self.scale = scale
        self.max_cache_size = max_cache_size
        self.center_cache = {}
        
        if self.session is None:
            if not onnx_file or not os.path.exists(onnx_file):
                raise FileNotFoundError(f"ONNX model file not found: {onnx_file}")
            
            # Sử dụng module onnx_config để cấu hình providers phù hợp
            from onnx_config import configure_onnx_providers
            providers = configure_onnx_providers()
            self.session = InferenceSession(onnx_file, providers=providers)
            
        self._init_vars()

    def _init_vars(self):
        """Initialize model configuration variables."""
        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape
        
        if isinstance(input_shape[2], str):
            self.input_size = None
        else:
            self.input_size = tuple(input_shape[2:4][::-1])
            
        input_name = input_cfg.name
        outputs = self.session.get_outputs()
        
        if len(outputs[0].shape) == 3:
            self.batched = True
            
        output_names = [o.name for o in outputs]
        self.input_name = input_name
        self.output_names = output_names
        self.use_kps = False
        self._num_anchors = 1
        
        # Configure model based on output shape
        if len(outputs) == 6:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
        elif len(outputs) == 9:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
            self.use_kps = True
        elif len(outputs) == 10:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
        elif len(outputs) == 15:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
            self.use_kps = True

    def _get_anchor_centers(self, height, width, stride):
        """Get anchor centers for a feature map.
        
        Args:
            height (int): Feature map height.
            width (int): Feature map width.
            stride (int): Feature stride.
            
        Returns:
            np.ndarray: Anchor centers.
        """
        key = (height, width, stride)
        
        if key in self.center_cache:
            return self.center_cache[key]
            
        # Calculate anchor centers
        anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
        anchor_centers = (anchor_centers * stride).reshape((-1, 2))
        
        if self._num_anchors > 1:
            anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1).reshape((-1, 2))
            
        # Manage cache size
        if len(self.center_cache) >= self.max_cache_size:
            # Remove oldest entry (first key)
            self.center_cache.pop(next(iter(self.center_cache)))
            
        self.center_cache[key] = anchor_centers
        return anchor_centers

    def _process_feature_level(self, scores, boxes, points, anchor_centers, stride, thresh):
        """Process detections from a single feature level.
        
        Args:
            scores (np.ndarray): Detection confidence scores.
            boxes (np.ndarray): Bounding box distances.
            points (np.ndarray): Keypoints if available.
            anchor_centers (np.ndarray): Anchor centers.
            stride (int): Feature stride.
            thresh (float): Confidence threshold.
            
        Returns:
            tuple: pos_scores, pos_bboxes, pos_kpss (if use_kps)
        """
        # Find positive detections
        pos_index = np.where(scores >= thresh)[0]
        if len(pos_index) == 0:
            return None, None, None
            
        # Convert distances to bboxes
        bboxes = distance2bbox(anchor_centers, boxes)
        
        # Get positive items
        pos_scores = scores[pos_index]
        pos_bboxes = bboxes[pos_index]
        
        # Handle keypoints if used
        pos_kpss = None
        if self.use_kps and points is not None:
            kpss = distance2kps(anchor_centers, points)
            kpss = kpss.reshape((kpss.shape[0], -1, 2))
            pos_kpss = kpss[pos_index]
            
        return pos_scores, pos_bboxes, pos_kpss

    def forward(self, img, thresh):
        """Perform forward pass for face detection.
        
        Args:
            img (np.ndarray): Input image.
            thresh (float): Confidence threshold.
            
        Returns:
            tuple: Lists of scores, bounding boxes, and keypoints.
        """
        scores_list = []
        bboxes_list = []
        kps_list = []
        
        try:
            input_size = tuple(img.shape[0:2][::-1])
            blob = cv2.dnn.blobFromImage(img, self.scale, input_size,
                                         self.mean_vals, swapRB=True)
            
            net_outs = self.session.run(self.output_names, {self.input_name: blob})
            
            input_height = blob.shape[2]
            input_width = blob.shape[3]
            
            # Process each feature stride level
            for idx, stride in enumerate(self._feat_stride_fpn):
                if self.batched:
                    scores = net_outs[idx][0]
                    boxes = net_outs[idx + self.fmc][0] * stride
                    points = net_outs[idx + self.fmc * 2][0] * stride if self.use_kps else None
                else:
                    scores = net_outs[idx]
                    boxes = net_outs[idx + self.fmc] * stride
                    points = net_outs[idx + self.fmc * 2] * stride if self.use_kps else None
                
                # Get feature map dimensions
                height = input_height // stride
                width = input_width // stride
                
                # Get anchor centers
                anchor_centers = self._get_anchor_centers(height, width, stride)
                
                # Process detections at this feature level
                pos_scores, pos_bboxes, pos_kpss = self._process_feature_level(
                    scores, boxes, points, anchor_centers, stride, thresh)
                
                # Add valid detections to results
                if pos_scores is not None:
                    scores_list.append(pos_scores)
                    bboxes_list.append(pos_bboxes)
                    if pos_kpss is not None:
                        kps_list.append(pos_kpss)
            
            return scores_list, bboxes_list, kps_list
            
        except Exception:
            # Silent error handling to prevent messages during runtime
            return [], [], []

    def detect(self, image, thresh=0.5, input_size=None, max_num=0, metric='default'):
        """Detect faces in an image.
        
        Args:
            image (np.ndarray): Input image.
            thresh (float): Confidence threshold.
            input_size (tuple): Input size for model.
            max_num (int): Maximum number of faces to return.
            metric (str): Selection metric ('default' or 'max').
            
        Returns:
            tuple: Detected bounding boxes and keypoints.
        """
        if image is None or image.size == 0:
            return np.array([]), None
            
        if input_size is None and self.input_size is None:
            raise ValueError("input_size or self.input_size must be provided.")
            
        input_size = self.input_size if input_size is None else input_size
        
        try:
            # Calculate image resize parameters
            im_ratio = float(image.shape[0]) / image.shape[1]
            model_ratio = float(input_size[1]) / input_size[0]
            
            if im_ratio > model_ratio:
                new_height = input_size[1]
                new_width = int(new_height / im_ratio)
            else:
                new_width = input_size[0]
                new_height = int(new_width * im_ratio)
                
            det_scale = float(new_height) / image.shape[0]
            resized_img = cv2.resize(image, (new_width, new_height))
            
            # Prepare padded image for detection
            det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
            det_img[:new_height, :new_width, :] = resized_img
            
            # Perform forward pass
            scores_list, bboxes_list, kps_list = self.forward(det_img, thresh)
            
            # Handle empty results
            if not scores_list:
                return np.array([]), None
                
            # Process detection results
            scores = np.vstack(scores_list)
            scores_ravel = scores.ravel()
            order = scores_ravel.argsort()[::-1]
            
            # Scale bboxes back to original image size
            bboxes = np.vstack(bboxes_list) / det_scale
            
            # Handle keypoints if available
            if self.use_kps and kps_list:
                kpss = np.vstack(kps_list) / det_scale
            else:
                kpss = None
                
            # Prepare detections for NMS
            pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
            pre_det = pre_det[order, :]
            
            # Apply NMS
            keep = self.nms(pre_det)
            det = pre_det[keep, :]
            
            # Handle keypoints after NMS
            if kpss is not None:
                kpss = kpss[order, :, :]
                kpss = kpss[keep, :, :]
                
            # Limit number of detections if requested
            if 0 < max_num < det.shape[0]:
                area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                img_center = image.shape[0] // 2, image.shape[1] // 2
                
                offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1],
                                    (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                
                # Select top detections by area or distance from center
                values = area if metric == 'max' else area - offset_dist_squared * 2.0
                bindex = np.argsort(values)[::-1][:max_num]
                
                det = det[bindex, :]
                if kpss is not None:
                    kpss = kpss[bindex, :]
                    
            return det, kpss
            
        except Exception:
            # Silent error handling to prevent messages during runtime
            return np.array([]), None

    def nms(self, dets):
        """Apply Non-Maximum Suppression to filter detections.
        
        Args:
            dets (np.ndarray): Array of detections [x1, y1, x2, y2, score].
            
        Returns:
            list: Indices of kept detections.
        """
        if dets.shape[0] == 0:
            return []
            
        thresh = self.nms_thresh
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
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
            index = np.where(ovr <= thresh)[0]
            order = order[index + 1]
            
        return keep