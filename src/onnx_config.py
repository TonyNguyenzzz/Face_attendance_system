import onnxruntime as ort
import logging

logger = logging.getLogger(__name__)
# Thiết lập mức độ logging cao hơn để loại bỏ thông báo không cần thiết
logger.setLevel(logging.WARNING)

def configure_onnx_providers():
    """
    Cấu hình ONNX Runtime để sử dụng CPU khi CUDA không khả dụng.
    Trả về danh sách providers phù hợp.
    """
    available_providers = ort.get_available_providers()
    logger.debug(f"ONNX Runtime providers khả dụng: {available_providers}")
    
    # Thử sử dụng CUDA nếu có, nếu không thì dùng CPU
    if 'CUDAExecutionProvider' in available_providers:
        try:
            # Thử tải CUDA provider
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            # Kiểm tra xem có thể tạo session với CUDA không
            logger.debug("Đang thử sử dụng CUDA Execution Provider...")
            return providers
        except Exception as e:
            logger.warning(f"Không thể sử dụng CUDA Execution Provider: {e}")
            logger.debug("Chuyển sang sử dụng CPU Execution Provider")
            return ['CPUExecutionProvider']
    else:
        logger.debug("CUDA Execution Provider không khả dụng, sử dụng CPU")
        return ['CPUExecutionProvider']