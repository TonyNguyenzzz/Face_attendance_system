"""
ONNX Runtime configuration module for Face Recognition Attendance System.
Provides centralized ONNX provider configuration for CPU/GPU acceleration.
"""
import onnxruntime as ort
import logging

logger = logging.getLogger(__name__)


def configure_onnx_providers(use_gpu: bool = True) -> list:
    """
    Configure ONNX Runtime providers based on hardware availability.
    
    Args:
        use_gpu: Whether to attempt GPU acceleration (default: True)
        
    Returns:
        List of provider names in priority order
    """
    available_providers = ort.get_available_providers()
    logger.debug(f"Available ONNX Runtime providers: {available_providers}")
    
    if not use_gpu:
        logger.info("GPU acceleration disabled, using CPU only")
        return ['CPUExecutionProvider']
    
    # Try CUDA first if available
    if 'CUDAExecutionProvider' in available_providers:
        try:
            logger.info("Using CUDA Execution Provider for GPU acceleration")
            return ['CUDAExecutionProvider', 'CPUExecutionProvider']
        except Exception as e:
            logger.warning(f"CUDA Execution Provider failed: {e}, falling back to CPU")
    
    # Fall back to CPU
    logger.info("Using CPU Execution Provider")
    return ['CPUExecutionProvider']


def get_available_providers() -> list:
    """Get list of all available ONNX Runtime providers."""
    return ort.get_available_providers()


def is_cuda_available() -> bool:
    """Check if CUDA execution provider is available."""
    return 'CUDAExecutionProvider' in ort.get_available_providers()
