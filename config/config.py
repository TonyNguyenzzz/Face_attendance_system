"""
Configuration module for Face Recognition Attendance System.
Centralized configuration management with type hints and validation.
"""
import os
from typing import Dict, Any, Tuple
from dataclasses import dataclass, field


@dataclass
class Config:
    """Strongly-typed configuration class for the attendance system."""
    
    # Directory paths
    model_dir: str = ""
    result_dir: str = ""
    
    # Model files
    scrfd_model: str = "scrfd_2.5g.onnx"
    face_model: str = "Facenet512"
    
    # Face detection parameters
    detector_confidence: float = 0.7
    
    # Face recognition parameters
    recognition_distance: float = 0.35
    recognition_confidence: float = 0.85
    distance_threshold: float = 0.35
    min_confidence_threshold: float = 0.6
    absolute_distance_threshold: float = 0.6
    min_confidence: float = 0.6
    
    # Image processing
    max_dim: int = 1080
    face_size: Tuple[int, int] = (160, 160)
    target_size: Tuple[int, int] = (160, 160)
    
    # Performance & caching
    cache_size: int = 1000
    max_workers: int = 4
    
    # Camera settings
    camera_id: int = 0
    camera_fps: int = 25
    
    # Hardware acceleration
    use_gpu: bool = True
    
    # Attendance settings
    attendance_cooldown: int = 300  # seconds
    late_threshold: int = 0  # minutes
    
    # Logging
    logging_level: str = "ERROR"
    disable_console_logging: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for backward compatibility."""
        return {
            "model_dir": self.model_dir,
            "result_dir": self.result_dir,
            "scrfd_model": self.scrfd_model,
            "face_model": self.face_model,
            "detector_confidence": self.detector_confidence,
            "recognition_distance": self.recognition_distance,
            "recognition_confidence": self.recognition_confidence,
            "distance_threshold": self.distance_threshold,
            "min_confidence_threshold": self.min_confidence_threshold,
            "absolute_distance_threshold": self.absolute_distance_threshold,
            "min_confidence": self.min_confidence,
            "max_dim": self.max_dim,
            "face_size": self.face_size,
            "target_size": self.target_size,
            "cache_size": self.cache_size,
            "max_workers": self.max_workers,
            "camera_id": self.camera_id,
            "camera_fps": self.camera_fps,
            "use_gpu": self.use_gpu,
            "attendance_cooldown": self.attendance_cooldown,
            "late_threshold": self.late_threshold,
            "logging_level": self.logging_level,
            "disable_console_logging": self.disable_console_logging,
        }


def ensure_dir(directory: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(directory, exist_ok=True)


def get_project_root() -> str:
    """Get the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def initialize_config() -> Config:
    """Initialize and return configuration with proper paths."""
    project_root = get_project_root()
    
    # Set up directories
    model_dir = os.path.join(project_root, "models")
    result_dir = os.path.join(project_root, "config", "results")
    
    ensure_dir(model_dir)
    ensure_dir(result_dir)
    
    return Config(
        model_dir=model_dir,
        result_dir=result_dir,
    )


# Initialize default configuration
DEFAULT_CONFIG = initialize_config().to_dict()
CONFIG = initialize_config()
