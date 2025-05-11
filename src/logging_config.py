import logging
import sys
from typing import Dict, Any, Optional

def configure_logging(config: Dict[str, Any]) -> None:
    """
    Cấu hình hệ thống logging dựa trên các tùy chọn trong config
    
    Args:
        config: Dictionary chứa cấu hình, bao gồm logging_level và disable_console_logging
    """
    # Lấy mức độ log từ config, mặc định là WARNING nếu không có
    log_level_str = config.get("logging_level", "WARNING").upper()
    log_level = getattr(logging, log_level_str, logging.WARNING)
    
    # Xóa tất cả các handler hiện có
    for handler in logging.root.handlers[:]:  
        logging.root.removeHandler(handler)
    
    # Cấu hình logging cơ bản
    logging.basicConfig(level=log_level, format='%(levelname)s:%(name)s:%(message)s')
    
    # Tắt log ra console nếu được yêu cầu
    if config.get("disable_console_logging", False):
        logging.root.handlers = []
    
    # Cấu hình cho các logger cụ thể
    loggers = ["GUI", "FaceEmbedding", "AttendanceSystem", "DatabaseHandler"]
    for logger_name in loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(log_level)
        
    # Thông báo cấu hình đã được áp dụng (chỉ hiển thị nếu không tắt console logging)
    if not config.get("disable_console_logging", False):
        logging.info(f"Đã cấu hình logging với mức độ: {log_level_str}")