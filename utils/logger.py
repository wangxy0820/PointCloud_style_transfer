"""
日志管理器
"""

import logging
import os
from datetime import datetime
from typing import Optional


class Logger:
    """统一的日志管理器"""
    
    def __init__(self, name: str, log_dir: str = 'logs', 
                 log_level: str = 'INFO',
                 console_output: bool = True,
                 file_output: bool = True):
        """
        Args:
            name: 日志器名称
            log_dir: 日志保存目录
            log_level: 日志级别
            console_output: 是否输出到控制台
            file_output: 是否输出到文件
        """
        self.name = name
        self.log_dir = log_dir
        
        # 创建logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # 格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 控制台处理器
        if console_output:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # 文件处理器
        if file_output:
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = os.path.join(log_dir, f'{name}_{timestamp}.log')
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def info(self, message: str):
        self.logger.info(message)
    
    def debug(self, message: str):
        self.logger.debug(message)
    
    def warning(self, message: str):
        self.logger.warning(message)
    
    def error(self, message: str):
        self.logger.error(message)
    
    def critical(self, message: str):
        self.logger.critical(message)
