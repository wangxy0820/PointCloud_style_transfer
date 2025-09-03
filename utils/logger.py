import logging
import os
import sys
from datetime import datetime

class Logger:
    """
    A simple logger utility that writes to both console and a file.
    FIXED: Now accepts an 'experiment_name' to create organized log directories.
    """
    def __init__(self, name: str, log_dir: str, experiment_name: str, file_output: bool = True):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # Prevent adding duplicate handlers
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch_formatter = logging.Formatter('%(asctime)s [%(levelname)s] [%(name)s]: %(message)s')
        ch.setFormatter(ch_formatter)
        self.logger.addHandler(ch)

        # File handler (if enabled)
        if file_output:
            # Create an experiment-specific log directory
            exp_log_dir = os.path.join(log_dir, experiment_name)
            os.makedirs(exp_log_dir, exist_ok=True)
            
            # Create a log file with a timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = os.path.join(exp_log_dir, f'{timestamp}.log')
            
            fh = logging.FileHandler(log_file)
            fh_formatter = logging.Formatter('%(asctime)s [%(levelname)s] [%(name)s]: %(message)s')
            fh.setFormatter(fh_formatter)
            self.logger.addHandler(fh)

    def info(self, msg: str):
        self.logger.info(msg)

    def warning(self, msg: str):
        self.logger.warning(msg)

    def error(self, msg: str):
        self.logger.error(msg)