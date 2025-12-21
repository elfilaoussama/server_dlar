"""
Centralized logging configuration with file and console output.
Logs are stored per date with size-based rotation within each day.
"""
import logging
import sys
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional
from contextvars import ContextVar
from logging.handlers import TimedRotatingFileHandler, RotatingFileHandler

from api.core import config

# Context variable for request correlation
request_id_ctx: ContextVar[Optional[str]] = ContextVar("request_id", default=None)


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        request_id = request_id_ctx.get()
        if request_id:
            log_entry["request_id"] = request_id
        
        if hasattr(record, "extra_data"):
            log_entry["data"] = record.extra_data
        
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """Human-readable text formatter."""
    
    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        request_id = request_id_ctx.get()
        req_str = f"[{request_id}] " if request_id else ""
        
        msg = f"{timestamp} | {record.levelname:8} | {req_str}{record.name} | {record.getMessage()}"
        
        if record.exc_info:
            msg += f"\n{self.formatException(record.exc_info)}"
        
        return msg


class DailyRotatingFileHandler(TimedRotatingFileHandler):
    """
    Custom handler that creates daily log files with size-based rotation.
    
    Files are named: app_2024-12-19.log, app_2024-12-19.1.log, etc.
    New file is created each day or when size limit is reached.
    """
    
    def __init__(self, log_dir: str, base_name: str = "app", 
                 max_bytes: int = 10 * 1024 * 1024, backup_count: int = 30):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.base_name = base_name
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self._current_date = datetime.now().strftime("%Y-%m-%d")
        self._file_counter = 0
        
        # Get initial log file path
        log_path = self._get_log_path()
        
        super().__init__(
            filename=str(log_path),
            when='midnight',
            interval=1,
            backupCount=backup_count,
            encoding='utf-8'
        )
    
    def _get_log_path(self) -> Path:
        """Get log file path for current date with counter."""
        if self._file_counter == 0:
            filename = f"{self.base_name}_{self._current_date}.log"
        else:
            filename = f"{self.base_name}_{self._current_date}.{self._file_counter}.log"
        return self.log_dir / filename
    
    def shouldRollover(self, record) -> bool:
        """Check if rollover is needed (new day or size exceeded)."""
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        # New day - reset counter
        if current_date != self._current_date:
            self._current_date = current_date
            self._file_counter = 0
            return True
        
        # Check file size
        if self.stream:
            self.stream.seek(0, 2)  # Seek to end
            if self.stream.tell() + len(self.format(record)) >= self.max_bytes:
                return True
        
        return False
    
    def doRollover(self):
        """Perform rollover to new file."""
        if self.stream:
            self.stream.close()
            self.stream = None
        
        # Increment counter for size-based rotation within same day
        current_date = datetime.now().strftime("%Y-%m-%d")
        if current_date == self._current_date:
            self._file_counter += 1
        else:
            self._current_date = current_date
            self._file_counter = 0
        
        # Update to new file path
        self.baseFilename = str(self._get_log_path())
        
        # Open new file
        self.stream = self._open()
        
        # Clean up old log files
        self._cleanup_old_logs()
    
    def _cleanup_old_logs(self):
        """Remove log files older than backup_count days."""
        try:
            cutoff = datetime.now().timestamp() - (self.backup_count * 24 * 60 * 60)
            for log_file in self.log_dir.glob(f"{self.base_name}_*.log"):
                if log_file.stat().st_mtime < cutoff:
                    log_file.unlink()
        except Exception:
            pass  # Don't fail on cleanup errors


def setup_logging(name: str = "dla") -> logging.Logger:
    """
    Configure application logging with console and daily rotating file handlers.
    
    Log files are stored as: logs/app_YYYY-MM-DD.log
    Size-based rotation creates: logs/app_YYYY-MM-DD.1.log, etc.
    Falls back to console-only if file logging fails.
    """
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger
    
    level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(level)
    
    # Choose formatter
    if config.LOG_FORMAT == "json":
        formatter = JSONFormatter()
    else:
        formatter = TextFormatter()
    
    # Console handler (always enabled)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Daily rotating file handler (optional, with error handling)
    if config.LOG_FILE_ENABLED:
        try:
            log_dir = Path(config.LOG_FILE_PATH).parent
            log_dir.mkdir(parents=True, exist_ok=True)
            
            file_handler = DailyRotatingFileHandler(
                log_dir=str(log_dir),
                base_name="app",
                max_bytes=config.LOG_MAX_BYTES,
                backup_count=config.LOG_BACKUP_COUNT
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except (PermissionError, OSError) as e:
            # Fall back to console-only logging if file logging fails
            logger.warning(f"File logging disabled: {e}. Using console only.")
    
    logger.propagate = False
    return logger


def get_logger(name: str = "dla") -> logging.Logger:
    """Get or create a logger instance."""
    return logging.getLogger(name)


def generate_request_id() -> str:
    """Generate unique request correlation ID."""
    return str(uuid.uuid4())[:8]


def set_request_id(request_id: str) -> None:
    """Set request ID in context."""
    request_id_ctx.set(request_id)


def get_request_id() -> Optional[str]:
    """Get current request ID from context."""
    return request_id_ctx.get()


# Default logger instance
_default_logger: Optional[logging.Logger] = None


def init_logger() -> logging.Logger:
    """Initialize and return the default application logger."""
    global _default_logger
    if _default_logger is None:
        _default_logger = setup_logging("dla")
    return _default_logger


# Convenience functions
def debug(msg: str, **kwargs):
    init_logger().debug(msg, extra={"extra_data": kwargs} if kwargs else None)

def info(msg: str, **kwargs):
    init_logger().info(msg, extra={"extra_data": kwargs} if kwargs else None)

def warning(msg: str, **kwargs):
    init_logger().warning(msg, extra={"extra_data": kwargs} if kwargs else None)

def error(msg: str, **kwargs):
    init_logger().error(msg, extra={"extra_data": kwargs} if kwargs else None)

def exception(msg: str, **kwargs):
    init_logger().exception(msg, extra={"extra_data": kwargs} if kwargs else None)
