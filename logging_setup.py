"""
Simple logging setup utility to centralize formatter/handlers.

Environment variables (optional):
- LOG_LEVEL: DEBUG/INFO/WARNING/ERROR/CRITICAL (default: INFO)
- LOG_TO_FILE: 1 to enable file handler (default: 0)
- LOG_FILE: path to log file (default: output_data/app.log)
"""
from __future__ import annotations

import logging
import os
from typing import Optional


_LEVEL_MAP = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL,
}


def _coerce_level(level: Optional[str | int]) -> int:
    if isinstance(level, int):
        return int(level)
    if isinstance(level, str):
        return _LEVEL_MAP.get(level.strip().upper(), logging.INFO)
    env = os.getenv('LOG_LEVEL', 'INFO').strip().upper()
    return _LEVEL_MAP.get(env, logging.INFO)


def setup_logging(level: Optional[str | int] = None, to_file: Optional[bool] = None, file_path: Optional[str] = None) -> logging.Logger:
    lvl = _coerce_level(level)
    if to_file is None:
        to_file = os.getenv('LOG_TO_FILE', '0') in ('1', 'true', 'True')
    if file_path is None:
        file_path = os.getenv('LOG_FILE', os.path.join('output_data', 'app.log'))

    logger = logging.getLogger()
    logger.setLevel(lvl)
    # Clear existing handlers to avoid duplicates
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)

    fmt = logging.Formatter('[%(asctime)s] %(levelname)s %(name)s: %(message)s')
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    sh.setLevel(lvl)
    logger.addHandler(sh)

    if to_file:
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        except Exception:
            pass
        fh = logging.FileHandler(file_path, encoding='utf-8')
        fh.setFormatter(fmt)
        fh.setLevel(lvl)
        logger.addHandler(fh)

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    return logging.getLogger(name)
