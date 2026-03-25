# logger.py (修复版)
import logging
import os
import sys

from config import BASE_DIR

# 获取 logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 防止重复添加 handler
if not logger.handlers:
    # 格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 1. 文件输出（加上 encoding='utf-8'）
    log_file_path = os.path.join(BASE_DIR, "app.log")
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 2. 控制台输出（加上 encoding='utf-8'）
    # 注意：Windows 控制台如果不支持 utf-8，可能需要改系统设置，但文件里肯定能存
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)