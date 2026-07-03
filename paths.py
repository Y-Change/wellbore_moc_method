# -*- coding: utf-8 -*-
"""自研 MOC 方法目录路径。"""
import os

METHOD_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(METHOD_DIR, "output")


def moc_output_dir() -> str:
    """返回 output 目录路径（不存在则创建）。"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR
