# -*- coding: utf-8 -*-
"""
moc_simulate.stratified_bench (向后兼容透明门面 / Backward Compatibility Facade)

透明转发至 moc_simulate.v1.stratified_bench，确保历史脚本与已有测试 100% 兼容。
"""
from __future__ import annotations

import runpy
from moc_simulate.v1.stratified_bench import *

if __name__ == "__main__":
    runpy.run_module("moc_simulate.v1.stratified_bench", run_name="__main__")
