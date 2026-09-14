# -*- coding: utf-8 -*-
"""
moc_simulate.run_lhs_batch_simulate (向后兼容透明门面 / Backward Compatibility Facade)

透明转发至 moc_simulate.v1.run_lhs_batch_simulate，确保历史脚本与已有测试 100% 兼容。
"""
from __future__ import annotations

import runpy
from moc_simulate.v1.run_lhs_batch_simulate import *
from moc_simulate.v1.run_lhs_batch_simulate import (
    generate_lhs_params,
    _worker_simulate,
)

if __name__ == "__main__":
    runpy.run_module("moc_simulate.v1.run_lhs_batch_simulate", run_name="__main__")
