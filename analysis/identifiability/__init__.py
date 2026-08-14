# -*- coding: utf-8 -*-
"""
analysis/identifiability — 反问题可辨识性分析（Fisher 信息矩阵 / Cramér-Rao 界）
与全波形 MLE 效率验证、高阶/阶数选择、失配迁移。

回答的问题与 analysis/resolvability 不同：
  - resolvability : 给定动态范围 DR，倒谱谱支撑宽度对应的深度分辨宽度（估计器相关，
                    且只描述某一特定变换的旁瓣宽度）。
  - identifiability: 给定观测噪声与传感器带宽，任何无偏估计器所能达到的位置方差下界
                    （估计器无关），并可显式把波速等讨厌参数边缘化掉。
  - efficiency    : 全波形 MLE 能否贴到该下界（run_efficiency / mle）。
  - order         : 缝数未知时的模型选择（run_order_selection）。
  - transfer      : 配对失配可迁移性（run_transfer_check）。
"""
