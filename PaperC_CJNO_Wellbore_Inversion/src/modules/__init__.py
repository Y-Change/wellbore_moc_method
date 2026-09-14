# -*- coding: utf-8 -*-
"""
PaperC_CJNO_Wellbore_Inversion.src.modules

TG-CJ-DeepONet 核心模块库:
1. time_gating: 三大多模态到时窗提取算法 (relative_window, gaussian_gating, ceps_patch);
2. acoustic_transformer: 声学时延偏置簇间解耦 Transformer 编码器;
3. dual_track_heads: 离散精准头与连续 Trunk 场解码双轨协同头。
"""
from __future__ import annotations

from .time_gating import (
    RelativeWindowGating,
    GaussianGating,
    CepstrumPatchGating,
    TimeGatingModule,
)
from .acoustic_transformer import AcousticBiasedTransformer
from .dual_track_heads import (
    DiscretePreciseHead,
    ContinuousTrunkHead,
    DualTrackHead,
)
from .embedding_modules import (
    extract_raw_wave_packet,
    PhysicsQueryEmbedding,
    FourierFeatureEmbedding,
    SincNetAcousticFilterbank,
    BaselineResampleEmbedding,
)

__all__ = [
    "RelativeWindowGating",
    "GaussianGating",
    "CepstrumPatchGating",
    "TimeGatingModule",
    "AcousticBiasedTransformer",
    "DiscretePreciseHead",
    "ContinuousTrunkHead",
    "DualTrackHead",
    "extract_raw_wave_packet",
    "PhysicsQueryEmbedding",
    "FourierFeatureEmbedding",
    "SincNetAcousticFilterbank",
    "BaselineResampleEmbedding",
]
