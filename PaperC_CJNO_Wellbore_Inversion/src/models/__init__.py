# -*- coding: utf-8 -*-
"""
PaperC_CJNO_Wellbore_Inversion.src.models
"""
from PaperC_CJNO_Wellbore_Inversion.src.models.resnet1d import ResNet1D
from PaperC_CJNO_Wellbore_Inversion.src.models.fno1d import FNO1D
from PaperC_CJNO_Wellbore_Inversion.src.models.deeponet import VanillaDeepONet
from PaperC_CJNO_Wellbore_Inversion.src.models.cj_cep_deeponet import CJCepDeepONet
from PaperC_CJNO_Wellbore_Inversion.src.models.tg_cj_deeponet import TGCJDeepONet
from PaperC_CJNO_Wellbore_Inversion.src.models.tg_dis_deeponet import TGDISDeepONet
from PaperC_CJNO_Wellbore_Inversion.src.models.tg_dis_embedding_variants import TGDISEmbeddingModel
from PaperC_CJNO_Wellbore_Inversion.src.models.cj_alphanet import CJAlphaNet
from PaperC_CJNO_Wellbore_Inversion.src.models.alpha_baselines import (
    AlphaResNet1D,
    AlphaFNO1D,
    AlphaCJCepNet,
    UniformFloorBaseline,
    TrainMeanBaseline,
)

__all__ = [
    "ResNet1D",
    "FNO1D",
    "VanillaDeepONet",
    "CJCepDeepONet",
    "TGCJDeepONet",
    "TGDISDeepONet",
    "TGDISEmbeddingModel",
    "CJAlphaNet",
    "AlphaResNet1D",
    "AlphaFNO1D",
    "AlphaCJCepNet",
    "UniformFloorBaseline",
    "TrainMeanBaseline",
]
