"""
关键层简化验证方法 - 理论框架
基于 α,β-CROWN 的 ReLU 神经网络鲁棒性验证简化方法
"""

from .theory_core import (
    TheoryFramework,
    CriticalLayerConfig,
    CompletenessEvaluator,
    KPICalculator
)
from .metrics import (
    VerificationKPI,
    ExperimentResult,
    ExperimentSummary
)

__all__ = [
    'TheoryFramework',
    'CriticalLayerConfig',
    'CompletenessEvaluator',
    'KPICalculator',
    'VerificationKPI',
    'ExperimentResult',
    'ExperimentSummary',
]
