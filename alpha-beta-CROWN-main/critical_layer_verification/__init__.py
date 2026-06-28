"""
基于关键层的简化验证方法 (Critical-Layer Based Simplified Verification)

基于α,β-CROWN框架,通过对ReLU神经网络中的关键层进行优先验证,
大幅提升鲁棒性验证效率,同时严格控制完备性损失。

核心模块:
- theory: 理论框架,关键指标定义
- sensitivity: 敏感性分析与神经元活跃度统计
- identification: 关键层与关键神经元识别
- verification: 简化验证引擎
- integration: 结果整合与推断
- experiment: 对比实验与数据分析
- utils: 工具函数
- config: 配置管理
"""

from .config import get_default_config, CriticalLayerConfig
from .theory.theory_core import (
    TheoryFramework, CompletenessEvaluator, KPICalculator,
    CriticalLayerConfig as TheoryConfig
)
from .sensitivity.sensitivity_analyzer import SensitivityAnalyzer
from .sensitivity.neuron_activity_analyzer import NeuronActivityAnalyzer
from .identification.layer_selector import CriticalLayerSelector
from .identification.neuron_identifier import CriticalNeuronIdentifier, CriticalNeuronStore
from .verification.critical_layer_verifier import CriticalLayerVerifier, VerificationMode
from .integration.result_integrator import ResultIntegrator, InferenceStrategy, CompletenessMonitor
from .experiment.comparison_experiment import ComparisonExperiment
from .experiment.data_analyzer import DataAnalyzer
from .experiment.parameter_optimizer import ParameterOptimizer
from .utils.model_utils import ModelUtils
from .utils.visualization import VisualizationTools

__version__ = "1.0.0"
