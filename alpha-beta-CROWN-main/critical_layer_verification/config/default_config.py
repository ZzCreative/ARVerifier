"""
默认配置文件
集中管理所有超参数和配置选项
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class CriticalLayerConfig:
    """关键层识别配置"""
    # === 敏感性分析 ===
    sensitivity_epsilon: float = 0.01        # 敏感性分析的扰动强度
    sensitivity_perturbation_levels: int = 5  # 扰动级别数
    
    # === 神经元分析 ===
    activation_batches: int = 10             # 激活统计的批次数
    always_on_threshold: float = 0.95        # 始终激活阈值
    always_off_threshold: float = 0.05       # 从未激活阈值
    
    # === 关键层选择 ===
    selection_method: str = "top_k"          # 选择方法: top_k / threshold / auto
    selection_threshold: float = 0.3         # 选择阈值/比例
    min_critical_layers: int = 1             # 最少关键层数
    max_critical_layers: Optional[int] = None  # 最大关键层数
    
    # === 评估权重 ===
    weight_sensitivity: float = 0.30         # 敏感性权重
    weight_activation: float = 0.25          # 激活贡献度权重
    weight_error_prop: float = 0.20          # 错误传播权重
    weight_neuron_importance: float = 0.15   # 神经元重要性权重
    weight_unstable: float = 0.10            # 不稳定神经元权重
    
    # === 验证参数 ===
    crown_use_alpha: bool = True             # 使用α-CROWN
    crown_use_beta: bool = False             # 使用β-CROWN(更精确但更慢)
    crown_alpha_iterations: int = 20         # α优化迭代次数
    
    # === 关键神经元识别 ===
    critical_neuron_ratio: float = 0.3       # 关键神经元占比
    min_critical_neurons: int = 5            # 每层最少关键神经元
    
    # === 推断策略 ===
    inference_strategy: str = "conservative"  # conservative / weighted / adaptive
    
    # === 内存优化 ===
    target_memory_reduction: float = 0.40    # 目标内存降低(40%)


def get_default_config() -> CriticalLayerConfig:
    """获取默认配置"""
    return CriticalLayerConfig()
