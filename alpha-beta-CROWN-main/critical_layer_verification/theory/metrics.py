"""
量化评估模型与关键性能指标定义
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np


@dataclass
class VerificationKPI:
    """验证KPI定义"""
    # 验证速度指标
    avg_speedup: float = 0.0          # 平均加速比
    max_speedup: float = 0.0          # 最大加速比
    min_speedup: float = float('inf') # 最小加速比
    std_speedup: float = 0.0          # 加速比标准差
    
    # 完备性指标
    completeness_loss: float = 0.0    # 平均完备性损失
    max_completeness_loss: float = 0.0 # 最大完备性损失
    
    # 准确率指标
    verification_accuracy: float = 0.0 # 验证准确率
    false_positive_rate: float = 0.0  # 假阳性率
    false_negative_rate: float = 0.0  # 假阴性率
    
    # 识别准确率指标
    identification_accuracy: float = 0.0  # 关键层识别准确率
    identification_precision: float = 0.0 # 识别精确率
    identification_recall: float = 0.0    # 识别召回率
    
    # 内存效率指标
    memory_reduction: float = 0.0     # 内存降低比例
    
    # 统计显著性
    p_value: float = 1.0
    is_statistically_significant: bool = False


@dataclass
class ExperimentResult:
    """单次实验结果"""
    model_name: str
    epsilon: float
    sample_index: int
    true_label: int
    full_verification_time: float
    simplified_verification_time: float
    full_result: str  # 'safe', 'unsafe', 'unknown'
    simplified_result: str
    full_memory_mb: float
    simplified_memory_mb: float
    identified_critical_layers: List[int]
    total_layers: int
    completeness_score: float


@dataclass
class ExperimentSummary:
    """实验总结"""
    model_name: str
    epsilon: float
    num_samples: int
    kpi: VerificationKPI = field(default_factory=VerificationKPI)
    results: List[ExperimentResult] = field(default_factory=list)
    
    def compute_kpi(self) -> VerificationKPI:
        """计算所有KPI"""
        if not self.results:
            return self.kpi
        
        # 验证时间
        full_times = [r.full_verification_time for r in self.results]
        simplified_times = [r.simplified_verification_time for r in self.results]
        
        speedups = [f / (s + 1e-10) for f, s in zip(full_times, simplified_times)]
        
        self.kpi.avg_speedup = float(np.mean(speedups))
        self.kpi.max_speedup = float(np.max(speedups))
        self.kpi.min_speedup = float(np.min(speedups))
        self.kpi.std_speedup = float(np.std(speedups))
        
        # 完备性损失
        completeness_losses = [1.0 - r.completeness_score for r in self.results]
        self.kpi.completeness_loss = float(np.mean(completeness_losses))
        self.kpi.max_completeness_loss = float(np.max(completeness_losses))
        
        # 验证准确率
        correct = sum(1 for r in self.results if r.full_result == r.simplified_result)
        self.kpi.verification_accuracy = correct / len(self.results)
        
        # 假阳性/假阴性率
        fps = sum(1 for r in self.results 
                  if r.full_result != 'safe' and r.simplified_result == 'safe')
        fns = sum(1 for r in self.results 
                  if r.full_result == 'safe' and r.simplified_result != 'safe')
        self.kpi.false_positive_rate = fps / len(self.results)
        self.kpi.false_negative_rate = fns / len(self.results)
        
        # 识别准确率
        correct_ids = sum(
            1 for r in self.results
            if len(set(r.identified_critical_layers) & 
                   set(range(r.total_layers))) >= r.total_layers * 0.3
        )
        self.kpi.identification_accuracy = correct_ids / len(self.results)
        
        # 内存降低
        memories = [1.0 - r.simplified_memory_mb / (r.full_memory_mb + 1e-10) 
                    for r in self.results]
        self.kpi.memory_reduction = float(np.mean(memories))
        
        return self.kpi
