"""
基于关键层的简化验证方法理论框架核心模块
参考: α,β-CROWN (Xu et al., 2020, 2021) 及神经元抽象方法
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class CriticalLayerConfig:
    """关键层识别配置"""
    # 敏感性系数阈值
    sensitivity_threshold: float = 0.15
    # 激活贡献度阈值
    activation_contribution_threshold: float = 0.1
    # 错误传播概率阈值
    error_propagation_threshold: float = 0.05
    # 关键神经元占比阈值 (选择前k%的神经元)
    critical_neuron_ratio: float = 0.3
    # 完备性损失上限
    max_completeness_loss: float = 0.05
    # 加速比目标
    target_speedup: float = 10.0


class TheoryFramework:
    """
    理论框架类 - 实现关键层识别与简化的理论推导
    
    核心思想:
    1. 在ReLU网络中,并非所有层对鲁棒性验证同等重要
    2. "关键层"是对验证结果边界影响最大的层
    3. 通过在关键层上精确计算,非关键层上近似计算,实现加速
    4. 完备性损失可量化且可控
    """
    
    def __init__(self, config: Optional[CriticalLayerConfig] = None):
        self.config = config or CriticalLayerConfig()
    
    def compute_sensitivity_coefficient(
        self, 
        pre_bounds: torch.Tensor, 
        post_bounds: torch.Tensor, 
        base_bounds: torch.Tensor
    ) -> float:
        """
        计算敏感性系数 (Sensitivity Coefficient, SC)
        
        SC = ||Φ(x+δ) - Φ(x)|| / ||δ||
        
        其中Φ是验证函数,x是输入界,δ是层输出界的扰动
        
        Args:
            pre_bounds: 扰动前的层边界 [2, n_neurons]
            post_bounds: 扰动后的层边界 [2, n_neurons]  
            base_bounds: 基准输出边界 [2, n_outputs]
        
        Returns:
            SC: 敏感性系数
        """
        delta = torch.norm(post_bounds - pre_bounds)
        if delta < 1e-10:
            return 0.0
        output_change = torch.norm(base_bounds)
        return (output_change / delta).item()
    
    def compute_activation_contribution(
        self,
        layer_output: torch.Tensor,
        final_weights: torch.Tensor,
        baseline_output: torch.Tensor
    ) -> torch.Tensor:
        """
        计算激活贡献度 (Activation Contribution, AC)
        
        AC_i = |w_i * ReLU(z_i)| / Σ|w_j * ReLU(z_j)|
        
        度量每个神经元对最终决策的贡献程度
        
        Args:
            layer_output: 层输出 [batch, n_neurons]
            final_weights: 后续层权重 [n_neurons, n_outputs]
            baseline_output: 基准输出 [batch, n_outputs]
        
        Returns:
            AC: 每个神经元的激活贡献度 [n_neurons]
        """
        activated = torch.relu(layer_output)
        contributions = torch.abs(activated @ final_weights)
        total = contributions.sum(dim=-1, keepdim=True) + 1e-10
        return (contributions / total).mean(dim=0)
    
    def compute_error_propagation_probability(
        self,
        layer_bounds: torch.Tensor,
        weight_matrix: torch.Tensor,
        next_layer_bounds: torch.Tensor
    ) -> float:
        """
        计算错误传播概率 (Error Propagation Probability, EPP)
        
        EPP = P(||ε_{l+1}|| > τ | ||ε_l|| > τ)
        
        其中ε_l是第l层的界估计误差
        
        Args:
            layer_bounds: 当前层边界 [2, n_neurons]
            weight_matrix: 权重矩阵 [n_neurons, n_next]
            next_layer_bounds: 下一层边界 [2, n_next]
        
        Returns:
            EPP: 错误传播概率
        """
        # 计算层边界的宽度（不确定性）
        layer_width = layer_bounds[1] - layer_bounds[0]
        next_width = next_layer_bounds[1] - next_layer_bounds[0]
        
        # 通过权重矩阵估计误差传播
        weight_norm = torch.norm(weight_matrix, p=float('inf'))
        propagated_error = weight_norm * layer_width.mean()
        
        # 传播误差与下一层固有不确定性的比值
        ratio = propagated_error / (next_width.mean() + 1e-10)
        return torch.clamp(ratio, 0.0, 1.0).item()
    
    def compute_layer_importance_score(
        self,
        sc: float,
        ac: float,
        epp: float,
        weights: Tuple[float, float, float] = (0.4, 0.35, 0.25)
    ) -> float:
        """
        计算层重要性综合得分 (Layer Importance Score, LIS)
        
        LIS = w_SC * SC + w_AC * AC + w_EPP * EPP
        
        Args:
            sc: 敏感性系数
            ac: 激活贡献度  
            epp: 错误传播概率
            weights: 权重元组 (w_sc, w_ac, w_epp)
        
        Returns:
            LIS: 综合重要性得分
        """
        w_sc, w_ac, w_epp = weights
        return w_sc * sc + w_ac * ac + w_epp * epp
    
    def bound_completeness_loss(
        self,
        full_network_bounds: torch.Tensor,
        simplified_bounds: torch.Tensor
    ) -> Tuple[float, float, float]:
        """
        计算完备性损失
        
        L_complete = ||bounds_full - bounds_simplified|| / ||bounds_full||
        
        Args:
            full_network_bounds: 全网络验证的输出边界 [2, n_outputs]
            simplified_bounds: 简化验证的输出边界 [2, n_outputs]
        
        Returns:
            (max_loss, mean_loss, std_loss): 最大、平均和标准差完备性损失
        """
        diff = torch.abs(full_network_bounds - simplified_bounds)
        norm_full = torch.norm(full_network_bounds) + 1e-10
        
        max_loss = (diff.max() / norm_full).item()
        mean_loss = (diff.mean() / norm_full).item()
        std_loss = (diff.std() / norm_full).item()
        
        return max_loss, mean_loss, std_loss
    
    def estimate_speedup(
        self,
        total_layers: int,
        critical_layers: int,
        critical_layer_cost: float,
        non_critical_layer_cost: float,
        overhead: float = 0.1
    ) -> float:
        """
        估计验证加速比
        
        Speedup = T_full / T_simplified
        
        其中 T_full = N * C_full
        T_simplified = N_critical * C_critical + N_non_critical * C_non_critical + overhead
        
        Args:
            total_layers: 总层数
            critical_layers: 关键层数
            critical_layer_cost: 关键层计算成本
            non_critical_layer_cost: 非关键层计算成本
            overhead: 额外开销比例
        
        Returns:
            estimated_speedup: 估计加速比
        """
        # 假设全网络验证每层成本相同
        full_cost = total_layers * critical_layer_cost
        
        # 简化网络成本
        simplified_cost = (
            critical_layers * critical_layer_cost +
            (total_layers - critical_layers) * non_critical_layer_cost
        ) * (1 + overhead)
        
        speedup = full_cost / (simplified_cost + 1e-10)
        return speedup
    
    def verify_theoretical_bounds(self, num_layers: int, 
                                   critical_ratio: float,
                                   computation_reduction: float) -> Dict[str, float]:
        """
        验证理论边界
        
        证明: 当关键层占比为r,计算量减少为α时,
        完备性损失上限为 O(r * (1-α)), 加速比为 O(1/(r*α + (1-r)))
        
        Args:
            num_layers: 网络总层数
            critical_ratio: 关键层占比
            computation_reduction: 非关键层计算量减少比例
        
        Returns:
            bounds: 理论边界字典
        """
        k = int(num_layers * critical_ratio)
        
        # 加速比上界 (假设非关键层计算可忽略)
        max_speedup = num_layers / (k + 1e-10)
        
        # 完备性损失上界
        completeness_loss_upper = (1 - critical_ratio) * (1 - computation_reduction)
        
        # 信息保留率下界
        info_preservation_lower = 1 - completeness_loss_upper
        
        return {
            "num_layers": num_layers,
            "critical_layers": k,
            "critical_ratio": critical_ratio,
            "max_speedup": max_speedup,
            "completeness_loss_upper_bound": completeness_loss_upper,
            "info_preservation_lower_bound": info_preservation_lower
        }


class CompletenessEvaluator:
    """
    完备性评估器 - 量化评估验证完备性
    
    定义完备性损失:
    L_complete = 1 - (简化验证正确率 / 全网络验证正确率)
    """
    
    def __init__(self):
        self.history = []
    
    def evaluate_completeness(
        self,
        full_result: bool,
        simplified_result: bool,
        full_bounds: torch.Tensor,
        simplified_bounds: torch.Tensor
    ) -> float:
        """
        评估单次验证的完备性
        
        Args:
            full_result: 全网络验证结果 (True=安全)
            simplified_result: 简化验证结果
            full_bounds: 全网络输出界
            simplified_bounds: 简化验证输出界
        
        Returns:
            completeness: 完备性得分 (0-1)
        """
        if full_result == simplified_result:
            # 结果一致时,计算边界差异导致的完备性损失
            bound_diff = torch.norm(full_bounds - simplified_bounds)
            bound_norm = torch.norm(full_bounds) + 1e-10
            loss = min(1.0, (bound_diff / bound_norm).item())
            completeness = 1.0 - loss
        else:
            # 结果不一致时,完备性较低
            completeness = 0.0
        
        self.history.append(completeness)
        return completeness
    
    def get_average_completeness(self) -> float:
        """获取平均完备性"""
        if not self.history:
            return 1.0
        return np.mean(self.history)
    
    def get_completeness_statistics(self) -> Dict[str, float]:
        """获取完备性统计信息"""
        if not self.history:
            return {"mean": 1.0, "std": 0.0, "min": 1.0, "max": 1.0}
        return {
            "mean": float(np.mean(self.history)),
            "std": float(np.std(self.history)),
            "min": float(np.min(self.history)),
            "max": float(np.max(self.history))
        }


# 预定义关键性能指标(KPI)计算方法
class KPICalculator:
    """关键性能指标计算器"""
    
    @staticmethod
    def compute_speedup_ratio(full_time: float, simplified_time: float) -> float:
        """计算加速比"""
        return full_time / (simplified_time + 1e-10)
    
    @staticmethod
    def compute_memory_reduction(full_memory: float, simplified_memory: float) -> float:
        """计算内存降低比例"""
        return 1.0 - simplified_memory / (full_memory + 1e-10)
    
    @staticmethod
    def compute_identification_accuracy(
        true_critical: List[int],
        identified_critical: List[int]
    ) -> Dict[str, float]:
        """
        计算关键层识别准确率
        
        Args:
            true_critical: 真实关键层索引列表
            identified_critical: 识别出的关键层索引列表
        
        Returns:
            包含 precision, recall, f1_score 的字典
        """
        true_set = set(true_critical)
        identified_set = set(identified_critical)
        
        tp = len(true_set & identified_set)
        fp = len(identified_set - true_set)
        fn = len(true_set - identified_set)
        
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy": tp / (len(true_set) + 1e-10)
        }
    
    @staticmethod
    def compute_statistical_significance(
        results_full: List[float],
        results_simplified: List[float]
    ) -> Dict[str, float]:
        """
        计算统计显著性
        
        Args:
            results_full: 全网络验证结果列表
            results_simplified: 简化验证结果列表
        
        Returns:
            统计显著性指标
        """
        import scipy.stats as stats
        
        # 配对t检验
        t_stat, p_value = stats.ttest_rel(results_full, results_simplified)
        
        # Cohen's d效应量
        diff = np.array(results_full) - np.array(results_simplified)
        cohens_d = np.mean(diff) / (np.std(diff) + 1e-10)
        
        return {
            "t_statistic": t_stat,
            "p_value": p_value,
            "cohens_d": cohens_d,
            "significant": p_value < 0.05
        }
