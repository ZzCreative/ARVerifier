"""
关键层筛选算法
设定多维度鲁棒性影响权重评估标准
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class LayerImportanceMetrics:
    """层重要性指标"""
    layer_idx: int
    sensitivity_coefficient: float      # 敏感性系数
    activation_contribution: float      # 激活贡献度
    error_propagation_probability: float  # 错误传播概率
    neuron_importance_mean: float       # 神经元平均重要性
    neuron_importance_std: float        # 神经元重要性标准差
    unstable_neuron_ratio: float        # 不稳定神经元占比
    combined_score: float               # 综合得分


class CriticalLayerSelector:
    """
    关键层选择器
    
    基于多维度评估指标筛选出对鲁棒性验证最关键的网络层
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device = torch.device('cpu'),
        sc_weight: float = 0.30,
        ac_weight: float = 0.25,
        epp_weight: float = 0.20,
        ni_weight: float = 0.15,
        ur_weight: float = 0.10,
        verbose: bool = True
    ):
        """
        Args:
            model: PyTorch神经网络
            device: 计算设备
            sc_weight: 敏感性系数权重
            ac_weight: 激活贡献度权重
            epp_weight: 错误传播概率权重
            ni_weight: 神经元重要性权重
            ur_weight: 不稳定神经元占比权重
            verbose: 是否打印详细信息
        """
        self.model = model
        self.device = device
        self.weights = {
            'sc': sc_weight,
            'ac': ac_weight,
            'epp': epp_weight,
            'ni': ni_weight,
            'ur': ur_weight
        }
        self.verbose = verbose
        self.model.eval()
        self.model.to(device)
    
    def get_relu_layer_count(self) -> int:
        """获取ReLU层数量"""
        return sum(1 for m in self.model.modules() if isinstance(m, nn.ReLU))
    
    def extract_layer_weights(self) -> Dict[int, torch.Tensor]:
        """
        提取各层权重矩阵
        
        Returns:
            layer_weights: {层索引: 权重矩阵}
        """
        layer_weights = {}
        linear_idx = 0
        relu_idx = 0
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                if linear_idx > 0:  # 跳过第一层(输入层)之前的权重
                    # 前一个ReLU层到当前Linear层的权重
                    layer_weights[relu_idx - 1] = module.weight.detach()
                linear_idx += 1
        
        return layer_weights
    
    def compute_combined_scores(
        self,
        sensitivity_dict: Dict[int, float],
        contribution_dict: Dict[int, float],
        epp_dict: Dict[int, float],
        neuron_importance_dict: Optional[Dict[int, float]] = None,
        unstable_ratio_dict: Optional[Dict[int, float]] = None
    ) -> List[LayerImportanceMetrics]:
        """
        计算各层的综合得分
        
        Args:
            sensitivity_dict: {层索引: 敏感性系数}
            contribution_dict: {层索引: 激活贡献度}
            epp_dict: {层索引: 错误传播概率}
            neuron_importance_dict: {层索引: 平均神经元重要性}
            unstable_ratio_dict: {层索引: 不稳定神经元占比}
        
        Returns:
            metrics_list: 按综合得分降序排列的指标列表
        """
        all_layers = set(sensitivity_dict.keys()) | set(contribution_dict.keys())
        
        metrics_list = []
        for layer_idx in all_layers:
            sc = sensitivity_dict.get(layer_idx, 0.0)
            ac = contribution_dict.get(layer_idx, 0.0)
            epp = epp_dict.get(layer_idx, 0.0)
            ni = neuron_importance_dict.get(layer_idx, 0.0) if neuron_importance_dict else 0.0
            ur = unstable_ratio_dict.get(layer_idx, 0.0) if unstable_ratio_dict else 0.0
            
            combined = (
                self.weights['sc'] * sc +
                self.weights['ac'] * ac +
                self.weights['epp'] * epp +
                self.weights['ni'] * ni +
                self.weights['ur'] * ur
            )
            
            metrics_list.append(LayerImportanceMetrics(
                layer_idx=layer_idx,
                sensitivity_coefficient=sc,
                activation_contribution=ac,
                error_propagation_probability=epp,
                neuron_importance_mean=ni,
                neuron_importance_std=0.0,
                unstable_neuron_ratio=ur,
                combined_score=combined
            ))
        
        metrics_list.sort(key=lambda m: m.combined_score, reverse=True)
        return metrics_list
    
    def select_critical_layers(
        self,
        metrics_list: List[LayerImportanceMetrics],
        method: str = 'top_k',
        threshold: float = 0.3,
        min_layers: int = 1,
        max_layers: Optional[int] = None
    ) -> List[int]:
        """
        选择关键层
        
        Args:
            metrics_list: 层重要性指标列表(已排序)
            method: 选择方法
                'top_k': 选择前k%的层
                'threshold': 选择综合得分超过阈值的层
                'auto': 自动选择(基于肘部法则)
            threshold: 阈值或比例
            min_layers: 最少层数
            max_layers: 最大层数
        
        Returns:
            critical_layer_indices: 关键层索引列表
        """
        if max_layers is None:
            max_layers = len(metrics_list)
        
        if method == 'top_k':
            k = max(min_layers, int(len(metrics_list) * threshold))
            k = min(k, max_layers)
            selected = [m.layer_idx for m in metrics_list[:k]]
        
        elif method == 'threshold':
            selected = []
            for m in metrics_list:
                if m.combined_score >= threshold and len(selected) < max_layers:
                    selected.append(m.layer_idx)
            if len(selected) < min_layers:
                selected = [m.layer_idx for m in metrics_list[:min_layers]]
        
        elif method == 'auto':
            # 肘部法则: 找到得分下降最剧烈的点
            if len(metrics_list) <= min_layers:
                selected = [m.layer_idx for m in metrics_list]
            else:
                scores = [m.combined_score for m in metrics_list]
                diffs = [scores[i] - scores[i+1] for i in range(len(scores)-1)]
                
                # 找最大下降点
                elbow = np.argmax(diffs) + 1
                elbow = max(min_layers, min(elbow, max_layers))
                selected = [m.layer_idx for m in metrics_list[:elbow]]
        
        else:
            raise ValueError(f"未知的选择方法: {method}")
        
        if self.verbose:
            print(f"\n关键层选择结果 (方法={method}):")
            print(f"  总层数: {len(metrics_list)}")
            print(f"  选择层数: {len(selected)}")
            print(f"  选择比例: {len(selected)/len(metrics_list)*100:.1f}%")
            print(f"  关键层索引: {selected}")
        
        return selected
    
    def estimate_computation_reduction(
        self,
        total_layers: int,
        critical_layers: List[int],
        critical_cost_ratio: float = 1.0,
        non_critical_cost_ratio: float = 0.1
    ) -> Dict[str, float]:
        """
        估计计算量减少
        
        Args:
            total_layers: 总层数
            critical_layers: 关键层索引列表
            critical_cost_ratio: 关键层相对计算成本
            non_critical_cost_ratio: 非关键层相对计算成本
        
        Returns:
            reduction: 计算量减少指标
        """
        n_critical = len(critical_layers)
        n_non_critical = total_layers - n_critical
        
        full_cost = total_layers * critical_cost_ratio
        simplified_cost = (
            n_critical * critical_cost_ratio +
            n_non_critical * non_critical_cost_ratio
        )
        
        reduction = 1.0 - simplified_cost / full_cost
        speedup = full_cost / simplified_cost
        
        return {
            "full_cost": full_cost,
            "simplified_cost": simplified_cost,
            "reduction_ratio": reduction,
            "estimated_speedup": speedup,
            "critical_layer_ratio": n_critical / total_layers
        }
    
    def print_selection_summary(
        self,
        metrics_list: List[LayerImportanceMetrics],
        critical_layers: List[int]
    ):
        """打印选择摘要"""
        print("\n" + "="*60)
        print("关键层筛选摘要")
        print("="*60)
        
        print(f"\n{'层索引':<8} {'敏感性系数':<12} {'激活贡献度':<12} {'错误传播':<10} {'综合得分':<10} {'关键层':<8}")
        print("-"*60)
        
        for m in metrics_list:
            is_critical = "★" if m.layer_idx in critical_layers else ""
            print(f"{m.layer_idx:<8} {m.sensitivity_coefficient:<12.4f} "
                  f"{m.activation_contribution:<12.4f} {m.error_propagation_probability:<10.4f} "
                  f"{m.combined_score:<10.4f} {is_critical:<8}")
