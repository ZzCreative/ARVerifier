"""
验证结果的整合与推断算法
实现从关键层验证结果到全网络鲁棒性结论的可靠推断
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum


class InferenceStrategy(Enum):
    """推断策略"""
    CONSERVATIVE = "conservative"       # 保守推断: 取最坏情况
    WEIGHTED = "weighted"               # 加权推断: 基于层重要性
    ADAPTIVE = "adaptive"               # 自适应推断: 动态调整


@dataclass
class IntegratedResult:
    """整合后的验证结果"""
    is_safe: bool                       # 最终安全性判定
    confidence: float                   # 判定置信度
    safety_margin: float                # 安全边际
    num_critical_layers: int            # 关键层数
    num_total_layers: int               # 总层数
    critical_layer_results: Dict[int, Dict[str, Any]]  # 各关键层的验证结果
    inference_strategy: InferenceStrategy  # 使用的推断策略
    explanation: str                   # 结果解释


class ResultIntegrator:
    """
    验证结果整合器
    
    将从关键层验证的结果整合为全网络鲁棒性结论
    支持保守、加权和自适应三种推断策略
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device = torch.device('cpu'),
        verbose: bool = False
    ):
        self.model = model
        self.device = device
        self.verbose = verbose
        self.model.eval()
        self.model.to(device)
    
    def infer_from_critical_layers(
        self,
        full_layer_bounds: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
        critical_layers: List[int],
        target_label: int,
        num_classes: int = 10,
        strategy: InferenceStrategy = InferenceStrategy.CONSERVATIVE,
        layer_importance: Optional[Dict[int, float]] = None
    ) -> IntegratedResult:
        """
        从关键层验证结果推断全网络鲁棒性
        
        Args:
            full_layer_bounds: 所有层的边界 {层索引: (下界, 上界)}
            critical_layers: 关键层索引列表
            target_label: 目标标签
            num_classes: 类别数
            strategy: 推断策略
            layer_importance: 层重要性权重字典
        
        Returns:
            result: 整合后的验证结果
        """
        if strategy == InferenceStrategy.CONSERVATIVE:
            return self._conservative_inference(
                full_layer_bounds, critical_layers, target_label, num_classes
            )
        elif strategy == InferenceStrategy.WEIGHTED:
            return self._weighted_inference(
                full_layer_bounds, critical_layers, target_label, 
                num_classes, layer_importance
            )
        elif strategy == InferenceStrategy.ADAPTIVE:
            return self._adaptive_inference(
                full_layer_bounds, critical_layers, target_label, num_classes
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _conservative_inference(
        self,
        full_layer_bounds: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
        critical_layers: List[int],
        target_label: int,
        num_classes: int
    ) -> IntegratedResult:
        """
        保守推断策略
        
        取所有关键层中最差的安全边距作为最终结果
        保证可靠性但可能过于保守
        """
        safety_margins = {}
        critical_results = {}
        
        for layer_idx in critical_layers:
            if layer_idx not in full_layer_bounds:
                continue
            
            lower, upper = full_layer_bounds[layer_idx]
            
            # 计算该层的安全边距
            # 对于输出层: target_logit - max_other_logit
            if lower.dim() > 1 and lower.shape[-1] == num_classes:
                target_lower = lower[0, target_label]
                
                # 构造spec: f_y(x) - f_i(x) > 0
                margins = []
                for other in range(num_classes):
                    if other == target_label:
                        continue
                    # 使用上界(最不利情况)
                    mar = target_lower - upper[0, other]
                    margins.append(mar.item())
                
                min_margin = min(margins)
                safety_margins[layer_idx] = min_margin
                
                critical_results[layer_idx] = {
                    "lower_bound": lower[0, target_label].item(),
                    "min_margin": min_margin,
                    "is_safe_layer": min_margin > 0
                }
        
        if not safety_margins:
            return IntegratedResult(
                is_safe=False,
                confidence=0.0,
                safety_margin=float('-inf'),
                num_critical_layers=len(critical_layers),
                num_total_layers=len(full_layer_bounds),
                critical_layer_results=critical_results,
                inference_strategy=InferenceStrategy.CONSERVATIVE,
                explanation="No critical layer bounds available"
            )
        
        # 取最保守(最小)的安全边距
        overall_margin = min(safety_margins.values())
        is_safe = overall_margin > 0
        
        # 置信度 = sigmoid(安全边距 * 缩放因子)
        confidence = 1.0 / (1.0 + np.exp(-overall_margin * 10))
        
        explanation = (
            f"保守推断: {len(critical_layers)}个关键层中,"
            f"最小安全边距={overall_margin:.4f},"
            f"结论: {'安全' if is_safe else '不安全'}"
        )
        
        return IntegratedResult(
            is_safe=is_safe,
            confidence=confidence,
            safety_margin=overall_margin,
            num_critical_layers=len(critical_layers),
            num_total_layers=len(full_layer_bounds),
            critical_layer_results=critical_results,
            inference_strategy=InferenceStrategy.CONSERVATIVE,
            explanation=explanation
        )
    
    def _weighted_inference(
        self,
        full_layer_bounds: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
        critical_layers: List[int],
        target_label: int,
        num_classes: int,
        layer_importance: Optional[Dict[int, float]] = None
    ) -> IntegratedResult:
        """
        加权推断策略
        
        基于层重要性权重加权综合判定
        """
        if layer_importance is None:
            # 均等权重
            total = len(critical_layers)
            layer_importance = {idx: 1.0/total for idx in critical_layers}
        
        # 归一化权重
        total_weight = sum(layer_importance.values())
        norm_weights = {k: v/total_weight for k, v in layer_importance.items()}
        
        weighted_margin = 0.0
        critical_results = {}
        
        for layer_idx in critical_layers:
            if layer_idx not in full_layer_bounds:
                continue
            
            lower, upper = full_layer_bounds[layer_idx]
            weight = norm_weights.get(layer_idx, 0.0)
            
            if lower.dim() > 1 and lower.shape[-1] == num_classes:
                target_lower = lower[0, target_label]
                margins = []
                for other in range(num_classes):
                    if other == target_label:
                        continue
                    mar = target_lower - upper[0, other]
                    margins.append(mar.item())
                
                min_margin = min(margins) if margins else 0.0
                weighted_margin += weight * min_margin
                
                critical_results[layer_idx] = {
                    "lower_bound": lower[0, target_label].item(),
                    "min_margin": min_margin,
                    "weight": weight,
                    "weighted_contribution": weight * min_margin
                }
        
        is_safe = weighted_margin > 0
        confidence = 1.0 / (1.0 + np.exp(-weighted_margin * 5))
        
        explanation = (
            f"加权推断: 加权安全边距={weighted_margin:.4f},"
            f"结论: {'安全' if is_safe else '不安全'}"
        )
        
        return IntegratedResult(
            is_safe=is_safe,
            confidence=confidence,
            safety_margin=weighted_margin,
            num_critical_layers=len(critical_layers),
            num_total_layers=len(full_layer_bounds),
            critical_layer_results=critical_results,
            inference_strategy=InferenceStrategy.WEIGHTED,
            explanation=explanation
        )
    
    def _adaptive_inference(
        self,
        full_layer_bounds: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
        critical_layers: List[int],
        target_label: int,
        num_classes: int
    ) -> IntegratedResult:
        """
        自适应推断策略
        
        根据边界的松弛程度动态调整推断策略
        边界紧时信任,边界松时保守
        """
        # 分析各边界的紧致程度
        bound_tightness = {}
        critical_results = {}
        
        for layer_idx in critical_layers:
            if layer_idx not in full_layer_bounds:
                continue
            
            lower, upper = full_layer_bounds[layer_idx]
            
            if lower.dim() > 1 and lower.shape[-1] == num_classes:
                # 边界宽度反映松弛程度
                bound_width = (upper - lower).mean().item()
                # 紧致度 = 1 / (1 + 宽度)
                tightness = 1.0 / (1.0 + bound_width)
                bound_tightness[layer_idx] = tightness
                
                target_lower = lower[0, target_label]
                margins = []
                for other in range(num_classes):
                    if other == target_label:
                        continue
                    mar = target_lower - upper[0, other]
                    margins.append(mar.item())
                
                min_margin = min(margins) if margins else 0.0
                critical_results[layer_idx] = {
                    "lower_bound": lower[0, target_label].item(),
                    "min_margin": min_margin,
                    "tightness": tightness
                }
        
        if not bound_tightness:
            return IntegratedResult(
                is_safe=False, confidence=0.0, safety_margin=float('-inf'),
                num_critical_layers=0,
                num_total_layers=len(full_layer_bounds),
                critical_layer_results={},
                inference_strategy=InferenceStrategy.ADAPTIVE,
                explanation="No bounds available"
            )
        
        # 根据紧致度调整阈值
        avg_tightness = np.mean(list(bound_tightness.values()))
        # 紧致度高时使用宽松阈值,紧致度低时使用严格阈值
        adaptive_threshold = 0.1 * (1.0 - avg_tightness)
        
        # 计算加权边距,权重为紧致度
        total_tightness = sum(bound_tightness.values())
        adaptive_margin = sum(
            critical_results[idx]["min_margin"] * bound_tightness[idx]
            for idx in critical_layers if idx in critical_results
        ) / (total_tightness + 1e-10)
        
        is_safe = adaptive_margin > adaptive_threshold
        confidence = 1.0 / (1.0 + np.exp(-adaptive_margin * 10 / (avg_tightness + 0.1)))
        
        explanation = (
            f"自适应推断: 紧致度={avg_tightness:.3f},"
            f"自适应阈值={adaptive_threshold:.4f},"
            f"加权边距={adaptive_margin:.4f},"
            f"结论: {'安全' if is_safe else '不安全'}"
        )
        
        return IntegratedResult(
            is_safe=is_safe,
            confidence=confidence,
            safety_margin=adaptive_margin,
            num_critical_layers=len(critical_layers),
            num_total_layers=len(full_layer_bounds),
            critical_layer_results=critical_results,
            inference_strategy=InferenceStrategy.ADAPTIVE,
            explanation=explanation
        )
    
    def compare_inference_strategies(
        self,
        full_layer_bounds: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
        critical_layers: List[int],
        target_label: int,
        num_classes: int = 10,
        layer_importance: Optional[Dict[int, float]] = None
    ) -> Dict[str, IntegratedResult]:
        """
        比较不同推断策略的结果
        
        Returns:
            strategies: {策略名: IntegratedResult}
        """
        strategies = {}
        for strategy in InferenceStrategy:
            strategies[strategy.value] = self.infer_from_critical_layers(
                full_layer_bounds, critical_layers, target_label,
                num_classes, strategy, layer_importance
            )
        return strategies


class CompletenessMonitor:
    """
    完备性监控器
    
    量化评估验证完备性,可视化展示验证完备性变化曲线
    """
    
    def __init__(self):
        self.history = {
            'full_results': [],
            'simplified_results': [],
            'completeness_scores': [],
            'bound_differences': [],
            'timestamps': []
        }
    
    def evaluate_completeness(
        self,
        full_bounds: Tuple[torch.Tensor, torch.Tensor],
        simplified_bounds: Tuple[torch.Tensor, torch.Tensor],
        full_result: bool,
        simplified_result: bool
    ) -> float:
        """
        评估完备性
        
        Args:
            full_bounds: 全网络边界 (lower, upper)
            simplified_bounds: 简化边界 (lower, upper)
            full_result: 全网络验证结果
            simplified_result: 简化验证结果
        
        Returns:
            completeness: 完备性得分 (0-1)
        """
        full_l, full_u = full_bounds
        simple_l, simple_u = simplified_bounds
        
        # 边界差异
        bound_diff = torch.norm(full_l - simple_l) + torch.norm(full_u - simple_u)
        bound_norm = torch.norm(full_l) + torch.norm(full_u) + 1e-10
        relative_diff = (bound_diff / bound_norm).item()
        
        # 结果一致性
        result_match = 1.0 if full_result == simplified_result else 0.0
        
        # 完备性: 结果一致性占70%, 边界差异占30%
        completeness = 0.7 * result_match + 0.3 * (1.0 - min(relative_diff, 1.0))
        
        # 记录历史
        self.history['full_results'].append(full_result)
        self.history['simplified_results'].append(simplified_result)
        self.history['completeness_scores'].append(completeness)
        self.history['bound_differences'].append(relative_diff)
        self.history['timestamps'].append(len(self.history['timestamps']))
        
        return completeness
    
    def get_completeness_trend(self) -> Dict[str, Any]:
        """获取完备性变化趋势"""
        scores = self.history['completeness_scores']
        if not scores:
            return {"trend": "no_data", "current": 1.0, "average": 1.0}
        
        recent = scores[-10:] if len(scores) >= 10 else scores
        trend = "improving" if len(recent) >= 3 and recent[-1] > recent[0] else \
                "declining" if len(recent) >= 3 and recent[-1] < recent[0] else "stable"
        
        return {
            "trend": trend,
            "current": scores[-1],
            "average": np.mean(scores),
            "std": np.std(scores),
            "min": min(scores),
            "max": max(scores),
            "recent_trend": recent[-3:] if len(recent) >= 3 else recent
        }
    
    def compute_completeness_loss(self) -> Dict[str, float]:
        """计算完备性损失统计"""
        if not self.history['completeness_scores']:
            return {"mean_loss": 0.0, "max_loss": 0.0}
        
        losses = [1.0 - s for s in self.history['completeness_scores']]
        return {
            "mean_loss": float(np.mean(losses)),
            "max_loss": float(np.max(losses)),
            "loss_std": float(np.std(losses))
        }
