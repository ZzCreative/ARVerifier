"""
基于α,β-CROWN框架的关键层专用验证框架
优化线性松弛过程,在关键层上使用精确边界,非关键层上使用近似边界
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum
import time
import sys
import os

# Add parent path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


class VerificationMode(Enum):
    """验证模式"""
    FULL = "full"               # 全网络精确验证
    CRITICAL_ONLY = "critical"  # 仅关键层精确验证
    HYBRID = "hybrid"           # 混合验证


@dataclass
class VerificationResult:
    """验证结果"""
    is_safe: bool                   # 是否安全
    verified_lower_bound: float     # 验证下界
    verified_upper_bound: float     # 验证上界
    verification_time: float        # 验证时间(秒)
    mode: VerificationMode         # 验证模式
    critical_layers: List[int]     # 关键层索引
    memory_usage_mb: float         # 内存使用(MB)


class CriticalLayerVerifier:
    """
    关键层验证器
    
    在α,β-CROWN框架基础上,选择性对关键层进行精确边界计算,
    非关键层使用更松的近似,从而大幅提升验证速度
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
        
        # 缓存
        self._bounds_cache = {}
        self._layer_cache = {}
    
    def get_relu_layers(self) -> List[int]:
        """获取所有ReLU层的索引"""
        if 'relu_layers' in self._layer_cache:
            return self._layer_cache['relu_layers']
        
        indices = []
        idx = 0
        for name, module in self.model.named_modules():
            if isinstance(module, nn.ReLU):
                indices.append(idx)
                idx += 1
        
        self._layer_cache['relu_layers'] = indices
        return indices
    
    def get_linear_layers(self) -> List[nn.Linear]:
        """获取所有Linear层(不包括Flatten)"""
        layers = []
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                layers.append(module)
        return layers
    
    def run_forward_pass(
        self,
        input_tensor: torch.Tensor
    ) -> Dict[int, torch.Tensor]:
        """
        运行前向传播,记录每层输出
        
        Args:
            input_tensor: 输入张量 [1, c, h, w]
        
        Returns:
            layer_outputs: {层索引: 输出张量}
        """
        layer_outputs = {}
        hooks = []
        
        relu_idx = 0
        for name, module in self.model.named_modules():
            if isinstance(module, nn.ReLU):
                idx = relu_idx
                def make_hook(layer_idx):
                    def hook(module, input, output):
                        layer_outputs[layer_idx] = output.detach()
                    return hook
                hooks.append(module.register_forward_hook(make_hook(idx)))
                relu_idx += 1
        
        with torch.no_grad():
            _ = self.model(input_tensor.to(self.device))
        
        for hook in hooks:
            hook.remove()
        
        return layer_outputs
    
    def compute_ibp_bounds(
        self,
        input_lower: torch.Tensor,
        input_upper: torch.Tensor,
        stop_layer: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算区间传播边界(IBP)
        
        Args:
            input_lower: 输入下界 [1, c, h, w]
            input_upper: 输入上界 [1, c, h, w]
            stop_layer: 停止计算的层索引
        
        Returns:
            (lower, upper): 输出边界
        """
        lower = input_lower.clone()
        upper = input_upper.clone()
        
        layer_idx = 0
        flattened = False
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Flatten):
                if not flattened:
                    lower = lower.reshape(lower.shape[0], -1)
                    upper = upper.reshape(upper.shape[0], -1)
                    flattened = True
            
            elif isinstance(module, nn.Linear):
                weight = module.weight
                bias = module.bias
                
                # IBP传播: μ = (l+u)/2, r = (u-l)/2
                mu = (lower + upper) / 2
                r = (upper - lower) / 2
                
                new_mu = mu @ weight.T + bias
                
                # |W| * r 需要处理正负号
                abs_weight = torch.abs(weight)
                new_r = r @ abs_weight.T
                
                lower = new_mu - new_r
                upper = new_mu + new_r
                
                if stop_layer is not None and layer_idx >= stop_layer:
                    break
                layer_idx += 1
            
            elif isinstance(module, nn.ReLU):
                if stop_layer is not None and layer_idx >= stop_layer:
                    break
                # ReLU: [l, u] -> [ReLU(l), ReLU(u)]
                lower = torch.clamp(lower, min=0.0)
                upper = torch.clamp(upper, min=0.0)
                layer_idx += 1
        
        return lower, upper
    
    def compute_crown_bounds(
        self,
        input_lower: torch.Tensor,
        input_upper: torch.Tensor,
        target_label: int,
        num_classes: int = 10,
        critical_layers: Optional[List[int]] = None,
        use_alpha: bool = True,
        use_beta: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        计算CROWN边界（简化版）
        
        对关键层使用精确线性松弛,非关键层使用快速近似
        
        Args:
            input_lower: 输入下界 [1, c, h, w]
            input_upper: 输入上界 [1, c, h, w]
            target_label: 目标标签
            num_classes: 类别数
            critical_layers: 关键层索引列表(None=全部精确)
            use_alpha: 是否使用α参数优化
            use_beta: 是否使用β参数优化(更精确但更慢)
        
        Returns:
            (lower_bound, upper_bound, computation_time)
        """
        start_time = time.time()
        
        if critical_layers is None:
            critical_layers = []
        
        # 构建目标函数: e_y - e_target
        batch_size = input_lower.shape[0]
        
        # 简化的CROWN边界计算
        # 实际项目中应调用auto_LiRPA的CROWN实现
        # 这里提供一个近似实现
        lower = input_lower.clone()
        upper = input_upper.clone()
        
        critical_set = set(critical_layers)
        layer_idx = 0
        flattened = False
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Flatten):
                if not flattened:
                    lower = lower.reshape(lower.shape[0], -1)
                    upper = upper.reshape(upper.shape[0], -1)
                    flattened = True
            
            elif isinstance(module, nn.Linear):
                weight = module.weight
                bias = module.bias
                
                mu = (lower + upper) / 2
                r = (upper - lower) / 2
                
                new_mu = mu @ weight.T + bias
                abs_weight = torch.abs(weight)
                new_r = r @ abs_weight.T
                
                lower = new_mu - new_r
                upper = new_mu + new_r
                layer_idx += 1
            
            elif isinstance(module, nn.ReLU):
                is_critical = (layer_idx in critical_set) if critical_layers else True
                
                if is_critical:
                    # 精确的CROWN线性松弛
                    # 使用triangle relaxation: ReLU(x) ≈ α*x
                    # 其中α取决于x的预激活上下界
                    beta_mask = (lower >= 0).float()  # 肯定激活
                    dead_mask = (upper <= 0).float()   # 肯定未激活
                    unstable_mask = 1 - beta_mask - dead_mask  # 不稳定区域
                    
                    # 对于不稳定区域使用最优α
                    if use_alpha and unstable_mask.sum() > 0:
                        # 使用α参数: α = l/(l-u) 或 启发式 0.5
                        alpha = torch.where(
                            (upper - lower).abs() > 1e-8,
                            lower / (lower - upper + 1e-10),
                            torch.ones_like(lower) * 0.5
                        )
                        alpha = torch.clamp(alpha, 0.0, 1.0)
                    else:
                        alpha = torch.ones_like(lower) * 0.5
                    
                    # 应用线性松弛
                    new_lower = lower * (beta_mask + unstable_mask * alpha)
                    new_upper = upper * (beta_mask + unstable_mask * 1.0)
                    
                    lower = new_lower
                    upper = new_upper
                else:
                    # 非关键层: 使用简单的IBP传播
                    lower = torch.clamp(lower, min=0.0)
                    upper = torch.clamp(upper, min=0.0)
                
                layer_idx += 1
        
        elapsed = time.time() - start_time
        
        return lower, upper, elapsed
    
    def verify_sample(
        self,
        input_tensor: torch.Tensor,
        epsilon: float,
        true_label: int,
        num_classes: int = 10,
        critical_layers: Optional[List[int]] = None,
        mode: VerificationMode = VerificationMode.HYBRID,
        norm_type: str = "linf"
    ) -> VerificationResult:
        """
        验证单个样本
        
        Args:
            input_tensor: 输入张量 [1, c, h, w]
            epsilon: 扰动半径
            true_label: 真实标签
            num_classes: 类别数
            critical_layers: 关键层索引列表
            mode: 验证模式
            norm_type: 范数类型
        
        Returns:
            result: 验证结果
        """
        start_time = time.time()
        
        # 计算输入边界
        if norm_type == "linf":
            input_lower = torch.clamp(input_tensor - epsilon, 0.0, 1.0)
            input_upper = torch.clamp(input_tensor + epsilon, 0.0, 1.0)
        else:
            raise ValueError(f"Unsupported norm: {norm_type}")
        
        # 对每个非目标类别计算验证边界
        is_safe = True
        verified_lower = float('inf')
        verified_upper = float('-inf')
        
        # 只需检查目标标签是否保持最高置信度
        for other_label in range(num_classes):
            if other_label == true_label:
                continue
            
            lower, upper, _ = self.compute_crown_bounds(
                input_lower, input_upper,
                true_label, num_classes,
                critical_layers if mode != VerificationMode.FULL else None,
                use_alpha=(mode != VerificationMode.CRITICAL_ONLY)
            )
            
            # 检查边界:d_y - d_o > 0
            # 这里使用最终输出的边界差值
            margin_lower = lower[0, true_label] - upper[0, other_label]
            
            if margin_lower < 0:
                is_safe = False
                break
            
            verified_lower = min(verified_lower, margin_lower.item())
        
        elapsed = time.time() - start_time
        
        # 估计内存使用
        memory_mb = self._estimate_memory_usage(critical_layers, mode)
        
        return VerificationResult(
            is_safe=is_safe,
            verified_lower_bound=verified_lower if verified_lower != float('inf') else 0.0,
            verified_upper_bound=verified_upper if verified_upper != float('-inf') else 0.0,
            verification_time=elapsed,
            mode=mode,
            critical_layers=critical_layers or [],
            memory_usage_mb=memory_mb
        )
    
    def _estimate_memory_usage(
        self,
        critical_layers: Optional[List[int]],
        mode: VerificationMode
    ) -> float:
        """
        估计内存使用
        
        Args:
            critical_layers: 关键层列表
            mode: 验证模式
        
        Returns:
            memory_mb: 估计内存使用(MB)
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        
        if mode == VerificationMode.FULL:
            # 全网络验证: 需要存储所有中间边界
            base_memory = total_params * 4 * 4 / (1024 * 1024)  # float32 * 4 copies
            return base_memory * 2
        
        elif mode == VerificationMode.CRITICAL_ONLY:
            # 仅关键层: 减少边界存储
            if not critical_layers:
                return 10.0
            reduction_factor = len(critical_layers) / max(self._get_total_relu(), 1)
            base_memory = total_params * 4 * 4 / (1024 * 1024)
            return base_memory * (0.3 + 0.7 * reduction_factor)
        
        else:  # HYBRID
            if not critical_layers:
                return self._estimate_memory_usage(None, VerificationMode.FULL)
            reduction_factor = len(critical_layers) / max(self._get_total_relu(), 1)
            base_memory = total_params * 4 * 4 / (1024 * 1024)
            return base_memory * (0.5 + 0.5 * reduction_factor)
    
    def _get_total_relu(self) -> int:
        """获取ReLU层总数"""
        return len(self.get_relu_layers())
    
    def verify_batch(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        epsilon: float,
        num_classes: int = 10,
        critical_layers: Optional[List[int]] = None,
        mode: VerificationMode = VerificationMode.HYBRID,
        verbose: bool = False
    ) -> List[VerificationResult]:
        """
        批量验证
        
        Args:
            images: 图像数据 [N, C, H, W]
            labels: 标签 [N]
            epsilon: 扰动半径
            num_classes: 类别数
            critical_layers: 关键层列表
            mode: 验证模式
            verbose: 是否显示进度
        
        Returns:
            results: 验证结果列表
        """
        results = []
        total = len(images)
        
        for i in range(total):
            input_tensor = torch.from_numpy(images[i:i+1]).float().to(self.device)
            true_label = int(labels[i])
            
            result = self.verify_sample(
                input_tensor, epsilon, true_label,
                num_classes, critical_layers, mode
            )
            results.append(result)
            
            if verbose and (i + 1) % 10 == 0:
                print(f"  完成 {i+1}/{total}, "
                      f"安全: {sum(1 for r in results if r.is_safe)}/{len(results)}")
        
        return results
