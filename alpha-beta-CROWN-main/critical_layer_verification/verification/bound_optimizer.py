"""
边界优化器 - 实现线性松弛过程的优化
对关键层使用更紧的边界估计
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Optional, Callable, Any


class BoundOptimizer:
    """
    边界优化器
    
    实现线性松弛过程的优化:
    1. 对关键层使用α-CROWN(可调参数)的紧致边界
    2. 对非关键层使用IBP的快速边界
    3. 支持自适应的松弛策略
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
    
    def optimize_alpha_parameters(
        self,
        pre_lower: torch.Tensor,
        pre_upper: torch.Tensor,
        num_iterations: int = 20,
        learning_rate: float = 0.1
    ) -> torch.Tensor:
        """
        优化α参数以获得更紧的ReLU松弛
        
        α-CROWN: ReLU(x)的下界松弛为 α*x (0 ≤ α ≤ 1)
        最优α = lower / (lower - upper) 当 upper > lower > 0 时
        
        Args:
            pre_lower: ReLU前下界 [batch, n_neurons]
            pre_upper: ReLU前上界 [batch, n_neurons]
            num_iterations: 优化迭代次数
            learning_rate: 学习率
        
        Returns:
            alpha: 最优α参数
        """
        # 肯定激活区域: α=1
        # 肯定未激活区域: α=0
        # 不稳定区域: 需要优化
        
        batch_size = pre_lower.shape[0]
        n_neurons = pre_lower.shape[1]
        
        device = pre_lower.device
        alpha = torch.full((batch_size, n_neurons), 0.5, device=device)
        
        # 设置边界约束
        active_mask = (pre_lower >= 0).float()
        dead_mask = (pre_upper <= 0).float()
        unstable_mask = (1 - active_mask - dead_mask).float()
        
        for iteration in range(num_iterations):
            # 对不稳定区域的α进行梯度更新
            if unstable_mask.sum() > 0:
                # 计算目标: 最小化松弛后的边界宽度
                # 边界宽度 = α * u - α * l = α * (u - l)
                # 最优α = l / (l - u)
                optimal_alpha = pre_lower / (pre_lower - pre_upper + 1e-10)
                optimal_alpha = torch.clamp(optimal_alpha, 0.0, 1.0)
                
                # 使用动量更新
                alpha = alpha * (1 - learning_rate) + optimal_alpha * learning_rate
                alpha = torch.clamp(alpha, 0.0, 1.0)
        
        # 应用边界
        final_alpha = active_mask * 1.0 + dead_mask * 0.0 + unstable_mask * alpha
        
        return final_alpha
    
    def compute_alpha_crown_bound(
        self,
        weight: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
        alpha: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用α-CROWN计算传播后的边界
        
        Args:
            weight: 权重矩阵 [n_out, n_in]
            lower: 输入下界 [batch, n_in]
            upper: 输入上界 [batch, n_in]
            alpha: α参数 [batch, n_in]
        
        Returns:
            (new_lower, new_upper): 传播后的边界
        """
        batch_size = lower.shape[0]
        n_out = weight.shape[0]
        n_in = weight.shape[1]
        
        # ReLU的CROWN线性界:
        # 下界系数: dL/dx = diag(α) * W^T (当W为正)
        # 上界系数: dU/dx = diag(1) * W^T (当W为正)
        # 对于负权重需要特殊处理
        
        # 分离正负权重
        W_plus = torch.clamp(weight, min=0.0)  # [n_out, n_in]
        W_minus = torch.clamp(weight, max=0.0)  # [n_out, n_in]
        
        # 计算下界系数
        diag_alpha = torch.diag_embed(alpha)  # [batch, n_in, n_in]
        
        # 传播下界: 使用α调整
        lower_coeff = W_plus.T.unsqueeze(0) * alpha.unsqueeze(1) + W_minus.T.unsqueeze(0)
        lower_bias = torch.zeros(batch_size, n_out, device=lower.device)
        
        # 简化边界计算
        new_lower = torch.zeros(batch_size, n_out, device=lower.device)
        new_upper = torch.zeros(batch_size, n_out, device=lower.device)
        
        for b in range(batch_size):
            for j in range(n_out):
                for i in range(n_in):
                    if weight[j, i] >= 0:
                        new_lower[b, j] += weight[j, i] * alpha[b, i] * lower[b, i]
                        new_upper[b, j] += weight[j, i] * upper[b, i]
                    else:
                        new_lower[b, j] += weight[j, i] * upper[b, i]
                        new_upper[b, j] += weight[j, i] * lower[b, i]
        
        return new_lower, new_upper
    
    def compute_crown_bounds_with_critical(
        self,
        input_lower: torch.Tensor,
        input_upper: torch.Tensor,
        model: nn.Module,
        critical_layers: Optional[List[int]] = None,
        use_beta: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        带关键层感知的CROWN边界计算
        
        Args:
            input_lower: 输入下界
            input_upper: 输入上界
            model: 神经网络
            critical_layers: 关键层索引列表
            use_beta: 是否使用β-CROWN
        
        Returns:
            (lower, upper, info): 边界和信息
        """
        if critical_layers is None:
            critical_layers = []
        
        critical_set = set(critical_layers)
        info = {
            "num_critical": len(critical_layers),
            "layers_processed": 0,
            "alpha_optimized_layers": 0
        }
        
        lower = input_lower.clone()
        upper = input_upper.clone()
        layer_idx = 0
        flattened = False
        
        for name, module in model.named_modules():
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
                
                new_lower = mu @ weight.T + bias - r @ torch.abs(weight).T
                new_upper = mu @ weight.T + bias + r @ torch.abs(weight).T
                
                lower = new_lower
                upper = new_upper
                layer_idx += 1
            
            elif isinstance(module, nn.ReLU):
                is_critical = (layer_idx in critical_set) if critical_layers else True
                
                if is_critical:
                    # 使用α-CROWN优化
                    alpha = self.optimize_alpha_parameters(lower, upper)
                    new_lower = alpha * lower
                    new_upper = torch.clamp(upper, min=0.0)
                    info["alpha_optimized_layers"] += 1
                else:
                    # 非关键层使用简单IBP
                    new_lower = torch.clamp(lower, min=0.0)
                    new_upper = torch.clamp(upper, min=0.0)
                
                lower = new_lower
                upper = new_upper
                layer_idx += 1
        
        info["layers_processed"] = layer_idx
        return lower, upper, info
