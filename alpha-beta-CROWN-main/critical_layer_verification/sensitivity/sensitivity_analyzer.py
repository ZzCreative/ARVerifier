"""
敏感性分析模块
基于输入扰动方法(L∞范数扰动、随机噪声注入)和输出变化度量指标
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Optional, Callable
from collections import defaultdict


class SensitivityAnalyzer:
    """
    敏感性分析器
    
    通过对每一层注入扰动,观测输出边界的相对变化量,
    量化各层对鲁棒性验证结果的影响力
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device = torch.device('cpu'),
        perturbation_norm: str = 'linf',
        num_perturbation_levels: int = 5,
        verbose: bool = True
    ):
        """
        Args:
            model: PyTorch神经网络模型
            device: 计算设备
            perturbation_norm: 扰动范数类型 ('linf', 'l2')
            num_perturbation_levels: 扰动级别数量
            verbose: 是否打印详细信息
        """
        self.model = model
        self.device = device
        self.perturbation_norm = perturbation_norm
        self.num_perturbation_levels = num_perturbation_levels
        self.verbose = verbose
        self.model.eval()
        self.model.to(device)
    
    def extract_layer_outputs(
        self,
        input_tensor: torch.Tensor,
        layer_indices: Optional[List[int]] = None
    ) -> Dict[int, torch.Tensor]:
        """
        提取指定层的输出
        
        Args:
            input_tensor: 输入张量 [batch, channels, height, width]
            layer_indices: 需要提取的层索引列表, None表示所有层
        
        Returns:
            layer_outputs: {layer_idx: output_tensor}
        """
        layer_outputs = {}
        hooks = []
        
        def make_hook(layer_idx):
            def hook(module, input, output):
                layer_outputs[layer_idx] = output.detach()
            return hook
        
        # 注册hook到所有ReLU层
        relu_idx = 0
        for name, module in self.model.named_modules():
            if isinstance(module, nn.ReLU):
                if layer_indices is None or relu_idx in layer_indices:
                    hooks.append(module.register_forward_hook(make_hook(relu_idx)))
                relu_idx += 1
        
        # 前向传播
        with torch.no_grad():
            _ = self.model(input_tensor.to(self.device))
        
        # 移除hooks
        for hook in hooks:
            hook.remove()
        
        return layer_outputs
    
    def compute_classification_confidence_change(
        self,
        base_output: torch.Tensor,
        perturbed_output: torch.Tensor
    ) -> float:
        """
        计算分类置信度变化率
        
        Args:
            base_output: 基准输出 [batch, n_classes]
            perturbed_output: 扰动后输出 [batch, n_classes]
        
        Returns:
            confidence_change: 置信度变化率
        """
        base_probs = torch.softmax(base_output, dim=-1)
        perturbed_probs = torch.softmax(perturbed_output, dim=-1)
        
        # 计算预测标签的置信度变化
        base_conf, base_pred = base_probs.max(dim=-1)
        perturbed_conf = perturbed_probs.gather(1, base_pred.unsqueeze(1)).squeeze()
        
        change = torch.abs(base_conf - perturbed_conf)
        return change.mean().item()
    
    def compute_prediction_stability(
        self,
        base_output: torch.Tensor,
        perturbed_output: torch.Tensor
    ) -> float:
        """
        计算预测标签稳定性
        
        Args:
            base_output: 基准输出
            perturbed_output: 扰动后输出
        
        Returns:
            stability: 稳定性得分 (0-1, 越高越稳定)
        """
        base_pred = base_output.argmax(dim=-1)
        perturbed_pred = perturbed_output.argmax(dim=-1)
        
        stable = (base_pred == perturbed_pred).float()
        return stable.mean().item()
    
    def inject_l_inf_perturbation(
        self,
        layer_output: torch.Tensor,
        epsilon: float
    ) -> torch.Tensor:
        """
        注入L∞范数扰动
        
        Args:
            layer_output: 原始层输出 [n_neurons]
            epsilon: 扰动半径
        
        Returns:
            perturbed_output: 扰动后的输出
        """
        noise = torch.randn_like(layer_output)
        noise = noise / (noise.norm(float('inf'), dim=-1, keepdim=True) + 1e-10)
        perturbed = layer_output + epsilon * noise
        return perturbed
    
    def inject_random_noise(
        self,
        layer_output: torch.Tensor,
        noise_std: float = 0.01
    ) -> torch.Tensor:
        """
        注入随机高斯噪声
        
        Args:
            layer_output: 原始层输出
            noise_std: 噪声标准差
        
        Returns:
            perturbed_output: 扰动后的输出
        """
        noise = torch.randn_like(layer_output) * noise_std
        return layer_output + noise
    
    def compute_layer_sensitivity(
        self,
        input_tensor: torch.Tensor,
        epsilon: float,
        layer_idx: int,
        use_random_noise: bool = False
    ) -> Dict[str, float]:
        """
        计算单层的敏感性指标
        
        Args:
            input_tensor: 输入张量
            epsilon: 扰动强度
            layer_idx: 层索引
            use_random_noise: 是否使用随机噪声代替L∞扰动
        
        Returns:
            sensitivity_metrics: 敏感性指标字典
        """
        # 获取基准输出
        with torch.no_grad():
            base_output = self.model(input_tensor.to(self.device))
        
        # 获取指定层的输出
        layer_outputs = self.extract_layer_outputs(input_tensor, [layer_idx])
        
        if layer_idx not in layer_outputs:
            return {"error": 1.0, "confidence_change": 1.0, "stability": 0.0}
        
        layer_output = layer_outputs[layer_idx]
        
        # 注入扰动
        if use_random_noise:
            perturbed_layer = self.inject_random_noise(layer_output, epsilon)
        else:
            perturbed_layer = self.inject_l_inf_perturbation(layer_output, epsilon)
        
        # 创建扰动后的输入（通过修改层输出模拟扰动传播）
        # 使用hook替换层输出
        perturbed_final = None
        
        def make_perturb_hook():
            nonlocal perturbed_final
            original_output = [None]
            
            def hook(module, input, output):
                if original_output[0] is None:
                    original_output[0] = output.detach()
                # 替换为扰动后的输出
                return perturbed_layer
            
            return hook
        
        hooks = []
        relu_idx = 0
        for name, module in self.model.named_modules():
            if isinstance(module, nn.ReLU):
                if relu_idx == layer_idx:
                    handle = module.register_forward_hook(make_perturb_hook())
                    hooks.append(handle)
                relu_idx += 1
        
        # 前向传播得到受扰动影响后的最终输出
        with torch.no_grad():
            try:
                perturbed_final = self.model(input_tensor.to(self.device))
            except Exception:
                perturbed_final = base_output.clone()
        
        for hook in hooks:
            hook.remove()
        
        # 计算敏感性指标
        confidence_change = self.compute_classification_confidence_change(
            base_output, perturbed_final
        )
        stability = self.compute_prediction_stability(base_output, perturbed_final)
        
        # 计算输出边界变化
        output_diff = torch.norm(perturbed_final - base_output).item()
        base_norm = torch.norm(base_output).item() + 1e-10
        
        return {
            "sensitivity": output_diff / base_norm,
            "confidence_change": confidence_change,
            "stability": stability,
            "output_diff": output_diff
        }
    
    def analyze_all_layers(
        self,
        input_tensor: torch.Tensor,
        epsilon: float = 0.01
    ) -> Dict[int, Dict[str, float]]:
        """
        分析所有层的敏感性
        
        Args:
            input_tensor: 输入张量
            epsilon: 扰动强度
        
        Returns:
            layer_sensitivity: {层索引: 敏感性指标}
        """
        # 先获取所有ReLU层的数量
        num_relu = sum(1 for m in self.model.modules() if isinstance(m, nn.ReLU))
        
        layer_sensitivity = {}
        for layer_idx in range(num_relu):
            metrics = self.compute_layer_sensitivity(
                input_tensor, epsilon, layer_idx
            )
            layer_sensitivity[layer_idx] = metrics
            
            if self.verbose:
                print(f"  层 {layer_idx}: 敏感性={metrics['sensitivity']:.4f}, "
                      f"置信度变化={metrics['confidence_change']:.4f}, "
                      f"稳定性={metrics['stability']:.4f}")
        
        return layer_sensitivity
    
    def multi_level_sensitivity_analysis(
        self,
        input_tensor: torch.Tensor,
        epsilons: Optional[List[float]] = None
    ) -> Dict[float, Dict[int, Dict[str, float]]]:
        """
        多级别敏感性分析
        
        Args:
            input_tensor: 输入张量
            epsilons: 扰动强度列表
        
        Returns:
            results: {epsilon: {layer_idx: metrics}}
        """
        if epsilons is None:
            epsilons = [0.001, 0.005, 0.01, 0.02, 0.05]
        
        results = {}
        for eps in epsilons:
            if self.verbose:
                print(f"\n扰动强度 ε={eps}:")
            results[eps] = self.analyze_all_layers(input_tensor, eps)
        
        return results
    
    def rank_layers_by_sensitivity(
        self,
        layer_sensitivity: Dict[int, Dict[str, float]]
    ) -> List[Tuple[int, float]]:
        """
        按敏感性对层进行排序
        
        Args:
            layer_sensitivity: {层索引: 敏感性指标}
        
        Returns:
            ranked_layers: [(层索引, 综合敏感得分)]
        """
        scores = []
        for layer_idx, metrics in layer_sensitivity.items():
            if "error" in metrics:
                continue
            # 综合得分: 敏感性权重0.5 + 置信度变化权重0.3 + (1-稳定性)权重0.2
            combined = (
                0.5 * metrics.get("sensitivity", 0) +
                0.3 * metrics.get("confidence_change", 0) +
                0.2 * (1 - metrics.get("stability", 1))
            )
            scores.append((layer_idx, combined))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores
