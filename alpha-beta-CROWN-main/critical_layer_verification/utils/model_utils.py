"""
模型工具函数
提供模型分析、参数统计等通用功能
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Optional


def get_model_size(model: nn.Module) -> float:
    """
    获取模型大小(MB)
    
    Args:
        model: PyTorch模型
    
    Returns:
        size_mb: 模型大小(MB)
    """
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size = param_size + buffer_size
    return total_size / (1024 * 1024)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    统计模型参数
    
    Args:
        model: PyTorch模型
    
    Returns:
        params: 参数字典
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": total - trainable
    }


def get_relu_layer_count(model: nn.Module) -> int:
    """获取ReLU层数量"""
    return sum(1 for m in model.modules() if isinstance(m, nn.ReLU))


def get_linear_layer_info(model: nn.Module) -> List[Dict]:
    """
    获取线性层信息
    
    Args:
        model: PyTorch模型
    
    Returns:
        layer_info: 每层的信息列表
    """
    info = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            info.append({
                "name": name,
                "in_features": module.in_features,
                "out_features": module.out_features,
                "weight_shape": list(module.weight.shape),
                "bias_shape": list(module.bias.shape) if module.bias is not None else None
            })
    return info


class ModelUtils:
    """模型工具类"""
    
    @staticmethod
    def analyze_model_architecture(model: nn.Module) -> Dict:
        """
        分析模型架构
        
        Args:
            model: PyTorch模型
        
        Returns:
            architecture: 架构信息
        """
        layers = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.ReLU, nn.Flatten)):
                layer_info = {
                    "name": name,
                    "type": type(module).__name__,
                }
                if isinstance(module, nn.Linear):
                    layer_info["in_features"] = module.in_features
                    layer_info["out_features"] = module.out_features
                elif isinstance(module, nn.Conv2d):
                    layer_info["in_channels"] = module.in_channels
                    layer_info["out_channels"] = module.out_channels
                    layer_info["kernel_size"] = module.kernel_size
                layers.append(layer_info)
        
        return {
            "num_layers": len(layers),
            "num_parameters": count_parameters(model),
            "model_size_mb": get_model_size(model),
            "layers": layers,
            "num_relu": get_relu_layer_count(model)
        }
    
    @staticmethod
    def compute_model_complexity(model: nn.Module, input_shape: Tuple = (1, 28*28)) -> Dict:
        """
        计算模型复杂度
        
        Args:
            model: PyTorch模型
            input_shape: 输入形状
        
        Returns:
            complexity: 复杂度信息
        """
        total_mult_add = 0
        
        for module in model.modules():
            if isinstance(module, nn.Linear):
                # FLOPs for Linear: 2 * in_features * out_features
                total_mult_add += 2 * module.in_features * module.out_features
        
        return {
            "flops": total_mult_add,
            "num_parameters": count_parameters(model)
        }
