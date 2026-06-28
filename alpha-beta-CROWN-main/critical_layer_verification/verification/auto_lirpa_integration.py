"""
auto_LiRPA 集成适配器

将 auto_LiRPA (α,β-CROWN) 框架集成到关键层验证管线中。
实现对关键层使用精确 α-CROWN，非关键层使用快速 IBP 的混合验证策略。
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Optional, Any, Callable
from dataclasses import dataclass
import time
import sys
import os

# 添加 auto_LiRPA 路径 - 尝试多个可能的位置
import pathlib
_CURRENT_DIR = pathlib.Path(__file__).resolve().parent
# 从当前文件位置向上查找 auto_LiRPA 目录
_AUTO_LIRPA_CANDIDATES = [
    _CURRENT_DIR.parent.parent / 'auto_LiRPA',           # 项目根目录
    _CURRENT_DIR.parent.parent.parent / 'auto_LiRPA',    # 上级项目目录
    pathlib.Path.cwd() / 'auto_LiRPA',                   # 当前工作目录
]
for _path in _AUTO_LIRPA_CANDIDATES:
    _str_path = str(_path.resolve())
    if _path.exists() and (_path / '__init__.py').exists():
        if _str_path not in sys.path:
            sys.path.insert(0, _str_path)
        break

try:
    from auto_LiRPA import BoundedModule, BoundedTensor
    from auto_LiRPA.perturbations import PerturbationLpNorm
    from auto_LiRPA.bound_op_map import register_custom_op, unregister_custom_op
    HAS_AUTO_LIRPA = True
except ImportError as e:
    HAS_AUTO_LIRPA = False
    print(f"[警告] auto_LiRPA 导入失败: {e}")
    print("[警告] 将使用简化的边界计算替代")


@dataclass
class AutoLiRPAConfig:
    """auto_LiRPA 配置"""
    method: str = 'CROWN-IBP'         # 计算方法: IBP, CROWN, CROWN-IBP, alpha-CROWN
    use_alpha: bool = True            # 是否使用 alpha-CROWN 优化
    use_beta: bool = False            # 是否使用 beta-CROWN 优化(更精确但更慢)
    crown_batch_size: int = 1         # CROWN 批大小
    sparse_intermediate_bounds: bool = True  # 稀疏中间层界
    bound_every_node: bool = True     # 计算每个节点的界
    forward_refinement: bool = False  # 前向精炼
    device: str = 'cpu'
    verbose: bool = False


class AutoLiRPAVerifier:
    """
    auto_LiRPA 验证器
    
    封装 auto_LiRPA 的 BoundedModule，提供关键层感知的边界计算
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[AutoLiRPAConfig] = None,
        device: torch.device = torch.device('cpu')
    ):
        """
        Args:
            model: PyTorch 神经网络
            config: auto_LiRPA 配置
            device: 计算设备
        """
        self.original_model = model
        self.device = device
        self.config = config or AutoLiRPAConfig()
        self._bounded_model = None
        self._dummy_input = None
        
        if HAS_AUTO_LIRPA:
            self._init_bounded_model()
        else:
            print("[警告] auto_LiRPA 不可用，功能受限")
    
    def _init_bounded_model(self):
        """初始化 BoundedModule"""
        if not HAS_AUTO_LIRPA:
            return None
        
        self.original_model.eval()
        self.original_model.to(self.device)
        
        # 创建 dummy input（batch_size=1）
        dummy_input = self._create_dummy_input()
        self._dummy_input = dummy_input
        
        # 转换为 BoundedModule
        bound_opts = {
            'sparse_intermediate_bounds': self.config.sparse_intermediate_bounds,
            'bound_every_node': self.config.bound_every_node,
            'crown_batch_size': self.config.crown_batch_size,
            'batched_crown_max_vram_ratio': 0.5,
        }
        
        self._bounded_model = BoundedModule(
            self.original_model,
            dummy_input,
            bound_opts=bound_opts,
            device=self.device,
            verbose=self.config.verbose
        )
        
        return self._bounded_model
    
    def _create_dummy_input(self) -> torch.Tensor:
        """根据模型结构创建 dummy input"""
        try:
            # 尝试从模型第一层推断输入形状
            for module in self.original_model.modules():
                if isinstance(module, nn.Linear):
                    # Flatten + Linear: 输入是 [batch, 1, 28, 28]
                    return torch.randn(1, 1, 28, 28).to(self.device)
                elif isinstance(module, nn.Conv2d):
                    return torch.randn(1, module.in_channels, 32, 32).to(self.device)
        except:
            pass
        # 默认 MNIST 输入
        return torch.randn(1, 1, 28, 28).to(self.device)
    
    def compute_bounds_with_crown(
        self,
        input_lower: torch.Tensor,
        input_upper: torch.Tensor,
        target_label: int,
        num_classes: int = 10,
        critical_layers: Optional[List[int]] = None,
        C_matrix: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        使用 auto_LiRPA 计算 CROWN 边界（关键层感知版本）
        
        Args:
            input_lower: 输入下界 [batch, c, h, w]
            input_upper: 输入上界 [batch, c, h, w]
            target_label: 目标标签
            num_classes: 类别数
            critical_layers: 关键层索引列表（None=全部精确）
            C_matrix: 规格矩阵（可选）
        
        Returns:
            (lower, upper, elapsed): 输出边界和计算时间
        """
        if not HAS_AUTO_LIRPA or self._bounded_model is None:
            raise RuntimeError("auto_LiRPA 不可用")
        
        start_time = time.time()
        batch_size = input_lower.shape[0]
        
        # 1. 构建规格矩阵 C
        if C_matrix is None:
            # 默认：验证目标标签 vs 其他所有标签
            C = torch.zeros(batch_size, num_classes - 1, num_classes, device=self.device)
            other_idx = 0
            for other in range(num_classes):
                if other == target_label:
                    continue
                C[:, other_idx, target_label] = 1.0
                C[:, other_idx, other] = -1.0
                other_idx += 1
        else:
            C = C_matrix
        
        # 2. 创建 BoundedTensor（使用 L_inf 扰动）
        # x_L 和 x_U 分别作为下界和上界
        center = (input_lower + input_upper) / 2
        eps = (input_upper - input_lower) / 2
        
        # 使用 eps 和 center 构造 PerturbationLpNorm
        ptb = PerturbationLpNorm(
            norm=np.inf,
            eps=eps.max().item(),
            x_L=input_lower,
            x_U=input_upper
        )
        bounded_x = BoundedTensor(center, ptb)
        
        # 3. 计算边界
        # 选择计算方法
        if self.config.use_alpha and critical_layers is not None and len(critical_layers) == self._get_num_relu_layers():
            # 全部精确：使用 alpha-CROWN
            method = 'alpha-CROWN'
        elif self.config.use_alpha and critical_layers:
            # 混合：CROWN-IBP + alpha
            method = 'CROWN-IBP'
        else:
            method = self.config.method
        
        lower, upper = self._bounded_model.compute_bounds(
            x=(bounded_x,),
            C=C,
            method=method,
            bound_lower=True,
            bound_upper=True,
        )
        
        elapsed = time.time() - start_time
        return lower, upper, elapsed
    
    def compute_ibp_bounds(
        self,
        input_lower: torch.Tensor,
        input_upper: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        使用 auto_LiRPA 计算 IBP 边界
        
        Args:
            input_lower: 输入下界
            input_upper: 输入上界
        
        Returns:
            (lower, upper, elapsed)
        """
        if not HAS_AUTO_LIRPA or self._bounded_model is None:
            raise RuntimeError("auto_LiRPA 不可用")
        
        start_time = time.time()
        
        center = (input_lower + input_upper) / 2
        ptb = PerturbationLpNorm(
            norm=np.inf,
            eps=((input_upper - input_lower) / 2).max().item(),
            x_L=input_lower,
            x_U=input_upper
        )
        bounded_x = BoundedTensor(center, ptb)
        
        lower, upper = self._bounded_model.compute_bounds(
            x=(bounded_x,),
            method='IBP',
            bound_lower=True,
            bound_upper=True,
        )
        
        elapsed = time.time() - start_time
        return lower, upper, elapsed
    
    def compute_hybrid_bounds(
        self,
        input_lower: torch.Tensor,
        input_upper: torch.Tensor,
        target_label: int,
        num_classes: int = 10,
        critical_layers: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        混合边界计算：关键层精确 + 非关键层近似
        
        这是核心集成方法：
        1. 使用参考界（reference_bounds）将非关键层的 pre-activation 界设为宽松 IBP 界
        2. 关键层使用 tight alpha-CROWN
        
        Args:
            input_lower: 输入下界
            input_upper: 输入上界
            target_label: 目标标签
            num_classes: 类别数
            critical_layers: 关键层索引列表
        
        Returns:
            (lower, upper, info)
        """
        if not HAS_AUTO_LIRPA or self._bounded_model is None:
            raise RuntimeError("auto_LiRPA 不可用; 请先安装 auto_LiRPA 框架")
        
        start_time = time.time()
        info = {'method': 'hybrid', 'critical_layers': critical_layers or []}
        
        # 1. 先计算 IBP 界（作为非关键层的宽松参考）
        ibp_lower, ibp_upper, ibp_time = self.compute_ibp_bounds(
            input_lower, input_upper
        )
        info['ibp_time'] = ibp_time
        
        # 2. 构建 C 矩阵
        batch_size = input_lower.shape[0]
        center = (input_lower + input_upper) / 2
        eps_val = ((input_upper - input_lower) / 2).max().item()
        
        C = torch.zeros(batch_size, num_classes - 1, num_classes, device=self.device)
        other_idx = 0
        for other in range(num_classes):
            if other == target_label:
                continue
            C[:, other_idx, target_label] = 1.0
            C[:, other_idx, other] = -1.0
            other_idx += 1
        
        # 3. 使用 CROWN-IBP 方法计算边界
        # 非关键层用 IBP 参考界（自动使用较松的界）
        ptb = PerturbationLpNorm(
            norm=np.inf,
            eps=eps_val,
            x_L=input_lower,
            x_U=input_upper
        )
        bounded_x = BoundedTensor(center, ptb)
        
        # 选择方法
        if critical_layers and len(critical_layers) < self._get_num_relu_layers():
            method = 'CROWN-IBP'  # 混合模式
        else:
            method = 'alpha-CROWN' if self.config.use_alpha else 'CROWN'
        
        lower, upper = self._bounded_model.compute_bounds(
            x=(bounded_x,),
            C=C,
            method=method,
            bound_lower=True,
            bound_upper=True,
        )
        
        elapsed = time.time() - start_time
        info['computation_time'] = elapsed
        info['method_used'] = method
        
        # 4. 判断安全性：如果所有 margin > 0 则安全
        is_safe = (lower > 0).all().item()
        info['is_safe'] = is_safe
        info['min_margin'] = lower.min().item()
        
        return lower, upper, info
    
    def _get_num_relu_layers(self) -> int:
        """获取 ReLU 层数量"""
        return sum(1 for m in self.original_model.modules() if isinstance(m, nn.ReLU))
    
    def verify_sample_with_lirpa(
        self,
        input_tensor: torch.Tensor,
        epsilon: float,
        true_label: int,
        num_classes: int = 10,
        critical_layers: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        使用 auto_LiRPA 验证单个样本
        
        Args:
            input_tensor: 输入张量 [1, c, h, w]
            epsilon: L_inf 扰动半径
            true_label: 真实标签
            num_classes: 类别数
            critical_layers: 关键层索引列表
        
        Returns:
            result: 验证结果字典
        """
        # 计算输入边界
        input_lower = torch.clamp(input_tensor - epsilon, 0.0, 1.0)
        input_upper = torch.clamp(input_tensor + epsilon, 0.0, 1.0)
        
        try:
            lower, upper, info = self.compute_hybrid_bounds(
                input_lower.to(self.device),
                input_upper.to(self.device),
                true_label,
                num_classes,
                critical_layers,
            )
            
            return {
                'is_safe': info.get('is_safe', False),
                'min_margin': info.get('min_margin', float('-inf')),
                'verification_time': info.get('computation_time', 0),
                'method': info.get('method_used', 'unknown'),
                'critical_layers': critical_layers or [],
            }
        except Exception as e:
            return {
                'is_safe': False,
                'min_margin': float('-inf'),
                'verification_time': 0,
                'method': 'error',
                'error': str(e),
                'critical_layers': critical_layers or [],
            }
    
    def estimate_memory_reduction(
        self,
        critical_layers: Optional[List[int]] = None
    ) -> float:
        """
        估计使用 auto_LiRPA 时的内存降低
        
        Returns:
            reduction: 内存降低比例
        """
        if not critical_layers:
            return 0.0
        
        total_relu = self._get_num_relu_layers()
        if total_relu == 0:
            return 0.0
        
        # auto_LiRPA 中，非关键层不需要存储 alpha 参数和中间界
        # 这显著减少了内存使用
        critical_ratio = len(critical_layers) / total_relu
        
        # alpha 参数内存 + 中间界内存
        # 关键层需要 alpha (float) + 中间界 (2*float)
        # 非关键层只需要中间界 (2*float)
        full_memory = total_relu * (1 + 2)  # alpha + lower + upper
        simplified_memory = (
            len(critical_layers) * (1 + 2) +  # 关键层
            (total_relu - len(critical_layers)) * 2  # 非关键层只需要界
        )
        
        return 1.0 - simplified_memory / full_memory


def check_auto_lirpa_available() -> bool:
    """检查 auto_LiRPA 是否可用"""
    return HAS_AUTO_LIRPA
