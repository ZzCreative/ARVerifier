"""
神经元活跃度统计工具
实现神经元激活频率、激活强度分布及激活模式相关性的量化分析
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import defaultdict


class NeuronActivityAnalyzer:
    """
    神经元活跃度分析器
    
    统计每个神经元在不同输入下的激活模式,
    识别出对网络决策影响最大的关键神经元
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
        
        # 存储激活记录的缓存
        self.activation_records = defaultdict(list)
        self._register_hooks()
    
    def _register_hooks(self):
        """注册hook以记录ReLU层激活"""
        self.hooks = []
        self.relu_outputs = {}
        
        relu_idx = 0
        for name, module in self.model.named_modules():
            if isinstance(module, nn.ReLU):
                idx = relu_idx
                
                def make_hook(layer_idx):
                    def hook(module, input, output):
                        self.relu_outputs[layer_idx] = output.detach()
                    return hook
                
                self.hooks.append(
                    module.register_forward_hook(make_hook(idx))
                )
                relu_idx += 1
    
    def remove_hooks(self):
        """移除所有hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def collect_activation_stats(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_batches: int = 10
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        收集神经元激活统计信息
        
        Args:
            dataloader: 数据加载器
            num_batches: 处理的批次数
        
        Returns:
            stats: {层索引: {pre_activations, post_activations, activation_mask}}
        """
        stats = defaultdict(dict)
        layer_pre = defaultdict(list)
        layer_post = defaultdict(list)
        
        self.model.eval()
        count = 0
        
        with torch.no_grad():
            for inputs, _ in dataloader:
                if count >= num_batches:
                    break
                
                inputs = inputs.to(self.device)
                self.relu_outputs.clear()
                
                # 前向传播前记录pre-activation
                pre_activations = {}
                pre_hooks = []
                
                relu_idx = 0
                for name, module in self.model.named_modules():
                    if isinstance(module, nn.ReLU):
                        idx = relu_idx
                        
                        def make_pre_hook(layer_idx):
                            def hook(module, input, output):
                                pre_activations[layer_idx] = input[0].detach()
                            return hook
                        
                        pre_hooks.append(
                            module.register_forward_hook(make_pre_hook(idx))
                        )
                        relu_idx += 1
                
                _ = self.model(inputs)
                
                for idx, pre_act in pre_activations.items():
                    layer_pre[idx].append(pre_act.cpu())
                
                for idx, post_act in self.relu_outputs.items():
                    layer_post[idx].append(post_act.cpu())
                
                for hook in pre_hooks:
                    hook.remove()
                
                count += 1
        
        for idx in layer_pre:
            pre_concat = torch.cat(layer_pre[idx], dim=0)
            post_concat = torch.cat(layer_post[idx], dim=0)
            
            stats[idx] = {
                "pre_activations": pre_concat,
                "post_activations": post_concat,
                "activation_mask": (post_concat > 0).float()
            }
        
        return stats
    
    def compute_activation_frequency(
        self,
        stats: Dict[int, Dict[str, torch.Tensor]]
    ) -> Dict[int, torch.Tensor]:
        """
        计算激活频率
        
        Args:
            stats: {层索引: 激活统计数据}
        
        Returns:
            freq: {层索引: 激活频率 [n_neurons]}
        """
        freq = {}
        for layer_idx, layer_stats in stats.items():
            mask = layer_stats["activation_mask"]  # [batch, n_neurons]
            freq[layer_idx] = mask.mean(dim=0)  # [n_neurons]
        
        return freq
    
    def compute_activation_strength_distribution(
        self,
        stats: Dict[int, Dict[str, torch.Tensor]]
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        计算激活强度分布
        
        Args:
            stats: {层索引: 激活统计数据}
        
        Returns:
            dist: {层索引: {mean, std, max, min, histogram}}
        """
        dist = {}
        for layer_idx, layer_stats in stats.items():
            post = layer_stats["post_activations"]  # [batch, n_neurons]
            mask = layer_stats["activation_mask"]
            
            # 只考虑激活的神经元
            activated = post[mask > 0]
            
            dist[layer_idx] = {
                "mean": post.mean(dim=0),
                "std": post.std(dim=0),
                "max": post.max(dim=0).values,
                "min": post.min(dim=0).values,
                "activated_mean": activated.mean() if activated.numel() > 0 else torch.tensor(0.0),
                "activated_std": activated.std() if activated.numel() > 0 else torch.tensor(0.0)
            }
        
        return dist
    
    def compute_activation_pattern_correlation(
        self,
        stats: Dict[int, Dict[str, torch.Tensor]]
    ) -> Dict[Tuple[int, int], torch.Tensor]:
        """
        计算层间激活模式相关性
        
        Args:
            stats: {层索引: 激活统计数据}
        
        Returns:
            correlation: {(层i, 层j): 相关性矩阵}
        """
        correlation = {}
        layers = sorted(stats.keys())
        
        for i in layers:
            for j in layers:
                if i >= j:
                    continue
                
                mask_i = stats[i]["activation_mask"]  # [batch, n_i]
                mask_j = stats[j]["activation_mask"]  # [batch, n_j]
                
                batch_size = mask_i.shape[0]
                n_i = mask_i.shape[1]
                n_j = mask_j.shape[1]
                
                # 批量计算Pearson相关系数
                corr_matrix = torch.zeros(n_i, n_j)
                for b in range(batch_size):
                    # 对每个样本计算外积
                    outer = mask_i[b:b+1].T @ mask_j[b:b+1]  # [n_i, n_j]
                    corr_matrix += outer.float()
                
                corr_matrix /= batch_size
                correlation[(i, j)] = corr_matrix
        
        return correlation
    
    def identify_stable_neurons(
        self,
        freq: Dict[int, torch.Tensor],
        always_on_threshold: float = 0.95,
        always_off_threshold: float = 0.05
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        识别稳定/不稳定神经元
        
        Args:
            freq: {层索引: 激活频率}
            always_on_threshold: 始终激活阈值
            always_off_threshold: 始终未激活阈值
        
        Returns:
            stable: {层索引: {always_on, always_off, unstable}}
        """
        stable = {}
        for layer_idx, f in freq.items():
            stable[layer_idx] = {
                "always_on": f >= always_on_threshold,
                "always_off": f <= always_off_threshold,
                "unstable": (f > always_off_threshold) & (f < always_on_threshold)
            }
        
        return stable
    
    def compute_neuron_importance_scores(
        self,
        stats: Dict[int, Dict[str, torch.Tensor]],
        final_weights: Optional[Dict[int, torch.Tensor]] = None
    ) -> Dict[int, torch.Tensor]:
        """
        计算神经元重要性得分
        
        结合激活频率、激活强度和后续权重计算综合重要性
        
        Args:
            stats: {层索引: 激活统计数据}
            final_weights: {层索引: 后续权重矩阵}
        
        Returns:
            importance: {层索引: 重要性得分 [n_neurons]}
        """
        importance = {}
        freq = self.compute_activation_frequency(stats)
        strength = self.compute_activation_strength_distribution(stats)
        
        for layer_idx in stats.keys():
            f = freq[layer_idx]
            s = strength[layer_idx]["mean"] / (strength[layer_idx]["std"] + 1e-10)
            
            # 综合得分: 频率 * (1 + 相对强度)
            combined = f * (1 + torch.nan_to_num(s, nan=0.0, posinf=1.0, neginf=0.0))
            
            importance[layer_idx] = combined
        
        return importance
    
    def get_critical_neurons_mask(
        self,
        importance_scores: Dict[int, torch.Tensor],
        top_k_ratio: float = 0.3
    ) -> Dict[int, torch.Tensor]:
        """
        获取关键神经元掩码
        
        Args:
            importance_scores: {层索引: 重要性得分}
            top_k_ratio: 选择前k%的神经元
        
        Returns:
            masks: {层索引: 关键神经元掩码}
        """
        masks = {}
        for layer_idx, scores in importance_scores.items():
            k = max(1, int(scores.shape[-1] * top_k_ratio))
            _, indices = torch.topk(scores, k, dim=-1)
            
            mask = torch.zeros_like(scores, dtype=torch.bool)
            mask[indices] = True
            masks[layer_idx] = mask
        
        return masks
    
    def print_activation_summary(
        self,
        freq: Dict[int, torch.Tensor],
        importance_scores: Dict[int, torch.Tensor]
    ):
        """打印激活统计摘要"""
        print("\n" + "="*60)
        print("神经元活跃度统计摘要")
        print("="*60)
        
        for layer_idx in sorted(freq.keys()):
            f = freq[layer_idx]
            imp = importance_scores.get(layer_idx, torch.zeros_like(f))
            
            n_always_on = (f >= 0.95).sum().item()
            n_always_off = (f <= 0.05).sum().item()
            n_unstable = ((f > 0.05) & (f < 0.95)).sum().item()
            
            print(f"\n层 {layer_idx}:")
            print(f"  总神经元数: {f.shape[0]}")
            print(f"  激活频率范围: [{f.min().item():.3f}, {f.max().item():.3f}]")
            print(f"  始终激活: {n_always_on} ({100*n_always_on/f.shape[0]:.1f}%)")
            print(f"  始终未激活: {n_always_off} ({100*n_always_off/f.shape[0]:.1f}%)")
            print(f"  不稳定: {n_unstable} ({100*n_unstable/f.shape[0]:.1f}%)")
            print(f"  平均重要性: {imp.mean().item():.4f}")
            print(f"  最高重要性: {imp.max().item():.4f}")
