"""
关键神经元子集识别与提取功能
开发高效的数据结构存储关键神经元信息
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class CriticalNeuronInfo:
    """关键神经元信息"""
    layer_idx: int
    neuron_indices: List[int]           # 神经元索引列表
    importance_scores: List[float]      # 重要性得分
    activation_frequencies: List[float] # 激活频率
    is_stable: List[bool]              # 是否稳定
    
    @property
    def num_neurons(self) -> int:
        return len(self.neuron_indices)
    
    def to_dict(self) -> Dict:
        return {
            "layer_idx": self.layer_idx,
            "neuron_indices": self.neuron_indices,
            "importance_scores": self.importance_scores,
            "activation_frequencies": self.activation_frequencies,
            "is_stable": self.is_stable,
            "num_neurons": self.num_neurons
        }


class CriticalNeuronStore:
    """
    关键神经元信息存储
    
    使用高效的数据结构存储和检索关键神经元信息
    """
    
    def __init__(self):
        # {layer_idx: CriticalNeuronInfo}
        self._store: Dict[int, CriticalNeuronInfo] = {}
        # 快速查找: {layer_idx: set(neuron_idx)}
        self._index: Dict[int, Set[int]] = defaultdict(set)
        # 标记: 当前活跃的关键神经元掩码
        self._masks: Dict[int, torch.Tensor] = {}
    
    def add_layer_info(self, info: CriticalNeuronInfo):
        """添加层的关键神经元信息"""
        self._store[info.layer_idx] = info
        self._index[info.layer_idx] = set(info.neuron_indices)
    
    def get_layer_info(self, layer_idx: int) -> Optional[CriticalNeuronInfo]:
        """获取指定层的关键神经元信息"""
        return self._store.get(layer_idx)
    
    def is_critical_neuron(self, layer_idx: int, neuron_idx: int) -> bool:
        """判断神经元是否为关键神经元"""
        return neuron_idx in self._index.get(layer_idx, set())
    
    def get_critical_count(self, layer_idx: int) -> int:
        """获取指定层的关键神经元数量"""
        return len(self._index.get(layer_idx, set()))
    
    def get_critical_mask(self, layer_idx: int, total_neurons: int) -> torch.Tensor:
        """
        获取关键神经元掩码张量
        
        Args:
            layer_idx: 层索引
            total_neurons: 该层总神经元数
        
        Returns:
            mask: bool类型掩码 [total_neurons]
        """
        if layer_idx in self._masks:
            return self._masks[layer_idx]
        
        mask = torch.zeros(total_neurons, dtype=torch.bool)
        if layer_idx in self._index:
            indices = list(self._index[layer_idx])
            if indices:
                mask[torch.tensor(indices)] = True
        
        self._masks[layer_idx] = mask
        return mask
    
    def get_all_critical_layers(self) -> List[int]:
        """获取所有包含关键神经元的层索引"""
        return list(self._store.keys())
    
    def summary(self) -> Dict:
        """获取存储摘要"""
        total_neurons = sum(info.num_neurons for info in self._store.values())
        return {
            "num_layers": len(self._store),
            "total_critical_neurons": total_neurons,
            "layers": {idx: self._store[idx].num_neurons 
                      for idx in sorted(self._store.keys())}
        }


class CriticalNeuronIdentifier:
    """
    关键神经元子集识别器
    
    基于多维度评估指标,识别每个层中的关键神经元子集
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device = torch.device('cpu'),
        verbose: bool = True
    ):
        self.model = model
        self.device = device
        self.verbose = verbose
        self.model.eval()
        self.model.to(device)
    
    def identify_critical_neurons(
        self,
        importance_scores: Dict[int, torch.Tensor],
        activation_frequencies: Dict[int, torch.Tensor],
        activation_masks: Dict[int, torch.Tensor],
        top_k_ratio: float = 0.3,
        min_neurons: int = 5,
        max_neurons: Optional[int] = None
    ) -> CriticalNeuronStore:
        """
        识别关键神经元子集
        
        Args:
            importance_scores: {层索引: 重要性得分 [n_neurons]}
            activation_frequencies: {层索引: 激活频率 [n_neurons]}
            activation_masks: {层索引: 激活掩码 [batch, n_neurons]}
            top_k_ratio: 选择前k%的神经元
            min_neurons: 每层最少神经元数
            max_neurons: 每层最多神经元数
        
        Returns:
            store: 关键神经元信息存储
        """
        store = CriticalNeuronStore()
        
        for layer_idx in sorted(importance_scores.keys()):
            scores = importance_scores[layer_idx]
            freqs = activation_frequencies.get(layer_idx, torch.ones_like(scores))
            mask = activation_masks.get(layer_idx)
            
            n_neurons = scores.shape[0]
            k = max(min_neurons, int(n_neurons * top_k_ratio))
            if max_neurons is not None:
                k = min(k, max_neurons)
            k = min(k, n_neurons)
            
            # 按重要性得分排序
            sorted_indices = torch.argsort(scores, descending=True)
            top_indices = sorted_indices[:k]
            
            # 判断稳定性: 平衡激活的神经元认为是不稳定的
            stable = torch.ones(k, dtype=torch.bool)
            for i, idx in enumerate(top_indices):
                if mask is not None:
                    freq = mask[:, idx].float().mean().item()
                    is_stable = freq >= 0.95 or freq <= 0.05
                else:
                    freq = freqs[idx].item()
                    is_stable = freq >= 0.95 or freq <= 0.05
                stable[i] = is_stable
            
            neuron_info = CriticalNeuronInfo(
                layer_idx=layer_idx,
                neuron_indices=top_indices.tolist(),
                importance_scores=scores[top_indices].tolist(),
                activation_frequencies=freqs[top_indices].tolist(),
                is_stable=stable.tolist()
            )
            
            store.add_layer_info(neuron_info)
            
            if self.verbose:
                n_unstable = sum(1 for s in stable if not s)
                print(f"  层 {layer_idx}: 识别出 {k} 个关键神经元 "
                      f"(其中 {n_unstable} 个不稳定)")
        
        return store
    
    def compute_neuron_overlap(
        self,
        store1: CriticalNeuronStore,
        store2: CriticalNeuronStore
    ) -> Dict[int, float]:
        """
        计算两个关键神经元集合的重叠率
        
        Args:
            store1: 第一个存储
            store2: 第二个存储
        
        Returns:
            overlap: {层索引: 重叠率}
        """
        overlap = {}
        all_layers = set(store1._index.keys()) | set(store2._index.keys())
        
        for layer_idx in all_layers:
            set1 = store1._index.get(layer_idx, set())
            set2 = store2._index.get(layer_idx, set())
            
            if not set1 or not set2:
                overlap[layer_idx] = 0.0
            else:
                intersection = len(set1 & set2)
                union = len(set1 | set2)
                overlap[layer_idx] = intersection / union
        
        return overlap
    
    def print_identification_summary(self, store: CriticalNeuronStore):
        """打印识别摘要"""
        summary = store.summary()
        
        print("\n" + "="*60)
        print("关键神经元子集识别摘要")
        print("="*60)
        print(f"\n包含关键神经元的层数: {summary['num_layers']}")
        print(f"关键神经元总数: {summary['total_critical_neurons']}")
        print(f"\n各层分布:")
        print(f"{'层索引':<10} {'关键神经元数':<15} {'占比':<10}")
        print("-"*35)
        
        for layer_idx, count in summary['layers'].items():
            info = store.get_layer_info(layer_idx)
            total = len(info.neuron_indices)  # approximate
            ratio = 1.0  # total is unknown here but we can show count
            print(f"{layer_idx:<10} {count:<15} {ratio:<10.2%}")
