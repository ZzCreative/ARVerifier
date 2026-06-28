"""
对比实验方案
与标准α,β-CROWN全网络验证方法进行系统性性能比较
"""
import torch
import torch.nn as nn
import numpy as np
import time
import json
import os
import sys
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from critical_layer_verification.theory.theory_core import (
    KPICalculator
)
from critical_layer_verification.theory.metrics import (
    VerificationKPI, ExperimentResult, ExperimentSummary
)
from critical_layer_verification.verification.critical_layer_verifier import (
    CriticalLayerVerifier, VerificationMode, VerificationResult
)
from critical_layer_verification.identification.layer_selector import (
    CriticalLayerSelector, LayerImportanceMetrics
)


class ComparisonExperiment:
    """
    对比实验管理器
    
    设计科学的对比实验方案,与标准全网络验证方法进行系统性比较
    """
    
    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        device: torch.device = torch.device('cpu'),
        results_dir: str = 'my_experiments',
        verbose: bool = True
    ):
        self.model = model
        self.model_name = model_name
        self.device = device
        self.results_dir = results_dir
        self.verbose = verbose
        
        self.model.eval()
        self.model.to(device)
        
        # 验证器
        self.verifier = CriticalLayerVerifier(model, device, verbose)
        
        # 确保结果目录存在
        os.makedirs(results_dir, exist_ok=True)
    
    def run_full_verification(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        epsilon: float,
        num_classes: int = 10,
        num_samples: int = 25
    ) -> List[VerificationResult]:
        """
        运行全网络验证(基线)
        
        Args:
            images: 图像数据 [N, C, H, W]
            labels: 标签 [N]
            epsilon: 扰动半径
            num_classes: 类别数
            num_samples: 验证样本数
        
        Returns:
            results: 验证结果列表
        """
        if self.verbose:
            print(f"\n运行全网络验证 (ε={epsilon}):")
        
        results = self.verifier.verify_batch(
            images[:num_samples], labels[:num_samples],
            epsilon, num_classes,
            critical_layers=None,
            mode=VerificationMode.FULL,
            verbose=self.verbose
        )
        
        return results
    
    def run_simplified_verification(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        epsilon: float,
        critical_layers: List[int],
        num_classes: int = 10,
        num_samples: int = 25
    ) -> List[VerificationResult]:
        """
        运行简化验证(仅关键层)
        
        Args:
            images: 图像数据
            labels: 标签
            epsilon: 扰动半径
            critical_layers: 关键层索引列表
            num_classes: 类别数
            num_samples: 验证样本数
        
        Returns:
            results: 验证结果列表
        """
        if self.verbose:
            print(f"\n运行简化验证 (ε={epsilon}, 关键层={critical_layers}):")
        
        results = self.verifier.verify_batch(
            images[:num_samples], labels[:num_samples],
            epsilon, num_classes,
            critical_layers=critical_layers,
            mode=VerificationMode.HYBRID,
            verbose=self.verbose
        )
        
        return results
    
    def run_hybrid_verification(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        epsilon: float,
        critical_layers: List[int],
        num_classes: int = 10,
        num_samples: int = 25
    ) -> List[VerificationResult]:
        """
        运行混合验证(关键层精确+非关键层近似)
        """
        return self.run_simplified_verification(
            images, labels, epsilon, critical_layers, num_classes, num_samples
        )
    
    def compare_verification_methods(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        epsilon: float,
        critical_layers: List[int],
        num_classes: int = 10,
        num_samples: int = 25
    ) -> Tuple[List[VerificationResult], List[VerificationResult]]:
        """
        比较全网络和简化验证
        
        Returns:
            (full_results, simplified_results)
        """
        full_results = self.run_full_verification(
            images, labels, epsilon, num_classes, num_samples
        )
        
        simplified_results = self.run_simplified_verification(
            images, labels, epsilon, critical_layers, num_classes, num_samples
        )
        
        if self.verbose:
            self._print_comparison(full_results, simplified_results)
        
        return full_results, simplified_results
    
    def _print_comparison(
        self,
        full_results: List[VerificationResult],
        simplified_results: List[VerificationResult]
    ):
        """打印对比结果"""
        print("\n" + "="*60)
        print("验证方法对比")
        print("="*60)
        
        full_safe = sum(1 for r in full_results if r.is_safe)
        simple_safe = sum(1 for r in simplified_results if r.is_safe)
        
        full_time = np.mean([r.verification_time for r in full_results])
        simple_time = np.mean([r.verification_time for r in simplified_results])
        
        full_mem = np.mean([r.memory_usage_mb for r in full_results])
        simple_mem = np.mean([r.memory_usage_mb for r in simplified_results])
        
        print(f"\n{'指标':<20} {'全网络验证':<16} {'简化验证':<16} {'变化':<16}")
        print("-"*68)
        print(f"{'安全样本数':<20} {full_safe:<16} {simple_safe:<16} "
              f"{simple_safe-full_safe:>+d}")
        print(f"{'平均验证时间(秒)':<20} {full_time:<16.4f} {simple_time:<16.4f} "
              f"{(full_time-simple_time)/full_time*100:>+.1f}%")
        print(f"{'内存使用(MB)':<20} {full_mem:<16.2f} {simple_mem:<16.2f} "
              f"{(full_mem-simple_mem)/full_mem*100:>+.1f}%")
        
        # 一致性分析
        consistent = sum(
            1 for f, s in zip(full_results, simplified_results)
            if f.is_safe == s.is_safe
        )
        print(f"{'结果一致数':<20} {consistent:<16} / {len(full_results)}")
        print(f"{'一致率':<20} {consistent/len(full_results)*100:<16.1f}%")
    
    def run_comprehensive_comparison(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        epsilons: List[float],
        critical_layers_dict: Dict[float, List[int]],
        num_classes: int = 10,
        num_samples: int = 25
    ) -> Dict[float, ExperimentSummary]:
        """
        运行全面对比实验
        
        Args:
            images: 图像数据
            labels: 标签
            epsilons: 多个扰动半径
            critical_layers_dict: {epsilon: 关键层列表}
            num_classes: 类别数
            num_samples: 样本数
        
        Returns:
            summaries: {epsilon: ExperimentSummary}
        """
        summaries = {}
        
        for eps in epsilons:
            if self.verbose:
                print(f"\n\n{'='*60}")
                print(f"开始对比实验: ε={eps}")
                print(f"{'='*60}")
            
            critical_layers = critical_layers_dict.get(eps, [])
            
            full_results, simplified_results = self.compare_verification_methods(
                images, labels, eps, critical_layers, num_classes, num_samples
            )
            
            # 构建实验结果
            exp_results = []
            for i, (fr, sr) in enumerate(zip(full_results, simplified_results)):
                exp_results.append(ExperimentResult(
                    model_name=self.model_name,
                    epsilon=eps,
                    sample_index=i,
                    true_label=int(labels[i]),
                    full_verification_time=fr.verification_time,
                    simplified_verification_time=sr.verification_time,
                    full_result="safe" if fr.is_safe else "unsafe",
                    simplified_result="safe" if sr.is_safe else "unsafe",
                    full_memory_mb=fr.memory_usage_mb,
                    simplified_memory_mb=sr.memory_usage_mb,
                    identified_critical_layers=critical_layers,
                    total_layers=len(critical_layers) + 10,  # rough estimate
                    completeness_score=1.0 if fr.is_safe == sr.is_safe else 0.0
                ))
            
            summary = ExperimentSummary(
                model_name=self.model_name,
                epsilon=eps,
                num_samples=num_samples,
                results=exp_results
            )
            summary.compute_kpi()
            summaries[eps] = summary
            
            # 保存结果
            self._save_results(summary, eps)
        
        return summaries
    
    def _save_results(self, summary: ExperimentSummary, epsilon: float):
        """保存实验结果"""
        filename = f"{self.results_dir}/comparison_{self.model_name}_eps{epsilon}.json"
        
        data = {
            "model_name": self.model_name,
            "epsilon": epsilon,
            "num_samples": summary.num_samples,
            "kpi": {
                "avg_speedup": summary.kpi.avg_speedup,
                "completeness_loss": summary.kpi.completeness_loss,
                "verification_accuracy": summary.kpi.verification_accuracy,
                "memory_reduction": summary.kpi.memory_reduction,
                "identification_accuracy": summary.kpi.identification_accuracy,
                "false_positive_rate": summary.kpi.false_positive_rate,
                "false_negative_rate": summary.kpi.false_negative_rate,
            },
            "results": [
                {
                    "index": r.sample_index,
                    "full_time": r.full_verification_time,
                    "simple_time": r.simplified_verification_time,
                    "full_result": r.full_result,
                    "simple_result": r.simplified_result,
                }
                for r in summary.results
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        if self.verbose:
            print(f"\n结果已保存至: {filename}")
