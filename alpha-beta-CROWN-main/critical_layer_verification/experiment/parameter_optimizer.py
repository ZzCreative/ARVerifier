"""
参数优化工具
实现关键层识别算法的自适应调整,支持交叉验证和网格搜索
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Optional, Any, Callable
from dataclasses import dataclass
from itertools import product
import json
import os


@dataclass
class OptimizationResult:
    """优化结果"""
    best_params: Dict[str, Any]
    best_score: float
    all_results: List[Tuple[Dict[str, Any], float]]
    search_method: str
    
    def summary(self) -> str:
        """获取优化摘要"""
        return (
            f"最佳参数: {self.best_params}\n"
            f"最佳得分: {self.best_score:.4f}\n"
            f"搜索方法: {self.search_method}\n"
            f"搜索次数: {len(self.all_results)}"
        )


class ParameterOptimizer:
    """
    参数优化器
    
    支持:
    1. 网格搜索 (Grid Search)
    2. 随机搜索 (Random Search)
    3. 自适应调整 (Adaptive Tuning)
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.best_params_ = None
        self.best_score_ = float('-inf')
    
    def grid_search(
        self,
        param_grid: Dict[str, List[Any]],
        evaluate_fn: Callable[..., float],
        maximize: bool = True
    ) -> OptimizationResult:
        """
        网格搜索
        
        Args:
            param_grid: 参数网格 {参数名: [候选值列表]}
            evaluate_fn: 评估函数,返回得分
            maximize: True表示最大化,False表示最小化
        
        Returns:
            result: 优化结果
        """
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(product(*values))
        
        if self.verbose:
            print(f"\n网格搜索: {len(combinations)} 组参数")
        
        results = []
        best_score = float('-inf') if maximize else float('inf')
        best_params = None
        
        for i, combo in enumerate(combinations):
            params = dict(zip(keys, combo))
            
            try:
                score = evaluate_fn(**params)
                results.append((params, score))
                
                if (maximize and score > best_score) or (not maximize and score < best_score):
                    best_score = score
                    best_params = params
                
                if self.verbose and (i + 1) % 5 == 0:
                    print(f"  进度: {i+1}/{len(combinations)} 当前最佳={best_score:.4f}")
            
            except Exception as e:
                if self.verbose:
                    print(f"  参数 {params} 评估失败: {e}")
                continue
        
        self.best_params_ = best_params
        self.best_score_ = best_score
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_results=results,
            search_method="grid_search"
        )
    
    def random_search(
        self,
        param_distributions: Dict[str, Tuple[Any, Any]],
        evaluate_fn: Callable[..., float],
        n_iter: int = 20,
        maximize: bool = True,
        seed: int = 42
    ) -> OptimizationResult:
        """
        随机搜索
        
        Args:
            param_distributions: 参数分布 {参数名: (最小值, 最大值)}
            n_iter: 迭代次数
            evaluate_fn: 评估函数
            maximize: 是否最大化
            seed: 随机种子
        
        Returns:
            result: 优化结果
        """
        rng = np.random.RandomState(seed)
        
        results = []
        best_score = float('-inf') if maximize else float('inf')
        best_params = None
        
        for i in range(n_iter):
            params = {}
            for key, (low, high) in param_distributions.items():
                if isinstance(low, int) and isinstance(high, int):
                    params[key] = int(rng.randint(low, high + 1))
                elif isinstance(low, float) or isinstance(high, float):
                    params[key] = float(rng.uniform(low, high))
                else:
                    params[key] = rng.uniform(low, high)
            
            try:
                score = evaluate_fn(**params)
                results.append((params, score))
                
                if (maximize and score > best_score) or (not maximize and score < best_score):
                    best_score = score
                    best_params = params
                
                if self.verbose and (i + 1) % 5 == 0:
                    print(f"  随机搜索进度: {i+1}/{n_iter} 当前最佳={best_score:.4f}")
            
            except Exception as e:
                if self.verbose:
                    print(f"  参数 {params} 评估失败: {e}")
                continue
        
        self.best_params_ = best_params
        self.best_score_ = best_score
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_results=results,
            search_method="random_search"
        )
    
    def adaptive_tune(
        self,
        initial_params: Dict[str, Any],
        param_bounds: Dict[str, Tuple[float, float]],
        evaluate_fn: Callable[..., float],
        max_iterations: int = 50,
        learning_rate: float = 0.1,
        maximize: bool = True
    ) -> OptimizationResult:
        """
        自适应参数调整
        
        使用简单的梯度上升/下降方法
        
        Args:
            initial_params: 初始参数
            param_bounds: 参数边界
            evaluate_fn: 评估函数
            max_iterations: 最大迭代次数
            learning_rate: 学习率
            maximize: 是否最大化
        
        Returns:
            result: 优化结果
        """
        params = initial_params.copy()
        results = []
        best_score = float('-inf') if maximize else float('inf')
        best_params = params.copy()
        
        for iteration in range(max_iterations):
            score = evaluate_fn(**params)
            results.append((params.copy(), score))
            
            if (maximize and score > best_score) or (not maximize and score < best_score):
                best_score = score
                best_params = params.copy()
            
            # 参数更新
            for key in params:
                low, high = param_bounds.get(key, (0.0, 1.0))
                if isinstance(params[key], float):
                    # 随机扰动
                    perturbation = np.random.randn() * learning_rate * (high - low)
                    params[key] = np.clip(params[key] + perturbation, low, high)
            
            if self.verbose and (iteration + 1) % 10 == 0:
                print(f"  自适应调整: {iteration+1}/{max_iterations} 最佳={best_score:.4f}")
        
        self.best_params_ = best_params
        self.best_score_ = best_score
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_results=results,
            search_method="adaptive_tune"
        )
