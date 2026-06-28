"""
实验数据记录与分析平台
自动生成实验报告和性能对比图表
"""
import numpy as np
import json
import os
from typing import List, Dict, Optional, Any, Tuple
from collections import defaultdict
from dataclasses import dataclass


class DataAnalyzer:
    """
    实验数据分析器
    
    分析实验结果,生成性能对比图表和报告
    """
    
    def __init__(self, results_dir: str = 'my_experiments'):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
    
    def compute_performance_summary(self, results: List[Dict]) -> Dict[str, Any]:
        """计算性能摘要"""
        if not results:
            return {}
        
        full_times = [r.get('full_verification_time', 0) for r in results]
        simple_times = [r.get('simplified_verification_time', 0) for r in results]
        
        speedups = [
            f / (s + 1e-10) 
            for f, s in zip(full_times, simple_times)
            if f > 0 and s > 0
        ]
        
        return {
            'avg_full_time': float(np.mean(full_times)),
            'avg_simple_time': float(np.mean(simple_times)),
            'avg_speedup': float(np.mean(speedups)) if speedups else 0,
            'max_speedup': float(np.max(speedups)) if speedups else 0,
            'min_speedup': float(np.min(speedups)) if speedups else 0,
            'speedup_std': float(np.std(speedups)) if len(speedups) > 1 else 0,
            'total_samples': len(results),
            'safe_count': sum(1 for r in results if r.get('is_safe', False)),
            'unsafe_count': sum(1 for r in results if not r.get('is_safe', True)),
        }
    
    def compute_identification_metrics(
        self,
        true_critical: List[int],
        identified_critical: List[int],
        total_layers: int
    ) -> Dict[str, float]:
        """
        计算识别准确率指标
        
        Args:
            true_critical: 真实关键层
            identified_critical: 识别出的关键层
            total_layers: 总层数
        
        Returns:
            metrics: 评估指标
        """
        true_set = set(true_critical)
        identified_set = set(identified_critical)
        
        tp = len(true_set & identified_set)
        fp = len(identified_set - true_set)
        fn = len(true_set - identified_set)
        tn = total_layers - tp - fp - fn
        
        accuracy = (tp + tn) / total_layers if total_layers > 0 else 0
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'true_negatives': tn
        }
    
    def generate_report(
        self,
        full_results: List[Dict],
        simplified_results: List[Dict],
        model_name: str,
        epsilon: float,
        critical_layers: List[int],
        output_file: Optional[str] = None
    ) -> str:
        """
        生成实验报告
        
        Args:
            full_results: 全网络验证结果
            simplified_results: 简化验证结果
            model_name: 模型名称
            epsilon: 扰动半径
            critical_layers: 关键层
            output_file: 输出文件路径
        
        Returns:
            report: 报告文本
        """
        report = []
        report.append("=" * 70)
        report.append(f"基于关键层的简化验证方法 - 实验报告")
        report.append("=" * 70)
        report.append(f"\n模型: {model_name}")
        report.append(f"扰动半径 (ε): {epsilon}")
        report.append(f"关键层: {critical_layers}")
        report.append(f"测试样本数: {len(full_results)}")
        
        # 性能对比
        report.append(f"\n{'─'*70}")
        report.append("一、性能对比")
        report.append(f"{'─'*70}")
        
        full_perf = self.compute_performance_summary(
            [{'full_verification_time': r['full_verification_time'] if isinstance(r, dict) else r.verification_time} 
             for r in full_results]
        )
        simple_perf = self.compute_performance_summary(
            [{'full_verification_time': r['simplified_verification_time'] if isinstance(r, dict) else r.verification_time}
             for r in simplified_results]
        )
        
        report.append(f"\n  {'指标':<20} {'全网络验证':<16} {'简化验证':<16}")
        report.append(f"  {'-'*52}")
        report.append(f"  {'平均验证时间(秒)':<20} {full_perf.get('avg_full_time', 0):<16.4f} "
                      f"{simple_perf.get('avg_simple_time', 0):<16.4f}")
        
        speedup = full_perf.get('avg_full_time', 0) / (simple_perf.get('avg_simple_time', 0) + 1e-10)
        report.append(f"\n  验证加速比: {speedup:.2f}x")
        
        # 完备性分析
        report.append(f"\n{'─'*70}")
        report.append("二、完备性分析")
        report.append(f"{'─'*70}")
        
        consistent = 0
        for fr, sr in zip(full_results, simplified_results):
            f_safe = fr.get('is_safe', fr.is_safe if hasattr(fr, 'is_safe') else False) if isinstance(fr, dict) else fr.is_safe
            s_safe = sr.get('is_safe', sr.is_safe if hasattr(sr, 'is_safe') else False) if isinstance(sr, dict) else sr.is_safe
            if f_safe == s_safe:
                consistent += 1
        
        consistency_rate = consistent / max(len(full_results), 1)
        report.append(f"\n  结果一致率: {consistency_rate:.2%}")
        report.append(f"  完备性损失: {1 - consistency_rate:.2%}")
        
        # 结论
        report.append(f"\n{'─'*70}")
        report.append("三、结论")
        report.append(f"{'─'*70}")
        
        if speedup >= 10:
            report.append(f"  ✓ 验证速度提升达到目标(≥10倍)")
        else:
            report.append(f"  △ 验证速度提升未达到目标10倍")
        
        if (1 - consistency_rate) <= 0.05:
            report.append(f"  ✓ 完备性损失控制在5%以内")
        else:
            report.append(f"  △ 完备性损失超过5%限制")
        
        report.append(f"\n{'='*70}")
        
        report_text = "\n".join(report)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"报告已保存至: {output_file}")
        
        return report_text
    
    def save_results_csv(self, results: List[Dict], filename: str):
        """保存结果为CSV"""
        import csv
        
        if not results:
            return
        
        filepath = os.path.join(self.results_dir, filename)
        keys = results[0].keys()
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"CSV已保存至: {filepath}")
    
    def plot_comparison(self, results_by_model: Dict[str, List[Dict]], 
                        save_path: Optional[str] = None):
        """
        生成对比图表(简单ASCII版本)
        
        Args:
            results_by_model: {模型名: [结果列表]}
            save_path: 保存路径
        """
        lines = []
        lines.append("\n性能对比图表")
        lines.append("="*60)
        lines.append(f"{'模型':<10} {'加速比':<10} {'一致率':<10} {'时间节省':<10}")
        lines.append("-"*60)
        
        for model_name, results in results_by_model.items():
            if not results:
                continue
            
            full_times = [r.get('full_verification_time', 0) for r in results]
            simple_times = [r.get('simplified_verification_time', 0) for r in results]
            
            speedup = np.mean(full_times) / (np.mean(simple_times) + 1e-10)
            time_saving = (1 - np.mean(simple_times) / (np.mean(full_times) + 1e-10)) * 100
            
            consistent = sum(1 for r in results if r.get('consistent', True))
            consistency = consistent / len(results) * 100
            
            lines.append(f"{model_name:<10} {speedup:<10.2f}x {consistency:<10.1f}% {time_saving:<10.1f}%")
        
        lines.append("="*60)
        
        chart = "\n".join(lines)
        print(chart)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(chart)
