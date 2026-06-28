"""
可视化工具模块
提供验证结果、完备性变化、性能对比等可视化功能
"""
import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass


class VisualizationTools:
    """
    可视化工具类
    
    提供完备性损失变化曲线、性能对比图表等可视化功能
    """
    
    @staticmethod
    def generate_completeness_chart(
        completeness_scores: List[float],
        width: int = 60,
        height: int = 12
    ) -> str:
        """
        生成完备性变化曲线(ASCII柱状图)
        
        Args:
            completeness_scores: 完备性得分列表
            width: 图表宽度
            height: 图表高度
        
        Returns:
            chart: ASCII图表字符串
        """
        if not completeness_scores:
            return "无数据"
        
        scores = np.array(completeness_scores)
        normalized = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
        
        lines = []
        lines.append("完备性变化曲线")
        lines.append("─" * (width + 10))
        
        # Y轴标签
        for h in range(height, -1, -1):
            threshold = h / height
            line = f"{threshold:.1f} │"
            
            for s in normalized:
                if s >= threshold:
                    line += "█"
                else:
                    line += " "
            
            lines.append(line)
        
        lines.append("  " + "└" + "─" * (width) + " 样本索引")
        
        # 统计信息
        lines.append(f"\n完备性统计:")
        lines.append(f"  平均: {scores.mean():.4f}")
        lines.append(f"  最大: {scores.max():.4f}")
        lines.append(f"  最小: {scores.min():.4f}")
        lines.append(f"  标准差: {scores.std():.4f}")
        
        return "\n".join(lines)
    
    @staticmethod
    def generate_speedup_chart(
        speedups: Dict[str, List[float]],
        width: int = 50
    ) -> str:
        """
        生成加速比对比图表
        
        Args:
            speedups: {模型名: [加速比列表]}
            width: 条图宽度
        
        Returns:
            chart: ASCII图表
        """
        lines = []
        lines.append("验证加速比对比")
        lines.append("=" * (width + 20))
        
        max_speedup = max(
            np.mean(speedups[model]) for model in speedups
        ) if speedups else 1.0
        
        for model_name, sps in speedups.items():
            avg_sp = np.mean(sps)
            bar_length = int(avg_sp / (max_speedup + 0.1) * width)
            bar = "█" * bar_length
            
            lines.append(f"{model_name:<10} │{bar:<{width}} {avg_sp:.2f}x")
        
        lines.append("=" * (width + 20))
        return "\n".join(lines)
    
    @staticmethod
    def generate_identification_chart(
        precision: float,
        recall: float,
        f1_score: float,
        accuracy: float
    ) -> str:
        """
        生成关键层识别准确率图表
        
        Args:
            precision: 精确率
            recall: 召回率
            f1_score: F1分数
            accuracy: 准确率
        
        Returns:
            chart: ASCII图表
        """
        def bar(value: float, width: int = 40) -> str:
            filled = int(value * width)
            return "█" * filled + "░" * (width - filled)
        
        lines = []
        lines.append("关键层识别准确率")
        lines.append("─" * 50)
        lines.append(f"  Precision  │{bar(precision)} {precision:.2%}")
        lines.append(f"  Recall     │{bar(recall)} {recall:.2%}")
        lines.append(f"  F1 Score   │{bar(f1_score)} {f1_score:.2%}")
        lines.append(f"  Accuracy   │{bar(accuracy)} {accuracy:.2%}")
        lines.append("─" * 50)
        
        return "\n".join(lines)
    
    @staticmethod
    def generate_comparison_table(
        headers: List[str],
        rows: List[List[Any]],
        title: str = ""
    ) -> str:
        """
        生成对比表格
        
        Args:
            headers: 列标题
            rows: 数据行
            title: 表格标题
        
        Returns:
            table: 表格字符串
        """
        if not headers or not rows:
            return "无数据"
        
        # 计算列宽
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(cell)))
        
        col_widths = [min(w + 2, 30) for w in col_widths]
        total_width = sum(col_widths) + len(headers) + 1
        
        lines = []
        if title:
            lines.append(title)
            lines.append("=" * total_width)
        
        # 表头
        header_str = "│"
        for i, h in enumerate(headers):
            header_str += f" {h:<{col_widths[i]-1}}│"
        lines.append(header_str)
        lines.append("─" * total_width)
        
        # 数据行
        for row in rows:
            row_str = "│"
            for i, cell in enumerate(row):
                row_str += f" {str(cell):<{col_widths[i]-1}}│"
            lines.append(row_str)
        
        lines.append("=" * total_width)
        
        return "\n".join(lines)
    
    @staticmethod
    def generate_memory_chart(
        memory_data: Dict[str, Tuple[float, float]],
        width: int = 50
    ) -> str:
        """
        生成内存使用对比图
        
        Args:
            memory_data: {模型名: (全网络内存, 简化内存)}
            width: 条图宽度
        
        Returns:
            chart: ASCII图表
        """
        lines = []
        lines.append("内存使用对比 (MB)")
        lines.append("=" * (width + 20))
        
        max_mem = max(
            max(full, simple) for full, simple in memory_data.values()
        ) if memory_data else 1.0
        
        for model_name, (full_mem, simple_mem) in memory_data.items():
            full_bar = int(full_mem / max_mem * width)
            simple_bar = int(simple_mem / max_mem * width)
            
            reduction = (1 - simple_mem / (full_mem + 1e-10)) * 100
            
            lines.append(f"{model_name:<10}")
            lines.append(f"  全网络: {'█' * full_bar:<{width}} {full_mem:.1f}MB")
            lines.append(f"  简化:   {'█' * simple_bar:<{width}} {simple_mem:.1f}MB")
            lines.append(f"  减少:   {reduction:.1f}%")
            lines.append("")
        
        lines.append("=" * (width + 20))
        return "\n".join(lines)
