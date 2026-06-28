"""
基于关键层的简化验证方法 - 主管道脚本

完整流程:
1. 加载数据和模型
2. 敏感性分析
3. 神经元活跃度统计
4. 关键层识别
5. 简化验证
6. 结果整合与评估
7. 生成报告

运行方式:
    python scripts/run_pipeline.py --model fc2 --epsilon 0.02 --samples 10
"""
import os
import sys
import argparse
import time
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from critical_layer_verification import (
    get_default_config, CriticalLayerConfig,
    TheoryFramework, CompletenessEvaluator, KPICalculator,
    SensitivityAnalyzer, NeuronActivityAnalyzer,
    CriticalLayerSelector, CriticalNeuronIdentifier, CriticalNeuronStore,
    CriticalLayerVerifier, VerificationMode,
    ResultIntegrator, InferenceStrategy, CompletenessMonitor,
    ComparisonExperiment, DataAnalyzer, ParameterOptimizer,
    ModelUtils, VisualizationTools
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="基于关键层的简化验证方法 - 主管道"
    )
    parser.add_argument('--model', type=str, default='fc2',
                        choices=['fc2', 'fc4', 'fc6', 'all'],
                        help='模型选择')
    parser.add_argument('--epsilon', type=float, default=0.02,
                        help='L∞扰动半径')
    parser.add_argument('--samples', type=int, default=10,
                        help='验证样本数')
    parser.add_argument('--method', type=str, default='top_k',
                        choices=['top_k', 'threshold', 'auto'],
                        help='关键层选择方法')
    parser.add_argument('--threshold', type=float, default=0.3,
                        help='选择阈值')
    parser.add_argument('--output', type=str, default='my_experiments',
                        help='输出目录')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='详细输出')
    parser.add_argument('--save_report', action='store_true', default=True,
                        help='保存实验报告')
    return parser.parse_args()


def create_model(num_layers=2, hidden_size=256):
    """创建MNIST FC模型"""
    layers = []
    layers.append(nn.Flatten())
    layers.append(nn.Linear(28*28, hidden_size))
    layers.append(nn.ReLU())
    for _ in range(num_layers - 1):
        layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(hidden_size, 10))
    return nn.Sequential(*layers)


def load_model(model_name: str, device: torch.device):
    """加载预训练模型"""
    model_files = {
        'fc2': ('mnist_fc2_256.pt', 2),
        'fc4': ('mnist_fc4_256.pt', 4),
        'fc6': ('mnist_fc6_256.pt', 6),
    }
    
    if model_name not in model_files:
        raise ValueError(f"未知模型: {model_name}")
    
    filename, num_layers = model_files[model_name]
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), filename)
    
    model = create_model(num_layers, 256).to(device)
    
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"  模型已加载: {model_path}")
    else:
        print(f"  警告: 未找到预训练模型 {model_path}, 使用随机初始化")
    
    model.eval()
    return model


def load_mnist_data(num_samples=10):
    """加载MNIST数据"""
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'my_experiments'
    )
    
    images_path = os.path.join(data_path, 'images.npy')
    labels_path = os.path.join(data_path, 'labels.npy')
    
    if os.path.exists(images_path) and os.path.exists(labels_path):
        images = np.load(images_path)
        labels = np.load(labels_path)
        print(f"  数据已加载: {images.shape}, {labels.shape}")
    else:
        print("  正在从MNIST数据集重新加载...")
        import torchvision
        import torchvision.transforms as transforms
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        testset = torchvision.datasets.MNIST(
            root=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'),
            train=False, download=True, transform=transform
        )
        
        images = []
        labels = []
        for i in range(num_samples):
            img, label = testset[i]
            images.append(img.numpy())
            labels.append(label)
        
        images = np.array(images)
        labels = np.array(labels)
        
        os.makedirs(data_path, exist_ok=True)
        np.save(images_path, images)
        np.save(labels_path, labels)
        print(f"  数据已保存: {images.shape}")
    
    return images[:num_samples], labels[:num_samples]


def step1_theoretical_framework(config: CriticalLayerConfig, verbose: bool = True):
    """步骤1: 理论框架构建"""
    if verbose:
        print("\n" + "="*60)
        print("步骤1: 理论框架构建")
        print("="*60)
    
    framework = TheoryFramework()
    
    # 验证理论边界
    theoretical_bounds = framework.verify_theoretical_bounds(
        num_layers=10,
        critical_ratio=config.selection_threshold,
        computation_reduction=0.9
    )
    
    if verbose:
        print(f"\n理论边界验证:")
        print(f"  关键层占比: {config.selection_threshold:.0%}")
        print(f"  理论最大加速比: {theoretical_bounds['max_speedup']:.2f}x")
        print(f"  完备性损失上界: {theoretical_bounds['completeness_loss_upper_bound']:.4f}")
        print(f"  信息保留率下界: {theoretical_bounds['info_preservation_lower_bound']:.4f}")
    
    return framework


def step2_sensitivity_analysis(
    model: nn.Module,
    images: np.ndarray,
    labels: np.ndarray,
    config: CriticalLayerConfig,
    device: torch.device,
    verbose: bool = True
):
    """步骤2: 敏感性分析"""
    if verbose:
        print("\n" + "="*60)
        print("步骤2: 敏感性分析")
        print("="*60)
    
    analyzer = SensitivityAnalyzer(model, device, verbose=verbose)
    
    input_tensor = torch.from_numpy(images[:1]).float().to(device)
    
    # 单级别敏感性分析
    if verbose:
        print(f"\n执行敏感性分析 (ε={config.sensitivity_epsilon}):")
    sensitivity = analyzer.analyze_all_layers(
        input_tensor, config.sensitivity_epsilon
    )
    
    # 层级排序
    ranked = analyzer.rank_layers_by_sensitivity(sensitivity)
    
    if verbose:
        print(f"\n层级敏感性排序:")
        for layer_idx, score in ranked:
            print(f"  层 {layer_idx}: 综合敏感性 = {score:.4f}")
    
    return sensitivity, ranked


def step3_neuron_activity_analysis(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    verbose: bool = True
):
    """步骤3: 神经元活跃度统计"""
    if verbose:
        print("\n" + "="*60)
        print("步骤3: 神经元活跃度分析")
        print("="*60)
    
    analyzer = NeuronActivityAnalyzer(model, device, verbose=verbose)
    
    # 收集激活统计
    stats = analyzer.collect_activation_stats(dataloader, num_batches=5)
    
    # 计算激活频率
    freq = analyzer.compute_activation_frequency(stats)
    
    # 计算重要性得分
    importance = analyzer.compute_neuron_importance_scores(stats)
    
    # 打印摘要
    if verbose:
        analyzer.print_activation_summary(freq, importance)
    
    return analyzer, stats, freq, importance


def step4_critical_layer_identification(
    model: nn.Module,
    sensitivity: dict,
    device: torch.device,
    config: CriticalLayerConfig,
    verbose: bool = True
):
    """步骤4: 关键层识别"""
    if verbose:
        print("\n" + "="*60)
        print("步骤4: 关键层识别")
        print("="*60)
    
    selector = CriticalLayerSelector(
        model, device,
        sc_weight=config.weight_sensitivity,
        ac_weight=config.weight_activation,
        epp_weight=config.weight_error_prop,
        ni_weight=config.weight_neuron_importance,
        ur_weight=config.weight_unstable,
        verbose=verbose
    )
    
    # 构建模拟指标
    sensitivity_dict = {}
    contribution_dict = {}
    epp_dict = {}
    
    for layer_idx, metrics in sensitivity.items():
        if "error" in metrics:
            continue
        sensitivity_dict[layer_idx] = metrics.get("sensitivity", 0.5)
        contribution_dict[layer_idx] = metrics.get("confidence_change", 0.3)
        epp_dict[layer_idx] = 1.0 - metrics.get("stability", 0.5)
    
    # 计算综合得分
    metrics_list = selector.compute_combined_scores(
        sensitivity_dict, contribution_dict, epp_dict
    )
    
    # 选择关键层
    critical_layers = selector.select_critical_layers(
        metrics_list,
        method=config.selection_method,
        threshold=config.selection_threshold,
        min_layers=config.min_critical_layers,
        max_layers=config.max_critical_layers
    )
    
    # 计算估计加速比
    estimation = selector.estimate_computation_reduction(
        total_layers=len(metrics_list),
        critical_layers=critical_layers
    )
    
    if verbose:
        print(f"\n估计加速比: {estimation['estimated_speedup']:.2f}x")
        print(f"计算量减少: {estimation['reduction_ratio']:.2%}")
    
    return selector, metrics_list, critical_layers, estimation


def step5_simplified_verification(
    model: nn.Module,
    images: np.ndarray,
    labels: np.ndarray,
    epsilon: float,
    critical_layers: list,
    device: torch.device,
    verbose: bool = True
):
    """步骤5: 简化验证"""
    if verbose:
        print("\n" + "="*60)
        print("步骤5: 简化验证")
        print("="*60)
    
    verifier = CriticalLayerVerifier(model, device, verbose=verbose)
    
    # 全网络验证
    if verbose:
        print("\n运行全网络验证...")
    full_results = verifier.verify_batch(
        images, labels, epsilon, num_classes=10,
        critical_layers=None, mode=VerificationMode.FULL,
        verbose=verbose
    )
    
    # 简化验证
    if verbose:
        print(f"\n运行简化验证 (关键层: {critical_layers})...")
    simplified_results = verifier.verify_batch(
        images, labels, epsilon, num_classes=10,
        critical_layers=critical_layers, mode=VerificationMode.HYBRID,
        verbose=verbose
    )
    
    return verifier, full_results, simplified_results


def step6_result_integration(
    full_results,
    simplified_results,
    critical_layers,
    verbose: bool = True
):
    """步骤6: 结果整合与评估"""
    if verbose:
        print("\n" + "="*60)
        print("步骤6: 结果整合与评估")
        print("="*60)
    
    monitor = CompletenessMonitor()
    integrator = None
    
    # 计算完备性
    total_completeness = 0
    for fr, sr in zip(full_results, simplified_results):
        # 模拟评估
        completeness = monitor.evaluate_completeness(
            (torch.tensor([fr.verified_lower_bound]), torch.tensor([fr.verified_upper_bound])),
            (torch.tensor([sr.verified_lower_bound]), torch.tensor([sr.verified_upper_bound])),
            fr.is_safe, sr.is_safe
        )
        total_completeness += completeness
    
    avg_completeness = total_completeness / max(len(full_results), 1)
    
    # 计算KPI
    kpi = KPICalculator()
    
    speedups = []
    for fr, sr in zip(full_results, simplified_results):
        sp = kpi.compute_speedup_ratio(
            fr.verification_time, sr.verification_time
        )
        speedups.append(sp)
    
    memory_reductions = []
    for fr, sr in zip(full_results, simplified_results):
        mr = kpi.compute_memory_reduction(
            fr.memory_usage_mb, sr.memory_usage_mb
        )
        memory_reductions.append(mr)
    
    if verbose:
        print(f"\n评估结果:")
        print(f"  平均完备性: {avg_completeness:.4f}")
        print(f"  完备性损失: {1 - avg_completeness:.4f}")
        print(f"  平均加速比: {np.mean(speedups):.2f}x")
        print(f"  最大加速比: {np.max(speedups):.2f}x")
        print(f"  平均内存降低: {np.mean(memory_reductions):.2%}")
        print(f"  内存降低(目标>=40%): {'[OK]' if np.mean(memory_reductions) >= 0.4 else '[FAIL]'}")
        
        # 结果一致性
        consistent = sum(
            1 for fr, sr in zip(full_results, simplified_results)
            if fr.is_safe == sr.is_safe
        )
        print(f"  结果一致率: {consistent}/{len(full_results)} ({consistent/len(full_results)*100:.1f}%)")
    
    return monitor, {
        'avg_completeness': avg_completeness,
        'avg_speedup': np.mean(speedups),
        'max_speedup': np.max(speedups),
        'avg_memory_reduction': np.mean(memory_reductions),
        'consistent_count': consistent,
        'total': len(full_results)
    }


def run_full_pipeline(args):
    """运行完整管道"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = get_default_config()
    
    # 配置覆盖
    config.selection_method = args.method
    config.selection_threshold = args.threshold
    
    print("="*70)
    print("基于关键层的简化验证方法 (Critical-Layer Based Simplified Verification)")
    print(f"  设备: {device}")
    print(f"  模型: {args.model}")
    print(f"  扰动半径: {args.epsilon}")
    print(f"  验证样本数: {args.samples}")
    print(f"  选择方法: {args.method} (阈值={args.threshold})")
    print("="*70)
    
    # 步骤1: 理论框架
    framework = step1_theoretical_framework(config, args.verbose)
    
    # 加载模型和数据
    print(f"\n{'='*60}")
    print("加载模型和数据")
    print(f"{'='*60}")
    
    models_to_test = ['fc2', 'fc4', 'fc6'] if args.model == 'all' else [args.model]
    
    all_results = {}
    
    for model_name in models_to_test:
        print(f"\n--- 处理模型: {model_name} ---")
        
        model = load_model(model_name, device)
        images, labels = load_mnist_data(args.samples)
        
        # 创建数据加载器用于神经元分析
        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(images).float(),
            torch.from_numpy(labels).long()
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=25, shuffle=False)
        
        # 步骤2: 敏感性分析
        sensitivity, ranked = step2_sensitivity_analysis(
            model, images, labels, config, device, args.verbose
        )
        
        # 步骤3: 神经元活跃度分析
        analyzer, stats, freq, importance = step3_neuron_activity_analysis(
            model, dataloader, device, args.verbose
        )
        
        # 步骤4: 关键层识别
        selector, metrics_list, critical_layers, estimation = step4_critical_layer_identification(
            model, sensitivity, device, config, args.verbose
        )
        
        # 步骤5: 简化验证
        verifier, full_results, simplified_results = step5_simplified_verification(
            model, images, labels, args.epsilon, critical_layers, device, args.verbose
        )
        
        # 步骤6: 结果整合
        monitor, evaluation = step6_result_integration(
            full_results, simplified_results, critical_layers, args.verbose
        )
        
        # 保存结果
        all_results[model_name] = {
            'model_name': model_name,
            'critical_layers': critical_layers,
            'estimation': estimation,
            'evaluation': evaluation,
            'num_full_safe': sum(1 for r in full_results if r.is_safe),
            'num_simple_safe': sum(1 for r in simplified_results if r.is_safe),
        }
        
        # 生成可视化
        if args.verbose:
            completeness_scores = monitor.history['completeness_scores']
            if completeness_scores:
                chart = VisualizationTools.generate_completeness_chart(completeness_scores)
                print(f"\n{chart}")
    
    # 总结报告
    print("\n" + "="*70)
    print("最终总结报告")
    print("="*70)
    
    headers = ['模型', '总层数', '关键层', '加速比', '完备性', '内存降低', '结果一致']
    rows = []
    
    for model_name, result in all_results.items():
        rows.append([
            model_name,
            result.get('total_layers', '?'),
            str(result['critical_layers']),
            f"{result['evaluation']['avg_speedup']:.2f}x",
            f"{result['evaluation']['avg_completeness']:.2%}",
            f"{result['evaluation']['avg_memory_reduction']:.1%}",
            f"{result['evaluation']['consistent_count']}/{result['evaluation']['total']}"
        ])
    
    table = VisualizationTools.generate_comparison_table(headers, rows, "模型性能对比")
    print(f"\n{table}")
    
    # 检查是否达到目标
    print(f"\n目标达成检查:")
    print(f"  {'指标':<30} {'目标':<15} {'结果':<15} {'状态':<10}")
    print(f"  {'-'*70}")
    
    for model_name, result in all_results.items():
        avg_sp = result['evaluation']['avg_speedup']
        avg_comp = result['evaluation']['avg_completeness']
        avg_mem = result['evaluation']['avg_memory_reduction']
        
        sp_ok = avg_sp >= 10.0
        comp_ok = (1 - avg_comp) <= 0.05
        mem_ok = avg_mem >= 0.40
        
        print(f"  {model_name}:")
        print(f"    {'验证加速比(>=10x)':<30} {'10.0x':<15} {f'{avg_sp:.2f}x':<15} {'[OK]' if sp_ok else '[FAIL]'}")
        print(f"    {'完备性损失(<=5%)':<30} {'5%':<15} {f'{(1-avg_comp)*100:.2f}%':<15} {'[OK]' if comp_ok else '[FAIL]'}")
        print(f"    {'内存降低(>=40%)':<30} {'40%':<15} {f'{avg_mem*100:.1f}%':<15} {'[OK]' if mem_ok else '[FAIL]'}")    
    # 保存报告
    if args.save_report:
        report_path = os.path.join(args.output, f'pipeline_report_{args.model}.txt')
        os.makedirs(args.output, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("基于关键层的简化验证方法 - 管道执行报告\n")
            f.write(f"{'='*70}\n")
            f.write(f"执行时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"模型: {args.model}\n")
            f.write(f"扰动半径: {args.epsilon}\n")
            f.write(f"验证样本: {args.samples}\n\n")
            
            for model_name, result in all_results.items():
                f.write(f"\n--- {model_name} ---\n")
                f.write(f"关键层: {result['critical_layers']}\n")
                f.write(f"预估加速比: {result['estimation']['estimated_speedup']:.2f}x\n")
                f.write(f"实际加速比: {result['evaluation']['avg_speedup']:.2f}x\n")
                f.write(f"完备性: {result['evaluation']['avg_completeness']:.2%}\n")
                f.write(f"内存降低: {result['evaluation']['avg_memory_reduction']:.1%}\n")
                f.write(f"结果一致: {result['evaluation']['consistent_count']}/{result['evaluation']['total']}\n")
            
            f.write(f"\n{'='*70}\n")
            f.write("管道执行完毕\n")
        
        print(f"\n报告已保存至: {report_path}")
    
    return all_results


if __name__ == "__main__":
    args = parse_args()
    results = run_full_pipeline(args)
    print("\n管道执行完毕!")
