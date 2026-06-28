import sys
sys.path.insert(0, '.')

# 测试所有导入
print("Testing imports...")

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

print("All imports OK!")

# 测试基本功能
import torch
import numpy as np

# 1. 测试理论框架
print("\n1. Testing TheoryFramework...")
config = get_default_config()
framework = TheoryFramework()
bounds = framework.verify_theoretical_bounds(6, 0.3, 0.9)
print(f"   Theoretical bounds: {bounds}")

# 2. 测试KPI计算
print("\n2. Testing KPICalculator...")
kpi = KPICalculator()
print(f"   Speedup ratio: {kpi.compute_speedup_ratio(10, 1):.2f}x")
print(f"   Memory reduction: {kpi.compute_memory_reduction(100, 50):.2%}")

# 3. 测试模型创建
print("\n3. Testing model creation...")
model = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(784, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 10)
)
model.eval()
print(f"   Model created. Parameters: {sum(p.numel() for p in model.parameters())}")

# 4. 测试关键层选择器
print("\n4. Testing CriticalLayerSelector...")
selector = CriticalLayerSelector(model, verbose=False)
metrics = selector.compute_combined_scores(
    {0: 0.8, 1: 0.3},
    {0: 0.7, 1: 0.2},
    {0: 0.6, 1: 0.1}
)
critical = selector.select_critical_layers(metrics, method='top_k', threshold=0.5)
print(f"   Critical layers: {critical}")

# 5. 测试验证器
print("\n5. Testing CriticalLayerVerifier...")
verifier = CriticalLayerVerifier(model, verbose=False)
input_tensor = torch.randn(1, 1, 28, 28)
result = verifier.verify_sample(input_tensor, 0.02, 0, 10, critical_layers=[0], mode=VerificationMode.HYBRID)
print(f"   Verification result: safe={result.is_safe}, time={result.verification_time:.4f}s, memory={result.memory_usage_mb:.2f}MB")

# 6. 测试结果整合
print("\n6. Testing ResultIntegrator...")
monitor = CompletenessMonitor()
lb1, ub1 = torch.tensor([[5.0, 3.0, 2.0]]), torch.tensor([[7.0, 5.0, 4.0]])
lb2, ub2 = torch.tensor([[4.5, 3.5, 2.5]]), torch.tensor([[7.5, 4.5, 3.5]])
completeness = monitor.evaluate_completeness((lb1, ub1), (lb2, ub2), True, True)
print(f"   Completeness score: {completeness:.4f}")
print(f"   Completeness stats: {monitor.get_completeness_trend()}")

# 7. 测试可视化
print("\n7. Testing VisualizationTools...")
chart = VisualizationTools.generate_completeness_chart([0.9, 0.85, 0.95, 0.8, 0.92])
print(chart[:200] + "...")

print("\n" + "="*50)
print("ALL TESTS PASSED!")
print("="*50)
