"""
auto_LiRPA 集成测试

验证 auto_LiRPA 框架是否能正常集成到关键层验证管线中。
"""
import sys
import os
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from critical_layer_verification.verification.auto_lirpa_integration import (
    AutoLiRPAVerifier, AutoLiRPAConfig, check_auto_lirpa_available
)

print("=" * 60)
print("auto_LiRPA 集成测试")
print("=" * 60)

# 1. 检查 auto_LiRPA 是否可用
print(f"\n[1] auto_LiRPA 可用性: {check_auto_lirpa_available()}")

# 2. 创建模型
print("\n[2] 创建测试模型 (MNIST FC-2)...")
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)
model.eval()
print(f"  参数数量: {sum(p.numel() for p in model.parameters())}")
print(f"  ReLU 层数: {sum(1 for m in model.modules() if isinstance(m, nn.ReLU))}")

# 3. 初始化 AutoLiRPAVerifier
print("\n[3] 初始化 AutoLiRPAVerifier...")
config = AutoLiRPAConfig(
    method='CROWN-IBP',
    use_alpha=True,
    use_beta=False,
    verbose=False,
    device='cpu'
)

try:
    verifier = AutoLiRPAVerifier(model, config, device=torch.device('cpu'))
    print(f"  BoundedModel 初始化: {'成功' if verifier._bounded_model is not None else '失败'}")
except Exception as e:
    print(f"  初始化失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 4. 测试边界计算
print("\n[4] 测试边界计算...")
input_tensor = torch.randn(1, 1, 28, 28)
epsilon = 0.02
input_lower = torch.clamp(input_tensor - epsilon, 0.0, 1.0)
input_upper = torch.clamp(input_tensor + epsilon, 0.0, 1.0)

# 4a. IBP 模式
print("\n  [4a] IBP 模式...")
try:
    ibp_lower, ibp_upper, ibp_time = verifier.compute_ibp_bounds(
        input_lower, input_upper
    )
    print(f"    IBP 形状: lower={ibp_lower.shape}, upper={ibp_upper.shape}")
    print(f"    IBP 时间: {ibp_time:.4f}s")
except Exception as e:
    print(f"    IBP 失败: {e}")

# 4b. CROWN 模式
print("\n  [4b] CROWN 模式...")
try:
    # 暂存配置，使用 CROWN
    original_method = config.method
    config.method = 'CROWN'
    verifier2 = AutoLiRPAVerifier(model, config, device=torch.device('cpu'))
    
    crown_lower, crown_upper, crown_time = verifier2.compute_bounds_with_crown(
        input_lower, input_upper,
        target_label=0,
        num_classes=10,
        critical_layers=None,
    )
    print(f"    CROWN 形状: lower={crown_lower.shape}, upper={crown_upper.shape}")
    print(f"    CROWN 时间: {crown_time:.4f}s")
    config.method = original_method
except Exception as e:
    print(f"    CROWN 失败: {e}")
    import traceback
    traceback.print_exc()

# 4c. alpha-CROWN 模式
print("\n  [4c] alpha-CROWN 模式...")
try:
    config.method = 'alpha-CROWN'
    config.use_alpha = True
    verifier3 = AutoLiRPAVerifier(model, config, device=torch.device('cpu'))
    
    alpha_lower, alpha_upper, alpha_time = verifier3.compute_bounds_with_crown(
        input_lower, input_upper,
        target_label=0,
        num_classes=10,
        critical_layers=[0, 1],  # 所有层都是关键层
    )
    print(f"    alpha-CROWN 形状: lower={alpha_lower.shape}, upper={alpha_upper.shape}")
    print(f"    alpha-CROWN 时间: {alpha_time:.4f}s")
    config.method = 'CROWN-IBP'
except Exception as e:
    print(f"    alpha-CROWN 失败: {e}")

# 4d. 混合模式 (关键层验证)
print("\n  [4d] 混合模式 (关键层=[0])...")
try:
    hybrid_lower, hybrid_upper, hybrid_info = verifier.compute_hybrid_bounds(
        input_lower, input_upper,
        target_label=0,
        num_classes=10,
        critical_layers=[0],  # 仅第0层是关键层
    )
    print(f"    混合模式 形状: lower={hybrid_lower.shape}, upper={hybrid_upper.shape}")
    print(f"    混合模式 时间: {hybrid_info.get('computation_time', 0):.4f}s")
    print(f"    使用方法: {hybrid_info.get('method_used', 'N/A')}")
    print(f"    安全判定: {hybrid_info.get('is_safe', 'N/A')}")
    print(f"    最小边距: {hybrid_info.get('min_margin', 'N/A')}")
except Exception as e:
    print(f"    混合模式 失败: {e}")

# 5. 完整验证流程测试
print("\n[5] 完整验证测试 (epsilon=0.02, 5个样本)...")
try:
    images = torch.randn(5, 1, 28, 28)
    labels = torch.randint(0, 10, (5,))
    
    for i in range(5):
        result = verifier.verify_sample_with_lirpa(
            images[i:i+1], 0.02, int(labels[i]),
            num_classes=10, critical_layers=[0]
        )
        print(f"  样本 {i}: safe={result['is_safe']}, "
              f"margin={result['min_margin']:.4f}, "
              f"time={result['verification_time']:.4f}s")
except Exception as e:
    print(f"  验证失败: {e}")

# 6. 内存降低估计
print(f"\n[6] 内存降低估计:")
print(f"  关键层 [0]: {verifier.estimate_memory_reduction([0]):.1%}")
print(f"  关键层 [0,1]: {verifier.estimate_memory_reduction([0,1]):.1%}")

print("\n" + "=" * 60)
print("测试完成!")
print("=" * 60)
