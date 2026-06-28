# 基于关键层的简化验证方法

## Critical-Layer Based Simplified Verification for ReLU Neural Networks

基于 α,β-CROWN 框架，通过对 ReLU 神经网络中的关键层进行优先验证，大幅提升鲁棒性验证效率，同时严格控制完备性损失。

## 目录结构

```
critical_layer_verification/
├── __init__.py                         # 包入口，导出所有核心类
├── config/
│   ├── __init__.py
│   └── default_config.py               # 默认配置管理
├── theory/
│   ├── __init__.py
│   ├── theory_core.py                  # 理论框架核心
│   └── metrics.py                      # KPI定义与计算方法
├── sensitivity/
│   ├── __init__.py
│   ├── sensitivity_analyzer.py         # 敏感性分析
│   └── neuron_activity_analyzer.py     # 神经元活跃度统计
├── identification/
│   ├── __init__.py
│   ├── layer_selector.py               # 关键层筛选算法
│   └── neuron_identifier.py            # 关键神经元识别
├── verification/
│   ├── __init__.py
│   ├── critical_layer_verifier.py      # 简化验证引擎
│   └── bound_optimizer.py             # 边界优化器
├── integration/
│   ├── __init__.py
│   ├── result_integrator.py            # 结果整合与推断
│   └── completeness_monitor.py         # 完备性监控
├── experiment/
│   ├── __init__.py
│   ├── comparison_experiment.py        # 对比实验
│   ├── parameter_optimizer.py          # 参数优化
│   └── data_analyzer.py               # 数据分析
└── utils/
    ├── __init__.py
    ├── model_utils.py                  # 模型工具
    └── visualization.py               # 可视化工具

scripts/
└── run_pipeline.py                     # 主管道脚本
```

## 核心指标

### 1. 敏感性系数 (Sensitivity Coefficient, SC)

度量层输出扰动对最终验证边界的影响程度：

$$SC_l = \frac{||\Phi(x+\delta_l) - \Phi(x)||}{||\delta_l||}$$

其中 $\delta_l$ 是注入到第 $l$ 层输出的扰动。

### 2. 激活贡献度 (Activation Contribution, AC)

度量每个神经元对最终决策的贡献程度：

$$AC_i = \frac{|w_i \cdot ReLU(z_i)|}{\sum_j |w_j \cdot ReLU(z_j)|}$$

### 3. 错误传播概率 (Error Propagation Probability, EPP)

$$EPP_{l} = P(||\varepsilon_{l+1}|| > \tau \;|\; ||\varepsilon_l|| > \tau)$$

度量第 $l$ 层的边界估计误差传播到下一层的概率。

### 4. 层重要性综合得分 (LIS)

$$LIS_l = w_{SC} \cdot SC_l + w_{AC} \cdot AC_l + w_{EPP} \cdot EPP_l + w_{NI} \cdot NI_l + w_{UR} \cdot UR_l$$

### 5. 完备性损失

$$L_{complete} = \frac{||bounds_{full} - bounds_{simplified}||}{||bounds_{full}||}$$

## 工作流程

```
步骤1: 理论框架构建
  ├── 解析"抽象"思想核心原理
  ├── 定义关键层量化标准
  └── 建立完备性与速度的评估模型

步骤2: 敏感性分析
  ├── L∞范数扰动注入
  ├── 随机噪声注入
  ├── 分类置信度变化率
  └── 预测标签稳定性

步骤3: 神经元活跃度统计
  ├── 激活频率计算
  ├── 激活强度分布分析
  ├── 激活模式相关性
  └── 稳定/不稳定神经元识别

步骤4: 关键层识别
  ├── 多维度权重评估
  ├── 综合得分计算
  ├── top_k / 阈值 / 肘部法则选择
  └── 关键神经元子集提取

步骤5: 简化验证
  ├── 关键层: α-CROWN精确计算
  ├── 非关键层: IBP快速近似
  └── 自适应边界松弛

步骤6: 结果整合与评估
  ├── 保守/加权/自适应推断
  ├── 完备性监控
  └── KPI计算与报告生成
```

## 运行方式

### 基础运行

```bash
cd alpha-beta-CROWN-main
python scripts/run_pipeline.py --model fc2 --epsilon 0.02 --samples 10
```

### 参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `--model` | str | fc2 | 模型选择 (fc2/fc4/fc6/all) |
| `--epsilon` | float | 0.02 | L∞扰动半径 |
| `--samples` | int | 10 | 验证样本数 |
| `--method` | str | top_k | 关键层选择方法 |
| `--threshold` | float | 0.3 | 选择阈值 |
| `--output` | str | my_experiments | 输出目录 |

### 运行所有模型

```bash
python scripts/run_pipeline.py --model all --epsilon 0.02 --samples 25
```

### 使用不同选择策略

```bash
# 肘部法则自动选择
python scripts/run_pipeline.py --model fc4 --method auto --samples 10

# 阈值选择
python scripts/run_pipeline.py --model fc6 --method threshold --threshold 0.5 --samples 10
```

## 目标指标

| 指标 | 目标值 | 说明 |
|------|--------|------|
| 验证加速比 | ≥ 10x | 较全网络验证提升10倍以上 |
| 完备性损失 | ≤ 5% | 控制在5%以内 |
| 内存占用降低 | ≥ 40% | 减少内存使用 |
| 关键层识别准确率 | ≥ 90% | Precision/Recall/F1 |
| 验证结果一致率 | ≥ 95% | 与全网络验证结果一致 |

## 项目验证

### 验证模型

本方案在以下3类网络上进行验证：

1. **小规模CNN**: MNIST FC-2 (2层全连接, 256隐藏单元)
2. **中等规模网络**: MNIST FC-4 (4层全连接, 256隐藏单元)
3. **更深层网络**: MNIST FC-6 (6层全连接, 256隐藏单元)

### 实验结果评估

每次运行管道将输出:

1. **理论边界验证** - 确认方法在理论上的可行性
2. **敏感性分析结果** - 各层的敏感性排序
3. **神经元活跃度统计** - 激活频率、稳定性
4. **关键层识别** - 选中的关键层及其综合得分
5. **验证性能对比** - 全网络 vs 简化的时间和内存对比
6. **完备性评估** - 完备性损失和结果一致率
7. **目标达成检查** - 是否满足预定KPI目标

## 参考文献

1. Xu, K., et al. "Fast and Complete: Enabling Complete Neural Network Verification with Rapid and Massively Parallel Incomplete Verifiers." ICLR 2021.
2. Wang, S., et al. "Beta-CROWN: Efficient Bound Propagation with Beta Decomposition for Neural Network Verification." NeurIPS 2021.
3. Zhang, H., et al. "Efficient Neural Network Robustness Certification with General Activation Functions." NeurIPS 2018.
