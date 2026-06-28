import os
import subprocess
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import time
import yaml
import pandas as pd
from tqdm import tqdm

# ========== 配置参数 ==========
MODELS = {
    "fc2": "mnist_fc2_256.pt",
    "fc4": "mnist_fc4_256.pt",
    "fc6": "mnist_fc6_256.pt",
}
EPSILONS = [0.02, 0.05]
NUM_SAMPLES = 25          # 验证前25张测试图片
TIMEOUT = 1800             # 单个验证超时时间（秒），30分钟
DEVICE = "cpu"             # 如果GPU可用且想用，改为"cuda"，但需确保PyTorch CUDA可用
USE_GPU = False            # α,β-CROWN 内部是否使用GPU，若GPU可用设为True

# 数据预处理（与训练时一致）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载测试集
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 提取前NUM_SAMPLES张图片和标签
images = []
labels = []
for i in range(NUM_SAMPLES):
    img, label = testset[i]
    images.append(img.numpy())
    labels.append(label)
images = np.array(images)   # shape: (NUM_SAMPLES, 1, 28, 28)
labels = np.array(labels)

# 保存为numpy文件，供配置文件引用
os.makedirs("my_experiments", exist_ok=True)
np.save("my_experiments/images.npy", images)
np.save("my_experiments/labels.npy", labels)

# ========== 辅助函数 ==========
def run_verification(model_path, epsilon, image_index, true_label):
    """对单个样本运行α,β-CROWN验证，返回 (result, elapsed_time)"""
    # 创建临时配置文件
    config = {
        "model": {
            "path": model_path,
            "type": "pytorch",
            "input_shape": [1, 28, 28]
        },
        "spec": {
            "type": "robustness",
            "epsilon": epsilon,
            "norm": "l_inf",
            "data": "my_experiments/images.npy",
            "labels": "my_experiments/labels.npy",
            "num_labels": 10,
            "image_index": image_index
        },
        "solver": {
            "batch_size": 1,
            "timeout": TIMEOUT,
            "use_gpu": USE_GPU
        }
    }
    # 生成唯一临时文件名
    tmp_config = f"my_experiments/temp_{os.path.basename(model_path)}_{epsilon}_{image_index}.yaml"
    with open(tmp_config, 'w') as f:
        yaml.dump(config, f)

    # 调用α,β-CROWN
    start = time.time()
    cmd = ["python", "complete_verifier/abcrown.py", "--config", tmp_config]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT, cwd=os.getcwd())
        elapsed = time.time() - start
        output = result.stdout + result.stderr
        if "Verified (unsat)" in output:
            status = "safe"
        elif "Counterexample found" in output:
            status = "unsafe"
        else:
            status = "unknown"
    except subprocess.TimeoutExpired:
        elapsed = TIMEOUT
        status = "timeout"
    except Exception as e:
        elapsed = -1
        status = f"error: {e}"
    finally:
        # 删除临时配置文件
        if os.path.exists(tmp_config):
            os.remove(tmp_config)
    return status, elapsed

# ========== 主循环 ==========
results = []
total_tests = len(MODELS) * len(EPSILONS) * NUM_SAMPLES
pbar = tqdm(total=total_tests, desc="验证进度")

for model_name, model_file in MODELS.items():
    model_path = os.path.join(os.getcwd(), model_file)  # 绝对路径
    for eps in EPSILONS:
        for idx in range(NUM_SAMPLES):
            true_label = labels[idx]
            status, elapsed = run_verification(model_path, eps, idx, true_label)
            results.append({
                "model": model_name,
                "epsilon": eps,
                "index": idx,
                "true_label": true_label,
                "result": status,
                "time_seconds": elapsed
            })
            pbar.update(1)

pbar.close()

# 保存结果
df = pd.DataFrame(results)
df.to_csv("my_experiments/verification_results.csv", index=False)
print("\n所有验证完成！结果已保存至 my_experiments/verification_results.csv")