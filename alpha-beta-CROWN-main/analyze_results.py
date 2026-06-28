import pandas as pd

df = pd.read_csv("my_experiments/verification_results.csv")

# 按模型和 epsilon 分组统计
summary = df.groupby(['model', 'epsilon']).agg(
    safe_count=('result', lambda x: (x == 'safe').sum()),
    unsafe_count=('result', lambda x: (x == 'unsafe').sum()),
    unknown_count=('result', lambda x: (x == 'unknown').sum()),
    timeout_count=('result', lambda x: (x == 'timeout').sum()),
    avg_time=('time_seconds', lambda x: x[x > 0].mean())
).reset_index()

print(summary)