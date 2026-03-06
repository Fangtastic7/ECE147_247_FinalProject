import os
import glob
import subprocess

# 1. 填入你昨晚跑出这16个文件夹的父目录路径
sweep_dir = r"logs\2026-03-02\01-58-11"

# 2. 自动搜索所有 job 文件夹下的 .ckpt 权重文件
# PyTorch Lightning 默认会保存类似 epoch=99-step=1000.ckpt 这样的文件
checkpoint_paths = glob.glob(os.path.join(sweep_dir, "job*", "checkpoints", "*.ckpt"))

if not checkpoint_paths:
    print("没有找到任何 .ckpt 文件，请检查路径是否正确！")
else:
    print(f"找到了 {len(checkpoint_paths)} 个模型权重，准备开始测试...")

# 3. 遍历每个权重文件，自动执行测试命令
for i, ckpt in enumerate(checkpoint_paths):
    print("\n" + "="*50)
    print(f"正在测试第 {i+1}/{len(checkpoint_paths)} 个模型: \n{ckpt}")
    print("="*50)
    
    # 构造命令行指令
    # 覆盖 train 为 False，并传入当前的 checkpoint 路径
    cmd = [
        "python", "-m", "emg2qwerty.train",
        "train=False",
        f"checkpoint={ckpt}"
    ]
    
    # 执行命令
    subprocess.run(cmd)

print("\n所有模型测试完成！")