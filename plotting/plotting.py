import os
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ================= 设置学术绘图样式 =================
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 14,             # 基础字体全局调大
    'axes.titlesize': 16,        # 子图标题调大
    'axes.labelsize': 14,        # X/Y轴标签调大
    'xtick.labelsize': 15,       # X轴刻度数字调大
    'ytick.labelsize': 15,       # Y轴刻度数字调大
    'legend.fontsize': 15,       # 图例字体适度调大 (保证5条线能放下)
    'figure.figsize': (12, 5.5)  # 画布整体稍微放大一点，防止大字体显得拥挤
})

def get_tensorboard_data(log_dir, tag):
    """读取指定文件夹下所有的 event 文件并提取 tag 数据"""
    if not os.path.exists(log_dir):
        print(f"❌ 错误: 路径不存在 {log_dir}")
        return [], []

    ea = EventAccumulator(log_dir, size_guidance={'scalars': 0})
    ea.Reload()
    
    if tag not in ea.Tags()['scalars']:
        print(f"⚠️ 警告: 找不到 '{tag}' 在 {log_dir} 中。可用 tags: {ea.Tags()['scalars']}")
        return [], []
        
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values

# ================= 替换为你的真实路径 =================
base_dir = r"plotting/"

dir_job1 = os.path.join(base_dir, r"log - CNN + BiLTSM Architecture")
dir_job2 = os.path.join(base_dir, r"log - Google Inception Network")
dir_job3 = os.path.join(base_dir, r"log - Temporal Convolutional Networks")
dir_job4 = os.path.join(base_dir, r"log- Conformer Architecture")


# ================= 模型配置 (图例标签, 路径, 颜色, 线型, 线宽) =================
model_configs = [
    ("CNN + BiLTSM Architecture", dir_job1, "#1f77b4", ":",  2),     # 蓝色 虚线 
    ("Google Inception Network", dir_job2, "#8c564b", "--", 2),     # 棕色 破折线 
    ("Temporal Convolutional Networks", dir_job3, "#ff7f0e", "-.", 2),     # 橙色 点划线 
    # ("Model 4 (k=31, N=6, D=128)", dir_job3, "#2ca02c", "-",  2.5),   # 绿色 实线加粗 (最佳)
    ("Conformer Architecture", dir_job4, "#d62728", "-",  2)      # 红色 实线 (停滞)
]

# ================= 创建 1x2 子图 =================
fig, (ax1, ax2) = plt.subplots(1, 2)

# 绘制左图: Training Loss
for label, path, color, linestyle, lw in model_configs:
    steps, loss_vals = get_tensorboard_data(path, "val/loss")
    if steps:
        ax1.plot(steps, loss_vals, label=label, color=color, linestyle=linestyle, linewidth=lw)

ax1.set_title("Training Loss", fontweight="bold")
ax1.set_xlabel("Training Steps")
# ax1.set_ylabel("Loss")
ax1.set_ylim(0, 7)  # 限制 Y 轴范围，突出有效对比区间
ax1.set_xlim(0,4800)

# 绘制右图: Testing CER
for label, path, color, linestyle, lw in model_configs:
    steps, cer_vals = get_tensorboard_data(path, "val/CER")
    if steps:
        ax2.plot(steps, cer_vals, label=label, color=color, linestyle=linestyle, linewidth=lw)

ax2.set_title("Validation CER", fontweight="bold")
ax2.set_xlabel("Training Steps")
# ax2.set_ylabel("Character Error Rate")
ax2.set_ylim(0, 130)  # 限制 Y 轴范围，过滤异常极端值
ax2.set_xlim(0,4800)


# 统一放置图例
ax1.legend(loc='upper right')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig("learning_dynamics.pdf", dpi=300, format="pdf")
print("✅ 图表已成功保存为 learning_dynamics.pdf")
plt.show()