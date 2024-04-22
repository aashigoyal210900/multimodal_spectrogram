import matplotlib.pyplot as plt
import numpy as np

# Define the metrics and model groups
metrics_labels = ['Train Accuracy', 'Train Loss', 'Test Accuracy', 'Test Loss']
group_labels = ['2D CNN', '3D CNN', 'Transformer']

# Specific encoders for each type
encoders = [
    ['ResNet18', 'GoogLeNet', 'VGG16'],
    ['Simple3DCNN', 'I3D'],
    ['ViT', 'TimeSformer']
]

# Simulated data for each encoder, each with: [Train Accuracy, Train Loss, Test Accuracy, Test Loss]
data = {
    'ResNet18': [0.7, 0.8, 0.6, 0.9],
    'GoogLeNet': [0.72, 0.85, 0.65, 0.88],
    'VGG16': [0.75, 0.9, 0.68, 0.93],
    'Simple3DCNN': [0.519, 1.213, 0.514, 1.294],
    'I3D': [0.623, 0.991, 0.605, 1.022],
    'ViT': [0.1634, 1.7931, 0.1738, 1.7908],
    'TimeSformer': [0.2, 1.7, 0.19, 1.7]  # Placeholder values
}

# colors = ['#FF9999', '#99CCFF', '#99FF99', '#FFCC99', '#FF99CC', '#CCFF99', '#99FFFF']  # Colors for each encoder
colors = ['red', 'green', 'blue', 'cyan', 'orange', 'yellow', 'purple']  # Colors for each encoder
# hatches = ['/', '\\', 'o', 'x']  # Different hatches for each metric

# Plotting
fig, ax = plt.subplots(figsize=(14, 8))

metric_group_width = 0.8
bar_width = metric_group_width / (len(encoders[0]) + len(encoders[1]) + len(encoders[2]) + 1)
x = np.arange(len(metrics_labels)) * (len(encoders[0]) + len(encoders[1]) + len(encoders[2]) + 3) * bar_width
alpha = {0: 0.6, 1: 0.8, 2: 0.8}
# Generate the bars
for i, metric in enumerate(metrics_labels):
    offset = 0  # reset offset for each metric group
    col_count = 0
    for j, group in enumerate(group_labels):
        off = alpha[j]
        ax.text(x[i] + (j + off) * (len(encoders[j]) + 1) * bar_width - 0.5 * len(group_labels) * bar_width, -0.05,
                group, ha='center', va='center')
        for k, encoder in enumerate(encoders[j]):
            metric_index = i
            val = data[encoder][metric_index]
            # rect = ax.bar(x[i] + offset, val, bar_width, label=metric if j == 0 and k == 0 else "",
            #               color=colors[j * len(encoders[j]) + k], edgecolor='black', hatch=hatches[i])
            # rect = ax.bar(x[i] + offset, val, bar_width, label=metric if j == 0 and k == 0 else "",
            #               color=colors[j * len(encoders[j]) + k], edgecolor='black')
            rect = ax.bar(x[i] + offset, val, bar_width, label=metric if j == 0 and k == 0 else "",
                          color=colors[col_count], edgecolor='black')
            ax.text(rect[0].get_x() + rect[0].get_width() / 2, 0,
                    encoder, ha='center', va='bottom', rotation=90, bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2')) # , weight='bold'
            offset += bar_width
            col_count += 1
        offset += bar_width * 0.5  # Add spacer after each group
    offset += bar_width * 1.5  # Extra space between metrics

# Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_xlabel('Metrics')
ax.set_ylabel('Values', fontsize=12)
ax.set_title('Unimodal audio pipeline for different encoders')
ax.set_xticks(x + (metric_group_width + 2.5 * bar_width) / 2)
ax.tick_params(axis='x', which='both', bottom=False)  # Remove x-tick markers
ax.set_xticklabels(metrics_labels, y=-0.05, fontsize=12)
# ax.legend([plt.Rectangle((0,0),1,1, color='gray', hatch=hatch) for hatch in hatches], metrics_labels, title='Metric')
# Add grid lines parallel to x-axis
ax.grid(axis='y', linestyle='--', alpha=0.7, which='both')

# Add separator lines between each metric group
for i in range(len(metrics_labels) - 1):
    ax.axvline(x[i] + (len(encoders[2]) + 6.5) * bar_width, color='black', linestyle='-', linewidth=2)


ax.axhline(1.0, color='red', linestyle='-', linewidth=2)

fig.tight_layout()
plt.show()
