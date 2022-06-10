from turtle import color
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize = fontsize, fontproperties = "Times New Roman")

colors = ["#CC3333", "#009999", "#66CC99", "#FFCC00"]

# x_labels = ["Test1", "Test2", "Test3", "Test4"]
# values1 = [81.19, 94.88, 98.47, 98.50]
# values2 = [96.69, 92.88, 99.66, 98.44]
# values3 = [96.19, 92.64, 99.47, 98.42]
# values4 = [14.07, 14.33, 14.07, 14.42]

# x_labels = ["Faster RCNN", "SSD", "RetinaNet", "CenterNet", "YOLOv5", "Ours"]
# p = [79.46, 81.52, 82.60, 86.80, 89.89, 92.07]
# r = [82.25, 82.26, 85.72, 87.55, 91.96, 93.40]
# ap = [78.80, 80.30, 82.36, 84.51, 90.33, 96.25]
# fps = [25.25, 24.68, 25.30, 23.46, 26.98, 13.50]
# size = [108.9, 100.3, 145.0, 77.2, 48.0, 27.6]

# x_labels = ["Test1", "Test2", "Test3", "Test4"]
# values1 = [98.31, 85.47, 100, 34.05]
# values2 = [98.31, 90.91, 100, 37.84]
# values3 = [97.22, 90.91, 100, 37.84]
# values4 = [6.73, 6.63, 6.86, 8.01]

# x_labels = ["1:0", "1:1", "1:2", "1:3"]
# values1 = [73.22, 77.33, 80.77]
# values2 = [78.92, 79.36, 80.77, 79.27]
# values3 = [80.27, 80.77, 79.89]

# x_labels = ["640x416", "720p", "1080p", "4K"]
# values1 = [78.6, 83.6, 83.9, 84.6]
# values2 = [145, 108, 99, 92]
# values3 = [0.270, 0.258, 0.256, 0.255]
# values4 = [27.75, 6.68, 6.32, 6.14]

x_labels = ["JDE", "FairMOT", "YOLOv5-DeepSORT", "Ours"]
values1 = [75.8, 77.4, 82.0, 84.6]
values2 = [171, 168, 100, 92]
values3 = [22.32, 27.74, 16.83, 6.14]
values4 = [557.7, 247.2, 166.0, 42.1]

x = np.arange(len(x_labels))  # the label locations
y_labels = list(range(0, 101, 20))
width = 0.3  # the width of the bars
fontsize = 14
markersize = 16
zorder = 90

fig, ax = plt.subplots(dpi = 120, figsize=(12,6))
# fig, ax = plt.subplots(dpi = 120, figsize=(4,6))
fig.patch.set_alpha(0.0)

ax.grid(True, which="major", axis="y")

# rects1 = ax.bar(x, values4, width, label='FPS', align="center", color=colors[3], zorder = zorder)

# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('FPS', fontsize = fontsize, fontproperties = "Times New Roman", color="white")
# ax.set_yticks(y_labels)
# ax.set_yticklabels(y_labels, fontsize = fontsize, fontproperties = "Times New Roman", color="white")
# # ax.set_title('Overview')
# ax.set_xticks(x)
# ax.set_xticklabels(x_labels, fontsize = fontsize, fontproperties = "Times New Roman", color="white")


ax.plot(x, values1, '-^',  color=colors[0], zorder = zorder, label='MOTA(%)', markersize = markersize)
ax.plot(x, values3, '-x',  color=colors[3], zorder = zorder, label='FPS', markersize = markersize)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel(' ', fontsize = fontsize, fontproperties = "Times New Roman", color="white")
ax.set_yticks(y_labels)
ax.set_yticklabels(y_labels, fontsize = fontsize, fontproperties = "Times New Roman", color="white")
# ax.set_title('Overview')
ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize = fontsize, fontproperties = "Times New Roman", color="white")

for a, b in zip(x, values1):
    ax.text(a, b, b, ha="center", va="center", fontsize = fontsize, fontproperties = "Times New Roman", zorder = 100)
for a, b in zip(x, values3):
    ax.text(a, b, b, ha="center", va="center", fontsize = fontsize, fontproperties = "Times New Roman", zorder = 100)

ax2 = ax.twinx()
tmp_ylabels = list(range(0, 600, 50))
ax2.plot(x, values2, '-+',  color=colors[1], label='IDs', markersize = markersize, zorder = zorder)
ax2.plot(x, values4, '-.',  color="gray", label='Size(MB)', markersize = markersize, zorder = zorder)

ax2.set_ylabel(' ', fontsize = fontsize, fontproperties = "Times New Roman", color="white")
ax2.set_yticks(tmp_ylabels)
ax2.set_yticklabels(tmp_ylabels, fontsize = fontsize, fontproperties = "Times New Roman", color="white")

for a, b in zip(x, values2):
    ax2.text(a, b, b, ha="center", va="center", fontsize = fontsize, fontproperties = "Times New Roman", zorder = 100)
for a, b in zip(x, values4):
    ax2.text(a, b, b, ha="center", va="center", fontsize = fontsize, fontproperties = "Times New Roman", zorder = 100)




# Show legend
ax.legend(loc = 6)

ax2.legend(loc = 7)

# autolabel(rects1)

fig.tight_layout()

plt.show()