import matplotlib.pyplot as plt

# 解决中文显示问题
import numpy as np

plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

x = np.array([0, 564, 1408, 2112, 2816])
y = np.array([75.09, 77.32, 79.14, 80.74, 81.47])
plt.plot(x, y, "o:r")
plt.title("fine-tuning实验结果")
plt.xlabel("带标签的图片数量（由于fine-tuning）")
plt.ylabel("测试集准确率/%")
plt.savefig("data1.jpg")
plt.show()
