import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

x =     [0.2, 0.3, 0.4, 0.6, 1]

y_t3 =     [89.62, 91.08, 91.59, 91.97, 92.03] # 100% 100% 92.13
y_t3_pib = [91.02, 92.12, 92.62, 93.02, 93.09] # 100% 100% 93.16

y_t2 =     [89.04, 90.80, 91.22, 91.66, 91.87] # 100% 100% 92.13
y_t2_pib = [89.82, 91.56, 92.01, 92.60, 92.80] # 100% 100% 93.16

y_t1 =     [87.57, 89.56, 90.06, 90.76, 91.21] # 100% 100% 92.13
y_t1_pib = [88.45, 90.61, 91.03, 91.72, 92.21] # 100% 100% 93.16


linewidth = 1.5
fontsize = 16
fontsize_kedu = fontsize - 2

fig = plt.figure(figsize=(7, 5))
ax1 = fig.add_subplot(111)

# 设置线条样式和颜色
ax1.plot(x, y_t3, '--x', linewidth=linewidth, label='Rt=0.3', color='k')
ax1.plot(x, y_t2, '--x', linewidth=linewidth, label='Rt=0.2', color='r')
ax1.plot(x, y_t1, '--x', linewidth=linewidth, label='Rt=0.1', color='b')

ax1.plot(x, y_t3_pib, '-^', linewidth=linewidth, label='Rt=0.3(PIB)', color='k')
ax1.plot(x, y_t2_pib, '-^', linewidth=linewidth, label='Rt=0.2(PIB)', color='r')
ax1.plot(x, y_t1_pib, '-^', linewidth=linewidth, label='Rt=0.1(PIB)', color='b')


ax1.set_title("Joint-way optimization (CIFAR10)",fontsize=fontsize,color='black',verticalalignment="center")#标题（表头）
# The results of applying three methods on the CIFAR10 datasets
ax1.legend(fontsize=14)  # 显示标签
ax1.grid(ls='--')  # 显示网格线

# 设置网格线
ax1.grid(color='#DDDDDD', linestyle='--', linewidth=0.5)
plt.xticks(fontsize=fontsize_kedu)
plt.yticks(fontsize=fontsize_kedu)
# 设置标签样式
ax1.set_xlabel('Rs(connectivity)',fontsize=fontsize)
ax1.set_ylabel("Test acc",fontsize=fontsize)  # y轴名称 只能是英文

# 显示图形
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.show()