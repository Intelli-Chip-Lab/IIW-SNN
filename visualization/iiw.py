import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

plt.rcParams['font.family'] = 'Arial'
fontsize = 28
fontsize_kedu = fontsize - 4
epochlist = [1,60,100]
info_dict = np.load('../info_dict.npy', mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII')
x = np.arange(len(info_dict))+1
IIW_array = np.zeros((3,10))
for t in range(len(info_dict)):
    for ep in range(len(epochlist)):
        IIW_array[ep][t] = info_dict[t][ep]

fig = plt.figure(figsize=(9, 7.5))
ax1 = fig.add_subplot(111)
ax1.plot(x, IIW_array[0],"ob-", label = "epoch-"+str(epochlist[0]),linewidth = 2)
ax1.plot(x, IIW_array[1],"ok-", label = "epoch-"+str(epochlist[1]),linewidth = 2)
ax1.plot(x, IIW_array[2],"or-", label = "epoch-"+str(epochlist[2]),linewidth = 2)

ax1.set_xlabel('timestep',fontsize=fontsize)
ax1.set_ylabel("I(W;S)",fontsize=fontsize)  # y轴名称 只能是英文
plt.xticks(fontsize=fontsize_kedu)
plt.yticks(fontsize=fontsize_kedu)
ax1.set_title("I(W;S) of CIFAR10",fontsize=fontsize,color='black',verticalalignment="center")#标题（表头）

# ax1.xlim(0, 11)  # 限制x坐标轴范围
ax1.legend(fontsize=fontsize_kedu)  # 显示标签
ax1.grid(ls='--')  # 显示网格线
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.show()