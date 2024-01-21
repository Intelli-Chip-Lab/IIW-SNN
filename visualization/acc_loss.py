import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
plt.rcParams['font.family'] = 'Arial'
fontsize = 28
fontsize_kedu = fontsize - 4
# acc_loss_dict = np.load('../snapshots/loss_acc_dict/80.npy',allow_pickle=True)
f_read = open('../snapshots/loss_acc_dict/ep120.pkl', 'rb')
acc_loss_dict = pickle.load(f_read)
f_read.close()
print(acc_loss_dict.keys())

x=np.arange(len(acc_loss_dict['tr_loss']))
# 双y轴画图参考  https://blog.csdn.net/weixin_39602776/article/details/109852535
fig = plt.figure(figsize=(9, 7.5))
# 左侧坐标
ax1 = fig.add_subplot(111)
lns1 = ax1.plot(x, acc_loss_dict['tr_acc'], '-', color='r', label='train acc',linewidth = 2)
lns2 = ax1.plot(x, acc_loss_dict['te_acc'], '-', color='b', label='test acc',linewidth = 2)
ax1.set_ylabel('acc',fontsize=fontsize)
ax1.set_xlabel('epoch',fontsize=fontsize)

plt.xticks(fontsize=fontsize_kedu)

plt.yticks(fontsize=fontsize_kedu)
ax1.set_title("Performance curve on CIFAR10",fontsize=fontsize,color='black',verticalalignment="center")#标题（表头）
# 右侧坐标
ax2 = ax1.twinx()
lns3 = ax2.plot(x, acc_loss_dict['tr_loss'], '-', color='k',label='train loss',linewidth = 2)
ax2.set_xlabel('Same')
plt.yticks(fontsize=fontsize_kedu)
lns = lns1 + lns2 + lns3
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=7, fontsize=fontsize_kedu)
ax1.grid(ls='--')
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.show()