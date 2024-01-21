import torchvision
import torch
import torch.nn as nn
from collections import defaultdict
from spikingjelly.clock_driven import surrogate, neuron
from spikingjelly.clock_driven.functional import reset_net # from spikingjelly.activation_based.functional import reset_net
from nni.compression.pruning import FPGMPruner
from nni.compression.speedup import ModelSpeedup
import pickle
import numpy as np
import os

from archs.resnet_snn import ResNet19
from src import utils, config
from src.utils import *


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
loss_acc_dict = defaultdict(list)
args = config.get_args()
n_class = 10

def main(train_loader, val_loader, n_class):
    # -----------------------------------------------------准备模型-------------------------------------------
    print("===========  load original model  ===========")
    model = ResNet19(num_classes=10, total_timestep=args.timestep)
    model.to(device)
    # 选择模型：选择时间步剪枝还是时间步 没有剪枝的模型
    model_index = 100
    model.load_state_dict(torch.load(f"./snapshots/T{str(args.timestep)}_cifar10/T{str(args.timestep)}_{str(args.dataset)}_ckpt_test_{str(model_index).zfill(4)}.pth.tar")['state_dict'])
    # # 这里需要载入微调模型
    # with open('pruning_timestep.pkl', 'rb') as f:
    #     model = pickle.load(f)
    # timestep_prune = 1
    # print("model total timestep:", model.total_timestep)

    val_top1 = validate(args, -1, val_loader, model, nn.CrossEntropyLoss().to(device))
    print("original acc", val_top1)

    param_num = 0
    param_num = sum([param.numel() for param in model.parameters()])
    param_num_in_millions = param_num / 1000000
    print("Original parameter number: {:.2f}M".format(param_num_in_millions))

    # -----------------------------------------------------剪枝模型-------------------------------------------
    # 将 LIF神经元换为 relu
    for n, m in model.named_modules():
        if (("lif" in n) and ('surrogate' not in n)):
            _set_module(model, n, nn.ReLU())

    config_list = [{
        'op_types': ['Conv2d'],
        'sparse_ratio': 0.70,  # 0是不剪枝    0.4是剪枝40%的权重     0.6是剪枝60%的权重
        'exclude_op_names': ['conv1']  # 排除执行剪枝的层名称
    }]

    print("===========  prune model  ===========")
    pruner = FPGMPruner(model, config_list)
    _, masks = pruner.compress()  # compress the model and generate the masks

    # show the masks sparsity
    for name, mask in masks.items():
        print(name, ' sparsity : ', '{:.2}'.format(mask['weight'].sum() / mask['weight'].numel()))
    pruner.unwrap_model()  # need to unwrap the model, if the model is wrapped before speedup
    ModelSpeedup(model, torch.rand(1, 3, 32, 32).to(device), masks, ).speedup_model()  # 这里要随数据集的改变而调整

    param_num = 0
    param_num = sum([param.numel() for param in model.parameters()])
    param_num_in_millions = param_num / 1000000
    print("Pruned model parameter number: {:.2f}M".format(param_num_in_millions))

    tau_global = 1. / (1. - 0.5)
    # 不能用同一个实例化的神经元 去 替换所有的Relu()函数！
    # LIF_neuron = neuron.LIFNode(v_threshold=1.0, v_reset=0.0, tau=tau_global, surrogate_function=surrogate.ATan(), detach_reset=True)
    # 将神经元换回来
    for n, m in model.named_modules():
        if (("lif" in n) and ('surrogate' not in n)):
            _set_module(model, n, neuron.LIFNode(v_threshold=1.0, v_reset=0.0, tau=tau_global, surrogate_function=surrogate.ATan(), detach_reset=True))  # detach_reset 冻结梯度
    print("finish pruning")

    # -----------------------------------------------------微调模型-------------------------------------------

    print("===========  fine tune model  ===========")
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, args.momentum, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(args.epochs), eta_min=0)

    best_te_acc = 0

    for epoch in range(30):  # 300 迭代
        train(args, epoch, train_loader, model, criterion, optimizer, scheduler)
        scheduler.step()
        val_top1 = validate(args, epoch, val_loader, model, criterion)
        # print("epoc={}, acc={})".format(epoch+1, val_top1))
        if val_top1 > best_te_acc:
            best_te_acc = val_top1
            print("save model")
            torch.save(model, 'spatioal_pruned_model.pth')


    # 导入微调后的 模型和模型的参数
    print("load fine turn model")
    _model = torch.load('spatioal_pruned_model.pth')

    val_top1 = validate(args, epoch, val_loader, _model, criterion)
    print("fine-turn test acc top1 = ", val_top1)
    return 1


def train(args, epoch, train_data,  model, criterion, optimizer, scheduler, get_piror=False,):
    model.train()
    top1 = utils.AvgrageMeter()
    train_loss = 0.0
    print('[%s%04d/%04d %s%f]' % ('Epoch:', epoch + 1, args.epochs, 'lr:', scheduler.get_last_lr()[0]))

    for step, (inputs, targets) in enumerate(train_data):

        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        output_list = model(inputs)
        output = sum(output_list)/ args.timestep
        loss = criterion(output, targets)
        loss.backward()

        prec1, prec5 = utils.accuracy(sum(output_list), targets, topk=(1, 5))

        n = inputs.size(0)
        top1.update(prec1.item(), n)
        train_loss += loss.item()

        reset_net(model)   # 优化一次参数后，需要重置网络的状态，因为SNN的神经元是有“记忆”的
        optimizer.step()
    #   optimizer.step()通常用在每个 mini-batch 之中，而scheduler.step()通常用在epoch里面,但是不绝对。可以根据具体的需求来做。
    #   只有用了optimizer.step()，模型才会更新，而scheduler.step()是对lr进行调整。

    print_loss = train_loss / len(train_data)
    print_tr_acc = top1.avg
    loss_acc_dict["tr_loss"].append(print_loss)
    loss_acc_dict["tr_acc"].append(print_tr_acc)
    print('train_loss: %.3f' % (print_loss), 'train_acc: %.3f' % print_tr_acc)
    return print_tr_acc

def validate(args, epoch, val_data, model, criterion):
    model.eval()
    val_loss = 0.0
    val_top1 = utils.AvgrageMeter()

    with torch.no_grad():
        # requires_grad 和 with torch.no_grad()的理解： https://blog.csdn.net/sazass/article/details/116668755
        for step, (inputs, targets) in enumerate(val_data):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(sum(outputs), targets)
            val_loss += loss.item()
            prec1, prec5 = utils.accuracy(sum(outputs), targets, topk=(1, 5))
            n = inputs.size(0)
            val_top1.update(prec1.item(), n)
            reset_net(model)
        print_te_acc = val_top1.avg
        print('[Val_Accuracy epoch:%d] val_acc:%f' % (epoch + 1,  print_te_acc))
        loss_acc_dict["te_acc"].append(print_te_acc)
        return print_te_acc

def _set_module(model, submodule_key, module):
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)

if __name__ == '__main__':
    train_transform, valid_transform = data_transforms(args)
    trainset = torchvision.datasets.CIFAR10(root=os.path.join(args.data_dir, 'cifar10'), train=True, download=True,
                                            transform=train_transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, pin_memory=False,
                                               num_workers=0)
    valset = torchvision.datasets.CIFAR10(root=os.path.join(args.data_dir, 'cifar10'), train=False, download=True,
                                          transform=valid_transform)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, pin_memory=False,
                                             num_workers=0)
    n_class = 10

    main(train_loader, val_loader, n_class)
    # 总结
    # 代码最终输出的是一个空间剪枝完成的模型  叫做 spatial_pruned_model 开源计算参数量(M)
    # 拿到这个模型再进行时间剪枝  最后拿到一个时空剪枝的模型  叫做 spayiotemporal_pruned_model
    # 再用compute_FLOPs.py计算 FLOPs 用科学计数法表示。