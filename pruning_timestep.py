import torchvision
import torch.nn as nn
from collections import defaultdict
from spikingjelly.clock_driven.functional import reset_net # from spikingjelly.activation_based.functional import reset_net
import pickle

from archs.resnet_snn import ResNet19
from src import utils, config
from src.utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
loss_acc_dict = defaultdict(list)
args = config.get_args()
n_class = 10
def main(train_loader, val_loader, n_class):
    # -----------------------------------------------------载入模型-------------------------------------------
    print("=====================load model=====================")
    choose_model = 1  # 1代表直接进行时间步剪枝   2代表对spatial pruning之后的模型进行时间步剪枝
    if choose_model==2:
        #  这里需要载入微调模型结构和参数!!!
        model = ResNet19(num_classes=n_class, total_timestep=args.timestep).to(device)
        epoch = 100
        model.load_state_dict(torch.load(f"./snapshots/T{str(args.timestep)}_cifar10/T{str(args.timestep)}_{str(args.dataset)}_ckpt_test_{str(100).zfill(4)}.pth.tar")['state_dict'])
    else:
        #  这里需要载入微调模型结构和参数
        model = torch.load("spatioal_pruned_model.pth").to(device)

    param_num = sum([param.numel() for param in model.parameters()])
    param_num_in_millions = param_num / 1000000
    print("weight pruned model's paramater num:  {:.2f}M".format(param_num_in_millions))
    val_top1 = validate(args, -1, val_loader, model, nn.CrossEntropyLoss().to(device))
    print("original acc", val_top1)
    # -----------------------------------------------------时间步剪枝-------------------------------------------
    print("=====================prune timestep=====================")
    model.total_timestep = 3
    timestep_prune = model.total_timestep
    print(timestep_prune)

    # -----------------------------------------------------微调模型-------------------------------------------
    print("=====================fine tune model=====================")
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, args.momentum, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(args.epochs), eta_min=0)

    best_te_acc = 0

    for epoch in range(30):  # 300 迭代
        train(args, epoch, train_loader, model, criterion, optimizer, scheduler)
        scheduler.step()
        val_top1 = validate(args, epoch, val_loader, model, criterion)  # print("epoc={}, acc={})".format(epoch, val_top1))
        if val_top1 > best_te_acc:
            best_te_acc = val_top1
            print("save model")
            torch.save(model, 'spatiotemporal_pruned_model.pth')
    # -----------------------------------------------------输出结果-------------------------------------------
    # 这里需要载入微调模型
    st_model = torch.load("spatiotemporal_pruned_model.pth").to(device)
    val_top1 = validate(args, epoch, val_loader, st_model, criterion)
    print("temporal_prune(best) acc", val_top1)

    param_num = 0
    param_num = sum([param.numel() for param in st_model.parameters()])
    param_num_in_millions = param_num / 1000000
    print("spatiotemporal_pruned_model's paramater number: {:.2f}M".format(param_num_in_millions))

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

        # # 计算模型的 FLOPs
        # print("compute FLOPs of the spatiotemporal_pruned_model:")
        #
        # # 创建一个模拟输入
        # input_size = (1, 3, 32, 32)
        # input_data = torch.randn(*input_size).to(device)
        # # 使用 thop 的 profile 函数计算模型的 FLOPs
        # flops, params = profile(st_model, inputs=(input_data,))
        # print(f"FLOPs of spatiotemporal_pruned_model:{flops}")
        # formatted_flops = "{:.3e}".format(flops)
        # print(f"FLOPs of temporal_pruned_model:{formatted_flops}")
        # # print(f"spatiotemporal_pruned_model's paramater number computed by thop tool:{formatted_flops}")