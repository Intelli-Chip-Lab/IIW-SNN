import torch
import torchvision
import torch.nn as nn
from collections import defaultdict
from spikingjelly.clock_driven.functional import reset_net # from spikingjelly.activation_based.functional import reset_net
import pickle
# from archs.vgg_snn import spiking_vgg16
from archs.vgg_snn import VGG16
from src import utils, config
from src.utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
loss_acc_dict = defaultdict(list)
args = config.get_args()
# C:\Users\wkn15/.cache\torch\hub\checkpoints\vgg16-397923af.pth 模型的位置
def main(train_loader, val_loader, n_class):
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>> GET piror_weights >>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # print("================== get piror_weights ==================")
    # model_piror = ResNet19(num_classes=n_class, total_timestep=args.timestep).to(device)
    # has_prior_model = True
    # if has_prior_model:
    #     model_piror.load_state_dict(torch.load("checkpoints_snn/Resnet_prior.pth"))
    # else:
    #     get_piror_weights(args, val_loader, model_piror, "True")

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>> GET posterior weights  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    print("================== train model ==================")
    model = VGG16().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, args.momentum, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max= int(args.epochs), eta_min= 0)

    best_te_acc = 0

    for epoch in range(args.epochs):  # 300 迭代
        print(epoch + 1)
        # =========================== train
        train(args, epoch, train_loader, model, criterion, optimizer, scheduler)
        scheduler.step()
        # =========================== validate
        val_top1 = validate(args, epoch, val_loader, model, criterion)
        # =========================== save results and model
        if val_top1 > best_te_acc :
            best_te_acc = val_top1
            if ((epoch + 1) > 60) or (epoch+1 ==1):
                utils.save_checkpoint({'state_dict': model.state_dict(), }, iters=epoch + 1, tag='T' + str(args.timestep) + '_' + str(args.dataset))
        if (epoch + 1) % args.save_freq == 0:
            utils.save_checkpoint({'state_dict': model.state_dict(), }, iters=epoch + 1, tag='T' + str(args.timestep) + '_' + str(args.dataset))
            f_save = open('./snapshots/loss_acc_dict/ep' + str(epoch + 1) + '.pkl', 'wb')
            pickle.dump(loss_acc_dict, f_save)
            f_save.close()

def get_piror_weights(args , val_loader, model, get_piror):
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, args.momentum, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(args.epochs), eta_min=0)
    early_stop_ckpt_path= "checkpoints_snn/Resnet_prior.pth"
    best_te_acc = 0
    loss_acc_dict_ =  defaultdict(list)
    for epoch in range(120):  # 40 迭代
        print(epoch)
        train(args, epoch, val_loader, model, criterion, optimizer, scheduler, get_piror)
        scheduler.step()
        val_top1 = validate(args, epoch, val_loader, model, criterion)
        # =========================== save results and model
        if val_top1 > best_te_acc:
            best_te_acc = val_top1
            torch.save(model.state_dict(), early_stop_ckpt_path)   # model.load_state_dict(torch.load(early_stop_ckpt_path))  导入的时候用

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

def calcalate_IWS(train_data):
    model = ResNet19(num_classes=10, total_timestep=args.timestep).to(device)
    model.load_state_dict(torch.load("checkpoints_snn/Resnet_prior.pth"))
    w0_dict = dict()
    for param in model.named_parameters():
        w0_dict[param[0]] = param[1].clone().detach()
    model.w0_dict = w0_dict

    epochlist = [1,60,100]
    # info_dict = defaultdict(list)
    info_dict = []
    for t in range(args.timestep):
        info_dict.append([])

    for ep in epochlist:
        print('Ep', str(ep), '--total time', str(args.timestep))
        model.load_state_dict(torch.load(f"./snapshots/T{str(args.timestep)}_cifar10/T{str(args.timestep)}_{str(args.dataset)}_ckpt_test_{str(ep).zfill(4)}.pth.tar")['state_dict'])

        ep_fisher_list = []
        for timestep in range(1, args.timestep + 1):
            info = model.compute_information_bp_fast(args, train_data, timestep, no_bp=True)
            temp = 0
            for k in info.keys():
                temp += info[k]
            info_dict[timestep-1].append(temp)
    np.save('info_dict.npy', info_dict, allow_pickle=True, fix_imports=True)  # 注意带上后缀名
    return 1

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
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>  计算IWS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    calculate_iws = False   # 记得把 resnet 也上传
    if calculate_iws:
        calcalate_IWS(train_loader)
    else:
        main(train_loader, val_loader, n_class)


