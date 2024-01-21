import math
import torchvision
import torch.nn as nn
from collections import defaultdict
from spikingjelly.clock_driven.functional import reset_net # from spikingjelly.activation_based.functional import reset_net
import pickle

from archs.resnet_snn import ResNet19
from src import utils, config
from src.utils import *
from src.pib_utils import *

__prior_ckpt__ = './checkpoints_snn/Resnet_prior.pt'
__save_ckpt__ = './checkpoints_snn/Resnet_pib.pt'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
loss_acc_dict = defaultdict(list)
args = config.get_args()

def main(train_loader, val_loader, n_class):
    model = ResNet19(num_classes=n_class, total_timestep=args.timestep).to(device)
    print("======================= load prior =======================")
    model.load_state_dict(torch.load("checkpoints_snn/Resnet_prior.pth"))
    w0_dict = dict()
    for param in model.named_parameters():
        w0_dict[param[0]] = param[1].clone().detach()  # detach but still on gpu
    model.w0_dict = w0_dict
    print("done get prior weights")
    print("======================= load pre-train model =======================")
    ep = 80
    model.load_state_dict(torch.load(f"./snapshots/T{str(args.timestep)}_cifar10/T{str(args.timestep)}_{str(args.dataset)}_ckpt_test_{str(ep).zfill(4)}.pth.tar")['state_dict'])
    val_top1 = validate(args, -1, val_loader, model, nn.CrossEntropyLoss().to(device))
    print("acc",val_top1)
    print("load finish")
    print("======================= pib train model =======================")
    info_dict = defaultdict(list)
    beta = 0.1
    best_te_acc = 0
    energy_decay = 0

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = SGLD(params=filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay, noise_scale=1e-10)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(args.epochs), eta_min=0)
    for epoch in range(60):
        train_SGLD(args, model, optimizer, criterion, scheduler, train_loader, device, epoch, energy_decay)
        scheduler.step()

        info = model.compute_information_bp_fast(args, train_loader, timestep=args.timestep, no_bp=True)
        energy_decay = 0
        for k in info.keys():
            energy_decay += info[k] # plus decay term for each weight
            info_dict[k].append(info[k]) # info_dict[k].append(info[k].item())
        energy_decay = beta * energy_decay

        val_top1 = validate(args, epoch, val_loader, model, criterion)
        if val_top1 > best_te_acc:
            best_te_acc = val_top1
            print("best acc", best_te_acc)
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

    main(train_loader, val_loader, n_class)