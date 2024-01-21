import torchvision
import torch.nn as nn
from collections import defaultdict
from spikingjelly.clock_driven.functional import reset_net # from spikingjelly.activation_based.functional import reset_net
import pickle

from archs.resnet_snn import ResNet19
from src import utils, config
from src.utils import *
from thop import profile

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
loss_acc_dict = defaultdict(list)
args = config.get_args()
n_class = 10

def main(train_loader, val_loader, n_class):
    print("================================================")
    org_model = ResNet19(num_classes=10, total_timestep=args.timestep)
    org_model.to(device)

    model_index = 100
    org_model.load_state_dict(torch.load(f"./snapshots/T{str(args.timestep)}_cifar10/T{str(args.timestep)}_{str(args.dataset)}_ckpt_test_{str(model_index).zfill(4)}.pth.tar")['state_dict'])
    # 创建一个模拟输入
    input_size = (1, 3, 32, 32)
    input_data_ = torch.randn(*input_size).to(device)
    # 使用 thop 的 profile 函数计算模型的 FLOPs
    flops, params = profile(org_model, inputs=(input_data_,))
    formatted_flops = "{:.3e}".format(flops)
    print(f"FLOPs of spatial_pruned_model:{formatted_flops}")

    val_top1 = validate(args, -1, val_loader, org_model, nn.CrossEntropyLoss().to(device))
    print("acc:", val_top1)

    param_num = 0
    param_num = sum([param.numel() for param in org_model.parameters()])
    param_num_in_millions = param_num / 1000000
    print("Original parameter number: {:.2f}M".format(param_num_in_millions))
    #  ===================================权重剪枝后的模型===================================
    # print("================================================")
    # s_model = torch.load("spatioal_pruned_model.pth")
    # s_model.to(device)
    #
    # # 创建一个模拟输入
    # input_size = (1, 3, 32, 32)
    # input_data_ = torch.randn(*input_size).to(device)
    # # 使用 thop 的 profile 函数计算模型的 FLOPs
    # flops, params = profile(s_model, inputs=(input_data_,))
    #
    # formatted_flops = "{:.3e}".format(flops)
    # print(f"FLOPs of spatial_pruned_model:{formatted_flops}")

    #  ===================================时空剪枝后的模型===================================
    print("================================================")
    st_model = torch.load("./spatiotemporal_pruned_model.pth").to(device)

    param_num = 0
    param_num = sum([param.numel() for param in st_model.parameters()])
    param_num_in_millions = param_num / 1000000
    print("spatiotemporal_pruned_model's paramater number: {:.2f}M".format(param_num_in_millions))

    val_top1 = validate(args, -1, val_loader, st_model, nn.CrossEntropyLoss().to(device))
    print("acc:", val_top1)

    # 计算模型的 FLOPs
    print("compute FLOPs of the spatiotemporal_pruned_model:")

    # 创建一个模拟输入
    input_size = (1, 3, 32, 32)
    input_data = torch.randn(*input_size).to(device)
    # 使用 thop 的 profile 函数计算模型的 FLOPs
    flops, params = profile(st_model, inputs=(input_data,))

    formatted_flops = "{:.3e}".format(flops)
    print(f"FLOPs of spatiotemporal_pruned_model:{formatted_flops}")
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
    trainset = torchvision.datasets.CIFAR10(root=os.path.join(args.data_dir, 'cifar10'), train=True, download=False,transform=train_transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, pin_memory=False,num_workers=0)
    valset = torchvision.datasets.CIFAR10(root=os.path.join(args.data_dir, 'cifar10'), train=False, download=False,transform=valid_transform)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, pin_memory=False,num_workers=0)
    n_class = 10

    main(train_loader, val_loader, n_class)