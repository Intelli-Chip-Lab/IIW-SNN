import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision
from spikingjelly.clock_driven import functional, layer, surrogate, neuron
from spikingjelly.clock_driven.functional import reset_net
from torch.autograd import grad

tau_global = 1. / (1. - 0.5)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        # 二维卷积 in_planes 输入张量通道数 planes 输出张量的输出通道数
        self.bn1 = nn.BatchNorm2d(planes)
        # BatchNorm2d进行数据的归一化处理，这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定，
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.lif1 = neuron.LIFNode(v_threshold=1.0, v_reset=0.0, tau=tau_global,
                                   surrogate_function=surrogate.ATan(),
                                   detach_reset=True)  # detach_reset 冻结梯度

        self.lif2 = neuron.LIFNode(v_threshold=1.0, v_reset=0.0, tau=tau_global,
                                   surrogate_function=surrogate.ATan(),  #使用了代理损失函数
                                   detach_reset=True)

        self.shortcut = nn.Sequential()   # 理解 https://blog.csdn.net/qq_23345187/article/details/121336352
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.lif1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.lif2(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, total_timestep=6):
        super(ResNet, self).__init__()
        self.in_planes = 128
        self.total_timestep = total_timestep

        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1, bias=False)
        #输入通道3 输出通道128
        self.bn1 = nn.BatchNorm2d(128)
        self.lif_input = neuron.LIFNode(v_threshold=1.0, v_reset=0.0, tau=tau_global,
                                        surrogate_function=surrogate.ATan(),
                                        detach_reset=True)

        self.layer1 = self._make_layer(block, 256, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, 512, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 512, num_blocks[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, input_img):
        output_list = []
        static_x = self.bn1(self.conv1(input_img))# 同一个图片，输入之后，会产生T个输出，时间步的数量个
        for t in range(self.total_timestep):
            out = self.lif_input(static_x)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.avgpool(out)
            out = out.view(out.size(0), -1)
            out = self.fc1(out)   #最后是全连接层

            output_list.append(out)
        return output_list
    def compute_information_bp_fast(self, args, train_data, timestep, no_bp=False):
        """Compute the full information with back propagation support.
        Using delta_w.T gw @ gw.T delta_w = (delta_w.T gw)^2 for efficient computation.
        Args:
            no_bp: detach the information term hence it won't be used for learning.
        """
        def one_hot_transform(y, num_class=10):
            one_hot_y = F.one_hot(y, num_classes=10)
            return one_hot_y.float()

        # delta_w_dict 计算
        param_keys = [p[0] for p in self.named_parameters()]
        delta_w_dict = dict().fromkeys(param_keys)
        for pa in self.named_parameters():
            if "weight" in pa[0]:
                w0 = self.w0_dict[pa[0]]
                delta_w = pa[1] - w0
                delta_w_dict[pa[0]] = delta_w

        info_dict = dict()
        gw_dict = dict().fromkeys(param_keys)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        for step, (inputs, targets) in enumerate(train_data):
            inputs, targets = inputs.to(device), targets.to(device)
            # 这个地方加 reset_net(self)
            reset_net(self)
            pred = self.forward(inputs)
            output = sum(pred[:timestep]) / timestep
            y_oh_batch = one_hot_transform(targets, 10)
            loss = F.cross_entropy(output, y_oh_batch, reduction="mean")  # none
            # way1:
            gradients = grad(loss, self.parameters())  # 求损失函数对 参数的导数
            for i, gw in enumerate(gradients):
                gw_ = gw.flatten()
                if gw_dict[param_keys[i]] is None:
                    gw_dict[param_keys[i]] = gw_
                else:
                    gw_dict[param_keys[i]] += gw_  # 对梯度进行累加
            if step==40:
                break;
        reset_net(self)

        for k in gw_dict.keys():
            if "weight" in k:
                gw_dict[k] *= ((1/32)*(1/40))
                delta_w = delta_w_dict[k]
                # delta_w.T gw @ gw.T delta_w = (delta_w.T gw)^2
                info_ = (delta_w.flatten() * gw_dict[k]).sum() ** 2
                if no_bp:
                    info_dict[k] = info_.item()
                else:
                    info_dict[k] = info_
        return info_dict

def ResNet19(num_classes, total_timestep):
    return ResNet(BasicBlock, [3, 3, 2], num_classes, total_timestep)
# return ResNet(BasicBlock, [2, 2, 1], num_classes, total_timestep)

# model = ResNet19(10,10)
# batch_size = 1
# channels = 3
# height = 32
# width = 32
# input_data = torch.randn(batch_size, channels, height, width)
# output_list = model(input_data)
# print(output_list)