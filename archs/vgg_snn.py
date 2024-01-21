import torch
import torch.nn as nn
from spikingjelly.clock_driven import functional, layer, surrogate, neuron
from spikingjelly.clock_driven.functional import reset_net

tau_global = 1. / (1. - 0.5)


# 定义VGG的卷积块模块，包含连续的卷积层和池化层
def vgg_block(in_channels, out_channels, num_convs):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(neuron.LIFNode(v_threshold=1.0, v_reset=0.0, tau=tau_global,
                                   surrogate_function=surrogate.ATan(),
                                   detach_reset=True))
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)

# 定义VGG16模型
class VGG16(nn.Module):
    def __init__(self, num_classes=10,total_timestep=10):
        super(VGG16, self).__init__()
        self.total_timestep = total_timestep
        # BatchNorm2d进行数据的归一化处理，这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定，
        self.encode_layer = vgg_block(3, 64, 2)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.features = nn.Sequential(
            vgg_block(64, 128, 2),
            vgg_block(128, 256, 3),
            vgg_block(256, 512, 3),
            vgg_block(512, 512, 3)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            neuron.LIFNode(v_threshold=1.0, v_reset=0.0, tau=tau_global,
                           surrogate_function=surrogate.ATan(),
                           detach_reset=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            neuron.LIFNode(v_threshold=1.0, v_reset=0.0, tau=tau_global,
                           surrogate_function=surrogate.ATan(),
                           detach_reset=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        output_list = []
        static_x = self.bn1(self.conv1(x))
        for t in range(self.total_timestep):
            out = self.features(static_x)
            out = self.avgpool(out)
            out = torch.flatten(out, 1)
            out = self.classifier(out)
            output_list.append(out)
        return output_list
    # def forward(self, x):
    #     output_list = []
    #     for t in range(self.total_timestep):
    #         x1 = self.features(x)
    #         x2 = self.avgpool(x1)
    #         x3 = torch.flatten(x2, 1)
    #         x4 = self.classifier(x3)
    #         output_list.append(x4)
    #     return output_list

# 创建一个VGG16实例
model = VGG16()
# Create a random input tensor
batch_size = 1
channels = 3
height = 32
width = 32
input_data = torch.randn(batch_size, channels, height, width)
output_list = model(input_data)
print(output_list)