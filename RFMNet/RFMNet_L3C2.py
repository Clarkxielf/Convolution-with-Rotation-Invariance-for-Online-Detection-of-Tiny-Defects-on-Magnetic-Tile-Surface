import torch
import torch.nn as nn

__all__ = ['rfmnet']

def conv3x3(in_planes, out_planes, stride=1, groups=1):
    return nn.Conv2d(in_planes, out_planes, stride=stride, groups=groups, kernel_size=3, padding=1, bias=False)

class RFMBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(RFMBlock, self).__init__()

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes//4, stride, groups=planes//4)
        self.bn1 = nn.BatchNorm2d(planes//4)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes//4, groups=planes//4)

        self.downsample = downsample
        self.bn = nn.BatchNorm2d(planes)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out0 = self.relu(out)
        x0 = torch.rot90(out0, k=1, dims=(-1, -2))
        x1 = torch.rot90(x0, k=1, dims=(-1, -2))
        x2 = torch.rot90(x1, k=1, dims=(-1, -2))
        out = torch.cat([out0, x0, x1, x2], dim=1)

        out0 = self.conv2(out)
        x0 = torch.rot90(out0, k=1, dims=(-1, -2))
        x1 = torch.rot90(x0, k=1, dims=(-1, -2))
        x2 = torch.rot90(x1, k=1, dims=(-1, -2))
        out = torch.cat([out0, x0, x1, x2], dim=1)

        if self.downsample is not None:
            identity0 = self.downsample(x)
            x0 = torch.rot90(identity0, k=1, dims=(-1, -2))
            x1 = torch.rot90(x0, k=1, dims=(-1, -2))
            x2 = torch.rot90(x1, k=1, dims=(-1, -2))
            identity = torch.cat([identity0, x0, x1, x2], dim=1)

        out += identity
        out = self.bn(out)
        out = self.relu(out)

        return out



class RFMNet(nn.Module):

    def __init__(self, block, layers, scale_channel=1, num_classes=3):
        super(RFMNet, self).__init__()

        self._norm_layer = nn.BatchNorm2d
        self.scale_channel = scale_channel
        self.inplanes = int(64*self.scale_channel)

        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, int(128*self.scale_channel), layers[0], stride=2)
        self.layer2 = self._make_layer(block, int(256*self.scale_channel), layers[1], stride=2)
        self.layer3 = self._make_layer(block, int(512*self.scale_channel), layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.fc = nn.Linear(int(4*512*self.scale_channel), num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # elif isinstance(m, RFMBlock):
            #     nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = conv3x3(self.inplanes, planes//4, stride, groups=planes//4)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _rfmnet(block, layers, scale_channel):

    model = RFMNet(block, layers, scale_channel)
    return model


def rfmnet(block=RFMBlock, scale_layer=3, scale_channel=2):

    return _rfmnet(block, [1, 1*scale_layer, 1*scale_layer], scale_channel)




# model = rfmnet()
# x = torch.randn((2, 1, 224, 224))
# y = model(x)
# print(y.shape)

# from thop.profile import profile
# from ptflops import get_model_complexity_info
# from torchstat import stat
#
# model = rfmnet()
#
# # # y = model(torch.randn((1, 1, 224, 224)))
# # stat(model, (1, 224, 224))
#
# print('%s|%s|%s'%('Params(M)', 'FLOPs(G)', 'FLOPs(M)'))
# print('---|---|---')
# input = torch.randn((1, 1, 224, 224))
# total_FLOPs, total_Params = profile(model, (input, ), verbose=False)
# print('%.3fM|%.3fG|%.3fM'%(total_Params/(1000**2), total_FLOPs/(1000**3), total_FLOPs/(1000**2)))
#
# print('%s|%s'%('Params(M)', 'FLOPs(G)'))
# print('---|---|---')
# flops, params = get_model_complexity_info(model, (1, 224, 224), as_strings=True, print_per_layer_stat=True)
# print(f'{params}|{flops}')


# import time
# device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
# input = torch.randn((6628*3, 1, 224, 224)).to(device)
# model = rfmnet().to(device)
#
# start = time.time()
# for i, data in enumerate(input):
#     model(data[None])
# end = time.time()
# print('FPS is %.1f'%(input.shape[0]/(end-start)))