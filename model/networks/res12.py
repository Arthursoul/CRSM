import torch.nn as nn
import torch
import torch.nn.functional as F
from model.networks.dropblock import DropBlock
from torchsummary import summary

# This ResNet network was designed following the practice of the following papers:
# TADAM: Task dependent adaptive metric for improved few-shot learning (Oreshkin et al., in NIPS 2018) and
# A Simple Neural Attentive Meta-Learner (Mishra et al., in ICLR 2018).

# 这是作者在文章中使用的一种baseline
# basicblock中有12层:
#        conv2d(3,64,3,2,1) [ch_in, ch_out, kernael size, stride, padding]
#        bn(64)   relu()
#        conv2d(64,64,3,2,1) [ch_in, ch_out, kernael size, stride, padding]
#        bn(64)   relu()
#        conv2d(64,64,3,2,1) [ch_in, ch_out, kernael size, stride, padding]
#        bn(64)   downsample()?   relu()
#                 这里这个downsample相当于在原来输入的基础上开出一条新分支，做了一次 conv2d 和 bn
#                 将结果在通道维度加给了当前位置的x
#        maxpooling(2)     dropblock()/dropout()

# 维度变化 就是通道数与convnet不同，在图片维度上面并没有进行更改
# 84*84*3 => 42*42*64 => 21*21*160 => 10*10*320 => 5*5*640 => 1*1*640 => 1*640

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.0, drop_block=False, block_size=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size
        self.DropBlock = DropBlock(block_size=self.block_size)

    def forward(self, x):
        self.num_batches_tracked += 1

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = self.maxpool(out)

        if self.drop_rate > 0:
            # 判断是否使用 dropblock 不用的话就直接 dropout
            if self.drop_block == True:
                feat_size = out.size()[2]

                # 随着训练的深入 drop rate 越大 效果越好， 别问，问就是 论文里面实验下来 就是这个结果
                keep_rate = max(1.0 - self.drop_rate / (20 * 2000) * (self.num_batches_tracked), 1.0 - self.drop_rate)

                # 这个就是dropblock论文中计算  gamma的公式
                gamma = (1 - keep_rate) / self.block_size ** 2 * feat_size ** 2 / (feat_size - self.block_size + 1) ** 2
                out = self.DropBlock(out, gamma=gamma)
            else:
                out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)

        return out

class ResNet(nn.Module):

    def __init__(self, block=BasicBlock, keep_prob=1.0, avg_pool=True, drop_rate=0.1, dropblock_size=5):
        self.inplanes = 3
        super(ResNet, self).__init__()

        self.layer1 = self._make_layer(block, 64, stride=2, drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, 160, stride=2, drop_rate=drop_rate)
        self.layer3 = self._make_layer(block, 320, stride=2, drop_rate=drop_rate, drop_block=True,
                                       block_size=dropblock_size)
        self.layer4 = self._make_layer(block, 640, stride=2, drop_rate=drop_rate, drop_block=True,
                                       block_size=dropblock_size)
        if avg_pool:
            self.avgpool = nn.AvgPool2d(5, stride=1)

        self.keep_prob = keep_prob
        self.keep_avg_pool = avg_pool
        self.dropout = nn.Dropout(p=1 - self.keep_prob, inplace=False)
        self.drop_rate = drop_rate

        # 初始化参数进行的操作
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1):
        downsample = None

        # 当需要特征图需要降维或通道数不匹配的时候调用
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, drop_rate, drop_block, block_size))
        self.inplanes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)  # [80,3,84,84] => [80,64,42,42]
        x = self.layer2(x)  # [80,64,42,42] => [80,160,21,21]
        x = self.layer3(x)  # [80,160,21,21] => [80,320,10,10]
        x = self.layer4(x)  # [80,320,10,10] => [80,640,5,5]
        if self.keep_avg_pool:
            x = self.avgpool(x)  # [80,640,5,5] => [80,640,1,1]
            x = x.view(x.size(0), -1)  # [80,640,1,1] => [80,640]   
        else:
            b, d, h, w = x.shape
            x = x.permute(0, 2, 3, 1).reshape(b, -1, d)
        return x

def Res12(keep_prob=1.0, avg_pool=False, **kwargs):
    """Constructs a ResNet-12 model.
    """
    model = ResNet(BasicBlock, keep_prob=keep_prob, avg_pool=avg_pool, **kwargs)
    return model

if __name__ == '__main__':
    summary(Res12().cuda(), (3, 84, 84))
