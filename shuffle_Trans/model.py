from typing import Callable

import torch
from torch import Tensor
import torch.nn as nn

import copy,math

class MultiHeadedAttention(nn.Module):
    def __init__(self,embedding , head=8, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert embedding % head == 0

        self.embed_cut = embedding // head  # W*H // head
        self.head = head
        self.linears = nn.ModuleList([copy.deepcopy(nn.Linear(embedding, embedding) ) for _ in range(4)])
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        BN,C,WH = query.size()

        query = self.linears[0](query).view(BN, -1, self.head, self.embed_cut).transpose(1, 2)
        key = self.linears[1](key).view(BN, -1, self.head, self.embed_cut).transpose(1, 2)
        value = self.linears[2](value).view(BN, -1, self.head, self.embed_cut).transpose(1, 2)

        embed_cut = query.size(3)
        scores = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(embed_cut)
        productAttention = torch.softmax(scores, dim=-1)
        self.attn = self.dropout(productAttention)
        output = torch.matmul(self.attn, value)

        output = output.transpose(1, 2).contiguous().view(BN, C, WH)
        output = self.linears[3](output)
        return output



class TransformerEncoderBlock(nn.Module):
    def __init__(self, drop_p: float = 0.,expansion: int = 4,** kwargs):
        super(TransformerEncoderBlock, self).__init__()
        self.emb_size = None
        self.drop_p = drop_p
        self.expansion = expansion

    def forward(self, input):
        BN, C, H, W = input.size()
        if not self.emb_size:
            self.emb_size = int(H*W)
            self.create_model()
        input = input.view(BN,C,int(H*W))

        branch_MHA = input
        input = self.LayerNorm1(input)
        input = self.MHA(input,input,input)
        input = self.Dropout1(input)
        input += branch_MHA
        branch_MLP = input
        input = self.LayerNorm2(input)
        input = self.MLP(input)
        input = self.Dropout2(input)
        input += branch_MLP

        output = input.view(BN,C,H,W)
        return output


    def create_model(self):
        self.LayerNorm1 = nn.LayerNorm(self.emb_size)
        self.MHA = MultiHeadedAttention(self.emb_size)
        self.Dropout1 = nn.Dropout(self.drop_p)

        self.LayerNorm2 = nn.LayerNorm(self.emb_size)
        self.MLP = nn.Sequential(
            nn.Linear(self.emb_size, self.expansion * self.emb_size),
            nn.GELU(),
            nn.Dropout(self.drop_p),
            nn.Linear(self.expansion * self.emb_size, self.emb_size),
        )
        self.Dropout2 = nn.Dropout(self.drop_p)




class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # print(b)
        # print(c)
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SKLayer(nn.Module):
    def __init__(self, in_channels):
        super(SKLayer, self).__init__()
        r = 2.0
        L = 32
        d = max(int(in_channels / r), L)
        self.in_channels = in_channels
        self.conv1 = nn.Sequential(
            self.depthwise_conv(in_channels, in_channels, kernel_size=3, dilation=1, padding=1),
            nn.BatchNorm2d(in_channels)
            )
        self.conv2 = nn.Sequential(
            self.depthwise_conv(in_channels, in_channels, kernel_size=3, dilation=2, padding=2),
            nn.BatchNorm2d(in_channels)
            )
        self.fc1 = nn.Linear(in_channels, d)
        self.fc2 = nn.Linear(d, in_channels*2)
        self.softmax = nn.Softmax(dim=1)

    @staticmethod
    def depthwise_conv(i, o, kernel_size, dilation=1, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, dilation=dilation, bias=bias, groups=i)

    def forward(self, x):
        U1 = self.conv1(x)
        U2 = self.conv2(x)
        U = U1 + U2
        S = U.mean(-1).mean(-1)
        Z1 = self.fc1(S)
        Z2 = self.fc2(Z1)
        A = self.softmax(Z2)
        V = U1 * A[:, :self.in_channels].unsqueeze(-1).unsqueeze(-1) + \
            U2 * A[:, self.in_channels:].unsqueeze(-1).unsqueeze(-1)
        return V


def channel_shuffle(x: Tensor, groups: int) -> Tensor:

    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch_size, -1, height, width)

    return x


class InvertedResidual_SE(nn.Module):
    def __init__(self, input_c: int, output_c: int, stride: int):
        super(InvertedResidual_SE, self).__init__()

        if stride not in [1, 2]:
            raise ValueError("illegal stride value.")
        self.stride = stride

        assert output_c % 2 == 0
        branch_features = output_c // 2
        # 当stride为1时，input_channel应该是branch_features的两倍
        # python中 '<<' 是位运算，可理解为计算×2的快速方法
        assert (self.stride != 1) or (input_c == branch_features << 1)

        if self.stride == 2:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(input_c, input_c, kernel_s=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(input_c),
                nn.Conv2d(input_c, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True)
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(input_c if self.stride > 1 else branch_features, branch_features, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_s=3, stride=self.stride, padding=1),
            SELayer(branch_features, reduction=16),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True)
        )

    @staticmethod
    def depthwise_conv(input_c: int,output_c: int,kernel_s: int,stride: int = 1,padding: int = 0,bias: bool = False) -> nn.Conv2d:
        return nn.Conv2d(in_channels=input_c, out_channels=output_c, kernel_size=kernel_s,stride=stride, padding=padding, bias=bias, groups=input_c)

    def forward(self, x: Tensor) -> Tensor:
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)
        return out


class InvertedResidual_SK(nn.Module):
    def __init__(self, input_c: int, output_c: int, stride: int):
        super(InvertedResidual_SK, self).__init__()

        if stride not in [1, 2]:
            raise ValueError("illegal stride value.")
        self.stride = stride

        assert output_c % 2 == 0
        branch_features = output_c // 2
        assert (self.stride != 1) or (input_c == branch_features << 1)

        if self.stride == 2:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(input_c, input_c, kernel_s=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(input_c),
                nn.Conv2d(input_c, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True)
            )
        else:
            self.branch1 = nn.Sequential()

        if self.stride > 1:
            self.branch2 = nn.Sequential(
                nn.Conv2d(input_c if (self.stride > 1) else branch_features,
                          branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
                self.depthwise_conv(branch_features, branch_features, kernel_s=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(branch_features),
                nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch2 = nn.Sequential(
                nn.Conv2d(input_c if (self.stride > 1) else branch_features,branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
                SKLayer(branch_features),
                nn.BatchNorm2d(branch_features),
                nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )

    @staticmethod
    def depthwise_conv(input_c: int,output_c: int,kernel_s: int,stride: int = 1,padding: int = 0,bias: bool = False) -> nn.Conv2d:
        return nn.Conv2d(in_channels=input_c, out_channels=output_c, kernel_size=kernel_s,stride=stride, padding=padding, bias=bias, groups=input_c)

    def forward(self, x: Tensor) -> Tensor:
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out


class InvertedResidual_Trans(nn.Module):
    def __init__(self, input_c: int, output_c: int, stride: int):
        super(InvertedResidual_Trans, self).__init__()

        if stride not in [1, 2]:
            raise ValueError("illegal stride value.")
        self.stride = stride

        assert output_c % 2 == 0
        branch_features = output_c // 2
        assert (self.stride != 1) or (input_c == branch_features << 1)

        if self.stride == 2:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(input_c, input_c, kernel_s=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(input_c),
                nn.Conv2d(input_c, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True)
            )
        else:
            self.branch1 = nn.Sequential()

        if self.stride > 1:
            self.branch2 = nn.Sequential(
                nn.Conv2d(input_c if (self.stride > 1) else branch_features,
                          branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
                self.depthwise_conv(branch_features, branch_features, kernel_s=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(branch_features),
                nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch2 = nn.Sequential(
                nn.Conv2d(input_c if (self.stride > 1) else branch_features,branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
                TransformerEncoderBlock(),  # 嗯，理论可行，当我感觉周围没必要加这么多了
                nn.BatchNorm2d(branch_features),
                nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )

    @staticmethod
    def depthwise_conv(input_c: int,output_c: int,kernel_s: int,stride: int = 1,padding: int = 0,bias: bool = False) -> nn.Conv2d:
        return nn.Conv2d(in_channels=input_c, out_channels=output_c, kernel_size=kernel_s,stride=stride, padding=padding, bias=bias, groups=input_c)

    def forward(self, x: Tensor) -> Tensor:
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out



class InvertedResidual(nn.Module):
    def __init__(self, input_c: int, output_c: int, stride: int):
        super(InvertedResidual, self).__init__()

        if stride not in [1, 2]:
            raise ValueError("illegal stride value.")
        self.stride = stride

        assert output_c % 2 == 0
        branch_features = output_c // 2
        assert (self.stride != 1) or (input_c == branch_features << 1)

        if self.stride == 2:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(input_c, input_c, kernel_s=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(input_c),
                nn.Conv2d(input_c, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True)
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(input_c if self.stride > 1 else branch_features, branch_features, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_s=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True)
        )

    @staticmethod
    def depthwise_conv(input_c: int,
                       output_c: int,
                       kernel_s: int,
                       stride: int = 1,
                       padding: int = 0,
                       bias: bool = False) -> nn.Conv2d:
        return nn.Conv2d(in_channels=input_c, out_channels=output_c, kernel_size=kernel_s,
                         stride=stride, padding=padding, bias=bias, groups=input_c)

    def forward(self, x: Tensor) -> Tensor:
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out


class ShuffleNetV2(nn.Module):
    def __init__(self,
                 stages_repeats: [4, 8, 4],
                 stages_out_channels: [24, 116, 232, 464, 1024],
                 num_classes: int = 1000,
                 inverted_residual: Callable[..., nn.Module] = InvertedResidual_Trans):  # 改这里
        super(ShuffleNetV2, self).__init__()

        if len(stages_repeats) != 3:
            raise ValueError("expected stages_repeats as list of 3 positive ints")
        if len(stages_out_channels) != 5:
            raise ValueError("expected stages_out_channels as list of 5 positive ints")
        self._stage_out_channels = stages_out_channels

        # input RGB image
        input_channels = 3
        output_channels = self._stage_out_channels[0]

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Static annotations for mypy
        self.stage2: nn.Sequential
        self.stage3: nn.Sequential
        self.stage4: nn.Sequential

        stage_names = ["stage{}".format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(stage_names, stages_repeats,
                                                  self._stage_out_channels[1:]):
            seq = [inverted_residual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(inverted_residual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

        self.fc = nn.Linear(output_channels, num_classes)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = x.mean([2, 3])  # global pool
        x = self.fc(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def shufflenet_v2_x1_0(num_classes=1000):
    """
    Constructs a ShuffleNetV2 with 1.0x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`.
    weight: https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth

    :param num_classes:
    :return:
    """
    model = ShuffleNetV2(stages_repeats=[4, 8, 4],
                         stages_out_channels=[24, 116, 232, 464, 1024],
                         num_classes=num_classes)

    return model


def shufflenet_v2_x0_5(num_classes=1000):
    """
    Constructs a ShuffleNetV2 with 0.5x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`.
    weight: https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth

    :param num_classes:
    :return:
    """
    model = ShuffleNetV2(stages_repeats=[4, 8, 4],
                         stages_out_channels=[24, 48, 96, 192, 1024],
                         num_classes=num_classes)

    return model


if __name__ == '__main__':
    model = shufflenet_v2_x1_0(10)
    # 注意长宽192这样的(64倍数)不行,得至少128倍数，
    # 可能是一共5次下采样(缩了32倍)，导致深层的W*H满足不了多头自注意力那W*H整除head(默认为8,调低可无视)
    input = torch.randn(1, 3, 128, 128)
    output = model(input)
    print('input size {} output size {}'.format(input.size(),output.size()))

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()
    writer.add_graph(model,input)
    writer.close()