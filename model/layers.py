import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict


class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, i, num_layer=2, pool=None):
        super(ConvBlock, self).__init__()

        self.i = i
        self.pool = pool

        self.BottleneckBlock = nn.Sequential(
            OrderedDict(
                [
                    (f'conv{i}_1', nn.Conv1d(in_dim, out_dim, 3, 1, 1)),
                    (f'relu{i}_1', nn.ReLU()),
                    (f'conv{i}_2', nn.Conv1d(out_dim, out_dim, 3, 1, 1)),
                    (f'relu{i}_2', nn.ReLU())
                ]
            )
        )
        if num_layer == 3:
            self.BottleneckBlock.add_module(f'conv{i}_3', nn.Conv1d(out_dim, out_dim, 3, 1, 1))
            self.BottleneckBlock.add_module(f'relu{i}_3', nn.ReLU())

        if pool == 'avg':
            self.BottleneckBlock.add_module(f'avgpool{i}', nn.AvgPool1d(3, 2, 1))
        else:
            pass

    def forward(self, x):
        x = self.BottleneckBlock(x)

        if self.pool == 'max':
            x = F.max_pool1d(x, 3, 2, 1)
        else:
            pass

        return x

class CBAMBlock(nn.Module):
    def __init__(self, in_dim, i, n):
        super(CBAMBlock, self).__init__()

        self.i = i
        self.n = n

        self.ChannelAttention = nn.ModuleDict(
            {
                f'cbam{i}_{n}_c_avg': nn.AdaptiveAvgPool1d(1),
                # f'cbam{i}_{n}_c_max': nn.AdaptiveMaxPool1d(1),
                f'cbam{i}_{n}_c_conv1': nn.Conv1d(in_dim, in_dim//8, kernel_size=1, bias=False),
                f'cbam{i}_{n}_c_relu_avg': nn.ReLU(),
                f'cbam{i}_{n}_c_relu_max': nn.ReLU(),
                f'cbam{i}_{n}_c_conv2': nn.Conv1d(in_dim//8, in_dim, kernel_size=1, bias=False),
                f'cbam{i}_{n}_c_sigmoid': nn.Sigmoid()
            }
        )

        self.SpatialAttention = nn.ModuleDict(
            {
                f'cbam{i}_{n}_s_conv': nn.Conv1d(2, 1, kernel_size=7, padding=7//2, bias=False),
                f'cbam{i}_{n}_s_sigmoid': nn.Sigmoid()
            }
        )

    def forward(self, x):
        # channel attention
        avg_x = self.ChannelAttention[f'cbam{self.i}_{self.n}_c_avg'](x)
        avg_x = self.ChannelAttention[f'cbam{self.i}_{self.n}_c_conv1'](avg_x)
        avg_x = self.ChannelAttention[f'cbam{self.i}_{self.n}_c_relu_avg'](avg_x)
        avg_x = self.ChannelAttention[f'cbam{self.i}_{self.n}_c_conv2'](avg_x)
        # max_x = self.ChannelAttention[f'cbam{self.i}_{self.n}_c_max'](x)
        max_x = F.adaptive_max_pool1d(x, 1)
        max_x = self.ChannelAttention[f'cbam{self.i}_{self.n}_c_conv1'](max_x)
        max_x = self.ChannelAttention[f'cbam{self.i}_{self.n}_c_relu_max'](max_x)
        max_x = self.ChannelAttention[f'cbam{self.i}_{self.n}_c_conv2'](max_x)
        c_x = avg_x + max_x
        c_w = self.ChannelAttention[f'cbam{self.i}_{self.n}_c_sigmoid'](c_x)
        x = c_w * x

        # spatial attention
        avg_x = torch.mean(x, dim=1, keepdim=True)
        max_x, _ = torch.max(x, dim=1, keepdim=True)
        s_x = torch.cat([avg_x, max_x], dim=1)
        s_x = self.SpatialAttention[f'cbam{self.i}_{self.n}_s_conv'](s_x)
        s_w = self.SpatialAttention[f'cbam{self.i}_{self.n}_s_sigmoid'](s_x)
        x = s_w * x

        return x


class ResidualCBAMLayer(nn.Module):
    def __init__(self, in_dim, out_dim, i, n):
        super(ResidualCBAMLayer, self).__init__()

        self.i = i
        self.n = n

        self.ResidualCBAM = nn.ModuleDict(
            {
                f'conv{i}_{n}_1': nn.Conv1d(in_dim, out_dim, 3, 1, 1),
                f'bn{i}_{n}_1': nn.BatchNorm1d(out_dim),
                f'relu{i}_{n}_1': nn.ReLU(),
                f'conv{i}_{n}_2': nn.Conv1d(out_dim, out_dim, 3, 1, 1),
                f'bn{i}_{n}_2': nn.BatchNorm1d(out_dim),
                f'cbam{i}_{n}': CBAMBlock(out_dim, i, n),
                f'relu{i}_{n}_2': nn.ReLU()
            }
        )

        self.residual_conv = False
        if in_dim != out_dim:
            self.residual_conv = True
            self.ResidualCBAM[f'res_conv{i}_{n}'] = nn.Conv1d(in_dim, out_dim, kernel_size=1, stride=1)
            self.ResidualCBAM[f'res_bn{i}_{n}'] = nn.BatchNorm1d(out_dim)

    def forward(self, x):
        x_skip = x
        if self.residual_conv:
            x_skip = self.ResidualCBAM[f'res_conv{self.i}_{self.n}'](x_skip)
            x_skip = self.ResidualCBAM[f'res_bn{self.i}_{self.n}'](x_skip)

        for layer_n, layer_name in enumerate(self.ResidualCBAM):
            if layer_n == 6:
                break
            x = self.ResidualCBAM[f'{layer_name}'](x)

        x += x_skip
        x = self.ResidualCBAM[f'relu{self.i}_{self.n}_2'](x)

        return x


class ResidualCBAMLayer_PreAct(nn.Module):
    def __init__(self, in_dim, out_dim, i, n):
        super(ResidualCBAMLayer_PreAct, self).__init__()

        self.i = i
        self.n = n

        self.ResidualCBAM_PreAct = nn.ModuleDict(
            {
                f'bn{i}_{n}_1': nn.BatchNorm1d(out_dim),
                f'relu{i}_{n}_1': nn.ReLU(),
                f'conv{i}_{n}_1': nn.Conv1d(in_dim, out_dim, 3, 1, 1),
                f'bn{i}_{n}_2': nn.BatchNorm1d(out_dim),
                f'relu{i}_{n}_2': nn.ReLU(),
                f'conv{i}_{n}_2': nn.Conv1d(out_dim, out_dim, 3, 1, 1),
                f'cbam{i}_{n}': CBAMBlock(out_dim, i, n)
            }
        )

        self.residual_conv = False
        if in_dim != out_dim:
            self.residual_conv = True
            self.ResidualCBAM_PreAct[f'res_bn{i}_{n}'] = nn.BatchNorm1d(out_dim)
            self.ResidualCBAM_PreAct[f'res_conv{i}_{n}'] = nn.Conv1d(in_dim, out_dim, kernel_size=1, stride=1)

    def forward(self, x):
        x_skip = x
        if self.residual_conv:
            x_skip = self.ResidualCBAM_PreAct[f'res_bn{self.i}_{self.n}'](x_skip)
            x_skip = self.ResidualCBAM_PreAct[f'res_conv{self.i}_{self.n}'](x_skip)

        for layer_name in self.ResidualCBAM_PreAct:
            x = self.ResidualCBAM_PreAct[f'{layer_name}'](x)

        x += x_skip

        return x


class ResidualCBAMBlock(nn.Module):
    def __init__(self, in_dim, out_dim, i, num_layer=2, preact=False, pool=None):
        super(ResidualCBAMBlock, self).__init__()

        self.pool = pool
        self.BottleneckBlock = nn.Sequential()
        for n in range(num_layer):
            if preact:
                self.BottleneckBlock.add_module(
                    f'residual_cbam{i}_{n+1}', ResidualCBAMLayer_PreAct(in_dim, out_dim, i, n+1)
                )
            else:
                self.BottleneckBlock.add_module(
                    f'residual_cbam{i}_{n+1}', ResidualCBAMLayer(in_dim, out_dim, i, n+1)
                )
            in_dim = out_dim

        if pool == 'avg':
            self.BottleneckBlock.add_module(f'avgpool{i}', nn.AvgPool1d(3, 2, 1))
        else:
            pass

    def forward(self, x):
        x = self.BottleneckBlock(x)

        if self.pool == 'max':
            x = F.max_pool1d(x, 3, 2, 1)
        else:
            pass

        return x