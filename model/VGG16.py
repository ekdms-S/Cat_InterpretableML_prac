from model.layers import *


class VGG16(nn.Module):
    def __init__(self,
                 input_dim = 2,
                 conv_dim = [64, 128, 256, 512, 512],
                 fc_dim = [256, 128, 64],
                 num_out=3):
        super(VGG16, self).__init__()

        self.convs = nn.Sequential()
        in_dim = input_dim
        for n_block in range(5):
            pool_type = None if n_block == 4 else 'max'
            n_layers = 2 if n_block < 2 else 3
            out_dim = conv_dim[n_block]
            self.convs.add_module(
                f'conv{n_block+1}',
                ConvBlock(in_dim, out_dim, i=n_block+1, num_layer=n_layers, pool=pool_type)
            )
            in_dim = out_dim

        self.adaptiveavgpool = nn.AdaptiveAvgPool1d(1)

        self.num_out = num_out
        self.fc_layers = nn.Sequential()
        in_dim = conv_dim[-1]
        for n_layer in range(len(fc_dim)):
            self.fc_layers.add_module(
                f'linear{n_layer + 1}', nn.Linear(in_dim, fc_dim[n_layer])
            )
            self.fc_layers.add_module(
                f'relu_lin{n_layer + 1}', nn.ReLU()
            )
            in_dim = fc_dim[n_layer]

        self.out1 = nn.Linear(in_features=fc_dim[-1], out_features=1)
        self.out2 = nn.Linear(in_features=fc_dim[-1], out_features=1)
        self.out3 = nn.Linear(in_features=fc_dim[-1], out_features=1)
        if num_out > 3:
            self.out4 = nn.Linear(in_features=fc_dim[-1], out_features=1)

    def forward(self, input):
        x = self.convs(input)

        x = self.adaptiveavgpool(x)
        x = torch.squeeze(x)

        x = self.fc_layers(x)
        y1 = self.out1(x)
        y2 = self.out2(x)
        y3 = self.out3(x)
        if self.num_out == 3:
            y = torch.stack([y1, y2, y3], dim=1)
        else:
            y4 = self.out4(x)
            y = torch.stack([y1, y2, y3, y4], dim=1)

        return y