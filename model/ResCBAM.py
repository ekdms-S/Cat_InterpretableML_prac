from model.layers import *


class ResCBAM(nn.Module):
    def __init__(self,
                 input_dim=2,
                 num_blocks=4,
                 conv_dim=[64, 128, 256, 512],
                 fc_dim=[256, 128, 64],
                 num_out=3):
        super(ResCBAM, self).__init__()

        self.conv0 = nn.Conv1d(in_channels=input_dim, out_channels=conv_dim[0],
                               kernel_size=20, stride=2, padding=9)
        self.bn0 = nn.BatchNorm1d(num_features=conv_dim[0])
        self.relu0 = nn.ReLU()
        self.avgpool0 = nn.AvgPool1d(3, 2, 1)

        self.residual_cbams = nn.Sequential()
        in_dim, out_dim = conv_dim[0], conv_dim[0]
        for n_block in range(num_blocks):
            pool_type = None if n_block == 3 else 'max'
            if n_block == 0:
                out_dim = conv_dim[0]
            else:
                out_dim = conv_dim[n_block]
            self.residual_cbams.add_module(
                f'residual_cbam{n_block + 1}',
                ResidualCBAMBlock(in_dim, out_dim, i=n_block+1, num_layer=3, pool=pool_type)
            )
            in_dim = out_dim

        self.adaptiveavgpool = nn.AdaptiveAvgPool1d(1)

        self.num_out = num_out
        self.fc_layers = nn.Sequential()
        in_dim = conv_dim[-1]
        for n_layer in range(len(fc_dim)):
            self.fc_layers.add_module(
                f'linear{n_layer+1}', nn.Linear(in_dim, fc_dim[n_layer])
            )
            self.fc_layers.add_module(
                f'relu_lin{n_layer+1}', nn.ReLU()
            )
            in_dim = fc_dim[n_layer]

        self.out1 = nn.Linear(in_features=fc_dim[-1], out_features=1)
        self.out2 = nn.Linear(in_features=fc_dim[-1], out_features=1)
        self.out3 = nn.Linear(in_features=fc_dim[-1], out_features=1)
        if num_out > 3:
            self.out4 = nn.Linear(in_features=fc_dim[-1], out_features=1)

    def forward(self, input):
        x = self.conv0(input)
        x = self.bn0(x)
        x = self.relu0(x)
        x = self.avgpool0(x)

        x = self.residual_cbams(x)

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