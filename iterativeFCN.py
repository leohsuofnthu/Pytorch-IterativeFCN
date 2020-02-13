import torch
import torch.nn as nn

class IterativeFCN(nn.Module):
    """
    Structure of Iterative FCN Model 
    """

    def consecutive_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(out_channels),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(out_channels))

    def __init__(self, num_channels=16):
        super(IterativeFCN, self).__init__()

        self.conv_initial = self.consecutive_conv(2, num_channels)

        self.conv_rest_1 = self.consecutive_conv(num_channels, num_channels)
        self.conv_rest_2 = self.consecutive_conv(num_channels, num_channels)
        self.conv_rest_3 = self.consecutive_conv(num_channels, num_channels)
        self.conv_rest_4 = self.consecutive_conv(num_channels, num_channels)
        self.conv_rest_5 = self.consecutive_conv(num_channels, num_channels)
        self.conv_rest_6 = self.consecutive_conv(num_channels, num_channels)
        self.conv_rest_7 = self.consecutive_conv(num_channels, num_channels)

        self.conv_up_1 = self.consecutive_conv(num_channels * 2, num_channels)
        self.conv_up_2 = self.consecutive_conv(num_channels * 2, num_channels)
        self.conv_up_3 = self.consecutive_conv(num_channels * 2, num_channels)

        self.conv_final = self.consecutive_conv(num_channels, 1)

        self.contract = nn.MaxPool3d(2, stride=2)
        self.expand = nn.Upsample(scale_factor=2)

        self.dense = nn.Linear(num_channels*2, 1)

    def forward(self, x):
        # 2*128*128*128 to 64*128*128*128
        x_128 = self.conv_initial(x)

        # 64*128*128*128 to 64*64*64*64
        x_128 = self.conv_rest_1(x_128)
        x_64 = self.contract(x_128)

        # 64*64*64*64 to 64*32*32*32
        x_64 = self.conv_rest_2(x_64)
        x_32 = self.contract(x_64)

        # 64*32*32*32 to 64*16*16*16
        x_32 = self.conv_rest_3(x_32)
        x_16 = self.contract(x_32)

        # 64*16*16*16 to 64*8*8*8
        x_16 = self.conv_rest_4(x_16)

        # upsmapling path
        u_32 = self.expand(x_16)
        u_32 = self.conv_up_1(torch.cat((x_32, u_32), 1))

        u_64 = self.expand(u_32)
        u_64 = self.conv_up_2(torch.cat((x_64, u_64), 1))

        u_128 = self.expand(u_64)
        u_128 = self.conv_up_3(torch.cat((x_128, u_128), 1))

        u_128 = self.conv_final(u_128)

        # classification path
        x_8 = self.conv_rest_5(self.contract(x_16))
        x_4 = self.conv_rest_6((self.contract(x_8)))

        x_2 = self.conv_rest_7(self.contract(x_4))
        x_1 = self.contract(x_2)

        cls = torch.sigmoid(self.dense(torch.flatten(x_1)))
        seg = torch.sigmoid(u_128)

        return seg, cls
