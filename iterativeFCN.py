"""
Iterative FCN for automatic vertebra segmentation (MIDL 2018).
"""
import torch
import torch.nn as nn


class IterativeFCN(nn.Module):
    """
    3D U-Net-style model with constant channel width and a completeness classifier.

    Input:  (N, 2, D, H, W)  — image patch + instance memory
    Output: seg (N, 1, D, H, W), cls (N,) completeness score in [0, 1]

    Note: encoder/decoder blocks intentionally share weights (matches the
    released pretrained checkpoint layout).
    """

    def consecutive_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(out_channels),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(out_channels),
        )

    def __init__(self, num_channels=64):
        super(IterativeFCN, self).__init__()

        self.conv_initial = self.consecutive_conv(2, num_channels)
        self.conv_final = self.consecutive_conv(num_channels, 1)
        self.conv_rest = self.consecutive_conv(num_channels, num_channels)
        self.conv_up = self.consecutive_conv(num_channels * 2, num_channels)

        self.contract = nn.MaxPool3d(2, stride=2)
        self.expand = nn.Upsample(scale_factor=2)
        self.dense = nn.Linear(num_channels, 1)

    def forward(self, x):
        # Expect spatial size divisible by 16 (default training patch = 128^3)
        x_128 = self.conv_rest(self.conv_initial(x))
        x_64 = self.contract(x_128)

        x_64 = self.conv_rest(x_64)
        x_32 = self.contract(x_64)

        x_32 = self.conv_rest(x_32)
        x_16 = self.contract(x_32)

        x_16 = self.conv_rest(x_16)

        u_32 = self.conv_up(torch.cat((x_32, self.expand(x_16)), 1))
        u_64 = self.conv_up(torch.cat((x_64, self.expand(u_32)), 1))
        u_128 = self.conv_up(torch.cat((x_128, self.expand(u_64)), 1))
        u_128 = self.conv_final(u_128)

        x_8 = self.conv_rest(self.contract(x_16))
        x_4 = self.conv_rest(self.contract(x_8))
        x_2 = self.conv_rest(self.contract(x_4))
        x_1 = self.contract(x_2)

        # Global pooled features -> completeness score
        pooled = torch.flatten(x_1, 1)
        seg = torch.sigmoid(u_128)
        cls = torch.sigmoid(self.dense(pooled)).view(-1)
        return seg, cls
