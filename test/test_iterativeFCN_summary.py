import torch
from torchsummary import summary
from iterativeFCN import IterativeFCN

# Test Purpose
model = IterativeFCN(num_channels=11)
if torch.cuda.is_available():
    model.cuda()
summary(model, (2, 128, 128, 128))
