import torch
from torch import nn
import numpy as np


class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(nn.AvgPool2d(2)(x1))
        x3 = self.conv3(nn.AvgPool2d(2)(x2))
        return [x1, x2, x3] # listで返しているので

def main():
    model = TestModel()
    input = torch.randn(4, 3, 16, 16) # batch_size=4, channel=3, x=y=16
    out = model(input) # 出力はtorch.Tensorのlist
    print(type(out))
    for x in out:
        print(x.size(), type(x))

if __name__ == "__main__":
    # # main()
    # mat1 = torch.cat([torch.arange(2*3).reshape((1, 2, 3)), torch.arange(2*3).reshape((1, 2, 3))*2])
    # mat2 = torch.cat([torch.ones(2*3).reshape((1, 3, 2)), torch.ones(2*3).reshape((1, 3, 2))*2])
    # torch.mm(mat1, mat2)
    # torch.tensor([[ 0.4851,  0.5037, -0.3633],
    #               [-0.0760, -3.6705,  2.4784]])

    a = torch.zeros(3, 2, 5, 6)
    b = torch.zeros(3, 2, 5, 6)
    a[1,:,3,4] = torch.tensor([0, 1])
    a[1,:,2,5] = torch.tensor([-np.sqrt(3), 1])
    b[1,:,1,3] = torch.tensor([1, 0])
    b[1,:,1,5] = torch.tensor([1, np.sqrt(3)])


    batch_size, channel_size, h, w = a.size()
    a_disamb = a.permute(0,2,3,1).view(batch_size, -1, channel_size)
    norm = torch.norm(a_disamb, dim=2).unsqueeze(-1)
    norm[norm == 0] = 1
    a_disamb = a_disamb / norm  # Don't set the operator as /= in order not to change original data of image feature.
    
    b_disamb = b.permute(0,2,3,1).view(batch_size, -1, channel_size)
    norm = torch.norm(b_disamb, dim=2).unsqueeze(-1)
    norm[norm == 0] = 1
    b_disamb = b_disamb / norm  # Don't set the operator as /= in order not to change original data of image feature.
    ab = torch.bmm(a_disamb, b_disamb.permute(0,2,1))

    weight = ab.view(batch_size,h,w,h,w)
    # weighted = weight.unsqueeze(3) * b.unsqueeze(1).unsqueeze(1)
    # a += weighted.sum(dim=(4,5)).permute(0,3,1,2)


    weighted_b = weight.unsqueeze(-1) * b.permute(0,2,3,1).unsqueeze(1).unsqueeze(1)
    mixed = a + weighted_b.sum(dim=(3,4)).permute(0,3,1,2)
    
    from ssd import Attn

    attn = Attn(weight_activation=None)
    mixed2, weight2 = attn.attn_func(a, b)
    
    
    print()
    
