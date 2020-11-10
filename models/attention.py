import torch

from models.nature import NatureCNNModel
from models.impala import ImpalaModel

class NatureAttention(NatureCNNModel):
    def forward(self, x):
        conv = self.conv.conv.conv
        softmax = torch.nn.Softmax2d()
        g0 = conv[1](conv[0](x))
        g1 = conv[3](conv[2](g0))
        g2 = conv[5](conv[4](g1))
        
        return [g.pow(2).mean(1) for g in (g0, g1, g2)]
      
class ImpalaAttention(ImpalaModel):
    def forward(self, x):
        conv = imp.conv.conv.blocks
        softmax = torch.nn.Softmax2d()
        g0 = conv[0](x)
        g1 = conv[1](g0)
        g2 = conv[2](g1)
        
        return [g.pow(2).mean(1) for g in (g0, g1, g2)]
      
class SelfAttention(torch.nn.Module):
  def __init__(self, in_channels):
    super().__init__()
    self.q = torch.nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1)
    self.v = torch.nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1)
    self.k = torch.nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1)
    self.y = torch.nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1)
    self.softmax = torch.nn.Softmax2d()
  def forward(self, x):
    f1 = self.q(x)
    f2 = self.v(x)
    g1 = self.k(x)

    z = torch.matmul(f1, f2.transpose(2, 3))
    z = self.softmax(z)
    z = torch.matmul(z, g1)

    yout = self.y(z)

    return yout + x

class ProjectorBlock(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(ProjectorBlock, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=1, padding=0, bias=False)
    def forward(self, inputs):
        return self.conv(inputs)

class LinearAttentionBlock(torch.nn.Module):
    def __init__(self, in_features, normalize_attn=True):
        super(LinearAttentionBlock, self).__init__()
        self.normalize_attn = normalize_attn
        self.op = torch.nn.Conv2d(in_channels=in_features, out_channels=1, kernel_size=1, padding=0, bias=False)
    def forward(self, l, g):
        N, C, W, H = l.size()
        c = self.op(l+g) # batch_sizex1xWxH
        if self.normalize_attn:
            a = torch.nn.functional.softmax(c.view(N,1,-1), dim=2).view(N,1,W,H)
        else:
            a = torch.sigmoid(c)
        g = torch.mul(a.expand_as(l), l)
        if self.normalize_attn:
            g = g.view(N,C,-1).sum(dim=2) # batch_sizexC
        else:
            g = torch.nn.functional.adaptive_avg_pool2d(g, (1,1)).view(N,C)
        return c.view(N,1,W,H), g