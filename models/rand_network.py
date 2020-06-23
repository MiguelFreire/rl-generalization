import torch
from torch import nn

class RandNetwork(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1, stride=1)
        #parameters are frozen
        self.requires_grad_(requires_grad=False)
    @torch.no_grad()
    def init_weights(self):
        def weights_initializer(m):
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)
        
        self.apply(weights_initializer)

    def forward(self, x):
        self.init_weights()
        return self.conv(x)


def random_convolution(imgs):
    '''
    random covolution in "network randomization"
    
    (imbs): B x (C x stack) x H x W, note: imgs should be normalized and torch tensor
    '''
    
    # initialize random covolution
    rand_conv = torch.nn.Conv2d(3, 3, kernel_size=3, bias=False, padding=1)
    rand_conv.weight.requires_grad = False
    if imgs.is_cuda:
          rand_conv.cuda()
    for i, img in enumerate(imgs):
      torch.nn.init.xavier_normal_(rand_conv.weight.data, gain=0.5)
      imgs[i] = rand_conv(img.unsqueeze(0))[0]
    return imgs