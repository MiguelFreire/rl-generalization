import torch
from torch.nn import functional as F
from rlpyt.models.utils import conv2d_output_shape
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels, layers=2, kernel_size=3, stride=1, padding=1):
        super().__init__()

        non_linearlity = torch.nn.ReLU()

        if layers == 2:
            conv1 = torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=stride, padding=padding)
            conv2 = torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=stride, padding=padding)

            self.block = torch.nn.Sequential(*[conv1, non_linearlity, conv2])
        else:
            conv = torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=stride, padding=padding)
            self.block = conv

        
    def forward(self, x):
        return F.relu(self.block(x) + x)

    def conv_out_size(self, h, w, c=None):
        """Helper function ot return the output size for a given input shape,
        without actually performing a forward pass through the model."""
        for child in self.block.children():
            try:
                h, w = conv2d_output_shape(h, w, child.kernel_size,
                    child.stride, child.padding)
            except AttributeError:
                pass  # Not a conv or maxpool layer.
            try:
                c = child.out_channels
            except AttributeError:
                pass  # Not a conv layer.
        return h, w, c

class ImpalaBlock(torch.nn.Module):
    def __init__(self, in_channels, channels):
        super().__init__()
        conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=channels, kernel_size=3, stride=1, padding=1)
        maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        resBlock1 = ResidualBlock(channels)
        resBlock2 = ResidualBlock(channels)

        self.block = torch.nn.Sequential(*[conv1, maxpool, resBlock1, resBlock2])

    def forward(self, x):
        return self.block(x)

    def conv_out_size(self, h, w, c=None):
        """Helper function ot return the output size for a given input shape,
        without actually performing a forward pass through the model."""
        for child in self.block.children():
            if isinstance(child, torch.nn.Conv2d):
                h, w = conv2d_output_shape(h, w, child.kernel_size,
                    child.stride, child.padding)
                c = child.out_channels
            elif isinstance(child, torch.nn.MaxPool2d):
                h, w = conv2d_output_shape(h, w, child.kernel_size,
                    child.stride, child.padding)
            elif isinstance(child, ResidualBlock):
                h, w, c = child.conv_out_size(h,w,c)
            
        return h, w, c

class ImpalaCNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        impala_blocks = ([ImpalaBlock(c_in,c_out) for (c_in, c_out) in zip(in_channels, out_channels)])

        self.blocks = torch.nn.Sequential(*impala_blocks)
        
    def forward(self,x):
        return self.blocks(x)

    def conv_out_size(self, h, w, c=None):
        for child in self.blocks.children():
            h, w, c = child.conv_out_size(h, w, c)
        
        return h * w * c

class ImpalaHead(torch.nn.Module):
    def __init__(
            self, 
            image_shape, 
            in_channels, 
            out_channels,
            hidden_size
            ):
        super().__init__()
        c, h, w = image_shape

        self.conv = ImpalaCNN(in_channels, out_channels)
        conv_output_size = self.conv.conv_out_size(h,w,c)
        print(conv_output_size)
        self.head = torch.nn.Linear(conv_output_size, hidden_size)
        self.output_size = hidden_size
    def forward(self,x):
        y = torch.nn.functional.relu(self.conv(x))
        y = self.head(y.view(x.shape[0], -1))
        return torch.nn.functional.relu(y)

class ImpalaModel(torch.nn.Module):
    def __init__(
            self, 
            image_shape,
            output_size,
            in_channels, 
            out_channels,
            hidden_size
            ):
        super().__init__()
        c, h, w = image_shape

        self.conv = ImpalaHead(image_shape, in_channels, out_channels, hidden_size)

        self.pi = torch.nn.Linear(self.conv.output_size, output_size) 
        #value function head 
        self.value = torch.nn.Linear(self.conv.output_size, 1)

    def forward(self, image, prev_action, prev_reward):
        #input normalization, cast to float then grayscale it
        img = image.type(torch.float)
        img = img.mul_(1. / 255)
        
        lead_dim, T, B, img_shape = infer_leading_dims(img, 3)
        
        fc_out = self.conv(img.view(T * B, *img_shape))
        pi = torch.nn.functional.softmax(self.pi(fc_out), dim=-1)
        v = self.value(fc_out).squeeze(-1)
        # Restore leading dimensions: [T,B], [B], or [], as input.
        #T -> transition
        #B -> batch_size?
        pi, v = restore_leading_dims((pi, v), lead_dim, T, B)

        return pi, v