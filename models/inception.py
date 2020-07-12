import torch
from rlpyt.models.utils import conv2d_output_shape
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims


class InceptionBlock(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        conv0 = torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1)
        conv2 = torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1)

        non_linearlity = torch.nn.ReLU()

        self.block = torch.nn.Sequential(*[non_linearlity, conv1, non_linearlity, conv2])
    def forward(self, x):
        return self.block(x) + x

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