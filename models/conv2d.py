import torch

from rlpyt.models.mlp import MlpModel
from rlpyt.models.utils import conv2d_output_shape
from models.impala import ResidualBlock

class Conv2dModel(torch.nn.Module):
    """2-D Convolutional model component, with option for max-pooling vs
    downsampling for strides > 1.  Requires number of input channels, but
    not input shape.  Uses ``torch.nn.Conv2d``.
    """

    def __init__(
            self,
            in_channels,
            channels,
            kernel_sizes,
            strides,
            paddings=None,
            nonlinearity=torch.nn.ReLU,  # Module, not Functional.
            use_maxpool=False,  # if True: convs use stride 1, maxpool downsample.
            head_sizes=None,  # Put an MLP head on top.
            batchNorm=False,
            dropout=0.0
            ):
        super().__init__()
        if paddings is None:
            paddings = [0 for _ in range(len(channels))]
        assert len(channels) == len(kernel_sizes) == len(strides) == len(paddings)
        in_channels = [in_channels] + channels[:-1]
        ones = [1 for _ in range(len(strides))]
        if use_maxpool:
            maxp_strides = strides
            strides = ones
        else:
            maxp_strides = ones
        conv_layers = [torch.nn.Conv2d(in_channels=ic, out_channels=oc,
            kernel_size=k, stride=s, padding=p) for (ic, oc, k, s, p) in
            zip(in_channels, channels, kernel_sizes, strides, paddings)]
        sequence = list()
        for conv_layer, maxp_stride in zip(conv_layers, maxp_strides):
            sequence.extend([conv_layer, nonlinearity()])
            if batchNorm:
                sequence.append(torch.nn.BatchNorm2d(conv_layer.out_channels))
            if dropout > 0.0:
                sequence.append(torch.nn.Dropout2d(dropout))
            if maxp_stride > 1:
                sequence.append(torch.nn.MaxPool2d(maxp_stride))  # No padding.
        self.conv = torch.nn.Sequential(*sequence)

    def forward(self, input):
        """Computes the convolution stack on the input; assumes correct shape
        already: [B,C,H,W]."""
        return self.conv(input)

    def conv_out_size(self, h, w, c=None):
        """Helper function ot return the output size for a given input shape,
        without actually performing a forward pass through the model."""
        for child in self.conv.children():
            try:
                h, w = conv2d_output_shape(h, w, child.kernel_size,
                    child.stride, child.padding)
            except AttributeError:
                pass  # Not a conv or maxpool layer.
            try:
                c = child.out_channels
            except AttributeError:
                pass  # Not a conv layer.
        return h * w * c

class Conv2dResModel(torch.nn.Module):
    def __init__(self, in_channels, layers=2, use_maxpool=False):
        super().__init__()
        seq = list()
        conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=32,
            kernel_size=8, stride=4, padding=0)
        seq.append(conv1)
        
        if use_maxpool:
          max_pool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
          seq.append(max_pool)
        
        resBlock = ResidualBlock(in_channels=32, channels=64, layers=layers, kernel_size=3, stride=1, padding=1, useConv1x1=True)
        resBlock2 = ResidualBlock(in_channels=64, channels=64, layers=layers, kernel_size=3, stride=1, padding=1)
        
        seq.append(resBlock)
        seq.append(resBlock2)
        
        
        self.conv = torch.nn.Sequential(*seq)

    def forward(self, x):
        return self.conv(x)

    def conv_out_size(self, h, w, c=None):
        """Helper function ot return the output size for a given input shape,
        without actually performing a forward pass through the model."""
        for child in self.conv.children():
            if isinstance(child, torch.nn.Conv2d):
                h, w = conv2d_output_shape(h, w, child.kernel_size,
                    child.stride, child.padding)
                c = child.out_channels
            elif isinstance(child, torch.nn.MaxPool2d):
                h, w = conv2d_output_shape(h, w, child.kernel_size,
                    child.stride, child.padding)
            elif isinstance(child, ResidualBlock):
                h, w, c = child.conv_out_size(h,w,c)
            print(h,w,c)
        return h * w * c

class Conv2dHeadModel(torch.nn.Module):
    """Model component composed of a ``Conv2dModel`` component followed by 
    a fully-connected ``MlpModel`` head.  Requires full input image shape to
    instantiate the MLP head.
    """

    def __init__(
            self,
            image_shape,
            channels,
            kernel_sizes,
            strides,
            hidden_sizes,
            output_size=None,  # if None: nonlinearity applied to output.
            paddings=None,
            nonlinearity=torch.nn.ReLU,
            use_maxpool=False,
            batchNorm=False,
            dropout=0.0,
            useResNet=False,
            resNetLayers=2,
            ):
        super().__init__()
        c, h, w = image_shape
        if useResNet:
            self.conv = Conv2dResModel(c, layers=resNetLayers, use_maxpool=use_maxpool)
        else:
            self.conv = Conv2dModel(
                in_channels=c,
                channels=channels,
                kernel_sizes=kernel_sizes,
                strides=strides,
                paddings=paddings,
                nonlinearity=nonlinearity,
                use_maxpool=use_maxpool,
                batchNorm=batchNorm,
                dropout=dropout
            )
        conv_out_size = self.conv.conv_out_size(h, w)
        if hidden_sizes or output_size:
            self.head = MlpModel(conv_out_size, hidden_sizes,
                output_size=output_size, nonlinearity=nonlinearity)
            if output_size is not None:
                self._output_size = output_size
            else:
                self._output_size = (hidden_sizes if
                    isinstance(hidden_sizes, int) else hidden_sizes[-1])
        else:
            self.head = lambda x: x
            self._output_size = conv_out_size

    def forward(self, input):
        """Compute the convolution and fully connected head on the input;
        assumes correct input shape: [B,C,H,W]."""
        return self.head(self.conv(input).view(input.shape[0], -1))

    @property
    def output_size(self):
        """Returns the final output size after MLP head."""
        return self._output_size
