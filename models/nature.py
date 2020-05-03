import torch

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from models.conv2d import Conv2dHeadModel
import numpy as np

class NatureCNNModel(torch.nn.Module):
    def __init__(self, image_shape, output_size, batchNorm=False, dropout=0.0):
        super().__init__()
    
        self.conv = Conv2dHeadModel(
            image_shape=image_shape,
            channels=[32,64,64],
            kernel_sizes=[8,4,3],
            strides=[4,3,1],
            paddings=[0,0,1],
            use_maxpool=False,
            hidden_sizes=512,
            batchNorm=batchNorm,
            dropout=dropout
        )
        #policy head
        
        self.pi = torch.nn.Linear(self.conv.output_size, output_size) 
        #value function head 
        self.value = torch.nn.Linear(self.conv.output_size, 1)
        #reset weights just like nature paper
        self.init_weights()
        
    def init_weights(self):
        #orthogonal initialization with gain of sqrt(2)
        def weights_initializer(m):
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.orthogonal_(m.weight, np.sqrt(2))
        
        self.apply(weights_initializer)
            
            
    def forward(self, image, prev_action, prev_reward):
        #input normalization, cast to float then grayscale it
        x = image.type(torch.float)
        x = x.mul_(1. / 255)
        
        lead_dim, T, B, img_shape = infer_leading_dims(img, 3)
        
        fc_out = self.conv(img.view(T * B, *img_shape))
        pi = F.softmax(self.pi(fc_out), dim=-1)
        v = self.value(fc_out).squeeze(-1)
        # Restore leading dimensions: [T,B], [B], or [], as input.
        #T -> transition
        #B -> batch_size?
        pi, v = restore_leading_dims((pi, v), lead_dim, T, B)

        return pi, v
    