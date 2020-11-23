import torch

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from models.conv2d import Conv2dHeadModel
from models.cutout import random_cutout_color
from models.layer_transform import ColorJitterLayer
from models.rand_network import random_convolution
from rlpyt.utils.collections import namedarraytuple
import torch.nn.functional as F
import numpy as np
import math

class NatureCNNModel(torch.nn.Module):
    def __init__(self, 
                 image_shape, output_size, batchNorm=False, dropout=0.0, 
                 augment_obs=None, use_maxpool=False, hidden_sizes=[512],
                 arch="original",
                ):
        super().__init__()

        self.augment_obs = augment_obs  
        self.transform = None
        channels=[32,64,64]
        kernel_sizes=[8,4,3]
        strides=[4,3,1]
        paddings=[0,0,1]
        
        if arch == "depth+1":
          channels=[32,64,64,128] 
          kernel_sizes=[8,4,3,2]
          strides=[4,3,1,1]
          paddings=[0,0,1,0]
        elif arch == "depth+2":
          channels=[32,64,64,128,256]
          kernel_sizes=[8,4,3,2,2]
          strides=[4,3,1,1,1]
          paddings=[0,0,1,0,0]
        elif arch == "channels/2":
          channels=[16,32,32]
          kernel_sizes=[8,4,3]
          strides=[4,3,1]
          paddings=[0,0,1]
        elif arch == "channels*2":
          channels=[64,128,128]
          kernel_sizes=[8,4,3]
          strides=[4,3,1]
          paddings=[0,0,1]
        elif arch == "a":
          channels=[16,32,64]
          kernel_sizes=[8,4,3]
          strides=[4,3,1]
          paddings=[0,0,1]
        elif arch == "b":
          channels=[16,32,64]
          kernel_sizes=[4,3,3]
          strides=[2,1,1]
          paddings=[0,1,1]
          
        resnet = arch=="resnet" or arch=="resnet2"

        self.conv = Conv2dHeadModel(
          image_shape=image_shape,
          channels=channels,
          kernel_sizes=kernel_sizes,
          strides=strides,
          paddings=paddings,
          use_maxpool=use_maxpool,
          hidden_sizes=hidden_sizes,
          batchNorm=batchNorm,
          dropout=dropout,
          useResNet=resnet,
          resNetLayers=2 if arch == "resnet2" else 1,
        )
          
        #policy head
        
        self.pi = torch.nn.Linear(self.conv.output_size, output_size) 
        #value function head 
        self.value = torch.nn.Linear(self.conv.output_size, 1)
        #reset weights just like nature paper
        self.init_weights()
    
    @torch.no_grad()    
    def init_weights(self):
        #orthogonal initialization with gain of sqrt(2)
        def weights_initializer(m):
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.orthogonal_(m.weight, np.sqrt(2))
        
        self.apply(weights_initializer)
            
            
    def forward(self, image, prev_action=None, prev_reward=None):
        #input normalization, cast to float then grayscale it
        img = image.type(torch.float)
        img = img.mul_(1. / 255)
        
        lead_dim, T, B, img_shape = infer_leading_dims(img, 3)
        
        img = img.view(T * B, *img_shape)
        if self.augment_obs != None:
            b, c, h, w = img.shape
            mask_vbox = torch.zeros(size=img.shape, dtype=torch.bool, device=img.device)

            mh = math.ceil(h * 0.20)
            #2 squares side by side
            mw = mh * 2
            ##create velocity mask -> False where velocity box is, True rest of the screen
            vmask = torch.ones((b, c, mh, mw), dtype=torch.bool, device=img.device)
            mask_vbox[:,:,:mh,:mw] = vmask
            obs_without_vbox = torch.where(mask_vbox, torch.zeros_like(img), img)
            
            if self.augment_obs == 'cutout':
                augmented = random_cutout_color(obs_without_vbox)
            elif self.augment_obs == 'jitter':
                if self.transform is None:
                    self.transform = ColorJitterLayer(b)
                augmented = self.transform(obs_without_vbox)
            elif self.augment_obs == 'rand_conv':
                augmented = random_convolution(obs_without_vbox)

            fixed = torch.where(mask_vbox, img, augmented)
            img = fixed


        fc_out = self.conv(img)
        pi = F.softmax(self.pi(fc_out), dim=-1)
        v = self.value(fc_out).squeeze(-1)
        # Restore leading dimensions: [T,B], [B], or [], as input.
        #T -> transition
        #B -> batch_size?
        pi, v = restore_leading_dims((pi, v), lead_dim, T, B)

        return pi, v
    
RnnState = namedarraytuple("RnnState", ["h", "c"])
class NatureLSTMModel(torch.nn.Module):
    def __init__(self, 
               image_shape, output_size, hidden_sizes=[512],
                lstm_size=256
              ):
        super().__init__()
        channels=[32,64,64]
        kernel_sizes=[8,4,3]
        strides=[4,3,1]
        paddings=[0,0,1]

        self.conv = Conv2dHeadModel(
          image_shape=image_shape,
          channels=channels,
          kernel_sizes=kernel_sizes,
          strides=strides,
          paddings=paddings,
          use_maxpool=False,
          hidden_sizes=hidden_sizes,
        )

        self.lstm = torch.nn.LSTM(self.conv.output_size + output_size + 1, lstm_size)
        self.pi = torch.nn.Linear(lstm_size, output_size)
        self.value = torch.nn.Linear(lstm_size, 1)
  
    def forward(self, image, prev_action=None, prev_reward=None, init_rnn_state=None):
        img = image.type(torch.float)
        img = img.mul_(1. / 255)

        lead_dim, T, B, img_shape = infer_leading_dims(img, 3)

        img = img.view(T * B, *img_shape)
        fc_out = self.conv(img.view(T * B, *img_shape))
        lstm_input = torch.cat([
            fc_out.view(T, B, -1),
            prev_action.view(T, B, -1),  # Assumed onehot.
            prev_reward.view(T, B, 1),
            ], dim=2)

        init_rnn_state = None if init_rnn_state is None else tuple(init_rnn_state)
        lstm_out, (hn, cn) = self.lstm(lstm_input, init_rnn_state)
        pi = F.softmax(self.pi(lstm_out.view(T * B, -1)), dim=-1)
        v = self.value(lstm_out.view(T * B, -1)).squeeze(-1)

        # Restore leading dimensions: [T,B], [B], or [], as input.
        pi, v = restore_leading_dims((pi, v), lead_dim, T, B)
        # Model should always leave B-dimension in rnn state: [N,B,H].
        next_rnn_state = RnnState(h=hn, c=cn)

        return pi, v, next_rnn_state
        