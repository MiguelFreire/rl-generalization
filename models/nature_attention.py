import torch
from rlpyt.models.utils import conv2d_output_shape
from models.attention import SelfAttention, ProjectorBlock, LinearAttentionBlock
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from environments.procgen import make_env

class NatureSelfAttention(torch.nn.Module):
    def __init__(self, image_shape, output_size):
        super().__init__()
        c, h, w = input_shape
        self.conv1 = torch.nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4, padding=0)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=3, padding=0)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.attention = SelfAttention(32)
        
        convs = [self.conv1, self.attention, self.conv2, self.conv3]

        
        conv_output_size = self.conv_out_size(convs, h, w)
        
        self.fc = torch.nn.Linear(conv_output_size, 512)
        
        self.pi = torch.nn.Linear(512, output_size) 
        #value function head 
        self.value = torch.nn.Linear(512, 1)
        #reset weights just like nature paper
        self.init_weights()
    
    @torch.no_grad()    
    def init_weights(self):
        #orthogonal initialization with gain of sqrt(2)
        def weights_initializer(m):
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.orthogonal_(m.weight, np.sqrt(2))
        
        self.apply(weights_initializer)
        
    def conv_out_size(self, convs, h, w, c=None):
        """Helper function ot return the output size for a given input shape,
        without actually performing a forward pass through the model."""
        for child in convs:
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
    
    def forward(self, image, prev_action=None, prev_reward=None):
        img = image.type(torch.float)
        img = img.mul_(1. / 255)
        
        lead_dim, T, B, img_shape = infer_leading_dims(img, 3)
        
        img = img.view(T * B, *img_shape)
        
        relu = torch.nn.functional.relu
        
        y = relu(self.conv1(img))
        y = self.attention(y)
        y = relu(self.conv2(y))
        y = relu(self.conv3(y))
        
        fc_out = self.fc(y.view(T * B, -1))
        
        pi = torch.nn.functional.softmax(self.pi(fc_out), dim=-1)
        v = self.value(fc_out).squeeze(-1)
        # Restore leading dimensions: [T,B], [B], or [], as input.
        #T -> transition
        #B -> batch_size?
        pi, v = restore_leading_dims((pi, v), lead_dim, T, B)

        return pi, v


class NatureAttention(torch.nn.Module):
    def __init__(self, image_shape, output_size):
        super().__init__()
        c, h, w = input_shape
        self.conv1 = torch.nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4, padding=0)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=3, padding=0)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        self.projector = ProjectorBlock(32, 512)
        self.projector2 = ProjectorBlock(64, 512)
        
        self.attn1 = LinearAttentionBlock(in_features=512, normalize_attn=True)
        self.attn2 = LinearAttentionBlock(in_features=512, normalize_attn=True)
        
        convs = [self.conv1, self.conv2, self.conv3]

        
        conv_output_size = self.conv_out_size(convs, h, w)
        
        #self.fc = torch.nn.Linear(conv_output_size, 512)
        
        self.fc = torch.nn.Conv2d(in_channels=64, out_channels=512, kernel_size=4, padding=0, bias=True)
        
        self.pi = torch.nn.Linear(512*2, output_size) 
        #value function head 
        self.value = torch.nn.Linear(512*2, 1)
        #reset weights just like nature paper
        self.init_weights()
    
    @torch.no_grad()    
    def init_weights(self):
        #orthogonal initialization with gain of sqrt(2)
        def weights_initializer(m):
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.orthogonal_(m.weight, np.sqrt(2))
        
        self.apply(weights_initializer)
        
    def conv_out_size(self, convs, h, w, c=None):
        """Helper function ot return the output size for a given input shape,
        without actually performing a forward pass through the model."""
        for child in convs:
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
    
    def forward(self, image, prev_action=None, prev_reward=None):
        img = image.type(torch.float)
        img = img.mul_(1. / 255)
        
        lead_dim, T, B, img_shape = infer_leading_dims(img, 3)
        
        img = img.view(T * B, *img_shape)
        
        relu = torch.nn.functional.relu
        
        l1 = relu(self.conv1(img))
        l2 = relu(self.conv2(l1))
        y = relu(self.conv3(l2))
        #fc_out = self.fc(y.view(T * B, -1))
        fc_out = self.fc(y)

        l1 = self.projector(l1)
        l2 = self.projector2(l2)
        
        c1, g1 = self.attn1(l1, fc_out)
        c2, g2 = self.attn2(l2, fc_out)
        g = torch.cat((g1,g2), dim=1) # batch_sizexC
        # classification layer

        pi = torch.nn.functional.softmax(self.pi(g), dim=-1)
        v = self.value(g).squeeze(-1)
        # Restore leading dimensions: [T,B], [B], or [], as input.
        #T -> transition
        #B -> batch_size?
        pi, v = restore_leading_dims((pi, v), lead_dim, T, B)

        return pi, v