import torch
import os
from models.attention import NatureAttention, ImpalaAttention
from matplotlib import pyplot as plt

class AttentionMaps():
  def __init__(
    self, 
    path="./", 
    obs="./",
    name="Default",
    batchNorm=False, 
    dropout=0.0, 
    augment_obs=None,  
    hidden_sizes=[512], 
    max_pooling=False, 
    arch="original"):
    
    self.name = name
    self.obs = torch.load(os.getcwd() + obs, map_location=torch.device('cpu'))
    
    dummy_x = torch.rand((3,64,64))
    
    if arch=="impala":
      in_channels = [3,16,32]
      out_channels = [16,32,32]
      self.model = ImpalaAttention(dummy_x.shape, 15, [], [], 512)
    else:
      self.model = NatureAttention(
        dummy_x.shape, 15, 
        batchNorm=batchNorm, 
        dropout=dropout, 
        augment_obs=augment_obs, 
        hidden_sizes=hidden_sizes, 
        use_maxpool=max_pooling, 
        arch=arch)
      
    saved_params = torch.load(os.getcwd() + path, map_location=torch.device('cpu'))

    self.model.load_state_dict(saved_params)
    
  def run(self):
    with torch.no_grad():
      #img = torch.from_numpy(self.obs).unsqueeze(0)
      img = self.obs.type(torch.float)
      img = img.mul_(1. / 255)
      gs = self.model(img)
      
      titles = ['Low Level', 'Medium Level', 'High Level']
      extent = 0, 64, 0, 64
      fig, axs = plt.subplots(3, figsize=(5,12))
      fig.suptitle('Attention Maps')
      
      for i, g in enumerate(gs):
        x = self.obs.squeeze(0).numpy().transpose((1,2,0))
        axs[i].imshow(x, extent=extent)
        axs[i].imshow(g[0], interpolation='bilinear', extent=extent, alpha=0.7)
        axs[i].set_title(titles[i])
      
      fig.savefig(self.name + '.png')