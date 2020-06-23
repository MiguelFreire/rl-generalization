import torch

def random_cutout_color(imgs, min_cut=10, max_cut=30):
    n, c, h, w = imgs.shape
    device = imgs.device
    
    for i in range(n):
      box_h = torch.randint(min_cut, max_cut, (1,), device=device).item()
      box_w = torch.randint(min_cut, max_cut, (1,), device=device).item()

      color = torch.randint(0, 255, (3,), device=device) / 255.
      r = color[0].item()
      g = color[1].item()
      b = color[2].item()

      x = torch.randint(0, w - box_w - 1, (1,), device=device).item()
      y = torch.randint(0, h - box_h -1, (1,), device=device).item()

      imgs[i,0, y:y+box_h,x:x+box_w] = r
      imgs[i,1, y:y+box_h,x:x+box_w] = g
      imgs[i,2, y:y+box_h,x:x+box_w] = b
    
    return imgs