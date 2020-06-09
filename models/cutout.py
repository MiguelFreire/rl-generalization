import numpy as np

def random_cutout_color(imgs, min_cut=10,max_cut=30, num_cutouts=1):
    """
        args:
        imgs: shape (B,C,H,W)
        out: output size (e.g. 84)
    """

    n, c, h, w = imgs.shape

    w1 = np.random.randint(min_cut, max_cut, (n, num_cutouts))
    h1 = np.random.randint(min_cut, max_cut, (n, num_cutouts))

    cutouts = np.empty((n, c, h, w), dtype=imgs.dtype)
    rand_box = np.random.randint(0, 255, (n, num_cutouts, c)) / 255.
    for i, (img, w11, h11) in enumerate(zip(imgs, w1, h1)):
        cut_img = img.copy()
        for j, (w22, h22) in enumerate(zip(w11, h11)):
            cut_img[:, h22:h22 + h22, w22:w22 + w22] = np.tile(
                rand_box[i][j].reshape(-1,1,1),                                                
                (1,) + cut_img[:, h22:h22 + h22, w22:w22 + w22].shape[1:])
        
        cutouts[i] = cut_img
    return cutouts