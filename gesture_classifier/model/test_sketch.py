import torch
import torch.nn as nn
from torchvision.utils import save_image

IMG_SIZE = 128
LATENT_DIM = 64
FLAT = IMG_SIZE * IMG_SIZE

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Flatten(), nn.Linear(FLAT,512), nn.ReLU(), nn.Linear(512,256), nn.ReLU())
        self.mu = nn.Linear(256, LATENT_DIM)
        self.log_var = nn.Linear(256, LATENT_DIM)
        self.dec = nn.Sequential(nn.Linear(LATENT_DIM,256), nn.ReLU(), nn.Linear(256,512), nn.ReLU(), nn.Linear(512,FLAT), nn.Sigmoid())

v = VAE()
v.load_state_dict(torch.load('vae_sketch.pt', weights_only=True))
v.eval()
with torch.no_grad():
    z = torch.randn(1, LATENT_DIM)
    img = v.dec(z).view(1, 1, IMG_SIZE, IMG_SIZE)
    save_image(img, '../data/generated/current_sketch.png')
    print('生成完成')
