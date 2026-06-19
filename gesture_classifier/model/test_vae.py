import torch
import torch.nn as nn
from torchvision.utils import save_image
import os

IMG_SIZE   = 128
CHANNELS   = 3
LATENT_DIM = 64
FLAT       = CHANNELS * IMG_SIZE * IMG_SIZE

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(FLAT, 512), nn.ReLU(),
            nn.Linear(512, 256),  nn.ReLU(),
        )
        self.mu      = nn.Linear(256, LATENT_DIM)
        self.log_var = nn.Linear(256, LATENT_DIM)
        self.dec = nn.Sequential(
            nn.Linear(LATENT_DIM, 256), nn.ReLU(),
            nn.Linear(256, 512),        nn.ReLU(),
            nn.Linear(512, FLAT),       nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.log_var(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.dec(z), mu, log_var

model = VAE()
model.load_state_dict(torch.load('vae_owl.pt', weights_only=True))
model.eval()

os.makedirs('../data/generated', exist_ok=True)

with torch.no_grad():
    z = torch.randn(16, LATENT_DIM)
    imgs = model.dec(z).view(16, 3, IMG_SIZE, IMG_SIZE)
    save_image(imgs, '../data/generated/sample_grid.png', nrow=4)
    print('✅ 生成完成！图片保存到 data/generated/sample_grid.png')
