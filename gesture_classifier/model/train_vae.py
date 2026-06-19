import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

# ── 参数 ──────────────────────────────────────────
IMG_SIZE   = 128
CHANNELS   = 3        # 彩色 RGB
LATENT_DIM = 64
BATCH_SIZE = 16
EPOCHS     = 100
LR         = 1e-3
DATA_DIR   = '../data/owl face'
SAVE_PATH  = 'vae_owl.pt'
# ─────────────────────────────────────────────────

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),   # [0,1], RGB
])

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

FLAT = CHANNELS * IMG_SIZE * IMG_SIZE

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

def vae_loss(recon, x, mu, log_var):
    recon_loss = F.binary_cross_entropy(recon, x.view(-1, FLAT))
    kl         = -0.5 * torch.mean(1 + log_var - mu**2 - log_var.exp())
    return recon_loss + 0.001 * kl

model = VAE()
opt   = torch.optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    total = 0
    for imgs, _ in loader:
        recon, mu, lv = model(imgs)
        loss = vae_loss(recon, imgs, mu, lv)
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item()
    if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1}/{EPOCHS}  loss={total/len(loader):.4f}')

torch.save(model.state_dict(), SAVE_PATH)
print('✅ 保存到', SAVE_PATH)
