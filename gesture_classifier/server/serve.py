import asyncio, websockets, torch, torch.nn as nn, json
from torchvision.utils import save_image
import os

IMG_SIZE   = 128
LATENT_DIM = 64

# 脸部猫头鹰（彩色）
CHANNELS_OWL  = 3
FLAT_OWL      = CHANNELS_OWL * IMG_SIZE * IMG_SIZE
OUTPUT_OWL    = os.path.expanduser('~/Desktop/gesture_classifier/data/generated/current_owl.png')

# 线稿猫头鹰（灰度）
CHANNELS_SKT  = 1
FLAT_SKT      = CHANNELS_SKT * IMG_SIZE * IMG_SIZE
OUTPUT_SKETCH = os.path.expanduser('~/Desktop/gesture_classifier/data/generated/current_sketch.png')

os.makedirs(os.path.dirname(OUTPUT_OWL), exist_ok=True)

# ── 手势分类器 ────────────────────────────────────
class GestureNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(42, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 2)
        )
    def forward(self, x):
        return self.net(x)

gesture_model = GestureNet()
gesture_model.load_state_dict(torch.load(
    os.path.expanduser('~/Desktop/gesture_classifier/model/gesture_model.pt'),
    weights_only=True))
gesture_model.eval()

# ── 脸部 VAE（彩色）────────────────────────────────
class VAEOwl(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(FLAT_OWL, 512), nn.ReLU(),
            nn.Linear(512, 256),      nn.ReLU(),
        )
        self.mu      = nn.Linear(256, LATENT_DIM)
        self.log_var = nn.Linear(256, LATENT_DIM)
        self.dec = nn.Sequential(
            nn.Linear(LATENT_DIM, 256), nn.ReLU(),
            nn.Linear(256, 512),        nn.ReLU(),
            nn.Linear(512, FLAT_OWL),   nn.Sigmoid(),
        )
    def forward(self, x):
        h = self.enc(x)
        mu, lv = self.mu(h), self.log_var(h)
        z = mu + torch.exp(0.5*lv) * torch.randn_like(mu)
        return self.dec(z), mu, lv

# ── 线稿 VAE（灰度）────────────────────────────────
class VAESketch(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(FLAT_SKT, 512), nn.ReLU(),
            nn.Linear(512, 256),      nn.ReLU(),
        )
        self.mu      = nn.Linear(256, LATENT_DIM)
        self.log_var = nn.Linear(256, LATENT_DIM)
        self.dec = nn.Sequential(
            nn.Linear(LATENT_DIM, 256), nn.ReLU(),
            nn.Linear(256, 512),        nn.ReLU(),
            nn.Linear(512, FLAT_SKT),   nn.Sigmoid(),
        )
    def forward(self, x):
        h = self.enc(x)
        mu, lv = self.mu(h), self.log_var(h)
        z = mu + torch.exp(0.5*lv) * torch.randn_like(mu)
        return self.dec(z), mu, lv

vae_owl = VAEOwl()
vae_owl.load_state_dict(torch.load(
    os.path.expanduser('~/Desktop/gesture_classifier/model/vae_owl.pt'),
    weights_only=True))
vae_owl.eval()

# 线稿模型（训练完才加载）
vae_sketch = None
sketch_path = os.path.expanduser('~/Desktop/gesture_classifier/model/vae_sketch.pt')
if os.path.exists(sketch_path):
    vae_sketch = VAESketch()
    vae_sketch.load_state_dict(torch.load(sketch_path, weights_only=True))
    vae_sketch.eval()
    print('✅ 线稿模型加载成功')
else:
    print('⚠️  线稿模型未找到，跑完 train_vae_sketch.py 再重启')

def generate_owl():
    with torch.no_grad():
        z = torch.randn(1, LATENT_DIM)
        img = vae_owl.dec(z).view(1, 3, IMG_SIZE, IMG_SIZE)
        save_image(img, OUTPUT_OWL)

def generate_sketch():
    if vae_sketch is None:
        return
    with torch.no_grad():
        z = torch.randn(1, LATENT_DIM)
        img = vae_sketch.dec(z).view(1, 1, IMG_SIZE, IMG_SIZE)
        save_image(img, OUTPUT_SKETCH)

# ── 持续生成循环 ──────────────────────────────────
current_gesture = 0

async def generation_loop():
    while True:
        generate_owl()
        generate_sketch()
        await asyncio.sleep(0.3)

# ── WebSocket 服务 ────────────────────────────────
async def handle(websocket):
    global current_gesture
    print("TD 已连接！")
    async for message in websocket:
        try:
            coords = json.loads(message)
            x = torch.tensor([coords], dtype=torch.float32)
            with torch.no_grad():
                pred = gesture_model(x).argmax(dim=1).item()
            current_gesture = pred
            await websocket.send(str(pred))
        except Exception as e:
            print("错误:", e)
            await websocket.send("-1")

async def main():
    print("✅ 模型加载成功")
    print("🚀 服务器运行在 ws://localhost:8765")
    async with websockets.serve(handle, "localhost", 8765, ping_interval=None):
        await generation_loop()

asyncio.run(main())
