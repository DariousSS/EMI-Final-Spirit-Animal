import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np

# ── 1. 读数据 ──────────────────────────────────────────
df = pd.read_csv('../data/gesture_data.csv')
X = df.drop('label', axis=1).values.astype(np.float32)
y = df['label'].values.astype(np.int64)

print(f"数据总量: {len(df)} 条")
print(f"Label 0: {(y==0).sum()}  Label 1: {(y==1).sum()}")

X_tensor = torch.tensor(X)
y_tensor = torch.tensor(y)

dataset = TensorDataset(X_tensor, y_tensor)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=32)

# ── 2. 定义模型 ────────────────────────────────────────
class GestureNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(42, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
    def forward(self, x):
        return self.net(x)

model = GestureNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# ── 3. 训练 ────────────────────────────────────────────
print("\n开始训练...")
for epoch in range(50):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                preds = model(xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += len(yb)
        print(f"Epoch {epoch+1}/50  验证准确率: {correct/total*100:.1f}%")

# ── 4. 保存模型 ────────────────────────────────────────
torch.save(model.state_dict(), 'gesture_model.pt')
print("\n✅ 模型已保存到 model/gesture_model.pt")