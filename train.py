import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from segnet2 import SegNet_spp
from KvasirDataset import KvasirSegDataset

def train_segnet():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = SegNet_spp(3, 2).to(device)

    # 加载完整数据集
    full_dataset = KvasirSegDataset(
        image_dir="../kvasir-seg/Kvasir-SEG/images/",
        mask_dir="../kvasir-seg/Kvasir-SEG/masks/",
        target_size=(512, 512)
    )

    # 数据集划分
    total_size = len(full_dataset)
    test_size = int(total_size * 0.1)
    train_val_size = total_size - test_size
    val_size = int(train_val_size * 0.1)
    train_size = train_val_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(88)
    )

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # 损失函数与优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-4)

    best_val_acc = 0.0
    num_epochs = 20
    print("dataset split done")
    for epoch in range(num_epochs):
        net.train()
        total_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device).long().squeeze(1)

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # 验证
        net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device).long().squeeze(1)
                outputs = net(images)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == masks).sum().item()
                total += masks.numel()

        val_acc = correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f} - Val Acc: {val_acc:.4f}")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(net.state_dict(), "best_segnet.pth")

if __name__ == "__main__":
    train_segnet()
