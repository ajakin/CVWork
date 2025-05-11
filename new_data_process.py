import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
from skimage.segmentation import slic
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
from segnet2 import SegNet_spp


class KvasirSegDataset(Dataset):
    def __init__(self, image_dir, mask_dir, target_size=(256, 256), debug=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_list = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
        self.mask_list = sorted([f for f in os.listdir(mask_dir) if f.endswith('.jpg')])
        self.target_size = target_size
        self.debug = debug
        self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_list[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_list[idx])

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        image, mask = self._resize_and_pad(image, mask)

        image_tensor = self.to_tensor(image)
        mask_tensor = self.to_tensor(mask).squeeze(0)
        mask_tensor = (mask_tensor > 0.5).float()

        if self.debug:
            self._debug_visualize(image_tensor, mask_tensor, idx)

        return image_tensor, mask_tensor

    def _resize_and_pad(self, image, mask):
        w, h = image.size
        scale = min(self.target_size[0] / h, self.target_size[1] / w)
        new_w, new_h = int(w * scale), int(h * scale)

        image = image.resize((new_w, new_h), Image.BILINEAR)
        mask = mask.resize((new_w, new_h), Image.NEAREST)

        pad_left = (self.target_size[1] - new_w) // 2
        pad_right = self.target_size[1] - new_w - pad_left
        pad_top = (self.target_size[0] - new_h) // 2
        pad_bottom = self.target_size[0] - new_h - pad_top

        fill_color = (128, 128, 128)
        image = TF.pad(image, (pad_left, pad_top, pad_right, pad_bottom), fill=fill_color)
        mask = TF.pad(mask, (pad_left, pad_top, pad_right, pad_bottom), fill=0)

        return image, mask

    def _generate_superpixels(self, img_tensor):
        """完全重写的超像素生成方法"""
        B, C, H, W = img_tensor.shape
        superpixel_maps = []

        for i in range(B):
            # 转换为numpy数组并处理
            img_np = (img_tensor[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            # 创建全图标签
            segments = np.zeros((H, W), dtype=np.int32)

            # 识别非填充区域（非灰色区域）
            non_gray = ~np.all(img_np == 128, axis=-1)

            if non_gray.any():
                # 提取非填充区域
                active_img = img_np[non_gray]

                # 重塑为2D数组（n_pixels x channels）
                active_img = active_img.reshape(-1, 3)

                # 计算超像素
                n_segments = min(100, active_img.shape[0] // 10)  # 确保足够像素
                if n_segments > 1:
                    labels = slic(
                        active_img,
                        n_segments=n_segments,
                        compactness=10,
                        channel_axis=-1
                    )
                    # 分配标签
                    segments[non_gray] = labels.reshape(segments[non_gray].shape)

            # 为填充区域分配最大标签+1
            segments[~non_gray] = segments.max() + 1 if segments.max() > 0 else 1

            superpixel_maps.append(torch.from_numpy(segments))

        return torch.stack(superpixel_maps).long()

    def _debug_visualize(self, image_tensor, mask_tensor, idx):
        """改进的调试可视化"""
        os.makedirs("debug", exist_ok=True)

        # 保存输入图像
        image_np = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        Image.fromarray(image_np).save(f"debug/{idx}_input.jpg")

        try:
            # 生成超像素
            spx = self._generate_superpixels(image_tensor.unsqueeze(0))

            # 可视化
            plt.figure(figsize=(12, 4))
            plt.subplot(131)
            plt.imshow(image_np)
            plt.title('Input Image')
            plt.subplot(132)
            plt.imshow(mask_tensor.numpy(), cmap='gray')
            plt.title('Mask')
            plt.subplot(133)
            plt.imshow(spx.squeeze().numpy(), cmap='jet')
            plt.title(f'Superpixels (max={spx.max().item()})')
            plt.savefig(f"debug/{idx}_preview.jpg")
            plt.close()

            print(f"Debug output saved for sample {idx}")
        except Exception as e:
            print(f"Debug visualization failed: {str(e)}")


def check_net():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 初始化模型
    net = SegNet_spp(3, 2).to(device)
    print("Model initialized:")
    print(f"  Input channels: 3, Output classes: 2")
    print(f"  Total parameters: {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M")

    # 加载数据集
    dataset = KvasirSegDataset(
        image_dir="../kvasir-seg/Kvasir-SEG/images/",
        mask_dir="../kvasir-seg/Kvasir-SEG/masks/",
        target_size=(256, 256),
        debug=True  # 启用调试模式
    )
    print("\nDataset loaded successfully")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Target size: {dataset.target_size}")

    # 获取第一个样本
    image, mask = dataset[0]
    image = image.unsqueeze(0).to(device)  # 添加batch维度并转移到设备

    # 打印输入信息
    print("\nInput details:")
    print(f"  Image shape: {image.shape} (range: {image.min():.2f}-{image.max():.2f})")
    print(f"  Mask shape: {mask.shape} (unique values: {torch.unique(mask).tolist()})")

    # 运行模型
    with torch.no_grad():
        output = net(image)

    # 打印输出信息
    print("\nModel output:")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: {output.min():.2f}-{output.max():.2f}")

    return output





if __name__ == "__main__":
    check_net()
