import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from segnet import SegNet


class KvasirSegDataset(Dataset):
    def __init__(self, image_dir, mask_dir, target_size=(256, 256)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_list = sorted(os.listdir(image_dir))
        self.mask_list = sorted(os.listdir(mask_dir))
        self.target_size = target_size

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_list[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_list[idx])

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        image, mask = self.resize_and_pad(image, mask, self.target_size)

        image = TF.to_tensor(image)
        mask = np.array(mask)
        mask = (mask >= 128).astype(np.uint8)
        mask = torch.from_numpy(mask).unsqueeze(0).float()

        return image, mask

    def resize_and_pad(self, image, mask, size):
        w, h = image.size
        scale = min(size[0] / h, size[1] / w)
        new_w, new_h = int(w * scale), int(h * scale)

        image = image.resize((new_w, new_h), Image.BILINEAR)
        mask = mask.resize((new_w, new_h), Image.NEAREST)

        pad_left = (size[1] - new_w) // 2
        pad_right = size[1] - new_w - pad_left
        pad_top = (size[0] - new_h) // 2
        pad_bottom = size[0] - new_h - pad_top

        image = TF.pad(image, (pad_left, pad_top, pad_right, pad_bottom), fill=0)
        mask = TF.pad(mask, (pad_left, pad_top, pad_right, pad_bottom), fill=0)

        return image, mask


import matplotlib.pyplot as plt

def show_sample_from_kvasir():
    dataset = KvasirSegDataset(
        image_dir="../kvasir-seg/Kvasir-SEG/images/",
        mask_dir="../kvasir-seg/Kvasir-SEG/masks/",
        target_size=(512, 512)
    )

    image, mask = dataset[0]  # 取第一张图像

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(image.permute(1, 2, 0))  # CHW -> HWC
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mask.squeeze(0), cmap='gray')
    plt.title("Mask")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

def check_net():
    net = SegNet(3,2)
    dataset = KvasirSegDataset(
        image_dir="../kvasir-seg/Kvasir-SEG/images/",
        mask_dir="../kvasir-seg/Kvasir-SEG/masks/",
        target_size=(512, 512)
    )

    image, mask = dataset[0]
    image = image.unsqueeze(0)

    print(net(image))


if __name__ == '__main__':
    # show_sample_from_kvasir()
    check_net()
