from skimage.segmentation import slic
from skimage.util import img_as_float
import numpy as np
import torch

def compute_superpixel_map(img_tensor, n_segments=100, compactness=10):
    """
    img_tensor: [B, 3, H, W] 的 PyTorch 图像张量，值范围 [0, 1] 或 [0, 255]
    return: [B, H, W] 的超像素标签张量，每个像素的值表示所属的超像素编号
    """
    B, C, H, W = img_tensor.shape
    superpixel_maps = []

    for i in range(B):
        img = img_tensor[i].cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]
        img = img_as_float(img)  # 转为 float 类型以便 SLIC 处理
        segments = slic(img, n_segments=n_segments, compactness=compactness, start_label=0)
        superpixel_maps.append(torch.from_numpy(segments).long())  # [H, W]

    return torch.stack(superpixel_maps)  # [B, H, W]

if __name__ == '__main__':
    from skimage.io import imread
    from skimage.segmentation import slic, mark_boundaries
    from skimage.util import img_as_float
    import matplotlib.pyplot as plt
    import cv2

    # 读取图像
    image_path = '../kvasir-seg/Kvasir-SEG/images/cju0qkwl35piu0993l0dewei2.jpg'
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV读取的是BGR，转为RGB
    image = img_as_float(image)

    # 使用 SLIC 进行超像素分割
    segments = slic(image, n_segments=100, compactness=10, sigma=1, start_label=0)

    # 可视化分割边界
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(mark_boundaries(image, segments))
    ax.set_title('SLIC Segmentation')
    ax.axis('off')
    plt.tight_layout()
    plt.show()


