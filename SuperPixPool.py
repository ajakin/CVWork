import torch
import torch.nn.functional as F

def superpixel_pooling(features, superpixel_map):
    """
    features: Tensor of shape [B, C, H, W]
    superpixel_map: Tensor of shape [B, H, W] (each pixel's value is superpixel label)
    return: list of length B, each element is Tensor [N_i, C] where N_i is number of superpixels in image i
    """
    B, C, H, W = features.shape
    features = features.view(B, C, -1)  # [B, C, H*W]
    superpixel_map = superpixel_map.view(B, -1)  # [B, H*W]

    pooled_outputs = []
    for i in range(B):
        sp_ids = torch.unique(superpixel_map[i])
        pooled = []
        for sp_id in sp_ids:
            mask = (superpixel_map[i] == sp_id)  # [H*W]
            mask = mask.unsqueeze(0).float()  # [1, H*W]
            area = mask.sum() + 1e-6  # avoid divide by zero
            sp_feat = (features[i] * mask).sum(dim=1) / area  # [C]
            pooled.append(sp_feat)
        pooled_outputs.append(torch.stack(pooled))  # [N_i, C]

    return pooled_outputs




