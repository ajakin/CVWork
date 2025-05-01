import torch

def superpixel_pooling(features, superpixel_map):
    """
    features: Tensor of shape [B, C, H, W]
    superpixel_map: Tensor of shape [B, H, W]
    """
    device = features.device
    B, C, H, W = features.shape
    out = torch.zeros_like(features)  # 自动在同一个 device 上

    superpixel_map_flat = superpixel_map.view(B, -1)
    features_flat = features.view(B, C, -1)

    for i in range(B):
        spmap = superpixel_map_flat[i]
        fmap = features_flat[i]
        sp_ids = torch.unique(spmap)

        for sp_id in sp_ids:
            mask = (spmap == sp_id).float().to(fmap.device)
            area = mask.sum() + 1e-6
            pooled_feat = (fmap * mask).sum(dim=1, keepdim=True) / area  # [C,1]
            mask_exp = mask.unsqueeze(0).expand(C, -1)
            out[i] += (pooled_feat * mask_exp).view(C, H, W)

    return out


def superpixel_unpooling(pooled_features, superpixel_map, H, W):
    """
    支持 GPU 和批量处理的超像素反池化操作
    pooled_features: Tensor of shape [B, C, H, W] (池化后的特征图)
    superpixel_map: Tensor of shape [B, H, W] (每个像素值是超像素标签)
    H, W: 原图尺寸
    return: Tensor of shape [B, C, H, W]
    """
    device = pooled_features.device
    B, C, _, _ = pooled_features.shape
    unpooled_features = torch.zeros(B, C, H, W, device=device)

    superpixel_map_flat = superpixel_map.view(B, -1)         # [B, H*W]
    pooled_features_flat = pooled_features.view(B, C, -1)    # [B, C, H*W]

    for i in range(B):
        spmap = superpixel_map_flat[i]
        pooled_fmap = pooled_features_flat[i]
        sp_ids = torch.unique(spmap)

        for sp_id in sp_ids:
            mask = (spmap == sp_id).float().to(pooled_fmap.device)
            pooled_feat = pooled_fmap[:, sp_id.item()]  # [C]
            unpooled_features[i] += pooled_feat.view(C, 1, 1) * mask.view(1, H, W)

    return unpooled_features


