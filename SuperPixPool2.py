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


def low_superpixel_pooling(features, superpixel_map):
    """
    features: Tensor of shape [B, C, H, W]
    superpixel_map: Tensor of shape [B, H, W]
    """
    device = features.device
    B, C, H, W = features.shape
    out = torch.zeros_like(features, device=device)

    superpixel_map_flat = superpixel_map.view(B, -1)
    features_flat = features.view(B, C, -1)

    for i in range(B):
        spmap = superpixel_map_flat[i]
        fmap = features_flat[i]
        sp_ids = torch.unique(spmap)

        pooled_feat = torch.zeros(C, len(sp_ids), device=device)
        for sp_id in sp_ids:
            mask = (spmap == sp_id).float().to(fmap.device)
            area = mask.sum() + 1e-6
            pooled_feat[:, sp_id.item()] = (fmap * mask).sum(dim=1) / area  # [C,1]

        # 将 pooled 特征广播回每个像素位置
        out[i] = (pooled_feat.view(C, len(sp_ids))[:, spmap].view(C, H, W))

        # 清理不再使用的临时变量
        del pooled_feat, mask

    torch.cuda.empty_cache()  # 手动清理缓存，释放显存
    return out



def low_superpixel_unpooling(pooled_features, superpixel_map, H, W):
    """
    pooled_features: Tensor [B, C, h, w]（superpixel_pooling 输出）
    superpixel_map: Tensor [B, H, W]，每个像素对应的超像素 ID
    H, W: 输出尺寸
    """
    B, C = pooled_features.shape[:2]
    device = pooled_features.device

    out = torch.zeros(B, C, H, W, device=device)
    superpixel_map = superpixel_map.to(device)

    for b in range(B):
        spmap = superpixel_map[b].view(-1).long()  # [H*W]
        sp_ids = torch.unique(spmap)
        num_sp = sp_ids.size(0)

        # 将 pooled 特征展平：[C, num_sp]
        feat = pooled_features[b].view(C, -1)[:, :num_sp]  # 假设 pooled 特征数量等于超像素数量

        # 构建映射表：每个像素所属的超像素在 sp_ids 中的下标
        sp_id_to_index = {int(sp.item()): idx for idx, sp in enumerate(sp_ids)}
        index_map = torch.tensor([sp_id_to_index[int(i.item())] for i in spmap],
                                 device=device, dtype=torch.long)  # [H*W]

        # 还原每像素特征
        out[b] = feat[:, index_map].view(C, H, W)

        # 删除临时变量，释放内存
        del spmap, sp_ids, feat, sp_id_to_index, index_map

    # 清理不再需要的缓存，释放显存
    torch.cuda.empty_cache()

    return out


# def low_superpixel_pooling(features, superpixel_map):
#     """
#     features: Tensor of shape [B, C, H, W]
#     superpixel_map: Tensor of shape [B, H, W]
#     """
#     device = features.device
#     B, C, H, W = features.shape
#     out = torch.zeros_like(features, device=device)
#
#     for i in range(B):
#         fmap = features[i].view(C, -1)  # [C, H*W]
#         spmap = superpixel_map[i].view(-1)  # [H*W]
#         sp_ids = torch.unique(spmap)
#         sp_id_to_index = {int(sp.item()): idx for idx, sp in enumerate(sp_ids)}
#
#         pooled_feat = torch.zeros(C, len(sp_ids), device=device)
#
#         for sp_id in sp_ids:
#             idx = sp_id_to_index[int(sp_id.item())]
#             mask = (spmap == sp_id).float().unsqueeze(0).to(fmap.device)  # [1, H*W]
#             area = mask.sum() + 1e-6
#             pooled_feat[:, idx] = (fmap * mask).sum(dim=1) / area
#
#         index_map = torch.tensor([sp_id_to_index[int(i.item())] for i in spmap],
#                                  device=device, dtype=torch.long)  # [H*W]
#         out[i] = pooled_feat[:, index_map].view(C, H, W)
#
#     return out
#
#
# def low_superpixel_unpooling(pooled_features, superpixel_map, H, W):
#     """
#     pooled_features: Tensor [B, C, N]（N 是超像素数）
#     superpixel_map: Tensor [B, H, W]，每个像素对应的超像素 ID
#     H, W: 输出尺寸
#     """
#     B, C = pooled_features.shape[:2]
#     device = pooled_features.device
#     out = torch.zeros(B, C, H, W, device=device)
#
#     for b in range(B):
#         spmap = superpixel_map[b].view(-1).long()  # [H*W]
#         sp_ids = torch.unique(spmap)
#         sp_id_to_index = {int(sp.item()): idx for idx, sp in enumerate(sp_ids)}
#         feat = pooled_features[b]  # [C, N]
#
#         index_map = torch.tensor([sp_id_to_index[int(i.item())] for i in spmap],
#                                  device=device, dtype=torch.long)  # [H*W]
#         out[b] = feat[:, index_map].view(C, H, W)
#
#     return out

