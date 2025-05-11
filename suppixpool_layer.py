import torch
import suppixpool_CUDA as spx_gpu
import numpy as np

class SupPixPoolFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, img, spx):
        B, C, H, W = img.shape
        spx = spx.to(torch.int32)
        K = len(torch.unique(spx))

        # 确保处理多通道输入
        outputs = []
        for c in range(C):  # 逐通道处理
            channel_img = img[:, c:c + 1]  # [B,1,H,W]
            out = spx_gpu.forward(channel_img, spx, K)
            outputs.append(out[0])  # 取池化结果

        # 合并通道 [B,C,K]
        pooled = torch.cat(outputs, dim=1)
        ctx.save_for_backward(out[1], img, spx, torch.tensor(K))
        return pooled
        
    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        indices, img, spx, K = ctx.saved_tensors
        grad_input, = spx_gpu.backward(grad_output.contiguous(), img, spx, indices, K)
        return grad_input, torch.zeros_like(spx)

class SupPixPool(torch.nn.Module):
    def __init__(self):
        super(SupPixPool, self).__init__()

    def forward(self, img, spx):
        return SupPixPoolFunction.apply(img, spx)

class SupPixUnpool(torch.nn.Module):
    def __init__(self):
        super(SupPixUnpool, self).__init__()

    def forward(self, pooled, spx):
        # 确保输入是3D [B,C,K]
        if pooled.dim() == 4:  # 如果是4D [B,C,1,K]
            pooled = pooled.squeeze(2)  # 移除多余的维度
        elif pooled.dim() != 3:
            raise ValueError(f"输入必须是3D或4D张量，但得到的是{pooled.dim()}D")

        B, C, K = pooled.shape
        _, H, W = spx.shape

        # 验证标签值范围
        max_label = spx.max().item()
        if max_label >= K:
            raise ValueError(f"超像素标签最大值{max_label} >= 池化特征数{K}")

        # 反池化操作
        output = pooled.gather(2, spx.view(B, 1, H * W).expand(-1, C, -1))
        return output.view(B, C, H, W)
