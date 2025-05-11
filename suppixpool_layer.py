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
        outShape = pooled.size()[0:2]+spx.size()[-2:]
        out = pooled.new_zeros(outShape)
        for batch in range(pooled.size()[0]):
            out[batch, :, :, :] = pooled[batch, :, spx[batch,:,:]]
        return out
