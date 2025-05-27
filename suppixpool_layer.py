import torch
import suppixpool_CUDA as spx_gpu
import numpy as np

class SupPixPoolFunction(torch.autograd.Function):
    # @staticmethod
    # def forward(ctx, img, spx):
    #     B, C, H, W = img.shape
    #     spx = spx.to(torch.int32)
    #     K = len(torch.unique(spx))
    #
    #     # 确保处理多通道输入
    #     outputs = []
    #     for c in range(C):  # 逐通道处理
    #         channel_img = img[:, c:c + 1]  # [B,1,H,W]
    #         out = spx_gpu.forward(channel_img.contiguous(), spx.contiguous(), K)
    #         outputs.append(out[0])  # 取池化结果
    #
    #     # 合并通道 [B,C,K]
    #     pooled = torch.cat(outputs, dim=1)
    #     ctx.save_for_backward(out[1], img, spx, torch.tensor(K))
    #     return pooled
    @staticmethod
    def forward(ctx, img, spx):
        B, C, H, W = img.shape
        spx = spx.to(torch.int32)
        K = len(torch.unique(spx))

        pooled_outputs = []
        all_indices = []

        for c in range(C):  # 每个通道分别处理
            channel_img = img[:, c:c + 1].contiguous()  # [B,1,H,W]
            pooled_c, indices_c = spx_gpu.forward(channel_img, spx, K)
            pooled_outputs.append(pooled_c)
            all_indices.append(indices_c)

        # [B, C, K]
        pooled = torch.cat(pooled_outputs, dim=1)

        # 保存所有通道的 indices
        ctx.save_for_backward(*all_indices, img, spx, torch.tensor(K))
        ctx.C = C  # 保存通道数，用于 backward 中还原
        return pooled

    # @staticmethod
    # def backward(ctx, grad_output):
    #     indices, img, spx, K = ctx.saved_tensors
    #     grad_input, = spx_gpu.backward(grad_output.contiguous(), img, spx, indices, K)
    #     print("In backward(): checking tensors")
    #     print("  grad_output:", grad_output.shape, grad_output.device, grad_output.dtype)
    #     print("  img:", img.shape, img.device, img.dtype)
    #     print("  spx:", spx.shape, spx.device, spx.dtype)
    #     print("  indices:", indices.shape, indices.device, indices.dtype)
    #
    #     assert img.shape[-2:] == spx.shape[-2:], "Shape mismatch"
    #     assert grad_output.is_cuda, "grad_output not on CUDA"
    #     assert spx.is_cuda, "spx not on CUDA"
    #     assert indices.min() >= 0, "Negative indices in backward!"
    #
    #     return grad_input, None
    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output: [B, C, K]
        返回每个输入通道的反向传播结果，再拼接为 [B, C, H, W]
        """
        saved = ctx.saved_tensors
        C = ctx.C
        indices_list = saved[:C]
        img = saved[C]
        spx = saved[C + 1]
        K = saved[C + 2].item()  # tensor 转 int

        B, _, H, W = img.shape
        grads = []

        for c in range(C):
            grad_c = grad_output[:, c:c + 1, :]  # [B,1,K]
            img_c = img[:, c:c + 1, :, :]  # [B,1,H,W]
            indices_c = indices_list[c]  # [B,1,K]

            grad_input_c, = spx_gpu.backward(
                grad_c.contiguous(), img_c.contiguous(), spx.contiguous(), indices_c.contiguous(), K
            )
            grads.append(grad_input_c)

        grad_input = torch.cat(grads, dim=1)  # [B, C, H, W]
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
        return output.view(B, C, H, W).contiguous()
