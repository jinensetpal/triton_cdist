#!/usr/bin/env python3

from torch.library import triton_op
import triton.language as tl
from typing import List
import logging
import triton
import torch


@triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE_X1": 2, "BLOCK_SIZE_X2": 2, "BLOCK_SIZE_RD": 2}, num_stages=0, num_warps=4),
            triton.Config({"BLOCK_SIZE_X1": 2, "BLOCK_SIZE_X2": 2, "BLOCK_SIZE_RD": 2}, num_stages=0, num_warps=8),
            triton.Config({"BLOCK_SIZE_X1": 4, "BLOCK_SIZE_X2": 4, "BLOCK_SIZE_RD": 4}, num_stages=0, num_warps=4),
            triton.Config({"BLOCK_SIZE_X1": 4, "BLOCK_SIZE_X2": 4, "BLOCK_SIZE_RD": 4}, num_stages=0, num_warps=8),
            triton.Config({"BLOCK_SIZE_X1": 8, "BLOCK_SIZE_X2": 8, "BLOCK_SIZE_RD": 8}, num_stages=0, num_warps=4),
            triton.Config({"BLOCK_SIZE_X1": 8, "BLOCK_SIZE_X2": 8, "BLOCK_SIZE_RD": 8}, num_stages=0, num_warps=8),
            triton.Config({"BLOCK_SIZE_X1": 16, "BLOCK_SIZE_X2": 16, "BLOCK_SIZE_RD": 16}, num_stages=0, num_warps=4),
            triton.Config({"BLOCK_SIZE_X1": 16, "BLOCK_SIZE_X2": 16, "BLOCK_SIZE_RD": 16}, num_stages=0, num_warps=8),
        ],
        key=['out_d1', 'out_d2', 'reduced_dim'],
        restore_value=['x1_ptr', 'x2_ptr'],
        reset_to_zero=['out_ptr'],
    )
@triton.jit
def pairwise_lp_kernel(
    x1_ptr,
    x2_ptr,
    out_ptr,
    out_d1, out_d2, reduced_dim,
    stride_x11, stride_x12,
    stride_x21, stride_x22,
    p,
    BLOCK_SIZE_RD: tl.constexpr,
    BLOCK_SIZE_X1: tl.constexpr,
    BLOCK_SIZE_X2: tl.constexpr,
):
    pid_x1 = tl.program_id(axis=0)
    pid_x2 = tl.program_id(axis=1)
    pid_rd = tl.program_id(axis=2)

    offs_d1 = (pid_x1 * BLOCK_SIZE_X1 + tl.arange(0, BLOCK_SIZE_X1))
    offs_d2 = (pid_x2 * BLOCK_SIZE_X2 + tl.arange(0, BLOCK_SIZE_X2))
    offs_rd = (pid_rd * BLOCK_SIZE_RD + tl.arange(0, BLOCK_SIZE_RD))

    offs_x1 = (offs_d1[:, None] * stride_x11 + offs_rd[None, :] * stride_x12)
    offs_x2 = (offs_d2[:, None] * stride_x21 + offs_rd[None, :] * stride_x22)
    offs_out = (offs_d1[:, None] * out_d2) + (offs_d2[None, :])

    x1_ptrs = x1_ptr + offs_x1
    x2_ptrs = x2_ptr + offs_x2
    out_ptrs = out_ptr + offs_out

    x1_mask = (offs_d1 < out_d1)[:, None] & (offs_rd < reduced_dim)[None, :]
    x2_mask = (offs_d2 < out_d2)[:, None] & (offs_rd < reduced_dim)[None, :]
    out_mask = (offs_d1 < out_d1)[:, None] & (offs_d2 < out_d2)[None, :]

    x1 = tl.expand_dims(tl.load(x1_ptrs, mask=x1_mask), -2)
    x2 = tl.load(x2_ptrs, mask=x2_mask)

    tl.atomic_add(out_ptrs, tl.sum(tl.exp(tl.log(tl.abs(x1 - x2)) * p), axis=-1), mask=out_mask)


@triton_op("triton_cdist::opt_cdist_singular", mutates_args={})
@torch.compile(fullgraph=True)
def opt_cdist_singular(x1: torch.Tensor, x2: torch.Tensor,
                       out_d1: int, out_d2: int,
                       p: float = 2.,) -> torch.Tensor:
    output = torch.zeros((out_d1, out_d2), device=x1.device)
    grid = lambda meta: (triton.cdiv(x1.size(0), meta["BLOCK_SIZE_X1"]), triton.cdiv(x2.size(0), meta["BLOCK_SIZE_X2"]), triton.cdiv(x1.size(1), meta["BLOCK_SIZE_RD"]))

    pairwise_lp_kernel[grid](x1, x2, output, *output.shape, x1.size(1),
                             x1.stride(0), x1.stride(1), x2.stride(0), x2.stride(1), p)

    return output.pow(1/p)


@triton_op("triton_cdist::opt_cdist", mutates_args={})
def opt_cdist(x1: torch.Tensor, x2: torch.Tensor, p: float = 2.) -> torch.Tensor:
    assert x1.size(-1) == x2.size(-1)

    batched_x1 = False
    batched_x2 = False
    n_batches = -1
    if x1.dim() == 3:
        n_batches = x1.size(0)
        batched_x1 = True
    if x2.dim() == 3:
        if n_batches != -1: assert n_batches == x2.size(0)
        else: n_batches = x2.size(0)
        batched_x2 = True

    if batched_x1 and batched_x2:
        batched_opt_cdist = torch.vmap(opt_cdist_singular, in_dims=(0, 0, None, None, None))
        return batched_opt_cdist(x1, x2, x1.size(1), x2.size(1), p)
    if not batched_x1 and not batched_x2: return opt_cdist_singular(x1, x2, x1.size(0), x2.size(0), p)

    out_d1 = x1.size(1 if batched_x1 else 0)
    out_d2 = x2.size(1 if batched_x2 else 0)
    rd = x1.size(-1)

    if batched_x1:
        x1 = x1.view(-1, rd)
        return opt_cdist_singular(x1, x2, x1.size(0), x2.size(0), p).view(n_batches, out_d1, out_d2)
    if batched_x2:
        x2 = x2.view(-1, rd)
        return opt_cdist_singular(x2, x1, x2.size(0), x1.size(0), p).view(n_batches, out_d2, out_d1).transpose(-1, -2)


@triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE_X1": 2, "BLOCK_SIZE_X2": 2, "BLOCK_SIZE_RD": 2}, num_stages=0, num_warps=4),
            triton.Config({"BLOCK_SIZE_X1": 2, "BLOCK_SIZE_X2": 2, "BLOCK_SIZE_RD": 2}, num_stages=0, num_warps=8),
            triton.Config({"BLOCK_SIZE_X1": 4, "BLOCK_SIZE_X2": 4, "BLOCK_SIZE_RD": 4}, num_stages=0, num_warps=4),
            triton.Config({"BLOCK_SIZE_X1": 4, "BLOCK_SIZE_X2": 4, "BLOCK_SIZE_RD": 4}, num_stages=0, num_warps=8),
            triton.Config({"BLOCK_SIZE_X1": 8, "BLOCK_SIZE_X2": 8, "BLOCK_SIZE_RD": 8}, num_stages=0, num_warps=4),
            triton.Config({"BLOCK_SIZE_X1": 8, "BLOCK_SIZE_X2": 8, "BLOCK_SIZE_RD": 8}, num_stages=0, num_warps=8),
            triton.Config({"BLOCK_SIZE_X1": 16, "BLOCK_SIZE_X2": 16, "BLOCK_SIZE_RD": 16}, num_stages=0, num_warps=4),
            triton.Config({"BLOCK_SIZE_X1": 16, "BLOCK_SIZE_X2": 16, "BLOCK_SIZE_RD": 16}, num_stages=0, num_warps=8),
        ],
        key=['out_d1', 'out_d2', 'reduced_dim'],
        restore_value=['x1_ptr', 'x2_ptr', 'fwd_res_ptr'],
        reset_to_zero=['x1_grad_ptr', 'x2_grad_ptr'],
    )
@triton.jit
def pairwise_lp_backward_kernel(
    x1_ptr,
    x2_ptr,
    fwd_res_ptr,
    fwd_grad_ptr,
    x1_grad_ptr,
    x2_grad_ptr,
    out_d1, out_d2, reduced_dim,
    stride_x11, stride_x12,
    stride_x21, stride_x22,
    p,
    BLOCK_SIZE_RD: tl.constexpr,
    BLOCK_SIZE_X1: tl.constexpr,
    BLOCK_SIZE_X2: tl.constexpr,
):
    pid_x1 = tl.program_id(axis=0)
    pid_x2 = tl.program_id(axis=1)
    pid_rd = tl.program_id(axis=2)

    offs_d1 = (pid_x1 * BLOCK_SIZE_X1 + tl.arange(0, BLOCK_SIZE_X1))
    offs_d2 = (pid_x2 * BLOCK_SIZE_X2 + tl.arange(0, BLOCK_SIZE_X2))
    offs_rd = (pid_rd * BLOCK_SIZE_RD + tl.arange(0, BLOCK_SIZE_RD))

    offs_x1 = (offs_d1[:, None] * stride_x11 + offs_rd[None, :] * stride_x12)
    offs_x2 = (offs_d2[:, None] * stride_x21 + offs_rd[None, :] * stride_x22)
    offs_fwd = (offs_d1[:, None] * out_d2) + (offs_d2[None, :])

    x1_ptrs = x1_ptr + offs_x1
    x2_ptrs = x2_ptr + offs_x2
    fwdr_ptrs = fwd_res_ptr + offs_fwd
    fwdg_ptrs = fwd_grad_ptr + offs_fwd
    x1_grad_ptrs = x1_grad_ptr + offs_x1
    x2_grad_ptrs = x2_grad_ptr + offs_x2

    x1_mask = (offs_d1 < out_d1)[:, None] & (offs_rd < reduced_dim)[None, :]
    x2_mask = (offs_d2 < out_d2)[:, None] & (offs_rd < reduced_dim)[None, :]
    fwd_mask = (offs_d1 < out_d1)[:, None] & (offs_d2 < out_d2)[None, :]

    x1 = tl.expand_dims(tl.load(x1_ptrs, mask=x1_mask), -2)
    x2 = tl.load(x2_ptrs, mask=x2_mask)
    fwdr = tl.load(fwdr_ptrs, mask=fwd_mask, other=float('inf'))
    fwdg = tl.load(fwdg_ptrs, mask=fwd_mask)

    pairwise_diff = (x1 - x2).permute(2, 0, 1)
    abs_log_diff = tl.log(pairwise_diff.abs())
    grad_partial = (((pairwise_diff > 0) * 2) - 1) * tl.exp(abs_log_diff * (p - 1)) / fwdr
    grad_partial = fwdg * tl.where(abs_log_diff == float('-inf'), 0., grad_partial)

    grad_mask = (tl.expand_dims(x1_mask, -2) & x2_mask).permute(2, 0, 1)
    x1_grad_partial = tl.sum(tl.where(grad_mask, grad_partial, 0.), -1).trans()
    x2_grad_partial = -tl.sum(tl.where(grad_mask, grad_partial, 0.), 1).trans()

    tl.atomic_add(x1_grad_ptrs, x1_grad_partial, mask=x1_mask)
    tl.atomic_add(x2_grad_ptrs, x2_grad_partial, mask=x2_mask)


@triton_op("triton_cdist::opt_cdist_singular_backward", mutates_args={})
@torch.compile(fullgraph=True)
def opt_cdist_singular_backward(x1: torch.Tensor, x2: torch.Tensor, fwd_res: torch.Tensor, fwd_grad: torch.Tensor, p: float) -> List[torch.Tensor]:
    x1_grad = torch.zeros_like(x1)
    x2_grad = torch.zeros_like(x2)

    grid = lambda meta: (triton.cdiv(x1.size(0), meta["BLOCK_SIZE_X1"]), triton.cdiv(x2.size(0), meta["BLOCK_SIZE_X2"]), triton.cdiv(x1.size(1), meta["BLOCK_SIZE_RD"]))

    pairwise_lp_backward_kernel[grid](x1, x2, fwd_res, fwd_grad, x1_grad, x2_grad, *fwd_res.shape, x1.size(1),
                                      x1.stride(0), x1.stride(1), x2.stride(0), x2.stride(1), p)

    return x1_grad, x2_grad


@triton_op("triton_cdist::opt_cdist_backward", mutates_args={})
def opt_cdist_backward(x1: torch.Tensor, x2: torch.Tensor, fwd_res: torch.Tensor, fwd_grad: torch.Tensor, p: float = 2.) -> List[torch.Tensor]:
    batched_x1 = False
    batched_x2 = False
    n_batches = -1
    if x1.dim() == 3:
        n_batches = x1.size(0)
        batched_x1 = True
    if x2.dim() == 3:
        if n_batches != -1: assert n_batches == x2.size(0)
        else: n_batches = x2.size(0)
        batched_x2 = True

    if batched_x1 and batched_x2:
        logging.warning('Naive and slow batching')
        gradients = [opt_cdist_singular_backward(x1i, x2j, frk, frg, p) for x1i, x2j, frk in zip(x1, x2, fwd_grad, fwd_res)]

        x1_grad = []
        x2_grad = []
        for i in range(len(gradients)):
            x1_grad.append(gradients[i][0][None,])
            x2_grad.append(gradients[i][1][None,])

        return torch.vstack(x1_grad), torch.vstack(x2_grad)
    if not batched_x1 and not batched_x2: return opt_cdist_singular_backward(x1, x2, fwd_res, fwd_grad, p)

    out_d1 = x1.size(1 if batched_x1 else 0)
    out_d2 = x2.size(1 if batched_x2 else 0)
    rd = x1.size(-1)

    if batched_x1:
        x1_grad, x2_grad = opt_cdist_singular_backward(x1.view(-1, rd), x2, fwd_res.view(-1, out_d2), fwd_grad, p)
        x1_grad = x1_grad.view(n_batches, out_d1, rd)
    elif batched_x2:
        x2_grad, x1_grad = opt_cdist_singular_backward(x2.view(-1, rd), x1, fwd_res.transpose(-1, -2).reshape(-1, out_d1), fwd_grad, p)
        x2_grad = x2_grad.view(n_batches, out_d2, rd)
    return x1_grad, x2_grad


def backward(ctx, grad):
    x1, x2, fwd_res = ctx.saved_tensors
    p = ctx.p

    x1_grad, x2_grad = opt_cdist_backward(x1, x2, fwd_res.pow(p - 1), grad, p)
    return x1_grad, x2_grad, None


@opt_cdist.register_kernel('cpu')
def cdist_fallback(x1, x2, p=2.):
    return torch.cdist(x1, x2, p=p)


@opt_cdist_backward.register_kernel('cpu')
def cdist_backward_fallback(x1, x2, fwd_res, fwd_grad, p=2.):
    batched_x1 = False
    batched_x2 = False
    n_batches = -1
    if x1.dim() == 3:
        n_batches = x1.size(0)
        batched_x1 = True
    if x2.dim() == 3:
        if n_batches != -1: assert n_batches == x2.size(0)
        else: n_batches = x2.size(0)
        batched_x2 = True

    out_d1 = x1.size(1 if batched_x1 else 0)
    out_d2 = x2.size(1 if batched_x2 else 0)
    rd = x1.size(-1)

    if batched_x1 and batched_x2:
        logging.warning('Naive and slow batching')
        gradients = [cdist_singular_backward_fallback(x1i, x2j, frk, frg, p) for x1i, x2j, frk in zip(x1, x2, fwd_res, fwd_grad)]

        x1_grad = []
        x2_grad = []
        for i in range(len(gradients)):
            x1_grad.append(gradients[i][0][None,])
            x2_grad.append(gradients[i][1][None,])

        return torch.vstack(x1_grad), torch.vstack(x2_grad)
    if not batched_x1 and not batched_x2: return cdist_singular_backward_fallback(x1, x2, fwd_res, fwd_grad, p)

    if batched_x1:
        x1_grad, x2_grad = cdist_singular_backward_fallback(x1.view(-1, rd), x2, fwd_res.view(-1, out_d2), fwd_grad, p)
        x1_grad = x1_grad.view(n_batches, out_d1, rd)
    elif batched_x2:
        x2_grad, x1_grad = cdist_singular_backward_fallback(x2.view(-1, rd), x1, fwd_res.transpose(-1, -2).reshape(-1, out_d1), fwd_grad, p)
        x2_grad = x2_grad.view(n_batches, out_d2, rd)
    return x1_grad, x2_grad


def cdist_singular_backward_fallback(x1, x2, fwd_res, fwd_grad, p):
    partial_grad = (x1.unsqueeze(1) - x2).permute(2, 0, 1)
    partial_grad *= partial_grad.abs().pow(p - 2)
    partial_grad /= fwd_res.pow(p - 1)
    partial_grad = fwd_grad * torch.where(partial_grad.isnan(), 0, partial_grad)

    x1_grad = partial_grad.sum(-1).transpose(0, 1)
    x2_grad = -partial_grad.sum(1).transpose(0, 1)
    return x1_grad, x2_grad


def setup_context(ctx, inputs, output):
    ctx.save_for_backward(*inputs[:2], output)
    ctx.p = inputs[-1]


opt_cdist.register_autograd(backward, setup_context=setup_context)


if __name__ == '__main__':
    x1 = torch.randn(2, 128, 32, device='cuda', requires_grad=True)
    x2 = torch.randn(128, 32, device='cuda', requires_grad=True)
    print(x1, x2, sep='\n')

    p = 2.
    out = opt_cdist(x1, x2, p=p)
    target = torch.cdist(x1, x2, p=p)
    print('Forward:', torch.allclose(out, target), (target - out).abs().max())

    out.sum().backward()
    manual_x1_grad = x1.grad.clone()
    manual_x2_grad = x2.grad.clone()
    x1.grad = None
    x2.grad = None
    target.sum().backward()
    print('Backward x1:', torch.allclose(x1.grad, manual_x1_grad, rtol=1e-4, atol=1e-5), (manual_x1_grad - x1.grad).abs().max())
    print('Backward x2:', torch.allclose(x2.grad, manual_x2_grad, rtol=1e-4, atol=1e-5), (manual_x2_grad - x2.grad).abs().max())
