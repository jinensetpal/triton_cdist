#!/usr/bin/env python3

from triton_cdist.lp_reduce import opt_cdist
import torch


def test_correctness(x1_size, x2_size, rd_size, p, reduciton_op):
    x1 = torch.randn(*x1_size, rd_size, device='cuda', requires_grad=True)
    x2 = torch.randn(*x2_size, rd_size, device='cuda', requires_grad=True)

    p = p
    out = opt_cdist(x1.clone(), x2.clone(), p=p)
    target = torch.cdist(x1.clone(), x2.clone(), p=p)

    print('Forward:', (target - out).abs().max())
    assert torch.allclose(out, target, rtol=1e-4, atol=1e-5)

    reduction_op(out).backward()
    manual_x1_grad = x1.grad.clone()
    manual_x2_grad = x2.grad.clone()
    x1.grad = None
    x2.grad = None
    reduction_op(target).backward()

    print('Backward x1:', (manual_x1_grad - x1.grad).abs().max())
    print('Backward x2:', (manual_x2_grad - x2.grad).abs().max())
    assert torch.allclose(x1.grad, manual_x1_grad, rtol=1e-4, atol=1e-5)
    assert torch.allclose(x2.grad, manual_x2_grad, rtol=1e-4, atol=1e-5)


if __name__ == '__main__':
    x_sizes = [(2 ** i,) for i in range(5, 8)]
    x_sizes.extend([(2, *x_size) for x_size in x_sizes])
    rd_sizes = [2 ** i for i in range(5, 8)]
    reduction_ops = [lambda x: x.sum(), lambda x: (x / 2).pow(2).log().mean(), lambda x: x.max()]
    p_s = torch.arange(1, 11)

    torch._dynamo.config.cache_size_limit = len(x_sizes) ** 2 * len(rd_sizes) * len(p_s)
    torch.autograd.set_detect_anomaly(True)

    for x1_size in x_sizes:
        for x2_size in x_sizes:
            for rd_size in rd_sizes:
                for p in p_s:
                    for rop_idx, reduction_op in enumerate(reduction_ops):
                        print('=' * 5, x1_size, x2_size, rd_size, p, rop_idx)
                        test_correctness(x1_size, x2_size, rd_size, p, reduction_op)
                        print('=' * 30)
