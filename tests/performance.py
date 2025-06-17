#!/usr/bin/env python3

from triton_cdist.lp_reduce import opt_cdist
import triton
import torch


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['x1_size', 'x2_size', 'rd_size'],
        x_vals=[2**i for i in range(5, 15, 1)],
        x_log=True,
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'Torch'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='TFLOPS',
        plot_name='lp-dist-performance',
        args={'p': 1.},
    ))
def benchmark(x1_size, x2_size, rd_size, p, provider):
    x1 = torch.randn(x1_size, rd_size, device='cuda', requires_grad=True)
    x2 = torch.randn(x2_size, rd_size, device='cuda', requires_grad=True)

    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.cdist(x1, x2, p=p), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: opt_cdist(x1, x2, p=p), quantiles=quantiles)
    perf = lambda ms: 2 * x1_size * x2_size * rd_size * 1e-12 / (ms * 1e-3)

    return perf(ms), perf(max_ms), perf(min_ms)


if __name__ == '__main__':
    for p in [1., 2., 10.]:
        benchmark.benchmarks.args = {'p': p}
        benchmark.run(print_data=True, show_plots=True)
