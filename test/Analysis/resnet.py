import torch
import torch.fx as fx
import torch.nn.functional as F
import torchvision.models as models
import torch.nn as nn

from concurrent.futures import ThreadPoolExecutor
import time
import operator
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Interval Analysis")
    parser.add_argument("--model", type=str, default="resnet18",
                        help="Model name to analyze")
    parser.add_argument("--shape", type=tuple, default=(1, 3, 224, 224),
                        help="Input shape for analysis")
    return parser.parse_args()


class Interval:
    __slots__ = ("l", "u")

    def __init__(self, l: torch.Tensor, u: torch.Tensor):
        self.l = l
        self.u = u

    def __repr__(self):
        return f"Interval(shape={tuple(self.l.shape)}, “[{self.l.min():.3f},{self.u.max():.3f}]”)"


def interval_conv2d(x: Interval, W, b, stride, padding, dilation, groups):
    Wp = torch.clamp(W, min=0)
    Wn = torch.clamp(W, max=0)
    lo = (F.conv2d(x.l, Wp, None, stride, padding, dilation, groups)
          + F.conv2d(x.u, Wn, None, stride, padding, dilation, groups))
    hi = (F.conv2d(x.u, Wp, None, stride, padding, dilation, groups)
          + F.conv2d(x.l, Wn, None, stride, padding, dilation, groups))
    if b is not None:
        lo = lo + b.view(1, -1, 1, 1)
        hi = hi + b.view(1, -1, 1, 1)
    return Interval(lo, hi)


def interval_linear(x: Interval, W, b):
    Wp = torch.clamp(W, min=0)
    Wn = torch.clamp(W, max=0)
    lo = x.l @ Wp.t() + x.u @ Wn.t()
    hi = x.u @ Wp.t() + x.l @ Wn.t()
    if b is not None:
        lo = lo + b
        hi = hi + b
    return Interval(lo, hi)


def interval_relu(x: Interval):
    return Interval(F.relu(x.l), F.relu(x.u))


def interval_add(a: Interval, b: Interval):
    return Interval(a.l + b.l, a.u + b.u)


def interval_mul(a: Interval, b: Interval):
    c0 = a.l * b.l
    c1 = a.l * b.u
    c2 = a.u * b.l
    c3 = a.u * b.u
    lo = torch.min(torch.min(c0, c1), torch.min(c2, c3))
    hi = torch.max(torch.max(c0, c1), torch.max(c2, c3))
    return Interval(lo, hi)


def interval_maxpool2d(x: Interval, k, s, p, d, ceil):
    return Interval(
        F.max_pool2d(x.l, k, s, p, d, ceil),
        F.max_pool2d(x.u, k, s, p, d, ceil),
    )


def interval_avgpool2d(x: Interval, k, s, p, ceil, cip):
    return Interval(
        F.avg_pool2d(x.l, k, s, p, ceil, cip),
        F.avg_pool2d(x.u, k, s, p, ceil, cip),
    )


def interval_adaptive_avgpool2d(x: Interval, out_sz):
    return Interval(
        F.adaptive_avg_pool2d(x.l, out_sz),
        F.adaptive_avg_pool2d(x.u, out_sz),
    )


def interval_batchnorm2d(x: Interval, bn: torch.nn.BatchNorm2d):
    gamma = bn.weight.view(1, -1, 1, 1)
    beta = bn.bias.view(1, -1, 1, 1)
    m = bn.running_mean.view(1, -1, 1, 1)
    v = bn.running_var.view(1, -1, 1, 1)
    eps = bn.eps
    std = torch.sqrt(v + eps)
    a = gamma / std
    b = beta - a * m
    lo = torch.where(a >= 0, x.l * a + b, x.u * a + b)
    hi = torch.where(a >= 0, x.u * a + b, x.l * a + b)
    return Interval(lo, hi)


def compute_interval(node: fx.Node, inputs, module: fx.GraphModule):
    op = node.op
    if op == "call_module":
        sub = module.get_submodule(node.target)
        x = inputs[0]
        if isinstance(sub, nn.Sequential):
            iv = x
            for m in sub:
                if isinstance(m, nn.Conv2d):
                    iv = interval_conv2d(iv, m.weight, m.bias,
                                         m.stride, m.padding,
                                         m.dilation, m.groups)
                elif isinstance(m, nn.BatchNorm2d):
                    iv = interval_batchnorm2d(iv, m)
                elif isinstance(m, nn.ReLU):
                    iv = interval_relu(iv)
                else:
                    raise NotImplementedError(f"Sequential child not supported: {type(m)}")
            return iv

        if isinstance(sub, torch.nn.Conv2d):
            return interval_conv2d(x, sub.weight, sub.bias,
                                   sub.stride, sub.padding,
                                   sub.dilation, sub.groups)
        if isinstance(sub, torch.nn.Linear):
            return interval_linear(x, sub.weight, sub.bias)
        if isinstance(sub, torch.nn.ReLU):
            return interval_relu(x)
        if isinstance(sub, torch.nn.MaxPool2d):
            return interval_maxpool2d(x,
                                      sub.kernel_size, sub.stride,
                                      sub.padding, sub.dilation,
                                      sub.ceil_mode)
        if isinstance(sub, torch.nn.AvgPool2d):
            return interval_avgpool2d(x,
                                      sub.kernel_size, sub.stride,
                                      sub.padding, sub.ceil_mode,
                                      sub.count_include_pad)
        if isinstance(sub, torch.nn.AdaptiveAvgPool2d):
            return interval_adaptive_avgpool2d(x, sub.output_size)
        if isinstance(sub, torch.nn.BatchNorm2d):
            return interval_batchnorm2d(x, sub)
        raise NotImplementedError(f"Unsupported module: {type(sub)}")

    if op == "call_function":
        fn = node.target
        if fn in (torch.add, operator.add):
            return interval_add(*inputs)
        if fn in (torch.mul, operator.mul):
            return interval_mul(*inputs)
        if fn is F.relu:
            return interval_relu(inputs[0])
        if fn is torch.flatten:
            iv = inputs[0]

            if len(inputs) >= 3:
                start_dim, end_dim = inputs[1], inputs[2]
            elif len(inputs) == 2:
                start_dim = inputs[1]
                end_dim = node.kwargs.get("end_dim", -1)
            else:
                start_dim = node.kwargs.get("start_dim", 0)
                end_dim = node.kwargs.get("end_dim", -1)
            l2 = iv.l.flatten(start_dim, end_dim)
            u2 = iv.u.flatten(start_dim, end_dim)
            return Interval(l2, u2)
        raise NotImplementedError(f"Unsupported function: {fn}")

    if op == "call_method":
        m = node.target
        if m == "view":
            iv = inputs[0]
            shape = node.args[1:]
            return Interval(iv.l.view(*shape), iv.u.view(*shape))
        if m == "size":
            iv = inputs[0]
            dim = node.args[1]
            return iv.l.size(dim)
        if m == "relu":
            return interval_relu(inputs[0])
        raise NotImplementedError(f"Unsupported method: {m}")
    if op == "output":
        return inputs[0]
    raise NotImplementedError(f"Unknown op: {op}")


def analyze(model: torch.nn.Module,
            input_lo: torch.Tensor,
            input_hi: torch.Tensor):
    device = input_lo.device
    model = model.eval().to(device)
    input_lo = input_lo.to(device)
    input_hi = input_hi.to(device)
    gm = fx.symbolic_trace(model)
    env = {}
    params = dict(model.named_parameters())
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            env[node] = Interval(input_lo, input_hi)
        elif node.op == "get_attr":
            w = params[node.target]
            env[node] = Interval(w, w)
    for node in gm.graph.nodes:
        if node not in env:
            inps = []
            for arg in node.args:
                if isinstance(arg, fx.Node):
                    inps.append(env[arg])
                else:
                    inps.append(arg)
            env[node] = compute_interval(node, inps, gm)
    t = {}
    for node in gm.graph.nodes:
        val = env[node]
        if isinstance(val, Interval):
            lo = val.l.min().item()
            hi = val.u.max().item()
            t[node.name.replace('_', '.')] = (lo, hi)
    return t


class OutputRangeMonitor:
    def __init__(self, model: nn.Module, orig_ranges: dict, max_workers: int = 4):
        self.model = model
        self.orig_ranges = orig_ranges
        self.slient = False
        self.failed_list = []
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.handles = []
        self._register_hooks()

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if name in self.orig_ranges:
                handle = module.register_forward_hook(self._make_hook(name))
                self.handles.append(handle)

    def _make_hook(self, layer_name):
        def hook(module, input_, output):
            out_cpu = output.detach().cpu()
            if not self.slient:
                self.executor.submit(self._check_range, layer_name, out_cpu)
            else:
                self.executor.submit(self._check_range_notty, layer_name, out_cpu)

        return hook

    def _check_range(self, layer_name, tensor_cpu: torch.Tensor):
        orig_min, orig_max = self.orig_ranges[layer_name]
        cur_min = tensor_cpu.min().item()
        cur_max = tensor_cpu.max().item()
        if cur_min < orig_min or cur_max > orig_max:
            print(f"[WARN] Layer {layer_name}: "
                  f"output range [{cur_min:.4f}, {cur_max:.4f}] "
                  f"exceeds original [{orig_min}, {orig_max}]")
        else:
            print(f"[OK]   Layer {layer_name}: [{cur_min:.4f}, {cur_max:.4f}]")
            pass

    def _check_range_notty(self, layer_name, tensor_cpu: torch.Tensor):
        orig_min, orig_max = self.orig_ranges[layer_name]
        cur_min = tensor_cpu.min().item()
        cur_max = tensor_cpu.max().item()
        self.failed_list = []
        if cur_min < orig_min or cur_max > orig_max:
            self.failed_list.append(layer_name)

    def close(self):
        for h in self.handles:
            h.remove()
        self.executor.shutdown(wait=True)


if __name__ == "__main__":
    model = models.resnet18(pretrained=True).cuda()
    BATCH = 1
    SHAPE = (BATCH, 3, 224, 224)
    TIMES = 1000
    print('Start to analyze')
    ranging = {}
    original_ranges = {}
    start_time = time.time()
    for i in range(TIMES):
        low_val = i * (1.0 / TIMES)
        high_val = (i + 1) * (1.0 / TIMES)
        lo = torch.full(SHAPE, low_val, device="cuda")
        hi = torch.full(SHAPE, high_val, device="cuda")
        ranges = analyze(model, lo, hi)
        ranging[(low_val, high_val)] = ranges
        original_ranges = ranges
    end_time = time.time()
    print(f'[Ok] Analyzing time: {end_time - start_time:.2f}s for {TIMES} times.')
    monitor = OutputRangeMonitor(model, original_ranges)
    monitor.slient = False
    x = torch.randn(1, 3, 224, 224).cuda()
    start_time = time.time()
    for _ in range(TIMES):
        with torch.no_grad():
            out = model(x)
    end_time = time.time()
    print(
        f'[Ok] Inference time: {end_time - start_time:.2f}s for {TIMES} times, per image time: {(end_time - start_time) / TIMES:.5f}s.')

    monitor.slient = True
    x = torch.randn(1, 3, 224, 224).cuda()
    start_time = time.time()
    for _ in range(TIMES):
        with torch.no_grad():
            out = model(x)
    end_time = time.time()
    print(
        f'[Ok] Inference time: {end_time - start_time:.2f}s for {TIMES} times, per image time: {(end_time - start_time) / TIMES:.5f}s.')

    monitor.close()
