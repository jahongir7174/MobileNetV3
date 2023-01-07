import copy
import math

import torch


def round_width(width):
    divisor = 8
    new_width = max(divisor, int(width + divisor / 2) // divisor * divisor)
    if new_width < 0.9 * width:
        new_width += divisor
    return int(new_width)


def fuse_conv(conv, norm):
    """
    [https://nenadmarkus.com/p/fusing-batchnorm-and-conv/]
    """
    fused_conv = torch.nn.Conv2d(conv.in_channels,
                                 conv.out_channels,
                                 conv.kernel_size,
                                 conv.stride,
                                 conv.padding,
                                 conv.dilation,
                                 conv.groups, True).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_norm = torch.diag(norm.weight.div(torch.sqrt(norm.eps + norm.running_var)))
    fused_conv.weight.copy_(torch.mm(w_norm, w_conv).view(fused_conv.weight.size()))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_norm = norm.bias - norm.weight.mul(norm.running_mean).div(torch.sqrt(norm.running_var + norm.eps))
    fused_conv.bias.copy_(torch.mm(w_norm, b_conv.reshape(-1, 1)).reshape(-1) + b_norm)

    return fused_conv


class SE(torch.nn.Module):
    """
    [https://arxiv.org/pdf/1709.01507.pdf]
    """

    def __init__(self, ch):
        super().__init__()
        self.se = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1),
                                      torch.nn.Conv2d(ch, round_width(ch // 4), 1),
                                      torch.nn.ReLU(),
                                      torch.nn.Conv2d(round_width(ch // 4), ch, 1),
                                      torch.nn.Hardsigmoid())

    def forward(self, x):
        return x * self.se(x)


class Conv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, activation, k=1, s=1, p=0, d=1, g=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, k, s, p, d, g, False)
        self.norm = torch.nn.BatchNorm2d(out_ch, 0.001, 0.01)
        self.relu = activation

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

    def fuse_forward(self, x):
        return self.relu(self.conv(x))


class Residual(torch.nn.Module):
    """
    [https://arxiv.org/pdf/1801.04381.pdf]
    """

    def __init__(self, in_ch, mid_ch, out_ch, activation, k, s, se):
        super().__init__()
        identity = torch.nn.Identity()
        self.add = s == 1 and in_ch == out_ch
        self.res = torch.nn.Sequential(Conv(in_ch, mid_ch, activation) if in_ch != mid_ch else identity,
                                       Conv(mid_ch, mid_ch, activation, k, s, k // 2, 1, mid_ch),
                                       SE(mid_ch) if se else identity,
                                       Conv(mid_ch, out_ch, identity))

    def forward(self, x):
        return x + self.res(x) if self.add else self.res(x)


class MobileNetV3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        out = [16, 24, 40, 80, 112, 160, 960, 1280, 1000]
        mid = [16, 64, 72, 120, 240, 200, 184, 480, 672, 960]

        self.p1 = []
        self.p2 = []
        self.p3 = []
        self.p4 = []
        self.p5 = []

        # p1/2
        self.p1.append(Conv(3, out[0], torch.nn.Hardswish(), 3, 2, 1))
        self.p1.append(Residual(out[0], mid[0], out[0], torch.nn.ReLU(), 3, 1, False))
        # p2/4
        self.p2.append(Residual(out[0], mid[1], out[1], torch.nn.ReLU(), 3, 2, False))
        self.p2.append(Residual(out[1], mid[2], out[1], torch.nn.ReLU(), 3, 1, False))
        # p3/8
        self.p3.append(Residual(out[1], mid[2], out[2], torch.nn.ReLU(), 5, 2, True))
        self.p3.append(Residual(out[2], mid[3], out[2], torch.nn.ReLU(), 5, 1, True))
        self.p3.append(Residual(out[2], mid[3], out[2], torch.nn.ReLU(), 5, 1, True))
        # p4/16
        self.p4.append(Residual(out[2], mid[4], out[3], torch.nn.Hardswish(), 3, 2, False))
        self.p4.append(Residual(out[3], mid[5], out[3], torch.nn.Hardswish(), 3, 1, False))
        self.p4.append(Residual(out[3], mid[6], out[3], torch.nn.Hardswish(), 3, 1, False))
        self.p4.append(Residual(out[3], mid[6], out[3], torch.nn.Hardswish(), 3, 1, False))
        self.p4.append(Residual(out[3], mid[7], out[4], torch.nn.Hardswish(), 3, 1, True))
        self.p4.append(Residual(out[4], mid[8], out[4], torch.nn.Hardswish(), 3, 1, True))
        # p5/32
        self.p5.append(Residual(out[4], mid[8], out[5], torch.nn.Hardswish(), 5, 2, True))
        self.p5.append(Residual(out[5], mid[9], out[5], torch.nn.Hardswish(), 5, 1, True))
        self.p5.append(Residual(out[5], mid[9], out[5], torch.nn.Hardswish(), 5, 1, True))

        self.p1 = torch.nn.Sequential(*self.p1)
        self.p2 = torch.nn.Sequential(*self.p2)
        self.p3 = torch.nn.Sequential(*self.p3)
        self.p4 = torch.nn.Sequential(*self.p4)
        self.p5 = torch.nn.Sequential(*self.p5)

        self.fc = torch.nn.Sequential(Conv(out[5], out[6], torch.nn.Hardswish()),
                                      torch.nn.AdaptiveAvgPool2d(1),
                                      torch.nn.Flatten(),
                                      torch.nn.Linear(out[6], out[7]),
                                      torch.nn.Hardswish(),
                                      torch.nn.Dropout(0.2),
                                      torch.nn.Linear(out[7], out[8]))

        # initialize weights
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out = fan_out * m.dilation[0] * m.dilation[1] // m.groups
                torch.nn.init.normal_(m.weight, 0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            if isinstance(m, torch.nn.Linear):
                init_range = 1.0 / math.sqrt(m.weight.size()[0])
                torch.nn.init.uniform_(m.weight, -init_range, init_range)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.p1(x)
        x = self.p2(x)
        x = self.p3(x)
        x = self.p4(x)
        x = self.p5(x)

        x = self.fc(x)
        return x

    def export(self):
        model = self.fuse()
        from timm.models.layers import activations
        for n, m in model.named_modules():
            if type(m) is Conv:
                if isinstance(m.relu, torch.nn.Hardswish):
                    m.relu = activations.HardSwish()
            if type(m) is SE:
                if isinstance(m.se[4], torch.nn.Hardsigmoid):
                    m.se[4] = activations.HardSigmoid()
            if n == 'fc':
                if isinstance(m[4], torch.nn.Hardswish):
                    m[4] = activations.HardSwish()
        return model

    def fuse(self):
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'norm'):
                m.conv = fuse_conv(m.conv, m.norm)
                m.forward = m.fuse_forward
                delattr(m, 'norm')
        return self


class EMA:
    """
    [https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage]
    """

    def __init__(self, model, decay=0.9999):
        super().__init__()
        self.decay = decay
        self.model = copy.deepcopy(model).eval()

        for param in self.model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, args, model):
        if args.distributed:
            model = model.module

        m_std = model.state_dict().values()
        e_std = self.model.state_dict().values()

        for m, e in zip(m_std, e_std):
            e.copy_(self.decay * e + (1. - self.decay) * m)


class StepLR:
    def __init__(self, optimizer):
        self.optimizer = optimizer

        self.decay_rate = 0.97
        self.decay_epochs = 2.4
        self.warmup_epochs = 3.0
        self.warmup_lr_init = 1e-6

        self.lr = [param_group['lr'] for param_group in self.optimizer.param_groups]

        self.warmup_lr = [(v - self.warmup_lr_init) / self.warmup_epochs for v in self.lr]
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.warmup_lr_init

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            values = [self.warmup_lr_init + epoch * s for s in self.warmup_lr]
        else:
            epoch = epoch - self.warmup_epochs
            values = [v * (self.decay_rate ** (epoch // self.decay_epochs)) for v in self.lr]
        for param_group, value in zip(self.optimizer.param_groups, values):
            param_group['lr'] = value


class LinearLR:
    def __init__(self, args, optimizer):
        self.optimizer = optimizer

        self.epochs = args.epochs
        self.values = [param_group['lr'] for param_group in self.optimizer.param_groups]

        self.warmup_epochs = 5
        self.warmup_values = [(v - 1e-4) / self.warmup_epochs for v in self.values]

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = 1e-4

    def step(self, epoch):
        epochs = self.epochs - self.warmup_epochs

        if epoch < self.warmup_epochs:
            values = [1e-4 + epoch * value for value in self.warmup_values]
        else:
            epoch = epoch - self.warmup_epochs
            if epoch < epochs:
                values = [(1 - epoch / epochs) * lr for lr in self.values]
            else:
                values = [1e-5 for _ in self.values]

        for param_group, value in zip(self.optimizer.param_groups, values):
            param_group['lr'] = value


class RMSprop(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-2, alpha=0.9, eps=1e-3, weight_decay=0.0,
                 momentum=0.9, centered=False, decoupled_decay=False, lr_in_momentum=True):

        defaults = dict(lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=momentum,
                        centered=centered, decoupled_decay=decoupled_decay, lr_in_momentum=lr_in_momentum)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for param_group in self.param_groups:
            param_group.setdefault('momentum', 0)
            param_group.setdefault('centered', False)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for param_group in self.param_groups:
            for param in param_group['params']:
                if param.grad is None:
                    continue
                grad = param.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Optimizer does not support sparse gradients')
                state = self.state[param]
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.ones_like(param.data)
                    if param_group['momentum'] > 0:
                        state['momentum_buffer'] = torch.zeros_like(param.data)
                    if param_group['centered']:
                        state['grad_avg'] = torch.zeros_like(param.data)

                square_avg = state['square_avg']
                one_minus_alpha = 1. - param_group['alpha']

                state['step'] += 1

                if param_group['weight_decay'] != 0:
                    if 'decoupled_decay' in param_group and param_group['decoupled_decay']:
                        param.data.add_(param.data, alpha=-param_group['weight_decay'])
                    else:
                        grad = grad.add(param.data, alpha=param_group['weight_decay'])

                square_avg.add_(grad.pow(2) - square_avg, alpha=one_minus_alpha)

                if param_group['centered']:
                    grad_avg = state['grad_avg']
                    grad_avg.add_(grad - grad_avg, alpha=one_minus_alpha)
                    avg = square_avg.addcmul(grad_avg, grad_avg, value=-1).add(param_group['eps']).sqrt_()
                else:
                    avg = square_avg.add(param_group['eps']).sqrt_()

                if param_group['momentum'] > 0:
                    buf = state['momentum_buffer']
                    if 'lr_in_momentum' in param_group and param_group['lr_in_momentum']:
                        buf.mul_(param_group['momentum']).addcdiv_(grad, avg, value=param_group['lr'])
                        param.data.add_(-buf)
                    else:
                        buf.mul_(param_group['momentum']).addcdiv_(grad, avg)
                        param.data.add_(-param_group['lr'], buf)
                else:
                    param.data.addcdiv_(grad, avg, value=-param_group['lr'])

        return loss


class CrossEntropyLoss(torch.nn.Module):
    """
    NLL Loss with label smoothing.
    """

    def __init__(self, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, outputs, targets):
        prob = self.softmax(outputs)
        mean = torch.mean(prob, dim=-1)

        index = torch.unsqueeze(targets, dim=1)

        nll_loss = torch.gather(prob, -1, index)
        nll_loss = torch.squeeze(nll_loss, dim=1)

        return ((self.epsilon - 1) * nll_loss - self.epsilon * mean).mean()
