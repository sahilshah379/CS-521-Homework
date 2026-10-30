import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print("device:", device)

NORMS = ("linf", "l2", "l1")
EPS = {"linf": 8 / 255, "l2": 0.5, "l1": 12.0}


def load_data(path):
    # The files store images as uint8. Convert them to floats in [0, 1] on the GPU.
    x, y = torch.load(path)
    return x.float().div(255).to(device), y.to(device)


# Training data: 10,000 images. Test data: the first 1,000 CIFAR-10 test images.
x_train, y_train = load_data("cifar10_train_10k.pt")
x_test, y_test = load_data("cifar10_test_1k.pt")
print("train:", tuple(x_train.shape), "test:", tuple(x_test.shape))


def augment(x):
    # Random horizontal flip and random crop (padding 4) for each image.
    n = x.shape[0]
    flip = torch.rand(n, device=x.device) < 0.5
    x = torch.where(flip.view(-1, 1, 1, 1), x.flip(3), x)
    padded = F.pad(x, (4, 4, 4, 4))
    i = torch.randint(0, 9, (n, 1, 1), device=x.device) + torch.arange(32, device=x.device).view(1, 32, 1)
    j = torch.randint(0, 9, (n, 1, 1), device=x.device) + torch.arange(32, device=x.device).view(1, 1, 32)
    return padded[torch.arange(n, device=x.device).view(n, 1, 1), :, i, j].permute(0, 3, 1, 2).contiguous()


def batches(x, y, batch_size, shuffle):
    idx = torch.randperm(len(x), device=x.device) if shuffle else torch.arange(len(x), device=x.device)
    for k in range(0, len(x), batch_size):
        j = idx[k:k + batch_size]
        yield x[j], y[j]


# PreActResNet-18 with half width (2.8M parameters). The input is an image in [0, 1].
class PreActBlock(nn.Module):
    def __init__(self, in_planes, planes, stride):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.shortcut = None
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Conv2d(in_planes, planes, 1, stride, bias=False)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if self.shortcut is not None else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        return out + shortcut


class PreActResNet18(nn.Module):
    def __init__(self, width=32, num_classes=10):
        super().__init__()
        self.register_buffer("mean", torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.2471, 0.2435, 0.2616]).view(1, 3, 1, 1))
        self.conv1 = nn.Conv2d(3, width, 3, 1, 1, bias=False)
        layers, in_planes = [], width
        for mult, stride in [(1, 1), (1, 1), (2, 2), (2, 1), (4, 2), (4, 1), (8, 2), (8, 1)]:
            layers.append(PreActBlock(in_planes, mult * width, stride))
            in_planes = mult * width
        self.layers = nn.Sequential(*layers)
        self.bn = nn.BatchNorm2d(in_planes)
        self.linear = nn.Linear(in_planes, num_classes)

    def forward(self, x):
        out = self.layers(self.conv1((x - self.mean) / self.std))
        out = F.adaptive_avg_pool2d(F.relu(self.bn(out)), 1).flatten(1)
        return self.linear(out)


def make_model():
    return PreActResNet18().to(device)


def load_base_model():
    model = make_model()
    model.load_state_dict(torch.load("linf_base.pt", map_location=device))
    return model


def clip_box(x, delta):
    # Make x + delta an image in [0, 1]. This never increases |delta_i|.
    return (x + delta).clamp(0, 1) - x


def input_grad(model, x, delta, y):
    delta = delta.detach().requires_grad_(True)
    loss = F.cross_entropy(model(x + delta), y, reduction="sum")
    (grad,) = torch.autograd.grad(loss, delta)
    return grad.detach()


def l2_norm(v):
    return v.flatten(1).norm(dim=1).view(-1, 1, 1, 1)


def pgd_linf(model, x, y, eps, alpha, steps, rand_init=True):
    delta = torch.zeros_like(x)
    if rand_init:
        delta = torch.distributions.Uniform(-eps, eps).sample(x.shape).to(x.device)
    delta = clip_box(x, delta)
    for _ in range(steps):
        grad = input_grad(model, x, delta, y)
        delta = (delta + alpha * grad.sign()).clamp(-eps, eps)
        delta = clip_box(x, delta)
    return (x + delta).detach()


def pgd_l2(model, x, y, eps, alpha, steps, rand_init=True):
    delta = torch.zeros_like(x)
    if rand_init:
        delta = torch.distributions.Normal(0.0, 1.0).sample(x.shape).to(x.device)
        r = torch.rand(x.shape[0], 1, 1, 1, device=x.device)
        delta = delta / l2_norm(delta) * eps * r
    delta = clip_box(x, delta)
    for _ in range(steps):
        grad = input_grad(model, x, delta, y)
        grad_norms = l2_norm(grad) + 1e-10
        grad = grad / grad_norms
        delta = delta + alpha * grad
        delta_norms = l2_norm(delta)
        factor = eps / delta_norms
        factor = torch.min(factor, torch.ones_like(delta_norms))
        delta = delta * factor
        delta = clip_box(x, delta)
    return (x + delta).detach()


def project_l1(delta, eps):
    # Projection of each image of delta onto the L1 ball of radius eps (Duchi et al., 2008).
    v = delta.flatten(1)
    a = v.abs()
    outside = a.sum(dim=1) > eps
    if not outside.any():
        return delta
    mu, _ = a.sort(dim=1, descending=True)
    cumsum = mu.cumsum(dim=1)
    idx = torch.arange(1, v.shape[1] + 1, device=v.device, dtype=v.dtype)
    rho = ((mu - (cumsum - eps) / idx > 0) * idx).argmax(dim=1, keepdim=True)
    theta = (cumsum.gather(1, rho) - eps) / (rho + 1).to(v.dtype)
    w = (a - theta).clamp_min(0) * v.sign()
    return torch.where(outside.unsqueeze(1), w, v).view(delta.shape)


def pgd_l1(model, x, y, eps, alpha, steps, rand_init=True, k=20):
    # Each step changes only the k coordinates with the largest gradient.
    # Coordinates at the image boundary that point outward are skipped.
    delta = torch.zeros_like(x)
    if rand_init:
        delta = torch.distributions.Laplace(0.0, 1.0).sample(x.shape).to(x.device)
        r = torch.rand(x.shape[0], 1, 1, 1, device=x.device)
        delta = delta / delta.flatten(1).abs().sum(1).view(-1, 1, 1, 1) * eps * r
    delta = clip_box(x, delta)
    for _ in range(steps):
        grad = input_grad(model, x, delta, y).flatten(1)
        x_cur = (x + delta).flatten(1)
        blocked = ((grad > 0) & (x_cur >= 1)) | ((grad < 0) & (x_cur <= 0))
        grad = grad.masked_fill(blocked, 0)
        top = grad.abs().topk(k, dim=1).indices
        step = torch.zeros_like(grad).scatter(1, top, grad.gather(1, top).sign())
        delta = clip_box(x, delta + alpha * (step / k).view_as(x))
        delta = project_l1(delta, eps)
    return (x + delta).detach()


ATTACKS = {"linf": pgd_linf, "l2": pgd_l2, "l1": pgd_l1}


def attack(model, x, y, norm, steps=10):
    # The attack for one norm with the default eps and step size.
    alpha = EPS["l1"] / steps ** 0.5 if norm == "l1" else 2.5 * EPS[norm] / steps
    return ATTACKS[norm](model, x, y, EPS[norm], alpha, steps)


def test_model(model, steps=50, restarts=2, batch_size=250):
    model.eval()
    torch.manual_seed(0)  # the same random starts of the attacks for each model, so the results are comparable
    start = time.time()
    correct = {k: 0 for k in ("clean", *NORMS, "union")}
    for i in range(0, len(x_test), batch_size):
        x, y = x_test[i:i + batch_size], y_test[i:i + batch_size]
        with torch.no_grad():
            clean = model(x).argmax(1) == y
        union = clean.clone()
        for norm in NORMS:
            robust = clean.clone()
            for _ in range(restarts):
                x_adv = attack(model, x, y, norm, steps)
                with torch.no_grad():
                    robust &= model(x_adv).argmax(1) == y
            correct[norm] += robust.sum().item()
            union &= robust
        correct["clean"] += clean.sum().item()
        correct["union"] += union.sum().item()
    acc = {k: 100.0 * v / len(y_test) for k, v in correct.items()}
    print("  ".join(f"{k} {v:.1f}" for k, v in acc.items()) + f"   ({time.time() - start:.0f} s)")
    return acc


def adv_train(model, loss_fn, epochs=10, lr=0.01, batch_size=128, gp_beta=None, seed=0, **kwargs):
    torch.manual_seed(seed)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * (len(x_train) // batch_size))
    params = list(model.parameters())
    start = time.time()
    for epoch in range(epochs):
        losses = []
        for x, y in batches(x_train, y_train, batch_size, shuffle=True):
            if len(y) < batch_size:  # skip the last, smaller batch
                continue
            x = augment(x)
            loss = loss_fn(model, x, y, **kwargs)
            opt.zero_grad()
            if gp_beta is None:
                loss.backward()
            else:
                # RAMP (part (b)): combine the adversarial gradient g_a and the natural gradient g_n
                # for each parameter tensor. gradient_projection is defined in part (b).
                g_a = torch.autograd.grad(loss, params)
                g_n = torch.autograd.grad(F.cross_entropy(model(x), y), params)
                for p, ga, gn in zip(params, g_a, g_n):
                    p.grad = gp_beta * gradient_projection(gn, ga) + (1 - gp_beta) * ga
            opt.step()
            sched.step()
            losses.append(loss.item())
        print(f"epoch {epoch + 1}: loss {np.mean(losses):.3f} ({time.time() - start:.0f} s)")
    model.eval()
    return model


def standard_loss(model, x, y):
    model.train()
    return F.cross_entropy(model(x), y)

def single_norm_loss(model, x, y, norm):
    model.eval()
    x_adv = attack(model, x, y, norm)
    return standard_loss(model, x_adv, y)


model_std = adv_train(make_model(), standard_loss, epochs=5)
test_model(model_std)

model_linf = adv_train(make_model(), single_norm_loss, epochs=5, norm="linf")
test_model(model_linf)


def linf_acc(model, restarts, pgd_args, batch_size=250):
    model.eval()
    torch.manual_seed(0)
    correct = 0
    for i in range(0, len(x_test), batch_size):
        x, y = x_test[i:i + batch_size], y_test[i:i + batch_size]
        with torch.no_grad():
            robust = model(x).argmax(1) == y
        for _ in range(restarts):
            x_adv = pgd_linf(model, x, y, *pgd_args)
            with torch.no_grad():
                robust &= model(x_adv).argmax(1) == y
        correct += robust.sum().item()
    return 100.0 * correct / len(y_test)


eps_ranges = [2**i for i in range(5)]
for k in eps_ranges:
    eps = k / 255
    print(f"std PGD  eps {k}/255: {linf_acc(model_std, 2, (eps, 2.5*eps/50, 50, True))}")
    print(f"adv PGD  eps {k}/255: {linf_acc(model_linf, 2, (eps, 2.5*eps/50, 50, True))}")
    print(f"std FGSM eps {k}/255: {linf_acc(model_std, 1, (eps, eps, 1, False))}")
    print(f"adv FGSM eps {k}/255: {linf_acc(model_linf, 1, (eps, eps, 1, False))}")
