# --------------------- Setup for dataset and model ---------------------
# Package installation, loading, and dataloaders
# There's also a resnet18 model defined.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

from torchvision import datasets, transforms
# from tensorboardX import SummaryWriter

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

batch_size = 64
k = 10

np.random.seed(42)
torch.manual_seed(42)

## Dataloaders
# train_dataset = datasets.CIFAR10('cifar10_data/', train=True, download=True, transform=transforms.Compose(
#     [transforms.ToTensor()]
# ))
test_dataset = datasets.CIFAR10('cifar10_data/', train=False, download=True, transform=transforms.Compose(
    [transforms.ToTensor()]
))

# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

def tp_relu(x, delta=1.):
    ind1 = (x < -1. * delta).float()
    ind2 = (x > delta).float()
    return .5 * (x + delta) * (1 - ind1) * (1 - ind2) + x * ind2

def tp_smoothed_relu(x, delta=1.):
    ind1 = (x < -1. * delta).float()
    ind2 = (x > delta).float()
    return (x + delta) ** 2 / (4 * delta) * (1 - ind1) * (1 - ind2) + x * ind2

class Normalize(nn.Module):
    def __init__(self, mu, std):
        super(Normalize, self).__init__()
        self.mu, self.std = mu, std

    def forward(self, x):
        return (x - self.mu) / self.std

class IdentityLayer(nn.Module):
    def forward(self, inputs):
        return inputs
    
class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, bn, learnable_bn, stride=1, activation='relu'):
        super(PreActBlock, self).__init__()
        self.collect_preact = True
        self.activation = activation
        self.avg_preacts = []
        self.bn1 = nn.BatchNorm2d(in_planes, affine=learnable_bn) if bn else IdentityLayer()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=not learnable_bn)
        self.bn2 = nn.BatchNorm2d(planes, affine=learnable_bn) if bn else IdentityLayer()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=not learnable_bn)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=not learnable_bn)
            )

    def act_function(self, preact):
        if self.activation == 'relu':
            act = F.relu(preact)
        elif self.activation[:6] == '3prelu':
            act = tp_relu(preact, delta=float(self.activation.split('relu')[1]))
        elif self.activation[:8] == '3psmooth':
            act = tp_smoothed_relu(preact, delta=float(self.activation.split('smooth')[1]))
        else:
            assert self.activation[:8] == 'softplus'
            beta = int(self.activation.split('softplus')[1])
            act = F.softplus(preact, beta=beta)
        return act

    def forward(self, x):
        out = self.act_function(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x  # Important: using out instead of x
        out = self.conv1(out)
        out = self.conv2(self.act_function(self.bn2(out)))
        out += shortcut
        return out

class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, n_cls, cuda=True, half_prec=False,
        activation='relu', fts_before_bn=False, normal='none'):
        super(PreActResNet, self).__init__()
        self.bn = True
        self.learnable_bn = True  # doesn't matter if self.bn=False
        self.in_planes = 64
        self.avg_preact = None
        self.activation = activation
        self.fts_before_bn = fts_before_bn
        if normal == 'cifar10':
            self.mu = torch.tensor((0.4914, 0.4822, 0.4465)).view(1, 3, 1, 1)
            self.std = torch.tensor((0.2471, 0.2435, 0.2616)).view(1, 3, 1, 1)
        else:
            self.mu = torch.tensor((0.0, 0.0, 0.0)).view(1, 3, 1, 1)
            self.std = torch.tensor((1.0, 1.0, 1.0)).view(1, 3, 1, 1)
            print('no input normalization')
        self.mu = self.mu.to(device)
        self.std = self.std.to(device)
        if half_prec:
            self.mu = self.mu.half()
            self.std = self.std.half()

        self.normalize = Normalize(self.mu, self.std)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=not self.learnable_bn)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.bn = nn.BatchNorm2d(512 * block.expansion)
        self.linear = nn.Linear(512*block.expansion, n_cls)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, self.bn, self.learnable_bn, stride, self.activation))
            # layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_features=False):
        for layer in [*self.layer1, *self.layer2, *self.layer3, *self.layer4]:
            layer.avg_preacts = []

        out = self.normalize(x)
        out = self.conv1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        if return_features and self.fts_before_bn:
            return out.view(out.size(0), -1)
        out = F.relu(self.bn(out))
        if return_features:
            return out.view(out.size(0), -1)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

def PreActResNet18(n_cls, cuda=True, half_prec=False, activation='relu', fts_before_bn=False,
    normal='none'):
    #print('initializing PA RN-18 with act {}, normal {}'.format())
    return PreActResNet(PreActBlock, [2, 2, 2, 2], n_cls=n_cls, cuda=cuda, half_prec=half_prec,
        activation=activation, fts_before_bn=fts_before_bn, normal=normal)

# intialize the model
model = PreActResNet18(10, cuda=True, activation='softplus1').to(device)
model.eval()



# --------------------- Implement the Attacks ---------------------
# Functions are given a simple useful signature that you can start with. Feel free to extend the signature as you see fit.
# You may find it useful to create a 'batched' version of PGD that you can use to create the adversarial attack.

def pgd_linf_untargeted(model, x, labels, k, eps, eps_step):
    model.eval()
    ce_loss = torch.nn.CrossEntropyLoss()
    adv_x = x.clone().detach()
    adv_x.requires_grad_(True) 
    for _ in range(k):
        adv_x.requires_grad_(True)
        model.zero_grad()
        output = model(adv_x)
        loss = ce_loss(output, labels)
        loss.backward()
        # find delta, clamp with eps
        grad = adv_x.grad.data
        adv_x = adv_x.detach() + eps_step * grad.sign()
        delta = torch.clamp(adv_x - x, min=-eps, max=eps)
        adv_x = torch.clamp(x + delta, min=0, max=1).detach()
   
    return adv_x

def pgd_l2_untargeted(model, x, labels, k, eps, eps_step):
    model.eval()
    ce_loss = torch.nn.CrossEntropyLoss()
    adv_x = x.clone().detach()
    adv_x.requires_grad_(True) 
    for _ in range(k):
        adv_x.requires_grad_(True)
        model.zero_grad()
        output = model(adv_x)
        batch_size = x.size()[0]
        loss = ce_loss(output, labels)
        loss.backward()
        grad = adv_x.grad.data
        # find delta, clamp with eps, project delta to the l2 ball
        grad_norms = (
            torch.norm(grad.view(batch_size, -1), p=2, dim=1)
            + 1e-10
        )
        grad = grad / grad_norms.view(batch_size, 1, 1, 1)
        adv_x = adv_x.detach() + eps_step * grad
        delta = adv_x - x
        delta_norms = torch.norm(delta.view(batch_size, -1), p=2, dim=1)
        factor = eps / delta_norms
        factor = torch.min(factor, torch.ones_like(delta_norms))
        delta = delta * factor.view(-1, 1, 1, 1)

        adv_x = torch.clamp(x + delta, min=0, max=1).detach()
        # HINT: https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/torchattacks/attacks/pgdl2.py 
   
    return adv_x



# --------------------- Evaluate Single and Multi-Norm Robust Accuracy ---------------------
# In this section, we evaluate the model on the Linf and L2 attacks as well as union accuracy.

def test_model_on_single_attack(model, attack='pgd_linf', eps=0.1):
    model.eval()
    tot_test, tot_acc = 0.0, 0.0
    for batch_idx, (x_batch, y_batch) in tqdm(enumerate(test_loader), total=len(test_loader), desc="Evaluating"):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        if attack == 'pgd_linf':
            x_adv = pgd_linf_untargeted(model, x_batch, y_batch, k, eps, eps/4)
        elif attack == 'pgd_l2':
            x_adv = pgd_l2_untargeted(model, x_batch, y_batch, k, eps, eps/4)
        else:
            x_adv = x_batch
        
        # get the testing accuracy and update tot_test and tot_acc
        with torch.no_grad():
            pred = torch.max(model(x_adv), dim=1)[1]
        tot_acc += (pred == y_batch).sum().item()
        tot_test += x_batch.size(0)
            
    print('Robust accuracy %.5lf' % (tot_acc/tot_test), f'on {attack} attack with eps = {eps}')


# Single-Norm Robust Accuracy

# standard accuracies
model.load_state_dict(torch.load('models/pretr_Linf.pth', map_location=device))
test_model_on_single_attack(model, attack='none', eps=0)

model.load_state_dict(torch.load('models/pretr_L2.pth', map_location=device))
test_model_on_single_attack(model, attack='none', eps=0)

model.load_state_dict(torch.load('models/pretr_RAMP.pth', map_location=device))
test_model_on_single_attack(model, attack='none', eps=0)

# Evaluate on Linf attack with different models with eps = 8/255
model.load_state_dict(torch.load('models/pretr_Linf.pth', map_location=device))
# Evaluate on Linf attack with model 1 with eps = 8/255
test_model_on_single_attack(model, attack='pgd_linf', eps=8./255.)

model.load_state_dict(torch.load('models/pretr_L2.pth', map_location=device))
# Evaluate on Linf attack with model 2 with eps = 8/255
test_model_on_single_attack(model, attack='pgd_linf', eps=8./255.)

model.load_state_dict(torch.load('models/pretr_RAMP.pth', map_location=device))
# Evaluate on Linf attack with model 3 with eps = 8/255
test_model_on_single_attack(model, attack='pgd_linf', eps=8./255.)

# Evaluate on L2 attack with different models with eps = 0.75
model.load_state_dict(torch.load('models/pretr_Linf.pth', map_location=device))
# Evaluate on L2 attack with model 1 with eps = 0.75
test_model_on_single_attack(model, attack='pgd_l2', eps=0.75)

model.load_state_dict(torch.load('models/pretr_L2.pth', map_location=device))
# Evaluate on L2 attack with model 2 with eps = 0.75
test_model_on_single_attack(model, attack='pgd_l2', eps=0.75)

model.load_state_dict(torch.load('models/pretr_RAMP.pth', map_location=device))
# Evaluate on L2 attack with model 3 with eps = 0.75
test_model_on_single_attack(model, attack='pgd_l2', eps=0.75)


# --------------------- Multi-Norm Robust Accuracy ---------------------
def test_model_on_multi_attacks(model, eps_linf=8./255., eps_l2=0.75):
    model.eval()
    tot_test, tot_acc = 0.0, 0.0
    for batch_idx, (x_batch, y_batch) in tqdm(enumerate(test_loader), total=len(test_loader), desc="Evaluating"):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        x_adv_linf = pgd_linf_untargeted(model, x_batch, y_batch, k, eps_linf, eps_linf/4)
        x_adv_l2 = pgd_l2_untargeted(model, x_batch, y_batch, k, eps_l2, eps_l2/4)
        
        ## calculate union accuracy: correct only if both attacks are correct
        
        out = model(x_adv_linf)
        pred_linf = torch.max(out, dim=1)[1]
        out = model(x_adv_l2)
        pred_l2 = torch.max(out, dim=1)[1]
        
        # TODO: get the testing accuracy with multi-norm robustness and update tot_test and tot_acc
        tot_acc += ((pred_linf == y_batch) & (pred_l2 == y_batch)).sum().item()
        tot_test += x_batch.size(0)
            
    print('Robust accuracy %.5lf' % (tot_acc/tot_test), f'on multi attacks')

# Evaluate on multi-norm attacks with different models with eps_linf = 8./255, eps_l2 = 0.75
model.load_state_dict(torch.load('models/pretr_Linf.pth', map_location=device))
# Evaluate on multi attacks with model 1
test_model_on_multi_attacks(model)

model.load_state_dict(torch.load('models/pretr_L2.pth', map_location=device))
# Evaluate on multi attacks with model 2
test_model_on_multi_attacks(model)

model.load_state_dict(torch.load('models/pretr_RAMP.pth', map_location=device))
# Evaluate on multi attacks with model 3
test_model_on_multi_attacks(model)
