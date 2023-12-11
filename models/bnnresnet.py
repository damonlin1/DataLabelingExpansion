import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample


class ResidualBlock(PyroModule):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        # self.conv1 = PyroModule[nn.Conv2d](in_channels=1, out_channels=28, kernel_size=(3,3), stride=1, padding=1)
        # self.conv1.weight = PyroSample(dist.Normal(0., 1.).expand(self.conv1.weight.shape).to_event(self.conv1.weight.dim()))
        # self.conv1.bias = PyroSample(dist.Normal(0., 1.).expand(self.conv1.bias.shape).to_event(self.conv1.bias.dim()))

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = self.relu(out)
        return out


class ResNetBNN(PyroModule):
    def __init__(self, num_classes=10, device=torch.device("cpu")):
        super(ResNetBNN, self).__init__()
        
        self.device = device
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = PyroModule[nn.Linear](512, num_classes)
        self.fc.weight = PyroSample(dist.Normal(0, torch.tensor(1., device=device)).expand([num_classes, 512]).to_event(2))
        self.fc.bias = PyroSample(dist.Normal(0, torch.tensor(1., device=device)).expand([num_classes]).to_event(1))

    def make_layer(self, block, out_channels, num_blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x, y=None):
        out = self.bn1(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        out = F.log_softmax(out, dim=-1)
        
        with pyro.plate("data", x.shape[0]):
            dim = out.shape[1]
            # From https://pyro.ai/examples/forecast_simple.html
            # trans_scale = pyro.sample(
            #     "trans_scale", dist.LogNormal(torch.zeros(dim, device=self.device), 0.1).to_event(1)
            # )
            # trans_corr = pyro.sample("trans_corr", dist.LKJCholesky(dim, torch.ones((), device=self.device)))
            # trans_scale_tril = trans_scale.unsqueeze(-1) * trans_corr 

            pyro.sample("obs", dist.Categorical(logits=out), obs=y)
            # pyro.sample("logits", dist.MultivariateNormal(out, torch.eye(dim, device=self.device)))
        return out
