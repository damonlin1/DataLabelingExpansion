import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.nn import PyroModule, PyroSample

import random
from models.bnnresnet import ResNetBNN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
num_samples = int(len(trainloader.dataset) * 0.25)
trainloader = torch.utils.data.DataLoader(
    trainloader.dataset,
    batch_size=trainloader.batch_size,
    sampler=torch.utils.data.sampler.SubsetRandomSampler(range(num_samples))
)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=True)
num_samples = int(len(testloader.dataset) * 0.1)
testloader = torch.utils.data.DataLoader(
    testloader.dataset,
    batch_size=testloader.batch_size,
    sampler=torch.utils.data.sampler.SubsetRandomSampler(range(num_samples))
)


model = ResNetBNN(device=device).to(device)


loss_fn = Trace_ELBO()

optimizer = pyro.optim.Adam({'lr': 0.001})

guide = pyro.infer.autoguide.AutoDiagonalNormal(model)

predictive = Predictive(model, guide=guide, num_samples=5)
svi = SVI(model, guide, optimizer, num_samples=10, loss=loss_fn)

num_epochs = 100
for epoch in range(num_epochs):
    running_loss = 0.0
    
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        loss = svi.step(inputs, labels)

        running_loss += loss
        if i % 10 == 9:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0

    # for name, param in model.named_parameters():
    #     if name == "layer2.1.conv2.weight":
    #         print (name, param.data)
    correct = 0
    total = 0
    for j, data in enumerate(testloader):
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        
        preds = predictive(images)
        print(preds.keys())
        print(preds['obs'].shape)
        predicted = torch.mode(preds['obs'], dim=0)[0]

        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print("accuracy: %d %%" % (100 * correct / total))


print('Training finished.')
