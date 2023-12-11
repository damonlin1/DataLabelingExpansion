import matplotlib.pyplot as plt
import numpy as np
import pyro
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.data import TensorDataset
import numpy as np

# TODO: move all hyperparameters to input arguments/config files
# EPOCHS = 10
# LR = 0.01
# NUM_FEATURES = 128


def test(gen_predictions, testing_set, batch_size, device):
    testloader = torch.utils.data.DataLoader(testing_set, batch_size=batch_size)
    total_length = 0
    total_corr = 0
    with torch.no_grad():
        for x, y in testloader:
            total_length += x.shape[0]

            x = x.to(device)
            y = y.to(device)
            pred = gen_predictions(x)

            total_corr += (y == pred).sum()
    print(f"Accuracy here: {(total_corr / total_length).item()}")
    return (total_corr / total_length).item()      
            
    
def train(batch_step, training_set, device, num_epochs, batch_size, testing_set=None, gen_predictions=None):
    # Training code originally templated from HW 3
    # bnn = BNNRegressor(dims=[3 * 32 * 32, 20, 20, 10])
    # guide = pyro.infer.autoguide.AutoDiagonalNormal(bnn)

    # adam = pyro.optim.Adam({"lr": LR})
    # svi = pyro.infer.SVI(bnn, guide, adam, loss=pyro.infer.Trace_ELBO())

    trainloader = torch.utils.data.DataLoader(training_set, batch_size=batch_size,
                                              shuffle=True,
                                            #   sampler=torch.utils.data.sampler.SubsetRandomSampler(range(500))
                                              )
    # pyro.clear_param_store()
    pbar = tqdm(range(num_epochs))
    losses = []
    accuracies = []

    cnt = 0

    for _ in pbar:
        for x, y in tqdm(trainloader):
            x = x.to(device)
            y = y.to(device)

            loss = batch_step(x, y)

            # elbo = svi.step(x, y)
            losses.append(loss / len(x))

            cnt += 1
            if cnt % 100 == 0:
                pbar.set_description(f"Loss: {loss / len(x):.3f}")
            # if elbo < -1.5:
            #     break
        if gen_predictions is not None:
            acc = test(gen_predictions, testing_set, batch_size, device)
            accuracies.append(acc)
    
    return accuracies if gen_predictions is not None else None
