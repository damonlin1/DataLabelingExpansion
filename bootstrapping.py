import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import torchvision
import torchvision.transforms as transforms
from train import train
import random as random
import numpy as np
from tqdm import tqdm
from torch.utils import data

import torch.optim as optim
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.nn import PyroModule, PyroSample
import math

from models.bnnresnet import ResNetBNN
from override_dataset import CustomCIFAR10, CustomSubset, CustomConcatDataset


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_unclassified = []
test_accuracies = []

proportion_right = []
num_added = []


def bootstrapping(training_set, testing_set, pred_conf_estimator, ledger, estimator_aux=None, conf_level=0.95):
    testloader = torch.utils.data.DataLoader(testing_set, batch_size=4096,
                                             shuffle=False,
                                            )
    predictions = []
    confidences = []
    
    for x, _ in tqdm(testloader):
        x = x.to(DEVICE)
        preds, confs = pred_conf_estimator(x)
        predictions.append(preds)
        confidences.append(confs)
    
    predictions = torch.cat(predictions)
    confidences = torch.cat(confidences)

    new_training_indices = []
    new_testing_indices = []

    curr_num_added = 0
    num_correct = 0

    for i, (prediction, confidence) in enumerate(zip(predictions, confidences)):
        if (confidence >= conf_level):
            new_training_indices.append(i)
            ledger.append([prediction.item(), testing_set.__getitem__(i)[1]])
            testing_set.change_label(i, prediction.item())

            curr_num_added += 1
            num_correct += prediction.item() == ledger[-1][1]
        else:
            new_testing_indices.append(i)

    transfer_set = CustomSubset(testing_set, new_training_indices) 
    new_training_set = CustomConcatDataset([training_set, transfer_set])
    new_testing_set = CustomSubset(testing_set, new_testing_indices)

    print("Number of transfer points: ", len(transfer_set))
    print("Number of test points remaining: ", len(new_testing_set))

    if curr_num_added:
        proportion_right.append(num_correct / curr_num_added)
    else:
        proportion_right.append(1)
    num_added.append(curr_num_added)

    return new_training_set, new_testing_set, ledger


def classification_with_bootstrapping(batch_step, pred_conf_estimator, gen_predictions,
                                      training_set, unlabeled_set, testing_set, 
                                      batch_size, epochs: int, first_frac: float, add_frac: float,
                                      conf_level: float = 0.95 
                                      ):
    ledger = []
    curr_epoch = int(epochs * first_frac)
    test_accuracies.extend(train(batch_step, training_set, DEVICE, curr_epoch, batch_size, testing_set, gen_predictions))

    while 1:
        training_set, unlabeled_set, ledger = bootstrapping(training_set, unlabeled_set, pred_conf_estimator, ledger,
                                                            conf_level=conf_level)
        num_unclassified.append(len(unlabeled_set))

        train_for = min(math.ceil(epochs * add_frac), epochs - curr_epoch)
        test_accuracies.extend(train(batch_step, training_set, DEVICE, train_for, batch_size, testing_set, gen_predictions)) 

        curr_epoch += train_for
        
        if curr_epoch >= epochs:
            break


    num_correct = 0
    for prediction, actual in ledger:
        if (prediction == actual):
            num_correct += 1
    accuracy = num_correct / len(ledger)

    print("Total guess accuracy: ", accuracy)
    print("Number of unclassified datapoints: ", len(unlabeled_set))

    
def traditional_batch_step(x, y):
    optim.zero_grad()
    loss = loss_fn(model(x), y)
    loss.backward()
    optim.step()
    return loss


if __name__ == "__main__":
    batch_size = 4096
    NUM_FEATURES = 128
    LR = 0.005

    EPOCHS = 40
    FIRST_FRAC = 0.5
    ADD_FRAC = 0.05

    # model = FC(NUM_FEATURES).to(DEVICE)

    # optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # loss_fn = nn.CrossEntropyLoss()
    # aux = DEVICE, optimizer, loss_fn

    NUM_CLASSES = 10
    NUM_TRAIN_SAMPLE = 10
    NUM_PRED_SAMPLE = 20

    model = ResNetBNN(device=DEVICE).to(DEVICE)

    loss_fn = Trace_ELBO()

    optimizer = pyro.optim.Adam({'lr': LR})

    guide = pyro.infer.autoguide.AutoDiagonalNormal(model)

    predictive = Predictive(model, guide=guide, num_samples=NUM_PRED_SAMPLE)
    svi = SVI(model, guide, optimizer, num_samples=NUM_TRAIN_SAMPLE, loss=loss_fn)


    def bnn_batch_step(images, labels):
        loss = svi.step(images, labels)
        return loss

    def bnn_pred_conf_estimator(images):
        preds = predictive(images)
        num_samps = preds['obs'].shape[0]

        counts = torch.zeros((images.shape[0], NUM_CLASSES), dtype=torch.long, device=DEVICE)
        fill = torch.ones(num_samps, dtype=torch.long, device=DEVICE)
        for i in range(images.shape[0]):
            counts[i].put_(preds['obs'][:, i], fill, accumulate=True)

        predicted_class = counts.argmax(dim=1)
        ratio_predicted = counts[range(counts.shape[0]), predicted_class] / num_samps

        return predicted_class, ratio_predicted

    def bnn_predict(images):
        return bnn_pred_conf_estimator(images)[0]


    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    training_set = CustomCIFAR10(root='./data', train=True,
                                                download=True, transform=transform,
                                                )

    # training_set = CustomSubset(training_set, torch.arange(len(training_set)//10))
    inds = torch.randperm(len(training_set)) 
    
    unlabeled_set = CustomSubset(training_set, inds[len(inds)//10:])
    training_set = CustomSubset(training_set, inds[:len(inds)//10])

    testing_set = CustomCIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testing_set = CustomSubset(testing_set, torch.arange(len(training_set)//8))

    classification_with_bootstrapping(bnn_batch_step, bnn_pred_conf_estimator, bnn_predict,
                                      training_set, unlabeled_set, testing_set,
                                      batch_size, EPOCHS, FIRST_FRAC, ADD_FRAC)

    print(f"Accuracies: {test_accuracies}")
    print(f"Number of unclassified: {num_unclassified}")
