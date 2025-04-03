# Matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
# Numpy
import numpy as np
# Pandas
import pandas as pd
# Torch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def load_dataset(file_path = 'data/weather_data_1hr.csv'):
    df = pd.read_csv(file_path)
    temperature = df['tempC'].values
    time = np.array([i for i in range(len(temperature))])
    return time, temperature


def plot_dataset(times, values):
    # Initialize plot
    plt.figure(figsize = (10, 7))
    plt.plot(times, values)
    plt.show()

def visualize_samples(inputs, outputs):
    plt.figure(figsize = (10, 7))
    inputs = inputs.cpu().numpy()
    outputs = outputs.cpu().numpy()
    times1 = [i for i in range(len(inputs))]
    times3 = [len(inputs) + i for i in range(len(outputs))]
    plt.scatter(times1, inputs, label = "Inputs", c = "b")
    plt.scatter(times3, outputs, label = "Outputs", c = "r")
    plt.plot(times1 + times3, np.hstack([inputs, outputs]),  "k--")
    plt.legend(loc = "best")
    plt.show()


def test_model(model, dataloader, seed_value):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed_value)
    # Draw sample from dataloader (reproducible thanks to seeding)
    inputs, outputs = next(iter(dataloader))
    inputs = inputs.unsqueeze(-1).permute(1, 0, 2)
    outputs = outputs.unsqueeze(-1).permute(1, 0, 2)
    # Predict
    pred = model(inputs)
    # Compute metrics
    print("Ground truth: ", outputs.cpu().numpy()[:, 0, :].transpose())
    print("Prediction: ", pred.detach().cpu().numpy()[:, 0, :].transpose())
    print("MAE: ", np.mean(np.abs(outputs.squeeze().cpu().numpy() - pred.squeeze().detach().cpu().numpy())))


def visualize_some_predictions(model, dataloader):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    index1 = 2486
    ax = axs[0, 0]
    torch.manual_seed(index1)
    torch.cuda.manual_seed(index1)
    torch.cuda.manual_seed_all(index1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(index1)
    inputs1, outputs1, mid1 = next(iter(dataloader))
    pred1 = model(inputs1, outputs1, mid1)
    inputs1, outputs1, mid1 = inputs1.detach().numpy()[128, :], outputs1.detach().numpy()[128,
                                                                :], mid1.detach().numpy()[128, :]
    pred1 = pred1.detach().numpy()[128, :]
    times1 = [i for i in range(len(inputs1))]
    times2 = [len(inputs1)]
    times3 = [len(inputs1) + i + 1 for i in range(len(outputs1))]
    ax.scatter(times1, inputs1, label="Inputs", c="b")
    ax.scatter(times2, mid1, label="Mid", c="g")
    ax.scatter(times3, outputs1, label="Outputs", c="r")
    err = np.mean((outputs1 - pred1) ** 2)
    ax.scatter(times3, pred1, label="Predictions - Error = {}".format(err), c="c", marker="x")
    ax.plot(times1 + times2 + times3, np.hstack([inputs1, mid1, outputs1]), "k--")
    ax.legend(loc="best")

    index1 = 2986
    ax = axs[0, 1]
    torch.manual_seed(index1)
    torch.cuda.manual_seed(index1)
    torch.cuda.manual_seed_all(index1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(index1)
    inputs1, outputs1, mid1 = next(iter(dataloader))
    pred1 = model(inputs1, outputs1, mid1)
    inputs1, outputs1, mid1 = inputs1.detach().numpy()[128, :], outputs1.detach().numpy()[128,
                                                                :], mid1.detach().numpy()[128, :]
    pred1 = pred1.detach().numpy()[128, :]
    times1 = [i for i in range(len(inputs1))]
    times2 = [len(inputs1)]
    times3 = [len(inputs1) + i + 1 for i in range(len(outputs1))]
    ax.scatter(times1, inputs1, label="Inputs", c="b")
    ax.scatter(times2, mid1, label="Mid", c="g")
    ax.scatter(times3, outputs1, label="Outputs", c="r")
    err = np.mean((outputs1 - pred1) ** 2)
    ax.scatter(times3, pred1, label="Predictions - Error = {}".format(err), c="c", marker="x")
    ax.plot(times1 + times2 + times3, np.hstack([inputs1, mid1, outputs1]), "k--")
    ax.legend(loc="best")

    index1 = 3486
    ax = axs[1, 0]
    torch.manual_seed(index1)
    torch.cuda.manual_seed(index1)
    torch.cuda.manual_seed_all(index1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(index1)
    inputs1, outputs1, mid1 = next(iter(dataloader))
    pred1 = model(inputs1, outputs1, mid1)
    inputs1, outputs1, mid1 = inputs1.detach().numpy()[128, :], outputs1.detach().numpy()[128,
                                                                :], mid1.detach().numpy()[128, :]
    pred1 = pred1.detach().numpy()[128, :]
    times1 = [i for i in range(len(inputs1))]
    times2 = [len(inputs1)]
    times3 = [len(inputs1) + i + 1 for i in range(len(outputs1))]
    ax.scatter(times1, inputs1, label="Inputs", c="b")
    ax.scatter(times2, mid1, label="Mid", c="g")
    ax.scatter(times3, outputs1, label="Outputs", c="r")
    err = np.mean((outputs1 - pred1) ** 2)
    ax.scatter(times3, pred1, label="Predictions - Error = {}".format(err), c="c", marker="x")
    ax.plot(times1 + times2 + times3, np.hstack([inputs1, mid1, outputs1]), "k--")
    ax.legend(loc="best")

    index1 = 3986
    ax = axs[1, 1]
    torch.manual_seed(index1)
    torch.cuda.manual_seed(index1)
    torch.cuda.manual_seed_all(index1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(index1)
    inputs1, outputs1, mid1 = next(iter(dataloader))
    pred1 = model(inputs1, outputs1, mid1)
    inputs1, outputs1, mid1 = inputs1.detach().numpy()[128, :], outputs1.detach().numpy()[128,
                                                                :], mid1.detach().numpy()[128, :]
    pred1 = pred1.detach().numpy()[128, :]
    times1 = [i for i in range(len(inputs1))]
    times2 = [len(inputs1)]
    times3 = [len(inputs1) + i + 1 for i in range(len(outputs1))]
    ax.scatter(times1, inputs1, label="Inputs", c="b")
    ax.scatter(times2, mid1, label="Mid", c="g")
    ax.scatter(times3, outputs1, label="Outputs", c="r")
    err = np.mean((outputs1 - pred1) ** 2)
    ax.scatter(times3, pred1, label="Predictions - Error = {}".format(err), c="c", marker="x")
    ax.plot(times1 + times2 + times3, np.hstack([inputs1, mid1, outputs1]), "k--")
    ax.legend(loc="best")

    plt.tight_layout()
    plt.show()