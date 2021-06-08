import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch
from torchvision import datasets, transforms
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from models.model import MyAwesomeModel


def plot():
    parser = argparse.ArgumentParser(description="Plotting arguments")
    parser.add_argument(
        "--load_model_from",
        default=str(Path(__file__).parents[2]) + "/models/model_state",
    )
    parser.add_argument("--batch_size", default=64)
    args = parser.parse_args(sys.argv[2:])

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    testset = datasets.MNIST(
        str(Path(__file__).parents[2]) + "/data/",
        train=False, transform=transform
    )
    num_classes = 10
    channels = 1
    height = testset.data.shape[-1]
    width = testset.data.shape[-2]

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=int(args.batch_size), shuffle=True
    )
    model = MyAwesomeModel(
        num_classes,
        channels,
        height,
        width,
        [
            {"nFilters": 40, "kernel": 3, "stride": 1, "padding": 1},
            {"nFilters": 80, "kernel": 3, "stride": 1, "padding": 1},
        ],
        [128, 64],
    )
    model.load_state_dict(torch.load(args.load_model_from))
    model.eval()

    labels = []
    preds = []
    for inputs, targets in testloader:
        preds.append(model(inputs).detach().numpy())
        labels.append(targets)
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)

    tsne = TSNE(n_components=2).fit_transform(preds)
    plt.scatter(tsne[:, 0], tsne[:, 1], c=labels)
    plt.savefig(str(Path(__file__).parents[2]) + "/reports/figures/T-SNE.png")


if __name__ == "__main__":
    plot()
