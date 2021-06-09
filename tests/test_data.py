import numpy as np
from torchvision import datasets, transforms
from pathlib import Path

def load():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                ])

    trainset = datasets.MNIST(str(Path(__file__).parents[1]) + '/data/', train=True, transform=transform)
    testset = datasets.MNIST(str(Path(__file__).parents[1]) + '/data/', train=False, transform=transform)
    return trainset, testset

def test_all():
    trainset, testset = load()
    assert len(trainset) == 60000
    assert len(testset) == 10000
    assert trainset.data.shape[1] == 28 and trainset.data.shape[2] == 28
    assert testset.data.shape[1] == 28 and testset.data.shape[2] == 28
    assert len(list(set(trainset.targets.detach().numpy()))) == 10