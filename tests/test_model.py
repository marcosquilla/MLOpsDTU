import os
import sys
from pathlib import Path
import pytest
from torchvision import datasets, transforms

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.models.model import MyAwesomeModel

def initialise():
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                              ])
    testset = datasets.MNIST(str(Path(__file__).parents[1]) + '/data/', train=False, transform=transform)
    num_classes = 10
    channels = 1
    height = testset.data.shape[-1]
    width = testset.data.shape[-2]
    model = MyAwesomeModel(num_classes, channels, height, width, 
                                    [{'nFilters':40, 'kernel':3, 'stride':1, 'padding':1},
                                    {'nFilters':80, 'kernel':3, 'stride':1, 'padding':1}],
                                    [128, 64])
    model.eval()
    return model, testset

@pytest.mark.parametrize("test_index", [1,2,3,4,5])
def test_model(test_index):
    model, testset = initialise()
    assert model(testset.data[test_index,:,:].unsqueeze(0).unsqueeze(0).float()).detach().numpy().shape[1:] == (10,)