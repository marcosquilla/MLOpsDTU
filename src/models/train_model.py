import sys
import argparse

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from torchvision import datasets, transforms
from pathlib import Path

from model import MyAwesomeModel

class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>"
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()
    
    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=0.0005)
        parser.add_argument('--batch_size', default=64)
        parser.add_argument('--weight_decay', default=1e-5)
        parser.add_argument('--epochs', default=15)
        
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        #print(args)
        
        # TODO: Implement training loop here
        transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                              ])

        trainset = datasets.MNIST(str(Path(__file__).parents[2]) + '/data/', train=True, transform=transform)
        num_classes = 10
        channels = 1
        height = trainset.data.shape[-1]
        width = trainset.data.shape[-2]
        
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=int(args.batch_size), shuffle=True)
        model = MyAwesomeModel(num_classes, channels, height, width, 
                                   [{'nFilters':40, 'kernel':3, 'stride':1, 'padding':1},
                                   {'nFilters':80, 'kernel':3, 'stride':1, 'padding':1}],
                                   [128, 64])
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=int(args.lr), weight_decay=int(args.weight_decay))
        
        losses = []
        model.train()
        
        for epoch in tqdm(range(int(args.epochs))):  # loop over the dataset multiple times
            running_loss = 0
            for inputs, labels in trainloader:
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                output = model(inputs)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            losses.append(running_loss/len(trainloader))
        plt.plot(losses)
        plt.savefig(str(Path(__file__).parents[2]) + '/reports/figures/training_curve.png')
        torch.save(model.state_dict(), str(Path(__file__).parents[2]) + '/models/model_state')
        
    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--load_model_from', default=str(Path(__file__).parents[2]) + "/models/model_state")
        parser.add_argument('--batch_size', default=64)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        #print(args)
        
        # TODO: Implement evaluation logic here
        transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                              ])

        testset = datasets.MNIST(str(Path(__file__).parents[2]) + '/data/', train=False, transform=transform)
        num_classes = 10
        channels = 1
        height = testset.data.shape[-1]
        width = testset.data.shape[-2]
        
        testloader = torch.utils.data.DataLoader(testset, batch_size=int(args.batch_size), shuffle=True)
        if args.load_model_from:
            model = MyAwesomeModel(num_classes, channels, height, width, 
                                   [{'nFilters':40, 'kernel':3, 'stride':1, 'padding':1},
                                   {'nFilters':80, 'kernel':3, 'stride':1, 'padding':1}],
                                   [128, 64])
            model.load_state_dict(torch.load(args.load_model_from))
            
        model.eval()
        total=0
        i = 0
        for inputs, labels in testloader: 
            preds = np.argmax(model(inputs).detach().numpy(), axis=1)
            total += np.sum(labels.detach().numpy() == preds)
            i += len(labels)
        print('Accuracy of the network on the {} test images: {:4.2f} %'.format(i, total/i*100))
            
if __name__ == '__main__':
    TrainOREvaluate()